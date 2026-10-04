"""Unit tests for kvstate: llama.cpp sequence-state layouts, builders and RoPE helpers."""

import dataclasses
import struct
import tempfile
from pathlib import Path
import unittest

import numpy as np

import kvstate
from kvstate import StateFormatError


def muse_metadata(n_layer=8, window=4, heads_kv=2, head_dim=8):
    return {"general.architecture": "muse-glimmer", "muse-glimmer.block_count": n_layer,
            "muse-glimmer.embedding_length": 32, "muse-glimmer.attention.head_count": 4,
            "muse-glimmer.attention.head_count_kv": heads_kv, "muse-glimmer.attention.key_length": head_dim,
            "muse-glimmer.attention.value_length": head_dim, "muse-glimmer.attention.sliding_window": window,
            "muse-glimmer.attention.sliding_window_pattern": 4, "muse-glimmer.rope.dimension_count": head_dim,
            "muse-glimmer.rope.freq_base": 500000.0}


def qwen_metadata(n_layer=8, heads_kv=1, head_dim=16, n_rot=8, sections=(2, 1, 1, 0)):
    return {"general.architecture": "qwen35", "qwen35.block_count": n_layer, "qwen35.embedding_length": 32,
            "qwen35.attention.head_count": 2, "qwen35.attention.head_count_kv": heads_kv,
            "qwen35.attention.key_length": head_dim, "qwen35.attention.value_length": head_dim,
            "qwen35.full_attention_interval": 4, "qwen35.rope.dimension_count": n_rot,
            "qwen35.rope.dimension_sections": list(sections), "qwen35.rope.freq_base": 1e7,
            "qwen35.ssm.conv_kernel": 4, "qwen35.ssm.inner_size": 8, "qwen35.ssm.state_size": 2,
            "qwen35.ssm.group_count": 1}


def qwen3_metadata(n_layer=4, heads=4, heads_kv=2, head_dim=16, n_rot=None, base=5e6):
    """Dense Qwen3 metadata; like the real Qwen3 GGUFs, rope.dimension_count is optional."""
    meta = {"general.architecture": "qwen3", "qwen3.block_count": n_layer, "qwen3.embedding_length": 32,
            "qwen3.attention.head_count": heads, "qwen3.attention.head_count_kv": heads_kv,
            "qwen3.attention.key_length": head_dim, "qwen3.attention.value_length": head_dim,
            "qwen3.rope.freq_base": base}
    if n_rot is not None:
        meta["qwen3.rope.dimension_count"] = n_rot
    return meta


def random_kv(spec, layers, cells, rng):
    k = {il: rng.normal(size=(cells,) + spec.head_shape(il, "k")).astype(np.float32) for il in layers}
    v = {il: rng.normal(size=(cells,) + spec.head_shape(il, "v")).astype(np.float32) for il in layers}
    return k, v


def qwen_state(spec, n_tokens, rng, *, dtype="f16", v_trans=False, flags=kvstate.FLAGS_NONE):
    full, recurrent = spec.layers_of("full"), spec.layers_of("recurrent")
    positions = np.arange(n_tokens)
    tokens = rng.integers(0, 1000, n_tokens)
    k, v = random_kv(spec, full, n_tokens, rng)
    parts = []
    if flags == kvstate.FLAGS_NONE:
        stream = kvstate.build_kv_stream(spec, full, positions, k, v, dtype=dtype, v_trans=v_trans, tokens_at=tokens)
        parts.append(kvstate.KVCache("attn", full, [stream]))
    r_row, s_row = spec.recurrent_row_bytes
    parts.append(kvstate.RecurrentState(
        "recurrent", recurrent, np.asarray([n_tokens - 1], dtype=np.int32), ((),), len(spec.layer_types),
        {il: kvstate.OpaqueRows(0, r_row, rng.bytes(r_row)) for il in recurrent},
        {il: kvstate.OpaqueRows(0, s_row, rng.bytes(s_row)) for il in recurrent}))
    return kvstate.SequenceState(spec, flags, parts), tokens, k, v


class SpecTests(unittest.TestCase):
    def test_real_model_geometry_from_metadata(self):
        muse = kvstate.ModelSpec.from_metadata({
            **muse_metadata(52, 2048, 2, 128), "muse-glimmer.embedding_length": 4096,
            "muse-glimmer.attention.head_count": 32, "muse-glimmer.rope.dimension_count": 128,
            "muse-glimmer.rope.freq_base_swa": 5e5})
        self.assertEqual(muse.layers_of("full"), tuple(range(3, 52, 4)))
        self.assertEqual(len(muse.layers_of("sliding")), 39)
        self.assertEqual([name for name, _, _ in muse.sections(0)], ["base", "swa"])
        self.assertEqual([name for name, _, _ in muse.sections(kvstate.FLAGS_PARTIAL_ONLY)], ["swa"])
        self.assertFalse(muse.cell_ext)
        qwen = kvstate.ModelSpec.from_metadata({
            **qwen_metadata(64, 4, 256, 64, (11, 11, 10, 0)), "qwen35.embedding_length": 5120,
            "qwen35.attention.head_count": 24})
        self.assertEqual(qwen.layers_of("full"), tuple(range(3, 64, 4)))
        self.assertEqual(len(qwen.layers_of("recurrent")), 48)
        # the live Qwen archive holds 16 F16 attention layers with 2048-byte rows
        self.assertEqual(2 * qwen.row_elements(3, "k"), 2048)
        self.assertTrue(qwen.cell_ext)
        self.assertEqual([name for name, _, _ in qwen.sections(0)], ["attn", "recurrent"])

    def test_dense_qwen3_geometry_from_metadata(self):
        # Qwen3-4B-Instruct-2507: 36 layers, 32 query / 8 KV heads of 128 (not n_embd / n_head = 80),
        # no rope.dimension_count (n_rot defaults to the head size), NEOX base 5e6
        spec = kvstate.ModelSpec.from_metadata({
            **qwen3_metadata(36, 32, 8, 128), "qwen3.embedding_length": 2560, "qwen3.context_length": 262144})
        self.assertEqual(spec.layer_types, ("full",) * 36)
        self.assertEqual((spec.head_shape(0, "k"), spec.head_shape(35, "v")), ((8, 128), (8, 128)))
        self.assertEqual((spec.rope_type, spec.n_rot, spec.rope_freq_base, spec.n_swa), ("neox", 128, 5e6, 0))
        self.assertEqual((spec.memory, spec.cell_ext, spec.recurrent_row_bytes), ("kv", False, None))
        # llama_kv_cache::state_write ignores the flags: one KV section either way
        for flags in (kvstate.FLAGS_NONE, kvstate.FLAGS_PARTIAL_ONLY):
            self.assertEqual(spec.sections(flags), [("kv", "kv", tuple(range(36)))])
        self.assertEqual(2 * spec.row_elements(0, "k"), 2048)
        with self.assertRaisesRegex(StateFormatError, "flags"):
            spec.sections(kvstate.FLAGS_ON_DEVICE)

    def test_ornith_geometry_with_a_nextn_block(self):
        # Ornith-1.5-9B (qwen35): 33 blocks of which 1 nextn -> 32 trunk layers, full attention every 4th
        spec = kvstate.ModelSpec.from_metadata({
            **qwen_metadata(33, 4, 256, 64, (11, 11, 10, 0)), "qwen35.nextn_predict_layers": 1,
            "qwen35.embedding_length": 4096, "qwen35.attention.head_count": 16, "qwen35.ssm.conv_kernel": 4,
            "qwen35.ssm.inner_size": 4096, "qwen35.ssm.state_size": 128, "qwen35.ssm.group_count": 16})
        self.assertEqual(len(spec.layer_types), 32)
        self.assertEqual(spec.layers_of("full"), tuple(range(3, 32, 4)))
        self.assertEqual(len(spec.layers_of("recurrent")), 24)
        self.assertEqual((spec.head_shape(3, "k"), spec.n_rot, spec.rope_sections), ((4, 256), 64, (11, 11, 10, 0)))
        # conv rows 4 * (4 - 1) * (4096 + 2 * 16 * 128), state rows 4 * 128 * 4096 bytes
        self.assertEqual(spec.recurrent_row_bytes, (98304, 2097152))

    def test_qwen3_refusals(self):
        with self.assertRaisesRegex(StateFormatError, "nextn"):
            kvstate.ModelSpec.from_metadata({**qwen3_metadata(), "qwen3.nextn_predict_layers": 1})
        # src/models/qwen3.cpp asserts n_rot == head size
        with self.assertRaisesRegex(StateFormatError, "whole head"):
            kvstate.ModelSpec.from_metadata(qwen3_metadata(n_rot=8))
        with self.assertRaisesRegex(StateFormatError, "RoPE scaling"):
            kvstate.ModelSpec.from_metadata({**qwen3_metadata(), "qwen3.rope.scaling.type": "yarn",
                                             "qwen3.rope.scaling.factor": 4.0})
        with self.assertRaisesRegex(StateFormatError, "NEOX"):
            kvstate.ModelSpec("qwen3", ("full",) * 2, (2, 2), (8, 8), (8, 8), rope_type="imrope", n_rot=8)
        with self.assertRaisesRegex(StateFormatError, "layers must be"):
            kvstate.ModelSpec("qwen3", ("full", "sliding"), (2, 2), (8, 8), (8, 8), rope_type="neox", n_rot=8)

    def test_qwen3_spec_from_a_fixture_gguf(self):
        from tiny_fixture import create_fixture
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen3.gguf"
            create_fixture(path, "qwen3")
            spec = kvstate.ModelSpec.from_gguf(path)
        self.assertEqual((spec.architecture, spec.layer_types, spec.n_head_kv, spec.head_dim_k),
                         ("qwen3", ("full",) * 4, (2,) * 4, (32,) * 4))
        self.assertEqual((spec.rope_type, spec.n_rot, spec.rope_freq_base), ("neox", 32, 500000.0))

    def test_unsupported_models_and_flags_are_refused(self):
        with self.assertRaisesRegex(StateFormatError, "unsupported architecture"):
            kvstate.ModelSpec.from_metadata({**muse_metadata(), "general.architecture": "llama"})
        with self.assertRaisesRegex(StateFormatError, "RoPE scaling"):
            kvstate.ModelSpec.from_metadata({**muse_metadata(), "muse-glimmer.rope.scaling.type": "yarn",
                                             "muse-glimmer.rope.scaling.factor": 4.0})
        spec = kvstate.ModelSpec.from_metadata(muse_metadata())
        with self.assertRaisesRegex(StateFormatError, "flags"):
            spec.sections(kvstate.FLAGS_ON_DEVICE)
        # Muse's iSWA cache would hold nextn layers in the base cache (no layer filter)
        with self.assertRaisesRegex(StateFormatError, "nextn"):
            kvstate.ModelSpec.from_metadata({**muse_metadata(), "muse-glimmer.nextn_predict_layers": 1})

    def test_sliding_layers_use_the_swa_head_sizes(self):
        spec = kvstate.ModelSpec.from_metadata({**muse_metadata(), "muse-glimmer.attention.key_length_swa": 4,
                                                "muse-glimmer.attention.value_length_swa": 6,
                                                "muse-glimmer.rope.dimension_count": 8})
        self.assertEqual([spec.head_shape(il, "k")[1] for il in range(8)], [4, 4, 4, 8, 4, 4, 4, 8])
        self.assertEqual([spec.head_shape(il, "v")[1] for il in range(8)], [6, 6, 6, 8, 6, 6, 6, 8])
        # n_rot_swa defaults to n_rot_full, which defaults to the full head size, not the sliding one
        self.assertEqual(spec.n_rot, 8)
        without = {k: v for k, v in muse_metadata(head_dim=8).items() if not k.endswith("rope.dimension_count")}
        spec = kvstate.ModelSpec.from_metadata({**without, "muse-glimmer.attention.key_length_swa": 16,
                                                "muse-glimmer.attention.value_length_swa": 16})
        self.assertEqual((spec.n_rot, spec.head_shape(0, "k"), spec.head_shape(3, "k")), (8, (2, 16), (2, 8)))
        # head sizes default to n_embd / n_head of layer 0
        bare = {k: v for k, v in muse_metadata().items() if "key_length" not in k and "value_length" not in k}
        self.assertEqual(kvstate.ModelSpec.from_metadata(bare).head_shape(3, "v"), (2, 32 // 4))


class MuseStateTests(unittest.TestCase):
    def setUp(self):
        self.spec = kvstate.ModelSpec.from_metadata(muse_metadata())
        self.rng = np.random.default_rng(5)
        self.layers = self.spec.layers_of("full", "sliding")

    def build(self, n=11, **options):
        k, v = random_kv(self.spec, self.layers, n, self.rng)
        tokens = self.rng.integers(0, 500, n)
        return kvstate.build_muse_state(self.spec, tokens, np.arange(n), k, v, **options), tokens, k, v

    def test_round_trip_and_logical_layout_for_every_dtype_and_v_layout(self):
        for dtype in ("f32", "f16"):
            for v_trans in (False, True):
                with self.subTest(dtype=dtype, v_trans=v_trans):
                    built, tokens, k, v = self.build(dtype=dtype, v_trans=v_trans)
                    data = built.to_bytes()
                    parsed = kvstate.parse_file(data, self.spec)
                    self.assertEqual(parsed.to_bytes(), data)
                    np.testing.assert_array_equal(parsed.tokens, tokens)
                    self.assertFalse(parsed.server_packed)
                    base, swa = parsed.state.part("base").stream(), parsed.state.part("swa").stream()
                    np.testing.assert_array_equal(base.positions, np.arange(11))
                    # a native save keeps pos > pos_max - n_swa in the SWA cache
                    np.testing.assert_array_equal(swa.positions, np.arange(7, 11))
                    self.assertEqual((base.v_trans, swa.v_trans), (v_trans, v_trans))
                    np_dtype = np.float16 if dtype == "f16" else np.float32
                    for il in self.spec.layers_of("full"):
                        np.testing.assert_array_equal(base.v[il], v[il].astype(np_dtype))
                        np.testing.assert_array_equal(base.k[il], k[il].astype(np_dtype))
                    for il in self.spec.layers_of("sliding"):
                        rows = np.arange(7, 11)
                        np.testing.assert_array_equal(swa.v[il], v[il][rows].astype(np_dtype))
                        rotated = kvstate.cache_keys(self.spec, il, k[il][rows], rows)
                        np.testing.assert_array_equal(swa.k[il], rotated.astype(np_dtype))
                        self.assertFalse(np.allclose(rotated[1:], k[il][rows][1:]))
                    size_v = 16 * (2 if dtype == "f16" else 4)
                    self.assertEqual(base.v[3].nbytes, 11 * size_v)

    def test_transposed_values_are_stored_element_major(self):
        built, _, _, v = self.build(n=5, dtype="f32", v_trans=True, swa_positions=[4])
        data = built.to_bytes()
        # base section: n_stream, cells, 5 x (pos, n_seq, seq), v_trans, n_layer, K (type, row, rows) x 2 layers
        offset = 12 + 4 * 5 + 4 + 4 + 5 * 12 + 8
        offset += 2 * (12 + 5 * 16 * 4)
        ggml_type, element, n_embd = struct.unpack_from("<iII", data, offset)
        self.assertEqual((ggml_type, element, n_embd), (0, 4, 16))
        first = np.frombuffer(data, "<f4", 5 * 16, offset + 12).reshape(16, 5)
        np.testing.assert_array_equal(first.T.reshape(5, 2, 8), v[3])

    def test_modified_layer_serializes_and_reparses(self):
        built, *_ = self.build(dtype="f16")
        parsed = kvstate.parse_file(built.to_bytes(), self.spec)
        stream = parsed.state.part("swa").stream()
        stream.v[1] = stream.v[1].copy()
        stream.v[1][0] += 1
        again = kvstate.parse_file(parsed.to_bytes(), self.spec)
        np.testing.assert_array_equal(again.state.part("swa").stream().v[1], stream.v[1])
        stream.v[1] = stream.v[1].astype(np.float32)
        with self.assertRaisesRegex(StateFormatError, "dtype changed"):
            parsed.to_bytes()

    def test_server_packed_tokens_blob_and_partial_checkpoint(self):
        built, tokens, *_ = self.build(server_tokens=True)
        words = built.token_words
        self.assertEqual(words[:3].tolist(), [-1, 1, 11])
        self.assertEqual(words[-1], 0)
        parsed = kvstate.parse_file(built.to_bytes(), self.spec)
        self.assertTrue(parsed.server_packed)
        np.testing.assert_array_equal(parsed.tokens, tokens)
        blob = kvstate.StateBlob(3, built.state)
        self.assertEqual(kvstate.parse_blob(blob.to_bytes(), self.spec).to_bytes(), blob.to_bytes())
        partial = kvstate.SequenceState(self.spec, kvstate.FLAGS_PARTIAL_ONLY, [built.state.part("swa")])
        data = kvstate.StateBlob(0, partial).to_bytes()
        reparsed = kvstate.parse_blob(data, self.spec, flags=kvstate.FLAGS_PARTIAL_ONLY)
        self.assertEqual([p.name for p in reparsed.state.parts], ["swa"])
        with self.assertRaises(StateFormatError):
            kvstate.parse_blob(data, self.spec)
        with self.assertRaisesRegex(StateFormatError, "magic"):
            kvstate.parse_blob(b"\0" * 8 + data[8:], self.spec)

    def test_malformed_files_are_refused(self):
        built, *_ = self.build(dtype="f16")
        data = built.to_bytes()
        n_words = len(built.token_words)
        payload = 12 + 4 * n_words
        cases = {
            "magic": b"xxxx" + data[4:],
            "version": data[:4] + struct.pack("<I", 2) + data[8:],
            "trailing": data + b"\0",
            "truncated": data[:-1],
            "stream count": data[:payload] + struct.pack("<I", 0) + data[payload + 4:],
        }
        for name, bad in cases.items():
            with self.subTest(name=name), self.assertRaises(StateFormatError):
                kvstate.parse_file(bad, self.spec)
        # n_seq_id of the first base cell, v_trans value, layer count, K type, K row size
        cell = payload + 8
        after_meta = cell + 11 * 12
        for offset, value, fragment in ((cell + 4, 2, "sequences"), (after_meta, 2, "v_trans"),
                                        (after_meta + 4, 3, "layers"), (after_meta + 8, 8, "ggml type"),
                                        (after_meta + 12, 17, "row size")):
            bad = bytearray(data)
            struct.pack_into("<I", bad, offset, value)
            with self.subTest(fragment=fragment), self.assertRaisesRegex(StateFormatError, fragment):
                kvstate.parse_file(bytes(bad), self.spec)
        other = kvstate.ModelSpec.from_metadata(muse_metadata(n_layer=12))
        with self.assertRaisesRegex(StateFormatError, "layers"):
            kvstate.parse_file(data, other)
        qwen = kvstate.ModelSpec.from_metadata(qwen_metadata())
        with self.assertRaises(StateFormatError):
            kvstate.parse_file(data, qwen)

    def test_builder_validation(self):
        k, v = random_kv(self.spec, self.layers, 6, self.rng)
        tokens = np.arange(6)
        build = kvstate.build_muse_state
        with self.assertRaisesRegex(StateFormatError, "tokens"):
            build(self.spec, tokens[:5], np.arange(6), k, v)
        with self.assertRaisesRegex(StateFormatError, "not among"):
            build(self.spec, tokens, np.arange(6), k, v, swa_positions=[5, 9])
        with self.assertRaisesRegex(StateFormatError, "at least one cell"):
            build(self.spec, tokens, np.arange(6), k, v, swa_positions=[])
        with self.assertRaisesRegex(StateFormatError, "same position"):
            build(self.spec, tokens, np.arange(6), k, v, swa_positions=[3, 4])
        with self.assertRaisesRegex(StateFormatError, "unique"):
            build(self.spec, tokens, np.asarray([0, 1, 2, 2, 4, 5]), k, v)
        with self.assertRaisesRegex(StateFormatError, "overflows"):
            build(self.spec, tokens, np.arange(6), k, {**v, 0: v[0] * 1e6})
        with self.assertRaisesRegex(StateFormatError, "missing"):
            build(self.spec, tokens, np.arange(6), {il: k[il] for il in (0, 1, 2)}, v)
        # sparse base cells and an SWA part reaching back beyond the window are allowed
        sparse = build(self.spec, tokens, np.asarray([0, 2, 3, 5]), {il: k[il][:4] for il in k},
                       {il: v[il][:4] for il in v}, swa_positions=[0, 2, 3, 5], dtype="f32")
        swa = kvstate.parse_file(sparse.to_bytes(), self.spec).state.part("swa").stream()
        np.testing.assert_array_equal(swa.positions, [0, 2, 3, 5])

    def test_serialization_refuses_sections_that_do_not_end_at_the_last_token(self):
        built, tokens, k, v = self.build(n=11, dtype="f32")
        self.assertIsNone(built.check_positions())
        base = built.state.part("base").stream()
        # a base cache past the SWA end is invisible to llama_memory_seq_pos_max after a restore
        base.positions = base.positions.copy()
        base.positions[-1] = 11
        with self.assertRaisesRegex(StateFormatError, "base section ends at position 11; with 11 tokens"):
            built.to_bytes()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(StateFormatError, "base section"):
                kvstate.write_file(Path(directory) / "bad.state", built)
            self.assertEqual(list(Path(directory).iterdir()), [])
        # deliberately inconsistent test files can still be written
        data = built.to_bytes(check_positions=False)
        self.assertEqual(kvstate.parse_file(data, self.spec).state.part("base").stream().positions[-1], 11)
        short = kvstate.StateFile(kvstate.token_words(tokens[:10]), self.build(n=11)[0].state)
        with self.assertRaisesRegex(StateFormatError, "section ends at position 10; with 10 tokens"):
            short.to_bytes()

    def test_write_new_never_overwrites(self):
        built, *_ = self.build()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slot.bin"
            kvstate.write_file(path, built)
            self.assertEqual(path.read_bytes(), built.to_bytes())
            with self.assertRaises(FileExistsError):
                kvstate.write_file(path, built)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["slot.bin"])
            self.assertEqual(kvstate.read_file(path, self.spec).to_bytes(), built.to_bytes())


class QwenStateTests(unittest.TestCase):
    def setUp(self):
        self.spec = kvstate.ModelSpec.from_metadata(qwen_metadata())
        self.rng = np.random.default_rng(9)

    def test_hybrid_round_trip_with_cell_ext_and_opaque_recurrent_rows(self):
        self.assertEqual(self.spec.recurrent_row_bytes, (4 * 3 * (8 + 4), 4 * 16))
        for dtype, v_trans in (("f16", False), ("f32", True)):
            with self.subTest(dtype=dtype, v_trans=v_trans):
                state, tokens, k, v = qwen_state(self.spec, 9, self.rng, dtype=dtype, v_trans=v_trans)
                file = kvstate.StateFile(kvstate.token_words(tokens, server_packed=True), state)
                data = file.to_bytes()
                parsed = kvstate.parse_file(data, self.spec)
                self.assertEqual(parsed.to_bytes(), data)
                attn = parsed.state.part("attn").stream()
                np.testing.assert_array_equal(attn.ext, np.stack([np.arange(9), np.arange(9), tokens], 1))
                np.testing.assert_array_equal(attn.v[3], v[3].astype(np.float16 if dtype == "f16" else np.float32))
                recurrent = parsed.state.part("recurrent")
                self.assertEqual((recurrent.layers, recurrent.positions.tolist(), recurrent.n_layer),
                                 ((0, 1, 2, 4, 5, 6), [8], 8))
                self.assertEqual(recurrent.nbytes, 6 * sum(self.spec.recurrent_row_bytes))

    def test_recurrent_layout_is_length_checked(self):
        state, *_ = qwen_state(self.spec, 4, self.rng)
        data = kvstate.StateFile(np.arange(4), state).to_bytes()
        smaller = kvstate.ModelSpec.from_metadata({**qwen_metadata(), "qwen35.ssm.state_size": 3})
        with self.assertRaisesRegex(StateFormatError, "row size"):
            kvstate.parse_file(data, smaller)
        with self.assertRaisesRegex(StateFormatError, "trailing|truncated"):
            kvstate.parse_file(data[:-3], self.spec)
        bad_rows = state.part("recurrent")
        bad_rows.r[0] = kvstate.OpaqueRows(0, bad_rows.r[0].row_bytes, b"\0")
        with self.assertRaisesRegex(StateFormatError, "bytes for"):
            state.to_bytes()

    def test_serialization_refuses_attention_and_recurrent_ends_that_disagree(self):
        state, tokens, *_ = qwen_state(self.spec, 6, self.rng)
        file = kvstate.StateFile(tokens, state)
        file.to_bytes()
        with self.assertRaisesRegex(StateFormatError, "attn section ends at position 5; with 5 tokens"):
            kvstate.StateFile(tokens[:5], state).to_bytes()
        recurrent = state.part("recurrent")
        recurrent.positions = np.asarray([4], dtype=np.int32)
        with self.assertRaisesRegex(StateFormatError, "recurrent state must sit at 5"):
            file.to_bytes()
        recurrent.positions = np.asarray([5], dtype=np.int32)
        # hybrid memory reports min(attention, recurrent) as pos_max, so a longer attention part hides there
        attn = state.part("attn").stream()
        attn.positions = attn.positions.copy()
        attn.positions[-1] = 7
        with self.assertRaisesRegex(StateFormatError, "attn section ends at position 7"):
            file.to_bytes()

    def test_partial_blob_holds_only_recurrent_state(self):
        state, *_ = qwen_state(self.spec, 3, self.rng, flags=kvstate.FLAGS_PARTIAL_ONLY)
        blob = kvstate.StateBlob(0, state).to_bytes()
        parsed = kvstate.parse_blob(blob, self.spec, flags=kvstate.FLAGS_PARTIAL_ONLY)
        self.assertEqual([p.name for p in parsed.state.parts], ["recurrent"])
        self.assertEqual(parsed.to_bytes(), blob)


class DenseStateTests(unittest.TestCase):
    def setUp(self):
        self.spec = kvstate.ModelSpec.from_metadata(qwen3_metadata())
        self.rng = np.random.default_rng(13)
        self.layers = self.spec.layers_of("full")

    def build(self, n=10, **options):
        k, v = random_kv(self.spec, self.layers, n, self.rng)
        tokens = self.rng.integers(0, 151936, n)
        return kvstate.build_dense_state(self.spec, tokens, np.arange(n), k, v, **options), tokens, k, v

    def test_round_trip_and_logical_layout_for_every_dtype_and_v_layout(self):
        for dtype in ("f32", "f16"):
            for v_trans in (False, True):
                with self.subTest(dtype=dtype, v_trans=v_trans):
                    built, tokens, k, v = self.build(dtype=dtype, v_trans=v_trans)
                    data = built.to_bytes()
                    parsed = kvstate.parse_file(data, self.spec)
                    self.assertEqual(parsed.to_bytes(), data)
                    np.testing.assert_array_equal(parsed.tokens, tokens)
                    self.assertEqual([p.name for p in parsed.state.parts], ["kv"])
                    stream = parsed.state.part("kv").stream()
                    self.assertEqual((stream.v_trans, stream.ext), (v_trans, None))
                    np.testing.assert_array_equal(stream.positions, np.arange(10))
                    np_dtype = np.float16 if dtype == "f16" else np.float32
                    for il in self.layers:
                        np.testing.assert_array_equal(stream.v[il], v[il].astype(np_dtype))
                        rotated = kvstate.cache_keys(self.spec, il, k[il], np.arange(10))
                        np.testing.assert_array_equal(stream.k[il], rotated.astype(np_dtype))
                        self.assertFalse(np.allclose(rotated[1:], k[il][1:]))
                    # meta: n_stream, cells, 10 x (pos, n_seq, seq id) with no cell ext, then v_trans, n_layer
                    payload = 12 + 4 * 10
                    self.assertEqual(struct.unpack_from("<II", data, payload), (1, 10))
                    self.assertEqual(struct.unpack_from("<iIi", data, payload + 8 + 12 * 9), (9, 1, 0))
                    self.assertEqual(struct.unpack_from("<II", data, payload + 8 + 12 * 10), (int(v_trans), 4))
                    self.assertEqual(kvstate.describe(parsed.state)["sections"][0]["streams"][0]["cells"], 10)

    def test_partial_flag_writes_the_whole_cache(self):
        built, *_ = self.build(dtype="f16")
        blob = kvstate.StateBlob(0, kvstate.SequenceState(self.spec, kvstate.FLAGS_PARTIAL_ONLY,
                                                          built.state.parts)).to_bytes()
        self.assertEqual(blob, kvstate.StateBlob(0, built.state).to_bytes())
        parsed = kvstate.parse_blob(blob, self.spec, flags=kvstate.FLAGS_PARTIAL_ONLY)
        self.assertEqual([p.name for p in parsed.state.parts], ["kv"])
        self.assertEqual(parsed.to_bytes(), blob)

    def test_builder_options_and_validation(self):
        k, v = random_kv(self.spec, self.layers, 6, self.rng)
        tokens = np.arange(6)
        build = kvstate.build_dense_state
        # cells are written in position order whatever the row order
        order = np.asarray([3, 0, 5, 1, 4, 2])
        shuffled = build(self.spec, tokens, order, {il: k[il][order] for il in k}, {il: v[il][order] for il in v},
                         dtype="f32")
        self.assertEqual(shuffled.to_bytes(), build(self.spec, tokens, np.arange(6), k, v, dtype="f32").to_bytes())
        # k_pre_rope=False stores K as given; server-packed token words; another stream
        raw = build(self.spec, tokens, np.arange(6), k, v, dtype="f32", k_pre_rope=False, server_tokens=True,
                    n_stream=2, stream=1)
        parsed = kvstate.parse_file(raw.to_bytes(), self.spec)
        self.assertTrue(parsed.server_packed)
        self.assertEqual(parsed.state.part("kv").streams[0], None)
        np.testing.assert_array_equal(parsed.state.part("kv").stream().k[2], k[2])
        with self.assertRaisesRegex(StateFormatError, "tokens"):
            build(self.spec, tokens[:5], np.arange(6), k, v)
        with self.assertRaisesRegex(StateFormatError, "strictly increasing"):
            build(self.spec, tokens, np.asarray([0, 1, 2, 2, 4, 5]), k, v)
        with self.assertRaisesRegex(StateFormatError, "missing"):
            build(self.spec, tokens, np.arange(6), {il: k[il] for il in (0, 1, 2)}, v)
        with self.assertRaisesRegex(StateFormatError, "rows"):
            build(self.spec, tokens, np.arange(6), {**k, 1: k[1][:5]}, v)
        with self.assertRaisesRegex(StateFormatError, "stream"):
            build(self.spec, tokens, np.arange(6), k, v, stream=1)
        with self.assertRaisesRegex(StateFormatError, "plain-KV"):
            build(kvstate.ModelSpec.from_metadata(muse_metadata()), tokens, np.arange(6), k, v)

    def test_malformed_and_foreign_files_are_refused(self):
        built, tokens, *_ = self.build(dtype="f16")
        data = built.to_bytes()
        after_meta = 12 + 4 * 10 + 8 + 12 * 10
        for offset, value, fragment in ((after_meta, 2, "v_trans"), (after_meta + 4, 5, "layers"),
                                        (12 + 4 * 10 + 8 + 4, 3, "sequences")):
            bad = bytearray(data)
            struct.pack_into("<I", bad, offset, value)
            with self.subTest(fragment=fragment), self.assertRaisesRegex(StateFormatError, fragment):
                kvstate.parse_file(bytes(bad), self.spec)
        with self.assertRaisesRegex(StateFormatError, "layers"):
            kvstate.parse_file(data, kvstate.ModelSpec.from_metadata(qwen3_metadata(n_layer=5)))
        with self.assertRaisesRegex(StateFormatError, "row size"):
            kvstate.parse_file(data, kvstate.ModelSpec.from_metadata(qwen3_metadata(heads_kv=1, heads=2)))
        for other in (kvstate.ModelSpec.from_metadata(muse_metadata()), kvstate.ModelSpec.from_metadata(qwen_metadata())):
            with self.assertRaises(StateFormatError):
                kvstate.parse_file(data, other)
        # decoding resumes at the token count, so the cache must end at the last token
        with self.assertRaisesRegex(StateFormatError, "kv section ends at position 9; with 9 tokens"):
            kvstate.StateFile(kvstate.token_words(tokens[:9]), built.state).to_bytes()


class TokenAndResumeTests(unittest.TestCase):
    def test_token_words(self):
        words = kvstate.token_words([5, 6, 7], server_packed=True)
        self.assertEqual(words.tolist(), [-1, 1, 3, 5, 6, 7, 0])
        self.assertEqual(kvstate.decode_token_words(words)[0].tolist(), [5, 6, 7])
        self.assertEqual(kvstate.decode_token_words([5, 6])[1], False)
        for bad, fragment in (([-1, 1, 1, 5, 2, 9, 9], "media"), ([-1, 2, 0, 0], "version"),
                              ([-1, 1, 1, 5, 0, 0], "trailing"), ([-1, 1, 5, 0], "truncated"), ([3, -2], "negative")):
            with self.subTest(bad=bad), self.assertRaisesRegex(StateFormatError, fragment):
                kvstate.decode_token_words(np.asarray(bad))

    def test_resume_file_round_trip_and_reader_checks(self):
        empty = kvstate.ResumeFile()
        self.assertEqual(empty.to_bytes(), struct.pack("<IIQQI", 0x51534C54, 1, 0, 0, 0))
        self.assertEqual(kvstate.parse_resume(empty.to_bytes()).checkpoints, [])
        resume = kvstate.ResumeFile(b"", b"spec", [kvstate.Checkpoint(40, 20, 39, b"x" * 50, b"", b"")])
        data = resume.to_bytes()
        self.assertEqual(kvstate.parse_resume(data), resume)
        with self.assertRaisesRegex(StateFormatError, "positions"):
            kvstate.parse_resume(data, n_prompt_tokens=30)
        with self.assertRaisesRegex(StateFormatError, "trailing"):
            kvstate.parse_resume(data + b"\0")
        with self.assertRaisesRegex(StateFormatError, "format"):
            kvstate.parse_resume(b"\0" * 8 + data[8:])


class RopeTests(unittest.TestCase):
    def test_norm_rotation_round_trip_and_pairing(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(40, 2, 8)).astype(np.float32)
        positions = np.arange(1000, 1040)
        y = kvstate.rope_norm(x, positions, n_rot=8, base=5e5)
        np.testing.assert_allclose(kvstate.rope_norm(y, positions, n_rot=8, base=5e5, inverse=True), x, atol=2e-6)
        # adjacent pairs keep their norm, position 0 is the identity
        np.testing.assert_allclose(np.hypot(y[..., 0::2], y[..., 1::2]), np.hypot(x[..., 0::2], x[..., 1::2]),
                                   rtol=1e-5)
        np.testing.assert_array_equal(kvstate.rope_norm(x[:1], [0], n_rot=8, base=5e5), x[:1])

    def test_qwen_partial_imrope_round_trip_with_real_geometry(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(64, 4, 256)).astype(np.float32)
        x /= np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True))
        positions = np.concatenate([np.arange(32), rng.integers(0, 200_000, 32)])
        sections = (11, 11, 10, 0)
        y = kvstate.imrope_text(x, positions, n_rot=64, base=1e7, sections=sections)
        np.testing.assert_array_equal(y[:, :, 64:], x[:, :, 64:])
        self.assertFalse(np.allclose(y[1:, :, :64], x[1:, :, :64]))
        # pairs (i, i + 32) rotate together and keep their norm
        np.testing.assert_allclose(np.hypot(y[..., :32], y[..., 32:64]), np.hypot(x[..., :32], x[..., 32:64]),
                                   rtol=1e-5)
        # with sections covering all 32 pairs no pair falls to the zero fourth position
        np.testing.assert_array_equal(y, kvstate.rope_neox(x, positions, n_rot=64, base=1e7))
        back = kvstate.imrope_text(y, positions, n_rot=64, base=1e7, sections=sections, inverse=True)
        np.testing.assert_allclose(back, x, atol=2e-5)
        spec = kvstate.ModelSpec.from_metadata({**qwen_metadata(64, 4, 256, 64, sections),
                                                "qwen35.attention.head_count": 24})
        np.testing.assert_array_equal(kvstate.cache_keys(spec, 3, x, positions), y)
        np.testing.assert_array_equal(kvstate.pre_rope_keys(spec, 3, y, positions), back)

    def test_qwen3_full_head_neox_round_trip_with_real_geometry(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(64, 8, 128)).astype(np.float32)
        positions = np.concatenate([np.arange(32), rng.integers(0, 262_144, 32)])
        spec = kvstate.ModelSpec.from_metadata(qwen3_metadata(36, 32, 8, 128))
        y = kvstate.cache_keys(spec, 7, x, positions)
        np.testing.assert_array_equal(y, kvstate.rope_neox(x, positions, n_rot=128, base=5e6))
        np.testing.assert_array_equal(y[0], x[0])
        # every dim rotates; pairs (i, i + 64) keep their norm
        self.assertTrue((np.abs(y[1:] - x[1:]).max(axis=(0, 1)) > 0).all())
        np.testing.assert_allclose(np.hypot(y[..., :64], y[..., 64:]), np.hypot(x[..., :64], x[..., 64:]), rtol=1e-5)
        np.testing.assert_allclose(kvstate.pre_rope_keys(spec, 7, y, positions), x, atol=2e-5)
        # NEOX pairs differ from NORMAL (adjacent) pairs
        self.assertFalse(np.allclose(y[1:], kvstate.rope_norm(x, positions, n_rot=128, base=5e6)[1:]))

    def test_theta_scale_is_libm_powf(self):
        import ctypes
        import ctypes.util
        libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
        libm.powf.restype, libm.powf.argtypes = ctypes.c_float, [ctypes.c_float, ctypes.c_float]
        for base, n_rot in ((5e5, 128), (1e7, 64), (5e5, 8), (1e7, 16), (1e4, 128), (5e5, 32)):
            with self.subTest(base=base, n_rot=n_rot):
                expected = np.float32(libm.powf(float(np.float32(base)), float(np.float32(-2.0) / np.float32(n_rot))))
                scale = kvstate.rope_theta_scale(base, n_rot)
                self.assertEqual(scale.view(np.uint32), expected.view(np.uint32))
                self.assertEqual(kvstate.rope_angles([1], n_rot, base)[0, 1], expected)
        # numpy's float32 power is 1 ulp off powf here (0.13335213 vs 0.13335215)
        self.assertEqual(kvstate.rope_theta_scale(1e7, 16), np.float32(0.13335215))

    def test_imrope_pairs_of_the_fourth_section_stay_unrotated_for_text(self):
        x = np.ones((3, 1, 32), dtype=np.float32)
        angles = kvstate.rope_angles([0, 5, 9], 32, 5e5, sections=(8, 4, 4, 0))
        self.assertEqual([i for i in range(16) if not angles[2, i]], [13, 14])
        y = kvstate.imrope_text(x, [0, 5, 9], n_rot=32, base=5e5, sections=(8, 4, 4, 0))
        np.testing.assert_array_equal(y[:, :, [13, 14, 29, 30]], 1)


def mla_metadata(architecture="deepseek2", n_layer=4, *, kv_lora=32, rope=16, heads=2, nextn=0, yarn=None,
                 context=256, **extra):
    """Synthetic MLA GGUF metadata: key_length kv_lora + rope, value_length kv_lora, one KV head.

    yarn: (factor, original context[, yarn_log_multiplier as stored, i.e. 0.1 * mscale_all_dim])."""
    p = architecture + "."
    meta = {"general.architecture": architecture, p + "block_count": n_layer + nextn, p + "context_length": context,
            p + "embedding_length": 64, p + "attention.head_count": heads, p + "attention.head_count_kv": 1,
            p + "attention.key_length": kv_lora + rope, p + "attention.value_length": kv_lora,
            p + "attention.key_length_mla": 32, p + "attention.value_length_mla": 16,
            p + "attention.kv_lora_rank": kv_lora, p + "rope.dimension_count": rope, p + "rope.freq_base": 10000.0,
            p + "attention.layer_norm_rms_epsilon": 1e-6}
    if nextn:
        meta[p + "nextn_predict_layers"] = nextn
    if yarn is not None:
        meta.update({p + "rope.scaling.type": "yarn", p + "rope.scaling.factor": yarn[0],
                     p + "rope.scaling.original_context_length": yarn[1]})
        if len(yarn) > 2:
            meta[p + "rope.scaling.yarn_log_multiplier"] = yarn[2]
    meta.update({p + key: value for key, value in extra.items()})
    return meta


def dsa_metadata(architecture="deepseek32", n_layer=4, types=None, **extra):
    meta = mla_metadata(architecture, n_layer, **{"attention.indexer.head_count": 2,
                                                  "attention.indexer.key_length": 64,
                                                  "attention.indexer.top_k": 8, **extra})
    if types is not None:
        meta[f"{architecture}.attention.indexer.types"] = list(types)
    return meta


def kda_metadata(architecture="kimi-linear", heads_kv=(1, 0, 0, 1), *, nextn=0, rope=16):
    meta = mla_metadata(architecture, len(heads_kv) - nextn, nextn=nextn, rope=rope)
    p = architecture + "."
    meta.update({p + "attention.head_count_kv": list(heads_kv), p + "attention.head_count": 2,
                 p + "ssm.conv_kernel": 4, p + "kda.head_dim": 8})
    return meta


def random_mla_stream(spec, layout, cells, rng, *, dtype="f16", v_trans=False):
    k = {il: rng.normal(size=(cells,) + spec.kv_shape(il, "k", layout.role)).astype(np.float32) for il in layout.layers}
    v = {il: rng.normal(size=(cells,) + spec.kv_shape(il, "v", layout.role)).astype(np.float32)
         for il in layout.layers if spec.kv_shape(il, "v", layout.role) is not None}
    return kvstate.build_kv_stream(spec, layout.layers, np.arange(cells), k, v, dtype=dtype, v_trans=v_trans,
                                   role=layout.role)


def opaque_rows(spec, layers, rng, cells=1):
    r_row, s_row = spec.recurrent_row_bytes
    return ({il: kvstate.OpaqueRows(0, r_row, rng.bytes(cells * r_row)) for il in layers},
            {il: kvstate.OpaqueRows(0, s_row, rng.bytes(cells * s_row)) for il in layers})


class MlaSpecTests(unittest.TestCase):
    def test_deepseek_v2_lite_geometry_and_yarn(self):
        # DeepSeek-V2-Lite: 27 layers, kv_lora 512, k_pe 64, YaRN x40 on 4096, mscale_all_dim 0.707
        spec = kvstate.ModelSpec.from_metadata(mla_metadata(
            "deepseek2", 27, kv_lora=512, rope=64, heads=16, yarn=(40.0, 4096, 0.0707), context=163840))
        self.assertEqual((spec.memory, spec.layer_types, spec.kv_lora), ("kv", ("full",) * 27, (512,) * 27))
        self.assertEqual(spec.sections(0), [("kv", "kv", tuple(range(27)))])
        self.assertEqual(spec.sections(kvstate.FLAGS_PARTIAL_ONLY), spec.sections(0))
        # K-only: the row is [c_kv | k_pe] = 576 elements, V is the c_kv view and is not stored
        self.assertEqual((spec.kv_shape(0, "k"), spec.kv_shape(0, "v"), spec.head_shape(0, "v")),
                         ((1, 576), None, (1, 512)))
        self.assertTrue(spec.k_only(5))
        self.assertEqual((spec.rope_freq_scale, spec.rope_ext_factor, spec.rope_n_ctx_orig),
                         (np.float32(1 / 40), 1.0, 4096))
        rope = spec.rope(3)
        self.assertEqual((rope.mode, rope.n_dims, rope.offset, rope.beta_fast, rope.beta_slow),
                         ("norm", 64, 512, 32.0, 1.0))
        # arch deepseek2 sets mscale = mscale_all_dim, so the k_pe magnitude is 1 up to float rounding
        self.assertAlmostEqual(float(rope.magnitude()), 1.0, places=6)

    def test_yarn_magnitude_per_architecture(self):
        f = 40.0
        expected = {"deepseek2": 1.0, "mistral4": (1 + 0.1 * np.log(f)) / (1 + 0.0707 * np.log(f)),
                    "deepseek32": (1 + 0.1 * np.log(f)) / (1 + 0.0707 * np.log(f))}
        for architecture, magnitude in expected.items():
            with self.subTest(architecture=architecture):
                meta = (dsa_metadata if architecture == "deepseek32" else mla_metadata)(
                    architecture, yarn=(f, 4096, 0.00707 * 10))
                spec = kvstate.ModelSpec.from_metadata(meta)
                self.assertAlmostEqual(float(spec.rope(0).magnitude()), magnitude, places=5)
        # m_all = 1 (DeepSeek V3 / V3.2, Kimi K2): magnitude 1 on every architecture
        spec = kvstate.ModelSpec.from_metadata(mla_metadata("mistral4", yarn=(f, 4096, 0.1)))
        self.assertAlmostEqual(float(spec.rope(0).magnitude()), 1.0, places=6)
        # loaders that do not read yarn_log_multiplier: 1 + 0.1 ln f
        spec = kvstate.ModelSpec.from_metadata(dsa_metadata("glm-dsa", yarn=(f, 4096, 0.0707)))
        self.assertAlmostEqual(float(spec.rope(0).magnitude()), 1 + 0.1 * np.log(f), places=5)
        # without YaRN nothing scales
        spec = kvstate.ModelSpec.from_metadata(mla_metadata("mistral4"))
        self.assertEqual((spec.rope_freq_scale, spec.rope_ext_factor, float(spec.rope(0).magnitude())), (1.0, 0.0, 1.0))

    def test_rope_runtime_mirrors_the_context_constructor_in_float32(self):
        f32, logf = np.float32, kvstate._logf
        meta = mla_metadata("deepseek2", yarn=(4.0, 64, 0.0707))
        rope = kvstate.rope_runtime(meta)
        factor = f32(4.0)
        m = f32(f32(0.0707) / f32(0.1))
        g = f32(f32(f32(0.1) * m) * logf(factor)) + f32(1)
        attn = f32(f32(g / g) * f32(f32(1) / f32(f32(1) + f32(f32(0.1) * logf(factor)))))
        self.assertEqual(f32(rope["attn_factor"]), attn)
        self.assertEqual((rope["freq_scale"], rope["ext_factor"], rope["n_ctx_orig"]), (0.25, 1.0, 64))
        # llama-server settings win over the GGUF: --rope-scaling none drops the scale and YaRN
        none = kvstate.rope_runtime(meta, runtime={"rope_scaling_type": "none"})
        self.assertEqual((none["freq_scale"], none["ext_factor"], none["attn_factor"]), (1.0, 0.0, 1.0))
        custom = kvstate.rope_runtime(mla_metadata(), runtime={"rope_scaling_type": "yarn", "rope_freq_scale": 0.125,
                                                               "yarn_orig_ctx": 32, "yarn_beta_fast": 16.0,
                                                               "rope_freq_base": 5e4})
        self.assertEqual((custom["freq_scale"], custom["n_ctx_orig"], custom["beta_fast"], custom["freq_base"]),
                         (0.125, 32, 16.0, 5e4))
        # --yarn-attn-factor is overwritten whenever ext_factor != 0 (src/llama-context.cpp:189-210)
        ignored = kvstate.rope_runtime(meta, runtime={"yarn_attn_factor": 3.0})
        self.assertEqual(ignored["attn_factor"], rope["attn_factor"])
        kept = kvstate.rope_runtime(mla_metadata(), runtime={"yarn_attn_factor": 3.0})
        self.assertEqual(kept["attn_factor"], 3.0)
        with self.assertRaisesRegex(StateFormatError, "unknown runtime"):
            kvstate.rope_runtime(meta, runtime={"rope_scale": 4})
        # None is an unset flag (an argv mapping may carry every key); numbers must be finite
        unset = dict.fromkeys(kvstate.RUNTIME_ROPE_KEYS)
        self.assertEqual(kvstate.rope_runtime(meta, runtime=unset), rope)
        self.assertEqual(kvstate.ModelSpec.from_metadata(meta, runtime=unset), kvstate.ModelSpec.from_metadata(meta))
        for key, value in (("yarn_ext_factor", float("nan")), ("yarn_attn_factor", float("inf")),
                           ("rope_freq_scale", "x")):
            with self.assertRaisesRegex(StateFormatError, "not a finite number"):
                kvstate.rope_runtime(meta, runtime={key: value})
        with self.assertRaisesRegex(StateFormatError, "longrope"):
            kvstate.ModelSpec.from_metadata(mla_metadata(**{"rope.scaling.type": "longrope"}))
        spec = kvstate.ModelSpec.from_metadata(mla_metadata(), runtime={"rope_scaling_type": "yarn",
                                                                        "rope_freq_scale": 0.25, "yarn_orig_ctx": 64})
        self.assertEqual((spec.rope_freq_scale, spec.rope_ext_factor, spec.rope_n_ctx_orig), (0.25, 1.0, 64))
        # the existing architectures keep refusing RoPE scaling, also as runtime settings
        with self.assertRaisesRegex(StateFormatError, "runtime RoPE"):
            kvstate.ModelSpec.from_metadata(qwen3_metadata(), runtime={"rope_freq_scale": 0.5})
        self.assertEqual(kvstate.ModelSpec.from_metadata(qwen3_metadata(), runtime=unset),
                         kvstate.ModelSpec.from_metadata(qwen3_metadata()))

    def test_nextn_layers_stay_in_the_plain_cache(self):
        # GLM-4.7-Flash style: deepseek2 with one nextn block; the cache keeps it (zero rows)
        spec = kvstate.ModelSpec.from_metadata(mla_metadata("deepseek2", 3, nextn=1))
        self.assertEqual((spec.layer_types, spec.n_layer_trunk), (("full",) * 3 + ("nextn",), 3))
        self.assertEqual(spec.sections(0), [("kv", "kv", (0, 1, 2, 3))])
        self.assertEqual((kvstate.nextn_layers(spec), spec.layers_of("full")), ((3,), (0, 1, 2)))
        # deepseek32 filters nextn layers out of both DSA sections
        spec = kvstate.ModelSpec.from_metadata({**dsa_metadata("deepseek32", 3), "deepseek32.block_count": 4,
                                                "deepseek32.nextn_predict_layers": 1})
        self.assertEqual(spec.sections(0), [("kv", "kv", (0, 1, 2)), ("indexer", "kv", (0, 1, 2))])
        with self.assertRaisesRegex(StateFormatError, "hy_v4 with nextn"):
            kvstate.ModelSpec.from_metadata(mla_metadata("hy_v4", 2, nextn=1))

    def test_dsa_layouts(self):
        spec = kvstate.ModelSpec.from_metadata(dsa_metadata("deepseek32", 3, yarn=(4.0, 64, 0.1)))
        ix = spec.indexer
        self.assertEqual((ix.layers, ix.head_size, ix.n_rot, ix.rope_type, ix.rope_offset, ix.hadamard, ix.top_k,
                          ix.norm_eps), ((0, 1, 2), 64, 16, "neox", 0, True, 8, 1e-6))
        # the DSA cache passes the flags to two plain caches, which ignore them
        for flags in (kvstate.FLAGS_NONE, kvstate.FLAGS_PARTIAL_ONLY):
            self.assertEqual([(lay.name, lay.role) for lay in spec.layouts(flags)],
                             [("kv", "attention"), ("indexer", "indexer")])
        self.assertEqual((spec.kv_shape(1, "k", "indexer"), spec.kv_shape(1, "v", "indexer")), ((1, 64), None))
        rope = spec.rope(2, "indexer")
        self.assertEqual((rope.mode, rope.offset, rope.freq_scale, rope.ext_factor), ("neox", 0, 0.25, 1.0))
        glm = kvstate.ModelSpec.from_metadata(dsa_metadata("glm-dsa", 4, types=[1, 0, 1, 0]))
        self.assertEqual((glm.indexer.layers, glm.indexer.rope_type, glm.indexer.norm_eps), ((0, 2), "norm", 0.0))
        # GLM-5/5.1 without types: every layer owns an indexer; GLM-5.2's built-in pattern is not modelled
        self.assertEqual(kvstate.ModelSpec.from_metadata(dsa_metadata("glm-dsa", 4)).indexer.layers, (0, 1, 2, 3))
        with self.assertRaisesRegex(StateFormatError, "indexer.types"):
            kvstate.ModelSpec.from_metadata(dsa_metadata("glm-dsa", 4, context=1 << 20))
        # hy_v4: DSA only with an indexer; [nope | rope] indexer keys, no Hadamard, RMS eps
        hy = kvstate.ModelSpec.from_metadata(dsa_metadata("hy_v4", 2, types=[1, 0]))
        self.assertEqual((hy.memory, hy.indexer.layers, hy.indexer.rope_offset, hy.indexer.hadamard,
                          hy.indexer.norm_eps), ("dsa", (0,), 48, False, np.float32(1e-6)))
        self.assertEqual(kvstate.ModelSpec.from_metadata(mla_metadata("hy_v4", 2)).memory, "kv")
        with self.assertRaisesRegex(StateFormatError, "layer 0"):
            kvstate.ModelSpec.from_metadata(dsa_metadata("hy_v4", 2, types=[0, 1]))
        with self.assertRaisesRegex(StateFormatError, "power-of-two"):
            kvstate.ModelSpec.from_metadata(dsa_metadata("deepseek32", 2, **{"attention.indexer.key_length": 96}))

    def test_dots3note_dsa_iswa_layout(self):
        meta = dsa_metadata("dots3note", 4, types=[1, 0, 1, 0], **{
            "attention.sliding_window": 8, "attention.sliding_window_pattern": [0, 1, 0, 1],
            "attention.kv_lora_rank_swa": 24, "attention.key_length_swa": 40, "attention.value_length_swa": 24,
            "rope.freq_base_swa": 500.0})
        spec = kvstate.ModelSpec.from_metadata(meta)
        self.assertEqual((spec.layer_types, spec.kv_lora, spec.n_swa), (("full", "sliding") * 2, (32, 24) * 2, 8))
        self.assertEqual([(lay.name, lay.layers, lay.sliding) for lay in spec.layouts(0)],
                         [("base", (0, 2), False), ("indexer", (0, 2), False), ("swa", (1, 3), True)])
        self.assertEqual(spec.sections(kvstate.FLAGS_PARTIAL_ONLY), [("swa", "kv", (1, 3))])
        self.assertEqual((spec.kv_shape(1, "k"), spec.rope(1).base, spec.rope(1).offset, spec.rope(0).base),
                         ((1, 40), 500.0, 24, 10000.0))
        # the sliding V view is the 24-wide sliding latent whatever value_length_swa says
        self.assertEqual(kvstate.ModelSpec.from_metadata({**meta, "dots3note.attention.value_length_swa": 96}), spec)
        with self.assertRaisesRegex(StateFormatError, "sliding_window_pattern"):
            kvstate.ModelSpec.from_metadata({**meta, "dots3note.attention.sliding_window_pattern": 2})

    def test_hybrid_k_only_layouts(self):
        spec = kvstate.ModelSpec.from_metadata(kda_metadata("kimi-linear"))
        self.assertEqual((spec.layer_types, spec.memory), (("full", "recurrent", "recurrent", "full"), "hybrid"))
        self.assertEqual(spec.sections(0), [("attn", "kv", (0, 3)), ("recurrent", "recurrent", (1, 2))])
        # KDA rows: R = 3 * (conv - 1) * n_head * head_dim, S = head_dim^2 * n_head (n_head of layer 0)
        self.assertEqual(spec.recurrent_row_bytes, (4 * 3 * 3 * 2 * 8, 4 * 8 * 8 * 2))
        self.assertIsNone(spec.rope(0))          # NoPE MLA: k_pe stored unrotated
        x = np.ones((2, 1, 48), np.float32)
        np.testing.assert_array_equal(kvstate.cache_keys(spec, 0, x, [0, 7]), x)
        # bailingmoe3 (RoPE MLA + KDA) keeps its nextn layer in the attention section
        spec = kvstate.ModelSpec.from_metadata(kda_metadata("bailingmoe3", (1, 0, 1, 1), nextn=1))
        self.assertEqual(spec.layer_types, ("full", "recurrent", "full", "nextn"))
        self.assertEqual(spec.sections(0), [("attn", "kv", (0, 2, 3)), ("recurrent", "recurrent", (1,))])
        self.assertEqual((spec.n_layer_trunk, spec.rope(2).offset), (3, 32))

    def test_value_length_plays_no_part_in_mla_specs(self):
        # an MLA cache stores no V; the kimi-linear and bailingmoe3 converters leave the base
        # class's value_length (head_dim, or nothing: n_embd / n_head) where DeepSeek's write kv_lora
        for architecture in ("deepseek2", "hy_v4", "kimi-k3", "kimi-linear", "bailingmoe3"):
            meta = kda_metadata(architecture) if architecture in kvstate.RECURRENT_FAMILY else mla_metadata(architecture)
            key = f"{architecture}.attention.value_length"
            reference = kvstate.ModelSpec.from_metadata(meta)
            for value in (None, 72, 16):
                with self.subTest(architecture=architecture, value_length=value):
                    edited = {k: v for k, v in meta.items() if k != key}
                    if value is not None:
                        edited[key] = value
                    spec = kvstate.ModelSpec.from_metadata(edited)
                    self.assertEqual(spec, reference)
                    for il in spec.layers_of("full"):
                        self.assertEqual((spec.head_shape(il, "v"), spec.kv_shape(il, "v")), ((1, 32), None))

    def test_mla_refusals(self):
        no_mla = {k: v for k, v in mla_metadata().items() if "length_mla" not in k}
        with self.assertRaisesRegex(StateFormatError, "expanded MHA"):
            kvstate.ModelSpec.from_metadata(no_mla)
        with self.assertRaisesRegex(StateFormatError, "one KV head"):
            kvstate.ModelSpec.from_metadata({**mla_metadata(), "deepseek2.attention.head_count_kv": 2})
        with self.assertRaisesRegex(StateFormatError, "V view is c_kv"):
            kvstate.ModelSpec("deepseek2", ("full",), (1,), (48,), (16,), rope_type="norm", n_rot=16, kv_lora=(32,))
        with self.assertRaisesRegex(StateFormatError, "k_pe has"):
            kvstate.ModelSpec.from_metadata({**mla_metadata(), "deepseek2.rope.dimension_count": 8})
        with self.assertRaisesRegex(StateFormatError, "kv_lora"):
            kvstate.ModelSpec("deepseek2", ("full",), (1,), (48,), (32,), rope_type="norm", n_rot=16)


class MlaStateTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)

    def round_trip(self, spec, parts, tokens, flags=kvstate.FLAGS_NONE):
        file = kvstate.StateFile(kvstate.token_words(tokens), kvstate.SequenceState(spec, flags, parts))
        data = file.to_bytes()
        parsed = kvstate.parse_file(data, spec)
        self.assertEqual(parsed.to_bytes(), data)
        return parsed, data

    def test_k_only_section_has_no_v_blocks_and_keeps_v_trans(self):
        spec = kvstate.ModelSpec.from_metadata(mla_metadata("deepseek2", 2, nextn=1))
        layout = spec.layout("kv")
        for dtype, v_trans in (("f16", False), ("f32", True)):
            with self.subTest(dtype=dtype, v_trans=v_trans):
                k = {il: self.rng.normal(size=(5, 1, 48)).astype(np.float32) for il in (0, 1)}
                stream = kvstate.build_kv_stream(spec, layout.layers, np.arange(5), k, None, dtype=dtype,
                                                 v_trans=v_trans)
                # the nextn layer was zero-filled, as a native save holds it
                self.assertFalse(stream.k[2].any())
                parsed, data = self.round_trip(spec, [kvstate.KVCache("kv", layout.layers, [stream])], np.arange(5))
                item = 2 if dtype == "f16" else 4
                # n_stream, cells, 5 x (pos, n_seq, seq), v_trans, n_layer, 3 x (type, row size, rows), nothing else
                self.assertEqual(len(data), 12 + 4 * 5 + 8 + 5 * 12 + 8 + 3 * (12 + 5 * 48 * item))
                st = parsed.state.part("kv").stream()
                self.assertEqual((st.v_trans, st.v, sorted(st.k)), (v_trans, {}, [0, 1, 2]))
                np.testing.assert_array_equal(kvstate.value_rows(spec, st, 1), st.k[1][..., :32])
        with self.assertRaisesRegex(StateFormatError, "K-only"):
            kvstate.build_kv_stream(spec, (0,), np.arange(2), {0: np.zeros((2, 1, 48))}, {0: np.zeros((2, 1, 32))})
        st.v[0] = st.k[0][..., :32]
        with self.assertRaisesRegex(StateFormatError, "K-only layers carry no V"):
            parsed.to_bytes()

    def test_dsa_and_dsa_iswa_states_round_trip(self):
        spec = kvstate.ModelSpec.from_metadata(dsa_metadata("glm-dsa", 4, types=[1, 0, 1, 0]))
        parts = [kvstate.KVCache(lay.name, lay.layers, [random_mla_stream(spec, lay, 6, self.rng)])
                 for lay in spec.layouts(0)]
        parsed, data = self.round_trip(spec, parts, np.arange(6))
        self.assertEqual([(p.name, p.layers) for p in parsed.state.parts], [("kv", (0, 1, 2, 3)), ("indexer", (0, 2))])
        self.assertEqual(parsed.state.part("indexer").stream().k[2].shape, (6, 1, 64))
        # a stale indexer end is caught like any other section end
        parsed.state.part("indexer").stream().positions[-1] = 9
        with self.assertRaisesRegex(StateFormatError, "indexer section ends at position 9"):
            parsed.to_bytes()
        dots = kvstate.ModelSpec.from_metadata(dsa_metadata("dots3note", 4, types=[1, 0, 1, 0], **{
            "attention.sliding_window": 8, "attention.sliding_window_pattern": [0, 1, 0, 1],
            "attention.kv_lora_rank_swa": 24, "attention.key_length_swa": 40, "attention.value_length_swa": 24,
            "rope.freq_base_swa": 500.0}))
        parts = [kvstate.KVCache(lay.name, lay.layers, [random_mla_stream(dots, lay, 6, self.rng, dtype="f32")])
                 for lay in dots.layouts(0)]
        parsed, _ = self.round_trip(dots, parts, np.arange(6))
        self.assertEqual(parsed.state.part("swa").stream().k[3].shape, (6, 1, 40))
        blob = kvstate.StateBlob(0, kvstate.SequenceState(dots, kvstate.FLAGS_PARTIAL_ONLY, parts[2:])).to_bytes()
        self.assertEqual([p.name for p in kvstate.parse_blob(blob, dots, flags=kvstate.FLAGS_PARTIAL_ONLY).state.parts],
                         ["swa"])

    def test_hybrid_k_only_state_and_recurrent_reset(self):
        spec = kvstate.ModelSpec.from_metadata(kda_metadata("bailingmoe3", (1, 0, 1, 1), nextn=1))
        attn = spec.layout("attn")
        stream = random_mla_stream(spec, attn, 4, self.rng)
        r, s = opaque_rows(spec, (1,), self.rng)
        parts = [kvstate.KVCache("attn", attn.layers, [stream]),
                 kvstate.RecurrentState("recurrent", (1,), np.asarray([3], np.int32), ((),), 3, r, s)]
        parsed, _ = self.round_trip(spec, parts, np.arange(4))
        recurrent = parsed.state.part("recurrent")
        self.assertEqual((recurrent.n_layer, recurrent.r[1].row_bytes), (3, spec.recurrent_row_bytes[0]))
        reset = kvstate.recurrent_reset(recurrent, 3)
        self.assertEqual((reset.positions.tolist(), reset.r[1].data, reset.s[1].row_bytes),
                         ([3], bytes(spec.recurrent_row_bytes[0]), spec.recurrent_row_bytes[1]))
        layout = kvstate.template_layout(parsed)
        self.assertEqual([(e["name"], e["role"], e["layers"]) for e in layout],
                         [("attn", "attention", (0, 2, 3)), ("recurrent", "recurrent", (1,))])
        self.assertEqual((layout[0]["dtype"], layout[0]["v_layers"], layout[0]["nextn_layers"], layout[1]["n_layer"]),
                         ("f16", (), (3,), 3))
        # the recurrent n_layer is the trunk count, not the cache's layer count
        bad = kvstate.SequenceState(spec, kvstate.FLAGS_NONE, [parts[0], kvstate.RecurrentState(
            "recurrent", (1,), np.asarray([3], np.int32), ((),), 4, r, s)])
        with self.assertRaisesRegex(StateFormatError, "records 4 model layers"):
            kvstate.parse_file(kvstate.StateFile(np.arange(4), bad).to_bytes(), spec)

    def test_build_dense_state_for_mla_and_dsa(self):
        spec = kvstate.ModelSpec.from_metadata(dsa_metadata("deepseek32", 2, yarn=(4.0, 64, 0.1)))
        positions = np.arange(7)
        k = {il: self.rng.normal(size=(7, 1, 48)).astype(np.float32) for il in (0, 1)}
        ik = {il: self.rng.normal(size=(7, 1, 64)).astype(np.float32) for il in (0, 1)}
        built = kvstate.build_dense_state(spec, np.arange(7), positions, k, indexer_k=ik, dtype="f32")
        parsed = kvstate.parse_file(built.to_bytes(), spec)
        st, ix = parsed.state.part("kv").stream(), parsed.state.part("indexer").stream()
        np.testing.assert_array_equal(st.k[1], kvstate.cache_keys(spec, 1, k[1], positions))
        np.testing.assert_array_equal(ix.k[0], kvstate.cache_keys(spec, 0, ik[0], positions, role="indexer"))
        np.testing.assert_allclose(kvstate.pre_rope_keys(spec, 0, ix.k[0], positions, role="indexer"), ik[0], atol=2e-6)
        # c_kv passes through; only the k_pe tail rotates
        np.testing.assert_array_equal(st.k[0][..., :32], k[0][..., :32])
        zeros = kvstate.build_dense_state(spec, np.arange(7), positions, k, indexer_k="zeros", dtype="f16")
        self.assertFalse(kvstate.parse_file(zeros.to_bytes(), spec).state.part("indexer").stream().k[1].any())
        with self.assertRaisesRegex(StateFormatError, "indexer_k"):
            kvstate.build_dense_state(spec, np.arange(7), positions, k)
        with self.assertRaisesRegex(StateFormatError, "plain-KV"):
            kvstate.build_dense_state(kvstate.ModelSpec.from_metadata(kda_metadata()), np.arange(7), positions, k)


class RecurrentRowTests(unittest.TestCase):
    def test_family_row_sizes(self):
        meta = {"general.architecture": "x", "x.embedding_length": 16, "x.attention.head_count": [4, 4],
                "x.ssm.conv_kernel": 4, "x.ssm.inner_size": 32, "x.ssm.state_size": 8, "x.ssm.group_count": 2,
                "x.kda.head_dim": 4, "x.wkv.head_size": 4, "x.shortconv.l_cache": 3}
        self.assertEqual(kvstate.recurrent_row_floats(meta, "ssm"), (3 * (32 + 2 * 2 * 8), 8 * 32))
        self.assertEqual(kvstate.recurrent_row_floats(meta, "kda"), (3 * 3 * 4 * 4, 4 * 4 * 4))
        self.assertEqual(kvstate.recurrent_row_floats({**meta, "x.ssm.conv_kernel": 0}, "kda")[0], 3 * 3 * 4 * 4)
        self.assertEqual(kvstate.recurrent_row_floats(meta, "rwkv"), (2 * 16, 16 * 4))
        self.assertEqual(kvstate.recurrent_row_floats(meta, "shortconv"), (16 * 2, 0))
        self.assertEqual(kvstate.recurrent_row_floats(meta, "lightning"), (0, 4 * 4 * 4))
        with self.assertRaisesRegex(StateFormatError, "unknown recurrent family"):
            kvstate.recurrent_row_floats(meta, "mamba3")
        ple = {**meta, "x.ple.ngram_size": 3, "x.ple.heads_per_ngram": 2, "x.ple.conv_kernel": 4,
               "x.hyper_connection.count": 4, "x.ple.layers": [0]}
        self.assertEqual(kvstate.ple_rows_from_metadata(ple), (3 * 3 * 4 * 16, (0,)))
        # ple_conv_state includes hc_mult: without hyper-connections there is no P row
        self.assertEqual(kvstate.ple_rows_from_metadata({**ple, "x.hyper_connection.count": 0}), (0, ()))

    def test_zero_byte_and_ple_rows(self):
        rng = np.random.default_rng(23)
        base = kvstate.ModelSpec.from_metadata(qwen_metadata())
        recurrent = base.layers_of("recurrent")
        # a family with 0-byte R rows (MiniMax-01) and a PLE row on the first recurrent layer (qwen4exp)
        spec = dataclasses.replace(base, recurrent_row_bytes=(0, 64), recurrent_p_bytes=24, ple_layers=(recurrent[0],))
        r = {il: kvstate.OpaqueRows(0, 0, b"") for il in recurrent}
        s = {il: kvstate.OpaqueRows(0, 64, rng.bytes(64)) for il in recurrent}
        p = {recurrent[0]: kvstate.OpaqueRows(None, 24, rng.bytes(24))}
        part = kvstate.RecurrentState("recurrent", recurrent, np.asarray([2], np.int32), ((),), 8, r, s, p)
        blob = kvstate.StateBlob(0, kvstate.SequenceState(spec, kvstate.FLAGS_PARTIAL_ONLY, [part])).to_bytes()
        # 4 + cell (8) + s_trans, n_layer (8) + 6 R headers (12, 0-byte rows) + P (8 + 24) + 6 S (12 + 64)
        self.assertEqual(len(blob), 8 + 4 + 8 + 8 + 6 * 12 + 8 + 24 + 6 * (12 + 64))
        parsed = kvstate.parse_blob(blob, spec, flags=kvstate.FLAGS_PARTIAL_ONLY)
        self.assertEqual(parsed.to_bytes(), blob)
        self.assertEqual(parsed.state.part("recurrent").p[recurrent[0]].data, p[recurrent[0]].data)
        reset = kvstate.recurrent_reset(parsed.state.part("recurrent"), 5)
        self.assertEqual((reset.p[recurrent[0]].data, reset.r[0].data), (bytes(24), b""))
        # 0-byte rows are refused when the spec does not expect them
        with self.assertRaisesRegex(StateFormatError, "row size 0"):
            kvstate.parse_blob(blob, dataclasses.replace(spec, recurrent_row_bytes=None),
                               flags=kvstate.FLAGS_PARTIAL_ONLY)
        with self.assertRaisesRegex(StateFormatError, "PLE rows"):
            kvstate.SequenceState(spec, kvstate.FLAGS_PARTIAL_ONLY, [dataclasses.replace(part, p={})]).to_bytes()
        with self.assertRaisesRegex(StateFormatError, "PLE rows"):
            dataclasses.replace(spec, ple_layers=(3,))


class MlaRopeTests(unittest.TestCase):
    def test_yarn_corr_dims_and_ramp(self):
        # DeepSeek-V2: 64 rotated dims, base 1e4, original context 4096, beta 32 / 1
        low, high = kvstate.yarn_corr_dims(64, 4096, 10000.0)
        expected = [64 * np.log(4096 / (b * 2 * np.pi)) / (2 * np.log(10000.0)) for b in (32.0, 1.0)]
        self.assertEqual((float(low), float(high)), (max(0.0, np.floor(expected[0])), min(63.0, np.ceil(expected[1]))))
        rope = kvstate.Rope("norm", 64, 10000.0, freq_scale=0.025, ext_factor=1.0, n_ctx_orig=4096)
        plain = kvstate.Rope("norm", 64, 10000.0)
        scaled = kvstate.Rope("norm", 64, 10000.0, freq_scale=0.025)
        positions = [3, 1000]
        cos, sin = rope.cos_sin(positions)
        m = rope.magnitude()
        angle = np.arctan2(sin, cos)
        # pairs below the low correction dim keep the original (extrapolated) angle,
        # pairs past the high one are fully interpolated (freq_scale * theta)
        c0, s0 = plain.cos_sin(positions)
        c1, s1 = scaled.cos_sin(positions)
        np.testing.assert_allclose(angle[:, :int(low)], np.arctan2(s0, c0)[:, :int(low)], atol=1e-6)
        np.testing.assert_allclose(angle[:, int(high) + 1:], np.arctan2(s1, c1)[:, int(high) + 1:], atol=1e-6)
        np.testing.assert_allclose(np.hypot(cos, sin), m, rtol=1e-6)
        self.assertAlmostEqual(float(m), 1 + 0.1 * np.log(40), places=6)

    def test_offset_window_rotation_and_inverse_with_magnitude(self):
        rng = np.random.default_rng(29)
        x = rng.normal(size=(20, 3, 48)).astype(np.float32)
        positions = rng.integers(0, 5000, 20)
        rope = kvstate.Rope("norm", 16, 10000.0, offset=32, freq_scale=0.25, ext_factor=1.0, attn_factor=0.8,
                            n_ctx_orig=64)
        y = rope.apply(x, positions)
        np.testing.assert_array_equal(y[..., :32], x[..., :32])
        # the window equals a plain rotation of the k_pe slice
        np.testing.assert_array_equal(y[..., 32:], dataclasses.replace(rope, offset=0).apply(x[..., 32:], positions))
        m = float(rope.magnitude())
        np.testing.assert_allclose(np.hypot(y[..., 32::2], y[..., 33::2]), m * np.hypot(x[..., 32::2], x[..., 33::2]),
                                   rtol=2e-6)
        np.testing.assert_allclose(rope.invert(y, positions), x, atol=3e-6)
        neox = kvstate.Rope("neox", 16, 10000.0, offset=8)
        z = neox.apply(x[..., :32], positions)
        np.testing.assert_array_equal(z[..., :8], x[..., :8])
        np.testing.assert_array_equal(z[..., 24:], x[..., 24:32])
        # plain parameters reproduce the old helpers bit for bit
        np.testing.assert_array_equal(kvstate.Rope("norm", 16, 5e5).apply(x[..., :16], positions),
                                      kvstate.rope_norm(x[..., :16], positions, n_rot=16, base=5e5))
        with self.assertRaisesRegex(ValueError, "offset"):
            rope.apply(x[..., :40], positions)

    def test_hadamard_and_indexer_round_trip(self):
        for n in (64, 128):
            h = kvstate.hadamard(n)
            self.assertEqual(h.dtype, np.float32)
            np.testing.assert_array_equal(h, h.T)
            np.testing.assert_array_equal(np.abs(h), np.float32(1.0 / float(np.sqrt(np.float32(n)))))
            np.testing.assert_allclose(h.astype(np.float64) @ h, np.eye(n), atol=1e-6)
        with self.assertRaises(ValueError):
            kvstate.hadamard(96)
        spec = kvstate.ModelSpec.from_metadata(dsa_metadata("deepseek32", 2))
        rng = np.random.default_rng(31)
        x = rng.normal(size=(9, 1, 64)).astype(np.float32)
        positions = np.arange(100, 109)
        y = kvstate.cache_keys(spec, 1, x, positions, role="indexer")
        rotated = spec.rope(1, "indexer").apply(x, positions)
        np.testing.assert_allclose(y, rotated @ kvstate.hadamard(64).T.astype(np.float64), atol=1e-6)
        np.testing.assert_allclose(kvstate.pre_rope_keys(spec, 1, y, positions, role="indexer"), x, atol=3e-6)
        with self.assertRaises(ValueError):
            kvstate.cache_keys(kvstate.ModelSpec.from_metadata(dsa_metadata("glm-dsa", 2, types=[1, 0])), 1, x,
                               positions, role="indexer")

    def test_mla_cache_keys_rotate_only_k_pe_and_invert(self):
        spec = kvstate.ModelSpec.from_metadata(mla_metadata("deepseek2", 2, yarn=(4.0, 64, 0.0707)))
        rng = np.random.default_rng(37)
        x = rng.normal(size=(12, 1, 48)).astype(np.float32)
        positions = np.arange(500, 512)
        y = kvstate.cache_keys(spec, 0, x, positions)
        np.testing.assert_array_equal(y[..., :32], x[..., :32])
        self.assertFalse(np.allclose(y[..., 32:], x[..., 32:]))
        np.testing.assert_allclose(kvstate.pre_rope_keys(spec, 0, y, positions), x, atol=2e-6)
        c_kv, k_pe = kvstate.mla_split(spec, 0, y)
        self.assertEqual((c_kv.shape, k_pe.shape), ((12, 1, 32), (12, 1, 16)))
        with self.assertRaises(ValueError):
            kvstate.mla_split(kvstate.ModelSpec.from_metadata(qwen3_metadata()), 0, y)

    def test_rms_shell_renormalization(self):
        rng = np.random.default_rng(41)
        g = (1 + 0.2 * rng.normal(size=32)).astype(np.float32)
        x = rng.normal(size=(10, 1, 32))
        native = (x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-6) * g).astype(np.float32)
        np.testing.assert_allclose(kvstate.mla_renormalize_latent(native, g), native, rtol=1e-5, atol=1e-6)
        decoded = 2.5 * native + 0.01 * rng.normal(size=native.shape)
        fixed = kvstate.mla_renormalize_latent(decoded, g)
        np.testing.assert_allclose(np.sqrt(np.mean((fixed / g) ** 2, axis=-1)), 1.0, rtol=1e-5)
        # a zero weight element stays zero; the RMS is taken over the other elements
        g0 = g.copy()
        g0[3] = 0
        fixed = kvstate.mla_renormalize_latent(decoded, g0)
        self.assertFalse(fixed[..., 3].any())
        kept = np.arange(32) != 3
        np.testing.assert_allclose(np.sqrt(np.mean((fixed[..., kept] / g0[kept]) ** 2, axis=-1)), 1.0, rtol=1e-5)
        # the model's RMS eps (attention.layer_norm_rms_epsilon): x * g comes back as RMSNorm_eps(x) * g
        loose = (x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-2) * g).astype(np.float32)
        np.testing.assert_allclose(kvstate.mla_renormalize_latent(x * g, g, eps=1e-2), loose, rtol=1e-5, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "zero everywhere"):
            kvstate.mla_renormalize_latent(decoded, np.zeros(32))
        with self.assertRaisesRegex(ValueError, "width"):
            kvstate.mla_renormalize_latent(decoded, g[:16])



def hybrid_metadata(architecture, *, remove=(), **extra):
    """Synthetic metadata of a plain-attention hybrid or recurrent-only model: four layers of 32,
    GQA heads 2/1 of 16, NEOX-style n_rot 8, a small SSM; `extra` overrides keys (no prefix)."""
    p = architecture + "."
    meta = {"general.architecture": architecture, p + "block_count": 4, p + "embedding_length": 32,
            p + "context_length": 256, p + "feed_forward_length": 64, p + "attention.head_count": 2,
            p + "attention.head_count_kv": 1, p + "attention.key_length": 16, p + "attention.value_length": 16,
            p + "rope.dimension_count": 8, p + "rope.freq_base": 1e6, p + "ssm.conv_kernel": 4,
            p + "ssm.inner_size": 8, p + "ssm.state_size": 2, p + "ssm.group_count": 1, p + "ssm.time_step_rank": 1}
    meta.update({p + key: value for key, value in extra.items()})
    return {key: value for key, value in meta.items() if key[len(p):] not in remove}


QWEN4EXP = {"rope.dimension_sections": [2, 1, 1, 0], "attention.indexer.head_count": 2,
            "attention.indexer.key_length": 8, "attention.indexer.top_k": 4, "ple.layers": [0], "ple.ngram_size": 3,
            "ple.heads_per_ngram": 2, "ple.conv_kernel": 4, "hyper_connection.count": 2}
# architecture -> (metadata overrides, layer kinds, memory, sections at flags 0, rope type)
HYBRIDS = {
    "qwen3next": ({}, ("recurrent",) * 3 + ("full",), "hybrid", [("attn", (3,)), ("recurrent", (0, 1, 2))], "neox"),
    "qwen35moe": ({"rope.dimension_sections": [2, 1, 1, 0]}, ("recurrent",) * 3 + ("full",), "hybrid",
                  [("attn", (3,)), ("recurrent", (0, 1, 2))], "imrope"),
    "qwen4exp": (QWEN4EXP, ("recurrent",) * 3 + ("full",), "hybrid_idx",
                 [("attn", (3,)), ("recurrent", (0, 1, 2)), ("indexer", (3,))], "imrope"),
    "minimax-01": ({"block_count": 8}, ("recurrent",) * 7 + ("full",), "hybrid",
                   [("attn", (7,)), ("recurrent", tuple(range(7)))], "neox"),
    "jamba": ({"attention.head_count_kv": [1, 0, 0, 1]}, ("full", "recurrent", "recurrent", "full"), "hybrid",
              [("attn", (0, 3)), ("recurrent", (1, 2))], "none"),
    "falcon-h1": ({}, ("parallel",) * 4, "hybrid", [("attn", (0, 1, 2, 3)), ("recurrent", (0, 1, 2, 3))], "neox"),
    "granitehybrid": ({"attention.head_count_kv": [0, 1, 0, 1]}, ("recurrent", "full", "recurrent", "full"), "hybrid",
                      [("attn", (1, 3)), ("recurrent", (0, 2))], "norm"),
    "nemotron_h": ({"block_count": 3, "attention.head_count_kv": [1, 0, 1], "feed_forward_length": [0, 0, 64]},
                   ("full", "recurrent", "mlp"), "hybrid", [("attn", (0,)), ("recurrent", (1,))], "none"),
    "plamo2": ({"attention.head_count": [0, 2, 0, 2], "attention.head_count_kv": [0, 1, 0, 1]},
               ("recurrent", "full", "recurrent", "full"), "hybrid", [("attn", (1, 3)), ("recurrent", (0, 2))], "none"),
    "lfm2": ({"attention.head_count_kv": [0, 1, 0, 1], "shortconv.l_cache": 3, "attention.sliding_window": 8},
             ("recurrent", "sliding", "recurrent", "sliding"), "hybrid_iswa",
             [("base", ()), ("swa", (1, 3)), ("recurrent", (0, 2))], "neox"),
    "lfm2moe": ({"attention.head_count_kv": [0, 1, 0, 1], "shortconv.l_cache": 3},
                ("recurrent", "full", "recurrent", "full"), "hybrid", [("attn", (1, 3)), ("recurrent", (0, 2))],
                "neox"),
    "mamba": ({"block_count": 2}, ("recurrent",) * 2, "recurrent", [("recurrent", (0, 1))], "none"),
    "rwkv6": ({"block_count": 2, "attention.head_count": 0, "wkv.head_size": 8}, ("recurrent",) * 2, "recurrent",
              [("recurrent", (0, 1))], "none"),
}


def random_state(spec, n, rng, *, dtype="f16", v_trans=False, flags=kvstate.FLAGS_NONE, tokens=None):
    """A SequenceState of `n` text tokens for any spec: random K/V (zero indexer V, as native) and
    random recurrent rows (PLE rows included) at n - 1."""
    tokens = rng.integers(0, 1000, n) if tokens is None else tokens
    parts = []
    for layout in spec.layouts(flags):
        if layout.kind == "recurrent":
            r_row, s_row = spec.recurrent_row_bytes
            parts.append(kvstate.RecurrentState(
                layout.name, layout.layers, np.asarray([n - 1], dtype=np.int32), ((),), spec.n_layer_trunk,
                {il: kvstate.OpaqueRows(0, r_row, rng.bytes(r_row)) for il in layout.layers},
                {il: kvstate.OpaqueRows(0, s_row, rng.bytes(s_row)) for il in layout.layers},
                {il: kvstate.OpaqueRows(None, spec.recurrent_p_bytes, rng.bytes(spec.recurrent_p_bytes))
                 for il in spec.ple_layers}))
            continue
        positions = np.arange(max(0, n - spec.n_swa) if layout.sliding else 0, n)
        k = {il: rng.normal(size=(len(positions),) + spec.kv_shape(il, "k", layout.role)).astype(np.float32)
             for il in layout.layers}
        v = {il: rng.normal(size=(len(positions),) + spec.kv_shape(il, "v", layout.role)).astype(np.float32)
             for il in layout.layers if layout.role == "attention"}
        stream = kvstate.build_kv_stream(spec, layout.layers, positions, k, v, dtype=dtype, v_trans=v_trans,
                                         tokens_at=np.asarray(tokens)[positions], role=layout.role)
        parts.append(kvstate.KVCache(layout.name, layout.layers, [stream]))
    return kvstate.SequenceState(spec, flags, parts), tokens


class HybridSpecTests(unittest.TestCase):
    def test_specs_of_every_hybrid_and_recurrent_family(self):
        for architecture, (extra, kinds, memory, sections, rope) in HYBRIDS.items():
            with self.subTest(architecture=architecture):
                remove = ("ssm.group_count",) if architecture == "mamba" else ()
                spec = kvstate.ModelSpec.from_metadata(hybrid_metadata(architecture, remove=remove, **extra))
                layout = [(name, layers) for name, _, layers in spec.sections(0)]
                self.assertEqual((spec.layer_types, spec.memory, layout, spec.rope_type), (kinds, memory, sections, rope))
                self.assertEqual(spec.n_layer_trunk, len(kinds))
                # plamo2's layer 0 is Mamba: no heads, so n_rot (and the head sizes) default to 0 and only
                # its own loader reads the head sizes back; nothing rotates
                self.assertEqual(spec.n_rot, 8 if rope != "none" else 0)
                for il in spec.attention_layers:
                    self.assertEqual(spec.kv_shape(il, "k"), (1, 16))
                    self.assertEqual(spec.kv_shape(il, "v"), (1, 16))
                # PARTIAL_ONLY (server checkpoints): the SWA KV of hybrid-iSWA memory stays, every
                # other KV section goes
                partial = [name for name, _, _ in spec.sections(kvstate.FLAGS_PARTIAL_ONLY)]
                self.assertEqual(partial, ["swa", "recurrent"] if memory == "hybrid_iswa" else ["recurrent"])
        # row sizes: Mamba-1 loaders read no group count; RWKV keeps token_shift_count rows of n_embd
        spec = kvstate.ModelSpec.from_metadata(hybrid_metadata("jamba", **HYBRIDS["jamba"][0]))
        self.assertEqual(spec.recurrent_row_bytes, (4 * 3 * 8, 4 * 2 * 8))
        mamba = kvstate.ModelSpec.from_metadata(hybrid_metadata("mamba", block_count=2))
        self.assertEqual(mamba.recurrent_row_bytes, (4 * 3 * 8, 4 * 2 * 8))
        rwkv = kvstate.ModelSpec.from_metadata(hybrid_metadata("rwkv6", **HYBRIDS["rwkv6"][0]))
        self.assertEqual(rwkv.recurrent_row_bytes, (4 * 2 * 32, 4 * 32 * 8))
        lfm2 = kvstate.ModelSpec.from_metadata(hybrid_metadata("lfm2moe", **HYBRIDS["lfm2moe"][0]))
        self.assertEqual(lfm2.recurrent_row_bytes, (4 * 32 * 2, 0))
        minimax = kvstate.ModelSpec.from_metadata(hybrid_metadata("minimax-01", block_count=8))
        self.assertEqual(minimax.recurrent_row_bytes, (0, 4 * 16 * 16 * 2))
        falcon = kvstate.ModelSpec.from_metadata(hybrid_metadata("falcon-h1"))
        self.assertEqual(falcon.recurrent_row_bytes, (4 * 3 * (8 + 2 * 2), 4 * 2 * 8))

    def test_qwen4exp_indexer_ple_and_cell_ext(self):
        spec = kvstate.ModelSpec.from_metadata(hybrid_metadata("qwen4exp", **QWEN4EXP))
        self.assertEqual((spec.indexer.layers, spec.indexer.head_size, spec.indexer.rope_type, spec.indexer.value_size),
                         ((3,), 8, "none", 16))
        self.assertEqual((spec.kv_shape(3, "k", "indexer"), spec.kv_shape(3, "v", "indexer")), ((1, 8), (1, 16)))
        self.assertIsNone(spec.rope(3, "indexer"))
        # (ngram - 1) * heads > 0, kernel 4: P = 3 * 3 * hc 2 * n_embd 32 floats on layer 0
        self.assertEqual((spec.recurrent_p_bytes, spec.ple_layers), (4 * 3 * 3 * 2 * 32, (0,)))
        # IMROPE gives the attention cells an ext; the indexer copies the hparams with rope NONE, so only
        # PLE gives its cells one
        self.assertEqual([layout.ext for layout in spec.layouts(0)], [True, False, True])
        no_ple = kvstate.ModelSpec.from_metadata(hybrid_metadata("qwen4exp", **{**QWEN4EXP, "ple.layers": []}))
        self.assertEqual([layout.ext for layout in no_ple.layouts(0)], [True, False, False])
        rng = np.random.default_rng(3)
        x = rng.normal(size=(5, 1, 8)).astype(np.float32)
        np.testing.assert_array_equal(kvstate.cache_keys(spec, 3, x, np.arange(5), role="indexer"), x)

    def test_refusals_and_runtime_rope(self):
        # default-filter hybrids would keep nextn layers in the main attention cache: not modelled;
        # so would nemotron's filters for a nextn layer with n_ff == 0
        for architecture in ("jamba", "falcon-h1", "nemotron_h", "lfm2", "mamba"):
            with self.subTest(architecture=architecture), self.assertRaisesRegex(StateFormatError, "nextn"):
                extra = dict(HYBRIDS[architecture][0])
                blocks = extra.pop("block_count", 4)
                extra = {key: value + [0 if key == "feed_forward_length" else value[-1]] if isinstance(value, list)
                         else value for key, value in extra.items()}
                kvstate.ModelSpec.from_metadata(hybrid_metadata(architecture, block_count=blocks + 1,
                                                                nextn_predict_layers=1, **extra))
        # the MTP block the converter writes for Nemotron-3 (attention + MoE, n_ff the MoE width) sits
        # in neither section
        for architecture in ("nemotron_h", "nemotron_h_moe"):
            mtp = kvstate.ModelSpec.from_metadata(hybrid_metadata(
                architecture, block_count=4, nextn_predict_layers=1, **{"attention.head_count_kv": [1, 0, 1, 1],
                                                                        "feed_forward_length": [0, 0, 64, 32]}))
            self.assertEqual((mtp.layer_types, mtp.sections(0)), (("full", "recurrent", "mlp"), [
                ("attn", "kv", (0,)), ("recurrent", "recurrent", (1,))]))
        # qwen3next/qwen35moe/qwen4exp/minimax filter il < n_layer(): nextn layers sit in neither section
        spec = kvstate.ModelSpec.from_metadata(hybrid_metadata("qwen3next", block_count=5, nextn_predict_layers=1))
        self.assertEqual((spec.layer_types, spec.n_layer_trunk), (("recurrent",) * 3 + ("full",), 4))
        with self.assertRaisesRegex(StateFormatError, "values for 4 layers"):
            kvstate.ModelSpec.from_metadata(hybrid_metadata("jamba", **{"attention.head_count_kv": [1, 0]}))
        # layer 0 without heads leaves the head sizes 0 (src/llama-model.cpp:1335-1366); plamo2 reads them again
        with self.assertRaisesRegex(StateFormatError, "attention dimensions"):
            kvstate.ModelSpec.from_metadata(hybrid_metadata("jamba", **{"attention.head_count": [0, 2, 0, 2],
                                                                        "attention.head_count_kv": [0, 1, 0, 1]}))
        # granitehybrid rotates only when rope.scaling.finetuned (default true)
        granite = kvstate.ModelSpec.from_metadata(hybrid_metadata("granitehybrid", **{
            "attention.head_count_kv": [0, 1, 0, 1], "rope.scaling.finetuned": False}))
        self.assertEqual((granite.rope_type, granite.rope(1)), ("none", None))
        with self.assertRaisesRegex(StateFormatError, "frequency factors"):
            dataclasses.replace(granite, rope_freq_factors=(1.0,) * 4)
        # rope_freqs.weight is an optional tensor metadata cannot show: a rotating granitehybrid spec
        # without it refuses rope() (layouts still parse); () says the GGUF has none
        meta = hybrid_metadata("granitehybrid", **HYBRIDS["granitehybrid"][0])
        unknown = kvstate.ModelSpec.from_metadata(meta)
        self.assertEqual(unknown.sections(0), [("attn", "kv", (1, 3)), ("recurrent", "recurrent", (0, 2))])
        with self.assertRaisesRegex(StateFormatError, "frequency factors are unknown"):
            unknown.rope(1)
        self.assertIsNone(kvstate.ModelSpec.from_metadata(meta, rope_freq_factors=()).rope(1).freq_factors)
        factors = (1.0, 2.0, 4.0, 8.0)
        self.assertEqual(kvstate.ModelSpec.from_metadata(meta, rope_freq_factors=factors).rope(3).freq_factors, factors)
        with self.assertRaisesRegex(StateFormatError, "frequency factors"):
            kvstate.ModelSpec.from_metadata(hybrid_metadata("qwen3next"), rope_freq_factors=factors)
        self.assertIsNotNone(kvstate.ModelSpec.from_metadata(hybrid_metadata("qwen3next"), rope_freq_factors=()).rope(3))
        # the receiving server's YaRN settings apply to every RoPE hybrid
        yarn = kvstate.ModelSpec.from_metadata(hybrid_metadata("qwen3next"), runtime={
            "rope_scaling_type": "yarn", "rope_freq_scale": 0.25, "yarn_orig_ctx": 64})
        self.assertEqual((yarn.rope_ext_factor, yarn.rope_freq_scale, yarn.rope(3).n_ctx_orig), (1.0, 0.25, 64))
        with self.assertRaisesRegex(StateFormatError, "longrope"):
            kvstate.ModelSpec.from_metadata(hybrid_metadata("lfm2moe", **{**HYBRIDS["lfm2moe"][0],
                                                                          "rope.scaling.type": "longrope"}))

    def test_states_round_trip_for_every_layout(self):
        rng = np.random.default_rng(31)
        for architecture, (extra, *_rest) in HYBRIDS.items():
            remove = ("ssm.group_count",) if architecture == "mamba" else ()
            spec = kvstate.ModelSpec.from_metadata(hybrid_metadata(architecture, remove=remove, **extra))
            for dtype, v_trans in (("f16", False), ("f32", True)):
                with self.subTest(architecture=architecture, dtype=dtype, v_trans=v_trans):
                    state, tokens = random_state(spec, 11, rng, dtype=dtype, v_trans=v_trans)
                    data = kvstate.StateFile(kvstate.token_words(tokens, server_packed=True), state).to_bytes()
                    parsed = kvstate.parse_file(data, spec)
                    self.assertEqual(parsed.to_bytes(), data)
                    for part, layout in zip(parsed.state.parts, spec.layouts(0)):
                        if layout.kind == "kv":
                            self.assertEqual(part.stream().ext is not None, layout.ext)
                    partial, _ = random_state(spec, 11, rng, flags=kvstate.FLAGS_PARTIAL_ONLY, tokens=tokens)
                    blob = kvstate.StateBlob(0, partial).to_bytes()
                    self.assertEqual(kvstate.parse_blob(blob, spec, flags=kvstate.FLAGS_PARTIAL_ONLY).to_bytes(), blob)
                    # R0 at t0 - 1 for every recurrent section, PLE rows included
                    reset = [kvstate.recurrent_reset(part, 10) if isinstance(part, kvstate.RecurrentState) else part
                             for part in parsed.state.parts]
                    zeroed = kvstate.StateFile(parsed.token_words, kvstate.SequenceState(spec, 0, reset)).to_bytes()
                    self.assertEqual(len(zeroed), len(data))
                    rows = [rows for part in reset if isinstance(part, kvstate.RecurrentState)
                            for rows in (*part.r.values(), *part.s.values(), *part.p.values())]
                    self.assertTrue(rows and not any(any(r.data) for r in rows))
        # the qwen4exp indexer V rows are never written: a builder leaves them zero
        spec = kvstate.ModelSpec.from_metadata(hybrid_metadata("qwen4exp", **QWEN4EXP))
        stream = kvstate.build_kv_stream(spec, (3,), np.arange(4), {3: rng.normal(size=(4, 1, 8))}, None,
                                         role="indexer", tokens_at=np.arange(4))
        self.assertEqual((stream.v[3].shape, bool(stream.v[3].any())), ((4, 1, 16), False))


class RopeFactorAndInverseTests(unittest.TestCase):
    def test_frequency_factors_divide_the_angle_of_each_pair(self):
        rng = np.random.default_rng(5)
        factors = tuple(float(f) for f in rng.uniform(0.5, 4.0, 4).astype(np.float32))
        rope = kvstate.Rope("norm", 8, 1e4, freq_factors=factors)
        positions = np.arange(0, 400, 7)
        cos, _ = rope.cos_sin(positions)
        angles = (kvstate.rope_angles(positions, 8, 1e4) / np.asarray(factors, np.float32)).astype(np.float32)
        np.testing.assert_array_equal(cos, kvstate._cos_sin(angles)[0])
        x = rng.normal(size=(len(positions), 2, 8)).astype(np.float32)
        np.testing.assert_allclose(rope.invert(rope.apply(x, positions), positions), x, atol=3e-6)
        self.assertFalse(np.allclose(rope.apply(x, positions), kvstate.Rope("norm", 8, 1e4).apply(x, positions)))
        with self.assertRaisesRegex(ValueError, "one value per rotated pair"):
            kvstate.Rope("norm", 8, 1e4, freq_factors=(1.0,))

    def test_inverse_divides_by_the_cached_norm(self):
        # float32 cos^2 + sin^2 is not exactly 1: dividing by it undoes the rotation of the cached
        # pairs more exactly, so forward(inverse(row)) keeps more float32 rows bit for bit
        rng = np.random.default_rng(6)
        x = rng.normal(size=(512, 4, 64)).astype(np.float32)
        positions = rng.integers(0, 100_000, 512)
        rope = kvstate.Rope("neox", 64, 1e6)
        y = rope.apply(x, positions)
        back = rope.invert(y, positions)
        c, s = (a.astype(np.float64)[:, None, :] for a in rope.cos_sin(positions))
        y0, y1 = y[..., :32].astype(np.float64), y[..., 32:].astype(np.float64)
        np.testing.assert_array_equal(back[..., :32], ((y0 * c + y1 * s) / (c * c + s * s)).astype(np.float32))
        plain = y.copy()
        plain[..., :32], plain[..., 32:] = (y0 * c + y1 * s).astype(np.float32), (y1 * c - y0 * s).astype(np.float32)
        changed = np.mean(rope.apply(back, positions) != y)
        self.assertLess(changed, np.mean(rope.apply(plain, positions) != y))


# -- DeepSeek V4 ------------------------------------------------------------------------------

def dsv4_metadata(*, ratios=(0, 0, 4, 128), nextn=0, head=64, n_rot=32, window=32, indexer=128, yarn=None):
    """deepseek4 metadata shaped like test-llama-archs' fixture (head 64, n_rot 32, window 32)."""
    p = "deepseek4."
    meta = {"general.architecture": "deepseek4", p + "block_count": len(ratios) + nextn,
            p + "nextn_predict_layers": nextn, p + "embedding_length": 512, p + "attention.head_count": 8,
            p + "attention.head_count_kv": 1, p + "attention.key_length": head, p + "attention.value_length": head,
            p + "rope.dimension_count": n_rot, p + "rope.freq_base": 10000.0, p + "attention.sliding_window": window,
            p + "attention.compress_ratios": list(ratios) + [0] * nextn,
            p + "attention.compress_rope_freq_base": 160000.0, p + "attention.indexer.key_length": indexer,
            p + "attention.indexer.head_count": 64, p + "attention.indexer.top_k": 8,
            p + "attention.layer_norm_rms_epsilon": 1e-5, p + "context_length": 256}
    if yarn:
        meta.update({p + "rope.scaling.type": "yarn", p + "rope.scaling.factor": float(yarn),
                     p + "rope.scaling.original_context_length": 64})
    return meta


def dsv4_state(spec, n, rng, *, cache="f16", flags=kvstate.FLAGS_NONE, n_stream=1, stream=0, v_trans=True):
    """A random, well-formed DeepSeek V4 sequence state of n tokens (flags 0 or PARTIAL_ONLY)."""
    code = kvstate.CACHE_TYPES[cache]

    def rows(shape):
        values = rng.normal(size=shape).astype(np.float32)
        return kvstate.quantize_q8_0(values) if cache == "q8_0" else values.astype(kvstate.GGML_DTYPES[code])

    def raw(name, layers, positions):
        k = {il: rows((len(positions), 1, spec.head_dim_k[il])) for il in layers}
        built = kvstate.KVStream(np.asarray(positions, np.int32), tuple((stream,) for _ in positions), None, v_trans,
                                 k, {}, {il: code for il in layers})
        return kvstate.KVCache(name, layers, [built if s == stream else None for s in range(n_stream)])
    parts = []
    for layout in spec.layouts(flags):
        if layout.name == "base":
            parts.append(raw("base", (), np.arange(n)))
        elif layout.name == "swa":
            parts.append(raw("swa", layout.layers, np.arange(max(0, n - spec.n_swa), n)))
        elif layout.kind == "rows":
            pool = spec.pool(layout.name)
            parts.append(kvstate.PooledRows(layout.name, layout.layers, 1, n // pool.ratio,
                                            {il: rows((1, n // pool.ratio, pool.width)) for il in layout.layers},
                                            {il: code for il in layout.layers}))
        else:
            pool = spec.pool(layout.name[:-len("_state")])
            shape = (1, pool.state_size, pool.state_width)
            parts.append(kvstate.CompressorRing(layout.name, layout.layers, 1,
                                                {il: rng.normal(size=shape).astype(np.float32) for il in layout.layers},
                                                {il: rng.normal(size=shape).astype(np.float32) for il in layout.layers}))
    return kvstate.SequenceState(spec, flags, parts)


class Dsv4SpecTests(unittest.TestCase):
    def test_spec_and_layouts(self):
        spec = kvstate.ModelSpec.from_metadata(dsv4_metadata(nextn=1))
        self.assertEqual((spec.memory, spec.layer_types, spec.n_swa), ("dsv4", ("sliding",) * 4 + ("nextn",), 32))
        self.assertEqual(spec.compress.ratios, (0, 0, 4, 128))
        self.assertEqual([(l.name, l.kind, l.layers) for l in spec.layouts(kvstate.FLAGS_NONE)],
                         [("base", "kv", ()), ("swa", "kv", (0, 1, 2, 3, 4)), ("csa", "rows", (2,)),
                          ("hca", "rows", (3,)), ("lid", "rows", (2,)), ("csa_state", "ring", (2,)),
                          ("hca_state", "ring", (3,)), ("lid_state", "ring", (2,))])
        self.assertEqual([l.name for l in spec.layouts(kvstate.FLAGS_PARTIAL_ONLY)],
                         ["swa", "csa_state", "hca_state", "lid_state"])
        self.assertEqual([(p.name, p.ratio, p.width, p.state_size, p.state_width) for p in spec.pools()],
                         [("csa", 4, 64, 8, 128), ("hca", 128, 64, 128, 64), ("lid", 4, 128, 8, 256)])
        # one K-only latent per cell, read as both K and V
        self.assertTrue(spec.k_only(0))
        self.assertEqual((spec.kv_shape(0, "k"), spec.kv_shape(0, "v"), spec.head_shape(0, "v")), ((1, 64), None, (1, 64)))
        for bad, message in ((dict(ratios=(0, 3)), "compress ratios"), (dict(n_rot=128), "n_rot")):
            with self.subTest(bad=bad), self.assertRaisesRegex(StateFormatError, message):
                kvstate.ModelSpec.from_metadata(dsv4_metadata(**bad))
        meta = dsv4_metadata(nextn=1)
        meta["deepseek4.attention.compress_ratios"][-1] = 4
        with self.assertRaisesRegex(StateFormatError, "MTP"):
            kvstate.ModelSpec.from_metadata(meta)

    def test_rope_per_layer_type(self):
        spec = kvstate.ModelSpec.from_metadata(dsv4_metadata())
        # window-only layers: freq_base, no YaRN; compressed layers and their pooled rows: the compress base
        self.assertEqual(spec.rope(0), kvstate.Rope("norm", 32, 10000.0, offset=32))
        self.assertEqual((spec.rope(2).base, spec.rope(2).offset, spec.rope(2, "pooled").offset), (160000.0, 32, 32))
        self.assertEqual((spec.rope(2, "lid").offset, spec.rope(2, "lid").base), (96, 160000.0))
        for layer, role in ((0, "pooled"), (3, "lid"), (2, "indexer")):
            with self.subTest(layer=layer, role=role), self.assertRaises(ValueError):
                spec.rope(layer, role)
        yarn = kvstate.ModelSpec.from_metadata(dsv4_metadata(yarn=4))
        self.assertEqual(yarn.rope(0), spec.rope(0))
        rope = yarn.rope(3)
        self.assertEqual((rope.freq_scale, rope.ext_factor, rope.n_ctx_orig), (0.25, 1.0, 64))
        # the DSV4 attn_factor cancels ggml's YaRN magnitude
        self.assertAlmostEqual(rope.attn_factor, 1 / (1 + 0.1 * np.log(4)), places=6)
        self.assertEqual(float(rope.magnitude()), 1.0)
        self.assertEqual(kvstate.dsv4_rope_attn_factor(0.25, 0.0), 1.0)

    def test_q8_0_blocks(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(5, 64)).astype(np.float32)
        x[0, :32] = 0
        blocks = kvstate.quantize_q8_0(x)
        self.assertEqual((blocks.shape, blocks.dtype.itemsize), ((5, 2), 34))
        back = kvstate.dequantize_q8_0(blocks)
        self.assertTrue((back[0, :32] == 0).all())
        np.testing.assert_allclose(back, x, atol=np.abs(x).max() / 127)
        # the largest magnitude of a block maps to +-127 exactly
        self.assertEqual(int(np.abs(blocks["qs"]).max()), 127)
        np.testing.assert_array_equal(kvstate.quantize_q8_0(back), blocks)


class Dsv4StateTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(9)
        self.spec = kvstate.ModelSpec.from_metadata(dsv4_metadata(nextn=1))

    def test_full_and_partial_round_trip(self):
        for cache in ("f16", "f32", "q8_0"):
            for n_stream, stream in ((1, 0), (2, 1)):
                with self.subTest(cache=cache, n_stream=n_stream):
                    state = dsv4_state(self.spec, 300, self.rng, cache=cache, n_stream=n_stream, stream=stream)
                    words = np.arange(300, dtype=np.int32) % 128
                    data = kvstate.StateFile(words, state).to_bytes()
                    self.assertEqual(struct.unpack_from("<III", data, 12 + 4 * 300), (0x34565344, 1, 0))
                    parsed = kvstate.parse_file(data, self.spec)
                    self.assertEqual(parsed.to_bytes(), data)
                    self.assertEqual(parsed.state.part("csa").n_rows, 75)
                    layout = {e["name"]: e for e in kvstate.template_layout(parsed)}
                    self.assertEqual((layout["swa"]["n_stream"], layout["swa"]["stream"], layout["swa"]["dtype"],
                                      layout["hca"]["n_rows"], layout["hca"]["dtype"]), (n_stream, stream, cache, 2, cache))
                    partial = dsv4_state(self.spec, 300, self.rng, cache=cache, flags=kvstate.FLAGS_PARTIAL_ONLY,
                                         n_stream=n_stream, stream=stream)
                    blob = kvstate.StateBlob(stream, partial).to_bytes()
                    self.assertEqual(struct.unpack_from("<IiIII", blob), (kvstate.IO_MAGIC, stream, 0x34565344, 1, 1))
                    self.assertEqual(kvstate.parse_blob(blob, self.spec, flags=kvstate.FLAGS_PARTIAL_ONLY).to_bytes(),
                                     blob)
                    with self.assertRaisesRegex(StateFormatError, "mode"):
                        kvstate.parse_blob(blob, self.spec, flags=kvstate.FLAGS_NONE)

    def test_refusals(self):
        state = dsv4_state(self.spec, 300, self.rng)
        words = np.arange(300, dtype=np.int32) % 128
        good = kvstate.StateFile(words, state)
        # one row per completed block: 299 tokens hold 74 CSA rows, not 75
        short = dsv4_state(self.spec, 299, self.rng)
        short.parts = [state.part("csa") if part.name == "csa" else part for part in short.parts]
        with self.assertRaisesRegex(StateFormatError, "csa section holds 75 rows"):
            kvstate.StateFile(words[:299], short).to_bytes()
        with self.assertRaisesRegex(StateFormatError, "cannot be cut"):
            kvstate.truncate_state(good, 200)
        data = bytearray(good.to_bytes())
        offset = 12 + 4 * 300
        data[offset:offset + 4] = struct.pack("<I", 0x12345678)
        with self.assertRaisesRegex(StateFormatError, "not a DeepSeek V4 state"):
            kvstate.parse_file(bytes(data), self.spec)
        # q8_0 rows stay DeepSeek V4 only
        qwen = kvstate.ModelSpec.from_metadata(qwen3_metadata())
        with self.assertRaisesRegex(StateFormatError, "F32 = 0 and F16 = 1 only"):
            kvstate._storage(kvstate.GGML_TYPE_Q8_0, (1, 64), "x")
        self.assertNotEqual(qwen.memory, "dsv4")
        ring = state.part("hca_state")
        ring.kv[3] = ring.kv[3].astype(np.float16)
        with self.assertRaisesRegex(StateFormatError, "must be float32"):
            good.to_bytes()


if __name__ == "__main__":
    unittest.main()
