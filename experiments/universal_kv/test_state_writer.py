"""state_writer.write: the bytes of kvstate.write_new(path, state.to_bytes()), streamed."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import state_writer
from test_translate import HYBRID, MUSE, TARGET, hybrid_state

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "linear_transfer"))
import kvstate  # noqa: E402


def mla_spec(architecture="deepseek2", indexer=False):
    """A tiny MLA (K-only) spec, plus DSA indexer keys when asked (as kvstate's GGUF reader builds it)."""
    p = architecture + "."
    meta = {"general.architecture": architecture, p + "block_count": 3, p + "context_length": 4096,
            p + "embedding_length": 64, p + "attention.head_count": 2, p + "attention.head_count_kv": 1,
            p + "attention.key_length": 32, p + "attention.value_length": 16, p + "attention.key_length_mla": 32,
            p + "attention.value_length_mla": 16, p + "attention.kv_lora_rank": 16, p + "rope.dimension_count": 16,
            p + "rope.freq_base": 10000.0, p + "attention.layer_norm_rms_epsilon": 1e-6}
    if indexer:
        meta.update({p + "attention.indexer.head_count": 2, p + "attention.indexer.key_length": 64,
                     p + "attention.indexer.top_k": 8})
    return kvstate.ModelSpec.from_metadata(meta)


def states():
    """(name, spec, StateFile) of every layout family, in both cache dtypes and V layouts."""
    rng = np.random.default_rng(7)
    out = []
    for dtype in ("f16", "f32"):
        for v_trans in (False, True):
            tag = f"{dtype}{'-vtrans' if v_trans else ''}"
            n = 23
            tokens = np.arange(n) + 100

            def kv(spec, layers):
                return ({l: rng.standard_normal((n,) + spec.head_shape(l, "k")).astype(np.float32) for l in layers},
                        {l: rng.standard_normal((n,) + spec.head_shape(l, "v")).astype(np.float32) for l in layers})
            k, v = kv(TARGET, TARGET.layers_of("full"))
            out.append((f"dense {tag}", TARGET, kvstate.build_dense_state(
                TARGET, tokens, np.arange(n), k, v, dtype=dtype, v_trans=v_trans, server_tokens=True)))
            k, v = kv(MUSE, MUSE.layers_of("full", "sliding"))
            out.append((f"iswa {tag}", MUSE, kvstate.build_muse_state(
                MUSE, tokens, np.arange(n), k, v, dtype=dtype, v_trans=v_trans)))
            k, v = kv(HYBRID, HYBRID.layers_of("full"))
            out.append((f"hybrid {tag}", HYBRID, hybrid_state(HYBRID, tokens, k, v, dtype=dtype, v_trans=v_trans)))
            for name, spec in (("mla", mla_spec()), ("dsa", mla_spec("deepseek32", indexer=True))):
                layers = spec.layers_of("full")
                k = {l: rng.standard_normal((n,) + spec.kv_shape(l, "k")).astype(np.float32) for l in layers}
                state = kvstate.build_dense_state(spec, tokens, np.arange(n), k, None, dtype=dtype, v_trans=v_trans,
                                                  **({"indexer_k": "zeros"} if spec.indexer else {}))
                out.append((f"{name} {tag}", spec, state))
    return out


class StateWriterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bytes_equal_write_new_for_every_layout(self):
        for name, spec, state in states():
            with self.subTest(name):
                parsed = kvstate.parse_file(state.to_bytes(), spec)        # as read from a slot file
                for source in (state, parsed):
                    reference = kvstate.write_new(self.dir / f"{name}-{id(source)}-ref", source.to_bytes())
                    ours = self.dir / f"{name}-{id(source)}-ours"
                    size = state_writer.write(ours, source)
                    self.assertEqual(ours.read_bytes(), reference.read_bytes())
                    self.assertEqual(size, reference.stat().st_size)

    def test_fast_path_covers_kv_sections_and_kvstate_writes_the_rest(self):
        _, spec, state = next(item for item in states() if item[0] == "hybrid f16")
        with mock.patch.object(kvstate, "_write_kv", wraps=kvstate._write_kv) as write_kv, \
                mock.patch.object(kvstate, "_write_recurrent", wraps=kvstate._write_recurrent) as write_recurrent:
            b"".join(bytes(chunk) for chunk in state_writer.chunks(state))
        # the attention section took the fast path: kvstate wrote only its first cells, to check it
        self.assertEqual(write_kv.call_count, 1)
        self.assertEqual(write_kv.call_args.args[1].stream().n_cells, state_writer.CHECK_CELLS)
        self.assertEqual(write_recurrent.call_count, 1)         # kvstate serializes the recurrent section

        class Custom(kvstate.KVCache):                           # anything unusual goes to kvstate's serializer
            pass
        part = state.state.parts[0]
        state.state.parts[0] = Custom(part.name, part.layers, part.streams)
        expected = kvstate.StateFile(state.token_words, kvstate.SequenceState(
            spec, state.state.flags, [part] + state.state.parts[1:])).to_bytes()
        with mock.patch.object(kvstate, "_write_kv", wraps=kvstate._write_kv) as write_kv:
            got = b"".join(bytes(chunk) for chunk in state_writer.chunks(state))
        self.assertEqual(write_kv.call_count, 1)
        self.assertEqual(got, expected)

    def test_a_changed_kv_format_in_kvstate_is_followed(self):
        original = kvstate._write_kv

        def changed(spec, cache, layout):                        # e.g. a future version field per section
            return b"\x02\x00\x00\x00" + original(spec, cache, layout)
        for name, spec, state in states()[:3]:
            with self.subTest(name), mock.patch.object(kvstate, "_write_kv", changed):
                expected = state.to_bytes()
                self.assertEqual(b"".join(bytes(chunk) for chunk in state_writer.chunks(state)), expected)

    def test_states_holding_media_are_written_like_kvstate(self):
        # a slot file whose token words hold an M-RoPE image (cells sharing one position), as
        # llama-server saves it with a projector: written whole, and truncated before the image
        from test_media_state import media_state, qwen_metadata
        spec = kvstate.ModelSpec.from_metadata(qwen_metadata())
        state = media_state(spec, np.random.default_rng(3))
        self.assertEqual(len(state.media.media), 1)
        for name, item in (("whole", state), ("cut", kvstate.truncate_state(state, 5, recurrent="reset"))):
            with self.subTest(name):
                size = state_writer.write(self.dir / f"{name}.bin", item)
                self.assertEqual((self.dir / f"{name}.bin").read_bytes(), item.to_bytes())
                self.assertEqual(size, len(item.to_bytes()))
        # malformed token words are refused as to_bytes refuses them
        words = np.asarray(state.token_words).copy()
        words[-1] = 1 << 20                                     # a chunk size past the end of the words
        with self.assertRaises(kvstate.StateFormatError):
            kvstate.StateFile(words, state.state).to_bytes()
        with self.assertRaises(kvstate.StateFormatError):
            state_writer.write(self.dir / "bad-media.bin", kvstate.StateFile(words, state.state))

    def test_new_file_is_private_atomic_and_never_replaces(self):
        _, _, state = states()[0]
        path = self.dir / "slot.bin"
        state_writer.write(path, state)
        if os.name == "posix":
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            state_writer.write(path, state)
        self.assertEqual(path.read_bytes(), before)
        # a refused state leaves nothing behind, not even the temporary file
        bad = kvstate.StateFile(np.arange(5, dtype=np.int32), state.state)       # tokens do not end the cache
        with self.assertRaises(kvstate.StateFormatError):
            state_writer.write(self.dir / "bad.bin", bad)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["slot.bin"])
        # check_positions=False writes what to_bytes(check_positions=False) gives
        state_writer.write(self.dir / "unchecked.bin", bad, check_positions=False)
        self.assertEqual((self.dir / "unchecked.bin").read_bytes(), bad.to_bytes(check_positions=False))


if __name__ == "__main__":
    unittest.main()
