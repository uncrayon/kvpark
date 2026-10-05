"""handoff.py: builder, token table and refusals on synthetic data; end to end with KVPARK_KV_PROBE."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import align
from attention_eval import Mapper
from captures import Capture, CaptureSet, write_synthetic_capture
import chat
import fit_layers
import handoff
import kvstate
from test_run_v2 import MUSE_BOS, MUSE_CONTROL, MUSE_TEMPLATE, QWEN_CONTROL, QWEN_TEMPLATE, SPLITS, conversation


PROBE = os.environ.get("KVPARK_KV_PROBE")
QWEN_IDS = {token: 258 + i for i, token in enumerate(QWEN_CONTROL)}
MUSE_IDS = {MUSE_BOS: 256, **{token: 258 + i for i, token in enumerate(MUSE_CONTROL)}}

QWEN_SPEC = kvstate.ModelSpec("qwen35", ("recurrent", "recurrent", "recurrent", "full") * 2, (0, 0, 0, 1) * 2,
                              (0, 0, 0, 16) * 2, (0, 0, 0, 16) * 2, rope_type="imrope", n_rot=8,
                              rope_freq_base=1e4, rope_freq_base_swa=1e4, rope_sections=(2, 1, 1, 0),
                              recurrent_row_bytes=(24, 40))
MUSE_SPEC = kvstate.ModelSpec("muse-glimmer", ("sliding", "sliding", "sliding", "full") * 2, (2,) * 8, (8,) * 8,
                              (8,) * 8, n_swa=4, rope_type="norm", n_rot=8, rope_freq_base=1e4,
                              rope_freq_base_swa=1e4)


def byte_tokens(text, controls):
    """Byte-level tokenization with control tokens as single ids (the tiny fixtures' tokenizer)."""
    ids, ends, special, offset, rest = [], [], [], 0, text
    while rest:
        control = next((token for token in controls if rest.startswith(token)), None)
        if control:
            ids.append(controls[control])
            offset += len(control.encode("utf-8"))
            ends.append(offset)
            special.append(1)
            rest = rest[len(control):]
            continue
        for byte in rest[0].encode("utf-8"):
            ids.append(byte)
            offset += 1
            ends.append(offset)
            special.append(0)
        rest = rest[1:]
    return {"token_id": np.asarray(ids, np.int32), "byte_end": np.asarray(ends, np.int64),
            "special": np.asarray(special, np.uint8)}


def render_records(numbers):
    qwen, muse = [], []
    for number in numbers:
        item = conversation(number)
        qwen.append(chat.rendered_record(item, "qwen", chat.render(item, QWEN_TEMPLATE)))
        muse.append(chat.rendered_record(item, "muse", chat.render(item, MUSE_TEMPLATE, bos_token=MUSE_BOS)))
    return qwen, muse


def write_mapper(directory, maps, *, source_shape, target_shape):
    """fit_layers-format mapper; maps: [(kind, target, source_layers, k_renorm, weight, bias, mean)]."""
    directory = Path(directory)
    directory.mkdir()
    entries = []
    for kind, target, sources, renorm, weight, bias, mean in maps:
        files = {"weight": f"W-{kind}-{target}.npy", "bias": f"b-{kind}-{target}.npy",
                 "mean": f"mean-{kind}-{target}.npy"}
        hashes = {}
        for name, array in (("weight", weight), ("bias", bias), ("mean", mean)):
            np.save(directory / files[name], np.asarray(array, dtype=np.float32))
            hashes[name] = hashlib.sha256((directory / files[name]).read_bytes()).hexdigest()
        entries.append({"kind": kind, "target": target, "source_layers": list(sources), "k": len(sources),
                        "alpha": 0.01, "lambda": 1.0, "k_renorm": bool(renorm), "files": files, "sha256": hashes})
    manifest = {"format_version": 1, "purpose": "kv_linear_transfer_mapper",
                "source": {"architecture": "qwen35", "head_shape": {k: list(source_shape) for k in "kv"}},
                "target": {"architecture": "muse-glimmer", "head_shape": {k: list(target_shape) for k in "kv"}},
                "maps": entries}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def unit_heads(values):
    return values / np.sqrt(np.mean(values ** 2, axis=-1, keepdims=True))


class SyntheticHandoffTests(unittest.TestCase):
    """Random captures on small specs; records rendered with the tiny chat templates."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-handoff-")
        root = cls.root = Path(cls.temp.name)
        rng = np.random.default_rng(7)
        cls.qwen_records, cls.muse_records = render_records(range(len(SPLITS)))
        cls.tokens = {}
        sides = {}
        for name, records, controls, spec in (("qwen", cls.qwen_records, QWEN_IDS, QWEN_SPEC),
                                              ("muse", cls.muse_records, MUSE_IDS, MUSE_SPEC)):
            documents, arrays, first = [], {}, 0
            for record in records:
                tokens = byte_tokens(record["text"], controls)
                cls.tokens[name, record["document_id"]] = tokens
                documents.append({"document_id": record["document_id"], **{k: v.tolist() for k, v in tokens.items()}})
            total = sum(len(d["byte_end"]) for d in documents)
            for layer in spec.layers_of("full", "sliding"):
                heads, dim = spec.head_shape(layer, "k")
                k = rng.normal(size=(total, heads, dim)).astype(np.float32)
                arrays["k", layer] = unit_heads(k).astype(np.float32) if name == "muse" else k
                arrays["v", layer] = rng.normal(size=(total, heads, dim)).astype(np.float32)
            path = write_synthetic_capture(root / name, documents, arrays, architecture=spec.architecture,
                                           layer_types=list(spec.layer_types))
            sides[name] = (CaptureSet([path]), arrays)
        (cls.qwen_set, cls.qwen_arrays), (cls.muse_set, cls.muse_arrays) = sides["qwen"], sides["muse"]
        cls.alignment = align.align(cls.qwen_records, cls.qwen_set, cls.muse_records, cls.muse_set)
        maps = []
        for kind in "kv":
            for target in MUSE_SPEC.layers_of("full", "sliding"):
                sources = [3] if target % 2 else [7, 3]
                maps.append((kind, target, sources, kind == "k" and target == 3,
                             rng.normal(0, .3, (16 * len(sources), 16)), rng.normal(0, .1, 16),
                             rng.normal(0, 1, 16)))
        cls.mapper_path = write_mapper(root / "mapper", maps, source_shape=(1, 16), target_shape=(2, 8))
        cls.table = handoff.build_token_table(cls.muse_set, cls.alignment, chunk_rows=1000)
        cls.table_path = handoff.write_token_table(root / "table.npz", cls.table)
        number = SPLITS.index("test")
        cls.qwen_record, cls.muse_record = cls.qwen_records[number], cls.muse_records[number]
        cls.doc = cls.muse_record["document_id"]
        cls.qwen_tokens, cls.muse_tokens = cls.tokens["qwen", cls.doc], cls.tokens["muse", cls.doc]
        cls.P = handoff.handoff_position(cls.muse_record, cls.muse_tokens)
        cls.n_q = handoff.handoff_position(cls.qwen_record, cls.qwen_tokens)
        cls.t0 = cls.P - 20                     # the SWA window then holds content and template cells

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def rows(self, name, kind, layer, stop):
        capture_set, arrays = (self.qwen_set, self.qwen_arrays) if name == "qwen" else (self.muse_set, self.muse_arrays)
        window = capture_set.rows(self.doc)
        return arrays[kind, layer][window.start:window.start + stop]

    def qwen_state(self, n=None, dtype="f32"):
        n = self.n_q if n is None else n
        layers = QWEN_SPEC.layers_of("full")
        state = handoff.qwen_state_from_rows(QWEN_SPEC, self.qwen_tokens["token_id"][:n],
                                             {l: self.rows("qwen", "k", l, n) for l in layers},
                                             {l: self.rows("qwen", "v", l, n) for l in layers}, dtype=dtype)
        return kvstate.parse_file(state.to_bytes(), QWEN_SPEC)

    def native(self, t0, dtype="f32", v_trans=False):
        layers = MUSE_SPEC.layers_of("full", "sliding")
        state = kvstate.build_muse_state(MUSE_SPEC, self.muse_tokens["token_id"][:t0], np.arange(t0),
                                         {l: self.rows("muse", "k", l, t0) for l in layers},
                                         {l: self.rows("muse", "v", l, t0) for l in layers}, dtype=dtype,
                                         v_trans=v_trans)
        data = state.to_bytes()
        return data, kvstate.parse_file(data, MUSE_SPEC)

    def build(self, **options):
        arguments = {"t0": self.t0, "muse_spec": MUSE_SPEC, "dtype": "f32", "fill_policy": "mean"}
        arguments.update(options)
        qwen_state = arguments.pop("qwen_state", "default")
        return handoff.build_handoff(self.qwen_state() if qwen_state == "default" else qwen_state, self.qwen_record,
                                     self.qwen_tokens, self.muse_record, self.muse_tokens,
                                     arguments.pop("mapper", self.mapper_path), **arguments)

    def stored(self, result, section, layer, kind):
        stream = result.state.state.part(section).stream()
        return np.asarray((stream.k if kind == "k" else stream.v)[layer], dtype=np.float32)

    def test_records_align_and_the_handoff_point_starts_the_final_message(self):
        self.assertEqual(self.alignment.rejected, [])
        start = handoff.final_message_start(self.muse_record)
        ends = self.muse_tokens["byte_end"]
        self.assertEqual(ends[self.P - 1], start)
        self.assertGreater(ends[self.P], start)
        self.assertGreater(self.t0, 2 * MUSE_SPEC.n_swa)

    def test_oracle_policies_reproduce_the_native_state_bytes(self):
        for dtype, v_trans in (("f32", False), ("f16", False), ("f16", True)):
            with self.subTest(dtype=dtype, v_trans=v_trans):
                data, native = self.native(self.t0, dtype, v_trans)
                result = self.build(content_policy="oracle", fill_policy="oracle", oracle_state=native, dtype=dtype,
                                    v_trans=v_trans, qwen_state=None, mapper=None)
                self.assertEqual(result.state.to_bytes(), data)
                base, swa = result.report["sections"]["base"], result.report["sections"]["swa"]
                self.assertEqual((base["cells"], base["pos_max"], swa["cells"], swa["pos_min"]),
                                 (self.t0, self.t0 - 1, 4, self.t0 - 4))
                self.assertEqual(base["sources"]["oracle"], self.t0 * 2)
        # a longer native state serves the base cache but not the attended sliding window
        _, later = self.native(self.t0 + 3)
        with self.assertRaisesRegex(ValueError, "oracle state lacks layer 0"):
            self.build(content_policy="oracle", fill_policy="oracle", oracle_state=later, qwen_state=None, mapper=None)

    def test_mapped_cells_equal_the_mapper_on_the_source_rows(self):
        _, native = self.native(self.t0)
        result = self.build(content_policy="mapped", fill_policy="oracle", oracle_state=native)
        mapper = Mapper(self.mapper_path)
        pairs = self.alignment.pair_mask("test") & (self.alignment.doc == self.alignment.document_ids.index(self.doc))
        src_of = dict(zip(self.alignment.tgt_position[pairs].tolist(), self.alignment.src_position[pairs].tolist()))
        for section in ("base", "swa"):
            cells = result.cells[section]
            positions = cells["positions"]
            expect_content = np.asarray([int(p) in src_of and src_of[int(p)] < self.n_q for p in positions])
            np.testing.assert_array_equal(cells["content"], expect_content)
            self.assertTrue(expect_content.any() and (section == "swa" or not expect_content.all()))
            at = np.flatnonzero(expect_content)
            src = np.asarray([src_of[int(p)] for p in positions[at]])
            for layer in result.state.state.part(section).layers:
                np.testing.assert_array_equal(cells["source"][layer] == handoff.SOURCE_NAMES.index("mapped"),
                                              expect_content)
                for kind in "kv":
                    sources = {s: self.rows("qwen", kind, s, self.n_q)[src] for s in (3, 7)}
                    expected = mapper.predict(kind, layer, sources, (2, 8))
                    if kind == "k" and MUSE_SPEC.layer_types[layer] == "sliding":
                        expected = kvstate.cache_keys(MUSE_SPEC, layer, expected, positions[at])
                    stored = self.stored(result, section, layer, kind)
                    np.testing.assert_allclose(stored[at], expected, atol=2e-5, rtol=0)
                    native_rows = np.asarray((native.state.part(section).stream().k if kind == "k"
                                              else native.state.part(section).stream().v)[layer])
                    fill = np.flatnonzero(~expect_content)
                    offset = int(native.state.part(section).stream().positions[0])
                    np.testing.assert_array_equal(stored[fill], native_rows[positions[fill] - offset])
        renormed = self.stored(result, "base", 3, "k")[np.flatnonzero(result.cells["base"]["content"])]
        np.testing.assert_allclose(np.sqrt(np.mean(renormed ** 2, axis=-1)), 1, rtol=1e-5)
        self.assertEqual(result.report["aligned_beyond_qwen_state"], 0)
        # aligned tokens whose Qwen partner is past the Qwen state take the fill policy
        short = self.build(content_policy="mapped", fill_policy="oracle", oracle_state=native,
                           qwen_state=self.qwen_state(self.n_q - 80))
        self.assertGreater(short.report["aligned_beyond_qwen_state"], 20)
        self.assertEqual(short.report["sections"]["base"]["content"],
                         result.report["sections"]["base"]["content"] - short.report["aligned_beyond_qwen_state"])

    def test_token_table_is_the_per_token_mean_of_non_aligned_train_rows(self):
        rows = np.concatenate([np.arange(self.muse_set.rows(d).start, self.muse_set.rows(d).stop)
                               for d, s in zip(self.alignment.document_ids, self.alignment.splits) if s == "train"])
        rows = rows[self.alignment.tgt_class[rows] != align.ALIGNED]
        arrays = self.table["arrays"]
        ids = self.muse_set.token_id[rows]
        np.testing.assert_array_equal(arrays["token_id"], np.unique(ids))
        np.testing.assert_array_equal(arrays["count"], [np.sum(ids == i) for i in arrays["token_id"]])
        self.assertIn(MUSE_IDS["<|start|>"], arrays["token_id"].tolist())
        for kind, layer in (("k", 0), ("v", 3), ("v", 7)):
            data = self.muse_arrays[kind, layer][rows].astype(np.float64)
            for index in (0, len(arrays["token_id"]) // 2, len(arrays["token_id"]) - 1):
                expected = data[ids == arrays["token_id"][index]].mean(axis=0)
                np.testing.assert_allclose(arrays[f"{kind}-{layer}"][index], expected, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(arrays[f"mean-{kind}-{layer}"], data.mean(axis=0), rtol=1e-6, atol=1e-6)
        chunked = handoff.build_token_table(self.muse_set, self.alignment, chunk_rows=7)
        for name, value in arrays.items():
            np.testing.assert_allclose(chunked["arrays"][name], value, rtol=1e-6, atol=1e-7)
        self.assertEqual(self.table["metadata"]["rows"], len(rows))
        table = handoff.TokenTable(self.table_path)
        looked, unseen = table.lookup("v", 3, [MUSE_IDS["<|start|>"], 255, 10**6])
        self.assertEqual(unseen.tolist(), [False, 255 not in arrays["token_id"], True])
        np.testing.assert_array_equal(looked[2], arrays["mean-v-3"])
        with self.assertRaises(FileExistsError):
            handoff.write_token_table(self.table_path, self.table)
        with self.assertRaisesRegex(ValueError, "alignment has no nope documents"):
            handoff.build_token_table(self.muse_set, self.alignment, split="nope")

    def test_token_table_refuses_target_captures_in_another_order(self):
        documents = self.muse_set.document_ids
        halves = documents[:5], documents[5:]
        paths = []
        for number, names in enumerate(halves):
            rows = np.concatenate([np.arange(self.muse_set.rows(d).start, self.muse_set.rows(d).stop) for d in names])
            entries = [{"document_id": d, **{k: v.tolist() for k, v in self.tokens["muse", d].items()}} for d in names]
            paths.append(write_synthetic_capture(self.root / f"muse-part{number}-{os.getpid()}", entries,
                                                 {key: value[rows] for key, value in self.muse_arrays.items()},
                                                 architecture=MUSE_SPEC.architecture,
                                                 layer_types=list(MUSE_SPEC.layer_types)))
        split = CaptureSet(paths)
        alignment = align.align(self.qwen_records, self.qwen_set, self.muse_records, split)
        rebuilt = handoff.build_token_table(split, alignment, chunk_rows=np.int64(1000))
        for name, value in self.table["arrays"].items():
            np.testing.assert_array_equal(rebuilt["arrays"][name], value)
        swapped = CaptureSet(paths[::-1])
        self.assertEqual(swapped.rows_total, split.rows_total)
        with self.assertRaisesRegex(ValueError, "different order"):
            handoff.build_token_table(swapped, alignment)
        shifted = align.Alignment(**{**alignment.__dict__, "tgt_row": alignment.tgt_row + 1})
        with self.assertRaisesRegex(ValueError, "outside their documents"):
            handoff.build_token_table(split, shifted)
        classes = alignment.tgt_class.copy()
        classes[split.rows(documents[0]).start] = align.EXCLUDED
        with self.assertRaisesRegex(ValueError, "row classes"):
            handoff.build_token_table(split, align.Alignment(**{**alignment.__dict__, "tgt_class": classes}))
        with self.assertRaisesRegex(ValueError, "lack aligned document"):
            handoff.build_token_table(CaptureSet([paths[0]]), align.Alignment(
                **{**alignment.__dict__, "tgt_class": alignment.tgt_class[:CaptureSet([paths[0]]).rows_total]}))
        with self.assertRaisesRegex(ValueError, "chunk_rows"):
            handoff.build_token_table(split, alignment, chunk_rows=0)

    def test_unseen_content_ids_take_the_content_mean(self):
        # Aligned content past a truncated Qwen state becomes fill; with every id unseen, those
        # content cells must take the mapper's content mean and template cells the table's mean.
        table, mapper = handoff.TokenTable(self.table_path), Mapper(self.mapper_path)
        original = table.lookup

        def all_unseen(kind, layer, token_ids):
            rows, _ = original(kind, layer, token_ids)
            mean = np.broadcast_to(table.mean(kind, layer), rows.shape).astype(np.float32)
            return mean.copy(), np.ones(len(token_ids), dtype=bool)

        table.lookup = all_unseen
        result = self.build(content_policy="mapped", fill_policy="table", table=table,
                            qwen_state=self.qwen_state(self.n_q - 80))
        checked = {"content": 0, "template": 0}
        for section, cells in result.cells.items():
            fill = np.flatnonzero(~cells["content"])
            content_class = cells["target_class"][fill] != align.TEMPLATE
            for layer in result.state.state.part(section).layers:
                stored = self.stored(result, section, layer, "v")[fill]
                expected = np.where(content_class[:, None, None], mapper.floor("v", layer, (2, 8)),
                                    table.mean("v", layer))
                np.testing.assert_array_equal(stored, expected.astype(np.float32))
                checked["content"] += int(content_class.sum())
                checked["template"] += int((~content_class).sum())
        self.assertGreater(checked["content"], 0)
        self.assertGreater(checked["template"], 0)

    def test_numpy_integer_arguments_are_accepted(self):
        plain = self.build(fill_policy="mean", swa_cells=5)
        numpy = self.build(fill_policy="mean", t0=np.int64(self.t0), swa_cells=np.int32(5), chunk_rows=np.uint16(3))
        self.assertEqual(numpy.state.to_bytes(), plain.state.to_bytes())
        self.assertIs(type(numpy.report["t0"]), int)
        self.assertIs(type(numpy.report["swa_cells"]), int)
        json.dumps(numpy.report)

    def test_table_and_mean_fills(self):
        table = handoff.TokenTable(self.table_path)
        result = self.build(content_policy="mapped", fill_policy="table", table=table)
        mapper = Mapper(self.mapper_path)
        for section in ("base", "swa"):
            cells = result.cells[section]
            fill = np.flatnonzero(~cells["content"])
            positions = cells["positions"][fill]
            token_ids = self.muse_tokens["token_id"][positions]
            content_class = cells["target_class"][fill] != align.TEMPLATE
            for layer in result.state.state.part(section).layers:
                values, unseen = table.lookup("v", layer, token_ids)
                # unseen ids of content tokens take the mapper's content mean, not the table's mean
                values[unseen & content_class] = mapper.floor("v", layer, values.shape[1:])
                np.testing.assert_array_equal(self.stored(result, section, layer, "v")[fill], values)
                keys, _ = table.lookup("k", layer, token_ids)
                keys[unseen & content_class] = mapper.floor("k", layer, keys.shape[1:])
                keys = handoff.renormalize_heads(keys)
                if MUSE_SPEC.layer_types[layer] == "sliding":
                    keys = kvstate.cache_keys(MUSE_SPEC, layer, keys, positions)
                np.testing.assert_array_equal(self.stored(result, section, layer, "k")[fill], keys)
                labels = cells["source"][layer][fill]
                np.testing.assert_array_equal(labels == handoff.SOURCE_NAMES.index("table_unseen"), unseen)
        counts = result.report["sections"]["base"]["sources"]
        self.assertEqual(counts["table"] + counts["table_unseen"], 2 * result.report["sections"]["base"]["fill"])
        floor = self.build(content_policy="mean", fill_policy="mean")
        for layer in (0, 3):
            section = "swa" if layer == 0 else "base"
            positions = floor.cells[section]["positions"]
            keys = np.broadcast_to(mapper.floor("k", layer, (2, 8)), (len(positions), 2, 8)).astype(np.float32)
            if layer == 0:
                keys = kvstate.cache_keys(MUSE_SPEC, layer, keys, positions)
            np.testing.assert_array_equal(self.stored(floor, section, layer, "k"), keys)
            np.testing.assert_array_equal(self.stored(floor, section, layer, "v")[0], mapper.mean("v", layer, (2, 8)))
        self.assertEqual(floor.report["sections"]["base"]["sources"]["mean"], 2 * self.t0)
        self.assertTrue(floor.report["floor"].startswith("harsher than v2_mean_all"))
        self.assertIsNone(result.report["floor"])
        _, native = self.native(self.t0)
        matched = self.build(content_policy="mean", fill_policy="oracle", oracle_state=native)
        self.assertTrue(matched.report["floor"].startswith("v2_mean_all"))
        fill = ~matched.cells["base"]["content"]
        np.testing.assert_array_equal(self.stored(matched, "base", 3, "v")[fill], self.rows("muse", "v", 3, self.t0)[fill])

    def test_native_prefix_takes_native_values_before_h(self):
        table = handoff.TokenTable(self.table_path)
        data, native = self.native(self.t0)
        oracle = self.build(content_policy="oracle", fill_policy="oracle", oracle_state=native, table=table,
                            qwen_state=None, mapper=None)
        whole = self.build(content_policy="mapped", fill_policy="table", table=table, oracle_state=native,
                           native_prefix=self.t0)
        self.assertEqual(whole.state.to_bytes(), oracle.state.to_bytes())
        self.assertEqual(whole.state.to_bytes(), data)
        self.assertEqual(whole.report["sections"]["base"]["sources"]["mapped"], 0)
        # H inside the history: [0, H) native from a state of exactly H tokens, [H, t0) as without a prefix
        h = self.t0 // 2
        _, prefix = self.native(h)
        plain = self.build(content_policy="mapped", fill_policy="table", table=table)
        hybrid = self.build(content_policy="mapped", fill_policy="table", table=table, oracle_state=prefix,
                            native_prefix=h)
        self.assertEqual(hybrid.report["native_prefix"], h)
        native_k = np.asarray(native.state.part("base").stream().k[3], dtype=np.float32)
        for section in ("base", "swa"):
            positions = hybrid.cells[section]["positions"]
            before, after = positions < h, positions >= h
            for layer in hybrid.state.state.part(section).layers:
                source = hybrid.cells[section]["source"][layer]
                self.assertTrue((source[before] == handoff.SOURCE_NAMES.index("oracle")).all())
                np.testing.assert_array_equal(source[after], plain.cells[section]["source"][layer][after])
                for kind in "kv":
                    np.testing.assert_array_equal(self.stored(hybrid, section, layer, kind)[after],
                                                  self.stored(plain, section, layer, kind)[after])
        np.testing.assert_array_equal(self.stored(hybrid, "base", 3, "k")[:h], native_k[:h])
        _, short = self.native(h - 1)
        with self.assertRaisesRegex(ValueError, "fewer than the native prefix"):
            self.build(content_policy="mapped", fill_policy="table", table=table, oracle_state=short,
                       native_prefix=h)
        with self.assertRaisesRegex(ValueError, "native prefix needs oracle_state"):
            self.build(content_policy="mapped", fill_policy="table", table=table, native_prefix=h)
        with self.assertRaises(ValueError):
            self.build(content_policy="mapped", fill_policy="table", table=table, oracle_state=native,
                       native_prefix=self.t0 + 1)
        # a mask gives the same cells as the equivalent prefix, and arbitrary positions work too
        mask = np.zeros(len(self.muse_tokens["token_id"]), dtype=bool)
        mask[:h] = True
        masked = self.build(content_policy="mapped", fill_policy="table", table=table, oracle_state=prefix,
                            native_mask=mask)
        self.assertEqual(masked.state.to_bytes(), hybrid.state.to_bytes())
        mask[:] = False
        mask[h:h + 5] = True
        middle = self.build(content_policy="mapped", fill_policy="table", table=table, oracle_state=native,
                            native_mask=mask)
        source = middle.cells["base"]["source"][3]
        self.assertEqual(middle.report["native_cells"], 5)
        self.assertTrue((source[h:h + 5] == handoff.SOURCE_NAMES.index("oracle")).all())
        np.testing.assert_array_equal(source[:h], plain.cells["base"]["source"][3][:h])
        with self.assertRaisesRegex(ValueError, "fewer than the native prefix"):
            self.build(content_policy="mapped", fill_policy="table", table=table, oracle_state=prefix,
                       native_mask=mask)
        with self.assertRaisesRegex(ValueError, "one entry per Muse token"):
            self.build(content_policy="mapped", fill_policy="table", table=table, oracle_state=native,
                       native_mask=mask[:-1])

    def test_sliding_cells_beyond_the_window_fall_back_and_short_prefixes_clip(self):
        _, native = self.native(self.t0)
        result = self.build(content_policy="oracle", fill_policy="oracle", oracle_state=native, swa_cells=7,
                            table=handoff.TokenTable(self.table_path), qwen_state=None, mapper=None)
        swa = result.report["sections"]["swa"]
        self.assertEqual((swa["cells"], swa["pos_min"], swa["oracle_fallback"]), (7, self.t0 - 7, 3 * 6))
        self.assertEqual(swa["sources"]["oracle"], 4 * 6)
        parsed = kvstate.parse_file(result.state.to_bytes(), MUSE_SPEC)
        np.testing.assert_array_equal(parsed.state.part("swa").stream().positions, np.arange(self.t0 - 7, self.t0))
        with self.assertRaisesRegex(ValueError, "need a table or a mapper"):
            self.build(content_policy="oracle", fill_policy="oracle", oracle_state=native, swa_cells=7,
                       qwen_state=None, mapper=None)
        floor = self.build(content_policy="oracle", fill_policy="oracle", oracle_state=native, swa_cells=7,
                           qwen_state=None)
        self.assertEqual(floor.report["sections"]["swa"]["sources"]["mean"], 3 * 6)
        _, early = self.native(3)
        clipped = self.build(t0=3, content_policy="oracle", fill_policy="oracle", oracle_state=early, swa_cells=10,
                             qwen_state=None, mapper=None)
        self.assertEqual(clipped.report["sections"]["swa"]["cells"], 3)

    def test_server_tokens_and_transposed_values_round_trip(self):
        result = self.build(content_policy="mapped", fill_policy="table", table=self.table_path, dtype="f16",
                            v_trans=True, server_tokens=True)
        parsed = kvstate.parse_file(result.state.to_bytes(), MUSE_SPEC)
        self.assertTrue(parsed.server_packed)
        np.testing.assert_array_equal(parsed.tokens, self.muse_tokens["token_id"][:self.t0])
        stream = parsed.state.part("base").stream()
        self.assertTrue(stream.v_trans)
        self.assertEqual(stream.k[3].dtype, np.float16)
        np.testing.assert_array_equal(stream.v[7], result.state.state.part("base").stream().v[7])

    def test_refusals(self):
        _, native = self.native(self.t0)
        bad_qwen = self.qwen_state()
        bad_qwen.token_words = bad_qwen.token_words.copy()
        bad_qwen.token_words[5] += 1
        other_muse = dict(self.muse_tokens, token_id=self.muse_tokens["token_id"].copy())
        other_muse["token_id"][2] += 1
        wrong_shape = write_mapper(self.root / f"mapper-shape-{os.getpid()}", [
            (kind, layer, [3], False, np.zeros((16, 8)), np.zeros(8), np.zeros(8))
            for kind in "kv" for layer in range(8)], source_shape=(1, 16), target_shape=(2, 4))
        partial = write_mapper(self.root / f"mapper-partial-{os.getpid()}", [
            ("k", 0, [3], False, np.zeros((16, 16)), np.zeros(16), np.zeros(16))], source_shape=(1, 16),
            target_shape=(2, 8))
        cases = [
            ({"qwen_state": bad_qwen}, "not a prefix of the rendered Qwen tokenization"),
            ({"qwen_state": self.qwen_state(len(self.qwen_tokens["token_id"]) + 0)}, None),
            ({"content_policy": "oracle", "fill_policy": "oracle"}, "needs oracle_state"),
            ({"fill_policy": "table"}, "needs a token table"),
            ({"content_policy": "mean", "fill_policy": "mean", "mapper": None}, "need a mapper"),
            ({"qwen_state": None}, "mapped policy needs qwen_state"),
            ({"content_policy": "copy"}, "content policy must be"),
            ({"t0": 0}, "t0 must be"), ({"t0": len(self.muse_tokens["token_id"]) + 1}, "t0 must be"),
            ({"t0": float(self.t0)}, "t0 must be"), ({"t0": True}, "t0 must be"),
            ({"t0": np.float64(self.t0)}, "t0 must be"), ({"t0": np.bool_(True)}, "t0 must be"),
            ({"chunk_rows": 0}, "chunk_rows must be"), ({"chunk_rows": 2.5}, "chunk_rows must be"),
            ({"swa_cells": np.int64(0)}, "swa_cells"),
            ({"swa_cells": 0, "fill_policy": "mean"}, "swa_cells"),
            ({"dtype": "q8_0", "fill_policy": "mean"}, "f16 or f32"),
            ({"fill_policy": "mean", "mapper": wrong_shape}, "target shape"),
            ({"fill_policy": "mean", "mapper": partial}, "mapper lacks maps"),
            ({"content_policy": "oracle", "fill_policy": "oracle", "oracle_state": native,
              "t0": self.t0 + 1}, "fewer than t0"),
            ({"fill_policy": "mean", "muse_spec": QWEN_SPEC}, "muse-glimmer"),
            ({"fill_policy": "mean", "muse_spec": None}, "muse_spec is required"),
        ]
        for options, fragment in cases:
            with self.subTest(fragment=fragment):
                if fragment is None:
                    self.build(fill_policy="mean", **options)   # a state of the whole document is fine
                    continue
                with self.assertRaisesRegex(ValueError, fragment):
                    self.build(**options)
        with self.assertRaisesRegex(ValueError, "oracle state tokens are not a prefix"):
            handoff.build_handoff(None, self.qwen_record, self.qwen_tokens, self.muse_record, other_muse, None,
                                  t0=self.t0, content_policy="oracle", fill_policy="oracle", oracle_state=native)
        with self.assertRaisesRegex(ValueError, "same document"):
            handoff.build_handoff(None, self.qwen_records[0], self.qwen_tokens, self.muse_record, self.muse_tokens,
                                  None, t0=self.t0, content_policy="oracle", fill_policy="oracle",
                                  oracle_state=native)
        cut = dict(self.muse_tokens, token_id=self.muse_tokens["token_id"][:-1],
                   byte_end=self.muse_tokens["byte_end"][:-1], special=self.muse_tokens["special"][:-1])
        with self.assertRaisesRegex(ValueError, "do not cover"):
            handoff.build_handoff(None, self.qwen_record, self.qwen_tokens, self.muse_record, cut, None,
                                  t0=self.t0, content_policy="oracle", fill_policy="oracle", oracle_state=native)
        with self.assertRaisesRegex(ValueError, "nondecreasing byte ends"):
            handoff.normalize_tokens(dict(self.muse_tokens, byte_end=self.muse_tokens["byte_end"][::-1]), "Muse")
        with self.assertRaisesRegex(ValueError, "nondecreasing byte ends"):
            handoff.normalize_tokens({"token_id": [1], "byte_end": [0], "special": [0]}, "Muse")
        # a token that spells nothing repeats the byte end before it (an emptied SentencePiece space)
        empty = handoff.normalize_tokens({"token_id": [1, 2, 3], "byte_end": [3, 3, 5], "special": [1, 0, 0]}, "Jamba")
        np.testing.assert_array_equal(empty["byte_end"], [3, 3, 5])
        inside = dict(self.muse_record, text=self.muse_record["text"])
        tokens = dict(self.muse_tokens, special=self.muse_tokens["special"].copy())
        start = self.muse_record["segments"][0][1]
        tokens["special"][int(np.searchsorted(tokens["byte_end"], start, side="right")) + 1] = 1
        with self.assertRaisesRegex(ValueError, "special token inside a segment"):
            handoff.align_document(self.qwen_record, self.qwen_tokens, inside, tokens)
        half = self.qwen_state()
        stream = half.state.part("attn").stream()
        stream.positions = stream.positions.copy()
        stream.positions[3] = stream.positions[4]
        with self.assertRaisesRegex(ValueError, "cover positions"):
            handoff.QwenSource(half, handoff.normalize_tokens(self.qwen_tokens, "Qwen"))


@contextlib.contextmanager
def quiet_descriptors():
    """Send fds 1 and 2 to /dev/null (probe subprocess logs); Python-level redirects still work."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in saved + [null]:
            os.close(fd)


def probe_run(mode, model, records, output, *extra, batch=16):
    docs = output.parent / f"{output.name}.jsonl"
    docs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    command = [PROBE, "--mode", mode, "--model", str(model), "--input", str(docs), "--output", str(output),
               "--ctx", "1024", "--threads", "2", "--batch", str(batch), *map(str, extra)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise AssertionError(result.stderr[-3000:])
    return output


DEPLOY = ["--cache-type", "f16", "--flash-attn", "on"]


@unittest.skipUnless(PROBE, "set KVPARK_KV_PROBE to run the handoff on tiny fixtures")
class ProbeHandoffTests(unittest.TestCase):
    """Real save-state (Qwen and Muse), a fitted tiny mapper, handoff build, seed-score."""

    @classmethod
    def setUpClass(cls):
        from tiny_fixture import create_fixture
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-handoff-probe-")
        root = cls.root = Path(cls.temp.name)
        cls.qwen_model, cls.muse_model = root / "qwen.gguf", root / "muse.gguf"
        create_fixture(cls.qwen_model, "qwen35", control_tokens=QWEN_CONTROL, chat_template=QWEN_TEMPLATE)
        create_fixture(cls.muse_model, "muse-glimmer", control_tokens=MUSE_CONTROL, chat_template=MUSE_TEMPLATE,
                       bos_token=MUSE_BOS, add_bos=True, sliding_window=16)
        cls.qwen_spec = kvstate.ModelSpec.from_gguf(cls.qwen_model)
        cls.muse_spec = kvstate.ModelSpec.from_gguf(cls.muse_model)
        qwen_records, muse_records = render_records(range(len(SPLITS)))
        for name, records in (("qwen", qwen_records), ("muse", muse_records)):
            (root / f"{name}-records.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        cls.qwen_capture = probe_run("collect", cls.qwen_model, qwen_records, root / "qwen-capture", batch=64)
        cls.muse_capture = probe_run("collect", cls.muse_model, muse_records, root / "muse-capture", batch=64)
        alignment = align.align(qwen_records, CaptureSet([cls.qwen_capture]), muse_records,
                                CaptureSet([cls.muse_capture]))
        cls.alignment_path = root / "alignment.npz"
        alignment.save(cls.alignment_path)
        cls.mapper = root / "mapper"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fit_layers.main(["--src-capture", str(cls.qwen_capture), "--tgt-capture", str(cls.muse_capture),
                             "--alignment", str(cls.alignment_path), "--output", str(cls.mapper), "--ks", "1",
                             "--threads", "1"])
        cls.table = root / "table.npz"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            handoff.main(["token-table", "--target-capture", str(cls.muse_capture), "--alignment",
                          str(cls.alignment_path), "--output", str(cls.table), "--chunk-rows", "100"])
        number = SPLITS.index("test")
        cls.qwen_record, cls.muse_record = qwen_records[number], muse_records[number]
        cls.doc = cls.muse_record["document_id"]
        tokens = {}
        for name, path in (("qwen", cls.qwen_capture), ("muse", cls.muse_capture)):
            capture = Capture(path)
            rows = capture.rows(cls.doc)
            tokens[name] = {key: np.array(getattr(capture, key)[rows]) for key in ("token_id", "byte_end", "special")}
        cls.qwen_tokens, cls.muse_tokens = tokens["qwen"], tokens["muse"]
        cls.P = handoff.handoff_position(cls.muse_record, cls.muse_tokens)
        cls.n_q = handoff.handoff_position(cls.qwen_record, cls.qwen_tokens)
        cls.t0 = (cls.P - 32) // 16 * 16            # on the --batch 16 grid: seeded ubatches repeat native ones
        cls.t0_late = cls.P - 1                    # tail of one token, off the grid
        states = {}
        for layout, options in (("deploy", DEPLOY), ("f32", ["--cache-type", "f32", "--flash-attn", "off"])):
            probe_run("save-state", cls.qwen_model, [{"document_id": cls.doc, "text": cls.qwen_record["text"],
                                                      "save_at": [cls.n_q]}], root / f"qwen-states-{layout}", *options)
            states["qwen", layout] = root / f"qwen-states-{layout}" / "states" / cls.doc / f"{cls.n_q}.state"
        probe_run("save-state", cls.muse_model, [{"document_id": cls.doc, "text": cls.muse_record["text"],
                                                  "save_at": [cls.t0, cls.t0_late]}], root / "muse-states", *DEPLOY)
        for t0 in (cls.t0, cls.t0_late):
            states["muse", t0] = root / "muse-states" / "states" / cls.doc / f"{t0}.state"
        cls.states = states

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def build(self, name, *, t0=None, qwen="deploy", oracle=None, **options):
        t0 = self.t0 if t0 is None else t0
        oracle_state = kvstate.read_file(self.states["muse", oracle], self.muse_spec) if oracle else None
        arguments = {"t0": t0, "muse_spec": self.muse_spec, "oracle_state": oracle_state, "table": self.table}
        arguments.update(options)
        result = handoff.build_handoff(kvstate.read_file(self.states["qwen", qwen], self.qwen_spec), self.qwen_record,
                                       self.qwen_tokens, self.muse_record, self.muse_tokens, self.mapper, **arguments)
        path = self.root / f"{name}.state"
        kvstate.write_file(path, result.state)
        return result, path

    def seed_score(self, name, conditions):
        start, end = handoff.final_message_start(self.muse_record), len(self.muse_record["text"].encode("utf-8"))
        record = {"document_id": self.doc, "text": self.muse_record["text"],
                  "expected_token_sha256": hashlib.sha256(
                      self.muse_tokens["token_id"].astype("<i4").tobytes()).hexdigest(),
                  "score_ranges": [[start, end]],
                  "conditions": [{"name": "native", "state_file": None}]
                  + [{"name": n, "state_file": str(p)} for n, p in conditions]}
        output = probe_run("seed-score", self.muse_model, [record], self.root / name, *DEPLOY, "--max-scored", "24")
        document = json.loads((output / "results.json").read_text())["documents"][0]
        scores = {c["name"]: np.load(output / "scores" / self.doc / f"{c['name']}.npy") for c in document["conditions"]}
        return {c["name"]: c for c in document["conditions"]}, scores

    def test_handoff_states_restore_and_score(self):
        self.assertEqual(self.t0 % 16, 0)
        self.assertGreater(self.t0, 2 * self.muse_spec.n_swa)
        oracle, oracle_path = self.build("oracle", oracle=self.t0, content_policy="oracle", fill_policy="oracle")
        # oracle/oracle is the native save, byte for byte
        self.assertEqual(oracle_path.read_bytes(), self.states["muse", self.t0].read_bytes())
        late, late_path = self.build("oracle-late", t0=self.t0_late, oracle=self.t0_late, content_policy="oracle",
                                     fill_policy="oracle", swa_cells=24)
        self.assertEqual(late.report["sections"]["swa"]["oracle_fallback"], 8 * 3)
        mapped_oracle = self.build("mapped-oracle", oracle=self.t0, content_policy="mapped", fill_policy="oracle")
        mapped_table = self.build("mapped-table", content_policy="mapped", fill_policy="table")
        mean_mean = self.build("mean-mean", content_policy="mean", fill_policy="mean")
        for result, _ in (mapped_oracle, mapped_table, mean_mean):
            self.assertGreater(result.report["sections"]["base"]["content"], 50)
            self.assertGreater(result.report["sections"]["base"]["fill"], 10)
        by_name, scores = self.seed_score("scores", [
            ("oracle", oracle_path), ("oracle_late", late_path), ("mapped_oracle", mapped_oracle[1]),
            ("mapped_table", mapped_table[1]), ("mean_mean", mean_mean[1])])
        np.testing.assert_array_equal(scores["oracle"], scores["native"])
        self.assertEqual(by_name["oracle"]["kl_mean"], 0)
        # 24 SWA cells (8 unattended, filled from the table), seed off the grid: summation-order noise only
        self.assertEqual((by_name["oracle_late"]["restored_pos_min"], by_name["oracle_late"]["seed_tokens"]),
                         (self.t0_late - 24, self.t0_late))
        self.assertLess(float(scores["oracle_late"][:, 1].max()), 1e-9)
        self.assertEqual(by_name["oracle_late"]["top1_agree"], 1)
        for name in ("mapped_oracle", "mapped_table", "mean_mean"):
            self.assertEqual((by_name[name]["seed_tokens"], by_name[name]["restored_pos_min"],
                              by_name[name]["restored_pos_max"]), (self.t0, self.t0 - 16, self.t0 - 1))
            self.assertTrue(np.isfinite(scores[name]).all())
            self.assertGreater(by_name[name]["kl_mean"], 0)
        self.assertEqual(len(scores["native"]), 24)

    def test_mapped_rows_match_the_mapper_on_captured_qwen_rows(self):
        result, _ = self.build("mapped-f32", qwen="f32", content_policy="mapped", fill_policy="table", dtype="f32")
        mapper = Mapper(self.mapper)
        alignment = align.load_alignment(self.alignment_path)
        number = alignment.document_ids.index(self.doc)
        pairs = alignment.doc == number
        src_of = dict(zip(alignment.tgt_position[pairs].tolist(), alignment.src_position[pairs].tolist()))
        qwen_set = CaptureSet([self.qwen_capture])
        start = qwen_set.rows(self.doc).start
        for section in ("base", "swa"):
            cells = result.cells[section]
            at = np.flatnonzero(cells["content"])
            positions = cells["positions"][at]
            src = start + np.asarray([src_of[int(p)] for p in positions])
            stream = result.state.state.part(section).stream()
            for layer in result.state.state.part(section).layers:
                for kind in "kv":
                    expected = mapper.predict(kind, layer, {3: qwen_set.gather(kind, 3, src)}, (2, 32))
                    stored = (stream.k if kind == "k" else stream.v)[layer][at]
                    if kind == "k" and self.muse_spec.layer_types[layer] == "sliding":
                        stored = kvstate.pre_rope_keys(self.muse_spec, layer, stored, positions)
                    np.testing.assert_allclose(stored, expected, atol=5e-5, rtol=0)

    def test_command_line_writes_a_server_slot_file(self):
        output = self.root / "cli" / "slot.bin"
        output.parent.mkdir()
        argv = ["build", "--qwen-state", str(self.states["qwen", "deploy"]), "--qwen-model", str(self.qwen_model),
                "--muse-model", str(self.muse_model), "--qwen-records", str(self.root / "qwen-records.jsonl"),
                "--muse-records", str(self.root / "muse-records.jsonl"), "--document-id", self.doc,
                "--probe", PROBE, "--tail", str(self.P - self.t0), "--mapper", str(self.mapper),
                "--table", str(self.table), "--content", "mapped", "--fill", "table", "--server-tokens",
                "--resume-sidecar", "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()) as stdout, quiet_descriptors():
            self.assertEqual(handoff.main(argv), 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["t0"], self.t0)
        report = json.loads(output.with_name("slot.bin.report.json").read_text())
        self.assertEqual((report["tail"], report["handoff_position"], report["bytes"]),
                         (self.P - self.t0, self.P, output.stat().st_size))
        parsed = kvstate.read_file(output, self.muse_spec)
        self.assertTrue(parsed.server_packed)
        np.testing.assert_array_equal(parsed.tokens, self.muse_tokens["token_id"][:self.t0])
        self.assertEqual(kvstate.parse_resume(output.with_name("slot.bin.resume").read_bytes()).checkpoints, [])
        api, _ = self.build("api", content_policy="mapped", fill_policy="table", server_tokens=True)
        self.assertEqual(api.state.to_bytes(), output.read_bytes())
        by_name, _ = self.seed_score("cli-score", [("cli", output)])
        self.assertTrue(by_name["cli"]["server_packed_tokens"])
        with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
            handoff.main(argv)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            handoff.main(argv[:-2] + ["--t0", "5", "--output", str(output) + "2"])


if __name__ == "__main__":
    unittest.main()
