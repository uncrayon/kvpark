"""Capture loading, token alignment and collect_v2 orchestration checks with fakes."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

import align as align_module
from align import ALIGNED, EXCLUDED, STRADDLING, TEMPLATE, UNMATCHED, align, load_alignment
from captures import Capture, CaptureSet, default_layer_types, open_capture, write_synthetic_capture
from chat import render, rendered_record
import collect_v2
from collect import sha256_file
from test_chat import MUSE_BOS, MUSE_LIKE, QWEN_LIKE, TEMPLATE_DIR, conversation


EXPERIMENT = Path(__file__).resolve().parent
SPECIAL = re.compile(r"<\|[a-z_]+\|>")


def doc_id(number):
    return hashlib.sha256(str(number).encode()).hexdigest()


def fake_tokenize(text, splitter):
    """Specials as single tokens; other text split by splitter(str) -> pieces."""
    ends, special, offset, position = [], [], 0, 0
    for match in list(SPECIAL.finditer(text)) + [None]:
        stop = match.start() if match else len(text)
        for piece in splitter(text[position:stop]):
            offset += len(piece.encode("utf-8"))
            ends.append(offset)
            special.append(0)
        if match:
            offset += len(match.group().encode("utf-8"))
            ends.append(offset)
            special.append(1)
            position = match.end()
    assert offset == len(text.encode("utf-8"))
    return ends, special


def words(text):
    return re.findall(r"\s+|\w+|[^\w\s]", text)


def byte_pair_tokenize(text):
    ends, special, offset, position = [], [], 0, 0
    for match in list(SPECIAL.finditer(text)) + [None]:
        stop = match.start() if match else len(text)
        size = len(text[position:stop].encode("utf-8"))
        ends.extend(offset + end for end in range(2, size, 2))
        special.extend([0] * len(range(2, size, 2)))
        if size:
            ends.append(offset + size)
            special.append(0)
        offset += size
        if match:
            offset += len(match.group().encode("utf-8"))
            ends.append(offset)
            special.append(1)
            position = match.end()
    return ends, special


def reference_alignment(src_record, src_tokens, tgt_record, tgt_tokens):
    """Direct loop implementation of the spec rule for one document."""
    def keyed(record, ends):
        result, starts = {}, [0] + ends[:-1]
        for row, (start, end) in enumerate(zip(starts, ends)):
            for index, seg_start, seg_end in record["segments"]:
                if seg_start <= start and end <= seg_end:
                    result[(index, end - seg_start)] = row
        return result
    src, tgt = keyed(src_record, src_tokens[0]), keyed(tgt_record, tgt_tokens[0])
    return sorted((src[key], tgt[key]) for key in src.keys() & tgt.keys())


def sample_conversations():
    first = conversation("train")
    second = conversation("test")
    second["messages"][1]["content"] = "Und morgen? Ça va — 東京 too."
    return [first, second]


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rng = np.random.default_rng(3)

    def documents(self, first=0, sizes=(3, 5)):
        return [{"document_id": doc_id(first + number), "byte_end": list(range(2, 2 * size + 1, 2)),
                 "token_id": list(range(size)), "special": [1] + [0] * (size - 1)}
                for number, size in enumerate(sizes)]

    def arrays(self, rows, layers=(3, 7)):
        return {(key, layer): self.rng.normal(size=(rows, 2, 4)).astype(np.float32)
                for key in ("k", "v") for layer in layers}

    def test_synthetic_capture_round_trip_and_derived_arrays(self):
        arrays = self.arrays(8)
        path = write_synthetic_capture(self.root / "cap", self.documents(), arrays, n_layer=8)
        capture = open_capture(path)
        self.assertEqual(capture.report["rows"], 8)
        np.testing.assert_array_equal(capture.byte_start, [0, 2, 4, 0, 2, 4, 6, 8])
        np.testing.assert_array_equal(capture.position, [0, 1, 2, 0, 1, 2, 3, 4])
        np.testing.assert_array_equal(capture.doc_index, [0, 0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(capture.rows(doc_id(1)), slice(3, 8))
        self.assertEqual(capture.layers("k"), [3, 7])
        self.assertEqual(capture.keys(), ["k", "v"])
        value = capture.array("v", 7)
        self.assertIsInstance(value, np.memmap)
        self.assertEqual(value.dtype.str, "<f4")
        np.testing.assert_array_equal(value, arrays["v", 7])
        self.assertEqual(capture.attention_layers(), list(range(8)))
        self.assertIsNone(capture.manifest)
        with self.assertRaises(KeyError):
            capture.array("q", 3)
        with self.assertRaises(FileExistsError):
            write_synthetic_capture(path, self.documents(), arrays, n_layer=8)

    def test_handwritten_npy_v1_header_as_the_cpp_writer_produces(self):
        path = write_synthetic_capture(self.root / "cap", self.documents(), self.arrays(8), n_layer=8)
        values = self.rng.normal(size=(8, 2, 4)).astype("<f4")
        header = "{'descr': '<f4', 'fortran_order': False, 'shape': (8, 2, 4), }"
        total = len(header) + 10 + 1
        header += " " * (-total % 64) + "\n"
        blob = b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header.encode() + values.tobytes()
        self.assertEqual((len(blob) - values.nbytes) % 64, 0)
        (path / "k-3.npy").write_bytes(blob)
        np.testing.assert_array_equal(open_capture(path).array("k", 3), values)

    def test_invalid_captures_are_rejected(self):
        def fresh(name, **options):
            return write_synthetic_capture(self.root / name, self.documents(), self.arrays(8),
                                           n_layer=8, **options)

        def edit_report(path, change):
            report = json.loads((path / "report.json").read_text())
            change(report)
            (path / "report.json").write_text(json.dumps(report))

        cases = {
            "gap": lambda p: edit_report(p, lambda r: r["documents"][1].update(first_row=4)),
            "rows": lambda p: edit_report(p, lambda r: r.update(rows=9)),
            "version": lambda p: edit_report(p, lambda r: r.update(format_version=1)),
            "recurrent": lambda p: edit_report(p, lambda r: r["layers"][3].update(attention="recurrent")),
            "file": lambda p: edit_report(p, lambda r: r["captures"][0].update(file="x.npy")),
            "duplicate": lambda p: edit_report(p, lambda r: r["captures"].append(r["captures"][0])),
            "dup_doc": lambda p: edit_report(p, lambda r: r["documents"][1].update(
                document_id=r["documents"][0]["document_id"])),
            "float64": lambda p: np.save(p / "k-3.npy", np.zeros((8, 2, 4))),
            "fortran": lambda p: np.save(p / "k-3.npy", np.asfortranarray(np.zeros((8, 2, 4), "<f4"))),
            "shape": lambda p: np.save(p / "k-3.npy", np.zeros((8, 4, 2), "<f4")),
            "token_dtype": lambda p: np.save(p / "token_id.npy", np.zeros(8, np.int64)),
            "byte_end": lambda p: np.save(p / "byte_end.npy", np.asarray([2, 2, 4, 2, 4, 6, 8, 10], "<i8")),
            "special": lambda p: np.save(p / "special.npy", np.full(8, 2, np.uint8)),
            "missing": lambda p: (p / "v-7.npy").unlink(),
            "symlink": lambda p: ((p / "k-3.npy").rename(p / "real.npy"),
                                  (p / "k-3.npy").symlink_to(p / "real.npy")),
        }
        for name, corrupt in cases.items():
            path = fresh(name)
            corrupt(path)
            with self.subTest(case=name), self.assertRaises(ValueError):
                open_capture(path)

    def test_deepseek4_compressor_captures(self):
        # kv_probe's deepseek4 row: K = the latent (kv_norm), and per compressed layer the compressor
        # inputs [a | s] of its pools: CSA and LID on compress-4 layers, HCA on compress-128 layers
        def write(name, keys, compress):
            arrays = {(key, layer): self.rng.normal(size=(8, 1, 8)).astype(np.float32) for key, layer in keys}
            path = write_synthetic_capture(self.root / name, self.documents(), arrays, architecture="deepseek4",
                                           layer_types=["sliding"] * 8)
            report = json.loads((path / "report.json").read_text())
            for layer, ratio in compress.items():
                report["layers"][layer]["compress"] = ratio
            (path / "report.json").write_text(json.dumps(report))
            return path
        good = {3: 4, 7: 128, 0: 0}
        capture = open_capture(write("ok", [("k", 3), ("csa", 3), ("lid", 3), ("hca", 7)], good))
        self.assertEqual((capture.architecture, capture.compress), ("deepseek4", {3: 4, 7: 128}))
        self.assertEqual(capture.keys(), ["csa", "hca", "k", "lid"])
        for name, keys, compress in (("hca-on-csa", [("hca", 3)], good), ("csa-on-hca", [("csa", 7)], good),
                                     ("window-only", [("lid", 0)], good), ("ratio", [("k", 3)], {3: 5}),
                                     ("text", [("k", 3)], {3: "4"})):
            with self.subTest(case=name), self.assertRaises(ValueError):
                open_capture(write(name, keys, compress))

    def test_tokenize_mode_and_checks(self):
        path = write_synthetic_capture(self.root / "tok", self.documents(), mode="tokenize",
                                       architecture="qwen35", n_layer=8)
        capture = open_capture(path)
        self.assertEqual(capture.keys(), [])
        self.assertEqual(capture.attention_layers(), [3, 7])
        records = [{"document_id": doc_id(0), "text": "abcdef"}, {"document_id": doc_id(1), "text": "x" * 10}]
        capture.check_documents(records)
        with self.assertRaises(ValueError):
            capture.check_documents(records[::-1])
        with self.assertRaises(ValueError):
            capture.check_documents([records[0], {**records[1], "text": "short"}])

    def test_check_finite(self):
        arrays = self.arrays(8)
        arrays["k", 7][5, 1, 2] = np.nan
        path = write_synthetic_capture(self.root / "nan", self.documents(), arrays, n_layer=8)
        with self.assertRaisesRegex(ValueError, "k-7"):
            open_capture(path).check_finite(chunk_rows=3)

    def test_capture_set_flattens_rows_and_gathers(self):
        first_arrays, second_arrays = self.arrays(8), self.arrays(4)
        manifest = {"model_sha256": "a" * 64}
        first = write_synthetic_capture(self.root / "a", self.documents(), first_arrays, n_layer=8,
                                        manifest=manifest)
        second = write_synthetic_capture(self.root / "b", self.documents(10, (4,)), second_arrays,
                                         n_layer=8, manifest=manifest)
        captures = CaptureSet([first, second])
        self.assertEqual(captures.rows_total, 12)
        self.assertEqual(captures.rows(doc_id(10)), slice(8, 12))
        self.assertEqual(captures.document_ids, [doc_id(0), doc_id(1), doc_id(10)])
        np.testing.assert_array_equal(captures.position, [0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3])
        gathered = captures.gather("v", 3, [9, 0, 7])
        np.testing.assert_array_equal(gathered[0], second_arrays["v", 3][1])
        np.testing.assert_array_equal(gathered[1], first_arrays["v", 3][0])
        np.testing.assert_array_equal(gathered[2], first_arrays["v", 3][7])
        with self.assertRaises(IndexError):
            captures.gather("v", 3, [12])
        with self.assertRaisesRegex(ValueError, "two captures"):
            CaptureSet([first, first])
        other = write_synthetic_capture(self.root / "c", self.documents(20), self.arrays(8, (3,)), n_layer=8,
                                        manifest=manifest)
        with self.assertRaisesRegex(ValueError, "layout"):
            CaptureSet([first, other])
        foreign = write_synthetic_capture(self.root / "d", self.documents(30), self.arrays(8), n_layer=8,
                                          manifest={"model_sha256": "b" * 64})
        with self.assertRaisesRegex(ValueError, "same model"):
            CaptureSet([first, foreign])

    def test_capture_set_accepts_extra_keys_on_some_captures(self):
        # The real run pairs a Muse train capture {k, v} with a dev/test capture {k, v, q, gate}.
        manifest = {"model_sha256": "a" * 64}
        base = self.arrays(8)
        train = write_synthetic_capture(self.root / "train", self.documents(0), base, n_layer=8,
                                        manifest=manifest)
        rows = len(next(iter(base.values())))
        extra = {**base, ("q", 3): np.ones((rows, 4, 2), np.float32)}
        evaluation = write_synthetic_capture(self.root / "eval", self.documents(40), extra, n_layer=8,
                                             manifest=manifest)
        captures = CaptureSet([train, evaluation])
        self.assertEqual(captures.layers("v"), sorted(layer for key, layer in base if key == "v"))
        self.assertEqual(captures.layers("q"), [])
        self.assertEqual(CaptureSet([evaluation]).layers("q"), [3])

    def test_default_layer_tables(self):
        self.assertEqual(default_layer_types("muse-glimmer", 8).count("full"), 2)
        self.assertEqual([i for i, kind in enumerate(default_layer_types("qwen35", 64)) if kind == "full"],
                         list(range(3, 64, 4)))
        self.assertEqual(default_layer_types("qwen3", 36), ["full"] * 36)
        with self.assertRaisesRegex(ValueError, "unknown architecture"):
            default_layer_types("llama", 4)

    def test_dense_qwen3_capture_has_every_layer_as_attention(self):
        arrays = {(key, layer): self.rng.normal(size=(8, 2, 4)).astype(np.float32)
                  for key in ("k", "v") for layer in range(4)}
        path = write_synthetic_capture(self.root / "cap", self.documents(), arrays, architecture="qwen3", n_layer=4)
        capture = open_capture(path)
        self.assertEqual((capture.architecture, capture.layer_types), ("qwen3", ["full"] * 4))
        self.assertEqual(capture.attention_layers(), [0, 1, 2, 3])
        self.assertEqual(CaptureSet([capture]).layers("k"), [0, 1, 2, 3])


class AlignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.counter = 0

    def build(self, conversations, src_tokenizer, tgt_tokenizer):
        src_records, tgt_records, src_tokens, tgt_tokens = [], [], [], []
        for conv in conversations:
            src_records.append(rendered_record(conv, "qwen", render(conv, QWEN_LIKE)))
            tgt_records.append(rendered_record(conv, "muse", render(conv, MUSE_LIKE, bos_token=MUSE_BOS)))
            src_tokens.append(src_tokenizer(src_records[-1]["text"]))
            tgt_tokens.append(tgt_tokenizer(tgt_records[-1]["text"]))
        return src_records, tgt_records, src_tokens, tgt_tokens

    def capture(self, records, tokens, architecture):
        self.counter += 1
        documents = [{"document_id": record["document_id"], "byte_end": ends, "special": special}
                     for record, (ends, special) in zip(records, tokens)]
        rows = sum(len(ends) for ends, _ in tokens)
        arrays = {("k", 3): np.zeros((rows, 1, 2), np.float32)}
        path = write_synthetic_capture(self.root / f"cap{self.counter}", documents, arrays,
                                       architecture=architecture, n_layer=4)
        return CaptureSet([path])

    def test_alignment_matches_reference_rule_and_classifies_tokens(self):
        src_records, tgt_records, src_tokens, tgt_tokens = self.build(
            sample_conversations(), lambda text: fake_tokenize(text, words), byte_pair_tokenize)
        src_set = self.capture(src_records, src_tokens, "qwen35")
        tgt_set = self.capture(tgt_records, tgt_tokens, "muse-glimmer")
        result = align({r["document_id"]: r for r in src_records}, src_set, tgt_records, tgt_set)
        self.assertEqual(result.document_ids, [r["document_id"] for r in tgt_records])
        self.assertEqual(result.splits, ["train", "test"])
        self.assertEqual(result.rejected, [])
        for number, record in enumerate(tgt_records):
            src_rows, tgt_rows = src_set.rows(record["document_id"]), tgt_set.rows(record["document_id"])
            expected = reference_alignment(src_records[number], src_tokens[number], record, tgt_tokens[number])
            chosen = result.doc == number
            actual = sorted(zip((result.src_row[chosen] - src_rows.start).tolist(),
                                (result.tgt_row[chosen] - tgt_rows.start).tolist()))
            self.assertEqual(actual, expected)
            self.assertGreater(len(expected), 20)
            np.testing.assert_array_equal(result.tgt_position[chosen], result.tgt_row[chosen] - tgt_rows.start)
            np.testing.assert_array_equal(result.src_position[chosen], result.src_row[chosen] - src_rows.start)
            self.assertTrue(np.all(np.diff(result.tgt_row[chosen]) > 0))
        encoded_src = [r["text"].encode() for r in src_records]
        spans = [{j: (s, e) for j, s, e in r["segments"]} for r in tgt_records]
        for pair in range(len(result.src_row)):
            number, segment, rel = int(result.doc[pair]), int(result.segment[pair]), int(result.rel_end[pair])
            src_row = int(result.src_row[pair])
            src_offset = src_set.rows(src_records[number]["document_id"]).start
            src_start = dict((j, s) for j, s, _ in src_records[number]["segments"])[segment]
            self.assertEqual(int(src_set.byte_end[src_row]), src_start + rel)
            tgt_row = int(result.tgt_row[pair])
            self.assertEqual(int(tgt_set.byte_end[tgt_row]), spans[number][segment][0] + rel)
            self.assertGreaterEqual(src_row, src_offset)
        self.assertTrue(len(encoded_src))
        classes = result.tgt_class
        self.assertEqual(set(np.unique(classes).tolist()), {TEMPLATE, ALIGNED, UNMATCHED, STRADDLING})
        self.assertEqual(int(np.sum(classes == ALIGNED)), len(result.tgt_row))
        self.assertTrue(np.all(classes[tgt_set.special.astype(bool)] == TEMPLATE))
        # Qwen tokens of the tool-call assistant content have no Muse counterpart.
        dropped = [index for index, key in enumerate(src_records[0]["segment_keys"]) if key == "2:content"][0]
        start, end = [(s, e) for j, s, e in src_records[0]["segments"] if j == dropped][0]
        rows = src_set.rows(src_records[0]["document_id"])
        inside = ((src_set.byte_start[rows] >= start) & (src_set.byte_end[rows] <= end))
        self.assertTrue(inside.any())
        self.assertTrue(np.all(result.src_class[rows][inside] == UNMATCHED))
        self.assertEqual(int(result.pair_mask("test").sum()), int(np.sum(result.doc == 1)))
        summary = result.summary()
        self.assertEqual(summary["pairs"], len(result.src_row))
        json.dumps(summary, allow_nan=False)

    def test_identical_tokenization_aligns_every_content_token(self):
        splitter = lambda text: fake_tokenize(text, words)
        src_records, tgt_records, src_tokens, tgt_tokens = self.build(sample_conversations(), splitter, splitter)
        src_set = self.capture(src_records, src_tokens, "qwen35")
        tgt_set = self.capture(tgt_records, tgt_tokens, "muse-glimmer")
        result = align(src_records, src_set, tgt_records, tgt_set)
        classes = result.tgt_class
        self.assertEqual(int(np.sum(classes == UNMATCHED)), 0)
        self.assertEqual(int(np.sum(classes == STRADDLING)), 0)
        self.assertGreater(int(np.sum(classes == ALIGNED)), 50)

    def test_special_token_inside_segment_rejects_document(self):
        conversations = sample_conversations()
        conversations[0]["messages"][1]["content"] = "Please print <|eot|> literally."
        splitter = lambda text: fake_tokenize(text, words)
        src_records, tgt_records, src_tokens, tgt_tokens = self.build(conversations, splitter, splitter)
        src_set = self.capture(src_records, src_tokens, "qwen35")
        tgt_set = self.capture(tgt_records, tgt_tokens, "muse-glimmer")
        result = align(src_records, src_set, tgt_records, tgt_set)
        self.assertEqual(result.document_ids, [tgt_records[1]["document_id"]])
        self.assertEqual(result.rejected, [{"document_id": tgt_records[0]["document_id"],
                                            "reason": "special token inside a segment",
                                            "models": ["source", "target"]}])
        self.assertTrue(np.all(result.tgt_class[tgt_set.rows(tgt_records[0]["document_id"])] == EXCLUDED))
        self.assertTrue(np.all(result.doc == 0))

    def test_inconsistent_inputs_are_rejected(self):
        splitter = lambda text: fake_tokenize(text, words)
        src_records, tgt_records, src_tokens, tgt_tokens = self.build(sample_conversations(), splitter, splitter)
        src_set = self.capture(src_records, src_tokens, "qwen35")
        tgt_set = self.capture(tgt_records, tgt_tokens, "muse-glimmer")
        with self.assertRaisesRegex(ValueError, "different documents"):
            align(src_records[:1], src_set, tgt_records, tgt_set)
        changed = [dict(tgt_records[0], segment_keys=tgt_records[0]["segment_keys"][:-1]), tgt_records[1]]
        with self.assertRaisesRegex(ValueError, "segment keys"):
            align(src_records, src_set, changed, tgt_set)
        changed = [dict(tgt_records[0], text=tgt_records[0]["text"] + "x"), tgt_records[1]]
        with self.assertRaisesRegex(ValueError, "cover"):
            align(src_records, src_set, changed, tgt_set)
        bad = [dict(tgt_records[0], segments=tgt_records[0]["segments"][::-1]), tgt_records[1]]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            align(src_records, src_set, bad, tgt_set)
        wrong_text = dict(tgt_records[0])
        j, s, e = wrong_text["segments"][0]
        encoded = wrong_text["text"].encode()
        wrong_text["text"] = (encoded[:s] + b"X" + encoded[s + 1:]).decode()
        with self.assertRaisesRegex(ValueError, "segment texts differ"):
            align(src_records, src_set, [wrong_text, tgt_records[1]], tgt_set)
        with self.assertRaisesRegex(ValueError, "lacks document"):
            align(src_records, src_set, tgt_records, self.capture(tgt_records[:1], tgt_tokens[:1], "muse-glimmer"))
        mixed = [dict(tgt_records[0], bos_token=""), tgt_records[1]]
        with self.assertRaisesRegex(ValueError, "target records were rendered with different"):
            align(src_records, src_set, mixed, tgt_set)
        documents = [{"document_id": r["document_id"], "byte_end": ends, "special": special}
                     for r, (ends, special) in zip(tgt_records, tgt_tokens)]
        rows = sum(len(ends) for ends, _ in tgt_tokens)
        stale = write_synthetic_capture(self.root / "stale", documents, {("k", 3): np.zeros((rows, 1, 2), np.float32)},
                                        architecture="muse-glimmer", n_layer=4,
                                        manifest={"render_bos_token": "", "render_template_sha256": None})
        with self.assertRaisesRegex(ValueError, "render_bos_token differs"):
            align(src_records, src_set, tgt_records, CaptureSet([stale]))

    @unittest.skipUnless((TEMPLATE_DIR / "qwen.jinja").is_file() and (TEMPLATE_DIR / "muse.jinja").is_file(),
                         "real chat templates not available")
    def test_real_templates_align_with_reference_rule(self):
        qwen = (TEMPLATE_DIR / "qwen.jinja").read_text(encoding="utf-8")
        muse = (TEMPLATE_DIR / "muse.jinja").read_text(encoding="utf-8")
        conversations = sample_conversations()
        src_records = [rendered_record(c, "qwen", render(c, qwen)) for c in conversations]
        tgt_records = [rendered_record(c, "muse", render(c, muse, bos_token=MUSE_BOS)) for c in conversations]
        src_tokens = [fake_tokenize(r["text"], words) for r in src_records]
        tgt_tokens = [byte_pair_tokenize(r["text"]) for r in tgt_records]
        src_set = self.capture(src_records, src_tokens, "qwen35")
        tgt_set = self.capture(tgt_records, tgt_tokens, "muse-glimmer")
        result = align(src_records, src_set, tgt_records, tgt_set)
        self.assertEqual(result.rejected, [])
        for number, record in enumerate(tgt_records):
            expected = reference_alignment(src_records[number], src_tokens[number], record, tgt_tokens[number])
            chosen = result.doc == number
            actual = sorted(zip((result.src_row[chosen] - src_set.rows(record["document_id"]).start).tolist(),
                                (result.tgt_row[chosen] - tgt_set.rows(record["document_id"]).start).tolist()))
            self.assertEqual(actual, expected)
        self.assertEqual(int(tgt_set.special[0]), 1)  # Muse BOS from the template
        self.assertNotIn(2, {int(key.split(":")[0]) for key in
                             [tgt_records[0]["segment_keys"][j] for j, _, _ in tgt_records[0]["segments"]]
                             if key.endswith(":content")})

    def test_save_and_load(self):
        splitter = lambda text: fake_tokenize(text, words)
        src_records, tgt_records, src_tokens, tgt_tokens = self.build(sample_conversations(), splitter,
                                                                      byte_pair_tokenize)
        result = align(src_records, self.capture(src_records, src_tokens, "qwen35"),
                       tgt_records, self.capture(tgt_records, tgt_tokens, "muse-glimmer"))
        result.save(self.root / "alignment.npz")
        loaded = load_alignment(self.root / "alignment.npz")
        for name in align_module.PAIR_FIELDS + ("src_class", "tgt_class"):
            np.testing.assert_array_equal(getattr(loaded, name), getattr(result, name))
        self.assertEqual(loaded.document_ids, result.document_ids)
        with self.assertRaises(FileExistsError):
            result.save(self.root / "alignment.npz")


FAKE_PROBE = textwrap.dedent('''\
    import argparse, hashlib, json, os, sys
    from pathlib import Path
    import numpy as np
    sys.path.insert(0, {experiment!r})
    from captures import default_layer_types, write_synthetic_capture
    parser = argparse.ArgumentParser()
    for name in ("mode", "model", "input", "output", "ctx", "batch", "threads"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--capture-spec")
    parser.add_argument("--max-scored")
    args = parser.parse_args()
    fault = os.environ.get("KVPARK_FAKE_FAULT", "")
    if os.environ.get("KVPARK_FAKE_ENV_LOG"):
        Path(os.environ["KVPARK_FAKE_ENV_LOG"]).write_text(json.dumps(dict(os.environ)))
    model = json.loads(Path(args.model).read_text())
    docs = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()]
    output = Path(args.output)
    settings = {{"ctx": int(args.ctx), "batch": int(args.batch), "threads": int(args.threads),
                "cache_type": "f32", "flash_attention": False, "swa_full": True, "add_special": False,
                "parse_special": True}}
    if args.mode == "tokenize":
        settings = {{"ctx": int(args.ctx), "add_special": fault == "tokenize-settings",
                    "parse_special": True}}
    if fault == "settings":
        settings["batch"] += 1
    if args.mode == "patch-score":
        settings["max_scored"] = int(args.max_scored) + (fault == "patch-settings")
        output.mkdir()
        results = []
        for doc in docs:
            scored = 2
            for condition in doc["conditions"]:
                folder = output / "scores" / doc["document_id"]
                folder.mkdir(parents=True, exist_ok=True)
                np.save(folder / (condition["name"] + ".npy"), np.zeros((scored, 3), "<f4"))
            results.append({{"document_id": doc["document_id"], "n_tokens": 5, "n_scored": scored,
                            "conditions": [{{"name": c["name"]}} for c in doc["conditions"]]}})
        (output / "results.json").write_text(json.dumps({{"format_version": 2, "mode": "patch-score",
                                                         "documents": results, "settings": settings}}))
        sys.exit(0)
    layer_types = default_layer_types(model["architecture"], model["n_layer"])
    attention = [i for i, kind in enumerate(layer_types) if kind != "recurrent"]
    documents = []
    bos = "<|begin_of_text|>".encode()
    for doc in docs:
        data = doc["text"].encode("utf-8")
        ends, ids, special = list(range(1, len(data) + 1)), list(data), [0] * len(data)
        if data.startswith(bos) and fault != "nobos":
            ends, ids, special = ends[len(bos) - 1:], [200000] + ids[len(bos):], [1] + special[len(bos):]
        documents.append({{"document_id": doc["document_id"], "byte_end": ends, "token_id": ids,
                          "special": special}})
    if fault == "order":
        documents.reverse()
    rows = sum(len(d["byte_end"]) for d in documents)
    arrays = {{}}
    if args.mode == "collect":
        spec = json.loads(Path(args.capture_spec).read_text())
        rng = np.random.default_rng(0)
        for key, layers in spec.items():
            for layer in attention if layers == "all" else layers:
                arrays[key, layer] = rng.normal(size=(rows, 2, 4)).astype(np.float32)
        if fault == "missing":
            arrays.pop(sorted(arrays)[0])
        if fault == "nan":
            arrays[sorted(arrays)[0]][0, 0, 0] = np.nan
    write_synthetic_capture(output, documents, arrays, architecture=model["architecture"],
                            layer_types=layer_types, mode=args.mode, settings=settings)
    ''')


class CollectV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model.gguf"
        self.model.write_text(json.dumps({"architecture": "muse-glimmer", "n_layer": 8}))
        self.library = self.root / "libllama.so"
        self.library.write_bytes(b"synthetic runtime fixture")
        self.probe = self.root / "kvpark-kv-probe"
        source = FAKE_PROBE.format(experiment=str(EXPERIMENT))
        self.probe.write_text(f"#!{sys.executable}\n" + source, encoding="utf-8")
        self.probe.chmod(0o755)
        manifest = {"format_version": 1, "runtime_source_revision": "pinned-test-runtime",
                    "collector_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "artifacts": [{"path": str(path), "sha256": sha256_file(path)}
                                  for path in (self.probe, self.library)]}
        (self.root / "kvpark-kv-probe.build.json").write_text(json.dumps(manifest))
        self.spec = self.root / "spec.json"
        self.spec.write_text(json.dumps({"k": "all", "v": [3], "q": [0, 7]}))
        self.template = self.root / "muse.jinja"
        self.template.write_text(MUSE_LIKE)
        self.records = [rendered_record(conv, "muse", render(conv, MUSE_LIKE, bos_token=MUSE_BOS))
                        for conv in sample_conversations()]
        self.docs = self.root / "docs.jsonl"
        self.write_docs(self.records)
        self.output = self.root / "capture"
        self.environment = patch.dict(os.environ, {"KVPARK_FAKE_FAULT": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.tokenizer_info = {"chat_template": MUSE_LIKE, "bos_token": MUSE_BOS, "add_bos": True}
        info = patch.object(collect_v2, "model_tokenizer_info", lambda model: dict(self.tokenizer_info))
        info.start()
        self.addCleanup(info.stop)

    def write_docs(self, records):
        self.docs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                             encoding="utf-8")

    def arguments(self, mode="collect", output=None, extra=()):
        arguments = ["--probe", str(self.probe), "--mode", mode, "--model", str(self.model),
                     "--model-id", "test/muse", "--docs", str(self.docs),
                     "--output", str(output or self.output), "--ctx", "4096", "--batch", "64",
                     "--threads", "2", "--skip-memory-check", *extra]
        if mode == "collect" and "--capture-spec" not in extra:
            arguments += ["--capture-spec", str(self.spec)]
        return arguments

    def invoke(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return collect_v2.main(self.arguments(*args, **kwargs))

    def leftovers(self):
        return sorted(path.name for path in self.root.glob(".kvpark-*"))

    def test_collect_publishes_validated_capture_with_manifest(self):
        cache = self.root / "hashes.json"
        self.assertEqual(self.invoke(extra=["--chat-template", str(self.template),
                                            "--hash-cache", str(cache)]), 0)
        capture = Capture(self.output)
        manifest = capture.manifest
        self.assertEqual(manifest["format_version"], 2)
        self.assertEqual(manifest["model_sha256"], sha256_file(self.model))
        self.assertEqual(manifest["rendered_docs_sha256"], sha256_file(self.docs))
        self.assertEqual(manifest["chat_template_sha256"], sha256_file(self.template))
        self.assertEqual(manifest["capture_spec"], {"k": "all", "v": [3], "q": [0, 7]})
        self.assertEqual(manifest["architecture"], "muse-glimmer")
        self.assertEqual(manifest["rows"], capture.rows_total)
        self.assertEqual(manifest["wrapper_sha256"], sha256_file(EXPERIMENT / "collect_v2.py"))
        self.assertEqual(manifest["settings"]["batch"], 64)
        self.assertEqual(manifest["settings"]["swa_full"], True)
        self.assertEqual((manifest["render_template_sha256"], manifest["render_bos_token"],
                          manifest["model_add_bos"]), (sha256_file(self.template), MUSE_BOS, True))
        self.assertNotIn(str(self.root), json.dumps(manifest))
        self.assertEqual(capture.layers("k"), list(range(8)))
        self.assertEqual(capture.layers("q"), [0, 7])
        capture.check_documents(self.records)
        self.assertEqual(self.leftovers(), [])
        with self.assertRaises(FileExistsError):
            self.invoke()

    def test_model_hash_is_cached_by_file_identity(self):
        cache = self.root / "hashes.json"
        calls = []
        original = collect_v2.sha256_file

        def counting(path):
            calls.append(Path(path).name)
            return original(path)

        with patch.object(collect_v2, "sha256_file", counting):
            self.invoke(extra=["--hash-cache", str(cache)])
            self.invoke(mode="tokenize", output=self.root / "tok", extra=["--hash-cache", str(cache)])
            self.assertEqual(calls.count("model.gguf"), 1)
            os.utime(self.model, ns=(1, 1))
            self.invoke(mode="tokenize", output=self.root / "tok2", extra=["--hash-cache", str(cache)])
            self.assertEqual(calls.count("model.gguf"), 2)
        entries = json.loads(cache.read_text())["entries"]
        self.assertEqual(len(entries), 2)
        self.assertEqual(set(entries.values()), {sha256_file(self.model)})
        self.assertEqual(Capture(self.root / "tok2").manifest["model_sha256"], sha256_file(self.model))

    def test_tokenize_helper_returns_per_document_arrays(self):
        tokens = collect_v2.tokenize(self.probe, self.model, self.records, workdir=self.root)
        self.assertEqual(list(tokens), [r["document_id"] for r in self.records])
        first = tokens[self.records[0]["document_id"]]
        data = self.records[0]["text"].encode()
        bos = len(MUSE_BOS.encode())
        np.testing.assert_array_equal(first["token_id"], [200000] + list(data[bos:]))
        np.testing.assert_array_equal(first["byte_end"], np.arange(bos, len(data) + 1))
        self.assertEqual(first["special"].dtype, np.uint8)
        self.assertEqual(first["special"].tolist(), [1] + [0] * (len(data) - bos))
        self.assertEqual(self.leftovers(), [])

    def test_bad_probe_outputs_are_rejected_without_publishing(self):
        for fault, message, mode in (("order", "order", "collect"), ("missing", "capture spec", "collect"),
                                     ("nan", "nonfinite", "collect"), ("settings", "settings", "collect"),
                                     ("nobos", "special BOS", "collect"), ("nobos", "special BOS", "tokenize"),
                                     ("tokenize-settings", "settings", "tokenize")):
            with self.subTest(fault=fault, mode=mode), patch.dict(os.environ, {"KVPARK_FAKE_FAULT": fault}):
                with self.assertRaisesRegex(ValueError, message):
                    self.invoke(mode=mode)
                self.assertFalse(self.output.exists())
                self.assertEqual(self.leftovers(), [])

    def test_render_provenance_is_checked_against_the_model(self):
        cases = [
            ("BOS", dict(self.tokenizer_info, bos_token="<s>"), self.records),
            ("exactly one", self.tokenizer_info,
             [rendered_record(c, "muse", render(c, QWEN_LIKE, bos_token=MUSE_BOS))
              for c in sample_conversations()]),
            ("carry template_sha256", self.tokenizer_info,
             [{k: v for k, v in r.items() if k not in ("template_sha256", "bos_token")} for r in self.records]),
            ("different templates", self.tokenizer_info,
             [self.records[0], dict(self.records[1], template_sha256="0" * 64)]),
        ]
        for message, info, records in cases:
            self.tokenizer_info = info
            self.write_docs(records)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.invoke(mode="tokenize")
            self.assertFalse(self.output.exists())
        self.tokenizer_info = {"chat_template": None, "bos_token": MUSE_BOS, "add_bos": True}
        self.write_docs(self.records)
        (self.root / "other.jinja").write_text(QWEN_LIKE)
        with self.assertRaisesRegex(ValueError, "--chat-template differs"):
            self.invoke(mode="tokenize", extra=["--chat-template", str(self.root / "other.jinja")])
        self.assertEqual(self.invoke(mode="tokenize"), 0)
        manifest = Capture(self.output).manifest
        self.assertEqual(manifest["settings"], {"ctx": 4096, "add_special": False, "parse_special": True,
                                                "vocab_only": True})
        self.assertEqual(manifest["render_bos_token"], MUSE_BOS)
        plain = [{"document_id": r["document_id"], "text": r["text"]} for r in self.records]
        self.write_docs(plain)
        self.tokenizer_info = None  # unrendered text never consults the model metadata
        self.assertEqual(self.invoke(mode="tokenize", output=self.root / "plain"), 0)
        self.assertIsNone(Capture(self.root / "plain").manifest["render_bos_token"])

    def test_hash_cache_merges_concurrent_writers(self):
        cache = self.root / "hashes.json"
        other = self.root / "other.gguf"
        other.write_text("other")
        first, second = collect_v2.ModelHashCache(cache), collect_v2.ModelHashCache(cache)
        first.sha256(self.model)
        second.sha256(other)
        entries = json.loads(cache.read_text())["entries"]
        self.assertEqual(sorted(entries.values()), sorted([sha256_file(self.model), sha256_file(other)]))

    def test_modified_runtime_is_rejected_before_execution(self):
        self.library.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.invoke()
        self.assertFalse(self.output.exists())

    def test_argument_and_spec_validation(self):
        cases = [
            (dict(extra=["--capture-spec", str(self.root / "bad.json")]), {"x": "all"}),
            (dict(extra=["--capture-spec", str(self.root / "bad.json")]), {"k": [1, 1]}),
            (dict(extra=["--capture-spec", str(self.root / "bad.json")]), {"k": []}),
            (dict(extra=["--capture-spec", str(self.root / "bad.json")]), {"k": "some"}),
            (dict(extra=["--capture-spec", str(self.root / "bad.json")]), []),
            (dict(mode="tokenize", extra=["--capture-spec", str(self.spec)]), None),
            (dict(extra=["--max-scored", "3"]), None),
            (dict(extra=["--batch", "8192"]), None),
        ]
        for options, spec in cases:
            if spec is not None:
                (self.root / "bad.json").write_text(json.dumps(spec))
            with self.subTest(options=options, spec=spec), self.assertRaises(ValueError):
                self.invoke(**options)
            self.assertFalse(self.output.exists())

    def test_document_validation(self):
        good = self.records[0]
        cases = [
            [dict(good, document_id="abc")],
            [dict(good, text="")],
            [good, good],
            [dict(good, segments=[[0, 5, 2]])],
            [dict(good, text="x" * (4 * 1024 * 1024 + 1))],
        ]
        for records in cases:
            self.write_docs(records)
            with self.subTest(records=str(records)[:60]), self.assertRaises(ValueError):
                collect_v2.read_documents(self.docs, "collect")
        self.docs.write_text("")
        with self.assertRaises(ValueError):
            collect_v2.read_documents(self.docs, "tokenize")

    def test_environment_hygiene(self):
        log = self.root / "env.json"
        with patch.dict(os.environ, {"GGML_NTHREADS": "9", "LD_PRELOAD": "/x.so",
                                     "KVPARK_FAKE_ENV_LOG": str(log)}):
            self.invoke(mode="tokenize")
        env = json.loads(log.read_text())
        self.assertNotIn("GGML_NTHREADS", env)
        self.assertNotIn("LD_PRELOAD", env)
        self.assertEqual(env["LD_LIBRARY_PATH"], str(self.root.resolve()))

    def test_memory_guard(self):
        with patch.object(collect_v2, "mem_available", return_value=1024):
            arguments = [item for item in self.arguments() if item != "--skip-memory-check"]
            with self.assertRaisesRegex(ValueError, "MemAvailable"), \
                    contextlib.redirect_stdout(io.StringIO()):
                collect_v2.main(arguments)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(collect_v2.main(
                    [item for item in self.arguments("tokenize") if item != "--skip-memory-check"]), 0)
        large = self.root / "large.gguf"
        with large.open("wb") as stream:
            stream.truncate(collect_v2.TINY_MODEL_BYTES)
        with self.assertRaisesRegex(ValueError, "tiny fixture"):
            collect_v2.memory_check(large, skip=True)
        collect_v2.memory_check(self.model, skip=True)
        self.assertGreater(collect_v2.mem_available(), 0)

    def test_publish_directory_never_replaces(self):
        source = self.root / "src"
        source.mkdir()
        destination = self.root / "dst"
        destination.mkdir()
        with self.assertRaises(OSError):
            collect_v2.publish_directory(source, destination)
        destination.rmdir()
        collect_v2.publish_directory(source, destination)
        self.assertTrue(destination.is_dir())
        self.assertFalse(source.exists())

    def test_patch_score_passthrough_and_validation(self):
        patch_dir = self.root / "patch"
        patch_dir.mkdir()
        np.save(patch_dir / "positions.npy", np.asarray([1, 2], np.int32))
        records = [dict(record, expected_token_sha256="0" * 64, score_ranges=[[0, 5]],
                        conditions=[{"name": "native", "patch_dir": None},
                                    {"name": "L27_v", "patch_dir": str(patch_dir)}])
                   for record in self.records]
        self.write_docs(records)
        self.assertEqual(self.invoke(mode="patch-score"), 0)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["settings"]["max_scored"], 256)
        with patch.dict(os.environ, {"KVPARK_FAKE_FAULT": "patch-settings"}), \
                self.assertRaisesRegex(ValueError, "settings"):
            self.invoke(mode="patch-score", output=self.root / "bad-patch")
        self.assertFalse((self.root / "bad-patch").exists())
        self.assertRegex(manifest["patch_inputs_sha256"], "^[0-9a-f]{64}$")
        self.assertTrue((self.output / "scores" / records[0]["document_id"] / "L27_v.npy").is_file())
        for change in ({"conditions": [{"name": "L27_v", "patch_dir": str(patch_dir)}]},
                       {"conditions": [{"name": "native", "patch_dir": None},
                                       {"name": "bad/name", "patch_dir": str(patch_dir)}]},
                       {"conditions": [{"name": "native", "patch_dir": None},
                                       {"name": "rel", "patch_dir": "patch"}]},
                       {"score_ranges": [[3, 3]]}, {"expected_token_sha256": "x"}):
            self.write_docs([dict(records[0], **change)])
            with self.subTest(change=change), self.assertRaises(ValueError):
                collect_v2.read_documents(self.docs, "patch-score")


PROBE = os.environ.get("KVPARK_KV_PROBE")


@unittest.skipUnless(PROBE, "set KVPARK_KV_PROBE to run collect_v2 against the real binary")
class CollectV2IntegrationTests(unittest.TestCase):
    def test_tiny_fixtures_collect_and_align(self):
        from tiny_fixture import create_fixture
        with tempfile.TemporaryDirectory(prefix="kvpark-collect-v2-") as temporary:
            root = Path(temporary)
            conversations = sample_conversations()
            spec = root / "spec.json"
            spec.write_text(json.dumps({"k": "all", "v": "all"}))
            sets, records = {}, {}
            # Tiny fixtures name their BOS "<bos>"; llama.cpp passes it to both templates.
            for architecture, name, template, bos in (("qwen35", "qwen", QWEN_LIKE, "<bos>"),
                                                      ("muse-glimmer", "muse", MUSE_LIKE, "<bos>")):
                model = root / f"{name}.gguf"
                create_fixture(model, architecture, context_length=4096)
                records[name] = [rendered_record(conv, name, render(conv, template, bos_token=bos))
                                 for conv in conversations]
                docs = root / f"{name}.jsonl"
                docs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records[name]),
                                encoding="utf-8")
                output = root / f"{name}-capture"
                with contextlib.redirect_stdout(io.StringIO()):
                    collect_v2.main(["--probe", PROBE, "--mode", "collect", "--model", str(model),
                                     "--model-id", f"tiny/{name}", "--docs", str(docs),
                                     "--output", str(output), "--ctx", "4096", "--batch", "64",
                                     "--threads", "2", "--capture-spec", str(spec),
                                     "--skip-memory-check", "--hash-cache", str(root / "hashes.json")])
                sets[name] = CaptureSet([output])
                tokens = collect_v2.tokenize(PROBE, model, records[name], ctx=4096, workdir=root)
                for record in records[name]:
                    rows = sets[name].rows(record["document_id"])
                    np.testing.assert_array_equal(tokens[record["document_id"]]["token_id"],
                                                  sets[name].token_id[rows])
            self.assertEqual(sets["qwen"].layers("k"), [3])
            self.assertEqual(sets["muse"].captures[0].manifest["render_bos_token"], "<bos>")
            self.assertEqual(int(sets["muse"].special[0]), 1)
            self.assertEqual(int(sets["qwen"].special[0]), 0)
            self.assertEqual(sets["muse"].layers("v"), [0, 1, 2, 3])
            result = align(records["qwen"], sets["qwen"], records["muse"], sets["muse"])
            # Byte-level fixture vocabularies: every rendered content byte aligns.
            self.assertEqual(int(np.sum(result.tgt_class == UNMATCHED)), 0)
            self.assertEqual(int(np.sum(result.tgt_class == STRADDLING)), 0)
            self.assertGreater(len(result.tgt_row), 100)


if __name__ == "__main__":
    unittest.main()
