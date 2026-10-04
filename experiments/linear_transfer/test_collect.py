"""Capture orchestration checks using tiny files and a fake offline executable."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from collect import main, read_corpus, read_raw_capture, sha256_file, verify_runtime
from probe import load_capture, validate_split


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"synthetic model fixture")
        self.library = self.root / "libllama.so"
        self.library.write_bytes(b"synthetic runtime fixture")
        self.collector = self.root / "collector"
        self.corpus = self.root / "corpus.jsonl"
        self.output = self.root / "capture.npz"
        self.write_corpus(["aé", "second"])
        self.write_collector()

    def write_corpus(self, texts):
        self.corpus.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts),
                               encoding="utf-8")

    def write_collector(self, tail=""):
        source = textwrap.dedent('''\
            import argparse
            import json
            from pathlib import Path
            import struct
            parser = argparse.ArgumentParser()
            for field in ("model", "input", "output", "layer", "ctx", "batch", "threads"):
                parser.add_argument("--" + field, required=True)
            args = parser.parse_args()
            directory = Path(args.output)
            directory.mkdir()
            documents = []
            for line in Path(args.input).read_text(encoding="utf-8").splitlines():
                document = json.loads(line)
                count = len(document["text"].encode("utf-8"))
                filename = document["document_sha256"] + ".f32"
                (directory / filename).write_bytes(struct.pack("<" + "f" * (count * 6),
                                                               *range(count * 6)))
                documents.append({"document_sha256": document["document_sha256"],
                                  "shape": [count, 2, 3], "byte_end": list(range(1, count + 1)),
                                  "token_ids": list(range(count)), "values_file": filename})
            report = {"format_version": 1, "architecture": "qwen35", "file_type": "7",
                      "layer": int(args.layer), "attention_type": "full", "cache_dtype": "f32",
                      "extraction": "Vcur_pre_cache_f32", "documents": documents}
            (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
            ''')
        self.collector.write_text(f"#!{sys.executable}\n" + source + tail, encoding="utf-8")
        self.collector.chmod(0o755)
        manifest = {"format_version": 1, "runtime_source_revision": "pinned-test-runtime",
                    "collector_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "artifacts": [{"path": str(path), "sha256": sha256_file(path)}
                                  for path in (self.collector, self.library)]}
        self.manifest_path = self.root / "collector.build.json"
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def arguments(self):
        return ["--collector", str(self.collector), "--model", str(self.model),
                "--model-id", "test/model", "--corpus", str(self.corpus),
                "--layer", "3", "--output", str(self.output)]

    def invoke(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(self.arguments())

    def test_collects_utf8_documents_with_identity_and_token_provenance(self):
        self.assertEqual(self.invoke(), 0)
        capture = load_capture(self.output)
        self.assertEqual(capture.values.shape, (9, 2, 3))
        self.assertEqual(capture.values.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(capture.byte_end[:3], [1, 2, 3])
        self.assertEqual(capture.metadata["model_revision"], sha256_file(self.model))
        self.assertEqual(capture.metadata["extraction"], "Vcur_pre_cache_f32")
        self.assertEqual(capture.metadata["settings"],
                         {"ctx": 1024, "batch": 128, "threads": 4, "device": "cpu",
                          "weight_loading": "mmap", "weight_repacking": False,
                          "flash_attention": False, "k_cache_dtype": "f32", "v_cache_dtype": "f32",
                          "add_special": False, "parse_special": False, "mtp": False})
        self.assertNotIn(str(self.root), json.dumps(capture.metadata))
        with np.load(self.output, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["token_id"][:3], [0, 1, 2])
        self.assertEqual(sorted(path.name for path in self.root.glob(".kvpark-capture-*")), [])

    def test_train_eval_metadata_remains_identical_for_different_documents(self):
        self.invoke()
        training = load_capture(self.output)
        self.write_corpus(["different held-out document"])
        self.output = self.root / "eval.npz"
        self.invoke()
        evaluation = load_capture(self.output)
        validate_split(training, training, evaluation, evaluation)

    def test_corpus_rejects_duplicates_empty_invalid_and_preserves_exact_text(self):
        self.write_corpus([" \r\né "])
        documents = read_corpus(self.corpus)
        self.assertEqual(documents[0]["text"], " \r\né ")
        for content in ('{"text":""}\n', '{"text":1}\n', 'not-json\n', '\n',
                        '{"text":"same"}\n{"text":"same"}\n', '{"text":"\\ud800"}\n'):
            self.corpus.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_corpus(self.corpus)

    def test_modified_collector_or_library_is_rejected_before_execution(self):
        for path in (self.collector, self.library):
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.invoke()
            self.assertFalse(self.output.exists())
            path.write_bytes(original)

    def test_runtime_identity_ignores_absolute_artifact_directory(self):
        revision, _, _ = verify_runtime(self.collector)
        moved = self.root / "moved"
        moved.mkdir()
        manifest = json.loads(self.manifest_path.read_text())
        for artifact in manifest["artifacts"]:
            original = Path(artifact["path"])
            replacement = moved / original.name
            replacement.write_bytes(original.read_bytes())
            artifact["path"] = str(replacement)
        (moved / "collector.build.json").write_text(json.dumps(manifest))
        self.assertEqual(revision, verify_runtime(moved / "collector")[0])

    def test_existing_output_and_publication_race_do_not_overwrite(self):
        self.output.write_bytes(b"existing capture")
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(self.output.read_bytes(), b"existing capture")
        self.output.unlink()
        self.write_collector(f"Path({str(self.output)!r}).write_bytes(b'other writer')\n")
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(self.output.read_bytes(), b"other writer")

    def test_process_failure_and_model_mutation_leave_no_capture(self):
        for tail, exception in (("raise SystemExit(7)\n", subprocess.CalledProcessError),
                                ("Path(args.model).write_bytes(b'changed')\n", ValueError)):
            self.write_collector(tail)
            with self.subTest(tail=tail), self.assertRaises(exception):
                self.invoke()
            self.assertFalse(self.output.exists())
            self.assertFalse(list(self.root.glob(".kvpark-capture-*")))

    def make_raw(self):
        documents = read_corpus(self.corpus)
        directory = self.root / "raw"
        directory.mkdir(exist_ok=True)
        command = [str(self.collector), "--model", str(self.model), "--input", str(self.root / "input"),
                   "--output", str(directory / "capture"), "--layer", "3", "--ctx", "1024",
                   "--batch", "128", "--threads", "4"]
        (self.root / "input").write_text("".join(json.dumps(item) + "\n" for item in documents))
        subprocess.run(command, check=True)
        raw = directory / "capture"
        return documents, raw, json.loads((raw / "report.json").read_text())

    def test_report_rejects_inconsistent_identity_alignment_shape_and_paths(self):
        documents, raw, original = self.make_raw()
        alterations = [
            lambda report: report.update(layer=4),
            lambda report: report.update(cache_dtype="f16"),
            lambda report: report.update(extraction="unknown"),
            lambda report: report["documents"].reverse(),
            lambda report: report["documents"].pop(),
            lambda report: report["documents"][0].update(shape=[3, 2, 4]),
            lambda report: report["documents"][0].update(byte_end=[1, 1, 3]),
            lambda report: report["documents"][0].update(byte_end=[1, 2, 4]),
            lambda report: report["documents"][0].update(byte_end=[1, 2, True]),
            lambda report: report["documents"][0].update(token_ids=[1, 2, -1]),
            lambda report: report["documents"][0].update(token_ids=[1, 2]),
            lambda report: report["documents"][0].update(values_file="../escape.f32"),
        ]
        for index, alteration in enumerate(alterations):
            report = json.loads(json.dumps(original))
            alteration(report)
            (raw / "report.json").write_text(json.dumps(report))
            with self.subTest(alteration=index), self.assertRaises(ValueError):
                read_raw_capture(raw, documents, 3)

    def test_rejects_nonfinite_truncated_and_symlinked_values(self):
        documents, raw, report = self.make_raw()
        path = raw / report["documents"][0]["values_file"]
        original = path.read_bytes()
        for data in (original[:-4], struct.pack("<f", float("nan")) + original[4:]):
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                read_raw_capture(raw, documents, 3)
        elsewhere = self.root / "elsewhere"
        elsewhere.write_bytes(original)
        path.unlink()
        path.symlink_to(elsewhere)
        with self.assertRaises(ValueError):
            read_raw_capture(raw, documents, 3)

    def test_rejects_model_changes_while_hashing(self):
        original_hash = sha256_file

        def changed_hash(path):
            result = original_hash(path)
            if path == self.model:
                path.write_bytes(b"changed during hashing")
            return result

        with patch("collect.sha256_file", side_effect=changed_hash), self.assertRaisesRegex(
                ValueError, "changed while computing"):
            self.invoke()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
