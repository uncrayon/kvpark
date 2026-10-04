"""End-to-end orchestration checks with a fake, tiny offline collector."""

import contextlib
import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np

import collect
from probe import load_capture
from run_prototype import main, read_split_corpus


class PrototypeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.collector = self.root / "collector"
        self.source = self.root / "source.gguf"
        self.target = self.root / "target.gguf"
        self.source.write_text("source")
        self.target.write_text("target")
        self.corpus = self.root / "corpus.jsonl"
        self.output = self.root / "run"
        self.failure = self.root / "fail-target"
        self.records = [{"id": "train-1", "split": "train", "text": "Alpha training document."},
                        {"id": "eval-1", "split": "eval", "text": "Held-out beta example."},
                        {"id": "train-2", "split": "train", "text": "Gamma training sentence."}]
        self.write_corpus()
        code = textwrap.dedent('''\
            import argparse
            import json
            from pathlib import Path
            import struct
            parser = argparse.ArgumentParser()
            for name in ("model", "input", "output", "layer", "ctx", "batch", "threads"):
                parser.add_argument("--" + name, required=True)
            args = parser.parse_args()
            target = Path(args.model).read_text() == "target"
            FAILURE_CHECK
            output = Path(args.output)
            output.mkdir()
            documents = []
            for line in Path(args.input).read_text().splitlines():
                document = json.loads(line)
                raw = document["text"].encode("utf-8")
                offsets = list(range(2, len(raw) + 1, 2)) if target else list(range(1, len(raw) + 1))
                if offsets[-1] != len(raw):
                    offsets.append(len(raw))
                values = []
                for end in offsets:
                    a, b = raw[end - 1] / 100, end / 10
                    values.extend([a + 2 * b + 3, 2 * a - b, b - 2] if target else [a, b])
                filename = document["document_sha256"] + ".f32"
                (output / filename).write_bytes(struct.pack("<" + "f" * len(values), *values))
                documents.append({"document_sha256": document["document_sha256"],
                                  "byte_end": offsets, "token_ids": [raw[end - 1] for end in offsets],
                                  "shape": [len(offsets), 1, 3 if target else 2],
                                  "values_file": filename})
            report = {"format_version": 1, "architecture": "fake", "file_type": "7",
                      "layer": int(args.layer), "attention_type": "full", "cache_dtype": "f32",
                      "extraction": "Vcur_pre_cache_f32", "documents": documents}
            (output / "report.json").write_text(json.dumps(report))
            print("COLLECTED", "target" if target else "source")
            ''').replace("FAILURE_CHECK", f"if target and Path({str(self.failure)!r}).exists(): raise SystemExit(7)")
        self.collector.write_text(f"#!{sys.executable}\n" + code)
        self.collector.chmod(0o755)
        manifest = {"format_version": 1, "runtime_source_revision": "fake-runtime",
                    "collector_source_sha256": collect.sha256_file(self.collector),
                    "artifacts": [{"path": str(self.collector),
                                   "sha256": collect.sha256_file(self.collector)}]}
        (self.root / "collector.build.json").write_text(json.dumps(manifest))

    def write_corpus(self):
        self.corpus.write_text("".join(json.dumps(record) + "\n" for record in self.records))

    def arguments(self):
        return ["--collector", str(self.collector), "--source-model", str(self.source),
                "--target-model", str(self.target), "--source-model-id", "fake/source",
                "--target-model-id", "fake/target", "--corpus", str(self.corpus),
                "--output", str(self.output)]

    def invoke(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(self.arguments() + list(extra))

    def test_runs_sequential_capture_document_split_fit_and_summary(self):
        self.assertEqual(self.invoke(), 0)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["documents"], {"train": 2, "eval": 1})
        self.assertLess(summary["held_out_values"]["mse_over_baseline"], 1e-4)
        self.assertLess(summary["eval_alignment"]["source_coverage"], 1)
        self.assertEqual(summary["eval_alignment"]["target_coverage"], 1)
        for side in ("source", "target"):
            all_rows = load_capture(self.output / f"{side}-all.npz")
            training = load_capture(self.output / f"{side}-train.npz")
            evaluation = load_capture(self.output / f"{side}-eval.npz")
            self.assertEqual(training.metadata, evaluation.metadata)
            self.assertFalse(set(training.documents) & set(evaluation.documents))
            self.assertEqual(len(all_rows.values), len(training.values) + len(evaluation.values))
            with np.load(self.output / f"{side}-eval.npz", allow_pickle=False) as split:
                self.assertEqual(len(split["token_id"]), len(evaluation.values))
        self.assertTrue((self.output / "fit" / "mapper.npz").is_file())

    def test_completed_resume_reuses_model_captures_and_fit(self):
        self.invoke()
        files = [self.output / name for name in ("source-all.npz", "target-all.npz", "source.log", "target.log")]
        files += [self.output / "fit" / "mapper.npz"]
        before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in files}
        self.assertEqual(self.invoke("--resume"), 0)
        for path, expected in before.items():
            self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), expected)
        with self.assertRaises(FileExistsError):
            self.invoke()

    def test_resume_after_target_failure_reuses_completed_source_capture(self):
        self.failure.touch()
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke()
        source = self.output / "source-all.npz"
        before = source.stat().st_mtime_ns
        self.assertFalse((self.output / "target-all.npz").exists())
        self.failure.unlink()
        self.invoke("--resume")
        self.assertEqual(source.stat().st_mtime_ns, before)
        self.assertEqual((self.output / "source.log").read_text().count("COLLECTED source"), 1)

    def test_resume_rejects_changed_model_corpus_and_settings(self):
        self.invoke()
        for path in (self.source, self.corpus):
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, "fingerprints differ"):
                self.invoke("--resume")
            path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "fingerprints differ"):
            self.invoke("--resume", "--threads", "2")

    def test_resume_rejects_changed_fit_artifact(self):
        self.invoke()
        mapper = self.output / "fit" / "mapper.npz"
        mapper.write_bytes(mapper.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "artifacts have changed"):
            self.invoke("--resume")

    def test_resume_rejects_split_that_no_longer_matches_full_capture(self):
        self.invoke()
        split = self.output / "source-train.npz"
        with np.load(split, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["values"][0, 0, 0] += 1
        np.savez(split, **arrays)
        with self.assertRaisesRegex(ValueError, "differs from its source"):
            self.invoke("--resume")

    def test_run_lock_prevents_duplicate_collection(self):
        self.invoke()
        with (self.output / ".run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "already running"):
                self.invoke("--resume")

    def test_corpus_rejects_missing_splits_duplicate_ids_and_document_overlap(self):
        alterations = [
            lambda: self.records[0].pop("split"),
            lambda: self.records[1].update(split="train"),
            lambda: self.records[2].update(split="eval"),
            lambda: self.records[1].update(id="train-1"),
            lambda: self.records[1].update(id=""),
            lambda: self.records[1].update(text=self.records[0]["text"]),
        ]
        original = json.dumps(self.records)
        for index, alteration in enumerate(alterations):
            self.records = json.loads(original)
            alteration()
            self.write_corpus()
            with self.subTest(alteration=index), self.assertRaises(ValueError):
                read_split_corpus(self.corpus)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
