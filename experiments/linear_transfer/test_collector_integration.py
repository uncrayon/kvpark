"""Real CPU graph checks; enable with KVPARK_COLLECTOR=/path/to/collector."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tiny_fixture import create_fixture


COLLECTOR = os.environ.get("KVPARK_COLLECTOR")


@unittest.skipUnless(COLLECTOR, "set KVPARK_COLLECTOR to enable real llama.cpp checks")
class CollectorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-tiny-collector-")
        cls.root = Path(cls.temp.name)
        cls.models = {}
        cls.weights = {}
        for architecture in ("muse-glimmer", "qwen35"):
            path = cls.root / f"{architecture}.gguf"
            cls.weights[architecture] = create_fixture(path, architecture)
            cls.models[architecture] = path
        cls.fused_muse = cls.root / "muse-fused.gguf"
        create_fixture(cls.fused_muse, "muse-glimmer", fused_qkv=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def capture(self, architecture, documents, *, batch=128, layer=3, fused=False):
        directory = Path(tempfile.mkdtemp(dir=self.root))
        corpus = directory / "corpus.jsonl"
        records = [{"text": text,
                    "document_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
                   for text in documents]
        corpus.write_text("".join(json.dumps(record) + "\n" for record in records),
                          encoding="utf-8")
        output = directory / "raw"
        model = self.fused_muse if fused else self.models[architecture]
        command = [COLLECTOR, "--model", str(model),
                   "--input", str(corpus), "--output", str(output),
                   "--layer", str(layer), "--ctx", "1024", "--batch", str(batch),
                   "--threads", "2"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        values = []
        for record, document in zip(records, report["documents"], strict=True):
            self.assertEqual(document["document_sha256"], record["document_sha256"])
            self.assertEqual(document["token_ids"], list(record["text"].encode("utf-8")))
            self.assertEqual(document["byte_end"],
                             list(range(1, len(record["text"].encode("utf-8")) + 1)))
            value = np.fromfile(output / document["values_file"], dtype="<f4")
            value = value.reshape(document["shape"])
            self.assertEqual(value.shape, (len(document["token_ids"]),
                                          2 if architecture == "muse-glimmer" else 1, 32))
            self.assertTrue(np.isfinite(value).all())
            self.assertGreater(float(np.std(value)), 0)
            values.append(value)
        return values

    def test_full_layer_values_match_across_batches_and_reset_between_documents(self):
        documents = ["A previous conversation must not leak into the next document.",
                     "Second café: 日本語, tabs\tand\nnewlines; different suffix."]
        for architecture in self.models:
            with self.subTest(architecture=architecture):
                whole = self.capture(architecture, documents)
                chunked = self.capture(architecture, documents, batch=7)
                isolated = self.capture(architecture, [documents[1]])
                for expected, actual in zip(whole, chunked, strict=True):
                    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
                np.testing.assert_allclose(whole[1], isolated[0], atol=2e-6, rtol=2e-5)

    def test_muse_first_layer_matches_independent_value_projection(self):
        text = "Independent V projection, including é."
        actual = self.capture("muse-glimmer", [text], layer=0)[0]
        weights = self.weights["muse-glimmer"]
        embedded = weights["token_embd.weight"][list(text.encode("utf-8"))]

        def rms_norm(value):
            return value / np.sqrt(np.mean(value ** 2, axis=-1, keepdims=True) + 1e-6)

        expected = rms_norm(rms_norm(embedded)) @ weights["blk.0.attn_v.weight"].T
        np.testing.assert_allclose(actual.reshape(len(expected), -1), expected,
                                   atol=2e-6, rtol=2e-5)

    def test_qwen_recurrent_layer_is_rejected(self):
        directory = Path(tempfile.mkdtemp(dir=self.root))
        corpus = directory / "corpus.jsonl"
        text = "Recurrent layers do not have the value cache this probe expects."
        corpus.write_text(json.dumps({"text": text,
                          "document_sha256": hashlib.sha256(text.encode()).hexdigest()}) + "\n",
                          encoding="utf-8")
        output = directory / "raw"
        result = subprocess.run([COLLECTOR, "--model", str(self.models["qwen35"]),
                                 "--input", str(corpus), "--output", str(output),
                                 "--layer", "0", "--ctx", "1024", "--batch", "128",
                                 "--threads", "2"],
                                capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("recurrent", result.stderr.lower())
        self.assertFalse((output / "report.json").exists())

    def test_fused_muse_values_match_separate_projections(self):
        documents = ["Strided value views must skip the query and key rows for every token."]
        separate = self.capture("muse-glimmer", documents)
        fused = self.capture("muse-glimmer", documents, fused=True)
        chunked = self.capture("muse-glimmer", documents, fused=True, batch=7)
        np.testing.assert_allclose(fused[0], separate[0], atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(chunked[0], separate[0], atol=2e-5, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
