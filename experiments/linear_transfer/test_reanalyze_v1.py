"""End-to-end reanalysis on synthetic v1-format captures and a tiny F32 Muse GGUF."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

import numpy as np

import gguf_tensors
import probe
import reanalyze_v1
import ridge


VOCAB, EMBED, HEADS, KV_HEADS, HEAD_DIM, LAYER = 40, 24, 4, 2, 4, 1


def write_gguf(path, embeddings, attn_output, kv_heads=KV_HEADS):
    gguf = gguf_tensors._gguf()
    writer = gguf.GGUFWriter(path, "muse-glimmer")
    writer.add_file_type(0)
    writer.add_embedding_length(EMBED)
    writer.add_block_count(2)
    writer.add_head_count(HEADS)
    writer.add_head_count_kv(kv_heads)
    writer.add_key_length(HEAD_DIM)
    writer.add_value_length(HEAD_DIM)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_tensor("token_embd.weight", embeddings.astype(np.float32))
    writer.add_tensor(f"blk.{LAYER}.attn_output.weight", attn_output.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_capture(path, values, documents, ends, tokens, model, architecture):
    metadata = dict(format_version=1, model_id=model, model_revision="r1", runtime_revision="rt",
                    weight_quantization="0", layer=LAYER, attention_type="full", cache_kind="value",
                    text_mode="raw_utf8_no_special_tokens", architecture=architecture)
    np.savez(path, values=values.astype(np.float32), document_sha256=np.array(documents),
             byte_end=np.asarray(ends, dtype=np.int64), token_id=np.asarray(tokens, dtype=np.int64),
             metadata_json=np.array(json.dumps(metadata)))


@unittest.skipUnless(gguf_tensors.GGUF_PY.is_dir(), "pinned gguf-py is not available")
class ReanalyzeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(gguf_tensors._READERS.clear)
        self.root = Path(self.temp.name)
        self.captures = self.root / "prototype-v1"
        self.captures.mkdir()
        rng = np.random.default_rng(17)
        self.embeddings = rng.normal(size=(VOCAB, EMBED))
        normalized = reanalyze_v1.rmsnorm(self.embeddings, 1e-5)
        lexical_map = rng.normal(size=(EMBED, KV_HEADS * HEAD_DIM)) * .4
        source_map = rng.normal(size=(6, KV_HEADS * HEAD_DIM))
        for split, count in (("train", 12), ("eval", 3)):
            source_docs, target_docs, source_rows, target_rows = [], [], [], []
            source_ends, target_ends, target_tokens, source_tokens = [], [], [], []
            for document in range(count):
                name = hashlib.sha256(f"{split}{document}".encode()).hexdigest()
                tokens = rng.integers(0, VOCAB, size=24)
                for position, token in enumerate(tokens):
                    x = rng.normal(size=6)
                    previous = normalized[tokens[position - 1]] if position else np.zeros(EMBED)
                    y = (normalized[token] + .5 * previous) @ lexical_map + x @ source_map
                    y += .3 * rng.normal(size=y.shape)
                    target_docs.append(name)
                    target_ends.append(2 * position + 2)
                    target_tokens.append(token)
                    target_rows.append(y)
                    if position % 5 != 3:
                        source_docs.append(name)
                        source_ends.append(2 * position + 2)
                        source_tokens.append(token)
                        source_rows.append(x)
            write_capture(self.captures / f"source-{split}.npz", np.array(source_rows)[:, None, :],
                          source_docs, source_ends, source_tokens, "src", "qwen35")
            write_capture(self.captures / f"target-{split}.npz",
                          np.array(target_rows).reshape(-1, KV_HEADS, HEAD_DIM),
                          target_docs, target_ends, target_tokens, "tgt", "muse-glimmer")
        # Non-orthogonal, with distinct per-head scales, so grouping and orientation matter.
        self.attn_output = rng.normal(size=(EMBED, HEADS * HEAD_DIM)) * np.repeat([.3, 2.0, 1.0, .5], HEAD_DIM)
        self.gguf = self.root / "muse.gguf"
        write_gguf(self.gguf, self.embeddings, self.attn_output)

    def gguf_py_attn_output(self):
        """W_o straight from gguf-py's reader, independent of gguf_tensors."""
        reader = gguf_tensors._gguf().GGUFReader(self.gguf)
        tensor = next(t for t in reader.tensors if t.name == f"blk.{LAYER}.attn_output.weight")
        self.assertEqual([int(n) for n in tensor.shape], [HEADS * HEAD_DIM, EMBED])
        return np.array(tensor.data, dtype=np.float64).reshape(EMBED, HEADS * HEAD_DIM)

    @staticmethod
    def direct_output_r2(attn_output, actual, predicted, kv_of_head):
        def energy(values):
            total = 0.0
            for head in range(HEADS):
                block = attn_output[:, head * HEAD_DIM:(head + 1) * HEAD_DIM]
                kv = kv_of_head(head)
                total += np.sum(np.square(values[:, kv * HEAD_DIM:(kv + 1) * HEAD_DIM] @ block.T))
            return total
        return 1 - energy(actual - predicted) / energy(actual - actual.mean(axis=0))

    def invoke(self, output, gguf=None, *extra):
        arguments = ["--captures", str(self.captures), "--muse-gguf", str(gguf or self.gguf),
                     "--output", str(output), "--folds", "4", "--bootstrap", "200",
                     "--lexical-rank", "12", "--lag-rank", "6", "--source-weights", "1,4", *extra]
        with contextlib.redirect_stdout(io.StringIO()):
            return reanalyze_v1.main(arguments)

    def test_end_to_end_report_matches_independent_computations(self):
        output = self.root / "reanalysis"
        self.assertEqual(self.invoke(output), 0)
        report = json.loads((output / "report.json").read_text())
        models = report["models"]
        captures = {name: probe.load_capture(self.captures / f"{name}.npz")
                    for name in reanalyze_v1.CAPTURES}
        x, y, _ = probe.aligned_values(captures["source-train"], captures["target-train"])
        x_eval, y_eval, _ = probe.aligned_values(captures["source-eval"], captures["target-eval"])
        weight, bias = probe.fit_ridge(x, y, .01)
        naive = probe.reconstruction_metrics(y_eval, x_eval @ weight + bias, y.mean(axis=0))
        self.assertAlmostEqual(models["source_naive"]["eval"]["mse"], naive["mse"])
        self.assertAlmostEqual(models["source_naive"]["eval"]["r2_vs_own_mean"], naive["variance_weighted_r2"])
        rows, _ = probe.aligned_rows(captures["source-train"], captures["target-train"])
        documents = captures["source-train"].documents[rows]
        cv = ridge.grouped_cv(x, y, documents, ridge.default_grid(), n_folds=4)
        self.assertEqual(models["source_cv"]["cv"]["chosen_lambda"], cv["chosen_lambda"])
        self.assertAlmostEqual(models["source_cv"]["cv"]["mean_fold_r2"], cv["mean_fold_r2"][cv["chosen_index"]])
        attn_output = self.gguf_py_attn_output()
        np.testing.assert_allclose(attn_output, self.attn_output.astype(np.float32))
        group = HEADS // KV_HEADS
        with np.load(output / "predictions.npz", allow_pickle=False) as arrays:
            actual = arrays["eval_target"]
            for name in ("source_naive", "source_cv", "lexical", "source_plus_lexical"):
                predicted = arrays[f"{name}_eval_prediction"]
                reported = models[name]["eval"]["output_space"]["r2_vs_own_mean"]
                with self.subTest(model=name):
                    self.assertAlmostEqual(reported, self.direct_output_r2(
                        attn_output, actual, predicted, lambda head: head // group), places=10)
                    interleaved = self.direct_output_r2(attn_output, actual, predicted,
                                                        lambda head: head % KV_HEADS)
                    self.assertGreater(abs(reported - interleaved), 1e-3)
                    self.assertGreater(abs(reported - models[name]["eval"]["r2_vs_own_mean"]), 1e-3)
        self.assertGreater(models["lexical"]["cv"]["mean_fold_r2"], .05)
        self.assertGreater(models["lexical"]["cv"]["mean_fold_r2"],
                           models["lexical_current"]["cv"]["mean_fold_r2"])
        increments = report["increments"]["source_beyond_lexical"]
        self.assertGreater(increments["cv_mean_fold_delta_r2"], .1)
        self.assertAlmostEqual(increments["eval_delta_r2"], models["source_plus_lexical"]["eval"]["r2_vs_own_mean"]
                               - models["lexical"]["eval"]["r2_vs_own_mean"])
        delta = increments["bootstrap_95"]["cv_oof_train_documents"]["delta_r2"]
        self.assertEqual((delta["n_documents"], delta["n_resamples"]), (12, 200))
        self.assertLessEqual(delta["low"], delta["high"])
        selection = models["source_plus_lexical"]["source_weight_selection"]
        self.assertIn(selection["chosen"], (1.0, 4.0))
        self.assertTrue(selection["chosen_at_grid_edge"])
        self.assertEqual(models["source_cv"]["cv"]["final_fit_solver"], "cholesky")
        self.assertIn("12 train documents", " ".join(report["limitations"]))
        self.assertIn("3 evaluation documents", " ".join(report["limitations"]))
        mask = os.umask(0o022)
        os.umask(mask)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o777 & ~mask)
        for check in report["invariance_output_transform"].values():
            self.assertLess(check["max_abs_weight_difference"], 1e-8 * max(1, check["max_abs_weight"]))
        self.assertEqual(report["output_space"]["q_heads_per_kv_head"], 2)
        self.assertEqual(report["inputs"]["muse_gguf"]["token_embd_rows_read"], VOCAB)
        self.assertEqual(report["data"]["train"]["documents"], 12)
        with np.load(output / "predictions.npz", allow_pickle=False) as arrays:
            np.testing.assert_allclose(arrays["source_naive_eval_prediction"], x_eval @ weight + bias)
            self.assertEqual(arrays["lexical_oof_prediction"].shape, y.shape)
        with self.assertRaises(FileExistsError):
            self.invoke(output)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ["muse.gguf", "prototype-v1", "reanalysis"])

    def test_output_metric_matches_direct_quadratic_form(self):
        rng = np.random.default_rng(4)
        attn_output = rng.normal(size=(EMBED, HEADS * HEAD_DIM))
        transform, metric = reanalyze_v1.output_metric(attn_output, HEADS, KV_HEADS, HEAD_DIM)
        error = rng.normal(size=KV_HEADS * HEAD_DIM)
        direct = 0.0
        for head in range(HEADS):
            group = head // (HEADS // KV_HEADS)
            block = attn_output[:, head * HEAD_DIM:(head + 1) * HEAD_DIM]
            direct += np.sum(np.square(block @ error[group * HEAD_DIM:(group + 1) * HEAD_DIM]))
        self.assertAlmostEqual(float(np.sum(np.square(error @ transform))), direct)
        self.assertEqual(len(metric["spectra"]), KV_HEADS)

    def test_rejects_rank_above_basis_rank(self):
        for flag in ("--lexical-rank", "--lag-rank"):
            output = self.root / f"rank{flag}"
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "exceeds the basis rank"):
                self.invoke(output, None, flag, "100")
            self.assertFalse(output.exists())

    def test_publish_refuses_existing_empty_directory(self):
        output = self.root / "occupied"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            reanalyze_v1.publish(output, {"a": 1}, {"b": np.zeros(2)})
        self.assertEqual(list(output.iterdir()), [])
        self.assertEqual([p.name for p in self.root.iterdir() if p.name.startswith(".pending")], [])

    def test_rejects_mismatched_gguf_and_unordered_captures(self):
        bad = self.root / "bad.gguf"
        write_gguf(bad, self.embeddings, np.eye(EMBED), kv_heads=4)
        with self.assertRaises(ValueError):
            self.invoke(self.root / "bad-output", bad)
        self.assertFalse((self.root / "bad-output").exists())
        with self.assertRaises(ValueError):
            reanalyze_v1.document_positions(np.array(["a", "b", "a"]), np.array([1, 1, 2]))
        with self.assertRaises(ValueError):
            reanalyze_v1.document_positions(np.array(["a", "a"]), np.array([2, 1]))
        np.testing.assert_array_equal(
            reanalyze_v1.document_positions(np.array(["a", "a", "b"]), np.array([1, 3, 2])), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
