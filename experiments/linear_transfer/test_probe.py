"""Small numerical and data-contract checks for the offline transfer probe."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import probe
from probe import aligned_values, fit_ridge, load_capture, main, reconstruction_metrics
import ridge


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def capture(self, name, values, ends, document="train", model="source", **changes):
        metadata = dict(format_version=1, model_id=model, model_revision="r1",
                        runtime_revision="runtime1", weight_quantization="fp32",
                        layer=0, attention_type="full", cache_kind="value",
                        text_mode="raw_utf8_no_special_tokens")
        metadata.update(changes)
        path = self.root / f"{name}.npz"
        np.savez(path, values=np.asarray(values, dtype=np.float64), byte_end=np.asarray(ends),
                 document_sha256=np.array([digest(document)] * len(ends)),
                 metadata_json=np.array(json.dumps(metadata)))
        return path

    def test_affine_recovery_with_different_feature_widths(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(80, 3))
        expected_w, expected_b = rng.normal(size=(3, 5)), rng.normal(size=5)
        w, b = fit_ridge(x, x @ expected_w + expected_b, 1e-8)
        unseen = rng.normal(size=(20, 3))
        np.testing.assert_allclose(unseen @ w + b, unseen @ expected_w + expected_b,
                                   atol=1e-7, rtol=1e-7)

    def test_collinear_features_remain_finite(self):
        column = np.linspace(-2, 2, 30)
        x = np.column_stack((column, column, np.ones(30)))
        y = np.column_stack((3 * column + 2, -column))
        w, b = fit_ridge(x, y, .001)
        self.assertTrue(np.isfinite(w).all() and np.isfinite(b).all())
        self.assertLess(np.mean((x @ w + b - y) ** 2), .001)

    def test_fit_rejects_invalid_inputs(self):
        for x, y, ridge in [(np.ones((2, 2)), np.ones((3, 1)), .01),
                            (np.full((2, 2), np.nan), np.ones((2, 1)), .01),
                            (np.ones((2, 2)), np.ones((2, 1)), 0),
                            (np.ones((2, 2)), np.ones((2, 1)), -1)]:
            with self.subTest(ridge=ridge, shape=y.shape), self.assertRaises(ValueError):
                fit_ridge(x, y, ridge)

    def test_alignment_uses_exact_shared_boundaries_independent_of_row_order(self):
        source = self.capture("source", np.array([30, 10, 20]).reshape(3, 1, 1), [6, 2, 4])
        target = self.capture("target", np.array([[400, 401], [600, 601], [500, 501]])
                              .reshape(3, 1, 2), [4, 6, 5], model="target")
        x, y, report = aligned_values(load_capture(source), load_capture(target))
        np.testing.assert_array_equal(x, [[20], [30]])
        np.testing.assert_array_equal(y, [[400, 401], [600, 601]])
        self.assertEqual(report["aligned_rows"], 2)

    def test_alignment_rejects_no_shared_boundaries_or_different_documents(self):
        source = load_capture(self.capture("source", np.ones((2, 1, 2)), [2, 4]))
        for document, ends in [("train", [3, 5]), ("other", [2, 4])]:
            target = self.capture("target", np.ones((2, 1, 3)), ends, document, "target")
            with self.subTest(document=document), self.assertRaises(ValueError):
                aligned_values(source, load_capture(target))

    def test_capture_rejects_duplicate_keys_and_invalid_boundaries(self):
        for ends in ([2, 2], [0, 2], [-1, 2], [1.5, 2.5]):
            path = self.capture("bad", np.ones((2, 1, 2)), ends)
            with self.subTest(ends=ends), self.assertRaises(ValueError):
                load_capture(path)

    def arguments(self, eval_document="eval", eval_changes=None):
        rng = np.random.default_rng(19)
        w = rng.normal(size=(3, 4))
        arguments = []
        for split, document in (("train", "train"), ("eval", eval_document)):
            x = rng.normal(size=(20, 3))
            for side, values in (("source", x), ("target", x @ w + 2)):
                changes = (eval_changes or {}) if split == "eval" and side == "source" else {}
                path = self.capture(f"{split}_{side}", values[:, None, :],
                                    np.arange(1, 21), document, side, **changes)
                arguments.extend([f"--{split}-{side}", str(path)])
        return arguments + ["--output", str(self.root / "result"), "--ridge", ".01"]

    def test_cli_writes_artifacts_and_refuses_existing_output(self):
        arguments = self.arguments()
        self.assertEqual(main(arguments), 0)
        output = self.root / "result"
        self.assertTrue((output / "mapper.npz").is_file())
        report = json.loads((output / "report.json").read_text())
        self.assertLess(report["held_out_values"]["mse_over_baseline"], 1e-5)
        x, y, _ = aligned_values(load_capture(self.root / "eval_source.npz"),
                                 load_capture(self.root / "eval_target.npz"))
        with np.load(output / "mapper.npz", allow_pickle=False) as mapper:
            prediction = x @ mapper["weight"] + mapper["bias"]
            self.assertAlmostEqual(float(np.mean((prediction - y) ** 2)),
                                   report["held_out_values"]["mse"])
            metadata = json.loads(mapper["metadata_json"].item())
            self.assertEqual(metadata["source"]["model_id"], "source")
            self.assertEqual(metadata["target"]["model_id"], "target")
        with self.assertRaises(FileExistsError):
            main(arguments)

    def test_constant_targets_have_defined_mse_and_undefined_normalized_metrics(self):
        values = np.ones((4, 2))
        metrics = reconstruction_metrics(values, values, np.ones(2))
        self.assertEqual(metrics["mse"], 0)
        self.assertIsNone(metrics["mse_over_baseline"])
        self.assertIsNone(metrics["variance_weighted_r2"])
        json.dumps(metrics, allow_nan=False)

    def test_held_out_report_groups_shuffled_rows_and_uses_training_baseline(self):
        train_x = np.arange(-2, 3, dtype=float)[:, None]
        coefficient = np.array([[1., 2., -1., 3.]])
        intercept = np.array([2., -4., 5., 1.])
        train_y = train_x @ coefficient + intercept
        eval_x = np.arange(3, 8, dtype=float)[:, None]
        eval_y = eval_x @ coefficient + intercept + np.array([
            [.5, -1., 5., -6.], [1., .25, 4., 2.],
            [8., -3., .1, .2], [10., 6., .2, .3], [12., 2., .3, .4],
        ])
        documents = np.array([digest("eval-a")] * 2 + [digest("eval-b")] * 3)
        ends = np.array([1, 2, 1, 2, 3])
        arguments = []
        for side, training, evaluation, order in (
            ("source", train_x[:, None, :], eval_x[:, None, :], [3, 0, 4, 2, 1]),
            ("target", train_y.reshape(5, 2, 2), eval_y.reshape(5, 2, 2), [1, 4, 0, 3, 2]),
        ):
            training_path = self.capture(f"train_{side}", training, np.arange(1, 6), model=side)
            evaluation_path = self.capture(f"eval_{side}", evaluation[order], ends[order],
                                           document="eval", model=side)
            with np.load(evaluation_path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["document_sha256"] = documents[order]
            np.savez(evaluation_path, **arrays)
            arguments.extend([f"--train-{side}", str(training_path),
                              f"--eval-{side}", str(evaluation_path)])
        output = self.root / "grouped-report"
        self.assertEqual(main(arguments + ["--output", str(output), "--ridge", "1"]), 0)
        report = json.loads((output / "report.json").read_text())

        # The centered one-feature training Gram is 10. Ridge=1 therefore
        # shrinks the slope by 10/11 and leaves this centered data's bias intact.
        # This reference never invokes the fitted mapper or reporting helpers.
        expected_prediction = eval_x @ (coefficient * 10 / 11) + intercept
        training_baseline = train_y.mean(axis=0)
        by_document = report["held_out_by_document"]
        self.assertEqual(set(by_document), set(documents))
        self.assertEqual(by_document[digest("eval-a")]["aligned_rows"], 2)
        self.assertEqual(by_document[digest("eval-b")]["aligned_rows"], 3)
        self.assertEqual(len(report["held_out_by_head"]), 2)
        groups = [
            ("all", report["held_out_values"], [0, 1, 2, 3, 4], [0, 1, 2, 3]),
            ("document-a", by_document[digest("eval-a")], [0, 1], [0, 1, 2, 3]),
            ("document-b", by_document[digest("eval-b")], [2, 3, 4], [0, 1, 2, 3]),
            ("head-0", report["held_out_by_head"][0], [0, 1, 2, 3, 4], [0, 1]),
            ("head-1", report["held_out_by_head"][1], [0, 1, 2, 3, 4], [2, 3]),
        ]
        for name, metrics, rows, columns in groups:
            with self.subTest(group=name):
                actual = eval_y[np.ix_(rows, columns)]
                residual = actual - expected_prediction[np.ix_(rows, columns)]
                error_sum = np.square(residual).sum()
                baseline_sum = np.square(actual - training_baseline[columns]).sum()
                centered_sum = np.square(actual - actual.mean(axis=0)).sum()
                self.assertAlmostEqual(metrics["mse"], error_sum / actual.size)
                self.assertAlmostEqual(metrics["train_mean_baseline_mse"],
                                       baseline_sum / actual.size)
                self.assertAlmostEqual(metrics["mse_over_baseline"], error_sum / baseline_sum)
                self.assertAlmostEqual(metrics["variance_weighted_r2"], 1 - error_sum / centered_sum)
                self.assertNotAlmostEqual(metrics["train_mean_baseline_mse"],
                                          centered_sum / actual.size)
        self.assertNotAlmostEqual(by_document[digest("eval-a")]["mse"],
                                  by_document[digest("eval-b")]["mse"])
        self.assertNotAlmostEqual(report["held_out_by_head"][0]["mse"],
                                  report["held_out_by_head"][1]["mse"])

    def test_cli_rejects_train_eval_document_overlap(self):
        with self.assertRaises(ValueError):
            main(self.arguments(eval_document="train"))
        self.assertFalse((self.root / "result").exists())

    def test_cli_rejects_changed_capture_identity(self):
        for changes in ({"model_revision": "r2"}, {"runtime_revision": "runtime2"},
                        {"weight_quantization": "q4"}, {"layer": 1},
                        {"tokenizer_options": {"normalize": True}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                main(self.arguments(eval_changes=changes))
            self.assertFalse((self.root / "result").exists())

    def multi_document_arguments(self, train_documents=10, noise=.5):
        rng = np.random.default_rng(23)
        w = rng.normal(size=(6, 4))
        arguments = []
        for split, count in (("train", train_documents), ("eval", 3)):
            rows = 12 * count
            x = rng.normal(size=(rows, 6))
            y = x @ w + 1 + noise * rng.normal(size=(rows, 4))
            documents = np.array([digest(f"{split}-{index // 12}") for index in range(rows)])
            ends = np.tile(np.arange(1, 13), count)
            for side, values in (("source", x[:, None, :]), ("target", y.reshape(rows, 2, 2))):
                path = self.capture(f"{split}_{side}_multi", values, ends, model=side)
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                arrays["document_sha256"] = documents
                np.savez(path, **arrays)
                arguments.extend([f"--{split}-{side}", str(path)])
        return arguments

    def test_default_ridge_is_grouped_cv_over_documented_grid(self):
        arguments = self.multi_document_arguments()
        output = self.root / "auto"
        self.assertEqual(main(arguments + ["--output", str(output), "--bootstrap", "300"]), 0)
        report = json.loads((output / "report.json").read_text())
        selection = report["ridge_selection"]
        self.assertEqual((selection["method"], selection["folds"], selection["group"]),
                         ("grouped_cv", 8, "document_sha256"))
        np.testing.assert_allclose(selection["grid"]["values"], np.logspace(-2, 7, 91))
        self.assertEqual(len(selection["curve"]), 91)
        source = load_capture(self.root / "train_source_multi.npz")
        target = load_capture(self.root / "train_target_multi.npz")
        x, y, _ = aligned_values(source, target)
        documents = source.documents[probe.aligned_rows(source, target)[0]]
        self.assertFalse(np.array_equal(documents, source.documents))
        expected = ridge.grouped_cv(x, y, documents, ridge.default_grid(), n_folds=8, seed=0)
        np.testing.assert_allclose([row["pooled_mse"] for row in selection["curve"]],
                                   expected["pooled_mse"], rtol=1e-12)
        np.testing.assert_allclose([row["mean_fold_r2"] for row in selection["curve"]],
                                   expected["mean_fold_r2"], rtol=1e-12)
        self.assertEqual(selection["fold_rows"], expected["fold_rows"])
        self.assertEqual(selection["chosen"], expected["chosen_lambda"])
        self.assertEqual(selection["final_fit_solver"], "cholesky")
        for name in ("probe.py", "ridge.py"):
            self.assertEqual(report["code_sha256"][name],
                             hashlib.sha256((Path(probe.__file__).parent / name).read_bytes()).hexdigest())
        self.assertEqual(report["ridge"], expected["chosen_lambda"])
        best = min(selection["curve"], key=lambda row: row["pooled_mse"])
        self.assertEqual(best["lambda"], selection["chosen"])
        with np.load(output / "mapper.npz", allow_pickle=False) as mapper:
            metadata = json.loads(mapper["metadata_json"].item())
            weight, bias = fit_ridge(x, y, selection["chosen"])
            np.testing.assert_allclose(mapper["weight"], weight)
            np.testing.assert_allclose(mapper["bias"], bias)
        self.assertEqual((metadata["ridge"], metadata["ridge_selection_method"]),
                         (selection["chosen"], "grouped_cv"))
        self.assertIn("variance_weighted_r2", report["train_values"])
        for metrics in report["held_out_by_document"].values():
            self.assertEqual(metrics["r2_vs_own_mean"], metrics["variance_weighted_r2"])
            self.assertAlmostEqual(metrics["r2_vs_train_mean"], 1 - metrics["mse_over_baseline"])
        interval = report["held_out_bootstrap_95"]["variance_weighted_r2"]
        self.assertEqual((interval["n_documents"], interval["n_resamples"], interval["level"]), (3, 300, .95))
        self.assertAlmostEqual(interval["estimate"], report["held_out_values"]["variance_weighted_r2"])
        self.assertLessEqual(interval["low"], interval["high"])

    def test_auto_caps_folds_at_training_documents_and_needs_two(self):
        output = self.root / "capped"
        self.assertEqual(main(self.multi_document_arguments(train_documents=3)
                              + ["--output", str(output)]), 0)
        selection = json.loads((output / "report.json").read_text())["ridge_selection"]
        self.assertEqual((selection["folds"], selection["folds_requested"]), (3, 8))
        with self.assertRaises(ValueError):
            main(self.arguments()[:-2] + ["--ridge", "auto"])
        self.assertFalse((self.root / "result").exists())

    def test_fixed_ridge_records_fixed_selection(self):
        self.assertEqual(main(self.arguments()), 0)
        report = json.loads((self.root / "result" / "report.json").read_text())
        self.assertEqual(report["ridge_selection"],
                         {"method": "fixed", "chosen": .01, "final_fit_solver": "cholesky"})
        self.assertEqual(report["ridge"], .01)

    def test_ridge_argument_parsing(self):
        self.assertEqual(probe.parse_ridge("auto"), "auto")
        self.assertEqual(probe.parse_ridge("2.5"), 2.5)
        for text in ("0", "-1", "nan", "inf", "Auto", "x"):
            with self.subTest(text=text), self.assertRaises(Exception):
                probe.parse_ridge(text)


if __name__ == "__main__":
    unittest.main()
