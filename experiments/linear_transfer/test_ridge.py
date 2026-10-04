"""Numerical checks for centered ridge, grouped CV, and the document bootstrap."""

import unittest
from unittest import mock

import numpy as np

import ridge


def reference_fit(x, y, lam):
    """Augmented normal equations with an unpenalized bias column."""
    design = np.column_stack((np.ones(len(x)), x))
    penalty = lam * np.eye(design.shape[1])
    penalty[0, 0] = 0
    solution = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return solution[1:], solution[0]


def problem(seed=3, rows=120, features=6, outputs=4, groups=12, noise=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, features)) * np.linspace(.5, 3, features)
    true = rng.normal(size=(features, outputs))
    y = x @ true + 2 + noise * rng.normal(size=(rows, outputs))
    documents = np.array([f"doc{index % groups:02d}" for index in range(rows)])
    return x, y, documents


class RidgeTests(unittest.TestCase):
    def test_fit_matches_augmented_normal_equations(self):
        x, y, _ = problem()
        for lam in (1e-6, .3, 50.0):
            with self.subTest(lam=lam):
                weight, bias = ridge.fit(x, y, lam)
                expected_weight, expected_bias = reference_fit(x, y, lam)
                np.testing.assert_allclose(weight, expected_weight, rtol=1e-9, atol=1e-10)
                np.testing.assert_allclose(bias, expected_bias, rtol=1e-9, atol=1e-10)

    def test_path_matches_cholesky_for_every_lambda(self):
        x, y, _ = problem(features=9)
        G, C, _, _ = ridge.centered_gram(x, y)
        lambdas = [1e-3, 1.0, 10.0, 1e4]
        for lam, weight in zip(lambdas, ridge.ridge_path(G, C, lambdas)):
            with self.subTest(lam=lam):
                np.testing.assert_allclose(weight, ridge.solve_cholesky(G, C, lam), rtol=1e-8, atol=1e-11)

    def test_more_features_than_rows_is_finite(self):
        x, y, _ = problem(rows=10, features=30)
        G, C, _, _ = ridge.centered_gram(x, y)
        path = ridge.RidgePath(G, C)
        self.assertTrue(np.all(path.eigenvalues >= 0))
        np.testing.assert_allclose(path.weight(.5), ridge.solve_cholesky(G, C, .5), rtol=1e-7, atol=1e-9)
        self.assertLessEqual(path.effective_dof(1e-4), 9 + 1e-3)

    def test_fit_falls_back_to_eigen_path_when_cholesky_fails(self):
        x, y, _ = problem()
        cholesky_weight, cholesky_bias, solver = ridge.fit_with_solver(x, y, .3)
        self.assertEqual(solver, "cholesky")
        failure = ValueError("ridge system is not positive definite; inspect input scale")
        with mock.patch.object(ridge, "solve_cholesky", side_effect=failure):
            weight, bias, solver = ridge.fit_with_solver(x, y, .3)
            self.assertEqual(ridge.fit(x, y, .3)[0].shape, weight.shape)
        self.assertEqual(solver, "eigen")
        np.testing.assert_allclose(weight, cholesky_weight, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(bias, cholesky_bias, rtol=1e-8, atol=1e-10)
        G, C, _, _ = ridge.centered_gram(x, y)
        np.testing.assert_allclose(weight, ridge.RidgePath(G, C).weight(.3), rtol=1e-12)

    def test_nearly_collinear_large_scale_fit_stays_finite(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            column = rng.normal(0, 1e4, 200)
            x = np.column_stack((column, column * 1.0000001, rng.normal(size=200)))
            y = rng.normal(size=(200, 2))
            weight, bias, solver = ridge.fit_with_solver(x, y, 1e-6)
            with self.subTest(seed=seed, solver=solver):
                self.assertTrue(np.isfinite(weight).all() and np.isfinite(bias).all())
                self.assertIn(solver, ("cholesky", "eigen"))

    def test_invariant_to_output_space_linear_transforms(self):
        x, y, _ = problem()
        transform = np.random.default_rng(5).normal(size=(4, 4))
        for lam in (.01, 100.0):
            weight, bias = ridge.fit(x, y, lam)
            mapped_weight, mapped_bias = ridge.fit(x, y @ transform, lam)
            np.testing.assert_allclose(mapped_weight, weight @ transform, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(mapped_bias, bias @ transform, rtol=1e-8, atol=1e-10)

    def test_rejects_invalid_inputs(self):
        x, y, documents = problem()
        with self.assertRaises(ValueError):
            ridge.fit(x, y[:-1], 1.0)
        for lam in (0, -1, np.nan, np.inf, True):
            with self.subTest(lam=lam), self.assertRaises(ValueError):
                ridge.fit(x, y, lam)
        with self.assertRaises(ValueError):
            ridge.solve_cholesky(-np.eye(3), np.ones((3, 1)), 1e-3)
        for lambdas in ([], [1, 1], [1, -2], [[1, 2]]):
            with self.subTest(lambdas=lambdas), self.assertRaises(ValueError):
                ridge.grouped_cv(x, y, documents, lambdas)

    def test_grouped_folds_follow_documented_rule(self):
        groups = np.array(["c", "a", "b", "a", "e", "d", "c", "f", "b"])
        folds = ridge.grouped_folds(groups, 3, seed=4)
        unique = np.array(sorted(set(groups)))
        order = np.random.default_rng(4).permutation(len(unique))
        expected = {unique[order[position]]: position % 3 for position in range(len(unique))}
        self.assertEqual({g: int(f) for g, f in zip(groups, folds)}, expected)
        self.assertEqual(sorted(np.bincount(list(expected.values()))), [2, 2, 2])
        shuffled = np.random.default_rng(0).permutation(len(groups))
        np.testing.assert_array_equal(ridge.grouped_folds(groups[shuffled], 3, seed=4), folds[shuffled])
        with self.assertRaises(ValueError):
            ridge.grouped_folds(groups, 7)
        with self.assertRaises(ValueError):
            ridge.grouped_folds(groups, 1)

    def test_grouped_cv_matches_brute_force(self):
        x, y, documents = problem(features=8, noise=4.0)
        lambdas = np.logspace(-2, 4, 13)
        result = ridge.grouped_cv(x, y, documents, lambdas, n_folds=4, seed=2, return_oof=True)
        folds = ridge.grouped_folds(documents, 4, seed=2)
        sse, baseline, fold_r2 = np.zeros(len(lambdas)), 0.0, np.zeros((len(lambdas), 4))
        chosen_oof = None
        for fold in range(4):
            held = folds == fold
            baseline += np.sum((y[held] - y[~held].mean(axis=0)) ** 2)
            for index, lam in enumerate(lambdas):
                weight, bias = reference_fit(x[~held], y[~held], lam)
                error = np.sum((y[held] - x[held] @ weight - bias) ** 2)
                sse[index] += error
                fold_r2[index, fold] = 1 - error / np.sum((y[held] - y[held].mean(axis=0)) ** 2)
        np.testing.assert_allclose(result["pooled_mse"], sse / y.size, rtol=1e-9)
        np.testing.assert_allclose(result["pooled_mse_over_baseline"], sse / baseline, rtol=1e-9)
        np.testing.assert_allclose(result["mean_fold_r2"], fold_r2.mean(axis=1), rtol=1e-9)
        chosen = int(np.argmin(sse))
        self.assertEqual(result["chosen_index"], chosen)
        self.assertEqual(result["chosen_lambda"], lambdas[chosen])
        chosen_oof = np.zeros_like(y)
        for fold in range(4):
            held = folds == fold
            weight, bias = reference_fit(x[~held], y[~held], lambdas[chosen])
            chosen_oof[held] = x[held] @ weight + bias
            np.testing.assert_allclose(result["oof_baseline"][held],
                                       np.broadcast_to(y[~held].mean(axis=0), y[held].shape))
        np.testing.assert_allclose(result["oof"], chosen_oof, rtol=1e-8, atol=1e-9)
        self.assertEqual(sum(result["fold_rows"]), len(y))
        self.assertEqual(sum(result["fold_groups"]), 12)

    def test_grouped_cv_prefers_regularization_for_pure_noise(self):
        rng = np.random.default_rng(11)
        x, y = rng.normal(size=(80, 40)), rng.normal(size=(80, 3))
        documents = np.repeat(np.arange(16), 5)
        result = ridge.grouped_cv(x, y, documents, ridge.default_grid(), n_folds=8)
        self.assertGreater(result["chosen_lambda"], 10)
        self.assertEqual(len(ridge.default_grid()), 91)
        self.assertAlmostEqual(ridge.default_grid()[10], .1)

    def test_bootstrap_estimate_ci_and_determinism(self):
        x, y, documents = problem(noise=2.0)
        weight, bias = ridge.fit(x, y, 1.0)
        prediction = x @ weight + bias
        train_mean = y.mean(axis=0) + .3
        names, stats = ridge.document_stats(y, prediction, train_mean, documents)
        self.assertEqual(list(names), sorted(set(documents)))
        first = ridge.document_bootstrap(stats, ridge.r2_from_stats, 500, seed=9)
        second = ridge.document_bootstrap(stats, ridge.r2_from_stats, 500, seed=9)
        self.assertEqual(first, second)
        direct = 1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean(axis=0)) ** 2)
        self.assertAlmostEqual(first["estimate"], direct)
        self.assertLessEqual(first["low"], direct)
        self.assertGreaterEqual(first["high"], direct)
        ratio = ridge.document_bootstrap(stats, ridge.mse_over_baseline_from_stats, 200)
        self.assertAlmostEqual(ratio["estimate"], np.sum((y - prediction) ** 2) / np.sum((y - train_mean) ** 2))
        per_row = np.broadcast_to(train_mean, y.shape)
        _, row_stats = ridge.document_stats(y, prediction, per_row, documents)
        np.testing.assert_allclose(row_stats["sse_baseline"], stats["sse_baseline"])
        single = ridge.document_bootstrap({k: v[:1] for k, v in stats.items()}, ridge.r2_from_stats, 50)
        self.assertEqual(single["low"], single["high"])
        with self.assertRaises(ValueError):
            ridge.document_bootstrap({"a": np.ones(3), "b": np.ones(4)}, lambda t: t["a"])

    def test_paired_bootstrap_of_a_difference(self):
        x, y, documents = problem(noise=2.0)
        full = x @ ridge.fit(x, y, 1.0)[0] + ridge.fit(x, y, 1.0)[1]
        partial = x[:, :2] @ ridge.fit(x[:, :2], y, 1.0)[0] + ridge.fit(x[:, :2], y, 1.0)[1]
        _, a = ridge.document_stats(y, full, y.mean(axis=0), documents)
        _, b = ridge.document_stats(y, partial, y.mean(axis=0), documents)
        stats = {**{"a_" + k: v for k, v in a.items()}, **{"b_" + k: v for k, v in b.items()}}

        def delta(total):
            pick = lambda prefix: {k[2:]: v for k, v in total.items() if k.startswith(prefix)}
            return ridge.r2_from_stats(pick("a_")) - ridge.r2_from_stats(pick("b_"))
        result = ridge.document_bootstrap(stats, delta, 300)
        self.assertGreater(result["low"], 0)


if __name__ == "__main__":
    unittest.main()
