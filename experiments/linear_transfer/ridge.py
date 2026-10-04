"""Centered ridge regression with an unpenalized bias, grouped CV, and document bootstrap.

Objective: ||X W + b - Y||² (summed over rows and outputs) + lambda * ||W||².
Lambda is in summed-squared-error units, so its effect depends on the row count.
"""

import numpy as np


def default_grid():
    """logspace(-2, 7) at ten points per decade."""
    return np.logspace(-2, 7, 91)


def _matrices(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2
            or x.shape[1] == 0 or y.shape[1] == 0
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("ridge requires finite matrices with matching rows and at least two observations")
    return x, y


def _lambdas(lambdas):
    values = np.asarray(lambdas, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(values <= 0) or len(np.unique(values)) != len(values)):
        raise ValueError("lambdas must be a nonempty list of distinct finite positive values")
    return values


def _lambda(lam):
    if isinstance(lam, bool) or not np.isfinite(lam) or lam <= 0:
        raise ValueError("ridge must be finite and positive")
    return float(lam)


def centered_gram(x, y):
    """Return G = Xc'Xc, C = Xc'Yc and the column means of x and y."""
    x, y = _matrices(x, y)
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    x_centered = x - x_mean
    return x_centered.T @ x_centered, x_centered.T @ (y - y_mean), x_mean, y_mean


class RidgePath:
    """All ridge solutions of one centered problem from a single eigendecomposition."""

    def __init__(self, G, C):
        G, C = np.asarray(G, dtype=np.float64), np.asarray(C, dtype=np.float64)
        if (G.ndim != 2 or G.shape[0] != G.shape[1] or C.ndim != 2 or len(C) != len(G)
                or not np.isfinite(G).all() or not np.isfinite(C).all()):
            raise ValueError("ridge path needs a finite square Gram and a matching cross-product")
        eigenvalues, self.basis = np.linalg.eigh((G + G.T) / 2)
        # A centered Gram is positive semidefinite; negative values are rounding.
        self.eigenvalues = np.clip(eigenvalues, 0, None)
        self.rotated = self.basis.T @ C

    def rotated_weight(self, lam):
        return self.rotated / (self.eigenvalues + _lambda(lam))[:, None]

    def weight(self, lam):
        return self.basis @ self.rotated_weight(lam)

    def project(self, x_centered):
        return np.asarray(x_centered, dtype=np.float64) @ self.basis

    def effective_dof(self, lam):
        return float(np.sum(self.eigenvalues / (self.eigenvalues + _lambda(lam))))


def ridge_path(G, C, lambdas):
    """Weights for every lambda, in order, from one eigendecomposition of G."""
    path = RidgePath(G, C)
    return [path.weight(lam) for lam in _lambdas(lambdas)]


def solve_cholesky(G, C, lam):
    """Solve (G + lam I) W = C; G must be symmetric positive semidefinite."""
    G, C = np.asarray(G, dtype=np.float64), np.asarray(C, dtype=np.float64)
    system = (G + G.T) / 2
    system.flat[::len(system) + 1] += _lambda(lam)
    try:
        lower = np.linalg.cholesky(system)
    except np.linalg.LinAlgError as exc:
        raise ValueError("ridge system is not positive definite; inspect input scale") from exc
    weight = np.linalg.solve(lower.T, np.linalg.solve(lower, C))
    if not np.isfinite(weight).all():
        raise ValueError("ridge produced nonfinite weights; inspect input scale")
    return weight


def fit_with_solver(x, y, lam):
    """Return (weight, bias, solver); solver is "cholesky" or "eigen".

    When rounding makes G + lam I fail Cholesky, the weights come from the
    eigendecomposition with eigenvalues clipped at zero, as in the CV path.
    """
    lam = _lambda(lam)
    G, C, x_mean, y_mean = centered_gram(x, y)
    try:
        weight, solver = solve_cholesky(G, C, lam), "cholesky"
    except ValueError:
        weight, solver = RidgePath(G, C).weight(lam), "eigen"
        if not np.isfinite(weight).all():
            raise ValueError("ridge produced nonfinite weights; inspect input scale") from None
    bias = y_mean - x_mean @ weight
    if not np.isfinite(bias).all():
        raise ValueError("ridge produced a nonfinite bias; inspect input scale")
    return weight, bias, solver


def fit(x, y, lam):
    """Return (weight, bias) of the centered ridge with an unpenalized bias."""
    weight, bias, _ = fit_with_solver(x, y, lam)
    return weight, bias


def grouped_folds(groups, n_folds, seed=0):
    """Fold index per row: sorted unique groups, seeded permutation, position mod n_folds."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("groups must be a nonempty one-dimensional array")
    if type(n_folds) is not int or n_folds < 2:
        raise ValueError("grouped CV needs at least two folds")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < n_folds:
        raise ValueError(f"grouped CV needs at least {n_folds} groups; found {len(unique)}")
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of_group = np.empty(len(unique), dtype=np.int64)
    fold_of_group[order] = np.arange(len(unique)) % n_folds
    return fold_of_group[inverse]


def sum_squares_about_mean(y):
    return float(np.sum(np.square(y - y.mean(axis=0))))


def grouped_cv(x, y, groups, lambdas, n_folds=8, seed=0, return_oof=False):
    """Grouped K-fold ridge path; lambda chosen by minimum pooled held-out MSE.

    Each fold is centered and fitted on its own training rows only. Returns a dict
    with per-lambda `pooled_mse`, `pooled_mse_over_baseline` (baseline = fold
    training mean), `mean_fold_r2` (each fold against its own mean), and the choice.
    """
    x, y = _matrices(x, y)
    lambdas = _lambdas(lambdas)
    groups = np.asarray(groups)
    if groups.shape != (len(x),):
        raise ValueError("groups must have one entry per row")
    folds = grouped_folds(groups, n_folds, seed)
    sse = np.zeros(len(lambdas))
    fold_r2 = np.zeros((len(lambdas), n_folds))
    baseline_sse, fold_rows, fold_groups = 0.0, [], []
    oof = np.zeros_like(y) if return_oof else None
    paths = []
    for fold in range(n_folds):
        held = folds == fold
        train = ~held
        if train.sum() < 2:
            raise ValueError("each CV training split needs at least two rows")
        G, C, x_mean, y_mean = centered_gram(x[train], y[train])
        path = RidgePath(G, C)
        projected = path.project(x[held] - x_mean)
        target = y[held]
        total = sum_squares_about_mean(target)
        baseline_sse += float(np.sum(np.square(target - y_mean)))
        for index, lam in enumerate(lambdas):
            error = float(np.sum(np.square(target - (projected @ path.rotated_weight(lam) + y_mean))))
            sse[index] += error
            fold_r2[index, fold] = 1 - error / total if total > 0 else np.nan
        fold_rows.append(int(held.sum()))
        fold_groups.append(int(len(np.unique(groups[held]))))
        if return_oof:
            paths.append((held, path.rotated, path.eigenvalues, projected, y_mean))
    elements = y.size
    chosen = int(np.argmin(sse))
    result = {
        "method": "grouped_kfold",
        "n_folds": n_folds,
        "seed": seed,
        "n_groups": int(len(np.unique(groups))),
        "fold_rows": fold_rows,
        "fold_groups": fold_groups,
        "lambdas": lambdas.tolist(),
        "pooled_mse": (sse / elements).tolist(),
        "pooled_baseline_mse": baseline_sse / elements,
        "pooled_mse_over_baseline": (sse / baseline_sse).tolist() if baseline_sse > 0 else None,
        "mean_fold_r2": [float(v) for v in np.mean(fold_r2, axis=1)],
        "chosen_index": chosen,
        "chosen_lambda": float(lambdas[chosen]),
        "chosen_fold_r2": fold_r2[chosen].tolist(),
        "chosen_at_grid_edge": chosen in (0, len(lambdas) - 1),
    }
    if return_oof:
        baseline = np.zeros_like(y)
        for held, rotated, eigenvalues, projected, y_mean in paths:
            oof[held] = projected @ (rotated / (eigenvalues + lambdas[chosen])[:, None]) + y_mean
            baseline[held] = y_mean
        result["oof"] = oof
        result["oof_baseline"] = baseline
        result["folds"] = folds
    return result


def document_bootstrap(stats, statistic, n_resamples=2000, seed=0, level=0.95):
    """Percentile CI by resampling documents with replacement.

    `stats` maps names to arrays whose first axis is the document and whose values
    are additive over rows; `statistic` maps the summed dict to a value per
    resample (vectorized over a leading resample axis).
    """
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in stats.items()}
    counts = {len(value) for value in arrays.values()}
    if len(counts) != 1 or 0 in counts:
        raise ValueError("bootstrap stats need one nonempty row per document")
    if type(n_resamples) is not int or n_resamples < 1 or not 0 < level < 1:
        raise ValueError("invalid bootstrap resample count or level")
    documents = counts.pop()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, documents, size=(n_resamples, documents))
    weights = np.zeros((n_resamples, documents))
    np.add.at(weights, (np.repeat(np.arange(n_resamples), documents), draws.ravel()), 1)

    def aggregate(w):
        return {name: np.tensordot(w, value, axes=(1, 0)) for name, value in arrays.items()}

    point = np.asarray(statistic(aggregate(np.ones((1, documents)))), dtype=np.float64)[0]
    values = np.asarray(statistic(aggregate(weights)), dtype=np.float64)
    tail = (1 - level) / 2 * 100
    low, high = np.nanpercentile(values, [tail, 100 - tail], axis=0)
    return {"estimate": point.tolist(), "low": low.tolist(), "high": high.tolist(),
            "level": level, "n_resamples": n_resamples, "seed": seed,
            "n_documents": documents, "method": "document_percentile_bootstrap"}


def document_stats(actual, predicted, train_mean, documents):
    """Additive per-document sums for R² (own mean) and MSE / train-mean baseline.

    `train_mean` is one baseline row, or one per row (e.g. fold training means).
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    train_mean = np.asarray(train_mean, dtype=np.float64)
    documents = np.asarray(documents)
    if (actual.ndim != 2 or predicted.shape != actual.shape or documents.shape != (len(actual),)
            or train_mean.shape not in (actual.shape[1:], actual.shape)):
        raise ValueError("document stats need matching actual/predicted rows, documents, and baseline")
    names = np.unique(documents)
    stats = {"rows": [], "sse": [], "sse_baseline": [], "sum": [], "sum_squares": []}
    for name in names:
        rows = documents == name
        target = actual[rows]
        stats["rows"].append(float(rows.sum()))
        stats["sse"].append(float(np.sum(np.square(target - predicted[rows]))))
        baseline = train_mean[rows] if train_mean.ndim == 2 else train_mean
        stats["sse_baseline"].append(float(np.sum(np.square(target - baseline))))
        stats["sum"].append(target.sum(axis=0))
        stats["sum_squares"].append(float(np.sum(np.square(target))))
    return names, {name: np.asarray(value) for name, value in stats.items()}


def r2_from_stats(total):
    """Variance-weighted R² about the pooled own mean, from summed document stats."""
    about_mean = total["sum_squares"] - np.sum(np.square(total["sum"]), axis=-1) / total["rows"]
    return 1 - total["sse"] / about_mean


def mse_over_baseline_from_stats(total):
    return total["sse"] / total["sse_baseline"]
