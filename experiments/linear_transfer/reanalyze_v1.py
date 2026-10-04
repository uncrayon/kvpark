"""Re-report the v1 value probe with CV ridge, a receiver-side baseline, and output-space R².

Reads the v1 captures (source/target train/eval NPZ) and, from the Muse GGUF, only
metadata, the token_embd rows of tokens present in the target captures, and the
target layer's attn_output. Writes a new directory; never loads a model.
"""

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "8")

import numpy as np

import gguf_tensors
import probe
import ridge


HERE = Path(__file__).resolve().parent
CODE_FILES = ("reanalyze_v1.py", "ridge.py", "probe.py", "gguf_tensors.py")
CAPTURES = ("source-train", "target-train", "source-eval", "target-eval")
ARCHITECTURE = "muse-glimmer"


def sha256_bytes(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_positions(documents, byte_end):
    """Token position within each document; rows must be contiguous and byte ordered."""
    positions = np.zeros(len(documents), dtype=np.int64)
    seen = set()
    for row in range(len(documents)):
        if row and documents[row] == documents[row - 1]:
            if byte_end[row] <= byte_end[row - 1]:
                raise ValueError("target capture rows are not in byte order within a document")
            positions[row] = positions[row - 1] + 1
        else:
            if documents[row] in seen:
                raise ValueError("target capture documents are not contiguous")
            seen.add(documents[row])
    return positions


def load_v1(directory, lags):
    captures = {name: probe.load_capture(directory / f"{name}.npz") for name in CAPTURES}
    probe.validate_split(*(captures[name] for name in CAPTURES))
    target = captures["target-train"].metadata
    if target.get("architecture") != ARCHITECTURE:
        raise ValueError(f"target captures must come from {ARCHITECTURE}")
    data = {"source_metadata": captures["source-train"].metadata, "target_metadata": target}
    for split in ("train", "eval"):
        source, captured = captures[f"source-{split}"], captures[f"target-{split}"]
        with np.load(directory / f"target-{split}.npz", allow_pickle=False) as archive:
            tokens = archive["token_id"].astype(np.int64)
        if tokens.shape != (len(captured.values),) or np.any(tokens < 0):
            raise ValueError(f"target-{split} lacks valid token ids")
        positions = document_positions(captured.documents, captured.byte_end)
        source_rows, target_rows = map(np.asarray, probe.aligned_rows(source, captured))
        history = np.full((len(target_rows), lags + 1), -1, dtype=np.int64)
        for lag in range(lags + 1):
            present = positions[target_rows] >= lag
            history[present, lag] = tokens[target_rows[present] - lag]
        data[split] = {
            "x": source.values[source_rows].reshape(len(source_rows), -1).astype(np.float64),
            "y": captured.values[target_rows].reshape(len(target_rows), -1).astype(np.float64),
            "documents": captured.documents[target_rows],
            "positions": positions[target_rows],
            "history": history,
            "stream_tokens": tokens,
            "target_rows": len(tokens),
            "source_rows": len(source.values),
        }
    data["heads"], data["head_dim"] = captures["target-train"].values.shape[1:]
    return data


def rmsnorm(values, eps):
    values = np.asarray(values, dtype=np.float64)
    return values / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + eps)


def load_muse(path, token_ids, layer, heads, head_dim):
    metadata = gguf_tensors.read_metadata(path)
    prefix = ARCHITECTURE + "."
    if metadata.get("general.architecture") != ARCHITECTURE:
        raise ValueError(f"GGUF architecture must be {ARCHITECTURE}")
    required = {name: metadata.get(prefix + name) for name in (
        "embedding_length", "block_count", "attention.head_count", "attention.head_count_kv",
        "attention.key_length", "attention.value_length", "attention.layer_norm_rms_epsilon")}
    if any(value is None for value in required.values()):
        raise ValueError("GGUF lacks required Muse hyperparameters")
    n_head, n_head_kv = required["attention.head_count"], required["attention.head_count_kv"]
    if (n_head_kv != heads or required["attention.value_length"] != head_dim
            or n_head % n_head_kv or not 0 <= layer < required["block_count"]):
        raise ValueError("GGUF head layout or layer does not match the target captures")
    embeddings = gguf_tensors.read_tensor(path, "token_embd.weight", token_ids)
    if embeddings.shape != (len(token_ids), required["embedding_length"]):
        raise ValueError("unexpected token_embd shape")
    output = gguf_tensors.read_tensor(path, f"blk.{layer}.attn_output.weight")
    if output.shape != (required["embedding_length"], n_head * head_dim):
        raise ValueError("unexpected attn_output shape")
    stat = os.stat(path)
    identity = {
        "file_name": Path(path).name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "file_type": metadata.get("general.file_type"),
        "name": metadata.get("general.name"),
        "hyperparameters": required,
        "token_embd": gguf_tensors.tensor_info(path, "token_embd.weight")["type"],
        "attn_output": gguf_tensors.tensor_info(path, f"blk.{layer}.attn_output.weight")["type"],
        "token_embd_rows_read": len(token_ids),
        "dequantized_sha256": {"token_embd_rows": sha256_bytes(token_ids, embeddings),
                               "attn_output": sha256_bytes(output)},
    }
    return {"embeddings": embeddings, "attn_output": output, "n_head": n_head,
            "eps": float(required["attention.layer_norm_rms_epsilon"]), "identity": identity}


def output_metric(attn_output, n_head, n_head_kv, head_dim):
    """M_g = sum over the q heads reading kv head g of W_h' W_h, and T = blockdiag(M_g^1/2)."""
    group = n_head // n_head_kv
    size = n_head_kv * head_dim
    transform, spectra = np.zeros((size, size)), []
    for kv_head in range(n_head_kv):
        metric = np.zeros((head_dim, head_dim))
        for head in range(kv_head * group, (kv_head + 1) * group):
            block = attn_output[:, head * head_dim:(head + 1) * head_dim].astype(np.float64)
            metric += block.T @ block
        eigenvalues, basis = np.linalg.eigh(metric)
        if eigenvalues.min() <= 0:
            raise ValueError("output metric is not positive definite")
        rows = slice(kv_head * head_dim, (kv_head + 1) * head_dim)
        transform[rows, rows] = (basis * np.sqrt(eigenvalues)) @ basis.T
        spectra.append({"trace": float(eigenvalues.sum()), "max": float(eigenvalues.max()),
                        "min": float(eigenvalues.min()),
                        "condition": float(eigenvalues.max() / eigenvalues.min())})
    return transform, {"q_heads_per_kv_head": group, "grouping": "contiguous: q head h reads kv head h // group",
                       "metric": "M_g = sum_h W_o[:, h]' W_o[:, h] (incoherent, gate and post-norm ignored)",
                       "spectra": spectra}


class LexicalBasis:
    """Occurrence-weighted PCA of RMSNorm(token_embd) over the training target token stream."""

    def __init__(self, token_ids, normalized, train_stream):
        self.index = {int(token): row for row, token in enumerate(token_ids)}
        unique, counts = np.unique(train_stream, return_counts=True)
        rows = np.array([self.index[int(token)] for token in unique])
        weights = counts / counts.sum()
        self.mean = weights @ normalized[rows]
        centered = np.sqrt(weights)[:, None] * (normalized[rows] - self.mean)
        eigenvalues, vectors = np.linalg.eigh(centered @ centered.T)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, vectors = eigenvalues[order], vectors[:, order]
        keep = eigenvalues > eigenvalues[0] * 1e-12
        self.eigenvalues = eigenvalues[keep]
        basis = centered.T @ vectors[:, keep] / np.sqrt(self.eigenvalues)
        self.rank = basis.shape[1]
        # Last row: absent history (before position 0) maps to the mean embedding.
        self.table = np.vstack(((normalized - self.mean) @ basis, np.zeros((1, self.rank))))

    def check_rank(self, rank):
        if not 1 <= rank <= self.rank:
            raise ValueError(f"requested lexical rank {rank} exceeds the basis rank {self.rank}")
        return rank

    def features(self, tokens, rank):
        rows = np.array([self.index[int(token)] if token >= 0 else len(self.table) - 1 for token in tokens])
        return self.table[rows, :self.check_rank(rank)]

    def variance_fraction(self, rank):
        return float(self.eigenvalues[:self.check_rank(rank)].sum() / self.eigenvalues.sum())


def position_features(positions):
    p = positions.astype(np.float64)
    return np.stack((p == 0, p == 1, p == 2, np.log1p(p)), axis=1).astype(np.float64)


def scaled_blocks(blocks):
    """Scale each train block to unit mean squared centered row norm, times sqrt(weight)."""
    train, evaluation, described = [], [], []
    for name, train_block, eval_block, weight in blocks:
        centered = train_block - train_block.mean(axis=0)
        norm = np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
        if not norm > 0:
            raise ValueError(f"feature block {name} is constant on the training rows")
        scale = np.sqrt(weight) / norm
        train.append(train_block * scale)
        evaluation.append(eval_block * scale)
        described.append({"block": name, "features": int(train_block.shape[1]), "weight": weight})
    return np.hstack(train), np.hstack(evaluation), described


def value_metrics(actual, predicted, baseline, heads, head_dim):
    error = float(np.mean(np.square(actual - predicted)))
    base = float(np.mean(np.square(actual - baseline)))
    variance = float(np.mean(np.square(actual - actual.mean(axis=0))))
    per_head = []
    for head in range(heads):
        columns = slice(head * head_dim, (head + 1) * head_dim)
        per_head.append(1 - float(np.mean(np.square(actual[:, columns] - predicted[:, columns])))
                        / float(np.mean(np.square(actual[:, columns] - actual[:, columns].mean(axis=0)))))
    return {"mse": error, "train_mean_baseline_mse": base, "mse_over_baseline": error / base,
            "r2_vs_own_mean": 1 - error / variance, "r2_vs_train_mean": 1 - error / base,
            "head_r2_vs_own_mean": per_head}


def mean_fold_r2(actual, predicted, folds):
    values = []
    for fold in np.unique(folds):
        rows = folds == fold
        error = float(np.sum(np.square(actual[rows] - predicted[rows])))
        values.append(1 - error / ridge.sum_squares_about_mean(actual[rows]))
    return float(np.mean(values)), values


def interval(stats, statistic, args):
    with np.errstate(divide="ignore", invalid="ignore"):
        return ridge.document_bootstrap(stats, statistic, args.bootstrap, args.bootstrap_seed)


def split_stats(actual, predicted, baseline, documents, transform):
    stats = {}
    for space, matrix in (("value", None), ("output", transform)):
        a, p, b = (actual, predicted, baseline) if matrix is None else (
            actual @ matrix, predicted @ matrix, baseline @ matrix)
        _, values = ridge.document_stats(a, p, b, documents)
        stats.update({f"{space}_{key}": value for key, value in values.items()})
    return stats


def pick(total, prefix):
    return {key[len(prefix):]: value for key, value in total.items() if key.startswith(prefix)}


def bootstrap_model(stats, args):
    return {
        "r2_vs_own_mean": interval(stats, lambda t: ridge.r2_from_stats(pick(t, "value_")), args),
        "mse_over_baseline": interval(stats, lambda t: ridge.mse_over_baseline_from_stats(pick(t, "value_")), args),
        "output_space_r2": interval(stats, lambda t: ridge.r2_from_stats(pick(t, "output_")), args),
    }


def run_model(name, blocks, data, args, transform, grid, raw=False):
    """Grouped-CV lambda on training documents; refit on all training rows; score eval once."""
    if raw:
        (block_name, train_x, eval_x, _), = blocks
        described = [{"block": block_name, "features": int(train_x.shape[1]), "weight": None,
                      "scaling": "raw"}]
    else:
        train_x, eval_x, described = scaled_blocks(blocks)
    train, evaluation = data["train"], data["eval"]
    cv = ridge.grouped_cv(train_x, train["y"], train["documents"], grid, n_folds=args.folds,
                          seed=args.seed, return_oof=True)
    lam = cv["chosen_lambda"]
    weight, bias, solver = ridge.fit_with_solver(train_x, train["y"], lam)
    prediction = eval_x @ weight + bias
    train_mean = train["y"].mean(axis=0)
    heads, head_dim = data["heads"], data["head_dim"]
    folds, oof = cv["folds"], cv["oof"]
    minimum = min(cv["pooled_mse"])
    plateau = {f"within_{int(tolerance * 1000) / 10}pct": [
        min(l for l, m in zip(cv["lambdas"], cv["pooled_mse"]) if m <= minimum * (1 + tolerance)),
        max(l for l, m in zip(cv["lambdas"], cv["pooled_mse"]) if m <= minimum * (1 + tolerance))]
        for tolerance in (.005, .01)}
    output_fold_mean, output_folds = mean_fold_r2(train["y"] @ transform, oof @ transform, folds)
    result = {
        "features": int(train_x.shape[1]),
        "blocks": described,
        "cv": {
            "chosen_lambda": lam,
            "chosen_at_grid_edge": cv["chosen_at_grid_edge"],
            "final_fit_solver": solver,
            "pooled_mse": cv["pooled_mse"][cv["chosen_index"]],
            "pooled_mse_over_baseline": cv["pooled_mse_over_baseline"][cv["chosen_index"]],
            "mean_fold_r2": cv["mean_fold_r2"][cv["chosen_index"]],
            "fold_r2": cv["chosen_fold_r2"],
            "output_space_mean_fold_r2": output_fold_mean,
            "output_space_fold_r2": output_folds,
            "lambda_plateau": plateau,
            "curve": [{"lambda": l, "pooled_mse": m, "pooled_mse_over_baseline": r, "mean_fold_r2": q}
                      for l, m, r, q in zip(cv["lambdas"], cv["pooled_mse"],
                                            cv["pooled_mse_over_baseline"], cv["mean_fold_r2"])],
        },
        "eval": {**value_metrics(evaluation["y"], prediction, train_mean, heads, head_dim),
                 "output_space": value_metrics(evaluation["y"] @ transform, prediction @ transform,
                                               train_mean @ transform, heads, head_dim)},
        "train_in_sample": value_metrics(train["y"], train_x @ weight + bias, train_mean, heads, head_dim),
    }
    eval_stats = split_stats(evaluation["y"], prediction, train_mean, evaluation["documents"], transform)
    oof_stats = split_stats(train["y"], oof, cv["oof_baseline"], train["documents"], transform)
    result["bootstrap_95"] = {"eval_documents": bootstrap_model(eval_stats, args),
                              "cv_oof_train_documents": bootstrap_model(oof_stats, args)}
    print(json.dumps({"model": name, "lambda": lam, "cv_mean_fold_r2": result["cv"]["mean_fold_r2"],
                      "eval_r2": result["eval"]["r2_vs_own_mean"]}), flush=True)
    arrays = {"eval_prediction": prediction, "oof_prediction": oof, "oof_folds": folds}
    return result, arrays, eval_stats, oof_stats, cv


def select_source_weight(name, make_blocks, weights, data, args, transform, grid):
    """Joint choice of the source block weight and lambda by minimum pooled CV MSE."""
    runs = {weight: run_model(f"{name}[source_weight={weight}]", make_blocks(weight), data, args,
                              transform, grid) for weight in weights}
    chosen = min(weights, key=lambda weight: runs[weight][0]["cv"]["pooled_mse"])
    result = runs[chosen]
    result[0]["source_weight_selection"] = {
        "chosen": chosen,
        "chosen_at_grid_edge": chosen in (min(weights), max(weights)),
        "candidates": [{"source_weight": weight, "lambda": runs[weight][0]["cv"]["chosen_lambda"],
                        "pooled_mse": runs[weight][0]["cv"]["pooled_mse"],
                        "mean_fold_r2": runs[weight][0]["cv"]["mean_fold_r2"]} for weight in weights]}
    return result


def increment(larger, smaller, args):
    """Paired R² difference (larger minus smaller) on CV folds, eval, and document bootstraps."""
    big, big_arrays, big_eval, big_oof, _ = larger
    small, small_arrays, small_eval, small_oof, _ = smaller
    fold_delta = np.subtract(big["cv"]["fold_r2"], small["cv"]["fold_r2"])
    result = {
        "cv_mean_fold_delta_r2": float(fold_delta.mean()),
        "cv_fold_delta_r2": fold_delta.tolist(),
        "cv_min_fold_delta_r2": float(fold_delta.min()),
        "cv_output_space_mean_fold_delta_r2": float(np.mean(np.subtract(
            big["cv"]["output_space_fold_r2"], small["cv"]["output_space_fold_r2"]))),
        "eval_delta_r2": big["eval"]["r2_vs_own_mean"] - small["eval"]["r2_vs_own_mean"],
        "eval_output_space_delta_r2": (big["eval"]["output_space"]["r2_vs_own_mean"]
                                       - small["eval"]["output_space"]["r2_vs_own_mean"]),
        "eval_partial_r2": 1 - big["eval"]["mse"] / small["eval"]["mse"],
    }
    for label, first, second in (("eval_documents", big_eval, small_eval),
                                 ("cv_oof_train_documents", big_oof, small_oof)):
        stats = {**{"a_" + k: v for k, v in first.items()}, **{"b_" + k: v for k, v in second.items()}}
        result.setdefault("bootstrap_95", {})[label] = {
            "delta_r2": interval(stats, lambda t: ridge.r2_from_stats(pick(t, "a_value_"))
                                 - ridge.r2_from_stats(pick(t, "b_value_")), args),
            "delta_output_space_r2": interval(stats, lambda t: ridge.r2_from_stats(pick(t, "a_output_"))
                                              - ridge.r2_from_stats(pick(t, "b_output_")), args),
        }
    return result


def invariance_check(x, y, transform, lambdas):
    inverse = np.linalg.inv(transform)
    checks = {}
    for lam in lambdas:
        weight, bias = ridge.fit(x, y, lam)
        mapped_weight, mapped_bias = ridge.fit(x, y @ transform, lam)
        checks[repr(float(lam))] = {
            "max_abs_weight_difference": float(np.abs(mapped_weight @ inverse - weight).max()),
            "max_abs_weight": float(np.abs(weight).max()),
            "max_abs_bias_difference": float(np.abs(mapped_bias @ inverse - bias).max()),
        }
    return checks


def rename_noreplace(source, destination):
    """Rename a directory into place; never replaces an existing path, even an empty directory."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (OSError, AttributeError):
        renameat2 = None
    if renameat2 is not None:
        at_fdcwd, no_replace = -100, 1
        if renameat2(at_fdcwd, os.fsencode(str(source)), at_fdcwd, os.fsencode(str(destination)),
                     no_replace) == 0:
            return
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(f"output already exists: {destination}")
        if code not in (errno.ENOSYS, errno.EINVAL):
            raise OSError(code, os.strerror(code), str(destination))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    os.rename(source, destination)


def publish(output, report, arrays):
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(dir=output.parent, prefix=".pending-" + output.name + "-"))
    try:
        np.savez(pending / "predictions.npz", **arrays)
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        (pending / "report.json").write_text(text, encoding="utf-8")
        mask = os.umask(0o022)
        os.umask(mask)
        os.chmod(pending, 0o777 & ~mask)
        rename_noreplace(pending, output)
    except BaseException:
        for child in pending.iterdir():
            child.unlink()
        pending.rmdir()
        raise


def parse_weights(text):
    try:
        values = [float(item) for item in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("--source-weights must be comma separated numbers") from None
    if not values or any(not np.isfinite(v) or v <= 0 for v in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("--source-weights must be distinct positive numbers")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, type=Path, help="v1 prototype directory")
    parser.add_argument("--muse-gguf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new report directory")
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--naive-ridge", type=float, default=.01)
    parser.add_argument("--lexical-rank", type=int, default=1024)
    parser.add_argument("--lag-rank", type=int, default=256)
    parser.add_argument("--lags", type=int, default=3)
    parser.add_argument("--lag-decay", type=float, default=.5)
    parser.add_argument("--source-weights", type=parse_weights, default=[1.0, 2.0, 4.0, 8.0, 16.0])
    args = parser.parse_args(argv)
    if (args.folds < 2 or args.bootstrap < 1 or min(args.seed, args.bootstrap_seed) < 0
            or not np.isfinite(args.naive_ridge) or args.naive_ridge <= 0
            or min(args.lexical_rank, args.lag_rank) < 1 or not 0 <= args.lags <= 8
            or not 0 < args.lag_decay <= 1):
        raise ValueError("invalid folds, bootstrap, seeds, ridge, ranks, lags, or decay")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    started = time.time()
    data = load_v1(args.captures, args.lags)
    train, evaluation = data["train"], data["eval"]
    layer = data["target_metadata"]["layer"]
    token_ids = np.unique(np.concatenate((train["stream_tokens"], evaluation["stream_tokens"])))
    muse = load_muse(args.muse_gguf, token_ids, layer, data["heads"], data["head_dim"])
    transform, metric = output_metric(muse["attn_output"], muse["n_head"], data["heads"], data["head_dim"])
    normalized = rmsnorm(muse["embeddings"], muse["eps"])
    basis = LexicalBasis(token_ids, normalized, train["stream_tokens"])
    basis.check_rank(args.lexical_rank)
    basis.check_rank(args.lag_rank)
    grid = ridge.default_grid()

    def lexical(k, rank):
        return (basis.features(train["history"][:, k], rank), basis.features(evaluation["history"][:, k], rank))

    def position_block():
        a, b = position_features(train["positions"]), position_features(evaluation["positions"])
        mean, spread = a.mean(axis=0), a.std(axis=0)
        spread[spread == 0] = 1
        return (a - mean) / spread, (b - mean) / spread

    current = ("token_0", *lexical(0, args.lexical_rank), 1.0)
    history = [(f"token_minus_{k}", *lexical(k, args.lag_rank), args.lag_decay ** k)
               for k in range(1, args.lags + 1)]
    lexical_blocks = [current, *history, ("position", *position_block(), 1.0)]
    source = ("source_values", train["x"], evaluation["x"])
    models, arrays, runs = {}, {}, {}

    naive_weight, naive_bias = ridge.fit(train["x"], train["y"], args.naive_ridge)
    naive_prediction = evaluation["x"] @ naive_weight + naive_bias
    train_mean = train["y"].mean(axis=0)
    naive_eval_stats = split_stats(evaluation["y"], naive_prediction, train_mean, evaluation["documents"], transform)
    runs["source_cv"] = run_model("source_cv", [(*source, None)], data, args, transform, grid, raw=True)
    source_curve = runs["source_cv"][0]["cv"]["curve"]
    naive_cv = min(source_curve, key=lambda row: abs(np.log(row["lambda"] / args.naive_ridge)))
    models["source_naive"] = {
        "lambda": args.naive_ridge,
        "features": int(train["x"].shape[1]),
        "eval": {**value_metrics(evaluation["y"], naive_prediction, train_mean, data["heads"], data["head_dim"]),
                 "output_space": value_metrics(evaluation["y"] @ transform, naive_prediction @ transform,
                                               train_mean @ transform, data["heads"], data["head_dim"])},
        "train_in_sample": value_metrics(train["y"], train["x"] @ naive_weight + naive_bias, train_mean,
                                         data["heads"], data["head_dim"]),
        "cv_at_nearest_grid_lambda": naive_cv,
        "bootstrap_95": {"eval_documents": bootstrap_model(naive_eval_stats, args)},
    }
    arrays["source_naive_eval_prediction"] = naive_prediction
    runs["lexical_current"] = run_model("lexical_current", [current], data, args, transform, grid)
    runs["lexical"] = run_model("lexical", lexical_blocks, data, args, transform, grid)
    runs["source_plus_current"] = select_source_weight(
        "source_plus_current", lambda w: [current, (*source, w)], args.source_weights, data, args, transform, grid)
    runs["source_plus_lexical"] = select_source_weight(
        "source_plus_lexical", lambda w: [*lexical_blocks, (*source, w)], args.source_weights, data, args,
        transform, grid)
    for name, (result, model_arrays, _, _, _) in runs.items():
        models[name] = result
        arrays.update({f"{name}_{key}": value for key, value in model_arrays.items()})
    increments = {
        "source_beyond_lexical": increment(runs["source_plus_lexical"], runs["lexical"], args),
        "source_beyond_current_token": increment(runs["source_plus_current"], runs["lexical_current"], args),
        "lexical_beyond_source": increment(runs["source_plus_lexical"], runs["source_cv"], args),
    }
    arrays.update({"eval_documents": evaluation["documents"], "train_documents": train["documents"],
                   "eval_target": evaluation["y"], "train_target": train["y"]})
    v1_fit = args.captures / "fit" / "report.json"
    v1_reference = None
    if v1_fit.is_file():
        reported = json.loads(v1_fit.read_text(encoding="utf-8"))["held_out_values"]
        v1_reference = {"v1_fit_report_held_out_values": reported,
                        "naive_mse_abs_difference": abs(reported["mse"] - models["source_naive"]["eval"]["mse"])}
    naive_eval, cv_eval = models["source_naive"]["eval"], models["source_cv"]["eval"]
    report_counts = {split: int(len(np.unique(data[split]["documents"]))) for split in ("train", "eval")}
    report = {
        "format_version": 1,
        "purpose": "v1_value_probe_reanalysis",
        "inputs": {
            "captures": {f"{name}.npz": sha256_file(args.captures / f"{name}.npz") for name in CAPTURES},
            "source": {k: data["source_metadata"][k] for k in ("model_id", "model_revision", "layer")},
            "target": {k: data["target_metadata"][k] for k in ("model_id", "model_revision", "layer")},
            "muse_gguf": muse["identity"],
            "muse_gguf_sha256_verified": False,
            "code_sha256": {name: sha256_file(HERE / name) for name in CODE_FILES},
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS")},
        "settings": {"folds": args.folds, "fold_seed": args.seed, "group": "document_sha256",
                     "lambda_grid": {"kind": "logspace", "start_exponent": -2, "stop_exponent": 7,
                                     "points_per_decade": 10},
                     "lambda_selection": "minimum pooled held-out MSE",
                     "bootstrap": {"resamples": args.bootstrap, "seed": args.bootstrap_seed, "level": .95,
                                   "unit": "document"},
                     "naive_ridge": args.naive_ridge, "lexical_rank": args.lexical_rank,
                     "lag_rank": args.lag_rank, "lags": args.lags, "lag_decay": args.lag_decay,
                     "source_weights": args.source_weights},
        "data": {split: {"aligned_rows": int(len(data[split]["y"])),
                         "documents": report_counts[split],
                         "target_rows": data[split]["target_rows"], "source_rows": data[split]["source_rows"]}
                 for split in ("train", "eval")},
        "receiver_features": {
            "embedding": "RMSNorm(token_embd) without weight, eps from GGUF (as embd_norm in the graph)",
            "basis": "occurrence-weighted PCA over the training target token stream (label free)",
            "basis_rank": basis.rank,
            "variance_fraction": {"token_0": basis.variance_fraction(args.lexical_rank),
                                  "history": basis.variance_fraction(args.lag_rank)},
            "history": "previous target tokens (including unaligned ones); absent before position 0 = mean",
            "position": ["pos == 0", "pos == 1", "pos == 2", "log1p(pos)"],
            "block_scaling": "unit mean squared centered row norm on training rows, times sqrt(weight)",
        },
        "output_space": metric,
        "headline": {
            "naive_eval": {"lambda": args.naive_ridge, "mse": naive_eval["mse"],
                           "mse_over_baseline": naive_eval["mse_over_baseline"], "r2": naive_eval["r2_vs_own_mean"]},
            "cv_ridge_eval": {"lambda": models["source_cv"]["cv"]["chosen_lambda"], "mse": cv_eval["mse"],
                              "mse_over_baseline": cv_eval["mse_over_baseline"], "r2": cv_eval["r2_vs_own_mean"],
                              "output_space_r2": cv_eval["output_space"]["r2_vs_own_mean"]},
            "receiver_lexical": {"cv_mean_fold_r2": models["lexical"]["cv"]["mean_fold_r2"],
                                 "eval_r2": models["lexical"]["eval"]["r2_vs_own_mean"]},
            "source_beyond_lexical": {k: increments["source_beyond_lexical"][k] for k in (
                "cv_mean_fold_delta_r2", "eval_delta_r2")},
        },
        "models": models,
        "increments": increments,
        "invariance_output_transform": invariance_check(
            train["x"], train["y"], transform, [args.naive_ridge, models["source_cv"]["cv"]["chosen_lambda"]]),
        "v1_reference": v1_reference,
        "limitations": [
            f"{report_counts['eval']} evaluation documents: eval confidence intervals are coarse; "
            f"the CV bootstrap uses {report_counts['train']} train documents.",
            "The lexical PCA basis and block scales use all training rows (label free) inside CV.",
            "Output-space R² ignores the input-dependent sigmoid gate, attention mixing, and post-attention RMSNorm.",
            "Muse GGUF identity is recorded by size/mtime/metadata and hashes of the bytes read, not a full-file hash.",
            "Lexical ranks, lag decay and position features follow the earlier audit's train-CV configuration.",
        ],
        "seconds": time.time() - started,
    }
    publish(args.output, report, arrays)
    print(json.dumps({"headline": report["headline"], "output": str(args.output)}, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
