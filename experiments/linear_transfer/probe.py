"""Offline value-cache regression probe; does not export or import model state."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

import ridge as ridge_lib


IDENTITY_FIELDS = (
    "format_version", "model_id", "model_revision", "runtime_revision",
    "weight_quantization", "layer", "attention_type", "cache_kind", "text_mode",
)


@dataclass
class Capture:
    values: np.ndarray
    documents: np.ndarray
    byte_end: np.ndarray
    metadata: dict


def load_capture(path):
    """Read one layer's values and exact end offsets in the original UTF-8 text."""
    with np.load(path, allow_pickle=False) as data:
        values = data["values"]
        documents = data["document_sha256"]
        byte_end = data["byte_end"]
        metadata = json.loads(data["metadata_json"].item())
    if (values.ndim != 3 or not all(values.shape)
            or values.dtype.kind != "f" or not np.isfinite(values).all()):
        raise ValueError("values must be finite floating point [rows, heads, head_dim]")
    if (documents.shape != (len(values),) or documents.dtype.kind != "U"
            or any(re.fullmatch(r"[0-9a-f]{64}", doc) is None for doc in documents)):
        raise ValueError("document_sha256 must contain one lowercase SHA-256 per row")
    if (byte_end.shape != (len(values),) or byte_end.dtype.kind not in "iu"
            or np.any(byte_end <= 0)):
        raise ValueError("byte_end must contain one positive integer UTF-8 offset per row")
    if len(set(zip(documents.tolist(), byte_end.tolist()))) != len(values):
        raise ValueError("duplicate document/byte_end rows are not allowed")
    if not isinstance(metadata, dict) or any(k not in metadata for k in IDENTITY_FIELDS):
        raise ValueError("capture metadata is missing required identity fields")
    for key in ("model_id", "model_revision", "runtime_revision", "weight_quantization"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f"metadata {key} must be a nonempty string")
    if (type(metadata["format_version"]) is not int or metadata["format_version"] != 1
            or type(metadata["layer"]) is not int or metadata["layer"] < 0
            or metadata["attention_type"] not in ("full", "sliding")
            or metadata["cache_kind"] != "value"
            or metadata["text_mode"] != "raw_utf8_no_special_tokens"):
        raise ValueError("unsupported capture metadata; expected raw-text value caches")
    return Capture(values, documents, byte_end, metadata)


def aligned_rows(source, target):
    """Indices at exact shared token ends, in document/byte order."""
    source_rows = {
        (doc, int(end)): row
        for row, (doc, end) in enumerate(zip(source.documents, source.byte_end))
    }
    target_rows = {
        (doc, int(end)): row
        for row, (doc, end) in enumerate(zip(target.documents, target.byte_end))
    }
    shared = sorted(source_rows.keys() & target_rows.keys())
    if not shared:
        raise ValueError("captures have no shared document/token-end boundaries")
    return [source_rows[key] for key in shared], [target_rows[key] for key in shared]


def aligned_values(source, target):
    """Keep shared token end boundaries; never use a source state from later text."""
    source_rows, target_rows = aligned_rows(source, target)
    count = len(source_rows)
    x = source.values[source_rows].reshape(count, -1)
    y = target.values[target_rows].reshape(count, -1)
    return x, y, {
        "source_rows": len(source.values),
        "target_rows": len(target.values),
        "aligned_rows": count,
        "aligned_documents": len(set(source.documents[source_rows])),
        "source_coverage": count / len(source.values),
        "target_coverage": count / len(target.values),
    }


def fit_ridge(x, y, ridge):
    """Minimize ||X W + b - Y||² + ridge * ||W||², with an unpenalized bias."""
    return ridge_lib.fit(x, y, ridge)


def code_sha256():
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in ("probe.py", "ridge.py")}


def parse_ridge(text):
    """`auto` (grouped CV over the default grid) or one finite positive float."""
    if text == "auto":
        return text
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("--ridge must be 'auto' or a positive number") from None
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("--ridge must be finite and positive")
    return value


def finite_or_none(value):
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    return float(value) if value is not None and np.isfinite(value) else None


def select_ridge(x, y, documents, ridge, folds, seed):
    """Return (lambda, selection report); auto uses document-grouped K-fold CV."""
    if ridge != "auto":
        return ridge, {"method": "fixed", "chosen": ridge}
    count = len(set(documents.tolist()))
    if count < 2:
        raise ValueError("--ridge auto needs at least two training documents; pass --ridge VALUE")
    used = min(folds, count)
    grid = ridge_lib.default_grid()
    cv = ridge_lib.grouped_cv(x, y, documents, grid, n_folds=used, seed=seed)
    ratios = cv["pooled_mse_over_baseline"] or [None] * len(grid)
    curve = [{"lambda": lam, "pooled_mse": mse, "pooled_mse_over_baseline": finite_or_none(ratio),
              "mean_fold_r2": finite_or_none(r2)}
             for lam, mse, ratio, r2 in zip(cv["lambdas"], cv["pooled_mse"], ratios, cv["mean_fold_r2"])]
    return cv["chosen_lambda"], {
        "method": "grouped_cv",
        "selection": "minimum_pooled_held_out_mse",
        "group": "document_sha256",
        "folds": used,
        "folds_requested": folds,
        "seed": seed,
        "fold_rows": cv["fold_rows"],
        "fold_documents": cv["fold_groups"],
        "grid": {"kind": "logspace", "start_exponent": -2, "stop_exponent": 7,
                 "points_per_decade": 10, "values": cv["lambdas"]},
        "curve": curve,
        "chosen": cv["chosen_lambda"],
        "chosen_fold_r2": finite_or_none(cv["chosen_fold_r2"]),
        "chosen_at_grid_edge": cv["chosen_at_grid_edge"],
    }


def bootstrap_metrics(actual, predicted, train_mean, documents, resamples, seed):
    _, stats = ridge_lib.document_stats(actual, predicted, train_mean, documents)
    result = {}
    for name, statistic in (("mse_over_baseline", ridge_lib.mse_over_baseline_from_stats),
                            ("variance_weighted_r2", ridge_lib.r2_from_stats)):
        with np.errstate(divide="ignore", invalid="ignore"):
            interval = ridge_lib.document_bootstrap(stats, statistic, resamples, seed)
        for key in ("estimate", "low", "high"):
            interval[key] = finite_or_none(interval[key])
        result[name] = interval
    return result


def reconstruction_metrics(actual, predicted, train_mean):
    error = float(np.mean((actual - predicted) ** 2))
    baseline = float(np.mean((actual - train_mean) ** 2))
    variance = float(np.mean((actual - actual.mean(axis=0)) ** 2))
    return {
        "mse": error,
        "train_mean_baseline_mse": baseline,
        "mse_over_baseline": error / baseline if baseline > 0 else None,
        "variance_weighted_r2": 1 - error / variance if variance > 0 else None,
    }


def validate_split(train_source, train_target, eval_source, eval_target):
    train_docs = set(train_source.documents) | set(train_target.documents)
    eval_docs = set(eval_source.documents) | set(eval_target.documents)
    if train_docs & eval_docs:
        raise ValueError("training and evaluation documents overlap; split whole documents")
    for name, train, evaluation in (
        ("source", train_source, eval_source), ("target", train_target, eval_target),
    ):
        if train.metadata != evaluation.metadata:
            raise ValueError(f"{name} train/evaluation metadata identity mismatch")
        if train.values.shape[1:] != evaluation.values.shape[1:]:
            raise ValueError(f"{name} train/evaluation head layout mismatch")
        if train.values.dtype != evaluation.values.dtype:
            raise ValueError(f"{name} train/evaluation capture dtype mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-source", "train-target", "eval-source", "eval-target"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new directory for mapper and report")
    parser.add_argument("--ridge", type=parse_ridge, default="auto",
                        help="positive lambda, or auto: grouped CV over logspace(-2, 7) (default)")
    parser.add_argument("--cv-folds", type=int, default=8,
                        help="document-grouped folds for --ridge auto (capped at the training document count)")
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000, help="document bootstrap resamples")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.cv_folds < 2 or args.bootstrap < 1 or min(args.cv_seed, args.bootstrap_seed) < 0:
        raise ValueError("--cv-folds must be at least 2, --bootstrap positive, and seeds nonnegative")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    train_source, train_target, eval_source, eval_target = [
        load_capture(path) for path in (
            args.train_source, args.train_target, args.eval_source, args.eval_target,
        )
    ]
    validate_split(train_source, train_target, eval_source, eval_target)
    x_train, y_train, train_alignment = aligned_values(train_source, train_target)
    x_eval, y_eval, eval_alignment = aligned_values(eval_source, eval_target)
    memory = {
        "float64_gram_bytes": 8 * x_train.shape[1] ** 2,
        "float64_weight_bytes": 8 * x_train.shape[1] * y_train.shape[1],
        "float64_train_matrices_bytes": 8 * (x_train.size + y_train.size),
    }
    print(json.dumps({"fit_dimensions": [*x_train.shape, y_train.shape[1]],
                      "allocation_estimates_not_peak_memory": memory}), flush=True)
    train_source_rows, _ = aligned_rows(train_source, train_target)
    train_documents = train_source.documents[train_source_rows]
    ridge, ridge_selection = select_ridge(x_train, y_train, train_documents, args.ridge,
                                          args.cv_folds, args.cv_seed)
    weight, bias, solver = ridge_lib.fit_with_solver(x_train, y_train, ridge)
    ridge_selection["final_fit_solver"] = solver
    prediction = np.asarray(x_eval, dtype=np.float64) @ weight + bias
    y_eval = np.asarray(y_eval, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("evaluation produced nonfinite predictions")
    train_mean = y_train.mean(axis=0, dtype=np.float64)
    metrics = reconstruction_metrics(y_eval, prediction, train_mean)
    y_train = np.asarray(y_train, dtype=np.float64)
    train_metrics = reconstruction_metrics(
        y_train, np.asarray(x_train, dtype=np.float64) @ weight + bias, train_mean)
    source_rows, _ = aligned_rows(eval_source, eval_target)
    eval_documents = eval_source.documents[source_rows]
    by_document = {}
    for document in sorted(set(eval_documents)):
        selected = eval_documents == document
        document_metrics = reconstruction_metrics(y_eval[selected], prediction[selected], train_mean)
        ratio = document_metrics["mse_over_baseline"]
        by_document[document] = {
            "aligned_rows": int(selected.sum()),
            **document_metrics,
            "r2_vs_own_mean": document_metrics["variance_weighted_r2"],
            "r2_vs_train_mean": None if ratio is None else 1 - ratio,
        }
    bootstrap = bootstrap_metrics(y_eval, prediction, train_mean, eval_documents,
                                  args.bootstrap, args.bootstrap_seed)
    heads, dimension = eval_target.values.shape[1:]
    by_head = [reconstruction_metrics(
        y_eval[:, head * dimension:(head + 1) * dimension],
        prediction[:, head * dimension:(head + 1) * dimension],
        train_mean[head * dimension:(head + 1) * dimension],
    ) for head in range(heads)]
    artifact_metadata = {
        "format_version": 1,
        "purpose": "value_reconstruction_probe_only",
        "source": train_source.metadata,
        "target": train_target.metadata,
        "source_head_shape": list(train_source.values.shape[1:]),
        "target_head_shape": list(train_target.values.shape[1:]),
        "source_capture_dtype": str(train_source.values.dtype),
        "target_capture_dtype": str(train_target.values.dtype),
        "flatten_order": "C: heads then head dimensions",
        "alignment": "exact document SHA-256 and UTF-8 token end offset",
        "ridge": ridge,
        "ridge_selection_method": ridge_selection["method"],
        "ridge_objective": "summed_squared_error_plus_lambda_weight_squared",
        "train_documents": sorted(set(train_source.documents) | set(train_target.documents)),
        "eval_documents": sorted(set(eval_source.documents) | set(eval_target.documents)),
    }
    report = {
        **artifact_metadata,
        "train_alignment": train_alignment,
        "eval_alignment": eval_alignment,
        "ridge_selection": ridge_selection,
        "train_values": train_metrics,
        "held_out_values": metrics,
        "held_out_bootstrap_95": bootstrap,
        "held_out_by_document": by_document,
        "held_out_by_head": by_head,
        "allocation_estimates_not_peak_memory": memory,
        "code_sha256": code_sha256(),
        "limitations": [
            "Value reconstruction does not establish generation quality or prefill speedup.",
            "Shared token boundaries omit unmatched target tokens; this cannot build a full cache.",
            "Single-layer values only: no keys, RoPE, recurrent state, or runtime import.",
        ],
    }
    report_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output / "mapper.npz", weight=weight, bias=bias,
             metadata_json=np.array(json.dumps(artifact_metadata, allow_nan=False)))
    (args.output / "report.json").write_text(report_text, encoding="utf-8")
    print(json.dumps({"ridge": ridge, "ridge_selection": ridge_selection["method"],
                      "held_out_values": metrics, "output": str(args.output)}, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
