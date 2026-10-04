"""Fit top-k source-layer ridge maps to every target layer, separately for K and V.

Inputs are a source CaptureSet (Qwen: k and v on its attention layers), a target
CaptureSet (Muse: k and v on all layers) and an align.py alignment whose documents
carry train/dev/test splits. Only aligned rows are used. A set may combine captures
that differ in keys other than k and v (target-train {k, v} with target-eval
{k, v, q, gate, attn_out}); every capture must hold k and v on the same layers and
head layouts. Per kind:

1. Stream the train rows in chunks into float64 centered sufficient statistics
   (Gram G over all source blocks, cross-product C with all target blocks, means,
   centered target sums of squares). Rows are shifted by the first chunk's mean
   before accumulation to avoid cancellation. G and C are spilled to a work
   directory and memory-mapped afterwards.
2. Screening: for each source layer, eigh of its diagonal block of G gives every
   ridge solution on the alpha grid; dev R^2 for every target layer. Lambda is
   alpha * mean(diag(G_block)), so it is in summed-squared-error units.
3. Ranking per target layer: mean over kinds of the best-alpha single-layer dev R^2
   (one shared order for K and V).
4. Top-k: concatenate the k best-ranked source blocks, solve with an in-place blocked
   Cholesky for every alpha (factorizations are shared by targets with the same
   layer set), score dev R^2, keep the best (k, alpha) per (target, kind). For K,
   dev metrics are also computed after per-head RMS renormalization (Muse k_norm
   has unit weight, so native K rows have per-head RMS 1); K selection uses the
   better of raw and renormalized dev R^2 and `k_renorm` records which one won, so
   the selected map is judged by the output consumers receive.
5. Test metrics for the selected maps only, once.

R^2 is variance weighted about the split's own mean; `mse_over_train_mean` uses the
train target mean as baseline. Dev/test predictions go through `apply`, in float32,
exactly as a consumer of the mapper would compute them.

Output directory (never overwritten, published atomically): `manifest.json`,
`report.json` (screening, ranking and the full top-k dev tables),
`W-<kind>-<t>.npy` float32 [k * source_features, target_features] with rows in
`source_layers` (rank) order, `b-<kind>-<t>.npy` float32 [target_features] and
`mean-<kind>-<t>.npy` float32 [target_features] (train target mean, for baselines).

Scale. Measured with `--threads 8` (OpenBLAS; ~0.7 TFLOP/s f64 and ~1.5 TFLOP/s f32
GEMM on the development machine) on synthetic captures of production width (16 source
layers x [4, 256], 8 target layers x [2, 128]) with 8.4K/1.8K/1.8K train/dev/test rows:
4 min 51 s wall; per kind accumulate 13 s, screen 5-6 s, top-k 126 s (k=8: 5.9 s per
target for 7 alphas; k=16: 62 s shared by all targets); peak anonymous memory 3.0 GB,
peak RSS 7.8 GB (the rest is clean file-backed pages of the spilled G/C and captures).
Extrapolated to production (52 target layers; corpus v2 has 106K/23.4K/24.7K Muse
train/dev/test tokens, so at most that many aligned rows): accumulate ~2.5 min per
kind (~80 TFLOP f64 plus reading ~13 GB of captures), screen ~3 min per kind
(~70 TFLOP f32), top-k ~14 min per kind (k=8 per-target Cholesky ~5 min, shared k=16
set ~3 min, dev predictions ~140 TFLOP f32 ~3-4 min), test ~1 min: ~40 min in total.
Dev (or test) rows of one kind take 116 KiB per row, ~2.7 GB at 23.4K rows. Peak
anonymous memory ~7 GB (k=16 system 2.1 GB, dev rows 2.7 GB, kept best weights
<= 0.9 GB, solve chunks 0.5 GB; accumulation holds G 2.1 GB + C 1.7 GB), plus up to
~8 GB of reclaimable memory-mapped G/C pages (~11 GB resident in practice). The work
directory next to the output holds 7.8 GB of spilled G/C. It and the `.pending-*`
directory are removed on success, errors, SIGINT, SIGTERM and SIGHUP; only after
SIGKILL or a crash can `.kvpark-fit-work-*` / `.pending-<output>-*` remain in the
output parent, and they are then safe to delete.
"""

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys
import tempfile
import threading
import time

THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _early_threads(argv):
    for index, item in enumerate(argv):
        if item == "--threads" and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith("--threads="):
            return item.split("=", 1)[1]
    return None


if __name__ == "__main__":
    _requested = _early_threads(sys.argv[1:])
    if _requested is not None and _requested.isdigit() and int(_requested) > 0:
        for _variable in THREAD_VARIABLES:
            os.environ[_variable] = _requested
for _variable in THREAD_VARIABLES:
    os.environ.setdefault(_variable, "8")

import numpy as np

from align import ALIGNED, load_alignment
from captures import Capture, CaptureSet
import ridge


HERE = Path(__file__).resolve().parent
CODE_FILES = ("fit_layers.py", "captures.py", "align.py", "ridge.py")
KINDS = ("k", "v")
SPLITS = ("train", "dev", "test")
DEFAULT_ALPHAS = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
DEFAULT_KS = (1, 2, 4, 8, 16)
CHOLESKY_BLOCK = 256
RHS_COLUMNS = 2048
RMS_FLOOR = 1e-12
# Row padding for large float64 squares: power-of-two strides make transposed block
# copies and BLAS packing alias in cache (a 1024 x 1024 mirror went from 170 ms to 1 ms).
ROW_PADDING = 16


def log(**fields):
    print(json.dumps({"time": round(time.time(), 3), **fields}), file=sys.stderr, flush=True)


def set_blas_threads(threads):
    """Best effort at runtime (env vars only act before numpy loads); returns the effective count."""
    libraries = set()
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                path = line.split()[-1] if line.strip() else ""
                if "openblas" in path.lower() and path.startswith("/"):
                    libraries.add(path)
    except OSError:
        pass
    effective = None
    for path in sorted(libraries):
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue
        for setter, getter in (("openblas_set_num_threads", "openblas_get_num_threads"),
                               ("scipy_openblas_set_num_threads64_", "scipy_openblas_get_num_threads64_"),
                               ("openblas_set_num_threads64_", "openblas_get_num_threads64_")):
            if hasattr(library, setter):
                getattr(library, setter)(ctypes.c_int(threads))
                if hasattr(library, getter):
                    effective = int(getattr(library, getter)())
    return {"requested": threads, "openblas_libraries": sorted(Path(p).name for p in libraries),
            "openblas_effective": effective,
            "environment": {name: os.environ.get(name) for name in THREAD_VARIABLES}}


# ---------------------------------------------------------------- linear algebra

def cholesky_in_place(matrix, block=CHOLESKY_BLOCK):
    """Blocked left-looking Cholesky of the lower triangle; returns inverse diagonal blocks.

    Only the lower triangle of `matrix` is read and overwritten with L; the strict
    upper triangle is left unchanged. Temporaries are at most n x block.
    """
    n = len(matrix)
    inverses = []
    for start in range(0, n, block):
        stop = min(start + block, n)
        if start:
            matrix[start:, start:stop] -= matrix[start:, :start] @ matrix[start:stop, :start].T
        try:
            diagonal = np.linalg.cholesky(matrix[start:stop, start:stop])
        except np.linalg.LinAlgError as exc:
            raise ValueError("ridge system is not positive definite; inspect input scale") from exc
        matrix[start:stop, start:stop] = np.tril(diagonal) + np.triu(matrix[start:stop, start:stop], 1)
        inverse = np.linalg.inv(diagonal)
        inverses.append(inverse)
        if stop < n:
            matrix[stop:, start:stop] = matrix[stop:, start:stop] @ inverse.T
    return inverses


def cholesky_solve_in_place(factor, inverses, rhs, block=CHOLESKY_BLOCK):
    """Solve L L' W = rhs in place, with L from `cholesky_in_place`."""
    n = len(factor)
    starts = list(range(0, n, block))
    for index, start in enumerate(starts):
        stop = min(start + block, n)
        if start:
            rhs[start:stop] -= factor[start:stop, :start] @ rhs[:start]
        rhs[start:stop] = inverses[index] @ rhs[start:stop]
    for index in reversed(range(len(starts))):
        start = starts[index]
        stop = min(start + block, n)
        if stop < n:
            rhs[start:stop] -= factor[stop:, start:stop].T @ rhs[stop:]
        rhs[start:stop] = inverses[index].T @ rhs[start:stop]
    if not np.isfinite(rhs).all():
        raise ValueError("ridge produced nonfinite weights; inspect input scale")
    return rhs


def padded_matrix(rows, columns, fill=None):
    """float64 [rows, columns] view whose row stride is padded by ROW_PADDING elements."""
    storage = np.empty((rows, columns + ROW_PADDING))
    view = storage[:, :columns]
    if fill is not None:
        view[...] = fill
    return view


def renormalize_heads(values, heads, dim):
    """Scale every head vector to RMS 1 (native Muse K rows after k_norm)."""
    shaped = np.asarray(values).reshape(len(values), heads, dim)
    rms = np.sqrt(np.mean(np.square(shaped, dtype=np.float64), axis=-1, keepdims=True))
    return (shaped / np.maximum(rms, RMS_FLOOR)).astype(np.float32).reshape(len(values), heads * dim)


# ---------------------------------------------------------------- data

class Problem:
    """Validated layer layout and aligned rows per split."""

    def __init__(self, src_set, tgt_set, alignment):
        self.src_set, self.tgt_set, self.alignment = src_set, tgt_set, alignment
        self.src_layers = self._layers(src_set, "source")
        self.tgt_layers = self._layers(tgt_set, "target")
        self.src_shape = {kind: self._shape(src_set, kind, self.src_layers, "source") for kind in KINDS}
        self.tgt_shape = {kind: self._shape(tgt_set, kind, self.tgt_layers, "target") for kind in KINDS}
        self._check_alignment()
        self.rows = {}
        for split in SPLITS:
            mask = alignment.pair_mask(split)
            documents = sorted({alignment.document_ids[i] for i in np.unique(alignment.doc[mask])})
            if mask.sum() < 2 or not documents:
                raise ValueError(f"split {split} needs aligned rows from at least one document")
            self.rows[split] = {"src": alignment.src_row[mask], "tgt": alignment.tgt_row[mask],
                                "doc": alignment.doc[mask], "documents": documents}

    @staticmethod
    def _layers(capture_set, name):
        layers = capture_set.captures[0].layers("k")
        for capture in capture_set.captures:
            if not layers or capture.layers("k") != layers or capture.layers("v") != layers:
                raise ValueError(f"every {name} capture needs k and v on the same nonempty layer list")
        return layers

    @staticmethod
    def _shape(capture_set, kind, layers, name):
        shapes = {tuple(capture.array(kind, layer).shape[1:])
                  for capture in capture_set.captures for layer in layers}
        if len(shapes) != 1:
            raise ValueError(f"{name} {kind} captures have different head layouts across layers or captures")
        return shapes.pop()

    def _check_alignment(self):
        alignment, src_set, tgt_set = self.alignment, self.src_set, self.tgt_set
        if any(split not in SPLITS for split in alignment.splits):
            raise ValueError("every aligned document needs split train, dev or test")
        if (len(alignment.src_class) != src_set.rows_total or len(alignment.tgt_class) != tgt_set.rows_total):
            raise ValueError("alignment was built for captures with different row counts")
        count = len(alignment.src_row)
        if (not count or any(len(getattr(alignment, name)) != count for name in ("tgt_row", "doc"))
                or alignment.doc.min() < 0 or alignment.doc.max() >= len(alignment.document_ids)):
            raise ValueError("alignment has no pairs or malformed pair arrays")
        for name, rows, capture_set, classes in (("source", alignment.src_row, src_set, alignment.src_class),
                                                 ("target", alignment.tgt_row, tgt_set, alignment.tgt_class)):
            if rows.min() < 0 or rows.max() >= capture_set.rows_total or len(np.unique(rows)) != count:
                raise ValueError(f"alignment {name} rows are out of range or repeated")
            if np.any(classes[rows] != ALIGNED):
                raise ValueError(f"alignment {name} rows are not all classified as aligned")
            for number, document_id in enumerate(alignment.document_ids):
                if document_id not in capture_set:
                    raise ValueError(f"{name} capture set lacks aligned document {document_id}")
                own = rows[alignment.doc == number]
                window = capture_set.rows(document_id)
                if len(own) and (own.min() < window.start or own.max() >= window.stop):
                    raise ValueError(f"alignment {name} rows leave document {document_id}")

    def features(self, side, kind):
        heads, dim = (self.src_shape if side == "src" else self.tgt_shape)[kind]
        return heads * dim

    def load(self, kind, split):
        """Dev/test rows in RAM (float32): per source layer blocks and per target layer blocks."""
        rows = self.rows[split]
        count = len(rows["src"])
        source = [self.src_set.gather(kind, layer, rows["src"]).reshape(count, -1) for layer in self.src_layers]
        target = [self.tgt_set.gather(kind, layer, rows["tgt"]).reshape(count, -1) for layer in self.tgt_layers]
        for name, blocks in (("source", source), ("target", target)):
            if not all(np.isfinite(block).all() for block in blocks):
                raise ValueError(f"{name} {kind} {split} rows contain nonfinite values")
        for layer, block in zip(self.tgt_layers, target):
            if np.array_equal(block.min(axis=0), block.max(axis=0)):
                raise ValueError(f"target {kind}-{layer} is constant on the {split} rows; R^2 is undefined")
        return source, target


class _KVCapture(Capture):
    """Capture whose set-compatibility layout covers only k and v, so a set may mix
    captures that differ in other keys (e.g. target-train {k, v} and target-eval
    {k, v, q, gate, attn_out})."""

    def layout(self):
        return {pair: shape for pair, shape in super().layout().items() if pair[0] in KINDS}


def open_capture_set(paths):
    """CaptureSet for fitting: the captures must agree on architecture, model and k/v layout only."""
    paths = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    return CaptureSet([_KVCapture(path) for path in paths])


class Statistics:
    """Centered train sufficient statistics of one kind (float64)."""

    def __init__(self, gram, cross, x_mean, y_mean, y_centered_squares, rows):
        self.gram, self.cross = gram, cross
        self.x_mean, self.y_mean = x_mean, y_mean
        self.y_centered_squares, self.rows = y_centered_squares, rows


def accumulate(problem, kind, chunk_rows, block=1024, progress=None):
    """Stream train rows into centered G = Xc'Xc, C = Xc'Yc, means and centered target sums of squares."""
    rows = problem.rows["train"]
    src_width, tgt_width = problem.features("src", kind), problem.features("tgt", kind)
    n_features = src_width * len(problem.src_layers)
    n_outputs = tgt_width * len(problem.tgt_layers)
    total = len(rows["src"])
    gram = padded_matrix(n_features, n_features, 0.0)
    cross = np.zeros((n_features, n_outputs))
    x_sum, y_sum, y_squares = np.zeros(n_features), np.zeros(n_outputs), np.zeros(n_outputs)
    x = np.empty((min(chunk_rows, total), n_features))
    y = np.empty((min(chunk_rows, total), n_outputs))
    x_shift = y_shift = None
    for start in range(0, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        count = stop - start
        xs, ys = x[:count], y[:count]
        for index, layer in enumerate(problem.src_layers):
            xs[:, index * src_width:(index + 1) * src_width] = problem.src_set.gather(
                kind, layer, rows["src"][start:stop]).reshape(count, -1)
        for index, layer in enumerate(problem.tgt_layers):
            ys[:, index * tgt_width:(index + 1) * tgt_width] = problem.tgt_set.gather(
                kind, layer, rows["tgt"][start:stop]).reshape(count, -1)
        if not np.isfinite(xs).all() or not np.isfinite(ys).all():
            raise ValueError(f"{kind} train rows contain nonfinite values")
        if x_shift is None:
            x_shift, y_shift = xs.mean(axis=0), ys.mean(axis=0)
        xs -= x_shift
        ys -= y_shift
        x_sum += xs.sum(axis=0)
        y_sum += ys.sum(axis=0)
        y_squares += np.einsum("ij,ij->j", ys, ys)
        for first in range(0, n_features, block):
            last = min(first + block, n_features)
            gram[first:last, first:] += xs[:, first:last].T @ xs[:, first:]
            cross[first:last] += xs[:, first:last].T @ ys
        if progress:
            progress(kind=kind, phase="accumulate", rows=stop, total=total)
    x_mean_shifted, y_mean_shifted = x_sum / total, y_sum / total
    for first in range(0, n_features, block):
        last = min(first + block, n_features)
        diagonal = gram[first:last, first:last]
        diagonal[...] = (diagonal + diagonal.T) / 2
        gram[last:, first:last] = gram[first:last, last:].T
        gram[first:last] -= total * np.outer(x_mean_shifted[first:last], x_mean_shifted)
        cross[first:last] -= total * np.outer(x_mean_shifted[first:last], y_mean_shifted)
    return Statistics(gram, cross, x_shift + x_mean_shifted, y_shift + y_mean_shifted,
                      y_squares - total * np.square(y_mean_shifted), total)


def spill(statistics, directory, kind):
    """Move G and C to disk and memory-map them read-only."""
    for name in ("gram", "cross"):
        path = Path(directory) / f"{name}-{kind}.npy"
        with path.open("xb") as stream:
            np.lib.format.write_array(stream, getattr(statistics, name), version=(1, 0))
        setattr(statistics, name, None)
        setattr(statistics, name, np.load(path, mmap_mode="r"))
    return statistics


# ---------------------------------------------------------------- metrics

def split_reference(target_blocks):
    """Per target block: own mean sum of squares (for R^2)."""
    return [float(np.sum(np.square(block - block.mean(axis=0, dtype=np.float64), dtype=np.float64)))
            for block in target_blocks]


def metrics(actual, predicted, train_mean, own_total=None):
    residual = np.asarray(actual, dtype=np.float64) - predicted
    sse = float(np.sum(np.square(residual)))
    baseline = float(np.sum(np.square(np.asarray(actual, dtype=np.float64) - train_mean)))
    if own_total is None:
        own_total = float(np.sum(np.square(actual - actual.mean(axis=0, dtype=np.float64))))
    if own_total <= 0 or baseline <= 0:
        raise ValueError("evaluation target has zero variance; R^2 is undefined")
    return {"r2": 1 - sse / own_total, "mse": sse / residual.size,
            "mse_over_train_mean": sse / baseline, "rows": int(len(actual))}


# ---------------------------------------------------------------- mapper

class LayerMap:
    """One (kind, target layer) map: y = concat(source blocks in source_layers order) @ W + b."""

    def __init__(self, kind, target, source_layers, weight, bias, k_renorm, source_shape, target_shape,
                 train_mean=None):
        self.kind, self.target = kind, int(target)
        self.source_layers = [int(layer) for layer in source_layers]
        self.weight = np.ascontiguousarray(weight, dtype=np.float32)
        self.bias = np.ascontiguousarray(bias, dtype=np.float32)
        self.k_renorm = bool(k_renorm)
        self.source_shape, self.target_shape = tuple(source_shape), tuple(target_shape)
        self.train_mean = None if train_mean is None else np.asarray(train_mean, dtype=np.float32)
        width = int(np.prod(self.source_shape))
        outputs = int(np.prod(self.target_shape))
        if (not self.source_layers or len(set(self.source_layers)) != len(self.source_layers)
                or self.weight.shape != (width * len(self.source_layers), outputs)
                or self.bias.shape != (outputs,) or not np.isfinite(self.weight).all()
                or not np.isfinite(self.bias).all() or (self.k_renorm and kind != "k")):
            raise ValueError(f"invalid map {kind}-{target}")

    def blocks(self):
        width = int(np.prod(self.source_shape))
        return [self.weight[i * width:(i + 1) * width] for i in range(len(self.source_layers))]


def _predict(layer_map, blocks):
    """float32 prediction from the source blocks listed in layer_map.source_layers order."""
    result = None
    for weight, rows in zip(layer_map.blocks(), blocks):
        part = rows @ weight
        result = part if result is None else np.add(result, part, out=result)
    result += layer_map.bias
    return result


def apply(mapper, source_rows_by_layer):
    """Predict target rows [n, heads, dim] from {source layer: [n, heads, dim] or [n, features]}."""
    width = int(np.prod(mapper.source_shape))
    blocks = []
    for layer in mapper.source_layers:
        if layer not in source_rows_by_layer:
            raise KeyError(f"map {mapper.kind}-{mapper.target} needs source layer {layer}")
        rows = np.asarray(source_rows_by_layer[layer], dtype=np.float32)
        if (rows.ndim < 2 or int(np.prod(rows.shape[1:])) != width
                or (blocks and len(rows) != len(blocks[0]))):
            raise ValueError(f"source layer {layer} rows have the wrong shape")
        rows = rows.reshape(len(rows), width)
        blocks.append(rows)
    predicted = _predict(mapper, blocks)
    if mapper.k_renorm:
        predicted = renormalize_heads(predicted, *mapper.target_shape)
    if not np.isfinite(predicted).all():
        raise ValueError("mapping produced nonfinite values")
    return predicted.reshape(len(predicted), *mapper.target_shape)


class Mapper:
    """All maps of a fit_layers output directory, verified against the manifest hashes."""

    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        if (self.manifest.get("format_version") != 1
                or self.manifest.get("purpose") != "kv_linear_transfer_mapper"):
            raise ValueError("unsupported mapper manifest")
        self.maps = {}
        for entry in self.manifest["maps"]:
            arrays = {}
            for field in ("weight", "bias", "mean"):
                file_path = self.path / entry["files"][field]
                if hashlib.sha256(file_path.read_bytes()).hexdigest() != entry["sha256"][field]:
                    raise ValueError(f"mapper file {file_path.name} does not match its manifest hash")
                arrays[field] = np.load(file_path, allow_pickle=False)
            kind = entry["kind"]
            self.maps[kind, entry["target"]] = LayerMap(
                kind, entry["target"], entry["source_layers"], arrays["weight"], arrays["bias"],
                entry["k_renorm"], self.manifest["source"]["head_shape"][kind],
                self.manifest["target"]["head_shape"][kind], arrays["mean"])

    def get(self, kind, target):
        return self.maps[kind, target]

    def apply(self, kind, target, source_rows_by_layer):
        return apply(self.get(kind, target), source_rows_by_layer)


def load_mapper(path):
    return Mapper(path)


# ---------------------------------------------------------------- fitting

def screen(problem, statistics, kind, dev, alphas, progress=None):
    """Single-layer dev R^2 [source, target, alpha] from one eigh per source block."""
    source_blocks, target_blocks = dev
    own_totals = split_reference(target_blocks)
    width = problem.features("src", kind)
    tgt_width = problem.features("tgt", kind)
    n_targets = len(problem.tgt_layers)
    chunk = max(1, RHS_COLUMNS // tgt_width)
    r2 = np.empty((len(problem.src_layers), n_targets, len(alphas)))
    lambdas = np.empty((len(problem.src_layers), len(alphas)))
    for index in range(len(problem.src_layers)):
        block = slice(index * width, (index + 1) * width)
        gram = np.asarray(statistics.gram[block, block])
        path = ridge.RidgePath(gram, np.asarray(statistics.cross[block]))
        scale = float(np.mean(np.diag(gram)))
        if not scale > 0:
            raise ValueError(f"source layer {problem.src_layers[index]} {kind} has no train variance")
        projected = path.project(source_blocks[index] - statistics.x_mean[block]).astype(np.float32)
        for a, alpha in enumerate(alphas):
            lam = alpha * scale
            lambdas[index, a] = lam
            weight = path.rotated_weight(lam).astype(np.float32)
            for first in range(0, n_targets, chunk):
                last = min(first + chunk, n_targets)
                columns = slice(first * tgt_width, last * tgt_width)
                predicted = projected @ weight[:, columns]
                predicted += statistics.y_mean[columns].astype(np.float32)
                for t in range(first, last):
                    local = slice((t - first) * tgt_width, (t - first + 1) * tgt_width)
                    sse = float(np.sum(np.square(target_blocks[t] - predicted[:, local], dtype=np.float64)))
                    r2[index, t, a] = 1 - sse / own_totals[t]
        if progress:
            progress(kind=kind, phase="screen", source_layer=problem.src_layers[index])
    return r2, lambdas


def rank_layers(problem, screening):
    """Per target layer: source layers by mean over kinds of best-alpha dev R^2 (ties: lower layer)."""
    score = np.mean([screening[kind][0].max(axis=2) for kind in KINDS], axis=0)
    ranking = {}
    for t, target in enumerate(problem.tgt_layers):
        order = sorted(range(len(problem.src_layers)), key=lambda s: (-score[s, t], problem.src_layers[s]))
        ranking[target] = {"order": [problem.src_layers[s] for s in order],
                           "score": [float(score[s, t]) for s in order]}
    return ranking


def top_k(problem, statistics, kind, dev, ranking, ks, alphas, progress=None):
    """Dev tables for every (target, k, alpha) and the best map per target."""
    source_blocks, target_blocks = dev
    own_totals = split_reference(target_blocks)
    width, tgt_width = problem.features("src", kind), problem.features("tgt", kind)
    position = {layer: index for index, layer in enumerate(problem.src_layers)}
    heads, dim = problem.tgt_shape[kind]
    chunk = max(1, RHS_COLUMNS // tgt_width)
    table, best = [], {}
    for k in ks:
        groups = {}
        for t, target in enumerate(problem.tgt_layers):
            chosen = ranking[target]["order"][:k]
            groups.setdefault(tuple(sorted(chosen, key=position.get)), []).append(t)
        for layers, members in groups.items():
            features = np.concatenate([np.arange(position[layer] * width, (position[layer] + 1) * width)
                                       for layer in layers])
            size = len(features)
            system = padded_matrix(size, size)
            diagonal_index = np.arange(size)
            diagonal = np.asarray(statistics.gram[features, features])
            scale = float(np.mean(diagonal))
            if not scale > 0:
                raise ValueError(f"source {kind} layers {list(layers)} have no train variance")
            for alpha in alphas:
                lam = alpha * scale
                for i, row_layer in enumerate(layers):
                    rows = slice(position[row_layer] * width, (position[row_layer] + 1) * width)
                    for j, column_layer in enumerate(layers[:i + 1]):
                        columns = slice(position[column_layer] * width, (position[column_layer] + 1) * width)
                        system[i * width:(i + 1) * width, j * width:(j + 1) * width] = statistics.gram[rows, columns]
                system[diagonal_index, diagonal_index] += lam
                inverses = cholesky_in_place(system)
                for first in range(0, len(members), chunk):
                    batch = members[first:first + chunk]
                    columns = np.concatenate([np.arange(t * tgt_width, (t + 1) * tgt_width) for t in batch])
                    weights = np.empty((size, len(columns)))
                    for i, layer in enumerate(layers):
                        weights[i * width:(i + 1) * width] = statistics.cross[
                            position[layer] * width:(position[layer] + 1) * width][:, columns]
                    cross = weights.copy()
                    cholesky_solve_in_place(system, inverses, weights)
                    for b, t in enumerate(batch):
                        local = slice(b * tgt_width, (b + 1) * tgt_width)
                        entry = _evaluate(problem, statistics, kind, t, layers, features, weights[:, local],
                                          cross[:, local], lam, source_blocks, target_blocks[t], own_totals[t],
                                          ranking, heads, dim)
                        entry.update({"k": k, "alpha": alpha, "lambda": lam})
                        layer_map = entry.pop("map")
                        table.append(entry)
                        current = best.get(t)
                        if current is None or selection_r2(entry) > selection_r2(current[0]):
                            best[t] = (entry, layer_map)
            if progress:
                progress(kind=kind, phase="top_k", k=k, source_layers=list(layers), targets=len(members))
    return table, best


def selection_r2(entry):
    """Dev R^2 of the output a consumer would get: K may be renormalized per head."""
    if entry["kind"] == "k":
        return max(entry["dev"]["r2"], entry["dev_renorm"]["r2"])
    return entry["dev"]["r2"]


def _evaluate(problem, statistics, kind, t, layers, features, weight, cross, lam, source_blocks, actual,
              own_total, ranking, heads, dim):
    target = problem.tgt_layers[t]
    width, tgt_width = problem.features("src", kind), problem.features("tgt", kind)
    columns = slice(t * tgt_width, (t + 1) * tgt_width)
    y_mean = statistics.y_mean[columns]
    bias = y_mean - statistics.x_mean[features] @ weight
    centered = statistics.y_centered_squares[columns].sum()
    train_sse = centered - np.sum(weight * cross) - lam * np.sum(np.square(weight))
    order = ranking[target]["order"][:len(layers)]
    sorted_index = {layer: i for i, layer in enumerate(layers)}
    ranked = np.concatenate([weight[sorted_index[layer] * width:(sorted_index[layer] + 1) * width]
                             for layer in order])
    layer_map = LayerMap(kind, target, order, ranked, bias, False, problem.src_shape[kind],
                         problem.tgt_shape[kind], y_mean)
    position = {layer: index for index, layer in enumerate(problem.src_layers)}
    predicted = _predict(layer_map, [source_blocks[position[layer]] for layer in order])
    entry = {"kind": kind, "target": target, "source_layers": order,
             "train_r2": float(1 - train_sse / centered) if centered > 0 else None,
             "dev": metrics(actual, predicted, y_mean, own_total)}
    if kind == "k":
        entry["dev_renorm"] = metrics(actual, renormalize_heads(predicted, heads, dim), y_mean, own_total)
    entry["map"] = layer_map
    return entry


def evaluate_test(problem, kind, best, test):
    source_blocks, target_blocks = test
    position = {layer: index for index, layer in enumerate(problem.src_layers)}
    heads, dim = problem.tgt_shape[kind]
    results = {}
    for t, (entry, layer_map) in best.items():
        rows = {layer: source_blocks[position[layer]] for layer in layer_map.source_layers}
        raw = LayerMap(kind, layer_map.target, layer_map.source_layers, layer_map.weight, layer_map.bias,
                       False, layer_map.source_shape, layer_map.target_shape)
        predicted = apply(raw, rows).reshape(len(target_blocks[t]), -1)
        mean = layer_map.train_mean
        result = {"raw": metrics(target_blocks[t], predicted, mean)}
        if kind == "k":
            result["renorm"] = metrics(target_blocks[t], renormalize_heads(predicted, heads, dim), mean)
        results[t] = result
    return results


# ---------------------------------------------------------------- orchestration

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _remove_tree(path):
    path = Path(path)
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


def _write_npy(path, array):
    with path.open("xb") as stream:
        np.lib.format.write_array(stream, np.ascontiguousarray(array), version=(1, 0))
    return sha256_file(path)


def _capture_identity(capture_set):
    return [{"path": str(capture.path.resolve()),
             "report_sha256": sha256_file(capture.path / "report.json"),
             "model_sha256": (capture.manifest or {}).get("model_sha256"),
             "documents": len(capture.documents), "rows": capture.rows_total}
            for capture in capture_set.captures]


def fit(problem, *, alphas=DEFAULT_ALPHAS, ks=DEFAULT_KS, chunk_rows=2048, work_dir, pending,
        progress=None):
    """Run screening, ranking and top-k; write arrays into `pending`; return (maps, report)."""
    alphas, ks = sorted(float(a) for a in alphas), sorted(int(k) for k in ks)
    if (not alphas or not ks or len(set(alphas)) != len(alphas) or len(set(ks)) != len(ks)
            or not all(np.isfinite(a) and a > 0 for a in alphas) or ks[0] < 1):
        raise ValueError("alphas must be distinct positive values and ks distinct positive integers")
    if ks[-1] > len(problem.src_layers):
        raise ValueError(f"k={ks[-1]} exceeds the {len(problem.src_layers)} captured source layers")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    timing, screening, stats = {}, {}, {}
    for kind in KINDS:
        started = time.perf_counter()
        stats[kind] = spill(accumulate(problem, kind, chunk_rows, progress=progress), work_dir, kind)
        timing[f"accumulate_{kind}"] = time.perf_counter() - started
        started = time.perf_counter()
        screening[kind] = screen(problem, stats[kind], kind, problem.load(kind, "dev"), alphas, progress)
        timing[f"screen_{kind}"] = time.perf_counter() - started
    ranking = rank_layers(problem, screening)
    maps, tables = [], []
    for kind in KINDS:
        started = time.perf_counter()
        table, best = top_k(problem, stats[kind], kind, problem.load(kind, "dev"), ranking, ks, alphas,
                            progress)
        timing[f"top_k_{kind}"] = time.perf_counter() - started
        tables.extend(table)
        started = time.perf_counter()
        tested = evaluate_test(problem, kind, best, problem.load(kind, "test"))
        timing[f"test_{kind}"] = time.perf_counter() - started
        for t in sorted(best):
            entry, layer_map = best[t]
            renorm = kind == "k" and entry["dev_renorm"]["r2"] > entry["dev"]["r2"]
            target = layer_map.target
            files = {"weight": f"W-{kind}-{target}.npy", "bias": f"b-{kind}-{target}.npy",
                     "mean": f"mean-{kind}-{target}.npy"}
            hashes = {"weight": _write_npy(pending / files["weight"], layer_map.weight),
                      "bias": _write_npy(pending / files["bias"], layer_map.bias),
                      "mean": _write_npy(pending / files["mean"], layer_map.train_mean)}
            record = {"kind": kind, "target": target, "source_layers": layer_map.source_layers,
                      "k": entry["k"], "alpha": entry["alpha"], "lambda": entry["lambda"],
                      "k_renorm": bool(renorm), "train_r2": entry["train_r2"],
                      "dev": {"raw": entry["dev"]}, "test": tested[t], "files": files, "sha256": hashes}
            if kind == "k":
                record["dev"]["renorm"] = entry["dev_renorm"]
            applied = "renorm" if renorm else "raw"
            record["dev"]["applied"] = applied
            record["test"]["applied"] = applied
            maps.append(record)
        stats[kind] = None
    screening_rows = []
    for kind in KINDS:
        r2, lambdas = screening[kind]
        for t, target in enumerate(problem.tgt_layers):
            for s, source in enumerate(problem.src_layers):
                best_alpha = int(np.argmax(r2[s, t]))
                screening_rows.append({"kind": kind, "target": target, "source": source,
                                       "dev_r2": r2[s, t].tolist(), "lambda": lambdas[s].tolist(),
                                       "best_alpha": alphas[best_alpha], "best_dev_r2": float(r2[s, t, best_alpha])})
    report = {"screening": screening_rows,
              "ranking": [{"target": target, **ranking[target]} for target in problem.tgt_layers],
              "top_k": tables, "timing_seconds": timing}
    return maps, report


def _parse_list(text, kind, name):
    try:
        values = [kind(item) for item in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{name} must be a comma separated list") from None
    if not values or len(set(values)) != len(values) or any(not np.isfinite(v) or v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"{name} must hold distinct positive values")
    return sorted(values)


class _TerminationSignals:
    """Turn SIGTERM/SIGHUP into SystemExit so cleanup in finally blocks runs; restores handlers."""

    SIGNALS = tuple(getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name))

    def __enter__(self):
        self.previous = {}
        if threading.current_thread() is threading.main_thread():
            for number in self.SIGNALS:
                self.previous[number] = signal.signal(number, self._raise)
        return self

    @staticmethod
    def _raise(number, frame):
        raise SystemExit(128 + number)

    def __exit__(self, *exc):
        for number, handler in self.previous.items():
            signal.signal(number, handler)
        return False


def main(argv=None):
    with _TerminationSignals():
        return _main(argv)


def _main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src-capture", required=True, type=Path, action="append",
                        help="source (Qwen) capture directory; repeat for several")
    parser.add_argument("--tgt-capture", required=True, type=Path, action="append",
                        help="target (Muse) capture directory; repeat for several")
    parser.add_argument("--alignment", required=True, type=Path, help="align.py NPZ")
    parser.add_argument("--output", required=True, type=Path, help="new mapper directory; never overwritten")
    parser.add_argument("--alphas", type=lambda t: _parse_list(t, float, "--alphas"),
                        default=list(DEFAULT_ALPHAS), help="ridge alpha grid relative to mean Gram diagonal")
    parser.add_argument("--ks", type=lambda t: _parse_list(t, int, "--ks"), default=list(DEFAULT_KS),
                        help="numbers of top-ranked source layers to try")
    parser.add_argument("--chunk-rows", type=int, default=2048, help="train rows per accumulation chunk")
    parser.add_argument("--threads", type=int, default=8, help="BLAS threads")
    args = parser.parse_args(argv)
    if args.chunk_rows < 1 or not 1 <= args.threads <= (os.cpu_count() or 1):
        raise ValueError("--chunk-rows must be positive and --threads between 1 and the CPU count")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    if not args.output.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {args.output.parent}")
    blas = set_blas_threads(args.threads)
    src_set, tgt_set = open_capture_set(args.src_capture), open_capture_set(args.tgt_capture)
    alignment_sha = sha256_file(args.alignment)
    problem = Problem(src_set, tgt_set, load_alignment(args.alignment))
    if max(args.ks) > len(problem.src_layers):
        raise ValueError(f"--ks exceeds the {len(problem.src_layers)} captured source layers")
    started = time.time()
    work = Path(tempfile.mkdtemp(dir=args.output.parent, prefix=".kvpark-fit-work-"))
    pending = Path(tempfile.mkdtemp(dir=args.output.parent, prefix=".pending-" + args.output.name + "-"))
    try:
        maps, report = fit(problem, alphas=args.alphas, ks=args.ks, chunk_rows=args.chunk_rows,
                           work_dir=work, pending=pending, progress=log)
        _remove_tree(work)
        settings = {"alphas": args.alphas, "ks": args.ks, "chunk_rows": args.chunk_rows,
                    "threads": args.threads, "cholesky_block": CHOLESKY_BLOCK,
                    "lambda_rule": "alpha * mean(diag(centered train Gram of the concatenated source blocks))",
                    "ranking": "mean over kinds of best-alpha single-layer dev R2; ties to the lower layer",
                    "selection": ("maximum dev R2 over (k, alpha) of the applied prediction (K: better of raw "
                                  "and per-head renormalized); ties to smaller k, then smaller alpha"),
                    "k_renorm": "K is renormalized per head when that beats raw dev R2 at the selection",
                    "r2": "variance weighted, about the evaluated split's own mean",
                    "prediction": "float32, concatenated source blocks in source_layers order @ W + b"}
        rows = {split: {"pairs": int(len(problem.rows[split]["src"])),
                        "documents": len(problem.rows[split]["documents"])} for split in SPLITS}
        manifest = {
            "format_version": 1, "purpose": "kv_linear_transfer_mapper",
            "source": {"architecture": src_set.architecture, "layers": problem.src_layers,
                       "head_shape": {kind: list(problem.src_shape[kind]) for kind in KINDS},
                       "captures": _capture_identity(src_set)},
            "target": {"architecture": tgt_set.architecture, "layers": problem.tgt_layers,
                       "head_shape": {kind: list(problem.tgt_shape[kind]) for kind in KINDS},
                       "captures": _capture_identity(tgt_set)},
            "alignment": {"path": str(args.alignment.resolve()), "sha256": alignment_sha},
            "rows": rows, "split_documents": {split: problem.rows[split]["documents"] for split in SPLITS},
            "settings": settings,
            "code_sha256": {name: sha256_file(HERE / name) for name in CODE_FILES},
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "blas": blas},
            "wall_seconds": time.time() - started,
            "maps": maps,
        }
        report = {"format_version": 1, "purpose": "kv_linear_transfer_layer_selection",
                  "alphas": args.alphas, "ks": args.ks, "rows": rows, **report}
        (pending / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")
        (pending / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
        mask = os.umask(0o022)
        os.umask(mask)
        os.chmod(pending, 0o777 & ~mask)
        rename_noreplace(pending, args.output)
    except BaseException:
        _remove_tree(pending)
        raise
    finally:
        _remove_tree(work)
    summary = [{"kind": m["kind"], "target": m["target"], "k": m["k"], "alpha": m["alpha"],
                "dev_r2": m["dev"][m["dev"]["applied"]]["r2"], "test_r2": m["test"][m["test"]["applied"]]["r2"]}
               for m in maps]
    print(json.dumps({"output": str(args.output), "rows": rows, "maps": summary}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
