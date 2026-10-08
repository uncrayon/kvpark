"""Where the hub fits' products run: fit_hub.py --fit-device auto|cpu|gpu (hub.fit_backend).

The fits (hub.fit_view, fit_hub_kind, add_model_kind, fit_head, evaluate) stream chunks of capture
rows through three kinds of work:

  projections    float32 rows times a fixed weight: stage-1 bases, stage-2 bases, encoders, decoders
  accumulations  float64 sums of chunk products: a chunk's Gram x'x (hub.Moments) and cross
                 products a'b (the encoders' and decoders' statistics)
  solves         eigh, solve and svd of the accumulated float64 matrices: LAPACK on the CPU, always
                 (top_eig: scipy's subset eigensolver for the top eigenpairs when scipy is importable)

NumpyBackend ("cpu") does them as hub.py always did: float32 products (OpenBLAS, its threads capped by
the resource guard), each chunk's product added to float64 host sums.

TorchBackend ("gpu": torch with ROCm or CUDA) runs projections and accumulations on the GPU, streamed
chunk by chunk, and keeps the float64 sums on the device until a statistic is read (fetch, e.g.
Moments.covariance). Inputs are float16, scaled by powers of two (exact) column by column to RMS 1:
  accumulations  a Triton kernel: float16 operands, float32 accumulation, and the float32 tile
                 unscaled and added to the float64 sum in its epilogue (accumulate_kernel); without
                 Triton, hipBLASLt float16 products (float16 output: 10x the rounding error)
  projections    hipBLASLt float16 products (float32 accumulation, float16 output: they feed float16
                 products), rows scaled to norm ~32 and weight columns to norm ~32 (|entry| <= 2048)
The error sits at the float16 input-rounding floor: a Gram entry within 4e-5 to 7e-5 of its scale
sqrt(G_ii G_jj) against float64 (4e-7 for float32 numpy), a projection within 5e-4 (its float16
output); it averages out over chunks and lies far below the fits' statistical noise. The onboarding of
LFM2-2.6B on the GPU gave held-out R^2 within 8e-6 of the CPU fit's (README.md "Fit placement").

Measured on the Radeon 8060S (2026-10-06, bench in the README): Triton float16 -> float32 A'B on
4096-square tiles 14-16 TFLOPS (7-10 on 8192-square ones); hipBLASLt float16 -> float16 14-29
TFLOPS; torch's float32 products 0.07 TFLOPS and its float16 products with float32 output 3-5 TFLOPS
(not used); OpenBLAS float32 on 8 threads 0.1-0.34 TFLOPS under the thermal guard.

GPU RULE. The desktop draws on the same GPU: a kernel that runs for seconds starves its graphics ring
(on 2026-10-06 at 09:53 float32 torch products of 8-33 s per kernel caused ring timeouts, GNOME Shell
aborted and the driver reset the GPU). Every product here is tiled so that one launch does at most
`budget` flops: the rate calibrate() measures at start (small sizes first) times target_s (0.04 s),
never more than MAX_LAUNCH_FLOPS (0.3 TFLOP: 0.02-0.04 s at the measured rates). GPU fits hold the GPU
lock and run under the resource guard's GPU-hang stop (fit_hub.py).
"""

from collections import OrderedDict
import json
import math
import os
import time

import numpy as np

DEVICES = ("auto", "cpu", "gpu")
ROWS = 8192                     # rows of one float16 accumulation launch
TILE = 4096                     # output tile edge of an accumulation launch (8192-square tiles run at half the rate)
TARGET_S = 0.04                 # one launch's intended duration at the calibrated rate
MAX_LAUNCH_FLOPS = 3e11         # and its cap, whatever the calibration measured
PROJECTION_NORM = 32.0          # rows and weight columns are scaled to about this norm
WEIGHT_CACHE_BYTES = 6 << 30    # float16 weights kept on the device (oldest dropped first)
_CURRENT = []
# methods a profiler times (the scratch profiler of the fit-placement measurements)
PROFILED = tuple(f"{owner}.{method}" for owner in ("NumpyBackend", "TorchBackend")
                 for method in ("mm", "add_gram", "add_cross", "fetch", "put", "col_sum64", "sq_sum64"))


# -- eigendecompositions ----------------------------------------------------------------------------

def _scipy_linalg():
    try:
        import scipy.linalg
    except ImportError:
        return None
    return scipy.linalg


SUBSET_MIN = 4096               # top_eig: from this size the subset solver beats a full eigh (measured)
LARGE_EIGH = os.environ.get("KVPARK_FIT_LARGE_EIGH", "float32")     # top_eig's precision for large subsets


def top_eig(cov, k, *, precision=None):
    """The k largest eigenvalues (descending, clipped at 0) and their eigenvectors of a symmetric
    float64 matrix: scipy.linalg.eigh with subset_by_index (dsyevr: only k eigenvectors are back-
    transformed) when scipy is importable, the matrix has at least SUBSET_MIN rows and k is at most half
    of them; else numpy's eigh. Measured with 8 threads under the guard, for the top quarter: 4.6 s
    against 6.8 s at 4,096 and 40 s against 58 s at 8,192; at 2,048 the full eigh is faster.

    precision (LARGE_EIGH, $KVPARK_FIT_LARGE_EIGH): "float32" (the default) solves such a large subset
    problem (n at least SUBSET_MIN, k at most n / 2) in float32: 22 s against 40 s at 8,192 (without scipy
    numpy's eigh gets the float32 copy, which numpy.linalg computes in float64: slower and 5.5 n x n float64
    of memory instead of 4; fit_hub.eigh_bytes), the eigenvalues within 1.2e-7 and the subspace within a
    cosine of 6e-6 of float64's; in the reduced joint fit of README.md "Fit placement" (the stage-2 PCA of an 18,432-square matrix, the GCCA
    of a 12,288-square one) 568 s of eigh against 821 s, every pair's held-out R^2 within 2.6e-5 of the
    float64 fit's. "float64" keeps them in float64. The results are float64 either way; the full
    eigendecompositions of the whitening, the decoders and the onboarding encoders stay float64."""
    n = len(cov)
    k = min(k, n)
    precision = precision or LARGE_EIGH
    if precision not in ("float64", "float32"):
        raise ValueError(f"top_eig precision must be float64 or float32 ({precision!r})")
    large = n >= SUBSET_MIN and 0 < k <= n // 2
    matrix = np.asarray(cov, dtype=np.float32) if large and precision == "float32" else cov
    linalg = _scipy_linalg()
    if linalg is not None and large:
        values, vectors = linalg.eigh(matrix, subset_by_index=[n - k, n - 1], driver="evr", check_finite=False)
        values, vectors = values[::-1], vectors[:, ::-1]
    else:
        values, vectors = np.linalg.eigh(matrix)
        order = np.argsort(values)[::-1][:k]
        values, vectors = values[order], vectors[:, order]
    if vectors.dtype != np.float64:
        values, vectors = values.astype(np.float64), np.ascontiguousarray(vectors, dtype=np.float64)
    return np.clip(values, 0, None), vectors


# -- the CPU backend --------------------------------------------------------------------------------

class NumpyBackend:
    """The fits' products in numpy (float32 products, float64 sums), as hub.py always did them."""

    name = "cpu"
    prefetch = False                    # IO and BLAS share the cores: no overlap

    def describe(self):
        return {"device": "cpu", "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "numpy": np.__version__}

    def put(self, x):
        return np.asarray(x, dtype=np.float32)

    def host(self, x):
        return np.asarray(x)

    def is_array(self, x):
        return isinstance(x, np.ndarray)

    def mm(self, x, weight):
        return np.asarray(x, dtype=np.float32) @ np.asarray(weight, dtype=np.float32)

    def sub(self, x, vector):
        return np.asarray(x, dtype=np.float32) - np.asarray(vector, dtype=np.float32)

    def add(self, x, vector):
        return np.asarray(x, dtype=np.float32) + np.asarray(vector, dtype=np.float32)

    def minus(self, x, y):
        """x - y, both [n, d]."""
        return np.asarray(x, dtype=np.float32) - np.asarray(y, dtype=np.float32)

    def scale_cols(self, x, vector):
        return np.asarray(x, dtype=np.float32) * np.asarray(vector, dtype=np.float32)

    def cat(self, parts, axis=1):
        return np.concatenate(parts, axis=axis)

    def total(self, parts):
        """The elementwise sum of equally shaped arrays (float32)."""
        out = np.array(parts[0], dtype=np.float32)
        for part in parts[1:]:
            out += part
        return out

    def mean_of(self, parts):
        return np.mean(parts, axis=0)

    def col_sum64(self, x):
        return np.asarray(x).sum(axis=0, dtype=np.float64)

    def col_mean32(self, x):
        return np.asarray(x).mean(axis=0, dtype=np.float64).astype(np.float32)

    def sq_sum64(self, x):
        return float(np.sum(np.square(x), dtype=np.float64))

    def zeros(self, rows, cols, symmetric=False):
        return np.zeros((rows, cols))

    def add_gram(self, acc, x):
        x = np.asarray(x, dtype=np.float32)
        acc += x.T @ x

    def add_cross(self, acc, a, b):
        acc += np.asarray(a, dtype=np.float32).T @ np.asarray(b, dtype=np.float32)

    def fetch(self, acc):
        return acc

    def synchronize(self):
        pass

    def release(self):
        pass

    def stats(self):
        return {}


# -- the GPU backend --------------------------------------------------------------------------------

def _triton_kernel():
    """The Triton accumulation kernel, or None without Triton (or when it cannot compile)."""
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    @triton.jit
    def accumulate_kernel(a_ptr, b_ptr, c_ptr, sa_ptr, sb_ptr, K, M, N, sak, sam, sbk, sbn, scm, scn,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """C[M, N] += diag(sa) A[K, M]^T B[K, N] diag(sb): float16 A and B, float32 accumulation, the
        float32 tile unscaled (sa, sb: float64) and added to the float64 C."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + offs_k
            a = tl.load(a_ptr + kk[:, None] * sak + offs_m[None, :] * sam,
                        mask=(kk[:, None] < K) & (offs_m[None, :] < M), other=0.0)
            b = tl.load(b_ptr + kk[:, None] * sbk + offs_n[None, :] * sbn,
                        mask=(kk[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(tl.trans(a), b)
        sa = tl.load(sa_ptr + offs_m, mask=offs_m < M, other=0.0)
        sb = tl.load(sb_ptr + offs_n, mask=offs_n < N, other=0.0)
        c = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        old = tl.load(c, mask=mask, other=0.0)
        tl.store(c, old + acc.to(tl.float64) * sa[:, None] * sb[None, :], mask=mask)

    return triton, accumulate_kernel


TRITON_CONFIG = {"BM": 128, "BN": 64, "BK": 32, "num_warps": 4, "num_stages": 2}   # 15.8 TFLOPS on 4096^2


class _Acc:
    """A float64 matrix on the device; symmetric: only its upper block triangle (TILE blocks) is summed."""

    def __init__(self, tensor, symmetric):
        self.tensor, self.symmetric = tensor, symmetric


class TorchBackend:
    """The fits' products on a torch GPU (see the module docstring)."""

    name = "gpu"
    prefetch = True                     # gather the next chunk while the GPU works on this one

    def __init__(self, device=None, *, target_s=TARGET_S, max_launch_flops=MAX_LAUNCH_FLOPS, triton=True,
                 calibrate=True, log=None):
        import torch
        self.torch = torch
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        self.device = torch.device(device or "cuda")
        self.target_s, self.max_launch_flops = target_s, max_launch_flops
        self.log = log or (lambda line: None)
        self._weights = OrderedDict()   # id(host array) -> (the array, float16 weight, float32 inverse column scale)
        self._weight_bytes = 0
        self._kernel = _triton_kernel() if triton else None
        self.accumulation = "triton f16->f32 (f64 epilogue)" if self._kernel else "hipBLASLt f16->f16"
        self.rate = {}                  # measured flop/s: "accumulate", "project"
        self.budget = max_launch_flops
        self.launches, self.flops, self.longest_s = 0, 0.0, 0.0
        self._scaled_last = {}          # id -> (tensor, its _version, its _scaled16): see _scaled16
        if calibrate:
            self.calibrate()

    # -- bookkeeping -------------------------------------------------------------------------------

    def describe(self):
        torch = self.torch
        info = {"device": str(self.device), "torch": torch.__version__, "accumulation": self.accumulation,
                "projection": "hipBLASLt f16->f16", "budget_gflop": round(self.budget / 1e9, 1),
                "rate_tflops": {k: round(v / 1e12, 2) for k, v in self.rate.items()}}
        if self.device.type == "cuda":
            info["gpu"] = torch.cuda.get_device_name(self.device)
        return info

    def stats(self):
        """Launches, work, the longest timed launch and the device memory peaks (GiB: on an APU these come
        out of the same RAM, but not out of the process's RSS)."""
        torch = self.torch
        return {"launches": self.launches, "tflop": round(self.flops / 1e12, 2),
                "longest_timed_launch_s": round(self.longest_s, 4),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated(self.device) / (1 << 30), 2),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved(self.device) / (1 << 30), 2)}

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def release(self):
        self._weights.clear()
        self._weight_bytes = 0
        self._scaled_last.clear()
        self.torch.cuda.empty_cache()

    def _time(self, fn, repeat=3):
        """The slowest of `repeat` synchronized launches of fn (seconds; one untimed warm-up first)."""
        torch = self.torch
        fn()
        torch.cuda.synchronize(self.device)
        worst = 0.0
        for _ in range(repeat):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(self.device)
            worst = max(worst, start.elapsed_time(end) / 1e3)
        self.longest_s = max(self.longest_s, worst)
        return worst

    def calibrate(self):
        """Measure the accumulation and projection rates, small sizes first (GPU RULE: a size runs only
        when the launch the smaller one predicts stays under 0.1 s), and set the launch budget."""
        torch = self.torch
        for name, sizes in (("accumulate", ((1024, 1024), (4096, TILE))), ("project", ((1024, 1024), (4096, TILE)))):
            rate = None
            for rows, cols in sizes:
                flops = 2.0 * rows * cols * cols
                if rate is not None and flops / rate > 0.1:
                    break
                x = torch.randn((rows, cols), device=self.device, dtype=torch.float32)
                if name == "accumulate":
                    acc = _Acc(torch.zeros((cols, cols), dtype=torch.float64, device=self.device), False)
                    h, inv = self._scaled16(x)
                    seconds = self._time(lambda: self._accumulate(acc.tensor, h, inv, h, inv, 0, cols, 0, cols))
                else:
                    weight = (torch.randn((cols, cols), device=self.device) / cols ** 0.5).half()
                    h = x.half()
                    seconds = self._time(lambda: h @ weight)
                rate = flops / seconds
                del x
            self.rate[name] = rate
        self.budget = min(self.max_launch_flops, min(self.rate.values()) * self.target_s)
        self.launches, self.flops = 0, 0.0           # stats count the fit's work, not the calibration's
        self._scaled_last.clear()
        torch.cuda.empty_cache()
        self.log(json.dumps({"stage": "fit-backend", **self.describe()}))

    # -- arrays ------------------------------------------------------------------------------------

    def is_array(self, x):
        return isinstance(x, self.torch.Tensor)

    def put(self, x):
        torch = self.torch
        if isinstance(x, torch.Tensor):
            return x if x.dtype == torch.float32 else x.float()
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)

    def host(self, x):
        if isinstance(x, self.torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)

    def _vector(self, values, dtype=None):
        torch = self.torch
        if isinstance(values, torch.Tensor):
            return values.to(self.device, dtype or values.dtype)
        values = np.asarray(values)
        tensor = torch.from_numpy(np.ascontiguousarray(values)).to(self.device)
        return tensor if dtype is None else tensor.to(dtype)

    def sub(self, x, vector):
        return self.put(x) - self._vector(vector, self.torch.float32)

    def add(self, x, vector):
        return self.put(x) + self._vector(vector, self.torch.float32)

    def minus(self, x, y):
        return self.put(x) - self.put(y)

    def scale_cols(self, x, vector):
        return self.put(x) * self._vector(vector, self.torch.float32)

    def cat(self, parts, axis=1):
        return self.torch.cat([self.put(p) for p in parts], dim=axis)

    def total(self, parts):
        out = self.put(parts[0]).clone()
        for part in parts[1:]:
            out += self.put(part)
        return out

    def mean_of(self, parts):
        return self.total(parts) / len(parts)

    def col_sum64(self, x):
        return self.put(x).sum(dim=0, dtype=self.torch.float64).cpu().numpy()

    def col_mean32(self, x):
        x = self.put(x)
        return (x.sum(dim=0, dtype=self.torch.float64) / len(x)).float().cpu().numpy()

    def sq_sum64(self, x):
        x = self.put(x)
        return float((x.double() ** 2).sum()) if x.numel() < (1 << 26) else \
            float(sum(float((x[i:i + 4096].double() ** 2).sum()) for i in range(0, len(x), 4096)))

    # -- products ----------------------------------------------------------------------------------

    def _scale(self, x, dim, statistic, target):
        """Power-of-two factors (float32, exact to apply) along `dim` that bring each slice's RMS
        (statistic "rms") or norm ("norm") to about `target`. Computed in two steps, first to max |x| = 1
        and then to the target, so that no square underflows: GPUs flush float32 subnormals to zero, and
        K rows hold columns of 1e-26 whose squares would vanish and blow the scale up (to float16 inf)."""
        torch = self.torch
        amax = x.abs().amax(dim=dim, keepdim=True)
        first = torch.round(-torch.log2(amax)).clamp(-100, 100)          # zero slices: +100, harmless
        y = x * torch.exp2(first)
        square = y.square().mean(dim=dim, keepdim=True) if statistic == "rms" else y.square().sum(dim=dim, keepdim=True)
        second = torch.round(math.log2(target) - 0.5 * torch.log2(square)).clamp(-60, 60)
        return torch.exp2((first + second).clamp(-120, 120))

    def _scaled16(self, x):
        """(float16 x * s, float64 1/s): column scales s powers of two that bring each column's RMS into
        [1/sqrt 2, sqrt 2] (|x s| <= sqrt(2 rows): no float16 overflow; inputs keep float16's precision).
        The last few results are kept for the same unchanged tensors (a decoder's estimates meet every
        layer); an entry holds its tensor, so the identity cannot pass to another one."""
        held = self._scaled_last.pop(id(x), None)
        if held is not None and held[0] is x and held[1] == x._version:
            self._scaled_last[id(x)] = held           # most recently used last
            return held[2]
        scale = self._scale(x, 0, "rms", 1.0)
        out = (x * scale).half(), (1.0 / scale.double()).reshape(-1)
        self._scaled_last[id(x)] = (x, x._version, out)
        while len(self._scaled_last) > 3:
            self._scaled_last.pop(next(iter(self._scaled_last)))
        return out

    def _accumulate(self, acc, ha, inv_a, hb, inv_b, i0, i1, j0, j1):
        """acc[i0:i1, j0:j1] += diag(inv_a) ha[:, i0:i1]' hb[:, j0:j1] diag(inv_b) (one launch)."""
        k, m, n = ha.shape[0], i1 - i0, j1 - j0
        if self._kernel is not None:
            triton, kernel = self._kernel
            a, b = ha[:, i0:i1], hb[:, j0:j1]
            c = acc[i0:i1, j0:j1]
            grid = (triton.cdiv(m, TRITON_CONFIG["BM"]), triton.cdiv(n, TRITON_CONFIG["BN"]))
            kernel[grid](a, b, c, inv_a[i0:i1], inv_b[j0:j1], k, m, n, a.stride(0), a.stride(1), b.stride(0),
                         b.stride(1), c.stride(0), c.stride(1), BM=TRITON_CONFIG["BM"], BN=TRITON_CONFIG["BN"],
                         BK=TRITON_CONFIG["BK"], num_warps=TRITON_CONFIG["num_warps"],
                         num_stages=TRITON_CONFIG["num_stages"])
        else:
            # without Triton: float16 output, so no more than 2 k <= 2 ROWS per entry after RMS scaling
            out = (ha[:, i0:i1].T @ hb[:, j0:j1]).double()
            out.mul_(inv_a[i0:i1, None]).mul_(inv_b[None, j0:j1])
            acc[i0:i1, j0:j1].add_(out)
        self.launches += 1
        self.flops += 2.0 * k * m * n

    def _rows_step(self, tile_flops_per_row):
        step = int(min(ROWS, self.budget // max(tile_flops_per_row, 1.0)))
        return max(256, step // 256 * 256)

    def zeros(self, rows, cols, symmetric=False):
        return _Acc(self.torch.zeros((rows, cols), dtype=self.torch.float64, device=self.device), symmetric)

    def add_gram(self, acc, x):
        """acc += x'x (only the upper block triangle when acc is symmetric; fetch mirrors it)."""
        x = self.put(x)
        n, d = x.shape
        tile = min(TILE, d)
        step = self._rows_step(2.0 * tile * tile)
        for r0 in range(0, n, step):
            h, inv = self._scaled16(x if step >= n else x[r0:r0 + step])
            for i0 in range(0, d, tile):
                for j0 in range(i0 if acc.symmetric else 0, d, tile):
                    self._accumulate(acc.tensor, h, inv, h, inv, i0, min(i0 + tile, d), j0, min(j0 + tile, d))

    def add_cross(self, acc, a, b):
        """acc += a'b."""
        a, b = self.put(a), self.put(b)
        n, da = a.shape
        db = b.shape[1]
        ta, tb = min(TILE, da), min(TILE, db)
        step = self._rows_step(2.0 * ta * tb)
        for r0 in range(0, n, step):
            # whole operands when they fit one step: the same estimates meet every layer (_scaled16 keeps them)
            ha, inv_a = self._scaled16(a if step >= n else a[r0:r0 + step])
            hb, inv_b = self._scaled16(b if step >= n else b[r0:r0 + step])
            for i0 in range(0, da, ta):
                for j0 in range(0, db, tb):
                    self._accumulate(acc.tensor, ha, inv_a, hb, inv_b, i0, min(i0 + ta, da), j0, min(j0 + tb, db))

    def fetch(self, acc):
        """The float64 host matrix of an accumulator (a symmetric one mirrored); FloatingPointError when
        it is not finite."""
        out = acc.tensor.cpu().numpy()
        if acc.symmetric:
            d = len(out)
            tile = min(TILE, d)
            for i0 in range(0, d, tile):
                for j0 in range(0, i0, tile):          # lower blocks from the upper ones
                    out[i0:i0 + tile, j0:j0 + tile] = out[j0:j0 + tile, i0:i0 + tile].T
        if not np.isfinite(out).all():
            raise FloatingPointError("non-finite sums from the GPU fit backend")
        return out

    def _weight(self, weight):
        """(float16 weight with columns scaled to norm ~PROJECTION_NORM, float32 inverse scales) on the
        device, cached by the host array's identity (the cache holds the array, so its id stays unique)."""
        key = id(weight)
        held = self._weights.get(key)
        if held is not None and held[0] is weight:
            self._weights.move_to_end(key)
            return held[1], held[2]
        torch = self.torch
        w = self.put(weight)
        scale = self._scale(w, 0, "norm", PROJECTION_NORM)
        w16, inverse = (w * scale).half(), (1.0 / scale).reshape(-1)
        del w
        self._weights[key] = (weight, w16, inverse)
        self._weight_bytes += w16.numel() * 2
        while self._weight_bytes > WEIGHT_CACHE_BYTES and len(self._weights) > 1:
            _, (_, old, _) = self._weights.popitem(last=False)
            self._weight_bytes -= old.numel() * 2
        if not torch.isfinite(w16).all():
            raise FloatingPointError("non-finite weight for the GPU fit backend")
        return w16, inverse

    def mm(self, x, weight):
        """x @ weight in float32: float16 product (rows and weight columns scaled by powers of two to norm
        ~32, so no entry exceeds ~2048), tiled by rows to the launch budget."""
        torch = self.torch
        x = self.put(x)
        w16, inverse = self._weight(weight)
        n, k = x.shape
        m = w16.shape[1]
        out = torch.empty((n, m), dtype=torch.float32, device=self.device)
        if n == 0:
            return out
        rows_scale = self._scale(x, 1, "norm", PROJECTION_NORM)         # [n, 1]
        step = max(64, int(self.budget // (2.0 * k * m)) // 64 * 64)
        for r0 in range(0, n, step):
            r1 = min(r0 + step, n)
            h = (x[r0:r1] * rows_scale[r0:r1]).half()
            product = (h @ w16).float()
            out[r0:r1] = product.mul_(1.0 / rows_scale[r0:r1]).mul_(inverse[None, :])
            self.launches += 1
            self.flops += 2.0 * (r1 - r0) * k * m
        return out


# -- selection --------------------------------------------------------------------------------------

def gpu_available():
    """(True, None) when torch sees a ROCm or CUDA GPU, else (False, why)."""
    try:
        import torch
    except ImportError:
        return False, "torch is not importable (the experiment venv has none: run fits with ~/venvs/rocm/bin/python)"
    try:
        if torch.cuda.is_available():
            return True, None
    except Exception as error:          # noqa: BLE001 (broken driver installs raise from the probe)
        return False, f"torch.cuda: {error}"
    return False, "torch sees no ROCm or CUDA GPU"


def resolve(device="auto"):
    """"cpu" or "gpu" for --fit-device: auto picks the GPU when torch sees one (measurably cheaper per fit,
    README "Fit placement"), else the CPU; gpu without one is a RuntimeError."""
    if device not in DEVICES:
        raise ValueError(f"fit device must be one of {', '.join(DEVICES)} ({device!r})")
    if device == "cpu":
        return "cpu"
    available, why = gpu_available()
    if available:
        return "gpu"
    if device == "gpu":
        raise RuntimeError(f"--fit-device gpu: {why}")
    return "cpu"


def make(device, *, log=None, **options):
    """The backend for a resolved device ("cpu" or "gpu")."""
    return TorchBackend(log=log, **options) if device == "gpu" else NumpyBackend()


CPU = NumpyBackend()


def current():
    """The backend the fits use now (hub's products): the one `using` installed, else the CPU."""
    return _CURRENT[-1] if _CURRENT else CPU


class using:
    """with fit_backend.using(backend): the fits run on it (restored after; nests)."""

    def __init__(self, backend):
        self.backend = backend

    def __enter__(self):
        _CURRENT.append(self.backend)
        return self.backend

    def __exit__(self, *exc):
        _CURRENT.pop()
        self.backend.release()
        return False


def timed_sync(backend):
    """time.perf_counter() after the backend's queued work finished (profiling)."""
    backend.synchronize()
    return time.perf_counter()
