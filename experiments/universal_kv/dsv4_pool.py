"""DeepSeek V4 pooled cache rows from per-token compressor inputs (numpy).

A DeepSeek V4 (llama.cpp arch deepseek4) layer of compress ratio 4 (CSA, with a lightning
indexer, LID) or 128 (HCA) keeps one pooled row per completed block of `ratio` tokens; the
tokens of an unfinished block live only in a float32 ring (kvstate.CompressSpec). Per token t
and pooling type the graph computes

    a_t = W_comp_kv x_t,   s_t = W_comp_gate x_t   (x_t the layer's attention input)

(graph names {csa,hca,lid}_state_kv and _state_score; src/models/deepseek4.cpp:946-971,
1030-1040) and adds APE[pos mod ratio] to s_t (_state_score_ape). A block's row is

    RMSNorm(sum_j softmax_j(s) * a_j) * comp_norm, RoPE'd at the block start

with the softmax per channel over the block's window: HCA, the block's 128 tokens; CSA and LID,
eight entries, the previous block's 4 tokens in channels [0, D) (a zero a and -inf score before
the first block) and the block's own 4 in channels [D, 2D) (build_hca_compressed_kv_from_state
and build_overlap_compressed_kv_from_state, src/models/deepseek4.cpp:471-589). RoPE is NORM on
the last n_rot dims with the compress base and the context's YaRN (kvstate.ModelSpec.rope, roles
"pooled" and "lid"); LID rows are then Hadamard-rotated, and in a quantized cache CSA and HCA
rows too (hadamard_rows). A row is written when the block's last token is decoded, at row
pos // ratio (src/llama-kv-cache-dsv4.cpp:497-528).

So pooled rows need, besides the per-token (a, s), only small tensors of the target GGUF (APE,
the compressor norms) and its RoPE metadata; no llama.cpp graph. The sums run in float64 and
the rows come out float32, which matches the stored float16 rows to float16 rounding (llama.cpp
sums in its own float32 order). `Pooler` streams the rows window by window of positions; the
rings at a split use ring_rows.
"""

from dataclasses import dataclass, field
import functools
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "linear_transfer"))

import gguf_tensors  # noqa: E402
import kvstate  # noqa: E402

# GGUF tensor names of each pooling type's APE and norm (src/llama-arch.cpp, src/models/deepseek4.cpp:136-150)
TENSORS = {"csa": ("attn_compressor_ape", "attn_compressor_norm"),
           "hca": ("attn_compressor_ape", "attn_compressor_norm"),
           "lid": ("indexer_compressor_ape", "indexer_compressor_norm")}
# capture names of the per-token inputs (state driver / eval callback names, before APE)
CAPTURES = {pool: (f"{pool}_state_kv", f"{pool}_state_score") for pool in TENSORS}
SPLIT = 128             # a split position a multiple of the HCA ratio leaves no HCA block open


@dataclass
class PoolTensors:
    """The target tensors pooling needs: per (pool, layer) the APE [ratio, state_width] and the
    compressor norm weight [width], float32."""

    ape: dict = field(default_factory=dict)
    norm: dict = field(default_factory=dict)


@functools.lru_cache(maxsize=4)
def _gguf_pool_tensors(path, size, mtime_ns, pools):
    tensors = PoolTensors()
    for name, layers in pools:
        ape_name, norm_name = TENSORS[name]
        for layer in layers:
            tensors.ape[name, layer] = gguf_tensors.read_tensor(path, f"blk.{layer}.{ape_name}.weight")
            tensors.norm[name, layer] = gguf_tensors.read_tensor(path, f"blk.{layer}.{norm_name}.weight")
    return tensors


def pool_tensors(gguf, spec):
    """PoolTensors of a deepseek4 GGUF (read once per file identity; gguf_tensors dequantizes F16, BF16
    and Q8_0, so a GGUF that quantizes APE otherwise is refused there)."""
    path = Path(gguf).resolve()
    status = path.stat()
    pools = tuple((p.name, tuple(p.layers)) for p in spec.pools())
    tensors = _gguf_pool_tensors(str(path), status.st_size, status.st_mtime_ns, pools)
    for p in spec.pools():
        for layer in p.layers:
            if tensors.ape[p.name, layer].shape != (p.ratio, p.state_width) or \
                    tensors.norm[p.name, layer].shape != (p.width,):
                raise ValueError(f"{gguf}: {p.name} layer {layer} APE/norm shapes differ from the model's")
    return tensors


def with_ape(pool, tensors, layer, score, positions):
    """float32 s + APE[pos mod ratio] (the score the graph pools and the ring keeps)."""
    ape = tensors.ape[pool.name, layer]
    return (np.asarray(score, dtype=np.float32) + ape[np.asarray(positions) % pool.ratio]).astype(np.float32)


def rot_size(width):
    """Hadamard block of llama_kv_cache::build_input_k_rot (src/llama-kv-cache.cpp:1435-1453): the largest
    power of two that divides the row width, at least 64."""
    n = 64
    while width % (2 * n) == 0:
        n *= 2
    return n


def hadamard_rows(rows):
    """rows [..., width] times the orthonormal Sylvester H blockwise (llama_mul_mat_hadamard; float64 sums)."""
    rows = np.asarray(rows, dtype=np.float32)
    n = rot_size(rows.shape[-1])
    h = kvstate.hadamard(n).astype(np.float64)
    blocks = rows.reshape(-1, n).astype(np.float64) @ h.T
    return blocks.reshape(rows.shape).astype(np.float32)


def _softmax_sum(a, s):
    """sum_j softmax_j(s) a_j per channel over axis 1 (float64; -inf scores weigh nothing)."""
    s = np.asarray(s, dtype=np.float64)
    w = np.exp(s - s.max(axis=1, keepdims=True))
    return (w * np.asarray(a, dtype=np.float64)).sum(axis=1) / w.sum(axis=1)


def pool_blocks(spec, pool, layer, a, s_ape, start, tensors, *, previous=None, rotate=None):
    """float32 [n / ratio, width] stored rows of the blocks of positions [start, start + n) (start and n
    multiples of the ratio), from float32 a and APE-added scores [n, state_width].

    previous: (a, s_ape) [ratio, state_width] of the block before `start` for an overlapping pool
    (CSA, LID), None at position 0. rotate: Hadamard rows (default: LID only; pass True for the CSA/HCA
    rows of a quantized cache)."""
    a, s_ape = np.asarray(a, dtype=np.float32), np.asarray(s_ape, dtype=np.float32)
    n, ratio, width = len(a), pool.ratio, pool.width
    if start % ratio or n % ratio or a.shape != (n, pool.state_width) or s_ape.shape != a.shape:
        raise ValueError(f"{pool.name} pooling needs whole blocks of {ratio} rows of width {pool.state_width}")
    n_blocks = n // ratio
    if n_blocks == 0:
        return np.zeros((0, width), dtype=np.float32)
    if pool.overlap:
        if previous is None:
            if start != 0:
                raise ValueError(f"{pool.name} pooling at {start} needs the previous block")
            prev_a = np.zeros((ratio, pool.state_width), dtype=np.float32)
            prev_s = np.full((ratio, pool.state_width), -np.inf, dtype=np.float32)
        else:
            prev_a, prev_s = (np.asarray(x, dtype=np.float32) for x in previous)
        all_a = np.concatenate([prev_a, a]).reshape(n_blocks + 1, ratio, pool.state_width)
        all_s = np.concatenate([prev_s, s_ape]).reshape(n_blocks + 1, ratio, pool.state_width)
        # block b: the previous block's first half of the channels, then its own second half
        window_a = np.concatenate([all_a[:-1, :, :width], all_a[1:, :, width:]], axis=1)
        window_s = np.concatenate([all_s[:-1, :, :width], all_s[1:, :, width:]], axis=1)
    else:
        window_a = a.reshape(n_blocks, ratio, width)
        window_s = s_ape.reshape(n_blocks, ratio, width)
    pooled = _softmax_sum(window_a, window_s)
    eps = spec.compress.norm_eps
    normed = pooled / np.sqrt(np.mean(pooled * pooled, axis=-1, keepdims=True) + eps)
    normed = (normed * tensors.norm[pool.name, layer].astype(np.float64)).astype(np.float32)
    rope = spec.rope(layer, "lid" if pool.name == "lid" else "pooled")
    starts = start + ratio * np.arange(n_blocks, dtype=np.int64)
    rows = rope.apply(normed[:, None, :], starts)[:, 0, :]
    if rotate if rotate is not None else pool.name == "lid":
        rows = hadamard_rows(rows)
    return rows


def ring_rows(pool, a, s_ape, positions, *, minimal=False):
    """(kv, score) float32 [state_size, state_width]: the ring of a native state whose last token sits
    at positions[-1], from the a and APE-added scores of `positions` (at least the last state_size of
    them, or, minimal, of the last ratio). Row p % state_size holds token p (llama_dsv4_comp_state).

    minimal keeps only what a decode continuing at a split p0 = positions[-1] + 1 with p0 a multiple of
    the HCA ratio reads (src/llama-kv-cache-dsv4.cpp:473-488, 530-538): for CSA/LID the first half of
    the channels of the last `ratio` tokens (the next block's previous half), for HCA nothing."""
    kv = np.zeros((pool.state_size, pool.state_width), dtype=np.float32)
    score = np.zeros_like(kv)
    positions = np.asarray(positions, dtype=np.int64)
    if minimal:
        if not pool.overlap:
            return kv, score
        keep, channels = pool.ratio, slice(0, pool.width)
    else:
        keep, channels = pool.state_size, slice(None)
    if len(positions) < keep or (np.diff(positions[-keep:]) != 1).any():
        raise ValueError(f"the {pool.name} ring needs the last {keep} consecutive positions")
    rows = positions[-keep:] % pool.state_size
    kv[rows, channels] = np.asarray(a, dtype=np.float32)[-keep:, channels]
    score[rows, channels] = np.asarray(s_ape, dtype=np.float32)[-keep:, channels]
    return kv, score


class Pooler:
    """Pooled rows of one layer and pooling type, fed in windows of consecutive positions from 0 (each
    a multiple of the ratio long, except possibly the last); keeps the previous block of an overlapping
    pool and the last state_size tokens for the ring."""

    def __init__(self, spec, pool, layer, tensors, *, rotate=None):
        self.spec, self.pool, self.layer, self.tensors, self.rotate = spec, pool, layer, tensors, rotate
        self.next, self.rows, self.closed = 0, [], False
        self.previous = None
        self.tail_a = np.zeros((0, pool.state_width), dtype=np.float32)
        self.tail_s = np.zeros((0, pool.state_width), dtype=np.float32)

    def feed(self, a, score):
        """a and pre-APE scores float32 [n, state_width] of positions [self.next, self.next + n)."""
        if self.closed:
            raise ValueError("a window ending inside a block must be the last one")
        a = np.asarray(a, dtype=np.float32)
        positions = np.arange(self.next, self.next + len(a), dtype=np.int64)
        s_ape = with_ape(self.pool, self.tensors, self.layer, score, positions)
        whole = len(a) - len(a) % self.pool.ratio
        if whole:
            self.rows.append(pool_blocks(self.spec, self.pool, self.layer, a[:whole], s_ape[:whole], self.next,
                                         self.tensors, previous=self.previous, rotate=self.rotate))
            if self.pool.overlap:
                self.previous = (a[whole - self.pool.ratio:whole], s_ape[whole - self.pool.ratio:whole])
        keep = self.pool.state_size
        self.tail_a = np.concatenate([self.tail_a, a])[-keep:]
        self.tail_s = np.concatenate([self.tail_s, s_ape])[-keep:]
        self.next += len(a)
        self.closed = len(a) != whole

    def result(self, *, minimal=False):
        """(rows float32 [completed blocks, width], ring (kv, score)) after the last window; `minimal`
        (ring_rows) only at a split that is a multiple of the HCA ratio."""
        if minimal and self.next % SPLIT:
            raise ValueError(f"a minimal ring needs a split at a multiple of {SPLIT}, not {self.next}")
        rows = np.concatenate(self.rows) if self.rows else np.zeros((0, self.pool.width), dtype=np.float32)
        positions = np.arange(self.next - len(self.tail_a), self.next, dtype=np.int64)
        return rows, ring_rows(self.pool, self.tail_a, self.tail_s, positions, minimal=minimal)
