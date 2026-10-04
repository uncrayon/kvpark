"""CPU backend: float32 numpy (the reference).

RoPE uses kvstate's own Rope (spec.rope: libm cos/sin, the CPU build's fused multiply-adds, YaRN
and the [c_kv | k_pe] offset of MLA rows) and its exact float64 expressions, applied in place to
the rotated pairs only (kvstate.cache_keys copies and widens the whole row; Qwen rotates 64 of
256 dims, an MLA row 64 of 576), so keys are bit-identical to kvstate.cache_keys /
pre_rope_keys. MLA latents are put back on their RMS shell with kvstate.mla_renormalize_latent
itself (renormalize_latents), and an MLA source's V rows are the c_kv view of its K rows.

The per-layer work outside BLAS (gathering source rows and widening them from float16,
rotations, the float16 cast of the cache: numpy runs each single-threaded) runs on a few
threads, one layer each (numpy releases the GIL in these loops; the results are those of a
sequential loop): on a 13K switch the assemble stage of Muse -> Qwen3.8 went from 1.2 s to
0.4 s and the encode stage of Qwen3.8 -> Muse from 2.6 s to 1.9 s (8 threads).
"""

from concurrent.futures import ThreadPoolExecutor
import functools
import os
import sys
import threading
import time

import numpy as np

from .base import Backend, cast_to_cache, hub_module, kvstate, latent_columns, stored_rows

import handoff  # noqa: E402  (on sys.path through .base)

OPTIMIZED = ("openblas", "mkl", "blis", "accelerate", "armpl", "aocl", "flexiblas", "atlas")
_POOL = []
_POOL_LOCK = threading.Lock()


def layer_threads():
    """Threads for per-layer work: as many as BLAS uses (OPENBLAS_NUM_THREADS / OMP_NUM_THREADS), else 8."""
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        try:
            return max(1, int(os.environ[name]))
        except (KeyError, ValueError):
            continue
    return max(1, min(8, os.cpu_count() or 1))


def map_layers(function, items):
    """[function(item) for item in items], on the layer threads when there is more than one item."""
    items = list(items)
    if len(items) < 2 or layer_threads() == 1:
        return [function(item) for item in items]
    with _POOL_LOCK:
        if not _POOL:
            _POOL.append(ThreadPoolExecutor(max_workers=layer_threads(), thread_name_prefix="numpy-layers"))
    return list(_POOL[0].map(function, items))


@functools.lru_cache(maxsize=1)
def blas_info():
    """numpy's BLAS: build name, loaded libraries (Linux), a 512^3 sgemm in GFLOPS and a warning
    when it is the reference BLAS (single-threaded, 10-100x slower: the engine then takes
    tens of seconds to minutes per switch)."""
    info = {"name": None, "libraries": [], "gflops": None, "reference": False}
    try:
        info["name"] = np.show_config(mode="dicts")["Build Dependencies"]["blas"].get("name")
    except Exception:                       # older numpy, or a build without the dict mode
        pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/maps", encoding="utf-8", errors="replace") as maps:
                paths = {line.split()[-1] for line in maps if line.count("/") and any(
                    word in line.rsplit("/", 1)[-1].lower() for word in ("blas", "mkl", "blis"))}
            info["libraries"] = sorted(paths)
        except OSError:
            pass
    a = np.full((512, 512), 0.5, dtype=np.float32)
    best = None
    for _ in range(3):
        started = time.perf_counter()
        a @ a
        elapsed = time.perf_counter() - started
        best = elapsed if best is None else min(best, elapsed)
    info["gflops"] = round(2 * 512 ** 3 / max(best, 1e-9) / 1e9, 1)
    return classify_blas(info)


def classify_blas(info):
    """Mark info["reference"] and add info["warning"] from the build name, libraries and speed."""
    libraries = " ".join(p.lower() for p in info["libraries"])        # Debian alternatives: .../openblas-pthread/
    name = (info["name"] or "").lower()
    optimized = any(word in libraries or word in name for word in OPTIMIZED)
    info["reference"] = (any("/blas/libblas.so" in p for p in info["libraries"])
                         or (name in ("blas", "cblas") and not optimized))
    if info["reference"]:
        info["warning"] = ("numpy uses the reference BLAS: translation will be 10-100x slower; run the engine "
                           "with a Python whose numpy bundles OpenBLAS (e.g. the experiment venv)")
    elif info["gflops"] is not None and info["gflops"] < 50:
        info["warning"] = f"numpy's BLAS is slow ({info['gflops']} GFLOPS on a 512^3 product)"
    return info


def rope_pairs(rope):
    """The two pair slices of a kvstate.Rope (Rope._pairs)."""
    o, n = rope.offset, rope.n_dims
    if rope.mode == "norm":
        return slice(o, o + n, 2), slice(o + 1, o + n, 2)
    return slice(o, o + n // 2), slice(o + n // 2, o + n)


def rotate_inplace(x, positions, rope, *, inverse, tables=None):
    """kvstate Rope.apply (or .invert) on float32 x [n, heads, dim], in place; tables caches cos/sin per rope."""
    if tables is not None and rope in tables:
        cos, sin = tables[rope]
    else:
        cos, sin = rope.cos_sin(positions)
        if tables is not None:
            tables[rope] = cos, sin
    cos, sin = cos[:, None, :], sin[:, None, :]
    a, b = rope_pairs(rope)
    x0, x1 = x[:, :, a], x[:, :, b]
    if not inverse:
        # rotate_pairs as the pinned CPU build compiles it (fused multiply-adds), as kvstate emulates it
        wide = x0.astype(np.float64)
        first = (wide * cos - (x1 * sin).astype(np.float64)).astype(np.float32)
        second = (wide * sin + (x1 * cos).astype(np.float64)).astype(np.float32)
    else:
        # kvstate.Rope.invert: the transposed rotation divided by cos^2 + sin^2 of the float32 table
        c, s, y0, y1 = (v.astype(np.float64) for v in (cos, sin, x0, x1))
        norm = c * c + s * s
        first, second = ((y0 * c + y1 * s) / norm).astype(np.float32), ((y1 * c - y0 * s) / norm).astype(np.float32)
    x[:, :, a] = first
    x[:, :, b] = second
    return x


class NumpyBackend(Backend):
    """The engine's compute interface, in float32 numpy:

    encoder(hub, kind, name) / decoder(...)   resident hub sides (EncoderSide / DecoderSide)
    encode(enc, cache, kind, positions, chunk_rows, center)
                         hub latents z [n, D] of the source rows at `positions` (K de-rotated),
                         minus center.z_mean when a decoder is given as center
    section(dec, z, layers, n_rows, mapped, mapped_rows, fill_rows, fill, centered)
                         float32 [n_rows, sum d_l]: rows mapped_rows decoded from z[mapped],
                         rows fill_rows the decoder mean (fill None) or `fill`
    renormalize_latents(spec, layers, values, norms)
                         in place on decoded K rows: MLA latents back on their RMS shell
    cache_rows(spec, layers, kind, values, positions, dtype, out=None)
                         {layer: [n, heads, dim] in the cache dtype}: K rotated at `positions`
                         (in place in values); with out ({layer: host array}) written into those arrays
    host(x)              a backend array as numpy
    """

    name = "numpy"
    device = "cpu"
    window_rows = 16384                   # BLAS products of fewer rows run 10-20% slower

    def describe(self):
        info = blas_info()
        out = {"backend": self.name, "device": self.device, "numpy": np.__version__,
               "blas": info["name"], "blas_libraries": [os.path.basename(p) for p in info["libraries"]],
               "blas_gflops": info["gflops"],
               "threads": os.environ.get("OPENBLAS_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")}
        if info.get("warning"):
            out["warning"] = info["warning"]
        return out

    def _load(self, hub, kind, name, role):
        if isinstance(hub, hub_module.Hub):
            return hub.side(kind, name, role)
        return hub[kind].encoder_side(name) if role == "encoder" else hub[kind].decoder_side(name)

    def rows(self, cache, kind, layers, positions):
        """{layer: float32 [n, heads * dim]} source rows at positions, K de-rotated (SourceCache.rows_many)."""
        tables, cells_of = {}, {}
        for layer in layers:
            stream, index = cache.where[layer]
            if id(stream) not in cells_of:
                cells_of[id(stream)] = handoff._cells(index, positions, f"source layer {layer}")
            rope = cache.spec.rope(layer) if kind == "k" else None
            if rope is not None and rope not in tables:
                tables[rope] = rope.cos_sin(positions)          # shared by the layers (and threads) using it

        def one(layer):
            stream, _ = cache.where[layer]
            block = np.asarray(stored_rows(cache.spec, stream, kind, layer)[cells_of[id(stream)]], dtype=np.float32)
            rope = cache.spec.rope(layer) if kind == "k" else None
            if rope is not None:
                rotate_inplace(block, positions, rope, inverse=True, tables=tables)
            return block.reshape(len(positions), -1)
        return dict(zip(layers, map_layers(one, layers)))

    def encode(self, encoder, cache, kind, positions, chunk_rows=4096, center=None):
        """z [n, D] float32; with center (a decoder) z - center.z_mean, ready for section(centered=True)."""
        positions = np.asarray(positions, dtype=np.int64)
        parts = [encoder.encode(self.rows(cache, kind, encoder.layers, positions[start:start + chunk_rows]))
                 for start in range(0, len(positions), chunk_rows)]
        z = np.concatenate(parts) if parts else np.zeros((0, encoder.dim), dtype=np.float32)
        if center is not None:
            z -= center.z_mean
        return z

    def section(self, decoder, z, layers, n_rows, mapped, mapped_rows, fill_rows, fill=None, centered=False):
        weight, mean = decoder.section(layers)
        product = None
        if len(mapped):
            rows = z if len(mapped) == len(z) and np.array_equal(mapped, np.arange(len(z))) else z[mapped]
            product = (rows if centered else rows - decoder.z_mean) @ weight
            product += mean
            if len(mapped) == n_rows and np.array_equal(mapped_rows, np.arange(n_rows)):
                return product                  # every row decoded, in order: no copy
        out = np.empty((n_rows, len(mean)), dtype=np.float32)
        if product is not None:
            out[mapped_rows] = product
        if len(fill_rows):
            out[fill_rows] = mean if fill is None else fill
        return out

    def renormalize_latents(self, spec, layers, values, norms):
        """In place on decoded K rows values [n, sum d_l] of `layers`: every MLA layer's c_kv columns put
        back on their RMS shell (kvstate.mla_renormalize_latent with norms[layer] = (weight, eps))."""
        def one(item):
            layer, start, lora = item
            weight, eps = norms[layer]
            values[:, start:start + lora] = kvstate.mla_renormalize_latent(values[:, start:start + lora], weight, eps)
        map_layers(one, latent_columns(spec, layers, norms))
        return values

    def cache_rows(self, spec, layers, kind, values, positions, dtype, out=None):
        n, blocks, offset, tables = len(positions), [], 0, {}
        positions = np.asarray(positions, dtype=np.int64)
        for layer in layers:
            shape = spec.head_shape(layer, kind)
            width = shape[0] * shape[1]
            rope = spec.rope(layer) if kind == "k" else None
            if rope is not None and rope not in tables:
                tables[rope] = rope.cos_sin(positions)
            # a view: rotated in place
            blocks.append((layer, values[:, offset:offset + width].reshape((n,) + shape), rope))
            offset += width

        def one(item):
            layer, block, rope = item
            if rope is not None:
                rotate_inplace(block, positions, rope, inverse=False, tables=tables)
            return cast_to_cache(block, dtype, f"layer {layer} {kind.upper()}", out=None if out is None else out[layer])
        return dict(zip(layers, map_layers(one, blocks)))

    def host(self, x):
        return np.asarray(x)
