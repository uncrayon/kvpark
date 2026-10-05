"""Vulkan backend: the engine's products as ggml graphs (libkvbridge, ggml/kvbridge.cpp), no torch.

ggml's Vulkan backend runs on AMD, NVIDIA and Intel GPUs (and on machines without ROCm or CUDA);
the library falls back to ggml's CPU backend when there is no Vulkan device (only when asked for:
backends.select prefers numpy on a CPU). The library is found at $KVPARK_GGML_BRIDGE, else at the
default build path runtime-ggml/build/libkvbridge.so (ggml/README.md builds it); the device is
$KVPARK_GGML_DEVICE ("auto": the first GPU; "cpu"; or a ggml device name such as "Vulkan1").

Arithmetic as torch_backend's: float16 weights and activations, float32 accumulation
(GGML_PREC_F32 on every product: ggml's default precision accumulates in float16 on Vulkan,
1.2e-2 relative error on the hub's products; this sits at the float16 input-rounding floor, a
few 1e-4 against the float32 numpy reference). Weights are uploaded once per side, transposed
and converted to float16 on the device (format 2 bundles: their float16 file as it is); means,
z_mean and the latents z stay float32.

RoPE: the device only does the rotation's multiply-adds; the cos/sin tables are computed here
per call from the positions, with kvstate's angles (the float32 theta recurrence of ggm-cpu,
freq factors, freq_scale, the YaRN mix and magnitude, kvstate's formulas) and float64 cos/sin
rounded to float32 (kvstate uses libm cosf/sinf: the same values up to their last-bit rounding).
The inverse uses c / (c^2 + s^2), s / (c^2 + s^2) as kvstate.Rope.invert. So keys agree with
kvstate.cache_keys / pre_rope_keys to float32 rounding at any position (ggml's own rope kernels
evaluate sin/cos in float32 on the GPU, far from kvstate at 262K positions). NoPE layers are not
rotated; an MLA row rotates its k_pe tail (the Rope offset). Every rotation is checked against
kvstate on first use (supports) and refused with NotImplementedError when it disagrees, and so
are the device's products against numpy, once (a driver can get them wrong: ggml-vulkan on
Mesa's llvmpipe does past 64 rows that are not a multiple of 64).

Device arrays (the latents z, a section's rows) are DeviceArray handles; host(x) reads one back.
No device tensor exceeds the device's largest buffer (ggml-vulkan's is 1 GiB): a DeviceArray is
held in row blocks of at most that size (one block for a 4096-row window up to 65536 floats
wide), and the library bounds the rows of each graph by it. encode checks the latents finite and
cache_rows the rows in the cache dtype, on the device (FloatingPointError otherwise, after which
the engine retries on numpy); cache_rows copies each layer into its host array. MLA latents are
put back on their RMS shell on the host (renormalize_latents: kvstate's arithmetic).
"""

import ctypes
import functools
import os
from pathlib import Path
import sys
import threading

import numpy as np

from .base import Backend, hub_module, kvstate, latent_columns, stored_rows

import handoff  # noqa: E402  (on sys.path through .base)

LIBRARY_ENV = "KVPARK_GGML_BRIDGE"
DEVICE_ENV = "KVPARK_GGML_DEVICE"
REPO = Path(__file__).resolve().parents[3]
BUILD = REPO / "runtime-ggml" / "build"
LIBRARY_NAMES = {"darwin": "libkvbridge.dylib", "win32": "kvbridge.dll"}
CONFIGS = ("", "Release", "RelWithDebInfo")      # multi-config generators (MSVC) build into a subdirectory
VERSION = 1                     # kvb_version() this module speaks
ROPE_TYPES = ("norm", "neox", "imrope", "mrope")
F32, F16, I32 = 0, 1, 2
DTYPES = {"f32": F32, "f16": F16}
OK, ERROR, OUT_OF_MEMORY, NOT_FINITE = 0, -1, -2, 1
CPU, GPU, IGPU = 0, 1, 2        # kvb_device_info kinds
# bytes of one graph's product (or stored rows) per chunk: 4096 rows of Qwen3.8's attention
# section; a wider section (a 70B-class target's 81920 floats) is decoded in fewer rows at a time
GRAPH_BYTES = 256 << 20


class _Rope(ctypes.Structure):
    _fields_ = [("neox", ctypes.c_int32), ("n_rot", ctypes.c_int32), ("offset", ctypes.c_int32),
                ("pad", ctypes.c_int32), ("cos", ctypes.c_void_p), ("sin", ctypes.c_void_p)]


class _EncodeLayer(ctypes.Structure):
    _fields_ = [("rows", ctypes.c_void_p), ("dtype", ctypes.c_int32), ("rope", ctypes.c_int32),
                ("span", ctypes.c_int64), ("index", ctypes.c_void_p), ("heads", ctypes.c_int64),
                ("head_dim", ctypes.c_int64), ("mean", ctypes.c_void_p), ("basis", ctypes.c_void_p)]


class _CacheLayer(ctypes.Structure):
    _fields_ = [("heads", ctypes.c_int64), ("head_dim", ctypes.c_int64), ("rope", ctypes.c_int32),
                ("pad", ctypes.c_int32), ("out", ctypes.c_void_p)]


def library_path():
    """$KVPARK_GGML_BRIDGE, else the default build's library (in its configuration subdirectory
    with a multi-config generator)."""
    value = os.environ.get(LIBRARY_ENV)
    if value:
        return Path(value)
    name = LIBRARY_NAMES.get(sys.platform, "libkvbridge.so")
    for config in CONFIGS:
        if (BUILD / config / name).is_file():
            return BUILD / config / name
    return BUILD / name


@functools.lru_cache(maxsize=None)
def _library(path):
    lib = ctypes.CDLL(path)
    p, i64, i32, vp = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p
    for name, restype, argtypes in (
            ("kvb_version", i32, []),
            ("kvb_last_error", ctypes.c_char_p, []),
            ("kvb_last_error_code", i32, []),
            ("kvb_ggml_version", ctypes.c_char_p, []),
            ("kvb_device_count", i32, []),
            ("kvb_device_info", i32, [i32, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
                                      ctypes.POINTER(i32), ctypes.POINTER(ctypes.c_uint64),
                                      ctypes.POINTER(ctypes.c_uint64)]),
            ("kvb_init", p, [ctypes.c_char_p, i32]),
            ("kvb_free", None, [p]),
            ("kvb_info", i32, [p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
                               ctypes.POINTER(i32), ctypes.POINTER(ctypes.c_uint64)]),
            ("kvb_memory", None, [p, ctypes.POINTER(i64)]),
            ("kvb_reset_peak", None, [p]),
            ("kvb_release", None, [p]),
            ("kvb_synchronize", i32, [p]),
            ("kvb_new", p, [p, i32, i64, i64, i32]),
            ("kvb_upload", p, [p, vp, i32, i64, i64, i32, i32, i32]),
            ("kvb_upload_many", i32, [p, i32, ctypes.POINTER(vp), ctypes.POINTER(i32), ctypes.POINTER(i64),
                                      ctypes.POINTER(i64), i32, i32, i32, ctypes.POINTER(p)]),
            ("kvb_concat_rows", p, [p, i32, ctypes.POINTER(p), ctypes.POINTER(i64), ctypes.POINTER(i64), i32]),
            ("kvb_read", i32, [p, p, i64, i64, vp]),
            ("kvb_write", i32, [p, p, i64, i64, vp]),
            ("kvb_shape", None, [p, ctypes.POINTER(i64), ctypes.POINTER(i64), ctypes.POINTER(i32),
                                 ctypes.POINTER(i64)]),
            ("kvb_tensor_free", None, [p, p]),
            ("kvb_encode", i32, [p, i64, i32, ctypes.POINTER(_EncodeLayer), i32, ctypes.POINTER(_Rope), p, p, p,
                                 i64]),
            ("kvb_section", i32, [p, p, i64, vp, vp, p, p, p, i64, vp, vp, p, i64]),
            ("kvb_cache_rows", i32, [p, p, i64, i64, i32, ctypes.POINTER(_CacheLayer), i32, ctypes.POINTER(_Rope),
                                     i32, i64, ctypes.POINTER(i32)]),
            ("kvb_rotate", i32, [p, vp, i64, i64, i64, ctypes.POINTER(_Rope), vp])):
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    if lib.kvb_version() != VERSION:
        raise OSError(f"{path}: libkvbridge version {lib.kvb_version()}, this backend speaks {VERSION} (rebuild it)")
    return lib


def library():
    """The loaded libkvbridge (OSError when it is missing or of another version)."""
    path = library_path()
    if not path.is_file():
        raise OSError(f"no libkvbridge at {path} (build it: experiments/universal_kv/ggml/README.md, or set "
                      f"{LIBRARY_ENV})")
    return _library(str(path))


def _error(lib, what):
    """The exception for the library's last failure on this thread."""
    message = f"{what}: {lib.kvb_last_error().decode(errors='replace')}"
    code = lib.kvb_last_error_code()
    if code == OUT_OF_MEMORY or "out of device memory" in message:
        return MemoryError(message)
    if "does not support" in message:
        return NotImplementedError(message)
    return RuntimeError(message)


def devices():
    """[{"index", "name", "description", "kind" (cpu|gpu|igpu|accel|other), "free", "total"}] ggml sees."""
    lib = library()
    kinds = ("cpu", "gpu", "igpu", "accel", "other")
    out = []
    for index in range(lib.kvb_device_count()):
        name, description = ctypes.create_string_buffer(128), ctypes.create_string_buffer(256)
        kind, free, total = ctypes.c_int(), ctypes.c_uint64(), ctypes.c_uint64()
        if lib.kvb_device_info(index, name, 128, description, 256, ctypes.byref(kind), ctypes.byref(free),
                               ctypes.byref(total)) == OK:
            out.append({"index": index, "name": name.value.decode(), "description": description.value.decode(),
                        "kind": kinds[min(kind.value, 4)], "free": int(free.value), "total": int(total.value)})
    return out


@functools.lru_cache(maxsize=None)
def _status(path, device, gpu):
    try:
        found = devices()
    except OSError as error:
        return False, str(error)
    if device not in ("", "auto"):
        chosen = [d for d in found if d["name"].lower() == device.lower() or str(d["index"]) == device]
        if not chosen:
            return False, f"no ggml device {device} (ggml sees {', '.join(d['name'] for d in found) or 'none'})"
        if gpu and chosen[0]["kind"] not in ("gpu", "igpu"):
            return False, f"{DEVICE_ENV}={device} is ggml's {chosen[0]['kind']} device, not a GPU"
        return True, f"{chosen[0]['name']}: {chosen[0]['description']}"
    gpus = [d for d in found if d["kind"] in ("gpu", "igpu")]
    if not gpus:
        return False, "ggml sees no Vulkan GPU"
    return True, f"{gpus[0]['name']}: {gpus[0]['description']}"


def status(*, gpu=True):
    """(available, detail): whether libkvbridge loads and its device ($KVPARK_GGML_DEVICE: a name or
    index, else the first GPU) exists; with gpu (backends.select for "auto" and "gpu"), only when that
    device is a GPU, not ggml's CPU backend."""
    return _status(str(library_path()), (os.environ.get(DEVICE_ENV) or "auto").strip(), bool(gpu))


# -- RoPE tables ---------------------------------------------------------------------------------

def _components(n_rot, sections, mode="imrope"):
    """The position component (0 t, 1 h, 2 w, 3 e) of each rotated pair (kvstate.rope_components:
    ggml_mrope_cache_init's interleaved IMROPE sectors or M-RoPE runs)."""
    if sections is None:
        return (0,) * (n_rot // 2)
    return kvstate.rope_components(n_rot // 2, sections, interleaved=mode == "imrope")


def rope_tables(rope, positions, *, inverse=False):
    """float32 (C, S) [n, n_rot/2] of a kvstate.Rope at positions [n] (text) or [n, 4] (M-RoPE components
    t, h, w, e): kvstate.Rope.cos_sin's values (angles and magnitude by its formulas, cos/sin in float64
    rounded to float32). The device rotates a = x0 * C - x1 * S, b = x1 * C + x0 * S: forward (C, S) =
    (cos, sin); inverse (cos, -sin) / (cos^2 + sin^2) in float64, kvstate.Rope.invert's division."""
    f32 = np.float32
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim == 1:
        extrap = kvstate.rope_angles(positions, rope.n_dims, rope.base, rope.sections,
                                     interleaved=rope.mode == "imrope")
    else:
        if rope.sections is None:
            raise NotImplementedError("M-RoPE positions need an IMROPE or M-RoPE rope")
        theta = positions.astype(f32)
        scale = kvstate.rope_theta_scale(rope.base, rope.n_dims)
        extrap = np.empty((len(theta), rope.n_dims // 2), dtype=f32)
        for index, component in enumerate(_components(rope.n_dims, tuple(int(s) for s in rope.sections), rope.mode)):
            extrap[:, index] = theta[:, component]
            theta = (theta * scale).astype(f32)
    if rope.freq_factors is not None:
        extrap = (extrap / np.asarray(rope.freq_factors, dtype=f32)).astype(f32)
    theta = (extrap * f32(rope.freq_scale)).astype(f32)
    if rope.ext_factor != 0:
        low, high = kvstate.yarn_corr_dims(rope.n_dims, rope.n_ctx_orig, rope.base, rope.beta_fast, rope.beta_slow)
        y = ((np.arange(rope.n_dims // 2, dtype=f32) - low) / max(f32(0.001), f32(high - low))).astype(f32)
        ramp = (f32(1) - np.minimum(f32(1), np.maximum(f32(0), y))).astype(f32)
        mix = (ramp * f32(rope.ext_factor)).astype(f32)
        theta = kvstate._fma32(theta, (f32(1) - mix).astype(f32), (extrap * mix).astype(f32))
    wide = theta.astype(np.float64)
    cos, sin = np.cos(wide).astype(f32), np.sin(wide).astype(f32)
    magnitude = rope.magnitude()
    if magnitude != 1:
        cos, sin = (cos * magnitude).astype(f32), (sin * magnitude).astype(f32)
    if not inverse:
        return np.ascontiguousarray(cos), np.ascontiguousarray(sin)
    c, s = cos.astype(np.float64), sin.astype(np.float64)
    norm = c * c + s * s
    return np.ascontiguousarray((c / norm).astype(f32)), np.ascontiguousarray((-s / norm).astype(f32))


class _Ropes:
    """The rope descriptors of one call: one entry (and its tables) per distinct kvstate.Rope."""

    def __init__(self, positions, inverse):
        self.positions, self.inverse = positions, inverse
        self.index, self.items, self.keep = {}, [], []

    def of(self, rope):
        """The index of a kvstate.Rope (-1 for None: NoPE)."""
        if rope is None:
            return -1
        if rope.mode not in ROPE_TYPES:
            raise NotImplementedError(f"no ggml RoPE for mode {rope.mode!r}")
        if rope not in self.index:
            cos, sin = rope_tables(rope, self.positions, inverse=self.inverse)
            self.keep += [cos, sin]
            self.index[rope] = len(self.items)
            self.items.append(_Rope(int(rope.mode != "norm"), int(rope.n_dims), int(rope.offset), 0,
                                    cos.ctypes.data, sin.ctypes.data))
        return self.index[rope]

    def array(self):
        return (_Rope * max(1, len(self.items)))(*self.items)


# -- device arrays and resident sides ------------------------------------------------------------

class _Tensor:
    """A device tensor of the library (weights, a section's rows, latents); freed with close() or
    when collected."""

    def __init__(self, backend, handle, what):
        if not handle:
            raise _error(backend.lib, what)
        self.backend, self.handle = backend, handle
        rows, cols, dtype, nbytes = ctypes.c_int64(), ctypes.c_int64(), ctypes.c_int(), ctypes.c_int64()
        backend.lib.kvb_shape(handle, ctypes.byref(rows), ctypes.byref(cols), ctypes.byref(dtype), ctypes.byref(nbytes))
        self.rows, self.cols, self.dtype, self.nbytes = rows.value, cols.value, dtype.value, nbytes.value

    @property
    def shape(self):
        return (self.rows, self.cols)

    def close(self):
        handle, self.handle = self.handle, None
        if handle and self.backend.ctx:
            self.backend.lib.kvb_tensor_free(self.backend.ctx, handle)

    def __del__(self):
        try:
            self.close()
        except Exception:                   # noqa: BLE001  (interpreter shutdown)
            pass


class DeviceArray:
    """A float32 [rows, cols] array on the device: hub latents (encode) or a section's rows (section).

    Held in row blocks of at most the device's largest buffer, each a library tensor: blocks
    [(first row, _Tensor)], one block unless the array is larger (none without rows)."""

    def __init__(self, backend, rows, cols):
        self.rows, self.cols = int(rows), int(cols)
        self.blocks = []
        per = backend.block_rows(self.cols)
        try:
            for row0 in range(0, self.rows, per):
                n = min(per, self.rows - row0)
                self.blocks.append((row0, _Tensor(backend, backend.lib.kvb_new(backend.ctx, F32, n, self.cols, 0),
                                                  "device array")))
        except BaseException:
            self.close()
            raise

    @property
    def shape(self):
        return (self.rows, self.cols)

    @property
    def nbytes(self):
        return _storage_bytes(tensor for _, tensor in self.blocks)

    def __len__(self):
        return self.rows

    def close(self):
        for _, tensor in self.blocks:
            tensor.close()


def _storage_bytes(tensors):
    return int(sum(t.nbytes for t in tensors if t is not None and t.handle))


class _Encoder:
    def __init__(self, layers, widths, means, bases, weight):
        self.layers = list(layers)
        self.widths = dict(zip(self.layers, widths))       # heads * dim of each source layer
        self.means = means                  # float32 [1, d_l] per layer
        self.bases = bases                  # float16 [r_l, d_l] per layer (transposed)
        self.weight = weight                # float16 [D, sum r_l] (the fused encoder, transposed)

    @property
    def dim(self):
        return self.weight.rows

    @property
    def nbytes(self):
        return _storage_bytes(self.means + self.bases + [self.weight])

    def close(self):
        for tensor in self.means + self.bases + [self.weight]:
            tensor.close()


class _Decoder:
    def __init__(self, backend, layers, z_mean, decoders, means):
        self.backend = backend
        self.layers = list(layers)
        self.z_mean = np.ascontiguousarray(z_mean, dtype=np.float32)      # float32 [D] (host)
        self.z_mean_device = backend.vector(self.z_mean)
        self.bias = backend.vector(-self.z_mean)            # what encode(center=...) adds
        self.decoders = dict(zip(self.layers, decoders))    # float16 [d_l, D] (transposed)
        self.means = {layer: np.ascontiguousarray(m, dtype=np.float32) for layer, m in zip(self.layers, means)}
        self.sections = {}
        self._lock = threading.Lock()

    @property
    def dim(self):
        return len(self.z_mean)

    @property
    def nbytes(self):
        # a snapshot: the engine daemon reads this while its worker may be building a section
        tensors = [self.z_mean_device, self.bias] + list(self.decoders.values())
        for weight, mean in list(self.sections.values()):
            tensors += [weight, mean]
        return _storage_bytes(tensors)

    def section(self, layers):
        """(float16 [sum d_l, D], float32 [1, sum d_l]) device tensors for one cache section, built once;
        when the sections cover every layer the per-layer upload is released."""
        key = tuple(int(layer) for layer in layers)
        with self._lock:
            if key not in self.sections:
                parts = [self._layer(layer) for layer in key]
                weight = self.backend.concat_rows(parts)
                mean = self.backend.vector(np.concatenate([self.means[layer] for layer in key]))
                self.sections[key] = (weight, mean)
                covered = {layer for k in self.sections for layer in k}
                if self.decoders and covered >= set(self.layers):
                    for tensor in self.decoders.values():
                        tensor.close()
                    self.decoders = {}
            return self.sections[key]

    def _layer(self, layer):
        """(tensor, first row, rows) of a layer's decoder: its upload, or its rows of a built section."""
        if layer in self.decoders:
            tensor = self.decoders[layer]
            return tensor, 0, tensor.rows
        for key, (weight, _) in self.sections.items():
            if layer in key:
                start = sum(len(self.means[other]) for other in key[:key.index(layer)])
                return weight, start, len(self.means[layer])
        raise KeyError(layer)

    def close(self):
        tensors = [self.z_mean_device, self.bias] + list(self.decoders.values())
        for weight, mean in self.sections.values():
            tensors += [weight, mean]
        for tensor in tensors:
            tensor.close()


class _OneSection:
    """A decoder of one section (weight, mean): GgmlBackend._check_products' product."""

    def __init__(self, weight, mean):
        self.pair = (weight, mean)

    def section(self, layers):
        return self.pair


class GgmlBackend(Backend):
    """NumpyBackend's interface on a ggml device (Vulkan, or ggml's CPU backend); see the module docstring.

    Beyond it: encode_rows (encode from given pre-RoPE rows, as hub.EncoderSide.encode),
    warm_products (the daemon's warm-up), rotate (one RoPE on host rows), memory / reset_peak."""

    name = "ggml"
    # the device-to-host copies of cache_rows would first-touch the output pages on one thread
    prefault_output = True

    def __init__(self, device=None, threads=None):
        """device: "auto" (the first GPU, else ggml's CPU backend), "cpu", a ggml device name such as
        "Vulkan1" or a device index (devices()); default $KVPARK_GGML_DEVICE, else "auto"."""
        super().__init__()
        self.lib = library()
        self.ctx = None
        want = (device or os.environ.get(DEVICE_ENV) or "auto").strip()
        ctx = self.lib.kvb_init(want.encode(), int(threads or 0))
        if not ctx:
            raise _error(self.lib, f"ggml device {want}")
        self.ctx = ctx
        name, description = ctypes.create_string_buffer(128), ctypes.create_string_buffer(256)
        kind, largest = ctypes.c_int(), ctypes.c_uint64()
        self.lib.kvb_info(ctx, name, 128, description, 256, ctypes.byref(kind), ctypes.byref(largest))
        self.device = name.value.decode()
        self.description = description.value.decode()
        self.kind = kind.value
        self.max_buffer = int(largest.value)       # bytes of the largest device tensor (DeviceArray blocks)
        self._checked = {}
        self._products = None                       # the product check's error (None: not run yet)

    def close(self):
        """Drop every side and free the device context (tests; a backend otherwise lives as long as the
        process). Device arrays still referenced elsewhere are freed with it and must not be used."""
        self.drop()
        ctx, self.ctx = self.ctx, None
        if ctx:
            self.lib.kvb_free(ctx)

    @property
    def on_gpu(self):
        return self.kind in (GPU, IGPU)

    def describe(self):
        return {"backend": self.name, "device": self.device, "gpu" if self.on_gpu else "cpu": self.description,
                "ggml": self.lib.kvb_ggml_version().decode(), "library": str(library_path()),
                "compute": "float16 weights/activations, float32 accumulation (GGML_PREC_F32)"}

    def memory(self):
        """Bytes: allocated (weights and live arrays), reserved (plus pooled buffers and the graph
        allocator's), peak_allocated (the peak of reserved since reset_peak), device_free / total,
        read_staging (pinned host memory reads go through on an integrated GPU)."""
        out = (ctypes.c_int64 * 8)()
        self.lib.kvb_memory(self.ctx, out)
        weights, arrays, pooled, graph, peak, free, total, staging = (int(v) for v in out)
        return {"allocated": weights + arrays, "reserved": weights + arrays + pooled + graph, "peak_allocated": peak,
                "weights": weights, "arrays": arrays, "pooled": pooled, "graph": graph, "device_free": free,
                "device_total": total, "read_staging": staging}

    def reset_peak(self):
        self.lib.kvb_reset_peak(self.ctx)

    def _released(self):
        self.lib.kvb_release(self.ctx)

    def synchronize(self):
        if self.lib.kvb_synchronize(self.ctx) != OK:
            raise _error(self.lib, "synchronize")

    def _call(self, code, what):
        if code != OK:
            raise _error(self.lib, what)

    # -- uploads ---------------------------------------------------------------------------------

    @staticmethod
    def _host_matrix(array, transposed):
        """(C-contiguous float16/float32 array, transpose on the device) for matrix(): a Fortran-ordered
        array (format 1 npz) is sent as its C-ordered transpose, already the stored layout."""
        a = np.asarray(array)
        if a.dtype not in (np.float16, np.float32):
            a = a.astype(np.float32)
        if a.ndim != 2:
            raise ValueError("a matrix must be 2-D")
        if transposed and a.flags.f_contiguous and not a.flags.c_contiguous:
            return np.ascontiguousarray(a.T), False
        return np.ascontiguousarray(a), transposed

    def matrix(self, array, *, transposed=False, weight=True, dtype="f16"):
        """A host matrix on the device as dtype: [rows, cols], or with transposed its transpose
        (ggml_mul_mat's operand layout for x @ array)."""
        a, transpose = self._host_matrix(array, transposed)
        source = F16 if a.dtype == np.float16 else F32
        handle = self.lib.kvb_upload(self.ctx, a.ctypes.data, source, a.shape[0], a.shape[1], DTYPES[dtype],
                                     int(transpose), int(weight))
        return _Tensor(self, handle, "upload")

    def matrices(self, arrays, *, transposed=False, weight=True, dtype="f16"):
        """[matrix(a) for a in arrays], uploaded together (one device graph per 256 MiB, not one per matrix)."""
        prepared = [self._host_matrix(array, transposed) for array in arrays]
        out = [None] * len(prepared)
        try:
            for flag in sorted({transpose for _, transpose in prepared}):
                chosen = [i for i, (_, transpose) in enumerate(prepared) if transpose == flag]
                n = len(chosen)
                hosts = [prepared[i][0] for i in chosen]
                handles = (ctypes.c_void_p * n)()
                code = self.lib.kvb_upload_many(
                    self.ctx, n, (ctypes.c_void_p * n)(*[a.ctypes.data for a in hosts]),
                    (ctypes.c_int * n)(*[F16 if a.dtype == np.float16 else F32 for a in hosts]),
                    (ctypes.c_int64 * n)(*[a.shape[0] for a in hosts]), (ctypes.c_int64 * n)(*[a.shape[1] for a in hosts]),
                    DTYPES[dtype], int(flag), int(weight), handles)
                if code != OK:
                    raise _error(self.lib, "upload")
                for i, handle in zip(chosen, handles):
                    out[i] = _Tensor(self, handle, "upload")
        except BaseException:
            for tensor in out:
                if tensor is not None:
                    tensor.close()
            raise
        return out

    def vector(self, values, *, weight=True):
        """A float32 [1, n] device tensor."""
        values = np.ascontiguousarray(values, dtype=np.float32).reshape(1, -1)
        return self.matrix(values, weight=weight, dtype="f32")

    def concat_rows(self, parts):
        """[sum rows, cols]: parts [(tensor, first row, rows)] one after the other, on the device."""
        handles = (ctypes.c_void_p * len(parts))(*[t.handle for t, _, _ in parts])
        first = (ctypes.c_int64 * len(parts))(*[r for _, r, _ in parts])
        rows = (ctypes.c_int64 * len(parts))(*[n for _, _, n in parts])
        return _Tensor(self, self.lib.kvb_concat_rows(self.ctx, len(parts), handles, first, rows, 1), "concat")

    def block_rows(self, cols):
        """Rows of a float32 DeviceArray block `cols` wide: as many as fit the largest buffer."""
        return max(1, self.max_buffer // (4 * max(1, int(cols))))

    def _chunk(self, chunk_rows, row_bytes):
        """Rows per library graph (at most chunk_rows) whose tensors of row_bytes a row stay within
        GRAPH_BYTES and the largest buffer."""
        return max(1, min(int(chunk_rows), min(GRAPH_BYTES, self.max_buffer) // max(1, int(row_bytes))))

    def array(self, rows, cols):
        """An uninitialized float32 DeviceArray [rows, cols]."""
        return DeviceArray(self, rows, cols)

    def from_host(self, values):
        """A host float32 [rows, cols] array as a DeviceArray."""
        values = np.ascontiguousarray(values, dtype=np.float32)
        out = self.array(*values.shape)
        self._write(out, values)
        return out

    def _write(self, x, values):
        """Host float32 C-contiguous [rows, cols] values into DeviceArray x."""
        for row0, tensor in x.blocks:
            self._call(self.lib.kvb_write(self.ctx, tensor.handle, 0, tensor.rows, values[row0:].ctypes.data), "write")

    def host(self, x):
        if not isinstance(x, DeviceArray):
            return np.asarray(x)
        out = np.empty(x.shape, dtype=np.float32)
        for row0, tensor in x.blocks:
            self._call(self.lib.kvb_read(self.ctx, tensor.handle, 0, tensor.rows, out[row0:].ctypes.data), "read")
        return out

    # -- sides -----------------------------------------------------------------------------------

    def _load(self, hub, kind, name, role):
        # head kinds (DeepSeek V4 compressor inputs) keep format 1 decoders, and encode the base kind
        if (isinstance(hub, hub_module.Hub) and hub.version == hub_module.FORMAT_V2
                and kind not in hub.manifest.get("heads", {})):
            return self._load_packed(hub, kind, name, role)
        side = (hub.side(kind, name, role) if isinstance(hub, hub_module.Hub)
                else hub[kind].encoder_side(name) if role == "encoder" else hub[kind].decoder_side(name))
        if role == "encoder":
            return self._encoder(side.layers, side.means, side.bases, side.weight)
        return self._decoder(side.layers, side.z_mean, side.decoders, side.means)

    def _encoder(self, layers, means, bases, weight):
        # a failed upload leaves nothing behind: the tensors made so far are freed when collected
        return _Encoder(layers, [np.shape(b)[0] for b in bases], [self.vector(mean) for mean in means],
                        self.matrices(bases, transposed=True), self.matrix(weight, transposed=True))

    def _decoder(self, layers, z_mean, decoders, means):
        return _Decoder(self, layers, z_mean, self.matrices(decoders, transposed=True), means)

    def _load_packed(self, hub, kind, name, role):
        """A format 2 side: matrices from its float16 file (else float32), means and z_mean in float32."""
        entry, small = hub.side_arrays(kind, name, role, mmap=True)
        _, flat = hub.side_flat(kind, name, role, dtype="f16" if "f16" in entry["files"] else "f32", mmap=True)
        table, layers = entry["arrays"], entry["layers"]

        def view(key):
            offset, shape = table[key]
            return flat[offset:offset + int(np.prod(shape))].reshape(shape)

        if role == "decoder":
            return self._decoder(layers, np.array(small["z_mean"]), [view(f"decoder_{l}") for l in layers],
                                 [np.array(small[f"decoder_mean_{l}"]) for l in layers])
        return self._encoder(layers, [np.array(small[f"mean_{l}"]) for l in layers],
                             [view(f"basis_{l}") for l in layers], view("weight"))

    # -- RoPE ------------------------------------------------------------------------------------

    def supports(self, spec):
        """Refuse (NotImplementedError) a device whose products disagree with numpy, or a spec whose
        RoPE disagrees with kvstate on the device."""
        self._check_products()
        ropes = {}
        for layer in spec.attention_layers:
            ropes.setdefault(spec.rope(layer), layer)
        for rope, layer in ropes.items():
            self._check_rope(spec, layer, rope)

    def _check_rope(self, spec, layer, rope):
        key = (spec, rope)
        if key in self._checked or rope is None:
            return
        if rope.mode not in ROPE_TYPES:
            raise NotImplementedError(f"no ggml RoPE for mode {rope.mode!r}")
        heads, dim = spec.head_shape(layer, "k")
        x = np.random.default_rng(0).standard_normal((5, heads, dim)).astype(np.float32) * 3
        positions = np.asarray([0, 1, 777, 32771, 262143], dtype=np.int64)
        for inverse, reference in ((False, kvstate.cache_keys), (True, kvstate.pre_rope_keys)):
            expected = reference(spec, layer, x, positions)
            error = float(np.max(np.abs(self.rotate(x, positions, rope, inverse=inverse) - expected)))
            if error > 1e-5 * float(np.max(np.abs(x))):
                raise NotImplementedError(f"ggml RoPE of layer {layer} disagrees with kvstate ({error:.3g})")
        self._checked[key] = True

    def _check_products(self):
        """One decode product (float16 inputs, float32 accumulation) of row counts with partial tiles
        against numpy in float64; NotImplementedError when it is off by more than float32 rounding
        allows (the inputs are float16 values: only the accumulation differs)."""
        if self._products is None:
            rng = np.random.default_rng(0)
            k, m = 160, 200
            weight = rng.standard_normal((k, m)).astype(np.float16)
            tensor = self.matrix(weight, transposed=True, weight=False)
            mean = self.vector(np.zeros(m), weight=False)
            worst = 0.0
            for n in (1, 9, 65, 300):
                z = rng.standard_normal((n, k)).astype(np.float16).astype(np.float32)
                rows = np.arange(n)
                out = self.section(_OneSection(tensor, mean), z, (), n, rows, rows, rows[:0], centered=True)
                reference = z.astype(np.float64) @ weight.astype(np.float64)
                worst = max(worst, float(np.abs(self.host(out) - reference).max() / np.abs(reference).max()))
            self._products = (f"ggml products on {self.device} ({self.description}) disagree with numpy "
                              f"({worst:.3g} of the largest value)") if worst > 1e-4 else ""
        if self._products:
            raise NotImplementedError(self._products)

    def rotate(self, x, positions, rope, *, inverse):
        """Host float32 x [n, heads, dim] rotated by a kvstate.Rope on the device (a new array)."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty_like(x)
        cos, sin = rope_tables(rope, positions, inverse=inverse)
        item = _Rope(int(rope.mode != "norm"), int(rope.n_dims), int(rope.offset), 0, cos.ctypes.data,
                     sin.ctypes.data)
        self._call(self.lib.kvb_rotate(self.ctx, x.ctypes.data, x.shape[0], x.shape[1], x.shape[2],
                                       ctypes.byref(item), out.ctypes.data), "rotate")
        return out

    # -- compute ---------------------------------------------------------------------------------

    @staticmethod
    def _rows(array):
        """A host row block as the library reads it: float16 or float32, C-contiguous [n, width]."""
        a = np.asarray(array)
        if a.dtype == kvstate.Q8_0_BLOCK:
            a = kvstate.dequantize_q8_0(a)
        a = a.reshape(len(a), -1)
        if a.dtype not in (np.float16, np.float32):
            a = a.astype(np.float32)
        return np.ascontiguousarray(a)

    def _encode(self, encoder, n, layers, ropes, center, out, row0):
        """kvb_encode of one chunk into rows [row0, row0 + n) of a library tensor `out`: layers [(rows,
        index, heads, dim, rope index)] in encoder.layers order."""
        items = []
        for layer, (rows, index, heads, dim, rope) in zip(encoder.layers, layers):
            if rows.shape[1] != encoder.widths[layer]:
                raise ValueError(f"source layer {layer} rows are {rows.shape[1]} wide, the hub's {encoder.widths[layer]}")
            items.append(_EncodeLayer(rows.ctypes.data, F16 if rows.dtype == np.float16 else F32, rope, len(rows),
                                      index.ctypes.data, heads, dim, encoder.means[len(items)].handle,
                                      encoder.bases[len(items)].handle))
        array = (_EncodeLayer * len(items))(*items)
        bias = center.bias.handle if center is not None else None
        code = self.lib.kvb_encode(self.ctx, n, len(items), array, len(ropes.items), ropes.array(),
                                   encoder.weight.handle, bias, out.handle, row0)
        if code == NOT_FINITE:
            # the centred rows or the stage one overflowed float16: the engine retries on numpy
            raise FloatingPointError(f"non-finite hub latents on {self.device} (float16 overflow)")
        self._call(code, "encode")

    @staticmethod
    def _chunks(array, chunk_rows):
        """(tensor, first row in it, first row of the array, rows) of a DeviceArray, chunk_rows at most
        and none across two blocks."""
        for row0, tensor in array.blocks:
            for start in range(0, tensor.rows, chunk_rows):
                yield tensor, start, row0 + start, min(chunk_rows, tensor.rows - start)

    def encode(self, encoder, cache, kind, positions, chunk_rows=4096, center=None):
        """Hub latents, a float32 DeviceArray [n, D]; with center (a decoder) z - center.z_mean (added as
        the product's bias). Source rows are uploaded per chunk of positions (the span of cells they
        occupy, from the slot file's arrays), gathered and de-rotated on the device."""
        positions = np.asarray(positions, dtype=np.int64)
        spec = cache.spec
        if kind == "k":
            for layer in encoder.layers:
                self._check_rope(spec, layer, spec.rope(layer))
        out = self.array(len(positions), encoder.dim)
        cells_of = {}
        for tensor, row0, start, count in self._chunks(out, chunk_rows):
            part = positions[start:start + count]
            ropes = _Ropes(part, inverse=True)
            spans, layers = {}, []
            for layer in encoder.layers:
                stream, index = cache.where[layer]
                if id(stream) not in cells_of:
                    cells_of[id(stream)] = handoff._cells(index, positions, f"source layer {layer}")
                if id(stream) not in spans:            # one span and gather index per stream and chunk
                    cells = cells_of[id(stream)][start:start + len(part)]
                    lo, hi = int(cells.min()), int(cells.max()) + 1
                    spans[id(stream)] = (lo, hi), np.ascontiguousarray(cells - lo, dtype=np.int32)
                (lo, hi), local = spans[id(stream)]
                rows = self._rows(stored_rows(spec, stream, kind, layer)[lo:hi])
                heads, dim = spec.head_shape(layer, "k") if kind == "k" else (1, rows.shape[1])
                layers.append((rows, local, heads, dim, ropes.of(spec.rope(layer)) if kind == "k" else -1))
            self._encode(encoder, len(part), layers, ropes, center, tensor, row0)
        return out

    def encode_rows(self, encoder, rows_by_layer, center=None):
        """Hub latents (DeviceArray [n, D]) of given pre-RoPE rows {layer: [n, width]} (hub.EncoderSide.encode
        on the device; with center, z - center.z_mean)."""
        blocks = [self._rows(rows_by_layer[layer]) for layer in encoder.layers]
        out = self.array(len(blocks[0]), encoder.dim)
        for tensor, row0, start, count in self._chunks(out, out.rows):
            index = np.arange(count, dtype=np.int32)
            self._encode(encoder, count, [(np.ascontiguousarray(b[start:start + count]), index, 1, b.shape[1], -1)
                                          for b in blocks], _Ropes(np.zeros(0), True), center, tensor, row0)
        return out

    def section(self, decoder, z, layers, n_rows, mapped, mapped_rows, fill_rows, fill=None, chunk_rows=4096,
                centered=False):
        """As NumpyBackend.section, a float32 DeviceArray [n_rows, sum d_l]; centered: z is already
        z - decoder.z_mean (encode(center=decoder)). One library call per block of the output (and of
        z) the rows fall in: one call unless an array exceeds the largest buffer."""
        weight, mean = decoder.section(layers)
        out = self.array(int(n_rows), weight.rows)
        mapped = np.asarray(mapped, dtype=np.int64)
        mapped_rows = np.asarray(mapped_rows, dtype=np.int64)
        fill_rows = np.asarray(fill_rows, dtype=np.int64)
        if len(mapped) and not isinstance(z, DeviceArray):
            z = self.from_host(z)                   # latents computed elsewhere (tests, numpy's)
        n_z = len(z) if len(mapped) else 0
        if (len(mapped) != len(mapped_rows) or (len(mapped) and (mapped.min() < 0 or mapped.max() >= n_z
                                                                 or mapped_rows.min() < 0
                                                                 or mapped_rows.max() >= out.rows))
                or (len(fill_rows) and (fill_rows.min() < 0 or fill_rows.max() >= out.rows))):
            raise IndexError(f"section: mapped or fill rows outside {n_z} latents / {out.rows} rows")
        if fill is not None:
            fill = np.broadcast_to(np.asarray(fill, dtype=np.float32), (len(fill_rows), weight.rows))
        z_mean = None if centered else decoder.z_mean_device.handle
        for row0, target in out.blocks:
            here = (mapped_rows >= row0) & (mapped_rows < row0 + target.rows)
            calls = []
            for start, source in (z.blocks if len(mapped) else ()):
                take = here & (mapped >= start) & (mapped < start + source.rows)
                if take.any():
                    calls.append((source, mapped[take] - start, mapped_rows[take] - row0))
            filled = (fill_rows >= row0) & (fill_rows < row0 + target.rows)
            if not calls and filled.any():
                calls = [(None, mapped[:0], mapped_rows[:0])]
            for index, (source, rows, at) in enumerate(calls):
                fills = fill_rows[filled] - row0 if index == 0 else fill_rows[:0]
                values = None if fill is None or not len(fills) else fill[filled]
                self._section(source, rows, at, z_mean, weight, mean, fills, values, target, chunk_rows)
        return out

    def _section(self, z, mapped, mapped_rows, z_mean, weight, mean, fill_rows, fill, out, chunk_rows):
        """kvb_section into one block `out` (library tensors; host index arrays, local to z and out)."""
        keep = []

        def indices(values, identity_ok=True):
            if identity_ok and np.array_equal(values, np.arange(len(values))):
                return None                     # 0 .. m - 1: views instead of gathers
            values = np.ascontiguousarray(values, dtype=np.int32)
            keep.append(values)
            return values.ctypes.data

        m = len(mapped)
        source = indices(mapped) if m else None
        target = indices(mapped_rows) if m else None
        fills = indices(fill_rows, identity_ok=False) if len(fill_rows) else None
        if fill is not None:
            fill = np.ascontiguousarray(fill, dtype=np.float32)
        chunk = self._chunk(chunk_rows, 4 * max(weight.rows, weight.cols))     # a chunk's product [rows, width]
        self._call(self.lib.kvb_section(self.ctx, z.handle if m else None, m, source, target, z_mean, weight.handle,
                                        mean.handle, len(fill_rows), fills, None if fill is None else fill.ctypes.data,
                                        out.handle, chunk), "section")

    def renormalize_latents(self, spec, layers, values, norms):
        """As NumpyBackend.renormalize_latents: the MLA latent columns read back, put on their shell by
        kvstate.mla_renormalize_latent, written back to the device."""
        columns = latent_columns(spec, layers, norms)
        if not columns:
            return values
        host = self.host(values)
        for layer, start, lora in columns:
            weight, eps = norms[layer]
            host[:, start:start + lora] = kvstate.mla_renormalize_latent(host[:, start:start + lora], weight, eps)
        if isinstance(values, DeviceArray):
            self._write(values, host)
            return values
        return host

    def cache_rows(self, spec, layers, kind, values, positions, dtype, out=None, chunk_rows=4096):
        """As NumpyBackend.cache_rows: K rotated at positions (in place in values where the rope covers
        part of a head), cast to the cache dtype and checked finite there on the device, each layer
        copied into its host array (out: {layer: C-contiguous array [n, heads, dim]}, else new arrays)."""
        positions = np.asarray(positions, dtype=np.int64)
        n = len(positions)
        np_dtype = kvstate.GGML_DTYPES[kvstate.DTYPE_NAMES[dtype]]
        result, shapes, ropes = {}, {}, {}
        for layer in layers:
            shape = spec.head_shape(layer, kind)
            target = out[layer] if out is not None else np.empty((n,) + shape, dtype=np_dtype)
            if target.dtype != np_dtype:
                raise ValueError(f"{kind.upper()} layer {layer}: the output array is {target.dtype}, not {dtype}")
            if not target.flags.c_contiguous or target.shape != (n,) + shape:
                raise ValueError(f"layer {layer}: cache_rows writes into C-contiguous [{n}, {shape[0]}, {shape[1]}] "
                                 "host arrays only")
            ropes[layer] = None
            if kind == "k":
                ropes[layer] = spec.rope(layer)
                self._check_rope(spec, layer, ropes[layer])
            result[layer], shapes[layer] = target, shape
        if not n:
            return result
        if not isinstance(values, DeviceArray):
            values = self.from_host(np.asarray(values, dtype=np.float32).reshape(n, -1))
        # a chunk's stored rows (the library also keeps each graph within the largest buffer)
        width = sum(heads * dim for heads, dim in shapes.values())
        chunk = self._chunk(chunk_rows, width * np.dtype(np_dtype).itemsize)
        for row0, tensor in values.blocks:
            rows = tensor.rows
            tables = _Ropes(positions[row0:row0 + rows], inverse=False)
            items = [_CacheLayer(shapes[layer][0], shapes[layer][1], tables.of(ropes[layer]), 0,
                                 result[layer][row0:].ctypes.data) for layer in layers]
            bad = ctypes.c_int(-1)
            code = self.lib.kvb_cache_rows(self.ctx, tensor.handle, 0, rows, len(items),
                                           (_CacheLayer * len(items))(*items), len(tables.items), tables.array(),
                                           DTYPES[dtype], chunk, ctypes.byref(bad))
            if code == NOT_FINITE:
                # float16 can overflow where float32 does not: the engine retries on numpy
                raise FloatingPointError(f"{kind.upper()} rows of layer {list(layers)[bad.value]} are not finite in "
                                         f"{dtype} on {self.device}")
            self._call(code, "cache rows")
        return result

    def warm_products(self, encoder, decoder, rows, sections, *, spec=None, kind="k"):
        """A switch's products on `rows` zero rows (engine_daemon.warm_products): encode centred on the
        decoder, each section's decode and, with the target's spec, its RoPE and float16 cast."""
        zeros = {layer: np.zeros((rows, width), dtype=np.float16) for layer, width in encoder.widths.items()}
        z = self.encode_rows(encoder, zeros, center=decoder)
        everything = np.arange(rows, dtype=np.int64)
        for layers in sections:
            values = self.section(decoder, z, layers, rows, everything, everything, np.zeros(0, dtype=np.int64),
                                  centered=True)
            if spec is not None:
                self.cache_rows(spec, layers, kind, values, everything, "f16")
