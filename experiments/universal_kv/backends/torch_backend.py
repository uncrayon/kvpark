"""GPU backend: torch with float16 weights and activations and float32 accumulation.

Matrix products run in float16 with float32 accumulation (hipBLASLt/cuBLAS;
allow_fp16_reduced_precision_reduction is switched off; bfloat16 is never used: 8x the
error). Measured on the Radeon 8060S the result sits at the float16 input-rounding floor
(relative error a few 1e-4 against the float32 numpy reference), 25-35 TFLOPS.

Arrays reach the device C-contiguous: the format 1 npz arrays are Fortran-ordered, and a
strided upload made both a 2.9 s product and a 3.5 s copy; hub bundles of format 2 are
packed C-contiguous (float16 copies are uploaded as they are).

RoPE is computed on the device from positions, never from precomputed tables (production
contexts reach 262K positions). It mirrors kvstate's Rope (spec.rope(layer)): angles by the
same float32 recurrence as ggml-cpu, frequency factors (llama3's, gemma4's proportional RoPE),
freq_scale and the YaRN mix with kvstate's float32 formulas, cos/sin in float64 rounded to
float32, the magnitude, the forward rotation with the
CPU build's fused multiply-adds in float32 (torch.addcmul, which kvstate emulates in float64)
and the inverse in float64, so keys agree with kvstate.cache_keys / pre_rope_keys to float32
rounding (MPS, without float64, computes in float32). NoPE layers (Muse's global layers, the
kimi MLA rows) are not rotated on either side; an MLA row rotates only its k_pe tail (the Rope
offset). Positions may be [n] (text) or [n, 4] (M-RoPE components t, h, w, e, for image
cells). Every rotation is
checked against kvstate on first use; one that disagrees (a RoPE variant kvstate learns later)
is refused with NotImplementedError, and the engine falls back to numpy.

MLA latents are put back on their RMS shell on the device (renormalize_latents: kvstate.
mla_renormalize_latent's arithmetic in float64), and an MLA source's V rows are the c_kv view
of its K rows.
"""

from dataclasses import dataclass
import functools
import warnings

import numpy as np
import torch

from .base import Backend, hub_module, kvstate, latent_columns, stored_rows

import handoff  # noqa: E402  (on sys.path through .base)

ROPE_TYPES = ("norm", "neox", "imrope", "mrope")
TORCH_DTYPES = {"f16": torch.float16, "f32": torch.float32}

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")


def default_device():
    """"cuda" (ROCm or CUDA), "mps" or None."""
    try:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:                       # broken driver installs raise from the probes
        pass
    return None


# -- RoPE ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Rope:
    """kvstate.Rope as the device evaluates it."""

    neox: bool                  # NEOX/IMROPE/M-RoPE pairs (i, i + n/2); else NORMAL pairs (2i, 2i + 1)
    n_rot: int
    offset: int                 # the pairs live in [offset, offset + n_rot) of the row
    theta_scale: float          # float32 powf(base, -2/n_rot), as kvstate.rope_theta_scale
    components: tuple           # per pair: position component 0 t, 1 h, 2 w, 3 e (IMROPE/M-RoPE sectors)
    freq_scale: float = 1.0
    mix: tuple = None           # YaRN: per pair float32 ramp * ext_factor; None without YaRN
    magnitude: float = 1.0      # float32 factor on cos/sin
    factors: tuple = None       # per pair frequency factor the angle is divided by (rope_freqs.weight)


def _components(n_rot, sections, mode="imrope"):
    """kvstate.rope_components of a rope's pairs (ggml_mrope_cache_init: interleaved IMROPE sectors,
    M-RoPE runs); every pair takes the token position without sections."""
    if sections is None:
        return (0,) * (n_rot // 2)
    return kvstate.rope_components(n_rot // 2, sections, interleaved=mode == "imrope")


@functools.lru_cache(maxsize=None)
def device_rope(rope):
    """The device form of a kvstate.Rope (None for NoPE)."""
    if rope is None:
        return None
    if rope.mode not in ROPE_TYPES:
        raise NotImplementedError(f"no device RoPE for mode {rope.mode!r}")
    f32 = np.float32
    mix = None
    if rope.ext_factor != 0:
        # Rope.cos_sin's ramp: 1 - clamp((i - low) / max(0.001, high - low), 0, 1), times ext_factor
        low, high = kvstate.yarn_corr_dims(rope.n_dims, rope.n_ctx_orig, rope.base, rope.beta_fast, rope.beta_slow)
        y = ((np.arange(rope.n_dims // 2, dtype=f32) - low) / max(f32(0.001), f32(high - low))).astype(f32)
        ramp = (f32(1) - np.minimum(f32(1), np.maximum(f32(0), y))).astype(f32)
        mix = tuple(float(m) for m in (ramp * f32(rope.ext_factor)).astype(f32))
    sections = None if rope.sections is None else tuple(int(x) for x in rope.sections)
    factors = None if rope.freq_factors is None else tuple(float(f32(x)) for x in rope.freq_factors)
    return Rope(rope.mode != "norm", int(rope.n_dims), int(rope.offset),
                float(kvstate.rope_theta_scale(rope.base, rope.n_dims)), _components(rope.n_dims, sections, rope.mode),
                float(f32(rope.freq_scale)), mix, float(rope.magnitude()), factors)


def layer_rope(spec, layer):
    """The device rotation kvstate.cache_keys applies to a layer's (attention) keys, or None (NoPE)."""
    if spec.layer_types[layer] == "recurrent":
        raise ValueError(f"layer {layer} is recurrent")
    return device_rope(spec.rope(layer))


# -- resident sides ------------------------------------------------------------------------------

def _storage_bytes(tensors):
    seen, total = set(), 0
    for t in tensors:
        if t is None:
            continue
        key = t.untyped_storage().data_ptr()
        if key not in seen:
            seen.add(key)
            total += t.untyped_storage().nbytes()
    return total


class _Encoder:
    def __init__(self, layers, means, bases, weight):
        self.layers = list(layers)
        self.means = means                  # float32 [L, d] (uniform) or list of [d_l]
        self.bases = bases                  # float16 [L, d, r] (uniform) or list of [d_l, r_l]
        self.weight = weight                # float16 [sum r_l, D]
        self.stacked = isinstance(bases, torch.Tensor)

    @property
    def dim(self):
        return self.weight.shape[1]

    @property
    def nbytes(self):
        tensors = [self.weight] + ([self.means, self.bases] if self.stacked else list(self.means) + list(self.bases))
        return _storage_bytes(tensors)


class _Decoder:
    def __init__(self, layers, z_mean, decoders, means):
        self.layers = list(layers)
        self.z_mean = z_mean                # float32 [D]
        # the encode product's bias (z - z_mean), rounded to float16 as addmm takes it: |z_mean| is
        # small next to z (at most 0.105 in hub4-d4096, a rounding of 3e-5)
        self.z_bias = (-z_mean).to(torch.float16)
        self.decoders = dict(zip(self.layers, decoders))     # float16 [D, d_l] (views of one upload)
        self.means = dict(zip(self.layers, means))           # float32 [d_l]
        self.sections = {}

    @property
    def nbytes(self):
        # snapshots: the engine daemon reads this while its worker may be building a section
        tensors = [self.z_mean] + list(self.decoders.values()) + list(self.means.values())
        for weight, mean in list(self.sections.values()):
            tensors += [weight, mean]
        return _storage_bytes(tensors)

    def section(self, layers):
        """(float16 [D, sum d_l], float32 [sum d_l]) for one cache section, built once; when the
        sections cover every layer the per-layer upload is released."""
        key = tuple(int(layer) for layer in layers)
        if key not in self.sections:
            self.sections[key] = (torch.cat([self._layer(layer) for layer in key], dim=1).contiguous(),
                                  torch.cat([self.means[layer] for layer in key]).contiguous())
            covered = {layer for k in self.sections for layer in k}
            if self.decoders and covered >= set(self.layers):
                self.decoders = {}
        return self.sections[key]

    def _layer(self, layer):
        if layer in self.decoders:
            return self.decoders[layer]
        for key, (weight, _) in self.sections.items():          # released: slice a built section
            if layer in key:
                start = sum(len(self.means[other]) for other in key[:key.index(layer)])
                return weight[:, start:start + len(self.means[layer])]
        raise KeyError(layer)


class TorchBackend(Backend):
    """NumpyBackend's interface on a torch device; see the module docstring."""

    name = "torch"
    # the device-to-host copies of cache_rows would first-touch the output pages on one thread
    prefault_output = True

    def __init__(self, device=None):
        super().__init__()
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        self.device = torch.device(device or default_device() or "cpu")
        self.half = torch.float16
        self.wide = torch.float32 if self.device.type == "mps" else torch.float64
        self._checked = {}
        self._weights = {}                      # id(latent norm weight) -> (the array, its device copy)

    def describe(self):
        info = {"backend": self.name, "device": str(self.device), "torch": torch.__version__,
                "compute": "float16 weights/activations, float32 accumulation"}
        if self.device.type == "cuda":
            info["gpu"] = torch.cuda.get_device_name(self.device)
            info["runtime"] = f"hip {torch.version.hip}" if torch.version.hip else f"cuda {torch.version.cuda}"
        return info

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def memory(self):
        if self.device.type != "cuda":
            return {}
        free, total = torch.cuda.mem_get_info(self.device)
        return {"allocated": int(torch.cuda.memory_allocated(self.device)),
                "reserved": int(torch.cuda.memory_reserved(self.device)),
                "peak_allocated": int(torch.cuda.max_memory_allocated(self.device)),
                "device_free": int(free), "device_total": int(total)}

    def reset_peak(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _released(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

    def _upload(self, array, dtype=None):
        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(self.device)
        return tensor if dtype is None else tensor.to(dtype)

    # -- sides ---------------------------------------------------------------------------------

    def _load(self, hub, kind, name, role):
        # head kinds (DeepSeek V4 compressor inputs) keep format 1 decoders, and encode the base kind
        if (isinstance(hub, hub_module.Hub) and hub.version == hub_module.FORMAT_V2
                and kind not in hub.manifest.get("heads", {})):
            return self._load_packed(hub, kind, name, role)
        side = (hub.side(kind, name, role) if isinstance(hub, hub_module.Hub)
                else hub[kind].encoder_side(name) if role == "encoder" else hub[kind].decoder_side(name))
        if role == "encoder":
            means = [np.asarray(m, dtype=np.float32) for m in side.means]
            bases = [np.ascontiguousarray(b, dtype=np.float32) for b in side.bases]
            weight = self._upload(side.weight, self.half)
            if len({b.shape for b in bases}) == 1:
                return _Encoder(side.layers, self._upload(np.stack(means)), self._upload(np.stack(bases), self.half),
                                weight)
            return _Encoder(side.layers, [self._upload(m) for m in means],
                            [self._upload(b, self.half) for b in bases], weight)
        weights = self._upload(np.concatenate(side.decoders, axis=1), self.half)
        widths = np.cumsum([0] + [d.shape[1] for d in side.decoders])
        return _Decoder(side.layers, self._upload(side.z_mean),
                        [weights[:, widths[i]:widths[i + 1]] for i in range(len(side.layers))],
                        [self._upload(m) for m in side.means])

    def _load_packed(self, hub, kind, name, role):
        """A format 2 side: the float16 file (or float32) in one upload; means and z_mean in float32."""
        entry, small = hub.side_arrays(kind, name, role, mmap=True)
        _, flat = hub.side_flat(kind, name, role, dtype="f16" if "f16" in entry["files"] else "f32")
        flat = self._upload(flat, self.half)
        table, layers = entry["arrays"], entry["layers"]

        def view(key):
            offset, shape = table[key]
            return flat[offset:offset + int(np.prod(shape))].view(shape)

        if role == "decoder":
            return _Decoder(layers, self._upload(np.array(small["z_mean"])), [view(f"decoder_{l}") for l in layers],
                            [self._upload(np.array(small[f"decoder_mean_{l}"])) for l in layers])
        shapes = {tuple(table[f"basis_{l}"][1]) for l in layers}
        means = [np.array(small[f"mean_{l}"]) for l in layers]
        if len(shapes) == 1:
            (d, r), = shapes
            offsets = [table[f"basis_{l}"][0] for l in layers]
            if offsets == [offsets[0] + i * d * r for i in range(len(layers))]:
                bases = flat[offsets[0]:offsets[0] + len(layers) * d * r].view(len(layers), d, r)
            else:
                bases = torch.stack([view(f"basis_{l}") for l in layers])
            return _Encoder(layers, self._upload(np.stack(means)), bases, view("weight"))
        return _Encoder(layers, [self._upload(m) for m in means], [view(f"basis_{l}") for l in layers],
                        view("weight"))

    # -- RoPE -----------------------------------------------------------------------------------

    def supports(self, spec):
        """Refuse (NotImplementedError) a spec whose RoPE is not mirrored or disagrees with kvstate."""
        ropes = {}
        for layer in spec.attention_layers:
            ropes.setdefault(layer_rope(spec, layer), layer)
        for rope, layer in ropes.items():
            self._check_rope(spec, layer, rope)

    def _check_rope(self, spec, layer, rope):
        key = (spec, rope)
        if key in self._checked:
            return
        heads, dim = spec.head_shape(layer, "k")
        x = np.random.default_rng(0).standard_normal((5, heads, dim)).astype(np.float32) * 3
        positions = np.asarray([0, 1, 777, 32771, 262143], dtype=np.int64)
        pos = torch.from_numpy(positions).to(self.device)
        for inverse, reference in ((False, kvstate.cache_keys), (True, kvstate.pre_rope_keys)):
            expected = reference(spec, layer, x, positions)
            got = self._upload(x) if rope is None else self.rotate(self._upload(x), pos, rope, inverse=inverse)
            error = float(np.max(np.abs(got.cpu().numpy() - expected)))
            if error > 1e-5 * float(np.max(np.abs(x))):
                raise NotImplementedError(f"device RoPE of layer {layer} disagrees with kvstate ({error:.3g})")
        self._checked[key] = True

    def cos_sin(self, positions, rope):
        """float32 cos, sin [n, n_rot/2] at int64 positions [n] (text) or [n, 4] (t, h, w, e)."""
        f32 = torch.float32
        theta = positions.to(f32)
        theta = theta[:, None] if theta.ndim == 1 else theta
        text = positions.ndim == 1
        extrap = torch.empty((len(theta), rope.n_rot // 2), dtype=f32, device=self.device)
        scale = torch.tensor(rope.theta_scale, dtype=f32, device=self.device)
        for index, component in enumerate(rope.components):
            if text and component == 3:
                extrap[:, index] = 0            # the fourth component of a text token is 0
            else:
                extrap[:, index] = theta[:, 0 if text else component]
            theta = theta * scale               # float32, as ggml's theta *= theta_scale
        if rope.factors is not None:
            # ggml_rope_cache_init's theta / freq_factors[i], in float32 (gemma4's 1e30 leaves a pair unrotated)
            extrap = extrap / torch.tensor(rope.factors, dtype=f32, device=self.device)
        angles = extrap * torch.tensor(rope.freq_scale, dtype=f32, device=self.device)
        if rope.mix is not None:
            mix = torch.tensor(rope.mix, dtype=f32, device=self.device)
            keep = torch.tensor(1, dtype=f32, device=self.device) - mix
            # fma(theta_interp, 1 - mix, theta_extrap * mix) of the pinned build (kvstate._fma32)
            angles = (angles.to(self.wide) * keep.to(self.wide) + (extrap * mix).to(self.wide)).to(f32)
        wide = angles.to(self.wide)
        cos, sin = torch.cos(wide).to(f32), torch.sin(wide).to(f32)
        if rope.magnitude != 1:
            magnitude = torch.tensor(rope.magnitude, dtype=f32, device=self.device)
            cos, sin = cos * magnitude, sin * magnitude
        return cos, sin

    def rotate(self, x, positions, rope, *, inverse, tables=None):
        """x float32 [..., n, heads, dim] rotated in place at positions (kvstate's arithmetic); returns x.

        tables: a dict reused across calls with the same positions (cos/sin per rope)."""
        if tables is None or rope not in tables:
            cos, sin = self.cos_sin(positions, rope)
            if tables is not None:
                tables[rope] = cos, sin
        else:
            cos, sin = tables[rope]
        cos, sin = cos[:, None, :], sin[:, None, :]
        o, n = rope.offset, rope.n_rot
        if rope.neox:
            a, b = x[..., o:o + n // 2], x[..., o + n // 2:o + n]
        else:
            a, b = x[..., o:o + n:2], x[..., o + 1:o + n:2]
        if inverse:
            # kvstate.Rope.invert: divided by cos^2 + sin^2 of the float32 table
            wide = self.wide
            c, s, y0, y1 = cos.to(wide), sin.to(wide), a.to(wide), b.to(wide)
            norm = c * c + s * s
            first, second = ((y0 * c + y1 * s) / norm).to(torch.float32), ((y1 * c - y0 * s) / norm).to(torch.float32)
        else:
            # fma(x0, cos, -(x1 * sin)) and fma(x0, sin, x1 * cos) of the CPU build in float32: addcmul is
            # one fused multiply-add per element on ROCm and CUDA, so this gives kvstate's float64
            # emulation bit for bit (99.09% of a Qwen3-4B window's floats equal kvstate.cache_keys'
            # either way) at 0.033 instead of 0.081 s per 4096-row window on the 8060S
            first = torch.addcmul(-(b * sin), a, cos)
            second = torch.addcmul(b * cos, a, sin)
        a.copy_(first)
        b.copy_(second)
        return x

    # -- compute --------------------------------------------------------------------------------

    def _source_rows(self, spec, stream, kind, layer, span, local):
        """One layer's source rows on the device, [n, width] in the cache dtype: the cells span
        (lo, hi) holding them is uploaded (a slice of the slot file's array: one copy), then the
        rows `local` (cells - lo, an int64 device tensor) are gathered on the device."""
        array = np.asarray(stored_rows(spec, stream, kind, layer))
        lo, hi = span
        return self._upload(array[lo:hi].reshape(hi - lo, -1)).index_select(0, local)

    def _derotate(self, x, spec, layer, positions, tables):
        """Source keys [n, heads * dim] (float32, in place) back to pre-RoPE; NoPE layers stay."""
        rope = layer_rope(spec, layer)
        if rope is not None:
            self.rotate(x.view((len(x),) + spec.head_shape(layer, "k")), positions, rope, inverse=True,
                        tables=tables)

    def encode(self, encoder, cache, kind, positions, chunk_rows=4096, center=None):
        """Hub latents [n, D] in float16; with center (a decoder) z - center.z_mean, added as the
        product's bias inside its float32 epilogue (the latent is rounded to float16 once, after the
        subtraction; the bias itself is held in float16, see _Decoder.z_bias). Source rows are
        uploaded per chunk of positions (the span of cells they occupy), never whole caches."""
        positions = np.asarray(positions, dtype=np.int64)
        layers = encoder.layers
        spec = cache.spec
        if kind == "k":
            for layer in layers:
                rope = layer_rope(spec, layer)
                if rope is not None:
                    self._check_rope(spec, layer, rope)
        out = torch.empty((len(positions), encoder.dim), dtype=self.half, device=self.device)
        cells_of = {}
        for start in range(0, len(positions), chunk_rows):
            part = positions[start:start + chunk_rows]
            n = len(part)
            pos = torch.from_numpy(part).to(self.device) if kind == "k" else None
            tables, spans = {}, {}
            blocks = []
            for i, layer in enumerate(layers):
                stream, index = cache.where[layer]
                if id(stream) not in cells_of:
                    cells_of[id(stream)] = handoff._cells(index, positions, f"source layer {layer}")
                if id(stream) not in spans:            # one span and gather index per stream and chunk
                    cells = cells_of[id(stream)][start:start + n]
                    lo, hi = int(cells.min()), int(cells.max()) + 1
                    spans[id(stream)] = (lo, hi), torch.from_numpy(cells - lo).to(self.device)
                blocks.append(self._source_rows(spec, stream, kind, layer, *spans[id(stream)]))
            if encoder.stacked:
                x = torch.stack(blocks).to(torch.float32)               # [L, n, width]
                del blocks
                if kind == "k":
                    for i, layer in enumerate(layers):
                        self._derotate(x[i], spec, layer, pos, tables)
                x = (x - encoder.means[:, None, :]).to(self.half)
                stage1 = torch.bmm(x, encoder.bases)                     # [L, n, r]
                stage1 = stage1.transpose(0, 1).reshape(n, -1)
            else:
                pieces = []
                for block, layer, mean, basis in zip(blocks, layers, encoder.means, encoder.bases):
                    x = block.to(torch.float32)
                    if kind == "k":
                        self._derotate(x, spec, layer, pos, tables)
                    pieces.append((x - mean).to(self.half) @ basis)
                stage1 = torch.cat(pieces, dim=1)
            del x
            if center is None:
                out[start:start + n] = stage1 @ encoder.weight
            else:
                out[start:start + n] = torch.addmm(center.z_bias, stage1, encoder.weight)
        if not bool(torch.isfinite(out).all()):
            raise FloatingPointError("non-finite hub latents on the GPU (float16 overflow)")
        return out

    def section(self, decoder, z, layers, n_rows, mapped, mapped_rows, fill_rows, fill=None, chunk_rows=4096,
                centered=False):
        """As NumpyBackend.section; centered: z is already z - decoder.z_mean (encode(center=decoder))."""
        weight, mean = decoder.section(layers)
        out = torch.empty((n_rows, len(mean)), dtype=torch.float32, device=self.device)
        mapped = torch.from_numpy(np.asarray(mapped, dtype=np.int64)).to(self.device)
        mapped_rows = torch.from_numpy(np.asarray(mapped_rows, dtype=np.int64)).to(self.device)
        for start in range(0, len(mapped), chunk_rows):
            zc = z.index_select(0, mapped[start:start + chunk_rows])
            if not centered:
                zc = (zc.to(torch.float32) - decoder.z_mean).to(self.half)
            out.index_copy_(0, mapped_rows[start:start + chunk_rows], (zc @ weight).to(torch.float32) + mean)
        if len(fill_rows):
            rows = torch.from_numpy(np.asarray(fill_rows, dtype=np.int64)).to(self.device)
            value = mean.expand(len(rows), -1) if fill is None else self._upload(np.asarray(fill, dtype=np.float32))
            out.index_copy_(0, rows, value.contiguous())
        return out

    def _weight(self, weight):
        """The device copy (wide dtype) of a latent norm weight, uploaded once per array."""
        held = self._weights.get(id(weight))
        if held is None or held[0] is not weight:
            device = self._upload(np.asarray(weight, dtype=np.float32)).to(self.wide)
            held = self._weights[id(weight)] = (weight, device)
        return held[1]

    def renormalize_latents(self, spec, layers, values, norms):
        """As NumpyBackend.renormalize_latents, on the device: kvstate.mla_renormalize_latent's float64
        arithmetic (float32 on MPS), written back into values."""
        for layer, start, lora in latent_columns(spec, layers, norms):
            weight, eps = norms[layer]
            g = self._weight(weight)
            block = values[:, start:start + lora]
            nonzero = g != 0
            c = block.to(self.wide)
            y = torch.where(nonzero, c / torch.where(nonzero, g, torch.ones_like(g)), torch.zeros_like(c))
            y = y / torch.sqrt((y * y).sum(dim=-1, keepdim=True) / int(nonzero.sum()) + float(eps))
            block.copy_((y * g).to(torch.float32))
        return values

    def cache_rows(self, spec, layers, kind, values, positions, dtype, out=None):
        """As NumpyBackend.cache_rows; with out ({layer: host array [n, heads, dim]}), each layer is
        copied from the device straight into its array."""
        n = len(positions)
        shapes = [spec.head_shape(layer, kind) for layer in layers]
        widths = [h * d for h, d in shapes]
        target = TORCH_DTYPES[dtype]
        uniform = len(set(widths)) == 1
        stacked = torch.empty((len(layers), n, widths[0]), dtype=target, device=self.device) if uniform else None
        pos = torch.from_numpy(np.asarray(positions, dtype=np.int64)).to(self.device)
        parts, offset, tables = [], 0, {}
        for i, layer in enumerate(layers):
            block = values[:, offset:offset + widths[i]]
            offset += widths[i]
            if kind == "k":
                rope = layer_rope(spec, layer)
                if rope is not None:
                    self._check_rope(spec, layer, rope)
                    block = self.rotate(block.contiguous().view((n,) + shapes[i]), pos, rope, inverse=False,
                                        tables=tables).view(n, widths[i])
            if uniform:
                stacked[i] = block
            else:
                parts.append(block.to(target))
        tensors = [stacked] if uniform else parts
        if not bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all()):
            # float16 arithmetic can overflow where float32 does not: the engine retries on numpy
            raise FloatingPointError(f"{kind.upper()} rows of layers {list(layers)} are not finite in {dtype} "
                                     "on the GPU")
        rows = [stacked[i] for i in range(len(layers))] if uniform else parts
        if out is not None:
            for layer, width, device_rows in zip(layers, widths, rows):
                if not out[layer].flags.c_contiguous:
                    raise ValueError(f"layer {layer}: cache_rows writes into C-contiguous host arrays only")
                torch.from_numpy(out[layer].reshape(n, width)).copy_(device_rows)    # a view of out[layer]
            return out
        if uniform:
            host = stacked.cpu().numpy()
            return {layer: host[i].reshape((n,) + shapes[i]) for i, layer in enumerate(layers)}
        return {layer: t.cpu().numpy().reshape((n,) + shapes[i]) for i, (layer, t) in enumerate(zip(layers, parts))}

    def host(self, x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
