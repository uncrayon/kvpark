"""DeepSeek V4 (llama.cpp arch deepseek4) as a translation target.

A DeepSeek V4 slot (kvstate.CompressSpec) keeps per-token latents only for its last n_swa tokens;
everything older lives in pooled rows (CSA, HCA, lightning indexer) and the compressor rings.
Those are functions of every old token's (a, s) (dsv4_pool), so a translated state for [0, T0)
is synthesized from per-token quantities the hub decodes, and the target prefills [T0, P)
natively. T0 is rounded down to a multiple of 128 (split_t0): no HCA block is then open at
the split, so the HCA ring is never read and the CSA/LID rings need only the first half of the
channels of their last 4 tokens (the next block's previous half, src/llama-kv-cache-dsv4.cpp:
473-488, 530-538).

  build_state   the target StateFile from a template (a native save of the target server: the
                same -np, flash attention, cache type and draft setup), the pre-RoPE latents of
                the last n_swa positions and the pooled rows and rings of every compressed layer;
  synthesize    build_state from per-token (latent, a, s) of [0, T0) (pools them with dsv4_pool);
  translate     the translator's DeepSeek V4 path (translate.translate dispatches here): hub
                latents of the source's aligned tokens, decoded into the raw latents ("k") and the
                compressor inputs (hub.HEAD_KINDS), pooled window by window, then build_state.

What a state holds and how it is written: the raw cache's base part (cell metadata of [0, T0), no
rows) and sliding part (the last n_swa latents of every layer, RoPE'd: ModelSpec.rope; zero rows
for nextn layers, which the trunk never writes), the CSA/HCA/LID rows of the completed blocks, the
rings. A quantized (q8_0) cache stores raw, CSA and HCA rows Hadamard-rotated (llama.cpp rotates
them when type_k is quantized and the head size a multiple of 64, unless LLAMA_ATTN_ROT_DISABLE
is set on the server, which the template cannot show) and LID rows always. Section stream counts,
the cell sequence id, v_trans and row types come from the template.

DeepSeek V4 as a source is not supported: a saved slot has per-token latents for its last n_swa
tokens only (a capture of a native run, CaptureRows, can stand in for one in tests).
"""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "linear_transfer"))

import handoff  # noqa: E402
import kvstate  # noqa: E402

import dsv4_pool  # noqa: E402
import hub as hub_module  # noqa: E402

SPLIT = dsv4_pool.SPLIT
LATENT = "k"                        # the hub kind of the raw latents (pre-RoPE kv_norm)


def split_t0(t0):
    """The largest multiple of SPLIT (128) not above t0: the translated part of a DeepSeek V4 target."""
    return int(t0) // SPLIT * SPLIT


def _quantized_rotation(spec, cache_type):
    """Whether a cache of this type stores raw, CSA and HCA rows Hadamard-rotated (attn_rot_k,
    src/llama-kv-cache.cpp:315-331; LLAMA_ATTN_ROT_DISABLE unset)."""
    return cache_type in kvstate.QUANT_BLOCKS and spec.head_dim_k[0] % 64 == 0


def _encode_rows(rows, cache_type, what):
    """float32 rows [..., width] as the cache stores them: float16/float32, or q8_0 blocks."""
    rows = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(rows).all():
        raise kvstate.StateFormatError(f"{what} must be finite")
    if cache_type == kvstate.GGML_TYPE_Q8_0:
        return kvstate.quantize_q8_0(rows)
    dtype = kvstate.GGML_DTYPES[cache_type]
    if dtype == np.float16 and np.abs(rows).max(initial=0) >= 65520.0:
        raise kvstate.StateFormatError(f"{what} overflows float16")
    return rows.astype(dtype)


def _template_parts(spec, template):
    if spec.memory != "dsv4":
        raise ValueError(f"{spec.architecture} is not a DeepSeek V4 model")
    state = template.state
    if state.flags != kvstate.FLAGS_NONE:
        raise ValueError("the template must be a full sequence save (slot save)")
    if [part.name for part in state.parts] != [layout.name for layout in spec.layouts(state.flags)]:
        raise ValueError("the template's sections differ from the target spec's (another -np, nextn setup or model?)")
    return {part.name: part for part in state.parts}


def _raw_stream(template_part, positions, k, k_types, *, seq_id, v_trans):
    """A raw-cache KVCache holding `positions` in the template's stream layout."""
    index = next(i for i, s in enumerate(template_part.streams) if s is not None)
    stream = kvstate.KVStream(np.asarray(positions, dtype=np.int32), tuple((seq_id,) for _ in positions), None,
                              v_trans, k, {}, k_types)
    streams = [None] * len(template_part.streams)
    streams[index] = stream
    return kvstate.KVCache(template_part.name, template_part.layers, streams)


def build_state(spec, template, tokens, t0, latents, pooled, *, server_tokens=None):
    """The target StateFile holding positions [0, t0).

    latents: {trunk layer: float32 [n, head] or [n, 1, head]} pre-RoPE latents (kv_norm) of the last
    n = min(t0, n_swa) positions; pooled: {(pool name, layer): (rows float32 [t0 // ratio, width],
    (ring kv, ring score) float32 [state_size, state_width])} for every pool and layer, rows as a
    float16 cache stores them (dsv4_pool.pool_blocks: LID rows rotated, CSA/HCA rows not; the
    rotation of a quantized cache is added here). server_tokens: pack the token words as
    llama-server does (default: as the template)."""
    parts = _template_parts(spec, template)
    tokens = np.asarray(tokens, dtype=np.int64)
    t0 = int(t0)
    if not 0 < t0 <= len(tokens):
        raise ValueError("t0 must lie in [1, len(tokens)]")
    swa_template = parts["swa"].stream()
    seq_id = int(swa_template.seq_ids[0][0])
    v_trans = swa_template.v_trans
    types = sorted(set(swa_template.k_types.values()))
    if len(types) != 1:
        raise ValueError("the template's raw rows mix cache types")
    cache_type = types[0]
    rotate = _quantized_rotation(spec, cache_type)
    head = spec.head_dim_k[0]

    out = {"base": _raw_stream(parts["base"], np.arange(t0), {}, {}, seq_id=seq_id, v_trans=v_trans)}
    window = np.arange(max(0, t0 - spec.n_swa), t0, dtype=np.int64)
    rows = {}
    for layer in parts["swa"].layers:
        if spec.layer_types[layer] == "nextn":
            rows[layer] = _encode_rows(np.zeros((len(window), 1, head), np.float32), cache_type, f"layer {layer}")
            continue
        if layer not in latents:
            raise ValueError(f"missing the latents of layer {layer}")
        latent = np.asarray(latents[layer], dtype=np.float32).reshape(-1, 1, head)
        if len(latent) != len(window):
            raise ValueError(f"layer {layer}: {len(latent)} latents for the {len(window)} window positions")
        cached = spec.rope(layer).apply(latent, window)
        if rotate:
            cached = dsv4_pool.hadamard_rows(cached)
        rows[layer] = _encode_rows(cached, cache_type, f"layer {layer} raw rows")
    out["swa"] = _raw_stream(parts["swa"], window, rows, {layer: cache_type for layer in rows}, seq_id=seq_id,
                             v_trans=v_trans)
    for pool in spec.pools():
        template_rows = parts[pool.name]
        n_rows = t0 // pool.ratio
        values, types_ = {}, {}
        for layer in pool.layers:
            pooled_rows, _ = pooled[pool.name, layer]
            pooled_rows = np.asarray(pooled_rows, dtype=np.float32)
            if pooled_rows.shape != (n_rows, pool.width):
                raise ValueError(f"{pool.name} layer {layer}: rows {pooled_rows.shape}, expected {(n_rows, pool.width)}")
            if rotate and pool.name != "lid":
                pooled_rows = dsv4_pool.hadamard_rows(pooled_rows)
            row_type = template_rows.types[layer]
            values[layer] = _encode_rows(pooled_rows, row_type, f"{pool.name} layer {layer} rows")[None].repeat(
                template_rows.n_stream, axis=0)
            types_[layer] = row_type
        out[pool.name] = kvstate.PooledRows(pool.name, template_rows.layers, template_rows.n_stream, n_rows, values,
                                            types_)
        template_ring = parts[pool.name + "_state"]
        kv, score = {}, {}
        for layer in pool.layers:
            ring_kv, ring_score = pooled[pool.name, layer][1]
            shape = (template_ring.n_stream, pool.state_size, pool.state_width)
            kv[layer] = np.broadcast_to(np.asarray(ring_kv, dtype=np.float32), shape).copy()
            score[layer] = np.broadcast_to(np.asarray(ring_score, dtype=np.float32), shape).copy()
        out[pool.name + "_state"] = kvstate.CompressorRing(template_ring.name, template_ring.layers,
                                                          template_ring.n_stream, kv, score)
    packed = template.server_packed if server_tokens is None else bool(server_tokens)
    state = kvstate.StateFile(kvstate.token_words(tokens[:t0], server_packed=packed),
                              kvstate.SequenceState(spec, kvstate.FLAGS_NONE,
                                                    [out[layout.name] for layout in spec.layouts(kvstate.FLAGS_NONE)]))
    state.check_positions()
    return state


def synthesize(spec, template, tokens, t0, inputs, tensors, *, ring="minimal", window=4096):
    """build_state from per-token quantities of [0, t0): inputs {("k", layer): latents [n, ...] of
    positions [0, t0) or of the last min(t0, n_swa); (pool name, layer): (a, pre-APE s) float32
    [t0, state_width] each}; tensors: dsv4_pool.PoolTensors of the target. ring "minimal" (what a
    split at a multiple of 128 reads; t0 must be one) or "full" (the last state_size tokens, as a
    native save holds them)."""
    if ring not in ("minimal", "full"):
        raise ValueError("ring must be 'minimal' or 'full'")
    t0 = int(t0)
    n_window = min(t0, spec.n_swa)
    latents = {}
    for layer in spec.layers_of("sliding"):
        rows = np.asarray(inputs[LATENT, layer], dtype=np.float32)
        latents[layer] = rows[-n_window:] if len(rows) >= n_window else rows
    pooled = {}
    for pool in spec.pools():
        for layer in pool.layers:
            a, s = inputs[pool.name, layer]
            pooler = dsv4_pool.Pooler(spec, pool, layer, tensors)
            for start in range(0, t0, window):
                pooler.feed(a[start:min(start + window, t0)], s[start:min(start + window, t0)])
            pooled[pool.name, layer] = pooler.result(minimal=ring == "minimal")
    return build_state(spec, template, tokens, t0, latents, pooled)


# -- per-token captures ------------------------------------------------------------------------

def driver_captures(directory, spec):
    """{(kind, layer): float32 [n, width]} per-token quantities of a kvpark-state-driver capture run
    (--capture kv_norm,{csa,hca,lid}_state_kv,{csa,hca,lid}_state_score): LATENT = the pre-RoPE latent
    (kv_norm), and per pool [a | s] side by side (the hub's head kinds); every ubatch in order."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    parts = {}
    for entry in manifest["captures"]:
        parts.setdefault(entry["name"], []).append((int(entry["pos_start"]), directory / entry["file"]))
    arrays = {}
    for name, files in parts.items():
        files.sort()
        arrays[name] = np.concatenate([np.load(path) for _, path in files], axis=0)
    out = {}
    for layer in spec.layers_of("sliding"):
        out[LATENT, layer] = arrays[f"kv_norm-{layer}"].reshape(-1, spec.head_dim_k[layer])
    for pool in spec.pools():
        a_name, s_name = dsv4_pool.CAPTURES[pool.name]
        for layer in pool.layers:
            a = arrays[f"{a_name}-{layer}"].reshape(-1, pool.state_width)
            s = arrays[f"{s_name}-{layer}"].reshape(-1, pool.state_width)
            out[pool.name, layer] = np.concatenate([a, s], axis=1)
    return out


def capture_names(spec):
    """The --capture names of a state-driver run that driver_captures reads."""
    names = ["kv_norm"]
    for pool in spec.pools():
        if pool.layers:
            names += list(dsv4_pool.CAPTURES[pool.name])
    return names


class CaptureRows:
    """Per-token hub rows of a native DeepSeek V4 run (driver_captures) as a translation source:
    rows_many(kind, layers, positions) like translate.SourceCache, for the latent kind and the head
    kinds. Only tests use it (an identity hub translating a model into itself); a saved slot has no
    such rows."""

    def __init__(self, rows, tokens):
        self.rows_by_key = rows
        self.tokens = np.asarray(tokens)

    def rows_many(self, kind, layers, positions):
        positions = np.asarray(positions, dtype=np.int64)
        return {layer: np.asarray(self.rows_by_key[kind, layer], dtype=np.float32)[positions] for layer in layers}

    def n_positions(self, layers, kind=LATENT):
        return min(len(self.rows_by_key[kind, layer]) for layer in layers) if layers else len(self.tokens)


# -- translation ---------------------------------------------------------------------------------

@dataclass
class _Decoded:
    """A decoder side and what it fills unmapped cells with (its train mean)."""

    side: object
    layers: list

    def rows(self, z, n_rows, mapped_rows, layers=None):
        """float32 [n_rows, sum d_l] of `layers`: rows mapped_rows decoded from centred z, the others the mean."""
        weight, mean = self.side.section(self.layers if layers is None else layers)
        out = np.empty((n_rows, len(mean)), dtype=np.float32)
        out[:] = mean
        if len(mapped_rows):
            out[mapped_rows] = z @ weight + mean
        return out


def translate(source_state, source, target, hub, *, t0, template, fill="mean", server_tokens=True, window_rows=1024,
              chunk_rows=4096, ring="minimal", tensors=None, backend=None):
    """(kvstate.StateFile, report) of a DeepSeek V4 target holding [0, split_t0(t0)); see the module
    docstring. source_state: the source's parsed slot (translate.SourceCache reads it), or a
    CaptureRows; fill: only "mean" (the decoders' train means; the token table holds no compressor
    inputs); tensors: dsv4_pool.PoolTensors (default: read from target.gguf). The arithmetic is
    numpy; a numpy `backend` lends its resident hub sides (the engine daemon keeps them between
    switches), other backends' sides are device arrays, so the hub is read instead."""
    import translate as translate_module            # translate dispatches here; imported late
    started = time.perf_counter()
    spec = target.spec
    if fill != "mean":
        raise ValueError("a DeepSeek V4 target fills unmapped cells with the decoder means (fill 'mean') only")
    requested = int(t0)
    t0 = split_t0(min(requested, len(target.tokens["token_id"])))
    if t0 < SPLIT:
        raise ValueError(f"a DeepSeek V4 target needs at least {SPLIT} tokens to translate (t0 = {requested})")
    if tensors is None:
        if target.gguf is None:
            raise ValueError(f"{target.name}: translating into DeepSeek V4 needs its GGUF (Side.gguf) for the "
                             "compressor APE and norm tensors")
        tensors = dsv4_pool.pool_tensors(target.gguf, spec)
    if window_rows % SPLIT:
        raise ValueError(f"window_rows must be a multiple of {SPLIT}")
    timings = {}

    # alignment: which source token feeds which target position (translate.prepare's rule)
    cache = source_state if isinstance(source_state, CaptureRows) else translate_module.SourceCache(source_state,
                                                                                                    source.spec)
    load = backend.side if getattr(backend, "name", None) == "numpy" else None
    if load is not None:
        encoder, latent_decoder = load(hub, LATENT, source.name, "encoder"), load(hub, LATENT, target.name, "decoder")
    else:
        encoder, latent_decoder = hub_module.sides(hub, LATENT, source.name, target.name)
    heads = {}
    for pool in spec.pools():
        if pool.layers:
            own, decoder = hub_module.head_sides(hub, pool.name, source.name, target.name, load=load)
            heads[pool.name] = (own, _Decoded(decoder, list(pool.layers)))
    alignment = handoff.align_document({**source.record, "document_id": target.record["document_id"]},
                                       source.tokens, target.record, target.tokens)
    n_source = cache.n_positions(encoder.layers)
    if n_source and not np.array_equal(np.asarray(cache.tokens)[:n_source],
                                       np.asarray(source.tokens["token_id"])[:n_source]):
        raise ValueError("source state tokens are not a prefix of the source tokenization")
    source_of = np.full(len(target.tokens["token_id"]), -1, dtype=np.int64)
    source_of[alignment.tgt_position] = alignment.src_position
    content = (alignment.tgt_class == handoff.ALIGNED) & (source_of >= 0) & (source_of < n_source)
    mapped = np.flatnonzero(content[:t0])
    timings["prepare_s"] = time.perf_counter() - started

    def encode(enc, kind, positions):
        parts = [enc.encode(cache.rows_many(kind, enc.layers, positions[i:i + chunk_rows]))
                 for i in range(0, len(positions), chunk_rows)]
        return np.concatenate(parts) if parts else np.zeros((0, enc.dim), dtype=np.float32)

    latent = _Decoded(latent_decoder, [layer for layer in spec.layers_of("sliding")])
    poolers = {(pool.name, layer): dsv4_pool.Pooler(spec, pool, layer, tensors)
               for pool in spec.pools() for layer in pool.layers}
    window_start = max(0, t0 - spec.n_swa)
    latents = {layer: [] for layer in latent.layers}
    for p0 in range(0, t0, window_rows):
        p1 = min(p0 + window_rows, t0)
        m0, m1 = (int(i) for i in np.searchsorted(mapped, [p0, p1]))
        here, rows_at = mapped[m0:m1], mapped[m0:m1] - p0
        t = time.perf_counter()
        z = encode(encoder, LATENT, source_of[here]) - latent_decoder.z_mean
        timings["encode_s"] = timings.get("encode_s", 0.0) + time.perf_counter() - t
        t = time.perf_counter()
        if p1 > window_start:
            # the raw latents of the window's last n_swa positions
            w0 = max(p0, window_start)
            keep = here >= w0
            values = latent.rows(z[keep], p1 - w0, here[keep] - w0)
            offset = 0
            for layer in latent.layers:
                latents[layer].append(values[:, offset:offset + spec.head_dim_k[layer]])
                offset += spec.head_dim_k[layer]
        for pool in spec.pools():
            if not pool.layers:
                continue
            own, decoded = heads[pool.name]
            if own is not None:         # a source that holds this kind itself (CaptureRows)
                z_head = encode(own, pool.name, source_of[here]) - decoded.side.z_mean
            else:                       # the latent kind's z, centred for this decoder
                z_head = z + (latent_decoder.z_mean - decoded.side.z_mean)
            for layer in pool.layers:   # one layer at a time: [window, 2 * state_width] rows
                values = decoded.rows(z_head, p1 - p0, rows_at, layers=[layer])
                poolers[pool.name, layer].feed(values[:, :pool.state_width], values[:, pool.state_width:])
        timings["decode_s"] = timings.get("decode_s", 0.0) + time.perf_counter() - t
    t = time.perf_counter()
    pooled = {key: pooler.result(minimal=ring == "minimal") for key, pooler in poolers.items()}
    state = build_state(spec, template, target.tokens["token_id"], t0,
                        {layer: np.concatenate(parts) for layer, parts in latents.items()}, pooled,
                        server_tokens=server_tokens)
    timings["assemble_s"] = time.perf_counter() - t
    report = {"t0": t0, "t0_requested": requested, "split": SPLIT, "fill": fill, "cells": t0,
              "mapped_cells": int(len(mapped)), "fill_cells": int(t0 - len(mapped)),
              "sections": {"base": t0, "swa": t0 - window_start,
                           **{pool.name: t0 // pool.ratio for pool in spec.pools()}},
              "kinds": [LATENT] + list(heads), "ring": ring, "recurrent_reset": False, "native_prefix": 0,
              "seconds": round(time.perf_counter() - started, 3), "backend": "numpy", "device": "cpu",
              "window_rows": window_rows, "timings": {key: round(value, 3) for key, value in timings.items()}}
    return state, report
