"""translate.translate on tiny synthetic models (dense, iSWA, hybrid, MLA, DSA, DSA-iSWA, K-only hybrid)
with hand-built linear hubs."""

import importlib.util
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import backends
import hub
import translate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "linear_transfer"))
import handoff  # noqa: E402
import kvstate  # noqa: E402

# the implementation before the prepare/compute/assemble split, when KVPARK_TRANSLATE_BASELINE
# names a snapshot of it; FROZEN below holds digests of its outputs for every other machine
BASELINE = Path(os.environ.get("KVPARK_TRANSLATE_BASELINE") or "/nonexistent")

SOURCE = kvstate.ModelSpec("qwen3", ("full",) * 2, (1, 1), (8, 8), (8, 8), rope_type="neox", n_rot=8,
                           rope_freq_base=10000.0)
TARGET = kvstate.ModelSpec("qwen3", ("full",) * 3, (2, 2, 2), (4, 4, 4), (4, 4, 4), rope_type="neox", n_rot=4,
                           rope_freq_base=500000.0)
D = 6
# Muse-like iSWA: NORM RoPE on sliding layers, NoPE global layers; window 3 (sections of 4 cells)
MUSE = kvstate.ModelSpec("muse-glimmer", ("sliding", "sliding", "full", "sliding", "sliding", "full"), (2,) * 6,
                         (4,) * 6, (4,) * 6, n_swa=3, rope_type="norm", n_rot=4, rope_freq_base=500000.0,
                         rope_freq_base_swa=10000.0)
# Qwen3.5-like hybrid: IMROPE over 8 of 16 dims, sections reaching the unrotated fourth component
HYBRID = kvstate.ModelSpec("qwen35", ("recurrent", "full", "recurrent", "full"), (0, 2, 0, 2), (0, 16, 0, 16),
                           (0, 16, 0, 16), rope_type="imrope", n_rot=8, rope_freq_base=1e7,
                           rope_sections=(1, 1, 1, 1), recurrent_row_bytes=(16, 32))


def mla_metadata(architecture, n_layer, *, kv_lora=8, rope=4, nextn=0, yarn=None, **extra):
    """Tiny MLA GGUF metadata (as kvstate's tests write it): rows [c_kv (kv_lora) | k_pe (rope)].
    yarn: (factor, original context, yarn_log_multiplier as stored)."""
    p = architecture + "."
    meta = {"general.architecture": architecture, p + "block_count": n_layer + nextn, p + "context_length": 256,
            p + "embedding_length": 64, p + "attention.head_count": 2, p + "attention.head_count_kv": 1,
            p + "attention.key_length": kv_lora + rope, p + "attention.value_length": kv_lora,
            p + "attention.key_length_mla": 12, p + "attention.value_length_mla": 8,
            p + "attention.kv_lora_rank": kv_lora, p + "rope.dimension_count": rope, p + "rope.freq_base": 10000.0,
            p + "attention.layer_norm_rms_epsilon": 1e-6}
    if nextn:
        meta[p + "nextn_predict_layers"] = nextn
    if yarn is not None:
        meta.update({p + "rope.scaling.type": "yarn", p + "rope.scaling.factor": yarn[0],
                     p + "rope.scaling.original_context_length": yarn[1],
                     p + "rope.scaling.yarn_log_multiplier": yarn[2]})
    meta.update({p + key: value for key, value in extra.items()})
    return meta


# top_k 9: the 9-token target prompt of the MLA tests fits (zero indexer keys are exact)
INDEXER = {"attention.indexer.head_count": 2, "attention.indexer.key_length": 64, "attention.indexer.top_k": 9}
# deepseek2 with YaRN (DeepSeek-V2 style magnitude 1) and a nextn layer, which 85d5703a3's cache keeps as
# zero rows (a7b94df2c's leaves it out: DEEPSEEK2.for_revision("a7b94df2c"))
DEEPSEEK2 = kvstate.ModelSpec.from_metadata(mla_metadata("deepseek2", 2, nextn=1, yarn=(4.0, 64, 0.0707)),
                                            llama_revision="85d5703a3")
# DSA: an MLA section and a lightning-indexer key section (NEOX, Hadamard), top_k 9 (and 8, 4)
DSA = kvstate.ModelSpec.from_metadata(mla_metadata("deepseek32", 2, **INDEXER))
DSA_TOP8 = kvstate.ModelSpec.from_metadata(mla_metadata("deepseek32", 2, **{**INDEXER, "attention.indexer.top_k": 8}))
DSA_TOP4 = kvstate.ModelSpec.from_metadata(mla_metadata("deepseek32", 2, **{**INDEXER, "attention.indexer.top_k": 4}))
# DSA-iSWA: full MLA layers 0, 2 with indexer keys, sliding layers 1, 3 (latent 6, base 500, window 3)
DOTS = kvstate.ModelSpec.from_metadata(mla_metadata(
    "dots3note", 4, **INDEXER, **{"attention.indexer.types": [1, 0, 1, 0], "attention.sliding_window": 3,
                                  "attention.sliding_window_pattern": [0, 1, 0, 1], "attention.kv_lora_rank_swa": 6,
                                  "attention.key_length_swa": 10, "rope.freq_base_swa": 500.0}))
# kimi-linear-like hybrid: NoPE K-only MLA layers 0, 3 and KDA recurrent layers 1, 2
KDA = kvstate.ModelSpec.from_metadata(mla_metadata(
    "kimi-linear", 4, **{"attention.head_count_kv": [1, 0, 0, 1], "ssm.conv_kernel": 4, "kda.head_dim": 8}))


def linear_hub(seed=0):
    """Source rows (2 layers x 8) -> z (6) -> target rows (3 layers x 8), plain matrices."""
    rng = np.random.default_rng(seed)
    hubs = {}
    for kind in hub.KINDS:
        source_view = hub.ViewTransform([0, 1], [np.zeros(8, np.float32)] * 2, [np.eye(8, dtype=np.float32)] * 2,
                                        np.eye(16, dtype=np.float32), np.ones(16, dtype=np.float32))
        target_view = hub.ViewTransform([0, 1, 2], [np.zeros(8, np.float32)] * 3, [np.eye(8, dtype=np.float32)] * 3,
                                        np.eye(24, dtype=np.float32), np.ones(24, dtype=np.float32))
        encoder = rng.standard_normal((16, D)).astype(np.float32) / 4
        decoders = [rng.standard_normal((D, 8)).astype(np.float32) for _ in range(3)]
        means = [rng.standard_normal(8).astype(np.float32) for _ in range(3)]
        hubs[kind] = hub.HubKind(["src", "tgt"], {"src": source_view, "tgt": target_view},
                                 {"src": encoder, "tgt": np.zeros((24, D), np.float32)}, np.ones(D),
                                 {"src": [np.zeros(8, np.float32)] * 2, "tgt": means},
                                 {"src": np.zeros(D, np.float32), "tgt": np.zeros(D, np.float32)},
                                 {"src": [np.zeros((D, 8), np.float32)] * 2, "tgt": decoders})
    return hubs


def record(text, segments, keys, document_id="doc"):
    return {"document_id": document_id, "text": text, "segments": segments, "segment_keys": keys,
            "split": "test", "final_message_segments": []}


def tokens_for(text, cuts, special=()):
    ends = np.asarray(cuts, dtype=np.int64)
    assert ends[-1] == len(text.encode())
    flags = np.zeros(len(ends), dtype=np.uint8)
    flags[list(special)] = 1
    return {"token_id": np.arange(len(ends), dtype=np.int32) + 10, "byte_end": ends, "special": flags}


def attention_widths(spec):
    return {layer: spec.n_head_kv[layer] * spec.head_dim_k[layer] for layer in spec.layers_of("full", "sliding")}


def kind_widths(spec):
    """{kind: {layer: width}} of a model's hub views: an MLA layer's K is its [c_kv | k_pe] row, its V c_kv."""
    layers = spec.layers_of("full", "sliding")
    return {kind: {layer: int(np.prod(spec.head_shape(layer, kind))) for layer in layers} for kind in hub.KINDS}


def latent_norm_weights(spec, seed=4):
    """{layer: (attn_kv_a_norm weight, eps)} for an MLA spec's attention layers (1 + noise, as the fixtures)."""
    rng = np.random.default_rng(seed)
    return {layer: ((1 + 0.2 * rng.standard_normal(spec.kv_lora[layer])).astype(np.float32), 1e-6)
            for layer in spec.layers_of("full", "sliding") if spec.k_only(layer)}


def random_indexer_ridge(spec, name="tgt", seed=8):
    """A hub.IndexerRidge with random weights for every indexer layer of a DSA spec."""
    rng = np.random.default_rng(seed)
    layers = list(spec.indexer.layers)
    width, size = spec.head_dim_k[layers[0]], spec.indexer.head_size
    return hub.IndexerRidge(name, layers, [rng.standard_normal(width).astype(np.float32) for _ in layers],
                            [(rng.standard_normal((width, size)) / 3).astype(np.float32) for _ in layers],
                            [rng.standard_normal(size).astype(np.float32) for _ in layers])


def random_hub(models, *, dim=D, seed=0, subsets=None, hub_kind=None):
    """{kind: HubKind} with random stage-1 bases (rank below the width), basis2 (Fortran-ordered, as
    save_hub's npz hold it), scale, encoders and decoders; models: {name: {layer: width}} or, when
    the kinds differ (MLA), {name: {kind: {layer: width}}} (kind_widths); subsets: {name: layers}
    for models that encode from a layer subset (sliding-window sources)."""
    rng = np.random.default_rng(seed)
    hub_kind = hub_kind or hub.HubKind
    hubs = {}
    by_kind = {kind: {name: widths[kind] if set(widths) == set(hub.KINDS) else widths
                      for name, widths in models.items()} for kind in hub.KINDS}

    def transform(widths, layers):
        ranks = [min(widths[layer], 5) for layer in layers]
        means = [rng.standard_normal(widths[layer]).astype(np.float32) for layer in layers]
        bases = [np.asfortranarray(rng.standard_normal((widths[l], r)).astype(np.float32) / 3)
                 for l, r in zip(layers, ranks)]
        basis2 = np.asfortranarray(rng.standard_normal((sum(ranks), 7)).astype(np.float32) / 2)
        scale = rng.uniform(0.5, 2.0, 7).astype(np.float32)
        return hub.ViewTransform(list(layers), means, bases, basis2, scale)

    for kind in hub.KINDS:
        models = by_kind[kind]
        names = list(models)
        transforms = {name: transform(widths, sorted(widths)) for name, widths in models.items()}
        encoders = {name: (rng.standard_normal((7, dim)) / 3).astype(np.float32) for name in names}
        decoders = {name: [(rng.standard_normal((dim, models[name][l])) / 2).astype(np.float32)
                           for l in sorted(models[name])] for name in names}
        means = {name: [rng.standard_normal(models[name][l]).astype(np.float32) for l in sorted(models[name])]
                 for name in names}
        z_mean = {name: rng.standard_normal(dim).astype(np.float32) for name in names}
        source_transforms = {name: transform(models[name], layers) for name, layers in (subsets or {}).items()}
        source_encoders = {name: (rng.standard_normal((7, dim)) / 3).astype(np.float32) for name in (subsets or {})}
        hubs[kind] = hub_kind(names, transforms, encoders, np.ones(dim), means, z_mean, decoders,
                              {name: {"alpha": 0.01} for name in names}, source_transforms, source_encoders)
    return hubs


def reference_rows(hubs, kind, source, target, rows_by_layer, layer):
    """The pre-split algebra: z = ((stage1 @ basis2) * scale) @ encoder, then per layer decoders."""
    hub_kind = hubs[kind]
    t = hub_kind.source_transforms.get(source) or hub_kind.transforms[source]
    encoder = hub_kind.source_encoders.get(source, hub_kind.encoders[source])
    blocks = [np.asarray(rows_by_layer[l], dtype=np.float64) for l in t.layers]
    stage1 = np.concatenate([(b - m) @ basis for b, m, basis in zip(blocks, t.means, t.bases)], axis=1)
    z = ((stage1 @ t.basis2) * t.scale) @ encoder
    index = hub_kind.transforms[target].layers.index(layer)
    return (z - hub_kind.decoder_z_mean[target]) @ hub_kind.decoders[target][index] + hub_kind.decoder_mean[target][index]


def hybrid_state(spec, tokens, k, v, *, dtype="f32", v_trans=False):
    """A Qwen3.5-like slot file: attention KV (pre-RoPE k rotated) and one recurrent cell at the end."""
    positions = np.arange(len(tokens))
    attention, recurrent = spec.layers_of("full"), spec.layers_of("recurrent")
    keys = {layer: kvstate.cache_keys(spec, layer, k[layer], positions) for layer in attention}
    stream = kvstate.build_kv_stream(spec, attention, positions, keys, {layer: v[layer] for layer in attention},
                                     dtype=dtype, v_trans=v_trans, tokens_at=np.asarray(tokens))
    rows_r = {layer: kvstate.OpaqueRows(0, 16, bytes(range(16))) for layer in recurrent}
    rows_s = {layer: kvstate.OpaqueRows(0, 32, bytes(range(32))) for layer in recurrent}
    parts = [kvstate.KVCache("attn", attention, [stream]),
             kvstate.RecurrentState("recurrent", recurrent, np.asarray([len(tokens) - 1], dtype=np.int32), ((),),
                                    len(spec.layer_types), rows_r, rows_s)]
    state = kvstate.StateFile(kvstate.token_words(tokens), kvstate.SequenceState(spec, kvstate.FLAGS_NONE, parts))
    return kvstate.parse_file(state.to_bytes(), spec)


def layout_state(spec, tokens, k, v, rng, *, dtype="f32", v_trans=False):
    """A slot file of any kvstate layout (used for the MLA, DSA and K-only hybrid specs): KV sections
    from pre-RoPE rows (sliding ones their last n_swa cells, indexer sections random keys), each
    recurrent section one cell at the end."""
    n = len(tokens)
    positions = np.arange(n)
    parts = []
    for layout in spec.layouts(kvstate.FLAGS_NONE):
        if layout.kind == "recurrent":
            r_bytes, s_bytes = spec.recurrent_row_bytes

            def rows(size):
                data = bytes(i % 251 for i in range(size))
                return {layer: kvstate.OpaqueRows(0, size, data) for layer in layout.layers}
            parts.append(kvstate.RecurrentState(layout.name, layout.layers, np.asarray([n - 1], dtype=np.int32), ((),),
                                                spec.n_layer_trunk, rows(r_bytes), rows(s_bytes)))
            continue
        cells = positions[-spec.n_swa:] if layout.sliding else positions
        pre = k
        if layout.role == "indexer":
            pre = {layer: rng.standard_normal((n, 1, spec.indexer.head_size)).astype(np.float32)
                   for layer in layout.layers}
        keys = {layer: kvstate.cache_keys(spec, layer, pre[layer][cells], cells, role=layout.role)
                for layer in layout.layers if layer in pre}
        values = {layer: v[layer][cells] for layer in layout.layers
                  if spec.kv_shape(layer, "v", layout.role) is not None}
        stream = kvstate.build_kv_stream(spec, layout.layers, cells, keys, values, dtype=dtype, v_trans=v_trans,
                                         tokens_at=np.asarray(tokens)[cells], role=layout.role)
        parts.append(kvstate.KVCache(layout.name, layout.layers, [stream]))
    state = kvstate.StateFile(kvstate.token_words(tokens), kvstate.SequenceState(spec, kvstate.FLAGS_NONE, parts))
    return kvstate.parse_file(state.to_bytes(), spec)


def native_state(spec, tokens, rng, *, dtype="f32", v_trans=False, pre=None):
    """(parsed slot file, pre-RoPE K, V) of any supported spec holding `tokens`, random values (an MLA
    layer's V: the c_kv view of its K)."""
    n = len(tokens)
    layers = spec.layers_of("full", "sliding")
    k = pre or {layer: rng.standard_normal((n,) + spec.head_shape(layer, "k")).astype(np.float32) for layer in layers}
    if spec.kv_lora is not None:
        v = {layer: k[layer][..., :spec.kv_lora[layer]] for layer in layers}
        return layout_state(spec, tokens, k, v, rng, dtype=dtype, v_trans=v_trans), k, v
    v = {layer: rng.standard_normal((n,) + spec.head_shape(layer, "v")).astype(np.float32) for layer in layers}
    if spec.memory == "hybrid":
        return hybrid_state(spec, tokens, k, v, dtype=dtype, v_trans=v_trans), k, v
    builder = kvstate.build_muse_state if spec.memory == "iswa" else kvstate.build_dense_state
    state = builder(spec, tokens, np.arange(n), k, v, dtype=dtype, v_trans=v_trans)
    return kvstate.parse_file(state.to_bytes(), spec), k, v


# the cases frozen below: (source spec, target spec, source layer subsets)
FROZEN_CASES = {"dense-dense": (SOURCE, TARGET, None), "dense-iswa": (SOURCE, MUSE, None),
                "dense-hybrid": (SOURCE, HYBRID, None), "iswa-hybrid": (MUSE, HYBRID, {"src": [2, 5]})}


def digests(state):
    """{"section.kind.layer": [sum, sum of squares, weighted sum]} of a parsed state's stored K/V
    (float64 sums; the weights sin(0.618 i + 0.1) catch any reordering or rotation of values)."""
    out = {}
    for part in state.state.parts:
        if not isinstance(part, kvstate.KVCache):
            continue
        stream = part.stream()
        for kind, arrays in (("k", stream.k), ("v", stream.v)):
            for layer in part.layers:
                x = np.asarray(arrays[layer], dtype=np.float64).ravel()
                weights = np.sin(np.arange(len(x)) * 0.618 + 0.1)
                out[f"{part.name}.{kind}.{layer}"] = [float(x.sum()), float(x @ x), float(x @ weights)]
    return out


# Digests of the pre-split module's outputs (the baseline-20261003 snapshot of translate.py and
# hub.py) on FROZEN_CASES with float32 caches and rng seed 5: its report (without seconds), the
# state's byte length and digests(). Regenerate them from that snapshot (load_baseline) only when
# a fixture above changes on purpose.
FROZEN = {
    "dense-dense": {
        "report": {"t0": 8, "swa_cells": 0, "fill": "mean", "cells": 8, "mapped_cells": 4, "fill_cells": 4,
                   "sections": {"kv": 8}, "recurrent_reset": False, "native_prefix": 0},
        "bytes": 1780,
        "values": {
            "kv.k.0": [-24.8030329, 521.059299, -29.3913162],
            "kv.k.1": [-23.56848703, 265.564166, -3.397179422],
            "kv.k.2": [-2.003408185, 172.7841864, 4.341261186],
            "kv.v.0": [2.219748288, 229.7494032, 9.131855498],
            "kv.v.1": [0.703148067, 204.3655731, -7.278742881],
            "kv.v.2": [-10.11618891, 173.8408829, -11.90075086],
        }},
    "dense-iswa": {
        "report": {"t0": 8, "swa_cells": 4, "fill": "mean", "cells": 8, "mapped_cells": 4, "fill_cells": 4,
                   "sections": {"base": 8, "swa": 4}, "recurrent_reset": False, "native_prefix": 0},
        "bytes": 2428,
        "values": {
            "base.k.2": [-7.736364245, 174.6237677, -10.15997141],
            "base.k.5": [-18.25940013, 137.3253808, -8.438539655],
            "base.v.2": [-2.866488516, 356.5176431, -17.32167644],
            "base.v.5": [-19.29065344, 452.5191793, 8.85799882],
            "swa.k.0": [4.245190967, 92.30289817, 12.64854216],
            "swa.k.1": [23.39623731, 135.1185483, 5.496424212],
            "swa.k.3": [-7.733788949, 114.1344607, -7.531597156],
            "swa.k.4": [-7.68360598, 110.2018014, -3.214301742],
            "swa.v.0": [-3.968666136, 282.963916, 6.290004634],
            "swa.v.1": [-3.27383849, 196.224416, 9.546943832],
            "swa.v.3": [-42.61291152, 388.2551136, 15.46543659],
            "swa.v.4": [-1.942623213, 200.8902664, 1.424753948],
        }},
    "dense-hybrid": {
        "report": {"t0": 8, "swa_cells": 0, "fill": "mean", "cells": 8, "mapped_cells": 4, "fill_cells": 4,
                   "sections": {"attn": 8}, "recurrent_reset": True, "native_prefix": 0},
        "bytes": 4576,
        "values": {
            "attn.k.1": [87.23644973, 1116.765479, 25.6323193],
            "attn.k.3": [38.03379105, 1481.110258, 4.123715106],
            "attn.v.1": [-73.92875317, 3216.411892, -18.01887915],
            "attn.v.3": [-26.46544258, 2215.104355, 3.584986941],
        }},
    "iswa-hybrid": {
        "report": {"t0": 8, "swa_cells": 0, "fill": "mean", "cells": 8, "mapped_cells": 4, "fill_cells": 4,
                   "sections": {"attn": 8}, "recurrent_reset": True, "native_prefix": 0},
        "bytes": 4576,
        "values": {
            "attn.k.1": [8.773005536, 1183.745053, 6.051699695],
            "attn.k.3": [64.32211368, 1814.23799, -12.7790702],
            "attn.v.1": [7.877029695, 710.4806049, -12.15769441],
            "attn.v.3": [119.3962592, 761.5206448, 10.2929184],
        }},
}


def load_baseline():
    """The translate and hub modules of the pre-split snapshot, or None."""
    if not (BASELINE / "translate.py").is_file():
        return None
    modules = {}
    for name in ("hub", "translate"):
        spec = importlib.util.spec_from_file_location(f"baseline_{name}", BASELINE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        saved = sys.modules.get(name)
        if name == "translate":
            sys.modules["hub"] = modules["hub"]          # the snapshot's translate sees the snapshot's hub
        try:
            spec.loader.exec_module(module)
        finally:
            if name == "translate":
                sys.modules["hub"] = saved
        modules[name] = module
    return modules


class WholeSections:
    """translate.translate's interface through the whole-section stages compute() and assemble()."""

    Side = translate.Side

    @staticmethod
    def translate(source_state, source, target, hub, *, t0, template, fill="mean", table=None, backend=None,
                  native_state=None, native_prefix=0, indexer_ridge=None):
        backend = backend or backends.select("cpu")
        layers = {kind: translate.source_layers(hub, kind, source.name) for kind in translate.KINDS}
        plan = translate.prepare(source_state, source, target, layers, t0=t0, template=template, fill=fill, table=table,
                                 native_state=native_state, native_prefix=native_prefix,
                                 indexer_ridge=indexer_ridge)
        return translate.assemble(plan, translate.compute(plan, hub, backend, table=table), backend), {}


class TranslateTest(unittest.TestCase):
    def setUp(self):
        # one chat segment "hello world foo bar" inside different template text in each model
        content = "hello world foo bar"
        self.s_text = "<s>" + content + "</s>"
        self.t_text = "[T]" + content + "[/T]"
        start = 3
        seg = [[0, start, start + len(content)]]
        self.s_rec = record(self.s_text, seg, ["1:content"])
        self.t_rec = record(self.t_text, seg, ["1:content"])
        word_ends = [start + 5, start + 11, start + 15, start + 19]          # hello| world| foo| bar
        self.s_tok = tokens_for(self.s_text, [3] + word_ends + [len(self.s_text)], special=(0, 5))
        # the target splits "world" differently: "wor" + "ld" -> one unmatched token
        t_cuts = [3, start + 5, start + 9, start + 11, start + 15, start + 19, len(self.t_text)]
        self.t_tok = tokens_for(self.t_text, t_cuts, special=(0, 6))
        rng = np.random.default_rng(1)
        n = len(self.s_tok["token_id"])
        self.s_pre = {l: rng.standard_normal((n, 1, 8)).astype(np.float32) for l in (0, 1)}
        self.s_v = {l: rng.standard_normal((n, 1, 8)).astype(np.float32) for l in (0, 1)}
        self.source_state = kvstate.parse_file(kvstate.build_dense_state(
            SOURCE, self.s_tok["token_id"], np.arange(n), self.s_pre, self.s_v, dtype="f32").to_bytes(), SOURCE)
        m = len(self.t_tok["token_id"])
        self.t_native_k = {l: rng.standard_normal((m, 2, 4)).astype(np.float32) for l in (0, 1, 2)}
        self.t_native_v = {l: rng.standard_normal((m, 2, 4)).astype(np.float32) for l in (0, 1, 2)}
        self.native = kvstate.parse_file(kvstate.build_dense_state(
            TARGET, self.t_tok["token_id"], np.arange(m), self.t_native_k, self.t_native_v, dtype="f32").to_bytes(),
            TARGET)
        self.hubs = linear_hub()

    def build(self, **options):
        return translate.translate(self.source_state, translate.Side("src", SOURCE, self.s_rec, self.s_tok),
                                   translate.Side("tgt", TARGET, self.t_rec, self.t_tok), self.hubs,
                                   t0=len(self.t_tok["token_id"]) - 1, template=self.native, **options)

    def stored(self, state, kind, layer):
        stream = state.state.part("kv").stream()
        values = np.asarray((stream.k if kind == "k" else stream.v)[layer], dtype=np.float32)
        positions = np.asarray(stream.positions, dtype=np.int64)
        if kind == "k":
            values = kvstate.pre_rope_keys(TARGET, layer, values, positions)
        return values.reshape(len(positions), -1)

    def test_mapped_fill_and_rotation(self):
        state, report = self.build()
        state = kvstate.parse_file(state.to_bytes(), TARGET)
        # aligned (same segment, same end byte): hello, "ld" (ends where "world" ends), foo, bar;
        # fill: the template token and the unmatched "wor"
        self.assertEqual((report["mapped_cells"], report["fill_cells"]), (4, 2))
        # target positions of aligned tokens and their source positions
        pairs = {1: 1, 3: 2, 4: 3, 5: 4}
        for kind in hub.KINDS:
            source_rows = self.s_pre if kind == "k" else self.s_v
            for layer in (0, 1, 2):
                stored = self.stored(state, kind, layer)
                for t_pos, s_pos in pairs.items():
                    x = np.concatenate([source_rows[0][s_pos].ravel(), source_rows[1][s_pos].ravel()])
                    expected = (x @ self.hubs[kind].encoders["src"]) @ self.hubs[kind].decoders["tgt"][layer] \
                        + self.hubs[kind].decoder_mean["tgt"][layer]
                    np.testing.assert_allclose(stored[t_pos], expected, atol=2e-4)
                for t_pos in (0, 2):                                              # fill: the decoder mean
                    np.testing.assert_allclose(stored[t_pos], self.hubs[kind].decoder_mean["tgt"][layer], atol=2e-4)

    def test_native_prefix_is_copied_exactly(self):
        state, report = self.build(native_state=self.native, native_prefix=2)
        self.assertEqual(report["native_prefix"], 2)
        stream = state.state.part("kv").stream()
        native = self.native.state.part("kv").stream()
        for layer in (0, 1, 2):
            np.testing.assert_array_equal(np.asarray(stream.k[layer][:2]), np.asarray(native.k[layer][:2]))
            np.testing.assert_array_equal(np.asarray(stream.v[layer][:2]), np.asarray(native.v[layer][:2]))
        with self.assertRaisesRegex(ValueError, "native prefix needs"):
            self.build(native_prefix=2)

    def test_native_mask_copies_only_those_cells(self):
        t0 = len(self.t_tok["token_id"]) - 1
        mask = np.zeros(t0, dtype=bool)
        mask[[1, 3]] = True
        plain, _ = self.build()
        state, _ = self.build(native_state=self.native, native_mask=mask)
        stream, native, base = (s.state.part("kv").stream() for s in (state, self.native, plain))
        for layer in (0, 1, 2):
            for kind in ("k", "v"):
                got, want, before = (np.asarray(getattr(s, kind)[layer]) for s in (stream, native, base))
                np.testing.assert_array_equal(got[[1, 3]], want[[1, 3]])
                keep = [p for p in range(t0) if not mask[p]]
                np.testing.assert_array_equal(got[keep], before[keep])
        with self.assertRaisesRegex(ValueError, "native mask needs"):
            self.build(native_mask=mask)
        with self.assertRaisesRegex(ValueError, "must cover"):
            self.build(native_state=self.native, native_mask=mask[:2])

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "t0 must lie"):
            translate.translate(self.source_state, translate.Side("src", SOURCE, self.s_rec, self.s_tok),
                                translate.Side("tgt", TARGET, self.t_rec, self.t_tok), self.hubs, t0=0,
                                template=self.native)
        with self.assertRaisesRegex(ValueError, "token table"):
            self.build(fill="table")


class SectionsTest(unittest.TestCase):
    """Every target state family, against the pre-split algebra and (when present) the old module."""

    def setUp(self):
        # one chat segment "hello world foo bar" inside different template text in each model
        content = "hello world foo bar"
        s_text, t_text, start = "<s>" + content + "</s>", "[T]" + content + "[/T]", 3
        seg = [[0, start, start + len(content)]]
        self.s_rec, self.t_rec = record(s_text, seg, ["1:content"]), record(t_text, seg, ["1:content"])
        word_ends = [start + 5, start + 11, start + 15, start + 19]
        self.s_tok = tokens_for(s_text, [3] + word_ends + [len(s_text)], special=(0, 5))
        t_cuts = [3, start + 2, start + 5, start + 9, start + 11, start + 13, start + 15, start + 19, len(t_text)]
        self.t_tok = tokens_for(t_text, t_cuts, special=(0, 8))
        self.rng = np.random.default_rng(3)

    def build(self, source_spec, target_spec, *, subsets=None, dtype="f32", v_trans=False, hub_kind=None,
              module=translate, hub_override=None, **options):
        source_state, pre, values = native_state(source_spec, self.s_tok["token_id"], self.rng)
        template, _, _ = native_state(target_spec, self.t_tok["token_id"], self.rng, dtype=dtype, v_trans=v_trans)
        hubs = random_hub({"src": kind_widths(source_spec), "tgt": kind_widths(target_spec)},
                          subsets=subsets, hub_kind=hub_kind)
        # an MLA target's latent norm weights (what translate reads from its GGUF), from their own generator
        self.norms = latent_norm_weights(target_spec) if target_spec.kv_lora is not None else None
        target_options = {} if self.norms is None else {"norms": self.norms}
        state, report = module.translate(source_state, module.Side("src", source_spec, self.s_rec, self.s_tok),
                                         module.Side("tgt", target_spec, self.t_rec, self.t_tok, **target_options),
                                         hubs if hub_override is None else hub_override,
                                         t0=len(self.t_tok["token_id"]) - 1, template=template, **options)
        return state, report, hubs, {"k": pre, "v": values}

    def stored(self, state, spec, part, kind, layer, role="attention"):
        stream = state.state.part(part).stream()
        values = np.asarray((stream.k if kind == "k" else stream.v)[layer], dtype=np.float32)
        if kind == "k":
            values = kvstate.pre_rope_keys(spec, layer, values, np.asarray(stream.positions, dtype=np.int64), role=role)
        return np.asarray(stream.positions), values.reshape(len(values), -1)

    def expected_row(self, hubs, rows, pairs, kind, spec, layer, position):
        """The decoded row (or the decoder mean) a target cell must hold before RoPE; an MLA K row
        with its c_kv put back on the RMS shell."""
        if position in pairs:
            source_rows = {l: rows[kind][l][pairs[position]].reshape(1, -1) for l in rows[kind]}
            expected = reference_rows(hubs, kind, "src", "tgt", source_rows, layer)[0]
        else:
            expected = hubs[kind].decoder_mean["tgt"][hubs[kind].transforms["tgt"].layers.index(layer)]
        expected = np.array(expected, dtype=np.float64)
        if kind == "k" and spec.k_only(layer):
            lora, (weight, eps) = spec.kv_lora[layer], self.norms[layer]
            expected[:lora] = kvstate.mla_renormalize_latent(expected[None, :lora], weight, eps)[0]
        return expected

    def check(self, source_spec, target_spec, subsets=None, **options):
        state, report, hubs, rows = self.build(source_spec, target_spec, subsets=subsets, **options)
        state = kvstate.parse_file(state.to_bytes(), target_spec)
        alignment = handoff.align_document({**self.s_rec, "document_id": "doc"}, self.s_tok, self.t_rec, self.t_tok)
        pairs = {int(t): int(s) for t, s in zip(alignment.tgt_position, alignment.src_position)
                 if alignment.tgt_class[t] == translate.ALIGNED}
        t0 = len(self.t_tok["token_id"]) - 1
        self.assertEqual(report["t0"], t0)
        decoded = {}
        for part, layout in zip(state.state.parts, target_spec.layouts(state.state.flags)):
            if isinstance(part, kvstate.RecurrentState):
                self.assertEqual(part.positions.tolist(), [t0 - 1])
                self.assertTrue(all(rows.data == bytes(rows.row_bytes) for rows in list(part.r.values())
                                    + list(part.s.values())))
                continue
            sliding = target_spec.layer_types[part.layers[0]] == "sliding"
            expected_positions = list(range(t0 - target_spec.n_swa - 1 if sliding else 0, t0))
            if layout.role == "indexer":            # filled from the K rows: test_dsa_indexer_keys
                self.assertEqual(part.stream().positions.tolist(), expected_positions)
                continue
            decoded[part.name] = sum(p in pairs for p in expected_positions)
            for kind in hub.KINDS:
                for layer in part.layers:
                    if target_spec.kv_shape(layer, kind) is None:
                        self.assertNotIn(layer, part.stream().v)          # MLA: V is the c_kv view of K
                        continue
                    positions, got = self.stored(state, target_spec, part.name, kind, layer)
                    self.assertEqual(positions.tolist(), expected_positions)
                    if target_spec.layer_types[layer] == "nextn":
                        self.assertFalse(got.any(), "nextn layers hold the zero rows of a native save")
                        continue
                    for row, position in zip(got, positions):
                        expected = self.expected_row(hubs, rows, pairs, kind, target_spec, layer, position)
                        np.testing.assert_allclose(row, expected, rtol=1e-4, atol=1e-4 * np.abs(expected).max(),
                                                   err_msg=f"{part.name} {kind} layer {layer} position {position}")
        self.assertEqual(report["decoded_rows"], decoded)
        self.assertEqual(report["mapped_cells"] + report["fill_cells"], report["cells"])
        return state, report

    def test_dense_to_dense(self):
        self.check(SOURCE, TARGET)

    def test_dense_to_iswa_decodes_sliding_sections_only_for_their_cells(self):
        state, report = self.check(SOURCE, MUSE)
        self.assertEqual(report["sections"], {"base": 8, "swa": 4})
        self.assertLess(report["decoded_rows"]["swa"], report["decoded_rows"]["base"])

    def test_dense_to_hybrid_resets_the_recurrent_state(self):
        _, report = self.check(SOURCE, HYBRID)
        self.assertTrue(report["recurrent_reset"])

    def test_hybrid_target_recurrent_strategies(self):
        # zero (R0) by default; the mean of native cells (recurrent_init) of the chosen group; the cell of a
        # native state of the prompt's start, never of a longer one (whose cell would be the answer)
        import recurrent_init
        source_state, _, _ = native_state(SOURCE, self.s_tok["token_id"], self.rng)
        template, _, _ = native_state(HYBRID, self.t_tok["token_id"], self.rng)
        hubs = random_hub({"src": kind_widths(SOURCE), "tgt": kind_widths(HYBRID)})
        source, target = (translate.Side("src", SOURCE, self.s_rec, self.s_tok),
                          translate.Side("tgt", HYBRID, self.t_rec, self.t_tok))

        def run(**options):
            return translate.translate(source_state, source, target, hubs, t0=8, template=template, **options)
        zero, report = run()
        self.assertEqual(report["recurrent"], {"strategy": "zero"})
        reset = kvstate.recurrent_reset(template.state.part("recurrent"), 7)
        self.assertEqual(kvstate._write_recurrent(HYBRID, zero.state.part("recurrent")),
                         kvstate._write_recurrent(HYBRID, reset))
        rng = np.random.default_rng(2)
        layout = template.state.part("recurrent")

        def cell():
            return recurrent_init.section(layout, 0, {key: {l: rng.standard_normal(size // 4)
                                                            for l, size in recurrent_init.row_sizes(
                                                                recurrent_init.Layout.of(layout), key)}
                                                      for key in ("r", "s")})
        mean = recurrent_init.build_mean([(cell(), "chat"), (cell(), "chat"), (cell(), "tools")])
        state, report = run(recurrent="mean", recurrent_mean=mean, recurrent_group="chat")
        self.assertEqual(report["recurrent"], {"strategy": "mean", "group": "chat"})
        part = state.state.part("recurrent")
        self.assertEqual(part.positions.tolist(), [7])
        got = recurrent_init.cell_rows(part)
        for key in ("r", "s"):
            for layer, values in mean.rows("chat")[key].items():
                np.testing.assert_array_equal(got[key][layer], values)
        # the attention cells do not depend on the strategy
        self.assertEqual(state.state.part("attn").stream().k[1].tobytes(),
                         zero.state.part("attn").stream().k[1].tobytes())
        for options, message in (({"recurrent": "mean"}, "needs a recurrent mean"),
                                 ({"recurrent": "mean", "recurrent_mean": mean, "recurrent_group": "agents"},
                                  "no group 'agents'"),
                                 ({"recurrent": "spiral"}, "unknown recurrent strategy"),
                                 ({"recurrent": "prefix"}, "needs a native state of the prompt's start")):
            with self.subTest(**{k: v for k, v in options.items() if k != "recurrent_mean"}), \
                    self.assertRaisesRegex(ValueError, message):
                run(**options)
        other = recurrent_init.build_mean([recurrent_init.section(
            kvstate.RecurrentState("recurrent", (0,), np.asarray([0], np.int32), ((),), 4,
                                   {0: kvstate.OpaqueRows(0, 16, bytes(16))},
                                   {0: kvstate.OpaqueRows(0, 32, bytes(32))}),
            0, {})])
        with self.assertRaisesRegex(ValueError, "another recurrent layout"):
            run(recurrent="mean", recurrent_mean=other)
        # prefix: a native state of the prompt's first 3 tokens gives its last cell
        prefix, _, _ = native_state(HYBRID, self.t_tok["token_id"][:3], self.rng)
        state, report = run(recurrent="prefix", native_state=prefix, native_prefix=3)
        self.assertEqual(report["recurrent"], {"strategy": "prefix"})
        self.assertEqual(recurrent_init.state_distance(prefix, state)["s"]["relative"], 0.0)
        with self.assertRaisesRegex(ValueError, "holds 9 tokens, more than the native prefix of 3"):
            run(recurrent="prefix", native_state=template, native_prefix=3)
        # a target without a recurrent section takes any strategy and reports none
        _, report = translate.translate(source_state, source, translate.Side("tgt", TARGET, self.t_rec, self.t_tok),
                                        random_hub({"src": kind_widths(SOURCE), "tgt": kind_widths(TARGET)}), t0=8,
                                        template=native_state(TARGET, self.t_tok["token_id"], self.rng)[0],
                                        recurrent="mean")
        self.assertNotIn("recurrent", report)

    def test_sliding_window_source_encodes_its_global_layers(self):
        self.check(MUSE, TARGET, subsets={"src": list(MUSE.layers_of("full"))})

    def test_hybrid_to_iswa(self):
        self.check(HYBRID, MUSE)

    # -- MLA, DSA and K-only hybrids ---------------------------------------------------------

    def test_dense_to_mla_decodes_only_k_and_puts_latents_on_their_shell(self):
        state, report = self.check(SOURCE, DEEPSEEK2)
        self.assertEqual((report["kinds"], report["latent_layers"], report["sections"]), (["k"], 2, {"kv": 8}))
        stream = state.state.part("kv").stream()
        self.assertEqual(stream.v, {})                                     # no V blocks at all
        positions = np.asarray(stream.positions, dtype=np.int64)
        for layer in (0, 1):
            c_kv = kvstate.mla_split(DEEPSEEK2, layer, stream.k[layer])[0]
            weight, _ = self.norms[layer]
            # RMSNorm(x) * g: RMS(c_kv / g) = 1 for every cell
            np.testing.assert_allclose(np.sqrt(np.mean((c_kv / weight) ** 2, axis=-1)), 1.0, rtol=1e-5)
            # only the k_pe tail rotates (YaRN): the latent is stored as decoded
            pre = kvstate.pre_rope_keys(DEEPSEEK2, layer, stream.k[layer], positions)
            np.testing.assert_array_equal(pre[..., :8], c_kv)
            self.assertGreater(np.abs(pre[..., 8:] - stream.k[layer][..., 8:]).max(), 1e-3)
        self.assertEqual(DEEPSEEK2.rope(0).ext_factor, 1.0)

    def test_mla_sources_encode_their_latent_as_v(self):
        self.check(DEEPSEEK2, TARGET)                     # V kind: the c_kv view, K: k_pe de-rotated (YaRN)
        self.check(KDA, MUSE)                             # NoPE rows: nothing de-rotated
        self.check(DOTS, TARGET, subsets={"src": list(DOTS.layers_of("full"))})   # global layers only
        self.check(DEEPSEEK2, DOTS)

    def test_the_states_readings_set_the_layouts(self):
        # specs of the current revision, states of 85d5703a3 (and the other way round): the template's
        # reading is the layout written, the source state's the RoPE undone
        current = DEEPSEEK2.for_revision("a7b94df2c")
        for side_spec, template_spec in ((current, DEEPSEEK2), (DEEPSEEK2, current)):
            with self.subTest(template=template_spec.llama_revision):
                source_state, _, _ = native_state(template_spec, self.s_tok["token_id"], self.rng)
                template, _, _ = native_state(template_spec, self.t_tok["token_id"], self.rng)
                spec = kvstate.parse_file(template.to_bytes(), side_spec).state.spec
                self.assertEqual(spec, template_spec)
                self.assertEqual(translate.state_side(translate.Side("tgt", side_spec, self.t_rec, self.t_tok),
                                                      template).spec, template_spec)
                hubs = random_hub({"src": kind_widths(side_spec), "tgt": kind_widths(side_spec)})
                state, _ = translate.translate(
                    source_state, translate.Side("src", side_spec, self.s_rec, self.s_tok),
                    translate.Side("tgt", side_spec, self.t_rec, self.t_tok, norms=latent_norm_weights(side_spec)),
                    hubs, t0=len(self.t_tok["token_id"]) - 1, template=template)
                self.assertEqual(state.state.spec, template_spec)
                parsed = kvstate.parse_file(state.to_bytes(), side_spec)
                self.assertEqual(parsed.state.spec, template_spec)
                stream = parsed.state.part("kv").stream()
                self.assertEqual(sorted(stream.k), [0, 1, 2] if template_spec.nextn_cached else [0, 1])
                if template_spec.nextn_cached:
                    self.assertFalse(stream.k[2].any())
        with self.assertRaisesRegex(ValueError, "another model's layout"):
            translate.state_side(translate.Side("tgt", DSA, self.t_rec, self.t_tok), template)
        # a native prefix of the other revision than the template's is refused
        other, _, _ = native_state(DEEPSEEK2, self.t_tok["token_id"], self.rng)
        with self.assertRaisesRegex(ValueError, "different llama.cpp revisions"):
            translate.translate(source_state, translate.Side("src", current, self.s_rec, self.s_tok),
                                translate.Side("tgt", current, self.t_rec, self.t_tok,
                                               norms=latent_norm_weights(current)),
                                hubs, t0=len(self.t_tok["token_id"]) - 1, template=template, native_state=other,
                                native_prefix=2)

    def test_k_only_hybrid_target_resets_the_recurrent_rows(self):
        state, report = self.check(HYBRID, KDA)
        self.assertTrue(report["recurrent_reset"])
        self.assertEqual([part.name for part in state.state.parts], ["attn", "recurrent"])
        stream = state.state.part("attn").stream()
        # NoPE: the stored rows are the pre-RoPE rows
        np.testing.assert_array_equal(kvstate.pre_rope_keys(KDA, 0, stream.k[0], stream.positions), stream.k[0])

    def test_dsa_iswa_target(self):
        state, report = self.check(SOURCE, DOTS)
        self.assertEqual(report["sections"], {"base": 8, "indexer": 8, "swa": 4})
        self.assertEqual(report["decoded_rows"].keys(), {"base", "swa"})
        self.assertEqual(state.state.part("swa").stream().k[1].shape, (4, 1, 10))
        self.assertFalse(any(np.asarray(k).any() for k in state.state.part("indexer").stream().k.values()))

    def test_dsa_indexer_keys(self):
        # the prompt (P = 9: t0 = 8 and one native tail token) <= top_k 9: zeros, exact whatever the keys
        # while every query sees at most top_k cells
        state, report = self.check(SOURCE, DSA)
        self.assertEqual((report["indexer"], report["sections"]), ("zeros", {"kv": 8, "indexer": 8}))
        stream = state.state.part("indexer").stream()
        self.assertEqual((sorted(stream.k), stream.v), ([0, 1], {}))
        self.assertFalse(any(np.asarray(k).any() for k in stream.k.values()))
        # a prompt above top_k without a ridge: refused (the bridge skips the switch), also when T0 fits
        # (top_k 8): the tail's queries would rank the translated cells by their keys
        for spec, top_k in ((DSA_TOP8, 8), (DSA_TOP4, 4)):
            refusal = f"no indexer keys for tgt: its prompt of 9 tokens \\(T0 = 8\\) exceeds .* top_k \\({top_k}\\)"
            with self.assertRaisesRegex(translate.IndexerKeysUnavailable, refusal):
                self.build(SOURCE, spec)
        # no translated cell (a native prefix covers [0, t0)): nothing to refuse, the native keys stay
        source_state, _, _ = native_state(SOURCE, self.s_tok["token_id"], self.rng)
        native, _, _ = native_state(DSA_TOP4, self.t_tok["token_id"], self.rng)
        hubs = random_hub({"src": kind_widths(SOURCE), "tgt": kind_widths(DSA_TOP4)})
        state, report = translate.translate(
            source_state, translate.Side("src", SOURCE, self.s_rec, self.s_tok),
            translate.Side("tgt", DSA_TOP4, self.t_rec, self.t_tok, norms=latent_norm_weights(DSA_TOP4)), hubs, t0=8,
            template=native, native_state=native, native_prefix=8)
        self.assertEqual((report["indexer"], report["native_prefix"]), ("native", 8))
        for layer in (0, 1):
            np.testing.assert_array_equal(state.state.part("indexer").stream().k[layer],
                                          native.state.part("indexer").stream().k[layer][:8])
        # with a ridge (explicit, or the hub's "indexer" entry): H . RoPE(ridge(decoded pre-RoPE K row))
        ridge = random_indexer_ridge(DSA_TOP4)
        for options in ({"indexer_ridge": ridge}, {"hub_override": "indexer"}):
            self.rng = np.random.default_rng(3)
            if options.get("hub_override") == "indexer":
                hubs = random_hub({"src": kind_widths(SOURCE), "tgt": kind_widths(DSA_TOP4)})
                options = {"hub_override": {**hubs, "indexer": {"tgt": ridge}}}
            state, report, _, _ = self.build(SOURCE, DSA_TOP4, **options)
            self.assertEqual(report["indexer"], "ridge")
            state = kvstate.parse_file(state.to_bytes(), DSA_TOP4)
            for layer in (0, 1):
                positions, keys = self.stored(state, DSA_TOP4, "kv", "k", layer)
                _, indexer = self.stored(state, DSA_TOP4, "indexer", "k", layer, role="indexer")
                np.testing.assert_allclose(indexer, ridge.predict(layer, keys), rtol=1e-4, atol=1e-4)
        with self.assertRaisesRegex(ValueError, "lacks indexer layers \\[1\\]"):
            partial = hub.IndexerRidge("tgt", [0], ridge.x_means[:1], ridge.weights[:1], ridge.y_means[:1])
            self.build(SOURCE, DSA_TOP4, indexer_ridge=partial)

    def test_indexer_ridge_needs_decoded_rows_at_its_cells(self):
        # the ridge fills an indexer row from the decoded K row of the same layer and cell: a layout where
        # some indexer row has none (no current architecture) is refused, not left unwritten
        from types import SimpleNamespace
        ridge, target, cells = random_indexer_ridge(DSA_TOP4), translate.Side("tgt", DSA_TOP4, {}, {}), np.arange(8)

        def section(role, decoded, positions=cells):
            return SimpleNamespace(role=role, layers=[0, 1], decoded={"k": decoded, "v": []}, positions=positions)
        indexer = section("indexer", [])
        self.assertIs(translate._indexer_fill(target, [section("attention", [0, 1]), indexer], 8, ridge, 0), ridge)
        for attention in (section("attention", [0]), section("attention", [0, 1], cells[4:])):
            with self.assertRaisesRegex(ValueError, "indexer layer . has no decoded K rows at its cells"):
                translate._indexer_fill(target, [attention, indexer], 8, ridge, 0)

    def test_mla_target_needs_its_latent_norms(self):
        source_state, _, _ = native_state(SOURCE, self.s_tok["token_id"], self.rng)
        template, _, _ = native_state(DEEPSEEK2, self.t_tok["token_id"], self.rng)
        hubs = random_hub({"src": kind_widths(SOURCE), "tgt": kind_widths(DEEPSEEK2)})
        with self.assertRaisesRegex(ValueError, "needs its GGUF"):
            translate.translate(source_state, translate.Side("src", SOURCE, self.s_rec, self.s_tok),
                                translate.Side("tgt", DEEPSEEK2, self.t_rec, self.t_tok), hubs, t0=8,
                                template=template)
        self.assertEqual(translate.target_kinds(DEEPSEEK2), ("k",))
        self.assertEqual(translate.target_kinds(KDA), ("k",))
        self.assertEqual(translate.target_kinds(MUSE), ("k", "v"))

    def test_latent_norms_of_a_split_gguf(self):
        # llama.cpp's split files (PREFIX-NNNNN-of-NNNNN.gguf): the metadata in the first, the tensors spread
        from tiny_fixture import _gguf_module
        gguf = _gguf_module()
        weights = latent_norm_weights(DEEPSEEK2)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"model-{number:05d}-of-00002.gguf" for number in (1, 2)]
            for number, path in enumerate(paths):
                writer = gguf.GGUFWriter(path, "deepseek2")
                writer.add_uint16("split.no", number)
                writer.add_uint16("split.count", 2)
                if number == 0:
                    writer.add_float32("deepseek2.attention.layer_norm_rms_epsilon", 1e-6)
                writer.add_tensor(f"blk.{number}.attn_kv_a_norm.weight", weights[number][0])
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
            norms = translate.latent_norms(translate.Side("tgt", DEEPSEEK2, self.t_rec, self.t_tok, gguf=paths[0]))
            self.assertEqual(sorted(norms), [0, 1])
            for layer in (0, 1):
                np.testing.assert_array_equal(norms[layer][0], weights[layer][0])
                self.assertAlmostEqual(norms[layer][1], 1e-6)
            self.assertEqual(translate.gguf_shards(paths[1]), [str(path) for path in paths])     # from any file
            paths[1].unlink()
            with self.assertRaisesRegex(ValueError, "missing split GGUF files"):
                translate.gguf_shards(paths[0])
            renamed = paths[0].with_name("model.gguf")
            paths[0].rename(renamed)
            with self.assertRaisesRegex(ValueError, "must be named PREFIX-NNNNN-of-00002.gguf"):
                translate.gguf_shards(renamed)

    def test_windows_match_whole_sections(self):
        class Table:
            def lookup(self, kind, layer, ids):
                rows = np.stack([np.full((2, 4), float(i) + layer, dtype=np.float32) for i in ids])
                return rows, np.zeros(len(ids), dtype=bool)
        cases = [(SOURCE, TARGET, None, {}), (SOURCE, MUSE, None, {}), (SOURCE, HYBRID, None, {}),
                 (MUSE, HYBRID, {"src": [2, 5]}, {}), (SOURCE, TARGET, None, {"fill": "table", "table": Table()}),
                 (SOURCE, DEEPSEEK2, None, {}), (DEEPSEEK2, KDA, None, {}), (SOURCE, DOTS, None, {}),
                 (SOURCE, DSA_TOP4, None, {"indexer_ridge": random_indexer_ridge(DSA_TOP4)})]
        for source_spec, target_spec, subsets, options in cases:
            for dtype in ("f32", "f16"):
                self.rng = np.random.default_rng(9)
                whole, _, _, _ = self.build(source_spec, target_spec, subsets=subsets, dtype=dtype,
                                            module=WholeSections, **options)
                expected = kvstate.parse_file(whole.to_bytes(), target_spec)
                for window in (1, 3, None):         # windows splitting sections, the sliding section's start, none
                    labels = {key: value for key, value in options.items() if key != "indexer_ridge"}
                    with self.subTest(target=target_spec.architecture, dtype=dtype, window=window, **labels):
                        self.rng = np.random.default_rng(9)
                        state, _, _, _ = self.build(source_spec, target_spec, subsets=subsets, dtype=dtype,
                                                    window_rows=window, chunk_rows=2, **options)
                        got = kvstate.parse_file(state.to_bytes(), target_spec)
                        self.assertEqual(len(state.to_bytes()), len(whole.to_bytes()))
                        for a, b in zip(got.state.parts, expected.state.parts):
                            if isinstance(a, kvstate.RecurrentState):
                                continue
                            sa, sb = a.stream(), b.stream()
                            np.testing.assert_array_equal(sa.positions, sb.positions)
                            self.assertEqual(sorted(sa.v), sorted(sb.v))            # K-only layers: no V
                            for layer in a.layers:
                                pairs = [(sa.k[layer], sb.k[layer])] + ([(sa.v[layer], sb.v[layer])] if layer in sb.v
                                                                        else [])
                                for x, y in pairs:
                                    # float32 rounding of other product shapes; in float16 at most one step
                                    tolerance = 1e-5 if dtype == "f32" else 1e-3
                                    np.testing.assert_allclose(np.asarray(x, np.float32), np.asarray(y, np.float32),
                                                               rtol=tolerance, atol=tolerance * float(np.abs(y).max()))

    def test_prefaulted_output_pages_give_the_same_state(self):
        # a backend whose output pages are touched ahead (Prefault, the GPU backend's setting): every
        # (section, kind) waits for its touch before its first rows land, so nothing is overwritten
        class Prefaulting(backends.select("cpu").__class__):
            prefault_output = True
        touched = []
        real_touch = translate._touch

        def touch(array):
            touched.append(array.shape)
            real_touch(array)
        source_state, _, _ = native_state(SOURCE, self.s_tok["token_id"], self.rng)
        for target_spec, options in ((TARGET, {}), (MUSE, {}), (HYBRID, {}),
                                     (DSA_TOP4, {"indexer_ridge": random_indexer_ridge(DSA_TOP4)})):
            states = []
            for backend in (backends.select("cpu"), Prefaulting()):
                self.rng = np.random.default_rng(5)
                with mock.patch.object(translate, "_touch", touch):
                    state, report, _, _ = self.build(SOURCE, target_spec, dtype="f16", backend=backend, window_rows=3,
                                                     **options)
                states.append(state.to_bytes())
            with self.subTest(target=target_spec.architecture):
                self.assertEqual(states[0], states[1])
                # one touch per K/V array of the prefaulting run (none for the indexer keys)
                self.assertEqual(len(touched), len(target_spec.layers_of("full", "sliding"))
                                 * len(translate.target_kinds(target_spec)))
            touched.clear()
        # with a native prefix copied into the same arrays after the windows
        template, _, _ = native_state(TARGET, self.t_tok["token_id"], self.rng)
        hubs = random_hub({"src": kind_widths(SOURCE), "tgt": kind_widths(TARGET)})
        out = []
        for backend in (backends.select("cpu"), Prefaulting()):
            state, report = translate.translate(source_state, translate.Side("src", SOURCE, self.s_rec, self.s_tok),
                                                translate.Side("tgt", TARGET, self.t_rec, self.t_tok), hubs, t0=8,
                                                template=template, native_state=template, native_prefix=3,
                                                backend=backend, window_rows=2)
            out.append(state.to_bytes())
        self.assertEqual(out[0], out[1])

    def test_a_failed_translation_drops_its_pending_touches(self):
        # the backend fails while PREFAULT_THREADS touches run and the rest wait: those are dropped
        # (an engine retrying on numpy would otherwise see the failed attempt's arrays faulted in)
        class Failing(backends.select("cpu").__class__):
            prefault_output = True

            def encode(self, *args, **kwargs):
                with running:                           # every pool thread holds a touch
                    running.wait_for(lambda: len(touched) >= translate.PREFAULT_THREADS, timeout=10)
                threading.Timer(0.5, gate.set).start()  # cancel waits for the running ones
                raise FloatingPointError("float16 overflow")
        running, gate, touched = threading.Condition(), threading.Event(), []

        def touch(array):
            with running:
                touched.append(array.shape)
                running.notify_all()
            gate.wait(10)
        with mock.patch.object(translate, "_touch", touch), self.assertRaisesRegex(FloatingPointError, "overflow"):
            self.build(SOURCE, TARGET, dtype="f16", backend=Failing())
        self.assertTrue(gate.is_set())
        self.assertEqual(len(touched), translate.PREFAULT_THREADS)
        self.assertGreater(len(TARGET.layers_of("full")) * len(translate.target_kinds(TARGET)), len(touched))

    def test_cache_dtype_and_v_layout_follow_the_template(self):
        for dtype in ("f16", "f32"):
            for v_trans in (False, True):
                state, _, _, _ = self.build(SOURCE, MUSE, dtype=dtype, v_trans=v_trans)
                parsed = kvstate.parse_file(state.to_bytes(), MUSE)
                for part in parsed.state.parts:
                    stream = part.stream()
                    self.assertEqual(stream.v_trans, v_trans)
                    self.assertEqual(set(stream.k_types.values()), {kvstate.DTYPE_NAMES[dtype]})

    def test_table_fill_renormalizes_keys(self):
        class Table:
            def lookup(self, kind, layer, ids):
                rows = np.tile(np.arange(1, 9, dtype=np.float32).reshape(2, 4), (len(ids), 1, 1)) * (layer + 1)
                return rows, np.zeros(len(ids), dtype=bool)
        state, report, hubs, _ = self.build(SOURCE, TARGET, fill="table", table=Table())
        self.assertGreater(report["fill_cells"], 0)
        _, k = self.stored(state, TARGET, "kv", "k", 1)
        _, v = self.stored(state, TARGET, "kv", "v", 1)
        filled = np.tile(np.arange(1, 9, dtype=np.float32).reshape(2, 4) * 2, (1, 1))
        np.testing.assert_allclose(v[0], filled.ravel(), rtol=1e-6)                 # position 0 is a fill cell
        np.testing.assert_allclose(k[0], handoff.renormalize_heads(filled[None])[0].ravel(), rtol=1e-5)

    def test_kv_stream_matches_kvstate_builder(self):
        positions = np.arange(3, 9)
        for spec, layers in ((MUSE, MUSE.layers_of("sliding")), (HYBRID, HYBRID.layers_of("full"))):
            k = {l: self.rng.standard_normal((6,) + spec.head_shape(l, "k")).astype(np.float16) for l in layers}
            v = {l: self.rng.standard_normal((6,) + spec.head_shape(l, "v")).astype(np.float16) for l in layers}
            tokens = np.arange(6) + 40
            for v_trans in (False, True):
                ours = translate.kv_stream(spec, layers, positions, k, v, dtype="f16", v_trans=v_trans, tokens_at=tokens)
                theirs = kvstate.build_kv_stream(spec, layers, positions, k, v, dtype="f16", v_trans=v_trans,
                                                 tokens_at=tokens)
                layout = kvstate.SectionLayout("x", "kv", tuple(layers), ext=spec.has_cell_ext())
                cache = lambda stream: kvstate._write_kv(spec, kvstate.KVCache("x", tuple(layers), [stream]), layout)  # noqa: E731
                self.assertEqual(cache(ours), cache(theirs))

    def test_hub_bundles_translate_like_hub_kinds(self):
        import tempfile
        import convert_hub
        state, _, hubs, _ = self.build(SOURCE, MUSE, subsets=None)
        with tempfile.TemporaryDirectory() as directory:
            views = [StubView(name, hubs, "qwen3" if name == "src" else "muse-glimmer") for name in ("src", "tgt")]
            hub.save_hub(Path(directory) / "v1", hubs, views, StubSpans(), {"note": "test"})
            convert_hub.convert(Path(directory) / "v1", Path(directory) / "v2", log=lambda line: None)
            for bundle in ("v1", "v2"):
                self.rng = np.random.default_rng(3)
                again, _, _, _ = self.build(SOURCE, MUSE, hub_override=hub.open_hub(Path(directory) / bundle))
                self.assertEqual(again.to_bytes(), state.to_bytes(), bundle)

    def test_outputs_match_the_frozen_pre_split_outputs(self):
        for name, (source_spec, target_spec, subsets) in FROZEN_CASES.items():
            self.rng = np.random.default_rng(5)
            state, report, _, _ = self.build(source_spec, target_spec, subsets=subsets, dtype="f32")
            frozen = FROZEN[name]
            self.assertEqual({key: report[key] for key in frozen["report"]}, frozen["report"], name)
            self.assertEqual(len(state.to_bytes()), frozen["bytes"], name)
            got = digests(kvstate.parse_file(state.to_bytes(), target_spec))
            self.assertEqual(set(got), set(frozen["values"]), name)
            for key, (total, squares, weighted) in frozen["values"].items():
                scale = 16 * np.sqrt(squares)               # bounds both sums of a layer's (at most 256) values
                for value, want, tolerance in zip(got[key], (total, squares, weighted),
                                                  (2e-5 * scale, 2e-5 * squares, 2e-5 * scale)):
                    self.assertLessEqual(abs(value - want), tolerance, f"{name} {key}")

    @unittest.skipUnless((BASELINE / "translate.py").is_file(), "no snapshot of the pre-split translate.py")
    def test_outputs_equal_the_pre_split_module(self):
        baseline = load_baseline()
        for source_spec, target_spec, subsets in ((SOURCE, TARGET, None), (SOURCE, MUSE, None), (SOURCE, HYBRID, None),
                                                  (MUSE, HYBRID, {"src": list(MUSE.layers_of("full"))})):
            outputs = []
            for module, hub_kind in ((translate, hub.HubKind), (baseline["translate"], baseline["hub"].HubKind)):
                self.rng = np.random.default_rng(5)
                state, report, _, _ = self.build(source_spec, target_spec, subsets=subsets, dtype="f32",
                                                 hub_kind=hub_kind, module=module)
                outputs.append((kvstate.parse_file(state.to_bytes(), target_spec), report))
            (new, new_report), (old, old_report) = outputs
            for key in old_report:
                if key != "seconds":
                    self.assertEqual(new_report[key], old_report[key], key)
            self.assertEqual(len(new.to_bytes()), len(old.to_bytes()))
            np.testing.assert_array_equal(new.token_words, old.token_words)
            for a, b in zip(new.state.parts, old.state.parts):
                if isinstance(a, kvstate.RecurrentState):
                    self.assertEqual(kvstate._write_recurrent(target_spec, a), kvstate._write_recurrent(target_spec, b))
                    continue
                sa, sb = a.stream(), b.stream()
                np.testing.assert_array_equal(sa.positions, sb.positions)
                self.assertEqual((sa.ext is None), (sb.ext is None))
                for layer in a.layers:
                    for got, want in ((sa.k[layer], sb.k[layer]), (sa.v[layer], sb.v[layer])):
                        np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5 * np.abs(want).max())


class StubView:
    """What save_hub reads from a view: name, architecture, layers and head shapes."""

    def __init__(self, name, hubs, architecture):
        self.name, self.architecture, self.hubs = name, architecture, hubs

    def layers(self, kind):
        return list(self.hubs[kind].transforms[self.name].layers)

    def shape(self, kind, layer):
        index = self.hubs[kind].transforms[self.name].layers.index(layer)
        return (1, self.hubs[kind].decoders[self.name][index].shape[1])


class StubSpans:
    def select(self, split):
        return np.zeros(0, dtype=np.int64)


if __name__ == "__main__":
    unittest.main()
