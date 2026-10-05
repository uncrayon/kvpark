"""Translate one model's llama.cpp slot state into another model's, through the universal KV hub.

    target_state, report = translate(source_state, source, target, hub, t0=..., template=...)

`source` and `target` describe each side of the same conversation: the model's name in the
hub, its kvstate.ModelSpec, its rendered record (text + chat segments) and its tokens
(token_id / byte_end / special); an MLA target also names its GGUF (the latent norm weights).
The target also brings a *template*: any native slot state of the target model (one token is
enough). Its sections give the target's state layout (section names, roles and layers, cache
dtype, V layout, recurrent row sizes), so no architecture is hard-coded here beyond what
kvstate parses.

Per target cell at position p < t0:

* aligned content (same chat segment, same end byte in both renderings, source token inside
  the source state): K/V decoded from the hub latent of the source token, which is encoded
  from the source's pre-RoPE K (de-rotated with the source's own RoPE) and V;
* every other cell (template, special, unmatched, straddling): `fill` = "table" (per token id
  mean of native target rows, TokenTable) or "mean" (the decoder's train mean);
* K is re-rotated at the target position with the target's RoPE (kvstate.cache_keys).

DeepSeek V4 targets (deepseek4) keep pooled caches that are synthesized from decoded per-token
compressor inputs (dsv4_target, dsv4_pool; the hub's head kinds); translate() dispatches them.

MLA caches (deepseek2 and relatives, kvstate's K-only layers) store one row per cell and
layer, [RMSNorm(c_kv) * g | k_pe], and no V: attention reads the c_kv view of K as V. The
hub's views of such a model are the pre-RoPE K row (K kind) and c_kv (V kind, kvstate.
value_rows). As a source, both kinds are encoded from the stored rows (k_pe de-rotated); as a
target only the K kind is decoded: c_kv is put back on its RMS shell (kvstate.
mla_renormalize_latent with the layer's attn_kv_a_norm weight from the target GGUF) and k_pe
rotated with the target's RoPE (YaRN included, the [c_kv | k_pe] offset; NoPE kimi rows
unrotated). nextn layers a section keeps are written as the zero rows a native save holds.

DSA targets (deepseek32, glm-dsa, hy_v4 with an indexer, dots3note) hold a second K-only
section of lightning-indexer keys. Those only decide which cells attention reads once a query
sees more than indexer top_k cells. The hub's indexer ridge for the target (hub.IndexerRidge:
the pre-RoPE indexer key as a ridge regression on the same layer's decoded pre-RoPE K row)
fills them when there is one (RoPE and the Hadamard rotation by kvstate.cache_keys). Without
one they are written as zeros while the whole target prompt (P = len(target.tokens), the
native tail [t0, P) included) fits in top_k cells: every query of the prompt then reads every
cell whatever the keys, so zeros are exact for it. A longer prompt is refused
(IndexerKeysUnavailable): its later queries would rank the translated cells by zero keys below
the native ones. Tokens generated after the prompt, and later turns, see more cells once the
conversation passes top_k; from there zero keys (and the ridge's) are approximate. qwen4exp's
block indexer is filled alike; on a7b94df2c its rows also hold pooled keys, which are written
zero and re-pooled from the keys on restore (kvstate.ModelSpec.indexer_row_slots). GLM5-Next's
rows hold a per-token pool gate beside the key ([key | gate | pooled]); its ridge predicts both
(kvstate.ModelSpec.indexer_key_slots: what pre_rope_keys gives), and zeros are exact below top_k
as above.

Plain-attention targets of the GQA family (kvstate.GQA_ARCHITECTURES: llama, mistral3, qwen3moe,
hy_v3, phi3; with a sliding window gemma3, gemma4, gpt-oss, laguna, mimo2) need nothing of their
own: their layers' head sizes, KV heads and RoPE (per layer kind: gemma4's proportional RoPE on
its full layers, laguna's plain RoPE on its sliding ones, frequency factors) come from the spec,
their sliding sections are written like any other, and gemma4's shared KV layers keep no cache.

The specs may describe another llama.cpp revision than the states (kvstate.LLAMA_REVISIONS: the
production Qwen3.8 server runs 85d5703a3, the repository's runtime a7b94df2c, and their layouts
differ for deepseek2/mistral4 nextn layers, qwen4exp's indexer and bailingmoe3 with rope sections):
kvstate reads each state as the revision whose layout it has, and the template's reading is the
target layout written, the source state's the RoPE undone (state_side).

Sections are written like the template's: full-attention and indexer sections hold [0, t0);
sliding sections hold the last `swa_cells` positions (default n_swa + 1, which llama-server
needs to reuse the cache); recurrent sections (hybrid targets) hold one cell at position t0 - 1
whose R, S and PLE rows the native tail [t0, P) carries on from: zeros by default (strategy R0,
kvstate.recurrent_reset), or what recurrent_init builds (`recurrent`: a per-model mean of native
cells, RESULTS.md section 7).

Three stages, timed in the report:

  prepare   alignment, the positions each target section keeps, which source row feeds which
            target cell (index arithmetic only), the hub kinds the target needs, the indexer fill;
  compute   encode the mapped source rows into hub latents (fused encoder), decode each section
            only for the positions it keeps (a sliding section decodes its last swa_cells, not
            [0, t0)), fill the rest;
  assemble  MLA latents back on their shell, indexer keys, RoPE at the target positions and the
            cache dtype, the native prefix, the state.

translate() runs compute and the per-row part of assemble window by window of target positions
(compute_cache; window_rows of them, by default the backend's: 4096 on a GPU, 16384 for numpy,
whose BLAS runs 10-20% slower on smaller products): per window the mapped cells are encoded,
each section's rows there decoded, finished (_store_rows) and copied into the section's host
arrays, so the float32 values of a whole section never exist and memory beyond the output stays
flat in t0 (a 13K switch held 1-3 GiB of them; a 262K context would hold 20-60 GiB). With a GPU
backend the output arrays' pages are first touched on a few threads meanwhile (Prefault), not in
its device-to-host copies. compute() and assemble() remain as whole-section stages (accuracy
comparisons between backends use them).

The arithmetic runs on a backend (backends/: numpy float32, the reference and the default;
torch float16 on a GPU). `hub` is {kind: HubKind} (hub.load_hub; an optional "indexer" entry
{model: IndexerRidge}) or a lazily read hub.Hub (hub.open_hub).
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import functools
from pathlib import Path
import re
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "linear_transfer"))

import gguf_tensors  # noqa: E402
import handoff  # noqa: E402
import kvstate  # noqa: E402

import backends  # noqa: E402
from backends.base import cast_to_cache  # noqa: E402
import hub as hub_module  # noqa: E402
import recurrent_init  # noqa: E402

KINDS = ("k", "v")
ALIGNED = handoff.ALIGNED
# llama.cpp's split GGUF files: PREFIX-00001-of-00003.gguf ... (llama_split_path)
SPLIT_NAME = re.compile(r"^(?P<prefix>.+)-(?P<number>\d{5})-of-(?P<count>\d{5})\.gguf$")
PREFAULT_THREADS = 4            # threads touching a GPU translation's fresh output pages (Prefault)
PAGE = 4096
_PREFAULT_POOL = []
_PREFAULT_LOCK = threading.Lock()


class IndexerKeysUnavailable(ValueError):
    """A DSA target whose prompt exceeds the indexer top_k, and no indexer ridge for it in the hub."""


@dataclass
class Side:
    name: str                   # model name in the hub
    spec: object                # kvstate.ModelSpec
    record: dict                # rendered record
    tokens: dict                # token_id, byte_end, special
    gguf: object = None         # the model's GGUF (a split one's first file): an MLA target's latent norm weights
    norms: dict = None          # MLA latent norms {layer: (attn_kv_a_norm weight, rms eps)}; default from gguf


def state_side(side, state):
    """`side` with the spec of a state's reading: kvstate's parsers read a state as the llama.cpp
    revision whose layout it has (kvstate.ModelSpec.layout_variants), so the template's reading is
    the layout to write and a source state's the RoPE to undo. The reading must be side.spec's model,
    on whichever revision; a state without a spec (a capture source) leaves the side as it is."""
    spec = getattr(getattr(state, "state", None), "spec", None)
    if spec is None or spec == side.spec:
        return side
    if spec.for_revision(side.spec.llama_revision) != side.spec:
        raise ValueError(f"{side.name}: the state holds another model's layout than its spec describes")
    return replace(side, spec=spec)


def target_kinds(spec):
    """The hub kinds translating into a model decodes: K, and V unless every attention layer is
    K-only (MLA: V is the c_kv view of K); a DeepSeek V4 target: K (its raw latents) and the head
    kinds of its compressed layers (hub.HEAD_KINDS)."""
    if spec.memory == "dsv4":
        return ("k",) + tuple(pool.name for pool in spec.pools() if pool.layers)
    attention = [layer for layer in spec.attention_layers if spec.layer_types[layer] != "nextn"]
    return tuple(kind for kind in KINDS if any(spec.kv_shape(layer, kind) is not None for layer in attention))


def encoder_kinds(spec):
    """The hub kinds whose source encoders a translation into this model reads: target_kinds but the
    head kinds, which decode the base kind's latent (a hub bundle keeps no encoders of them)."""
    return tuple(kind for kind in target_kinds(spec) if kind not in hub_module.HEAD_KINDS)


def numpy_only(spec):
    """Whether translating into this model runs on numpy whatever the backend: a DeepSeek V4 target
    (dsv4_target pools on the CPU and reads only a numpy backend's resident hub sides)."""
    return spec.memory == "dsv4"


def gguf_shards(path):
    """The files of a GGUF model: [path], or every shard of a split GGUF in order (named like
    llama.cpp's PREFIX-NNNNN-of-NNNNN.gguf, split.count in the metadata), from any one of them."""
    count = int(gguf_tensors.metadata_value(str(path), "split.count", 1) or 1)
    if count <= 1:
        return [str(path)]
    path = Path(path)
    match = SPLIT_NAME.match(path.name)
    if match is None or int(match["count"]) != count:
        raise ValueError(f"{path}: a GGUF split in {count} files must be named PREFIX-NNNNN-of-{count:05d}.gguf")
    shards = [path.with_name(f"{match['prefix']}-{number:05d}-of-{count:05d}.gguf") for number in range(1, count + 1)]
    missing = [shard.name for shard in shards if not shard.is_file()]
    if missing:
        raise ValueError(f"{path}: missing split GGUF files {missing}")
    return [str(shard) for shard in shards]


@functools.lru_cache(maxsize=8)
def _gguf_norms(path, size, mtime_ns, architecture, layers):
    shards = gguf_shards(path)
    # a split GGUF keeps the model's metadata in its first file
    eps = gguf_tensors.metadata_value(shards[0], f"{architecture}.attention.layer_norm_rms_epsilon")
    if eps is None:
        raise ValueError(f"{path}: no {architecture}.attention.layer_norm_rms_epsilon")
    norms = {}
    for layer in layers:
        name = f"blk.{layer}.attn_kv_a_norm.weight"
        shard = next((shard for shard in shards if name in gguf_tensors.reader(shard)[1]), None)
        if shard is None:
            raise ValueError(f"{path}: no tensor {name}")
        norms[layer] = (gguf_tensors.read_tensor(shard, name), float(eps))
    return norms


def latent_norms(side):
    """{layer: (float32 attn_kv_a_norm weight [kv_lora], rms eps)} of a side's MLA attention layers
    (empty for other models): side.norms, else read from side.gguf (any shard of a split GGUF) once
    per file identity."""
    spec = side.spec
    layers = tuple(layer for layer in spec.attention_layers
                   if spec.k_only(layer) and spec.layer_types[layer] != "nextn")
    if not layers:
        return {}
    if side.norms is not None:
        norms = side.norms
    elif side.gguf is None:
        raise ValueError(f"{side.name}: translating into an MLA model needs its GGUF (Side.gguf) for the "
                         "attn_kv_a_norm weights")
    else:
        path = Path(side.gguf).resolve()
        status = path.stat()
        norms = _gguf_norms(str(path), status.st_size, status.st_mtime_ns, spec.architecture, layers)
    for layer in layers:
        if layer not in norms or np.shape(norms[layer][0]) != (spec.kv_lora[layer],):
            raise ValueError(f"{side.name}: layer {layer} needs a latent norm weight of {spec.kv_lora[layer]} values")
    return norms


def attention_parts(state_file):
    """(KVCache, SectionLayout) of a parsed state's attention KV sections (not DSA indexer keys)."""
    state = state_file.state
    return [(part, layout) for part, layout in zip(state.parts, state.spec.layouts(state.flags))
            if layout.kind == "kv" and layout.role == "attention"]


class SourceCache:
    """Pre-RoPE K and V rows of a parsed source state, by source position, for any attention layer
    (an MLA layer's V is the c_kv view of its K rows, kvstate.value_rows)."""

    def __init__(self, state_file, spec):
        self.spec, self.tokens = spec, np.asarray(state_file.tokens)
        self.where = {}
        for part, _ in attention_parts(state_file):
            stream = part.stream()
            index = handoff._stream_index(stream, f"source {part.name} cache")
            for layer in part.layers:
                self.where[layer] = (stream, index)

    def stored(self, kind, layer):
        """The stored rows of one layer and kind (K, V or the c_kv view of an MLA row), all cells."""
        stream, _ = self.where[layer]
        return stream.k[layer] if kind == "k" else kvstate.value_rows(self.spec, stream, layer)

    def rows(self, kind, layer, positions):
        return self.rows_many(kind, [layer], positions)[layer]

    def rows_many(self, kind, layers, positions):
        """{layer: float32 [n, heads * dim]}; K de-rotated, one call per RoPE (layers sharing one are
        concatenated along the heads axis)."""
        positions = np.asarray(positions, dtype=np.int64)
        blocks = {}
        for layer in layers:
            _, index = self.where[layer]
            cells = handoff._cells(index, positions, f"source layer {layer}")
            blocks[layer] = np.asarray(self.stored(kind, layer)[cells], dtype=np.float32)
        if kind == "k":
            groups = {}
            for layer in layers:
                groups.setdefault(self.spec.rope(layer), []).append(layer)
            for group in groups.values():
                heads = [blocks[layer].shape[1] for layer in group]
                stacked = kvstate.pre_rope_keys(self.spec, group[0], np.concatenate([blocks[l] for l in group], axis=1),
                                                positions)
                blocks.update(zip(group, np.split(stacked, np.cumsum(heads)[:-1], axis=1)))
        return {layer: block.reshape(len(positions), -1) for layer, block in blocks.items()}

    def n_positions(self, layers):
        """Positions 0..n-1 held by every given layer (a slot can hold a sampled token without a cell)."""
        n = len(self.tokens)
        for layer in layers:
            stream, _ = self.where[layer]
            positions = np.asarray(stream.positions, dtype=np.int64)
            n = min(n, int(positions.max()) + 1 if len(positions) else 0)
        return n

    def has(self, layer, positions):
        _, index = self.where[layer]
        return np.asarray([int(p) in index for p in positions], dtype=bool)


def _section_positions(layout, t0, swa_cells):
    """Target positions a KV section holds: [0, t0), the last swa_cells of them for a sliding section."""
    start = max(0, t0 - swa_cells) if layout.sliding else 0
    return np.arange(start, t0, dtype=np.int64)


def _decoded_layers(spec, layout, kind):
    """Layers of a section whose `kind` rows the hub decodes: an attention section's layers but the
    nextn ones (zero rows) and, for V, the K-only ones; an indexer section decodes nothing."""
    if layout.role != "attention":
        return []
    return [layer for layer in layout.layers
            if spec.layer_types[layer] != "nextn" and spec.kv_shape(layer, kind, layout.role) is not None]


@dataclass
class Section:
    """One KV section of the target state and where the values of its cells come from."""

    part: object                # the template's kvstate.KVCache
    layout: object              # its kvstate.SectionLayout (role "attention" or "indexer")
    positions: np.ndarray       # int64 target positions it holds, increasing
    mapped: np.ndarray          # rows of the plan's mapped cells (hub latents) it holds
    mapped_rows: np.ndarray     # their rows in this section
    fill_rows: np.ndarray       # its rows without a source token
    decoded: dict = field(default_factory=dict)     # kind -> layers decoded from the hub (_decoded_layers)

    @property
    def name(self):
        return self.part.name

    @property
    def layers(self):
        return list(self.part.layers)

    @property
    def role(self):
        return self.layout.role


@dataclass
class Plan:
    """Everything a translation decides before any hub arithmetic (translate.prepare)."""

    t0: int
    swa_cells: int
    fill: str
    source: Side
    target: Side
    template: object
    cache: SourceCache
    source_positions: np.ndarray    # [mapped cells] source position of each, in target order
    sections: list                  # Section per template KV section, in template order
    cells: int                      # target positions held by any section
    native_state: object
    native_prefix: int
    server_tokens: bool
    mapped_positions: np.ndarray    # [mapped cells] target position of each, increasing
    kinds: tuple = KINDS            # hub kinds the target decodes (target_kinds)
    norms: dict = field(default_factory=dict)       # MLA latent norms of the target (latent_norms)
    indexer: object = None          # DSA targets: "zeros", "native" or the hub.IndexerRidge filling the indexer keys
    recurrent: str = "zero"         # hybrid targets: the recurrent_init strategy of the cell at t0 - 1
    recurrent_mean: object = None   # its recurrent_init.RecurrentMean (strategies reading the mean)
    recurrent_group: str = recurrent_init.ALL       # the mean's group

    @property
    def mapped_cells(self):
        return int(len(self.source_positions))

    @property
    def fill_cells(self):
        return self.cells - self.mapped_cells


def source_layers(hub, kind, name):
    if isinstance(hub, hub_module.Hub):
        return hub.source_layers(kind, name)
    return list(hub[kind].source_layers(name))


def _indexer_fill(target, sections, t0, ridge, native_prefix):
    """The indexer keys of a DSA target: the ridge; zeros while the prompt fits in top_k cells, or
    "native" when a native prefix covers [0, t0); else refused (see the module docstring)."""
    if not any(section.role == "indexer" for section in sections):
        return None
    indexer = target.spec.indexer
    if ridge is not None:
        missing = sorted(set(indexer.layers) - set(ridge.layers))
        if missing:
            raise ValueError(f"the indexer ridge of {target.name} lacks indexer layers {missing}")
        # the ridge fills the indexer rows at the positions of the decoded K rows they come from
        decoded = {layer: section.positions for section in sections if section.role == "attention"
                   for layer in section.decoded["k"]}
        for section in sections:
            for layer in section.layers if section.role == "indexer" else ():
                if layer not in decoded or not np.array_equal(decoded[layer], section.positions):
                    raise ValueError(f"{target.name}: indexer layer {layer} has no decoded K rows at its cells")
        return ridge
    if native_prefix >= t0:
        return "native"
    prompt = len(target.tokens["token_id"])
    if prompt <= indexer.top_k:
        return "zeros"
    raise IndexerKeysUnavailable(
        f"no indexer keys for {target.name}: its prompt of {prompt} tokens (T0 = {t0}) exceeds its DSA indexer "
        f"top_k ({indexer.top_k}), so the indexer keys of the translated cells would decide which cells attention "
        f"reads, and the hub has no indexer ridge for it (fit_hub.py --indexer-ridge {target.name})")


def _recurrent_strategy(template, strategy, mean, group, native_state, native_prefix):
    """The checked name of a hybrid target's recurrent strategy (recurrent_init.STRATEGIES): a mean
    of the template's layout holding `group` when the strategy reads the mean; for "prefix" a native
    state of the prompt's start holding no more than the native prefix (pair_eval passes the native
    save at T0, whose cell would be the answer itself). Not checked for targets without one."""
    rule = recurrent_init.parse_strategy(strategy)
    if not any(isinstance(part, kvstate.RecurrentState) for part in template.state.parts):
        return rule.name
    if "mean" in rule.needs:
        if mean is None:
            raise ValueError(f"the recurrent strategy {rule.name} needs a recurrent mean of the target")
        mean.check(template)
        if group not in mean.means:
            raise ValueError(f"the recurrent mean has no group {group!r} (groups: {sorted(mean.means)})")
    if "prefix" in rule.needs:
        if native_state is None or not native_prefix:
            raise ValueError(f"the recurrent strategy {rule.name} needs a native state of the prompt's start "
                             "(native_state, native_prefix)")
        if len(native_state.tokens) > native_prefix:
            raise ValueError(f"the recurrent strategy {rule.name} reads the cell of a native state of the prompt's "
                             f"start, and this one holds {len(native_state.tokens)} tokens, more than the native "
                             f"prefix of {native_prefix}")
    return rule.name


def prepare(source_state, source, target, layers, *, t0, template, fill="mean", table=None, swa_cells=None,
            server_tokens=True, native_state=None, native_prefix=0, indexer_ridge=None, recurrent="zero",
            recurrent_mean=None, recurrent_group=recurrent_init.ALL):
    """The Plan of a translation; layers: {kind: source layers the hub encodes} (see source_layers);
    indexer_ridge: the target's hub.IndexerRidge (DSA targets) or None; recurrent, recurrent_mean,
    recurrent_group: what a hybrid target's recurrent cell at t0 - 1 starts from (translate)."""
    t0 = int(t0)
    source, target = state_side(source, source_state), state_side(target, template)
    if not 0 < t0 <= len(target.tokens["token_id"]):
        raise ValueError("t0 must lie in [1, target tokens]")
    if fill == "table" and table is None:
        raise ValueError("fill 'table' needs a token table")
    swa_cells = int(swa_cells if swa_cells is not None else (target.spec.n_swa + 1 if target.spec.n_swa else 0))
    native_prefix = int(min(native_prefix, t0))
    if native_prefix:
        if native_state is None or len(native_state.tokens) < native_prefix:
            raise ValueError("a native prefix needs a native target state holding it")
        if state_side(target, native_state).spec != target.spec and len(target.spec.layout_variants()) > 1:
            raise ValueError("the native prefix state and the template hold different llama.cpp revisions' layouts")
        if not np.array_equal(np.asarray(native_state.tokens)[:native_prefix],
                              np.asarray(target.tokens["token_id"])[:native_prefix]):
            raise ValueError("the native state's tokens differ from the target prompt")
    layouts = target.spec.layouts(template.state.flags)
    if [part.name for part in template.state.parts] != [layout.name for layout in layouts]:
        raise ValueError("the template's sections differ from the target spec's")
    recurrent = _recurrent_strategy(template, recurrent, recurrent_mean, recurrent_group, native_state, native_prefix)
    held = [(part, layout, _section_positions(layout, t0, swa_cells))
            for part, layout in zip(template.state.parts, layouts) if layout.kind == "kv"]
    decoded = [{kind: _decoded_layers(target.spec, layout, kind) for kind in KINDS} for _, layout, _ in held]
    kinds = tuple(kind for kind in KINDS if any(d[kind] for d in decoded))
    if not kinds:
        raise ValueError(f"{target.name} has no attention cache to translate into")
    alignment = handoff.align_document({**source.record, "document_id": target.record["document_id"]},
                                       source.tokens, target.record, target.tokens)
    cache = SourceCache(source_state, source.spec)
    n_source = min(cache.n_positions(layers[kind]) for kind in kinds)
    if n_source and not np.array_equal(cache.tokens[:n_source], np.asarray(source.tokens["token_id"])[:n_source]):
        raise ValueError("source state tokens are not a prefix of the source tokenization")
    source_of = np.full(len(target.tokens["token_id"]), -1, dtype=np.int64)
    source_of[alignment.tgt_position] = alignment.src_position
    content = (alignment.tgt_class == ALIGNED) & (source_of >= 0) & (source_of < n_source)

    needed = np.unique(np.concatenate([positions for _, _, positions in held]))
    mapped_at = needed[content[needed]]
    sections = []
    for (part, layout, positions), by_kind in zip(held, decoded):
        is_mapped = content[positions]
        mapped_rows = np.flatnonzero(is_mapped)
        sections.append(Section(part, layout, positions, np.searchsorted(mapped_at, positions[mapped_rows]),
                                mapped_rows, np.flatnonzero(~is_mapped), by_kind))
    norms = latent_norms(target) if any(target.spec.k_only(l) for d in decoded for l in d["k"]) else {}
    return Plan(t0, swa_cells, fill, source, target, template, cache, source_of[mapped_at], sections,
                int(len(needed)), native_state, native_prefix, bool(server_tokens), mapped_at, kinds, norms,
                _indexer_fill(target, sections, t0, indexer_ridge, native_prefix), recurrent, recurrent_mean,
                recurrent_group)


def _table_fill(plan, positions, layers, kind, table, renorm_fill_k):
    """float32 [len(positions), sum d_l]: per token id means of native target rows (TokenTable).
    K rows of k_norm models are renormalized per head; MLA rows go through the latent shell later."""
    ids = np.asarray(plan.target.tokens["token_id"])[positions]
    blocks = []
    for layer in layers:
        filled, _ = table.lookup(kind, layer, ids)
        if kind == "k" and renorm_fill_k and not plan.target.spec.k_only(layer):
            filled = handoff.renormalize_heads(filled)
        blocks.append(filled.reshape(len(ids), -1))
    return np.concatenate(blocks, axis=1)


def compute(plan, hub, backend, *, table=None, renorm_fill_k=True, chunk_rows=4096, timings=None):
    """{(section name, kind): backend float32 [section rows, sum d_l]}: pre-RoPE target K/V of the
    layers each section decodes (Section.decoded; indexer sections and the V of K-only layers have none)."""
    timings = {} if timings is None else timings
    values = {}
    for kind in plan.kinds:
        started = time.perf_counter()
        encoder = backend.encoder(hub, kind, plan.source.name)
        decoder = backend.decoder(hub, kind, plan.target.name)
        timings["sides_s"] = timings.get("sides_s", 0.0) + time.perf_counter() - started
        started = time.perf_counter()
        z = backend.encode(encoder, plan.cache, kind, plan.source_positions, chunk_rows, center=decoder)
        backend.synchronize()
        timings["encode_s"] = timings.get("encode_s", 0.0) + time.perf_counter() - started
        started = time.perf_counter()
        for section in plan.sections:
            layers = section.decoded[kind]
            if not layers:
                continue
            fill = None
            if plan.fill == "table" and len(section.fill_rows):
                fill = _table_fill(plan, section.positions[section.fill_rows], layers, kind, table, renorm_fill_k)
            values[section.name, kind] = backend.section(decoder, z, layers, len(section.positions), section.mapped,
                                                         section.mapped_rows, section.fill_rows, fill, centered=True)
        del z
        backend.synchronize()
        timings["decode_s"] = timings.get("decode_s", 0.0) + time.perf_counter() - started
    return values


def kv_stream(spec, layers, positions, k, v, *, dtype, v_trans, tokens_at=None, seq_id=0, role="attention"):
    """kvstate.build_kv_stream for rows already in the cache dtype (cast and checked by a backend);
    nextn layers missing from k, and the V rows of an indexer section that stores some (qwen4exp on
    85d5703a3, never written), get the zero rows a native save holds."""
    positions = np.asarray(positions)
    if (positions.ndim != 1 or len(positions) == 0 or positions.dtype.kind not in "iu" or (positions < 0).any()
            or (np.diff(positions) <= 0).any() or positions.max() > 2**31 - 1):
        raise kvstate.StateFormatError("cell positions must be a nonempty strictly increasing int32 array")
    cells, np_dtype = len(positions), kvstate.GGML_DTYPES[kvstate.DTYPE_NAMES[dtype]]
    out_k, out_v = {}, {}
    for il in layers:
        for key, source, target in (("k", k, out_k), ("v", v, out_v)):
            shape = spec.kv_shape(il, key, role)
            if shape is None:               # the V of a K-only layer
                continue
            if il not in source and ((key == "k" and spec.layer_types[il] == "nextn")
                                     or (key == "v" and role == "indexer")):
                target[il] = np.zeros((cells,) + shape, dtype=np_dtype)
                continue
            value = source[il]
            if value.dtype != np_dtype or value.shape != (cells,) + shape:
                raise kvstate.StateFormatError(f"layer {il} {key.upper()} is {value.dtype} {value.shape}, expected "
                                               f"{np_dtype} {(cells,) + shape}")
            target[il] = value
    ext = None
    if spec.has_cell_ext(role):
        if tokens_at is None or len(tokens_at) != cells:
            raise kvstate.StateFormatError("IMROPE and PLE caches need the token id of every cell")
        ext = np.stack([positions, positions, np.asarray(tokens_at)], axis=1).astype(np.int32)
    code = kvstate.DTYPE_NAMES[dtype]
    return kvstate.KVStream(positions.astype(np.int32), tuple((int(seq_id),) for _ in range(cells)), ext,
                            bool(v_trans), out_k, out_v, {il: code for il in out_k}, {il: code for il in out_v})


def _cache_dtype(part):
    """The cache dtype name ("f16" | "f32") of a template KV section (f16 for a section of no layers,
    e.g. the base cache of a hybrid whose attention layers all slide: it stores cells but no rows)."""
    codes = part.stream().k_types.values()
    code = next(iter(codes)) if codes else kvstate.GGML_TYPE_F16
    return next(name for name, value in kvstate.DTYPE_NAMES.items() if value == code)


def _output_arrays(plan):
    """{section name: ({layer: K}, {layer: V})}: empty host arrays in the cache dtype for the rows each
    section stores (indexer sections: their keys, zeros until the ridge or a native prefix fills them)."""
    spec = plan.target.spec
    out = {}
    for section in plan.sections:
        np_dtype = kvstate.GGML_DTYPES[kvstate.DTYPE_NAMES[_cache_dtype(section.part)]]
        n = len(section.positions)
        if section.role == "indexer":
            out[section.name] = ({layer: np.zeros((n,) + spec.kv_shape(layer, "k", "indexer"), dtype=np_dtype)
                                  for layer in section.layers}, {})
            continue
        out[section.name] = tuple({layer: np.empty((n,) + spec.kv_shape(layer, kind), dtype=np_dtype)
                                   for layer in section.decoded[kind]} for kind in KINDS)
    return out


def _touch(array):
    """One zero byte per page of a fresh output array (its rows are all written later): its pages
    are faulted in now."""
    if array.size:
        array.reshape(-1).view(np.uint8)[::PAGE] = 0


class Prefault:
    """The first touch of a translation's empty output arrays (_output_arrays), on a few threads
    while the backend encodes and decodes.

    np.empty arrays get their memory only when first written. A GPU backend first writes them in
    its device-to-host copies on the calling thread, inside the assemble stage: for the 1.5 GB of
    a 10.6K-row Qwen3-4B target 0.08 s on a calm machine (transparent huge pages; 0.03 s on 8
    threads), up to 0.8 s with direct compaction under memory pressure (bench_engine.py job).
    Here PREFAULT_THREADS threads touch every page, (section, kind) by (section, kind) in the order
    compute_cache writes them, and wait() holds back a (section, kind)'s first write until its
    arrays are done (a touch after the rows were written would overwrite them). Indexer keys
    (zeros until written) are left alone."""

    def __init__(self, plan, out):
        with _PREFAULT_LOCK:
            if not _PREFAULT_POOL:
                _PREFAULT_POOL.append(ThreadPoolExecutor(max_workers=PREFAULT_THREADS,
                                                         thread_name_prefix="kvpark-prefault"))
        self.pending = {}
        for kind in plan.kinds:
            for section in plan.sections:
                if section.role != "indexer":
                    arrays = out[section.name][KINDS.index(kind)].values()
                    self.pending[section.name, kind] = [_PREFAULT_POOL[0].submit(_touch, a) for a in arrays]

    def wait(self, key=None):
        """Block until the arrays of key = (section name, kind), or (None) every array, are touched."""
        for item in list(self.pending) if key is None else [key]:
            for future in self.pending.pop(item, ()):
                future.result()

    def cancel(self):
        """Drop the touches not yet started and wait out the running ones: the arrays of a failed
        translation are not faulted in for nothing while its error is handled (a numpy retry)."""
        futures = [future for item in self.pending.values() for future in item]
        self.pending.clear()
        for future in [future for future in futures if not future.cancel()]:
            future.exception()          # running: wait, whatever its outcome


def _indexer_rows(plan, backend, section, values, r0, r1, out):
    """The ridge-filled indexer keys at the positions of a section's decoded K rows [r0, r1)."""
    spec, ridge = plan.target.spec, plan.indexer
    positions = section.positions[r0:r1]
    host, offsets, offset = None, {}, 0
    for layer in section.decoded["k"]:
        width = int(np.prod(spec.head_shape(layer, "k")))
        offsets[layer] = (offset, width)
        offset += width
    for other in plan.sections:
        if other.role != "indexer":
            continue
        wanted = [layer for layer in other.layers if layer in offsets]
        if not wanted:
            continue
        i0, i1 = (int(i) for i in np.searchsorted(other.positions, [positions[0], positions[-1] + 1]))
        if not np.array_equal(other.positions[i0:i1], positions):
            raise ValueError(f"{other.name} cells differ from the {section.name} cells they are computed from")
        if host is None:
            host = backend.host(values)
        dtype = _cache_dtype(other.part)
        for layer in wanted:
            start, width = offsets[layer]
            keys = kvstate.cache_keys(spec, layer, ridge.predict(layer, host[:, start:start + width])[:, None, :],
                                      positions, role="indexer")
            cast_to_cache(keys, dtype, f"layer {layer} indexer keys", out=out[other.name][0][layer][i0:i1])


def _store_rows(plan, backend, section, kind, values, r0, r1, out):
    """A section's decoded float32 rows [r0, r1) (values [rows, sum d_l] of section.decoded[kind]) into
    its cache arrays in out: MLA latents back on their RMS shell, the ridge-filled indexer keys of
    these positions (from the pre-RoPE K rows), then RoPE at the target positions and the cache dtype
    (backend.cache_rows rotates values in place)."""
    spec, layers = plan.target.spec, section.decoded[kind]
    if kind == "k" and plan.norms:
        backend.renormalize_latents(spec, layers, values, plan.norms)
    if kind == "k" and isinstance(plan.indexer, hub_module.IndexerRidge):
        _indexer_rows(plan, backend, section, values, r0, r1, out)
    rows = out[section.name][KINDS.index(kind)]
    backend.cache_rows(spec, layers, kind, values, section.positions[r0:r1], _cache_dtype(section.part),
                       out={layer: rows[layer][r0:r1] for layer in layers})


def compute_cache(plan, hub, backend, *, table=None, renorm_fill_k=True, window=4096, chunk_rows=4096,
                  timings=None):
    """{section name: ({layer: K}, {layer: V})}: every section's rows in its cache dtype, K rotated at
    the target positions, computed window by window of `window` target positions.

    Per kind and window: the mapped cells there are encoded (one encode, centred on the target's
    decoder), then each section's rows inside the window are decoded or filled and finished
    (_store_rows: latent shell, indexer keys, RoPE, cast) into that section's host arrays. The
    arithmetic per row is compute's and assemble's; only the float32 temporaries are bounded by
    the window."""
    timings = {} if timings is None else timings
    out = _output_arrays(plan)
    # a GPU backend's device-to-host copies would first-touch them inside the assemble stage
    prefault = Prefault(plan, out) if backend.prefault_output else None
    try:
        for kind in plan.kinds:
            started = time.perf_counter()
            encoder = backend.encoder(hub, kind, plan.source.name)
            decoder = backend.decoder(hub, kind, plan.target.name)
            timings["sides_s"] = timings.get("sides_s", 0.0) + time.perf_counter() - started
            for p0 in range(0, plan.t0, window):
                p1 = min(p0 + window, plan.t0)
                m0, m1 = (int(i) for i in np.searchsorted(plan.mapped_positions, [p0, p1]))
                z = None
                if m1 > m0:
                    started = time.perf_counter()
                    z = backend.encode(encoder, plan.cache, kind, plan.source_positions[m0:m1], chunk_rows,
                                       center=decoder)
                    backend.synchronize()
                    timings["encode_s"] = timings.get("encode_s", 0.0) + time.perf_counter() - started
                for section in plan.sections:
                    layers = section.decoded[kind]
                    r0, r1 = (int(i) for i in np.searchsorted(section.positions, [p0, p1]))
                    if r1 == r0 or not layers:
                        continue
                    started = time.perf_counter()
                    i0, i1 = np.searchsorted(section.mapped_rows, [r0, r1])
                    f0, f1 = np.searchsorted(section.fill_rows, [r0, r1])
                    fill = None
                    if plan.fill == "table" and f1 > f0:
                        fill = _table_fill(plan, section.positions[section.fill_rows[f0:f1]], layers, kind, table,
                                           renorm_fill_k)
                    values = backend.section(decoder, z, layers, r1 - r0, section.mapped[i0:i1] - m0,
                                             section.mapped_rows[i0:i1] - r0, section.fill_rows[f0:f1] - r0, fill,
                                             centered=True)
                    backend.synchronize()
                    timings["decode_s"] = timings.get("decode_s", 0.0) + time.perf_counter() - started
                    started = time.perf_counter()
                    if prefault is not None:
                        prefault.wait((section.name, kind))
                    _store_rows(plan, backend, section, kind, values, r0, r1, out)
                    del values
                    timings["assemble_s"] = timings.get("assemble_s", 0.0) + time.perf_counter() - started
                del z
    except BaseException:
        if prefault is not None:
            prefault.cancel()
        raise
    if prefault is not None:
        prefault.wait()                 # build_state copies a native prefix into them
    return out


def assemble(plan, values, backend, *, timings=None):
    """The target kvstate.StateFile from compute's values (consumed): latent shell, indexer keys, RoPE
    and the cache dtype on the backend (_store_rows), then build_state."""
    timings = {} if timings is None else timings
    started = time.perf_counter()
    out = _output_arrays(plan)
    for section in plan.sections:
        for kind in plan.kinds:
            if section.decoded[kind]:
                _store_rows(plan, backend, section, kind, values.pop((section.name, kind)), 0, len(section.positions),
                            out)
    state = build_state(plan, out)
    timings["assemble_s"] = timings.get("assemble_s", 0.0) + time.perf_counter() - started
    return state


def build_state(plan, arrays):
    """The target kvstate.StateFile from {section name: ({layer: K}, {layer: V})} in the cache dtype
    (K rotated): the native prefix copied exactly, sections and recurrent rows like the template's."""
    target, t0 = plan.target, plan.t0
    by_part = {id(section.part): section for section in plan.sections}
    new_parts = []
    for part in plan.template.state.parts:
        if isinstance(part, kvstate.KVCache):
            section = by_part[id(part)]
            positions, layers = section.positions, section.layers
            template_stream = part.stream()
            dtype = _cache_dtype(part)
            k, v = arrays[part.name]
            if plan.native_prefix and (positions < plan.native_prefix).any():
                native_stream = plan.native_state.state.part(part.name).stream()
                native_index = handoff._stream_index(native_stream, "native target state")
                rows = np.flatnonzero(positions < plan.native_prefix)
                available = [i for i in rows if int(positions[i]) in native_index]
                if available:
                    cells = handoff._cells(native_index, positions[available], "native target state")
                    for stored, native in ((k, native_stream.k), (v, native_stream.v)):
                        for layer in stored:
                            stored[layer][available] = np.asarray(native[layer][cells])
            stream = kv_stream(target.spec, layers, positions, k, v, dtype=dtype, v_trans=template_stream.v_trans,
                               tokens_at=np.asarray(target.tokens["token_id"])[positions], role=section.role)
            streams = [None] * len(part.streams)
            streams[next(i for i, s in enumerate(part.streams) if s is not None)] = stream
            new_parts.append(kvstate.KVCache(part.name, part.layers, streams))
        elif isinstance(part, kvstate.RecurrentState):
            if part.n_cells != 1:
                raise ValueError("the target template must hold exactly one recurrent cell")
            # one cell at t0 - 1, written like the template's (R, S and PLE rows): zero (R0,
            # kvstate.recurrent_reset's bytes) or the plan's strategy (recurrent_init)
            new_parts.append(recurrent_init.initial_state(part, t0 - 1, plan.recurrent, mean=plan.recurrent_mean,
                                                          prefix=plan.native_state, group=plan.recurrent_group))
        else:
            raise ValueError(f"unsupported template section {type(part).__name__}")
    state = kvstate.StateFile(kvstate.token_words(target.tokens["token_id"][:t0], server_packed=plan.server_tokens),
                              kvstate.SequenceState(target.spec, plan.template.state.flags, new_parts))
    state.check_positions()
    return state


def translate(source_state, source, target, hub, *, t0, template, fill="mean", table=None, swa_cells=None,
              server_tokens=True, chunk_rows=4096, renorm_fill_k=True, native_state=None, native_prefix=0,
              backend=None, window_rows=None, indexer_ridge=None, recurrent="zero", recurrent_mean=None,
              recurrent_group=recurrent_init.ALL):
    """kvstate.StateFile for the target holding positions [0, t0); see the module docstring.

    native_state / native_prefix H: cells at positions < H are copied from a native target state
    holding at least H tokens. K/V of the first H tokens depend only on those tokens, so a short
    native prefill of the prompt's start (the attention-sink tokens, the system prompt, tools)
    gives exact values whatever the rest of the conversation is.

    backend: a backends instance (default numpy, the float32 reference); window_rows: target
    positions per compute_cache window (default backend.window_rows); chunk_rows: source rows per
    encode product; indexer_ridge: the DSA target's hub.IndexerRidge (default: the hub's, if any).

    recurrent: what a hybrid target's recurrent cell at t0 - 1 holds (recurrent_init.STRATEGIES):
    "zero" (R0, the default), "mean" (recurrent_mean, a recurrent_init.RecurrentMean of the target,
    its group recurrent_group: RESULTS.md section 7 measured it closer to the native cell than zero
    on every document), "prefix" (the cell of native_state, which may then hold no more than
    native_prefix tokens), ...; report["recurrent"] = {"strategy", "group"} for hybrid targets.

    A DeepSeek V4 target goes to dsv4_target.translate (pooled caches from the decoded compressor
    inputs; T0 rounded down to a multiple of 128, report["t0"]); it runs on numpy whatever `backend`
    is (a numpy backend lends its resident hub sides), takes fill "mean" only and no native prefix.

    The source and target specs may describe another llama.cpp revision than the states' (both
    pins' states parse with either; kvstate.LLAMA_REVISIONS): the template's reading gives the
    target's layout and the source state's its RoPE (state_side)."""
    source, target = state_side(source, source_state), state_side(target, template)
    if target.spec.memory == "dsv4":
        import dsv4_target
        if native_prefix or table is not None:
            raise ValueError("a DeepSeek V4 target takes no native prefix and no token table")
        return dsv4_target.translate(source_state, source, target, hub, t0=t0, template=template, fill=fill,
                                     server_tokens=server_tokens, chunk_rows=chunk_rows, backend=backend)
    started = time.perf_counter()
    backend = backend or backends.select("cpu")
    for spec in (source.spec, target.spec):
        backend.supports(spec)
    timings = {}
    layers = {kind: source_layers(hub, kind, source.name) for kind in KINDS}
    if indexer_ridge is None and target.spec.indexer is not None:
        indexer_ridge = hub_module.indexer_ridge(hub, target.name)
    plan = prepare(source_state, source, target, layers, t0=t0, template=template, fill=fill, table=table,
                   swa_cells=swa_cells, server_tokens=server_tokens, native_state=native_state,
                   native_prefix=native_prefix, indexer_ridge=indexer_ridge, recurrent=recurrent,
                   recurrent_mean=recurrent_mean, recurrent_group=recurrent_group)
    timings["prepare_s"] = time.perf_counter() - started
    window = int(window_rows or backend.window_rows)
    arrays = compute_cache(plan, hub, backend, table=table, renorm_fill_k=renorm_fill_k, window=window,
                           chunk_rows=chunk_rows, timings=timings)
    built = time.perf_counter()
    state = build_state(plan, arrays)
    timings["assemble_s"] = timings.get("assemble_s", 0.0) + time.perf_counter() - built
    report = {"t0": plan.t0, "swa_cells": plan.swa_cells, "fill": fill, "cells": plan.cells,
              "mapped_cells": plan.mapped_cells, "fill_cells": plan.fill_cells,
              "sections": {section.name: int(len(section.positions)) for section in plan.sections},
              "recurrent_reset": any(isinstance(p, kvstate.RecurrentState) for p in template.state.parts),
              "native_prefix": plan.native_prefix,
              "seconds": round(time.perf_counter() - started, 3),
              "backend": backend.name, "device": str(backend.device),
              "decoded_rows": {section.name: int(len(section.mapped)) for section in plan.sections
                               if any(section.decoded.values())},
              "window_rows": window,
              "timings": {key: round(value, 3) for key, value in timings.items()}}
    if plan.kinds != KINDS:
        report["kinds"] = list(plan.kinds)
    if plan.norms:
        report["latent_layers"] = len(plan.norms)
    if plan.indexer is not None:
        report["indexer"] = plan.indexer if isinstance(plan.indexer, str) else "ridge"
    if report["recurrent_reset"]:
        report["recurrent"] = {"strategy": plan.recurrent}
        if "mean" in recurrent_init.parse_strategy(plan.recurrent).needs:
            report["recurrent"]["group"] = plan.recurrent_group
    return state, report
