"""Universal KV hub: a shared per-token latent between the attention caches of many models.

Per kind (k or v), every model is a *view*: the concatenation over its KV layers of the
pre-RoPE K (or V) rows of one token. The hub is fitted on *hub spans*: tokens of the same
chat segment that end at the same byte in every model (align.py's pairwise rule extended to
N models). Train split only:

  stage 1  per (model, layer): centered PCA to at most `layer_rank` components;
  stage 2  per model: the concatenated stage-1 scores, PCA to at most `view_dim`
           components, whitened: u = scores / sqrt(eigenvalue + eps * largest eigenvalue);
  stage 3  multi-view CCA (MAXVAR): each view block is whitened with its own (ridged) train
           covariance W_m; the top `hub_dim` eigenvectors B of W C W (C the covariance of
           [u_1 .. u_M]) define the consensus Z = [u_1 .. u_M] W B diag(1 / sqrt(eig)), with
           eigenvalues the sums of canonical correlations (at most M); the encoder of model m
           (z from model m alone) is the ridge regression u_m -> Z, read off the same C;
  stage 4  decoders: per model n, ridge from the encoder estimates z_m of every *other*
           model (pooled, so any source works) to each of model n's KV layers.

A translation m -> n is decode_n(encode_m(rows)). With M models the hub stores 2M adapters
instead of M(M - 1) pairwise maps; adding a model needs stages 1-2 for it and two ridge
fits against the hub latents of the hub corpus (add_model_kind). Its decoder regresses its rows on
the pooled pivot estimates, as stage 4 does; its encoder (solve_onboard_encoder, ONBOARD_ENCODERS)
cannot regress onto its own estimate of Z, since Z was fitted without it: "ridge" regresses onto the
mean of the pivots' estimates, which a MAXVAR hub shrinks by lambda / M per dimension, "consensus"
onto Z itself, and match, tls, reverse and cca rescale those encodings (README "Onboarding
encoders"). reencode_model_kind fits only the encoder again, its transforms and decoders untouched.

Views are whatever the captures hold per attention layer, so cache families mix freely: GQA
heads (Qwen3, Muse, the Qwen3.5 hybrids' attention layers) and MLA rows (deepseek2 and
relatives: K = the pre-RoPE row [c_kv | k_pe], one "head" of kv_lora + n_rot; V = c_kv, the
view of K that attention reads as V). A DSA model also gets an IndexerRidge (fit_indexer_ridge):
its own pre-RoPE lightning-indexer keys as a ridge on its K rows, which fills the indexer
section of a translated state. A DeepSeek V4 model's K view is its pre-RoPE latent per layer;
its compressor inputs (HEAD_KINDS: per CSA, HCA or LID layer the row [a | s]) get head decoders
out of the K latent (HeadKind, fit_head), which dsv4_pool pools into its compressed caches.

All statistics are float64 sums over float32 chunks; nothing is held for all rows at once. The fits'
products (projections, Grams, cross products) run on the fit backend (fit_backend.current(): numpy on
the CPU, or a torch GPU in float16 with float32 accumulation, fit_hub.py --fit-device); eigh, solve and
svd stay LAPACK on the CPU, in float64 except the top eigenvectors of a large matrix (_top_eig: float32
subset solves, fit_backend.top_eig). Decoders are fitted from pooled statistics (_DecoderStats): the
cross moments of the pooled source estimates with the target rows are those of their sum (the target
rows repeat for every source), the pooled Gram is pooled from per-source moments, and the ridge
strength is chosen on the dev spans from their Gram and cross moments in closed form, without
predicting the dev rows for every alpha.

Translation needs one *side* per kind and model: the encoder side of the source (stage-1
centering and bases, then the fused encoder W = (basis2 * scale) @ encoder, one matrix
instead of two) and the decoder side of the target. Bundles: format 1 (save_hub, one npz per
model and kind) and format 2 (convert_hub.py: one flat C-contiguous .npy per model, kind and
role, fused encoders precomputed). open_hub reads either lazily, side by side; load_hub reads
a whole bundle of either format into HubKinds. An onboarding's bundle can hard-link the base's
files (save_hub(link_base=...), after checking they hold the arrays in memory), and merge_bundles
assembles a base and the bundles of models onboarded into it separately into one bundle of hard
links (campaign.py assemble); linked files are read-only and never cross a filesystem (link_file).
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "linear_transfer"))

from align import _segment_bytes, classify_tokens, segment_table  # noqa: E402
from captures import CaptureSet  # noqa: E402
import fit_backend  # noqa: E402

KINDS = ("k", "v")
INDEXER_KEY = "ik"              # capture key of the pre-RoPE DSA indexer keys (kv_probe)
FORMAT = 1
FORMAT_V2 = 2
_POOL = ThreadPoolExecutor(8)   # memory-mapped reads: page faults in parallel threads


def _no_check(where=None):
    return None


# the resource guard's hook (resource_guard.Guard.check): every fit loop calls it between chunks and
# before its large eigendecompositions, so a fit sleeps while the CPU is too hot and stops when it is
# critical. fit_hub.py (and diagnostics) set it for a fit; a no-op otherwise.
guard_check = _no_check


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


# -- data ---------------------------------------------------------------------------------------

class CaptureView:
    """One model's captured K/V rows (a CaptureSet) and its rendered records; a DSA model's captures
    may also hold its pre-RoPE indexer keys (key INDEXER_KEY), a DeepSeek V4 model's its compressor
    inputs (HEAD_KINDS). A DeepSeek V4 cache reads its one latent as both K and V, so kv_probe captures
    no V for it: its V view is its K rows."""

    def __init__(self, name, capture_paths, records):
        self.name = name
        self.captures = CaptureSet(capture_paths)
        records = read_jsonl(records) if isinstance(records, (str, Path)) else list(records)
        self.records = {record["document_id"]: record for record in records}
        self.architecture = self.captures.architecture
        extra = (INDEXER_KEY,) + HEAD_KINDS
        self._layers = {kind: list(self.captures.layers(kind)) for kind in KINDS + extra}
        self._alias = {"v": "k"} if self.architecture == "deepseek4" and not self._layers["v"] else {}
        for kind, source in self._alias.items():
            self._layers[kind] = list(self._layers[source])
        self._shape = {}
        for kind in KINDS + extra:
            if not self._layers[kind] and kind in KINDS:
                raise ValueError(f"{name}: captures hold no {kind} rows")
            for layer in self._layers[kind]:
                stored = self._alias.get(kind, kind)
                self._shape[kind, layer] = tuple(self.captures.captures[0].array(stored, layer).shape[1:])

    def layers(self, kind):
        return self._layers[kind]

    def split_rows(self, split):
        """Global capture rows of the documents whose record is in `split` (every token, not only hub spans)."""
        parts = [np.arange(self.captures.rows(d).start, self.captures.rows(d).stop) for d in self.captures.document_ids
                 if self.records.get(d, {}).get("split") == split]
        return np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, dtype=np.int64)

    def shape(self, kind, layer):
        return self._shape[kind, layer]

    def width(self, kind, layer):
        return int(np.prod(self._shape[kind, layer]))

    def rows(self, kind, layer, rows):
        """float32 [n, heads * dim] for global capture rows."""
        return self.captures.gather(self._alias.get(kind, kind), layer, rows).reshape(len(rows), -1)


@dataclass
class Spans:
    """Hub spans: per span its document number and, per model, its global capture row."""

    document_ids: list
    splits: list
    doc: np.ndarray
    rows: dict                          # model name -> int64 [n_spans]
    rejected: list = field(default_factory=list)

    def select(self, split):
        chosen = np.asarray([s == split for s in self.splits], dtype=bool)
        return np.flatnonzero(chosen[self.doc]) if len(chosen) else np.zeros(0, dtype=np.int64)


def joint_spans(views):
    """Tokens that end at the same byte of the same segment in every view's rendering."""
    first = views[0]
    order = [d for d in first.captures.document_ids
             if all(d in view.records and d in view.captures for view in views)]
    document_ids, splits, rejected = [], [], []
    doc_numbers, rows = [], {view.name: [] for view in views}
    for document_id in order:
        records = [view.records[document_id] for view in views]
        keys0, split0 = records[0].get("segment_keys"), records[0].get("split")
        if any(r.get("segment_keys") != keys0 or r.get("split") != split0 for r in records[1:]):
            raise ValueError(f"segment keys or split differ between models for {document_id}")
        tables = [segment_table(r) for r in records]
        texts = [_segment_bytes(r, t) for r, t in zip(records, tables)]
        # templates may drop segments (e.g. earlier turns' reasoning); compare the ones both render
        if any(texts[i][j] != texts[0][j] for i in range(1, len(texts)) for j in texts[0].keys() & texts[i].keys()):
            raise ValueError(f"segment texts differ between models for {document_id}")
        stride = max(len(r["text"].encode("utf-8")) for r in records) + 1
        sides, bad = [], []
        for view, record, table in zip(views, records, tables):
            span = view.captures.rows(document_id)
            start, end = view.captures.byte_start[span], view.captures.byte_end[span]
            if int(end[-1]) != len(record["text"].encode("utf-8")):
                raise ValueError(f"{view.name} capture does not cover the rendered text of {document_id}")
            segment, inside, overlap = classify_tokens(start, end, table)
            # the capture's special flags are kv_probe's (control, user-defined and unknown tokens), stricter than the
            # engine's and pair_eval's (control tokens only, pair_eval.engine_special): a document whose text spells a
            # user-defined token (Falcon-H1-3B's 192 LaTeX commands) is left out of the fit and the token tables (17 of
            # Falcon's 248) although a live switch and pair_eval translate it. Kept so: every fitted
            # hub's spans stay as they were
            if np.any(overlap & view.captures.special[span].astype(bool)):
                bad.append(view.name)
            ids, starts, _ = table
            seg_start = starts[np.searchsorted(starts, start, side="right").clip(1) - 1] if len(ids) else 0 * start
            # a token that spells nothing (kv_probe's emptied SentencePiece space) would repeat the key of the
            # one before it: no span (align.align does the same)
            local = np.flatnonzero(inside & (end > start))
            sides.append((span, local, segment[local] * stride + end[local] - seg_start[local]))
        if bad:
            rejected.append({"document_id": document_id, "reason": "special token inside a segment", "models": bad})
            continue
        common = sides[0][2]
        for _, _, keys in sides[1:]:
            common = np.intersect1d(common, keys, assume_unique=True)
        positions = []
        for span, local, keys in sides:
            _, _, index = np.intersect1d(common, keys, assume_unique=True, return_indices=True)
            positions.append(local[index])
        order_in_text = np.argsort(positions[0], kind="stable")
        number = len(document_ids)
        document_ids.append(document_id)
        splits.append(records[0].get("split"))
        doc_numbers.append(np.full(len(common), number, dtype=np.int64))
        for view, (span, _, _), position in zip(views, sides, positions):
            rows[view.name].append(span.start + position[order_in_text])
    cat = lambda parts: np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, dtype=np.int64)  # noqa: E731
    return Spans(document_ids, splits, cat(doc_numbers), {name: cat(parts) for name, parts in rows.items()},
                 rejected)


# -- streaming statistics -----------------------------------------------------------------------

class Moments:
    """Streaming mean and covariance (float64 sums of float32 chunks, shifted by the first chunk's mean).
    The Gram of each chunk runs on the fit backend current at construction (backend=); its sum stays
    there until it is first read (squares, covariance), after which no chunk may be added. The
    covariance is computed once (callers must not modify it in place)."""

    def __init__(self, dim, backend=None):
        self.backend = backend or fit_backend.current()
        self.n, self.shift = 0, None
        self.sum = np.zeros(dim)
        self._squares, self._held, self._covariance = self.backend.zeros(dim, dim, symmetric=True), None, None

    def add(self, x):
        if self._held is not None:
            raise RuntimeError("Moments: a chunk added after the sums were read")
        backend = self.backend
        x = backend.put(x)
        if self.shift is None:
            self.shift = backend.col_mean32(x)
        xc = backend.sub(x, self.shift)
        self.n += len(x)
        self.sum += backend.col_sum64(xc)
        backend.add_gram(self._squares, xc)

    @property
    def squares(self):
        if self._held is None:
            self._held, self._squares = self.backend.fetch(self._squares), None
        return self._held

    @property
    def mean(self):
        return self.shift.astype(np.float64) + self.sum / self.n

    def covariance(self):
        if self._covariance is None:
            centered = self.sum / self.n
            cov = self.squares / self.n - np.outer(centered, centered)
            self._covariance = (cov + cov.T) / 2
        return self._covariance


def _pool(moments):
    """(rows, mean, scatter) of the rows of several Moments together: the parallel-axis theorem."""
    n = sum(m.n for m in moments)
    mean = sum(m.n * m.mean for m in moments) / n
    return n, mean, _scatter_about(moments, mean)


def _scatter_about(moments, center):
    """sum over the rows of several Moments of (x - center)(x - center)' (float64)."""
    scatter = 0
    for m in moments:
        offset = m.mean - center
        scatter = scatter + m.n * m.covariance() + m.n * np.outer(offset, offset)
    return scatter


def chunks(n, size):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def _loaded_chunks(n, size, load):
    """(part, load(part)) for the chunks of range(n). With a backend that prefetches (the GPU), the next
    chunk's load (the capture reads) runs on a thread while this one is processed."""
    parts = list(chunks(n, size))
    if not fit_backend.current().prefetch or len(parts) < 2:
        for part in parts:
            yield part, load(part)
        return
    with ThreadPoolExecutor(1) as reader:
        pending = reader.submit(load, parts[0])
        for index, part in enumerate(parts):
            data = pending.result()
            if index + 1 < len(parts):
                pending = reader.submit(load, parts[index + 1])
            yield part, data
            del data


def subsample(indices, limit, seed=0, doc=None):
    """At most about `limit` of `indices`; whole documents (contiguous rows) when `doc` is given."""
    if limit is None or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    if doc is None:
        return np.sort(rng.choice(indices, size=limit, replace=False))
    docs = rng.permutation(np.unique(doc[indices]))
    sizes = np.bincount(doc[indices], minlength=int(doc.max()) + 1)[docs]
    keep = docs[:int(np.searchsorted(np.cumsum(sizes), limit)) + 1]
    return indices[np.isin(doc[indices], keep)]


def gather_layers(view, kind, layers, rows):
    """[float32 [n, d_l]] for the given layers, read in parallel."""
    return list(_POOL.map(lambda layer: view.rows(kind, layer, rows), layers))


# -- the model-side transform (stages 1-2) ------------------------------------------------------

@dataclass
class ViewTransform:
    """Rows of one model (one kind) -> whitened view coordinates u (stages 1 and 2)."""

    layers: list
    means: list                 # per layer [d_l]
    bases: list                 # per layer [d_l, r_l]
    basis2: np.ndarray          # [sum r_l, p]
    scale: np.ndarray           # [p]
    explained: dict = field(default_factory=dict)

    @property
    def dim(self):
        return len(self.scale)

    def stage1(self, blocks):
        return np.concatenate([(np.asarray(b, dtype=np.float32) - m) @ basis
                               for b, m, basis in zip(blocks, self.means, self.bases)], axis=1)

    def __call__(self, blocks):
        """blocks: per layer float32 [n, d_l] in self.layers order -> float32 [n, p]."""
        return (self.stage1(blocks) @ self.basis2) * self.scale


def _top_eig(cov, k):
    """The top k eigenpairs (fit_backend.top_eig: scipy's subset solver when importable, else numpy; a
    large subset problem in float32)."""
    return fit_backend.top_eig(cov, k)


def _stage1(backend, transform, blocks):
    """transform.stage1(blocks) on a fit backend (blocks: per layer, transform.layers order)."""
    return backend.cat([backend.mm(backend.sub(block, mean), basis)
                        for block, mean, basis in zip(blocks, transform.means, transform.bases)], axis=1)


def _coords(backend, transform, blocks):
    """transform(blocks) on a fit backend: the whitened view coordinates u."""
    return backend.scale_cols(backend.mm(_stage1(backend, transform, blocks), transform.basis2), transform.scale)


def _encode(backend, hub_kind, name, rows_by_layer):
    """hub_kind.encode(name, rows_by_layer) on a fit backend."""
    transform = hub_kind.source_transforms.get(name) or hub_kind.transforms[name]
    return backend.mm(_stage1(backend, transform, [rows_by_layer[layer] for layer in transform.layers]),
                      hub_kind.encoder_weight(name))


def fit_view(view, kind, rows, *, layer_rank=512, view_dim=2048, eps=1e-3, chunk=4096, log=print, layers=None):
    """Stages 1-2 for one model and kind on the given global capture rows (optionally a layer subset)."""
    backend = fit_backend.current()
    layers = list(view.layers(kind) if layers is None else layers)

    def load(part):
        return gather_layers(view, kind, layers, rows[part])
    moments = [Moments(view.width(kind, layer)) for layer in layers]
    for part, blocks in _loaded_chunks(len(rows), chunk, load):
        guard_check("view")
        for moment, block in zip(moments, blocks):
            moment.add(block)
    means, bases, explained = [], [], []
    for index, moment in enumerate(moments):
        guard_check("view")
        cov = moment.covariance()
        values, vectors = _top_eig(cov, min(layer_rank, len(cov)))
        means.append(moment.mean.astype(np.float32))
        bases.append(vectors.astype(np.float32))
        explained.append(float(values.sum() / max(np.trace(cov), 1e-30)))
        # its float64 sums and covariance (16 bytes per width^2 cell: 13 GiB for Gemma-4-31B's 60 layers)
        # are not needed by stage 2, whose own covariance and eigendecomposition are the peak
        moments[index] = None
    moment = cov = None
    partial = ViewTransform(layers, means, bases, np.zeros((0, 0), np.float32), np.zeros(0, np.float32))
    width = sum(b.shape[1] for b in bases)
    second = Moments(width)
    for part, blocks in _loaded_chunks(len(rows), chunk, load):
        guard_check("view")
        second.add(_stage1(backend, partial, blocks))
    guard_check("view")
    cov2 = second.covariance()
    values, vectors = _top_eig(cov2, min(view_dim, width))
    scale = 1 / np.sqrt(values + eps * values[0])
    log(json.dumps({"stage": "view", "model": view.name, "kind": kind, "layers": len(layers),
                    "stage1_explained_min": round(min(explained), 4), "stage1_width": width,
                    "view_dim": len(values), "stage2_explained": round(float(values.sum() / np.trace(cov2)), 4)}))
    return ViewTransform(layers, means, bases, vectors.astype(np.float32), scale.astype(np.float32),
                         {"stage1": explained, "stage2": float(values.sum() / np.trace(cov2))})


# -- translation sides --------------------------------------------------------------------------

def fuse_encoder(basis2, scale, encoder):
    """W = (basis2 * scale) @ encoder in float64, as float32: (s1 @ basis2) * scale @ encoder == s1 @ W."""
    return ((np.asarray(basis2, dtype=np.float64) * np.asarray(scale, dtype=np.float64))
            @ np.asarray(encoder, dtype=np.float64)).astype(np.float32)


@dataclass
class EncoderSide:
    """What translating *from* a model needs for one kind: z = stage1(rows) @ weight."""

    name: str
    layers: list
    means: list                 # per layer [d_l]
    bases: list                 # per layer [d_l, r_l]
    weight: np.ndarray          # [sum r_l, D], the fused (basis2 * scale) @ encoder

    @property
    def dim(self):
        return self.weight.shape[1]

    @property
    def nbytes(self):
        return int(sum(a.nbytes for a in self.means) + sum(a.nbytes for a in self.bases) + self.weight.nbytes)

    def encode(self, rows_by_layer):
        """z [n, D] float32 from {layer: float32 [n, heads * dim]} for self.layers."""
        stage1 = np.concatenate([(np.asarray(rows_by_layer[layer], dtype=np.float32) - mean) @ basis
                                 for layer, mean, basis in zip(self.layers, self.means, self.bases)], axis=1)
        return stage1 @ self.weight


@dataclass
class DecoderSide:
    """What translating *into* a model needs for one kind: per-layer ridge decoders out of the hub."""

    name: str
    layers: list
    z_mean: np.ndarray          # [D]
    decoders: list              # per layer [D, d_l]
    means: list                 # per layer [d_l]
    _sections: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def nbytes(self):
        own = self.z_mean.nbytes + sum(a.nbytes for a in self.decoders) + sum(a.nbytes for a in self.means)
        # a snapshot: the engine daemon reads this while its worker may be adding a section
        return int(own + sum(w.nbytes + m.nbytes for w, m in list(self._sections.values())))

    def section(self, layers):
        """(weight [D, sum d_l], mean [sum d_l]) for a run of layers (one GEMM per cache section), cached."""
        key = tuple(int(layer) for layer in layers)
        if key not in self._sections:
            index = {layer: i for i, layer in enumerate(self.layers)}
            self._sections[key] = (np.concatenate([self.decoders[index[layer]] for layer in key], axis=1),
                                   np.concatenate([self.means[index[layer]] for layer in key]))
        return self._sections[key]

    def decode(self, z, layers):
        """float32 [n, sum d_l]: the rows of `layers` (concatenated in that order) for hub latents z."""
        weight, mean = self.section(layers)
        return (z - self.z_mean) @ weight + mean


# -- the hub (stages 3-4) -----------------------------------------------------------------------

@dataclass
class HubKind:
    """The hub of one kind: per model a view transform, an encoder and per-layer decoders."""

    names: list
    transforms: dict            # name -> ViewTransform
    encoders: dict              # name -> [p_m, D]
    eigenvalues: np.ndarray     # [D] GCCA eigenvalues (sum over views of the canonical correlations)
    decoder_mean: dict          # name -> per layer [d_l] (train mean of the rows)
    decoder_z_mean: dict        # name -> [D] (mean of the pooled encoder estimates)
    decoders: dict              # name -> per layer [D, d_l]
    alphas: dict = field(default_factory=dict)
    source_transforms: dict = field(default_factory=dict)   # name -> ViewTransform on a layer subset
    source_encoders: dict = field(default_factory=dict)     # name -> [p_sub, D]
    _fused: dict = field(default_factory=dict, repr=False, compare=False)   # name -> (arrays, W)

    def source_layers(self, name):
        """Layers the model needs as a translation source (all, or e.g. only global-attention layers)."""
        return (self.source_transforms.get(name) or self.transforms[name]).layers

    def encoder_weight(self, name):
        """The fused encoder of the transform encode() uses, computed once per side.

        Cached against the identity of basis2, scale and the encoder: the cache holds those
        arrays, so a replaced array (fitting and onboarding assign new ones) is never mistaken
        for its successor. Modifying one of them in place is not detected."""
        transform = self.source_transforms.get(name) or self.transforms[name]
        parts = (transform.basis2, transform.scale, self.source_encoders.get(name, self.encoders[name]))
        cached = self._fused.get(name)
        if cached is None or any(held is not part for held, part in zip(cached[0], parts)):
            cached = self._fused[name] = (parts, fuse_encoder(*parts))
        return cached[1]

    def encode(self, name, rows_by_layer):
        """z [n, D] from a source model's rows: {layer: float32 [n, heads * dim]} for source_layers(name)."""
        transform = self.source_transforms.get(name) or self.transforms[name]
        return transform.stage1([rows_by_layer[layer] for layer in transform.layers]) @ self.encoder_weight(name)

    def decode(self, name, z, layer_index):
        return (z - self.decoder_z_mean[name]) @ self.decoders[name][layer_index] + self.decoder_mean[name][layer_index]

    def encoder_side(self, name):
        transform = self.source_transforms.get(name) or self.transforms[name]
        return EncoderSide(name, list(transform.layers), list(transform.means), list(transform.bases),
                           self.encoder_weight(name))

    def decoder_side(self, name):
        return DecoderSide(name, list(self.transforms[name].layers), self.decoder_z_mean[name],
                           list(self.decoders[name]), list(self.decoder_mean[name]))


def _gather_blocks(view, kind, rows):
    return gather_layers(view, kind, view.layers(kind), rows)


class _DecoderStats:
    """The statistics of one target's ridge decoders (stage 4, onboarding, head decoders): the target's
    row blocks regressed on the pooled hub estimates of `repeat` sources (each source's estimate of a
    span is one regression row, the target's rows repeat for every source), streamed chunk by chunk.

      add(z_total, z_col_sum, ys)  a train chunk: the sum over the sources of their estimates [n, D] (on
                                   the backend), the sum of their column sums (float64) and the target's
                                   blocks: the cross moments of the pooled estimates with the repeated
                                   rows are those of z_total with the rows
      finish(source_moments)       after the train pass, with the Moments of each source's estimates:
                                   the pooled Gram (_pool), the centered cross moments, their projection
                                   on the Gram's eigenvectors
      add_dev(z_total, ys)         a dev chunk (after finish): the dev cross moments and row sums
      solve(alphas, dev_moments)   (decoder per block, block means, z mean, alpha, dev R^2 per alpha):
                                   the ridge strength with the best dev R^2, the dev sum of squared
                                   errors of every alpha in closed form from the dev Gram (dev_moments:
                                   Moments of each source's dev estimates) and cross moments.
    """

    def __init__(self, backend, dim, widths, repeat):
        self.backend, self.dim, self.widths, self.repeat = backend, dim, list(widths), repeat
        self.cross = [backend.zeros(dim, w) for w in self.widths]
        self.y_sum = [np.zeros(w) for w in self.widths]
        self.count, self.z_shift, self.y_shift = 0, None, None
        self.dev_rows = 0

    def add(self, z_total, z_col_sum, ys):
        backend = self.backend
        n = len(ys[0])
        if self.z_shift is None:        # the mean of the first chunk's pooled estimates
            self.z_shift = (np.asarray(z_col_sum) / (self.repeat * n)).astype(np.float32)
            self.y_shift = [backend.col_mean32(y) for y in ys]
        zc = backend.sub(z_total, self.repeat * self.z_shift)
        for index, y in enumerate(ys):
            yc = backend.sub(y, self.y_shift[index])
            self.y_sum[index] += self.repeat * backend.col_sum64(yc)
            backend.add_cross(self.cross[index], zc, yc)
        self.count += self.repeat * n

    def finish(self, source_moments):
        backend = self.backend
        count, self.z_mean, gram = _pool(source_moments)
        if count != self.count:
            raise ValueError(f"decoder statistics of {count} pooled estimates for {self.count} target rows")
        self.pooled_var = np.diag(gram) / count
        z_mean_c = self.z_mean - self.z_shift
        self.y_means = [self.y_shift[i] + self.y_sum[i] / count for i in range(len(self.widths))]
        evals, self.evecs = np.linalg.eigh(gram)
        self.evals = np.clip(evals, 0, None)
        self.projected = []
        for index in range(len(self.widths)):
            # sum (z - z_mean)(y - y_mean)' = sum (z - shift)(y - shift)' - n (z_mean - shift)(y_mean - shift)'
            cross = backend.fetch(self.cross[index]) - count * np.outer(z_mean_c, self.y_sum[index] / count)
            self.cross[index] = None
            self.projected.append(self.evecs.T @ cross)
        del gram
        self.dev_cross = None

    def add_dev(self, z_total, ys):
        backend = self.backend
        if self.dev_cross is None:
            self.dev_cross = [backend.zeros(self.dim, w) for w in self.widths]
            self.dev_y_sum, self.dev_y_sq = [np.zeros(w) for w in self.widths], 0.0
            self._z_center = (self.repeat * self.z_mean).astype(np.float32)
            self._y_centers = [m.astype(np.float32) for m in self.y_means]
        zc = backend.sub(z_total, self._z_center)       # sum over the sources of (z - z_mean)
        for index, y in enumerate(ys):
            yc = backend.sub(y, self._y_centers[index])
            self.dev_y_sq += backend.sq_sum64(yc)
            self.dev_y_sum[index] += backend.col_sum64(yc)
            backend.add_cross(self.dev_cross[index], zc, yc)
        self.dev_rows += len(ys[0])

    def dev_r2(self, alphas, dev_moments):
        """{alpha: dev R^2} of the decoders of each alpha, from the dev statistics: with Z the dev
        estimates less the train mean, Y the dev rows less the train mean and w = (G + lam)^-1 C =
        V diag(1 / (e + lam)) P (G = V diag(e) V', P = V' C), summed over the repeated rows,
        SSE = repeat |Y|^2 - 2 <Z'Y, w> + <Z'Z w, w> = repeat |Y|^2 - 2 sum_i r_i d_i + d' (Gd o PP') d
        with d = 1 / (e + lam), r_i = sum_j P_ij (V' Z'Y)_ij, Gd = V' Z'Z V."""
        backend = self.backend
        evecs = self.evecs
        gram_dev = evecs.T @ _scatter_about(dev_moments, self.z_mean) @ evecs
        r, pp = np.zeros(self.dim), np.zeros((self.dim, self.dim))
        for index, projected in enumerate(self.projected):
            cross_dev = evecs.T @ backend.fetch(self.dev_cross[index])
            self.dev_cross[index] = None
            r += np.einsum("ij,ij->i", projected, cross_dev)
            pp += projected @ projected.T
        quadratic = gram_dev * pp
        del gram_dev, pp
        y_sq = self.repeat * self.dev_y_sq
        # the dev rows' own spread (the R^2 denominator), about their own mean
        total = self.repeat * (self.dev_y_sq - sum(float(np.sum(np.square(s))) for s in self.dev_y_sum) / self.dev_rows)
        results = {}
        for alpha in alphas:
            d = 1.0 / (self.evals + alpha * float(self.evals.mean()))
            sse = y_sq - 2.0 * float(r @ d) + float(d @ quadratic @ d)
            results[alpha] = 1 - sse / max(total, 1e-30)
        return results

    def solve(self, alphas, dev_moments=None):
        results = self.dev_r2(alphas, dev_moments) if self.dev_rows else {}
        best = max(results, key=results.get) if results else alphas[len(alphas) // 2]
        lam = best * float(self.evals.mean())
        weights = [(self.evecs @ (p / (self.evals + lam)[:, None])).astype(np.float32) for p in self.projected]
        return (weights, [m.astype(np.float32) for m in self.y_means], self.z_mean.astype(np.float32), best,
                results)


def fit_hub_kind(views, spans, kind, *, layer_rank=512, view_dim=2048, hub_dim=1024, eps=1e-3,
                 encoder_ridge=1e-3, alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), fit_rows=None, chunk=4096, log=print,
                 source_layers=None, gcca_ridge=1e-3):
    """source_layers: {model name: layers} for models that, as a translation source, only hold some layers
    for old tokens (sliding-window models keep only their global layers); those get a second encoder."""
    backend = fit_backend.current()
    source_layers = {name: list(layers) for name, layers in (source_layers or {}).items()}
    names = [view.name for view in views]
    train, dev = spans.select("train"), spans.select("dev")
    stats_rows = subsample(train, fit_rows, doc=spans.doc)
    started = time.perf_counter()
    transforms = {view.name: fit_view(view, kind, spans.rows[view.name][stats_rows], layer_rank=layer_rank,
                                      view_dim=view_dim, eps=eps, chunk=chunk, log=log) for view in views}
    dims = [transforms[name].dim for name in names]
    offsets = np.cumsum([0] + dims)

    def loader(selected):
        """The row blocks of every view for a chunk of `selected` spans (on the backend)."""
        def load(part):
            return {v.name: _gather_blocks(v, kind, spans.rows[v.name][selected[part]]) for v in views}
        return load

    def uploaded(blocks_of):
        return {name: [backend.put(block) for block in blocks] for name, blocks in blocks_of.items()}

    joint = Moments(int(offsets[-1]))
    for part, blocks_of in _loaded_chunks(len(train), chunk, loader(train)):
        guard_check("gcca")
        joint.add(backend.cat([_coords(backend, transforms[v.name], blocks_of[v.name]) for v in views], axis=1))
    cov = joint.covariance()
    del joint                   # its float64 sums are as large as cov
    # MAXVAR as a generalized eigenproblem: whiten every view block with its own (ridged) train
    # covariance, so eigenvalues are sums of canonical correlations (at most the number of views)
    # and no view's leftover variance can masquerade as a shared direction. The block-diagonal
    # whitening W is applied block by block (W C W without a dense W or dense temporaries: a
    # seven-view hub of 4,096-dimensional views holds two 28,672-square float64 matrices, not four).
    blocks = [slice(offsets[index], offsets[index + 1]) for index in range(len(names))]
    whitens = []
    for block in blocks:
        guard_check("gcca")
        own = cov[block, block]
        evals, evecs = np.linalg.eigh(own + gcca_ridge * np.trace(own) / len(own) * np.eye(len(own)))
        whitens.append((evecs / np.sqrt(np.clip(evals, 1e-12, None))) @ evecs.T)
    whitened = np.empty_like(cov)
    for row_block, row_whiten in zip(blocks, whitens):
        rows = row_whiten @ cov[row_block]
        for column_block, column_whiten in zip(blocks, whitens):
            whitened[row_block, column_block] = rows[:, column_block] @ column_whiten
        del rows
    guard_check("gcca")
    values, vectors = _top_eig(whitened, min(hub_dim, len(cov)))
    del whitened
    # [sum p, D]: Z = U @ projection
    projection = np.concatenate([w @ vectors[block] for block, w in zip(blocks, whitens)]) \
        / np.sqrt(np.maximum(values, 1e-12))
    encoders = {}
    for index, name in enumerate(names):
        guard_check("gcca")
        block = slice(offsets[index], offsets[index + 1])
        own = cov[block, block]
        own_reg = own + encoder_ridge * np.trace(own) / len(own) * np.eye(len(own))
        encoders[name] = np.linalg.solve(own_reg, cov[block] @ projection).astype(np.float32)
    log(json.dumps({"stage": "gcca", "kind": kind, "views": dims, "hub_dim": len(values),
                    "top_eigenvalues": [round(float(v), 3) for v in values[:5]],
                    "eigenvalue_at_hub_dim": round(float(values[-1]), 4),
                    "seconds": round(time.perf_counter() - started, 1)}))
    del cov
    projection32 = projection.astype(np.float32)        # one array: the GPU backend keeps its upload

    # encoders from a layer subset: ridge from the subset's own view to the consensus Z
    source_transforms, source_encoders = {}, {}
    if source_layers:
        for view in views:
            if view.name in source_layers:
                source_transforms[view.name] = fit_view(view, kind, spans.rows[view.name][stats_rows],
                                                        layer_rank=layer_rank, view_dim=view_dim, eps=eps,
                                                        chunk=chunk, log=log, layers=source_layers[view.name])
        sub_moments = {name: Moments(t.dim) for name, t in source_transforms.items()}
        sub_cross = {name: backend.zeros(t.dim, len(values)) for name, t in source_transforms.items()}
        sub_shift, z_sum = {}, np.zeros(len(values))
        for part, blocks_of in _loaded_chunks(len(train), chunk, loader(train)):
            guard_check("source_encoder")
            blocks_of = uploaded(blocks_of)
            us = [_coords(backend, transforms[v.name], blocks_of[v.name]) for v in views]
            z = backend.mm(backend.cat(us, axis=1), projection32)
            del us
            z_sum += backend.col_sum64(z)
            for name, t in source_transforms.items():
                index = {layer: i for i, layer in enumerate(transforms[name].layers)}
                u_sub = _coords(backend, t, [blocks_of[name][index[layer]] for layer in t.layers])
                if name not in sub_shift:
                    sub_shift[name] = backend.col_mean32(u_sub)
                sub_moments[name].add(u_sub)
                backend.add_cross(sub_cross[name], backend.sub(u_sub, sub_shift[name]), z)
        for name, t in source_transforms.items():
            m = sub_moments[name]
            # sum (u - u_mean)(z - z_mean)' = sum (u - shift) z' - n (u_mean - shift) z_mean'
            centered_cross = backend.fetch(sub_cross[name]) - m.n * np.outer(m.mean - sub_shift[name], z_sum / m.n)
            cov = m.covariance()
            reg = cov + encoder_ridge * np.trace(cov) / len(cov) * np.eye(len(cov))
            source_encoders[name] = np.linalg.solve(reg, centered_cross / m.n).astype(np.float32)
            log(json.dumps({"stage": "source_encoder", "kind": kind, "model": name, "layers": len(t.layers),
                            "view_dim": t.dim}))
        del sub_moments, sub_cross

    def estimates(blocks_of):
        """Per view: its z estimate [n, D] from its source encoder (on the backend)."""
        out = {}
        for v in views:
            blocks = blocks_of[v.name]
            if v.name in source_transforms:
                t = source_transforms[v.name]
                index = {layer: i for i, layer in enumerate(transforms[v.name].layers)}
                out[v.name] = backend.mm(_coords(backend, t, [blocks[index[layer]] for layer in t.layers]),
                                         source_encoders[v.name])
            else:
                out[v.name] = backend.mm(_coords(backend, transforms[v.name], blocks), encoders[v.name])
        return out

    # one pass over the train spans accumulates every decoder's statistics: the input of model n's
    # decoder is the pool of the z estimates of every other model (its Gram pooled from each model's
    # own Moments of its estimates, its cross moments those of their sum)
    D = len(values)
    repeat = len(views) - 1
    estimate_moments = {v.name: Moments(D) for v in views}
    stats = {target.name: _DecoderStats(backend, D, [target.width(kind, layer) for layer in target.layers(kind)],
                                        repeat) for target in views}
    for part, blocks_of in _loaded_chunks(len(train), chunk, loader(train)):
        guard_check("decoder")
        blocks_of = uploaded(blocks_of)
        z = estimates(blocks_of)
        sums = {name: backend.col_sum64(z[name]) for name in names}
        z_all = backend.total([z[name] for name in names])
        for name in names:
            estimate_moments[name].add(z[name])
        for target in views:
            stats[target.name].add(backend.minus(z_all, z[target.name]),
                                   sum(sums[name] for name in names if name != target.name), blocks_of[target.name])
        del z, z_all, blocks_of
    for target in views:
        stats[target.name].finish([estimate_moments[name] for name in names if name != target.name])
    del estimate_moments
    dev_moments = None
    if len(dev):
        dev_moments = {v.name: Moments(D) for v in views}
        for part, blocks_of in _loaded_chunks(len(dev), chunk, loader(dev)):
            guard_check("decoder")
            blocks_of = uploaded(blocks_of)
            z = estimates(blocks_of)
            z_all = backend.total([z[name] for name in names])
            for name in names:
                dev_moments[name].add(z[name])
            for target in views:
                stats[target.name].add_dev(backend.minus(z_all, z[target.name]), blocks_of[target.name])
            del z, z_all, blocks_of

    decoders, decoder_mean, decoder_z_mean, chosen = {}, {}, {}, {}
    for target in views:
        guard_check("decoder")
        others = [dev_moments[name] for name in names if name != target.name] if dev_moments else None
        decoders[target.name], decoder_mean[target.name], decoder_z_mean[target.name], best, results = \
            stats[target.name].solve(alphas, others)
        stats[target.name] = None
        chosen[target.name] = {"alpha": best, "dev_r2": results}
        log(json.dumps({"stage": "decoder", "kind": kind, "model": target.name, "alpha": best,
                        "dev_r2": {str(a): round(r, 4) for a, r in results.items()}}))
    return HubKind(names, transforms, encoders, values.astype(np.float64), decoder_mean, decoder_z_mean,
                   decoders, chosen, source_transforms, source_encoders)


# -- onboarding a model into an existing hub ------------------------------------------------------

ONBOARD_ENCODERS = ("ridge", "consensus", "match", "tls", "reverse", "cca", "reverse_z")
CONSENSUS_ENCODERS = ("consensus", "cca", "reverse_z")   # regress onto the consensus Z: need every view it was fitted on
# the onboarding encoder by default (Track A, 2026-10-06, RESULTS.md section 9), chosen for robustness: about 6 times
# ridge's share of the floor's KL removed in both end-to-end runs (DeepSeek-V2-Lite and LFM2-2.6B -> Qwen3.8), held-out
# R^2 within 0.03-0.05 (K) of the best encoder; reverse_z removed as much on average, but significantly more for one
# source and less for the other, with a chat-level loss of 45% of a small floor and 0.19-0.27 less K R^2
DEFAULT_ONBOARD_ENCODER = "reverse"
# reverse and reverse_z scale dimension d by var(target_d) / cov(target_d, encoding_d), about 1 / R^2_d: unbounded as
# R^2_d -> 0 (a view that barely predicts a dimension, e.g. the layer-subset encoder of a sliding-window model). Their
# per-dimension variance is capped at ONBOARD_VARIANCE_CAP times the pooled pivot estimates' (what every decoder was
# fitted on), i.e. the scale at sqrt(cap) times match's. 4 is the largest the end-to-end comparison covered: reverse
# reached at most 2.4-3.75 on the pilots' train spans and won; reverse_z (median 8-10, up to 41) lost on LFM2's
# low-floor chats (RESULTS.md section 9). It binds on none of the four models onboarded on 2026-10-07. 0: no cap.
ONBOARD_VARIANCE_CAP = 4.0
ONBOARD_WARN_R2 = 0.05          # an onboarding logs a warning when a dimension's train R^2 is below this


def consensus_views(manifest, *, read=None):
    """The models a bundle's hub latent was fitted on (the views of its GCCA): settings
    "consensus_views" when recorded; else those of the base an assembled bundle (merge_bundles) or an
    onboarded one grew from, all its models for a bundle fitted without --base; None when unknown."""
    settings = manifest.get("settings") or {}
    if settings.get("consensus_views"):
        return list(settings["consensus_views"])
    base = manifest.get("assembled_base") or settings.get("base")
    if not base:
        return list(manifest["models"])
    base = Path(base)
    if not (base / "manifest.json").is_file():
        return None
    return consensus_views((read or read_manifest)(base), read=read)


def solve_onboard_encoder(method, cov_u, cross, target_cov, pooled_var, *, ridge=1e-3,
                          variance_cap=ONBOARD_VARIANCE_CAP, info=None):
    """The encoder [p, D] of an onboarded view from its train statistics.

    cov_u [p, p]: the covariance of the new view's coordinates u; cross [p, D]: their covariance with
    the regression target (the mean of the pivots' estimates, or for CONSENSUS_ENCODERS the consensus
    Z); target_cov [D, D]: the target's covariance; pooled_var [D]: the per-dimension variance of the
    pivots' estimates pooled (what every decoder was fitted on). Methods (ONBOARD_ENCODERS):

      ridge      the ridge regression u -> target (target: the pivots' mean, onboarding until 2026-10-05)
      consensus  the ridge regression u -> Z, an estimate of the consensus as a jointly fitted
                 encoder is (cross and target_cov must be the consensus's)
      match      ridge, rescaled per dimension to the pooled pivot estimates' variance
      tls        ridge, rescaled per dimension to the target's variance: the geometric mean of the forward
                 and the inverted reverse regression slopes, i.e. reduced-major-axis scaling (not orthogonal
                 total least squares; the name stayed)
      reverse    ridge, rescaled per dimension by var(target) / cov(target, encoding): the inverted
                 reverse regression of the encoding on the target, which makes the encoding unbiased
                 given the target (E[encoding | target] = target), not shrunk toward its mean
      cca        the canonical correlation of u with the target, every canonical pair at unit gain:
                 encodings with the target's covariance (cross and target_cov of the consensus), not
                 scaled like the base views' estimates (that variant, RESULTS.md section 9's ccam, is the
                 cca encoder times sqrt(pooled_var / var(Z)) per dimension; not an option here)
      reverse_z  reverse onto the consensus Z (cross and target_cov of the consensus): unbiased given Z
                 itself, a per-dimension variance up to 1 / R^2 of Z's (8-40 times the pooled pivot
                 estimates' on the pilots)

    variance_cap (reverse, reverse_z; ONBOARD_VARIANCE_CAP, 0 or None: none): each dimension's train variance at
    most variance_cap times pooled_var's (the scale at most sqrt(variance_cap) times match's). info: a dict, filled
    with the fit's per-dimension diagnostics on the train statistics (encoder_diagnostics: min_r2 and median_r2 of
    the ridge fit onto the target, max_ratio of the encoding's variance over pooled_var, before the cap for the
    reverse family as max_ratio_uncapped, and capped, the number of dimensions the cap limited).
    """
    if method not in ONBOARD_ENCODERS:
        raise ValueError(f"unknown onboarding encoder {method!r} (one of {', '.join(ONBOARD_ENCODERS)})")
    cov_u, cross = np.asarray(cov_u, dtype=np.float64), np.asarray(cross, dtype=np.float64)
    target_var = np.diag(np.asarray(target_cov, dtype=np.float64))
    pooled_var = np.asarray(pooled_var, dtype=np.float64)
    reg = cov_u + ridge * np.trace(cov_u) / len(cov_u) * np.eye(len(cov_u))
    out = info if info is not None else {}
    out.clear()
    out.update(method=method, cap=float(variance_cap) if variance_cap and method in ("reverse", "reverse_z") else None)
    if method == "cca":
        eu, vu = np.linalg.eigh(reg)
        u_inv_root = (vu / np.sqrt(np.clip(eu, 1e-30, None))) @ vu.T
        ez, vz = np.linalg.eigh(np.asarray(target_cov, dtype=np.float64))
        ez = np.clip(ez, 1e-9 * max(float(ez.max()), 1e-30), None)
        left, _, right = np.linalg.svd(u_inv_root @ cross @ ((vz / np.sqrt(ez)) @ vz.T), full_matrices=False)
        encoder = u_inv_root @ left @ right @ ((vz * np.sqrt(ez)) @ vz.T)
        out.update(encoder_diagnostics(encoder, cov_u, cross, target_var, pooled_var))
        return encoder.astype(np.float32)
    weight = np.linalg.solve(reg, cross)
    variance = np.maximum(np.einsum("ij,ij->j", weight, cov_u @ weight), 1e-30)
    covariance = np.einsum("ij,ij->j", weight, cross)          # cov(encoding_d, target_d) >= 0: reg is PD
    scale = np.ones_like(variance)
    if method == "match":
        scale = np.sqrt(pooled_var / variance)
    elif method == "tls":
        scale = np.sqrt(target_var / variance)
    elif method in ("reverse", "reverse_z"):
        scale = target_var / np.maximum(covariance, 1e-30)
        out["max_ratio_uncapped"] = round(float(np.max(scale ** 2 * variance / np.maximum(pooled_var, 1e-30))), 4)
        limit = np.sqrt(variance_cap * pooled_var / variance) if variance_cap else np.inf
        out["capped"] = int(np.count_nonzero(scale > limit))
        scale = np.minimum(scale, limit)
    encoder = weight * scale
    out.update(encoder_diagnostics(encoder, cov_u, cross, target_var, pooled_var, ridge_fit=(variance, covariance)))
    return encoder.astype(np.float32)


def encoder_diagnostics(encoder, cov_u, cross, target_var, pooled_var, *, ridge_fit=None):
    """Per-dimension train diagnostics of an onboarded encoder [p, D]: min_r2 and median_r2 (the squared
    correlation of the encoding with the regression target, which a per-dimension rescaling keeps: for the
    ridge-based methods the ridge fit's), min_ratio, median_ratio and max_ratio (its variance over the pooled
    pivot estimates'), rounded for the log."""
    variance = np.maximum(np.einsum("ij,ij->j", encoder, cov_u @ encoder), 1e-30)
    covariance = np.einsum("ij,ij->j", encoder, cross)
    if ridge_fit is not None:
        variance_r, covariance_r = ridge_fit
        r2 = covariance_r ** 2 / (np.maximum(variance_r, 1e-30) * np.maximum(target_var, 1e-30))
    else:
        r2 = covariance ** 2 / (variance * np.maximum(target_var, 1e-30))
    ratio = variance / np.maximum(pooled_var, 1e-30)
    return {"min_r2": round(float(np.min(r2)), 4), "median_r2": round(float(np.median(r2)), 4),
            "min_ratio": round(float(np.min(ratio)), 4), "median_ratio": round(float(np.median(ratio)), 4),
            "max_ratio": round(float(np.max(ratio)), 4)}


def encoder_warnings(info, *, warn_r2=ONBOARD_WARN_R2):
    """Why an onboarded encoder's diagnostics (solve_onboard_encoder's info) deserve a look: dimensions the
    variance cap limited, a dimension the view barely predicts; [] when none."""
    out = []
    if info.get("capped"):
        out.append(f"the variance cap ({info['cap']:g} x the pivots') limited {info['capped']} dimension(s); "
                   f"uncapped the largest ratio was {info.get('max_ratio_uncapped')}")
    if info.get("min_r2") is not None and info["min_r2"] < warn_r2:
        out.append(f"a dimension's train R^2 is {info['min_r2']} (below {warn_r2}): the view barely predicts it")
    return out


def _onboard_target(encoder, consensus, pivots):
    """The models whose full-view estimates define an onboarded encoder's regression target: for
    CONSENSUS_ENCODERS the models the hub latent was fitted on (`consensus`, every one a pivot), else
    none (the target is the mean of the pivots' estimates). ValueError for an unknown encoder."""
    if encoder not in ONBOARD_ENCODERS:
        raise ValueError(f"unknown onboarding encoder {encoder!r} (one of {', '.join(ONBOARD_ENCODERS)})")
    if encoder not in CONSENSUS_ENCODERS:
        return []
    if not consensus:
        raise ValueError(f"the {encoder} encoder needs the models the hub latent was fitted on (consensus=)")
    absent = sorted(set(consensus) - {p.name for p in pivots})
    if absent:
        raise ValueError(f"the {encoder} encoder needs every model the hub latent was fitted on as a pivot; "
                         f"missing {absent}")
    return list(consensus)


def _onboard_loader(hub_kind, new_view, pivots, spans, kind, selected, full_views=()):
    """load(part) for _loaded_chunks: the new model's row blocks and each pivot's rows by layer (the
    layers it provides as a source; for the models in full_views every layer) for a chunk of `selected`."""
    name = new_view.name

    def load(part):
        chosen = selected[part]
        out = {name: _gather_blocks(new_view, kind, spans.rows[name][chosen])}
        for p in pivots:
            wanted = hub_kind.transforms[p.name].layers if p.name in full_views else hub_kind.source_layers(p.name)
            out[p.name] = dict(zip(wanted, gather_layers(p, kind, wanted, spans.rows[p.name][chosen])))
        return out
    return load


def _pivot_estimates(backend, hub_kind, pivots, rows_of, full_views=()):
    """([each pivot's estimate from the layers it provides as a source], the onboarded encoder's
    regression target) on the backend: the consensus Z = M / lambda times the mean of the full-view
    estimates of full_views, or without full_views the mean of the pivots' estimates."""
    estimates, full = [], []
    for p in pivots:
        by_layer = rows_of[p.name]
        estimates.append(_encode(backend, hub_kind, p.name, {l: by_layer[l] for l in hub_kind.source_layers(p.name)}))
        if p.name in full_views:
            t = hub_kind.transforms[p.name]
            full.append(backend.mm(_coords(backend, t, [by_layer[l] for l in t.layers]), hub_kind.encoders[p.name])
                        if p.name in hub_kind.source_transforms else estimates[-1])
    if not full_views:
        return estimates, backend.mean_of(estimates)
    factor = (len(full_views) / len(full) / np.asarray(hub_kind.eigenvalues, dtype=np.float64)).astype(np.float32)
    return estimates, backend.scale_cols(backend.total(full), factor)


class _EncoderStats:
    """The train statistics of an onboarded model's encoders (solve_onboard_encoder), streamed chunk by
    chunk: per transform (key "full", and "sub" for a layer subset it provides as a source) the Moments
    of its coordinates u and their cross moments with the regression target, the target's Moments and,
    per pivot, the Moments of its estimates (pooled: what every decoder was fitted on)."""

    def __init__(self, backend, transforms, dim, pivots):
        self.backend, self.transforms = backend, dict(transforms)
        self.u = {key: Moments(t.dim) for key, t in self.transforms.items()}
        self.cross = {key: backend.zeros(t.dim, dim) for key, t in self.transforms.items()}
        self.shift = {}
        self.target, self.z_sum = Moments(dim), np.zeros(dim)
        self.pivots = {name: Moments(dim) for name in pivots}

    def add(self, blocks_by_layer, z_ref, estimates=None):
        """A chunk: the new model's blocks {layer: rows on the backend}, the target, and (unless a caller
        pools them itself) the pivots' estimates in self.pivots order."""
        backend = self.backend
        self.target.add(z_ref)
        self.z_sum += backend.col_sum64(z_ref)
        for key, t in self.transforms.items():
            u = _coords(backend, t, [blocks_by_layer[layer] for layer in t.layers])
            if key not in self.shift:
                self.shift[key] = backend.col_mean32(u)
            self.u[key].add(u)
            backend.add_cross(self.cross[key], backend.sub(u, self.shift[key]), z_ref)
        for moments, estimate in zip(self.pivots.values(), estimates or ()):
            moments.add(estimate)

    def pooled_var(self):
        """Per hub dimension, the variance of the pivots' estimates pooled (each estimate one row)."""
        count, _, scatter = _pool(list(self.pivots.values()))
        return np.diag(scatter) / count

    def solve(self, key, encoder, pooled_var, ridge, variance_cap=ONBOARD_VARIANCE_CAP, info=None):
        m = self.u[key]
        # sum (u - u_mean)(z - z_mean)' = sum (u - shift) z' - n (u_mean - shift) z_mean'
        cross = self.backend.fetch(self.cross[key]) - m.n * np.outer(m.mean - self.shift[key], self.z_sum / m.n)
        return solve_onboard_encoder(encoder, m.covariance(), cross / m.n, self.target.covariance(), pooled_var,
                                     ridge=ridge, variance_cap=variance_cap, info=info)


def _solve_encoders(stats, keys, encoder, pooled_var, ridge, variance_cap, *, kind, name, log):
    """{key: encoder} of an onboarded model (stats: _EncoderStats) and {key: diagnostics}; logs the
    diagnostics (stage "onboard-encoder") and any encoder_warnings (stage "onboard-warning", also on stderr)."""
    encoders, infos = {}, {}
    for key in keys:
        info = {}
        encoders[key] = stats.solve(key, encoder, pooled_var, ridge, variance_cap=variance_cap, info=info)
        infos[key] = info
        log(json.dumps({"stage": "onboard-encoder", "kind": kind, "model": name, "transform": key, **info}))
        for warning in encoder_warnings(info):
            log(json.dumps({"stage": "onboard-warning", "kind": kind, "model": name, "transform": key,
                            "warning": warning}))
            print(f"onboarding {name} ({kind}, {key}): {warning}", file=sys.stderr, flush=True)
    return encoders, infos


def add_model_kind(hub_kind, new_view, pivots, spans, kind, *, layer_rank=512, view_dim=2048, eps=1e-3,
                   encoder_ridge=1e-3, alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), fit_rows=None, chunk=4096,
                   source_layers=None, log=print, encoder=DEFAULT_ONBOARD_ENCODER, consensus=None,
                   variance_cap=ONBOARD_VARIANCE_CAP):
    """Add `new_view` to a fitted hub without refitting the other models.

    pivots: views of models already in the hub, with captures on the same corpus. The new model's
    decoder is the ridge regression from the pooled pivot estimates onto its rows (as in stage 4).
    Its encoder (solve_onboard_encoder) regresses its own view onto the hub latent of a span:
    `encoder` "ridge" (and match, tls, reverse) onto the mean of the pivots' estimates, which for a
    MAXVAR hub is the consensus shrunk by lambda_d / M per dimension (M views, eigenvalues lambda: the
    estimates of all M views sum to lambda_d Z_d), so its encodings spread about half as far as a
    jointly fitted model's; "consensus" (and cca) onto the consensus itself, Z = M / lambda * the mean
    of the full-view estimates of `consensus`, the models the hub latent was fitted on (all of them
    must be pivots; consensus_views reads them from a bundle's manifest). The decoder does not depend on
    the encoder: reencode_model_kind fits the encoder again with another method. variance_cap: the reverse
    family's per-dimension variance cap (solve_onboard_encoder); the encoders' train diagnostics are logged
    (stage "onboard-encoder") and kept in hub_kind.alphas[name]["encoder_stats"].
    """
    backend = fit_backend.current()
    name = new_view.name
    if name in hub_kind.names:
        raise ValueError(f"{name} is already in the hub")
    missing = [p.name for p in pivots if p.name not in hub_kind.names]
    if missing:
        raise ValueError(f"pivots {missing} are not in the hub")
    full_views = _onboard_target(encoder, consensus, pivots)
    train, dev = spans.select("train"), spans.select("dev")
    stats_rows = spans.rows[name][subsample(train, fit_rows, doc=spans.doc)]
    transform = fit_view(new_view, kind, stats_rows, layer_rank=layer_rank, view_dim=view_dim, eps=eps,
                         chunk=chunk, log=log)
    sub = (fit_view(new_view, kind, stats_rows, layer_rank=layer_rank, view_dim=view_dim, eps=eps, chunk=chunk,
                    log=log, layers=source_layers) if source_layers else None)
    D = len(hub_kind.eigenvalues)
    layers = new_view.layers(kind)
    widths = [new_view.width(kind, layer) for layer in layers]
    stats = _EncoderStats(backend, {"full": transform, **({"sub": sub} if sub is not None else {})}, D,
                          [p.name for p in pivots])
    decoder = _DecoderStats(backend, D, widths, len(pivots))
    load = _onboard_loader(hub_kind, new_view, pivots, spans, kind, train, full_views)
    for part, rows_of in _loaded_chunks(len(train), chunk, load):
        guard_check("onboard")
        blocks = [backend.put(block) for block in rows_of[name]]
        estimates, z_ref = _pivot_estimates(backend, hub_kind, pivots, rows_of, full_views)
        stats.add(dict(zip(layers, blocks)), z_ref, estimates)
        decoder.add(backend.total(estimates), sum(backend.col_sum64(e) for e in estimates), blocks)
        del rows_of, blocks, estimates, z_ref

    guard_check("onboard")
    decoder.finish(list(stats.pivots.values()))
    pooled_var = decoder.pooled_var
    dev_moments = None
    if len(dev):
        dev_moments = {p.name: Moments(D) for p in pivots}
        for part, rows_of in _loaded_chunks(len(dev), chunk, _onboard_loader(hub_kind, new_view, pivots, spans,
                                                                             kind, dev)):
            guard_check("onboard")
            estimates, _ = _pivot_estimates(backend, hub_kind, pivots, rows_of)
            for p, estimate in zip(pivots, estimates):
                dev_moments[p.name].add(estimate)
            decoder.add_dev(backend.total(estimates), [backend.put(block) for block in rows_of[name]])
            del rows_of, estimates
    weights, means, z_mean, best, results = decoder.solve(
        alphas, [dev_moments[p.name] for p in pivots] if dev_moments else None)
    encoders, infos = _solve_encoders(stats, ["full"] + (["sub"] if sub is not None else []), encoder, pooled_var,
                                      encoder_ridge, variance_cap, kind=kind, name=name, log=log)
    hub_kind.names.append(name)
    hub_kind.transforms[name] = transform
    hub_kind.encoders[name] = encoders["full"]
    hub_kind.decoders[name] = weights
    hub_kind.decoder_mean[name] = means
    hub_kind.decoder_z_mean[name] = z_mean
    hub_kind.alphas[name] = {"alpha": best, "dev_r2": results, "onboarded_with": [p.name for p in pivots],
                             "encoder": encoder, "encoder_stats": infos}
    if sub is not None:
        hub_kind.source_transforms[name] = sub
        hub_kind.source_encoders[name] = encoders["sub"]
    log(json.dumps({"stage": "onboard", "kind": kind, "model": name, "pivots": [p.name for p in pivots],
                    "encoder": encoder, "alpha": best, "dev_r2": {str(a): round(r, 4) for a, r in results.items()}}))
    return hub_kind


def reencode_model_kind(hub_kind, new_view, pivots, spans, kind, *, encoder, consensus=None, encoder_ridge=1e-3,
                        chunk=4096, log=print, variance_cap=ONBOARD_VARIANCE_CAP):
    """Fit an onboarded model's encoder(s) again with another method (solve_onboard_encoder) from one
    pass over the train spans, leaving its view transforms and decoders exactly as they are (no stage 1-2
    refit, no decoder statistics): translations into the model, and between the other models, stay
    bit-identical; only translations out of it change. pivots and consensus as for add_model_kind."""
    backend = fit_backend.current()
    name = new_view.name
    if name not in hub_kind.names:
        raise ValueError(f"{name} is not in the hub: onboard it first")
    missing = [p.name for p in pivots if p.name not in hub_kind.names or p.name == name]
    if missing or not pivots:
        raise ValueError(f"pivots {missing or '(none)'} are not other models of the hub")
    full_views = _onboard_target(encoder, consensus, pivots)
    transforms = {"full": hub_kind.transforms[name]}
    if name in hub_kind.source_transforms:
        transforms["sub"] = hub_kind.source_transforms[name]
    layers = transforms["full"].layers
    stats = _EncoderStats(backend, transforms, len(hub_kind.eigenvalues), [p.name for p in pivots])
    train = spans.select("train")
    for part, rows_of in _loaded_chunks(len(train), chunk, _onboard_loader(hub_kind, new_view, pivots, spans, kind,
                                                                           train, full_views)):
        guard_check("onboard")
        blocks = dict(zip(new_view.layers(kind), (backend.put(block) for block in rows_of[name])))
        estimates, z_ref = _pivot_estimates(backend, hub_kind, pivots, rows_of, full_views)
        stats.add({layer: blocks[layer] for layer in layers}, z_ref, estimates)
        del rows_of, blocks, estimates, z_ref
    guard_check("onboard")
    pooled_var = stats.pooled_var()
    previous = (hub_kind.alphas.get(name) or {}).get("encoder")
    encoders, infos = _solve_encoders(stats, list(transforms), encoder, pooled_var, encoder_ridge, variance_cap,
                                      kind=kind, name=name, log=log)
    hub_kind.encoders[name] = encoders["full"]
    if "sub" in transforms:
        hub_kind.source_encoders[name] = encoders["sub"]
    hub_kind.alphas[name] = {**(hub_kind.alphas.get(name) or {}), "encoder": encoder, "reencoded_from": previous,
                             "onboarded_with": [p.name for p in pivots], "encoder_stats": infos}
    log(json.dumps({"stage": "reencode", "kind": kind, "model": name, "pivots": [p.name for p in pivots],
                    "encoder": encoder, "previous": previous, "rows": int(stats.u["full"].n)}))
    return hub_kind


# -- DSA indexer keys -----------------------------------------------------------------------------

@dataclass
class IndexerRidge:
    """A DSA model's lightning-indexer keys from its own attention K rows.

    Per indexer layer: the pre-RoPE indexer key (the k_norm LayerNorm output, what kvstate.
    pre_rope_keys gives for role "indexer") as a ridge regression on the same layer's pre-RoPE K
    row ([c_kv | k_pe] of the MLA cache), fitted on the model's native captures. Both are
    projections of the layer's normalized input, so the K rows a translation decodes carry much of
    what the keys need; translate.py fills a DSA target's indexer section with it."""

    name: str
    layers: list
    x_means: list               # per layer [d_k]
    weights: list               # per layer [d_k, head_size]
    y_means: list               # per layer [head_size]
    fit: dict = field(default_factory=dict)     # layer -> {"alpha", "dev_r2"}

    @property
    def nbytes(self):
        return int(sum(a.nbytes for part in (self.x_means, self.weights, self.y_means) for a in part))

    def predict(self, layer, rows):
        """float32 [n, head_size] pre-RoPE indexer keys of `layer` from its pre-RoPE K rows [n, d_k]."""
        index = self.layers.index(layer)
        rows = np.asarray(rows, dtype=np.float32)
        return ((rows - self.x_means[index]) @ self.weights[index] + self.y_means[index]).astype(np.float32)


def fit_indexer_ridge(view, *, alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), fit_rows=None, chunk=4096, log=print):
    """IndexerRidge of a DSA model from its captures: K rows (key "k") -> pre-RoPE indexer keys (key
    INDEXER_KEY) per indexer layer, on every token of the train documents (template tokens included:
    a translation fills those cells too); the ridge strength is picked on the dev documents."""
    layers = list(view.layers(INDEXER_KEY))
    if not layers:
        raise ValueError(f"{view.name}: captures hold no indexer keys ({INDEXER_KEY}); collect them with kv_probe")
    missing = [layer for layer in layers if layer not in view.layers("k")]
    if missing:
        raise ValueError(f"{view.name}: indexer layers {missing} have no captured K rows")
    train, dev = view.split_rows("train"), view.split_rows("dev")
    if not len(train):
        raise ValueError(f"{view.name}: no train documents for the indexer ridge")
    train = subsample(train, fit_rows)
    x_means, weights, y_means, fit = [], [], [], {}
    for layer in layers:
        width = view.width("k", layer)
        joint = Moments(width + view.width(INDEXER_KEY, layer))
        for part in chunks(len(train), chunk):
            guard_check("indexer_ridge")
            rows = train[part]
            joint.add(np.concatenate([view.rows("k", layer, rows), view.rows(INDEXER_KEY, layer, rows)], axis=1))
        cov, mean = joint.covariance(), joint.mean
        cov_x, cross = cov[:width, :width], cov[:width, width:]
        evals, evecs = np.linalg.eigh(cov_x)
        evals = np.clip(evals, 0, None)
        projected = evecs.T @ cross
        solve = lambda alpha: evecs @ (projected / (evals + alpha * float(evals.mean()))[:, None])  # noqa: E731
        results = {}
        if len(dev):
            x_dev, y_dev = view.rows("k", layer, dev), view.rows(INDEXER_KEY, layer, dev)
            total = float(np.sum(np.square(y_dev - y_dev.mean(axis=0))))
            for alpha in alphas:
                sse = float(np.sum(np.square(y_dev - ((x_dev - mean[:width]) @ solve(alpha) + mean[width:]))))
                results[alpha] = 1 - sse / max(total, 1e-30)
            best = max(results, key=results.get)
        else:
            best = alphas[len(alphas) // 2]
        x_means.append(mean[:width].astype(np.float32))
        weights.append(solve(best).astype(np.float32))
        y_means.append(mean[width:].astype(np.float32))
        fit[layer] = {"alpha": best, "dev_r2": results}
        log(json.dumps({"stage": "indexer_ridge", "model": view.name, "layer": layer, "alpha": best,
                        "dev_r2": {str(a): round(r, 4) for a, r in results.items()}}))
    return IndexerRidge(view.name, layers, x_means, weights, y_means, fit)


def indexer_ridge(hub, name):
    """The IndexerRidge of a model: from a Hub bundle, or the optional "indexer" entry ({name:
    IndexerRidge}) of a {kind: HubKind} dict; None when the hub has none for it."""
    if isinstance(hub, Hub):
        return hub.indexer_ridge(name)
    return (hub.get("indexer") or {}).get(name) if isinstance(hub, dict) else None


# -- head kinds (DeepSeek V4 compressor inputs) ------------------------------------------------------

@dataclass
class HeadKind:
    """Decoders of per-token quantities that only some models hold, read out of the hub latent of
    another kind (`base`, the K kind): DeepSeek V4's compressor inputs (HEAD_KINDS: per CSA, HCA or
    LID layer the row [a | s], a = W_comp_kv x and s = W_comp_gate x before APE), which pool into its
    compressed caches (dsv4_pool). Decoders are fitted like stage 4 (fit_head): ridge regressions
    from the pooled base-kind latent estimates of the other models onto the target's rows, so any
    source of the base kind translates into them, with the base kind's z.

    `sources` holds the own encoders of models that hold the kind themselves ({name: (ViewTransform,
    encoder [p, D])}): only an identity hub has one (z from the model's own rows, decoded back
    exactly); a translation from such a source encodes this kind from its own rows."""

    kind: str
    base: str
    names: list                 # models with decoders
    layers: dict                # name -> layers
    decoder_mean: dict          # name -> per layer [d_l]
    decoder_z_mean: dict        # name -> [D]
    decoders: dict              # name -> per layer [D, d_l]
    alphas: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    base_kind: object = field(default=None, repr=False, compare=False)    # the base HubKind (not saved)

    def decoder_side(self, name):
        return DecoderSide(name, list(self.layers[name]), self.decoder_z_mean[name], list(self.decoders[name]),
                           list(self.decoder_mean[name]))

    def encoder_side(self, name):
        """The own EncoderSide of a model in `sources`, else the base kind's encoder side (when this
        HeadKind knows its base HubKind: load_hub and with_heads set it)."""
        if name in self.sources:
            transform, encoder = self.sources[name]
            return EncoderSide(name, list(transform.layers), list(transform.means), list(transform.bases),
                               fuse_encoder(transform.basis2, transform.scale, encoder))
        if self.base_kind is None:
            raise KeyError(f"{name} encodes the {self.base} kind for {self.kind} heads")
        return self.base_kind.encoder_side(name)

    def source_layers(self, name):
        if name in self.sources:
            return list(self.sources[name][0].layers)
        return self.base_kind.source_layers(name)


HEAD_KINDS = ("csa", "hca", "lid")      # kvstate.ModelSpec.pools() names
HEAD_BASE = "k"


def with_heads(hubs, heads):
    """{kind: HubKind} with HeadKinds added (merged per kind: one HeadKind per kind holds every model's
    decoders), each pointed at its base HubKind."""
    out = dict(hubs)
    for head in heads:
        head.base_kind = out.get(head.base)
        held = out.get(head.kind)
        if isinstance(held, HeadKind):
            for name in head.names:
                if name not in held.names:
                    held.names.append(name)
                for attr in ("layers", "decoder_mean", "decoder_z_mean", "decoders", "alphas"):
                    if name in getattr(head, attr):
                        getattr(held, attr)[name] = getattr(head, attr)[name]
            held.sources.update(head.sources)
        else:
            out[head.kind] = head
    return out


def head_sides(hub, kind, source, target, *, load=None):
    """(own EncoderSide of `source` for head kind `kind`, or None when the source encodes the base
    kind; DecoderSide of `target`) from {kind: HubKind | HeadKind} or a Hub (only the target's
    decoders are read). load(hub, kind, name, role), e.g. a backend's resident side, gives the
    decoder side instead."""
    if isinstance(hub, Hub):
        own = None                      # a bundle keeps no own head encoders
        known = target in hub.manifest.get("heads", {}).get(kind, {}).get("models", {})
    else:
        head = hub.get(kind) if isinstance(hub, dict) else None
        known = isinstance(head, HeadKind) and target in head.names
        own = head.encoder_side(source) if known and source in head.sources else None
    if not known:
        raise ValueError(f"the hub has no {kind} head decoders for {target} (fit_hub.py --heads {target})")
    if load is not None:
        return own, load(hub, kind, target, "decoder")
    return own, (hub.head_decoder_side(kind, target) if isinstance(hub, Hub) else head.decoder_side(target))


def fit_head(base_kind, pivots, target_view, spans, kind, *, alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), chunk=4096,
             log=print):
    """A HeadKind for `target_view`: per layer of its `kind` rows (target_view.layers(kind)), the ridge
    regression from the pooled base-kind latent estimates of the `pivots` (models in base_kind, captured
    on the same corpus; hub spans `spans`) onto the target's rows, the ridge strength picked on the dev
    spans (as add_model_kind fits a decoder). alphas[name] holds the alpha, the dev R^2 of every alpha
    (the selection criterion) and test_r2: per pivot, the chosen decoders' variance-weighted R^2 on the
    test spans over all layers (as evaluate scores a kind)."""
    backend = fit_backend.current()
    name = target_view.name
    missing = [p.name for p in pivots if p.name not in base_kind.names]
    if missing or not pivots:
        raise ValueError(f"pivots {missing or '(none)'} are not in the {HEAD_BASE} hub")
    layers = list(target_view.layers(kind))
    if not layers:
        raise ValueError(f"{name}: captures hold no {kind} rows")
    train, dev, test = spans.select("train"), spans.select("dev"), spans.select("test")
    D = len(base_kind.eigenvalues)
    widths = [target_view.width(kind, layer) for layer in layers]

    def loader(selected, with_pivots=True):
        """Each pivot's base-kind source rows by layer and the target's rows of `kind` for a chunk."""
        def load(part):
            chosen = selected[part]
            out = {name: gather_layers(target_view, kind, layers, spans.rows[name][chosen])}
            for p in pivots if with_pivots else ():
                wanted = base_kind.source_layers(p.name)
                out[p.name] = dict(zip(wanted, gather_layers(p, HEAD_BASE, wanted, spans.rows[p.name][chosen])))
            return out
        return load

    def estimate(pivot, rows_of):
        return _encode(backend, base_kind, pivot.name, rows_of[pivot.name])

    pivot_moments = {p.name: Moments(D) for p in pivots}
    decoder = _DecoderStats(backend, D, widths, len(pivots))
    for part, rows_of in _loaded_chunks(len(train), chunk, loader(train)):
        guard_check("head")
        estimates = [estimate(p, rows_of) for p in pivots]
        sums = 0
        for p, z in zip(pivots, estimates):
            pivot_moments[p.name].add(z)
            sums = sums + backend.col_sum64(z)
        decoder.add(backend.total(estimates), sums, [backend.put(y) for y in rows_of[name]])
        del rows_of, estimates
    if not decoder.count:
        raise ValueError(f"no train spans to fit the {kind} heads of {name}")
    decoder.finish([pivot_moments[p.name] for p in pivots])
    del pivot_moments
    dev_moments = None
    if len(dev):
        dev_moments = {p.name: Moments(D) for p in pivots}
        for part, rows_of in _loaded_chunks(len(dev), chunk, loader(dev)):
            guard_check("head")
            estimates = [estimate(p, rows_of) for p in pivots]
            for p, z in zip(pivots, estimates):
                dev_moments[p.name].add(z)
            decoder.add_dev(backend.total(estimates), [backend.put(y) for y in rows_of[name]])
            del rows_of, estimates
    weights, means, z_mean, best, results = decoder.solve(
        alphas, [dev_moments[p.name] for p in pivots] if dev_moments else None)
    test_r2 = {}
    if len(test):
        sse = {p.name: 0.0 for p in pivots}
        y_sum, y_sq = [np.zeros(w) for w in widths], 0.0
        for part, rows_of in _loaded_chunks(len(test), chunk, loader(test)):
            guard_check("head")
            ys = [backend.put(y) for y in rows_of[name]]
            for i, y in enumerate(ys):
                y_sum[i] += backend.col_sum64(y)
                y_sq += backend.sq_sum64(y)
            for pivot in pivots:
                z = backend.sub(estimate(pivot, rows_of), z_mean)
                for i, y in enumerate(ys):
                    sse[pivot.name] += backend.sq_sum64(backend.minus(y, backend.add(backend.mm(z, weights[i]), means[i])))
            del rows_of, ys
        total = y_sq - sum(float(np.sum(np.square(s))) for s in y_sum) / len(test)
        test_r2 = {p.name: float(1 - sse[p.name] / max(total, 1e-30)) for p in pivots}
    log(json.dumps({"stage": "head", "kind": kind, "model": name, "layers": layers, "alpha": best,
                    "dev_r2": {str(a): round(r, 4) for a, r in results.items()},
                    "test_r2": {p: round(r, 4) for p, r in test_r2.items()}}))
    return HeadKind(kind, HEAD_BASE, [name], {name: layers}, {name: means}, {name: z_mean}, {name: weights},
                    {name: {"alpha": best, "dev_r2": results, "test_r2": test_r2, "pivots": [p.name for p in pivots]}})


# -- evaluation ---------------------------------------------------------------------------------

def evaluate(hub_kind, views, spans, kind, split="test", chunk=4096, involving=None, pairs=None):
    """Variance-weighted R^2 (about the split's own mean) of every ordered pair, all target layers;
    with `involving` (a model name) only the pairs that have it as source or target, with `pairs`
    (an iterable of (source, target) names) only those pairs. One pass over the split: each chunk's rows
    are read once, each source encoded once for all its targets."""
    backend = fit_backend.current()
    selected = spans.select(split)
    wanted = None if pairs is None else {tuple(pair) for pair in pairs}
    todo = [(source, target) for source in views for target in views
            if (involving is None or involving in (source.name, target.name))
            and (wanted is None or (source.name, target.name) in wanted)]
    if not todo:
        return {}
    sources = [v for v in views if any(s is v for s, _ in todo)]
    targets = [v for v in views if any(t is v for _, t in todo)]

    def load(part):
        """{name: {layer: rows}} of every layer a source or target of this chunk needs."""
        out = {}
        for v in views:
            needed = set(hub_kind.source_layers(v.name)) if any(s is v for s in sources) else set()
            if any(t is v for t in targets):
                needed |= set(v.layers(kind))
            ordered = [layer for layer in v.layers(kind) if layer in needed]
            if ordered:
                out[v.name] = dict(zip(ordered, gather_layers(v, kind, ordered, spans.rows[v.name][selected[part]])))
        return out

    sse = {(s.name, t.name): np.zeros(len(t.layers(kind))) for s, t in todo}
    y_sum = {t.name: [np.zeros(t.width(kind, layer)) for layer in t.layers(kind)] for t in targets}
    y_sq = {t.name: np.zeros(len(t.layers(kind))) for t in targets}
    n = 0
    for part, rows_of in _loaded_chunks(len(selected), chunk, load):
        guard_check("evaluate")
        z = {s.name: _encode(backend, hub_kind, s.name, rows_of[s.name]) for s in sources}
        for target in targets:
            ys = [backend.put(rows_of[target.name][layer]) for layer in target.layers(kind)]
            for index, y in enumerate(ys):
                y_sum[target.name][index] += backend.col_sum64(y)
                y_sq[target.name][index] += backend.sq_sum64(y)
            for source in sources:
                if (source.name, target.name) not in sse:
                    continue
                zc = backend.sub(z[source.name], hub_kind.decoder_z_mean[target.name])
                for index, y in enumerate(ys):
                    prediction = backend.add(backend.mm(zc, hub_kind.decoders[target.name][index]),
                                             hub_kind.decoder_mean[target.name][index])
                    sse[source.name, target.name][index] += backend.sq_sum64(backend.minus(y, prediction))
            del ys
        n += part.stop - part.start
        del rows_of, z
    results = {}
    for source, target in todo:
        total = np.asarray([y_sq[target.name][i] - np.sum(np.square(y_sum[target.name][i])) / n
                            for i in range(len(target.layers(kind)))])
        errors = sse[source.name, target.name]
        results[source.name, target.name] = {
            "r2": float(1 - errors.sum() / total.sum()), "rows": int(n),
            "per_layer": [float(1 - s / t) for s, t in zip(errors, total)]}
    return results


# -- persistence --------------------------------------------------------------------------------

def link_file(source, destination):
    """Hard-link source to destination (a new path) and make the file read-only (0444: every name
    of it, the base bundle's included, so no writer rewrites a shared file in place; a replacement
    goes through a new file). Refused across filesystems, where a link would be a full copy."""
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        raise ValueError(f"cannot hard-link {source} to {destination}: they lie on different filesystems "
                         "(a link would be a copy; build the bundle on the filesystem of its members)") from None
    os.chmod(destination, 0o444)


def _device(path):
    """st_dev of path, or of its nearest existing ancestor."""
    path = Path(path).absolute()
    while not path.exists():
        path = path.parent
    return path.stat().st_dev


def _arrays_match(path, arrays):
    """Whether the npz at `path` holds exactly these arrays (the same names, shapes and values)."""
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(arrays):
            return False
        for key, value in arrays.items():
            value = np.asarray(value)
            other = stored[key]
            if other.shape != value.shape or not np.array_equal(other, value):
                return False
    return True


def _kind_arrays(hub, name):
    """The arrays save_hub writes to <name>.<kind>.npz for a HubKind's model."""
    t = hub.transforms[name]
    arrays = {"basis2": t.basis2, "scale": t.scale, "encoder": hub.encoders[name],
              "decoder_z_mean": hub.decoder_z_mean[name]}
    for i, layer in enumerate(t.layers):
        arrays[f"mean_{layer}"] = t.means[i]
        arrays[f"basis_{layer}"] = t.bases[i]
        arrays[f"decoder_{layer}"] = hub.decoders[name][i]
        arrays[f"decoder_mean_{layer}"] = hub.decoder_mean[name][i]
    if name in hub.source_transforms:
        st = hub.source_transforms[name]
        arrays.update(source_basis2=st.basis2, source_scale=st.scale, source_encoder=hub.source_encoders[name],
                      source_layers=np.asarray(st.layers, dtype=np.int64))
        for i, layer in enumerate(st.layers):
            arrays[f"source_mean_{layer}"] = st.means[i]
            arrays[f"source_basis_{layer}"] = st.bases[i]
    return arrays


def _ridge_arrays(ridge):
    """The arrays save_hub writes to <name>.indexer.npz for an IndexerRidge."""
    arrays = {"layers": np.asarray(ridge.layers, dtype=np.int64)}
    for i, layer in enumerate(ridge.layers):
        arrays.update({f"x_mean_{layer}": ridge.x_means[i], f"weight_{layer}": ridge.weights[i],
                       f"y_mean_{layer}": ridge.y_means[i]})
    return arrays


def _head_arrays(head, name):
    """The arrays save_hub writes to <name>.<kind>.npz for a HeadKind's model."""
    layers = list(head.layers[name])
    arrays = {"layers": np.asarray(layers, dtype=np.int64), "decoder_z_mean": head.decoder_z_mean[name]}
    for i, layer in enumerate(layers):
        arrays[f"decoder_{layer}"] = head.decoders[name][i]
        arrays[f"decoder_mean_{layer}"] = head.decoder_mean[name][i]
    return arrays


def save_hub(directory, hubs, views, spans, settings, base_models=None, indexer_ridges=None, link_base=None):
    """Write a hub bundle; base_models (a loaded manifest's "models") covers models not given as views;
    indexer_ridges ({name: IndexerRidge}) go to <name>.indexer.npz; HeadKinds among `hubs` go to
    <name>.<kind>.npz (decoders only; own sources are not saved) and manifest["heads"].

    link_base: the format 1 bundle an onboarding started from (add_model_kind leaves the arrays of
    its models untouched): the files of its models are hard links to its own (read-only, link_file),
    with its manifest entries, so only the onboarded model's files are written. Before a file is
    linked, its sha256 is checked against the base manifest and its arrays against the ones in
    memory: a base model whose arrays changed is refused (ValueError), never linked."""
    directory = Path(directory)
    heads = {kind: hub for kind, hub in hubs.items() if isinstance(hub, HeadKind)}
    hubs = {kind: hub for kind, hub in hubs.items() if not isinstance(hub, HeadKind)}
    linked = {}
    if link_base is not None:
        link_base = Path(link_base)
        base_manifest = read_manifest(link_base)
        if base_manifest["format_version"] != FORMAT:
            raise ValueError(f"{link_base} is not a format {FORMAT} bundle: its files cannot be linked")
        linked = {key: entry for key, entry in base_manifest["files"].items()
                  if _owner(key, base_manifest["models"]) is not None}
    directory.mkdir(parents=True, exist_ok=False)

    verifier = Verifier()

    def linked_entry(key, arrays):
        """Link the base bundle's file of `key` into the new bundle once it holds exactly `arrays`
        (the arrays in memory); its manifest entry, or None when the base has no such file."""
        entry = linked.get(key)
        if entry is not None:
            path = link_base / entry["file"]
            verifier.check(path, entry["sha256"])
            if not _arrays_match(path, arrays):
                raise ValueError(f"{key}: the arrays in memory differ from {path}: a base model changed, "
                                 "its file cannot be linked")
            link_file(path, directory / entry["file"])
            manifest["files"][key] = dict(entry)
        return entry

    models = dict(base_models or {})
    models.update({view.name: {"architecture": view.architecture,
                               "layers": {kind: view.layers(kind) for kind in KINDS},
                               "head_shape": {kind: list(view.shape(kind, view.layers(kind)[0])) for kind in KINDS}}
                   for view in views})
    manifest = {"format_version": FORMAT, "purpose": "universal_kv_hub", "settings": settings,
                "models": models,
                "spans": {split: int(len(spans.select(split))) for split in ("train", "dev", "test")},
                "files": {}}
    for kind, hub in hubs.items():
        manifest[f"{kind}_eigenvalues"] = [float(v) for v in hub.eigenvalues]
        manifest[f"{kind}_alphas"] = {name: hub.alphas[name]["alpha"] for name in hub.names}
        for name in hub.names:
            arrays = _kind_arrays(hub, name)
            if linked_entry(f"{name}.{kind}", arrays):
                continue
            path = directory / f"{name}.{kind}.npz"
            np.savez(path, **arrays)
            manifest["files"][f"{name}.{kind}"] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if indexer_ridges:
        manifest["indexer_ridges"] = {}
        for name, ridge in sorted(indexer_ridges.items()):
            if name not in models:
                raise ValueError(f"indexer ridge for {name}, which is not in the hub")
            manifest["indexer_ridges"][name] = {"layers": list(ridge.layers),
                                                "alphas": {str(l): ridge.fit.get(l, {}).get("alpha")
                                                           for l in ridge.layers}}
            arrays = _ridge_arrays(ridge)
            if linked_entry(f"{name}.indexer", arrays):
                continue
            path = directory / f"{name}.indexer.npz"
            np.savez(path, **arrays)
            manifest["files"][f"{name}.indexer"] = {"file": path.name, "sha256": sha256_file(path)}
    if heads:
        manifest["heads"] = {}
        for kind, head in sorted(heads.items()):
            manifest["heads"][kind] = {"base": head.base, "models": {}}
            for name in head.names:
                if name not in models:
                    raise ValueError(f"{kind} heads for {name}, which is not in the hub")
                layers = list(head.layers[name])
                manifest["heads"][kind]["models"][name] = {"layers": layers,
                                                           "alpha": head.alphas.get(name, {}).get("alpha")}
                arrays = _head_arrays(head, name)
                if linked_entry(f"{name}.{kind}", arrays):
                    continue
                path = directory / f"{name}.{kind}.npz"
                np.savez(path, **arrays)
                manifest["files"][f"{name}.{kind}"] = {"file": path.name, "sha256": sha256_file(path)}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def read_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("purpose") != "universal_kv_hub" or manifest.get("format_version") not in (FORMAT, FORMAT_V2):
        raise ValueError(f"{directory} is not a universal KV hub")
    return manifest


def sha256_file(path, block=16 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(block):
            digest.update(chunk)
    return digest.hexdigest()


class Verifier:
    """Checks hub files against their manifest sha256 once per file identity.

    A file whose (size, mtime_ns, inode) matches a stamp recorded when its hash was last
    verified is trusted without rereading it; the stamps live in memory and, with
    `cache_path`, in a small JSON file shared by later processes (an integrity check
    against corrupt or replaced files, not against a user rewriting their own timestamps)."""

    def __init__(self, cache_path=None):
        self.cache_path = Path(cache_path) if cache_path else None
        self.hashed = 0                 # files hashed by this verifier (tests, reports)
        self._lock = threading.Lock()
        self._stamps = {}
        if self.cache_path is not None and self.cache_path.is_file():
            try:
                self._stamps = {k: list(v) for k, v in json.loads(self.cache_path.read_text(encoding="utf-8")).items()}
            except (OSError, ValueError):
                self._stamps = {}

    @staticmethod
    def _identity(status):
        return [status.st_size, status.st_mtime_ns, status.st_ino]

    def check(self, path, sha256):
        path = Path(path).resolve()
        key = str(path)
        before = self._identity(path.stat())
        with self._lock:
            if self._stamps.get(key) == before + [sha256]:
                return
        if sha256_file(path) != sha256:
            raise ValueError(f"{path.name} does not match its manifest hash")
        with self._lock:
            self.hashed += 1
            if self._identity(path.stat()) == before:       # unchanged while it was hashed
                self._stamps[key] = before + [sha256]
                self._save()

    def _save(self):
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            pending = self.cache_path.with_name(f"{self.cache_path.name}.{os.getpid()}.tmp")
            fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self._stamps, stream)
            os.replace(pending, self.cache_path)
        except OSError:
            pass                        # the stamps are an optimisation only


def _flat_views(flat, table):
    """{name: view} of one packed side file; table: name -> [offset (elements), shape]."""
    out = {}
    for name, (offset, shape) in table.items():
        size = int(np.prod(shape)) if shape else 1
        out[name] = flat[offset:offset + size].reshape(shape)
    return out


class Hub:
    """A hub bundle (format 1 or 2) opened lazily: translation reads only the sides it needs.

        hub = open_hub(directory)
        encoder = hub.encoder_side("k", "qwen38")      # EncoderSide (fused encoder)
        decoder = hub.decoder_side("k", "muse")        # DecoderSide

    Format 1 reads the model's npz (hash-verified once, see Verifier) and fuses its encoder
    on first use; format 2 reads one packed .npy per side. Sides are not cached here: the
    caller (a backend, the engine daemon) keeps what it uses. indexer_ridge(name) reads a DSA
    model's IndexerRidge (kept once read)."""

    def __init__(self, directory, *, verifier=None):
        self.directory = Path(directory).resolve()
        self.manifest = read_manifest(self.directory)
        self.version = self.manifest["format_version"]
        self.names = list(self.manifest["models"])
        self.verifier = verifier or Verifier()
        self._source_layers = {}
        self._ridges = {}

    def __repr__(self):
        return f"Hub({str(self.directory)!r}, format {self.version})"

    def target_layers(self, kind, name):
        self._known(name)
        return list(self.manifest["models"][name]["layers"][kind])

    def source_layers(self, kind, name):
        self._known(name)
        if self.version == FORMAT_V2:
            return list(self.manifest["sides"][f"{name}.{kind}.encoder"]["layers"])
        if (name, kind) not in self._source_layers:
            path = self._v1_path(name, kind)
            with np.load(path, allow_pickle=False) as arrays:
                self._source_layers[name, kind] = ([int(l) for l in arrays["source_layers"]]
                                                   if "source_layers" in arrays.files
                                                   else self.target_layers(kind, name))
        return list(self._source_layers[name, kind])

    def _known(self, name):
        if name not in self.manifest["models"]:
            raise ValueError(f"the hub has no adapter for {name}")

    def _v1_path(self, name, kind):
        entry = self.manifest["files"][f"{name}.{kind}"]
        path = self.directory / entry["file"]
        self.verifier.check(path, entry["sha256"])
        return path

    def side_file(self, kind, name, role, dtype="f32"):
        """The hash-verified path of a format 2 side's packed file ("f32", or "f16" when converted with it)."""
        item = self.manifest["sides"][f"{name}.{kind}.{role}"]["files"][dtype]
        path = self.directory / item["file"]
        self.verifier.check(path, item["sha256"])
        return path

    def side_flat(self, kind, name, role, *, dtype="f32", mmap=False):
        """(entry, flat array) of a format 2 side: its packed .npy (entry["arrays"] locates each array)."""
        self._known(name)
        if self.version != FORMAT_V2:
            raise ValueError("packed sides exist only in format 2 hubs (convert_hub.py)")
        entry = self.manifest["sides"][f"{name}.{kind}.{role}"]
        if dtype == "f32" and "f32" not in entry["files"] and "f16" in entry["files"]:
            # a published bundle may carry only the f16 copies: the float32 readers get them widened
            path = self.side_file(kind, name, role, "f16")
            return entry, np.load(path, allow_pickle=False).astype(np.float32)
        path = self.side_file(kind, name, role, dtype)
        return entry, np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)

    def side_arrays(self, kind, name, role, *, dtype="f32", mmap=False):
        """(entry, {name: array}) of a format 2 side, read from its packed .npy."""
        entry, flat = self.side_flat(kind, name, role, dtype=dtype, mmap=mmap)
        return entry, _flat_views(flat, entry["arrays"])

    def encoder_side(self, kind, name):
        if self.version == FORMAT_V2:
            entry, arrays = self.side_arrays(kind, name, "encoder")
            layers = entry["layers"]
            return EncoderSide(name, list(layers), [arrays[f"mean_{l}"] for l in layers],
                               [arrays[f"basis_{l}"] for l in layers], arrays["weight"])
        path = self._v1_path(name, kind)
        with np.load(path, allow_pickle=False) as arrays:
            prefix = "source_" if "source_encoder" in arrays.files else ""
            layers = ([int(l) for l in arrays["source_layers"]] if prefix
                      else self.target_layers(kind, name))
            weight = fuse_encoder(arrays[f"{prefix}basis2"], arrays[f"{prefix}scale"], arrays[f"{prefix}encoder"])
            return EncoderSide(name, layers, [arrays[f"{prefix}mean_{l}"] for l in layers],
                               [arrays[f"{prefix}basis_{l}"] for l in layers], weight)

    def decoder_side(self, kind, name):
        layers = self.target_layers(kind, name)
        if self.version == FORMAT_V2:
            _, arrays = self.side_arrays(kind, name, "decoder")
        else:
            with np.load(self._v1_path(name, kind), allow_pickle=False) as npz:
                arrays = {key: npz[key] for key in ["decoder_z_mean"] + [f"decoder_{l}" for l in layers]
                          + [f"decoder_mean_{l}" for l in layers]}
            arrays["z_mean"] = arrays.pop("decoder_z_mean")
        return DecoderSide(name, layers, arrays["z_mean"], [arrays[f"decoder_{l}"] for l in layers],
                           [arrays[f"decoder_mean_{l}"] for l in layers])

    def side(self, kind, name, role):
        entry = self.manifest.get("heads", {}).get(kind)
        if entry is not None:
            # a head kind: sources encode its base kind (a bundle keeps no own head encoders)
            return (self.encoder_side(entry["base"], name) if role == "encoder"
                    else self.head_decoder_side(kind, name))
        return self.encoder_side(kind, name) if role == "encoder" else self.decoder_side(kind, name)

    def head_decoder_side(self, kind, name):
        """The DecoderSide of one model's head kind: its <name>.<kind>.npz (format 1 in any bundle)."""
        if name not in self.manifest.get("heads", {}).get(kind, {}).get("models", {}):
            raise ValueError(f"the hub has no {kind} head decoders for {name} (fit_hub.py --heads {name})")
        if f"{name}.{kind}" not in self.manifest.get("files", {}):
            raise ValueError(f"{self.directory} lists {kind} heads for {name} without their file (a format "
                             f"{self.version} bundle converted without its heads?)")
        with np.load(self._v1_path(name, kind), allow_pickle=False) as npz:
            layers = [int(l) for l in npz["layers"]]
            return DecoderSide(name, layers, np.array(npz["decoder_z_mean"]),
                               [np.ascontiguousarray(npz[f"decoder_{l}"]) for l in layers],
                               [np.array(npz[f"decoder_mean_{l}"]) for l in layers])

    def head(self, kind):
        """The HeadKind of a head kind (every model's decoders, read now), or None; its encoder sides
        are the base kind's, read through this Hub. Translation reads single sides (side)."""
        entry = self.manifest.get("heads", {}).get(kind)
        if entry is None:
            return None
        head = HeadKind(kind, entry["base"], [], {}, {}, {}, {}, base_kind=_HubKindSides(self, entry["base"]))
        for name, item in entry["models"].items():
            side = self.head_decoder_side(kind, name)
            head.names.append(name)
            head.layers[name] = list(side.layers)
            head.decoder_z_mean[name] = side.z_mean
            head.decoders[name] = list(side.decoders)
            head.decoder_mean[name] = list(side.means)
            head.alphas[name] = {"alpha": item.get("alpha")}
        return head

    def indexer_ridge(self, name):
        """The model's IndexerRidge (read once and kept: a few KB per layer), or None."""
        self._known(name)
        if name not in self._ridges:
            entry = self.manifest.get("indexer_ridges", {}).get(name)
            ridge = None
            if entry is not None:
                if self.version == FORMAT_V2:
                    _, arrays = self.side_arrays("indexer", name, "ridge")
                    layers = [int(l) for l in self.manifest["sides"][f"{name}.indexer.ridge"]["layers"]]
                else:
                    item = self.manifest["files"][f"{name}.indexer"]
                    path = self.directory / item["file"]
                    self.verifier.check(path, item["sha256"])
                    with np.load(path, allow_pickle=False) as npz:
                        arrays = {key: npz[key] for key in npz.files}
                    layers = [int(l) for l in arrays["layers"]]
                fit = {int(l): {"alpha": alpha} for l, alpha in entry.get("alphas", {}).items()}
                ridge = IndexerRidge(name, layers, [np.array(arrays[f"x_mean_{l}"]) for l in layers],
                                     [np.ascontiguousarray(arrays[f"weight_{l}"]) for l in layers],
                                     [np.array(arrays[f"y_mean_{l}"]) for l in layers], fit)
            self._ridges[name] = ridge
        return self._ridges[name]


class _HubKindSides:
    """The encoder sides of one kind of a Hub, as a HeadKind's base."""

    def __init__(self, hub, kind):
        self.hub, self.kind = hub, kind

    def encoder_side(self, name):
        return self.hub.encoder_side(self.kind, name)

    def source_layers(self, name):
        return self.hub.source_layers(self.kind, name)


def open_hub(directory, *, cache_dir=None, verifier=None):
    """A lazily read Hub; with cache_dir, verified-file stamps persist in <cache_dir>/hub-verified.json."""
    if verifier is None:
        verifier = Verifier(Path(cache_dir) / "hub-verified.json" if cache_dir else None)
    return Hub(directory, verifier=verifier)


def sides(hub, kind, source, target):
    """(EncoderSide of source, DecoderSide of target) from {kind: HubKind} or a Hub."""
    if isinstance(hub, Hub):
        return hub.side(kind, source, "encoder"), hub.side(kind, target, "decoder")
    return hub[kind].encoder_side(source), hub[kind].decoder_side(target)


def load_hub(directory, *, verifier=None):
    """({kind: HubKind}, manifest) from a whole bundle directory (format 1 or 2, hash-verified)."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    verifier = verifier or Verifier()
    if manifest["format_version"] == FORMAT_V2:
        return _load_heads(_load_hub_v2(Hub(directory, verifier=verifier)), directory, manifest, verifier), manifest
    hubs = {}
    for kind in KINDS:
        names = list(manifest["models"])
        transforms, encoders, decoders, decoder_mean, z_mean = {}, {}, {}, {}, {}
        source_transforms, source_encoders = {}, {}
        for name in names:
            entry = manifest["files"][f"{name}.{kind}"]
            path = directory / entry["file"]
            verifier.check(path, entry["sha256"])
            with np.load(path, allow_pickle=False) as arrays:
                layers = manifest["models"][name]["layers"][kind]
                transforms[name] = ViewTransform(layers, [arrays[f"mean_{l}"] for l in layers],
                                                 [arrays[f"basis_{l}"] for l in layers], arrays["basis2"],
                                                 arrays["scale"])
                encoders[name] = arrays["encoder"]
                decoders[name] = [arrays[f"decoder_{l}"] for l in layers]
                decoder_mean[name] = [arrays[f"decoder_mean_{l}"] for l in layers]
                z_mean[name] = arrays["decoder_z_mean"]
                if "source_encoder" in arrays:
                    sl = [int(l) for l in arrays["source_layers"]]
                    source_transforms[name] = ViewTransform(sl, [arrays[f"source_mean_{l}"] for l in sl],
                                                            [arrays[f"source_basis_{l}"] for l in sl],
                                                            arrays["source_basis2"], arrays["source_scale"])
                    source_encoders[name] = arrays["source_encoder"]
        hubs[kind] = HubKind(names, transforms, encoders, np.asarray(manifest[f"{kind}_eigenvalues"]),
                             decoder_mean, z_mean, decoders,
                             {name: {"alpha": manifest[f"{kind}_alphas"][name]} for name in names},
                             source_transforms, source_encoders)
    return _load_heads(hubs, directory, manifest, verifier), manifest


def _load_heads(hubs, directory, manifest, verifier):
    """hubs with the bundle's HeadKinds (manifest["heads"]) added."""
    if not manifest.get("heads"):
        return hubs
    hub = Hub(directory, verifier=verifier)
    return with_heads(hubs, [hub.head(kind) for kind in manifest["heads"]])


def _owner(key, names):
    """The model a manifest key ("<name>.<kind>", "<name>.<kind>.<role>") belongs to: the longest
    name it starts with (model names may hold dots)."""
    found = [name for name in names if key.startswith(name + ".")]
    return max(found, key=len) if found else None


def _bundle_entries(manifest):
    """{(table, key): entry} of the files a bundle lists: format 1 "files", format 2 "sides" (and the
    format 1 "files" it copies, DeepSeek V4 heads)."""
    out = {("files", key): entry for key, entry in (manifest.get("files") or {}).items()}
    out.update({("sides", key): entry for key, entry in (manifest.get("sides") or {}).items()})
    return out


def _entry_files(table, entry):
    return [item["file"] for item in entry["files"].values()] if table == "sides" else [entry["file"]]


def merge_bundles(output, bundles):
    """Assemble one bundle `output` (new) from bundles of one format: the first is the base, each
    other one was grown from it by onboarding (fit_hub.py --base: the models it shares with the base
    hold the very same files) and adds its own models. Every file is a hard link, so assembling
    refits nothing and costs no disk; the order does not matter, and two bundles may not add the
    same model. The manifest keeps the base's settings and spans and records each member bundle's
    settings under "assembled". `output` must lie on the members' filesystem (a link across
    filesystems would be a copy: refused before anything is written). Returns the manifest."""
    import copy
    output, bundles = Path(output), [Path(b) for b in bundles]
    manifests = [read_manifest(b) for b in bundles]
    base = manifests[0]
    version = base["format_version"]
    if any(m["format_version"] != version for m in manifests):
        raise ValueError("bundles of different formats cannot be merged (convert_hub.py first)")
    for path, manifest in zip(bundles[1:], manifests[1:]):
        for kind in KINDS:
            if manifest[f"{kind}_eigenvalues"] != base[f"{kind}_eigenvalues"]:
                raise ValueError(f"{path} was not grown from {bundles[0]}: its {kind} hub latent differs")
    devices = {_device(path) for path in bundles}
    if devices != {_device(output)}:
        raise ValueError(f"{output} and the bundles it assembles lie on different filesystems: "
                         "hard links cannot cross them (assemble next to the members)")
    merged = copy.deepcopy(base)
    entries = {item: (bundles[0], entry) for item, entry in _bundle_entries(base).items()}
    base_names = set(base["models"])
    for path, manifest in zip(bundles[1:], manifests[1:]):
        names = set(manifest["models"])
        for (table, key), entry in _bundle_entries(manifest).items():
            owner = _owner(key, names)
            if owner in base_names:
                if entries.get((table, key), (None, None))[1] != entry:
                    raise ValueError(f"{path}: {key} differs from the base bundle's (not grown from it)")
                continue
            if (table, key) in entries:
                raise ValueError(f"{key} is in {entries[table, key][0]} and in {path}: a model added twice")
            entries[table, key] = (path, entry)
        for name in sorted(names - base_names):
            merged["models"][name] = manifest["models"][name]
            for kind in KINDS:
                merged[f"{kind}_alphas"][name] = manifest[f"{kind}_alphas"][name]
            if name in manifest.get("indexer_ridges", {}):
                merged.setdefault("indexer_ridges", {})[name] = manifest["indexer_ridges"][name]
            for kind, head in (manifest.get("heads") or {}).items():
                if name in head["models"]:
                    merged.setdefault("heads", {}).setdefault(kind, {"base": head["base"], "models": {}})
                    merged["heads"][kind]["models"][name] = head["models"][name]
            if version == FORMAT_V2:
                for key, item in manifest["converted_from"]["files"].items():
                    if _owner(key, names) == name:
                        merged["converted_from"]["files"][key] = item
        merged.setdefault("assembled", {})[str(path.resolve())] = {
            "models": sorted(names - base_names), "settings": manifest.get("settings")}
    output.mkdir(parents=True, exist_ok=False)
    merged["files"], merged["sides"] = {}, {}
    for (table, key), (path, entry) in sorted(entries.items()):
        for name in _entry_files(table, entry):
            link_file(path / name, output / name)
        merged[table][key] = entry
    if version == FORMAT:
        merged.pop("sides")
    elif not merged["files"]:
        merged.pop("files")
    merged["assembled_base"] = str(bundles[0].resolve())
    (output / "manifest.json").write_text(json.dumps(merged, indent=1) + "\n", encoding="utf-8")
    return merged


def load_indexer_ridges(directory, *, verifier=None):
    """{name: IndexerRidge} of a bundle (format 1 or 2), hash-verified."""
    hub = Hub(directory, verifier=verifier)
    return {name: hub.indexer_ridge(name) for name in hub.manifest.get("indexer_ridges", {})}


def _load_hub_v2(hub):
    """HubKinds from a format 2 bundle: the view role holds what translation does not need."""
    manifest, hubs = hub.manifest, {}
    for kind in KINDS:
        transforms, encoders, decoders, decoder_mean, z_mean = {}, {}, {}, {}, {}
        source_transforms, source_encoders, fused = {}, {}, {}
        for name in hub.names:
            layers = hub.target_layers(kind, name)
            entry, enc = hub.side_arrays(kind, name, "encoder")
            _, view = hub.side_arrays(kind, name, "view")
            _, dec = hub.side_arrays(kind, name, "decoder")
            full = view if entry.get("subset") else enc
            transforms[name] = ViewTransform(layers, [full[f"mean_{l}"] for l in layers],
                                             [full[f"basis_{l}"] for l in layers], view["basis2"], view["scale"])
            encoders[name] = view["encoder"]
            if entry.get("subset"):
                sl = list(entry["layers"])
                source_transforms[name] = ViewTransform(sl, [enc[f"mean_{l}"] for l in sl],
                                                        [enc[f"basis_{l}"] for l in sl], view["source_basis2"],
                                                        view["source_scale"])
                source_encoders[name] = view["source_encoder"]
            used = source_transforms.get(name) or transforms[name]
            fused[name] = ((used.basis2, used.scale, source_encoders.get(name, encoders[name])), enc["weight"])
            decoders[name] = [dec[f"decoder_{l}"] for l in layers]
            decoder_mean[name] = [dec[f"decoder_mean_{l}"] for l in layers]
            z_mean[name] = dec["z_mean"]
        hubs[kind] = HubKind(list(hub.names), transforms, encoders, np.asarray(manifest[f"{kind}_eigenvalues"]),
                             decoder_mean, z_mean, decoders,
                             {name: {"alpha": manifest[f"{kind}_alphas"][name]} for name in hub.names},
                             source_transforms, source_encoders, fused)
    return hubs
