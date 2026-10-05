"""Build a Muse slot state from a Qwen slot state: the cross-model cache handoff (design A).

Muse continues a conversation that Qwen has processed. The handoff point is Muse token
position T0 (usually P - W, P = the first Muse token of the final assistant message, so
Muse natively decodes the W-token tail [T0, P) on top of the built cache). The built
llama_state_seq_save_file state holds:

* the base cache (full, NoPE layers): a cell for every position in [0, T0);
* the SWA cache (sliding layers, RoPE NORM at Muse positions): the last `swa_cells`
  positions before T0 (default n_swa; cells older than T0 - n_swa are never attended
  and only exist so llama-server does not re-process the prompt).

Every cell's K/V comes from a policy. Aligned content tokens (align.py rules: same
segment, same end byte in both renderings, and the Qwen token inside the Qwen state)
use `content_policy`: `mapped` (fit_layers.apply on the Qwen K/V, K de-rotated from the
Qwen cache with kvstate.pre_rope_keys), `oracle` (Muse's own cache from a native Muse
state) or `mean` (the mapper's train target mean, K renormalized per head: the v2
mean_all content rows). All other tokens (BOS, template, special, unmatched,
straddling, or aligned beyond the Qwen state) use `fill_policy`: `oracle`, `table`
(per-token-id mean of native Muse K/V over non-aligned train rows, build_token_table;
unseen ids take the per-layer mean, except unseen ids of content tokens, which take the
mapper's content mean when a mapper is given) or `mean`.

Floors: v2 mean_all (inject.py) replaced only the aligned content cells and kept BOS,
template and special cells native, so the condition matching it here is content `mean`
with fill `oracle` (or fill `table` when no native cells may be used). mean/mean also
puts the content mean (a mean over aligned content rows) into every fill cell,
including the BOS attention sink, so it is a harsher floor than v2 mean_all and
contrasts against it are not comparable with v2's. The report's `floor` names which.

Computed K rows are pre-RoPE and rotated at their Muse positions with kvstate.cache_keys on sliding layers; oracle rows are
copied from the native state as stored. Table K rows (unless disabled) and mean K rows
have per-head RMS 1 like every native Muse K row (k_norm weight is ones). Oracle cells
that a native state cannot hold (sliding cells older than t0 - n_swa, which no query
at or after t0 attends) fall back to the table, else the mean.

Memory: the Qwen state stays as parsed (views of the file bytes); mapper maps are cached
while building (~1.7 GB at production size for 104 maps with k = 16) and source rows are
processed in chunks of `chunk_rows` positions.

Subcommands: `build` writes a Muse slot file (plain or server-packed token words,
optional empty `.resume` sidecar) plus a JSON report; `token-table` writes the token
table NPZ from target captures and an alignment, streaming one (kind, layer) at a time.
"""

import os

if __name__ == "__main__":
    for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, "8")

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import operator
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

import align as align_module
from align import ALIGNED, EXCLUDED, STRADDLING, TEMPLATE, UNMATCHED, segment_table
import kvstate


KINDS = ("k", "v")
CONTENT_POLICIES = ("mapped", "oracle", "mean")
FILL_POLICIES = ("oracle", "table", "mean")
TABLE_FORMAT = 1
TABLE_PURPOSE = "kv_handoff_token_table"
CLASS_NAMES = {TEMPLATE: "template_or_special", ALIGNED: "aligned", UNMATCHED: "unmatched",
               STRADDLING: "straddling"}
RMS_FLOOR = 1e-12
# What a content `mean` state is a floor of, by fill policy (reported; see the module docstring).
MEAN_FLOORS = {"oracle": "v2_mean_all: content cells get the mean, fill cells stay native",
               "table": "content cells get the mean, fill cells come from the token table",
               "mean": "harsher than v2_mean_all: every cell, BOS and template included, gets the content mean"}


def renormalize_heads(values):
    """[n, heads, dim] float32 with every head vector scaled to RMS 1 (as fit_layers.renormalize_heads)."""
    values = np.asarray(values, dtype=np.float32)
    rms = np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=-1, keepdims=True))
    return (values / np.maximum(rms, RMS_FLOOR)).astype(np.float32)


def _integer(value, what, low, high=None):
    """int(value) for any integer type (numpy included; not bool or float) within [low, high]."""
    bad = f"{what} must be an integer in [{low}, {'inf' if high is None else high}]"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(bad)
    try:
        value = operator.index(value)
    except TypeError:
        raise ValueError(bad) from None
    if value < low or (high is not None and value > high):
        raise ValueError(bad)
    return int(value)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, data):
    """Create path atomically with bytes or str content; never replaces an existing path."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    if isinstance(data, str):
        data = data.encode("utf-8")
    return kvstate.write_new(path, data)


# -- tokens and alignment -----------------------------------------------------------

def normalize_tokens(tokens, what):
    """{token_id int32, byte_end int64, special uint8} from a dict or an object with those attributes."""
    get = tokens.get if isinstance(tokens, dict) else (lambda name: getattr(tokens, name, None))
    arrays = {}
    for name, dtype in (("token_id", np.int32), ("byte_end", np.int64), ("special", np.uint8)):
        value = get(name)
        if value is None:
            raise ValueError(f"{what} tokens lack {name}")
        value = np.asarray(value)
        if value.ndim != 1 or value.dtype.kind not in "iub":
            raise ValueError(f"{what} {name} must be a 1-D integer array")
        arrays[name] = value.astype(dtype)
    n = len(arrays["token_id"])
    if n == 0 or any(len(value) != n for value in arrays.values()):
        raise ValueError(f"{what} token arrays must be nonempty and of equal length")
    # a token may spell nothing (byte_end repeats; captures.Capture): kv_probe empties the space a
    # SentencePiece tokenizer puts after a special token
    ends = arrays["byte_end"]
    if (ends[0] < 0 or ends[-1] <= 0 or np.any(np.diff(ends) < 0) or np.any(arrays["token_id"] < 0)
            or np.any(arrays["special"] > 1)):
        raise ValueError(f"{what} tokens need nondecreasing byte ends that reach past 0, ids >= 0, special 0/1")
    return arrays


class _DocumentTokens:
    """One document's tokenization with the CaptureSet interface align.align reads."""

    def __init__(self, document_id, tokens):
        self.document_id = document_id
        self.token_id, self.byte_end, self.special = tokens["token_id"], tokens["byte_end"], tokens["special"]
        self.byte_start = np.concatenate([[0], self.byte_end[:-1]]).astype(np.int64)
        self.rows_total = len(self.token_id)
        self.document_ids = [document_id]

    def __contains__(self, document_id):
        return document_id == self.document_id

    def rows(self, document_id):
        if document_id != self.document_id:
            raise KeyError(document_id)
        return slice(0, self.rows_total)


def align_document(qwen_record, qwen_tokens, muse_record, muse_tokens):
    """align.align on one document (source Qwen, target Muse) from token arrays; rejects bad documents."""
    document_id = muse_record.get("document_id")
    if qwen_record.get("document_id") != document_id or not isinstance(document_id, str):
        raise ValueError("Qwen and Muse records must render the same document")
    alignment = align_module.align([qwen_record], _DocumentTokens(document_id, qwen_tokens),
                                   [muse_record], _DocumentTokens(document_id, muse_tokens))
    if alignment.rejected:
        raise ValueError(f"alignment rejected {document_id}: {alignment.rejected[0]['reason']}")
    return alignment


def final_message_start(record):
    """Start byte of the record's final message (its first rendered final-message segment)."""
    ids, starts, _ = segment_table(record)
    final = set(record.get("final_message_segments") or [])
    chosen = [int(start) for index, start in zip(ids, starts) if int(index) in final]
    if not chosen:
        raise ValueError(f"record {record.get('document_id')} renders no final-message segment")
    return min(chosen)


def handoff_position(record, tokens):
    """P: the number of tokens ending at or before the final message start (its first token's position)."""
    tokens = normalize_tokens(tokens, "record")
    return int(np.searchsorted(tokens["byte_end"], final_message_start(record), side="right"))


# -- token table ----------------------------------------------------------------------

def check_alignment_rows(target_set, alignment):
    """Refuse a target capture set whose flattened rows are not the ones the alignment was built on.

    align.align visits documents in capture-set order and stores global target rows, so the
    set must hold the same documents at the same rows (e.g. capture directories given in the
    same order). The alignment stores no token ids; this checks the row total, the document
    order, every pair's row against its document's rows and local position, and that the
    classified (non-excluded) rows are exactly the aligned documents' rows.
    """
    if len(alignment.tgt_class) != target_set.rows_total:
        raise ValueError("alignment was built for a target capture set with a different row count")
    missing = [d for d in alignment.document_ids if d not in target_set]
    if missing:
        raise ValueError(f"target captures lack aligned document {missing[0]}")
    order = {d: i for i, d in enumerate(target_set.document_ids)}
    index = np.asarray([order[d] for d in alignment.document_ids], dtype=np.int64)
    if np.any(np.diff(index) <= 0):
        raise ValueError("target captures list the aligned documents in a different order than the alignment "
                         "(give the capture directories in the order the alignment was built with)")
    windows = [target_set.rows(d) for d in alignment.document_ids]
    starts = np.asarray([w.start for w in windows], dtype=np.int64)
    stops = np.asarray([w.stop for w in windows], dtype=np.int64)
    doc = np.asarray(alignment.doc, dtype=np.int64)
    if len(doc) and (doc.min() < 0 or doc.max() >= len(windows)):
        raise ValueError("alignment pairs name unknown documents")
    tgt_row, tgt_position = np.asarray(alignment.tgt_row), np.asarray(alignment.tgt_position)
    if np.any(tgt_row != starts[doc] + tgt_position) or np.any(tgt_row >= stops[doc]):
        raise ValueError("alignment target rows fall outside their documents' rows in these captures")
    classified = alignment.tgt_class != EXCLUDED
    inside = np.zeros(target_set.rows_total, dtype=bool)
    for window in windows:
        inside[window] = True
    if np.any(classified != inside) or np.any(alignment.tgt_class[tgt_row] != ALIGNED):
        raise ValueError("alignment row classes do not match the aligned documents' rows in these captures")


def build_token_table(target_set, alignment, *, split="train", chunk_rows=16384, progress=None):
    """Per-(kind, layer, token id) mean native K (pre-RoPE) / V over non-aligned rows of `split`.

    target_set: the target CaptureSet the alignment was built on (same flattened rows).
    Returns {"arrays": {name: array}, "metadata": dict}; arrays are `token_id` int32 [U]
    (ascending), `count` int64 [U], `<kind>-<layer>` float32 [U, heads, dim] and
    `mean-<kind>-<layer>` float32 [heads, dim] (mean over all selected rows). Streams one
    (kind, layer) at a time in chunks of `chunk_rows` rows; float64 accumulation.
    """
    chunk_rows = _integer(chunk_rows, "chunk_rows", 1)
    check_alignment_rows(target_set, alignment)
    documents =[d for d, s in zip(alignment.document_ids, alignment.splits) if s == split]
    if not documents:
        raise ValueError(f"alignment has no {split} documents")
    rows = np.concatenate([np.arange(target_set.rows(d).start, target_set.rows(d).stop) for d in documents])
    classes = alignment.tgt_class[rows]
    if np.any(classes == EXCLUDED):
        raise ValueError("aligned documents contain excluded target rows")
    class_counts = {CLASS_NAMES[value]: int(np.sum(classes == value)) for value in CLASS_NAMES}
    rows = rows[classes != ALIGNED]
    if not len(rows):
        raise ValueError(f"no non-aligned target rows in split {split}")
    token_ids, inverse = np.unique(target_set.token_id[rows], return_inverse=True)
    inverse = inverse.reshape(-1)
    count = np.bincount(inverse, minlength=len(token_ids)).astype(np.int64)
    layers = target_set.layers("k")
    if not layers or target_set.layers("v") != layers:
        raise ValueError("target captures need k and v on the same layers")
    arrays = {"token_id": token_ids.astype(np.int32), "count": count}
    shapes = {}
    started = time.perf_counter()
    for layer in layers:
        for kind in KINDS:
            sums, total, shape = None, None, None
            for start in range(0, len(rows), chunk_rows):
                block = target_set.gather(kind, layer, rows[start:start + chunk_rows])
                if not np.isfinite(block).all():
                    raise ValueError(f"{kind}-{layer} holds nonfinite values")
                if sums is None:
                    shape = block.shape[1:]
                    sums = np.zeros((len(token_ids), int(np.prod(shape))), dtype=np.float64)
                    total = np.zeros(int(np.prod(shape)), dtype=np.float64)
                flat = block.reshape(len(block), -1).astype(np.float64)
                local = inverse[start:start + chunk_rows]
                order = np.argsort(local, kind="stable")
                groups, first = np.unique(local[order], return_index=True)
                sums[groups] += np.add.reduceat(flat[order], first, axis=0)
                total += flat.sum(axis=0)
            arrays[f"{kind}-{layer}"] = (sums / count[:, None]).astype(np.float32).reshape((-1,) + shape)
            arrays[f"mean-{kind}-{layer}"] = (total / len(rows)).astype(np.float32).reshape(shape)
            shapes[kind] = list(shape)
            if progress is not None:
                progress({"event": "token_table", "kind": kind, "layer": layer,
                          "seconds": round(time.perf_counter() - started, 3)})
    metadata = {"format_version": TABLE_FORMAT, "purpose": TABLE_PURPOSE, "split": split,
                "architecture": target_set.architecture, "layers": list(layers), "head_shape": shapes,
                "rows": int(len(rows)), "unique_tokens": int(len(token_ids)), "documents": len(documents),
                "split_target_tokens": class_counts,
                "definition": "mean of native target K (pre-RoPE, Kcur_normed) and V per token id over the "
                              "split's target rows that are not aligned content (template/special, unmatched, "
                              "straddling); mean-<kind>-<layer> averages all those rows"}
    return {"arrays": arrays, "metadata": metadata}


def write_token_table(path, table, provenance=None):
    """Write the table NPZ (uncompressed, so members load lazily) atomically; never overwrites."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    metadata = dict(table["metadata"], **(provenance or {}))
    with tempfile.TemporaryDirectory(prefix=".kvpark-token-table-", dir=path.parent) as temporary:
        pending = Path(temporary) / "table.npz"
        with pending.open("xb") as stream:
            np.savez(stream, **table["arrays"], metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
        os.chmod(pending, 0o644)
        os.link(pending, path)
    return path


class TokenTable:
    """Reader of a token table NPZ; (kind, layer) arrays load on first use."""

    def __init__(self, path):
        self.path = Path(path)
        self._data = np.load(self.path, allow_pickle=False)
        try:
            self.metadata = json.loads(str(self._data["metadata_json"]))
        except KeyError:
            raise ValueError(f"{self.path} is not a token table") from None
        if self.metadata.get("format_version") != TABLE_FORMAT or self.metadata.get("purpose") != TABLE_PURPOSE:
            raise ValueError(f"{self.path} is not a token table (format {TABLE_FORMAT})")
        self.token_id = np.asarray(self._data["token_id"])
        self.count = np.asarray(self._data["count"])
        if (self.token_id.ndim != 1 or len(self.token_id) != len(self.count)
                or np.any(np.diff(self.token_id) <= 0) or np.any(self.count <= 0)):
            raise ValueError("token table ids must be strictly increasing with positive counts")
        self._cache = {}

    def _array(self, name):
        if name not in self._cache:
            if name not in self._data.files:
                raise KeyError(f"token table has no {name}")
            value = np.asarray(self._data[name], dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f"token table {name} is nonfinite")
            self._cache[name] = value
        return self._cache[name]

    def mean(self, kind, layer):
        return self._array(f"mean-{kind}-{layer}")

    def lookup(self, kind, layer, token_ids):
        """(float32 [n, heads, dim], unseen mask): per-id means, unseen ids take the per-layer mean."""
        token_ids = np.asarray(token_ids, dtype=np.int64)
        table, mean = self._array(f"{kind}-{layer}"), self.mean(kind, layer)
        if table.shape[1:] != mean.shape or len(table) != len(self.token_id):
            raise ValueError(f"token table {kind}-{layer} is malformed")
        index = np.searchsorted(self.token_id, token_ids).clip(0, max(len(self.token_id) - 1, 0))
        seen = (self.token_id[index] == token_ids) if len(self.token_id) else np.zeros(len(token_ids), bool)
        rows = np.empty((len(token_ids),) + mean.shape, dtype=np.float32)
        rows[seen] = table[index[seen]]
        rows[~seen] = mean
        return rows, ~seen

    def close(self):
        self._data.close()


# -- sources --------------------------------------------------------------------------

class _PositionIndex(dict):
    """position -> cell index of a parsed stream. Positions that several cells share (every
    cell of an M-RoPE image sits at the image's first position) are kept apart in `shared`:
    they are not keys, and _cells refuses to look one up."""

    def __init__(self, items, shared=()):
        super().__init__(items)
        self.shared = frozenset(shared)


def _stream_index(stream, what, pos_cut=None):
    """_PositionIndex of a stream's cells; with `pos_cut` (a position, not a token-list index),
    cells at positions >= pos_cut are left out. kvstate.truncate_state(state, cut) drops the
    cells from MediaTokens.pos_next(cut) on, so pass that: after an M-RoPE image the position
    and the token index differ. Shared positions are an error only when looked up, so media
    cells past the cut never get in the way (`what` stays for the callers; _cells names the
    source in its errors)."""
    positions = np.asarray(stream.positions, dtype=np.int64)
    if pos_cut is not None:
        positions = np.where(positions < pos_cut, positions, -1)
    unique, counts = np.unique(positions[positions >= 0], return_counts=True)
    shared = set(unique[counts > 1].tolist())
    return _PositionIndex({int(p): i for i, p in enumerate(positions) if p >= 0 and int(p) not in shared}, shared)


def _cells(index, positions, what):
    shared = sorted({int(p) for p in positions} & getattr(index, "shared", frozenset()))
    if shared:
        raise ValueError(f"{what} holds several cells at positions {shared[:5]}{' ...' if len(shared) > 5 else ''} "
                         "(media cells share their position; cut the state before them)")
    missing = [int(p) for p in positions if int(p) not in index]
    if missing:
        raise ValueError(f"{what} lacks positions {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return np.fromiter((index[int(p)] for p in positions), dtype=np.int64, count=len(positions))


class QwenSource:
    """Pre-RoPE K and V rows of a parsed Qwen state (attention section), by Qwen position."""

    def __init__(self, state_file, tokens):
        spec = state_file.state.spec
        if spec.architecture != "qwen35":
            raise ValueError("the source state must be a qwen35 state")
        self.spec, self.tokens = spec, np.asarray(state_file.tokens)
        n = len(self.tokens)
        if n == 0 or n > len(tokens["token_id"]) or not np.array_equal(self.tokens, tokens["token_id"][:n]):
            raise ValueError("Qwen state tokens are not a prefix of the rendered Qwen tokenization")
        self.stream = state_file.state.part("attn").stream()
        positions = np.asarray(self.stream.positions, dtype=np.int64)
        if len(positions) != n or not np.array_equal(np.sort(positions), np.arange(n)):
            raise ValueError(f"Qwen attention cells must cover positions 0..{n - 1} exactly once")
        if self.stream.ext is not None and not np.array_equal(self.stream.ext[:, 2], self.tokens[positions]):
            raise ValueError("Qwen cell token ids differ from the state's token list")
        self.index = _stream_index(self.stream, "Qwen attention cache")
        self.layers = tuple(state_file.state.part("attn").layers)
        self.n_tokens = n

    def rows(self, kind, positions, layers):
        """{layer: float32 [n, heads, dim]}; K de-rotated at the Qwen positions (all layers in one call)."""
        cells = _cells(self.index, positions, "Qwen state")
        layers = list(layers)
        unknown = set(layers) - set(self.layers)
        if unknown:
            raise ValueError(f"Qwen state has no attention layers {sorted(unknown)}")
        if kind == "v":
            return {layer: np.asarray(self.stream.v[layer][cells], dtype=np.float32) for layer in layers}
        stored = [np.asarray(self.stream.k[layer][cells], dtype=np.float32) for layer in layers]
        heads = [array.shape[1] for array in stored]
        stacked = kvstate.pre_rope_keys(self.spec, layers[0], np.concatenate(stored, axis=1),
                                        np.asarray(positions, dtype=np.int64))
        return dict(zip(layers, np.split(stacked, np.cumsum(heads)[:-1], axis=1)))


class OracleSource:
    """Native Muse cache rows (as stored: post-RoPE K on sliding layers) of a parsed Muse state."""

    def __init__(self, state_file, muse_tokens):
        spec = state_file.state.spec
        if spec.architecture != "muse-glimmer":
            raise ValueError("the oracle state must be a muse-glimmer state")
        tokens = np.asarray(state_file.tokens)
        if (len(tokens) == 0 or len(tokens) > len(muse_tokens["token_id"])
                or not np.array_equal(tokens, muse_tokens["token_id"][:len(tokens)])):
            raise ValueError("oracle state tokens are not a prefix of the rendered Muse tokenization")
        self.n_tokens = len(tokens)
        self.streams = {name: state_file.state.part(name).stream() for name in ("base", "swa")}
        self.index = {name: _stream_index(stream, f"oracle {name} cache") for name, stream in self.streams.items()}
        self.spec = spec

    def section(self, layer):
        return "swa" if self.spec.layer_types[layer] == "sliding" else "base"

    def has(self, layer, positions):
        index = self.index[self.section(layer)]
        return np.asarray([int(p) in index for p in positions], dtype=bool)

    def rows(self, kind, layer, positions):
        name = self.section(layer)
        cells = _cells(self.index[name], positions, f"oracle {name} cache")
        stream = self.streams[name]
        return np.asarray((stream.k if kind == "k" else stream.v)[layer][cells])


# -- builder --------------------------------------------------------------------------

SOURCE_NAMES = ("oracle", "mapped", "table", "table_unseen", "mean")


@dataclass
class Handoff:
    """A built Muse state, its report, and per-section cell provenance (for checks)."""

    state: kvstate.StateFile
    report: dict
    cells: dict = field(default_factory=dict)   # section -> {"positions", "content", "source" (per layer)}


def _check_mapper(mapper, muse_spec, qwen_spec, pairs):
    for kind in KINDS:
        layer = muse_spec.layers_of("full", "sliding")[0]
        if tuple(mapper.target_shape[kind]) != muse_spec.head_shape(layer, kind):
            raise ValueError(f"mapper {kind} target shape {mapper.target_shape[kind]} differs from Muse "
                             f"{muse_spec.head_shape(layer, kind)}")
        if qwen_spec is not None:
            source = qwen_spec.layers_of("full")[0]
            if tuple(mapper.source_shape[kind]) != qwen_spec.head_shape(source, kind):
                raise ValueError(f"mapper {kind} source shape {mapper.source_shape[kind]} differs from Qwen "
                                 f"{qwen_spec.head_shape(source, kind)}")
    for architecture, expected in ((mapper.source_architecture, "qwen35"), (mapper.target_architecture,
                                                                           "muse-glimmer")):
        if architecture not in (None, expected):
            raise ValueError(f"mapper architectures must be qwen35 -> muse-glimmer, got {architecture}")
    missing = [f"{kind}-{layer}" for kind, layer in pairs if not mapper.has(kind, layer)]
    if missing:
        raise ValueError(f"mapper lacks maps {missing[:6]}")


def _as_mapper(mapper):
    if mapper is None or hasattr(mapper, "predict"):
        return mapper
    import attention_eval
    return attention_eval.Mapper(mapper)


def build_handoff(qwen_state, qwen_record, qwen_tokens, muse_record, muse_tokens, mapper=None, *, t0,
                  muse_spec=None, swa_cells=None, content_policy="mapped", fill_policy="table",
                  oracle_state=None, table=None, dtype="f16", v_trans=False, server_tokens=False,
                  n_stream=1, stream=0, seq_id=0, renorm_table_k=True, chunk_rows=2048, native_prefix=0,
                  native_mask=None):
    """Build the Muse state for positions [0, t0); see the module docstring for the policies.

    qwen_state: parsed Qwen kvstate.StateFile (None only when content_policy is not
    `mapped`; then every aligned pair counts as content). qwen_tokens / muse_tokens:
    token_id, byte_end, special of each model's rendered record (e.g. kv_probe tokenize);
    the Qwen state's tokens must be a prefix of qwen_tokens. mapper: attention_eval.Mapper
    or a fit_layers mapper directory (needed by `mapped` and `mean`). oracle_state: native
    Muse kvstate.StateFile whose tokens are a prefix of muse_tokens with >= t0 tokens,
    needed by `oracle`. table: TokenTable or NPZ path, needed by `table`.
    dtype / v_trans must match the receiving context (cache type; v_trans = flash-attn off).
    native_prefix H: positions [0, H) take Muse's native values from oracle_state whatever
    their class (the hybrid: e.g. the system prompt and tool schemas prefilled natively by
    Muse, only the conversation history mapped); oracle_state then needs >= H tokens, or
    >= t0 when a policy is `oracle`. native_mask (bool per Muse token) does the same for
    any set of positions (a diagnostic: e.g. tool-result messages native, the rest mapped).
    """
    started = time.perf_counter()
    if content_policy not in CONTENT_POLICIES or fill_policy not in FILL_POLICIES:
        raise ValueError(f"content policy must be one of {CONTENT_POLICIES}, fill policy one of {FILL_POLICIES}")
    qwen_tokens, muse_tokens = normalize_tokens(qwen_tokens, "Qwen"), normalize_tokens(muse_tokens, "Muse")
    if muse_spec is None:
        if oracle_state is None:
            raise ValueError("muse_spec is required without an oracle state")
        muse_spec = oracle_state.state.spec
    if muse_spec.architecture != "muse-glimmer":
        raise ValueError("muse_spec must describe a muse-glimmer model")
    if oracle_state is not None and oracle_state.state.spec != muse_spec:
        raise ValueError("the oracle state was parsed with a different Muse spec")
    if dtype not in kvstate.DTYPE_NAMES:
        raise ValueError("dtype must be f16 or f32")
    t0 = _integer(t0, "t0", 1, len(muse_tokens["token_id"]))
    native_prefix = _integer(native_prefix, "native_prefix", 0, t0)
    native = np.zeros(len(muse_tokens["token_id"]), dtype=bool)
    native[:native_prefix] = True
    if native_mask is not None:
        native_mask = np.asarray(native_mask, dtype=bool)
        if native_mask.shape != native.shape:
            raise ValueError("native_mask needs one entry per Muse token")
        native |= native_mask
    native_end = int(np.flatnonzero(native[:t0])[-1]) + 1 if native[:t0].any() else 0
    swa_cells = _integer(muse_spec.n_swa if swa_cells is None else swa_cells, "swa_cells", 1)
    chunk_rows = _integer(chunk_rows, "chunk_rows", 1)
    policies = {content_policy, fill_policy}
    if "oracle" in policies and oracle_state is None:
        raise ValueError("the oracle policy needs oracle_state")
    if native_end and oracle_state is None:
        raise ValueError("a native prefix needs oracle_state")
    needed = t0 if "oracle" in policies else native_end
    if oracle_state is not None and len(oracle_state.tokens) < needed:
        raise ValueError(f"the oracle state holds {len(oracle_state.tokens)} tokens, fewer than "
                         + (f"t0 = {t0}" if needed == t0 else f"the native prefix = {native_end}"))
    if content_policy == "mapped" and qwen_state is None:
        raise ValueError("the mapped policy needs qwen_state")
    mapper = _as_mapper(mapper)
    if ({"mapped", "mean"} & policies) and mapper is None:
        raise ValueError("the mapped and mean policies need a mapper")
    if isinstance(table, (str, os.PathLike)):
        table = TokenTable(table)
    if fill_policy == "table" and table is None:
        raise ValueError("the table fill policy needs a token table")
    if table is not None and table.metadata.get("architecture") not in (None, "muse-glimmer"):
        raise ValueError("the token table was not built from Muse captures")

    alignment = align_document(qwen_record, qwen_tokens, muse_record, muse_tokens)
    qwen = QwenSource(qwen_state, qwen_tokens) if qwen_state is not None else None
    oracle = OracleSource(oracle_state, muse_tokens) if oracle_state is not None else None
    attention = muse_spec.layers_of("full", "sliding")
    if mapper is not None:
        needed = [(kind, layer) for kind in KINDS for layer in attention] if {"mapped", "mean"} & policies else []
        _check_mapper(mapper, muse_spec, None if qwen is None else qwen.spec, needed)
        if content_policy == "mapped":
            sources = {source for kind in KINDS for target in attention
                       for source in mapper.source_layers(kind, target)[1]}
            unknown = sources - set(qwen.layers)
            if unknown:
                raise ValueError(f"mapper reads Qwen layers {sorted(unknown)} that the Qwen state lacks")

    n_source = qwen.n_tokens if qwen is not None else len(qwen_tokens["token_id"])
    target_class = alignment.tgt_class
    source_of = np.full(len(muse_tokens["token_id"]), -1, dtype=np.int64)
    source_of[alignment.tgt_position] = alignment.src_position
    content = (target_class == ALIGNED) & (source_of >= 0) & (source_of < n_source)
    beyond = int(np.sum((target_class[:t0] == ALIGNED) & ~content[:t0]))

    shapes = {kind: muse_spec.head_shape(attention[0], kind) for kind in KINDS}

    def fill_rows(kind, layer, positions, policy, source):
        """float32 pre-RoPE rows for a table/mean policy; records sources."""
        if policy == "table":
            rows, unseen = table.lookup(kind, layer, muse_tokens["token_id"][positions])
            source[:] = np.where(unseen, SOURCE_NAMES.index("table_unseen"), SOURCE_NAMES.index("table"))
            # The table's per-layer fallback is mostly template rows; an unseen id of a content
            # token (unmatched, straddling, or aligned past the Qwen state) takes the content mean.
            content_unseen = unseen & (target_class[positions] != TEMPLATE)
            if mapper is not None and content_unseen.any():
                rows = np.array(rows, dtype=np.float32, copy=True)
                rows[content_unseen] = mapper.floor(kind, layer, shapes[kind])
        else:
            rows = np.broadcast_to(mapper.floor(kind, layer, shapes[kind]),
                                   (len(positions),) + shapes[kind]).astype(np.float32)
            source[:] = SOURCE_NAMES.index("mean")
        if kind == "k" and renorm_table_k and policy == "table":
            rows = renormalize_heads(rows)
        return rows

    def mapped_rows(positions, layers):
        """{(kind, layer): float32 [n, heads, dim]} mapper predictions from the Qwen rows (row chunks)."""
        out = {(kind, layer): np.empty((len(positions),) + shapes[kind], dtype=np.float32)
               for kind in KINDS for layer in layers}
        for start in range(0, len(positions), chunk_rows):
            src = source_of[positions[start:start + chunk_rows]]
            for kind in KINDS:
                wanted = sorted({s for layer in layers for s in mapper.source_layers(kind, layer)[1]})
                rows = qwen.rows(kind, src, wanted)
                for layer in layers:
                    out[kind, layer][start:start + len(src)] = mapper.predict(kind, layer, rows, shapes[kind])
        return out

    fallback_policy = "table" if table is not None else "mean"

    def section(name, layers, positions):
        n = len(positions)
        is_content = content[positions]
        policy = np.where(is_content, content_policy, fill_policy).astype("<U6")
        policy[native[positions]] = "oracle"
        values = {(kind, layer): np.empty((n,) + shapes[kind], dtype=np.float32) for kind in KINDS for layer in layers}
        sources = {layer: np.full(n, -1, dtype=np.int8) for layer in layers}
        fallback = {layer: np.zeros(n, dtype=bool) for layer in layers}
        mapped_at = np.flatnonzero(policy == "mapped")
        mapped = mapped_rows(positions[mapped_at], layers) if len(mapped_at) else {}
        computed = {}
        for layer in layers:
            oracle_at = np.flatnonzero(policy == "oracle")
            if len(oracle_at):
                present = oracle.has(layer, positions[oracle_at])
                if not present.all():
                    # only sliding cells that no query at >= t0 can attend may lack native values
                    lacking = positions[oracle_at[~present]]
                    if muse_spec.layer_types[layer] != "sliding" or lacking.max() >= t0 - muse_spec.n_swa:
                        raise ValueError(f"oracle state lacks layer {layer} cells at positions {lacking[:5].tolist()}")
                    fallback[layer][oracle_at[~present]] = True
                    oracle_at = oracle_at[present]
                for kind in KINDS:
                    values[kind, layer][oracle_at] = oracle.rows(kind, layer, positions[oracle_at]).astype(np.float32)
                sources[layer][oracle_at] = SOURCE_NAMES.index("oracle")
            if len(mapped_at):
                for kind in KINDS:
                    values[kind, layer][mapped_at] = mapped[kind, layer]
                sources[layer][mapped_at] = SOURCE_NAMES.index("mapped")
            for fill in ("table", "mean"):
                mask = policy == fill
                if fill == fallback_policy:
                    mask |= fallback[layer]
                at = np.flatnonzero(mask)
                if not len(at):
                    continue
                if fill == "mean" and mapper is None:
                    raise ValueError("unreachable sliding cells without oracle values need a table or a mapper")
                for kind in KINDS:
                    label = np.empty(len(at), dtype=np.int8)
                    values[kind, layer][at] = fill_rows(kind, layer, positions[at], fill, label)
                    sources[layer][at] = label
            computed[layer] = sources[layer] != SOURCE_NAMES.index("oracle")
            if (sources[layer] < 0).any():
                raise AssertionError("a cell was left without a source")
        sliding = [layer for layer in layers if muse_spec.layer_types[layer] == "sliding"]
        if sliding:
            # one rotation call for every sliding layer: RoPE NORM is elementwise per head
            at = [np.flatnonzero(computed[layer]) for layer in sliding]
            if all(np.array_equal(at[0], other) for other in at[1:]) and len(at[0]):
                stacked = np.concatenate([values["k", layer][at[0]] for layer in sliding], axis=1)
                rotated = kvstate.cache_keys(muse_spec, sliding[0], stacked, positions[at[0]])
                for layer, part in zip(sliding, np.split(rotated, len(sliding), axis=1)):
                    values["k", layer][at[0]] = part
            else:
                for layer, rows in zip(sliding, at):
                    if len(rows):
                        values["k", layer][rows] = kvstate.cache_keys(muse_spec, layer, values["k", layer][rows],
                                                                      positions[rows])
        built = kvstate.build_kv_stream(muse_spec, layers, positions,
                                        {layer: values["k", layer] for layer in layers},
                                        {layer: values["v", layer] for layer in layers},
                                        dtype=dtype, v_trans=v_trans, seq_id=seq_id)
        streams = [None] * n_stream
        streams[stream] = built
        counts = {"cells": n, "content": int(is_content.sum()), "fill": int(n - is_content.sum()),
                  "by_class": {CLASS_NAMES[value]: int(np.sum(target_class[positions] == value))
                               for value in CLASS_NAMES},
                  "sources": {SOURCE_NAMES[code]: int(sum(np.sum(sources[layer] == code) for layer in layers))
                              for code in range(len(SOURCE_NAMES))},
                  "oracle_fallback": int(sum(fallback[layer].sum() for layer in layers)),
                  "pos_min": int(positions.min()), "pos_max": int(positions.max())}
        return kvstate.KVCache(name, layers, streams), counts, {"positions": positions, "content": is_content,
                                                                 "source": sources, "fallback": fallback,
                                                                 "target_class": target_class[positions]}

    if not 0 <= stream < n_stream:
        raise ValueError("stream must lie in [0, n_stream)")
    base_positions = np.arange(t0, dtype=np.int64)
    swa_positions = np.arange(max(0, t0 - swa_cells), t0, dtype=np.int64)
    base, base_counts, base_cells = section("base", muse_spec.layers_of("full"), base_positions)
    swa, swa_counts, swa_cells_info = section("swa", muse_spec.layers_of("sliding"), swa_positions)
    state = kvstate.StateFile(kvstate.token_words(muse_tokens["token_id"][:t0], server_packed=server_tokens),
                              kvstate.SequenceState(muse_spec, kvstate.FLAGS_NONE, [base, swa]))
    state.check_positions()
    if mapper is not None and hasattr(mapper, "clear"):
        mapper.clear()
    report = {"document_id": muse_record["document_id"], "t0": t0, "swa_cells": swa_cells,
              "n_swa": muse_spec.n_swa, "muse_tokens": int(len(muse_tokens["token_id"])),
              "qwen_tokens": int(len(qwen_tokens["token_id"])),
              "qwen_state_tokens": None if qwen is None else qwen.n_tokens,
              "oracle_state_tokens": None if oracle is None else oracle.n_tokens,
              "content_policy": content_policy, "fill_policy": fill_policy, "dtype": dtype, "v_trans": bool(v_trans),
              "server_packed_tokens": bool(server_tokens), "renorm_table_k": bool(renorm_table_k),
              "aligned_beyond_qwen_state": beyond, "native_prefix": native_prefix,
              "native_cells": int(native[:t0].sum()),
              "floor": MEAN_FLOORS[fill_policy] if content_policy == "mean" else None,
              "sections": {"base": base_counts, "swa": swa_counts},
              "seconds": round(time.perf_counter() - started, 3)}
    return Handoff(state, report, {"base": base_cells, "swa": swa_cells_info})


def build_muse_state(qwen_state, qwen_record, muse_record, muse_tokens, mapper, *, qwen_tokens, t0, swa_cells=None,
                     content_policy="mapped", fill_policy="table", oracle_state=None, table=None, dtype="f16",
                     v_trans=False, **options):
    """kvstate.StateFile for Muse (see build_handoff; this returns only the state)."""
    return build_handoff(qwen_state, qwen_record, qwen_tokens, muse_record, muse_tokens, mapper, t0=t0,
                         swa_cells=swa_cells, content_policy=content_policy, fill_policy=fill_policy,
                         oracle_state=oracle_state, table=table, dtype=dtype, v_trans=v_trans, **options).state


# -- helpers for tests and offline checks ----------------------------------------------------

def qwen_state_from_rows(spec, tokens, k, v, *, dtype="f32", v_trans=False, server_tokens=False):
    """A Qwen (hybrid) StateFile from pre-RoPE K / V rows at positions 0..n-1 of every attention layer.

    The recurrent section is zero-filled with the spec's row sizes (the handoff never reads
    it); use only for offline checks, never to seed a real Qwen context.
    """
    if spec.architecture != "qwen35" or spec.recurrent_row_bytes is None:
        raise ValueError("need a qwen35 spec with recurrent row sizes")
    tokens = np.asarray(tokens)
    positions = np.arange(len(tokens), dtype=np.int64)
    layers = spec.layers_of("full")
    cached = {layer: kvstate.cache_keys(spec, layer, np.asarray(k[layer], dtype=np.float32), positions)
              for layer in layers}
    attn = kvstate.build_kv_stream(spec, layers, positions, cached, {layer: v[layer] for layer in layers},
                                   dtype=dtype, v_trans=v_trans, tokens_at=tokens)
    recurrent_layers = spec.layers_of("recurrent")
    r_bytes, s_bytes = spec.recurrent_row_bytes
    recurrent = kvstate.RecurrentState(
        "recurrent", recurrent_layers, np.asarray([len(tokens) - 1], dtype=np.int32), ((),), len(spec.layer_types),
        {layer: kvstate.OpaqueRows(kvstate.GGML_TYPE_F32, r_bytes, bytes(r_bytes)) for layer in recurrent_layers},
        {layer: kvstate.OpaqueRows(kvstate.GGML_TYPE_F32, s_bytes, bytes(s_bytes)) for layer in recurrent_layers})
    state = kvstate.SequenceState(spec, kvstate.FLAGS_NONE,
                                  [kvstate.KVCache("attn", layers, [attn]), recurrent])
    return kvstate.StateFile(kvstate.token_words(tokens, server_packed=server_tokens), state)


# -- command line ----------------------------------------------------------------------

def _record(path, document_id):
    for record in align_module.read_records(path):
        if record.get("document_id") == document_id:
            return record
    raise ValueError(f"{path} has no record {document_id}")


def _tokens_from_capture(path, document_id):
    from captures import Capture
    capture = Capture(path)
    if document_id not in capture.document_index:
        raise ValueError(f"{path} does not hold document {document_id}")
    rows = capture.rows(document_id)
    return {name: np.array(getattr(capture, name)[rows]) for name in ("token_id", "byte_end", "special")}


def _tokens(args, role, model, record):
    given = getattr(args, f"{role}_tokens")
    if given is not None:
        return _tokens_from_capture(given, record["document_id"])
    if args.probe is None:
        raise ValueError(f"--{role}-tokens or --probe is required")
    import collect_v2
    ctx = max(len(record["text"].encode("utf-8")) + 16, 5120)
    return collect_v2.tokenize(args.probe, model, [record], ctx=ctx)[record["document_id"]]


def _json(value):
    return json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n"


def command_build(args):
    output = args.output
    report_path = args.report or output.with_name(output.name + ".report.json")
    for path in (output, report_path, output.with_name(output.name + ".resume")):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"output already exists: {path}")
    qwen_spec, muse_spec = kvstate.ModelSpec.from_gguf(args.qwen_model), kvstate.ModelSpec.from_gguf(args.muse_model)
    qwen_record = _record(args.qwen_records, args.document_id)
    muse_record = _record(args.muse_records, args.document_id)
    qwen_tokens = _tokens(args, "qwen", args.qwen_model, qwen_record)
    muse_tokens = _tokens(args, "muse", args.muse_model, muse_record)
    handoff_at = handoff_position(muse_record, muse_tokens)
    t0 = args.t0 if args.t0 is not None else handoff_at - args.tail
    # read strictly with the specs the build goes on to use (kvstate.parse_state: another revision's
    # reading would lay the rows out otherwise; Qwen3.5 and Muse states are the same on both pins)
    qwen_state = kvstate.read_file(args.qwen_state, qwen_spec, strict=True) if args.qwen_state else None
    oracle_state = kvstate.read_file(args.oracle_state, muse_spec, strict=True) if args.oracle_state else None
    result = build_handoff(qwen_state, qwen_record, qwen_tokens, muse_record, muse_tokens, args.mapper, t0=t0,
                           muse_spec=muse_spec, swa_cells=args.swa_cells, content_policy=args.content,
                           fill_policy=args.fill, oracle_state=oracle_state, table=args.table,
                           dtype=args.cache_type, v_trans=args.flash_attn == "off", server_tokens=args.server_tokens,
                           renorm_table_k=not args.no_renorm_table_k, chunk_rows=args.chunk_rows,
                           native_prefix=args.native_prefix)
    data = result.state.to_bytes()
    kvstate.parse_file(data, muse_spec, strict=True)
    write_new(output, data)
    if args.resume_sidecar:
        write_new(output.with_name(output.name + ".resume"), kvstate.ResumeFile().to_bytes())
    inputs = {"qwen_state": args.qwen_state, "oracle_state": args.oracle_state, "table": args.table,
              "mapper_manifest": None if args.mapper is None else args.mapper / "manifest.json",
              "qwen_records": args.qwen_records, "muse_records": args.muse_records}
    report = dict(result.report, format_version=1, purpose="kv_handoff_state", output=str(output.resolve()),
                  bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), handoff_position=handoff_at,
                  tail=handoff_at - t0, resume_sidecar=bool(args.resume_sidecar),
                  inputs={name: None if path is None else {"path": str(Path(path).resolve()),
                                                           "sha256": sha256_file(path)}
                          for name, path in inputs.items()},
                  models={"qwen": str(Path(args.qwen_model).resolve()), "muse": str(Path(args.muse_model).resolve())})
    write_new(report_path, _json(report))
    print(json.dumps({"output": str(output), "report": str(report_path), "bytes": len(data),
                      "t0": t0, "sections": report["sections"]}, allow_nan=False))
    return 0


def command_token_table(args):
    from captures import CaptureSet
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    target_set = CaptureSet(args.target_capture)
    alignment = align_module.load_alignment(args.alignment)

    def progress(event):
        print(json.dumps(event), file=sys.stderr, flush=True)

    started = time.perf_counter()
    table = build_token_table(target_set, alignment, split=args.split, chunk_rows=args.chunk_rows, progress=progress)
    provenance = {"alignment": {"path": str(args.alignment.resolve()), "sha256": sha256_file(args.alignment)},
                  "target_captures": [{"path": str(Path(path).resolve()),
                                       "report_sha256": sha256_file(Path(path) / "report.json"),
                                       "model_sha256": (capture.manifest or {}).get("model_sha256")}
                                      for path, capture in zip(args.target_capture, target_set.captures)],
                  "code_sha256": {name: sha256_file(Path(__file__).resolve().parent / name)
                                  for name in ("handoff.py", "align.py", "captures.py")},
                  "seconds": round(time.perf_counter() - started, 3)}
    write_token_table(args.output, table, provenance)
    print(json.dumps({"output": str(args.output), "rows": table["metadata"]["rows"],
                      "unique_tokens": table["metadata"]["unique_tokens"],
                      "layers": len(table["metadata"]["layers"])}))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="write a Muse slot file from a Qwen slot file")
    build.add_argument("--qwen-state", type=Path, help="Qwen llama_state_seq_save_file (slot file)")
    build.add_argument("--qwen-model", type=Path, required=True, help="Qwen GGUF (metadata only)")
    build.add_argument("--muse-model", type=Path, required=True, help="Muse GGUF (metadata only)")
    build.add_argument("--qwen-records", type=Path, required=True, help="Qwen rendered JSONL")
    build.add_argument("--muse-records", type=Path, required=True, help="Muse rendered JSONL")
    build.add_argument("--document-id", required=True)
    build.add_argument("--qwen-tokens", type=Path, help="kv_probe tokenize/collect directory holding the document")
    build.add_argument("--muse-tokens", type=Path, help="kv_probe tokenize/collect directory holding the document")
    build.add_argument("--probe", type=Path, help="kvpark-kv-probe for vocab-only tokenization")
    where = build.add_mutually_exclusive_group(required=True)
    where.add_argument("--t0", type=int, help="Muse token count of the built state")
    where.add_argument("--tail", type=int, help="t0 = P - TAIL, P = first Muse token of the final message")
    build.add_argument("--swa-cells", type=int, help="sliding-layer cells before t0 (default n_swa)")
    build.add_argument("--content", choices=CONTENT_POLICIES, default="mapped")
    build.add_argument("--fill", choices=FILL_POLICIES, default="table")
    build.add_argument("--mapper", type=Path, help="fit_layers mapper directory")
    build.add_argument("--oracle-state", type=Path,
                       help="native Muse state with >= t0 tokens (>= the native prefix without oracle policies)")
    build.add_argument("--native-prefix", type=int, default=0,
                       help="positions [0, N) take the oracle state's native values whatever their class")
    build.add_argument("--table", type=Path, help="token table NPZ (handoff.py token-table)")
    build.add_argument("--cache-type", choices=("f16", "f32"), default="f16")
    build.add_argument("--flash-attn", choices=("on", "off"), default="on", help="off writes transposed V")
    build.add_argument("--server-tokens", action="store_true", help="server_tokens::serialize() token words")
    build.add_argument("--resume-sidecar", action="store_true", help="also write an empty <output>.resume")
    build.add_argument("--no-renorm-table-k", action="store_true", help="keep table K rows unnormalized")
    build.add_argument("--chunk-rows", type=int, default=2048)
    build.add_argument("--output", type=Path, required=True, help="new slot file; never overwritten")
    build.add_argument("--report", type=Path, help="report JSON (default <output>.report.json)")
    table = commands.add_parser("token-table", help="per-token-id mean Muse K/V over non-aligned train rows")
    table.add_argument("--target-capture", type=Path, action="append", required=True,
                       help="target capture directories in the alignment's order (repeatable)")
    table.add_argument("--alignment", type=Path, required=True)
    table.add_argument("--split", default="train")
    table.add_argument("--chunk-rows", type=int, default=16384)
    table.add_argument("--output", type=Path, required=True, help="new NPZ; never overwritten")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "build":
        if args.tail is not None and args.tail < 0:
            raise ValueError("--tail must be nonnegative")
        if args.chunk_rows < 1:
            raise ValueError("--chunk-rows must be positive")
        return command_build(args)
    return command_token_table(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, kvstate.StateFormatError) as exc:
        raise SystemExit(str(exc)) from exc
