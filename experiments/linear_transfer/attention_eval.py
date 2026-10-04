"""Offline attention-output check of mapped Muse K/V against Muse's own attention.

For each evaluated Muse layer and dev/test document, the captured pre-RoPE K/V of
aligned content tokens in the prefix (tokens before the final message) are replaced
by mapper predictions from the Qwen capture, and causal GQA attention is recomputed
in numpy with the runtime's conventions (RoPE NORM with ggml's float32 angle
recurrence and the sliding window on SWA layers, NoPE on full layers). Native and
mapped SDPA outputs are compared per query head, and after the gate, W_o and the
post-attention RMSNorm. Layers with a captured attn_out validate the numpy attention.
"""

import os

if __name__ == "__main__":
    for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, "8")

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from align import classify_tokens, load_alignment, read_records, segment_table
from captures import CaptureSet
from collect import sha256_file
import fit_layers


ARCHITECTURE = "muse-glimmer"
POST_NORM_EPS = 1e-8
VALIDATION_TOLERANCE = 1e-4
VARIANTS = ("kv", "k", "v", "mean_kv")
QUERY_SETS = ("all", "continuation", "affected", "scored")
QUERY_SET_NOTES = {
    "all": "every token of the document",
    "continuation": "tokens inside the final message's segments",
    "affected": "continuation queries that can attend to at least one patched key on this layer",
    "scored": "queries p - 1 for the first --max-scored continuation tokens p >= 1: the positions whose "
              "logits inject.py scores",
}
SPLIT_ORDER = ("test", "dev", "train")
MAPPER_KINDS = ("k", "v")


# -- RoPE and attention ------------------------------------------------------

def rope_angles(positions, n_rot, base, freq_scale=1.0):
    """float32 angles [n, n_rot/2] built like ggml_rope_cache_init (no YaRN, no freq factors)."""
    if n_rot <= 0 or n_rot % 2:
        raise ValueError("n_rot must be a positive even number")
    theta_scale = np.power(np.float32(base), np.float32(-2.0) / np.float32(n_rot), dtype=np.float32)
    theta = np.asarray(positions).astype(np.float32)
    angles = np.empty((len(theta), n_rot // 2), dtype=np.float32)
    for index in range(n_rot // 2):
        angles[:, index] = theta
        theta = (theta * theta_scale).astype(np.float32)
    return (angles * np.float32(freq_scale)).astype(np.float32)


def rope_norm(x, positions, *, n_rot, base, freq_scale=1.0):
    """GGML_ROPE_TYPE_NORMAL: rotate adjacent pairs (2i, 2i+1) of the first n_rot dims.

    x is [n, heads, dim]; theta_i = pos * base^(-2i/n_rot). Returns float32.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or len(positions) != len(x) or n_rot > x.shape[2]:
        raise ValueError("rope_norm needs x [n, heads, dim] with n positions and n_rot <= dim")
    angles = rope_angles(positions, n_rot, base, freq_scale)
    cos = np.cos(angles.astype(np.float64)).astype(np.float32)[:, None, :]
    sin = np.sin(angles.astype(np.float64)).astype(np.float32)[:, None, :]
    out = x.copy()
    x0, x1 = x[:, :, 0:n_rot:2], x[:, :, 1:n_rot:2]
    out[:, :, 0:n_rot:2] = x0 * cos - x1 * sin
    out[:, :, 1:n_rot:2] = x0 * sin + x1 * cos
    return out


def causal_attention(q, k, v, *, scale, window=None, query_index=None, chunk=128, dtype=np.float32):
    """Causal GQA SDPA for one document with positions 0..n-1.

    q [n, Hq, D], k/v [n, Hkv, D]; q head h reads kv head h // (Hq / Hkv). With a
    window, key j is masked for query i iff i - j >= window. Returns [len(query_index), Hq, D].
    """
    n, heads, dim = q.shape
    kv_heads = k.shape[1]
    if k.shape != (n, kv_heads, dim) or v.shape != (n, kv_heads, dim) or heads % kv_heads:
        raise ValueError("q, k and v shapes are inconsistent")
    if window is not None and window <= 0:
        raise ValueError("window must be positive")
    group = heads // kv_heads
    queries = np.arange(n) if query_index is None else np.asarray(query_index, dtype=np.int64)
    if queries.ndim != 1 or (len(queries) and (queries.min() < 0 or queries.max() >= n)):
        raise ValueError("query index outside the document")
    output = np.empty((len(queries), heads, dim), dtype=np.float32)
    keys = np.ascontiguousarray(k.astype(dtype).transpose(1, 2, 0))
    values = np.ascontiguousarray(v.astype(dtype).transpose(1, 0, 2))
    for start in range(0, len(queries), chunk):
        chosen = queries[start:start + chunk]
        count = len(chosen)
        low = 0 if window is None else max(0, int(chosen.min()) - window + 1)
        high = int(chosen.max()) + 1
        block = q[chosen].astype(dtype).reshape(count, kv_heads, group, dim)
        block = block.transpose(1, 2, 0, 3).reshape(kv_heads, group * count, dim)
        scores = np.matmul(block, keys[:, :, low:high]) * dtype(scale)
        scores = scores.reshape(kv_heads, group, count, high - low)
        distance = chosen[:, None] - np.arange(low, high)[None, :]
        allowed = distance >= 0
        if window is not None:
            allowed &= distance < window
        scores = np.where(allowed[None, None], scores, -np.inf)
        scores -= scores.max(axis=-1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(axis=-1, keepdims=True)
        mixed = np.matmul(scores.reshape(kv_heads, group * count, high - low), values[:, low:high])
        output[start:start + count] = mixed.reshape(kv_heads, group, count, dim).transpose(
            2, 0, 1, 3).reshape(count, heads, dim)
    return output


def attention_contribution(sdpa, gate, w_o, post_norm, eps=POST_NORM_EPS, chunk=1024):
    """RMSNorm(W_o (sdpa * gate)) * post_norm per token: the residual update of the layer."""
    n = len(sdpa)
    width = sdpa.shape[1] * sdpa.shape[2]
    if gate.shape != sdpa.shape or w_o.shape[1] != width or post_norm.shape != (w_o.shape[0],):
        raise ValueError("gate, W_o or post-attention norm shapes do not match the attention output")
    output = np.empty((n, w_o.shape[0]), dtype=np.float32)
    for start in range(0, n, chunk):
        gated = (sdpa[start:start + chunk] * gate[start:start + chunk]).reshape(-1, width)
        projected = gated.astype(np.float32) @ w_o.T
        mean_square = np.mean(projected.astype(np.float64) ** 2, axis=1, keepdims=True)
        output[start:start + chunk] = projected / np.sqrt(mean_square + eps) * post_norm
    return output


def row_cosine(a, b):
    """Cosine along the last axis in float64; rows where either vector is zero count as 0."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.where(denominator > 0, dot / np.where(denominator > 0, denominator, 1), 0.0)


# -- model parameters ----------------------------------------------------------

@dataclass
class ModelParams:
    n_head: int
    n_head_kv: int
    head_dim: int
    n_rot: int
    rope_base: float
    window: int
    freq_scale: float = 1.0

    @property
    def kq_scale(self):
        return 1.0 / math.sqrt(self.head_dim)


def _scalar(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("expected a scalar GGUF metadata value")
        value = value[0]
    return value


def load_model_params(model):
    """Muse attention hyperparameters from GGUF metadata (sliding-layer RoPE)."""
    import gguf_tensors

    def value(key, default=None):
        found = gguf_tensors.metadata_value(model, f"{ARCHITECTURE}.{key}")
        return default if found is None else _scalar(found)

    architecture = gguf_tensors.metadata_value(model, "general.architecture")
    if architecture != ARCHITECTURE:
        raise ValueError(f"model architecture {architecture!r} is not {ARCHITECTURE}")
    n_head, n_head_kv = int(value("attention.head_count")), int(value("attention.head_count_kv"))
    head_dim = value("attention.key_length")
    head_dim = int(head_dim) if head_dim is not None else int(value("embedding_length")) // n_head
    if value("attention.value_length", head_dim) != head_dim:
        raise ValueError("key and value head sizes differ")
    n_rot = int(value("rope.dimension_count_swa", value("rope.dimension_count", head_dim)))
    base = float(value("rope.freq_base_swa", value("rope.freq_base", 10000.0)))
    window = value("attention.sliding_window")
    if window is None:
        raise ValueError("model has no sliding window")
    scaling = value("rope.scaling.type")
    factor = value("rope.scaling.factor", value("rope.scale_linear", 0.0))
    if scaling not in (None, "none", "linear") or float(factor) not in (0.0, 1.0):
        raise ValueError(f"RoPE scaling {scaling!r} with factor {factor} is not supported")
    return ModelParams(n_head=n_head, n_head_kv=n_head_kv, head_dim=head_dim, n_rot=n_rot,
                       rope_base=base, window=int(window))


def load_output_weights(model, layer):
    """(W_o as numpy [n_embd, n_head*head_dim], post-attention norm weight [n_embd]) float32."""
    import gguf_tensors
    w_o = gguf_tensors.read_tensor(model, f"blk.{layer}.attn_output.weight")
    norm = gguf_tensors.read_tensor(model, f"blk.{layer}.post_attention_norm.weight")
    if w_o.ndim != 2 or norm.shape != (w_o.shape[0],):
        raise ValueError(f"unexpected output tensor shapes for layer {layer}")
    return w_o, norm


# -- mapper ------------------------------------------------------------------

MAPPER_PURPOSE = "kv_linear_transfer_mapper"
MAPPER_FILES = ("weight", "bias", "mean")


def _head_shape(value, name):
    if (not isinstance(value, list) or len(value) != 2
            or any(type(item) is not int or item <= 0 for item in value)):
        raise ValueError(f"mapper manifest {name} head shape is malformed: {value!r}")
    return tuple(value)


class Mapper:
    """Lazy reader of a fit_layers.py mapper directory.

    The manifest schema, hash verification and prediction (fit_layers.LayerMap and
    fit_layers.apply: source blocks of source_layers concatenated in order @ W + b,
    per-head RMS renormalization when k_renorm) are fit_layers' own, so consumers
    compute exactly what fit_layers evaluated. Source rows of a K map are source K
    rows, likewise for V. Maps load on first use; clear() drops them. A file's hash is
    checked on its first load; later loads in the same process skip the hash while the
    file's identity (device, inode, size, mtime) is unchanged.
    """

    def __init__(self, path):
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"mapper manifest is missing: {self.path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict) or manifest.get("format_version") != 1
                or manifest.get("purpose") != MAPPER_PURPOSE):
            raise ValueError(f"{manifest_path} is not a fit_layers mapper manifest (format_version 1)")
        self.manifest = manifest
        try:
            self.source_shape = {kind: _head_shape(manifest["source"]["head_shape"][kind], "source")
                                 for kind in MAPPER_KINDS}
            self.target_shape = {kind: _head_shape(manifest["target"]["head_shape"][kind], "target")
                                 for kind in MAPPER_KINDS}
        except (KeyError, TypeError):
            raise ValueError("mapper manifest lacks source/target head_shape for k and v") from None
        self.source_architecture = manifest["source"].get("architecture")
        self.target_architecture = manifest["target"].get("architecture")
        entries = manifest.get("maps")
        if not isinstance(entries, list) or not entries:
            raise ValueError("mapper manifest needs a nonempty 'maps' list")
        self.entries, self._maps, self._verified = {}, {}, {}
        for entry in entries:
            if (not isinstance(entry, dict) or entry.get("kind") not in MAPPER_KINDS
                    or type(entry.get("target")) is not int or entry["target"] < 0
                    or not isinstance(entry.get("source_layers"), list) or not entry["source_layers"]
                    or any(type(layer) is not int or layer < 0 for layer in entry["source_layers"])
                    or len(set(entry["source_layers"])) != len(entry["source_layers"])
                    or type(entry.get("k_renorm")) is not bool
                    or (entry["k_renorm"] and entry["kind"] != "k")
                    or not isinstance(entry.get("files"), dict) or not isinstance(entry.get("sha256"), dict)
                    or any(not isinstance(entry["files"].get(name), str)
                           or Path(entry["files"][name]).name != entry["files"][name]
                           or not isinstance(entry["sha256"].get(name), str) for name in MAPPER_FILES)):
                raise ValueError(f"invalid mapper entry: {entry!r}")
            pair = entry["kind"], entry["target"]
            if pair in self.entries:
                raise ValueError(f"duplicate mapper for {pair}")
            self.entries[pair] = entry

    def has(self, kind, layer):
        return (kind, layer) in self.entries

    def entry(self, kind, layer):
        try:
            return self.entries[kind, layer]
        except KeyError:
            raise KeyError(f"mapper has no {kind}-{layer}") from None

    def layer_map(self, kind, layer):
        """fit_layers.LayerMap for (kind, layer), loaded and hash-verified on first use."""
        if (kind, layer) not in self._maps:
            entry = self.entry(kind, layer)
            arrays = {}
            for name in MAPPER_FILES:
                path = self.path / entry["files"][name]
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"mapper file is missing: {path.name}")
                data = path.read_bytes()
                status = path.stat()
                identity = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
                if self._verified.get(path) != identity or len(data) != status.st_size:
                    if hashlib.sha256(data).hexdigest() != entry["sha256"][name]:
                        raise ValueError(f"mapper file {path.name} does not match its manifest hash")
                    self._verified[path] = identity
                arrays[name] = np.load(io.BytesIO(data), allow_pickle=False)
                if arrays[name].dtype != np.float32:
                    raise ValueError(f"mapper file {path.name} is {arrays[name].dtype}, not float32")
            layer_map = fit_layers.LayerMap(kind, layer, entry["source_layers"], arrays["weight"], arrays["bias"],
                                            entry["k_renorm"], self.source_shape[kind], self.target_shape[kind],
                                            arrays["mean"])
            if (layer_map.train_mean.shape != (int(np.prod(self.target_shape[kind])),)
                    or not np.isfinite(layer_map.train_mean).all()):
                raise ValueError(f"mapper train mean {kind}-{layer} is malformed")
            self._maps[kind, layer] = layer_map
        return self._maps[kind, layer]

    def clear(self):
        self._maps.clear()

    def _check_shape(self, kind, target_shape):
        if tuple(target_shape) != self.target_shape[kind]:
            raise ValueError(f"mapper {kind} head shape {self.target_shape[kind]} differs from {tuple(target_shape)}")

    def source_layers(self, kind, layer):
        """(source capture key, source layers in feature order) for the map."""
        return kind, list(self.entry(kind, layer)["source_layers"])

    def predict(self, kind, layer, source_rows, target_shape):
        """source_rows: {source_layer: [n, heads, dim]} -> float32 [n, *target_shape]."""
        self._check_shape(kind, target_shape)
        return fit_layers.apply(self.layer_map(kind, layer), source_rows)

    def mean(self, kind, layer, target_shape):
        """Train target mean of the map as [*target_shape] float32."""
        self._check_shape(kind, target_shape)
        return self.layer_map(kind, layer).train_mean.reshape(target_shape)

    def floor(self, kind, layer, target_shape):
        """Information-free substitute row: the train target mean, with K renormalized per
        head to RMS 1 like every native Muse K row (k_norm), so floor keys are not short."""
        mean = self.mean(kind, layer, target_shape)
        if kind == "k":
            mean = fit_layers.renormalize_heads(mean.reshape(1, -1), *target_shape).reshape(target_shape)
        return mean


# -- document plans ----------------------------------------------------------

@dataclass
class DocumentPlan:
    document_id: str
    split: str
    n_tokens: int
    tgt_start: int
    src_start: int
    prefix_tokens: int
    positions: np.ndarray       # aligned content target positions inside the prefix, ascending
    src_positions: np.ndarray   # matching source positions
    segments: np.ndarray        # segment index of each patched position
    system: np.ndarray          # patched position lies in a message-0 (system) segment
    continuation: np.ndarray    # target positions inside the final message's segments
    score_ranges: list          # [[start_byte, end_byte]] of the final message's segments

    def summary(self):
        return {"document_id": self.document_id, "split": self.split, "n_tokens": self.n_tokens,
                "prefix_tokens": self.prefix_tokens, "patched_positions": int(len(self.positions)),
                "system_positions": int(self.system.sum()),
                "continuation_tokens": int(len(self.continuation))}


def _records_by_id(records):
    result = {}
    for record in records:
        if record["document_id"] in result:
            raise ValueError(f"duplicate rendered record {record['document_id']}")
        result[record["document_id"]] = record
    return result


def _document_tokens(capture_set, document_id, record, name):
    rows = capture_set.rows(document_id)
    ends = capture_set.byte_end[rows]
    if int(ends[-1]) != len(record["text"].encode("utf-8")):
        raise ValueError(f"{name} capture tokens do not cover the rendered text of {document_id}")
    return rows.start, capture_set.byte_start[rows], ends


def _check_pairs(name, starts_bytes, ends_bytes, table, positions, segments, rel_end, document_id):
    ids, seg_starts, seg_ends = table
    lookup = {int(index): (int(start), int(end)) for index, start, end in zip(ids, seg_starts, seg_ends)}
    if len(positions) and (positions.min() < 0 or positions.max() >= len(ends_bytes)):
        raise ValueError(f"alignment {name} positions exceed the capture of {document_id}")
    for position, segment, relative in zip(positions, segments, rel_end):
        if int(segment) not in lookup:
            raise ValueError(f"alignment segment {segment} is not rendered in {name} {document_id}")
        start, end = lookup[int(segment)]
        if (int(ends_bytes[position]) - start != int(relative) or starts_bytes[position] < start
                or ends_bytes[position] > end):
            raise ValueError(f"alignment does not match the {name} capture of {document_id}; "
                             "was it built from different captures?")


def plan_documents(alignment, tgt_records, tgt_set, src_records, src_set, splits):
    """Document plans for aligned documents in the given splits; returns (plans, skipped)."""
    tgt_records, src_records = _records_by_id(tgt_records), _records_by_id(src_records)
    plans, skipped = [], []
    for number, document_id in enumerate(alignment.document_ids):
        split = alignment.splits[number]
        if split not in splits:
            continue
        for name, records, capture_set in (("target", tgt_records, tgt_set), ("source", src_records, src_set)):
            if document_id not in records:
                raise ValueError(f"{name} rendered records lack document {document_id}")
            if document_id not in capture_set:
                raise ValueError(f"{name} capture set lacks document {document_id}")
        tgt_record, src_record = tgt_records[document_id], src_records[document_id]
        if tgt_record.get("split") != split:
            raise ValueError(f"split of {document_id} differs between alignment and records")
        tgt_start, tgt_byte_start, tgt_byte_end = _document_tokens(tgt_set, document_id, tgt_record, "target")
        src_start, src_byte_start, src_byte_end = _document_tokens(src_set, document_id, src_record, "source")
        selected = np.flatnonzero(alignment.doc == number)
        tgt_pos, src_pos = alignment.tgt_position[selected], alignment.src_position[selected]
        segment, rel_end = alignment.segment[selected], alignment.rel_end[selected]
        tgt_table, src_table = segment_table(tgt_record), segment_table(src_record)
        _check_pairs("target", tgt_byte_start, tgt_byte_end, tgt_table, tgt_pos, segment, rel_end, document_id)
        _check_pairs("source", src_byte_start, src_byte_end, src_table, src_pos, segment, rel_end, document_id)
        final = set(tgt_record.get("final_message_segments") or [])
        ids, seg_starts, seg_ends = tgt_table
        rendered_final = [(int(s), int(e)) for j, s, e in zip(ids, seg_starts, seg_ends) if int(j) in final]
        if not rendered_final:
            skipped.append({"document_id": document_id, "reason": "final message has no rendered segment"})
            continue
        final_start = min(start for start, _ in rendered_final)
        prefix_tokens = int(np.searchsorted(tgt_byte_end, final_start, side="right"))
        in_prefix = tgt_pos < prefix_tokens
        if not in_prefix.any():
            skipped.append({"document_id": document_id, "reason": "no aligned content tokens in the prefix"})
            continue
        order = np.argsort(tgt_pos[in_prefix], kind="stable")
        positions = tgt_pos[in_prefix][order].astype(np.int64)
        if np.any(np.diff(positions) <= 0):
            raise ValueError(f"alignment maps two pairs to one target position in {document_id}")
        segments = segment[in_prefix][order].astype(np.int64)
        keys = tgt_record["segment_keys"]
        system = np.asarray([keys[int(j)].startswith("0:") for j in segments], dtype=bool)
        token_segment, inside, _ = classify_tokens(tgt_byte_start, tgt_byte_end, tgt_table)
        continuation = np.flatnonzero(inside & np.isin(token_segment, sorted(final))).astype(np.int64)
        plans.append(DocumentPlan(
            document_id=document_id, split=split, n_tokens=len(tgt_byte_end), tgt_start=int(tgt_start),
            src_start=int(src_start), prefix_tokens=prefix_tokens, positions=positions,
            src_positions=src_pos[in_prefix][order].astype(np.int64), segments=segments, system=system,
            continuation=continuation, score_ranges=[[start, end] for start, end in sorted(rendered_final)]))
    return plans, skipped


def mapped_rows(mapper, kind, layer, plan, src_set, target_shape, _source_cache=None):
    """Mapper predictions [P, *target_shape] for the plan's patched positions."""
    source_key, source_layers = mapper.source_layers(kind, layer)
    cache = {} if _source_cache is None else _source_cache
    rows = plan.src_start + plan.src_positions
    source_rows = {}
    for source_layer in source_layers:
        if (source_key, source_layer) not in cache:
            cache[source_key, source_layer] = src_set.gather(source_key, source_layer, rows)
        source_rows[source_layer] = cache[source_key, source_layer]
    return mapper.predict(kind, layer, source_rows, target_shape)


# -- evaluation --------------------------------------------------------------

class Accumulator:
    FIELDS = ("queries", "head_pairs", "sdpa_cos_sum", "contribution_rows", "contribution_cos_sum",
              "contribution_rel_sum", "contribution_diff_sq", "contribution_native_sq")

    def __init__(self):
        self.values = dict.fromkeys(self.FIELDS, 0.0)

    def add(self, other):
        for name in self.FIELDS:
            self.values[name] += other.values[name]

    def result(self):
        v = self.values
        result = {"queries": int(v["queries"]),
                  "sdpa_head_cosine_mean": v["sdpa_cos_sum"] / v["head_pairs"] if v["head_pairs"] else None}
        if v["contribution_rows"]:
            result.update(
                contribution_cosine_mean=v["contribution_cos_sum"] / v["contribution_rows"],
                contribution_relative_error_mean=v["contribution_rel_sum"] / v["contribution_rows"],
                contribution_relative_error_pooled=(math.sqrt(v["contribution_diff_sq"] / v["contribution_native_sq"])
                                                    if v["contribution_native_sq"] > 0 else None))
        else:
            result.update(contribution_cosine_mean=None, contribution_relative_error_mean=None,
                          contribution_relative_error_pooled=None)
        return result


def affected_queries(n, positions, window):
    """Queries that can attend to at least one patched key position."""
    queries = np.arange(n)
    if not len(positions):
        return queries[:0]
    last_patched = positions[np.searchsorted(positions, queries, side="right").clip(1) - 1]
    visible = queries >= positions[0]
    if window is not None:
        visible &= queries - last_patched < window
    return queries[visible]


def compare_layer(q, k, v, patched_positions, substitutes, *, params, sliding, query_sets,
                  gate=None, weights=None, attn_out=None, chunk=128):
    """Native vs substituted attention for one document and layer.

    substitutes: {variant: (k_rows or None, v_rows or None)} with rows [P, Hkv, D] replacing
    the pre-RoPE k/v at patched_positions. Contribution metrics need gate and
    weights = (W_o, post_norm). Returns (validation relative error or None,
    {variant: {query_set: Accumulator}}).
    """
    n = len(q)
    positions = np.arange(n)
    window = params.window if sliding else None

    def rotate(x):
        return rope_norm(x, positions, n_rot=params.n_rot, base=params.rope_base,
                         freq_scale=params.freq_scale) if sliding else np.asarray(x, dtype=np.float32)

    q_rot, k_rot, v = rotate(q), rotate(k), np.asarray(v, dtype=np.float32)
    native = causal_attention(q_rot, k_rot, v, scale=params.kq_scale, window=window, chunk=chunk)
    validation = None
    if attn_out is not None:
        reference = np.asarray(attn_out, dtype=np.float64)
        validation = float(np.linalg.norm(native - reference) / max(np.linalg.norm(reference), 1e-30))
    contribution = gate is not None and weights is not None
    if contribution:
        native_values = attention_contribution(native, gate, *weights)
        native_sq = np.sum(native_values.astype(np.float64) ** 2, axis=1)
    touched = affected_queries(n, patched_positions, window)
    results = {}
    for variant, (k_rows, v_rows) in substitutes.items():
        k_new, v_new = k_rot, v
        if k_rows is not None:
            k_new = np.array(k, dtype=np.float32)
            k_new[patched_positions] = k_rows
            k_new = rotate(k_new)
        if v_rows is not None:
            v_new = v.copy()
            v_new[patched_positions] = v_rows
        mapped = native.copy()
        if len(touched):
            mapped[touched] = causal_attention(q_rot, k_new, v_new, scale=params.kq_scale, window=window,
                                               query_index=touched, chunk=chunk)
        changed = np.flatnonzero(np.any(mapped != native, axis=(1, 2)))
        head_cos = np.ones((n, q.shape[1]))
        head_cos[changed] = row_cosine(native[changed], mapped[changed])
        if contribution:
            cos, rel, diff_sq = np.ones(n), np.zeros(n), np.zeros(n)
            if len(changed):
                mapped_values = attention_contribution(mapped[changed], gate[changed], *weights)
                cos[changed] = row_cosine(native_values[changed], mapped_values)
                diff_sq[changed] = np.sum((mapped_values.astype(np.float64) - native_values[changed]) ** 2, axis=1)
                rel[changed] = np.sqrt(diff_sq[changed] / np.where(native_sq[changed] > 0, native_sq[changed], 1))
        per_set = {}
        for name, index in query_sets.items():
            acc = Accumulator()
            values = acc.values
            values["queries"] = len(index)
            values["head_pairs"] = len(index) * q.shape[1]
            values["sdpa_cos_sum"] = float(head_cos[index].sum())
            if contribution:
                values["contribution_rows"] = len(index)
                values["contribution_cos_sum"] = float(cos[index].sum())
                values["contribution_rel_sum"] = float(rel[index].sum())
                values["contribution_diff_sq"] = float(diff_sq[index].sum())
                values["contribution_native_sq"] = float(native_sq[index].sum())
            per_set[name] = acc
        results[variant] = per_set
    return validation, results


def variant_substitutes(variants, mapper, layer, plan, src_set, target_shape):
    substitutes, cache, predicted = {}, {}, {}

    def prediction(kind):
        if kind not in predicted:
            predicted[kind] = mapped_rows(mapper, kind, layer, plan, src_set, target_shape, cache)
        return predicted[kind]

    for variant in variants:
        if variant == "kv":
            substitutes[variant] = prediction("k"), prediction("v")
        elif variant == "k":
            substitutes[variant] = prediction("k"), None
        elif variant == "v":
            substitutes[variant] = None, prediction("v")
        elif variant == "mean_kv":
            count = len(plan.positions)
            substitutes[variant] = tuple(np.broadcast_to(mapper.floor(kind, layer, target_shape),
                                                         (count, *target_shape)).astype(np.float32)
                                         for kind in ("k", "v"))
        else:
            raise ValueError(f"unknown variant {variant}")
    return substitutes


def query_sets(plan, window, max_scored):
    """{name: query positions} for QUERY_SETS (see QUERY_SET_NOTES)."""
    continuation = plan.continuation
    scored = continuation[continuation >= 1][:max_scored] - 1
    affected = np.intersect1d(continuation, affected_queries(plan.n_tokens, plan.positions, window))
    return {"all": np.arange(plan.n_tokens), "continuation": continuation, "affected": affected, "scored": scored}


def evaluate(plans, tgt_set, src_set, mapper, params, layers, layer_types, variants, *,
             load_weights=None, chunk=128, tolerance=VALIDATION_TOLERANCE, log=None, max_scored=256):
    """Run all layers (validation layers first) over all plans; returns report sections.

    Layer metrics are per split only: fit_layers selected (k, alpha) and k_renorm on dev,
    so dev numbers are optimistic and are never pooled with test.
    """
    validation_layers = set(tgt_set.layers("attn_out"))
    ordered = sorted(layers, key=lambda layer: (layer not in validation_layers, layer))
    gate_layers = set(tgt_set.layers("gate"))
    target_shape = (params.n_head_kv, params.head_dim)
    totals, per_document, validation = {}, [], []
    for layer in ordered:
        sliding = layer_types[layer] == "sliding"
        weights = load_weights(layer) if load_weights is not None and layer in gate_layers else None
        errors = []
        for plan in plans:
            rows = np.arange(plan.tgt_start, plan.tgt_start + plan.n_tokens)
            q = tgt_set.gather("q", layer, rows)
            k = tgt_set.gather("k", layer, rows)
            v = tgt_set.gather("v", layer, rows)
            if q.shape[1:] != (params.n_head, params.head_dim) or k.shape[1:] != target_shape:
                raise ValueError(f"capture shapes of layer {layer} differ from the model parameters")
            gate = tgt_set.gather("gate", layer, rows) if weights is not None else None
            attn_out = tgt_set.gather("attn_out", layer, rows) if layer in validation_layers else None
            substitutes = variant_substitutes(variants, mapper, layer, plan, src_set, target_shape)
            error, results = compare_layer(q, k, v, plan.positions, substitutes, params=params, sliding=sliding,
                                           query_sets=query_sets(plan, params.window if sliding else None,
                                                                 max_scored), gate=gate,
                                           weights=weights, attn_out=attn_out, chunk=chunk)
            if error is not None:
                errors.append(error)
                if not error < tolerance:
                    raise ValueError(f"numpy attention does not reproduce captured attn_out on layer {layer} "
                                     f"({layer_types[layer]}) for {plan.document_id}: relative error "
                                     f"{error:.3g} >= {tolerance:g}")
            for variant, per_set in results.items():
                for name, acc in per_set.items():
                    totals.setdefault((plan.split, layer, variant, name), Accumulator()).add(acc)
                    per_document.append({"document_id": plan.document_id, "split": plan.split, "layer": layer,
                                         "variant": variant, "query_set": name, **acc.result()})
            if log is not None:
                log(f"layer {layer} document {plan.document_id[:12]} done")
        if errors:
            validation.append({"layer": layer, "attention": layer_types[layer], "documents": len(errors),
                               "max_relative_error": max(errors), "tolerance": tolerance})
    layers_report = [{"split": split, "layer": layer, "attention": layer_types[layer], "variant": variant,
                      "query_set": name, **acc.result()}
                     for (split, layer, variant, name), acc in sorted(totals.items(), key=lambda item: (
                         SPLIT_ORDER.index(item[0][0]), item[0][1], item[0][2], item[0][3]))]
    return {"validation": validation, "layers": layers_report, "documents_by_layer": per_document}


# -- CLI -----------------------------------------------------------------------

def _write_new_json(path, payload):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".kvpark-attn-", suffix=".json",
                                     delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.link(stream.name, path)
    finally:
        os.unlink(stream.name)


def _capture_model_sha(capture_set):
    found = {(capture.manifest or {}).get("model_sha256") for capture in capture_set.captures}
    return found.pop() if len(found) == 1 else None


def _parse_list(text, cast=str):
    items = [cast(item) for item in text.split(",") if item.strip()]
    if not items or len(set(items)) != len(items):
        raise ValueError(f"expected a nonempty comma-separated list without duplicates: {text!r}")
    return items


def check_mapper(mapper, params, layers, src_set, kinds=MAPPER_KINDS):
    """Fail early unless the mapper maps Qwen to Muse K/V with this head layout on every layer."""
    if mapper.target_architecture not in (None, ARCHITECTURE) or mapper.source_architecture not in (None, "qwen35"):
        raise ValueError(f"mapper maps {mapper.source_architecture} to {mapper.target_architecture}, "
                         f"not qwen35 to {ARCHITECTURE}")
    for kind in kinds:
        if mapper.target_shape[kind] != (params.n_head_kv, params.head_dim):
            raise ValueError(f"mapper {kind} head shape {mapper.target_shape[kind]} differs from the model's "
                             f"{(params.n_head_kv, params.head_dim)}")
        for layer in layers:
            source_key, source_layers = mapper.source_layers(kind, layer)
            missing = sorted(set(source_layers) - set(src_set.layers(source_key)))
            if missing:
                raise ValueError(f"source capture lacks {source_key} layers {missing} for mapper {kind}-{layer}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Muse GGUF (metadata, W_o, post norm)")
    parser.add_argument("--tgt-docs", required=True, type=Path, help="Muse rendered JSONL")
    parser.add_argument("--tgt-capture", required=True, type=Path, action="append",
                        help="Muse capture with k, v, q (and gate, attn_out) on the eval layers")
    parser.add_argument("--src-docs", required=True, type=Path, help="Qwen rendered JSONL")
    parser.add_argument("--src-capture", required=True, type=Path, action="append")
    parser.add_argument("--alignment", required=True, type=Path, help="align.py NPZ for these captures")
    parser.add_argument("--mapper", required=True, type=Path, help="fit_layers.py mapper directory")
    parser.add_argument("--output", required=True, type=Path, help="new report JSON; never overwritten")
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--layers", help="comma-separated Muse layers (default: every layer with q captured)")
    parser.add_argument("--variants", default="kv,k,v,mean_kv",
                        help=f"comma-separated subset of {','.join(VARIANTS)} (mean_kv, the information-free "
                        "floor, needs mean-<kind>-<layer>.npy in the mapper)")
    parser.add_argument("--max-scored", type=int, default=256,
                        help="continuation tokens of the 'scored' query set (as inject.py --max-scored)")
    parser.add_argument("--chunk", type=int, default=128, help="queries per attention block")
    parser.add_argument("--hash-cache", type=Path, help="model SHA-256 cache (collect_v2 format)")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    if not 0 < args.chunk <= 4096 or not 0 < args.max_scored:
        raise ValueError("--chunk must be in 1..4096 and --max-scored positive")
    splits = _parse_list(args.splits)
    if set(splits) - {"train", "dev", "test"}:
        raise ValueError("--splits must name train, dev or test")
    variants = _parse_list(args.variants)
    if set(variants) - set(VARIANTS):
        raise ValueError(f"--variants must be a subset of {VARIANTS}")
    tgt_set, src_set = CaptureSet(args.tgt_capture), CaptureSet(args.src_capture)
    if tgt_set.architecture != ARCHITECTURE or src_set.architecture != "qwen35":
        raise ValueError("target captures must be Muse and source captures Qwen")
    layer_types = tgt_set.captures[0].layer_types
    q_layers = tgt_set.layers("q")
    layers = _parse_list(args.layers, int) if args.layers else q_layers
    for layer in layers:
        if layer not in q_layers or layer not in tgt_set.layers("k") or layer not in tgt_set.layers("v"):
            raise ValueError(f"layer {layer} needs captured q, k and v")
    if not set(layers) & set(tgt_set.layers("attn_out")):
        raise ValueError("no evaluated layer has a captured attn_out to validate the numpy attention")
    capture_sha = _capture_model_sha(tgt_set)
    model_sha = None
    if capture_sha is not None:
        from collect_v2 import ModelHashCache
        model_sha = ModelHashCache(args.hash_cache).sha256(args.model)
        if model_sha != capture_sha:
            raise ValueError("--model differs from the model recorded in the target capture manifest")
    params = load_model_params(args.model)
    mapper = Mapper(args.mapper)
    check_mapper(mapper, params, layers, src_set)
    alignment = load_alignment(args.alignment)
    plans, skipped = plan_documents(alignment, read_records(args.tgt_docs), tgt_set,
                                    read_records(args.src_docs), src_set, set(splits))
    if not plans:
        raise ValueError("no documents to evaluate")
    sections = evaluate(plans, tgt_set, src_set, mapper, params, layers, layer_types, variants,
                        load_weights=lambda layer: load_output_weights(args.model, layer), chunk=args.chunk,
                        log=lambda message: print(message, flush=True), max_scored=args.max_scored)
    missing_types = sorted({layer_types[layer] for layer in layers}
                           - {item["attention"] for item in sections["validation"]})
    report = {
        "format_version": 1,
        "tool": "attention_eval",
        "settings": {"splits": splits, "layers": layers, "variants": variants, "chunk": args.chunk,
                     "validation_tolerance": VALIDATION_TOLERANCE, "post_norm_eps": POST_NORM_EPS,
                     "model_params": params.__dict__, "prefix": "tokens ending at or before the first "
                     "rendered segment of the final message", "patched": "aligned content tokens in the prefix",
                     "continuation_queries": "tokens inside the final message's segments",
                     "max_scored": args.max_scored, "query_sets": QUERY_SET_NOTES,
                     "mean_kv": "information-free floor: train target mean K/V at every patched position, "
                                "K renormalized per head to RMS 1"},
        "inputs": {"model_sha256": model_sha, "tgt_docs_sha256": sha256_file(args.tgt_docs),
                   "src_docs_sha256": sha256_file(args.src_docs),
                   "alignment_sha256": sha256_file(args.alignment),
                   "mapper_manifest_sha256": sha256_file(args.mapper / "manifest.json"),
                   "source_model_sha256": _capture_model_sha(src_set),
                   "wrapper_sha256": sha256_file(Path(__file__).resolve())},
        "headline_split": "test" if "test" in splits else None,
        "split_note": "fit_layers selects (k, alpha) and k_renorm on dev; dev rows are optimistic, test is "
                      "the headline; splits are never pooled",
        "unvalidated_attention_types": missing_types,
        "documents": [plan.summary() for plan in plans],
        "skipped": skipped,
        **sections,
    }
    _write_new_json(args.output, report)
    print(json.dumps({"output": str(args.output), "documents": len(plans), "layers": layers,
                      "validation": sections["validation"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
