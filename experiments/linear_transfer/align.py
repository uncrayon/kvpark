"""Align source and target model tokens that end at the same byte of the same chat segment."""

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from captures import CaptureSet


TEMPLATE, ALIGNED, UNMATCHED, STRADDLING, EXCLUDED = 0, 1, 2, 3, -1
PAIR_FIELDS = ("src_row", "tgt_row", "doc", "segment", "rel_end", "src_position", "tgt_position")


def _records_by_id(records):
    if isinstance(records, dict):
        items = list(records.items())
    else:
        items = [(record["document_id"], record) for record in records]
    result = {}
    for document_id, record in items:
        if not isinstance(record, dict) or record.get("document_id") != document_id:
            raise ValueError("rendered records must be keyed by their document_id")
        if document_id in result:
            raise ValueError(f"duplicate rendered record {document_id}")
        result[document_id] = record
    return result


def _check_render_provenance(name, records, capture_set):
    """One template and BOS per model, matching what the capture manifests recorded."""
    found = {(record.get("template_sha256"), record.get("bos_token")) for record in records.values()}
    if len(found) > 1:
        raise ValueError(f"{name} records were rendered with different templates or BOS texts")
    template_sha, bos = found.pop() if found else (None, None)
    for capture in getattr(capture_set, "captures", ()):
        manifest = capture.manifest or {}
        for field, value in (("render_template_sha256", template_sha), ("render_bos_token", bos)):
            if manifest.get(field) is not None and manifest[field] != value:
                raise ValueError(f"{name} capture {field} differs from the rendered records")


def segment_table(record):
    """Validated (segment ids, start bytes, end bytes) of a rendered record, in text order."""
    text, segments, keys = record.get("text"), record.get("segments"), record.get("segment_keys")
    if (not isinstance(text, str) or not isinstance(segments, list)
            or not isinstance(keys, list) or any(not isinstance(key, str) for key in keys)):
        raise ValueError(f"malformed rendered record {record.get('document_id')}")
    size, previous, seen = len(text.encode("utf-8")), 0, set()
    for segment in segments:
        if (not isinstance(segment, (list, tuple)) or len(segment) != 3
                or any(type(value) is not int for value in segment)):
            raise ValueError("segments must be [index, start_byte, end_byte] integers")
        index, start, end = segment
        if not 0 <= index < len(keys) or index in seen or not previous <= start < end <= size:
            raise ValueError(f"invalid or overlapping segment {segment} in {record['document_id']}")
        seen.add(index)
        previous = end
    table = np.asarray(segments, dtype=np.int64).reshape(-1, 3)
    return table[:, 0], table[:, 1], table[:, 2]


def _segment_bytes(record, table):
    encoded = record["text"].encode("utf-8")
    return {int(j): encoded[s:e] for j, s, e in zip(*table)}


def classify_tokens(byte_start, byte_end, table):
    """Per token: containing segment id (or -1), inside flag, overlap flag."""
    ids, starts, ends = table
    if not len(ids):
        empty = np.zeros(len(byte_end), dtype=bool)
        return np.full(len(byte_end), -1, dtype=np.int64), empty, empty.copy()
    first = np.searchsorted(starts, byte_start, side="right") - 1
    inside = (first >= 0) & (byte_end <= ends[first.clip(0)])
    last = np.searchsorted(starts, byte_end, side="left") - 1
    overlap = (last >= 0) & (ends[last.clip(0)] > byte_start)
    segment = np.where(inside, ids[first.clip(0)], -1)
    return segment, inside, overlap


@dataclass
class Alignment:
    document_ids: list
    splits: list
    src_row: np.ndarray
    tgt_row: np.ndarray
    doc: np.ndarray
    segment: np.ndarray
    rel_end: np.ndarray
    src_position: np.ndarray
    tgt_position: np.ndarray
    src_class: np.ndarray
    tgt_class: np.ndarray
    rejected: list = field(default_factory=list)

    def pair_mask(self, split):
        chosen = np.asarray([item == split for item in self.splits], dtype=bool)
        return chosen[self.doc] if len(chosen) else np.zeros(len(self.doc), dtype=bool)

    def summary(self):
        counts = {name: int(np.sum(self.tgt_class == value)) for name, value in
                  (("template_or_special", TEMPLATE), ("aligned", ALIGNED),
                   ("unmatched", UNMATCHED), ("straddling", STRADDLING), ("excluded", EXCLUDED))}
        return {"documents": len(self.document_ids), "pairs": int(len(self.src_row)),
                "target_tokens": counts, "rejected": self.rejected,
                "pairs_by_split": {split: int(self.pair_mask(split).sum())
                                   for split in sorted(set(self.splits))}}

    def save(self, path):
        path = Path(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"output already exists: {path}")
        metadata = {"format_version": 1, "document_ids": self.document_ids, "splits": self.splits,
                    "rejected": self.rejected}
        arrays = {name: getattr(self, name) for name in PAIR_FIELDS + ("src_class", "tgt_class")}
        with tempfile.TemporaryDirectory(prefix=".kvpark-align-", dir=path.parent) as temporary:
            pending = Path(temporary) / "alignment.npz"
            with pending.open("xb") as stream:
                np.savez(stream, **arrays,
                         metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
            os.link(pending, path)


def load_alignment(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if metadata.get("format_version") != 1:
            raise ValueError("unsupported alignment file")
        arrays = {name: np.array(data[name]) for name in PAIR_FIELDS + ("src_class", "tgt_class")}
    return Alignment(document_ids=metadata["document_ids"], splits=metadata["splits"],
                     rejected=metadata["rejected"], **arrays)


def align(src_docs, src_capture_set, tgt_docs, tgt_capture_set):
    src_docs, tgt_docs = _records_by_id(src_docs), _records_by_id(tgt_docs)
    if set(src_docs) != set(tgt_docs):
        raise ValueError("source and target rendered records cover different documents")
    _check_render_provenance("source", src_docs, src_capture_set)
    _check_render_provenance("target", tgt_docs, tgt_capture_set)
    for document_id in src_docs:
        for name, capture_set in (("source", src_capture_set), ("target", tgt_capture_set)):
            if document_id not in capture_set:
                raise ValueError(f"{name} capture set lacks document {document_id}")
    src_class = np.full(src_capture_set.rows_total, EXCLUDED, dtype=np.int8)
    tgt_class = np.full(tgt_capture_set.rows_total, EXCLUDED, dtype=np.int8)
    order = [document_id for document_id in tgt_capture_set.document_ids if document_id in tgt_docs]
    pairs = {name: [] for name in PAIR_FIELDS}
    document_ids, splits, rejected = [], [], []
    for document_id in order:
        src_record, tgt_record = src_docs[document_id], tgt_docs[document_id]
        if src_record.get("segment_keys") != tgt_record.get("segment_keys"):
            raise ValueError(f"segment keys differ between models for {document_id}")
        if src_record.get("split") != tgt_record.get("split"):
            raise ValueError(f"split differs between models for {document_id}")
        tables = segment_table(src_record), segment_table(tgt_record)
        src_bytes, tgt_bytes = _segment_bytes(src_record, tables[0]), _segment_bytes(tgt_record, tables[1])
        if any(src_bytes[j] != tgt_bytes[j] for j in src_bytes.keys() & tgt_bytes.keys()):
            raise ValueError(f"segment texts differ between models for {document_id}")
        sides = []
        for record, table, capture_set in ((src_record, tables[0], src_capture_set),
                                           (tgt_record, tables[1], tgt_capture_set)):
            rows = capture_set.rows(document_id)
            start, end = capture_set.byte_start[rows], capture_set.byte_end[rows]
            if int(end[-1]) != len(record["text"].encode("utf-8")):
                raise ValueError(f"capture tokens do not cover the rendered text of {document_id}")
            segment, inside, overlap = classify_tokens(start, end, table)
            special = capture_set.special[rows].astype(bool)
            sides.append((rows, start, end, segment, inside, overlap, special, table))
        bad = [name for name, side in zip(("source", "target"), sides) if np.any(side[5] & side[6])]
        if bad:
            rejected.append({"document_id": document_id, "reason": "special token inside a segment",
                             "models": bad})
            continue
        keys, stride = [], max(len(src_record["text"].encode("utf-8")),
                               len(tgt_record["text"].encode("utf-8"))) + 1
        for rows, start, end, segment, inside, _, _, table in sides:
            ids, starts, _ = table
            seg_start = np.zeros(len(end), dtype=np.int64)
            if len(ids):
                seg_start = starts[np.searchsorted(starts, start, side="right").clip(1) - 1]
            # a token that spells nothing (kv_probe's emptied SentencePiece space) repeats the key of the
            # one before it: it pairs with nothing (unmatched), which keeps the keys unique
            local = np.flatnonzero(inside & (end > start))
            keys.append((local, segment[local] * stride + end[local] - seg_start[local]))
        (src_local, src_keys), (tgt_local, tgt_keys) = keys
        _, src_match, tgt_match = np.intersect1d(src_keys, tgt_keys, assume_unique=True,
                                                 return_indices=True)
        by_target = np.argsort(tgt_local[tgt_match], kind="stable")
        src_match, tgt_match = src_match[by_target], tgt_match[by_target]
        src_pos, tgt_pos = src_local[src_match], tgt_local[tgt_match]
        doc_number = len(document_ids)
        document_ids.append(document_id)
        splits.append(tgt_record.get("split"))
        src_rows, tgt_rows = sides[0][0], sides[1][0]
        pairs["src_row"].append(src_rows.start + src_pos)
        pairs["tgt_row"].append(tgt_rows.start + tgt_pos)
        pairs["doc"].append(np.full(len(tgt_pos), doc_number, dtype=np.int64))
        pairs["segment"].append(sides[1][3][tgt_pos])
        pairs["rel_end"].append(tgt_keys[tgt_match] % stride)
        pairs["src_position"].append(src_pos.astype(np.int64))
        pairs["tgt_position"].append(tgt_pos.astype(np.int64))
        for classes, (rows, _, _, _, inside, overlap, _, _), matched in (
                (src_class, sides[0], src_pos), (tgt_class, sides[1], tgt_pos)):
            local = np.full(rows.stop - rows.start, TEMPLATE, dtype=np.int8)
            local[inside] = UNMATCHED
            local[matched] = ALIGNED
            local[overlap & ~inside] = STRADDLING
            classes[rows] = local
    arrays = {name: (np.concatenate(values).astype(np.int64) if values else np.zeros(0, dtype=np.int64))
              for name, values in pairs.items()}
    return Alignment(document_ids=document_ids, splits=splits, src_class=src_class,
                     tgt_class=tgt_class, rejected=rejected, **arrays)


def read_records(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid rendered JSON on line {number} of {path}") from exc
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-docs", required=True, type=Path, help="source rendered JSONL")
    parser.add_argument("--src-capture", required=True, type=Path, action="append")
    parser.add_argument("--tgt-docs", required=True, type=Path, help="target rendered JSONL")
    parser.add_argument("--tgt-capture", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path, help="new alignment NPZ; never overwritten")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    alignment = align(read_records(args.src_docs), CaptureSet(args.src_capture),
                      read_records(args.tgt_docs), CaptureSet(args.tgt_capture))
    alignment.save(args.output)
    print(json.dumps({"output": str(args.output), **alignment.summary()}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
