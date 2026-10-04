"""Read and validate kvpark-kv-probe capture directories (format version 2)."""

import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np


SHA256 = re.compile(r"[0-9a-f]{64}")
# k/v: pre-RoPE K and V of an attention layer (an MLA layer: the [c_kv | k_pe] row and its c_kv
# view; a DeepSeek V4 layer: its latent before RoPE, kv_norm); ik: DSA lightning-indexer keys
# before RoPE and the Hadamard rotation; csa/hca/lid: a DeepSeek V4 compressed layer's compressor
# inputs [a | s] (the score before APE) of its CSA, HCA or lightning-indexer pool
KEYS = ("k", "v", "q", "gate", "attn_out", "ik", "csa", "hca", "lid")
# the DeepSeek V4 compress ratio of the layers holding each compressor key (report layers' "compress")
COMPRESS = {"csa": 4, "lid": 4, "hca": 128}
# "parallel": attention next to an SSM in the same layer (falcon-h1); "mlp": neither (nemotron-h)
ATTENTION = ("full", "sliding", "parallel")
LAYER_TYPES = ("full", "sliding", "recurrent", "parallel", "mlp")
TOKEN_ARRAYS = {"token_id": "<i4", "byte_end": "<i8", "special": "|u1"}
# the architectures kv_probe captures (its arch_table rows with capture keys), and those whose
# captures come from their saved states (collect_v2.py --mode states)
ARCHITECTURES = ("qwen35", "muse-glimmer", "qwen3", "qwen3next", "qwen35moe", "qwen4exp", "plamo2", "minimax-01",
                 "jamba", "granitehybrid", "nemotron_h", "nemotron_h_moe", "falcon-h1", "lfm2", "lfm2moe", "deepseek2",
                 "mistral4", "deepseek32", "glm-dsa", "dots3note", "kimi-linear", "kimi-k3", "bailingmoe3", "hy_v4",
                 "deepseek4")


def _load_npy(path, dtype, shape, *, mmap):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"capture file is missing or not a regular file: {path.name}")
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            header = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            header = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"{path.name} uses unsupported NPY version {version}")
    stored_shape, fortran, stored_dtype = header
    if stored_dtype.str != dtype or fortran or tuple(stored_shape) != tuple(shape):
        raise ValueError(f"{path.name} has dtype {stored_dtype.str}, shape {tuple(stored_shape)}, "
                         f"fortran_order {fortran}; expected {dtype} {tuple(shape)} C order")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return array if mmap else np.array(array)


def _positive_int(value):
    return type(value) is int and value > 0


def _nonnegative_int(value):
    return type(value) is int and value >= 0


class Capture:
    """One kv_probe output directory; arrays are memory-mapped and never copied whole."""

    def __init__(self, path):
        self.path = Path(path)
        report_path = self.path / "report.json"
        if report_path.is_symlink() or not report_path.is_file():
            raise ValueError(f"capture report is missing: {self.path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest_path = self.path / "manifest.json"
        self.manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                         if manifest_path.is_file() else None)
        self.report = report
        self._validate_report()
        rows = report["rows"]
        for name, dtype in TOKEN_ARRAYS.items():
            setattr(self, name, _load_npy(self.path / f"{name}.npy", dtype, (rows,), mmap=False))
        self._validate_tokens()
        self._arrays = {}
        for capture in report.get("captures", []):
            path = self.path / capture["file"]
            self._arrays[capture["key"], capture["layer"]] = _load_npy(
                path, "<f4", (rows, capture["heads"], capture["dim"]), mmap=True)

    def _validate_report(self):
        report = self.report
        if (not isinstance(report, dict) or type(report.get("format_version")) is not int
                or report["format_version"] != 2 or report.get("mode") not in ("collect", "tokenize")
                or report.get("architecture") not in ARCHITECTURES
                or not isinstance(report.get("file_type"), str) or not _positive_int(report.get("n_layer"))
                or not _nonnegative_int(report.get("rows"))):
            raise ValueError("unsupported or malformed capture report")
        layers = report.get("layers")
        if (not isinstance(layers, list) or len(layers) != report["n_layer"]
                or any(not isinstance(item, dict) or item.get("layer") != index
                       or item.get("attention") not in LAYER_TYPES
                       for index, item in enumerate(layers))):
            raise ValueError("capture report layer table is malformed")
        self.layer_types = [item["attention"] for item in layers]
        if any(item.get("indexer") is not None and not _positive_int(item["indexer"]) for item in layers):
            raise ValueError("capture report layer table is malformed")
        # DSA: the layers that store lightning-indexer keys (their key length)
        self.indexer = {item["layer"]: item["indexer"] for item in layers if item.get("indexer")}
        if any(type(item.get("compress", 0)) is not int or item.get("compress", 0) not in (0, 4, 128)
               for item in layers):
            raise ValueError("capture report layer table is malformed")
        # DeepSeek V4: the compressed layers (their compress ratio; 0 or absent for window-only layers)
        self.compress = {item["layer"]: item["compress"] for item in layers if item.get("compress")}
        documents = report.get("documents")
        if not isinstance(documents, list) or not documents:
            raise ValueError("capture report has no documents")
        seen, first = set(), 0
        for document in documents:
            if (not isinstance(document, dict) or not isinstance(document.get("document_id"), str)
                    or not SHA256.fullmatch(document["document_id"])
                    or not _positive_int(document.get("n_tokens")) or document.get("first_row") != first):
                raise ValueError("capture report documents are malformed or not contiguous")
            if document["document_id"] in seen:
                raise ValueError("duplicate document in capture report")
            seen.add(document["document_id"])
            first += document["n_tokens"]
        if first != report["rows"]:
            raise ValueError("capture report row count differs from its documents")
        captures = report.get("captures", [] if report["mode"] == "tokenize" else None)
        if not isinstance(captures, list) or (report["mode"] == "tokenize" and captures):
            raise ValueError("capture list is malformed for this mode")
        pairs = set()
        for capture in captures:
            if (not isinstance(capture, dict) or capture.get("key") not in KEYS
                    or not _nonnegative_int(capture.get("layer"))
                    or capture["layer"] >= report["n_layer"]
                    or self.layer_types[capture["layer"]] not in ATTENTION
                    or (capture["key"] in COMPRESS and self.compress.get(capture["layer"]) != COMPRESS[capture["key"]])
                    or not _positive_int(capture.get("heads")) or not _positive_int(capture.get("dim"))
                    or capture.get("dtype") != "<f4"
                    or capture.get("file") != f"{capture['key']}-{capture['layer']}.npy"
                    or not isinstance(capture.get("tensor"), str)):
                raise ValueError(f"invalid capture entry: {capture!r}")
            pair = capture["key"], capture["layer"]
            if pair in pairs:
                raise ValueError(f"duplicate capture {pair}")
            pairs.add(pair)
        self.documents = documents
        self.document_index = {document["document_id"]: index
                               for index, document in enumerate(documents)}

    def _validate_tokens(self):
        counts = [document["n_tokens"] for document in self.documents]
        self.doc_index = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        starts = np.asarray([document["first_row"] for document in self.documents], dtype=np.int64)
        first = np.zeros(len(self.byte_end), dtype=bool)
        first[starts] = True
        self.position = np.arange(len(self.byte_end), dtype=np.int64) - starts[self.doc_index]
        self.byte_start = np.empty_like(self.byte_end)
        self.byte_start[0] = 0
        self.byte_start[1:] = self.byte_end[:-1]
        self.byte_start[first] = 0
        if np.any(self.byte_end <= self.byte_start):
            raise ValueError("token byte_end offsets must increase strictly within each document")
        if np.any(self.token_id < 0) or np.any(self.special > 1):
            raise ValueError("token ids must be nonnegative and special flags 0/1")

    @property
    def rows_total(self):
        return self.report["rows"]

    @property
    def architecture(self):
        return self.report["architecture"]

    def rows(self, document_id):
        document = self.documents[self.document_index[document_id]]
        return slice(document["first_row"], document["first_row"] + document["n_tokens"])

    def keys(self):
        return sorted({key for key, _ in self._arrays})

    def layers(self, key):
        return sorted(layer for stored, layer in self._arrays if stored == key)

    def layout(self):
        return {pair: array.shape[1:] for pair, array in sorted(self._arrays.items())}

    def array(self, key, layer):
        try:
            return self._arrays[key, layer]
        except KeyError:
            raise KeyError(f"capture has no {key}-{layer}") from None

    def attention_layers(self):
        return [index for index, kind in enumerate(self.layer_types) if kind in ATTENTION]

    def indexer_layers(self):
        """Layers storing DSA indexer keys (what key "ik" with "all" captures)."""
        return sorted(self.indexer)

    def check_finite(self, chunk_rows=4096):
        for (key, layer), array in self._arrays.items():
            for start in range(0, len(array), chunk_rows):
                if not np.isfinite(array[start:start + chunk_rows]).all():
                    raise ValueError(f"{key}-{layer} contains nonfinite values")

    def check_documents(self, records):
        """Verify document order and that token offsets cover each record's UTF-8 text."""
        if [record["document_id"] for record in records] != [d["document_id"] for d in self.documents]:
            raise ValueError("capture documents differ from the input documents or their order")
        for record, document in zip(records, self.documents):
            last = document["first_row"] + document["n_tokens"] - 1
            if int(self.byte_end[last]) != len(record["text"].encode("utf-8")):
                raise ValueError(f"tokens do not cover document {record['document_id']}")


def open_capture(path):
    return Capture(path)


class CaptureSet:
    """Several capture directories of one model with disjoint documents; rows are flattened."""

    def __init__(self, paths):
        paths = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
        if not paths:
            raise ValueError("a capture set needs at least one capture")
        self.captures = [item if isinstance(item, Capture) else Capture(item) for item in paths]
        first = self.captures[0]
        model = (first.manifest or {}).get("model_sha256")
        for capture in self.captures[1:]:
            if capture.architecture != first.architecture or capture.layer_types != first.layer_types:
                raise ValueError("captures in a set must share architecture and layer types")
            # Captures may hold different extra keys (e.g. q/gate only on eval documents), but a
            # key present in both must have the same layers and row shapes (never truncated).
            mine, theirs = first.layout(), capture.layout()
            shared = {key for key, _ in mine} & {key for key, _ in theirs}
            if any({p: v for p, v in mine.items() if p[0] == key} != {p: v for p, v in theirs.items() if p[0] == key}
                   for key in shared):
                raise ValueError("captures in a set must share the capture layout of every common key")
            if (capture.manifest or {}).get("model_sha256") != model:
                raise ValueError("captures in a set must come from the same model")
        self.offsets = np.cumsum([0] + [capture.rows_total for capture in self.captures])
        self.locations = {}
        for index, capture in enumerate(self.captures):
            for document in capture.documents:
                if document["document_id"] in self.locations:
                    raise ValueError(f"document {document['document_id']} appears in two captures")
                self.locations[document["document_id"]] = index
        self.token_id = np.concatenate([capture.token_id for capture in self.captures])
        self.byte_end = np.concatenate([capture.byte_end for capture in self.captures])
        self.byte_start = np.concatenate([capture.byte_start for capture in self.captures])
        self.special = np.concatenate([capture.special for capture in self.captures])
        self.position = np.concatenate([capture.position for capture in self.captures])

    @property
    def architecture(self):
        return self.captures[0].architecture

    @property
    def rows_total(self):
        return int(self.offsets[-1])

    @property
    def document_ids(self):
        return [document["document_id"] for capture in self.captures for document in capture.documents]

    def __contains__(self, document_id):
        return document_id in self.locations

    def rows(self, document_id):
        """Flattened global row slice of a document."""
        index = self.locations[document_id]
        local = self.captures[index].rows(document_id)
        offset = int(self.offsets[index])
        return slice(local.start + offset, local.stop + offset)

    def layers(self, key):
        """Layers of `key`; empty unless every capture in the set holds the key."""
        if any(not capture.layers(key) for capture in self.captures):
            return []
        return self.captures[0].layers(key)

    def gather(self, key, layer, rows):
        """Copy the given flattened global rows of one capture array into RAM (float32)."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or (len(rows) and (rows.min() < 0 or rows.max() >= self.rows_total)):
            raise IndexError("global rows are out of range")
        which = np.searchsorted(self.offsets, rows, side="right") - 1
        owners = np.unique(which) if len(rows) else np.arange(len(self.captures))
        shapes = {self.captures[index].array(key, layer).shape[1:] for index in owners}
        if len(shapes) != 1:
            raise ValueError(f"{key}-{layer} has inconsistent shapes across captures")
        result = np.empty((len(rows),) + shapes.pop(), dtype=np.float32)
        for index in np.unique(which):
            selected = which == index
            local = rows[selected] - self.offsets[index]
            result[selected] = self.captures[index].array(key, layer)[local]
        return result


def default_layer_types(architecture, n_layer):
    if architecture == "muse-glimmer":
        return ["sliding" if layer % 4 < 3 else "full" for layer in range(n_layer)]
    if architecture == "qwen35":
        return ["full" if layer % 4 == 3 else "recurrent" for layer in range(n_layer)]
    if architecture == "qwen3":
        return ["full"] * n_layer
    raise ValueError("unknown architecture")


def _write_npy(path, array):
    with path.open("xb") as stream:
        np.lib.format.write_array(stream, np.ascontiguousarray(array), version=(1, 0))


def write_synthetic_capture(path, documents, arrays=None, *, architecture="muse-glimmer",
                            n_layer=None, layer_types=None, mode="collect", manifest=None,
                            file_type="7", settings=None):
    """Write a capture directory in the kv_probe format for tests.

    documents: [{"document_id", "byte_end": [...], "token_id"?: [...], "special"?: [...]}]
    arrays: {(key, layer): float32 [N, heads, dim]} with rows in document order.
    """
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    arrays = dict(arrays or {})
    if layer_types is None:
        layers_needed = max([layer for _, layer in arrays] + [3]) + 1
        layer_types = default_layer_types(architecture, n_layer or layers_needed)
    token_id, byte_end, special, entries, first = [], [], [], [], 0
    for document in documents:
        ends = list(document["byte_end"])
        token_id.extend(document.get("token_id", range(len(ends))))
        special.extend(document.get("special", [0] * len(ends)))
        byte_end.extend(ends)
        entries.append({"document_id": document["document_id"], "n_tokens": len(ends),
                        "first_row": first})
        first += len(ends)
    captures = []
    for (key, layer), array in sorted(arrays.items()):
        array = np.asarray(array)
        captures.append({"key": key, "layer": layer, "tensor": f"synthetic-{key}-{layer}",
                         "heads": int(array.shape[1]), "dim": int(array.shape[2]),
                         "file": f"{key}-{layer}.npy", "dtype": "<f4"})
    report = {"format_version": 2, "mode": mode, "architecture": architecture,
              "file_type": file_type, "n_layer": len(layer_types),
              "layers": [{"layer": index, "attention": kind} for index, kind in enumerate(layer_types)],
              "documents": entries, "rows": first}
    if mode == "collect":
        report["settings"] = settings or {"ctx": 5120, "batch": 512, "threads": 1, "cache_type": "f32",
                                          "flash_attention": False, "swa_full": True,
                                          "add_special": False, "parse_special": True}
        report["captures"] = captures
    else:
        report["settings"] = settings or {"ctx": 5120, "add_special": False, "parse_special": True}
    parent = path.parent
    with tempfile.TemporaryDirectory(prefix=".kvpark-synthetic-", dir=parent) as temporary:
        staging = Path(temporary) / "capture"
        staging.mkdir()
        _write_npy(staging / "token_id.npy", np.asarray(token_id, dtype="<i4"))
        _write_npy(staging / "byte_end.npy", np.asarray(byte_end, dtype="<i8"))
        _write_npy(staging / "special.npy", np.asarray(special, dtype="u1"))
        for (key, layer), array in arrays.items():
            _write_npy(staging / f"{key}-{layer}.npy", np.asarray(array, dtype="<f4"))
        (staging / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                             encoding="utf-8")
        if manifest is not None:
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False),
                                                   encoding="utf-8")
        os.rename(staging, path)
    return path
