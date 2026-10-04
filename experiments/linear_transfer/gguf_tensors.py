"""Read GGUF metadata and selected tensor rows through gguf-py's memory map.

Only the requested rows are read and dequantized; whole large tensors are never
materialized unless explicitly requested with rows=None.
"""

import os
from pathlib import Path
import sys

import numpy as np


GGUF_PY = Path(__file__).resolve().parents[2] / "runtime/source/gguf-py"
SUPPORTED = ("F32", "F16", "BF16", "Q8_0")
Q8_0_BLOCK, Q8_0_BYTES = 32, 34
_READERS = {}


def _gguf():
    if not GGUF_PY.is_dir():
        raise RuntimeError(f"pinned gguf-py not found: {GGUF_PY}")
    if str(GGUF_PY) not in sys.path:
        sys.path.insert(0, str(GGUF_PY))
    import gguf
    return gguf


def _identity(path):
    stat = os.stat(path)
    return os.path.realpath(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def reader(path):
    """Cached read-only GGUFReader; reopened when the file identity changes."""
    key = _identity(path)
    cached = _READERS.get(key[0])
    if cached is None or cached[0] != key:
        opened = _gguf().GGUFReader(path, "r")
        if opened.byte_order != "I" or sys.byteorder != "little":
            raise ValueError("only little-endian GGUF files on little-endian hosts are supported")
        cached = (key, opened, {tensor.name: tensor for tensor in opened.tensors})
        _READERS[key[0]] = cached
    return cached[1], cached[2]


def _field_value(field, max_array):
    gguf = _gguf()
    kind = field.types[0] if field.types else None
    if kind == gguf.GGUFValueType.ARRAY:
        if len(field.data) > max_array:
            return {"array_length": len(field.data), "element_type": field.types[-1].name}
        return field.contents()
    return field.contents()


def read_metadata(path, *, max_array=64):
    """Metadata key/values; arrays longer than max_array are summarized by length."""
    opened, _ = reader(path)
    return {key: _field_value(field, max_array) for key, field in opened.fields.items()
            if not key.startswith("GGUF.")}


def metadata_value(path, key, default=None):
    opened, _ = reader(path)
    field = opened.fields.get(key)
    return default if field is None else field.contents()


def chat_template(path):
    return metadata_value(path, "tokenizer.chat_template")


def token_text(path, token_id):
    opened, _ = reader(path)
    field = opened.fields.get("tokenizer.ggml.tokens")
    if field is None:
        raise ValueError("GGUF has no tokenizer.ggml.tokens")
    if type(token_id) is not int and not isinstance(token_id, np.integer):
        raise ValueError("token id must be an integer")
    if not 0 <= int(token_id) < len(field.data):
        raise ValueError(f"token id {token_id} outside vocabulary of {len(field.data)}")
    return bytes(field.parts[field.data[int(token_id)]]).decode("utf-8")


def tensor_info(path, name):
    """Type name, ggml ne order shape, numpy row-major shape, and file byte offset."""
    _, tensors = reader(path)
    if name not in tensors:
        raise KeyError(f"tensor not found: {name}")
    tensor = tensors[name]
    ne = [int(size) for size in tensor.shape]
    return {"name": name, "type": tensor.tensor_type.name, "ne": ne,
            "shape": list(reversed(ne)), "data_offset": int(tensor.data_offset),
            "n_bytes": int(tensor.n_bytes)}


def dequantize_rows(raw, type_name):
    """Dequantize raw row bytes/elements of one ggml type to float32."""
    raw = np.asarray(raw)
    if type_name == "F32":
        return raw.astype(np.float32)
    if type_name == "F16":
        return raw.astype(np.float32)
    if type_name == "BF16":
        bits = np.ascontiguousarray(raw).view(np.uint16)
        return (bits.astype(np.uint32) << 16).view(np.float32)
    if type_name == "Q8_0":
        if raw.dtype != np.uint8 or raw.shape[-1] % Q8_0_BYTES:
            raise ValueError("Q8_0 rows must be uint8 multiples of 34 bytes")
        blocks = np.ascontiguousarray(raw).reshape(*raw.shape[:-1], raw.shape[-1] // Q8_0_BYTES, Q8_0_BYTES)
        scale = blocks[..., :2].copy().view(np.float16).astype(np.float32)
        quant = blocks[..., 2:].copy().view(np.int8).astype(np.float32)
        return (scale * quant).reshape(*raw.shape[:-1], raw.shape[-1] // Q8_0_BYTES * Q8_0_BLOCK)
    raise ValueError(f"unsupported tensor type {type_name}; supported: {', '.join(SUPPORTED)}")


def _row_index(rows, count):
    if rows is None:
        return np.arange(count)
    if isinstance(rows, slice):
        return np.arange(count)[rows]
    index = np.asarray(rows)
    if index.ndim == 1 and index.size == 0:
        return np.zeros(0, dtype=np.int64)
    if index.ndim != 1 or index.dtype.kind not in "iu":
        raise ValueError("rows must be None, a slice, or a one-dimensional integer sequence")
    if len(index) and (index.min() < 0 or index.max() >= count):
        raise ValueError(f"row index outside 0..{count - 1}")
    return index.astype(np.int64)


def read_tensor(path, name, rows=None, *, chunk_rows=4096):
    """Float32 tensor in numpy (row-major, reversed ne) order.

    `rows` selects along the first numpy axis (ne[-1]); only those rows are read.
    One-dimensional tensors accept rows=None only.
    """
    _, tensors = reader(path)
    if name not in tensors:
        raise KeyError(f"tensor not found: {name}")
    tensor = tensors[name]
    type_name = tensor.tensor_type.name
    if type_name not in SUPPORTED:
        raise ValueError(f"unsupported tensor type {type_name} for {name}")
    data = tensor.data
    shape = tuple(reversed([int(size) for size in tensor.shape]))
    if len(shape) == 1:
        if rows is not None:
            raise ValueError("row selection needs a tensor with at least two dimensions")
        values = dequantize_rows(np.asarray(data).reshape(1, -1), type_name).reshape(shape)
        return _checked(values, name)
    index = _row_index(rows, shape[0])
    output = np.empty((len(index), *shape[1:]), dtype=np.float32)
    for start in range(0, len(index), chunk_rows):
        part = index[start:start + chunk_rows]
        output[start:start + len(part)] = dequantize_rows(np.asarray(data[part]), type_name).reshape(
            len(part), *shape[1:])
    return _checked(output, name)


def _checked(values, name):
    if not np.isfinite(values).all():
        raise ValueError(f"tensor {name} contains nonfinite values")
    return values
