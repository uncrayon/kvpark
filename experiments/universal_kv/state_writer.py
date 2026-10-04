"""Write a kvstate.StateFile to disk part by part: the bytes of kvstate.write_new(path, state.to_bytes()).

    state_writer.write(path, state_file)

to_bytes() joins the whole file in memory (each array copied twice, a Python loop per cell
for the metadata) and write_new fsyncs it: 0.85 s for a 963 MB Qwen3.8 state against 0.17 s
here. The header, token words and every section are written as they are serialized, in the
order and with the checks of SequenceState.to_bytes (spec.layouts). KV sections take a fast
path (cell metadata built with numpy, arrays written from memoryviews, no copies) when they
hold exactly what kvstate._write_kv accepts (K-only layers and indexer rows included) and the
fast path's bytes for the section's first cells equal kvstate._write_kv's (so a later change
to kvstate's KV format sends sections to kvstate instead); every other section goes to
kvstate's own serializer for its kind (`_write_<kind>`), so section types kvstate learns later
keep working; a kind without such a serializer falls back to the whole payload of
SequenceState.to_bytes().

The file is created next to `path` with mode 0600, then linked into place (atomic, and like
write_new it never replaces an existing file); no fsync: the bridge's output lives seconds,
until the target server restores it.
"""

import dataclasses
import os
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "linear_transfer"))

import kvstate  # noqa: E402

CODES = {np.dtype("<f4"): kvstate.GGML_TYPE_F32, np.dtype("<f2"): kvstate.GGML_TYPE_F16}
CHECK_CELLS = 2                 # cells per stream the fast path is checked on against kvstate


def _raw(array):
    return memoryview(np.ascontiguousarray(array).reshape(-1).view(np.uint8))


def _little(array):
    """(little-endian array, ggml type) as kvstate._layer_array serializes a layer."""
    array = np.asarray(array)
    little = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return little, CODES.get(little.dtype)


def _shape(spec, layer, key, role):
    """(heads, dim) a section stores for a layer, or None for the V of a K-only layer."""
    return spec.kv_shape(layer, key, role)


def _kv_fast_path(spec, cache, role):
    """Whether a KV section holds exactly what kvstate._write_kv accepts (then _kv_chunks writes its bytes)."""
    if type(cache) is not kvstate.KVCache:
        return False
    with_v = {il for il in cache.layers if _shape(spec, il, "v", role) is not None}
    for stream in cache.streams:
        if stream is None:
            continue
        if type(stream) is not kvstate.KVStream:
            return False
        cells = stream.n_cells
        positions = np.asarray(stream.positions)
        if (cells == 0 or positions.shape != (cells,) or len(stream.seq_ids) != cells
                or spec.cell_ext != (stream.ext is not None)
                or (stream.ext is not None and np.shape(stream.ext) != (cells, 3))
                or any(len(ids) != 1 for ids in stream.seq_ids)
                or set(stream.k) != set(cache.layers) or set(stream.v) != with_v):
            return False
        for il in cache.layers:
            for key, arrays, types in (("k", stream.k, stream.k_types), ("v", stream.v, stream.v_types)):
                if key == "v" and il not in with_v:
                    continue
                array, code = _little(arrays[il])
                if (code is None or (il in types and types[il] != code)
                        or array.shape != (cells,) + _shape(spec, il, key, role)):
                    return False
    return True


def _kv_chunks(spec, cache, role):
    """kvstate._write_kv's bytes, as chunks (numpy cell metadata, arrays as memoryviews)."""
    yield struct.pack("<I", len(cache.streams))
    for stream in cache.streams:
        if stream is None:
            yield struct.pack("<I", 0)
            continue
        cells = stream.n_cells
        yield struct.pack("<I", cells)
        fields = [("pos", "<i4"), ("n", "<u4")]
        if stream.ext is not None:
            fields.append(("ext", "<i4", (3,)))
        fields.append(("seq", "<i4"))
        meta = np.empty(cells, dtype=np.dtype(fields))
        meta["pos"] = np.asarray(stream.positions)
        meta["n"] = 1
        if stream.ext is not None:
            meta["ext"] = np.asarray(stream.ext)
        meta["seq"] = [ids[0] for ids in stream.seq_ids]
        yield meta.tobytes()
        yield struct.pack("<II", int(bool(stream.v_trans)), len(cache.layers))
        for il in cache.layers:
            array, code = _little(stream.k[il])
            yield struct.pack("<iQ", code, array.nbytes // cells)
            yield _raw(array)
        for il in cache.layers:
            shape = _shape(spec, il, "v", role)
            if shape is None:
                continue
            array, code = _little(stream.v[il])
            n_embd = shape[0] * shape[1]
            if stream.v_trans:
                yield struct.pack("<iII", code, array.dtype.itemsize, n_embd)
                yield _raw(array.reshape(cells, n_embd).T)
            else:
                yield struct.pack("<iQ", code, array.dtype.itemsize * n_embd)
                yield _raw(array)


def _head(cache, cells=CHECK_CELLS):
    """The section cut to its first cells per stream."""
    def cut(stream):
        return dataclasses.replace(
            stream, positions=np.asarray(stream.positions)[:cells], seq_ids=tuple(stream.seq_ids[:cells]),
            ext=None if stream.ext is None else np.asarray(stream.ext)[:cells],
            k={il: np.asarray(a)[:cells] for il, a in stream.k.items()},
            v={il: np.asarray(a)[:cells] for il, a in stream.v.items()})
    return kvstate.KVCache(cache.name, cache.layers, [None if s is None else cut(s) for s in cache.streams])


def _fast_path_agrees(spec, cache, layout):
    """Whether _kv_chunks writes kvstate._write_kv's bytes, checked on the section's first cells."""
    head = _head(cache)
    return b"".join(_kv_chunks(spec, head, layout.role)) == kvstate._write_kv(spec, head, layout)


def _payload_chunks(state):
    """The memory payload of a SequenceState (as SequenceState.to_bytes checks and emits it)."""
    spec = state.spec
    layouts = spec.layouts(state.flags)
    if [p.name for p in state.parts] != [layout.name for layout in layouts]:
        raise kvstate.StateFormatError("state sections do not match the model spec")
    serializers = []
    for part, layout in zip(state.parts, layouts):
        if tuple(part.layers) != layout.layers:
            raise kvstate.StateFormatError(f"{part.name} layers {part.layers} differ from the spec {layout.layers}")
        serializers.append(getattr(kvstate, f"_write_{layout.kind}", None))
    if any(serializer is None for serializer in serializers):
        yield state.to_bytes()          # a section kind without a per-part serializer: kvstate does it all
        return
    for part, layout, serializer in zip(state.parts, layouts, serializers):
        if (layout.kind == "kv" and serializer is kvstate._write_kv and _kv_fast_path(spec, part, layout.role)
                and _fast_path_agrees(spec, part, layout)):
            yield from _kv_chunks(spec, part, layout.role)
        elif layout.kind == "kv":
            yield serializer(spec, part, layout)
        else:
            yield serializer(spec, part)


def chunks(state_file, *, check_positions=True):
    """The bytes of state_file.to_bytes(check_positions=...), as a sequence of buffers. The token
    words are checked as to_bytes checks them (kvstate.parse_media_tokens: plain ids, or
    server_tokens words that may hold media chunks)."""
    words = np.asarray(state_file.token_words, dtype="<i4")
    kvstate.parse_media_tokens(words)
    if check_positions:
        state_file.check_positions()
    yield struct.pack("<III", kvstate.LLAMA_STATE_SEQ_MAGIC, kvstate.LLAMA_STATE_SEQ_VERSION, len(words))
    yield _raw(words)
    yield from _payload_chunks(state_file.state)


def write(path, state_file, *, check_positions=True):
    """Write a new file holding state_file's bytes (0600, atomic, never overwriting); returns its size."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.partial-", dir=path.parent)
    size = 0
    try:
        with os.fdopen(fd, "wb", buffering=0) as stream:
            for chunk in chunks(state_file, check_positions=check_positions):
                view = memoryview(chunk)
                while view:
                    written = stream.write(view)
                    view = view[written:]
                    size += written
        try:
            os.link(temp, path)                 # fails if path exists: never replaces a file
        except FileExistsError:
            raise
        except OSError:                         # file systems without hard links
            if path.exists():
                raise FileExistsError(path) from None
            os.rename(temp, path)
            temp = None
    finally:
        if temp is not None:
            os.unlink(temp)
    return size
