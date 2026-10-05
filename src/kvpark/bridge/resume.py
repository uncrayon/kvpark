"""Slot resume sidecars of kvpark-patched llama-servers, and the draft state a translated restore needs.

A kvpark-patched llama-server (runtime/ as scripts/build_runtime.py builds it at llama.cpp
a7b94df2c, and the production Qwen build that carries the same patch on 85d5703a3) writes
``<slot file>.resume`` next to every slot save and requires it on every restore
(tools/server/server-context.cpp:42-122, save 2805, restore 2870 at a7b94df2c; the file is the
same on both pins). Upstream servers neither write nor read it. Layout, little-endian::

    u32 magic 0x51534c54, u32 version 1
    u64 size, draft blob          llama_state_seq_get_data_ext(ctx_dft, slot, flags 0); empty
                                  when the server has no draft context
    u64 size, speculative blob    common_speculative_get_state: only EAGLE3 keeps one (for
                                  hybrid targets); DFlash and MTP write nothing, and every
                                  set_state ignores an empty blob (common/speculative.cpp:214, 944-977)
    u32 checkpoint count; per checkpoint i64 n_tokens, i32 pos_min, i32 pos_max and three
                                  sized blobs (target, draft, speculative)

On restore a server with a draft context requires a non-empty draft blob that
llama_state_seq_set_data_ext consumes completely; a server without one requires an empty blob
("Slot archive requires draft context"). Any failure clears the slot and fails the restore;
on a7b94df2c a refused state also leaves no rows behind (a failed restore zeroes the cells it
had written, llama_kv_cache::state_clear, and hybrid memories drop every part they restored).

A draft blob is a sequence-state blob: u32 io magic 0xaf143cd8, i32 seq_id, then the draft
memory's state_write (src/llama-context.cpp:3190-3262). Memories are sequences of sections:
a plain KV cache is one KV section, iSWA is a base then a sliding-window KV section
(src/llama-kv-cache-iswa.cpp:259-273), hybrid memory is a KV then a recurrent section. A KV
section (src/llama-kv-cache.cpp state_write 2065-2133, meta 2211-2242, data 2244-2341; the
sections and rows of the draft memories below are the same on both pins)::

    u32 n_stream; per stream: u32 cell_count, and when it is not 0:
      per cell: i32 pos, u32 n_seq_id (1), [i32 x, i32 y, i32 tok when the model uses M-RoPE], i32 seq_id
      u32 v_trans, u32 n_layer
      per layer: i32 k_type, u64 k_row_bytes, k rows
      per layer (none for K-only caches): v_trans 0: i32 v_type, u64 v_row_bytes, v rows;
                                          v_trans 1: i32 v_type, u32 element size, u32 n_embd_v, elements

A recurrent section (src/llama-memory-recurrent.cpp:784-1008) is u32 cell_count, per cell i32 pos
and u32 n_seq_id (0 for a single-sequence save), u32 s_trans, u32 n_layer, then (type, row size,
rows) for every recurrent layer's r rows and again for its s rows.

The blob does not say whether cells carry the M-RoPE extension or whether V exists, so
``parse_state_blob`` tries every reading and keeps the ones that consume the blob exactly;
the draft of a native slot save holds exactly the saved prompt's positions, which settles the
rare remaining ambiguity. (On a7b94df2c a DFlash draft skips the rows of an M-RoPE image, so
a save with images holds a position gap; the bridge's layout templates are one-line text saves.)

Drafts after a translated restore. The translated target state covers [0, T0) and the server
prefills [T0, P) natively, feeding the draft as it goes:

* DFlash/DFlash2/DSpark (an iSWA blob whose base cache has no layers; every layer slides over
  a window of usually 2048 tokens). Its K/V at a position are a function of the target's
  hidden states at that position only (common/speculative.cpp:1157-1239), so with a native
  tail of at least window - 1 tokens the draft's whole visible window is native. The blob is
  an "aligned dummy": two zero K/V cells at T0 - 2 and T0 - 1 in the template's layout. They
  sit outside every later query's window, keep the draft's last position equal to the
  target's (which the 85d5703a3 production build needs: its contiguous-positions patch counts
  draft positions from the draft's own last position; on a7b94df2c the draft takes the
  target's positions, so an empty draft would line up too), pass the strict "next position =
  last + 1" check of non-M-RoPE (NEOX) drafts, and two of them survive the server
  re-evaluating the last restored token.
* MTP (a plain KV blob with one layer: the nextn head's own full-attention cache). An empty
  marker (no cells): zero cells would be attended by every later draft and dilute it. MTP
  then sees only [T0, P) and what follows, as after any restore. The marker does not clear
  the draft sequence, so the slot must be erased before the restore (the bridge always does).
  On a7b94df2c a qwen4exp MTP draft holds the MTP block's attention and indexer and a
  recurrent memory without layers (src/llama-model.cpp:2720-2727): no reading here fits it, so
  it is classified "other" and such a target is skipped (not run: no qwen4exp MTP model).
* Shared (gemma4-assistant, Gemma 4's MTP head: an iSWA cache whose layers are views of the
  target's last sliding and last full layer, src/llama-model.cpp:2825-2853). A cache sharing another
  context's cells writes and reads nothing (llama_kv_cache::state_write/state_read_sinfo return
  first, src/llama-kv-cache.cpp:2066-2069, 2145-2148), so its blob is the io magic and the sequence
  id alone, and that same 8-byte blob is the translated draft: the draft reads the restored target
  rows. (Measured on the Gemma-4 26B-A4B and 31B servers with their gemma4-assistant MTP heads.)
* No draft context: no blob.
"""

import dataclasses
import os
from pathlib import Path
import struct

RESUME_MAGIC = 0x51534C54
RESUME_VERSION = 1
IO_MAGIC = 0xAF143CD8
STATE_FILE_MAGIC = 0x67677371         # llama_state_seq_save_file, include/llama.h
TOKEN_NULL = -1
MAX_STREAMS = 256                     # LLAMA_MAX_SEQ
MAX_LAYERS = 4096
GGML_TYPES = 64                       # ggml types are small enums (GGML_TYPE_COUNT is 43 on both pins)
MAX_SECTIONS = 3                      # iSWA hybrid: base, sliding window, recurrent


class StateBlobError(ValueError):
    pass


# -- resume file ---------------------------------------------------------------------------------

@dataclasses.dataclass
class Checkpoint:
    n_tokens: int
    pos_min: int
    pos_max: int
    data_tgt: bytes = b""
    data_dft: bytes = b""
    data_spec: bytes = b""


@dataclasses.dataclass
class Resume:
    draft: bytes = b""
    speculative: bytes = b""
    checkpoints: list = dataclasses.field(default_factory=list)

    def to_bytes(self):
        def blob(data):
            return struct.pack("<Q", len(data)) + bytes(data)
        out = [struct.pack("<II", RESUME_MAGIC, RESUME_VERSION), blob(self.draft), blob(self.speculative),
               struct.pack("<I", len(self.checkpoints))]
        for cp in self.checkpoints:
            if cp.n_tokens < 0 or cp.pos_min < 0 or cp.pos_max < cp.pos_min:
                raise ValueError("invalid checkpoint positions")
            out += [struct.pack("<qii", cp.n_tokens, cp.pos_min, cp.pos_max),
                    blob(cp.data_tgt), blob(cp.data_dft), blob(cp.data_spec)]
        return b"".join(out)


def parse_resume(data):
    """A resume file with the server's own checks (sizes, counts, positions, no trailing bytes)."""
    reader = _Reader(data)
    if reader.u32() != RESUME_MAGIC or reader.u32() != RESUME_VERSION:
        raise StateBlobError("not a slot resume file (magic, version)")
    remaining = len(data)

    def blob():
        nonlocal remaining
        size = reader.u64()
        if size > remaining:
            raise StateBlobError("invalid slot resume blob size")
        remaining -= size
        return bytes(reader.take(size))

    draft, speculative = blob(), blob()
    count = reader.u32()
    if count > 65536 or count > remaining // 40:
        raise StateBlobError("invalid checkpoint count")
    checkpoints = []
    for _ in range(count):
        n_tokens, pos_min, pos_max = reader.i64(), reader.i32(), reader.i32()
        checkpoint = Checkpoint(n_tokens, pos_min, pos_max, blob(), blob(), blob())
        if n_tokens < 0 or pos_min < 0 or pos_max < pos_min:
            raise StateBlobError("invalid checkpoint positions")
        checkpoints.append(checkpoint)
    if reader.offset != len(data):
        raise StateBlobError("trailing slot resume data")
    return Resume(draft, speculative, checkpoints)


def slot_token_count(path):
    """Prompt tokens in a slot file (llama_state_seq_save_file; llama-server packs its tokens as
    [-1, version, n, ...], tools/server/server-common.cpp:295-333)."""
    with open(path, "rb") as stream:
        head = stream.read(24)
    if len(head) < 12:
        raise StateBlobError(f"{path} is not a slot file")
    magic, _, count = struct.unpack_from("<III", head)
    if magic != STATE_FILE_MAGIC:
        raise StateBlobError(f"{path} is not a slot file")
    if count >= 3 and len(head) >= 24 and struct.unpack_from("<i", head, 12)[0] == TOKEN_NULL:
        return struct.unpack_from("<I", head, 20)[0]
    return count


def write_private(path, data):
    """Create a new owner-only file (0600) holding `data`; never overwrite."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return Path(path)


# -- sequence-state blobs ------------------------------------------------------------------------

class _Reader:
    def __init__(self, data, offset=0):
        self.data, self.offset = memoryview(data), offset

    def take(self, size):
        if size < 0 or self.offset + size > len(self.data):
            raise StateBlobError("state data is truncated")
        view = self.data[self.offset:self.offset + size]
        self.offset += size
        return view

    def _unpack(self, fmt, size):
        if self.offset + size > len(self.data):
            raise StateBlobError("state data is truncated")
        value = struct.unpack_from(fmt, self.data, self.offset)[0]
        self.offset += size
        return value

    def u32(self):
        return self._unpack("<I", 4)

    def i32(self):
        return self._unpack("<i", 4)

    def u64(self):
        return self._unpack("<Q", 8)

    def i64(self):
        return self._unpack("<q", 8)


@dataclasses.dataclass(frozen=True)
class LayerRows:
    """One cache layer's row headers in a KV section; v_type is None for K-only caches."""

    k_type: int
    k_row: int
    v_type: int = None
    v_row: int = None             # v_trans 0: bytes per cell
    v_element: int = None         # v_trans 1: element size and elements per cell
    v_embd: int = None


@dataclasses.dataclass
class KVStream:
    positions: tuple
    ext: tuple                    # (x, y, tok) per cell, or None without M-RoPE
    seq_ids: tuple


@dataclasses.dataclass
class KVSection:
    n_stream: int
    streams: list                 # KVStream, or None for an empty stream
    ext: bool
    v_trans: int = None           # None (and layers None) when every stream is empty
    layers: tuple = None

    @property
    def cells(self):
        return sum(len(s.positions) for s in self.streams if s is not None)

    def describe(self):
        if self.layers is None:
            return "empty"
        kinds = "K-only" if self.layers and self.layers[0].v_type is None else "K/V"
        return (f"{len(self.layers)} layers {kinds}, {self.cells} cells, v_trans {self.v_trans}"
                + (", M-RoPE cells" if self.ext else ""))


@dataclasses.dataclass
class RecurrentSection:
    positions: tuple
    n_layer: int
    r: tuple                      # (type, row bytes) per recurrent layer
    s: tuple

    def describe(self):
        return f"recurrent, {len(self.r)} layers, {len(self.positions)} cells"


@dataclasses.dataclass
class StateBlob:
    seq_id: int
    sections: list


def _kv_section(reader, ext, v_present):
    n_stream = reader.u32()
    if not 1 <= n_stream <= MAX_STREAMS:
        raise StateBlobError(f"implausible stream count {n_stream}")
    streams, layout = [], None
    for _ in range(n_stream):
        cells = reader.u32()
        if cells == 0:
            streams.append(None)
            continue
        if cells > (len(reader.data) - reader.offset) // 12:
            raise StateBlobError("cell count exceeds the data")
        positions, exts, seq_ids = [], [] if ext else None, []
        for _ in range(cells):
            pos, n_seq = reader.i32(), reader.u32()
            if pos < 0 or n_seq != 1:      # single-sequence saves (state_read_meta requires n_seq_id 1)
                raise StateBlobError("not a single-sequence cell")
            if ext:
                exts.append((reader.i32(), reader.i32(), reader.i32()))
            positions.append(pos)
            seq_ids.append(reader.i32())
        v_trans, n_layer = reader.u32(), reader.u32()
        if v_trans not in (0, 1) or n_layer > MAX_LAYERS:
            raise StateBlobError("implausible KV data header")
        keys = []
        for _ in range(n_layer):
            k_type, k_row = reader.i32(), reader.u64()
            if not 0 <= k_type < GGML_TYPES or not 0 < k_row < 1 << 32:
                raise StateBlobError("implausible key rows")
            reader.take(cells * k_row)
            keys.append((k_type, k_row))
        layers = []
        for k_type, k_row in keys:
            if not v_present:
                layers.append(LayerRows(k_type, k_row))
            elif not v_trans:
                v_type, v_row = reader.i32(), reader.u64()
                if not 0 <= v_type < GGML_TYPES or not 0 < v_row < 1 << 32:
                    raise StateBlobError("implausible value rows")
                reader.take(cells * v_row)
                layers.append(LayerRows(k_type, k_row, v_type, v_row))
            else:
                v_type, element, n_embd = reader.i32(), reader.u32(), reader.u32()
                if not 0 <= v_type < GGML_TYPES or not 0 < element <= 8 or not 0 < n_embd < 1 << 24:
                    raise StateBlobError("implausible transposed value rows")
                reader.take(cells * element * n_embd)
                layers.append(LayerRows(k_type, k_row, v_type, element * n_embd, element, n_embd))
        if layout is not None and layout != (v_trans, tuple(layers)):
            raise StateBlobError("streams disagree on the layer layout")
        layout = (v_trans, tuple(layers))
        streams.append(KVStream(tuple(positions), tuple(exts) if ext else None, tuple(seq_ids)))
    return KVSection(n_stream, streams, ext, *(layout or (None, None)))


def _recurrent_section(reader):
    cells = reader.u32()
    if cells > (len(reader.data) - reader.offset) // 8:
        raise StateBlobError("cell count exceeds the data")
    positions = []
    for _ in range(cells):
        pos, n_seq = reader.i32(), reader.u32()
        if n_seq != 0:
            raise StateBlobError("not a single-sequence recurrent cell")
        positions.append(pos)
    s_trans, n_layer = reader.u32(), reader.u32()
    if s_trans != 0 or not 0 < n_layer <= MAX_LAYERS:
        raise StateBlobError("implausible recurrent data header")
    groups = []
    while reader.offset < len(reader.data):
        kind, row = reader.i32(), reader.u64()
        if not 0 <= kind < GGML_TYPES or not 0 < row < 1 << 40:
            raise StateBlobError("implausible recurrent rows")
        reader.take(cells * row)
        groups.append((kind, row))
    half = len(groups) // 2
    if not groups or len(groups) % 2 or half > n_layer or len(set(groups[:half])) != 1 or len(set(groups[half:])) != 1:
        raise StateBlobError("recurrent rows do not pair up")
    return RecurrentSection(tuple(positions), n_layer, tuple(groups[:half]), tuple(groups[half:]))


def _readings(data, offset, ext, v_present, depth=0):
    """Every way the bytes from `offset` split into sections; a recurrent section can only be last."""
    if offset == len(data):
        if depth:
            yield []
        return
    if depth == MAX_SECTIONS:
        return
    reader = _Reader(data, offset)
    try:
        section = _kv_section(reader, ext, v_present)
    except StateBlobError:
        pass
    else:
        for rest in _readings(data, reader.offset, ext, v_present, depth + 1):
            yield [section] + rest
    reader = _Reader(data, offset)
    try:
        yield [_recurrent_section(reader)]
    except StateBlobError:
        pass


def _signature(sections):
    out = []
    for section in sections:
        if isinstance(section, KVSection):
            cells = section.layers is not None
            out.append(("kv", section.n_stream, section.ext if cells else None, section.v_trans, section.layers,
                        tuple(s.positions if s else None for s in section.streams)))
        else:
            out.append(("recurrent", section.positions, section.r, section.s))
    return tuple(out)


def _fits(sections, n_tokens):
    """A draft saved with a prompt of n tokens holds a run of positions ending at n - 1."""
    for section in sections:
        streams = section.streams if isinstance(section, KVSection) else [section]
        for stream in streams:
            if stream is None or not stream.positions:
                continue
            positions = list(stream.positions)
            if positions != list(range(n_tokens - len(positions), n_tokens)):
                return False
            if isinstance(section, KVSection) and section.ext and any(
                    (x, y) != (p, p) for (x, y, _), p in zip(stream.ext, positions)):
                return False
    return True


def parse_state_blob(data, *, n_tokens=None):
    """The sections of a llama_state_seq_get_data blob. `n_tokens` (the prompt length of the
    save it came from) settles readings that the bytes alone leave open. A memory that shares
    another context's cells writes nothing: no sections."""
    reader = _Reader(data)
    if reader.u32() != IO_MAGIC:
        raise StateBlobError("not a sequence state blob")
    seq_id = reader.i32()
    if reader.offset == len(data):
        return StateBlob(seq_id, [])
    found = {}
    for ext in (False, True):
        for v_present in (True, False):
            for sections in _readings(data, reader.offset, ext, v_present):
                found.setdefault(_signature(sections), sections)
    readings = list(found.values())
    if len(readings) > 1 and n_tokens is not None:
        readings = [sections for sections in readings if _fits(sections, n_tokens)] or readings
    if not readings:
        raise StateBlobError("the bytes do not form KV or recurrent sections")
    if len(readings) > 1:
        raise StateBlobError(f"{len(readings)} different layouts fit these bytes")
    return StateBlob(seq_id, readings[0])


# -- drafts --------------------------------------------------------------------------------------

DRAFT_KINDS = ("none", "dflash", "mtp", "shared", "other")


@dataclasses.dataclass
class DraftLayout:
    kind: str
    note: str
    blob: StateBlob = None


def classify_draft(draft, *, n_tokens=None):
    """What kind of draft context wrote this blob (a native save's .resume draft blob)."""
    if not draft:
        return DraftLayout("none", "no draft context")
    try:
        blob = parse_state_blob(draft, n_tokens=n_tokens)
    except StateBlobError as error:
        return DraftLayout("other", f"unreadable draft state ({error})")
    sections = blob.sections
    if not sections:
        return DraftLayout("shared", "draft memory that writes no cells of its own (it shares another context's, "
                           "e.g. gemma4-assistant's views of the target's cache)", blob)
    described = "; ".join(section.describe() for section in sections)
    kv = all(isinstance(section, KVSection) for section in sections)
    if kv and len(sections) == 2 and sections[0].layers == () and sections[1].layers:
        return DraftLayout("dflash", f"sliding-window draft (DFlash/DSpark): base {described}", blob)
    if kv and len(sections) == 1 and sections[0].layers is not None and len(sections[0].layers) == 1:
        return DraftLayout("mtp", f"single-layer full-attention draft (MTP): {described}", blob)
    return DraftLayout("other", f"draft memory the bridge cannot synthesize: {described}", blob)


def _kv_bytes(section, positions, seq_id, *, v_trans=None):
    """A KV section holding `positions` (one sequence) with zero rows, in `section`'s layout."""
    if not positions:
        return struct.pack("<I", section.n_stream) + struct.pack("<I", 0) * section.n_stream
    layers = section.layers if section.layers is not None else ()
    v_trans = section.v_trans if section.v_trans is not None else v_trans
    if v_trans is None:
        raise ValueError("the template draft section holds no cells, so its layout is unknown")
    home = next((s for s, stream in enumerate(section.streams) if stream is not None), 0)
    n = len(positions)
    out = [struct.pack("<I", section.n_stream)]
    for s in range(section.n_stream):
        if s != home:
            out.append(struct.pack("<I", 0))
            continue
        out.append(struct.pack("<I", n))
        for pos in positions:
            out.append(struct.pack("<iI", pos, 1))
            if section.ext:
                # what DFlash injection stores: a temporal-only M-RoPE position, no token
                out.append(struct.pack("<iii", pos, pos, TOKEN_NULL))
            out.append(struct.pack("<i", seq_id))
        out.append(struct.pack("<II", v_trans, len(layers)))
        for layer in layers:
            out += [struct.pack("<iQ", layer.k_type, layer.k_row), bytes(layer.k_row * n)]
        for layer in layers:
            if layer.v_type is None:
                continue
            if not v_trans:
                out += [struct.pack("<iQ", layer.v_type, layer.v_row), bytes(layer.v_row * n)]
            else:
                out += [struct.pack("<iII", layer.v_type, layer.v_element, layer.v_embd),
                        bytes(layer.v_element * layer.v_embd * n)]
    return b"".join(out)


def translated_draft(layout, t0):
    """The draft blob for a translated restore of [0, t0), in the layout of a native save."""
    if layout.kind == "none":
        return b""
    if layout.kind not in ("mtp", "dflash", "shared"):
        # (an unreadable draft has no blob at all)
        raise ValueError(f"no translated draft for {layout.kind} drafts: {layout.note}")
    blob = layout.blob
    head = struct.pack("<Ii", IO_MAGIC, blob.seq_id)
    if layout.kind == "shared":
        return head
    if layout.kind == "mtp":
        return head + _kv_bytes(blob.sections[0], (), blob.seq_id)
    if t0 < 2:
        raise ValueError("an aligned draft needs at least two restored tokens")
    base, window = blob.sections
    positions = (t0 - 2, t0 - 1)
    # both caches use the context's V layout; an empty base stream does not record it
    return head + _kv_bytes(base, positions, blob.seq_id, v_trans=window.v_trans) + _kv_bytes(
        window, positions, blob.seq_id)


def translated_resume(template_resume, t0, *, n_tokens=None):
    """(resume file bytes, DraftLayout) for a translated slot of t0 tokens, from the .resume of a
    native template save (whose prompt had n_tokens tokens): the synthesized draft blob, no
    speculative state and no checkpoints."""
    layout = classify_draft(parse_resume(template_resume).draft, n_tokens=n_tokens)
    return Resume(translated_draft(layout, t0)).to_bytes(), layout
