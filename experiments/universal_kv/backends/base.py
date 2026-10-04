"""What every backend shares: the resident hub sides (an LRU keyed by hub, kind, model, role)."""

from collections import OrderedDict
from pathlib import Path
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent.parent
for _path in (HERE, HERE.parent / "linear_transfer"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import hub as hub_module  # noqa: E402
import kvstate  # noqa: E402

def side_key(hub, kind, name, role):
    """(key, owner) of a hub side: a bundle is keyed by its directory, a {kind: HubKind} by object."""
    if isinstance(hub, hub_module.Hub):
        return (str(hub.directory), kind, name, role), hub
    return (f"HubKind@{id(hub[kind])}", kind, name, role), hub[kind]


def stored_rows(spec, stream, kind, layer):
    """A source stream's stored rows of one layer and kind: K, V, or for an MLA layer (no V stored)
    the c_kv view of its K rows (kvstate.value_rows)."""
    return stream.k[layer] if kind == "k" else kvstate.value_rows(spec, stream, layer)


def latent_columns(spec, layers, norms):
    """[(layer, first column, kv_lora)] of the MLA layers among `layers` (concatenated K rows, in order)
    that have a latent norm."""
    out, offset = [], 0
    for layer in layers:
        heads, dim = spec.head_shape(layer, "k")
        if layer in norms and spec.k_only(layer):
            out.append((layer, offset, spec.kv_lora[layer]))     # one KV head: c_kv leads the row
        offset += heads * dim
    return out


def cast_to_cache(array, dtype, what, out=None):
    """`array` as the cache dtype ("f16" | "f32"), refused like kvstate._cast when not finite or overflowing;
    with out (an array of that dtype and shape), written into it."""
    array = np.asarray(array)
    if array.size:
        high, low = float(array.max()), float(array.min())
        if not (np.isfinite(high) and np.isfinite(low)):
            raise kvstate.StateFormatError(f"{what} must be finite floating point")
        if dtype == "f16" and max(high, -low) >= 65520.0:       # rounds to inf in float16
            raise kvstate.StateFormatError(f"{what} overflows float16")
    if out is None:
        return np.ascontiguousarray(array, dtype=kvstate.GGML_DTYPES[kvstate.DTYPE_NAMES[dtype]])
    if out.dtype != kvstate.GGML_DTYPES[kvstate.DTYPE_NAMES[dtype]]:
        raise ValueError(f"{what}: the output array is {out.dtype}, not {dtype}")
    np.copyto(out, array, casting="same_kind")
    return out


class Backend:
    """Interface (see NumpyBackend): encoder/decoder sides, encode, section, renormalize_latents,
    cache_rows.

    Sides are loaded on first use and kept, most recently used last; the engine daemon trims
    them with drop() and evict()."""

    name = None
    device = None
    window_rows = 4096                    # target positions per translate.compute_cache window
    prefault_output = False               # translate.Prefault touches the fresh output pages ahead

    def __init__(self):
        self._sides = OrderedDict()           # key -> (owner, handle, last used)
        self._lock = threading.RLock()

    # -- resident sides --------------------------------------------------------------------------

    def side(self, hub, kind, name, role):
        key, owner = side_key(hub, kind, name, role)
        with self._lock:
            entry = self._sides.get(key)
            if entry is not None and entry[0] is owner:
                self._sides[key] = (owner, entry[1], time.monotonic())
                self._sides.move_to_end(key)
                return entry[1]
        handle = self._load(hub, kind, name, role)
        with self._lock:
            self._sides[key] = (owner, handle, time.monotonic())
            self._sides.move_to_end(key)
        return handle

    def encoder(self, hub, kind, name):
        return self.side(hub, kind, name, "encoder")

    def decoder(self, hub, kind, name):
        return self.side(hub, kind, name, "decoder")

    def resident(self):
        """[{"key", "bytes", "idle_s"}] oldest first."""
        now = time.monotonic()
        with self._lock:
            return [{"key": list(key), "bytes": int(handle.nbytes), "idle_s": round(now - used, 1)}
                    for key, (_, handle, used) in self._sides.items()]

    def resident_bytes(self):
        with self._lock:
            return int(sum(handle.nbytes for _, handle, _ in self._sides.values()))

    def drop(self, keys=None):
        """Forget resident sides (all, or the given keys); returns the bytes released."""
        with self._lock:
            chosen = list(self._sides) if keys is None else [tuple(k) for k in keys if tuple(k) in self._sides]
            released = sum(self._sides.pop(key)[1].nbytes for key in chosen)
        self._released()
        return int(released)

    def evict(self, nbytes, keep=()):
        """Drop least recently used sides (not in `keep`) until `nbytes` are released; returns released."""
        keep = {tuple(k) for k in keep}
        with self._lock:
            chosen, total = [], 0
            for key, (_, handle, _) in self._sides.items():
                if total >= nbytes:
                    break
                if key not in keep:
                    chosen.append(key)
                    total += handle.nbytes
        return self.drop(chosen) if chosen else 0

    def _released(self):
        """Hook: give freed memory back (the GPU backend empties its allocator cache)."""

    # -- per backend -----------------------------------------------------------------------------

    def _load(self, hub, kind, name, role):
        raise NotImplementedError

    def describe(self):
        return {"backend": self.name, "device": self.device}

    def supports(self, spec):
        """Raise NotImplementedError when this backend cannot translate into/from a model spec."""

    def synchronize(self):
        pass

    def memory(self):
        return {}
