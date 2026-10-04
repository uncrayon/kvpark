"""The recurrent cell a hybrid target restores at the handoff point t0 - 1.

A hybrid target (gated DeltaNet: Qwen3.5/3.8, Ornith; and every other recurrent family kvstate
parses) keeps one recurrent cell per sequence: per recurrent layer a short row R (the conv taps:
the last K - 1 qkv projections for GDN), a long-memory row S (per head the delta-rule matrix)
and, for qwen4exp, a PLE row P. They summarize the whole history and have no per-token form, so
the translator cannot carry them over from the source: it writes one cell at t0 - 1 and the
native tail [t0, P) updates it. This module decides what that cell holds:

    state = initial_state(template_part, t0 - 1, "mean", mean=RecurrentMean.load(path))

Strategies (STRATEGIES; `mean_s<x>` scales the mean's S rows by x, e.g. mean_s0.5):

  zero        every row zero (R0): exactly llama.cpp's fresh-sequence state, what translate.py
              writes today (kvstate.recurrent_reset)
  mean        per-layer mean of native recurrent cells of the target (R0-mean), saved by native
              prefills of training documents at many depths (build_mean / MeanAccumulator,
              stored per model as an npz, RecurrentMean)
  mean_conv   R and P from the mean, S zero (only the short conv window gets the average)
  prefix      the rows of a native save of the prompt's start (R0-prefix): the system prompt the
              bridge already prefills natively for Qwen targets (translate's native_state)

A mean may hold several groups (e.g. "chat" and "tools", by whether the conversation declares
tools) besides "all", the mean over every sample; initial_state takes the group to use, and
match_group picks it for a conversation.

Rows are F32 in every family (src/llama-model.cpp:2503-2511); the P rows of qwen4exp are written
without a type and read here as F32 too. Templates, means and prefix states must share one
layout (section name, layers, row sizes), and a mean of another layout is refused. Only the
layout is compared, not the model: a mean of another quantization or fine-tune of the same
architecture is accepted (RecurrentMean.model records the GGUF it was built from).
"""

import dataclasses
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "linear_transfer"))

import kvstate  # noqa: E402

FORMAT_VERSION = 1
ROWS = ("r", "s", "p")
ALL = "all"
# strategy name -> (source of the R and P rows, source of the S rows, scale of the S rows)
STRATEGIES = {
    "zero": ("zero", "zero", 1.0),
    "mean": ("mean", "mean", 1.0),
    "mean_conv": ("mean", "zero", 1.0),
    "prefix": ("prefix", "prefix", 1.0),
}
SCALED = re.compile(r"^mean_s(?P<scale>\d+(\.\d+)?)$")
F16_MAX = 65504.0


@dataclasses.dataclass(frozen=True)
class Rule:
    """Where a strategy takes each row from ("zero" | "mean" | "prefix") and the scale of S."""

    name: str
    r: str
    s: str
    s_scale: float = 1.0

    @property
    def needs(self):
        return {self.r, self.s} - {"zero"}


def parse_strategy(name):
    """The Rule of a strategy name (STRATEGIES, or mean_s<x>)."""
    if isinstance(name, Rule):
        return name
    if name in STRATEGIES:
        return Rule(name, *STRATEGIES[name])
    match = SCALED.match(str(name))
    if match:
        return Rule(name, "mean", "mean", float(match["scale"]))
    raise ValueError(f"unknown recurrent strategy {name!r}: one of {sorted(STRATEGIES)} or mean_s<scale>")


@dataclasses.dataclass(frozen=True)
class Layout:
    """What a recurrent section's rows look like: its layers, model layer count and row bytes."""

    name: str
    layers: tuple
    n_layer: int
    r_bytes: tuple                  # per layer, in `layers` order
    s_bytes: tuple
    p_layers: tuple = ()
    p_bytes: int = 0

    @classmethod
    def of(cls, part):
        p = getattr(part, "p", None) or {}
        p_bytes = {rows.row_bytes for rows in p.values()}
        if len(p_bytes) > 1:
            raise ValueError(f"{part.name}: PLE rows of different sizes {sorted(p_bytes)}")
        return cls(part.name, tuple(int(l) for l in part.layers), int(part.n_layer),
                   tuple(int(part.r[l].row_bytes) for l in part.layers),
                   tuple(int(part.s[l].row_bytes) for l in part.layers),
                   tuple(sorted(int(l) for l in p)), int(next(iter(p_bytes), 0)))

    def to_json(self):
        return {key: list(value) if isinstance(value, tuple) else value
                for key, value in dataclasses.asdict(self).items()}

    @classmethod
    def from_json(cls, data):
        return cls(**{key: tuple(value) if isinstance(value, list) else value for key, value in data.items()})

    def require(self, other, what):
        if self != other:
            raise ValueError(f"{what} has another recurrent layout ({_layout_difference(self, other)})")


def _layout_difference(a, b):
    for field in dataclasses.fields(Layout):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if x != y:
            return f"{field.name}: {_short(x)} vs {_short(y)}"
    return "same"


def _short(value):
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def row_sizes(layout, key):
    """[(layer, row bytes)] of one kind of row ("r" | "s" | "p") in a Layout."""
    if key == "p":
        return [(layer, layout.p_bytes) for layer in layout.p_layers]
    return list(zip(layout.layers, layout.r_bytes if key == "r" else layout.s_bytes))


def recurrent_part(state):
    """The one recurrent section of a kvstate.StateFile (or the RecurrentState itself)."""
    if isinstance(state, kvstate.RecurrentState):
        return state
    parts = [part for part in state.state.parts if isinstance(part, kvstate.RecurrentState)]
    if len(parts) != 1:
        raise ValueError(f"the state holds {len(parts)} recurrent sections, expected 1")
    return parts[0]


def cell_rows(part, cell=-1):
    """{"r" | "s" | "p": {layer: float32 [row floats]}} of one cell of a recurrent section (views
    into its bytes; the last cell by default, the one a single-sequence save holds)."""
    n = part.n_cells
    if not -n <= cell < n:
        raise IndexError(f"cell {cell} of a section with {n} cells")
    cell %= n
    out = {}
    for key in ROWS:
        out[key] = {}
        for layer, rows in (getattr(part, key, None) or {}).items():
            if rows.ggml_type not in (kvstate.GGML_TYPE_F32, None):
                raise ValueError(f"{part.name} layer {layer} {key.upper()} rows are ggml type {rows.ggml_type}, not F32")
            if len(rows.data) != n * rows.row_bytes or rows.row_bytes % 4:
                raise ValueError(f"{part.name} layer {layer} {key.upper()}: {len(rows.data)} bytes for {n} rows "
                                 f"of {rows.row_bytes}")
            size = rows.row_bytes // 4
            out[key][int(layer)] = np.frombuffer(rows.data, dtype="<f4", count=size, offset=cell * rows.row_bytes)
    return out


def section(template, position, rows):
    """A one-cell recurrent section at `position` with the template's layout and these float rows
    ({"r" | "s" | "p": {layer: array}}; a missing kind or layer is zero)."""
    def build(key, source):
        out = {}
        for layer, model in source.items():
            values = rows.get(key, {}).get(layer)
            if values is None:
                data = bytes(model.row_bytes)
            else:
                values = np.asarray(values, dtype="<f4").reshape(-1)
                if values.nbytes != model.row_bytes:
                    raise ValueError(f"layer {layer} {key.upper()}: {values.nbytes} bytes for a row of "
                                     f"{model.row_bytes}")
                if not np.isfinite(values).all():
                    raise ValueError(f"layer {layer} {key.upper()}: non-finite values")
                data = values.tobytes()
            out[layer] = kvstate.OpaqueRows(model.ggml_type, model.row_bytes, data)
        return out
    return kvstate.RecurrentState(template.name, template.layers, np.asarray([position], dtype=np.int32), ((),),
                                  template.n_layer, build("r", template.r), build("s", template.s),
                                  build("p", getattr(template, "p", None) or {}))


def replace_recurrent(state_file, part):
    """state_file with its recurrent section replaced by `part` (token words and KV sections kept)."""
    current = recurrent_part(state_file)
    Layout.of(part).require(Layout.of(current), "the new recurrent section")
    parts = [part if candidate is current else candidate for candidate in state_file.state.parts]
    return kvstate.StateFile(state_file.token_words,
                             kvstate.SequenceState(state_file.state.spec, state_file.state.flags, parts))


class MeanAccumulator:
    """Running float64 sums of native recurrent cells, per group label (build_mean).

        accumulator = MeanAccumulator()
        for state, group, depth in samples:          # native saves of training documents
            accumulator.add(state, group=group, depth=depth)
        accumulator.result(model={...}).save(path)

    Besides the sums it keeps, per group and layer, the sum of squared row norms, so stats() can
    say how much of the rows' energy the mean explains (1 - E|x - mean|^2 / E|x|^2)."""

    def __init__(self):
        self.layout = None
        self.groups = {}

    def add(self, state, *, group=ALL, depth=None):
        """Add every cell of a state's recurrent section (a StateFile or RecurrentState)."""
        part = recurrent_part(state)
        layout = Layout.of(part)
        if self.layout is None:
            self.layout = layout
        layout.require(self.layout, "a calibration state")
        sums = self.groups.get(group)
        if sums is None:
            sums = self.groups[group] = {
                "count": 0, "depth": [None, None],
                "sum": {key: {layer: np.zeros(size // 4, dtype=np.float64) for layer, size in self._sizes(key)}
                        for key in ROWS},
                "sq": {key: {layer: 0.0 for layer, _ in self._sizes(key)} for key in ROWS}}
        for cell in range(part.n_cells):
            rows = cell_rows(part, cell)
            for key in ROWS:
                for layer, values in rows[key].items():
                    sums["sum"][key][layer] += values
                    sums["sq"][key][layer] += float(np.dot(values.astype(np.float64), values))
            sums["count"] += 1
        if depth is not None:
            low, high = sums["depth"]
            sums["depth"] = [depth if low is None else min(low, depth), depth if high is None else max(high, depth)]

    def _sizes(self, key):
        return row_sizes(self.layout, key)

    def _pooled(self, group):
        """The sums of one group; ALL merges every group (samples added without a label included)."""
        if group != ALL:
            if group not in self.groups:
                raise KeyError(f"no calibration samples in group {group!r}")
            return self.groups[group]
        if not self.groups:
            raise ValueError("no calibration samples")
        if list(self.groups) == [ALL]:
            return self.groups[ALL]
        pooled = {"count": 0, "depth": [None, None],
                  "sum": {key: {layer: 0.0 for layer, _ in self._sizes(key)} for key in ROWS},
                  "sq": {key: {layer: 0.0 for layer, _ in self._sizes(key)} for key in ROWS}}
        for sums in self.groups.values():
            pooled["count"] += sums["count"]
            for key in ROWS:
                for layer in sums["sum"][key]:
                    pooled["sum"][key][layer] = pooled["sum"][key][layer] + sums["sum"][key][layer]
                    pooled["sq"][key][layer] += sums["sq"][key][layer]
            depths = [d for d in sums["depth"] + pooled["depth"] if d is not None]
            pooled["depth"] = [min(depths), max(depths)] if depths else [None, None]
        return pooled

    def mean(self, group=ALL):
        """{"r" | "s" | "p": {layer: float32 mean row}} of a group (ALL: every sample)."""
        sums = self._pooled(group)
        return {key: {layer: (np.asarray(total) / sums["count"]).astype(np.float32)
                      for layer, total in sums["sum"][key].items()} for key in ROWS}

    def stats(self, group=ALL):
        """Per kind: samples, depth range, and per layer the mean energy E|x|^2 and the share of it
        the mean explains on these samples (1 - E|x - mean|^2 / E|x|^2 = |mean|^2 / E|x|^2)."""
        sums = self._pooled(group)
        n = sums["count"]
        out = {"count": n, "depth": sums["depth"]}
        for key in ROWS:
            layers = {}
            for layer, total in sums["sum"][key].items():
                energy = sums["sq"][key][layer] / n
                mean = np.asarray(total) / n
                layers[layer] = {"energy": energy,
                                 "explained": float(np.dot(mean, mean) / energy) if energy > 0 else None}
            if layers:
                total_energy = sum(item["energy"] for item in layers.values())
                explained = sum(item["energy"] * (item["explained"] or 0.0) for item in layers.values())
                out[key] = {"energy": total_energy,
                            "explained": explained / total_energy if total_energy > 0 else None,
                            "layers": layers}
        return out

    def result(self, *, model=None, groups=None):
        """A RecurrentMean of the accumulated samples: ALL plus each named group (default: every
        group label added, when there is more than ALL)."""
        if self.layout is None:
            raise ValueError("no calibration samples")
        labels = sorted(self.groups) if groups is None else list(groups)
        names = [ALL] + [label for label in labels if label != ALL]
        means = {name: self.mean(name) for name in names}
        info = {}
        for name in names:
            sums = self._pooled(name)
            info[name] = {"count": sums["count"], "depth": sums["depth"]}
        return RecurrentMean(self.layout, means, info, model=model)


class RecurrentMean:
    """Per-layer mean recurrent rows of one model, by group (ALL and optional labels).

    means holds the rows of the groups loaded; groups (sample counts and depth range) may list
    more, the groups of the file they were loaded from. Saved as an npz: `meta` (JSON: format,
    layout, model identity, the groups it holds rows for) and one array per (group, kind, layer),
    float16 where the values fit (S rows of a GDN model are small; anything beyond the float16
    range is kept as float32). A file may hold any groups, e.g. only chat and tools:

        RecurrentMean.load(path, groups=["chat", "tools"]).save(smaller)"""

    def __init__(self, layout, means, groups, *, model=None):
        self.layout, self.means, self.groups, self.model = layout, means, groups, dict(model or {})
        if not means or set(means) - set(groups):
            raise ValueError("a recurrent mean needs rows of at least one group, and only of its groups")
        for name, rows in means.items():
            for key in ROWS:
                sizes = dict(row_sizes(layout, key))
                if set(rows.get(key, {})) != set(sizes):
                    raise ValueError(f"mean group {name!r} {key.upper()} rows cover other layers than the layout")
                for layer, values in rows.get(key, {}).items():
                    if np.asarray(values).size * 4 != sizes[layer]:
                        raise ValueError(f"mean group {name!r} layer {layer} {key.upper()}: wrong row size")

    def rows(self, group=ALL):
        if group not in self.means:
            raise KeyError(f"the mean has no group {group!r} (groups: {sorted(self.means)})")
        return self.means[group]

    def check(self, part):
        """Refuse a template (RecurrentState or StateFile) of another layout."""
        Layout.of(recurrent_part(part)).require(self.layout, "the recurrent mean")

    def save(self, path, *, dtype="float16"):
        """Write the npz (a new file); dtype float16 or float32 for the stored rows."""
        path = Path(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        if dtype not in ("float16", "float32"):
            raise ValueError("dtype must be float16 or float32")
        arrays = {}
        for name, rows in self.means.items():
            for key in ROWS:
                for layer, values in rows[key].items():
                    values = np.asarray(values, dtype=np.float32)
                    small = dtype == "float16" and (not values.size or float(np.abs(values).max()) < F16_MAX)
                    arrays[f"{name}/{key}/{layer}"] = values.astype(np.float16 if small else np.float32)
        meta = {"format_version": FORMAT_VERSION, "layout": self.layout.to_json(), "model": self.model,
                "groups": {name: self.groups[name] for name in self.means}, "dtype": dtype}
        with open(path, "xb") as stream:
            np.savez(stream, meta=np.asarray(json.dumps(meta, sort_keys=True)), **arrays)
        return path

    @staticmethod
    def _meta(data, path):
        meta = json.loads(str(data["meta"]))
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported recurrent mean format {meta.get('format_version')}")
        return meta

    @classmethod
    def stored_groups(cls, path):
        """The groups a saved mean holds rows for (reads only its meta)."""
        with open(path, "rb") as stream, np.load(stream, allow_pickle=False) as data:   # closed if not a zip
            return sorted(cls._meta(data, path)["groups"])

    @classmethod
    def load(cls, path, *, groups=None):
        """Read a saved mean; groups: the groups to read (default every one; a switch needs one)."""
        with open(path, "rb") as stream, np.load(stream, allow_pickle=False) as data:
            meta = cls._meta(data, path)
            layout = Layout.from_json(meta["layout"])
            missing = sorted(set(groups or ()) - set(meta["groups"]))
            if missing:
                raise KeyError(f"{path} has no groups {missing} (groups: {sorted(meta['groups'])})")
            stored, means = set(data.files), {}
            for name in meta["groups"] if groups is None else groups:
                means[name] = {key: {} for key in ROWS}
                for key, layers in (("r", layout.layers), ("s", layout.layers), ("p", layout.p_layers)):
                    for layer in layers:
                        array = f"{name}/{key}/{layer}"
                        if array not in stored:
                            raise KeyError(f"{path}: group {name!r} is listed but has no {array} array")
                        means[name][key][int(layer)] = np.asarray(data[array], dtype=np.float32)
        return cls(layout, means, meta["groups"], model=meta.get("model"))


def match_group(available, tools, *, tools_group=True):
    """The mean group of a conversation (mean-match): "tools" when it declares tools, else "chat";
    ALL when the mean lacks that group; None when it has neither (then use zero).

    tools_group=False: a conversation with tools takes ALL even from a mean with a "tools" group
    (RESULTS.md section 7: a tools mean only when that group was calibrated on the deployment's
    own agent prompts and tools; the pooled mean was level with zero or better everywhere, and a
    tools mean from other agents' prompts was not measured)."""
    wanted = ("tools" if tools_group else ALL) if tools else "chat"
    for group in (wanted, ALL):
        if group in available:
            return group
    return None


def translate_options(strategy="zero", mean=None, group=None, *, tools=None, means=None):
    """translate.translate's recurrent options for an evaluation's --recurrent, --recurrent-mean and
    --recurrent-group (pair_eval.py, score_translate.py): the strategy and, for one that reads the
    mean (the npz path), its group (default: match_group of the conversation's tools as the engine
    picks it, the pooled mean for tools) and that group's rows (means: {group: RecurrentMean}, a
    cache kept across documents)."""
    rule = parse_strategy(strategy)
    if "mean" not in rule.needs:
        return {"recurrent": rule.name}
    if mean is None:
        raise ValueError(f"the recurrent strategy {rule.name} needs a recurrent mean file")
    available = RecurrentMean.stored_groups(mean)
    group = group or match_group(available, tools, tools_group=False)
    if group not in available:
        raise ValueError(f"{mean} has no group {group!r} (groups: {', '.join(available)})")
    means = {} if means is None else means
    if group not in means:
        means[group] = RecurrentMean.load(mean, groups=[group])
    return {"recurrent": rule.name, "recurrent_mean": means[group], "recurrent_group": group}


def build_mean(samples, *, model=None, groups=None):
    """A RecurrentMean from native states: samples yields a state (StateFile or RecurrentState),
    or (state, group) or (state, group, depth) tuples."""
    accumulator = MeanAccumulator()
    for item in samples:
        item = item if isinstance(item, tuple) else (item,)
        accumulator.add(item[0], group=item[1] if len(item) > 1 else ALL, depth=item[2] if len(item) > 2 else None)
    return accumulator.result(model=model, groups=groups)


def initial_state(template, position, strategy="zero", *, mean=None, prefix=None, group=ALL):
    """The recurrent section to restore at `position` (t0 - 1): the template's layout, one cell.

    template: the target's recurrent section (any native save of the target, e.g. translate's
    template) or a StateFile holding one; mean: a RecurrentMean of the target (strategies taking
    rows from the mean); prefix: a native state of the prompt's start (a StateFile or its
    recurrent section; its last cell is used)."""
    template = recurrent_part(template)
    rule = parse_strategy(strategy)
    layout = Layout.of(template)
    sources = {"zero": {key: {} for key in ROWS}}
    if "mean" in rule.needs:
        if mean is None:
            raise ValueError(f"strategy {rule.name} needs a recurrent mean of the target")
        layout.require(mean.layout, "the recurrent mean")
        sources["mean"] = mean.rows(group)
    if "prefix" in rule.needs:
        if prefix is None:
            raise ValueError(f"strategy {rule.name} needs a native state of the prompt's start")
        prefix = recurrent_part(prefix)
        Layout.of(prefix).require(layout, "the prefix state")
        sources["prefix"] = cell_rows(prefix)
    rows = {"r": sources[rule.r]["r"], "p": sources[rule.r]["p"], "s": sources[rule.s]["s"]}
    if rule.s_scale != 1.0:
        rows["s"] = {layer: np.asarray(values, dtype=np.float32) * np.float32(rule.s_scale)
                     for layer, values in rows["s"].items()}
    return section(template, position, rows)


def state_distance(reference, candidate):
    """How far a candidate recurrent cell lies from a reference (e.g. the native cell at t0 - 1):
    per kind, the relative squared error sum|c - x|^2 / sum|x|^2 over all layers, and per layer."""
    a, b = cell_rows(recurrent_part(reference)), cell_rows(recurrent_part(candidate))
    out = {}
    for key in ROWS:
        if not a[key]:
            continue
        layers, error_total, energy_total = {}, 0.0, 0.0
        for layer, x in a[key].items():
            x = x.astype(np.float64)
            diff = b[key][layer].astype(np.float64) - x
            error, energy = float(np.dot(diff, diff)), float(np.dot(x, x))
            layers[layer] = error / energy if energy > 0 else None
            error_total += error
            energy_total += energy
        out[key] = {"relative": error_total / energy_total if energy_total > 0 else None, "layers": layers}
    return out
