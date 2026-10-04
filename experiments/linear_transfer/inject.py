"""Prefix-injection scoring on Muse: patch mapped K/V into the prefix and score the continuation.

For each test document, patch directories (positions.npy + k-/v-<layer>.npy) are built
per condition at the aligned content tokens of the prefix (tokens before the final
message), `kvpark-kv-probe --mode patch-score` runs through collect_v2 (runtime
verification, environment hygiene, manifest, atomic publish), and the continuation
NLL delta, KL(native || condition) and top-1 agreement are summarized with document
bootstrap confidence intervals (shared resamples; paired contrasts against mean_all and
all_kv; long/short patched-prefix strata). Positions overridden by all_kv_tail<N> and
all_kv_system_native are left out of the patch, so Muse recomputes them on the patched
prefix. Groups of documents are scored into OUTPUT.partial and kept there when a later
group fails; `run --resume` validates them and scores the rest.
"""

import os

if __name__ == "__main__":
    for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, "8")

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import attention_eval as ae
from align import load_alignment, read_records
from captures import CaptureSet
from collect import sha256_file


SANITY = "sanity_native_all"
SANITY_THRESHOLD = 1e-6  # per-document mean KL(native || sanity)
BOOTSTRAP_RESAMPLES = 2000
KINDS = ("k", "v")
SOURCES = ("native", "mapped", "mean")


# -- conditions ----------------------------------------------------------------

@dataclass
class Condition:
    name: str
    sources: dict = field(default_factory=dict)   # {(kind, layer): "native" | "mapped" | "mean"}
    override: str = None                          # patched positions forced native: None, "tail", "system"
    omitted: list = field(default_factory=list)   # native arrays left out as no-ops

    def describe(self):
        counts = {source: sum(1 for value in self.sources.values() if value == source) for source in SOURCES}
        return {"name": self.name, "arrays": counts, "override": self.override,
                "mapped": sorted(f"{kind}-{layer}" for (kind, layer), source in self.sources.items()
                                 if source != "native"),
                "omitted_noop_native": [f"{kind}-{layer}" for kind, layer in self.omitted]}


def define_conditions(layer_types, *, focus_layer=27, tail=64, names=None, prune_noop_native=True):
    """Conditions in canonical order (native first). layer_types: Muse per-layer attention types.

    native and sanity_native_all are always included: every run must prove the patch is a no-op.
    """
    attention = [layer for layer, kind in enumerate(layer_types) if kind in ("full", "sliding")]
    full = [layer for layer in attention if layer_types[layer] == "full"]
    if focus_layer not in attention:
        raise ValueError(f"focus layer {focus_layer} is not an attention layer")
    if tail <= 0:
        raise ValueError("tail must be positive")

    def everything(source):
        return {(kind, layer): source for layer in attention for kind in KINDS}

    def focus(kinds):
        sources = everything("native")
        sources.update({(kind, focus_layer): "mapped" for kind in kinds})
        return sources

    full_mapped = everything("native")
    full_mapped.update({(kind, layer): "mapped" for layer in full for kind in KINDS})
    catalog = [
        Condition("native"),
        Condition(SANITY, everything("native")),
        Condition(f"L{focus_layer}_v", focus(["v"])),
        Condition(f"L{focus_layer}_kv", focus(KINDS)),
        Condition("full_kv", full_mapped),
        Condition("all_kv", everything("mapped")),
        Condition(f"all_kv_tail{tail}", everything("mapped"), override="tail"),
        Condition("all_kv_system_native", everything("mapped"), override="system"),
        Condition("mean_all", everything("mean")),
    ]
    if names is not None:
        known = [condition.name for condition in catalog]
        unknown = set(names) - set(known)
        if unknown or len(set(names)) != len(names):
            raise ValueError(f"unknown or repeated conditions {sorted(unknown)}; known: {known}")
        catalog = [condition for condition in catalog if condition.name in ("native", SANITY) or condition.name in names]
    if prune_noop_native:
        for condition in catalog:
            if condition.name != SANITY:
                _prune(condition)
    return catalog


def _prune(condition):
    """Drop native arrays at or before the first non-native layer: the runtime already computes them.

    K/V of layer L depend only on layers < L, so writing captured native values there changes
    nothing (sanity_native_all measures that no-op directly).
    """
    changed = [layer for (_, layer), source in condition.sources.items() if source != "native"]
    if not changed:
        return
    first = min(changed)
    condition.omitted = sorted(pair for pair, source in condition.sources.items()
                               if source == "native" and pair[1] <= first)
    for pair in condition.omitted:
        del condition.sources[pair]


# -- patch values --------------------------------------------------------------

class PlanValues:
    """Native, mapped and mean K/V rows for one document plan's patched positions (cached)."""

    def __init__(self, plan, tgt_set, src_set, mapper, target_shape):
        self.plan, self.tgt_set, self.src_set, self.mapper = plan, tgt_set, src_set, mapper
        self.target_shape = tuple(target_shape)
        self.rows = plan.tgt_start + plan.positions
        self._native, self._mapped, self._source = {}, {}, {}

    def native(self, kind, layer):
        if (kind, layer) not in self._native:
            self._native[kind, layer] = self.tgt_set.gather(kind, layer, self.rows)
        return self._native[kind, layer]

    def mapped(self, kind, layer):
        if (kind, layer) not in self._mapped:
            self._mapped[kind, layer] = ae.mapped_rows(self.mapper, kind, layer, self.plan, self.src_set,
                                                       self.target_shape, self._source)
        return self._mapped[kind, layer]

    def mean(self, kind, layer):
        return np.broadcast_to(self.mapper.floor(kind, layer, self.target_shape),
                               (len(self.rows), *self.target_shape)).astype(np.float32)

    def get(self, source, kind, layer):
        return getattr(self, source)(kind, layer)


def override_mask(condition, plan, tail):
    if condition.override is None:
        return None
    if condition.override == "tail":
        return plan.positions >= plan.prefix_tokens - tail
    if condition.override == "system":
        return plan.system.copy()
    raise ValueError(f"unknown override {condition.override}")


def condition_rows(condition, plan, tail):
    """Indices into plan.positions that the condition's patch writes.

    Overridden positions (tail, system) are left out of the patch, so Muse recomputes
    them on top of the patched prefix like template and unmatched tokens. When the
    override covers every patched position nothing is mapped and the rows are written
    with native values: with no mapped row in the document, captured native K/V are
    exactly what Muse recomputes (sanity_native_all shows such a write is a no-op).
    """
    mask = override_mask(condition, plan, tail)
    if mask is None or mask.all():
        return np.arange(len(plan.positions))
    return np.flatnonzero(~mask)


def mapped_row_count(condition, plan, tail):
    """Patched positions that receive non-native values in this condition."""
    if all(source == "native" for source in condition.sources.values()):
        return 0
    mask = override_mask(condition, plan, tail)
    return len(plan.positions) if mask is None else int(np.sum(~mask))


def condition_arrays(condition, plan, values, tail):
    """{(kind, layer): float32 [R, heads, dim]} at plan.positions[condition_rows(...)]."""
    mask = override_mask(condition, plan, tail)
    rows = condition_rows(condition, plan, tail)
    arrays = {}
    for (kind, layer), source in sorted(condition.sources.items(), key=lambda item: (item[0][1], item[0][0])):
        if mask is not None and mask.all():
            source = "native"
        arrays[kind, layer] = np.array(values.get(source, kind, layer)[rows], dtype=np.float32)
    return arrays


def _write_npy(path, array):
    with path.open("xb") as stream:
        np.lib.format.write_array(stream, np.ascontiguousarray(array), version=(1, 0), allow_pickle=False)


def write_patch_dir(path, positions, arrays):
    """Create a new kv_probe patch directory; never replaces an existing path."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"patch directory already exists: {path}")
    positions = np.asarray(positions)
    if (positions.ndim != 1 or not len(positions) or positions.dtype.kind not in "iu"
            or positions.min() < 0 or positions.max() > np.iinfo(np.int32).max
            or np.any(np.diff(positions) <= 0)):
        raise ValueError("patch positions must be a nonempty strictly increasing nonnegative int32 range")
    if not arrays:
        raise ValueError("a patch directory needs at least one array")
    for (kind, layer), array in arrays.items():
        array = np.asarray(array)
        if (kind not in KINDS or type(layer) is not int or layer < 0 or array.dtype != np.float32
                or array.ndim != 3 or len(array) != len(positions) or not np.isfinite(array).all()):
            raise ValueError(f"patch array {kind}-{layer} must be finite float32 [P, heads, dim]")
    staging = Path(tempfile.mkdtemp(prefix=".kvpark-patch-", dir=path.parent))
    try:
        _write_npy(staging / "positions.npy", positions.astype("<i4"))
        for (kind, layer), array in arrays.items():
            _write_npy(staging / f"{kind}-{layer}.npy", np.asarray(array, dtype="<f4"))
        os.rename(staging, path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return path


def token_sha256(token_ids):
    return hashlib.sha256(np.asarray(token_ids, dtype="<i4").tobytes()).hexdigest()


def expected_applied(conditions, plan, tail):
    """{condition: {"<kind>-<layer>": rows}} that kv_probe must report for one document."""
    return {condition.name: {f"{kind}-{layer}": len(condition_rows(condition, plan, tail))
                             for kind, layer in condition.sources}
            for condition in conditions}


def build_document_patches(directory, plan, record, conditions, values, tgt_set, tail):
    """Write every condition's patch dir for one document; return (patch-score record, expected applied)."""
    directory = Path(directory)
    directory.mkdir()
    if not conditions or conditions[0].name != "native":
        raise ValueError("the first condition must be native")
    entries = [{"name": "native", "patch_dir": None}]
    for condition in conditions[1:]:
        if not condition.sources:
            raise ValueError(f"condition {condition.name} patches nothing")
        path = write_patch_dir(directory / condition.name, plan.positions[condition_rows(condition, plan, tail)],
                               condition_arrays(condition, plan, values, tail))
        entries.append({"name": condition.name, "patch_dir": str(path.resolve())})
    expected = expected_applied(conditions, plan, tail)
    token_ids = tgt_set.token_id[plan.tgt_start:plan.tgt_start + plan.n_tokens]
    patch_record = {"document_id": plan.document_id, "text": record["text"],
                    "expected_token_sha256": token_sha256(token_ids),
                    "score_ranges": plan.score_ranges, "conditions": entries}
    for name in ("template_sha256", "bos_token"):
        if name in record:
            patch_record[name] = record[name]
    return patch_record, expected


# -- running -------------------------------------------------------------------

def score_group(probe, model, model_id, docs, output, *, ctx, batch, threads, max_scored,
                hash_cache=None, skip_memory_check=False):
    """patch-score through collect_v2 (runtime checks, env hygiene, manifest, atomic publish)."""
    import collect_v2
    argv = ["--probe", str(probe), "--mode", "patch-score", "--model", str(model), "--model-id", model_id,
            "--docs", str(docs), "--output", str(output), "--ctx", str(ctx), "--batch", str(batch),
            "--threads", str(threads), "--max-scored", str(max_scored)]
    if hash_cache is not None:
        argv += ["--hash-cache", str(hash_cache)]
    if skip_memory_check:
        argv.append("--skip-memory-check")
    collect_v2.main(argv)


def read_results(directory):
    directory = Path(directory)
    results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    if (not isinstance(results, dict) or results.get("format_version") != 2
            or results.get("mode") != "patch-score" or not isinstance(results.get("documents"), list)):
        raise ValueError(f"unsupported patch-score results in {directory}")
    return results


def check_applied(results, expected):
    """Exactly the expected documents and conditions, each written array applied with P rows."""
    reported = [document["document_id"] for document in results["documents"]]
    if reported != list(expected):
        raise ValueError(f"patch-score results cover documents {reported}, expected {list(expected)}")
    for document in results["documents"]:
        wanted = expected[document["document_id"]]
        names = [condition["name"] for condition in document["conditions"]]
        if names != list(wanted):
            raise ValueError(f"patch-score results of {document['document_id']} have conditions {names}, "
                             f"expected {list(wanted)}")
        for condition in document["conditions"]:
            if condition.get("applied_rows", {}) != wanted[condition["name"]]:
                raise ValueError(f"applied rows differ for {document['document_id']} {condition['name']}: "
                                 f"{condition.get('applied_rows')} != {wanted[condition['name']]}")


def check_sanity(directory, name=SANITY, threshold=SANITY_THRESHOLD):
    """Raise unless every document has the sanity condition with mean KL(native || sanity) below threshold.

    Returns (worst document mean KL, worst single-token KL).
    """
    directory = Path(directory)
    means, tokens = [], []
    for document in read_results(directory)["documents"]:
        if name not in [condition["name"] for condition in document["conditions"]]:
            raise ValueError(f"{document['document_id']} was scored without {name}; the no-op check is required")
        scores = np.load(directory / "scores" / document["document_id"] / f"{name}.npy", allow_pickle=False)
        mean = float(np.mean(scores[:, 1], dtype=np.float64))
        means.append(mean)
        tokens.append(float(scores[:, 1].max()))
        if not mean < threshold:
            raise ValueError(f"{name} changed predictions of {document['document_id']}: mean KL "
                             f"{mean:.3g} >= {threshold:g}; patching is not a no-op, aborting")
    if not means:
        raise ValueError(f"no documents in {directory}")
    return max(means), max(tokens)


# -- summary -------------------------------------------------------------------

def _bootstrap(sums, counts, rng_seed, resamples):
    sums, counts = np.asarray(sums, dtype=np.float64), np.asarray(counts, dtype=np.float64)
    mean = float(sums.sum() / counts.sum())
    doc_mean = float(np.mean(sums / counts))
    if len(sums) < 2:
        return {"mean": mean, "doc_mean": doc_mean, "ci95": None}
    rng = np.random.default_rng(rng_seed)
    index = rng.integers(0, len(sums), size=(resamples, len(sums)))
    stats = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    low, high = np.percentile(stats, [2.5, 97.5])
    return {"mean": mean, "doc_mean": doc_mean, "ci95": [float(low), float(high)]}


def load_scores(directories):
    """{document_id: {"n_scored", "conditions": {name: float64 [S, 3]}, "applied": {...}}} across runs."""
    documents = {}
    for directory in directories:
        directory = Path(directory)
        for document in read_results(directory)["documents"]:
            document_id = document["document_id"]
            if document_id in documents:
                raise ValueError(f"document {document_id} appears in two patch-score runs")
            names = [condition["name"] for condition in document["conditions"]]
            if not names or names[0] != "native":
                raise ValueError(f"first condition of {document_id} is not native")
            scores, applied = {}, {}
            for condition in document["conditions"]:
                path = directory / "scores" / document_id / f"{condition['name']}.npy"
                array = np.load(path, allow_pickle=False)
                if (array.dtype.str != "<f4" or array.shape != (document["n_scored"], 3)
                        or not np.isfinite(array).all()):
                    raise ValueError(f"score file {path} is malformed")
                if not np.isclose(float(np.mean(array[:, 0], dtype=np.float64)), condition["nll_mean"],
                                  rtol=1e-4, atol=1e-5):
                    raise ValueError(f"score file {path} disagrees with results.json")
                scores[condition["name"]] = array.astype(np.float64)
                applied[condition["name"]] = condition.get("applied_rows", {})
            native = scores["native"]
            if np.any(native[:, 1] != 0) or np.any(native[:, 2] != 1):
                raise ValueError(f"native scores of {document_id} are not self-consistent")
            documents[document_id] = {"n_scored": document["n_scored"], "conditions": scores,
                                      "applied": applied, "n_eligible": document.get("n_eligible"),
                                      "scored_truncated": document.get("scored_truncated")}
    if not documents:
        raise ValueError("no scored documents")
    return documents


CONTRAST_BASELINES = ("mean_all", "all_kv")


def _condition_order(documents):
    order = []
    for document in documents.values():
        for name in document["conditions"]:
            if name not in order:
                order.append(name)
    return order


def _condition_summaries(documents, order, seed, resamples):
    """Token-pooled means with document bootstrap CIs. Every condition and metric uses the
    same seed, so conditions scored on the same documents share their resample indices."""
    conditions = []
    for name in order:
        present = [document for document in documents.values() if name in document["conditions"]]
        counts = [document["n_scored"] for document in present]
        sums = {"nll_delta": [], "kl": [], "top1_agree": [], "nll": []}
        for document in present:
            scores, native = document["conditions"][name], document["conditions"]["native"]
            sums["nll"].append(scores[:, 0].sum())
            sums["nll_delta"].append((scores[:, 0] - native[:, 0]).sum())
            sums["kl"].append(scores[:, 1].sum())
            sums["top1_agree"].append(scores[:, 2].sum())
        summary = {"name": name, "documents": len(present), "tokens": int(sum(counts))}
        for metric, values in sums.items():
            summary[metric] = _bootstrap(values, counts, seed, resamples)
        summary["kl"]["max_token"] = float(max(document["conditions"][name][:, 1].max() for document in present))
        conditions.append(summary)
    return conditions


def _contrasts(documents, order, seed, resamples):
    """Paired per-document differences (condition minus baseline) with document bootstrap CIs."""
    contrasts = []
    for baseline in CONTRAST_BASELINES:
        if baseline not in order:
            continue
        for name in order:
            if name in ("native", SANITY, baseline) or (baseline != "mean_all" and name == "mean_all"):
                continue
            present = [document for document in documents.values()
                       if name in document["conditions"] and baseline in document["conditions"]]
            counts = [document["n_scored"] for document in present]
            item = {"name": name, "baseline": baseline, "documents": len(present), "tokens": int(sum(counts))}
            for metric, column in (("nll", 0), ("kl", 1), ("top1_agree", 2)):
                item[metric] = _bootstrap([np.sum(document["conditions"][name][:, column]
                                                  - document["conditions"][baseline][:, column])
                                           for document in present], counts, seed, resamples)
            contrasts.append(item)
    return contrasts


def summarize(directories, *, seed=0, resamples=BOOTSTRAP_RESAMPLES, sanity=SANITY,
              sanity_threshold=SANITY_THRESHOLD, groups=None):
    """groups: optional {name: [document ids]} summarized separately (e.g. prefix-length strata)."""
    documents = load_scores(directories)
    order = _condition_order(documents)
    conditions = _condition_summaries(documents, order, seed, resamples)
    per_document = []
    for document_id, document in documents.items():
        native = document["conditions"]["native"]
        per_document.append({"document_id": document_id, "n_scored": document["n_scored"],
                             "n_eligible": document["n_eligible"], "scored_truncated": document["scored_truncated"],
                             "conditions": {
            name: {"nll_delta_mean": float(np.mean(scores[:, 0] - native[:, 0])),
                   "kl_mean": float(np.mean(scores[:, 1])), "top1_agree": float(np.mean(scores[:, 2])),
                   "applied_rows": document["applied"][name]}
            for name, scores in document["conditions"].items()}})
    sanity_summary = None
    if sanity in order:
        worst = next(item for item in conditions if item["name"] == sanity)["kl"]["max_token"]
        worst_mean = max(item["conditions"][sanity]["kl_mean"] for item in per_document
                         if sanity in item["conditions"])
        sanity_summary = {"condition": sanity, "max_document_kl_mean": worst_mean, "max_token_kl": worst,
                          "threshold": sanity_threshold, "passed": worst_mean < sanity_threshold}
    strata = {}
    for name, members in (groups or {}).items():
        unknown = sorted(set(members) - set(documents))
        if unknown:
            raise ValueError(f"group {name} names unscored documents {unknown[:3]}")
        chosen = {document_id: documents[document_id] for document_id in documents if document_id in set(members)}
        strata[name] = {"documents": len(chosen), "document_ids": list(chosen),
                        "scored_tokens": int(sum(document["n_scored"] for document in chosen.values())),
                        "conditions": _condition_summaries(chosen, order, seed, resamples) if chosen else [],
                        "contrasts": _contrasts(chosen, order, seed, resamples) if chosen else []}
    return {"format_version": 1, "documents": len(documents),
            "scored_tokens": int(sum(document["n_scored"] for document in documents.values())),
            "truncated_documents": sum(1 for document in documents.values() if document["scored_truncated"]),
            "bootstrap": {"unit": "document", "statistic": "token-pooled mean", "resamples": resamples,
                          "seed": seed, "interval": "percentile 95%",
                          "shared_indices": "one seed for every condition, metric and contrast",
                          "contrasts": "paired per-document sums of condition minus baseline"},
            "sanity": sanity_summary, "conditions": conditions,
            "contrasts": _contrasts(documents, order, seed, resamples), "groups": strata,
            "per_document": per_document}


# -- orchestration -------------------------------------------------------------

def _write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _groups(items, size):
    return [items[start:start + size] for start in range(0, len(items), size)]


def _remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _build_group(patches, group, records, tgt_set, src_set, mapper, conditions, tail):
    """Write a group's patch dirs and return its patch-score records.

    The per-document row caches (native, mapped and gathered source rows, up to ~1 GB)
    and the loaded maps are released before the probe loads the model.
    """
    shape = tgt_set.captures[0].array("k", tgt_set.layers("k")[0]).shape[1:]
    patch_records = []
    try:
        for plan in group:
            values = PlanValues(plan, tgt_set, src_set, mapper, shape)
            record, _ = build_document_patches(patches / plan.document_id, plan, records[plan.document_id],
                                               conditions, values, tgt_set, tail)
            del values
            patch_records.append(record)
    finally:
        mapper.clear()
    return patch_records


def validate_group(output, group, conditions, tail, *, check_fn=None, sanity_threshold=SANITY_THRESHOLD):
    """check_fn, applied rows of every document and condition, then the no-op sanity check."""
    if check_fn is not None:
        check_fn(output)
    check_applied(read_results(output), {plan.document_id: expected_applied(conditions, plan, tail)
                                         for plan in group})
    return check_sanity(output, threshold=sanity_threshold)


def run(plans, tgt_records, tgt_set, src_set, mapper, conditions, work, *, tail, docs_per_run, score_fn,
        check_fn=None, resume=False, sanity_threshold=SANITY_THRESHOLD, log=print):
    """Score document groups into work/out/runs/group-NNN; each group's patches are deleted after scoring.

    score_fn(docs_path, output_dir) must publish a patch-score directory at output_dir;
    check_fn(output_dir) adds checks such as the model hash. Scored groups stay in `work`
    when a later group fails; with resume=True they are validated again and kept, and
    only missing groups are scored. Returns the run directories.
    """
    work = Path(work)
    runs_dir, inputs, patch_root = work / "out" / "runs", work / "inputs", work / "patches"
    for directory in (runs_dir, inputs, patch_root):
        directory.mkdir(parents=True, exist_ok=resume)
    if resume:
        # Leftovers of an interrupted attempt: collect_v2 staging dirs and unscored patches.
        for stale in [*runs_dir.glob(".kvpark-*"), *patch_root.iterdir()]:
            _remove(stale)
    records = {record["document_id"]: record for record in tgt_records}
    runs = []
    for number, group in enumerate(_groups(plans, docs_per_run)):
        name = f"group-{number:03d}"
        output = runs_dir / name
        if output.exists() or output.is_symlink():
            if not resume:
                raise FileExistsError(f"patch-score output already exists: {output}")
            worst = validate_group(output, group, conditions, tail, check_fn=check_fn,
                                   sanity_threshold=sanity_threshold)
            runs.append(output)
            log(json.dumps({"group": number, "documents": len(group), "resumed": True,
                            "sanity_worst_kl": worst[0]}))
            continue
        patches = patch_root / name
        patches.mkdir()
        try:
            patch_records = _build_group(patches, group, records, tgt_set, src_set, mapper, conditions, tail)
            docs_path = inputs / f"{name}.jsonl"
            if resume and docs_path.exists():
                docs_path.unlink()
            with docs_path.open("x", encoding="utf-8") as stream:
                for record in patch_records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            del patch_records
            score_fn(docs_path, output)
        finally:
            shutil.rmtree(patches)
        worst = validate_group(output, group, conditions, tail, check_fn=check_fn,
                               sanity_threshold=sanity_threshold)
        runs.append(output)
        log(json.dumps({"group": number, "documents": len(group), "resumed": False, "sanity_worst_kl": worst[0]}))
    return runs


def _capture_model_sha(capture_set):
    found = {(capture.manifest or {}).get("model_sha256") for capture in capture_set.captures}
    return found.pop() if len(found) == 1 else None


def check_inputs(conditions, tgt_set, src_set, mapper):
    if mapper.target_architecture not in (None, ae.ARCHITECTURE) or mapper.source_architecture not in (None, "qwen35"):
        raise ValueError(f"mapper maps {mapper.source_architecture} to {mapper.target_architecture}, "
                         f"not qwen35 to {ae.ARCHITECTURE}")
    needed = {pair for condition in conditions for pair, source in condition.sources.items()}
    mapped = {pair for condition in conditions for pair, source in condition.sources.items() if source != "native"}
    for kind, layer in needed:
        if layer not in tgt_set.layers(kind):
            raise ValueError(f"target capture lacks {kind}-{layer}")
        shape = tuple(tgt_set.captures[0].array(kind, layer).shape[1:])
        if kind in mapper.target_shape and mapper.target_shape[kind] != shape:
            raise ValueError(f"mapper {kind} head shape {mapper.target_shape[kind]} differs from the capture's {shape}")
    for kind, layer in mapped:
        source_key, source_layers = mapper.source_layers(kind, layer)
        for source_layer in source_layers:
            if source_layer not in src_set.layers(source_key):
                raise ValueError(f"source capture lacks {source_key}-{source_layer} for mapper {kind}-{layer}")


def select_documents(plans, skipped, path):
    """Plans for the document ids listed in `path` (one per line), in file order."""
    wanted = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not wanted or len(set(wanted)) != len(wanted):
        raise ValueError(f"{path} must list distinct document ids, one per line")
    by_id = {plan.document_id: plan for plan in plans}
    reasons = {item["document_id"]: item["reason"] for item in skipped}
    for document_id in wanted:
        if document_id not in by_id:
            reason = f" ({reasons[document_id]})" if document_id in reasons else ""
            raise ValueError(f"--documents names {document_id}, which is not a planned document of the "
                             f"selected splits{reason}")
    return [by_id[document_id] for document_id in wanted]


def publish_directory(source, destination):
    import collect_v2
    collect_v2.publish_directory(source, destination)


def _capture_identity(capture_set):
    return [{"path": str(capture.path.resolve()), "report_sha256": sha256_file(capture.path / "report.json")}
            for capture in capture_set.captures]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="build patches, score them with kv_probe, summarize")
    runner.add_argument("--probe", required=True, type=Path, help="kvpark-kv-probe binary")
    runner.add_argument("--model", required=True, type=Path, help="Muse GGUF")
    runner.add_argument("--model-id", required=True)
    runner.add_argument("--tgt-docs", required=True, type=Path, help="Muse rendered JSONL")
    runner.add_argument("--tgt-capture", required=True, type=Path, action="append",
                        help="Muse capture with k and v on every attention layer")
    runner.add_argument("--src-docs", required=True, type=Path, help="Qwen rendered JSONL")
    runner.add_argument("--src-capture", required=True, type=Path, action="append")
    runner.add_argument("--alignment", required=True, type=Path)
    runner.add_argument("--mapper", required=True, type=Path, help="fit_layers.py mapper directory")
    runner.add_argument("--output", required=True, type=Path,
                        help="new directory; never overwritten. Work in progress lives in OUTPUT.partial")
    runner.add_argument("--resume", action="store_true",
                        help="continue OUTPUT.partial with identical settings, keeping its validated groups")
    runner.add_argument("--splits", default="test")
    runner.add_argument("--documents", type=Path,
                        help="file with one document id per line: score exactly these, in this order")
    runner.add_argument("--conditions", help="comma-separated subset (native and sanity_native_all always run)")
    runner.add_argument("--focus-layer", type=int, default=27)
    runner.add_argument("--tail", type=int, default=64)
    runner.add_argument("--limit-docs", type=int, help="score only the first N selected documents")
    runner.add_argument("--docs-per-run", type=int, default=4, help="documents per kv_probe invocation")
    runner.add_argument("--write-noop-native", action="store_true",
                        help="also write native arrays at layers the patch cannot have changed")
    runner.add_argument("--ctx", type=int, default=5120)
    runner.add_argument("--batch", type=int, default=512)
    runner.add_argument("--threads", type=int, default=16)
    runner.add_argument("--max-scored", type=int, default=256)
    runner.add_argument("--long-prefix", type=int, default=256,
                        help="report documents with at least this many patched positions as a separate stratum")
    runner.add_argument("--hash-cache", type=Path)
    runner.add_argument("--skip-memory-check", action="store_true", help="tiny fixtures only")
    runner.add_argument("--seed", type=int, default=0)
    runner.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    summary = commands.add_parser("summarize", help="summarize existing patch-score directories")
    summary.add_argument("runs", type=Path, nargs="+")
    summary.add_argument("--output", required=True, type=Path, help="new injection-report.json")
    summary.add_argument("--seed", type=int, default=0)
    summary.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    if not 100 <= args.bootstrap <= 100000:
        raise ValueError("--bootstrap must be in 100..100000")
    if args.command == "summarize":
        report = summarize(args.runs, seed=args.seed, resamples=args.bootstrap)
        with tempfile.TemporaryDirectory(prefix=".kvpark-inject-", dir=args.output.parent) as temporary:
            pending = Path(temporary) / "report.json"
            _write_json(pending, report)
            os.link(pending, args.output)
        print(json.dumps({"output": str(args.output), "documents": report["documents"]}))
        return 0
    if (not 0 < args.docs_per_run <= 64 or not 0 < args.tail or not 0 < args.max_scored
            or not 0 < args.long_prefix or (args.limit_docs is not None and args.limit_docs <= 0)):
        raise ValueError("--docs-per-run (1..64), --tail, --max-scored, --long-prefix and --limit-docs "
                         "must be positive")
    if not args.output.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {args.output.parent}")
    partial = args.output.with_name(args.output.name + ".partial")
    if not args.resume and (partial.exists() or partial.is_symlink()):
        raise FileExistsError(f"an interrupted run exists at {partial}; pass --resume to continue it "
                              "or remove it")
    if args.resume and not (partial / "config.json").is_file():
        raise ValueError(f"nothing to resume: {partial / 'config.json'} does not exist")
    splits = set(ae._parse_list(args.splits))
    if splits - {"train", "dev", "test"}:
        raise ValueError("--splits must name train, dev or test")
    tgt_set, src_set = CaptureSet(args.tgt_capture), CaptureSet(args.src_capture)
    if tgt_set.architecture != ae.ARCHITECTURE or src_set.architecture != "qwen35":
        raise ValueError("target captures must be Muse and source captures Qwen")
    names = ae._parse_list(args.conditions) if args.conditions else None
    conditions = define_conditions(tgt_set.captures[0].layer_types, focus_layer=args.focus_layer,
                                   tail=args.tail, names=names, prune_noop_native=not args.write_noop_native)
    mapper = ae.Mapper(args.mapper)
    check_inputs(conditions, tgt_set, src_set, mapper)
    tgt_records = read_records(args.tgt_docs)
    plans, skipped = ae.plan_documents(load_alignment(args.alignment), tgt_records, tgt_set,
                                       read_records(args.src_docs), src_set, splits)
    planned = len(plans)
    if args.documents is not None:
        plans = select_documents(plans, skipped, args.documents)
    if args.limit_docs is not None:
        plans = plans[:args.limit_docs]
    if not plans:
        raise ValueError("no documents to score")
    probe, model = args.probe.resolve(), args.model.resolve()
    target_sha = _capture_model_sha(tgt_set)
    settings = {"splits": sorted(splits), "focus_layer": args.focus_layer, "tail": args.tail,
                "limit_docs": args.limit_docs, "docs_per_run": args.docs_per_run,
                "write_noop_native": args.write_noop_native, "ctx": args.ctx, "batch": args.batch,
                "threads": args.threads, "max_scored": args.max_scored, "long_prefix": args.long_prefix,
                "sanity_threshold": SANITY_THRESHOLD, "patched": "aligned content tokens in the prefix",
                "overrides": "tail/system positions are left out of the patch, so Muse recomputes them on the "
                             "patched prefix (native values are written only when nothing else is mapped)",
                "mean_floor": "train target mean; K renormalized per head to RMS 1 like native Muse K",
                "conditions": [condition.describe() for condition in conditions]}
    inputs = {"probe": str(probe), "probe_sha256": sha256_file(probe), "model": str(model),
              "model_id": args.model_id, "target_model_sha256": target_sha,
              "source_model_sha256": _capture_model_sha(src_set),
              "target_captures": _capture_identity(tgt_set), "source_captures": _capture_identity(src_set),
              "tgt_docs_sha256": sha256_file(args.tgt_docs), "src_docs_sha256": sha256_file(args.src_docs),
              "alignment_sha256": sha256_file(args.alignment),
              "mapper_manifest_sha256": sha256_file(args.mapper / "manifest.json"),
              "documents_file_sha256": sha256_file(args.documents) if args.documents is not None else None,
              "documents": [plan.document_id for plan in plans]}
    config = json.loads(json.dumps({"format_version": 1, "settings": settings, "inputs": inputs}))
    if args.resume:
        previous = json.loads((partial / "config.json").read_text(encoding="utf-8"))
        if previous != config:
            differing = sorted(f"{section}.{key}" for section in ("settings", "inputs")
                               for key in set(previous.get(section, {})) | set(config[section])
                               if previous.get(section, {}).get(key) != config[section].get(key))
            raise ValueError(f"--resume needs the settings and inputs of {partial}; differing: {differing}")
    else:
        partial.mkdir()
        _write_json(partial / "config.json", config)

    def score_fn(docs_path, output):
        score_group(probe, model, args.model_id, docs_path, output, ctx=args.ctx, batch=args.batch,
                    threads=args.threads, max_scored=args.max_scored, hash_cache=args.hash_cache,
                    skip_memory_check=args.skip_memory_check)

    def check_fn(output):
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if target_sha is not None and manifest.get("model_sha256") != target_sha:
            raise ValueError(f"{output} scored a model that differs from the model of the target capture")

    try:
        runs = run(plans, tgt_records, tgt_set, src_set, mapper, conditions, partial, tail=args.tail,
                   docs_per_run=args.docs_per_run, score_fn=score_fn, check_fn=check_fn, resume=args.resume)
    except BaseException:
        print(json.dumps({"interrupted": str(partial), "resume": "rerun with the same arguments and --resume"}),
              file=sys.stderr, flush=True)
        raise
    out = partial / "out"
    strata = {"long_prefix": [plan.document_id for plan in plans if len(plan.positions) >= args.long_prefix],
              "short_prefix": [plan.document_id for plan in plans if len(plan.positions) < args.long_prefix]}
    report = summarize(runs, seed=args.seed, resamples=args.bootstrap, groups=strata)
    report["groups_definition"] = (f"long_prefix: >= {args.long_prefix} patched prefix positions; "
                                   f"short_prefix: fewer")
    report["conditions_defined"] = settings["conditions"]
    report["skipped"] = skipped
    report["planned_documents"] = planned
    report["scored_documents"] = [{**plan.summary(), "mapped_rows": {
        condition.name: mapped_row_count(condition, plan, args.tail) for condition in conditions}} for plan in plans]
    for name in ("injection-report.json", "manifest.json"):
        (out / name).unlink(missing_ok=True)
    _write_json(out / "injection-report.json", report)
    _write_json(out / "manifest.json", {
        "format_version": 1, "tool": "inject",
        "settings": {**settings, "seed": args.seed, "bootstrap": args.bootstrap, "resumed": args.resume},
        "inputs": {**inputs, "target_capture_settings": [capture.report.get("settings")
                                                         for capture in tgt_set.captures],
                   "wrapper_sha256": sha256_file(Path(__file__).resolve()),
                   "attention_eval_sha256": sha256_file(Path(ae.__file__).resolve()),
                   "fit_layers_sha256": sha256_file(Path(ae.fit_layers.__file__).resolve())},
        "runs": [run_dir.name for run_dir in runs]})
    publish_directory(out, args.output)
    shutil.rmtree(partial)
    print(json.dumps({"output": str(args.output), "documents": len(plans), "sanity": report["sanity"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
