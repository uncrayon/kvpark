"""Evaluate the Qwen -> Muse cache handoff (design A) on long prefixes, end to end and resumably.

Muse receives a state for positions [0, T0) built from Qwen's cache (handoff.py), natively
decodes the tail [T0, P) and is scored on the first --max-scored tokens of the final
message (P = its first token) against its own full prefill, all with kv_probe seed-score.

Stages, in order; each writes stages/<stage>.json (settings, input digests, output
fingerprints, wall time, metrics; the run_v2 format) and logs to logs/long-<stage>.log:

  prepare      render the long corpus with each model's GGUF template (must be the v2 run's), tokenize
               vocab-only, compute P, the Qwen prefix, T0 per tail W and the conditions -> long/plan.json
  grow-train   (--grow-train) adopt or produce: rendered-train-extra-*.jsonl, captures/train-extra-*
               (k/v on both models), refit/alignment.npz (v2 + extra docs) and refit/mapper (fit_layers on
               v2 train + extra train, dev/test = v2 dev/test): the v3 mapper condition family
  token-table  per-token-id mean Muse K/V over non-aligned v2 train rows -> long/token-table.npz
  qwen-states  save-state on Qwen at the end of every prefix -> long/states/source/group-*
  muse-states  save-state on Muse at every T0 (the native states) -> long/states/target/group-*
  build        handoff.py states per document x condition -> long/handoff/<document_id>/
  score        seed-score on Muse: native reference, native states, built states -> long/scores/group-*
  report       long/report.json + long/report.md (per condition, bucket and contrast; verdict)

Per-group and per-document outputs are published atomically with a key (settings, inputs,
documents); --resume reuses those whose key matches, so an interrupted stage continues where
it stopped, and verifies completed stages like run_v2. qwen-states, muse-states, score and a
grow-train collection load real weights: --device vulkan (default; the GPU probe) or cpu,
a memory guard first (weights + KV + headroom; on this shared-memory machine Vulkan buffers
are system RAM), and every GPU report must show all layers offloaded and the expected graph
split count. Progress and ETAs are one JSON object per line on stdout.

Conditions come from the plan (--conditions, fixed by prepare) plus --extra-conditions: kinds that
need no native Muse state (default mean_table, the information-free floor of the deployable
table_fill), added by build and score without redoing prepare or the state stages. Score stops
at the first group whose native-state checks exceed the gates (--sanity-kl, --sanity-mean-kl,
--sanity-nll); that group stays published, and since the gates are not part of any key, a rerun
with looser gates accepts it without re-scoring. The verdict's headline condition is fixed in
advance (--headline). --device cpu is a fallback for reduced plans only (see --device).
"""

import os
import sys

THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

if __name__ == "__main__":
    _threads = "8"
    for _index, _item in enumerate(sys.argv):
        if _item == "--build-threads" and _index + 1 < len(sys.argv):
            _threads = sys.argv[_index + 1]
        elif _item.startswith("--build-threads="):
            _threads = _item.split("=", 1)[1]
    for _name in THREAD_VARIABLES:
        os.environ.setdefault(_name, _threads)

import argparse
import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time

import numpy as np

from align import ALIGNED, EXCLUDED, STRADDLING, TEMPLATE, UNMATCHED, load_alignment
from captures import Capture, CaptureSet
import chat
from collect import file_identity, sha256_file, verify_runtime
import collect_v2
import handoff
import kvstate
import run_v2
from run_v2 import (RunError, digest, fingerprint, json_text, jsonable, now_iso, read_json, read_jsonl,
                    stop_process_group, write_new)


HERE = Path(__file__).resolve().parent
LONG = "long"
STAGES = ("prepare", "grow-train", "token-table", "qwen-states", "muse-states", "build", "score", "report")
DEPENDS = {
    "prepare": (),
    "grow-train": ("prepare",),
    "token-table": ("prepare",),
    "qwen-states": ("prepare",),
    "muse-states": ("prepare",),
    "build": ("prepare", "grow-train", "token-table", "qwen-states", "muse-states"),
    "score": ("prepare", "muse-states", "build"),
    "report": ("prepare", "grow-train", "token-table", "qwen-states", "muse-states", "build", "score"),
}
CODE = {
    "prepare": ("chat.py", "collect_v2.py", "handoff.py", "align.py"),
    "grow-train": ("chat.py", "collect_v2.py", "align.py", "fit_layers.py", "captures.py"),
    "token-table": ("handoff.py", "captures.py", "align.py"),
    "qwen-states": ("collect_v2.py",),
    "muse-states": ("collect_v2.py",),
    "build": ("handoff.py", "kvstate.py", "attention_eval.py", "fit_layers.py"),
    "score": ("collect_v2.py",),
    "report": ("inject.py",),
}
ROLES = {"source": "qwen35", "target": "muse-glimmer"}
MODEL_NAMES = {"source": "qwen", "target": "muse"}
STATE_STAGES = {"qwen-states": "source", "muse-states": "target"}
TOKENIZE_CTX = 32768
FAMILIES = ("v2", "v3")
# kind -> (content policy, fill policy, needs the native Muse state at T0, depends on the mapper family,
#          possible at W = 0, role in the report); "sanity" scores the native state file itself
CONDITION_KINDS = {
    "sanity": (None, None, True, False, False, "check: the native Muse state at T0"),
    "oracle": ("oracle", "oracle", True, False, False, "check: handoff.py rebuilds the native state"),
    "oracle_fill": ("mapped", "oracle", True, True, False, "upper bound: mapped content, native fill cells"),
    "table_fill": ("mapped", "table", False, True, True, "deployable: mapped content, token-table fill"),
    "mean_oracle": ("mean", "oracle", True, False, False, "reference: content mean, native fill cells (not "
                    "information-free: native fill cells carry content; not the v2 mean_all protocol)"),
    "mean_old": ("mean", "mean", False, False, True, "floor (information-free, harsh): every cell the content mean"),
    "mean_table": ("mean", "table", False, False, True, "floor of table_fill (information-free): content mean, "
                   "token-table fill"),
}
# --conditions default: the kinds planned by prepare (the plan of run-v3 was prepared with exactly these);
# kinds that need no new native state can be added after prepare with --extra-conditions (build and score)
PLAN_KINDS = ("sanity", "oracle", "oracle_fill", "table_fill", "mean_oracle", "mean_old")
DEFAULT_EXTRA_KINDS = ("mean_table",)
CHECK_KINDS = ("sanity", "oracle")
FLOOR_KINDS = ("mean_old", "mean_table")
DEPLOYABLE_KIND = "table_fill"
HEADLINE_TAIL = 2048
SAVED = re.compile(r"saved ([0-9]+) states of ([0-9a-f]{12}) after ([0-9]+)/([0-9]+) tokens")
GPU_COMPUTE_BYTES = 4 * 1024 ** 3
DISK_HEADROOM = 20 * 1024 ** 3
# ETA priors (tokens/s, including model loads): GPU from the v3 grow-train collection (Qwen 191, Muse 198
# tokens/s at <= 4.6K tokens) less ~10% for 16-29K contexts; CPU from the v2 collection
DEFAULT_RATES = {"vulkan": (170.0, 170.0), "cpu": (19.0, 17.0)}
# Gates of the native-state checks (sanity, oracle), per document: (max token KL, mean KL, |mean delta NLL|).
# Exactly 0 on the CPU (T0 on the batch grid). On the GPU flash attention sums restored cells in another order
# (max token KL 1.5e-7 on the tiny fixtures, not measured at 16-29K tokens), so the GPU gates only ask the
# checks to stay an order of magnitude below the 0.05 target: a broken restore moves every token, not a few.
DEFAULT_SANITY_GATES = {"vulkan": (1e-2, 1e-3, 5e-3), "cpu": (1e-6, 1e-6, 1e-6)}
MODEL_LOAD_SECONDS = 20.0
# build throughput measured with the real v2 mapper and token table (a 2.8K-token v2 document, 8 BLAS threads):
# 200K (row, target map) predictions in 4.3 s with the map load; 1.4 s per 119 MB state for the rest
MAP_ROWS_PER_SECOND = 47000.0
STATE_BYTES_PER_SECOND = 8.5e7


def condition_name(kind, tail, family=None):
    return f"{kind}_{tail}" + ("" if family in (None, "v2") else f"_{family}")


def chunks(items, size):
    return [items[start:start + size] for start in range(0, len(items), size)]


def parse_tails(text):
    try:
        values = [int(item) for item in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("--tails must be comma-separated integers") from None
    if not values or len(set(values)) != len(values) or min(values) < 0:
        raise argparse.ArgumentTypeError("--tails must list distinct nonnegative tails")
    return sorted(values)


def parse_kinds(text):
    kinds = list(CONDITION_KINDS) if text == "all" else text.split(",")
    unknown = sorted(set(kinds) - set(CONDITION_KINDS))
    if unknown or len(set(kinds)) != len(kinds):
        raise argparse.ArgumentTypeError(f"--conditions must be 'all' or distinct kinds of {list(CONDITION_KINDS)}")
    return [kind for kind in CONDITION_KINDS if kind in kinds]


def parse_extra_kinds(text):
    if text == "none":
        return []
    kinds = text.split(",")
    allowed = [kind for kind, value in CONDITION_KINDS.items() if not value[2]]
    if sorted(set(kinds) - set(allowed)) or len(set(kinds)) != len(kinds):
        raise argparse.ArgumentTypeError(f"--extra-conditions must be 'none' or distinct kinds of {allowed} "
                                         "(kinds that need no native Muse state)")
    return [kind for kind in CONDITION_KINDS if kind in kinds]


def parse_buckets(text):
    try:
        edges = [int(item) for item in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("--prefix-buckets must be comma-separated integers") from None
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])) or edges[0] < 0:
        raise argparse.ArgumentTypeError("--prefix-buckets must be at least two increasing edges")
    return edges


def token_sha256(token_ids):
    return hashlib.sha256(np.asarray(token_ids, dtype="<i4").tobytes()).hexdigest()


def plan_t0(prefix, tail, batch, grid, minimum):
    """T0 for a tail W (None when the handoff would hold fewer than `minimum` tokens).

    W = 0 hands over everything but the last prefix token (seed-score needs the logits at P - 1
    from a decoded token). With `grid`, T0 = P - W rounded down to the --batch grid, so the
    seeded decode repeats the native ubatches and the sanity condition is exact on the CPU.
    """
    if tail == 0:
        t0 = prefix - 1
    else:
        t0 = prefix - tail
        if grid:
            t0 = t0 // batch * batch
    return t0 if t0 >= max(1, minimum) else None


def document_conditions(prefix, kinds, tails, families, batch, grid, minimum):
    """The planned conditions of one document: [{name, kind, family, tail, t0, content, fill, native}]."""
    out = []
    for tail in tails:
        t0 = plan_t0(prefix, tail, batch, grid, minimum)
        if t0 is None:
            continue
        for kind in kinds:
            content, fill, native, per_family, at_zero, _ = CONDITION_KINDS[kind]
            if tail == 0 and not at_zero:
                continue
            for family in (families if per_family else (None,)):
                out.append({"name": condition_name(kind, tail, family), "kind": kind, "family": family,
                            "tail": tail, "t0": t0, "content": content, "fill": fill, "native_state": native})
    return out


def with_extra_conditions(run, documents):
    """Copies of plan documents whose conditions include run.extra_kinds (planned like prepare would)."""
    a = run.args
    out = []
    for document in documents:
        names = {c["name"] for c in document["conditions"]}
        extra = [c for c in document_conditions(document["prefix"], run.extra_kinds, a.tails, run.families, a.batch,
                                                a.t0_grid == "on", a.min_handoff_tokens) if c["name"] not in names]
        out.append(dict(document, conditions=document["conditions"] + extra))
    return out


def bucket_label(bucket):
    return f"{bucket[0]}-{bucket[1]}"


# -- the run ----------------------------------------------------------------------------

class LongRun:
    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.records = {}
        self.lock_fds = ()
        self.current = None
        self._hashes = None
        self._cache = {}

    def event(self, event, **fields):
        print(json.dumps({"time": now_iso(), "event": event, **fields}, ensure_ascii=False, allow_nan=False,
                         default=str), flush=True)

    def path(self, rel):
        return self.root / rel

    def exists(self, rel):
        path = self.path(rel)
        return path.exists() or path.is_symlink()

    def v2(self, rel):
        return self.args.v2_run.resolve() / rel

    @property
    def families(self):
        return ["v2", "v3"] if self.args.grow_train else ["v2"]

    @property
    def stages(self):
        return tuple(stage for stage in STAGES if stage != "grow-train" or self.args.grow_train)

    def depends(self, stage):
        return tuple(name for name in DEPENDS[stage] if name in self.stages)

    @property
    def hashes(self):
        if self._hashes is None:
            self._hashes = collect_v2.ModelHashCache(self.path("model-hashes.json"))
        return self._hashes

    def model(self, role):
        return (self.args.source_model if role == "source" else self.args.target_model).resolve()

    def model_id(self, role):
        return self.args.source_model_id if role == "source" else self.args.target_model_id

    def model_sha(self, role):
        key = f"sha-{role}"
        if key not in self._cache:
            self._cache[key] = self.hashes.sha256(self.model(role))
        return self._cache[key]

    def sha(self, path):
        path = Path(path).resolve()
        key = ("sha", str(path), file_identity(path))
        if key not in self._cache:
            self._cache[key] = sha256_file(path)
        return self._cache[key]

    def spec(self, role):
        key = f"spec-{role}"
        if key not in self._cache:
            self._cache[key] = kvstate.ModelSpec.from_gguf(self.model(role))
        return self._cache[key]

    def probe(self):
        """(runtime revision, runtime identity, artifacts) of --probe (verified once per process)."""
        if "probe" not in self._cache:
            self._cache["probe"] = verify_runtime(self.args.probe.resolve())
        return self._cache["probe"]

    def plan(self):
        if "plan" not in self._cache:
            self._cache["plan"] = read_json(self.path(f"{LONG}/plan.json"))
        return self._cache["plan"]

    @property
    def extra_kinds(self):
        """--extra-conditions kinds that prepare did not plan (built and scored after the plan's own)."""
        return [kind for kind in self.args.extra_conditions if kind not in self.args.conditions]

    def documents(self):
        """The plan's documents, each with its planned conditions plus the --extra-conditions ones.

        Extra kinds need no native Muse state, so they add nothing to qwen-states or muse-states; their
        T0 per tail follows the prepare settings exactly as if prepare had planned them.
        """
        if "documents" not in self._cache:
            self._cache["documents"] = with_extra_conditions(self, self.plan()["documents"])
        return self._cache["documents"]

    def headline_name(self):
        if self.args.headline:
            return self.args.headline
        return condition_name(DEPLOYABLE_KIND, HEADLINE_TAIL, self.families[-1])

    def rendered(self, role):
        key = f"rendered-{role}"
        if key not in self._cache:
            self._cache[key] = {record["document_id"]: record
                                for record in read_jsonl(self.path(f"{LONG}/rendered-{role}.jsonl"))}
        return self._cache[key]

    def tokens(self, role, document_id):
        key = f"tokens-{role}"
        if key not in self._cache:
            self._cache[key] = Capture(self.path(f"{LONG}/tokens/{role}"))
        capture = self._cache[key]
        rows = capture.rows(document_id)
        return {name: np.array(getattr(capture, name)[rows]) for name in ("token_id", "byte_end", "special")}

    def rate(self, role):
        default = DEFAULT_RATES[self.args.device][0 if role == "source" else 1]
        given = self.args.source_tokens_per_second if role == "source" else self.args.target_tokens_per_second
        return given or default

    # -- stage records (the run_v2 format) -----------------------------------------------

    def settings(self, stage):
        a = self.args
        probe = {"device": a.device, "ctx": a.ctx, "batch": a.batch, "cache_type": a.cache_type,
                 "flash_attn": a.flash_attn, "docs_per_run": a.docs_per_run,
                 "expect_graph_splits": a.expect_graph_splits if a.device != "cpu" else None}
        if stage == "prepare":
            value = {"ctx": a.ctx, "batch": a.batch, "tails": a.tails, "t0_grid": a.t0_grid,
                     "min_handoff_tokens": a.min_handoff_tokens, "docs": a.docs, "docs_per_bucket": a.docs_per_bucket,
                     "prefix_buckets": a.prefix_buckets, "max_scored": a.max_scored, "conditions": a.conditions,
                     "families": self.families, "source_model_id": a.source_model_id,
                     "target_model_id": a.target_model_id}
        elif stage == "grow-train":
            value = {"train_ctx": a.train_ctx, "train_batch": a.train_batch, "device": a.device,
                     "fit_ks": a.fit_ks, "fit_alphas": a.fit_alphas}
        elif stage == "token-table":
            value = {"split": "train", "adopt": None if a.token_table is None else str(a.token_table.resolve())}
        elif stage in STATE_STAGES:
            value = probe
        elif stage == "build":
            value = {"swa_cells": a.swa_cells, "cache_type": a.cache_type, "flash_attn": a.flash_attn,
                     "chunk_rows": a.chunk_rows, "renorm_table_k": True, "extra_conditions": self.extra_kinds}
        elif stage == "score":
            # the sanity gates are not settings: they are re-applied to every group, reused ones included, so
            # changing them never invalidates scores (a failing group stays published for inspection)
            value = dict(probe, max_scored=a.max_scored, extra_conditions=self.extra_kinds,
                         probe_runtime=self.probe()[0])
        else:
            value = {"target_nll": a.target_nll, "bootstrap": a.bootstrap, "seed": a.seed,
                     "headline": self.headline_name()}
        return jsonable(value)

    def inputs(self, stage):
        if stage == "prepare":
            manifest = corpus_manifest_path(self.args)
            return {"corpus_sha256": self.sha(self.args.corpus),
                    "corpus_manifest_sha256": None if manifest is None else self.sha(manifest),
                    "source_model_sha256": self.model_sha("source"), "target_model_sha256": self.model_sha("target"),
                    "v2_rendered_sha256": {role: self.sha(self.v2(f"rendered-{role}.jsonl")) for role in ROLES}}
        value = {name: self.records[name]["outputs_sha256"] for name in self.depends(stage)}
        if stage == "grow-train":
            value.update(train_corpus_sha256=self.sha(self.args.train_corpus), **v2_inputs(self))
        elif stage == "token-table":
            value.update(v2_inputs(self))
            if self.args.token_table is not None:
                value["adopted_table_sha256"] = self.sha(self.args.token_table)
        elif stage == "build":
            value["v2_mapper_manifest_sha256"] = self.sha(self.v2("mapper/manifest.json"))
        return value

    def code(self, stage):
        return {name: sha256_file(HERE / name) for name in ("run_long.py", "run_v2.py") + CODE[stage]}

    def read_record(self, stage):
        path = self.path(f"stages/{stage}.json")
        if not path.exists():
            return None
        try:
            record = read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RunError(f"stage record {path} is corrupt: {exc}") from None
        if (not isinstance(record, dict) or record.get("format_version") != 1 or record.get("stage") != stage
                or record.get("status") != "complete" or not isinstance(record.get("outputs"), dict)):
            raise RunError(f"stage record {path} is malformed")
        return record

    def write_record(self, stage, outputs, metrics, started, wall, adopted=False):
        outputs = sorted(set(outputs))
        prints = fingerprint(self.root, outputs)
        record = {"format_version": 1, "stage": stage, "status": "complete", "adopted": adopted,
                  "settings": self.settings(stage), "inputs": self.inputs(stage), "output_paths": outputs,
                  "outputs": prints, "outputs_sha256": digest(prints), "code_sha256": self.code(stage),
                  "started": started, "finished": now_iso(), "wall_seconds": wall, "metrics": jsonable(metrics)}
        write_new(self.path(f"stages/{stage}.json"), json_text(record))
        self.records[stage] = record
        return record

    def verify(self, stage, record):
        for what, expected, found in (("settings", self.settings(stage), record.get("settings")),
                                      ("inputs", self.inputs(stage), record.get("inputs"))):
            if found != expected:
                found = found or {}
                changed = sorted(key for key in set(expected) | set(found) if expected.get(key) != found.get(key))
                if what == "settings":
                    raise RunError(f"stage {stage} completed with different settings ({', '.join(changed)}); "
                                   "rerun with the original settings or use a new --output")
                raise RunError(f"stage {stage} was completed from different inputs ({', '.join(changed)}); "
                               "an upstream output or model changed")
        paths = record.get("output_paths")
        if not isinstance(paths, list):
            raise RunError(f"stage record {stage} lacks output_paths")
        try:
            current = fingerprint(self.root, paths)
        except ValueError as exc:
            raise RunError(f"stage {stage} output is missing or invalid: {exc}") from None
        if current != record["outputs"] or digest(current) != record.get("outputs_sha256"):
            changed = sorted(key for key in set(current) | set(record["outputs"])
                             if current.get(key) != record["outputs"].get(key))
            raise RunError(f"stage {stage} outputs changed since the stage completed (corrupted or modified): "
                           f"{', '.join(changed[:10])}{' ...' if len(changed) > 10 else ''}")
        code = self.code(stage)
        if record.get("code_sha256") != code:
            self.event("warning", stage=stage, message="code changed since this stage completed",
                       files=sorted(name for name in code if (record.get("code_sha256") or {}).get(name) != code[name]))

    def key(self, stage, **extra):
        """Identity of a per-group / per-document output: the stage settings and inputs plus `extra`."""
        return digest({"stage": stage, "settings": self.settings(stage), "inputs": self.inputs(stage), **extra})

    # -- commands -------------------------------------------------------------------------

    def python_env(self, threads):
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for name in THREAD_VARIABLES:
            env[name] = str(threads)
        return env

    def command(self, stage, argv, log_name, *, env, on_line=None):
        """Run argv with stdout+stderr appended to logs/<log_name>; raise RunError on failure."""
        self.path("logs").mkdir(exist_ok=True)
        log_path = self.path(f"logs/{log_name}")
        command = [str(item) for item in argv]
        with log_path.open("ab") as log:
            log.write(f"==== run_long {stage} {now_iso()}: {shlex.join(command)}\n".encode("utf-8"))
            log.flush()
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=HERE,
                                  start_new_session=True, pass_fds=self.lock_fds) as process:
                try:
                    for raw in process.stdout:
                        log.write(raw)
                        log.flush()
                        if on_line is not None:
                            on_line(raw.decode("utf-8", "replace").rstrip("\n"))
                    code = process.wait()
                except BaseException:
                    stop_process_group(process)
                    raise
        if code != 0:
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 1500))
                tail = stream.read().decode("utf-8", "replace").strip().splitlines()[-6:]
            raise RunError(f"{stage}: {Path(command[1] if command[0] == sys.executable else command[0]).name} "
                           f"exited with status {code}; see {log_path}\n" + "\n".join(tail))

    def python(self, stage, script, arguments, log_name, *, threads=2, on_line=None):
        self.command(stage, [sys.executable, HERE / script, *arguments], log_name, env=self.python_env(threads),
                     on_line=on_line)

    def run_probe(self, stage, mode, role, records, output, log_name, *, extra=(), on_line=None):
        """kv_probe mode on documents -> output (created by the probe, atomically)."""
        _, _, artifacts = self.probe()
        a = self.args
        input_path = output.parent / (output.name + ".input.jsonl")
        with input_path.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        argv = [a.probe.resolve(), "--mode", mode, "--model", self.model(role), "--input", input_path,
                "--output", output, "--ctx", a.ctx, "--batch", a.batch, "--threads", a.threads,
                "--cache-type", a.cache_type, "--flash-attn", a.flash_attn, *extra]
        if a.device != "cpu":
            argv += ["--device", a.device]
        before = file_identity(self.model(role))
        started = time.monotonic()
        self.command(stage, argv, log_name, env=collect_v2.probe_environment(artifacts), on_line=on_line)
        if file_identity(self.model(role)) != before:
            raise RunError(f"{stage}: the model file changed during the run")
        return time.monotonic() - started

    def check_probe_report(self, report, stage, mode, role, extra_settings=None):
        a = self.args
        if report.get("format_version") != 2 or report.get("mode") != mode:
            raise RunError(f"{stage}: unexpected kv_probe report (mode {report.get('mode')!r})")
        if report.get("architecture") != ROLES[role]:
            raise RunError(f"{stage}: kv_probe ran a {report.get('architecture')} model, expected {ROLES[role]}")
        # --threads is not checked: it does not change outputs, and it is not part of the stage settings
        expected = {"ctx": a.ctx, "batch": a.batch, "cache_type": a.cache_type,
                    "flash_attention": a.flash_attn == "on", "swa_full": True, "add_special": False,
                    "parse_special": True, **(extra_settings or {})}
        if a.device != "cpu":
            expected["device"] = a.device
        try:
            collect_v2.check_settings(report.get("settings"), expected)
        except ValueError as exc:
            raise RunError(f"{stage}: {exc}") from None
        return check_device(report, a, stage)

    def memory_guard(self, stage, role):
        """MemAvailable must exceed weights + KV cache (+ GPU compute buffers) + 6 GiB."""
        a = self.args
        model = self.model(role)
        size = model.stat().st_size
        if a.skip_memory_check:
            try:
                collect_v2.memory_check(model, True)
            except ValueError as exc:
                raise RunError(f"{stage}: {exc}") from None
            return None
        need = size + kv_cache_bytes(self.spec(role), a.ctx, a.cache_type) + collect_v2.MEMORY_HEADROOM
        if a.device != "cpu":
            need += GPU_COMPUTE_BYTES
        available = collect_v2.mem_available()
        busy = processes_using(model)
        if busy:
            self.event("warning", stage=stage, message="another process has the model on its command line",
                       pids=busy)
        self.event("memory", stage=stage, role=role, available_gib=round(available / 1024 ** 3, 1),
                   needed_gib=round(need / 1024 ** 3, 1))
        if available <= need:
            raise RunError(f"{stage}: MemAvailable {available / 1024 ** 3:.1f} GiB does not exceed "
                           f"{need / 1024 ** 3:.1f} GiB (weights, KV cache at --ctx {a.ctx}, "
                           f"{'GPU buffers, ' if a.device != 'cpu' else ''}6 GiB headroom); refusing to load "
                           f"{model.name}. Stop the live servers first.")
        return {"available": available, "needed": need}


def kv_cache_bytes(spec, ctx, cache_type):
    """KV bytes of a context with every attention layer holding ctx cells (swa_full on)."""
    width = 2 if cache_type == "f16" else 4
    per_token = sum(spec.row_elements(layer, "k") + spec.row_elements(layer, "v")
                    for layer in spec.layers_of("full", "sliding"))
    return per_token * width * ctx


def state_bytes_per_token(spec, cache_type, layers):
    width = 2 if cache_type == "f16" else 4
    return sum(spec.row_elements(layer, "k") + spec.row_elements(layer, "v") for layer in layers) * width


def disk_guard(run, stage, needed):
    free = shutil.disk_usage(run.root).free
    run.event("disk", stage=stage, free_gib=round(free / 1024 ** 3, 1), needed_gib=round(needed / 1024 ** 3, 1))
    if free <= needed + DISK_HEADROOM:
        raise RunError(f"{stage}: {free / 1024 ** 3:.1f} GiB free on {run.root}, need about "
                       f"{needed / 1024 ** 3:.1f} GiB + {DISK_HEADROOM / 1024 ** 3:.0f} GiB headroom")


def state_file_bytes(spec, cache_type, n_tokens, swa_cells=None):
    """Approximate bytes of a state holding n tokens (Muse: all base cells, at most swa_cells sliding cells)."""
    if spec.architecture == "muse-glimmer":
        window = min(n_tokens, swa_cells or spec.n_swa)
        return (n_tokens * state_bytes_per_token(spec, cache_type, spec.layers_of("full"))
                + window * state_bytes_per_token(spec, cache_type, spec.layers_of("sliding")))
    recurrent = 0
    if spec.recurrent_row_bytes is not None:
        recurrent = sum(spec.recurrent_row_bytes) * len(spec.layers_of("recurrent"))
    return n_tokens * state_bytes_per_token(spec, cache_type, spec.layers_of("full")) + recurrent


def score_tokens(doc, batch):
    """Tokens seed-score decodes for a document: native through the last needed ubatch, each state from T0."""
    needed = min(doc["muse_tokens"], -(-doc["last_scored"] // batch) * batch)
    return needed + sum(needed - condition["t0"] for condition in doc["conditions"])


def mapped_row_maps(doc, spec, swa_cells, family):
    """(row, target map) predictions of one document and mapper family with the prediction cache."""
    mapped = [c for c in doc["conditions"] if c["content"] == "mapped" and (c["family"] or "v2") == family]
    if not mapped:
        return 0
    full, sliding = len(spec.layers_of("full")), len(spec.layers_of("sliding"))
    base = max(c["t0"] for c in mapped) * full * 2
    return base + sum(min(t0, swa_cells) * sliding * 2 for t0 in {c["t0"] for c in mapped})


def estimate_plan(run, plan):
    """Time, RAM and disk per stage from the plan, the model specs and the ETA priors."""
    a = run.args
    qwen, muse = run.spec("source"), run.spec("target")
    swa_cells = a.swa_cells or muse.n_swa
    documents = with_extra_conditions(run, plan["documents"])
    groups = -(-len(documents) // a.docs_per_run)
    gib = 1024 ** 3

    def ram(role):
        spec = run.spec(role)
        return (run.model(role).stat().st_size + kv_cache_bytes(spec, a.ctx, a.cache_type)
                + (GPU_COMPUTE_BYTES if a.device != "cpu" else 0)) / gib

    qwen_tokens = sum(doc["qwen_prefix"] for doc in documents)
    native = {(doc["document_id"], c["t0"]) for doc in documents for c in doc["conditions"] if c["native_state"]}
    muse_tokens = sum(max([c["t0"] for c in doc["conditions"] if c["native_state"]] or [0]) for doc in documents)
    built = [(doc, c) for doc in documents for c in doc["conditions"] if c["kind"] != "sanity"]
    built_bytes = sum(state_file_bytes(muse, a.cache_type, c["t0"], swa_cells) for _, c in built)
    row_maps = sum(mapped_row_maps(doc, muse, swa_cells, family) for doc in documents for family in run.families)
    largest = max(documents, key=lambda doc: doc["prefix"])
    build_ram = (state_file_bytes(qwen, a.cache_type, largest["qwen_prefix"])
                 + state_file_bytes(muse, a.cache_type, largest["prefix"], swa_cells) * 3
                 + mapped_row_maps(largest, muse, swa_cells, "v2") * 1024 + 2 * 1.7e9) / gib
    score = sum(score_tokens(doc, a.batch) for doc in documents)
    stages = {
        "qwen-states": {"tokens": qwen_tokens, "hours": (qwen_tokens / run.rate("source")
                                                         + groups * MODEL_LOAD_SECONDS) / 3600,
                        "ram_gib": ram("source"),
                        "disk_gib": sum(state_file_bytes(qwen, a.cache_type, doc["qwen_prefix"])
                                        for doc in documents) / gib},
        "muse-states": {"tokens": muse_tokens, "hours": (muse_tokens / run.rate("target")
                                                         + groups * MODEL_LOAD_SECONDS) / 3600,
                        "ram_gib": ram("target"),
                        "disk_gib": sum(state_file_bytes(muse, a.cache_type, t0, muse.n_swa + a.batch)
                                        for _, t0 in native) / gib},
        "build": {"states": len(built), "row_maps": row_maps,
                  "hours": (row_maps / MAP_ROWS_PER_SECOND + built_bytes / STATE_BYTES_PER_SECOND) / 3600,
                  "ram_gib": build_ram, "disk_gib": built_bytes / gib},
        "score": {"tokens": score, "hours": (score / run.rate("target") + groups * MODEL_LOAD_SECONDS) / 3600,
                  "ram_gib": ram("target"), "disk_gib": 0.1},
    }
    for item in stages.values():
        for key in ("hours", "ram_gib", "disk_gib"):
            item[key] = round(item[key], 2)
    return {"stages": stages, "total_hours": round(sum(item["hours"] for item in stages.values()), 2),
            "total_disk_gib": round(sum(item["disk_gib"] for item in stages.values()), 1),
            "assumptions": {"tokens_per_second": {"source": run.rate("source"), "target": run.rate("target")},
                            "model_load_seconds": MODEL_LOAD_SECONDS, "map_rows_per_second": MAP_ROWS_PER_SECOND,
                            "state_bytes_per_second": STATE_BYTES_PER_SECOND}}


def processes_using(model):
    """PIDs whose command line names the model file (e.g. a live llama-server)."""
    target = str(Path(model).resolve())
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            words = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        for word in words:
            try:
                text = word.decode("utf-8")
                if text and (text == target or (os.path.isabs(text) and os.path.realpath(text) == target)):
                    found.append(int(entry.name))
                    break
            except (UnicodeDecodeError, ValueError):
                continue
    return found


def check_device(report, args, stage):
    """None on the CPU; on a GPU every layer must be offloaded with the expected graph splits."""
    device = report.get("device")
    if args.device == "cpu":
        if device is not None:
            raise RunError(f"{stage}: a CPU run reported device {device}")
        return None
    if not isinstance(device, dict) or device.get("backend") != args.device:
        raise RunError(f"{stage}: the probe report lacks the {args.device} device section")
    offloaded = str(device.get("offloaded_layers") or "")
    done, _, total = offloaded.partition("/")
    if not done or done != total:
        raise RunError(f"{stage}: only {offloaded or 'unknown'} layers were offloaded to the GPU")
    if args.expect_graph_splits != "any" and str(device.get("graph_splits")) != args.expect_graph_splits:
        raise RunError(f"{stage}: graph splits = {device.get('graph_splits')!r}, expected "
                       f"{args.expect_graph_splits!r}: an operation may have fallen back to the CPU; stop and "
                       "investigate (see the log), or pass --expect-graph-splits any")
    return device


def seed_hash_cache(path, seed, models):
    """Start RUN/model-hashes.json from another run's cache (read-only), keeping entries of these models."""
    data = json.loads(Path(seed).read_text(encoding="utf-8"))
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        raise RunError(f"--hash-cache-seed {seed} is not a model hash cache")
    keep = {}
    for model in models:
        key = collect_v2.ModelHashCache.key(model)
        if isinstance(entries.get(key), str) and collect_v2.SHA256.fullmatch(entries[key]):
            keep[key] = entries[key]
    if keep:
        write_new(path, json.dumps({"format_version": 1, "entries": keep}, indent=2))


def corpus_manifest_path(args):
    if args.corpus_manifest is not None:
        return args.corpus_manifest
    candidate = args.corpus.parent / "manifest.json"
    return candidate if candidate.is_file() else None


def v2_inputs(run):
    """Digests of the v2 run files the token table and the refit read."""
    captures = {name: run.sha(run.v2(f"captures/{name}/report.json")) for name in ("source", "target-train",
                                                                                   "target-eval")}
    return {"v2_alignment_sha256": run.sha(run.v2("alignment/alignment.npz")), "v2_capture_reports": captures,
            "v2_mapper_manifest_sha256": run.sha(run.v2("mapper/manifest.json"))}


def write_or_match(run, rel, content):
    """Write a new file, or accept an identical existing one (an earlier attempt); True when written."""
    data = content.encode("utf-8") if isinstance(content, str) else content
    path = run.path(rel)
    if run.exists(rel):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise RunError(f"{path} exists with other content than this run produces; remove it or use a new "
                           "--output")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, data)
    return True


def staging_directory(run, stage):
    """A temp directory under long/.staging (never part of a stage's fingerprinted outputs)."""
    parent = run.path(f"{LONG}/.staging")
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{stage}-", dir=parent))


def publish(staging, destination):
    os.chmod(staging, 0o755)
    destination.parent.mkdir(parents=True, exist_ok=True)
    collect_v2.publish_directory(staging, destination)


def eta(run, stage, amount, rate, provenance, unit="tokens", **extra):
    seconds = amount / rate if rate else None
    finish = None if seconds is None else (datetime.datetime.now().astimezone()
                                           + datetime.timedelta(seconds=seconds)).isoformat(timespec="minutes")
    run.event("eta", stage=stage, amount=amount, unit=unit, per_second=rate, rate_source=provenance,
              eta_seconds=None if seconds is None else round(seconds), eta_finish=finish, **extra)


# -- prepare ----------------------------------------------------------------------------

def read_corpus_manifest(args):
    path = corpus_manifest_path(args)
    if path is None:
        return None, {}
    manifest = read_json(path)
    documents = ((manifest.get("long_eval") or {}).get("documents")) if isinstance(manifest, dict) else None
    if not isinstance(documents, list):
        raise RunError(f"{path} has no long_eval.documents (a build_corpus.py long-eval manifest)")
    by_id = {}
    for item in documents:
        if not isinstance(item, dict) or not isinstance(item.get("document_id"), str):
            raise RunError(f"{path}: malformed long_eval document entry")
        by_id[item["document_id"]] = item
    return path, by_id


def scored_positions(tokens, start_byte, end_byte, cap):
    """Positions p >= 1 whose token lies inside [start_byte, end_byte) (kv_probe's rule), first `cap`."""
    ends = tokens["byte_end"]
    starts = np.concatenate([[0], ends[:-1]])
    inside = np.flatnonzero((starts >= start_byte) & (ends <= end_byte))
    inside = inside[inside >= 1]
    return inside[:cap], int(len(inside))


def stage_prepare(run):
    a = run.args
    manifest_path, manifest_docs = read_corpus_manifest(a)
    conversations = list(chat.read_conversations(a.corpus))
    if not conversations:
        raise RunError("the long corpus is empty")
    lines, templates, records = {}, {}, {}
    for role in ROLES:
        architecture = run_v2._architecture(run.model(role))
        if architecture != ROLES[role]:
            raise RunError(f"--{role}-model architecture is {architecture!r}, expected {ROLES[role]!r}")
        info = chat.gguf_tokenizer_info(run.model(role))
        template, bos = info["chat_template"], info["bos_token"]
        if template is None:
            raise RunError(f"--{role}-model has no tokenizer.chat_template metadata")
        with run.v2(f"rendered-{role}.jsonl").open(encoding="utf-8") as stream:
            v2_record = json.loads(stream.readline())
        if (v2_record.get("template_sha256"), v2_record.get("bos_token")) != (chat.template_sha256(template), bos):
            raise RunError(f"{role}: the model's chat template or BOS differs from the v2 run's renders "
                           f"({run.v2(f'rendered-{role}.jsonl')}); the mapper was fitted on those")
        out, seen = [], set()
        for number, conversation in enumerate(conversations, 1):
            try:
                record = chat.rendered_record(conversation, MODEL_NAMES[role],
                                              chat.render(conversation, template, bos_token=bos))
            except ValueError as exc:
                raise RunError(f"{role}: conversation {conversation.get('id')!r} (line {number}): {exc}") from None
            if record["document_id"] in seen:
                raise RunError(f"duplicate conversation {conversation.get('id')!r} (line {number})")
            if record["split"] != "test":
                raise RunError(f"conversation {conversation.get('id')!r} is not a test conversation")
            seen.add(record["document_id"])
            out.append(record)
        collect_v2.check_bos(out, bos, info)
        lines[role] = "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in out)
        templates[role] = template
        records[role] = out
    written = []
    for role in ROLES:
        for rel, content in ((f"{LONG}/templates/{MODEL_NAMES[role]}.jinja", templates[role]),
                             (f"{LONG}/rendered-{role}.jsonl", lines[role])):
            if write_or_match(run, rel, content):
                written.append(rel)
    run.path(f"{LONG}/tokens").mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        rel = f"{LONG}/tokens/{role}"
        if not run.exists(rel):
            run.python("prepare", "collect_v2.py", [
                "--probe", a.probe.resolve(), "--mode", "tokenize", "--model", run.model(role),
                "--model-id", run.model_id(role), "--docs", run.path(f"{LONG}/rendered-{role}.jsonl"),
                "--chat-template", run.path(f"{LONG}/templates/{MODEL_NAMES[role]}.jinja"),
                "--hash-cache", run.path("model-hashes.json"), "--output", run.path(rel),
                "--ctx", TOKENIZE_CTX, "--batch", 512, "--threads", 4], "long-prepare.log")
            written.append(rel)
        capture = Capture(run.path(rel))
        capture.check_documents(records[role])
        if (capture.manifest or {}).get("model_sha256") != run.model_sha(role):
            raise RunError(f"{run.path(rel)} was tokenized with another model")
        bad = run_v2._special_inside_segments(capture, records[role])
        if bad:
            raise RunError(f"{role}: special tokens inside segments of {len(bad)} documents: {bad[:3]}")
    documents = []
    for source_record, target_record in zip(records["source"], records["target"]):
        document_id = target_record["document_id"]
        entry = manifest_docs.get(document_id)
        if manifest_docs and entry is None:
            raise RunError(f"conversation {target_record['conversation_id']!r} is not in the corpus manifest")
        q_tokens, m_tokens = run.tokens("source", document_id), run.tokens("target", document_id)
        for role, tokens in (("source", q_tokens), ("target", m_tokens)):
            if len(tokens["token_id"]) > a.ctx:
                raise RunError(f"{role}: document {document_id[:12]} has {len(tokens['token_id'])} tokens, more "
                               f"than --ctx {a.ctx}")
        try:
            alignment = handoff.align_document(source_record, q_tokens, target_record, m_tokens)
            prefix = handoff.handoff_position(target_record, m_tokens)
            qwen_prefix = handoff.handoff_position(source_record, q_tokens)
            final_start = handoff.final_message_start(target_record)
        except ValueError as exc:
            raise RunError(f"prepare: document {document_id[:12]}: {exc}") from None
        text_bytes = len(target_record["text"].encode("utf-8"))
        scored, eligible = scored_positions(m_tokens, final_start, text_bytes, a.max_scored)
        if not len(scored) or scored[0] < prefix:
            raise RunError(f"prepare: document {document_id[:12]} has no scored token at or after P = {prefix}")
        classes = alignment.tgt_class[:prefix]
        if entry is not None:
            bucket, stratum, role_name = entry.get("bucket"), entry.get("stratum"), entry.get("role")
            if entry.get("muse_prefix_tokens") not in (None, prefix):
                raise RunError(f"prepare: document {document_id[:12]}: P = {prefix} but the corpus manifest says "
                               f"{entry['muse_prefix_tokens']} (different tokenizer or template)")
        else:
            edges = a.prefix_buckets
            index = int(np.searchsorted(edges, prefix, side="right")) - 1
            bucket = [edges[index], edges[index + 1]] if 0 <= index < len(edges) - 1 else None
            stratum, role_name = None, "primary"
        documents.append({
            "document_id": document_id, "conversation_id": target_record["conversation_id"],
            "source": target_record["source"], "bucket": bucket_label(bucket) if bucket else "other",
            "stratum": stratum, "role": role_name, "muse_tokens": int(len(m_tokens["token_id"])),
            "qwen_tokens": int(len(q_tokens["token_id"])), "prefix": prefix, "qwen_prefix": qwen_prefix,
            "final_start_byte": final_start, "text_bytes": text_bytes, "first_scored": int(scored[0]),
            "last_scored": int(scored[-1]), "n_scored": int(len(scored)), "n_eligible": eligible,
            "muse_token_sha256": token_sha256(m_tokens["token_id"]),
            "qwen_token_sha256": token_sha256(q_tokens["token_id"]),
            "prefix_classes": {"aligned": int(np.sum(classes == ALIGNED)),
                               "template_or_special": int(np.sum(classes == TEMPLATE)),
                               "unmatched": int(np.sum(classes == UNMATCHED)),
                               "straddling": int(np.sum(classes == STRADDLING))},
            "conditions": document_conditions(prefix, a.conditions, a.tails, run.families, a.batch,
                                              a.t0_grid == "on", a.min_handoff_tokens)})
    if a.docs == "primary":
        if not manifest_docs:
            raise RunError("--docs primary needs a corpus manifest (build_corpus.py long-eval manifest.json)")
        documents = [item for item in documents if item["role"] == "primary"]
    if a.docs_per_bucket:
        kept, counts = [], {}
        for item in documents:
            counts[item["bucket"]] = counts.get(item["bucket"], 0) + 1
            if counts[item["bucket"]] <= a.docs_per_bucket:
                kept.append(item)
        documents = kept
    documents = [item for item in documents if item["conditions"]]
    if not documents:
        raise RunError("prepare: no document has an applicable condition")
    buckets = {}
    for item in documents:
        group = buckets.setdefault(item["bucket"], {"documents": 0, "muse_prefix_tokens": [], "conditions": 0})
        group["documents"] += 1
        group["muse_prefix_tokens"].append(item["prefix"])
        group["conditions"] += len(item["conditions"])
    for group in buckets.values():
        values = group.pop("muse_prefix_tokens")
        group.update(prefix_min=min(values), prefix_median=float(np.median(values)), prefix_max=max(values))
    plan = {"format_version": 1, "purpose": "kv_handoff_long_eval_plan",
            "corpus": {"path": str(a.corpus.resolve()), "sha256": run.sha(a.corpus),
                       "manifest": None if manifest_path is None else str(Path(manifest_path).resolve())},
            "settings": run.settings("prepare"), "families": run.families,
            "condition_kinds": {kind: {"content": value[0], "fill": value[1], "native_state": value[2],
                                       "per_family": value[3], "at_tail_0": value[4], "role": value[5]}
                                for kind, value in CONDITION_KINDS.items() if kind in a.conditions},
            "definitions": {
                "prefix": "P = Muse tokens ending at or before the start of the final message's first rendered "
                          "segment; the first scored token is at P (or later when a token straddles)",
                "t0": "tail W > 0: P - W rounded down to the --batch grid (--t0-grid on); W = 0: P - 1 (the "
                      "last prefix token is decoded natively so seed-score has logits at P - 1)",
                "scored": "the first --max-scored tokens inside the final message (kv_probe score_ranges)"},
            "buckets": dict(sorted(buckets.items())), "documents": documents}
    if write_or_match(run, f"{LONG}/plan.json", json_text(plan)):
        written.append(f"{LONG}/plan.json")
    run._cache.pop("plan", None)
    outputs = [f"{LONG}/templates", f"{LONG}/rendered-source.jsonl", f"{LONG}/rendered-target.jsonl",
               f"{LONG}/tokens/source", f"{LONG}/tokens/target", f"{LONG}/plan.json"]
    metrics = {"documents": len(documents), "corpus_documents": len(conversations), "buckets": plan["buckets"],
               "conditions": sum(len(item["conditions"]) for item in documents),
               "muse_prefix_tokens": sum(item["prefix"] for item in documents),
               "qwen_prefix_tokens": sum(item["qwen_prefix"] for item in documents),
               "max_tokens": {"source": max(item["qwen_tokens"] for item in documents),
                              "target": max(item["muse_tokens"] for item in documents)},
               "written": written, "estimate": estimate_plan(run, plan)}
    run.event("estimate", stage="prepare", **metrics["estimate"])
    return outputs, metrics, False


# -- grow-train ---------------------------------------------------------------------------

def refit_capture_paths(run):
    return ([run.v2("captures/source"), run.path("captures/train-extra-source")],
            [run.v2("captures/target-train"), run.v2("captures/target-eval"), run.path("captures/train-extra-target")])


def validate_train_capture(run, role, records, rendered_rel):
    rel = f"captures/train-extra-{role}"
    capture = Capture(run.path(rel))
    manifest = capture.manifest
    template = run.path(f"{LONG}/templates/{MODEL_NAMES[role]}.jinja").read_text(encoding="utf-8")
    expected = {"mode": "collect", "model_sha256": run.model_sha(role), "architecture": ROLES[role],
                "rendered_docs_sha256": run.sha(run.path(rendered_rel)), "capture_spec": {"k": "all", "v": "all"},
                "render_template_sha256": chat.template_sha256(template)}
    if not isinstance(manifest, dict):
        raise RunError(f"{run.path(rel)} has no collect_v2 manifest.json")
    wrong = sorted(key for key, value in expected.items() if manifest.get(key) != value)
    if wrong:
        raise RunError(f"grow-train: {run.path(rel)} does not match this run ({', '.join(wrong)})")
    capture.check_documents(records)
    collect_v2.check_capture_spec(capture, {"k": "all", "v": "all"})
    settings = manifest.get("settings") or {}
    return capture, {key: settings.get(key) for key in ("device", "ctx", "batch", "cache_type", "flash_attention")}


def validate_refit_alignment(run, path, v2_alignment, extra_ids):
    alignment = load_alignment(path)
    if alignment.rejected:
        raise RunError(f"grow-train: {path} rejected documents {alignment.rejected[:3]}")
    splits = dict(zip(v2_alignment.document_ids, v2_alignment.splits))
    expected = {**splits, **{document_id: "train" for document_id in extra_ids}}
    found = dict(zip(alignment.document_ids, alignment.splits))
    if found != expected:
        raise RunError(f"grow-train: {path} does not align exactly the v2 documents (same splits) plus the extra "
                       "train documents")
    _, targets = refit_capture_paths(run)
    try:
        handoff.check_alignment_rows(CaptureSet(targets), alignment)
    except ValueError as exc:
        raise RunError(f"grow-train: {path}: {exc}") from None
    return alignment


def validate_refit_mapper(run, path, alignment_path):
    manifest = read_json(Path(path) / "manifest.json")
    v2 = read_json(run.v2("mapper/manifest.json"))
    sources, targets = refit_capture_paths(run)
    problems = []
    if manifest.get("purpose") != "kv_linear_transfer_mapper":
        problems.append("purpose")
    if (manifest.get("alignment") or {}).get("sha256") != run.sha(alignment_path):
        problems.append("alignment")
    for side, paths in (("source", sources), ("target", targets)):
        reports = [item.get("report_sha256") for item in (manifest.get(side) or {}).get("captures") or []]
        if reports != [run.sha(Path(item) / "report.json") for item in paths]:
            problems.append(f"{side} captures")
    for split in ("dev", "test"):
        if (manifest.get("split_documents") or {}).get(split) != v2["split_documents"][split]:
            problems.append(f"{split} documents")
    if problems:
        raise RunError(f"grow-train: {path} was not fitted on v2 + extra train with v2 dev/test "
                       f"({', '.join(problems)})")
    for name in ("alphas", "ks"):
        if (manifest.get("settings") or {}).get(name) != (v2.get("settings") or {}).get(name):
            run.event("warning", stage="grow-train", message=f"the v3 mapper used other {name} than the v2 mapper",
                      v2=(v2.get("settings") or {}).get(name), v3=(manifest.get("settings") or {}).get(name))
    return manifest


def r2_comparison(run, v2_manifest, v3_manifest):
    types = run.spec("target").layer_types

    def table(manifest):
        out = {}
        for item in manifest["maps"]:
            applied = item["dev"]["applied"]
            out[item["kind"], item["target"]] = {"dev": item["dev"][applied]["r2"], "test": item["test"][applied]["r2"],
                                                 "k": item["k"]}
        return out

    old, new = table(v2_manifest), table(v3_manifest)
    means = {}
    for kind in ("k", "v"):
        for attention in ("full", "sliding", "all"):
            keys = [key for key in old if key[0] == kind and attention in ("all", types[key[1]]) and key in new]
            if keys:
                means[f"{kind}/{attention}"] = {
                    split: {"v2": float(np.mean([old[key][split] for key in keys])),
                            "v3": float(np.mean([new[key][split] for key in keys]))} for split in ("dev", "test")}
    per_map = [{"kind": kind, "target": target, "attention": types[target], "v2": old[kind, target],
                "v3": new[kind, target]} for kind, target in sorted(old) if (kind, target) in new]
    return {"means": means, "maps": per_map,
            "rows": {"v2": v2_manifest["rows"], "v3": v3_manifest["rows"]}}


def stage_grow_train(run):
    a = run.args
    conversations = list(chat.read_conversations(a.train_corpus))
    if not conversations:
        raise RunError("the train corpus is empty")
    adopted, metrics, lines, records = [], {"parts": {}}, {}, {}
    for role in ROLES:
        template = run.path(f"{LONG}/templates/{MODEL_NAMES[role]}.jinja").read_text(encoding="utf-8")
        bos = chat.gguf_tokenizer_info(run.model(role))["bos_token"]
        out = []
        for number, conversation in enumerate(conversations, 1):
            try:
                record = chat.rendered_record(conversation, MODEL_NAMES[role],
                                              chat.render(conversation, template, bos_token=bos))
            except ValueError as exc:
                raise RunError(f"grow-train {role}: conversation {conversation.get('id')!r} (line {number}): "
                               f"{exc}") from None
            if record["split"] != "train":
                raise RunError(f"grow-train: conversation {conversation.get('id')!r} is not a train conversation")
            out.append(record)
        records[role] = out
        lines[role] = "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in out)
        rel = f"rendered-train-extra-{role}.jsonl"
        if not write_or_match(run, rel, lines[role]):
            adopted.append(rel)
    v2_records = {role: read_jsonl(run.v2(f"rendered-{role}.jsonl")) for role in ROLES}
    overlap = {record["document_id"] for record in v2_records["target"]} & {r["document_id"] for r in records["target"]}
    if overlap:
        raise RunError(f"grow-train: {len(overlap)} train corpus documents are already in the v2 corpus")
    manual = run_v2.collection_times(run)
    for role in ROLES:
        rel = f"captures/train-extra-{role}"
        rendered = f"rendered-train-extra-{role}.jsonl"
        seconds, collected = None, False
        if run.exists(rel):
            adopted.append(rel)
            seconds = manual.get(f"train-extra-{role}")
        else:
            run.memory_guard("grow-train", role)
            tokens = sum(len(record["text"].encode("utf-8")) for record in records[role]) // 4
            eta(run, "grow-train", tokens, run.rate(role), "prior (about 4 bytes per token)", part=rel)
            progress = run_v2.Progress(run, "grow-train", tokens, run_v2.CAPTURED, lambda m: int(m.group(2)), rel)
            run.path("captures").mkdir(exist_ok=True)
            started = time.monotonic()
            arguments = ["--probe", a.probe.resolve(), "--mode", "collect", "--model", run.model(role),
                         "--model-id", run.model_id(role), "--docs", run.path(rendered),
                         "--capture-spec", run.v2("spec-kv.json"),
                         "--chat-template", run.path(f"{LONG}/templates/{MODEL_NAMES[role]}.jinja"),
                         "--hash-cache", run.path("model-hashes.json"), "--output", run.path(rel),
                         "--ctx", a.train_ctx, "--batch", a.train_batch, "--threads", a.threads, "--device", a.device]
            if a.skip_memory_check:
                arguments.append("--skip-memory-check")
            run.python("grow-train", "collect_v2.py", arguments, f"long-grow-train-{role}.log", on_line=progress)
            seconds, collected = time.monotonic() - started, True
        capture, settings = validate_train_capture(run, role, records[role], rendered)
        metrics["parts"][rel] = {"documents": len(capture.documents), "tokens": capture.rows_total,
                                 "seconds": seconds, "adopted": not collected, "collect_settings": settings,
                                 "tokens_per_second": capture.rows_total / seconds if seconds else None}
    for role in ROLES:
        rel = f"refit/rendered-{role}-all.jsonl"
        content = run.v2(f"rendered-{role}.jsonl").read_text(encoding="utf-8") + lines[role]
        if not write_or_match(run, rel, content):
            adopted.append(rel)
    sources, targets = refit_capture_paths(run)
    alignment_rel = "refit/alignment.npz"
    v2_alignment = load_alignment(run.v2("alignment/alignment.npz"))
    if run.exists(alignment_rel):
        adopted.append(alignment_rel)
    else:
        arguments = ["--src-docs", run.path("refit/rendered-source-all.jsonl"),
                     "--tgt-docs", run.path("refit/rendered-target-all.jsonl"), "--output", run.path(alignment_rel)]
        for path in sources:
            arguments += ["--src-capture", path]
        for path in targets:
            arguments += ["--tgt-capture", path]
        run.python("grow-train", "align.py", arguments, "long-grow-train-align.log", threads=a.build_threads)
    alignment = validate_refit_alignment(run, run.path(alignment_rel), v2_alignment,
                                         [record["document_id"] for record in records["target"]])
    mapper_rel = "refit/mapper"
    if run.exists(mapper_rel):
        adopted.append(mapper_rel)
    else:
        pending = sorted(p.name for p in run.path("refit").glob(".*") if p.name.startswith((".pending-mapper-",
                                                                                         ".kvpark-fit-work-")))
        if pending:
            raise RunError(f"grow-train: {run.path('refit')} holds {pending}: a refit is still running (or died); "
                           "wait for it to publish refit/mapper, or remove the leftovers")
        arguments = ["--alignment", run.path(alignment_rel), "--output", run.path(mapper_rel),
                     "--threads", a.build_threads, "--ks", a.fit_ks]
        for path in sources:
            arguments += ["--src-capture", path]
        for path in targets:
            arguments += ["--tgt-capture", path]
        if a.fit_alphas:
            arguments += ["--alphas", a.fit_alphas]
        run.python("grow-train", "fit_layers.py", arguments, "long-grow-train-fit.log", threads=a.build_threads)
    v3 = validate_refit_mapper(run, run.path(mapper_rel), run.path(alignment_rel))
    metrics["alignment"] = alignment.summary()
    metrics["alignment"].pop("rejected", None)
    metrics["r2"] = r2_comparison(run, read_json(run.v2("mapper/manifest.json")), v3)
    metrics["fit_wall_seconds"] = v3.get("wall_seconds")
    metrics["adopted"] = sorted(adopted)
    outputs = ["rendered-train-extra-source.jsonl", "rendered-train-extra-target.jsonl", "captures/train-extra-source",
               "captures/train-extra-target", "refit/rendered-source-all.jsonl", "refit/rendered-target-all.jsonl",
               alignment_rel, mapper_rel]
    return outputs, metrics, bool(adopted)


# -- token table ------------------------------------------------------------------------------

def v2_target_paths(run):
    return [run.v2("captures/target-train"), run.v2("captures/target-eval")]


def check_token_table(run, path):
    try:
        table = handoff.TokenTable(path)
    except (ValueError, KeyError, OSError) as exc:
        raise RunError(f"token-table: {path}: {exc}") from None
    try:
        metadata = table.metadata
        spec = run.spec("target")
        problems = []
        if metadata.get("architecture") != "muse-glimmer" or metadata.get("split") != "train":
            problems.append("architecture/split")
        if list(metadata.get("layers") or []) != list(spec.layers_of("full", "sliding")):
            problems.append("layers")
        if (metadata.get("alignment") or {}).get("sha256") != run.sha(run.v2("alignment/alignment.npz")):
            problems.append("alignment")
        reports = [item.get("report_sha256") for item in metadata.get("target_captures") or []]
        if reports != [run.sha(p / "report.json") for p in v2_target_paths(run)]:
            problems.append("target captures")
        if problems:
            raise RunError(f"token-table: {path} was not built from the v2 run's train captures and alignment "
                           f"({', '.join(problems)})")
        return {key: metadata.get(key) for key in ("rows", "unique_tokens", "documents", "split_target_tokens")}
    finally:
        table.close()


def stage_token_table(run):
    rel = f"{LONG}/token-table.npz"
    target = run.path(rel)
    adopted = False
    if run.exists(rel):
        adopted = True
    elif run.args.token_table is not None:
        source = run.args.token_table.resolve()
        check_token_table(run, source)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        adopted = True
    else:
        started = time.perf_counter()
        target_set = CaptureSet(v2_target_paths(run))
        alignment = load_alignment(run.v2("alignment/alignment.npz"))

        def progress(item):
            run.event("progress", stage="token-table", part=f"{item['kind']}-{item['layer']}",
                      seconds=item["seconds"])

        try:
            table = handoff.build_token_table(target_set, alignment, split="train", progress=progress)
        except ValueError as exc:
            raise RunError(f"token-table: {exc}") from None
        provenance = {"alignment": {"path": str(run.v2("alignment/alignment.npz")),
                                    "sha256": run.sha(run.v2("alignment/alignment.npz"))},
                      "target_captures": [{"path": str(path), "report_sha256": run.sha(path / "report.json"),
                                           "model_sha256": (capture.manifest or {}).get("model_sha256")}
                                          for path, capture in zip(v2_target_paths(run), target_set.captures)],
                      "code_sha256": {name: sha256_file(HERE / name) for name in ("handoff.py", "align.py",
                                                                                  "captures.py")},
                      "seconds": round(time.perf_counter() - started, 3)}
        handoff.write_token_table(target, table, provenance)
    metrics = check_token_table(run, target)
    metrics["adopted_from"] = None if run.args.token_table is None else str(run.args.token_table.resolve())
    return [rel], metrics, adopted


# -- save-state stages ------------------------------------------------------------------------

def state_items(run, role):
    """[(document plan, save_at)] for one model."""
    items = []
    for document in run.plan()["documents"]:
        if role == "source":
            save_at = [document["qwen_prefix"]]
        else:
            save_at = sorted({c["t0"] for c in document["conditions"] if c["native_state"]})
        if save_at:
            items.append((document, save_at))
    return items


def check_state_group(run, stage, role, directory, group):
    report = read_json(directory / "report.json")
    device = run.check_probe_report(report, stage, "save-state", role)
    documents = report.get("documents")
    if [item.get("document_id") for item in documents or []] != [doc["document_id"] for doc, _ in group]:
        raise RunError(f"{stage}: {directory} holds other documents than planned")
    decoded, states = 0, {}
    for (doc, save_at), item in zip(group, documents):
        saved = [state.get("n_tokens") for state in item.get("states") or []]
        if saved != save_at:
            raise RunError(f"{stage}: {directory} saved {saved} for {doc['document_id'][:12]}, planned {save_at}")
        for state in item["states"]:
            path = directory / state["file"]
            if (path.is_symlink() or not path.is_file() or path.stat().st_size != state["bytes"]
                    or state["pos_max"] != state["n_tokens"] - 1):
                raise RunError(f"{stage}: state {path} is missing or inconsistent with the report")
            states[doc["document_id"], state["n_tokens"]] = str(path)
        decoded += item["prefill_tokens"]
    return {"decoded_tokens": decoded, "device": device, "states": states}


def state_paths(run, role):
    """{(document_id, n_tokens): absolute state path} over the published groups of a save-state stage."""
    stage = "qwen-states" if role == "source" else "muse-states"
    paths = {}
    items = state_items(run, role)
    for index, group in enumerate(chunks(items, run.args.docs_per_run)):
        paths.update(check_state_group(run, stage, role, run.path(f"{LONG}/states/{role}/group-{index:03d}"),
                                       group)["states"])
    return paths


def stage_states(run, stage):
    role = STATE_STAGES[stage]
    a = run.args
    items = state_items(run, role)
    if not items:
        raise RunError(f"{stage}: no document needs a {role} state")
    groups = chunks(items, a.docs_per_run)
    records = run.rendered(role)
    total = sum(save_at[-1] for _, save_at in items)
    rels = [f"{LONG}/states/{role}/group-{index:03d}" for index in range(len(groups))]
    pending = [index for index, rel in enumerate(rels) if not run.exists(rel)]
    pending_tokens = sum(save_at[-1] for index in pending for _, save_at in groups[index])
    if pending:
        spec = run.spec(role)
        disk_guard(run, stage, sum(state_file_bytes(spec, a.cache_type, n, spec.n_swa + a.batch if spec.n_swa else None)
                                   for index in pending for _, save_at in groups[index] for n in save_at))
        eta(run, stage, pending_tokens, run.rate(role), "prior --" + role + "-tokens-per-second",
            groups=len(pending), documents=sum(len(groups[index]) for index in pending))
    progress = run_v2.Progress(run, stage, pending_tokens, SAVED, lambda m: int(m.group(3)), role)
    run.path(f"{LONG}/states/{role}").mkdir(parents=True, exist_ok=True)
    seconds, decoded_new, reused, devices = 0.0, 0, [], []
    for index, group in enumerate(groups):
        rel = rels[index]
        key = run.key(stage, documents=[[doc["document_id"], save_at] for doc, save_at in group])
        if run.exists(rel):
            found = read_json(run.path(rel) / "group.json") if (run.path(rel) / "group.json").is_file() else {}
            if found.get("key") != key:
                raise RunError(f"{stage}: {run.path(rel)} was produced with other settings, inputs or documents; "
                               "remove it or use a new --output")
            info = check_state_group(run, stage, role, run.path(rel), group)
            devices.append(info["device"])
            reused.append(rel)
            run.event("group_reuse", stage=stage, group=rel)
            continue
        # every group loads the model again, and the live servers may have been restarted meanwhile
        run.memory_guard(stage, role)
        staging = staging_directory(run, stage)
        try:
            inputs = []
            for doc, save_at in group:
                record = records[doc["document_id"]]
                inputs.append({"document_id": doc["document_id"], "text": record["text"], "save_at": save_at,
                               "expected_token_sha256": doc["qwen_token_sha256" if role == "source"
                                                            else "muse_token_sha256"]})
            elapsed = run.run_probe(stage, "save-state", role, inputs, staging / "raw", f"long-{stage}.log",
                                    on_line=progress)
            info = check_state_group(run, stage, role, staging / "raw", group)
            (staging / "raw" / "group.json").write_text(json_text(
                {"key": key, "documents": [doc["document_id"] for doc, _ in group], "seconds": elapsed,
                 "decoded_tokens": info["decoded_tokens"], "device": info["device"], "finished": now_iso()}),
                encoding="utf-8")
            publish(staging / "raw", run.path(rel))
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        seconds += elapsed
        decoded_new += info["decoded_tokens"]
        devices.append(info["device"])
        run.event("group_done", stage=stage, group=rel, seconds=round(elapsed, 1), tokens=info["decoded_tokens"])
    sizes = [path.stat().st_size for rel in rels for path in run.path(rel).rglob("*.state")]
    metrics = {"documents": len(items), "groups": len(groups), "reused_groups": reused, "tokens": total,
               "decoded_tokens_this_invocation": decoded_new, "seconds": seconds if decoded_new else None,
               "tokens_per_second": decoded_new / seconds if decoded_new and seconds else None,
               "state_files": len(sizes), "state_bytes": sum(sizes), "max_state_bytes": max(sizes),
               "device": next((item for item in devices if item), None)}
    return [f"{LONG}/states/{role}"], metrics, False


# -- build --------------------------------------------------------------------------------------

class CachingMapper:
    """A mapper for handoff.build_handoff that memoizes predictions across the states of a document.

    build_handoff maps source-row chunks (the same dict object for every target layer of a chunk);
    predictions are a pure function of those rows, so they are cached by a digest of the rows'
    bytes. States of one document share most chunks (oracle and table fill map the same content
    positions; the base cache of a shorter T0 is a prefix of a longer one's). clear() (called by
    build_handoff after each state) keeps everything; release() drops the predictions (between
    documents) and unload() also the loaded maps.
    """

    def __init__(self, mapper):
        self.inner = mapper
        self._predictions = {}
        self._last = (None, None)
        self.hits = self.misses = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _digest(self, rows):
        if rows is not self._last[0]:
            state = hashlib.blake2b(digest_size=20)
            for layer in sorted(rows):
                array = np.ascontiguousarray(rows[layer])
                state.update(f"{layer}:{array.dtype.str}:{array.shape};".encode())
                state.update(memoryview(array).cast("B"))
            self._last = (rows, state.hexdigest())
        return self._last[1]

    def predict(self, kind, layer, source_rows, target_shape):
        key = (kind, layer, tuple(target_shape), self._digest(source_rows))
        value = self._predictions.get(key)
        if value is None:
            self.misses += 1
            value = self.inner.predict(kind, layer, source_rows, target_shape)
            value.flags.writeable = False
            self._predictions[key] = value
        else:
            self.hits += 1
        return value

    def clear(self):
        pass

    def release(self):
        self._predictions.clear()
        self._last = (None, None)

    def unload(self):
        self.release()
        self.inner.clear()


def build_document(run, document, staging, mappers, table, qwen_path, native_paths, swa_cells):
    """Build every planned state of one document into `staging`; returns the per-condition reports."""
    a = run.args
    muse_spec, qwen_spec = run.spec("target"), run.spec("source")
    document_id = document["document_id"]
    q_record, m_record = run.rendered("source")[document_id], run.rendered("target")[document_id]
    q_tokens, m_tokens = run.tokens("source", document_id), run.tokens("target", document_id)
    qwen_state = kvstate.read_file(qwen_path, qwen_spec)
    reports = []
    wanted = [c for c in document["conditions"] if c["kind"] != "sanity"]
    order = sorted(wanted, key=lambda c: (FAMILIES.index(c["family"] or "v2"), -c["t0"], c["name"]))
    native, family = {}, None
    try:
        for condition in order:
            if (condition["family"] or "v2") != family:
                if family is not None:
                    mappers[family].release()
                family = condition["family"] or "v2"
            t0 = condition["t0"]
            oracle_state = None
            if condition["native_state"]:
                if t0 not in native:
                    native.clear()
                    data = Path(native_paths[document_id, t0]).read_bytes()
                    native[t0] = (data, kvstate.parse_file(data, muse_spec))
                oracle_state = native[t0][1]
            mapper = mappers[family]
            started = time.perf_counter()
            result = handoff.build_handoff(
                qwen_state, q_record, q_tokens, m_record, m_tokens, mapper, t0=t0, muse_spec=muse_spec,
                swa_cells=swa_cells, content_policy=condition["content"], fill_policy=condition["fill"],
                oracle_state=oracle_state, table=table, dtype=a.cache_type, v_trans=a.flash_attn == "off",
                chunk_rows=a.chunk_rows)
            data = result.state.to_bytes()
            kvstate.write_new(staging / f"{condition['name']}.state", data)
            report = dict(result.report, name=condition["name"], kind=condition["kind"], family=condition["family"],
                          tail=condition["tail"], bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                          seconds_including_write=round(time.perf_counter() - started, 3))
            if condition["kind"] == "oracle":
                report["identical_to_native_state"] = data == native[t0][0]
            (staging / f"{condition['name']}.json").write_text(json_text(report), encoding="utf-8")
            reports.append(report)
            run.event("built", stage="build", document=document_id[:12], condition=condition["name"],
                      t0=t0, seconds=report["seconds"], bytes=len(data))
    finally:
        native.clear()
        for mapper in mappers.values():
            mapper.release()
    return reports


def stage_build(run):
    a = run.args
    muse_spec = run.spec("target")
    swa_cells = a.swa_cells or muse_spec.n_swa
    import attention_eval
    mapper_dirs = {"v2": run.v2("mapper")}
    if a.grow_train:
        mapper_dirs["v3"] = run.path("refit/mapper")
    table = handoff.TokenTable(run.path(f"{LONG}/token-table.npz"))
    qwen_paths, native_paths = state_paths(run, "source"), state_paths(run, "target")
    run.path(f"{LONG}/handoff").mkdir(parents=True, exist_ok=True)
    documents = [doc for doc in run.documents() if any(c["kind"] != "sanity" for c in doc["conditions"])]
    disk_guard(run, "build", sum(state_file_bytes(muse_spec, a.cache_type, c["t0"], swa_cells)
                                 for doc in documents if not run.exists(f"{LONG}/handoff/{doc['document_id']}")
                                 for c in doc["conditions"] if c["kind"] != "sanity"))
    total_rows = sum(c["t0"] for doc in documents for c in doc["conditions"] if c["kind"] != "sanity")
    done_rows, seconds, built, reused, cache = 0, 0.0, 0, [], {"hits": 0, "misses": 0}
    mappers = {family: CachingMapper(attention_eval.Mapper(path)) for family, path in mapper_dirs.items()}
    try:
        for number, document in enumerate(documents, 1):
            document_id = document["document_id"]
            rel = f"{LONG}/handoff/{document_id}"
            rows = sum(c["t0"] for c in document["conditions"] if c["kind"] != "sanity")
            key = run.key("build", document=document)
            if run.exists(rel):
                found = run.path(rel) / "document.json"
                if not found.is_file() or read_json(found).get("key") != key:
                    raise RunError(f"build: {run.path(rel)} was built with other settings or inputs; remove it or "
                                   "use a new --output")
                reused.append(document_id)
                done_rows += rows
                continue
            before = {family: (mapper.hits, mapper.misses) for family, mapper in mappers.items()}
            staging = staging_directory(run, "build")
            started = time.perf_counter()
            try:
                reports = build_document(run, document, staging, mappers, table,
                                         qwen_paths[document_id, document["qwen_prefix"]], native_paths, swa_cells)
                elapsed = time.perf_counter() - started
                hits = sum(m.hits - before[f][0] for f, m in mappers.items())
                misses = sum(m.misses - before[f][1] for f, m in mappers.items())
                (staging / "document.json").write_text(json_text(
                    {"key": key, "document_id": document_id, "conditions": [r["name"] for r in reports],
                     "seconds": round(elapsed, 3), "prediction_cache": {"hits": hits, "misses": misses},
                     "finished": now_iso()}), encoding="utf-8")
                publish(staging, run.path(rel))
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
            cache["hits"] += hits
            cache["misses"] += misses
            built += 1
            seconds += elapsed
            done_rows += rows
            rate = done_rows / seconds if seconds and built else None
            new_rows = sum(sum(c["t0"] for c in d["conditions"] if c["kind"] != "sanity")
                           for d in documents[number:])
            run.event("progress", stage="build", part="documents", unit="state rows", done=done_rows,
                      total=total_rows, documents=f"{number}/{len(documents)}", seconds=round(elapsed, 1),
                      eta_seconds=round(new_rows / (rows / elapsed)) if elapsed else None)
    finally:
        table.close()
        for mapper in mappers.values():
            mapper.unload()
    sizes = [path.stat().st_size for path in run.path(f"{LONG}/handoff").rglob("*.state")]
    metrics = {"documents": len(documents), "built_documents": built, "reused_documents": len(reused),
               "states": len(sizes), "state_bytes": sum(sizes), "seconds": seconds if built else None,
               "rows": total_rows, "swa_cells": swa_cells, "prediction_cache": cache,
               "blas_threads": {name: os.environ.get(name) for name in THREAD_VARIABLES}}
    return [f"{LONG}/handoff"], metrics, False


# -- score --------------------------------------------------------------------------------------

def score_conditions(run, document, native_paths):
    conditions = [{"name": "native", "state_file": None}]
    for condition in document["conditions"]:
        if condition["kind"] == "sanity":
            path = native_paths[document["document_id"], condition["t0"]]
        else:
            path = run.path(f"{LONG}/handoff/{document['document_id']}/{condition['name']}.state")
        conditions.append({"name": condition["name"], "state_file": str(path)})
    return conditions


def check_score_group(run, directory, group):
    a = run.args
    results = read_json(directory / "results.json")
    device = run.check_probe_report(results, "score", "seed-score", "target",
                                    {"max_scored": a.max_scored, "decode": "through_last_needed_ubatch"})
    documents = results.get("documents") or []
    if [item.get("document_id") for item in documents] != [doc["document_id"] for doc in group]:
        raise RunError(f"score: {directory} holds other documents than planned")
    decoded, checks = 0, {}
    for doc, item in zip(group, documents):
        kinds = {condition["name"]: condition["kind"] for condition in doc["conditions"]}
        by_name = {condition["name"]: condition for condition in item.get("conditions") or []}
        names = ["native"] + [condition["name"] for condition in doc["conditions"]]
        if [condition.get("name") for condition in item["conditions"]] != names:
            raise RunError(f"score: conditions of {doc['document_id'][:12]} differ from the plan")
        if item.get("n_scored") != doc["n_scored"]:
            raise RunError(f"score: {doc['document_id'][:12]} scored {item.get('n_scored')} tokens, planned "
                           f"{doc['n_scored']}")
        for condition in doc["conditions"]:
            found = by_name[condition["name"]]
            if (found.get("seed_tokens") != condition["t0"] or found.get("restored_pos_max") != condition["t0"] - 1):
                raise RunError(f"score: {condition['name']} of {doc['document_id'][:12]} restored "
                               f"{found.get('seed_tokens')} tokens ending at {found.get('restored_pos_max')}, "
                               f"planned {condition['t0']}")
        arrays = {}
        for name in names:
            path = directory / "scores" / doc["document_id"] / f"{name}.npy"
            scores = np.load(path, allow_pickle=False)
            if scores.dtype.str != "<f4" or scores.shape != (doc["n_scored"], 3) or not np.isfinite(scores).all():
                raise RunError(f"score: {path} is malformed")
            arrays[name] = scores.astype(np.float64)
            decoded += by_name[name].get("decoded_tokens", 0)
        for name, scores in arrays.items():
            if kinds.get(name) in CHECK_KINDS:
                checks[f"{doc['document_id'][:12]}/{name}"] = {
                    "max_token_kl": float(scores[:, 1].max()), "mean_kl": float(scores[:, 1].mean()),
                    "nll_delta": float(np.mean(scores[:, 0] - arrays["native"][:, 0])),
                    "top1_agree": float(scores[:, 2].mean())}
    return {"decoded_tokens": decoded, "device": device, "checks": checks}


SANITY_GATES = (("max_token_kl", "sanity_kl", "--sanity-kl"), ("mean_kl", "sanity_mean_kl", "--sanity-mean-kl"),
                ("nll_delta", "sanity_nll", "--sanity-nll"))


def sanity_verdict(run, checks):
    """The native-state checks against the gates (per document and check: max token KL, mean KL, |mean dNLL|)."""
    if not checks:
        return None
    a = run.args
    out = {"checks": len(checks), "thresholds": {}, "failures": []}
    for statistic, attribute, flag in SANITY_GATES:
        values = {name: abs(item[statistic]) for name, item in checks.items()}
        worst = max(values, key=values.get)
        threshold = getattr(a, attribute)
        out[statistic] = {"max": values[worst], "worst": worst}
        out["thresholds"][statistic] = threshold
        out["failures"] += [{"check": name, "statistic": statistic, "value": value, "threshold": threshold,
                             "flag": flag} for name, value in sorted(values.items()) if value > threshold]
    out["max_token_kl"] = out["max_token_kl"]["max"]
    out["exact"] = all(item["max_token_kl"] == 0.0 and item["nll_delta"] == 0.0 for item in checks.values())
    out["passed"] = not out["failures"]
    return out


def sanity_error(sanity, where):
    shown = "; ".join(f"{item['check']} {item['statistic']} {item['value']:.3g} > {item['flag']} {item['threshold']:g}"
                      for item in sanity["failures"][:4])
    more = len(sanity["failures"]) - 4
    return (f"score: the native-state checks changed predictions ({shown}{f'; {more} more' if more > 0 else ''}); "
            f"the restore path is not a no-op at these gates, so no handoff number can be trusted. The scores are "
            f"in {where} (published: the gates are not part of the group key). Investigate; if the differences "
            f"are numerical noise, rerun the same command with looser gates to accept them without re-scoring")


def stage_score(run):
    a = run.args
    native_paths = state_paths(run, "target")
    records = run.rendered("target")
    documents = [doc for doc in run.documents() if doc["conditions"]]
    groups = chunks(documents, a.docs_per_run)
    rels = [f"{LONG}/scores/group-{index:03d}" for index in range(len(groups))]

    def planned(doc):
        return score_tokens(doc, a.batch)

    pending = [index for index, rel in enumerate(rels) if not run.exists(rel)]
    pending_tokens = sum(planned(doc) for index in pending for doc in groups[index])
    if pending:
        eta(run, "score", pending_tokens, run.rate("target"), "prior --target-tokens-per-second",
            groups=len(pending), documents=sum(len(groups[index]) for index in pending),
            conditions=sum(len(doc["conditions"]) for index in pending for doc in groups[index]))
    progress = run_v2.Progress(run, "score", pending_tokens, run_v2.SCORED, lambda m: int(m.group(3)), "seed-score")
    run.path(f"{LONG}/scores").mkdir(parents=True, exist_ok=True)
    seconds, decoded_new, reused, checks, devices = 0.0, 0, [], {}, []
    for index, group in enumerate(groups):
        rel = rels[index]
        key = run.key("score", documents=[doc["document_id"] for doc in group])
        if run.exists(rel):
            found = read_json(run.path(rel) / "group.json") if (run.path(rel) / "group.json").is_file() else {}
            if found.get("key") != key:
                raise RunError(f"score: {run.path(rel)} was scored with other settings, inputs or documents; "
                               "remove it or use a new --output")
            info = check_score_group(run, run.path(rel), group)
            reused.append(rel)
            run.event("group_reuse", stage="score", group=rel)
        else:
            run.memory_guard("score", "target")
            staging = staging_directory(run, "score")
            try:
                inputs = [{"document_id": doc["document_id"], "text": records[doc["document_id"]]["text"],
                           "expected_token_sha256": doc["muse_token_sha256"],
                           "score_ranges": [[doc["final_start_byte"], doc["text_bytes"]]],
                           "conditions": score_conditions(run, doc, native_paths)} for doc in group]
                elapsed = run.run_probe("score", "seed-score", "target", inputs, staging / "raw", "long-score.log",
                                        extra=["--max-scored", a.max_scored], on_line=progress)
                info = check_score_group(run, staging / "raw", group)
                (staging / "raw" / "group.json").write_text(json_text(
                    {"key": key, "documents": [doc["document_id"] for doc in group], "seconds": elapsed,
                     "decoded_tokens": info["decoded_tokens"], "device": info["device"], "finished": now_iso()}),
                    encoding="utf-8")
                publish(staging / "raw", run.path(rel))
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            seconds += elapsed
            decoded_new += info["decoded_tokens"]
            run.event("group_done", stage="score", group=rel, seconds=round(elapsed, 1),
                      tokens=info["decoded_tokens"])
        # the gates apply to every group, reused ones too, and stop the stage at the first failing group
        group_sanity = sanity_verdict(run, info["checks"])
        if group_sanity is not None:
            run.event("sanity", stage="score", group=rel, passed=group_sanity["passed"],
                      **{statistic: group_sanity[statistic] for statistic in ("mean_kl", "nll_delta")},
                      max_token_kl=group_sanity["max_token_kl"])
            if not group_sanity["passed"]:
                raise RunError(sanity_error(group_sanity, run.path(rel)))
        checks.update(info["checks"])
        devices.append(info["device"])
    sanity = sanity_verdict(run, checks)
    metrics = {"documents": len(documents), "groups": len(groups), "reused_groups": reused,
               "conditions": sum(len(doc["conditions"]) for doc in documents), "decoded_tokens_planned":
               sum(planned(doc) for doc in documents), "decoded_tokens_this_invocation": decoded_new,
               "seconds": seconds if decoded_new else None,
               "tokens_per_second": decoded_new / seconds if decoded_new and seconds else None,
               "sanity": sanity, "checks": checks, "device": next((item for item in devices if item), None)}
    return [f"{LONG}/scores"], metrics, False


# -- report -------------------------------------------------------------------------------------

def load_scores(run):
    """{document_id: {name: float64 [S, 3]}} over the score groups, native included."""
    a = run.args
    documents = [doc for doc in run.documents() if doc["conditions"]]
    out = {}
    for index, group in enumerate(chunks(documents, a.docs_per_run)):
        directory = run.path(f"{LONG}/scores/group-{index:03d}")
        results = read_json(directory / "results.json")
        for doc, item in zip(group, results["documents"]):
            scores = {}
            for condition in item["conditions"]:
                array = np.load(directory / "scores" / doc["document_id"] / f"{condition['name']}.npy")
                if not np.isclose(float(np.mean(array[:, 0], dtype=np.float64)), condition["nll_mean"],
                                  rtol=1e-4, atol=1e-5):
                    raise RunError(f"report: scores of {doc['document_id'][:12]}/{condition['name']} disagree "
                                   "with results.json")
                scores[condition["name"]] = array.astype(np.float64)
            native = scores["native"]
            if np.any(native[:, 1] != 0) or np.any(native[:, 2] != 1):
                raise RunError(f"report: native scores of {doc['document_id'][:12]} are not self-consistent")
            out[doc["document_id"]] = scores
    return out


def summarize_condition(scores, name, members, seed, resamples):
    import inject
    present = [document_id for document_id in members if name in scores[document_id]]
    if not present:
        return None
    counts = [len(scores[d]["native"]) for d in present]
    delta = [float(np.sum(scores[d][name][:, 0] - scores[d]["native"][:, 0])) for d in present]
    kl = [float(np.sum(scores[d][name][:, 1])) for d in present]
    top1 = [float(np.sum(scores[d][name][:, 2])) for d in present]
    return {"name": name, "documents": len(present), "tokens": int(sum(counts)),
            "nll_delta": inject._bootstrap(delta, counts, seed, resamples),
            "kl": dict(inject._bootstrap(kl, counts, seed, resamples),
                       max_token=float(max(scores[d][name][:, 1].max() for d in present))),
            "top1_agree": inject._bootstrap(top1, counts, seed, resamples)}


def contrast(scores, name, baseline, members, seed, resamples):
    import inject
    present = [d for d in members if name in scores[d] and baseline in scores[d]]
    if not present:
        return None
    counts = [len(scores[d]["native"]) for d in present]
    item = {"name": name, "baseline": baseline, "documents": len(present), "tokens": int(sum(counts))}
    for metric, column in (("nll", 0), ("kl", 1), ("top1_agree", 2)):
        item[metric] = inject._bootstrap([float(np.sum(scores[d][name][:, column] - scores[d][baseline][:, column]))
                                          for d in present], counts, seed, resamples)
    return item


def slope(scores, name, documents, seed, resamples):
    """OLS slope of per-document mean delta NLL on P (per 1K tokens) with a document bootstrap CI."""
    points = [(doc["prefix"] / 1000.0, float(np.mean(scores[doc["document_id"]][name][:, 0]
                                                     - scores[doc["document_id"]]["native"][:, 0])))
              for doc in documents if name in scores[doc["document_id"]]]
    if len(points) < 3:
        return None
    x, y = np.asarray(points).T

    def fit(xs, ys):
        spread = np.sum((xs - xs.mean()) ** 2)
        return float(np.sum((xs - xs.mean()) * (ys - ys.mean())) / spread) if spread > 0 else math.nan

    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(x), size=(resamples, len(x)))
    draws = np.asarray([fit(x[i], y[i]) for i in index])
    draws = draws[np.isfinite(draws)]
    correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else None
    return {"name": name, "documents": len(x), "slope_per_1k_tokens": fit(x, y),
            "ci95": [float(v) for v in np.percentile(draws, [2.5, 97.5])] if len(draws) else None,
            "pearson_r": correlation, "prefix_range": [float(x.min() * 1000), float(x.max() * 1000)]}


def verdict(item, target):
    """Against the ΔNLL target: meets (CI below), misses (CI above) or inconclusive (CI straddles).

    `handoff_fraction` (mean T0/P over the documents) says how much of the prefix the condition hands over.
    """
    value = item["nll_delta"]
    ci = value.get("ci95")
    if ci is None:
        status = "meets" if value["mean"] < target else "misses"
        status += " (no CI: one document)"
    elif ci[1] < target:
        status = "meets"
    elif ci[0] > target:
        status = "misses"
    else:
        status = "inconclusive"
    return {"condition": item["name"], "mean": value["mean"], "ci95": ci, "target": target,
            "gap": value["mean"] - target, "status": status, "documents": item["documents"],
            "handoff_fraction": item.get("handoff_fraction")}


def report_contrasts(conditions, families):
    """(condition, baseline) pairs: vs mean_old_W, table_fill vs mean_table_W (its information-free floor), vs
    oracle_fill_W, mapper v3 vs v2, tail vs the shortest tail."""
    by = {(c["kind"], c["tail"], c["family"]): name for name, c in conditions.items()}
    tails = sorted({c["tail"] for c in conditions.values()})
    pairs = []
    for tail in tails:
        floor = by.get(("mean_old", tail, None))
        for name, c in conditions.items():
            if floor and c["tail"] == tail and c["kind"] not in CHECK_KINDS + ("mean_old",):
                pairs.append((name, floor))
        table_floor = by.get(("mean_table", tail, None))
        for family in families:
            if table_floor and ("table_fill", tail, family) in by:
                pairs.append((by["table_fill", tail, family], table_floor))
        for family in families:
            upper = by.get(("oracle_fill", tail, family))
            for key in (("table_fill", tail, family), ("mean_oracle", tail, None)):
                if upper and key in by:
                    pairs.append((by[key], upper))
        for kind in ("oracle_fill", "table_fill"):
            if (kind, tail, "v3") in by and (kind, tail, "v2") in by:
                pairs.append((by[kind, tail, "v3"], by[kind, tail, "v2"]))
    positive = [tail for tail in tails if tail > 0]
    for family in families:
        if positive and ("table_fill", positive[0], family) in by:
            for tail in tails:
                if tail != positive[0] and ("table_fill", tail, family) in by:
                    pairs.append((by["table_fill", tail, family], by["table_fill", positive[0], family]))
    return list(dict.fromkeys(pairs))


def stage_report(run):
    a = run.args
    plan = run.plan()
    scores = load_scores(run)
    documents = [doc for doc in run.documents() if doc["document_id"] in scores]
    order = []
    for doc in documents:
        for condition in doc["conditions"]:
            if condition["name"] not in order:
                order.append(condition["name"])
    kinds = {c["name"]: c for doc in documents for c in doc["conditions"]}
    # tail, then kind, then family (extra kinds are appended to each document's planned conditions)
    order.sort(key=lambda name: (kinds[name]["tail"], list(CONDITION_KINDS).index(kinds[name]["kind"]),
                                 FAMILIES.index(kinds[name]["family"] or "v2")))
    groups = {"all": [doc["document_id"] for doc in documents]}
    for doc in documents:
        groups.setdefault(f"bucket {doc['bucket']}", []).append(doc["document_id"])
    pairs = report_contrasts(kinds, plan["families"])
    sections = {}
    for group, members in groups.items():
        sections[group] = {
            "documents": len(members), "prefix_tokens": [min(doc["prefix"] for doc in documents
                                                             if doc["document_id"] in members),
                                                         max(doc["prefix"] for doc in documents
                                                             if doc["document_id"] in members)],
            "conditions": [item for item in (summarize_condition(scores, name, members, a.seed, a.bootstrap)
                                             for name in order) if item],
            "contrasts": [item for item in (contrast(scores, name, base, members, a.seed, a.bootstrap)
                                            for name, base in pairs) if item]}
    t0_of = {(doc["document_id"], c["name"]): c["t0"] / doc["prefix"] for doc in documents for c in doc["conditions"]}
    for group, section in sections.items():
        for item in section["conditions"]:
            condition = kinds[item["name"]]
            fractions = [t0_of[d, item["name"]] for d in groups[group] if (d, item["name"]) in t0_of]
            item.update(kind=condition["kind"], family=condition["family"], tail=condition["tail"],
                        role=CONDITION_KINDS[condition["kind"]][5],
                        handoff_fraction=float(np.mean(fractions)) if fractions else None)
    deployable = [item for item in sections["all"]["conditions"] if item["kind"] == DEPLOYABLE_KIND]
    verdicts = [verdict(item, a.target_nll) for item in deployable]
    # the headline condition is fixed before scoring (--headline); the best deployable one is reported as such
    wanted = run.headline_name()
    headline = next((dict(v, selection="pre-registered") for v in verdicts if v["condition"] == wanted), None)
    best = dict(min(verdicts, key=lambda v: v["mean"]), selection=f"selected post hoc as the best of {len(verdicts)}") \
        if verdicts else None
    if headline is None and best is not None:
        headline = dict(best, selection=f"{best['selection']}; the pre-registered {wanted} was not scored")
    by_bucket = {}
    if headline:
        for group, section in sections.items():
            if group.startswith("bucket "):
                item = next((c for c in section["conditions"] if c["name"] == headline["condition"]), None)
                if item:
                    by_bucket[group[len("bucket "):]] = verdict(item, a.target_nll)
    checks = [item for item in sections["all"]["conditions"] if item["kind"] in CHECK_KINDS]
    floors = [item for item in sections["all"]["conditions"] if item["kind"] in FLOOR_KINDS]
    control = {"condition_kinds": list(FLOOR_KINDS), "passed": any(item["kl"]["max_token"] > 0 for item in floors),
               "documents_changed": {item["name"]: sum(1 for d in groups["all"] if item["name"] in scores[d]
                                                       and scores[d][item["name"]][:, 1].max() > 0)
                                     for item in floors}} if floors else None
    per_document = []
    for doc in documents:
        native = scores[doc["document_id"]]["native"]
        per_document.append({
            "document_id": doc["document_id"], "conversation_id": doc["conversation_id"], "source": doc["source"],
            "bucket": doc["bucket"], "prefix": doc["prefix"], "qwen_prefix": doc["qwen_prefix"],
            "n_scored": doc["n_scored"], "native_nll": float(native[:, 0].mean()),
            "t0": {str(c["tail"]): c["t0"] for c in doc["conditions"]},
            "conditions": {name: {"nll_delta": float(np.mean(array[:, 0] - native[:, 0])),
                                  "kl": float(np.mean(array[:, 1])), "top1_agree": float(np.mean(array[:, 2]))}
                           for name, array in scores[doc["document_id"]].items() if name != "native"}})
    covariate = [item for item in (slope(scores, name, documents, a.seed, a.bootstrap) for name in order
                                   if kinds[name]["kind"] not in CHECK_KINDS) if item]
    build = {}
    for doc in documents:
        for condition in doc["conditions"]:
            path = run.path(f"{LONG}/handoff/{doc['document_id']}/{condition['name']}.json")
            if path.is_file():
                item = read_json(path)
                entry = build.setdefault(condition["name"], {"states": 0, "seconds": 0.0, "bytes": 0,
                                                              "identical_to_native_state": [], "table_cells": 0,
                                                              "table_unseen_cells": 0})
                entry["states"] += 1
                entry["seconds"] += item["seconds"]
                entry["bytes"] += item["bytes"]
                for section in (item.get("sections") or {}).values():
                    sources = section.get("sources") or {}
                    entry["table_cells"] += sources.get("table", 0) + sources.get("table_unseen", 0)
                    entry["table_unseen_cells"] += sources.get("table_unseen", 0)
                if "identical_to_native_state" in item:
                    entry["identical_to_native_state"].append(item["identical_to_native_state"])
    for entry in build.values():
        values = entry.pop("identical_to_native_state")
        entry["identical_to_native_state"] = f"{sum(values)}/{len(values)}" if values else None
        # token-table cells whose id was never seen in v2 train (they get the per-layer, mostly template, mean)
        entry["table_unseen_share"] = (entry["table_unseen_cells"] / entry["table_cells"]
                                       if entry["table_cells"] else None)
    timings = []
    for stage in run.stages[:-1]:
        record = run.records.get(stage)
        if record:
            metrics = record["metrics"]
            timings.append({"stage": stage, "adopted": record["adopted"], "wall_seconds": record["wall_seconds"],
                            "tokens": metrics.get("decoded_tokens_this_invocation", metrics.get("tokens")),
                            "tokens_per_second": metrics.get("tokens_per_second"),
                            "bytes": metrics.get("state_bytes")})
    prepare = run.records["prepare"]
    report = {
        "format_version": 1, "purpose": "kv_handoff_long_eval_report", "generated": now_iso(),
        "models": {role: {"id": run.model_id(role), "architecture": ROLES[role],
                          "sha256": prepare["inputs"][f"{role}_model_sha256"]} for role in ROLES},
        "corpus": plan["corpus"], "settings": {stage: run.records[stage]["settings"] for stage in run.stages[:-1]},
        "families": plan["families"], "target_nll": a.target_nll,
        "definitions": dict(plan["definitions"], nll_delta="condition NLL minus native NLL per scored token "
                            "(nats); token-pooled mean, document bootstrap 95% CI (shared seed)",
                            kl="KL(native || condition) per scored token",
                            handoff_fraction="mean over documents of T0 / P (the share of the prefix handed over)",
                            conditions={kind: {"content": value[0], "fill": value[1], "native_state": value[2],
                                               "per_family": value[3], "at_tail_0": value[4], "role": value[5],
                                               "planned_by": "prepare" if kind in a.conditions else
                                               "--extra-conditions"}
                                        for kind, value in CONDITION_KINDS.items()
                                        if any(c["kind"] == kind for c in kinds.values())}),
        "verdict": {"target": a.target_nll, "headline": headline, "best_post_hoc": best, "deployable": verdicts,
                    "headline_by_bucket": by_bucket},
        "sanity": run.records["score"]["metrics"].get("sanity"), "positive_control": control,
        "checks": checks, "groups": sections, "prefix_covariate": covariate, "per_document": per_document,
        "build": build, "grow_train": (run.records.get("grow-train") or {}).get("metrics"),
        "timings": timings, "caveats": CAVEATS,
        "stage_digests": {stage: run.records[stage]["outputs_sha256"] for stage in run.stages[:-1]},
    }
    for rel in (f"{LONG}/report.json", f"{LONG}/report.md"):
        if run.exists(rel):
            run.path(rel).unlink()
    write_new(run.path(f"{LONG}/report.json"), json_text(report))
    write_new(run.path(f"{LONG}/report.md"), markdown(report))
    run.event("verdict", **(headline or {}))
    return [f"{LONG}/report.json", f"{LONG}/report.md"], {"headline": headline, "sections": sorted(report)}, False


CAVEATS = [
    "Scoring is teacher-forced on the dataset's reference final message (its first --max-scored tokens), not free "
    "generation; delta NLL is condition minus Muse's own full prefill on the same tokens.",
    "table_fill is the deployable handoff (every cell from Qwen via the mapper or from the token table); "
    "oracle_fill uses Muse's native values for template/special/unmatched cells and is an upper bound. The "
    "information-free floors are mean_table (content mean, token-table fill: the floor of table_fill; "
    "table_fill minus mean_table is what the mapper adds) and mean_old (every cell the content mean, harsher). "
    "mean_oracle (content mean, native fill) is neither a floor nor v2's mean_all: its fill cells come from Muse's "
    "clean native state (templates plus the 2-5% unmatched or straddling content cells before T0), whereas v2 "
    "recomputed every unpatched cell on top of the patched context, so it is not comparable with v2's 0.648.",
    "The token table knows only the ids seen in v2 train non-aligned rows (mostly template/special rows; see the "
    "token-table stage metrics); unseen ids take the per-layer mean of those rows, except unseen ids of content "
    "tokens (most unmatched content cells of long documents), which take the mapper's content mean "
    "(table_unseen_share in the build table counts both). The table_fill minus oracle_fill gap includes this.",
    "With a tail W >= the sliding window (2048), sliding-layer state cells are never attended by the scored "
    "tokens directly; they act through the natively decoded tail. T0 is rounded down to the --batch grid, so the "
    "actual tail is W to W + batch - 1 tokens (t0 per document is in per_document). W = 0 hands over P - 1 tokens.",
    "The sanity (native state) and oracle (handoff.py rebuild of it) conditions must reproduce the native "
    "scores within the gates (max token KL, mean KL and |mean delta NLL| per document); on the GPU with flash "
    "attention small differences are expected from summation order.",
    "Buckets confound length with content: 4-8K prefixes are multi-turn chat, 8K+ are LongAlign reading; the "
    "8-16K vs 16-29.5K comparison is the cleaner length contrast, and prefix_covariate gives the slope over P.",
    "Documents are few (10 per bucket); bootstrap intervals resample documents. Paired contrasts share "
    "documents and resample indices and are the way to compare conditions.",
    "The headline condition is fixed before scoring (--headline, default table_fill_2048 of the newest mapper "
    "family); the best deployable condition is also shown but was selected after the fact. Longer tails hand over "
    "less: see handoff_fraction (mean T0/P), which is small for W = 4096 in the 4-8K bucket.",
    "States come from --device (Vulkan by default) with the deployment cache layout (f16, flash attention); the "
    "v2 mapper was fitted on CPU F32 captures, and GPU K/V differ from CPU ones by 2-5% relative, so the mapper "
    "sees slightly shifted Qwen inputs. The native reference and every condition are scored on the same device.",
    "The v3 family's mapper adds GPU-collected extra train data (Vulkan numerics differ from the CPU v2 captures "
    "by 2-5% relative); it shares the v2 token table and floors, so v3 minus v2 isolates the mapper.",
]


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}" if math.isfinite(value) else "n/a"
    return str(value)


def _sci(value):
    return "n/a" if value is None else f"{value:.3g}"


def _ci(item, digits=4):
    ci = item.get("ci95")
    return f"{item['mean']:.{digits}f}" + ("" if ci is None else f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]")


def markdown(report):
    verdict_section = report["verdict"]
    headline = verdict_section["headline"]
    lines = ["# Long-context cache handoff (Qwen -> Muse, design A)", "",
             f"Source {report['models']['source']['id']} -> target {report['models']['target']['id']}; "
             f"families {', '.join(report['families'])}; generated {report['generated']}.", "", "## Verdict", ""]
    if headline:
        best = verdict_section.get("best_post_hoc")
        lines += [f"Target: delta NLL < {report['target_nll']} nats/token. Headline condition "
                  f"**{headline['condition']}** ({headline['selection']}): {headline['mean']:.4f}"
                  + (f" [{headline['ci95'][0]:.4f}, {headline['ci95'][1]:.4f}]" if headline["ci95"] else "")
                  + f" over {headline['documents']} documents, handing over {_fmt(headline['handoff_fraction'], 2)} "
                  f"of the prefix on average -> **{headline['status']}** (gap {headline['gap']:+.4f}).", ""]
        if best and best["condition"] != headline["condition"]:
            lines += [f"Best deployable condition, {best['selection']}: {best['condition']} {best['mean']:.4f} "
                      f"({best['status']}, T0/P {_fmt(best['handoff_fraction'], 2)}).", ""]
        lines += ["| deployable condition | docs | T0/P | delta NLL [95% CI] | status | gap to target |",
                  "|---|---|---|---|---|---|"]
        for item in verdict_section["deployable"]:
            ci = "" if item["ci95"] is None else f" [{item['ci95'][0]:.4f}, {item['ci95'][1]:.4f}]"
            lines.append(f"| {item['condition']} | {item['documents']} | {_fmt(item['handoff_fraction'], 2)} | "
                         f"{item['mean']:.4f}{ci} | {item['status']} | {item['gap']:+.4f} |")
        if verdict_section["headline_by_bucket"]:
            lines += ["", f"{headline['condition']} by prefix bucket:", "",
                      "| bucket | docs | T0/P | delta NLL [95% CI] | status |", "|---|---|---|---|---|"]
            for bucket, item in verdict_section["headline_by_bucket"].items():
                ci = "" if item["ci95"] is None else f" [{item['ci95'][0]:.4f}, {item['ci95'][1]:.4f}]"
                lines.append(f"| {bucket} | {item['documents']} | {_fmt(item['handoff_fraction'], 2)} | "
                             f"{item['mean']:.4f}{ci} | {item['status']} |")
    else:
        lines.append("No deployable (table_fill) condition was scored.")
    sanity = report.get("sanity") or {}
    control = report.get("positive_control") or {}
    worst = {statistic: (sanity.get(statistic) or {}) for statistic in ("mean_kl", "nll_delta")}
    lines += ["", "## Checks", "",
              f"Native-state checks (sanity = the native Muse state, oracle = handoff.py's rebuild of it), worst per "
              f"document over {sanity.get('checks')} checks: max token KL {_sci(sanity.get('max_token_kl'))}, "
              f"mean KL {_sci(worst['mean_kl'].get('max'))}, |mean delta NLL| {_sci(worst['nll_delta'].get('max'))} "
              f"(gates {sanity.get('thresholds')}; exact: {sanity.get('exact')}; passed: {sanity.get('passed')}). "
              f"Positive control (the floors change predictions): {control.get('passed')} "
              f"{control.get('documents_changed')}.", ""]
    for name, section in report["groups"].items():
        lines += [f"## {name[0].upper() + name[1:]}: {section['documents']} documents, prefix "
                  f"{section['prefix_tokens'][0]}-{section['prefix_tokens'][1]} tokens", "",
                  "| condition | role | docs | tokens | T0/P | delta NLL vs native | KL(native, cond) | max token KL "
                  "| top-1 |", "|---|---|---|---|---|---|---|---|---|"]
        for item in section["conditions"]:
            lines.append(f"| {item['name']} | {item['role'].split(':')[0]} | {item['documents']} | {item['tokens']} | "
                         f"{_fmt(item['handoff_fraction'], 2)} | {_ci(item['nll_delta'])} | {_ci(item['kl'])} | "
                         f"{item['kl']['max_token']:.3g} | {_ci(item['top1_agree'], 3)} |")
        if section["contrasts"]:
            lines += ["", "Paired contrasts (condition minus baseline, shared document resamples):", "",
                      "| condition | minus | docs | delta NLL | delta KL | delta top-1 |", "|---|---|---|---|---|---|"]
            for item in section["contrasts"]:
                lines.append(f"| {item['name']} | {item['baseline']} | {item['documents']} | {_ci(item['nll'])} | "
                             f"{_ci(item['kl'])} | {_ci(item['top1_agree'], 3)} |")
        lines.append("")
    if report["prefix_covariate"]:
        lines += ["## Delta NLL against prefix length", "",
                  "| condition | docs | slope per 1K tokens [95% CI] | Pearson r | P range |", "|---|---|---|---|---|"]
        for item in report["prefix_covariate"]:
            ci = "" if item["ci95"] is None else f" [{item['ci95'][0]:.4f}, {item['ci95'][1]:.4f}]"
            lines.append(f"| {item['name']} | {item['documents']} | {item['slope_per_1k_tokens']:.4f}{ci} | "
                         f"{_fmt(item['pearson_r'])} | {item['prefix_range'][0]:.0f}-{item['prefix_range'][1]:.0f} |")
        lines.append("")
    grow = report.get("grow_train")
    if grow and grow.get("r2"):
        lines += ["## Mapper v3 vs v2 (R^2 of the applied prediction)", "",
                  f"Aligned train rows: v2 {grow['r2']['rows']['v2']['train']['pairs']}, v3 "
                  f"{grow['r2']['rows']['v3']['train']['pairs']}.", "",
                  "| maps | dev v2 | dev v3 | test v2 | test v3 |", "|---|---|---|---|---|"]
        for key, value in grow["r2"]["means"].items():
            lines.append(f"| {key} | {value['dev']['v2']:.3f} | {value['dev']['v3']:.3f} | {value['test']['v2']:.3f} | "
                         f"{value['test']['v3']:.3f} |")
        lines.append("")
    names = [n for n in (report["groups"]["all"]["conditions"]) if n["kind"] not in CHECK_KINDS]
    names = [item["name"] for item in names]
    lines += ["## Per document (delta NLL / KL)", "",
              "| document | bucket | P | native NLL | " + " | ".join(names) + " |", "|" + "---|" * (4 + len(names))]
    for item in report["per_document"]:
        cells = " | ".join(f"{item['conditions'][n]['nll_delta']:.3f} / {item['conditions'][n]['kl']:.3f}"
                           if n in item["conditions"] else "n/a" for n in names)
        lines.append(f"| {item['document_id'][:10]} | {item['bucket']} | {item['prefix']} | "
                     f"{item['native_nll']:.3f} | {cells} |")
    lines += ["", "## Build", "", "| condition | states | seconds | GB | identical to native | table cells unseen |",
              "|---|---|---|---|---|---|"]
    for name, item in report["build"].items():
        lines.append(f"| {name} | {item['states']} | {item['seconds']:.1f} | {item['bytes'] / 1e9:.2f} | "
                     f"{_fmt(item['identical_to_native_state'])} | {_fmt(item['table_unseen_share'], 3)} |")
    lines += ["", "## Timings", "", "| stage | adopted | wall s | tokens | tokens/s | GB written |",
              "|---|---|---|---|---|---|"]
    for item in report["timings"]:
        lines.append(f"| {item['stage']} | {'yes' if item['adopted'] else 'no'} | {_fmt(item['wall_seconds'], 0)} | "
                     f"{_fmt(item['tokens'])} | {_fmt(item['tokens_per_second'], 1)} | "
                     f"{_fmt(None if item['bytes'] is None else item['bytes'] / 1e9, 2)} |")
    lines += ["", "## Caveats", ""] + [f"- {text}" for text in report["caveats"]] + [""]
    return "\n".join(lines)


STAGE_FUNCTIONS = {
    "prepare": stage_prepare, "grow-train": stage_grow_train, "token-table": stage_token_table,
    "qwen-states": lambda run: stage_states(run, "qwen-states"),
    "muse-states": lambda run: stage_states(run, "muse-states"),
    "build": stage_build, "score": stage_score, "report": stage_report,
}


def clean_staging(run):
    """Remove long/.staging leftovers of a killed run (the stage lock rules out a concurrent one)."""
    parent = run.path(f"{LONG}/.staging")
    leftovers = sorted(parent.iterdir()) if parent.is_dir() else []
    if leftovers:
        run.event("cleanup", removed=[path.name for path in leftovers],
                  bytes=sum(item.stat().st_size for path in leftovers for item in
                            ([path] if not path.is_dir() else path.rglob("*")) if item.is_file()))
        for path in leftovers:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()


def execute(run):
    a = run.args
    stages = run.stages
    clean_staging(run)
    last = stages.index(a.until) if a.until in stages else len(stages) - 1
    if not a.grow_train:
        run.event("stage_skip", stage="grow-train", reason="disabled (pass --grow-train for the v3 mapper family)")
    for stage in stages[:last + 1]:
        run.current = stage
        record = run.read_record(stage)
        if record is not None:
            run.records[stage] = record
            run.verify(stage, record)
            run.event("stage_skip", stage=stage, reason="complete and verified", adopted=record["adopted"])
            continue
        run.event("stage_start", stage=stage)
        started, clock = now_iso(), time.monotonic()
        outputs, metrics, adopted = STAGE_FUNCTIONS[stage](run)
        wall = time.monotonic() - clock
        record = run.write_record(stage, outputs, metrics, started, wall, adopted=adopted)
        run.event("stage_done", stage=stage, adopted=adopted, wall_seconds=round(wall, 1),
                  outputs_sha256=record["outputs_sha256"],
                  **{key: metrics[key] for key in ("tokens", "tokens_per_second", "decoded_tokens_this_invocation")
                     if metrics.get(key) is not None})
        run.current = None


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True, type=Path, help="run directory (new unless --resume)")
    parser.add_argument("--resume", action="store_true", help="continue an existing run directory")
    parser.add_argument("--until", choices=STAGES, default="report", help="last stage to run")
    parser.add_argument("--probe", required=True, type=Path,
                        help="kvpark-kv-probe binary (built against a Vulkan runtime for --device vulkan)")
    parser.add_argument("--device", choices=("vulkan", "cpu"), default="vulkan",
                        help="device of save-state, seed-score and grow-train collection. cpu is a fallback for a "
                             "reduced plan only: the device is part of the stage settings, so it needs a new "
                             "--output, and at ~17-19 tokens/s the full plan's score stage alone takes ~35 h")
    parser.add_argument("--corpus", required=True, type=Path, help="long-eval conversations.jsonl (split test)")
    parser.add_argument("--corpus-manifest", type=Path,
                        help="build_corpus.py manifest with long_eval.documents (default: next to --corpus)")
    parser.add_argument("--docs", choices=("primary", "all"), default="primary",
                        help="primary documents only (needs the manifest) or primaries and reserves")
    parser.add_argument("--docs-per-bucket", type=int, help="keep at most this many documents per bucket")
    parser.add_argument("--prefix-buckets", type=parse_buckets, default=[4096, 8192, 16384, 29500],
                        help="bucket edges on P when the corpus has no manifest")
    parser.add_argument("--v2-run", required=True, type=Path, help="run_v2 directory (captures, alignment, mapper)")
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-model", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--hash-cache-seed", type=Path,
                        help="model hash cache of another run to start RUN/model-hashes.json from (read only)")
    parser.add_argument("--tails", type=parse_tails, default=[0, 2048, 4096],
                        help="natively decoded tails W before P (0: only table_fill and mean_old)")
    parser.add_argument("--t0-grid", choices=("on", "off"), default="on",
                        help="round T0 = P - W down to the --batch grid (exact sanity on the CPU)")
    parser.add_argument("--min-handoff-tokens", type=int, default=512,
                        help="skip a tail for a document whose T0 would be smaller")
    parser.add_argument("--conditions", type=parse_kinds, default=list(PLAN_KINDS),
                        help=f"kinds planned by prepare: 'all' or comma-separated kinds of {list(CONDITION_KINDS)} "
                             f"(default {','.join(PLAN_KINDS)})")
    parser.add_argument("--extra-conditions", type=parse_extra_kinds, default=list(DEFAULT_EXTRA_KINDS),
                        help="kinds needing no native Muse state that build and score add to the plan's (planned "
                             "with the prepare settings; 'none' for none; default "
                             f"{','.join(DEFAULT_EXTRA_KINDS)}); changing them after build or score started "
                             "invalidates those stages")
    parser.add_argument("--swa-cells", type=int, help="sliding cells in built states (default: the model's window)")
    parser.add_argument("--cache-type", choices=("f16", "f32"), default="f16")
    parser.add_argument("--flash-attn", choices=("on", "off"), default="on")
    parser.add_argument("--ctx", type=int, default=32768, help="probe context (every document must fit)")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16, help="probe CPU threads")
    parser.add_argument("--build-threads", type=int, default=8, help="BLAS threads of build, align and fit")
    parser.add_argument("--chunk-rows", type=int, default=2048, help="handoff.py mapping chunk")
    parser.add_argument("--docs-per-run", type=int, default=3,
                        help="documents per probe invocation (a failure loses at most one group)")
    parser.add_argument("--max-scored", type=int, default=256)
    parser.add_argument("--sanity-kl", type=float,
                        help="gate: max token KL of each native-state check (sanity, oracle) per document "
                             "(default 1e-6 on the CPU, where T0 on the grid gives exactly 0; 1e-2 on the GPU, where "
                             "flash attention reorders sums). The gates are re-applied to reused score groups and "
                             "are not part of their key: loosening one never forces re-scoring")
    parser.add_argument("--sanity-mean-kl", type=float,
                        help="gate: mean KL per document of each check (default 1e-6 CPU, 1e-3 GPU)")
    parser.add_argument("--sanity-nll", type=float,
                        help="gate: |mean delta NLL| per document of each check (default 1e-6 CPU, 5e-3 GPU)")
    parser.add_argument("--expect-graph-splits", default="2",
                        help="graph splits every GPU report must show ('any' disables the check)")
    parser.add_argument("--grow-train", action="store_true", help="add the v3 mapper family (see grow-train)")
    parser.add_argument("--train-corpus", type=Path, help="corpus-v3-train conversations.jsonl (--grow-train)")
    parser.add_argument("--train-ctx", type=int, default=5120)
    parser.add_argument("--train-batch", type=int, default=512)
    parser.add_argument("--fit-ks", default=run_v2.DEFAULT_KS)
    parser.add_argument("--fit-alphas", help="fit_layers --alphas (default: its grid)")
    parser.add_argument("--token-table", type=Path, help="adopt this handoff.py token table (checked against v2)")
    parser.add_argument("--target-nll", type=float, default=0.05, help="delta NLL target (nats/token)")
    parser.add_argument("--headline",
                        help="pre-registered headline condition of the verdict (default: table_fill_2048 of the "
                             "newest mapper family, e.g. table_fill_2048_v3 with --grow-train)")
    parser.add_argument("--bootstrap", type=int, default=2000, help="document bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-tokens-per-second", type=float, help="ETA prior (default by device)")
    parser.add_argument("--target-tokens-per-second", type=float, help="ETA prior (default by device)")
    parser.add_argument("--skip-memory-check", action="store_true", help="tiny fixtures only")
    return parser


def check_args(args):
    for (_, attribute, _), default in zip(SANITY_GATES, DEFAULT_SANITY_GATES[args.device]):
        if getattr(args, attribute) is None:
            setattr(args, attribute, default)
    if not 0 < args.batch <= args.ctx <= TOKENIZE_CTX or not 0 < args.threads <= 256:
        raise RunError("need 0 < batch <= ctx <= 32768 and threads in 1..256")
    if not 0 < args.build_threads <= (os.cpu_count() or 1) or args.chunk_rows < 1 or args.docs_per_run < 1:
        raise RunError("build threads must be in 1..CPU count; chunk rows and docs per run positive")
    if args.max_scored < 1 or args.min_handoff_tokens < 1 or args.bootstrap < 100 or args.target_nll <= 0:
        raise RunError("max scored, min handoff tokens, target and bootstrap (>= 100) must be positive")
    if (any(not 0 <= getattr(args, attribute) < 1 for _, attribute, _ in SANITY_GATES)
            or (args.swa_cells is not None and args.swa_cells < 1)):
        raise RunError("--sanity-kl, --sanity-mean-kl and --sanity-nll must be in [0, 1) and --swa-cells positive")
    if args.headline is not None and not re.fullmatch(rf"{DEPLOYABLE_KIND}_[0-9]+(_v3)?", args.headline):
        raise RunError(f"--headline must name a {DEPLOYABLE_KIND} condition, e.g. {DEPLOYABLE_KIND}_2048_v3")
    if args.docs_per_bucket is not None and args.docs_per_bucket < 1:
        raise RunError("--docs-per-bucket must be positive")
    if args.expect_graph_splits != "any" and not args.expect_graph_splits.isdigit():
        raise RunError("--expect-graph-splits must be a count or 'any'")
    if args.grow_train and args.train_corpus is None:
        raise RunError("--grow-train needs --train-corpus")
    if not args.grow_train and args.until == "grow-train":
        raise RunError("--until grow-train needs --grow-train")
    if not 0 < args.train_batch <= args.train_ctx <= 65536:
        raise RunError("need 0 < train batch <= train ctx")
    run_v2.parse_ints(args.fit_ks, "--fit-ks")
    for rate in (args.source_tokens_per_second, args.target_tokens_per_second):
        if rate is not None and rate <= 0:
            raise RunError("ETA priors must be positive")
    files = [(args.probe, "--probe"), (args.corpus, "--corpus"), (args.source_model, "--source-model"),
             (args.target_model, "--target-model")]
    files += [(path, name) for path, name in ((args.corpus_manifest, "--corpus-manifest"),
                                              (args.train_corpus, "--train-corpus"), (args.token_table, "--token-table"),
                                              (args.hash_cache_seed, "--hash-cache-seed")) if path is not None]
    for path, name in files:
        if not path.is_file():
            raise RunError(f"{name} {path} is not a file")
    for rel in ("rendered-source.jsonl", "rendered-target.jsonl", "mapper/manifest.json", "alignment/alignment.npz",
                "captures/source", "captures/target-train", "captures/target-eval", "spec-kv.json"):
        if not (args.v2_run / rel).exists():
            raise RunError(f"--v2-run {args.v2_run} lacks {rel}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    check_args(args)
    root = args.output.resolve()
    if args.resume:
        if not root.is_dir():
            raise RunError(f"--resume: {root} is not an existing run directory")
    else:
        if root.exists() or root.is_symlink():
            raise RunError(f"{root} already exists; pass --resume to continue it")
        root.mkdir()
    (root / "stages").mkdir(exist_ok=True)
    run = LongRun(args)
    with (root / "stages" / ".lock").open("a") as lock, run_v2._TerminationSignals():
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunError(f"another run_long/run_v2 process (or a child it left running) is using {root}") from None
        run.lock_fds = (lock.fileno(),)
        if args.hash_cache_seed is not None and not (root / "model-hashes.json").exists():
            seed_hash_cache(root / "model-hashes.json", args.hash_cache_seed, [run.model(role) for role in ROLES])
        run.event("run_start", output=str(root), resume=args.resume, until=args.until, device=args.device,
                  families=run.families)
        started = time.monotonic()
        try:
            execute(run)
        except BaseException as exc:
            text = str(exc) or type(exc).__name__
            run.event("error", stage=run.current, type=type(exc).__name__, message=text.splitlines()[0],
                      detail=text)
            raise
        run.event("run_done", until=args.until, wall_seconds=round(time.monotonic() - started, 1))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
