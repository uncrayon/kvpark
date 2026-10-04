"""Run the v2 cross-model K/V transfer experiment (Qwen -> Muse) end to end, resumably.

Stages, in order; each writes stages/<stage>.json (settings, input digests, output
fingerprints, wall time, metrics) and logs to logs/:

  prepare         render the corpus with each model's GGUF chat template, tokenize both
                  vocab-only, abort on docs over --ctx or special tokens inside segments;
                  rendered-*.jsonl, templates/, spec-*.json, splits.json
  smoke           shortest train doc on each model with the real capture specs -> smoke/;
                  numpy attention must reproduce Muse attn_out; ETA for the collection
  collect-source  Qwen, all docs, k/v -> captures/source
  collect-target  Muse train docs k/v -> captures/target-train; dev+test docs with
                  q/gate/attn_out on the eval layers -> captures/target-eval
  align           align.py -> alignment/ (+ coverage per split/subset)
  fit             fit_layers.py over target [target-train, target-eval] -> mapper/
  inject          inject.py patch-score on up to --inject-docs test docs -> injection/
                  (before attention: it is the last stage that needs the servers down)
  attention       attention_eval.py on dev+test (test is the headline; mean_kv floor) -> attention/
  report          summary.json + summary.md

--resume verifies completed stages (settings, inputs, output fingerprints) and skips them.
Output fingerprints hash files up to 32 MiB and record size + mtime of larger ones, so
verification never rereads the multi-GB capture arrays. --adopt (with --resume) validates
outputs of prepare/smoke/collect stages produced outside this tool (same file layout) and
records them; --from STAGE refuses to run any earlier stage. An align/fit/inject/attention/
report output published by an attempt that died before writing its stage record is reused
when stages/<stage>.started.json shows the same settings and inputs. Children run in their
own process group, which is terminated as a whole on failure or SIGINT/SIGTERM. Progress is
one JSON object per line on stdout.
"""

import os

if __name__ == "__main__":
    for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, "4")

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
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from align import ALIGNED, STRADDLING, TEMPLATE, UNMATCHED, classify_tokens, load_alignment, segment_table
from captures import Capture, CaptureSet
import chat
from collect import sha256_file
import collect_v2


HERE = Path(__file__).resolve().parent
STAGES = ("prepare", "smoke", "collect-source", "collect-target", "align", "fit", "inject", "attention", "report")
ADOPTABLE = ("prepare", "smoke", "collect-source", "collect-target")
DEPENDS = {
    "prepare": (),
    "smoke": ("prepare",),
    "collect-source": ("prepare",),
    "collect-target": ("prepare",),
    "align": ("prepare", "collect-source", "collect-target"),
    "fit": ("align", "collect-source", "collect-target"),
    "inject": ("prepare", "collect-source", "collect-target", "align", "fit"),
    "attention": ("prepare", "collect-source", "collect-target", "align", "fit"),
    "report": ("prepare", "collect-source", "collect-target", "align", "fit", "inject", "attention"),
}
CODE = {
    "prepare": ("chat.py", "collect_v2.py", "align.py"),
    "smoke": ("collect_v2.py", "captures.py", "attention_eval.py"),
    "collect-source": ("collect_v2.py", "captures.py"),
    "collect-target": ("collect_v2.py", "captures.py"),
    "align": ("align.py", "captures.py"),
    "fit": ("fit_layers.py", "ridge.py", "captures.py"),
    "inject": ("inject.py", "attention_eval.py", "collect_v2.py"),
    "attention": ("attention_eval.py", "fit_layers.py"),
    "report": (),
}
ROLES = {"source": "qwen35", "target": "muse-glimmer"}
MODEL_NAMES = {"qwen35": "qwen", "muse-glimmer": "muse"}
SPLITS = ("train", "dev", "test")
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
HASH_LIMIT = 32 * 1024 ** 2
TOKENIZE_CTX = 32768
DEFAULT_KS = "1,2,4,8,16"
FLOOR_VARIANT = "mean_kv"
ATTENTION_TOLERANCE = 1e-4
ATTENDED = re.compile(r"layer ([0-9]+) document ([0-9a-f]{12}) done")
CAPTURED = re.compile(r"captured document ([0-9a-f]{12}): ([0-9]+) tokens")
SCORED = re.compile(r"scored ([0-9a-f]{12}) (\S+): .* decoded ([0-9]+)/([0-9]+) tokens")


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def jsonable(value):
    return json.loads(json.dumps(value, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_fingerprint(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"output is not a regular file: {path}")
    if info.st_size <= HASH_LIMIT:
        return {"size": info.st_size, "sha256": sha256_file(path)}
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def fingerprint(root, relpaths):
    """{relative file path: fingerprint} for files and (recursively) directories."""
    result = {}
    for rel in relpaths:
        path = root / rel
        if path.is_symlink() or not path.exists():
            raise ValueError(f"output is missing: {path}")
        if path.is_dir():
            files = []
            for item in sorted(path.rglob("*")):
                if item.is_symlink() or not (item.is_dir() or item.is_file()):
                    raise ValueError(f"unexpected entry in output: {item}")
                if item.is_file():
                    files.append(item)
            if not files:
                raise ValueError(f"output directory is empty: {path}")
            for item in files:
                result[str(item.relative_to(root))] = file_fingerprint(item)
        else:
            result[str(rel)] = file_fingerprint(path)
    return result


def write_new(path, data):
    """Create path with bytes or str content via a temp file and link; never replaces."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    if isinstance(data, str):
        data = data.encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".run_v2-", delete=False) as stream:
        stream.write(data)
    try:
        os.chmod(stream.name, 0o644)
        os.link(stream.name, path)
    finally:
        os.unlink(stream.name)


def json_text(value, indent=2):
    return json.dumps(value, indent=indent, ensure_ascii=False, allow_nan=False) + "\n"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {number} of {path}") from exc
    return records


def parse_layers(text, name):
    if text == "auto":
        return "auto"
    try:
        layers = [int(item) for item in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{name} must be 'auto' or comma-separated layers") from None
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0:
        raise argparse.ArgumentTypeError(f"{name} must list distinct nonnegative layers")
    return sorted(layers)


def parse_ints(text, name):
    try:
        values = [int(item) for item in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from None
    if not values or len(set(values)) != len(values) or min(values) < 1:
        raise argparse.ArgumentTypeError(f"{name} must list distinct positive integers")
    return sorted(values)


def auto_eval_layers(layer_types):
    """E = every full layer + first, middle (nearest n_layer/2) and last sliding layer;
    A = [middle sliding, middle full] for attn_out validation of both attention types."""
    n = len(layer_types)
    full = [layer for layer, kind in enumerate(layer_types) if kind == "full"]
    sliding = [layer for layer, kind in enumerate(layer_types) if kind == "sliding"]
    if not full or not sliding:
        raise ValueError("the target model needs full and sliding attention layers for auto eval layers")
    middle_sliding = min(sliding, key=lambda layer: (abs(layer - n // 2), layer))
    middle_full = middle_full_layer(layer_types)
    evaluated = sorted(set(full) | {sliding[0], middle_sliding, sliding[-1]})
    return evaluated, sorted({middle_sliding, middle_full})


def middle_full_layer(layer_types):
    full = [layer for layer, kind in enumerate(layer_types) if kind == "full"]
    if not full:
        raise ValueError("the target model has no full attention layer")
    return full[len(full) // 2]


class RunError(ValueError):
    pass


def stop_process_group(process, grace=120):
    """SIGTERM a child's whole process group (kv_probe grandchildren included), SIGKILL leftovers."""
    for number, wait in ((signal.SIGTERM, True), (signal.SIGKILL, False)):
        try:
            os.killpg(process.pid, number)
        except (ProcessLookupError, PermissionError):
            pass
        if wait:
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
    process.wait()


class Run:
    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.records = {}
        self.lock_fds = ()
        self.current = None
        self.adopt_only = False
        self._hashes = None
        self._cache = {}

    # -- bookkeeping -----------------------------------------------------------

    def event(self, event, **fields):
        print(json.dumps({"time": now_iso(), "event": event, **fields}, ensure_ascii=False,
                         allow_nan=False, default=str), flush=True)

    def path(self, rel):
        return self.root / rel

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

    def exists(self, rel):
        path = self.path(rel)
        return path.exists() or path.is_symlink()

    def require_new_work(self, stage, what):
        if self.adopt_only:
            raise RunError(f"--from {self.args.from_stage}: stage {stage} would have to run {what}; "
                           "its outputs must already exist to be adopted")

    def require_adopt(self, stage, rel):
        if not self.args.adopt:
            raise RunError(f"{stage}: {self.path(rel)} already exists but the stage has no record; "
                           "pass --resume --adopt to validate and adopt it, or use a new --output")

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

    def settings(self, stage):
        a = self.args
        if stage == "prepare":
            value = {"ctx": a.ctx, "eval_layers": a.eval_layers, "attn_out_layers": a.attn_out_layers,
                     "source_model_id": a.source_model_id, "target_model_id": a.target_model_id}
        elif stage in ("smoke", "collect-source", "collect-target"):
            value = {"ctx": a.ctx, "batch": a.batch}
        elif stage == "fit":
            value = {"ks": a.fit_ks, "alphas": a.fit_alphas}
        elif stage == "attention":
            value = {"splits": ["dev", "test"], "variants": a.attention_variants, "max_scored": a.max_scored}
        elif stage == "inject":
            value = {"inject_docs": a.inject_docs, "min_tokens": a.inject_min_tokens,
                     "max_tokens": a.inject_max_tokens, "max_scored": a.max_scored,
                     "conditions": a.inject_conditions, "focus_layer": a.inject_focus_layer,
                     "tail": a.inject_tail, "docs_per_run": a.inject_docs_per_run, "ctx": a.ctx, "batch": a.batch,
                     "long_prefix": a.inject_long_prefix, "selection": "most patched prefix positions per subset"}
        else:
            value = {}
        return jsonable(value)

    def inputs(self, stage):
        if stage == "prepare":
            return {"corpus_sha256": sha256_file(self.args.corpus), "source_model_sha256": self.model_sha("source"),
                    "target_model_sha256": self.model_sha("target")}
        return {name: self.records[name]["outputs_sha256"] for name in DEPENDS[stage]}

    def code(self, stage):
        return {name: sha256_file(HERE / name) for name in ("run_v2.py",) + CODE[stage]}

    def write_record(self, stage, outputs, metrics, started, wall, adopted=False):
        outputs = sorted(set(outputs))
        prints = fingerprint(self.root, outputs)
        record = {"format_version": 1, "stage": stage, "status": "complete", "adopted": adopted,
                  "settings": self.settings(stage), "inputs": self.inputs(stage), "output_paths": outputs,
                  "outputs": prints, "outputs_sha256": digest(prints), "code_sha256": self.code(stage),
                  "started": started, "finished": now_iso(), "wall_seconds": wall, "metrics": jsonable(metrics)}
        write_new(self.path(f"stages/{stage}.json"), json_text(record))
        self.records[stage] = record
        self.clear_started(stage)
        return record

    def mark_started(self, stage):
        """stages/<stage>.started.json: settings and inputs of the attempt about to publish outputs."""
        marker = {"stage": stage, "settings": self.settings(stage), "inputs": self.inputs(stage), "started": now_iso()}
        path = self.path(f"stages/{stage}.started.json")
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".run_v2-", delete=False,
                                         encoding="utf-8") as stream:
            stream.write(json_text(marker))
        os.replace(stream.name, path)

    def clear_started(self, stage):
        self.path(f"stages/{stage}.started.json").unlink(missing_ok=True)

    def reusable(self, stage, rel):
        """False when rel does not exist. True when it exists and an earlier attempt of this stage
        with the same settings and inputs published it but died before writing the stage record
        (the tools publish atomically, so an existing output is complete). Otherwise an error."""
        if not self.exists(rel):
            return False
        path = self.path(f"stages/{stage}.started.json")
        try:
            marker = read_json(path) if path.is_file() else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            marker = None
        if (isinstance(marker, dict) and marker.get("settings") == self.settings(stage)
                and marker.get("inputs") == self.inputs(stage)):
            self.event("stage_reuse", stage=stage, output=rel,
                       message="output published by an interrupted attempt with the same settings and inputs")
            return True
        raise RunError(f"{stage}: {self.path(rel)} exists without a stage record and was not published by an "
                       "attempt with these settings and inputs; remove it or use a new --output")

    def verify(self, stage, record):
        settings = self.settings(stage)
        if record.get("settings") != settings:
            changed = sorted(key for key in set(settings) | set(record.get("settings") or {})
                             if settings.get(key) != (record.get("settings") or {}).get(key))
            raise RunError(f"stage {stage} completed with different settings ({', '.join(changed)}); "
                           "rerun with the original settings or use a new --output")
        inputs = self.inputs(stage)
        if record.get("inputs") != inputs:
            changed = sorted(key for key in set(inputs) | set(record.get("inputs") or {})
                             if inputs.get(key) != (record.get("inputs") or {}).get(key))
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

    # -- commands ----------------------------------------------------------------

    def env(self, threads):
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for name in THREAD_VARIABLES:
            env[name] = str(threads)
        return env

    def command(self, stage, argv, log_name, *, threads, on_line=None):
        """Run a module CLI with stdout+stderr appended to logs/<log_name>; raise on failure."""
        self.path("logs").mkdir(exist_ok=True)
        log_path = self.path(f"logs/{log_name}")
        command = [sys.executable, *map(str, argv)]
        with log_path.open("ab") as log:
            log.write(f"==== run_v2 {stage} {now_iso()}: {shlex.join(command)}\n".encode("utf-8"))
            log.flush()
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  env=self.env(threads), cwd=HERE, start_new_session=True,
                                  pass_fds=self.lock_fds) as process:
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
            raise RunError(f"{stage}: {Path(argv[0]).name} exited with status {code}; see {log_path}\n"
                           + "\n".join(tail))

    def collect(self, stage, role, docs_rel, spec_rel, template_rel, out_rel, log_name, *, progress=None):
        argv = [HERE / "collect_v2.py", "--probe", self.args.probe.resolve(), "--mode", "collect",
                "--model", self.model(role), "--model-id", self.model_id(role), "--docs", self.path(docs_rel),
                "--capture-spec", self.path(spec_rel), "--chat-template", self.path(template_rel),
                "--hash-cache", self.path("model-hashes.json"), "--output", self.path(out_rel),
                "--ctx", self.args.ctx, "--batch", self.args.batch, "--threads", self.args.threads]
        if self.args.skip_memory_check:
            argv.append("--skip-memory-check")
        started = time.monotonic()
        self.command(stage, argv, log_name, threads=2, on_line=progress)
        return time.monotonic() - started

    # -- shared validation ------------------------------------------------------

    def rendered(self, name):
        key = f"rendered-{name}"
        if key not in self._cache:
            self._cache[key] = read_jsonl(self.path(f"rendered-{name}.jsonl"))
        return self._cache[key]

    def spec(self, name):
        return collect_v2.read_capture_spec(self.path(f"spec-{name}.json"))

    def validate_capture(self, rel, docs_rel, spec_name, role):
        """A published collect_v2 capture of exactly these docs, spec, model and settings."""
        path = self.path(rel)
        capture = Capture(path)
        manifest = capture.manifest
        records = read_jsonl(self.path(docs_rel))
        spec = self.spec(spec_name)
        expected = {"mode": "collect", "model_sha256": self.model_sha(role),
                    "rendered_docs_sha256": sha256_file(self.path(docs_rel)), "capture_spec": spec,
                    "render_template_sha256": records[0].get("template_sha256"),
                    "architecture": ROLES[role]}
        if not isinstance(manifest, dict):
            raise RunError(f"{path} has no collect_v2 manifest.json")
        wrong = sorted(key for key, value in expected.items() if manifest.get(key) != value)
        settings = manifest.get("settings") or {}
        wrong += [f"settings.{key}" for key in ("ctx", "batch") if settings.get(key) != getattr(self.args, key)]
        if wrong:
            raise RunError(f"{path} does not match this run ({', '.join(wrong)})")
        capture.check_documents(records)
        collect_v2.check_capture_spec(capture, spec)
        return capture


# -- prepare ---------------------------------------------------------------------

def _architecture(model):
    import gguf_tensors
    return gguf_tensors.metadata_value(model, "general.architecture")


def _tokenize(run, role, docs_path, template_path, work):
    """Vocab-only tokenization through collect_v2 (runtime checks, BOS checks)."""
    output = work / f"{role}-tokens"
    argv = [HERE / "collect_v2.py", "--probe", run.args.probe.resolve(), "--mode", "tokenize",
            "--model", run.model(role), "--model-id", run.model_id(role), "--docs", docs_path,
            "--chat-template", template_path, "--hash-cache", run.path("model-hashes.json"),
            "--output", output, "--ctx", TOKENIZE_CTX, "--batch", 512, "--threads", 4]
    run.command("prepare", argv, "prepare.log", threads=2)
    return Capture(output)


def _special_inside_segments(capture, records):
    bad = []
    for record in records:
        rows = capture.rows(record["document_id"])
        _, _, overlap = classify_tokens(capture.byte_start[rows], capture.byte_end[rows], segment_table(record))
        if np.any(overlap & capture.special[rows].astype(bool)):
            bad.append(record["document_id"])
    return bad


def stage_prepare(run):
    a = run.args
    conversations = list(chat.read_conversations(a.corpus))
    if not conversations:
        raise RunError("the corpus is empty")
    outputs, lines, templates, tokenizer = {}, {}, {}, {}
    for role in ("source", "target"):
        architecture = _architecture(run.model(role))
        if architecture != ROLES[role]:
            raise RunError(f"--{role}-model architecture is {architecture!r}, expected {ROLES[role]!r}")
        info = chat.gguf_tokenizer_info(run.model(role))
        if info["chat_template"] is None:
            raise RunError(f"--{role}-model has no tokenizer.chat_template metadata")
        template, bos = info["chat_template"], info["bos_token"]
        if not bos and chat.template_uses_bos(template):
            raise RunError(f"the {role} chat template uses bos_token but the model's BOS text is empty")
        records, seen = [], set()
        for number, conversation in enumerate(conversations, 1):
            try:
                record = chat.rendered_record(conversation, MODEL_NAMES[architecture],
                                              chat.render(conversation, template, bos_token=bos))
            except ValueError as exc:
                raise RunError(f"{role}: conversation {conversation.get('id')!r} (line {number}): {exc}") from None
            if record["document_id"] in seen:
                raise RunError(f"duplicate conversation {conversation.get('id')!r} (line {number})")
            seen.add(record["document_id"])
            records.append(record)
        collect_v2.check_bos(records, bos, info)
        lines[role] = [json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records]
        templates[role] = (MODEL_NAMES[architecture], template)
        tokenizer[role] = records
    source_records, target_records = tokenizer["source"], tokenizer["target"]
    counts, layer_types = {}, {}
    with tempfile.TemporaryDirectory(prefix=".run_v2-prepare-", dir=run.root) as temporary:
        work = Path(temporary)
        for role, records in tokenizer.items():
            (work / f"{role}.jsonl").write_text("".join(lines[role]), encoding="utf-8")
            (work / f"{role}.jinja").write_text(templates[role][1], encoding="utf-8")
            capture = _tokenize(run, role, work / f"{role}.jsonl", work / f"{role}.jinja", work)
            layer_types[role] = capture.layer_types
            counts[role] = {document["document_id"]: document["n_tokens"] for document in capture.documents}
            over = [f"{doc[:12]} ({count} tokens)" for doc, count in counts[role].items() if count > a.ctx]
            if over:
                raise RunError(f"{role}: {len(over)} documents exceed --ctx {a.ctx}: {', '.join(over[:10])}")
            bad = _special_inside_segments(capture, records)
            if bad:
                raise RunError(f"{role}: special tokens inside segments of {len(bad)} documents: "
                               f"{', '.join(doc[:12] for doc in bad[:10])}")
    if not any(kind in ("full", "sliding") for kind in layer_types["source"]):
        raise RunError("the source model has no attention layers")
    if a.eval_layers == "auto" or a.attn_out_layers == "auto":
        auto_eval, auto_out = auto_eval_layers(layer_types["target"])
    evaluated = auto_eval if a.eval_layers == "auto" else a.eval_layers
    attn_out = auto_out if a.attn_out_layers == "auto" else a.attn_out_layers
    attention = [layer for layer, kind in enumerate(layer_types["target"]) if kind in ("full", "sliding")]
    if set(evaluated) - set(attention) or set(attn_out) - set(evaluated):
        raise RunError("eval layers must be target attention layers and attn_out layers a subset of them")
    splits = {split: [record for record in target_records if record["split"] == split] for split in SPLITS}
    if any(not splits[split] for split in SPLITS):
        raise RunError("the corpus needs train, dev and test conversations")
    target_lines = dict(zip([record["document_id"] for record in target_records], lines["target"]))
    documents = [{"document_id": s["document_id"], "conversation_id": s["conversation_id"], "source": s["source"],
                  "split": s["split"], "source_tokens": counts["source"][s["document_id"]],
                  "target_tokens": counts["target"][s["document_id"]]}
                 for s in source_records]
    summary = {split: {"documents": len(splits[split]),
                       "source_tokens": sum(d["source_tokens"] for d in documents if d["split"] == split),
                       "target_tokens": sum(d["target_tokens"] for d in documents if d["split"] == split)}
               for split in SPLITS}
    files = {
        "rendered-source.jsonl": "".join(lines["source"]),
        "rendered-target.jsonl": "".join(lines["target"]),
        "rendered-target-train.jsonl": "".join(target_lines[r["document_id"]] for r in splits["train"]),
        "rendered-target-eval.jsonl": "".join(target_lines[r["document_id"]] for r in target_records
                                              if r["split"] != "train"),
        f"templates/{templates['source'][0]}.jinja": templates["source"][1],
        f"templates/{templates['target'][0]}.jinja": templates["target"][1],
        "spec-kv.json": json.dumps({"k": "all", "v": "all"}),
        "spec-target-eval.json": json.dumps({"k": "all", "v": "all", "q": evaluated, "gate": evaluated,
                                             "attn_out": attn_out}),
        "splits.json": json_text({"format_version": 1, "splits": summary, "documents": documents,
                                  "layer_types": layer_types}),
    }
    run.path("templates").mkdir(exist_ok=True)
    written, adopted = [], []
    for rel, content in files.items():
        path = run.path(rel)
        if run.exists(rel):
            run.require_adopt("prepare", rel)
            existing = path.read_bytes()
            same = (json.loads(existing) == json.loads(content) if rel.endswith(".json")
                    else existing == content.encode("utf-8"))
            if not same:
                raise RunError(f"prepare: existing {path} differs from what this run renders "
                               "(different corpus, template, chat.py or layer selection)")
            adopted.append(rel)
        else:
            write_new(path, content)
            written.append(rel)
    metrics = {"documents": len(documents), "splits": summary, "eval_layers": evaluated,
               "attn_out_layers": attn_out, "max_tokens": {role: max(counts[role].values()) for role in counts},
               "written": written, "adopted_files": adopted}
    return list(files), metrics, bool(adopted)


# -- smoke -----------------------------------------------------------------------

def validate_attn_out(capture, model):
    import attention_eval as ae
    params = ae.load_model_params(model)
    errors = {}
    for layer in capture.layers("attn_out"):
        sliding = capture.layer_types[layer] == "sliding"
        worst = 0.0
        for document in capture.documents:
            rows = capture.rows(document["document_id"])
            q, k, v = (np.asarray(capture.array(key, layer)[rows]) for key in ("q", "k", "v"))
            error, _ = ae.compare_layer(q, k, v, np.zeros(0, dtype=np.int64), {}, params=params, sliding=sliding,
                                        query_sets={}, attn_out=np.asarray(capture.array("attn_out", layer)[rows]))
            worst = max(worst, error)
        errors[layer] = worst
        if not worst < ATTENTION_TOLERANCE:
            raise RunError(f"smoke: numpy attention does not reproduce captured attn_out-{layer} "
                           f"({capture.layer_types[layer]}): relative error {worst:.3g}")
    if not errors:
        raise RunError("smoke: the target capture has no attn_out layer to validate")
    return errors


def collection_times(run):
    """Wall seconds per step from logs/collection-steps.jsonl (manual collection chain), if present."""
    path = run.path("logs/collection-steps.jsonl")
    times = {}
    if not path.is_file():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            stamp = datetime.datetime.fromisoformat(item["time"])
        except (ValueError, KeyError, TypeError):
            continue
        times.setdefault(item.get("step"), {})[item.get("event")] = stamp
    return {step: (events["done"] - events["start"]).total_seconds()
            for step, events in times.items() if "start" in events and "done" in events}


def stage_smoke(run):
    source_records = run.rendered("source")
    train = [record for record in source_records if record["split"] == "train"]
    chosen = min(train, key=lambda record: (len(record["text"].encode("utf-8")), record["document_id"]))["document_id"]
    run.path("smoke").mkdir(exist_ok=True)
    adopted, metrics = False, {"document_id": chosen}
    for role, rendered, spec in (("source", "source", "kv"), ("target", "target", "target-eval")):
        docs_rel, out_rel = f"smoke-{role}.jsonl", f"smoke/{role}"
        line = next(line for line in run.path(f"rendered-{rendered}.jsonl").read_text(encoding="utf-8")
                    .splitlines(keepends=True) if json.loads(line)["document_id"] == chosen)
        if run.exists(docs_rel):
            run.require_adopt("smoke", docs_rel)
            if run.path(docs_rel).read_text(encoding="utf-8") != line:
                raise RunError(f"smoke: existing {run.path(docs_rel)} is not the shortest train document")
        else:
            write_new(run.path(docs_rel), line)
        seconds = None
        if run.exists(out_rel):
            run.require_adopt("smoke", out_rel)
            adopted = True
        else:
            run.require_new_work("smoke", f"a {role} collection")
            collect_v2.memory_check(run.model(role), run.args.skip_memory_check)
            seconds = run.collect("smoke", role, docs_rel, f"spec-{spec}.json", template_rel(run, role), out_rel,
                                  f"smoke-{role}.log")
        capture = run.validate_capture(out_rel, docs_rel, spec, role)
        capture.check_finite()
        metrics[role] = {"tokens": capture.rows_total, "seconds_including_load": seconds,
                         "tokens_per_second_including_load": (capture.rows_total / seconds if seconds else None)}
        if role == "target":
            metrics["attn_out_relative_error"] = validate_attn_out(capture, run.model("target"))
    return ["smoke-source.jsonl", "smoke-target.jsonl", "smoke/source", "smoke/target"], metrics, adopted


def template_rel(run, role):
    return f"templates/{MODEL_NAMES[ROLES[role]]}.jinja"


# -- collection ------------------------------------------------------------------

class Progress:
    """Throughput and ETA from per-item log lines of a child (the first item includes model load)."""

    def __init__(self, run, stage, total, pattern, amount, label, unit="tokens"):
        self.run, self.stage, self.total, self.pattern, self.tokens_of = run, stage, total, pattern, amount
        self.label, self.unit, self.done, self.first = label, unit, 0, None
        self.lock = threading.Lock()

    def __call__(self, line):
        match = self.pattern.search(line)
        if not match:
            return
        tokens = self.tokens_of(match)
        with self.lock:
            now = time.monotonic()
            self.done += tokens
            if self.first is None:
                self.first = (now, self.done)
                rate = None
            else:
                elapsed = now - self.first[0]
                rate = (self.done - self.first[1]) / elapsed if elapsed > 0 else None
            remaining = max(0, self.total - self.done)
            self.run.event("progress", stage=self.stage, part=self.label, unit=self.unit, done=self.done,
                           total=self.total, per_second=None if rate is None else round(rate, 3),
                           eta_seconds=None if not rate else round(remaining / rate))


def rate_for(run, role):
    """(tokens/s, provenance) for ETAs: measured in this run when possible, else the prior."""
    stage = "collect-source" if role == "source" else "collect-target"
    record = run.records.get(stage)
    if record and record["metrics"].get("tokens_per_second"):
        return record["metrics"]["tokens_per_second"], f"measured in {stage}"
    prior = run.args.source_tokens_per_second if role == "source" else run.args.target_tokens_per_second
    return prior, f"prior --{role}-tokens-per-second"


def eta_event(run, stage, tokens, rate, provenance, **extra):
    seconds = tokens / rate if rate else None
    finish = None if seconds is None else (datetime.datetime.now().astimezone()
                                           + datetime.timedelta(seconds=seconds)).isoformat(timespec="minutes")
    run.event("eta", stage=stage, tokens=tokens, tokens_per_second=rate, rate_source=provenance,
              eta_seconds=None if seconds is None else round(seconds), eta_finish=finish, **extra)
    return seconds


def remaining_eta(run):
    """After smoke: ETA of the collection stages (priors, or rates measured in this run)."""
    info = splits_info(run)
    total = 0.0
    for stage, role, splits in (("collect-source", "source", set(SPLITS)), ("collect-target", "target", set(SPLITS))):
        if stage in run.records or run.read_record(stage) is not None:
            continue
        rate, provenance = rate_for(run, role)
        tokens = sum(d[f"{role}_tokens"] for d in info["documents"] if d["split"] in splits)
        total += eta_event(run, stage, tokens, rate, provenance) or 0
    run.event("eta", stage="collection", eta_seconds=round(total), note="injection ETA is printed when it starts")


def splits_info(run):
    return read_json(run.path("splits.json"))


def stage_collect(run, stage):
    role = "source" if stage == "collect-source" else "target"
    info = splits_info(run)
    parts = ([("source", "rendered-source.jsonl", "kv", "captures/source", "collect-source.log", None)]
             if role == "source" else
             [("target-train", "rendered-target-train.jsonl", "kv", "captures/target-train",
               "collect-target-train.log", {"train"}),
              ("target-eval", "rendered-target-eval.jsonl", "target-eval", "captures/target-eval",
               "collect-target-eval.log", {"dev", "test"})])
    manual = collection_times(run)
    pending = [part for part in parts if not run.exists(part[3])]
    for part in parts:
        if part not in pending:
            run.require_adopt(stage, part[3])
    tokens = {part[0]: sum(d[f"{role}_tokens"] for d in info["documents"]
                           if part[5] is None or d["split"] in part[5]) for part in parts}
    if pending:
        run.require_new_work(stage, f"collections {[part[0] for part in pending]}")
        collect_v2.memory_check(run.model(role), run.args.skip_memory_check)
        rate, provenance = rate_for(run, role)
        eta_event(run, stage, sum(tokens[part[0]] for part in pending), rate, provenance,
                  parts=[part[0] for part in pending])
    run.path("captures").mkdir(exist_ok=True)
    metrics, seconds_total, adopted = {"parts": {}}, 0.0, False
    for name, docs_rel, spec, out_rel, log_name, _ in parts:
        seconds = None
        if (name, docs_rel, spec, out_rel, log_name, _) in pending:
            progress = Progress(run, stage, tokens[name], CAPTURED, lambda m: int(m.group(2)), name)
            seconds = run.collect(stage, role, docs_rel, f"spec-{spec}.json", template_rel(run, role), out_rel,
                                  log_name, progress=progress)
        else:
            adopted = True
            seconds = manual.get(name)
        capture = run.validate_capture(out_rel, docs_rel, spec, role)
        if capture.rows_total != tokens[name]:
            raise RunError(f"{out_rel} holds {capture.rows_total} tokens, prepare counted {tokens[name]}")
        metrics["parts"][name] = {"tokens": capture.rows_total, "documents": len(capture.documents),
                                  "seconds": seconds, "adopted": name not in [p[0] for p in pending],
                                  "tokens_per_second": capture.rows_total / seconds if seconds else None}
        if seconds is None:
            seconds_total = None
        elif seconds_total is not None:
            seconds_total += seconds
    total_tokens = sum(tokens.values())
    metrics["tokens"] = total_tokens
    metrics["seconds"] = seconds_total
    metrics["tokens_per_second"] = total_tokens / seconds_total if seconds_total else None
    return [part[3] for part in parts], metrics, adopted


# -- post-collection stages -------------------------------------------------------

def target_capture_paths(run):
    return [run.path("captures/target-train"), run.path("captures/target-eval")]


def coverage(run, alignment):
    src_set = CaptureSet([run.path("captures/source")])
    tgt_set = CaptureSet(target_capture_paths(run))
    records = {record["document_id"]: record for record in run.rendered("target")}
    groups = {}
    for number, document_id in enumerate(alignment.document_ids):
        record = records[document_id]
        classes = alignment.tgt_class[tgt_set.rows(document_id)]
        src_classes = alignment.src_class[src_set.rows(document_id)]
        counts = {"documents": 1, "target_tokens": len(classes),
                  "aligned": int(np.sum(classes == ALIGNED)), "unmatched": int(np.sum(classes == UNMATCHED)),
                  "straddling": int(np.sum(classes == STRADDLING)),
                  "template_or_special": int(np.sum(classes == TEMPLATE)), "source_tokens": len(src_classes),
                  "source_content": int(np.sum(np.isin(src_classes, (ALIGNED, UNMATCHED, STRADDLING))))}
        for key in (("by_split", record["split"]), ("by_subset", record["source"]),
                    ("by_split_subset", f"{record['split']}/{record['source']}"), ("all", "all")):
            total = groups.setdefault(key[0], {}).setdefault(key[1], dict.fromkeys(counts, 0))
            for name, value in counts.items():
                total[name] += value
    for section in groups.values():
        for total in section.values():
            content = total["aligned"] + total["unmatched"] + total["straddling"]
            total["target_content"] = content
            total["aligned_fraction_of_content"] = total["aligned"] / content if content else None
            total["aligned_fraction_of_tokens"] = total["aligned"] / total["target_tokens"]
            total["source_aligned_fraction_of_content"] = (total["aligned"] / total["source_content"]
                                                           if total["source_content"] else None)
    return {"format_version": 1, "pairs": int(len(alignment.src_row)), "rejected": alignment.rejected,
            "definition": "target tokens; content = tokens inside rendered segments (aligned + unmatched + "
                          "straddling); aligned = same segment and same end byte in both models",
            **{name: dict(sorted(section.items())) for name, section in sorted(groups.items())}}


def staged_directory(run, stage):
    return Path(tempfile.mkdtemp(prefix=f".run_v2-{stage}-", dir=run.root))


def publish(run, staging, final_rel):
    os.chmod(staging, 0o755)
    collect_v2.publish_directory(staging, run.path(final_rel))


def stage_align(run):
    if not run.reusable("align", "alignment"):
        run.mark_started("align")
        staging = staged_directory(run, "align")
        try:
            argv = [HERE / "align.py", "--src-docs", run.path("rendered-source.jsonl"),
                    "--src-capture", run.path("captures/source"), "--tgt-docs", run.path("rendered-target.jsonl"),
                    "--output", staging / "alignment.npz"]
            for path in target_capture_paths(run):
                argv += ["--tgt-capture", path]
            run.command("align", argv, "align.log", threads=run.args.fit_threads)
            alignment = load_alignment(staging / "alignment.npz")
            if alignment.rejected:
                raise RunError(f"align rejected {len(alignment.rejected)} documents: {alignment.rejected[:3]}")
            write_new(staging / "coverage.json", json_text(coverage(run, alignment)))
            publish(run, staging, "alignment")
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    alignment = load_alignment(run.path("alignment/alignment.npz"))
    if alignment.rejected:
        raise RunError(f"align rejected {len(alignment.rejected)} documents: {alignment.rejected[:3]}")
    report = read_json(run.path("alignment/coverage.json"))
    return ["alignment"], {"pairs": report["pairs"], "coverage": report["by_split"]}, False


def fit_ks(run):
    requested = parse_ints(run.args.fit_ks, "--fit-ks")
    layers = Capture(run.path("captures/source")).layers("k")
    ks = [k for k in requested if k <= len(layers)]
    if not ks:
        raise RunError(f"--fit-ks {run.args.fit_ks}: the source capture has only {len(layers)} layers")
    return ks, layers


def stage_fit(run):
    ks, layers = fit_ks(run)
    if not run.reusable("fit", "mapper"):
        run.mark_started("fit")
        argv = [HERE / "fit_layers.py", "--src-capture", run.path("captures/source"),
                "--alignment", run.path("alignment/alignment.npz"), "--output", run.path("mapper"),
                "--ks", ",".join(map(str, ks)), "--threads", run.args.fit_threads]
        for path in target_capture_paths(run):
            argv += ["--tgt-capture", path]
        if run.args.fit_alphas:
            argv += ["--alphas", run.args.fit_alphas]
        run.command("fit", argv, "fit.log", threads=run.args.fit_threads)
    import attention_eval as ae
    mapper = ae.Mapper(run.path("mapper"))
    target_layers = Capture(run.path("captures/target-eval")).layers("k")
    missing = [f"{kind}-{layer}" for layer in target_layers for kind in ("k", "v") if not mapper.has(kind, layer)]
    if missing:
        raise RunError(f"fit: the mapper lacks maps {missing[:8]}")
    manifest = mapper.manifest
    return ["mapper"], {"ks": ks, "ks_requested": run.args.fit_ks, "source_layers": layers,
                        "rows": manifest["rows"], "fit_wall_seconds": manifest.get("wall_seconds")}, False


def stage_attention(run):
    import attention_eval as ae
    if not run.reusable("attention", "attention"):
        run.mark_started("attention")
        staging = staged_directory(run, "attention")
        try:
            argv = [HERE / "attention_eval.py", "--model", run.model("target"),
                    "--tgt-docs", run.path("rendered-target.jsonl"),
                    "--tgt-capture", run.path("captures/target-eval"),
                    "--src-docs", run.path("rendered-source.jsonl"), "--src-capture", run.path("captures/source"),
                    "--alignment", run.path("alignment/alignment.npz"), "--mapper", run.path("mapper"),
                    "--output", staging / "attention-report.json", "--splits", "dev,test",
                    "--variants", run.args.attention_variants, "--max-scored", run.args.max_scored,
                    "--hash-cache", run.path("model-hashes.json")]
            alignment = load_alignment(run.path("alignment/alignment.npz"))
            documents = sum(1 for split in alignment.splits if split in ("dev", "test"))
            layers = run.spec("target-eval")["q"]
            progress = Progress(run, "attention", documents * len(layers), ATTENDED, lambda match: 1, "attention",
                                unit="layer-documents")
            run.command("attention", argv, "attention.log", threads=run.args.fit_threads, on_line=progress)
            check_attention_validation(read_json(staging / "attention-report.json"))
            publish(run, staging, "attention")
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    report = read_json(run.path("attention/attention-report.json"))
    validation = check_attention_validation(report)
    return ["attention"], {"documents": len(report["documents"]), "validation": validation,
                           "layers": report["settings"]["layers"]}, False


def check_attention_validation(report):
    import attention_eval as ae
    validation = report.get("validation") or []
    if (not validation or report.get("unvalidated_attention_types")
            or any(not item["max_relative_error"] < ae.VALIDATION_TOLERANCE for item in validation)):
        raise RunError(f"attention: attn_out validation did not pass: {validation}, unvalidated types "
                       f"{report.get('unvalidated_attention_types')}")
    return validation


def inject_plans(run):
    import attention_eval as ae
    tgt_set = CaptureSet([run.path("captures/target-eval")])
    src_set = CaptureSet([run.path("captures/source")])
    plans, skipped = ae.plan_documents(load_alignment(run.path("alignment/alignment.npz")), run.rendered("target"),
                                       tgt_set, run.rendered("source"), src_set, {"test"})
    return plans, skipped, tgt_set


def select_inject_documents(plans, records, count, low, high):
    """Up to `count` plans with low <= Muse tokens <= high, round-robin over subsets (sorted).
    Within a subset the plans with the most patched prefix positions come first (ties by
    document_id), so injection tests the longest mapped prefixes rather than system + user stubs."""
    by_subset = {}
    for plan in sorted(plans, key=lambda plan: (-len(plan.positions), plan.document_id)):
        if low <= plan.n_tokens <= high and np.any(plan.continuation >= 1):
            by_subset.setdefault(records[plan.document_id]["source"], []).append(plan)
    chosen, subsets = [], sorted(by_subset)
    while len(chosen) < count and any(by_subset.values()):
        for subset in subsets:
            if by_subset[subset] and len(chosen) < count:
                chosen.append(by_subset[subset].pop(0))
    return chosen


def decoded_tokens(plan, max_scored, batch, patched_last):
    """Tokens kv_probe decodes for one condition: whole ubatches through the last needed position."""
    scored = plan.continuation[plan.continuation >= 1][:max_scored]
    needed = int(scored[-1])
    if patched_last is not None:
        needed = max(needed, int(patched_last) + 1)
    return min(plan.n_tokens, -(-needed // batch) * batch)


def _decoded_in(run_dirs):
    total = 0
    for run_dir in run_dirs:
        for document in read_json(run_dir / "results.json")["documents"]:
            total += sum(condition.get("decoded_tokens", document["n_tokens"]) for condition in document["conditions"])
    return total


def positive_control(report):
    """Raise unless some non-native condition changed predictions; mean_all must, if it ran."""
    import inject
    effect = [item for item in report["conditions"] if item["name"] not in ("native", inject.SANITY)]
    if not effect:
        return None
    if all(item["kl"]["max_token"] == 0 for item in effect):
        raise RunError("inject: positive control failed: no condition changed any prediction (KL exactly 0 "
                       "everywhere), so the patches had no effect")
    floor = next((item for item in effect if item["name"] == "mean_all"), None)
    if floor is None:
        return {"condition": None, "passed": True}
    documents = [item["conditions"]["mean_all"] for item in report["per_document"] if "mean_all" in item["conditions"]]
    changed_kl = sum(1 for item in documents if item["kl_mean"] > 0)
    changed_top1 = sum(1 for item in documents if item["top1_agree"] < 1)
    if not changed_kl:
        raise RunError("inject: positive control failed: mean_all left every document's predictions unchanged")
    return {"condition": "mean_all", "documents": len(documents), "documents_kl_above_0": changed_kl,
            "documents_top1_below_1": changed_top1, "passed": True, "top1_changed": changed_top1 > 0}


def stage_inject(run):
    import inject
    a = run.args
    plans, skipped, tgt_set = inject_plans(run)
    capture_batch = tgt_set.captures[0].report["settings"]["batch"]
    if capture_batch != a.batch:
        raise RunError(f"inject: --batch {a.batch} differs from the target-eval capture batch {capture_batch}; "
                       "the sanity condition is exact only with identical ubatches")
    layer_types = tgt_set.captures[0].layer_types
    focus = middle_full_layer(layer_types) if a.inject_focus_layer == "auto" else int(a.inject_focus_layer)
    names = None if a.inject_conditions == "all" else a.inject_conditions.split(",")
    conditions = inject.define_conditions(layer_types, focus_layer=focus, tail=a.inject_tail, names=names)
    records = {record["document_id"]: record for record in run.rendered("target")}
    chosen = select_inject_documents(plans, records, a.inject_docs, a.inject_min_tokens, a.inject_max_tokens)
    if not chosen:
        raise RunError(f"inject: no test document has {a.inject_min_tokens}..{a.inject_max_tokens} Muse tokens")
    documents_rel = "stages/inject-documents.txt"
    text = "".join(plan.document_id + "\n" for plan in chosen)
    if run.exists(documents_rel):
        if run.path(documents_rel).read_text(encoding="utf-8") != text:
            raise RunError(f"inject: {run.path(documents_rel)} lists other documents than this run selects")
    else:
        write_new(run.path(documents_rel), text)
    per_condition = [decoded_tokens(plan, a.max_scored, a.batch, plan.positions[-1]) for plan in chosen]
    decoded = len(conditions) * sum(per_condition)
    full = len(conditions) * sum(plan.n_tokens for plan in chosen)
    partial = run.path("injection.partial")
    resume, seconds, earlier = partial.exists(), None, set()
    if not run.reusable("inject", "injection"):
        run.mark_started("inject")
        if resume:
            earlier = {path.name for path in (partial / "out" / "runs").glob("group-*")}
        collect_v2.memory_check(run.model("target"), a.skip_memory_check)
        rate, provenance = rate_for(run, "target")
        eta_event(run, "inject", decoded, rate, provenance, documents=len(chosen), conditions=len(conditions),
                  full_decode_tokens=full, resume=resume, groups_already_scored=len(earlier))
        argv = [HERE / "inject.py", "run", "--probe", a.probe.resolve(), "--model", run.model("target"),
                "--model-id", a.target_model_id, "--tgt-docs", run.path("rendered-target.jsonl"),
                "--tgt-capture", run.path("captures/target-eval"), "--src-docs", run.path("rendered-source.jsonl"),
                "--src-capture", run.path("captures/source"), "--alignment", run.path("alignment/alignment.npz"),
                "--mapper", run.path("mapper"), "--output", run.path("injection"), "--splits", "test",
                "--documents", run.path(documents_rel), "--focus-layer", focus, "--tail", a.inject_tail,
                "--docs-per-run", a.inject_docs_per_run, "--ctx", a.ctx, "--batch", a.batch, "--threads", a.threads,
                "--max-scored", a.max_scored, "--long-prefix", a.inject_long_prefix,
                "--hash-cache", run.path("model-hashes.json")]
        if names is not None:
            argv += ["--conditions", a.inject_conditions]
        if a.skip_memory_check:
            argv.append("--skip-memory-check")
        if resume:
            argv.append("--resume")
        started = time.monotonic()
        progress = Progress(run, "inject", decoded, SCORED, lambda m: int(m.group(3)), "patch-score")
        run.command("inject", argv, "inject.log", threads=a.fit_threads, on_line=progress)
        seconds = time.monotonic() - started
    report = read_json(run.path("injection/injection-report.json"))
    sanity = report.get("sanity") or {}
    if not sanity.get("passed"):
        raise RunError(f"inject: the sanity condition did not pass: {sanity}")
    if sanity.get("max_token_kl") != 0:
        run.event("warning", stage="inject", message="sanity KL is below the threshold but not exactly 0",
                  sanity=sanity)
    control = positive_control(report)
    if control is not None and not control.get("top1_changed", True):
        run.event("warning", stage="inject", message="mean_all changed KL but no top-1 prediction", control=control)
    run_dirs = sorted(run.path("injection/runs").iterdir())
    actual = _decoded_in(run_dirs)
    this_invocation = None if seconds is None else _decoded_in([path for path in run_dirs if path.name not in earlier])
    metrics = {"documents": len(chosen), "conditions": [condition.name for condition in conditions],
               "focus_layer": focus, "decoded_tokens": actual, "decoded_tokens_planned": decoded,
               "full_decode_tokens": full, "seconds": seconds, "resumed": resume,
               "groups_scored_before": sorted(earlier), "decoded_tokens_this_invocation": this_invocation,
               "tokens_per_second": this_invocation / seconds if seconds and this_invocation else None,
               "sanity": sanity, "positive_control": control, "planned_test_documents": len(plans),
               "skipped": skipped,
               "prefix_patched_fraction": [
                   {"document_id": plan.document_id, "n_tokens": plan.n_tokens, "prefix_tokens": plan.prefix_tokens,
                    "patched": int(len(plan.positions))} for plan in chosen]}
    return ["injection", documents_rel], metrics, False


# -- report ----------------------------------------------------------------------

def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _ci(item, digits=4):
    ci = item.get("ci95")
    return f"{item['mean']:.{digits}f}" + ("" if ci is None else f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]")


def fit_summary(run):
    manifest = read_json(run.path("mapper/manifest.json"))
    report = read_json(run.path("mapper/report.json"))
    types = read_json(run.path("splits.json"))["layer_types"]["target"]
    by_k = {}
    for entry in report["top_k"]:
        value = entry["dev"]["r2"]
        if entry["kind"] == "k":
            value = max(value, entry["dev_renorm"]["r2"])
        key = (entry["kind"], entry["target"], entry["k"])
        by_k[key] = max(by_k.get(key, -math.inf), value)
    rows = []
    for item in manifest["maps"]:
        applied = item["dev"]["applied"]
        rows.append({"kind": item["kind"], "target": item["target"], "attention": types[item["target"]],
                     "k": item["k"], "alpha": item["alpha"], "k_renorm": item["k_renorm"],
                     "source_layers": item["source_layers"], "train_r2": item["train_r2"],
                     "dev_r2": item["dev"][applied]["r2"], "test_r2": item["test"][applied]["r2"],
                     "test_mse_over_train_mean": item["test"][applied]["mse_over_train_mean"],
                     "dev_r2_by_k": {str(k): by_k[item["kind"], item["target"], k]
                                     for k in report["ks"] if (item["kind"], item["target"], k) in by_k}})
    rows.sort(key=lambda row: (row["kind"], row["target"]))
    means = {}
    for kind in ("k", "v"):
        for attention in ("full", "sliding", "all"):
            chosen = [row for row in rows if row["kind"] == kind and attention in ("all", row["attention"])]
            if chosen:
                means[f"{kind}/{attention}"] = {"dev_r2": float(np.mean([row["dev_r2"] for row in chosen])),
                                                "test_r2": float(np.mean([row["test_r2"] for row in chosen])),
                                                "layers": len(chosen)}
    return {"rows": manifest["rows"], "ks": report["ks"], "alphas": report["alphas"], "maps": rows, "means": means}


def skill(value, floor):
    """(cos - floor) / (1 - floor): 0 = information-free floor, 1 = native; None when undefined."""
    if value is None or floor is None or not floor < 1:
        return None
    return (value - floor) / (1 - floor)


def attention_summary(run):
    report = read_json(run.path("attention/attention-report.json"))
    keep = ("split", "layer", "attention", "variant", "query_set", "queries", "sdpa_head_cosine_mean",
            "contribution_cosine_mean", "contribution_relative_error_pooled")
    layers = [{key: item.get(key) for key in keep} for item in report["layers"]]
    floors = {(item["split"], item["layer"], item["query_set"]): item for item in layers
              if item["variant"] == FLOOR_VARIANT}
    for item in layers:
        floor = floors.get((item["split"], item["layer"], item["query_set"]))
        if item["variant"] == FLOOR_VARIANT or floor is None:
            continue
        for metric in ("sdpa_head_cosine_mean", "contribution_cosine_mean"):
            item[f"floor_{metric}"] = floor[metric]
            item[f"skill_{metric}"] = skill(item[metric], floor[metric])
    return {"headline_split": report.get("headline_split"), "validation": report["validation"],
            "documents": {split: sum(1 for d in report["documents"] if d["split"] == split) for split in ("dev", "test")},
            "floor_variant": FLOOR_VARIANT if floors else None, "query_sets": report["settings"].get("query_sets"),
            "skipped": report.get("skipped"), "layers": layers}


def injection_summary(run):
    report = read_json(run.path("injection/injection-report.json"))
    fields = ("name", "documents", "tokens", "nll", "nll_delta", "kl", "top1_agree")
    conditions = [{key: item[key] for key in fields} for item in report["conditions"]]
    metrics = run.records["inject"]["metrics"]
    subsets = {record["document_id"]: record["source"] for record in run.rendered("target")}
    scored = []
    for item in report.get("scored_documents") or []:
        scored.append({**item, "subset": subsets.get(item["document_id"])})
    mapped = {}
    for item in scored:
        for name, count in (item.get("mapped_rows") or {}).items():
            mapped[name] = mapped.get(name, 0) + count
    for item in conditions:
        item["mapped_rows"] = mapped.get(item["name"])
    groups = {name: {"documents": group["documents"], "scored_tokens": group["scored_tokens"],
                     "document_ids": group["document_ids"],
                     "conditions": [{key: item[key] for key in fields} for item in group["conditions"]],
                     "contrasts": group["contrasts"]}
              for name, group in (report.get("groups") or {}).items()}
    return {"documents": report["documents"], "scored_tokens": report["scored_tokens"],
            "truncated_documents": report["truncated_documents"], "bootstrap": report["bootstrap"],
            "sanity": report["sanity"], "positive_control": metrics.get("positive_control"),
            "conditions": conditions, "contrasts": report.get("contrasts") or [], "groups": groups,
            "groups_definition": report.get("groups_definition"), "defined": report.get("conditions_defined"),
            "scored_documents": scored, "per_document": report.get("per_document"),
            "focus_layer": metrics["focus_layer"], "decoded_tokens": metrics["decoded_tokens"],
            "full_decode_tokens": metrics["full_decode_tokens"], "max_scored": run.args.max_scored}


CAVEATS = [
    "fit_layers selects (k, alpha) and K renormalization on dev, so dev R2 and dev attention numbers are "
    "optimistic; test is the headline and splits are never pooled.",
    "Only aligned content tokens (same chat segment, same end byte in both tokenizations) are mapped and "
    "patched; template/special, unmatched and straddling prefix tokens keep Muse's native K/V in every "
    "condition, mean_all included. That covers chat headers and text the template renders itself, such as "
    "tool schemas in the system turn, which can be most of the prefix of tool-calling documents.",
    "Injection patches K/V in the prefix (before the final message) and scores at most --max-scored tokens "
    "at the start of the final assistant message per document; the continuation itself is always native. "
    "When the final message carries reasoning, the prefix is only the earlier turns (often just system + "
    "user, a few dozen tokens). Injection documents are picked per subset by most patched prefix positions; "
    "results are also given separately for long and short patched prefixes and per document.",
    "all_kv_tail<N> and all_kv_system_native leave the overridden prefix positions out of the patch: Muse "
    "recomputes them on top of the mapped prefix (as a deployment would), not from oracle native values. "
    "When the override covers every patched position of a document nothing is mapped there.",
    "The test set is small (a few dozen documents, fewer injected); bootstrap intervals resample documents "
    "and are wide. Paired contrasts (condition minus mean_all / all_kv) share resample indices and are the "
    "way to compare conditions. One corpus, one split, one seed.",
    "mean_all (injection) and mean_kv (attention) put the train target mean at every patched position, K "
    "renormalized per head to RMS 1 like native Muse K: an information-free floor. Attention cosines are high "
    "even at this floor because queries attend mostly to native tokens; read them through the skill column "
    "(cos - floor) / (1 - floor) and prefer the 'scored' and 'affected' query sets.",
    "Attention cosines are computed offline in numpy and are trusted only because the numpy attention "
    "reproduces captured attn_out on one sliding and one full layer; other layers reuse the same code. The "
    "smoke document is short, so the sliding window is exercised by the real validation documents only.",
    "The sanity condition proves the patching path is a no-op only for identical ubatches (inject --batch "
    "equals the capture batch); the positive control requires mean_all to change predictions.",
    "Throughput of adopted collection stages comes from the external collection log timestamps and includes "
    "model loading; smoke throughput includes model loading; injection throughput counts only groups scored "
    "in the last invocation and includes model loads and patch building.",
    "Untrained tiny fixtures (tests) exercise the plumbing only; no number from them measures transfer.",
]


def _attention_metric(item):
    """(name, value, floor, skill) of the contribution cosine, or the SDPA cosine without a gate."""
    if item.get("contribution_cosine_mean") is not None:
        metric = "contribution_cosine_mean"
    else:
        metric = "sdpa_head_cosine_mean"
    return item.get(metric), item.get(f"floor_{metric}"), item.get(f"skill_{metric}")


def _injection_table(conditions, mapped=True):
    lines = ["| condition | docs | tokens |" + (" mapped rows |" if mapped else "")
             + " delta NLL vs native | KL(native, cond) | top-1 agreement |",
             "|---|---|---|" + ("---|" if mapped else "") + "---|---|---|"]
    for item in conditions:
        lines.append(f"| {item['name']} | {item['documents']} | {item['tokens']} |"
                     + (f" {_fmt(item.get('mapped_rows'))} |" if mapped else "")
                     + f" {_ci(item['nll_delta'])} | {_ci(item['kl'])} | {_ci(item['top1_agree'], 3)} |")
    return lines


def _contrast_table(contrasts):
    lines = ["| condition | minus | docs | delta NLL | delta KL | delta top-1 |", "|---|---|---|---|---|---|"]
    for item in contrasts:
        lines.append(f"| {item['name']} | {item['baseline']} | {item['documents']} | {_ci(item['nll'])} | "
                     f"{_ci(item['kl'])} | {_ci(item['top1_agree'], 3)} |")
    return lines


def markdown(summary):
    lines = ["# v2 cross-model K/V transfer run", "",
             f"Source {summary['models']['source']['id']} -> target {summary['models']['target']['id']}; "
             f"generated {summary['generated']}.", "", "## Coverage (target tokens)", "",
             "| group | docs | tokens | content | aligned/content | aligned/tokens |", "|---|---|---|---|---|---|"]
    for section in ("by_split", "by_subset"):
        for name, item in summary["coverage"][section].items():
            lines.append(f"| {name} | {item['documents']} | {item['target_tokens']} | {item['target_content']} | "
                         f"{_fmt(item['aligned_fraction_of_content'])} | {_fmt(item['aligned_fraction_of_tokens'])} |")
    fit = summary["fit"]
    lines += ["", "## Fit (R^2 of the applied prediction; dev selects, test once)", "",
              f"Aligned rows: {', '.join(f'{s} {fit['rows'][s]['pairs']}' for s in SPLITS)}.", ""]
    for kind in ("k", "v"):
        header = " | ".join(f"dev k={k}" for k in fit["ks"])
        lines += [f"### {kind.upper()}", "", f"| layer | type | {header} | k | renorm | dev | test |",
                  "|" + "---|" * (len(fit["ks"]) + 6)]
        for row in fit["maps"]:
            if row["kind"] != kind:
                continue
            cells = " | ".join(_fmt(row["dev_r2_by_k"].get(str(k))) for k in fit["ks"])
            lines.append(f"| {row['target']} | {row['attention']} | {cells} | {row['k']} | "
                         f"{'yes' if row['k_renorm'] else 'no'} | {_fmt(row['dev_r2'])} | {_fmt(row['test_r2'])} |")
        lines.append("")
    lines += ["Means over layers: " + "; ".join(f"{key} dev {_fmt(value['dev_r2'])} test {_fmt(value['test_r2'])}"
                                                 for key, value in fit["means"].items()), ""]
    injection = summary["injection"]
    sanity = injection["sanity"] or {}
    control = injection.get("positive_control") or {}
    lines += ["## Prefix injection (Muse test documents; mean [doc-bootstrap 95% CI])", "",
              f"{injection['documents']} documents, {injection['scored_tokens']} scored tokens; sanity max token "
              f"KL {sanity.get('max_token_kl')} (passed: {sanity.get('passed')}); positive control "
              f"{control.get('condition')}: KL > 0 on {control.get('documents_kl_above_0')} and top-1 changed on "
              f"{control.get('documents_top1_below_1')} of {control.get('documents')} documents. Mapped rows = "
              "patched positions that receive non-native values, summed over documents.", ""]
    lines += _injection_table(injection["conditions"])
    if injection["contrasts"]:
        lines += ["", "### Paired contrasts (condition minus baseline, shared document resamples)", ""]
        lines += _contrast_table(injection["contrasts"])
    for name, group in injection["groups"].items():
        lines += ["", f"### Stratum {name}: {group['documents']} documents, {group['scored_tokens']} scored tokens", ""]
        if group["documents"]:
            lines += _injection_table(group["conditions"], mapped=False)
            floor_contrasts = [item for item in group["contrasts"] if item["baseline"] == "mean_all"]
            if floor_contrasts:
                lines += [""] + _contrast_table(floor_contrasts)
    if injection["groups_definition"]:
        lines += ["", f"Strata: {injection['groups_definition']}."]
    names = [item["name"] for item in injection["conditions"] if item["name"] not in ("native", "sanity_native_all")]
    per_document = {item["document_id"]: item for item in injection.get("per_document") or []}
    lines += ["", "### Per document (delta NLL / KL per condition)", "",
              "| document | subset | tokens | prefix | patched | scored | " + " | ".join(names) + " |",
              "|" + "---|" * (6 + len(names))]
    for item in injection["scored_documents"]:
        scores = (per_document.get(item["document_id"]) or {}).get("conditions", {})
        cells = " | ".join(f"{_fmt(scores[name]['nll_delta_mean'])} / {_fmt(scores[name]['kl_mean'])}"
                           if name in scores else "n/a" for name in names)
        lines.append(f"| {item['document_id'][:10]} | {item.get('subset')} | {item['n_tokens']} | "
                     f"{item['prefix_tokens']} | {item['patched_positions']} | "
                     f"{min(item['continuation_tokens'], injection['max_scored'])} | {cells} |")
    attention = summary["attention"]
    lines += ["", "## Attention output (offline, test headline)", "",
              "Validation: " + "; ".join(f"layer {item['layer']} ({item['attention']}) max rel. error "
                                         f"{item['max_relative_error']:.2e}" for item in attention["validation"]),
              "", "Cosine of the gated, W_o-projected, post-normed contribution (SDPA head cosine on layers without "
              f"a gate). floor = {attention.get('floor_variant')} (information-free); skill = (cos - floor) / "
              "(1 - floor). Query sets: scored = the queries whose logits injection scores; affected = continuation "
              "queries that can see a patched key; cont. = every final-message token; all = every token.", "",
              "| layer | type | variant | scored cos | scored floor | scored skill | affected cos | affected floor | "
              "affected skill | cont. cos | all cos |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    table = {}
    for item in attention["layers"]:
        if item["split"] == attention["headline_split"] and item["variant"] != FLOOR_VARIANT:
            table.setdefault((item["layer"], item["attention"], item["variant"]), {})[item["query_set"]] = item
    for (layer, kind, variant), sets in sorted(table.items()):
        cells = []
        for name in ("scored", "affected"):
            cells += [_fmt(value) for value in _attention_metric(sets.get(name, {}))]
        cells += [_fmt(_attention_metric(sets.get(name, {}))[0]) for name in ("continuation", "all")]
        lines.append(f"| {layer} | {kind} | {variant} | " + " | ".join(cells) + " |")
    lines += ["", "## Timings", "", "| stage | adopted | wall s | tokens | tokens/s |", "|---|---|---|---|---|"]
    for item in summary["timings"]:
        lines.append(f"| {item['stage']} | {'yes' if item['adopted'] else 'no'} | {_fmt(item['wall_seconds'], 0)} | "
                     f"{_fmt(item.get('tokens'))} | {_fmt(item.get('tokens_per_second'), 1)} |")
    lines += ["", "## Caveats", ""] + [f"- {text}" for text in summary["caveats"]] + [""]
    return "\n".join(lines)


def stage_report(run):
    existing = [rel for rel in ("summary.json", "summary.md") if run.exists(rel)]
    if existing and run.reusable("report", existing[0]):
        for rel in existing:
            run.path(rel).unlink()
    run.mark_started("report")
    prepare = run.records["prepare"]
    timings = []
    for stage in STAGES[:-1]:
        record = run.records.get(stage)
        if record is None:
            continue
        metrics = record["metrics"]
        tokens = (metrics.get("decoded_tokens_this_invocation") if stage == "inject"
                  else metrics.get("tokens"))
        timings.append({"stage": stage, "adopted": record["adopted"], "wall_seconds": record["wall_seconds"],
                        "tokens": tokens, "tokens_per_second": metrics.get("tokens_per_second")})
    summary = {
        "format_version": 1, "generated": now_iso(),
        "models": {role: {"id": run.model_id(role), "architecture": ROLES[role],
                          "sha256": prepare["inputs"][f"{role}_model_sha256"]} for role in ROLES},
        "corpus": {"sha256": prepare["inputs"]["corpus_sha256"], "splits": prepare["metrics"]["splits"]},
        "coverage": read_json(run.path("alignment/coverage.json")),
        "fit": fit_summary(run),
        "attention": attention_summary(run),
        "injection": injection_summary(run),
        "timings": timings,
        "stage_digests": {stage: run.records[stage]["outputs_sha256"] for stage in DEPENDS["report"]},
        "caveats": CAVEATS,
    }
    write_new(run.path("summary.json"), json_text(summary))
    write_new(run.path("summary.md"), markdown(summary))
    return ["summary.json", "summary.md"], {"sections": sorted(summary)}, False


STAGE_FUNCTIONS = {
    "prepare": stage_prepare, "smoke": stage_smoke,
    "collect-source": lambda run: stage_collect(run, "collect-source"),
    "collect-target": lambda run: stage_collect(run, "collect-target"),
    "align": stage_align, "fit": stage_fit, "attention": stage_attention, "inject": stage_inject,
    "report": stage_report,
}


def execute(run):
    a = run.args
    first = STAGES.index(a.from_stage) if a.from_stage else 0
    last = STAGES.index(a.until)
    for index, stage in enumerate(STAGES[:last + 1]):
        run.current = stage
        record = run.read_record(stage)
        if record is not None:
            run.records[stage] = record
            run.verify(stage, record)
            run.event("stage_skip", stage=stage, reason="complete and verified", adopted=record["adopted"])
            continue
        if index < first and not (a.adopt and stage in ADOPTABLE):
            raise RunError(f"--from {a.from_stage}: stage {stage} has no record"
                           + (" (pass --adopt to validate and adopt its existing outputs)"
                              if stage in ADOPTABLE else ""))
        run.adopt_only = index < first
        run.current = stage
        run.event("stage_start", stage=stage)
        started, clock = now_iso(), time.monotonic()
        outputs, metrics, adopted = STAGE_FUNCTIONS[stage](run)
        wall = time.monotonic() - clock
        if adopted and stage != "prepare":
            wall = metrics.get("seconds") if isinstance(metrics.get("seconds"), (int, float)) else None
        record = run.write_record(stage, outputs, metrics, started, wall, adopted=adopted)
        run.event("stage_done", stage=stage, adopted=adopted, wall_seconds=None if wall is None else round(wall, 1),
                  outputs_sha256=record["outputs_sha256"],
                  **{key: metrics[key] for key in ("tokens", "tokens_per_second", "decoded_tokens",
                                                   "decoded_tokens_this_invocation")
                     if metrics.get(key) is not None})
        run.current = None
        if stage == "smoke" and not run.adopt_only:
            remaining_eta(run)


class _TerminationSignals:
    """SIGTERM/SIGHUP -> SystemExit so running children are terminated and cleaned up.

    A signal that is ignored on entry (SIGHUP under nohup) stays ignored.
    """

    SIGNALS = tuple(getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name))

    def __enter__(self):
        self.previous = {}
        if threading.current_thread() is threading.main_thread():
            for number in self.SIGNALS:
                if signal.getsignal(number) is signal.SIG_IGN:
                    continue
                self.previous[number] = signal.signal(number, self._raise)
        return self

    @staticmethod
    def _raise(number, frame):
        raise SystemExit(128 + number)

    def __exit__(self, *exc):
        for number, handler in self.previous.items():
            signal.signal(number, handler)
        return False


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True, type=Path, help="run directory (new unless --resume)")
    parser.add_argument("--resume", action="store_true", help="continue an existing run directory")
    parser.add_argument("--adopt", action="store_true",
                        help="with --resume: validate and record existing prepare/smoke/collect outputs")
    parser.add_argument("--from", dest="from_stage", choices=STAGES,
                        help="never run stages before this one (they must be complete or adoptable)")
    parser.add_argument("--until", choices=STAGES, default="report", help="last stage to run")
    parser.add_argument("--probe", required=True, type=Path, help="kvpark-kv-probe binary")
    parser.add_argument("--corpus", required=True, type=Path, help="corpus v2 conversations.jsonl")
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-model", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--ctx", type=int, default=5120)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16, help="kv_probe threads (collection, injection)")
    parser.add_argument("--fit-threads", type=int, default=16, help="BLAS threads of the numpy stages")
    parser.add_argument("--eval-layers", default="auto", type=lambda t: parse_layers(t, "--eval-layers"))
    parser.add_argument("--attn-out-layers", default="auto", type=lambda t: parse_layers(t, "--attn-out-layers"))
    parser.add_argument("--fit-ks", default=DEFAULT_KS, help="top-k source layer counts (clamped to the source)")
    parser.add_argument("--fit-alphas", help="fit_layers --alphas (default: its grid)")
    parser.add_argument("--attention-variants", default="kv,k,v,mean_kv",
                        help="attention_eval variants; mean_kv is the information-free floor of the skill column")
    parser.add_argument("--inject-docs", type=int, default=12)
    parser.add_argument("--inject-min-tokens", type=int, default=600)
    parser.add_argument("--inject-max-tokens", type=int, default=3500)
    parser.add_argument("--inject-conditions", default="all", help="all or comma-separated inject.py conditions")
    parser.add_argument("--inject-focus-layer", default="auto", help="auto (middle full layer) or a layer")
    parser.add_argument("--inject-tail", type=int, default=64)
    parser.add_argument("--inject-docs-per-run", type=int, default=4)
    parser.add_argument("--inject-long-prefix", type=int, default=256,
                        help="patched prefix positions from which a document counts as long-prefix in the report")
    parser.add_argument("--max-scored", type=int, default=256)
    parser.add_argument("--source-tokens-per-second", type=float, default=19.0,
                        help="ETA prior (measured real Qwen collection with k,v on 16 layers)")
    parser.add_argument("--target-tokens-per-second", type=float, default=17.0,
                        help="ETA prior for Muse collection and patch-score")
    parser.add_argument("--skip-memory-check", action="store_true", help="tiny fixtures only")
    return parser


def check_args(args):
    if args.adopt and not args.resume:
        raise RunError("--adopt requires --resume")
    if args.from_stage and STAGES.index(args.from_stage) > STAGES.index(args.until):
        raise RunError("--from must not come after --until")
    if (not 0 < args.ctx <= TOKENIZE_CTX or not 0 < args.batch <= args.ctx or not 0 < args.threads <= 256
            or not 0 < args.fit_threads <= (os.cpu_count() or 1)):
        raise RunError("need 0 < batch <= ctx <= 32768, threads in 1..256, fit threads in 1..CPU count")
    if (args.inject_docs < 1 or args.inject_min_tokens < 1 or args.inject_max_tokens < args.inject_min_tokens
            or args.inject_tail < 1 or not 0 < args.inject_docs_per_run <= 64 or args.max_scored < 1
            or args.inject_long_prefix < 1
            or args.source_tokens_per_second <= 0 or args.target_tokens_per_second <= 0):
        raise RunError("injection sizes and ETA priors must be positive")
    if args.inject_focus_layer != "auto" and not args.inject_focus_layer.isdigit():
        raise RunError("--inject-focus-layer must be auto or a layer index")
    parse_ints(args.fit_ks, "--fit-ks")
    for path, name in ((args.probe, "--probe"), (args.corpus, "--corpus"), (args.source_model, "--source-model"),
                       (args.target_model, "--target-model")):
        if not path.is_file():
            raise RunError(f"{name} {path} is not a file")


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
    run = Run(args)
    with (root / "stages" / ".lock").open("a") as lock, _TerminationSignals():
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunError(f"another run_v2 process (or a child it left running) is using {root}") from None
        run.lock_fds = (lock.fileno(),)
        run.event("run_start", output=str(root), resume=args.resume, adopt=args.adopt,
                  from_stage=args.from_stage, until=args.until)
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
