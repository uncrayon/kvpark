"""Onboard one model into the universal KV hub end to end, resumable stage by stage.

    campaign.py run --name NAME [--catalog-id ID] [--gguf GGUF] [--stages STAGE,...] [--redo STAGE,...]
    campaign.py plan --name NAME [--catalog-id ID]   (statuses, what a run would do, the capture forecast; loads no model)
    campaign.py status [NAME ...]       (and assembled hubs an onboarding has outdated since)
    campaign.py assemble --output DIR [--evaluate] NAME [NAME ...]
    campaign.py evaluate-onboarded --hub DIR NAME [NAME ...]   (assemble --evaluate's R^2 again, e.g. after a guard stop)
    campaign.py reencode [--encoder METHOD] NAME [NAME ...]   (fit onboarded models' encoders again, nothing else)
    campaign.py exec --memory-bytes N [--timeout S] -- COMMAND ...   (one command under the GPU/memory lock)

One model per `run`. NAME is the model's name in the hub and, by default, its id in
evidence/models/catalog.json (GGUF, architecture, smoke prefill speed; --catalog-id names another
entry); --gguf and --capture-tps cover a model the catalog lacks. Everything lands in evidence/universal-kv/campaign/NAME/: state.json (per
stage its status, wall time, lock waits, disk, peak RSS, commands and result), log.jsonl (events),
commands.sh (every command a stage ran, runnable as it stands: a model-loading command through
`campaign.py exec`, which takes the lock and checks memory as the run did), logs/ (their output)
and the stage outputs:

  render     the hub corpus (corpus v2 + v3-train: the 248 conversations hub4-d4096 was fitted on)
             rendered with the model's own chat template, chat.py corpus mode -> render/records.jsonl
             and template.jinja. Parity with llama.cpp on a sample (two conversations per corpus
             source) through template_check's server mode on a tokenizer twin (ports 18690-18699,
             KVPARK_TWIN_PORT moves them): the sample's prompts (the final assistant message left
             out) under every option set, and its whole conversations as the records hold them (no
             generation prompt). Every whole conversation must match with no server error, and every
             prompt but the server's refusals of option sets named with --parity-tolerate (none by
             default), at least one.
             A record that differs from the server's own whole-conversation bytes (a server context
             such as Qwen3.6's, which keeps earlier reasoning) is recorded with that context and the
             first difference, not refused. Then the probe tokenizes every record -> render/tokens.json.
  capture    the capture forecast (capture_plan: every dev and test document, then train documents
             round-robin over the corpus sources, until the size or time budget: 60 GB, one hour, by
             the model's cache width and 0.4 of its smoke prefill speed) -> capture-plan.json; then
             the capture in chunks (collect_v2.py; the Vulkan probe, or --mode states on the CPU probe
             for STATES_ARCHITECTURES, whose graphs name no pre-RoPE key, or collect on the CPU probe
             for CPU_CAPTURE_ARCHITECTURES, whose Vulkan capture runs differ from plain runs): chunk 0 the dev and test
             documents, then train chunks in the forecast's order while the measured speed (rows per
             probe second, lock waits excluded) and the bytes so far say the next one fits (a states
             chunk counts twice: its state files and its rows exist together). Every chunk is killed
             at 1.5 x the time budget of probe time; a train chunk that is, ends the capture there.
             -> capture/chunk-NN/ (+ .docs.jsonl, .json), capture-docs.jsonl (all captured documents)
  onboard    fit_hub.py --base hub4-d4096 with every base model as a pivot, --link-base (the base's
             files are read-only hard links, checked against the arrays in memory first),
             --onboard-encoder ONBOARD_ENCODER and --evaluate new, run with FIT_PYTHON (torch with
             ROCm: the fit's products on the GPU, fit_hub --fit-device auto) -> hub/ (format 1: held-out
             R^2 per kind against each pivot in hub/evaluation.json), then convert_hub.py --reuse
             hub4-d4096-v2 -> hub-v2/ (only the new model's sides are converted); the new model's
             files' sha256, its encoder, the reverse family's variance cap (hub.ONBOARD_VARIANCE_CAP) and
             the encoders' train diagnostics (the smallest per-dimension R^2, the variance ratios, the
             dimensions capped, any warning) go to state.json and the report
  tables     token_tables.py on the onboarding's spans (the model and the pivots, --only the
             model) -> token-tables/NAME.npz
  recurrent  hybrids whose recurrent cell holds a state (S rows: SSM, delta rule, WKV, lightning):
             r0_experiments.py calibrate (training conversations of corpus v2 and v3-train) ->
             recurrent/recurrent-mean.npz. Skipped for a cell of convolution taps only (lfm2): it
             holds the last few inputs, which the native tail replaces after as many tokens.
  evaluate   pair_eval.py qwen38 -> NAME and NAME -> qwen38 through hub-v2 on RESULTS.md section 2's
             eight held-out chats (EVAL_IDS) where the target renders them 4,096-16,384 tokens
             before the final message, at least six, else pair_eval's own pick from the same corpus
             and sources. A direction whose target renders some of them shorter also runs the
             longer set: pair_eval's pick of every chat it renders 6,144-16,384 tokens long (at least
             4,096 translated). Token tables; into a hybrid the recurrent mean (into a conv-only one
             the zero cell). Then Qwen3-4B -> Qwen3.8 under the very settings of NAME -> qwen38
             (campaign/_baseline/, run once for every campaign): same chats, same floors, so the
             source is the only difference. When the render stage found that the model's server
             renders conversations other than its template's defaults (Qwen3.6 keeps earlier
             reasoning), both directions run again with both models rendered as their servers
             render them (pair_eval --server-render; -server), what a live switch holds.
             -> eval/<source>-to-<target>[-server][-long]/, each with <dir>.command.json (a finished
             direction is kept only for the same command; an unfinished one of the same command, e.g.
             one the resource guard ended, continues: pair_eval --resume keeps its saved states and
             scored documents)
  report     report.json and report.md: wall time (and the command time of every attempt since the
             stage last started over, from log.jsonl), lock waits, disk and peak RSS per stage, the
             capture, R^2, ΔNLL / KL / top-1 with document-bootstrap 95% CIs, paired differences
             (hub - floor, hub_prefix - hub, ...) with their CIs, the share of the floor's KL removed
             and how much of it one chat carries, the comparison with Qwen3-4B, the -server runs
             against the templates' rendering chat by chat, the recommended
             settings (the native prompt start: into the Qwen family always, RESULTS.md section 2;
             otherwise when the paired CI of hub_prefix - hub KL says so, and on by default when it
             spans 0; with a DFlash draft in the catalog, --draft-window from its sliding window)

GPU/memory lock: every command that loads a model or a hub (capture, onboard, tables, recurrent,
evaluate, assemble --evaluate) holds an exclusive flock on /tmp/kvpark-campaign-gpu.lock, and starts
only once MemAvailable is at least what it needs resident (the largest model it loads, and for
pair_eval the hub bundle it loads too; fit_hub the base hub + fit_hub_extra, which grows with the
new model's K/V width and the base hub's dev spans, token_tables TABLES_MEMORY, all measured: the captures they memory-map are
file pages) + 25 GiB (otherwise it releases the lock, waits 60 s and tries again; every wait is
logged). Every command's peak anonymous memory (its process group's RssAnon) and RSS are recorded. The command inherits the locked descriptor (pass_fds), so the lock stays
held while it runs even if campaign.py dies; it runs in its own process group, which campaign.py
kills on SIGINT, SIGTERM, SIGHUP or a timeout. render loads no model (a tokenizer twin).

Resource guard (resource_guard.py; after 2026-10-05's thermal shutdown): every command runs with
BLAS/OpenMP threads capped at 8 unless set (the probe with --threads 8), starts only once the CPU is
at most 88 C (after its locks), and runs under a Watch that samples it every 2 s into
logs/<label>.guard.jsonl, stops its process group at 90 C, at 85% swap use or below 8 GiB
MemAvailable, continues it at 88 C with memory recovered, and ends it at 98 C on two samples in a
row or when the sensors are lost for 60 s (the stage fails: GuardStop). Pauses count neither toward
a command's seconds nor its timeout. The CPU-heavy commands (fit_hub, token_tables, r0 calibrate,
pair_eval, a capture on the CPU probe) also hold the CPU lock,
/tmp/kvpark-campaign-cpu.lock, taken after the GPU/memory lock and inherited like it
(KVPARK_CPU_LOCK_FD): one CPU-heavy job at a time, hand-run fits (fit_hub.py takes it too)
included. The GPU evaluations (the evaluate stage's pair_eval runs and the baseline, assemble
--evaluate) run with resource_guard.GPU_EVALUATION's defaults: stopped at 90 C, continued at 88 C,
4 BLAS and probe threads (commands.sh: exec --gpu-evaluation). KVPARK_GUARD_* change the thresholds
(KVPARK_GUARD=off disables pausing and aborting, with a warning).

Re-encoding: `campaign.py reencode [--encoder METHOD] NAME ...` fits an onboarded model's encoder again
with another onboarding encoder (fit_hub.py --reencode: its view transforms and decoders, and every other
model's adapters, stay exactly as they are), swaps hub/ and hub-v2/ (the previous ones, the fit log and
the evaluation runs out of the model go to history/onboard-<encoder>-<time>/), and marks evaluate and
report "outdated": the next `run` evaluates only the directions out of the model again (the runs into it
read unchanged sides and are kept) and reports. The same method again is refused, except that a reverse
or reverse_z encoder fitted before the variance cap existed (none recorded) may be fitted again with it.

Resume: a stage runs when it is not done; --redo STAGE runs it again with every stage that depends
on it. A stage that failed or was interrupted is cleaned and run again, except that each published
capture chunk whose documents, probe build and settings match is adopted (collect_v2's manifest),
and each finished evaluation direction run by the same command is kept. Only one run per model at a
time (a lock in its directory). Every command entry records git HEAD, the changed files under
experiments/ and src/, and the sha256 of the scripts and probe it ran.

How the hub grows: every model is onboarded against the same base (hub4-d4096, pivots qwen38, muse,
qwen3-4b, ornith), so its adapters depend on the base and its own captures only, never on another
onboarded model, and no onboarding refits anything. `assemble` builds one hub for the bridge from
the base and any number of onboarded models (hub.merge_bundles: read-only hard links on the same
filesystem, no disk, any order), in both formats: DIR and DIR-v2, and records it in each model's
assembled.json; `status` flags an assembled hub whose files are no longer the model's onboarding
(after --redo onboard). Adding a model later means assembling a new directory with one more name; a
hub the bridge reads is never modified. `assemble --evaluate` also measures held-out R^2 between
the onboarded models (they all captured the same dev and test documents) -> DIR/evaluation-onboarded.json:
as a child process (`campaign.py evaluate-onboarded`, FIT_PYTHON: its products on the GPU) under the
GPU/memory and CPU locks and a Watch with the GPU-evaluation defaults, which can stop it at any moment (an
in-process evaluation on the CPU could be paused only between chunks: on 2026-10-07 the package went from
82 C to 98 C within one). `campaign.py evaluate-onboarded --hub DIR NAME ...` runs it again (after a guard
stop, the assembled directory is kept).
"""

import os

import resource_guard

if __name__ == "__main__":
    resource_guard.cap_threads(os.environ)

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import types

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "linear_transfer"))

EVIDENCE = ROOT / "evidence"
CAMPAIGN_DIR = EVIDENCE / "universal-kv" / "campaign"
CATALOG = EVIDENCE / "models" / "catalog.json"
BASE_HUB = EVIDENCE / "universal-kv" / "hub4-d4096"
BASE_HUB_V2 = EVIDENCE / "universal-kv" / "hub4-d4096-v2"
HUB_CORPUS = (EVIDENCE / "linear-transfer" / "corpus-v2" / "conversations.jsonl",
              EVIDENCE / "linear-transfer" / "corpus-v3-train" / "conversations.jsonl")
EVAL_CORPUS = EVIDENCE / "linear-transfer" / "corpus-long-chat" / "conversations.jsonl"
EVAL_SOURCES = "multi_turn_reasoning_if_think,multi_turn_concat_sessions"
# RESULTS.md section 2: the eight held-out long chats of every pair into Qwen3.8 and Ornith
# (evidence/universal-kv/pair-qwen38-ornith/report.json)
EVAL_IDS = ("f3fcee861dffa373004b559a8783d2ec6c2c4d38d14dc5ad7254702a465e491c",
            "2ddc213738d7d26c1db43711a9155a04c6da0a7e55e70dcf4d511e3b27c7dd47",
            "1749e42d342820f1f019d42efcf230243576024602e541fa921d2a336a2608a2",
            "b19deab5418ec186122b4a31b06875b23d5397abdc1aa2d869ad7bc9e2b628c4",
            "f2eef633b953b3831920db700f13243977b419473c1ad918c7d7988465b5566c",
            "08e28e454d949435be59b185245534a460759cd67601f94db40a904c9b9ea313",
            "a6c4726d120234ca7539c9aace4fd5c57dcb1de5a7e3365def1cf314f530c1bd",
            "3b211f08c7afcc53e539ff6b2262fa067ba580835ddba6418225dcb6e1d94138")
EVAL_MIN_DOCUMENTS = 6
EVAL_PREFIX = (4096, 16384)
# the longer set, for a target that renders the fixed set short: every chat of the same corpus and
# sources it renders 6,144-16,384 tokens before the final message (T0 >= 4,096 with the 2,048 tail)
LONG_MIN_PREFIX = 6144
LONG_LIMIT = 12
LONG_MIN_DOCUMENTS = 8
REFERENCE = "qwen38"
REFERENCE_GGUF = Path("/srv/models/qwen3.8-27b-q8/Qwen3.8-27B-Q8_0.gguf")
REFERENCE_MEAN = EVIDENCE / "universal-kv" / "recurrent-init-20261003" / "calibrate-qwen38" / "recurrent-mean.npz"
REFERENCE_TABLES = EVIDENCE / "universal-kv" / "token-tables"
# the like-for-like comparison of NAME -> qwen38: a base model into Qwen3.8 under the same settings
BASELINE_SOURCE = "qwen3-4b"
BASELINE_GGUF = Path("/srv/models/qwen3-4b-instruct/Qwen3-4B-Instruct-2507-Q8_0.gguf")
BASELINE_DIR = CAMPAIGN_DIR / "_baseline"
# RESULTS.md section 2: the Qwen family (dense Qwen3, the Qwen3.5 hybrids) needs its native prompt start
NATIVE_PREFIX_ARCHITECTURES = {"qwen3", "qwen35", "qwen35moe", "qwen3next"}
# the DFlash draft window the bridge assumes without --draft-window (src/kvpark/bridge/switch.py DRAFT_WINDOW)
BRIDGE_DRAFT_WINDOW = 2048
# a target outside it, when the paired CI of hub_prefix - hub KL spans 0: the native prefix stays on (the
# target's exact values from a 14-30 token prefill; the attention-sink collapse it prevents, Qwen3-4B KL
# 0.77 without it, is far larger than its cost where it hurts, Muse +0.05 nats)
NATIVE_PREFIX_DEFAULT = True
PROBE_VULKAN = ROOT / "runtime-kvtools" / "kvpark-kv-probe-vulkan" / "kvpark-kv-probe"
PROBE_CPU = ROOT / "runtime-kvtools" / "kvpark-kv-probe" / "kvpark-kv-probe"
LLAMA_SERVER = ROOT / "runtime" / "build" / "bin" / "llama-server"
PYTHON = HERE.parent / "linear_transfer" / ".venv" / "bin" / "python"      # a venv: never resolve symlinks
# fit_hub's Python: torch with ROCm runs the fit's products on the GPU (fit_backend.py; README.md "Fit placement":
# the LFM2-2.6B onboarding in 1.8x less wall time and 2.5x less energy, held-out R^2 within 1e-5 of the CPU fit's);
# $KVPARK_FIT_PYTHON names another; without either the experiment venv fits on the CPU (fit_hub --fit-device auto)
FIT_PYTHON = Path.home() / "venvs" / "rocm" / "bin" / "python"
# how an onboarded model's encoder is fitted (fit_hub.py --onboard-encoder, hub.solve_onboard_encoder): the hub's
# default (hub.DEFAULT_ONBOARD_ENCODER, chosen end to end in RESULTS.md section 9; the pilots used "ridge")
ONBOARD_ENCODER = "reverse"
GPU_LOCK = Path(resource_guard.GPU_LOCK)
CPU_LOCK = Path(resource_guard.CPU_LOCK)   # CPU-heavy commands, taken after GPU_LOCK
GIB = 1024 ** 3
GPU_MARGIN = 25 * GIB
GPU_POLL_S = 60
CHILD_POLL_S = 0.2
MEMORY_SAMPLE_S = 1.0           # how often a command's anonymous memory is sampled
STOP_GRACE_S = 30
# the render stage's tokenizer-twin servers (KVPARK_TWIN_PORT moves the ten ports, as in template_check)
_TWIN_PORT = int(os.environ.get("KVPARK_TWIN_PORT", "18690"))
TWIN_PORTS = range(_TWIN_PORT, _TWIN_PORT + 10)
# graphs that name no pre-RoPE key: their captures come from f32 state saves (collect_v2 --mode states, CPU)
STATES_ARCHITECTURES = {"kimi-k3", "bailingmoe3", "hy_v4", "hy_v3", "glm5-next"}
# graphs whose Vulkan capture run and a plain run differ by more than rounding (the eval callback splits
# the graph; linear_transfer/README.md, Gemma 4): Gemma-4 26B-A4B's full layers hold V exactly in 2% of
# the values, K up to 0.075 off, and a state rebuilt from the captures scores KL 0.014 against the saved
# one. Collected on the CPU probe, where the two agree (V exact, K within 0.7 ulp, rebuilt = saved).
CPU_CAPTURE_ARCHITECTURES = {"gemma4"}
REFUSED_ARCHITECTURES = {"deepseek4": "DeepSeek V4 needs fit_hub.py --heads and a split GGUF the probe cannot load"}
BUDGET_BYTES = 60 * 10 ** 9
BUDGET_SECONDS = 3600
LOAD_SECONDS = 120              # model load and tokenization, outside the per-token rate
# capture tokens/s as a share of the catalog's smoke prefill (ROCm, flash attention, f16 cache): the
# hub4 captures (Vulkan, f32 cache, flash attention off, rows written out) ran Ornith at 0.49 of it
# and Qwen3-4B at 0.33, LFM2 at 0.72; --mode states runs on the CPU (a guess until measured); collect
# on the CPU probe: Gemma-4 26B-A4B's capture ran 80 tokens/s of probe time with 8 threads (0.062 of its
# smoke prefill; a short trial with 4 threads, 45), 2026-10-06, guard pauses excluded: it reaches 90 C
# within 2 s of running and was paused half of the time
CPU_COLLECT_SPEED = 0.06
CAPTURE_SPEED = {("collect", "vulkan"): 0.4, ("collect", "cpu"): CPU_COLLECT_SPEED, ("states", "cpu"): 0.05}
# disk a chunk holds at its peak per byte of rows: states mode keeps every state file of the chunk
# until its rows are written (collect_v2.run_states)
PEAK_FACTOR = {"collect": 1, "states": 2}
CAPTURE_CHUNKS = 6              # train chunks of the forecast budget: the speed is measured between them
CHUNK_MIN_TOKENS = 16384
HARD_FACTOR = 1.5               # probe time limit of the whole capture, x the time budget
CAPTURE_BATCH = 512
CAPTURE_THREADS = resource_guard.DEFAULT_THREADS   # the probe's CPU threads (16 until the 2026-10-05 shutdown)
CAPTURE_CTX = 5120
PROBE_MAX_CTX = 32768           # kv_probe refuses a larger context
PARITY_PER_SOURCE = 2
# what the CPU stages need resident beyond what they load (anonymous memory and, for a GPU fit, the device's:
# GTT on this APU, which no RssAnon shows and which comes off MemAvailable all the same; their RSS is mostly the
# memory-mapped captures, file pages the kernel can drop). fit_hub (fit_hub_extra; fit_hub.memory_estimate is the
# same model) holds the new model's stage-1 covariance and its eigendecomposition (float64 sums and copies of W1 x W1,
# W1 = its per-layer PCA ranks summed, on the host and the device: 28 bytes per cell), its decoder's statistics
# (float64 cross moments and float32 weights: 10 bytes per hub dim x column), its rows in 4,096-row chunks (two on the
# host while the next is read, one on the backend: 12 bytes per chunk row x column), 12 bytes per dev span x pivot x
# hub dim for what does not grow with the model (the pivots' chunk rows and estimates, their dev Moments) and on the
# GPU FIT_HUB_GPU (torch's context and calibration, the pivots' rows on the device). Calibrated on 2026-10-09 on the
# twelve onboardings' anonymous peaks plus torch's reserved peak on the device (the three CPU fits: LFM2-2.6B 11.0 GiB,
# DeepSeek-V2-Lite 16.7, Qwen3.6-35B-A3B 13.0; the GPU fits: Jamba, Nemotron-3.5, Granite, Ling, Kimi-Linear and
# Falcon-H1 13.5-16.5, Gemma-4-26B-A4B 24.2, GLM-4.7-Flash 31.3, Gemma-4-31B 56.4), with the 4.6 GiB base hub; at each
# onboarding's own dev spans the estimate lies 0.5-4 GiB above each. The estimate until then counted anonymous memory
# only: 7 of the 9 GPU onboardings' peaks exceeded it by 1.0-3.9 GiB (and until 2026-10-09 it also counted the dev
# rows tiled over the pivots, 4 bytes per dev span x pivot x column, which the fit has streamed since 2026-10-06: 98
# GiB for Gemma-4-31B, which no lock could grant). The dev spans are bounded by the base hub's (a new rendering only
# removes joint spans).
# token_tables 0.5-0.6 GiB
FIT_HUB_BASE = int(5.5 * GIB)
FIT_HUB_GPU = int(3.5 * GIB)        # a fit on the GPU
FIT_HUB_COVARIANCE_BYTES = 28       # per W1 x W1 cell
FIT_HUB_DECODER_BYTES = 10          # per hub dimension x column of the widest kind
FIT_HUB_CHUNK_BYTES = 12            # per chunk row x column of the widest kind
FIT_HUB_CHUNK = 4096                # fit_hub.py's --chunk
FIT_HUB_DEV_LATENT_BYTES = 12       # per dev span x pivot x hub dimension
TABLES_MEMORY = 2 * GIB

STAGES = ("render", "capture", "onboard", "tables", "recurrent", "evaluate", "report")
REQUIRES = {"render": (), "capture": ("render",), "onboard": ("capture",), "tables": ("onboard",),
            "recurrent": (), "evaluate": ("onboard", "tables", "recurrent"),
            "report": ("render", "capture", "onboard", "tables", "recurrent", "evaluate")}
OUTPUTS = {"render": ("render",), "capture": ("capture", "capture-plan.json", "capture-docs.jsonl",
                                              "capture-spec.json"),
           "onboard": ("hub", "hub.log.jsonl", "hub-v2"), "tables": ("token-tables",), "recurrent": ("recurrent",),
           "evaluate": ("eval",), "report": ("report.json", "report.md")}
STATE_FORMAT = 2


class StageError(RuntimeError):
    """A stage found something wrong with its inputs or outputs (the message says what)."""


class CommandTimeout(StageError):
    """A command ran past its timeout and was stopped (its whole process group)."""


class GuardStop(StageError):
    """The resource guard ended a command: too hot, sensors lost or a pause that never cleared."""


class Terminated(BaseException):
    """SIGTERM or SIGHUP: raised in the main thread so that running commands are stopped too."""


def _terminate(signum, frame):
    raise Terminated(f"signal {signum}")


# -- planning (no model, no I/O beyond what is passed in) ---------------------------------------

def applicable(stage, recurrent):
    """Whether a stage applies to a model (the recurrent mean only to a cell that holds a state)."""
    return stage != "recurrent" or bool(recurrent)


def downstream(stage):
    """The stages that need `stage`, directly or through others."""
    out, frontier = set(), {stage}
    while frontier:
        found = {other for other in STAGES if set(REQUIRES[other]) & frontier} - out
        out |= found
        frontier = found
    return out


def parse_stages(text):
    """['render', ...] from 'render,capture' (any order; kept in STAGES order); '' or None: []."""
    names = [item.strip() for item in (text or "").split(",") if item.strip()]
    unknown = sorted(set(names) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown stages {unknown} (stages: {', '.join(STAGES)})")
    return [stage for stage in STAGES if stage in names]


def plan_run(statuses, *, recurrent, only=None, redo=()):
    """The stages a run executes now, in order.

    statuses: {stage: "done" | "running" | "failed" | "stale" | None}. A stage runs when it is not
    done or when it or a stage it depends on is redone; `only` restricts the run to those stages
    (the rest stay as they are). Every stage that runs needs the stages it requires done and not
    redone, or running before it in this run; otherwise ValueError names what is missing.
    recurrent: whether the model needs a recurrent mean (recurrent_cell)."""
    stale = set()
    for stage in redo:
        stale |= {stage} | downstream(stage)
    todo = [stage for stage in STAGES if applicable(stage, recurrent)
            and (statuses.get(stage) != "done" or stage in stale) and (not only or stage in only)]
    for stage in todo:
        for needed in REQUIRES[stage]:
            if not applicable(needed, recurrent) or needed in todo[:todo.index(stage)]:
                continue
            if statuses.get(needed) != "done" or needed in stale:
                raise ValueError(f"stage {stage} needs {needed}, which is not done (run it too)")
    return todo


def command_seconds(events, recurrent):
    """{stage: seconds} its commands ran (lock waits excluded) over every attempt since its outputs
    last started over, from log.jsonl's events in order: a run's --redo starts over the stages it
    makes stale, while a resumed stage keeps what earlier attempts left (published capture chunks,
    finished evaluations), so their commands count. state.json's stage seconds are the last
    attempt's wall time only."""
    out = {}
    for event in events:
        if event.get("event") == "run":
            for stage in stale_stages(event.get("redo") or [], recurrent):
                out.pop(stage, None)
        elif event.get("event") == "command-done" and event.get("stage") in STAGES:
            out[event["stage"]] = round(out.get(event["stage"], 0.0) + float(event.get("seconds") or 0.0), 1)
    return out


def stale_stages(redo, recurrent):
    """The stages a --redo invalidates (marked stale before anything runs)."""
    out = set()
    for stage in redo:
        out |= {stage} | downstream(stage)
    return [stage for stage in STAGES if stage in out and applicable(stage, recurrent)]


def bytes_per_token(spec):
    """Capture bytes per token: float32 K and V rows of every attention layer (nextn excluded), the
    DSA indexer keys when the model has an indexer, and the token id, byte end and special flag."""
    total = 0
    for layer in spec.attention_layers:
        if spec.layer_types[layer] == "nextn":
            continue
        for key in ("k", "v"):
            heads, dim = spec.head_shape(layer, key)
            total += heads * dim * 4
    if spec.indexer is not None:
        total += len(spec.indexer.layers) * spec.indexer.head_size * spec.indexer_key_slots * 4
    return total + 4 + 8 + 1


def refusal(architecture):
    """Why the campaign cannot onboard a model of this architecture, or None."""
    reason = REFUSED_ARCHITECTURES.get(architecture)
    return None if reason is None else f"{architecture}: {reason}"


def capture_mode(architecture):
    if refusal(architecture):
        raise ValueError(refusal(architecture))
    return "states" if architecture in STATES_ARCHITECTURES else "collect"


def capture_device(architecture):
    """Where the capture runs: "cpu" (the CPU probe) for --mode states and CPU_CAPTURE_ARCHITECTURES,
    else "vulkan"."""
    if capture_mode(architecture) == "states" or architecture in CPU_CAPTURE_ARCHITECTURES:
        return "cpu"
    return "vulkan"


def recurrent_cell(spec):
    """What a model's recurrent cell holds and whether a hybrid target needs its recurrent mean:
    {"layers", "r_bytes", "s_bytes", "mean", "reason"} (bytes per layer row, kvstate's
    recurrent_row_bytes). A cell with a state (S rows: SSM, delta rule, WKV, lightning) carries the
    whole context: a translated cache restores it from the mean (RESULTS.md section 7). A cell of
    convolution taps only (R rows, S = 0: lfm2's short convolution) holds the last few inputs, which
    the native tail (2,048 tokens in pair_eval and the bridge) replaces after as many tokens: its
    mean cannot matter, and the zero cell is restored."""
    layers = list(spec.recurrent_layers)
    if not layers:
        return {"layers": 0, "r_bytes": 0, "s_bytes": 0, "mean": False, "reason": "no recurrent layers"}
    r_bytes, s_bytes = spec.recurrent_row_bytes or (None, None)
    if s_bytes == 0:
        return {"layers": len(layers), "r_bytes": r_bytes, "s_bytes": 0, "mean": False,
                "reason": f"the recurrent cell holds convolution taps only ({r_bytes:,} bytes of recent inputs per "
                          "layer, no state): the native tail replaces them after a few tokens, so its mean cannot "
                          "matter (the zero cell is restored)"}
    return {"layers": len(layers), "r_bytes": r_bytes, "s_bytes": s_bytes, "mean": True,
            "reason": "the recurrent cell holds a state of the whole context: restored from its mean"}


def capture_plan(documents, *, bytes_per_token, tokens_per_second, budget_bytes=BUDGET_BYTES,
                 budget_seconds=BUDGET_SECONDS, load_seconds=LOAD_SECONDS, reserve_tokens=0):
    """The capture forecast: every dev and test document (the onboarding's alpha choice and held-out
    R^2, the same for every model), then train documents round-robin over the corpus sources (each
    source in document-id order) while they fit the budget, the smaller of
    budget_bytes / bytes_per_token - reserve_tokens (the room a chunk's peak needs) and
    (budget_seconds - load_seconds) * tokens_per_second tokens.

    documents: [{"document_id", "split", "source", "tokens"}] (render order, which the plan keeps).
    Returns {"documents": [ids], "held_out": [ids], "order": [every train id, in priority order],
    "tokens", "bytes", "seconds", "budget_tokens", "limit", "by_split", "train_fraction"}; "limit" is
    "corpus" when everything fits, else "bytes" or "time". The capture stage measures the speed and
    decides chunk by chunk; this is what it expects."""
    if bytes_per_token <= 0 or tokens_per_second <= 0:
        raise ValueError("bytes_per_token and tokens_per_second must be positive")
    by_bytes = max(0, budget_bytes // bytes_per_token - reserve_tokens)
    by_time = max(0.0, budget_seconds - load_seconds) * tokens_per_second
    budget = int(min(by_bytes, by_time))
    held = [d for d in documents if d["split"] in ("dev", "test")]
    total = sum(d["tokens"] for d in held)
    if total > budget:
        raise ValueError(f"the dev and test documents alone ({total} tokens) exceed the budget ({budget} tokens)")
    sources = {}
    for document in sorted((d for d in documents if d["split"] == "train"), key=lambda d: d["document_id"]):
        sources.setdefault(document["source"], []).append(document)
    order, queues = [], [sources[name] for name in sorted(sources)]
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.pop(0))
    chosen = {d["document_id"] for d in held}
    train_total = sum(d["tokens"] for d in order)
    for document in order:
        if total + document["tokens"] <= budget:
            chosen.add(document["document_id"])
            total += document["tokens"]
    picked = [d for d in documents if d["document_id"] in chosen]
    train_tokens = sum(d["tokens"] for d in picked if d["split"] == "train")
    limit = "corpus" if len(picked) == len(documents) else ("bytes" if by_bytes <= by_time else "time")
    by_split = {}
    for document in picked:
        item = by_split.setdefault(document["split"], {"documents": 0, "tokens": 0})
        item["documents"] += 1
        item["tokens"] += document["tokens"]
    return {"documents": [d["document_id"] for d in picked], "held_out": [d["document_id"] for d in held],
            "order": [d["document_id"] for d in order], "tokens": int(total),
            "bytes": int(total * bytes_per_token), "seconds": round(load_seconds + total / tokens_per_second, 1),
            "budget_tokens": budget, "limit": limit, "by_split": by_split,
            "train_fraction": round(train_tokens / train_total, 4) if train_total else None}


def held_out_need(documents, *, bytes_per_token, tokens_per_second, load_seconds=LOAD_SECONDS):
    """What the dev and test documents alone take, for a budget that cannot hold them (a wide model:
    Gemma-4 31B's 1.8 MB per token puts them at 76 GB, past the default 60)."""
    tokens = sum(d["tokens"] for d in documents if d["split"] in ("dev", "test"))
    return (f"the held-out documents alone take {tokens * bytes_per_token / 10 ** 9:.1f} GB and "
            f"{(load_seconds + tokens / tokens_per_second) / 60:.0f} minutes at {tokens_per_second:.1f} tokens/s: give "
            "--budget-gb and --budget-minutes above that, with room for the training documents")


def chunk_tokens_for(budget_tokens):
    """Tokens per train chunk: a sixth of the forecast budget, at least CHUNK_MIN_TOKENS."""
    return max(CHUNK_MIN_TOKENS, int(budget_tokens) // CAPTURE_CHUNKS)


def next_chunk(pending, tokens, *, chunk_tokens, budget_tokens):
    """The next train chunk from the front of `pending` (the forecast's priority order, consumed):
    a document is taken when it fits the remaining budget (budget_tokens), dropped when it does not
    (the budget only shrinks), and the chunk closes before the document that would take it past
    chunk_tokens (a larger document makes a chunk of its own). Returns (chunk ids, dropped ids)."""
    chunk, total, dropped = [], 0, []
    while pending:
        n = tokens[pending[0]]
        if n > budget_tokens - total:
            dropped.append(pending.pop(0))
            continue
        if chunk and total + n > chunk_tokens:
            break
        chunk.append(pending.pop(0))
        total += n
    return chunk, dropped


def parity_sample(conversations, per_source=PARITY_PER_SOURCE):
    """Up to per_source conversations of each source (the first by id), in corpus order."""
    taken, out = {}, []
    for conversation in sorted(conversations, key=lambda c: c["id"]):
        if taken.get(conversation["source"], 0) < per_source:
            taken[conversation["source"]] = taken.get(conversation["source"], 0) + 1
            out.append(conversation["id"])
    keep = set(out)
    return [c for c in conversations if c["id"] in keep]


def prompt_shape(conversation):
    """The request a server sees before the final answer: the conversation without its trailing
    assistant messages."""
    messages = list(conversation["messages"])
    while messages and messages[-1]["role"] == "assistant":
        messages.pop()
    return {**conversation, "messages": messages}


def option_label(options):
    """A request option set as --parity-tolerate names it: "default" or its compact sorted JSON."""
    return "default" if not options else json.dumps(options, sort_keys=True, separators=(",", ":"))


def parity_problems(prompts, whole, tolerate=()):
    """Why template parity fails ([] when it holds): every whole conversation must match the
    server's bytes with no server error; among the prompts, every case must match except the
    server's refusals of the option sets in `tolerate` (option_label), and at least one must
    match. A twin that refuses everything therefore fails instead of passing unchecked."""
    problems = []
    counts = whole["counts"]
    if not counts["cases"] or counts["matched"] != counts["cases"]:
        problems.append(f"whole conversations: {counts['matched']} of {counts['cases']} match (server errors "
                        f"{counts['server_errors']}, mismatches {counts['mismatch']}, chat.py errors "
                        f"{counts['chat_py_errors']})")
    counts = prompts["counts"]
    refused = [case for case in prompts["cases"] if "server_error" in case]
    untolerated = sorted({option_label(case["options"]) for case in refused
                          if option_label(case["options"]) not in set(tolerate)})
    if counts["matched"] == 0:
        problems.append("prompts: no case matches")
    if untolerated:
        problems.append(f"prompts: the server refused option sets not tolerated: {untolerated}")
    if counts["matched"] + len(refused) != counts["cases"]:
        problems.append(f"prompts: {counts['mismatch']} mismatches, {counts['chat_py_errors']} chat.py errors of "
                        f"{counts['cases']} cases")
    return problems


def text_difference(record, server, width=60):
    """Where a record's bytes leave the server's: {"at", "record_chars", "server_chars", "record",
    "server"} with `width` characters around the first difference (None: equal)."""
    at = next((i for i, (a, b) in enumerate(zip(record, server)) if a != b), None)
    if at is None:
        if len(record) == len(server):
            return None
        at = min(len(record), len(server))
    return {"at": at, "record_chars": len(record), "server_chars": len(server),
            "record": record[max(0, at - width):at + width], "server": server[max(0, at - width):at + width]}


# -- evaluation statistics ----------------------------------------------------------------------

def boot_ratio(sums, counts, seed=0, draws=10000):
    """Token-weighted mean sum(sums) / sum(counts) and its document-bootstrap 95% interval."""
    sums, counts = np.asarray(sums, float), np.asarray(counts, float)
    index = np.random.default_rng(seed).integers(0, len(sums), (draws, len(sums)))
    stats = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    return float(sums.sum() / counts.sum()), [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]


def condition_names(rows):
    return [key for key in rows[0] if isinstance(rows[0][key], dict) and "nll_delta" in rows[0][key]]


def conditions_of(rows):
    """{condition: {"nll_delta", "nll_ci95", "kl", "kl_ci95", "top1", "top1_ci95"}} of pair_eval
    document rows: token-weighted over the documents (each scores n_scored tokens), with
    document-bootstrap 95% intervals."""
    counts = [row["n_scored"] for row in rows]
    out = {}
    for name in condition_names(rows):
        item = {}
        for metric, label in (("nll_delta", "nll"), ("kl", "kl"), ("top1", "top1")):
            item[metric], item[f"{label}_ci95"] = boot_ratio([row[name][metric] * row["n_scored"] for row in rows],
                                                             counts)
        out[name] = item
    return out


def condition_table(report):
    """conditions_of a pair_eval report's documents (pair_eval's own summary has the ΔNLL interval
    and plain document means of KL and top-1)."""
    return conditions_of(report["documents"])


SIGN_FLIP_EXACT = 16             # sign_flip_p enumerates all 2^n sign patterns up to this many documents
SIGN_FLIP_DRAWS = 1 << 16


def sign_flip_p(diff, weight, seed=0):
    """Two-sided p of the paired sign-flip (permutation) test of a token-weighted mean difference: the share of the
    2^n patterns of per-document signs whose weighted mean is at least as far from 0 as the observed one (exact up to
    SIGN_FLIP_EXACT documents, else SIGN_FLIP_DRAWS random patterns). With 8 chats the percentile bootstrap's interval
    excludes 0 more often than this test rejects (its smallest p is then 2/256): read both."""
    diff, weight = np.asarray(diff, float), np.asarray(weight, float)
    n = len(diff)
    if n == 0 or weight.sum() <= 0:
        return None
    exact = n <= SIGN_FLIP_EXACT
    signs = (1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) if exact
             else np.random.default_rng(seed).choice(np.array([-1, 1]), size=(SIGN_FLIP_DRAWS, n)))
    stats = (signs * diff) @ weight / weight.sum()
    observed = abs(float(diff @ weight / weight.sum()))
    extreme = int(np.sum(np.abs(stats) >= observed * (1 - 1e-9)))
    return extreme / len(stats) if exact else (extreme + 1) / (len(stats) + 1)


def paired(rows, a, b, metric, seed=0, draws=10000):
    """The token-weighted mean of the per-document difference a - b of a metric, its
    document-bootstrap 95% interval (both conditions resampled together), the exact sign-flip test's
    two-sided p (sign_flip_p) and on how many documents a is below b."""
    diff = np.asarray([row[a][metric] - row[b][metric] for row in rows], float)
    weight = np.asarray([row["n_scored"] for row in rows], float)
    index = np.random.default_rng(seed).integers(0, len(diff), (draws, len(diff)))
    stats = (diff[index] * weight[index]).sum(axis=1) / weight[index].sum(axis=1)
    return {"diff": float(diff @ weight / weight.sum()),
            "ci95": [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))],
            "p": sign_flip_p(diff, weight), "lower": int((diff < 0).sum()), "documents": len(rows)}


PAIRS = (("hub", "floor"), ("hub_prefix", "floor_prefix"), ("hub_prefix", "hub"), ("hub_table_prefix", "hub_prefix"))


def paired_table(rows):
    """{"a-b": {"kl": paired, "nll_delta": paired}} for the condition pairs a report has."""
    names = set(condition_names(rows))
    return {f"{a}-{b}": {metric: paired(rows, a, b, metric) for metric in ("kl", "nll_delta")}
            for a, b in PAIRS if a in names and b in names}


def recovered_share(rows, hub, floor, metric="kl", seed=0, draws=10000):
    """The share of the floor's token-weighted metric the hub condition removes, 1 - sum(w hub) /
    sum(w floor), its document-bootstrap 95% interval, and the document that carries the largest
    part of the floor's mass: its id, that part, and the share and both means without it."""
    weight = np.asarray([row["n_scored"] for row in rows], float)
    h = np.asarray([row[hub][metric] for row in rows], float) * weight
    f = np.asarray([row[floor][metric] for row in rows], float) * weight
    if f.sum() <= 0:
        return {"share": None, "ci95": None, "largest": None}
    index = np.random.default_rng(seed).integers(0, len(rows), (draws, len(rows)))
    denominators = f[index].sum(axis=1)
    stats = 1 - h[index].sum(axis=1) / np.where(denominators > 0, denominators, np.nan)
    out = {"share": float(1 - h.sum() / f.sum()),
           "ci95": [float(np.nanquantile(stats, 0.025)), float(np.nanquantile(stats, 0.975))], "largest": None}
    if len(rows) > 1:
        top = int(np.argmax(f))
        keep = np.arange(len(rows)) != top
        out["largest"] = {"document_id": rows[top]["document_id"], "floor_share": float(f[top] / f.sum()),
                          "share_without": float(1 - h[keep].sum() / f[keep].sum()) if f[keep].sum() > 0 else None,
                          "hub_without": float(h[keep].sum() / weight[keep].sum()),
                          "floor_without": float(f[keep].sum() / weight[keep].sum())}
    return out


def verdict(interval):
    """How a paired difference a - b reads: "lower", "higher" or "no measurable difference"."""
    if interval[1] < 0:
        return "lower"
    if interval[0] > 0:
        return "higher"
    return "no measurable difference"


def headline(rows, prefix):
    """The condition a switch into this target uses (hub or hub_prefix) against its floor: the
    paired KL and ΔNLL differences with their CIs and a verdict on the KL one, the share of the
    floor's KL removed (CI, and how much of the floor one chat carries), and the share of its ΔNLL
    only when the floor's ΔNLL interval lies above 0 (on the reference answer ΔNLL can be negative,
    so KL is the fidelity measure)."""
    hub_name, floor_name = ("hub_prefix", "floor_prefix") if prefix else ("hub", "floor")
    kl, nll = paired(rows, hub_name, floor_name, "kl"), paired(rows, hub_name, floor_name, "nll_delta")
    counts = [row["n_scored"] for row in rows]
    floor_nll, floor_nll_ci = boot_ratio([row[floor_name]["nll_delta"] * row["n_scored"] for row in rows], counts)
    hub_nll = boot_ratio([row[hub_name]["nll_delta"] * row["n_scored"] for row in rows], counts)[0]
    share = recovered_share(rows, hub_name, floor_name)
    return {"condition": hub_name, "floor": floor_name, "paired_kl": kl, "paired_nll": nll, "verdict": verdict(kl["ci95"]),
            "recovered_kl": share["share"], "recovered_kl_ci95": share["ci95"], "largest": share["largest"],
            "recovered_nll": (1 - hub_nll / floor_nll) if floor_nll_ci[0] > 0 else None}


def switch_setting(rows, qwen_target):
    """(native prefix?, rule): into the Qwen family always (RESULTS.md section 2); otherwise the
    condition with the lower KL when the paired 95% CI of hub_prefix - hub excludes 0, and
    NATIVE_PREFIX_DEFAULT when it spans 0."""
    if qwen_target:
        return True, "RESULTS.md section 2 (a Qwen-family target)"
    pair = paired(rows, "hub_prefix", "hub", "kl")
    if pair["ci95"][1] < 0:
        return True, "measured: hub_prefix has the lower KL (paired 95% CI below 0)"
    if pair["ci95"][0] > 0:
        return False, "measured: hub has the lower KL (paired 95% CI above 0)"
    return NATIVE_PREFIX_DEFAULT, (f"default: the paired 95% CI of hub_prefix - hub KL spans 0, so the native prefix "
                                   f"stays {'on' if NATIVE_PREFIX_DEFAULT else 'off'}")


def pool_rows(reports):
    """The documents of several pair_eval reports of one direction, each once (the first report's
    row wins), and the largest |difference| of any condition metric between two copies of a
    document (the same chat, hub and settings: 0 up to scoring noise; None without copies)."""
    rows, seen, largest = [], {}, None
    for report in reports:
        for row in report["documents"]:
            first = seen.get(row["document_id"])
            if first is None:
                seen[row["document_id"]] = row
                rows.append(row)
                continue
            for name in condition_names([row]):
                for metric in ("nll_delta", "kl", "top1"):
                    gap = abs(row[name][metric] - first[name][metric])
                    largest = gap if largest is None else max(largest, gap)
    return rows, largest


def baseline_comparison(rows, baseline_rows, condition="hub_prefix", floor="floor_prefix"):
    """NAME -> target against the baseline source -> the same target on their common documents:
    the paired difference of `condition` (NAME - baseline) in KL and ΔNLL with CIs, the largest gap
    between their floors (identical settings give identical floors: then the source is the only
    difference), the recovered KL share of each and the per-chat figures."""
    by_id = {row["document_id"]: row for row in baseline_rows}
    common = [row for row in rows if row["document_id"] in by_id]
    if not common:
        return {"documents": 0}
    base = [by_id[row["document_id"]] for row in common]
    pairs = [{"n_scored": row["n_scored"], "model": row[condition], "baseline": other[condition]}
             for row, other in zip(common, base)]
    gaps = [abs(row[floor][metric] - other[floor][metric]) for row, other in zip(common, base)
            for metric in ("kl", "nll_delta")]

    def fraction(row):
        total = row["mapped_cells"] + row["fill_cells"]
        return row["mapped_cells"] / total if total else None
    per_chat = [{"document_id": row["document_id"], "P": row["P"], "t0": row["t0"],
                 "mapped_fraction": fraction(row), "baseline_mapped_fraction": fraction(other),
                 "floor_kl": row[floor]["kl"], "kl": row[condition]["kl"], "baseline_kl": other[condition]["kl"],
                 "recovered_kl": 1 - row[condition]["kl"] / row[floor]["kl"] if row[floor]["kl"] > 0 else None,
                 "baseline_recovered_kl": (1 - other[condition]["kl"] / other[floor]["kl"]
                                           if other[floor]["kl"] > 0 else None)}
                for row, other in zip(common, base)]
    return {"documents": len(common), "condition": condition, "floor": floor, "floor_max_abs_diff": max(gaps),
            "kl": paired(pairs, "model", "baseline", "kl"), "nll_delta": paired(pairs, "model", "baseline", "nll_delta"),
            "recovered_kl": recovered_share(common, condition, floor)["share"],
            "baseline_recovered_kl": recovered_share(base, condition, floor)["share"], "per_chat": per_chat}


def rendering_comparison(template_rows, server_rows, condition="hub_prefix", floor="floor_prefix"):
    """The chats evaluated under both renderings (templates' defaults, as llama-server renders them):
    per chat the target's P, the share of translated cells mapped from the source, the floor's KL
    and the share of it `condition` removes, under each rendering."""
    by_id = {row["document_id"]: row for row in template_rows}

    def fraction(row):
        total = row["mapped_cells"] + row["fill_cells"]
        return row["mapped_cells"] / total if total else None

    def removed(row):
        return 1 - row[condition]["kl"] / row[floor]["kl"] if row[floor]["kl"] > 0 else None
    out = []
    for row in server_rows:
        other = by_id.get(row["document_id"])
        if other is None:
            continue
        out.append({"document_id": row["document_id"], "P": [other["P"], row["P"]],
                    "mapped_fraction": [fraction(other), fraction(row)],
                    "floor_kl": [other[floor]["kl"], row[floor]["kl"]], "kl": [other[condition]["kl"], row[condition]["kl"]],
                    "recovered_kl": [removed(other), removed(row)]})
    return out


def evaluation_summary(report):
    """One pair_eval run as the campaign reports it."""
    rows = report["documents"]
    mapped, fill = sum(row["mapped_cells"] for row in rows), sum(row["fill_cells"] for row in rows)
    return {"source": report["source"], "target": report["target"], "documents": len(rows),
            "rendering": report.get("rendering") or "template",
            "selection": report.get("selection"), "left_out": sorted(report.get("left_out") or {}),
            "recurrent": report.get("recurrent"), "recurrent_mean": report.get("recurrent_mean"),
            "summary": report["summary"], "conditions": condition_table(report), "paired": paired_table(rows),
            "document_ids": [row["document_id"] for row in rows],
            "P": [row["P"] for row in rows], "t0": [row["t0"] for row in rows],
            "mapped_fraction": mapped / (mapped + fill) if mapped + fill else None}


# -- state, logs, locks, processes --------------------------------------------------------------

def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, data):
    """Write JSON atomically (a temporary file in the same directory, then rename)."""
    path = Path(path)
    pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pending.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(pending, path)


def mem_available(meminfo="/proc/meminfo"):
    with open(meminfo, encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


@contextlib.contextmanager
def gpu_lock(memory_bytes, log, *, path=GPU_LOCK, margin=GPU_MARGIN, poll=GPU_POLL_S, available=mem_available,
             sleep=time.sleep):
    """Hold the campaign's GPU/memory lock: an exclusive flock on `path`, and MemAvailable at least
    memory_bytes + margin once it is held (otherwise release, wait `poll` s and try again). Every
    wait is logged. Yields a namespace: `stream` (the locked file; a command given its descriptor,
    pass_fds, holds the lock too), `free` (the MemAvailable found) and `waited` (seconds). The lock
    belongs to the open file: leaving the block closes this process's descriptor, so the lock is
    free once no command that inherited it still runs. Never taken while this process holds the CPU
    lock (resource_guard.LockOrderError): the GPU/memory lock comes first."""
    resource_guard.check_lock_order()
    started = time.monotonic()
    while True:
        stream = open(path, "a")
        try:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                log({"event": "gpu-lock-wait", "lock": str(path)})
                fcntl.flock(stream, fcntl.LOCK_EX)
            free = available()
            need = memory_bytes + margin
            if free >= need:
                waited = time.monotonic() - started
                log({"event": "gpu-lock", "available_gib": round(free / GIB, 1), "need_gib": round(need / GIB, 1),
                     "waited_s": round(waited)})
                yield types.SimpleNamespace(stream=stream, free=free, waited=waited)
                return
            log({"event": "memory-wait", "available_gib": round(free / GIB, 1), "need_gib": round(need / GIB, 1),
                 "waited_s": round(time.monotonic() - started)})
        finally:
            stream.close()
        sleep(poll)


def _reap(child, block):
    """wait4 on a command; its rusage once it has exited (and child.returncode set), else None."""
    pid, status, usage = os.wait4(child.pid, 0 if block else os.WNOHANG)
    if not pid:
        return None
    child.returncode = os.waitstatus_to_exitcode(status)
    return usage


def stop_group(child, grace=STOP_GRACE_S):
    """End a command and its process group: SIGTERM (and SIGCONT: a group the resource guard
    stopped handles it only once continued), then SIGKILL to whatever is left after `grace` seconds;
    the command is reaped."""
    def signal_group(number):
        try:
            os.killpg(child.pid, number)
        except ProcessLookupError:
            pass
    signal_group(signal.SIGTERM)
    signal_group(signal.SIGCONT)
    deadline = time.monotonic() + grace
    while child.returncode is None and time.monotonic() < deadline:
        if _reap(child, block=False) is None:
            time.sleep(0.1)
    signal_group(signal.SIGKILL)
    if child.returncode is None:
        _reap(child, block=True)


def group_anon_bytes(pgid, proc="/proc"):
    """Resident anonymous memory (RssAnon: not file pages the kernel can drop) of every process of a
    process group, from /proc."""
    total = 0
    for entry in os.scandir(proc):
        if not entry.name.isdigit():
            continue
        try:
            with open(os.path.join(entry.path, "stat"), "rb") as stream:
                stat = stream.read()
            if int(stat[stat.rindex(b")") + 2:].split()[2]) != pgid:
                continue
            with open(os.path.join(entry.path, "status"), "rb") as stream:
                for line in stream:
                    if line.startswith(b"RssAnon:"):
                        total += int(line.split()[1]) * 1024
                        break
        except (OSError, ValueError, IndexError):
            continue
    return total


def run_child(argv, *, cwd=None, env=None, output=None, keep_fds=(), timeout=None, poll=CHILD_POLL_S,
              sample=MEMORY_SAMPLE_S, watch=None, gpu_memory=None):
    """Run a command in its own session (process group) with the descriptors keep_fds inherited, and
    wait for it; on a timeout, or when this process is interrupted (KeyboardInterrupt, Terminated),
    its whole group is stopped. watch: a resource_guard.Watch attached to the group and stepped while
    it runs (it stops and continues the group); its pauses count neither toward the timeout nor the
    seconds, and when it ends the command (GuardAbort) the group is stopped and `aborted` says why.
    Once the command has exited, whatever it left running in its group (a server, a daemon) is stopped
    too, before the caller releases its locks.
    Returns a namespace: returncode, seconds, max_rss (bytes, wait4's ru_maxrss of the command or the
    largest descendant it waited for: file pages it touched included), max_anon (the group's peak
    RssAnon, sampled every `sample` seconds: what it needs resident), max_gpu (the peak rise of the GPU's
    GTT use since the command started, resource_guard.read_gpu_memory, or None where it cannot be read: on
    this APU a GPU job's memory, which RssAnon does not show), max_total (the largest anonymous + GPU memory
    of one sample: what the command takes off MemAvailable), timed_out, paused_s, aborted and guard (the
    Watch's summary)."""
    gpu_memory = gpu_memory or (lambda: resource_guard.read_gpu_memory()["gtt_used"])
    gtt_base = gpu_memory()
    started = time.perf_counter()
    child = subprocess.Popen([str(item) for item in argv], cwd=cwd, env=env, stdout=output,
                             stderr=subprocess.STDOUT if output is not None else None, pass_fds=tuple(keep_fds),
                             start_new_session=True)
    usage, timed_out, max_anon, sampled, aborted = None, False, 0, 0.0, None
    max_gpu = max_total = 0
    try:
        if watch is not None:
            watch.attach(child.pid)
        while True:
            usage = _reap(child, block=False)
            if usage is not None:
                break
            if watch is not None:
                watch.step()
            elapsed = time.perf_counter() - started
            if timeout is not None and elapsed - (watch.paused_seconds() if watch is not None else 0.0) > timeout:
                stop_group(child)
                timed_out = True
                break
            if elapsed - sampled >= sample:
                anon, used = group_anon_bytes(child.pid), gpu_memory() if gtt_base is not None else None
                gpu = 0 if used is None else max(0, used - gtt_base)
                max_anon, max_gpu, max_total, sampled = (max(max_anon, anon), max(max_gpu, gpu),
                                                         max(max_total, anon + gpu), elapsed)
            time.sleep(poll)
    except resource_guard.GuardAbort as error:
        stop_group(child)
        aborted = str(error)
    except BaseException:
        stop_group(child)
        if watch is not None:
            watch.close()
        raise
    seconds = time.perf_counter() - started
    if not timed_out and aborted is None:
        stop_leftovers(child.pid)
    guard = watch.close() if watch is not None else None
    paused = guard["paused_s"] if guard else 0.0
    return types.SimpleNamespace(returncode=child.returncode, seconds=seconds - paused,
                                 max_rss=None if usage is None else usage.ru_maxrss * 1024, max_anon=max_anon,
                                 max_gpu=None if gtt_base is None else max_gpu,
                                 max_total=max_anon if gtt_base is None else max_total,
                                 timed_out=timed_out, paused_s=paused, aborted=aborted, guard=guard)


def group_members(pgid, proc="/proc"):
    """The pids of a process group's live members (zombies left out), from /proc."""
    out = []
    for entry in os.scandir(proc):
        if not entry.name.isdigit():
            continue
        try:
            with open(os.path.join(entry.path, "stat"), "rb") as stream:
                stat = stream.read()
            fields = stat[stat.rindex(b")") + 2:].split()
            if int(fields[2]) == pgid and fields[0] != b"Z":
                out.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return out


def stop_leftovers(pgid, grace=STOP_GRACE_S, proc="/proc"):
    """After a command has exited: end what it left running in its process group (a llama-server or an engine
    daemon live_bridge.py kept there under a Watch, which would hold GPU memory after the locks are released):
    SIGTERM and SIGCONT, SIGKILL after `grace` seconds. Returns the pids it found."""
    left = group_members(pgid, proc)
    if not left:
        return []
    for number in (signal.SIGTERM, signal.SIGCONT):
        try:
            os.killpg(pgid, number)
        except ProcessLookupError:
            return left
    deadline = time.monotonic() + grace
    while group_members(pgid, proc) and time.monotonic() < deadline:
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return left


@contextlib.contextmanager
def guarded_locks(memory_bytes, log, *, cpu=False, label=None):
    """The locks of one command, in order: the GPU/memory lock (when memory_bytes is not None), then
    the CPU lock (cpu). Yields (gpu lock namespace or None, the descriptors the command inherits, its
    environment additions: KVPARK_CPU_LOCK_FD)."""
    with contextlib.ExitStack() as stack:
        held = stack.enter_context(gpu_lock(memory_bytes, log, path=GPU_LOCK, poll=GPU_POLL_S,
                                            available=mem_available)) if memory_bytes is not None else None
        fds, extra = [] if held is None else [held.stream.fileno()], {}
        if cpu:
            locked = stack.enter_context(resource_guard.cpu_lock(CPU_LOCK, log=log, label=label))
            if locked.fd is not None:
                fds.append(locked.fd)
                extra[resource_guard.CPU_LOCK_FD] = str(locked.fd)
        yield held, fds, extra


def guarded_child(argv, *, settings, guard_log, label, events, environment, **options):
    """run_child under the resource guard (the locks already held): wait_cool, then a Watch logging
    to guard_log; the child's environment says it is watched. Returns run_child's namespace with
    cool_wait_s."""
    cooled = resource_guard.wait_cool(settings, log=resource_guard.JsonLog(guard_log, label, events))
    watch = resource_guard.Watch(settings, log_path=guard_log, label=label, events=events)
    outcome = run_child(argv, env={**environment, resource_guard.WATCHED: "1"}, watch=watch, **options)
    outcome.cool_wait_s = cooled
    return outcome


def disk_usage(paths):
    """{"bytes": allocated bytes of the files only these paths hold, "linked_bytes": of files with
    other links too (a hub's base files), "files"}."""
    own = linked = files = 0
    seen = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        items = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file() and not p.is_symlink()]
        for item in items:
            status = item.stat()
            if (status.st_dev, status.st_ino) in seen:
                continue
            seen.add((status.st_dev, status.st_ino))
            files += 1
            if status.st_nlink > 1:
                linked += status.st_blocks * 512
            else:
                own += status.st_blocks * 512
    return {"bytes": own, "linked_bytes": linked, "files": files}


def file_bytes(paths):
    """Apparent size of the files under these paths."""
    total = 0
    for path in paths:
        path = Path(path)
        items = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        total += sum(item.stat().st_size for item in items)
    return total


def sha256_file(path, block=16 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(block):
            digest.update(chunk)
    return digest.hexdigest()


_HASHES = {}
_GIT = {}


def cached_sha256(path):
    """sha256 of a file, cached in this process by (path, size, mtime)."""
    status = Path(path).stat()
    key = (str(path), status.st_size, status.st_mtime_ns)
    if key not in _HASHES:
        _HASHES[key] = sha256_file(path)
    return _HASHES[key]


def git_state():
    """{"head": git HEAD, "changed": paths changed under experiments/ and src/ (git status)}; once per process."""
    if not _GIT:
        def git(*arguments):
            try:
                result = subprocess.run(["git", "-C", str(ROOT), *arguments], capture_output=True, text=True,
                                        timeout=60)
            except (OSError, subprocess.SubprocessError):
                return None
            return result.stdout if result.returncode == 0 else None
        head = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--", "experiments", "src")
        _GIT.update(head=head.strip() if head else None,
                    changed=None if status is None else [line[3:] for line in status.splitlines() if line])
    return dict(_GIT)


def script_hashes(argv):
    """{path: sha256} of the Python scripts a command runs and of the probe it names (--probe)."""
    out = {}
    paths = [item for item in argv[1:] if item.endswith(".py")]
    if "--probe" in argv[:-1]:
        paths.append(argv[argv.index("--probe") + 1])
    for item in paths:
        if Path(item).is_file():
            out[item] = cached_sha256(item)
    return out


class Campaign:
    """One model's campaign directory: its state file, event log, command log and model facts."""

    def __init__(self, name, *, directory=None, gguf=None, catalog=CATALOG, catalog_id=None):
        """catalog_id: the model's id in the catalog when it is not its hub name (default: name)."""
        self.name = name
        self.dir = Path(directory or CAMPAIGN_DIR / name)
        self.state_path = self.dir / "state.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if gguf is not None and Path(gguf).resolve() != Path(self.state["gguf"]):
                raise ValueError(f"{name} was started with {self.state['gguf']}, not {gguf}")
            started_id = (self.state.get("catalog") or {}).get("id")
            if catalog_id is not None and catalog_id != started_id:
                raise ValueError(f"{name} was started with catalog entry {started_id}, not {catalog_id}")
        else:
            entry = catalog_entry(catalog_id or name, catalog)
            if catalog_id is not None and entry is None:
                raise ValueError(f"{catalog_id} is not in the catalog")
            refused = refusal((entry or {}).get("arch"))
            if refused:
                raise ValueError(f"{name} is refused: {refused}")
            gguf = Path(gguf) if gguf else model_file(entry, catalog_id or name)
            self.state = {"format_version": STATE_FORMAT, "name": name, "gguf": str(Path(gguf).resolve()),
                          "catalog": entry, "created": now(), "stages": {}}
        self.gguf = Path(self.state["gguf"])

    # state
    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        write_json(self.state_path, self.state)

    def stage(self, stage):
        return self.state["stages"].setdefault(stage, {})

    def statuses(self):
        return {stage: self.state["stages"].get(stage, {}).get("status") for stage in STAGES}

    def log(self, item):
        item = {"time": now(), **item}
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "log.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(json.dumps(item, ensure_ascii=False), flush=True)

    @contextlib.contextmanager
    def exclusive(self):
        """Only one run per model: an exclusive, non-blocking flock in its directory."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / ".lock", "a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(f"another campaign run of {self.name} is active") from None
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    # commands
    def command_line(self, argv, *, cwd, env, memory_bytes, timeout, cpu=False, guard_log=None, gpu_evaluation=False):
        """The command as commands.sh holds it: a model-loading command through `campaign.py exec`,
        which takes the GPU/memory lock (and the CPU lock: cpu), checks memory and runs it under the
        resource guard as the run does (with the GPU-evaluation defaults: gpu_evaluation)."""
        if memory_bytes is not None:
            argv = [str(PYTHON), str(HERE / "campaign.py"), "exec", "--memory-bytes", str(int(memory_bytes))] \
                + (["--timeout", str(int(timeout))] if timeout is not None else []) \
                + (["--cpu-lock"] if cpu else []) + (["--gpu-evaluation"] if gpu_evaluation else []) \
                + (["--guard-log", str(guard_log)] if guard_log else []) + ["--", *argv]
        extra = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted((env or {}).items()))
        return f"(cd {shlex.quote(str(cwd))} && {extra + ' ' if extra else ''}{shlex.join(argv)})"

    def run(self, argv, *, stage, label, memory_bytes=None, env=None, cwd=HERE, timeout=None, cpu=False,
            gpu_evaluation=False):
        """Run one command with its output in logs/<label>.log; under the GPU/memory lock when it
        loads a model or a hub (memory_bytes: what it loads), and the CPU lock after it when it is
        CPU-heavy (cpu), both of which the command itself holds while it runs; under the resource
        guard (BLAS threads capped, wait_cool, a Watch logging to logs/<label>.guard.jsonl; for a GPU
        evaluation, gpu_evaluation, with resource_guard.GPU_EVALUATION's defaults). Recorded
        in commands.sh and the stage's state entry (with git HEAD and the sha256 of the scripts it
        runs), so it can be run again by hand. Returns the command's own seconds (lock, memory and
        cooling waits and guard pauses excluded); a timeout stops its process group and raises
        CommandTimeout, the guard ending it GuardStop."""
        argv = [str(item) for item in argv]
        settings = resource_guard.Settings.from_env(
            defaults=resource_guard.GPU_EVALUATION if gpu_evaluation else None)
        logs = self.dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        guard_log = logs / f"{label}.guard.jsonl"
        shell = self.command_line(argv, cwd=cwd, env=env, memory_bytes=memory_bytes, timeout=timeout, cpu=cpu,
                                  guard_log=guard_log, gpu_evaluation=gpu_evaluation)
        with open(self.dir / "commands.sh", "a", encoding="utf-8") as stream:
            stream.write(f"# {now()} {self.name} {stage}: {label}\n{shell}\n")
        entry = self.stage(stage).setdefault("commands", [])
        entry.append({"label": label, "argv": argv, "cwd": str(cwd), "env": env or {}, "gpu_lock": memory_bytes is not None,
                      "cpu_lock": bool(cpu), "memory_bytes": memory_bytes, "timeout": timeout,
                      "log": str(logs / f"{label}.log"), "guard_log": str(guard_log), "git": git_state(),
                      "scripts": script_hashes(argv)})
        self.save()
        environment = resource_guard.thread_env({**os.environ, **(env or {})}, settings.threads)
        entry[-1]["threads"] = environment["OPENBLAS_NUM_THREADS"]

        def events(item):
            self.log({**item, "stage": stage, "label": label})
        with guarded_locks(memory_bytes, events, cpu=cpu, label=f"campaign {self.name} {label}") as (held, fds, extra):
            self.log({"event": "command", "stage": stage, "label": label, "command": shell})
            with open(logs / f"{label}.log", "a", encoding="utf-8") as output:
                outcome = guarded_child(argv, settings=settings, guard_log=guard_log, label=label, events=events,
                                        environment={**environment, **extra}, cwd=cwd, output=output, keep_fds=fds,
                                        timeout=timeout)
        entry[-1].update(seconds=round(outcome.seconds, 1), waited_s=0.0 if held is None else round(held.waited, 1),
                         cool_wait_s=round(outcome.cool_wait_s, 1), paused_s=outcome.paused_s,
                         returncode=outcome.returncode, timed_out=outcome.timed_out, guard_abort=outcome.aborted,
                         max_c=(outcome.guard or {}).get("max_c"),
                         max_rss_gib=None if outcome.max_rss is None else round(outcome.max_rss / GIB, 2),
                         max_anon_gib=round(outcome.max_anon / GIB, 2),
                         max_gpu_gib=None if outcome.max_gpu is None else round(outcome.max_gpu / GIB, 2),
                         max_total_gib=round(outcome.max_total / GIB, 2))
        self.save()
        self.log({"event": "command-done", "stage": stage, "label": label, "seconds": round(outcome.seconds, 1),
                  "returncode": outcome.returncode, "timed_out": outcome.timed_out, "max_rss_gib": entry[-1]["max_rss_gib"],
                  "max_anon_gib": entry[-1]["max_anon_gib"], "max_gpu_gib": entry[-1]["max_gpu_gib"],
                  "max_total_gib": entry[-1]["max_total_gib"], "paused_s": outcome.paused_s, "max_c": entry[-1]["max_c"],
                  "guard_abort": outcome.aborted})
        if outcome.aborted:
            raise GuardStop(f"{label} was ended by the resource guard: {outcome.aborted} (samples in {guard_log})")
        if outcome.timed_out:
            raise CommandTimeout(f"{label} ran past its {timeout:.0f} s timeout and was stopped")
        if outcome.returncode != 0:
            tail = (logs / f"{label}.log").read_text(encoding="utf-8", errors="replace")[-3000:]
            raise StageError(f"{label} exited with {outcome.returncode}; the end of its log:\n{tail}")
        return outcome.seconds

    # model facts
    def spec(self, probe=None):
        import kvstate
        return kvstate.ModelSpec.from_gguf(self.gguf, llama_revision=kvstate.llama_revision_of(Path(probe or PROBE_VULKAN)))

    def facts(self):
        """The model's architecture, whether it is a hybrid, its recurrent cell and whether that needs
        a recurrent mean (computed once, kept in state.json); ValueError for a refused architecture."""
        if "recurrent_mean" not in self.state:
            spec = self.spec()
            cell = recurrent_cell(spec)
            self.state.update(architecture=spec.architecture, hybrid=bool(spec.recurrent_layers), recurrent_cell=cell,
                              recurrent_mean=cell["mean"])
        refused = refusal(self.state.get("architecture"))
        if refused:
            raise ValueError(f"{self.name} is refused: {refused}")
        return self.state

    @property
    def needs_mean(self):
        return bool(self.facts()["recurrent_mean"])

    def captures(self):
        """The capture directories the capture stage published (its chunks)."""
        result = self.state["stages"].get("capture", {}).get("result") or {}
        return [Path(path) for path in result.get("captures") or [self.path("capture")]]

    def path(self, *parts):
        return self.dir.joinpath(*parts)


def catalog_entry(name, catalog=CATALOG):
    """The catalog entry with this id, or None."""
    if catalog is None or not Path(catalog).is_file():
        return None
    for entry in json.loads(Path(catalog).read_text(encoding="utf-8"))["models"]:
        if entry["id"] == name:
            return entry
    return None


def dflash_drafts(entry, read=None):
    """[{"path", "window"}] of a catalog entry's DFlash drafts (companions of kind dflash) with the
    sliding window each attends over target features (dflash.attention.sliding_window; None when
    it cannot be read). A switch into the model while its server runs the draft keeps a native
    tail of window - 1 tokens, so that the draft sees native context only: the bridge needs
    --draft-window when the window is not BRIDGE_DRAFT_WINDOW. read(path, key): gguf_tensors'
    metadata_value by default."""
    if read is None:
        from gguf_tensors import metadata_value as read
    out = []
    for companion in (entry or {}).get("companions") or []:
        if companion.get("kind") != "dflash":
            continue
        for path in companion.get("paths") or []:
            try:
                window = read(path, "dflash.attention.sliding_window")
            except (OSError, ValueError, RuntimeError):
                window = None
            out.append({"path": str(path), "window": None if window is None else int(window)})
    return out


def model_file(entry, name):
    """The single GGUF file of a catalog entry (the probe loads one file)."""
    if entry is None:
        raise ValueError(f"{name} is not in the catalog: give --gguf (and --capture-tps)")
    parts = (entry.get("variants") or [{}])[0].get("parts") or []
    if len(parts) != 1:
        raise ValueError(f"{name}: the probe loads a single GGUF file, the catalog lists {len(parts)} parts")
    return Path(parts[0])


# -- stages -------------------------------------------------------------------------------------

def base_models(base=BASE_HUB):
    """[(name, [capture directories], records)] of the base hub's models (its settings)."""
    manifest = json.loads((Path(base) / "manifest.json").read_text(encoding="utf-8"))
    return [(name, [Path(p) for p in item["captures"]], Path(item["records"]))
            for name, item in manifest["settings"]["models"].items()]


def hub_bytes(directory):
    """Bytes of a hub bundle's files (what loading it whole reads into memory)."""
    return file_bytes([p for p in Path(directory).iterdir() if p.name != "manifest.json"])


def sides_bytes(directory, source, target):
    """Bytes of what a translation source -> target reads from a format 2 bundle (pair_eval and the engine
    read it lazily): the source's encoder sides, the target's decoder sides and indexer ridge (float32),
    and the target's head kinds; a format 1 bundle: both models' files."""
    import hub
    manifest = hub.read_manifest(directory)
    names = []
    for (table, key), entry in hub._bundle_entries(manifest).items():
        owner = hub._owner(key, manifest["models"])
        role = key[len(owner) + 1:].split(".") if owner else []
        if table == "sides" and ((owner == source and role[-1] == "encoder")
                                 or (owner == target and role[-1] in ("decoder", "ridge"))):
            names.append(entry["files"]["f32"]["file"])
        elif table == "files" and (owner == target or (owner == source and manifest["format_version"] == hub.FORMAT)):
            if manifest["format_version"] == hub.FORMAT or role[-1] not in hub.KINDS:
                names.append(entry["file"])
    return file_bytes([Path(directory) / name for name in names])


def model_argument(name, captures, records):
    return f"{name}={','.join(str(c) for c in captures)}@{records}"


def stage_render(campaign, args):
    import chat
    import collect_v2
    import template_check
    out = campaign.path("render")
    out.mkdir()
    info = chat.gguf_tokenizer_info(campaign.gguf)
    if info["chat_template"] is None:
        raise StageError(f"{campaign.gguf.name} has no chat template")
    template, bos = info["chat_template"], info["bos_token"]
    context = {"eos_token": info["eos_token"]}          # llama.cpp gives every template its EOS text
    conversations, records, seen = [], [], set()
    for path in HUB_CORPUS:
        for conversation in chat.read_conversations(path):
            record = chat.rendered_record(conversation, campaign.name,
                                          chat.render(conversation, template, bos_token=bos, extra_context=context))
            if record["document_id"] in seen:
                raise StageError(f"conversation {conversation['id']} appears twice in the hub corpus")
            seen.add(record["document_id"])
            conversations.append(conversation)
            records.append(record)
    (out / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                                       encoding="utf-8")
    (out / "template.jinja").write_text(template, encoding="utf-8")
    sample = parity_sample(conversations)
    started = time.perf_counter()
    prompts = template_check.check_server(campaign.gguf, server_binary=LLAMA_SERVER, workdir=out, ports=TWIN_PORTS,
                                          conversations=[prompt_shape(c) for c in sample])
    whole = template_check.check_server(campaign.gguf, server_binary=LLAMA_SERVER, workdir=out, ports=TWIN_PORTS,
                                        conversations=sample, option_sets=[{}], add_generation_prompt=False)
    write_json(out / "parity-prompts.json", prompts)
    write_json(out / "parity-records.json", whole)
    parity_seconds = time.perf_counter() - started
    tolerate = tuple(getattr(args, "parity_tolerate", None) or ())
    problems = parity_problems(prompts, whole, tolerate)
    # whether each sampled record holds the very bytes the server renders for its conversation, and
    # where and why not (a server context such as preserve_reasoning: recorded, not fatal)
    differing = record_differences(records, sample, whole, template)
    write_json(out / "records-differing.json", differing)
    campaign.log({"event": "parity", "prompts": prompts["counts"], "records": whole["counts"], "problems": problems,
                  "records_identical": not differing,
                  "records_differing": [{k: item.get(k) for k in ("conversation", "context", "context_differs", "at")}
                                        for item in differing]})
    if problems:
        raise StageError("chat.py and llama-server's /apply-template disagree: " + "; ".join(problems)
                         + " (render/parity-*.json)")
    started = time.perf_counter()
    tokens = collect_v2.tokenize(PROBE_VULKAN, campaign.gguf, records, ctx=PROBE_MAX_CTX, workdir=out)
    documents = [{"document_id": r["document_id"], "split": r["split"], "source": r["source"],
                  "tokens": int(len(tokens[r["document_id"]]["token_id"]))} for r in records]
    write_json(out / "tokens.json", {"probe": str(PROBE_VULKAN), "documents": documents})
    by_split = {}
    for document in documents:
        item = by_split.setdefault(document["split"], {"documents": 0, "tokens": 0})
        item["documents"] += 1
        item["tokens"] += document["tokens"]
    return {"documents": len(records), "tokens": sum(d["tokens"] for d in documents),
            "max_tokens": max(d["tokens"] for d in documents), "by_split": by_split,
            "template_sha256": chat.template_sha256(template), "bos_token": bos, "add_bos": info["add_bos"],
            "parity": {"prompts": prompts["counts"], "records": whole["counts"], "tolerated": list(tolerate),
                       "records_identical": not differing,
                       "records_differing": [{k: item.get(k) for k in ("conversation", "context", "context_differs", "at",
                                                                       "reproduced")} for item in differing],
                       "contexts": sorted({c.get("context") for c in prompts["cases"] + whole["cases"]} - {None}),
                       "server": str(LLAMA_SERVER), "build_info": prompts.get("build_info"),
                       "seconds": round(parity_seconds, 1)},
            "tokenize_seconds": round(time.perf_counter() - started, 1)}


def served_text(text, report):
    """A rendering as llama-server's /apply-template returns it: without the template's BOS text when
    the vocabulary adds BOS, and without its EOS text when it adds EOS (template_check.compare_server)."""
    bos, eos = report.get("bos_token") or "", report.get("eos_token") or ""
    if report.get("add_bos") and bos and text.startswith(bos):
        text = text[len(bos):]
    if report.get("add_eos") and eos and text.endswith(eos):
        text = text[:-len(eos)]
    return text


def _plain(value):
    return json.loads(json.dumps(value, default=str))


def record_differences(records, sample, whole, template):
    """The sampled records whose bytes are not the server's whole-conversation rendering
    (template_check's prompt_sha256), each with why: the server context that reproduces the server's
    bytes (chat.server_contexts; "template" is what a record renders with), the variables in which
    it differs from the template's defaults, whether it reproduces them, and the first difference."""
    import chat
    by_conversation = {record["conversation_id"]: record for record in records}
    conversations = {conversation["id"]: conversation for conversation in sample}
    contexts = dict(chat.server_contexts({}, eos_token=whole.get("eos_token") or ""))
    # what a record renders with: the template's defaults and the EOS text (stage_render)
    defaults = {**{key: value for key, value in contexts["template"].items() if key != "strftime_now"},
                "eos_token": whole.get("eos_token") or ""}
    out = []
    for case in whole["cases"]:
        record = served_text(by_conversation[case["conversation"]]["text"], whole)
        if hashlib.sha256(record.encode("utf-8")).hexdigest() == case.get("prompt_sha256"):
            continue
        item = {"conversation": case["conversation"], "context": case.get("context"),
                "server_error": case.get("server_error")}
        context = contexts.get(case.get("context"))
        if context is not None:
            conversation = conversations[case["conversation"]]
            server = served_text(chat.render({"messages": conversation["messages"], "tools": conversation.get("tools")},
                                             template, bos_token=whole.get("bos_token") or "",
                                             add_generation_prompt=False, extra_context=context, live=True).text, whole)
            item["reproduced"] = hashlib.sha256(server.encode("utf-8")).hexdigest() == case.get("prompt_sha256")
            item["context_differs"] = {key: _plain(value) for key, value in context.items()
                                       if key != "strftime_now" and defaults.get(key, None) != value}
            item.update(text_difference(record, server) or {})
        out.append(item)
    return out


def fit_hub_extra(spec, layer_rank, *, dev_spans, pivots, hub_dim, gpu=True, chunk=FIT_HUB_CHUNK):
    """What fit_hub needs beyond the base hub to onboard a model of this spec (bytes of anonymous memory and, on
    the GPU, the device's): FIT_HUB_BASE, FIT_HUB_GPU when it fits on the GPU (gpu), and per kind (the larger counts,
    one kind is fitted at a time) its stage-1 covariance (W1 = the per-layer ranks min(layer_rank, width) summed over
    the attention layers), its decoder's statistics, its chunk rows, and per dev span and pivot the part that does
    not grow with the model (see FIT_HUB_BASE). dev_spans: the base hub's, an upper bound."""
    out = 0
    for kind in ("k", "v"):
        widths = [int(np.prod(spec.head_shape(layer, kind))) for layer in spec.attention_layers
                  if spec.layer_types[layer] != "nextn"]
        stage1 = sum(min(layer_rank, width) for width in widths)
        columns = sum(widths)
        out = max(out, FIT_HUB_COVARIANCE_BYTES * stage1 ** 2 + FIT_HUB_DECODER_BYTES * hub_dim * columns
                  + FIT_HUB_CHUNK_BYTES * chunk * columns + dev_spans * pivots * FIT_HUB_DEV_LATENT_BYTES * hub_dim)
    return FIT_HUB_BASE + (FIT_HUB_GPU if gpu else 0) + out


def capture_ctx(max_tokens):
    """The probe context for documents of up to max_tokens tokens (the hub4 captures used 5,120)."""
    return max(CAPTURE_CTX, -(-(max_tokens + 64) // CAPTURE_BATCH) * CAPTURE_BATCH)


def _normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def capture_identity(campaign, probe, mode, ctx, spec_file, device):
    """What a published capture chunk's manifest must say to be adopted (besides its documents):
    the mode, the model's sha256, the probe's runtime revision and build (collect.verify_runtime),
    collect_v2.py's sha256, the capture spec and the probe settings (device: capture_device)."""
    import collect
    import collect_v2
    revision, identity, _ = collect.verify_runtime(Path(probe).resolve())
    return {"mode": mode, "model_sha256": collect_v2.ModelHashCache(CAMPAIGN_DIR / "model-hashes.json").sha256(campaign.gguf),
            "runtime_revision": revision, "collector_build": _normalized(identity),
            "wrapper_sha256": sha256_file(HERE.parent / "linear_transfer" / "collect_v2.py"),
            "capture_spec": None if spec_file is None else collect_v2.read_capture_spec(spec_file),
            "settings": {"ctx": ctx, "batch": CAPTURE_BATCH, "threads": CAPTURE_THREADS,
                         "device": device}}


def adopt_chunk(directory, docs, expected):
    """(True, None) when `directory` is a published capture of exactly the documents in `docs` made
    with the `expected` identity (capture_identity), else (False, why)."""
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        return False, "no published capture"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("rendered_docs_sha256") != sha256_file(docs):
        return False, "other documents"
    for key, value in expected.items():
        have = manifest.get(key)
        if key == "settings":
            wrong = sorted(name for name in value if (have or {}).get(name) != value[name])
            if wrong:
                return False, f"other probe settings: {wrong}"
        elif _normalized(have) != _normalized(value):
            return False, f"another {key}"
    return True, None


def remove_chunks(out, start):
    """Remove capture chunks numbered `start` and above (their directories, docs and run files)."""
    for path in sorted(Path(out).glob("chunk-*")):
        number = path.name.split(".", 1)[0][len("chunk-"):]
        if number.isdigit() and int(number) >= start:
            shutil.rmtree(path) if path.is_dir() else path.unlink()


def stage_capture(campaign, args):
    spec = campaign.spec()
    mode, device = capture_mode(spec.architecture), capture_device(spec.architecture)
    probe = PROBE_VULKAN if device == "vulkan" else PROBE_CPU
    documents = json.loads(campaign.path("render", "tokens.json").read_text(encoding="utf-8"))["documents"]
    tokens_of = {d["document_id"]: d["tokens"] for d in documents}
    smoke = ((campaign.state.get("catalog") or {}).get("smoke") or {}).get("prefill_tps")
    tps = args.capture_tps or (smoke * CAPTURE_SPEED[mode, device] if smoke else None)
    if not tps:
        raise StageError(f"no prefill speed for {campaign.name} in the catalog: give --capture-tps")
    per_token, peak = bytes_per_token(spec), PEAK_FACTOR[mode]
    budget_bytes, budget_seconds = int(args.budget_gb * 10 ** 9), args.budget_minutes * 60
    chunk_tokens = chunk_tokens_for(min(budget_bytes // per_token, (budget_seconds - LOAD_SECONDS) * tps))
    try:
        plan = capture_plan(documents, bytes_per_token=per_token, tokens_per_second=tps, budget_bytes=budget_bytes,
                            budget_seconds=budget_seconds, reserve_tokens=(peak - 1) * chunk_tokens)
    except ValueError as error:
        raise StageError(f"{error}: {held_out_need(documents, bytes_per_token=per_token, tokens_per_second=tps)}") \
            from None
    hard_limit = HARD_FACTOR * budget_seconds
    plan.update(mode=mode, device=device, probe=str(probe), bytes_per_token=per_token, tokens_per_second=round(tps, 1),
                speed_source="--capture-tps" if args.capture_tps else f"catalog smoke {smoke} x {CAPTURE_SPEED[mode, device]}",
                attention_layers=[l for l in spec.attention_layers if spec.layer_types[l] != "nextn"],
                budget_gb=args.budget_gb, budget_minutes=args.budget_minutes, chunk_tokens=chunk_tokens,
                peak_factor=peak, hard_limit_seconds=hard_limit)
    write_json(campaign.path("capture-plan.json"), plan)
    campaign.log({"event": "capture-plan", **{k: plan[k] for k in ("tokens", "bytes", "seconds", "limit", "by_split",
                                                                    "chunk_tokens")}})
    records = {json.loads(line)["document_id"]: line
               for line in campaign.path("render", "records.jsonl").read_text(encoding="utf-8").splitlines()}
    spec_file = None
    if mode == "collect":
        spec_file = campaign.path("capture-spec.json")
        write_json(spec_file, {"k": "all", "v": "all", **({"ik": "all"} if spec.indexer is not None else {})})
    out = campaign.path("capture")
    out.mkdir(exist_ok=True)
    identities = {}

    def identity(ctx):
        if ctx not in identities:
            identities[ctx] = capture_identity(campaign, probe, mode, ctx, spec_file, device)
        return identities[ctx]
    pending, chunks, dropped = list(plan["order"]), [], []
    probe_seconds = waited = 0.0
    rows = captured_tokens = 0
    limit = "corpus"
    for index in itertools.count():
        name = f"chunk-{index:02d}"
        directory, docs, info_path = out / name, out / f"{name}.docs.jsonl", out / f"{name}.json"
        if index > 0 and not pending:
            break
        if docs.is_file() and info_path.is_file():
            # a chunk an interrupted run published: adopted when it holds the documents this run would
            # still capture, from the same probe build and settings
            ids = [json.loads(line)["document_id"] for line in docs.read_text(encoding="utf-8").splitlines()]
            fits = ids == plan["held_out"] if index == 0 else bool(ids) and set(ids) <= set(pending)
            ok, why = adopt_chunk(directory, docs, identity(capture_ctx(max(tokens_of[d] for d in ids)))) \
                if fits else (False, "documents this run does not capture next")
            if ok:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                pending = [d for d in pending if d not in set(ids)]
                probe_seconds += info["seconds"]
                waited += info.get("waited_s") or 0.0
                rows += info["rows"]
                captured_tokens += info["tokens"]
                chunks.append({**info, "name": name, "adopted": True})
                campaign.log({"event": "capture-chunk-adopted", "chunk": name, "rows": info["rows"]})
                continue
            campaign.log({"event": "capture-chunk-discarded", "chunk": name, "why": why})
        remove_chunks(out, index)
        if index == 0:
            ids = list(plan["held_out"])
        else:
            rate = rows / probe_seconds if probe_seconds > 0 else tps
            by_time = max(0.0, budget_seconds - probe_seconds) * rate
            by_bytes = max(0, budget_bytes - captured_tokens * per_token) // (per_token * peak)
            ids, gone = next_chunk(pending, tokens_of, chunk_tokens=chunk_tokens, budget_tokens=min(by_time, by_bytes))
            if gone:
                dropped += gone
                limit = "time" if by_time <= by_bytes else "bytes"
            if not ids:
                break
        chunk_tokens_now = sum(tokens_of[d] for d in ids)
        ctx = capture_ctx(max(tokens_of[d] for d in ids))
        docs.write_text("".join(records[d] + "\n" for d in ids), encoding="utf-8")
        argv = [PYTHON, HERE.parent / "linear_transfer" / "collect_v2.py", "--probe", probe, "--mode", mode,
                "--model", campaign.gguf, "--model-id", campaign.name, "--docs", docs, "--output", directory,
                "--ctx", ctx, "--batch", CAPTURE_BATCH, "--threads", CAPTURE_THREADS, "--chat-template",
                campaign.path("render", "template.jinja"), "--hash-cache", CAMPAIGN_DIR / "model-hashes.json"]
        if mode == "collect":
            argv += ["--capture-spec", spec_file]
        if device == "vulkan":
            argv += ["--device", "vulkan"]
        commands_before = len(campaign.stage("capture").get("commands", []))
        try:
            seconds = campaign.run(argv, stage="capture", label=f"collect-{name}",
                                   memory_bytes=campaign.gguf.stat().st_size, cwd=HERE.parent / "linear_transfer",
                                   timeout=max(60.0, hard_limit - probe_seconds), cpu=device == "cpu")
        except CommandTimeout:
            remove_chunks(out, index)
            if index == 0:
                raise StageError(f"the dev and test documents were not captured within {hard_limit:.0f} s of probe "
                                 f"time ({HARD_FACTOR} x the time budget)") from None
            dropped += ids + pending
            pending, limit = [], "timeout"
            campaign.log({"event": "capture-timeout", "chunk": name, "documents": len(ids)})
            break
        command = campaign.stage("capture")["commands"][commands_before]
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        info = {"documents": len(ids), "tokens": chunk_tokens_now, "rows": manifest["rows"], "ctx": ctx,
                "seconds": round(seconds, 1), "waited_s": command.get("waited_s"),
                "max_rss_gib": command.get("max_rss_gib")}
        write_json(info_path, info)
        probe_seconds += seconds
        waited += command.get("waited_s") or 0.0
        rows += manifest["rows"]
        captured_tokens += chunk_tokens_now
        chunks.append({**info, "name": name, "adopted": False})
        campaign.log({"event": "capture-chunk", "chunk": name, **info,
                      "tokens_per_second": round(rows / probe_seconds, 1)})
    remove_chunks(out, len(chunks))
    if pending:
        dropped += pending
    captured = [line for chunk in chunks
                for line in (out / f"{chunk['name']}.docs.jsonl").read_text(encoding="utf-8").splitlines()]
    campaign.path("capture-docs.jsonl").write_text("".join(line + "\n" for line in captured), encoding="utf-8")
    speed = rows / probe_seconds if probe_seconds > 0 else None
    return {"captures": [str(out / chunk["name"]) for chunk in chunks], "chunks": chunks, "documents": len(captured),
            "tokens": captured_tokens, "rows": rows, "bytes": rows * per_token, "limit": limit, "dropped": len(dropped),
            "mode": mode, "device": device, "probe": str(probe), "probe_seconds": round(probe_seconds, 1), "waited_s": round(waited, 1),
            "tokens_per_second": None if speed is None else round(speed, 1),
            "speed_share_of_smoke": None if speed is None or not smoke else round(speed / smoke, 3),
            "adopted": bool(chunks) and all(chunk["adopted"] for chunk in chunks),
            "planned": {k: plan[k] for k in ("tokens", "bytes", "seconds", "limit", "budget_tokens", "train_fraction")}}


def sliding(spec):
    return any(kind == "sliding" for kind in spec.layer_types)


def own_files(directory, name):
    """{file: sha256} of a model's own files in a bundle (its manifest entries, either format)."""
    import hub
    manifest = hub.read_manifest(directory)
    out = {}
    for (table, key), entry in hub._bundle_entries(manifest).items():
        if hub._owner(key, manifest["models"]) == name:
            for item in (entry["files"].values() if table == "sides" else [entry]):
                out[item["file"]] = item["sha256"]
    return out


def fit_python(environ=None):
    """The Python fit_hub.py runs with: $KVPARK_FIT_PYTHON, else FIT_PYTHON (torch with ROCm) when it exists,
    else the experiment venv (no torch: fit_hub's --fit-device auto then fits on the CPU)."""
    named = (os.environ if environ is None else environ).get("KVPARK_FIT_PYTHON")
    if named:
        return Path(named)
    return FIT_PYTHON if FIT_PYTHON.is_file() else PYTHON


def fits_on_gpu(python):
    """Whether fit_hub.py run with `python` (--fit-device auto) may fit on the GPU: any Python but the experiment
    venv, which has no torch (the memory estimate then counts the device's memory too)."""
    return Path(python) != PYTHON


def stage_onboard(campaign, args):
    spec = campaign.spec()
    base = json.loads((BASE_HUB / "manifest.json").read_text(encoding="utf-8"))
    settings = base["settings"]
    python = fit_python()
    argv = [python, HERE / "fit_hub.py", "--base", BASE_HUB, "--link-base", "--evaluate", "new",
            "--onboard-encoder", ONBOARD_ENCODER, "--fit-device", "auto",
            "--model", model_argument(campaign.name, campaign.captures(), campaign.path("render", "records.jsonl"))]
    for name, captures, records in base_models():
        argv += ["--model", model_argument(name, captures, records)]
    argv += ["--output", campaign.path("hub"), "--view-dim", settings["view_dim"], "--layer-rank", settings["layer_rank"],
             "--fit-rows", settings["fit_rows"]]
    if sliding(spec):
        argv += ["--source-global", campaign.name]
    if spec.indexer is not None:
        argv += ["--indexer-ridge", campaign.name]
    fit_seconds = campaign.run(argv, stage="onboard", label="fit_hub", cpu=True,
                               memory_bytes=hub_bytes(BASE_HUB) + fit_hub_extra(
                                   spec, settings["layer_rank"], dev_spans=base["spans"]["dev"],
                                   pivots=len(base_models()), hub_dim=settings["hub_dim"], gpu=fits_on_gpu(python)))
    convert_seconds = campaign.run([PYTHON, HERE / "convert_hub.py", campaign.path("hub"), campaign.path("hub-v2"),
                                    "--reuse", BASE_HUB_V2], stage="onboard", label="convert_hub")
    return {**onboard_result(campaign, ONBOARD_ENCODER, python), "source_global": sliding(spec),
            "fit_seconds": round(fit_seconds, 1), "convert_seconds": round(convert_seconds, 1)}


def onboard_result(campaign, encoder, python):
    """The onboard stage's result from its bundles and fit_hub's log: spans, held-out R^2 per kind against
    each pivot, the decoders' alphas, the encoder, where the fit ran, and the sha256 of the model's files."""
    evaluation = json.loads(campaign.path("hub", "evaluation.json").read_text(encoding="utf-8"))
    log = [json.loads(line) for line in campaign.path("hub.log.jsonl").read_text(encoding="utf-8").splitlines()]
    spans = next(item for item in log if item.get("stage") == "spans")
    r2 = {kind: {pair: round(item["r2"], 4) for pair, item in evaluation[kind].items()} for kind in ("k", "v")}
    alphas = {item["kind"]: {"alpha": item["alpha"], "dev_r2": item["dev_r2"]} for item in log
              if item.get("stage") == "onboard"}
    backend = next((item for item in log if item.get("stage") == "fit-backend-done"), {})
    manifest = campaign.path("hub", "manifest.json")
    settings = (json.loads(manifest.read_text(encoding="utf-8")).get("settings") or {}) if manifest.is_file() else {}
    # the encoders' train diagnostics (hub.solve_onboard_encoder): per kind and transform, the last fit's
    encoder_stats = {}
    for item in log:
        if item.get("stage") == "onboard-encoder":
            encoder_stats.setdefault(item["kind"], {})[item["transform"]] = {
                key: value for key, value in item.items() if key not in ("stage", "kind", "model", "transform")}
    return {"spans": spans["spans"], "documents": spans["documents"], "rejected": spans["rejected"],
            "r2": r2, "alphas": alphas, "pivots": [name for name, _, _ in base_models()],
            "encoder": encoder, "variance_cap": settings.get("onboard_variance_cap"),
            "encoder_stats": encoder_stats or None, "warnings": [item["warning"] for item in log
                                                                 if item.get("stage") == "onboard-warning"],
            "fit_python": str(python), "fit_device": backend.get("device"),
            "files": {"hub": own_files(campaign.path("hub"), campaign.name),
                      "hub-v2": own_files(campaign.path("hub-v2"), campaign.name)}}


# a re-encode's statistics beyond FIT_HUB_BASE (and FIT_HUB_GPU on the GPU) and the bundle it loads: Granite-4.0-H-Tiny's
# (2026-10-09, GPU, 4.7 GiB bundle) peaked at 9.6 GiB anonymous and 4.1 GiB reserved on the device
REENCODE_EXTRA = 2 * GIB
OUTDATED = "outdated"           # a stage status: not done, its finished parts kept (no stale wipe)


def reencode(campaign, encoder=None, *, stamp=None):
    """Fit the onboarded model's encoder again with `encoder` (default ONBOARD_ENCODER; fit_hub.py --reencode:
    its view transforms and decoders stay exactly as they are) into hub/ and hub-v2/, under the GPU/memory
    and CPU locks and the guard. The previous bundles, fit log and the evaluation runs out of the model
    (the only ones whose translations change) move to history/onboard-<encoder>-<stamp>/; evaluate and
    report become "outdated": the next `run` evaluates the directions out of the model again, keeps the
    runs into it (kept_run: the same command; their hub sides are unchanged), and reports. Assembled
    hubs keep the old adapters (`status` flags them). Returns the onboard stage's new result."""
    encoder = encoder or ONBOARD_ENCODER
    import hub
    if encoder not in hub.ONBOARD_ENCODERS:
        raise SystemExit(f"unknown onboarding encoder {encoder} (one of {', '.join(hub.ONBOARD_ENCODERS)})")
    onboard = campaign.state["stages"].get("onboard", {})
    if onboard.get("status") != "done":
        raise SystemExit(f"{campaign.name} is not onboarded (campaign.py run --name {campaign.name})")
    previous = (onboard.get("result") or {}).get("encoder") or "ridge"
    # an encoder fitted before the reverse family's variance cap existed records none: fitting it again with the
    # cap is a change (it is bit-identical where the cap does not bind)
    capped = encoder in ("reverse", "reverse_z")
    previous_cap = (onboard.get("result") or {}).get("variance_cap") if capped else None
    if previous == encoder and previous_cap == (hub.ONBOARD_VARIANCE_CAP if capped else None):
        raise SystemExit(f"{campaign.name}'s encoder is already {encoder}")
    new, new_v2 = campaign.path("hub.reencode"), campaign.path("hub-v2.reencode")
    for leftover in (new, new_v2, campaign.path("hub.reencode.log.jsonl"), campaign.path("hub.reencode.guard.jsonl")):
        if leftover.is_dir():
            shutil.rmtree(leftover)
        elif leftover.exists():
            leftover.unlink()
    python = fit_python()
    argv = [python, HERE / "fit_hub.py", "--base", campaign.path("hub"), "--reencode", "--link-base", "--evaluate", "new",
            "--onboard-encoder", encoder, "--fit-device", "auto",
            "--model", model_argument(campaign.name, campaign.captures(), campaign.path("render", "records.jsonl"))]
    for name, captures, records in base_models():
        argv += ["--model", model_argument(name, captures, records)]
    argv += ["--output", new]
    entry = onboard.setdefault("reencoded", [])
    fit_seconds = campaign.run(argv, stage="onboard", label=f"fit_hub-reencode-{encoder}", cpu=True,
                               memory_bytes=hub_bytes(campaign.path("hub")) + FIT_HUB_BASE + REENCODE_EXTRA
                               + (FIT_HUB_GPU if fits_on_gpu(python) else 0))
    convert_seconds = campaign.run([PYTHON, HERE / "convert_hub.py", new, new_v2, "--reuse", BASE_HUB_V2],
                                   stage="onboard", label=f"convert_hub-reencode-{encoder}")
    history = campaign.path("history", f"onboard-{previous}-{stamp or time.strftime('%Y%m%dT%H%M%S')}")
    history.mkdir(parents=True)
    moved = []
    for name in ("hub", "hub-v2", "hub.log.jsonl", "hub.guard.jsonl"):
        if campaign.path(name).exists():
            campaign.path(name).rename(history / name)
            moved.append(name)
    out_of = [p for p in campaign.path("eval").glob(f"{campaign.name}-to-*")] if campaign.path("eval").is_dir() else []
    if out_of:
        (history / "eval").mkdir()
        for path in out_of:
            path.rename(history / "eval" / path.name)
            moved.append(f"eval/{path.name}")
    new.rename(campaign.path("hub"))
    new_v2.rename(campaign.path("hub-v2"))
    for suffix in ("log.jsonl", "guard.jsonl"):
        if campaign.path(f"hub.reencode.{suffix}").exists():
            campaign.path(f"hub.reencode.{suffix}").rename(campaign.path(f"hub.{suffix}"))
    record = {"from": previous, "to": encoder, "at": now(), "history": str(history), "moved": moved,
              "fit_seconds": round(fit_seconds, 1), "convert_seconds": round(convert_seconds, 1)}
    entry.append(record)
    result = {**onboard.get("result"), **onboard_result(campaign, encoder, python),
              "alphas": (onboard.get("result") or {}).get("alphas")}     # the decoders are the onboarding's
    onboard["result"] = result
    for stage in ("evaluate", "report"):
        item = campaign.state["stages"].get(stage)
        if item and item.get("status") == "done":
            item.update(status=OUTDATED, reason=f"the encoder was fitted again ({previous} -> {encoder}): "
                                                "the directions out of the model run again")
    campaign.save()
    campaign.log({"event": "reencoded", "from": previous, "to": encoder, "history": str(history), "moved": moved})
    return result


def stage_tables(campaign, args):
    argv = [PYTHON, HERE / "token_tables.py",
            "--model", model_argument(campaign.name, campaign.captures(), campaign.path("render", "records.jsonl"))]
    for name, captures, records in base_models():
        argv += ["--model", model_argument(name, captures, records)]
    argv += ["--output", campaign.path("token-tables"), "--only", campaign.name]
    campaign.run(argv, stage="tables", label="token_tables", memory_bytes=TABLES_MEMORY, cpu=True)
    path = campaign.path("token-tables", f"{campaign.name}.npz")
    with np.load(path, allow_pickle=False) as npz:
        unique = int(len(npz["token_id"]))
        rows = int(npz["count"].sum())
    return {"table": str(path), "unique_tokens": unique, "rows": rows}


def stage_recurrent(campaign, args):
    argv = [PYTHON, HERE / "r0_experiments.py", "calibrate", "--model", campaign.gguf, "--name", campaign.name]
    for corpus in reversed(HUB_CORPUS):
        argv += ["--corpus", corpus]
    argv += ["--probe", PROBE_VULKAN, "--output", campaign.path("recurrent")]
    campaign.run(argv, stage="recurrent", label="calibrate", memory_bytes=campaign.gguf.stat().st_size, cpu=True)
    report = json.loads(campaign.path("recurrent", "report.json").read_text(encoding="utf-8"))
    return {"npz": report["npz"], "npz_bytes": report["npz_bytes"], "documents": report["documents"],
            "saves": report["saves"], "probe_seconds": report["probe_seconds"],
            "groups": {name: {"count": item["count"], "explained": item["explained"]}
                       for name, item in report["groups"].items()}}


def directions(campaign):
    """[(label, source name, source GGUF, target name, target GGUF, target is the new model)]."""
    return [(f"{REFERENCE}-to-{campaign.name}", REFERENCE, REFERENCE_GGUF, campaign.name, campaign.gguf, True),
            (f"{campaign.name}-to-{REFERENCE}", campaign.name, campaign.gguf, REFERENCE, REFERENCE_GGUF, False)]


def target_recurrent(campaign, into_new):
    """pair_eval's recurrent options for a direction's target: Qwen3.8's mean; the new model's mean
    when its cell holds a state, the zero cell when it holds only convolution taps."""
    if not into_new:
        return ["--recurrent", "mean", "--recurrent-mean", REFERENCE_MEAN]
    if campaign.needs_mean:
        return ["--recurrent", "mean", "--recurrent-mean", campaign.path("recurrent", "recurrent-mean.npz")]
    if campaign.facts()["hybrid"]:
        return ["--recurrent", "zero"]
    return []


def eval_argv(source, target, output, *, hub, tables, recurrent, fixed=True, server_render=False):
    """pair_eval.py's command for one run: source and target as (name, GGUF); fixed: RESULTS.md
    section 2's chats (--ids), else the longer set; server_render: both models rendered as
    llama-server renders them (pair_eval --server-render)."""
    argv = [PYTHON, HERE / "pair_eval.py", "--corpus", EVAL_CORPUS, "--sources", EVAL_SOURCES]
    if fixed:
        argv += ["--ids", ",".join(EVAL_IDS), "--min-documents", EVAL_MIN_DOCUMENTS, "--limit", len(EVAL_IDS),
                 "--min-prefix", EVAL_PREFIX[0]]
    else:
        argv += ["--limit", LONG_LIMIT, "--min-prefix", LONG_MIN_PREFIX]
    argv += ["--max-prefix", EVAL_PREFIX[1], "--hub", hub, "--source", f"{source[0]}={source[1]}",
             "--target", f"{target[0]}={target[1]}", "--probe", PROBE_VULKAN, "--tables", tables,
             "--output", output, *recurrent] + (["--server-render"] if server_render else []) + ["--resume"]
    return [str(item) for item in argv]


def server_renders_differently(render_result):
    """Whether the model's llama-server renders conversations other than its template's defaults
    do: the render stage found sampled records that differ from the server's bytes, and the server
    context reproduces every one of them (Qwen3.6: the server keeps earlier reasoning, the template
    drops it). The evaluation then also runs with both models rendered as their servers do."""
    differing = ((render_result or {}).get("parity") or {}).get("records_differing") or []
    return bool(differing) and all(item.get("reproduced") and item.get("context") == "server" for item in differing)


RESULT_NEUTRAL = ("--resume",)       # pair_eval flags that do not change a run's results
# an evaluation's memory beyond the larger model and the two hub sides it reads: the probe's KV cache and compute buffers
# (GTT on this APU), pair_eval's states and scores. Nemotron-3.5 -> Qwen3.8 (2026-10-09, the first run that recorded its
# GPU memory) took 34.0 GiB (7.8 anonymous, 30.5 on the GPU) where those two came to 33.2
EVALUATION_EXTRA = 4 * GIB


def command_digest(argv):
    """sha256 of a command, without the flags that do not change its results (a run made before pair_eval
    had --resume is the same run)."""
    kept = [str(item) for item in argv if str(item) not in RESULT_NEUTRAL]
    return hashlib.sha256(json.dumps(kept).encode("utf-8")).hexdigest()


def kept_run(output, argv):
    """Whether `output` holds a finished pair_eval run of exactly this command (<output>.command.json)."""
    record = Path(output).with_name(Path(output).name + ".command.json")
    if not (Path(output) / "report.json").is_file() or not record.is_file():
        return False
    return json.loads(record.read_text(encoding="utf-8")).get("sha256") == command_digest(argv)


def run_evaluation(campaign, label, argv, output, memory_bytes):
    """pair_eval into `output`, unless it holds a finished run of this very command; an unfinished run of
    this command (the guard stopped it) continues (pair_eval --resume), any other output is removed."""
    output = Path(output)
    if kept_run(output, argv):
        campaign.log({"event": "evaluation-kept", "run": label})
        return
    record = output.with_name(output.name + ".command.json")
    same = record.is_file() and json.loads(record.read_text(encoding="utf-8")).get("sha256") == command_digest(argv)
    if output.exists() and not same:
        shutil.rmtree(output)
    if same and output.exists():
        campaign.log({"event": "evaluation-resumed", "run": label})
    write_json(record, {"argv": argv, "sha256": command_digest(argv), "written": now()})
    campaign.run(argv, stage="evaluate", label=f"pair_eval-{label}", memory_bytes=memory_bytes, cpu=True,
                 gpu_evaluation=True)


def baseline_label():
    return f"{BASELINE_SOURCE}-to-{REFERENCE}"


def run_baseline(campaign):
    """Qwen3-4B -> Qwen3.8 on the fixed set with the settings of NAME -> Qwen3.8 (Qwen3.8's tables
    and recurrent mean, the new pin's probe), through the base hub: shared by every campaign and run
    once (again only when its command changes). Returns its directory."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    output = BASELINE_DIR / baseline_label()
    argv = eval_argv((BASELINE_SOURCE, BASELINE_GGUF), (REFERENCE, REFERENCE_GGUF), output, hub=BASE_HUB_V2,
                     tables=REFERENCE_TABLES, recurrent=target_recurrent(campaign, into_new=False))
    with open(BASELINE_DIR / ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run_evaluation(campaign, f"baseline-{baseline_label()}", argv, output,
                       max(BASELINE_GGUF.stat().st_size, REFERENCE_GGUF.stat().st_size)
                       + sides_bytes(BASE_HUB_V2, BASELINE_SOURCE, REFERENCE) + EVALUATION_EXTRA)
    return output


def stage_evaluate(campaign, args):
    out = campaign.path("eval")
    out.mkdir(exist_ok=True)
    runs = {}
    server = server_renders_differently(campaign.state["stages"].get("render", {}).get("result"))
    if server:
        campaign.log({"event": "evaluation-server-rendering",
                      "why": "the model's server renders the sampled conversations other than its template's defaults "
                             "(render/records-differing.json): both directions run again as the servers render them"})
    for variant in ("", "-server") if server else ("",):
        for label, s_name, s_gguf, t_name, t_gguf, into_new in directions(campaign):
            label += variant
            tables = campaign.path("token-tables") if into_new else REFERENCE_TABLES
            recurrent = target_recurrent(campaign, into_new)
            # the larger model (the probe runs one at a time) and the two hub sides pair_eval reads
            memory = (max(Path(s_gguf).stat().st_size, Path(t_gguf).stat().st_size)
                      + sides_bytes(campaign.path("hub-v2"), s_name, t_name) + EVALUATION_EXTRA)
            argv = eval_argv((s_name, s_gguf), (t_name, t_gguf), out / label, hub=campaign.path("hub-v2"),
                             tables=tables, recurrent=recurrent, server_render=bool(variant))
            run_evaluation(campaign, label, argv, out / label, memory)
            report = json.loads((out / label / "report.json").read_text(encoding="utf-8"))
            runs[label] = {"report": str(out / label / "report.json"), "documents": len(report["documents"]),
                           "selection": report.get("selection"), "left_out": len(report.get("left_out") or {})}
            if report.get("left_out"):
                # the target renders some of the fixed chats too short: the longer set as well
                long_label = f"{label}-long"
                argv = eval_argv((s_name, s_gguf), (t_name, t_gguf), out / long_label, hub=campaign.path("hub-v2"),
                                 tables=tables, recurrent=recurrent, fixed=False, server_render=bool(variant))
                run_evaluation(campaign, long_label, argv, out / long_label, memory)
                report = json.loads((out / long_label / "report.json").read_text(encoding="utf-8"))
                runs[long_label] = {"report": str(out / long_label / "report.json"),
                                    "documents": len(report["documents"]), "selection": report.get("selection")}
    baseline = run_baseline(campaign)
    return {"runs": runs, "baseline": str(baseline / "report.json"), "server_rendering": server}


def read_runs(campaign):
    """{label: pair_eval report} of the evaluate stage's runs."""
    result = campaign.state["stages"].get("evaluate", {}).get("result") or {}
    paths = {label: item["report"] for label, item in (result.get("runs") or {}).items()} or \
        {label: str(campaign.path("eval", label, "report.json")) for label, *_ in directions(campaign)}
    return {label: json.loads(Path(path).read_text(encoding="utf-8")) for label, path in paths.items()}


def stage_report(campaign, args):
    state = campaign.state
    stages = state["stages"]
    arch = state.get("architecture")
    # read again from the pair_eval reports, so the summary can change without evaluating again
    runs = read_runs(campaign)
    evaluation = {label: evaluation_summary(report) for label, report in runs.items()}
    summary, pooled = {}, {}
    # per direction: the templates' rendering, then (when the evaluate stage ran it) the servers'
    for variant in ("", "-server"):
        for label, s_name, _, t_name, _, into_new in directions(campaign):
            members = [name for name in (label + variant, f"{label}{variant}-long") if name in runs]
            if not members:
                continue
            rows, duplicates = pool_rows([runs[name] for name in members])
            pooled[label + variant] = rows
            prefix, rule = switch_setting(rows, t_name == REFERENCE or (into_new and arch in NATIVE_PREFIX_ARCHITECTURES))
            summary[label + variant] = {
                "source": s_name, "target": t_name, "rendering": "server" if variant else "template", "runs": members,
                "documents": len(rows), "duplicate_max_abs_diff": duplicates, "native_prefix": prefix, "rule": rule,
                **headline(rows, prefix), "conditions": conditions_of(rows), "paired": paired_table(rows)}
            if variant:
                condition, floor = (("hub_prefix", "floor_prefix") if prefix else ("hub", "floor"))
                summary[label + variant]["against_template"] = rendering_comparison(pooled[label], rows, condition, floor)
    # the bridge renders a live conversation as the target's server does: its runs decide when there are any
    into = f"{REFERENCE}-to-{campaign.name}"
    into = into + "-server" if into + "-server" in summary else into
    out_of = f"{campaign.name}-to-{REFERENCE}"
    out_of = out_of + "-server" if out_of + "-server" in summary else out_of
    # a switch out of the model that does worse than the floor (paired 95% CI of the switch's KL against the floor's
    # above 0): register it as a target only (kvpark bridge add --no-source), so the next model prefills natively
    baseline_path = Path((stages.get("evaluate", {}).get("result") or {}).get("baseline")
                         or BASELINE_DIR / baseline_label() / "report.json")
    baseline = None
    if baseline_path.is_file():
        baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
        reverse = f"{campaign.name}-to-{REFERENCE}"
        baseline = {"label": baseline_label(), "report": str(baseline_path), **evaluation_summary(baseline_report),
                    "comparison": baseline_comparison(runs[reverse]["documents"], baseline_report["documents"])}
    recurrent = stages.get("recurrent", {})
    drafts = dflash_drafts(state.get("catalog"))
    log_path = campaign.path("log.jsonl")
    spent = command_seconds([json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                            if log_path.is_file() else [], bool(state.get("recurrent_mean")))
    windows = [item["window"] for item in drafts if item["window"] is not None]
    report = {"name": campaign.name, "gguf": str(campaign.gguf), "architecture": arch, "hybrid": state.get("hybrid"),
              "recurrent_cell": state.get("recurrent_cell"), "generated": now(), "git": git_state(),
              "campaign_sha256": cached_sha256(Path(__file__).resolve()),
              "stages": {stage: {**{key: stages.get(stage, {}).get(key)
                                    for key in ("status", "seconds", "waited_s", "disk", "max_rss_gib", "max_anon_gib",
                                                "max_total_gib", "reason")}, "command_seconds": spent.get(stage)}
                         for stage in STAGES if stage != "report"},
              "render": stages["render"]["result"], "capture": stages["capture"]["result"],
              "capture_plan": {k: v for k, v in json.loads(campaign.path("capture-plan.json").read_text(
                  encoding="utf-8")).items() if k not in ("documents", "order", "held_out")},
              "onboard": {**stages["onboard"]["result"],
                          "reencoded": stages["onboard"].get("reencoded") or stages["onboard"]["result"].get("reencoded")},
              "tables": stages["tables"]["result"],
              "recurrent": recurrent.get("result"), "recurrent_status": recurrent.get("status"),
              "evaluate": evaluation, "directions": summary, "baseline": baseline,
              "notes": ["Floors into Qwen3.8 here start its recurrent cell from its recurrent mean (calibrated with "
                        "the old-pin probe) on llama.cpp a7b94df2c; RESULTS.md section 2's start it from zero on the "
                        "old pin, so their floors and recovered shares are not comparable with these. The comparison "
                        f"with {BASELINE_SOURCE} -> {REFERENCE} runs under the very settings of this report."],
              "bridge": {"hub_name": campaign.name,
                         "native_prefix": summary[into]["native_prefix"], "native_prefix_from": into,
                         "source": summary[out_of]["verdict"] != "higher" if out_of in summary else True,
                         "source_from": out_of if out_of in summary else None,
                         "recurrent_mean": (recurrent.get("result") or {}).get("npz") if state.get("recurrent_mean")
                         else None,
                         "dflash_drafts": drafts, "draft_window": max(windows) if windows else None}}
    write_json(campaign.path("report.json"), report)
    campaign.path("report.md").write_text(report_markdown(report), encoding="utf-8")
    return {"report": str(campaign.path("report.json")), "markdown": str(campaign.path("report.md"))}


def _size(value):
    if value is None:
        return "—"
    return f"{value / 10 ** 9:.2f} GB" if value >= 10 ** 7 else f"{value / 10 ** 6:.1f} MB"


def _share(value):
    return "—" if value is None else f"{value:.0%}"


def _cell(item, metric, ci, digits=3, sign=True):
    value = item[metric]
    low, high = item[ci]
    form = f"{{:+.{digits}f}}" if sign else f"{{:.{digits}f}}"
    return f"{form.format(value)} [{form.format(low)}, {form.format(high)}]"


def _diff(item, digits=4):
    """A paired difference with its bootstrap interval and, when known, the sign-flip test's p."""
    text = f"{item['diff']:+.{digits}f} [{item['ci95'][0]:+.{digits}f}, {item['ci95'][1]:+.{digits}f}]"
    return text + (f" (p {item['p']:.3f})" if item.get("p") is not None else "")


def direction_markdown(item):
    """The paragraph and tables of one direction's pooled chats."""
    head = item
    largest = head.get("largest") or {}
    kl, nll = head["paired_kl"], head["paired_nll"]
    text = (f"A switch uses `{head['condition']}` ({head['rule']}). Against `{head['floor']}` on the "
            f"{item['documents']} chats: KL {_diff(kl)} (lower on {kl['lower']}/{kl['documents']}; "
            f"{head['verdict']}), ΔNLL {_diff(nll)}; it removes {_share(head['recovered_kl'])}"
            + ("" if head["recovered_kl_ci95"] is None else
               f" [{_share(head['recovered_kl_ci95'][0])}, {_share(head['recovered_kl_ci95'][1])}]")
            + f" of the floor's KL"
            + ("" if head["recovered_nll"] is None else f" and {_share(head['recovered_nll'])} of its ΔNLL")
            + ("." if head["recovered_nll"] is not None else " (the floor's ΔNLL is not clearly above 0).")
            + (f" One chat ({largest['document_id'][:8]}) carries {_share(largest['floor_share'])} of the floor's KL; "
               f"without it KL is {largest['hub_without']:.4f} against {largest['floor_without']:.4f} "
               f"({_share(largest['share_without'])} removed)." if largest else ""))
    lines = [text, "", "| Paired difference | KL | ΔNLL | Lower on |", "| --- | --- | --- | ---: |"]
    for name, pair in item["paired"].items():
        lines.append(f"| {name.replace('-', ' − ')} | {_diff(pair['kl'])} | {_diff(pair['nll_delta'])} | "
                     f"{pair['kl']['lower']}/{pair['kl']['documents']} |")
    return lines


def encoder_stats_markdown(stats, warnings=None):
    """One sentence of the encoders' train diagnostics (onboard_result's encoder_stats) and any warnings."""
    if not stats:
        return ""
    parts = []
    for kind in sorted(stats):
        for key, item in sorted(stats[kind].items()):
            text = (f"{kind.upper()}{'' if key == 'full' else ' (layer subset)'}: per-dimension train R² from "
                    f"{item['min_r2']:.3f} (median {item['median_r2']:.3f}), variance {item['min_ratio']:.2f}-"
                    f"{item['max_ratio']:.2f}x the pivots' estimates'")
            if item.get("capped"):
                text += f", {item['capped']} dimension(s) capped"
            parts.append(text)
    out = " " + "; ".join(parts) + "."
    if warnings:
        out += " **Warnings:** " + "; ".join(warnings) + "."
    return out


def report_markdown(report):
    """The report as markdown: stages, render and capture, R^2, end-to-end tables, the comparison
    with the baseline source, the recurrent cell, settings and notes."""
    lines = [f"# Campaign: {report['name']}", "",
             f"`{report['gguf']}`, architecture `{report['architecture']}`"
             f"{' (hybrid)' if report['hybrid'] else ''}; generated {report['generated']}"
             + (f" at git {report['git']['head'][:9]}" if (report.get("git") or {}).get("head") else "")
             + (f" with {len(report['git']['changed'])} changed files" if (report.get("git") or {}).get("changed")
                else "") + ".", "",
             "## Stages", "", "Wall time of the stage's last attempt; command time of every attempt since it last started "
             "over (a resumed stage keeps what earlier attempts finished: capture chunks, evaluations), from log.jsonl. "
             "Peak memory: anonymous (RssAnon of the command's process group, what it needs resident) / anonymous + GPU "
             "(with the rise of the GPU's GTT use, which RssAnon does not show: recorded since 2026-10-09) / RSS (with the "
             "file pages it touched, the captures it memory-maps).", "",
             "| Stage | Status | Wall time | Commands, all attempts | Lock wait | Disk (own) | Disk (linked) | Peak memory |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for stage, item in report["stages"].items():
        disk = item.get("disk") or {}
        seconds, waited, spent = item.get("seconds"), item.get("waited_s"), item.get("command_seconds")
        rss, anon, total = item.get("max_rss_gib"), item.get("max_anon_gib"), item.get("max_total_gib")
        memory = "—" if rss is None and anon is None else \
            (f"{'—' if anon is None else f'{anon:.1f}'} / {'—' if total is None else f'{total:.1f}'} / "
             f"{'—' if rss is None else f'{rss:.1f}'} GiB")
        lines.append(f"| {stage} | {item.get('status') or '—'} | "
                     f"{'—' if seconds is None else f'{seconds / 60:.1f} min'} | "
                     f"{'—' if spent is None else f'{spent / 60:.1f} min'} | "
                     f"{'—' if waited is None else f'{waited:.0f} s'} | {_size(disk.get('bytes'))} | "
                     f"{_size(disk.get('linked_bytes'))} | {memory} |")
    plan, capture, render = report["capture_plan"], report["capture"], report["render"]
    parity = render["parity"]
    differing = parity.get("records_differing") or []
    lines += ["", "## Render and capture", "",
              f"- Hub corpus: {render['documents']} documents, {render['tokens']:,} tokens with this tokenizer "
              f"(longest {render['max_tokens']:,}).",
              f"- Template parity (tokenizer twin, {parity.get('build_info')}): prompts {parity['prompts']['matched']}"
              f"/{parity['prompts']['cases']} (server errors {parity['prompts']['server_errors']}, tolerated option sets "
              f"{parity.get('tolerated') or 'none'}), whole conversations {parity['records']['matched']}/"
              f"{parity['records']['cases']} (server errors {parity['records']['server_errors']}); records identical "
              f"to the server's bytes: {parity['records_identical']} (contexts {', '.join(parity['contexts'])})."]
    for item in differing:
        lines.append(f"  - `{item['conversation']}` differs at character {item.get('at')}: the server renders it with "
                     f"the `{item.get('context')}` context ({', '.join(sorted(item.get('context_differs') or {}))}), "
                     "the record with the template's defaults.")
    lines += [f"- Capture forecast ({plan['mode']}, {plan['bytes_per_token']:,} bytes per token, "
              f"{plan['tokens_per_second']} tokens/s assumed, chunks of {plan['chunk_tokens']:,} tokens): "
              f"{plan['tokens']:,} tokens, {_size(plan['bytes'])}, limit `{plan['limit']}` "
              f"(budget {plan['budget_tokens']:,} tokens; "
              f"{'no' if plan['train_fraction'] is None else format(plan['train_fraction'], '.0%')} of the train tokens). "
              f"Per split: {', '.join(f'{k} {v['documents']} docs / {v['tokens']:,} tokens' for k, v in sorted(plan['by_split'].items()))}.",
              f"- Captured {capture['rows']:,} rows of {capture['documents']} documents ({_size(capture['bytes'])}) in "
              f"{len(capture['chunks'])} chunks, limit `{capture['limit']}`, {capture['dropped']} documents left out; "
              f"{capture['probe_seconds'] / 60:.1f} min of probe time ({capture['tokens_per_second']} tokens/s measured, "
              f"{capture['speed_share_of_smoke']} of the smoke prefill; lock waits {capture['waited_s']:.0f} s)."]
    onboard = report["onboard"]
    lines += ["", "## Onboarding (held-out R², test spans)", "",
              f"Pivots {', '.join(onboard['pivots'])}; spans train {onboard['spans']['train']:,}, dev "
              f"{onboard['spans']['dev']:,}, test {onboard['spans']['test']:,} ({onboard['documents']} documents, "
              f"{onboard['rejected']} rejected). Encoder `{onboard.get('encoder') or 'ridge'}`"
              + (f" (variance cap {onboard['variance_cap']:g})" if onboard.get("variance_cap")
                 and (onboard.get("encoder") or "ridge") in ("reverse", "reverse_z") else "")
              + "".join(f"; fitted again from `{item['from']}` on {item['at'][:10]} (campaign.py reencode: the "
                        "transforms and decoders are the onboarding's)" for item in onboard.get("reencoded") or [])
              + "." + encoder_stats_markdown(onboard.get("encoder_stats"), onboard.get("warnings")), "",
              "| Pair | K R² | V R² |", "| --- | ---: | ---: |"]
    for pair in onboard["r2"]["k"]:
        lines.append(f"| {pair} | {onboard['r2']['k'][pair]:.3f} | {onboard['r2']['v'][pair]:.3f} |")
    lines += ["", "## End to end (seed-score, first 256 tokens of the final message)", "",
              "Token-weighted means with document-bootstrap 95% CIs; paired differences resample both conditions "
              "of a chat together, and p is the exact sign-flip test's (two-sided; with 8 chats the bootstrap "
              "interval can exclude 0 at p above 0.05)."
              + (" Runs labelled `-server` render both models as llama-server renders them (pair_eval "
                 "--server-render: its default template kwargs, which keep earlier reasoning where a template reads "
                 "them), what a live switch holds; the others with the templates' defaults, as RESULTS.md section 2."
                 if any(item.get("rendering") == "server" for item in report["directions"].values()) else ""), ""]
    for label, item in report["directions"].items():
        server = item.get("rendering") == "server"
        lines += [f"### {item['source']} → {item['target']}{', as the servers render' if server else ''}", ""]
        for run in item["runs"]:
            summary = report["evaluate"][run]
            kind = "the longer set (pair_eval's pick)" if run.endswith("-long") else "RESULTS.md section 2's chats"
            kind += ", rendered as llama-server renders them" if server else ""
            left_out = f"; {len(summary['left_out'])} of the set render too short" if summary["left_out"] else ""
            lines += [f"`{run}`: {summary['documents']} documents, {kind} ({summary['selection']}{left_out}), "
                      f"P {min(summary['P']):,}–{max(summary['P']):,} and T0 {min(summary['t0']):,}–"
                      f"{max(summary['t0']):,} target tokens, recurrent `{summary['recurrent']}`; "
                      f"{_share(summary['mapped_fraction'])} of the translated cells come from the source.", "",
                      "| Condition | ΔNLL | KL | Top-1 |", "| --- | --- | --- | --- |"]
            for name, cell in summary["conditions"].items():
                lines.append(f"| {name} | {_cell(cell, 'nll_delta', 'nll_ci95')} | "
                             f"{_cell(cell, 'kl', 'kl_ci95', sign=False)} | {_cell(cell, 'top1', 'top1_ci95', sign=False)} |")
            lines.append("")
        if len(item["runs"]) > 1:
            lines += [f"Both sets together: {item['documents']} distinct chats (a chat in both scores "
                      f"{'the same' if not item['duplicate_max_abs_diff'] else 'within ' + format(item['duplicate_max_abs_diff'], '.1e')}).", ""]
        lines += direction_markdown(item) + [""]
        if item.get("against_template"):
            lines += [f"The same chats under the templates' rendering (`{label[:-len('-server')]}`) and the servers' "
                      f"(`{label}`): the target's P, the share of translated cells mapped from the source, the floor's "
                      f"KL and the share of it `{item['condition']}` removes.", "",
                      "| Chat | P (template / server) | Mapped | Floor KL | Removed |", "| --- | --- | --- | --- | --- |"]
            for chat in item["against_template"]:
                lines.append(f"| {chat['document_id'][:8]} | {chat['P'][0]:,} / {chat['P'][1]:,} | "
                             f"{_share(chat['mapped_fraction'][0])} / {_share(chat['mapped_fraction'][1])} | "
                             f"{chat['floor_kl'][0]:.4f} / {chat['floor_kl'][1]:.4f} | "
                             f"{_share(chat['recovered_kl'][0])} / {_share(chat['recovered_kl'][1])} |")
            lines.append("")
    baseline = report.get("baseline")
    if baseline and baseline["comparison"].get("documents"):
        comparison = baseline["comparison"]
        same = comparison["floor_max_abs_diff"] < 1e-6
        lines += [f"## {report['name']} → {baseline['target']} against {baseline['source']} → {baseline['target']}", "",
                  f"`{baseline['label']}` ran with the very settings of `{report['name']}-to-{baseline['target']}` "
                  f"(same chats, hub sides, token tables, recurrent `{baseline['recurrent']}`, probe): its floors "
                  + ("are identical, so the source is the only difference." if same else
                     f"differ by up to {comparison['floor_max_abs_diff']:.2e}.")
                  + f" `{comparison['condition']}` KL {report['name']} − {baseline['source']}: {_diff(comparison['kl'])} "
                  f"(lower on {comparison['kl']['lower']}/{comparison['kl']['documents']}), ΔNLL "
                  f"{_diff(comparison['nll_delta'])}; floor KL removed {_share(comparison['recovered_kl'])} against "
                  f"{_share(comparison['baseline_recovered_kl'])}.", "",
                  f"| Chat | P | T0 | Mapped ({report['name']} / {baseline['source']}) | Floor KL | KL {report['name']} | "
                  f"KL {baseline['source']} | Removed ({report['name']} / {baseline['source']}) |",
                  "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |"]
        for chat in comparison["per_chat"]:
            lines.append(f"| {chat['document_id'][:8]} | {chat['P']:,} | {chat['t0']:,} | "
                         f"{_share(chat['mapped_fraction'])} / {_share(chat['baseline_mapped_fraction'])} | "
                         f"{chat['floor_kl']:.4f} | {chat['kl']:.4f} | {chat['baseline_kl']:.4f} | "
                         f"{_share(chat['recovered_kl'])} / {_share(chat['baseline_recovered_kl'])} |")
        lines.append("")
    cell = report.get("recurrent_cell") or {}
    lines += ["## Recurrent cell", ""]
    if not report["hybrid"]:
        lines.append("Not a hybrid: no recurrent cell.")
    elif report.get("recurrent"):
        lines.append(f"Recurrent mean of {report['recurrent']['saves']} native saves: `{report['recurrent']['npz']}`.")
    else:
        lines.append(f"No recurrent mean ({report.get('recurrent_status')}): {cell.get('reason')}.")
    bridge = report["bridge"]
    command = (f"kvpark bridge add ... --hub-name {bridge['hub_name']}"
               + (" --native-prefix" if bridge["native_prefix"] else "")
               + (f" --recurrent-mean {bridge['recurrent_mean']}" if bridge["recurrent_mean"] else "")
               + (" --no-source" if bridge.get("source") is False else ""))
    lines += ["", "## Bridge settings", "", f"`{command}`", ""]
    if bridge.get("source") is False:
        lines += [f"`--no-source`: a switch out of {report['name']} did worse than the floor (`{bridge['source_from']}`: "
                  "the paired 95% CI of its KL against the floor's lies above 0), so the bridge registers it as a target "
                  "only: switches out of it are skipped and the next model prefills natively.", ""]
    if (bridge.get("native_prefix_from") or "").endswith("-server"):
        lines += [f"The native prefix follows `{bridge['native_prefix_from']}`: the bridge renders a live conversation "
                  "as the target's server does.", ""]
    by_window = {}
    for draft in bridge.get("dflash_drafts") or []:
        by_window.setdefault(draft["window"], []).append(f"`{draft['path']}`")
    for window, paths in by_window.items():
        if window is None:
            lines += [f"Its DFlash draft ({', '.join(paths)}) names no sliding window "
                      "(dflash.attention.sliding_window): read it before running the draft on its server.", ""]
            continue
        lines += [f"With its DFlash draft on the server ({', '.join(paths)}; sliding window {window:,} tokens): add "
                  f"`--draft-window {window}`"
                  + ("" if window != BRIDGE_DRAFT_WINDOW else " (the bridge's default)")
                  + f". A switch then keeps a native tail of at least {window - 1:,} tokens, so that the draft "
                  f"sees native context only"
                  + (f" (without the flag the bridge assumes {BRIDGE_DRAFT_WINDOW:,}, and the draft's window would "
                     f"reach into translated cells)" if window != BRIDGE_DRAFT_WINDOW else "")
                  + "; it hands over only prompts longer than that tail plus `--min-handover`. The evaluation "
                  "above scores the 2,048-token tail with no draft.", ""]
    lines += ["## Notes", ""]
    lines += [f"- {note}" for note in report.get("notes") or []]
    lines.append("")
    return "\n".join(lines)


STAGE_FUNCTIONS = {"render": stage_render, "capture": stage_capture, "onboard": stage_onboard, "tables": stage_tables,
                   "recurrent": stage_recurrent, "evaluate": stage_evaluate, "report": stage_report}


def clean_outputs(campaign, stage, *, redo):
    """Remove what an unfinished (or redone) stage left: its outputs, the probe's temporary
    directories; a resumed capture keeps its published chunks (stage_capture adopts or discards
    each) and a resumed evaluation its finished runs (kept_run checks their commands)."""
    for leftover in [*campaign.dir.glob(".kvpark-*"), *campaign.dir.glob("capture/.kvpark-*"),
                     *campaign.dir.glob("eval/.kvpark-*")]:
        shutil.rmtree(leftover, ignore_errors=True)
    for name in OUTPUTS[stage]:
        path = campaign.path(name)
        if not path.exists():
            continue
        if not redo and stage == "capture" and name == "capture":
            continue
        if not redo and stage == "evaluate":
            continue        # finished runs are kept (kept_run), unfinished ones resume or start over (run_evaluation)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def run_stages(campaign, args):
    """Run the planned stages in order, recording each one's status, times, disk and result."""
    redo = parse_stages(args.redo)
    only = parse_stages(args.stages)
    facts = campaign.facts()
    recurrent = bool(facts["recurrent_mean"])
    todo = plan_run(campaign.statuses(), recurrent=recurrent, only=only, redo=redo)
    if "onboard" in todo:
        for directory, _ in stale_hubs(campaign, assume_redo=True):
            campaign.log({"event": "assembled-hub-outdated", "hub": str(directory),
                          "why": "this run onboards the model again: assemble a new hub afterwards"})
    for stage in stale_stages(redo, recurrent):
        if campaign.statuses().get(stage) == "done":
            campaign.stage(stage)["status"] = "stale"
    for stage in STAGES:
        if not applicable(stage, recurrent):
            if campaign.statuses().get(stage) != "skipped":
                clean_outputs(campaign, stage, redo=True)
            campaign.state["stages"][stage] = {"status": "skipped",
                                               "reason": (facts.get("recurrent_cell") or {}).get("reason")}
    campaign.save()
    campaign.log({"event": "run", "todo": todo, "redo": redo, "only": only, "pid": os.getpid(), "git": git_state(),
                  "campaign_sha256": cached_sha256(Path(__file__).resolve())})
    for stage in todo:
        entry = campaign.stage(stage)
        clean_outputs(campaign, stage, redo=stage in redo or entry.get("status") == "stale")
        campaign.state["stages"][stage] = entry = {"status": "running", "started": now(), "pid": os.getpid()}
        campaign.save()
        campaign.log({"event": "stage", "stage": stage})
        with open(campaign.path("commands.sh"), "a", encoding="utf-8") as stream:
            stream.write(f"# {now()} stage {stage}, as a whole: (cd {shlex.quote(str(HERE))} && {shlex.quote(str(PYTHON))} "
                         f"campaign.py run --name {shlex.quote(campaign.name)} --stages {stage})\n")
        started = time.perf_counter()
        try:
            result = STAGE_FUNCTIONS[stage](campaign, args)
        except BaseException as error:
            entry.update(status="failed", finished=now(), seconds=round(time.perf_counter() - started, 1),
                         error=f"{type(error).__name__}: {error}"[-4000:])
            campaign.save()
            campaign.log({"event": "stage-failed", "stage": stage, "error": entry["error"][-600:]})
            raise
        commands = entry.get("commands") or []
        rss = [command["max_rss_gib"] for command in commands if command.get("max_rss_gib") is not None]
        anon = [command["max_anon_gib"] for command in commands if command.get("max_anon_gib") is not None]
        total = [command["max_total_gib"] for command in commands if command.get("max_total_gib") is not None]
        entry.update(status="done", finished=now(), seconds=round(time.perf_counter() - started, 1),
                     waited_s=round(sum(command.get("waited_s") or 0.0 for command in commands), 1),
                     max_rss_gib=max(rss) if rss else None, max_anon_gib=max(anon) if anon else None,
                     max_total_gib=max(total) if total else None,
                     disk=disk_usage([campaign.path(name) for name in OUTPUTS[stage]]), result=result)
        campaign.save()
        campaign.log({"event": "stage-done", "stage": stage, "seconds": entry["seconds"], "disk": entry["disk"]})
    return todo


# -- assemble ----------------------------------------------------------------------------------

def assembled_hubs(campaign):
    """[{"directory", "bundle" ("hub" or "hub-v2"), "assembled"}] of the hubs assembled with this model."""
    path = campaign.path("assembled.json")
    return json.loads(path.read_text(encoding="utf-8"))["hubs"] if path.is_file() else []


def record_assembly(campaign, directory, bundle, when=None):
    """Note in the model's assembled.json that `directory` holds its `bundle` files."""
    hubs = [item for item in assembled_hubs(campaign) if item["directory"] != str(directory)]
    hubs.append({"directory": str(directory), "bundle": bundle, "assembled": when or now()})
    write_json(campaign.path("assembled.json"), {"hubs": hubs})


def stale_hubs(campaign, *, assume_redo=False):
    """[(directory, why)] of the hubs assembled with this model that do not hold its onboarding's
    files (by sha256: state.json's onboard result, else its hub/ and hub-v2/ manifests); with
    assume_redo every hub that holds the model (an onboarding about to run again)."""
    onboard = campaign.state["stages"].get("onboard", {})
    recorded = (onboard.get("result") or {}).get("files") or {}
    out = []
    for item in assembled_hubs(campaign):
        directory = Path(item["directory"])
        if not (directory / "manifest.json").is_file():
            continue
        if assume_redo:
            out.append((directory, "the onboarding runs again"))
            continue
        if onboard.get("status") != "done":
            out.append((directory, f"the onboarding is {onboard.get('status') or 'not done'}"))
            continue
        expected = recorded.get(item["bundle"])
        if expected is None:
            expected = own_files(campaign.path(item["bundle"]), campaign.name)
        if own_files(directory, campaign.name) != expected:
            out.append((directory, "built from an older onboarding of this model"))
    return out


def check_member(campaign):
    """An onboarded model's hub/ and hub-v2/, refused (SystemExit) when the onboarding is not done or
    its bundles no longer hold the files state.json records."""
    onboard = campaign.state["stages"].get("onboard", {})
    if onboard.get("status") != "done":
        raise SystemExit(f"{campaign.name} is not onboarded (campaign.py run --name {campaign.name})")
    recorded = (onboard.get("result") or {}).get("files")
    for bundle in ("hub", "hub-v2"):
        if recorded is not None and own_files(campaign.path(bundle), campaign.name) != recorded[bundle]:
            raise SystemExit(f"{campaign.path(bundle)} does not hold the files of {campaign.name}'s onboarding "
                             f"(state.json): campaign.py run --name {campaign.name} --redo onboard")
    return campaign.path("hub"), campaign.path("hub-v2")


def onboarded_evaluation(output, campaigns, log=print):
    """Held-out R^2 (test spans) of every ordered pair of the onboarded models in an assembled hub
    (format 1), from their captures: they all captured the same dev and test documents. Computed in this
    process on the current fit backend (fit_backend.current()); `campaign.py evaluate-onboarded` runs it as
    the watched child of evaluate_onboarded. Returns {"pairs", "spans", kind: {"a->b": {"r2", "rows"}}}."""
    import hub
    views = [hub.CaptureView(camp.name, camp.captures(), camp.path("render", "records.jsonl")) for camp in campaigns]
    spans = hub.joint_spans(views)
    names = [camp.name for camp in campaigns]
    pairs = [(a, b) for a in names for b in names if a != b]
    hubs, _ = hub.load_hub(output)
    result = {}
    for kind in hub.KINDS:
        started = time.perf_counter()
        result[kind] = {f"{s}->{t}": {"r2": round(item["r2"], 4), "rows": item["rows"]}
                        for (s, t), item in hub.evaluate(hubs[kind], views, spans, kind, pairs=pairs).items()}
        log({"event": "evaluated", "kind": kind, "seconds": round(time.perf_counter() - started, 1)})
    return {"pairs": [f"{a}->{b}" for a, b in pairs],
            "spans": {split: int(len(spans.select(split))) for split in ("train", "dev", "test")}, **result}


# what evaluate_onboarded takes off MemAvailable, per byte of the assembled hub's files: hub6-20261008's evaluation
# (2026-10-09, 15 models, 21.6 GiB of files, on the GPU) peaked at 36.2 GiB anonymous (the whole hub loaded, the
# test chunks) and 12.2 GiB reserved on the device, and MemAvailable fell by 46.9 GiB; the lock asked the hub's
# bytes alone until then
EVALUATE_ONBOARDED_FACTOR = 2.3


def evaluate_onboarded(output, campaigns, log=print, *, campaign_dir=CAMPAIGN_DIR, python=None):
    """onboarded_evaluation of an assembled hub as a child process, `campaign.py evaluate-onboarded` with
    fit_python() (its products on the GPU, --fit-device auto), under the GPU/memory lock (the whole hub is
    loaded), the CPU lock and the resource guard with the GPU-evaluation defaults (resource_guard.GPU_EVALUATION),
    whose Watch can stop the child's process group at any moment (evaluating in this process on the CPU, a Guard
    could pause only between chunks: on 2026-10-07 the package went from 82 C to 98 C within one). Its output in
    output/evaluation-onboarded.log, guard samples in output/evaluation-onboarded.guard.jsonl. GuardStop when the
    guard ended it, StageError when it failed; returns its result (onboarded_evaluation's)."""
    output = Path(output)
    result_path = output / "evaluation-onboarded.json"
    argv = [str(python or fit_python()), str(HERE / "campaign.py"), "evaluate-onboarded", "--hub", str(output),
            "--campaign-dir", str(campaign_dir), "--fit-device", "auto", *[camp.name for camp in campaigns]]
    settings = resource_guard.Settings.from_env(defaults=resource_guard.GPU_EVALUATION)
    environment = resource_guard.thread_env(os.environ, settings.threads)
    guard_log = output / "evaluation-onboarded.guard.jsonl"
    log_path = output / "evaluation-onboarded.log"
    result_path.unlink(missing_ok=True)
    with guarded_locks(int(EVALUATE_ONBOARDED_FACTOR * hub_bytes(output)), log, cpu=True,
                       label="campaign assemble --evaluate") as (held, fds, extra):
        log({"event": "command", "label": "evaluate-onboarded", "command": shlex.join(argv)})
        with open(log_path, "a", encoding="utf-8") as stream:
            outcome = guarded_child(argv, settings=settings, guard_log=guard_log, label="assemble-evaluate", events=log,
                                    environment={**environment, **extra}, cwd=HERE, output=stream, keep_fds=fds)
    log({"event": "command-done", "label": "evaluate-onboarded", "returncode": outcome.returncode,
         "seconds": round(outcome.seconds, 1), "paused_s": outcome.paused_s, "guard_abort": outcome.aborted})
    if outcome.aborted:
        raise GuardStop(f"the evaluation of {output} was ended by the resource guard: {outcome.aborted} (samples "
                        f"in {guard_log}); run it again: campaign.py evaluate-onboarded --hub {output} "
                        + " ".join(camp.name for camp in campaigns))
    if outcome.returncode != 0 or not result_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:] if log_path.is_file() else ""
        raise StageError(f"the evaluation of {output} exited with {outcome.returncode}; the end of its log:\n{tail}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def evaluate_onboarded_command(args):
    """`campaign.py evaluate-onboarded --hub DIR NAME ...`: held-out R^2 between the onboarded models of an
    assembled hub -> DIR/evaluation-onboarded.json. Run by hand it is evaluate_onboarded (the locks and a watched
    child); as that child (KVPARK_GUARD_WATCHED: also under `campaign.py exec`) onboarded_evaluation on the fit
    backend of --fit-device."""
    import fit_backend
    campaigns = []
    for name in args.names:
        if not (Path(args.campaign_dir) / name / "state.json").is_file():
            raise SystemExit(f"{name} has no campaign in {args.campaign_dir}")
        campaigns.append(Campaign(name, directory=Path(args.campaign_dir) / name, catalog=None))

    def log(item):
        print(json.dumps({"time": now(), **item}), flush=True)
    if os.environ.get(resource_guard.WATCHED) != "1":
        try:
            evaluate_onboarded(args.hub, campaigns, log=log, campaign_dir=args.campaign_dir)
        except (StageError, resource_guard.GuardError) as error:
            raise SystemExit(f"evaluate-onboarded: {error}") from None
        return 0
    device = fit_backend.resolve(args.fit_device)
    with fit_backend.using(fit_backend.make(device, log=lambda line: log({"event": "fit-backend", "line": line}))) \
            as backend:
        log({"event": "fit-backend", "device": device})
        result = onboarded_evaluation(args.hub, campaigns, log=log)
        log({"event": "fit-backend-done", "device": device, **backend.stats()})
    write_json(Path(args.hub) / "evaluation-onboarded.json", {**result, "fit_device": device})
    return 0


def assemble(output, names, *, evaluate=False, base=BASE_HUB, base_v2=BASE_HUB_V2, campaign_dir=CAMPAIGN_DIR,
             log=print):
    """One hub of the base and the named onboarded models: output (format 1) and output-v2 (on the
    members' filesystem), recorded in each model's assembled.json; no run of a member may be active.
    evaluate: also R^2 between the onboarded models -> output/evaluation-onboarded.json."""
    import hub
    output = Path(os.path.normpath(Path(output).absolute()))
    output_v2 = output.with_name(output.name + "-v2")
    for path in (output, output_v2):
        if path.exists():
            raise SystemExit(f"{path} exists: assemble a new directory")
    with contextlib.ExitStack() as stack:
        members, members_v2, campaigns = [Path(base)], [Path(base_v2)], []
        for name in names:
            if not (Path(campaign_dir) / name / "state.json").is_file():
                raise SystemExit(f"{name} has no campaign in {campaign_dir}")
            camp = Campaign(name, directory=Path(campaign_dir) / name, catalog=None)
            stack.enter_context(camp.exclusive())
            hub_dir, hub_v2_dir = check_member(camp)
            members.append(hub_dir)
            members_v2.append(hub_v2_dir)
            campaigns.append(camp)
        try:
            manifest = hub.merge_bundles(output, members)
            hub.merge_bundles(output_v2, members_v2)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        for camp in campaigns:
            record_assembly(camp, output, "hub")
            record_assembly(camp, output_v2, "hub-v2")
        result = {"output": str(output), "output_v2": str(output_v2), "models": list(manifest["models"])}
        if evaluate:
            if len(campaigns) < 2:
                result["evaluation"] = "one onboarded model: its pairs with the base are in its hub/evaluation.json"
            else:
                evaluate_onboarded(output, campaigns, log=lambda item: log(json.dumps(item)), campaign_dir=campaign_dir)
                result["evaluation"] = str(output / "evaluation-onboarded.json")
    return result


# -- CLI ----------------------------------------------------------------------------------------

def show_plan(campaign, args):
    facts = campaign.facts()
    statuses, recurrent = campaign.statuses(), bool(facts["recurrent_mean"])
    out = {"name": campaign.name, "gguf": str(campaign.gguf), "architecture": facts.get("architecture"),
           "hybrid": facts.get("hybrid"), "recurrent_cell": facts.get("recurrent_cell"), "statuses": statuses}
    try:
        out["todo"] = plan_run(statuses, recurrent=recurrent, only=parse_stages(args.stages),
                               redo=parse_stages(args.redo))
    except ValueError as error:
        out["todo_error"] = str(error)
    spec = campaign.spec()
    out["capture_mode"] = mode = capture_mode(spec.architecture)
    out["capture_device"] = device = capture_device(spec.architecture)
    out["bytes_per_token"] = per_token = bytes_per_token(spec)
    tokens_path = campaign.path("render", "tokens.json")
    if tokens_path.is_file():
        smoke = ((campaign.state.get("catalog") or {}).get("smoke") or {}).get("prefill_tps")
        tps = args.capture_tps or (smoke * CAPTURE_SPEED[mode, device] if smoke else None)
        if tps:
            budget_bytes, budget_seconds = int(args.budget_gb * 10 ** 9), args.budget_minutes * 60
            chunk_tokens = chunk_tokens_for(min(budget_bytes // per_token, (budget_seconds - LOAD_SECONDS) * tps))
            documents = json.loads(tokens_path.read_text(encoding="utf-8"))["documents"]
            try:
                plan = capture_plan(documents, bytes_per_token=per_token, tokens_per_second=tps,
                                    budget_bytes=budget_bytes, budget_seconds=budget_seconds,
                                    reserve_tokens=(PEAK_FACTOR[mode] - 1) * chunk_tokens)
            except ValueError as error:
                out["capture_plan_error"] = (f"{error}: "
                                             f"{held_out_need(documents, bytes_per_token=per_token, tokens_per_second=tps)}")
            else:
                out["capture_plan"] = {k: v for k, v in plan.items() if k not in ("documents", "order", "held_out")}
                out["capture_plan"].update(documents=len(plan["documents"]), chunk_tokens=chunk_tokens)
    out["memory_need_gib"] = round((campaign.gguf.stat().st_size + GPU_MARGIN) / GIB, 1)
    print(json.dumps(out, indent=1))


def show_status(names):
    if not names and CAMPAIGN_DIR.is_dir():
        names = sorted(p.name for p in CAMPAIGN_DIR.iterdir() if (p / "state.json").is_file())
    for name in names:
        if not (CAMPAIGN_DIR / name / "state.json").is_file():
            print(f"{name}: not started")
            continue
        camp = Campaign(name, catalog=None)
        cells = []
        for stage in STAGES:
            item = camp.state["stages"].get(stage, {})
            seconds = item.get("seconds")
            cells.append(f"{stage}={item.get('status') or '-'}" + (f"({seconds / 60:.1f}m)" if seconds else ""))
        print(f"{name}: " + " ".join(cells))
        for item in assembled_hubs(camp):
            print(f"  assembled into {item['directory']}")
        for directory, why in stale_hubs(camp):
            print(f"  OUTDATED {directory}: {why}")


def exec_command(args):
    """`campaign.py exec`: one command under the GPU/memory lock (and the CPU lock, --cpu-lock), which
    it inherits, and the resource guard (--guard-log: its samples); its exit status
    (resource_guard.ABORT_EXIT when the guard ended it)."""
    command = list(args.argv)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("exec: no command")

    def log(item):
        print(json.dumps({"time": now(), **item}), file=sys.stderr, flush=True)
    settings = resource_guard.Settings.from_env(defaults=resource_guard.GPU_EVALUATION if args.gpu_evaluation else None)
    environment = resource_guard.thread_env(os.environ, settings.threads)
    with guarded_locks(args.memory_bytes, log, cpu=args.cpu_lock, label="campaign exec") as (held, fds, extra):
        outcome = guarded_child(command, settings=settings, guard_log=args.guard_log, label="exec", events=log,
                                environment={**environment, **extra}, keep_fds=fds, timeout=args.timeout)
    log({"event": "exec-done", "returncode": outcome.returncode, "seconds": round(outcome.seconds, 1),
         "timed_out": outcome.timed_out, "max_rss_gib": None if outcome.max_rss is None else round(outcome.max_rss / GIB, 2),
         "max_anon_gib": round(outcome.max_anon / GIB, 2),
         "max_gpu_gib": None if outcome.max_gpu is None else round(outcome.max_gpu / GIB, 2),
         "max_total_gib": round(outcome.max_total / GIB, 2), "paused_s": outcome.paused_s,
         "guard_abort": outcome.aborted})
    if outcome.aborted:
        return resource_guard.ABORT_EXIT
    if outcome.timed_out:
        return 124
    return outcome.returncode if outcome.returncode >= 0 else 128 - outcome.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "plan"):
        sub = commands.add_parser(command, help="run the stages" if command == "run" else "show what a run would do")
        sub.add_argument("--name", required=True, help="the model's hub name and catalog id")
        sub.add_argument("--catalog-id", help="the model's catalog id when it is not its hub name (a new campaign only)")
        sub.add_argument("--gguf", type=Path, help="the model file (default: the catalog's)")
        sub.add_argument("--stages", help="comma list: run only these (default: every stage not done)")
        sub.add_argument("--redo", help="comma list: run these again, with every stage that depends on them")
        sub.add_argument("--capture-tps", type=float, help="capture tokens/s (default: from the catalog's smoke run)")
        sub.add_argument("--budget-gb", type=float, default=BUDGET_BYTES / 10 ** 9, help="capture size budget")
        sub.add_argument("--budget-minutes", type=float, default=BUDGET_SECONDS / 60, help="capture time budget")
        sub.add_argument("--parity-tolerate", action="append", metavar="OPTIONS",
                         help="an option set whose /apply-template refusals the render stage accepts: 'default' or "
                         "its compact sorted JSON, e.g. '{\"reasoning_effort\":\"none\"}' (repeatable)")
    sub = commands.add_parser("status", help="stage statuses of campaign models and their assembled hubs")
    sub.add_argument("names", nargs="*")
    sub = commands.add_parser("assemble", help="one hub (DIR and DIR-v2) of the base and onboarded models")
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--evaluate", action="store_true", help="also R^2 between the onboarded models (loads the hub)")
    sub.add_argument("names", nargs="+")
    sub = commands.add_parser("reencode", help="fit onboarded models' encoders again with another method")
    sub.add_argument("--encoder", default=None, help=f"the onboarding encoder (default {ONBOARD_ENCODER})")
    sub.add_argument("names", nargs="+")
    sub = commands.add_parser("evaluate-onboarded", help="held-out R^2 between the onboarded models of an assembled "
                              "hub (what assemble --evaluate runs: the locks, then a watched child on the GPU)")
    sub.add_argument("--hub", type=Path, required=True, help="the assembled hub (format 1)")
    sub.add_argument("--campaign-dir", type=Path, default=CAMPAIGN_DIR)
    sub.add_argument("--fit-device", default="auto", choices=("auto", "cpu", "gpu"))
    sub.add_argument("names", nargs="+")
    sub = commands.add_parser("exec", help="run one command under the GPU/memory lock (commands.sh uses it)")
    sub.add_argument("--memory-bytes", type=int, required=True, help="what the command loads")
    sub.add_argument("--timeout", type=float, help="stop the command's process group after this many seconds")
    sub.add_argument("--cpu-lock", action="store_true", help="also hold the CPU lock (a CPU-heavy command)")
    sub.add_argument("--gpu-evaluation", action="store_true",
                     help="a GPU evaluation: the guard's defaults are resource_guard.GPU_EVALUATION (stop at 90 C, "
                     "continue at 88 C, 4 threads) where KVPARK_GUARD_* do not set them")
    sub.add_argument("--guard-log", type=Path, help="the resource guard's JSONL samples of the command")
    sub.add_argument("argv", nargs=argparse.REMAINDER, metavar="-- COMMAND ...")
    args = parser.parse_args(argv)
    if args.command == "status":
        show_status(args.names)
        return 0
    if args.command == "evaluate-onboarded":
        return evaluate_onboarded_command(args)
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGHUP, _terminate)
    if args.command == "exec":
        try:
            return exec_command(args)
        except resource_guard.GuardError as error:
            raise SystemExit(f"exec: {error}") from None
    if args.command == "assemble":
        try:
            print(json.dumps(assemble(args.output, args.names, evaluate=args.evaluate)))
        except (StageError, resource_guard.GuardError) as error:
            raise SystemExit(f"assemble: {error}") from None
        except Terminated as error:
            raise SystemExit(f"assemble: stopped ({error})") from None
        return 0
    if args.command == "reencode":
        for name in args.names:
            if not (CAMPAIGN_DIR / name / "state.json").is_file():
                raise SystemExit(f"{name} has no campaign in {CAMPAIGN_DIR}")
            camp = Campaign(name, catalog=None)
            with camp.exclusive():
                try:
                    result = reencode(camp, args.encoder)
                except (StageError, resource_guard.GuardError) as error:
                    raise SystemExit(f"{name}: {error}") from None
                except Terminated as error:
                    raise SystemExit(f"{name}: stopped ({error})") from None
            print(json.dumps({"name": name, "encoder": result["encoder"], "r2": result["r2"]}))
        show_status(args.names)
        return 0
    try:
        campaign = Campaign(args.name, gguf=args.gguf, catalog_id=args.catalog_id)
        campaign.facts()
    except ValueError as error:                 # not in the catalog, a split GGUF, a refused architecture
        raise SystemExit(f"{args.name}: {error}") from None
    if args.command == "plan":
        show_plan(campaign, args)
        return 0
    with campaign.exclusive():
        campaign.save()
        try:
            run_stages(campaign, args)
        except (StageError, resource_guard.GuardError) as error:
            raise SystemExit(f"{campaign.name}: {error}") from None
        except Terminated as error:
            raise SystemExit(f"{campaign.name}: stopped ({error})") from None
    show_status([campaign.name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
