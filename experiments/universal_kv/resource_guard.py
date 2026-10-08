"""Thermal and resource guard for heavy experiment jobs (standard library only).

On 2026-10-05 at 22:02 this machine (Ryzen AI Max+ 395 with the Radeon 8060S in the same package,
123 GiB of unified memory) powered off: the CPU's internal thermal limit tripped (AMD reset reason
0x00080a00; no OOM kill, swap 93% used) while a joint seven-model hub refit (numpy BLAS on every
core) and an onboarding diagnostic ran together on the CPU after hours of GPU captures. Every heavy
job now runs under this guard:

  read_sample()  the package temperature (k10temp Tctl, Tdie and Tccd*, and the iGPU's amdgpu edge,
                 which sits in the same package; amdgpu junction, its PPT power and the NVMe composite
                 temperatures are logged only) from /sys/class/hwmon, MemAvailable, swap and the load
  wait_cool()    before a heavy step: wait until the package is at most start_c (75 C) and memory
                 is not short (MemAvailable at least mem_resume, swap use below swap_resume)
  Watch          around a child process group: a JSONL sample every poll_s (2 s: temperatures,
                 memory, swap, load, the group's RSS); the group is stopped (SIGSTOP) when the package
                 reaches pause_c (90 C), swap use swap_pause (85%) or MemAvailable falls below
                 mem_floor (8 GiB), and continued (SIGCONT) once the package is at most resume_c (80 C)
                 and memory has recovered; GuardAbort, after which the caller ends the group (SIGTERM,
                 SIGCONT, SIGKILL after 30 s: stop_group), when the package is at abort_c (95 C) on
                 abort_samples (2) samples in a row (the group is stopped at the first: Tctl spikes
                 that fast on this machine, and stopping the job alone brings it down 30 C within 2 s),
                 when no CPU temperature can be read for sensor_loss_s (60 s) or when a pause lasts
                 max_pause_s (30 min). Swap use counts only while MemAvailable is below swap_ignore_gib
                 (24 GiB): above it, full swap holds pages swapped out earlier (mostly a stopped job's own,
                 which return only when it runs), and holding the job for it never ended (2026-10-07)
  Guard          the same for loops in this process: check() between chunks sleeps while too hot or
                 short of memory and raises GuardAbort when critical; a background thread logs a
                 sample every poll_s meanwhile and ends the process at abort_c (on abort_samples
                 samples in a row) when no check() comes
                 (one LAPACK call can run for many minutes: run such a job under `run` to have it
                 paused there too)
  KernelLog      GPU hangs: the kernel log from a job's start (journalctl -k -f -o short-unix --since,
                 or /dev/kmsg when readable) scanned for amdgpu "ring ... timeout", "GPU reset", "ring
                 reset" and "device wedged" lines (GPU_HANG). On 2026-10-06 at 09:53 plain float32 torch
                 products of a fit benchmark (8-33 s per kernel on the 8060S) starved the desktop's
                 graphics ring, which shares the iGPU: repeated ring timeouts, GNOME Shell aborted and the
                 driver reset the GPU. Watch (always, gpu_watch=auto) and Guard (Guard(gpu=True), for
                 in-process GPU fits) stop the job at the first such line, like an abort, and log it
                 (event gpu-hang, the summary's gpu_hang)
  cpu_lock()     /tmp/kvpark-campaign-cpu.lock: one CPU-heavy numpy job at a time. A job that needs
                 the GPU/memory lock too (/tmp/kvpark-campaign-gpu.lock, campaign.gpu_lock) takes that
                 one first (check_lock_order refuses the other order); a child handed the locked
                 descriptor (KVPARK_CPU_LOCK_FD) holds the lock without locking again
  thread_env()   BLAS, OpenMP and MKL threads capped (8) for heavy children unless set (OpenBLAS
                 would start one thread per hardware thread, 32)
  run()          one command under all of this; on the command line `resource_guard.py run [--cpu-lock]
                 [--log FILE] -- COMMAND ...`, and `resource_guard.py status` prints one sample

A watched child gets KVPARK_GUARD_WATCHED=1: its own steps (pair_eval's and r0_experiments' probe runs
through run(), a fit's Guard) then leave pausing and logging to the outer Watch, which stops its whole
process group.

GPU evaluations (pair_eval's probe runs; campaign.py's evaluate stage and assemble --evaluate; campaign.py exec
--gpu-evaluation) take GPU_EVALUATION's defaults instead: stop at 85 C, continue at 65 C, 4 threads (a probe
burst on the iGPU heats the CPU dies 4-8 s after the job is already stopped; with these, ~0.3% of their samples
reached 95 C, peaks 97.9 C, and every run completed, some after a resume). KVPARK_GUARD_* and --guard still
override them.

Settings (Settings.from_env): KVPARK_GUARD=off turns waiting, pausing, aborting and fit_hub.py's
memory refusal off with a loud warning (samples are still logged; the CPU lock and the thread caps
stay); KVPARK_GUARD_START_C, _PAUSE_C, _RESUME_C, _ABORT_C, _SWAP_PAUSE and _SWAP_RESUME (fractions of
swap), _SWAP_IGNORE_GIB (0: swap always counts), _ABORT_SAMPLES, _MEM_FLOOR_GIB, _MEM_RESUME_GIB, _FIT_MARGIN_GIB (fit_hub.py: what stays free beyond a fit's
estimated peak), _POLL_S, _SENSOR_LOSS_S, _STOP_GRACE_S, _MAX_PAUSE_S and _MAX_WAIT_S (0: no limit;
wait_cool waits without one by default), _THREADS, _GPU_WATCH (auto, journal, kmsg or off: where the
GPU-hang stop reads the kernel log; with the guard off a hang is logged, never acted on);
KVPARK_HWMON_ROOT, KVPARK_GUARD_MEMINFO,
KVPARK_GUARD_CPU_LOCK and KVPARK_GUARD_GPU_LOCK point elsewhere (tests). Command lines that use the guard take --guard KEY=VALUE
with the same keys in lower case (pause_c=88, threads=4, off).
"""

import argparse
import contextlib
import dataclasses
import datetime
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
import types

GIB = 1024 ** 3
CPU_LOCK = "/tmp/kvpark-campaign-cpu.lock"
GPU_LOCK = "/tmp/kvpark-campaign-gpu.lock"
DEFAULT_THREADS = 8
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")
WATCHED = "KVPARK_GUARD_WATCHED"        # in a watched child's environment: the outer Watch pauses it
CPU_LOCK_FD = "KVPARK_CPU_LOCK_FD"      # the inherited descriptor that holds the CPU lock
ABORT_EXIT = 75                         # `resource_guard.py run`'s exit status when the guard ended the command
GPU_WATCH_SOURCES = ("auto", "journal", "kmsg", "off")
# the defaults of GPU evaluations (pair_eval's probe runs, campaign.py's evaluate stage and assemble --evaluate, exec
# --gpu-evaluation), where KVPARK_GUARD_* do not set them: a 2 s burst of the probe at ~125 W is followed by a Tctl
# peak of 97-98 C 4-8 s later, while the job is already stopped at 90 C (2026-10-06/07). Stopping at 85 C, resuming at
# 65 C and 4 BLAS/probe threads: 69 of the 24,487 samples of the twelve 2026-10-07 pair_eval runs (0.28%) at or above
# 95 C, peaks up to 97.9 C, 7 guard ends, every run completed (resumed); at the defaults they ended again and again.
GPU_EVALUATION = {"pause_c": 85.0, "resume_c": 65.0, "threads": 4}


class GuardError(RuntimeError):
    """The guard refused to start or continue a job (the message says why)."""


class GuardAbort(GuardError):
    """A watched job had to end: the package reached abort_c, the sensors were lost or a pause lasted too long."""


class LockOrderError(RuntimeError):
    """The GPU/memory lock was asked for while this process holds the CPU lock."""


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# -- settings -----------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Settings:
    enabled: bool = True
    start_c: float = 75.0           # wait_cool: the package at most this before a heavy step starts
    pause_c: float = 90.0           # stop the job at or above
    resume_c: float = 80.0          # continue at or below (and memory recovered)
    abort_c: float = 95.0           # end the job at or above ...
    abort_samples: int = 2          # ... on this many samples in a row (stopped at the first)
    swap_pause: float = 0.85        # stop when this fraction of swap is used
    swap_resume: float = 0.80       # continue below
    mem_floor_gib: float = 8.0      # stop when MemAvailable falls below
    mem_resume_gib: float = 12.0    # continue at or above
    swap_ignore_gib: float = 24.0   # swap use neither stops nor holds a job while MemAvailable is at least this (0: always)
    fit_margin_gib: float = 25.0    # fit_hub.py refuses a fit whose estimated peak leaves less MemAvailable
    poll_s: float = 2.0
    sensor_loss_s: float = 60.0
    stop_grace_s: float = 30.0
    max_pause_s: float = 1800.0     # 0: pause without limit
    max_wait_s: float = 0.0         # wait_cool: 0 waits without limit
    threads: int = DEFAULT_THREADS
    hwmon_root: str = "/sys/class/hwmon"
    meminfo: str = "/proc/meminfo"
    cpu_lock: str = CPU_LOCK
    gpu_lock: str = GPU_LOCK        # gpu_lock(): the campaign's GPU/memory lock file
    gpu_watch: str = "auto"         # the kernel log for GPU hangs: auto, journal, kmsg or off (KernelLog)

    def __post_init__(self):
        if self.gpu_watch not in GPU_WATCH_SOURCES:
            raise ValueError(f"guard: gpu_watch must be one of {', '.join(GPU_WATCH_SOURCES)} ({self.gpu_watch!r})")
        if not self.resume_c < self.pause_c < self.abort_c:
            raise ValueError(f"guard: resume_c < pause_c < abort_c needed ({self.resume_c}, {self.pause_c}, "
                             f"{self.abort_c})")
        if self.start_c > self.pause_c:
            raise ValueError(f"guard: start_c ({self.start_c}) above pause_c ({self.pause_c})")
        if not 0 < self.swap_resume <= self.swap_pause <= 1:
            raise ValueError(f"guard: 0 < swap_resume <= swap_pause <= 1 needed "
                             f"({self.swap_resume}, {self.swap_pause})")
        if self.swap_ignore_gib and self.swap_ignore_gib < self.mem_resume_gib:
            raise ValueError(f"guard: swap_ignore_gib ({self.swap_ignore_gib}) below mem_resume_gib "
                             f"({self.mem_resume_gib}) (0 turns it off)")
        if self.mem_resume_gib < self.mem_floor_gib:
            raise ValueError(f"guard: mem_resume_gib ({self.mem_resume_gib}) below mem_floor_gib "
                             f"({self.mem_floor_gib})")
        if self.poll_s <= 0 or self.threads < 1 or self.sensor_loss_s <= 0 or self.abort_samples < 1:
            raise ValueError("guard: poll_s and sensor_loss_s must be positive, threads and abort_samples at least 1")

    @property
    def mem_floor(self):
        return int(self.mem_floor_gib * GIB)

    @property
    def mem_resume(self):
        return int(self.mem_resume_gib * GIB)

    @property
    def fit_margin(self):
        return int(self.fit_margin_gib * GIB)

    def as_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_env(cls, environ=None, overrides=(), defaults=None):
        """Settings from `defaults` (a dict of field values, e.g. GPU_EVALUATION; else the class's), then
        KVPARK_GUARD_* (and KVPARK_HWMON_ROOT), then `overrides`: "KEY=VALUE" strings (--guard) or a dict, keys
        the field names; "off" / "on" alone switch the guard."""
        environ = os.environ if environ is None else environ
        values = dict(defaults or {})
        for field in dataclasses.fields(cls):
            if field.name == "enabled":
                text = environ.get("KVPARK_GUARD")
                if text is not None:
                    values["enabled"] = _switch(text, "KVPARK_GUARD")
                continue
            name = {"hwmon_root": "KVPARK_HWMON_ROOT"}.get(field.name, "KVPARK_GUARD_" + field.name.upper())
            if name in environ and environ[name] != "":
                values[field.name] = _parse(field, environ[name], name)
        items = overrides.items() if isinstance(overrides, dict) else [_split(item) for item in overrides or ()]
        names = {field.name: field for field in dataclasses.fields(cls)}
        for key, value in items:
            key = key.strip().lower()
            if key in ("off", "on") and value is None:
                values["enabled"] = key == "on"
                continue
            if key not in names:
                raise ValueError(f"--guard: unknown setting {key!r} (one of off, on, {', '.join(names)})")
            if value is None:
                raise ValueError(f"--guard: {key} needs a value ({key}=VALUE)")
            values[key] = value if not isinstance(value, str) else (
                _switch(value, key) if key == "enabled" else _parse(names[key], value, key))
        return cls(**values)

    @classmethod
    def from_args(cls, args, environ=None, defaults=None):
        return cls.from_env(environ, getattr(args, "guard", None) or (), defaults=defaults)


def _split(item):
    key, sep, value = str(item).partition("=")
    return key, (value if sep else None)


def _switch(text, name):
    lowered = str(text).strip().lower()
    if lowered in ("0", "off", "false", "no"):
        return False
    if lowered in ("1", "on", "true", "yes"):
        return True
    raise ValueError(f"{name}: {text!r} is not on or off")


def _parse(field, text, name):
    try:
        if field.type in (int, "int"):
            return int(text)
        if field.type in (float, "float"):
            return float(text)
    except ValueError:
        raise ValueError(f"{name}: {text!r} is not a number") from None
    return str(text)


def add_arguments(parser):
    """--guard KEY=VALUE (repeatable): resource guard settings for this run (Settings.from_args)."""
    parser.add_argument("--guard", action="append", default=[], metavar="KEY=VALUE",
                        help="resource guard setting (resource_guard.Settings: pause_c=90, abort_c=95, threads=8, "
                             "mem_floor_gib=8, ...; 'off' disables it with a warning); repeatable")


def warn_off(where, stream=None):
    """The loud warning of a run with the guard off."""
    line = "!" * 78
    print(f"{line}\n!! WARNING: the resource guard is OFF for {where} (KVPARK_GUARD=off or --guard off):\n"
          f"!! no cooling waits, no pauses at high temperature or memory pressure, no aborts.\n"
          f"!! On 2026-10-05 an unguarded CPU job tripped this machine's thermal shutdown.\n{line}",
          file=stream or sys.stderr, flush=True)


# -- threads ------------------------------------------------------------------------------------

def thread_env(environ=None, threads=None):
    """A copy of `environ` (os.environ) with the BLAS, OpenMP and MKL thread counts set where they are not:
    KVPARK_GUARD_THREADS, else 8. Given `threads` (a run's Settings.threads, e.g. GPU_EVALUATION's 4), they are
    capped at it: set to it where unset, lowered to it where larger (a parent that capped its own at 8 before
    importing numpy hands its children no more than `threads`)."""
    environ = dict(os.environ if environ is None else environ)
    explicit = threads is not None
    threads = threads or _env_threads(environ)
    for name in THREAD_VARIABLES:
        current = environ.get(name)
        if not current:
            environ[name] = str(threads)
        elif explicit and current.isdigit() and int(current) > threads:
            environ[name] = str(threads)
    return environ


def cap_threads(environ=None, threads=None):
    """thread_env in place (os.environ by default): call it before numpy is imported."""
    environ = os.environ if environ is None else environ
    environ.update({k: v for k, v in thread_env(environ, threads).items() if k in THREAD_VARIABLES})
    return environ


def _env_threads(environ):
    text = environ.get("KVPARK_GUARD_THREADS")
    return int(text) if text else DEFAULT_THREADS


# -- sensors ------------------------------------------------------------------------------------

def _read(path):
    try:
        with open(path, encoding="ascii", errors="replace") as stream:
            return stream.read().strip()
    except OSError:
        return None


def _number(path, scale):
    text = _read(path)
    try:
        return None if text is None else int(text) / scale
    except ValueError:
        return None


def read_temperatures(root="/sys/class/hwmon"):
    """{"cpu": {label: C} (k10temp Tctl, Tdie, Tccd*), "edge", "junction" (amdgpu, C), "ppt_w" (amdgpu
    PPT, the APU's package power), "nvme": [composite C]}; what cannot be read is missing or None."""
    out = {"cpu": {}, "edge": None, "junction": None, "ppt_w": None, "nvme": []}
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return out
    for entry in entries:
        path = os.path.join(root, entry)
        name = _read(os.path.join(path, "name"))
        if name not in ("k10temp", "amdgpu", "nvme"):
            continue
        try:
            files = os.listdir(path)
        except OSError:
            continue
        for item in sorted(files):
            if not (item.startswith("temp") and item.endswith("_input")):
                continue
            stem = item[:-len("_input")]
            value = _number(os.path.join(path, item), 1000)
            if value is None:
                continue
            label = _read(os.path.join(path, stem + "_label"))
            if name == "k10temp":
                label = label or ("Tctl" if stem == "temp1" else stem)
                if label in ("Tctl", "Tdie") or label.startswith("Tccd"):
                    out["cpu"][label] = value
            elif name == "amdgpu" and label in ("edge", "junction"):
                out[label] = value
            elif name == "nvme" and stem == "temp1":
                out["nvme"].append(value)
        if name == "amdgpu":
            for item in ("power1_average", "power1_input"):
                watts = _number(os.path.join(path, item), 1e6)
                if watts is not None:
                    out["ppt_w"] = watts
                    break
    return out


def read_memory(meminfo="/proc/meminfo"):
    """{"mem_total", "mem_available", "swap_total", "swap_free"} in bytes (None when unreadable)."""
    keys = {"MemTotal:": "mem_total", "MemAvailable:": "mem_available", "SwapTotal:": "swap_total",
            "SwapFree:": "swap_free"}
    out = dict.fromkeys(keys.values())
    text = _read(meminfo)
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in keys:
            try:
                out[keys[parts[0]]] = int(parts[1]) * 1024
            except ValueError:
                pass
    return out


def read_sample(settings=None):
    """One reading: temperatures, package_c (the hottest CPU sensor or the iGPU edge; None when no CPU
    sensor can be read), memory, swap use and the 1-minute load."""
    settings = settings or Settings.from_env()
    temperatures = read_temperatures(settings.hwmon_root)
    memory = read_memory(settings.meminfo)
    cpu = temperatures["cpu"]
    package = max(list(cpu.values()) + ([temperatures["edge"]] if temperatures["edge"] is not None else [])) \
        if cpu else None
    swap_total, swap_free = memory["swap_total"], memory["swap_free"]
    swap_used = None if swap_total is None or swap_free is None else swap_total - swap_free
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = None
    return {"package_c": package, "cpu_c": cpu, "edge_c": temperatures["edge"],
            "junction_c": temperatures["junction"], "ppt_w": temperatures["ppt_w"], "nvme_c": temperatures["nvme"],
            "mem_total": memory["mem_total"], "mem_available": memory["mem_available"], "swap_total": swap_total,
            "swap_used": swap_used,
            "swap_fraction": (swap_used / swap_total) if swap_used is not None and swap_total else 0.0,
            "load1": load}


def group_memory(pgid, proc="/proc"):
    """{"rss", "anon", "processes"} of every process of a process group (bytes, from /proc)."""
    rss = anon = count = 0
    try:
        entries = list(os.scandir(proc))
    except OSError:
        entries = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            with open(os.path.join(entry.path, "stat"), "rb") as stream:
                stat = stream.read()
            if int(stat[stat.rindex(b")") + 2:].split()[2]) != pgid:
                continue
            with open(os.path.join(entry.path, "status"), "rb") as stream:
                for line in stream:
                    if line.startswith(b"VmRSS:"):
                        rss += int(line.split()[1]) * 1024
                    elif line.startswith(b"RssAnon:"):
                        anon += int(line.split()[1]) * 1024
            count += 1
        except (OSError, ValueError, IndexError):
            continue
    return {"rss": rss, "anon": anon, "processes": count}


def own_memory():
    """{"rss", "anon"} of this process."""
    out = {"rss": 0, "anon": 0}
    for line in (_read("/proc/self/status") or "").splitlines():
        if line.startswith("VmRSS:"):
            out["rss"] = int(line.split()[1]) * 1024
        elif line.startswith("RssAnon:"):
            out["anon"] = int(line.split()[1]) * 1024
    return out


def swap_counts(sample, settings):
    """Whether swap use counts for the memory decisions: not while MemAvailable is at least swap_ignore_gib.
    Swap that fills while memory is plentiful holds pages swapped out earlier, mostly a stopped job's own
    (on 2026-10-07 a stopped joint refit's 18 GiB and a stopped pair_eval's 12 GiB filled swap at 40-50 GiB
    MemAvailable): they come back only when the job runs, so holding it until swap use falls never ends."""
    available = sample.get("mem_available")
    return not (settings.swap_ignore_gib and available is not None and available >= settings.swap_ignore_gib * GIB)


def memory_trouble(sample, settings):
    """Why memory is short (a reason) or None: MemAvailable below the floor, or swap use at swap_pause while
    it counts (swap_counts)."""
    available = sample.get("mem_available")
    if available is not None and available < settings.mem_floor:
        return f"MemAvailable {available / GIB:.1f} GiB < floor {settings.mem_floor_gib:g} GiB"
    if sample.get("swap_total") and sample.get("swap_fraction", 0.0) >= settings.swap_pause \
            and swap_counts(sample, settings):
        return f"swap {100 * sample['swap_fraction']:.0f}% used >= {100 * settings.swap_pause:.0f}%"
    return None


def memory_recovered(sample, settings):
    available = sample.get("mem_available")
    if available is not None and available < settings.mem_resume:
        return False
    return (not sample.get("swap_total") or sample.get("swap_fraction", 0.0) < settings.swap_resume
            or not swap_counts(sample, settings))


def assess(sample, settings, paused):
    """(action, reason) for one sample: "lost" (no CPU temperature), "abort", "pause" (a running job
    to stop), "run" (keep running), "resume" (a paused job may continue) or "hold" (stay paused)."""
    package = sample.get("package_c")
    if package is None:
        return "lost", "no CPU temperature readable"
    if package >= settings.abort_c:
        return "abort", f"package {package:.1f} C >= abort {settings.abort_c:g} C"
    if not paused:
        if package >= settings.pause_c:
            return "pause", f"package {package:.1f} C >= pause {settings.pause_c:g} C"
        trouble = memory_trouble(sample, settings)
        return ("pause", trouble) if trouble else ("run", None)
    if package <= settings.resume_c and memory_recovered(sample, settings):
        return "resume", None
    return "hold", None


def _gib(value):
    return None if value is None else round(value / GIB, 2)


def sample_record(sample, **extra):
    """The JSONL form of a sample (rounded, GiB)."""
    item = {"event": "sample", "package_c": sample.get("package_c"), "cpu_c": sample.get("cpu_c"),
            "edge_c": sample.get("edge_c"), "ppt_w": None if sample.get("ppt_w") is None else round(sample["ppt_w"], 1),
            "mem_available_gib": _gib(sample.get("mem_available")), "swap_used_gib": _gib(sample.get("swap_used")),
            "swap_pct": round(100 * sample.get("swap_fraction", 0.0), 1),
            "load1": None if sample.get("load1") is None else round(sample["load1"], 2)}
    if sample.get("junction_c") is not None:
        item["junction_c"] = sample["junction_c"]
    if sample.get("nvme_c"):
        item["nvme_c"] = sample["nvme_c"]
    item.update({key: value for key, value in extra.items() if value is not None})
    return item


class JsonLog:
    """Appends guard events and samples to a JSONL file (when given); events other than samples also
    go to `events` (a callable taking the dict, e.g. a campaign's log)."""

    def __init__(self, path=None, label=None, events=None):
        self.path = None if path is None else Path(path)
        self.label, self.events = label, events
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, item):
        item = {"time": now(), **({"label": self.label} if self.label else {}), **item}
        if self.path is not None:
            with self._lock, open(self.path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(item) + "\n")
        if self.events is not None and item.get("event") != "sample":
            self.events(item)


# -- waiting until cool ---------------------------------------------------------------------------

def wait_cool(settings=None, *, start_c=None, log=None, sample=None, clock=time.monotonic, sleep=time.sleep,
              log_every_s=60.0):
    """Before a heavy step: wait until the package is at most start_c (settings.start_c, 75 C) and
    memory is not short (MemAvailable at least mem_resume, swap use below swap_resume). Logs the wait
    (every log_every_s) and returns its seconds; GuardError when no CPU temperature can be read for
    sensor_loss_s or the wait passes max_wait_s (when set). With the guard off: a warning, no wait."""
    settings = settings if settings is not None else Settings.from_env()
    log = log if log is not None else JsonLog()
    if not settings.enabled:
        log.write({"event": "guard-off", "where": "wait_cool"})
        return 0.0
    start_c = settings.start_c if start_c is None else start_c
    sample = sample or (lambda: read_sample(settings))
    started = clock()
    lost_since, logged = None, None
    while True:
        current = sample()
        elapsed = clock() - started
        package = current.get("package_c")
        if package is None:
            lost_since = clock() if lost_since is None else lost_since
            if clock() - lost_since >= settings.sensor_loss_s:
                log.write({"event": "abort", "reason": "no CPU temperature readable", "where": "wait_cool"})
                raise GuardError(f"guard: no CPU temperature readable under {settings.hwmon_root} for "
                                 f"{settings.sensor_loss_s:g} s")
        else:
            lost_since = None
            if package <= start_c and memory_recovered(current, settings):
                if elapsed >= settings.poll_s:
                    log.write(sample_record(current, event="cool", waited_s=round(elapsed, 1), start_c=start_c))
                return elapsed
        if settings.max_wait_s and elapsed >= settings.max_wait_s:
            log.write({"event": "abort", "reason": "still hot or short of memory", "where": "wait_cool",
                       "waited_s": round(elapsed, 1)})
            raise GuardError(f"guard: the package did not cool to {start_c:g} C (or memory did not recover) within "
                             f"{settings.max_wait_s:g} s (now {package} C)")
        if logged is None or clock() - logged >= log_every_s:
            logged = clock()
            log.write(sample_record(current, event="cool-wait", waited_s=round(elapsed, 1), start_c=start_c,
                                    why=memory_trouble(current, settings) or ("sensors" if package is None else "hot")))
        sleep(settings.poll_s)


# -- GPU hangs in the kernel log ------------------------------------------------------------------

# what amdgpu logs when a job starves or hangs the GPU (lines of 2026-10-06 09:53):
#   amdgpu 0000:c6:00.0: ring gfx_0.0.0 timeout, signaled seq=6167215, emitted seq=6167217
#   amdgpu 0000:c6:00.0: Starting gfx_0.0.0 ring reset / Ring gfx_0.0.0 reset succeeded
#   amdgpu 0000:c6:00.0: [drm] device wedged, but recovered through reset
#   amdgpu 0000:c6:00.0: GPU reset begin!. Source:  1
GPU_HANG = re.compile(r"\bring \S+ timeout|\bring reset|\bring \S+ reset|\bGPU reset|device wedged|\bGPU hang|"
                      r"MES failed to respond", re.IGNORECASE)
GPU_DRIVERS = ("amdgpu", "[drm]", "drm:")


def gpu_hang_line(line):
    """Whether a kernel log line reports a GPU ring timeout, reset or hang (GPU_HANG, on a GPU driver line)."""
    return any(driver in line for driver in GPU_DRIVERS) and GPU_HANG.search(line) is not None


def _line_epoch(line):
    """The epoch of a `journalctl -o short-unix` line (its first field), or None."""
    head = line.split(None, 1)[0] if line.strip() else ""
    try:
        return float(head)
    except ValueError:
        return None


def _die_with_parent():
    """preexec_fn: the follower gets SIGTERM when this process dies (Linux PR_SET_PDEATHSIG)."""
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)       # PR_SET_PDEATHSIG = 1
    except Exception:
        pass


def journal_readable(run=subprocess.run):
    """Whether `journalctl -k` shows this user kernel messages (the adm or systemd-journal group)."""
    if shutil.which("journalctl") is None:
        return False
    try:
        result = run(["journalctl", "-k", "-n", "1", "-o", "short-unix", "--no-pager", "-q"], capture_output=True,
                     text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


class KernelLog:
    """The kernel log from `since` (epoch seconds, default now) on, scanned for GPU hangs (gpu_hang_line)
    on a reader thread. source: "journal" (`journalctl -k -f -o short-unix --since @since`: its lines
    before `since` are skipped), "kmsg" (/dev/kmsg from its end), "auto" (the journal when readable,
    else /dev/kmsg when readable) or an iterable of lines (tests: a fake log; it may block for more).
    start() returns self; poll() returns the hits not returned before ({"line", "epoch"}); hits holds them
    all; `source` says what is read ("unavailable" with `error` when nothing could be); close() ends the
    follower."""

    def __init__(self, since=None, *, source="auto", clock=time.time):
        self.since = clock() if since is None else since
        self._source, self._clock = source, clock
        self.source, self.error = None, None
        self.hits, self._returned = [], 0
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._process, self._fd, self._thread = None, None, None

    def start(self):
        lines = None
        if not isinstance(self._source, str):
            self.source, lines = "lines", iter(self._source)
        else:
            wanted = self._source
            if wanted in ("auto", "journal") and journal_readable():
                wanted = "journal"
            elif wanted in ("auto", "kmsg") and os.access("/dev/kmsg", os.R_OK):
                wanted = "kmsg"
            else:
                self.source, self.error = "unavailable", (
                    f"cannot read the kernel log ({self._source}): journalctl -k shows nothing to this user and "
                    "/dev/kmsg is not readable")
                return self
            self.source = wanted
            try:
                lines = self._journal() if wanted == "journal" else self._kmsg()
            except OSError as error:
                self.source, self.error = "unavailable", f"cannot read the kernel log ({wanted}): {error}"
                return self
        self._thread = threading.Thread(target=self._read, args=(lines,), name="kernel-log", daemon=True)
        self._thread.start()
        return self

    def _journal(self):
        self._process = subprocess.Popen(
            ["journalctl", "-k", "-f", "-o", "short-unix", "--no-pager", "-q", "--since", f"@{int(self.since)}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, errors="replace", bufsize=1,
            preexec_fn=_die_with_parent)
        return self._process.stdout

    def _kmsg(self):
        self._fd = os.open("/dev/kmsg", os.O_RDONLY | os.O_NONBLOCK)
        os.lseek(self._fd, 0, os.SEEK_END)                      # only what is logged from now on

        def records():
            while not self._stop.is_set():
                ready, _, _ = select.select([self._fd], [], [], 0.5)
                if not ready:
                    continue
                try:
                    record = os.read(self._fd, 8192)
                except OSError as error:
                    if error.errno in (errno.EAGAIN, errno.EPIPE):     # none yet / overwritten: go on
                        continue
                    raise
                yield record.decode("utf-8", "replace").partition(";")[2]
        return records()

    def _read(self, lines):
        try:
            for line in lines:
                if self._stop.is_set():
                    break
                line = line.rstrip("\n")
                epoch = _line_epoch(line) if self.source != "kmsg" else None      # short-unix lines
                if epoch is not None and epoch < self.since:
                    continue
                if gpu_hang_line(line):
                    with self._lock:
                        self.hits.append({"line": line.strip(), "epoch": epoch if epoch is not None else self._clock()})
        except (OSError, ValueError) as error:
            if not self._stop.is_set():
                self.error = f"kernel log reader stopped: {error}"

    def poll(self):
        """The hits found since the last poll."""
        with self._lock:
            out = self.hits[self._returned:]
            self._returned = len(self.hits)
        return out

    def describe(self):
        return {"source": self.source, **({"error": self.error} if self.error else {}),
                "since": round(self.since, 3)}

    def close(self):
        self._stop.set()
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process.stdout.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def open_kernel_log(settings, since=None, where=None):
    """A started KernelLog for settings.gpu_watch, or None when it is off; a warning on stderr when the
    kernel log cannot be read (the job then runs without the GPU-hang stop)."""
    if settings.gpu_watch == "off":
        return None
    log = KernelLog(since, source=settings.gpu_watch).start()
    if log.source == "unavailable":
        print(f"resource guard: WARNING: no GPU-hang stop for {where or 'this job'}: {log.error}", file=sys.stderr,
              flush=True)
    return log


# -- watching a child process group ---------------------------------------------------------------

def signal_group(pgid, number, kill=os.killpg):
    try:
        kill(pgid, number)
        return True
    except ProcessLookupError:
        return False


def stop_group(child, grace=30.0, kill=os.killpg):
    """End a Popen child and its process group: SIGTERM and SIGCONT (a stopped group handles neither
    until continued), then SIGKILL to whatever is left after `grace` seconds; the child is reaped."""
    signal_group(child.pid, signal.SIGTERM, kill)
    signal_group(child.pid, signal.SIGCONT, kill)
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    signal_group(child.pid, signal.SIGKILL, kill)
    child.wait()


class Watch:
    """Watches a child process group (attach(pgid), then step() as often as convenient: it samples
    every poll_s and reads the kernel log for GPU hangs on every step). See the module docstring for
    what it does; GuardAbort leaves ending the group to the caller (stop_group). paused_seconds() is
    the time the group spent stopped. kernel_log: a KernelLog (not started) to read instead of the one
    settings.gpu_watch opens at attach()."""

    def __init__(self, settings=None, *, log_path=None, label=None, events=None, sample=None, clock=time.monotonic,
                 kill=os.killpg, proc="/proc", kernel_log=None):
        self.settings = settings if settings is not None else Settings.from_env()
        self.log = JsonLog(log_path, label, events)
        self._sample = sample or (lambda: read_sample(self.settings))
        self._clock, self._kill, self._proc = clock, kill, proc
        self.pgid = None
        self.paused_since, self.reason = None, None
        self.paused_total, self.pauses, self.samples = 0.0, 0, 0
        self.max_c, self.min_available, self.max_rss, self.max_anon = None, None, 0, 0
        self.lost_since, self.aborted, self._last = None, None, None
        self.critical = 0               # samples in a row at abort_c
        self.kernel, self._kernel_given = kernel_log, kernel_log is not None
        self.gpu_hang = None            # the first GPU-hang line of the kernel log

    def attach(self, pgid):
        self.pgid = pgid
        self.log.write({"event": "watch", "pgid": pgid, "settings": self.settings.as_dict()})
        if not self.settings.enabled:
            warn_off(self.log.label or f"process group {pgid}")
            self.log.write({"event": "guard-off", "where": "watch"})
        self.kernel = (self.kernel.start() if self._kernel_given
                       else open_kernel_log(self.settings, where=self.log.label or f"process group {pgid}"))
        if self.kernel is not None:
            self.log.write({"event": "gpu-watch", **self.kernel.describe()})
        self.step(force=True)
        return self

    def _check_kernel(self):
        """GuardAbort at the first GPU-hang line in the kernel log (logged as gpu-hang either way)."""
        hits = self.kernel.poll() if self.kernel is not None else []
        if not hits or self.gpu_hang is not None:
            return
        self.gpu_hang = hits[0]["line"]
        self.log.write({"event": "gpu-hang", "line": self.gpu_hang, "kernel_epoch": hits[0]["epoch"],
                        "lines": len(hits), "acted": self.settings.enabled})
        if self.settings.enabled:
            self._abort(f"GPU hang in the kernel log: {self.gpu_hang}", self._sample(), gpu_hang=self.gpu_hang)

    def paused_seconds(self):
        current = 0.0 if self.paused_since is None else self._clock() - self.paused_since
        return self.paused_total + current

    def step(self, force=False):
        """Read the kernel log (GuardAbort at a GPU hang); sample when poll_s has passed (or force): log
        it, then stop, continue or abort the group."""
        self._check_kernel()
        moment = self._clock()
        if not force and self._last is not None and moment - self._last < self.settings.poll_s:
            return
        self._last = moment
        current = self._sample()
        memory = group_memory(self.pgid, self._proc) if self.pgid is not None else None
        self.samples += 1
        if current.get("package_c") is not None:
            self.max_c = max(self.max_c or current["package_c"], current["package_c"])
        if current.get("mem_available") is not None:
            self.min_available = min(self.min_available or current["mem_available"], current["mem_available"])
        if memory:
            self.max_rss, self.max_anon = max(self.max_rss, memory["rss"]), max(self.max_anon, memory["anon"])
        self.log.write(sample_record(current, state="paused" if self.paused_since is not None else "running",
                                     child_rss_gib=_gib(memory["rss"]) if memory else None,
                                     child_anon_gib=_gib(memory["anon"]) if memory else None))
        self._act(current, moment)

    def _act(self, current, moment):
        action, reason = assess(current, self.settings, self.paused_since is not None)
        if action == "lost":
            self.lost_since = moment if self.lost_since is None else self.lost_since
            if self.settings.enabled and moment - self.lost_since >= self.settings.sensor_loss_s:
                self._abort(f"no CPU temperature readable for {moment - self.lost_since:.0f} s "
                            f"({self.settings.hwmon_root})", current)
            return
        self.lost_since = None
        if not self.settings.enabled:
            return
        if action == "abort":
            self.critical += 1
            if self.critical >= self.settings.abort_samples:
                self._abort(reason, current)
            if self.paused_since is None:
                signal_group(self.pgid, signal.SIGSTOP, self._kill)
                self.paused_since, self.reason = moment, reason
                self.pauses += 1
                self.log.write(sample_record(current, event="pause", reason=reason, critical=self.critical))
            return
        self.critical = 0
        if action == "pause":
            signal_group(self.pgid, signal.SIGSTOP, self._kill)
            self.paused_since, self.reason = moment, reason
            self.pauses += 1
            self.log.write(sample_record(current, event="pause", reason=reason))
        elif action == "resume":
            signal_group(self.pgid, signal.SIGCONT, self._kill)
            paused = moment - self.paused_since
            self.paused_total += paused
            self.log.write(sample_record(current, event="resume", reason=self.reason, paused_s=round(paused, 1)))
            self.paused_since, self.reason = None, None
        elif action == "hold" and self.settings.max_pause_s and moment - self.paused_since >= self.settings.max_pause_s:
            self._abort(f"paused for {moment - self.paused_since:.0f} s and {self.reason} has not cleared", current)

    def _abort(self, reason, current, **extra):
        self.aborted = reason
        self.log.write(sample_record(current, event="abort", reason=reason, **extra))
        raise GuardAbort(f"guard: {reason}")

    def summary(self):
        return {"samples": self.samples, "max_c": self.max_c, "paused_s": round(self.paused_seconds(), 1),
                "pauses": self.pauses, "min_available_gib": _gib(self.min_available), "max_rss_gib": _gib(self.max_rss),
                "max_anon_gib": _gib(self.max_anon), "aborted": self.aborted, "gpu_hang": self.gpu_hang,
                "gpu_watch": None if self.kernel is None else self.kernel.source,
                "log": None if self.log.path is None else str(self.log.path)}

    def close(self):
        """Log the summary (once the group has ended); a pause still open is counted. A GPU hang logged
        after the last step is still read (and logged, not acted on: the group has ended)."""
        if self.paused_since is not None:
            self.paused_total += self._clock() - self.paused_since
            self.paused_since = None
        if self.kernel is not None:
            hits = self.kernel.poll()
            if hits and self.gpu_hang is None:
                self.gpu_hang = hits[0]["line"]
                self.log.write({"event": "gpu-hang", "line": self.gpu_hang, "kernel_epoch": hits[0]["epoch"],
                                "lines": len(hits), "acted": False, "after": "the group ended"})
            self.kernel.close()
        item = self.summary()
        self.log.write({"event": "watch-done", **item})
        return item


# -- the guard for loops in this process ------------------------------------------------------------

def _end_process(reason, grace):
    """What Guard's background thread does at abort_c when no check() handles it: SIGTERM to this
    process (Python's default disposition ends it at once), SIGKILL after `grace` seconds."""
    print(f"resource guard: {reason}: ending process {os.getpid()}", file=sys.stderr, flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(grace)
    os.kill(os.getpid(), signal.SIGKILL)


class Guard:
    """The guard for loops in this process (a fit, a diagnostic): use it as a context manager and call
    check() between chunks. check() samples at most every poll_s; while the package is at pause_c or
    memory is short it sleeps (logging) until resume_c and memory recovered, and raises GuardAbort at
    abort_c, after sensor_loss_s without a CPU temperature or after max_pause_s of pausing. A background
    thread logs a sample (with this process's RSS) every poll_s meanwhile and, at abort_c while no
    check() runs, calls on_critical(reason) (default: end this process). Under an outer Watch
    (KVPARK_GUARD_WATCHED=1), which pauses this whole process and logs, check() only raises at abort_c
    and there is no background thread.

    gpu=True (a fit that runs GPU kernels in this process): check() also reads the kernel log
    (KernelLog, settings.gpu_watch; or kernel_log, not started) and raises GuardAbort at the first GPU
    hang, and the background thread calls on_critical when no check() has seen one within two poll_s
    (a GPU call that does not return). Under an outer Watch, which reads the kernel log itself and ends
    the whole group, the Guard reads it only when given kernel_log."""

    def __init__(self, settings=None, *, log_path=None, label=None, events=None, sample=None, clock=time.monotonic,
                 sleep=time.sleep, background=True, on_critical=None, environ=None, gpu=False, kernel_log=None):
        self.settings = settings if settings is not None else Settings.from_env()
        self.log = JsonLog(log_path, label, events)
        self._sample = sample or (lambda: read_sample(self.settings))
        self._clock, self._sleep = clock, sleep
        self.watched = (os.environ if environ is None else environ).get(WATCHED) == "1"
        self._background = background and not self.watched
        self._on_critical = on_critical or (lambda reason: _end_process(reason, self.settings.stop_grace_s))
        self._next = None
        self._in_check = False
        self._lost_since = None
        self._stop = threading.Event()
        self._thread = None
        self.paused_total, self.pauses, self.max_c, self.critical = 0.0, 0, None, None
        self._critical = 0              # check() samples in a row at abort_c
        self.gpu, self.kernel, self._kernel_given = gpu, kernel_log, kernel_log is not None
        self.gpu_hang, self._hang_seen = None, None     # the first GPU-hang line; when the background saw it

    def __enter__(self):
        self.log.write({"event": "guard", "pid": os.getpid(), "watched": self.watched, "gpu": self.gpu,
                        "settings": self.settings.as_dict()})
        if not self.settings.enabled:
            warn_off(self.log.label or f"process {os.getpid()}")
            self.log.write({"event": "guard-off", "where": "guard"})
        if self._kernel_given:
            self.kernel = self.kernel.start()
        elif self.gpu and not self.watched:
            self.kernel = open_kernel_log(self.settings, where=self.log.label or f"process {os.getpid()}")
        if self.kernel is not None:
            self.log.write({"event": "gpu-watch", **self.kernel.describe()})
        if self._background:
            self._thread = threading.Thread(target=self._run_background, name="resource-guard", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5 * self.settings.poll_s)
        if self.kernel is not None:
            self._gpu_check("exit", act=False)
            self.kernel.close()
        self.log.write({"event": "guard-done", "paused_s": round(self.paused_total, 1), "pauses": self.pauses,
                        "max_c": self.max_c, "gpu_hang": self.gpu_hang,
                        **{f"{k}_gib": _gib(v) for k, v in own_memory().items()}})
        return False

    def _gpu_check(self, where, act=True):
        """GuardAbort at the first GPU-hang line in the kernel log (logged as gpu-hang either way)."""
        hits = self.kernel.poll() if self.kernel is not None else []
        if not hits or self.gpu_hang is not None:
            return
        self.gpu_hang = hits[0]["line"]
        acted = act and self.settings.enabled
        self.log.write({"event": "gpu-hang", "line": self.gpu_hang, "kernel_epoch": hits[0]["epoch"],
                        "lines": len(hits), "where": where, "acted": acted})
        if acted:
            current = self._sample()
            self.log.write(sample_record(current, event="abort", reason="GPU hang in the kernel log", where=where,
                                         gpu_hang=self.gpu_hang))
            raise GuardAbort(f"guard: GPU hang in the kernel log: {self.gpu_hang}")

    def wait_cool(self, start_c=None):
        """wait_cool before the job (not under an outer Watch, which did it)."""
        if self.watched:
            return 0.0
        return wait_cool(self.settings, start_c=start_c, log=self.log, sample=self._sample, clock=self._clock,
                         sleep=self._sleep)

    def _record(self, current, **extra):
        if current.get("package_c") is not None:
            self.max_c = max(self.max_c or current["package_c"], current["package_c"])
        memory = own_memory()
        self.log.write(sample_record(current, rss_gib=_gib(memory["rss"]), anon_gib=_gib(memory["anon"]), **extra))

    def _run_background(self):
        lost_since, critical = None, 0
        while not self._stop.wait(self.settings.poll_s):
            current = self._sample()
            self._record(current, state="paused" if self._in_check and self._paused else "running")
            if not self.settings.enabled:
                continue
            if self.kernel is not None and self.kernel.hits and self.gpu_hang is None:
                # a hang no check() has handled: once it has waited two polls, the main thread is stuck
                self._hang_seen = self._hang_seen or self._clock()
                if self._clock() - self._hang_seen >= 2 * self.settings.poll_s and self.gpu_hang is None:
                    self.critical = f"GPU hang in the kernel log outside check(): {self.kernel.hits[0]['line']}"
                    self.log.write(sample_record(current, event="abort", reason=self.critical,
                                                 gpu_hang=self.kernel.hits[0]["line"]))
                    self._on_critical(self.critical)
                    return
            package = current.get("package_c")
            if package is None:
                lost_since = self._clock() if lost_since is None else lost_since
                continue
            lost_since = None
            critical = critical + 1 if package >= self.settings.abort_c and not self._in_check else 0
            if critical >= self.settings.abort_samples:
                self.critical = (f"package {package:.1f} C >= abort {self.settings.abort_c:g} C on {critical} "
                                 f"samples in a row outside check()")
                self.log.write(sample_record(current, event="abort", reason=self.critical))
                self._on_critical(self.critical)
                return

    _paused = False

    def check(self, where=None):
        """Between chunks of a heavy loop: read the kernel log (gpu=True: GuardAbort at a GPU hang), sample
        (at most every poll_s), sleep while too hot or short of memory, raise GuardAbort when critical."""
        self._gpu_check(where)
        moment = self._clock()
        if self._next is not None and moment < self._next:
            return
        self._next = moment + self.settings.poll_s
        self._in_check = True
        try:
            self._check(where)
        finally:
            self._in_check, self._paused = False, False

    def _check(self, where):
        current = self._sample()
        if self._thread is None and not self.watched:
            self._record(current, where=where)
        if not self.settings.enabled:
            return
        action, reason = assess(current, self.settings, False)
        if action == "lost":
            moment = self._clock()
            self._lost_since = moment if self._lost_since is None else self._lost_since
            if moment - self._lost_since >= self.settings.sensor_loss_s:
                self._abort(f"no CPU temperature readable for {moment - self._lost_since:.0f} s", current, where)
            return
        self._lost_since = None
        if action == "abort":
            self._critical += 1
            if self._critical >= self.settings.abort_samples:
                self._abort(reason, current, where)
        else:
            self._critical = 0
        if action not in ("pause", "abort") or self.watched:
            return
        started, self._paused = self._clock(), True
        self.pauses += 1
        self.log.write(sample_record(current, event="pause", reason=reason, where=where))
        lost_since = None
        while True:
            self._sleep(self.settings.poll_s)
            current = self._sample()
            if self._thread is None:
                self._record(current, state="paused", where=where)
            moment = self._clock()
            action, why = assess(current, self.settings, True)
            if action == "lost":
                lost_since = moment if lost_since is None else lost_since
                if moment - lost_since >= self.settings.sensor_loss_s:
                    self._abort(f"no CPU temperature readable for {moment - lost_since:.0f} s", current, where)
                continue
            lost_since = None
            if action == "abort":
                self._critical += 1
                if self._critical >= self.settings.abort_samples:
                    self._abort(why, current, where)
                continue
            self._critical = 0
            if action == "resume":
                paused = moment - started
                self.paused_total += paused
                self.log.write(sample_record(current, event="resume", reason=reason, where=where,
                                             paused_s=round(paused, 1)))
                return
            if self.settings.max_pause_s and moment - started >= self.settings.max_pause_s:
                self._abort(f"paused for {moment - started:.0f} s and {reason} has not cleared", current, where)

    def _abort(self, reason, current, where):
        self.log.write(sample_record(current, event="abort", reason=reason, where=where))
        raise GuardAbort(f"guard: {reason}")


# -- the CPU lock -----------------------------------------------------------------------------------

_HELD = {"cpu": 0}


def holds_cpu_lock():
    return _HELD["cpu"] > 0


def check_lock_order(what="the GPU/memory lock"):
    """Refuse to take `what` while this process holds the CPU lock: the GPU/memory lock comes first."""
    if holds_cpu_lock():
        raise LockOrderError(f"{what} is taken before the CPU lock, never while holding it "
                             f"({CPU_LOCK}; resource_guard.cpu_lock)")


def _inherited(path, environ):
    """The descriptor named by KVPARK_CPU_LOCK_FD when it is open on the lock file at `path` and holds
    (or can take) its lock, else None."""
    text = environ.get(CPU_LOCK_FD)
    if not text:
        return None
    try:
        fd = int(text)
        held, target = os.fstat(fd), os.stat(path)
    except (ValueError, OSError):
        return None
    if (held.st_dev, held.st_ino) != (target.st_dev, target.st_ino):
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)      # the open file description we share holds it
    except OSError:
        return None
    return fd


@contextlib.contextmanager
def cpu_lock(path=None, *, log=None, label=None, environ=None):
    """Hold the CPU lock (an exclusive flock on `path`, Settings.cpu_lock by default) for CPU-heavy
    numpy work: blocks while another job holds it (logged as cpu-lock-wait with the holder the lock file
    names). Re-entrant in a process. A child handed the locked descriptor (pass_fds, and its number in
    KVPARK_CPU_LOCK_FD) holds it already: no wait, and leaving the block never unlocks the parent's lock.
    Yields a namespace: stream (None when inherited), fd, waited (seconds), inherited."""
    environ = os.environ if environ is None else environ
    path = str(path or Settings.from_env(environ).cpu_lock)
    log = log or (lambda item: None)
    if holds_cpu_lock():
        _HELD["cpu"] += 1
        try:
            yield types.SimpleNamespace(stream=None, fd=None, waited=0.0, inherited=True)
        finally:
            _HELD["cpu"] -= 1
        return
    fd = _inherited(path, environ)
    if fd is not None:
        _HELD["cpu"] += 1
        try:
            yield types.SimpleNamespace(stream=None, fd=fd, waited=0.0, inherited=True)
        finally:
            _HELD["cpu"] -= 1
        return
    started = time.monotonic()
    stream = open(path, "a+")
    held = False
    try:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.seek(0)
            log({"event": "cpu-lock-wait", "lock": path, "holder": stream.read().strip() or None})
            fcntl.flock(stream, fcntl.LOCK_EX)
        waited = time.monotonic() - started
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid {os.getpid()} {label or ' '.join(sys.argv[:2])} since {now()}\n")
        stream.flush()
        log({"event": "cpu-lock", "lock": path, "waited_s": round(waited, 1)})
        _HELD["cpu"] += 1
        held = True
        yield types.SimpleNamespace(stream=stream, fd=stream.fileno(), waited=waited, inherited=False)
    finally:
        if held:
            _HELD["cpu"] -= 1
        stream.close()


# -- the GPU lock -----------------------------------------------------------------------------------

def _inherited_lock(path, fd_dir="/proc/self/fd"):
    """A descriptor of this process open on the lock file at `path` whose open file description holds
    (or can take) its flock: a parent that holds the lock and passed its descriptor (campaign.py exec,
    pass_fds) lends it to us. None when there is none."""
    try:
        target = os.stat(path)
        names = os.listdir(fd_dir)
    except OSError:
        return None
    for name in names:
        try:
            fd = int(name)
            status = os.fstat(fd)
        except (ValueError, OSError):
            continue
        if (status.st_dev, status.st_ino) != (target.st_dev, target.st_ino):
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            continue
        return fd
    return None


@contextlib.contextmanager
def gpu_lock(path=None, *, log=None, label=None, environ=None):
    """Hold the GPU/memory lock (an exclusive flock on `path`, Settings.gpu_lock: the file campaign.gpu_lock
    takes, with its MemAvailable check) for GPU work in this process, e.g. a GPU fit. A process handed the
    locked descriptor (campaign.py exec) holds it already: no wait. Blocks while another job holds it
    (logged as gpu-lock-wait). Taken before the CPU lock, never while holding it (LockOrderError). Yields a
    namespace: fd, waited (seconds), inherited."""
    environ = os.environ if environ is None else environ
    path = str(path or Settings.from_env(environ).gpu_lock)
    log = log or (lambda item: None)
    check_lock_order()
    fd = _inherited_lock(path)
    if fd is not None:
        log({"event": "gpu-lock", "lock": path, "inherited": True, "waited_s": 0.0})
        yield types.SimpleNamespace(fd=fd, waited=0.0, inherited=True)
        return
    started = time.monotonic()
    stream = open(path, "a")
    try:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log({"event": "gpu-lock-wait", "lock": path, "label": label})
            fcntl.flock(stream, fcntl.LOCK_EX)
        waited = time.monotonic() - started
        log({"event": "gpu-lock", "lock": path, "inherited": False, "waited_s": round(waited, 1)})
        yield types.SimpleNamespace(fd=stream.fileno(), waited=waited, inherited=False)
    finally:
        stream.close()


# -- one command under the guard --------------------------------------------------------------------

class Interrupted(BaseException):
    """SIGTERM or SIGHUP while run() waits for its command (whose group it then stops)."""


def _interrupt(signum, frame):
    raise Interrupted(f"signal {signum}")


@contextlib.contextmanager
def _interruptible():
    """While run() waits: SIGTERM and SIGHUP raise Interrupted where they would end this process
    (default handlers, main thread), so the command's process group is stopped too."""
    installed = []
    if threading.current_thread() is threading.main_thread():
        for number in (signal.SIGTERM, signal.SIGHUP):
            if signal.getsignal(number) is signal.SIG_DFL:
                signal.signal(number, _interrupt)
                installed.append(number)
    try:
        yield
    finally:
        for number in installed:
            signal.signal(number, signal.SIG_DFL)


def run(argv, *, settings=None, log_path=None, label=None, events=None, cwd=None, env=None, stdout=None,
        stderr=None, cpu=False, pass_fds=(), timeout=None, check=False, poll=0.2, sample=None, kernel_log=None):
    """Run one command under the guard and wait for it.

    In a process an outer Watch already watches (KVPARK_GUARD_WATCHED=1) the command runs in this
    process group, which that Watch pauses as a whole, with capped threads. Otherwise: wait_cool, the
    CPU lock when `cpu`, capped threads (thread_env), its own process group under a Watch logging to
    log_path; the group is stopped on a timeout (not counting pauses), on KeyboardInterrupt, SIGTERM
    or SIGHUP, and on GuardAbort (a GPU hang in the kernel log included; kernel_log: a KernelLog to read
    instead of settings.gpu_watch's), which is raised after. Returns a namespace: returncode, seconds
    (pauses excluded), paused_s, cool_wait_s, timed_out, summary (the Watch's). check: a nonzero exit
    raises subprocess.CalledProcessError."""
    settings = settings if settings is not None else Settings.from_env()
    argv = [str(item) for item in argv]
    environment = thread_env(env if env is not None else os.environ, settings.threads)
    if os.environ.get(WATCHED) == "1":
        started = time.perf_counter()
        with cpu_lock(settings.cpu_lock, label=label) if cpu else contextlib.nullcontext():
            completed = subprocess.run(argv, cwd=cwd, env=environment, stdout=stdout, stderr=stderr,
                                       pass_fds=tuple(pass_fds), timeout=timeout)
        outcome = types.SimpleNamespace(returncode=completed.returncode, seconds=time.perf_counter() - started,
                                        paused_s=0.0, cool_wait_s=0.0, timed_out=False, summary=None)
    else:
        log = JsonLog(log_path, label, events)
        with contextlib.ExitStack() as stack:
            fds = list(pass_fds)
            if cpu:
                held = stack.enter_context(cpu_lock(settings.cpu_lock, log=log.write, label=label))
                if held.fd is not None:
                    fds.append(held.fd)
                    environment[CPU_LOCK_FD] = str(held.fd)
            cool = wait_cool(settings, log=log, sample=sample)
            environment[WATCHED] = "1"
            watch = Watch(settings, log_path=log_path, label=label, events=events, sample=sample, kernel_log=kernel_log)
            outcome = _watched(argv, watch, cwd=cwd, env=environment, stdout=stdout, stderr=stderr, pass_fds=fds,
                               timeout=timeout, poll=poll)
            outcome.cool_wait_s = cool
    if check and outcome.returncode != 0:
        raise subprocess.CalledProcessError(outcome.returncode, argv)
    return outcome


def _watched(argv, watch, *, cwd, env, stdout, stderr, pass_fds, timeout, poll):
    started = time.perf_counter()
    child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, pass_fds=tuple(pass_fds),
                             start_new_session=True)
    timed_out = False
    try:
        with _interruptible():
            watch.attach(child.pid)
            while child.poll() is None:
                watch.step()
                if timeout is not None and time.perf_counter() - started - watch.paused_seconds() > timeout:
                    stop_group(child, watch.settings.stop_grace_s)
                    timed_out = True
                    break
                time.sleep(poll)
    except BaseException:
        stop_group(child, watch.settings.stop_grace_s)
        watch.close()
        raise
    summary = watch.close()
    seconds = time.perf_counter() - started - summary["paused_s"]
    return types.SimpleNamespace(returncode=child.returncode, seconds=seconds, paused_s=summary["paused_s"],
                                 cool_wait_s=0.0, timed_out=timed_out, summary=summary)


# -- CLI ----------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("status", help="one sample and the settings, as JSON")
    add_arguments(sub)
    sub = commands.add_parser("run", help="one command under the guard (wait_cool, Watch, capped threads)")
    add_arguments(sub)
    sub.add_argument("--cpu-lock", action="store_true", help="hold the CPU lock while it runs (CPU-heavy numpy work)")
    sub.add_argument("--log", type=Path, help="the JSONL guard log (samples every poll_s, pauses, the summary)")
    sub.add_argument("--label", help="a name for the log lines")
    sub.add_argument("--timeout", type=float, help="stop the command's process group after this many running seconds")
    sub.add_argument("argv", nargs=argparse.REMAINDER, metavar="-- COMMAND ...")
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.command == "status":
        print(json.dumps({"sample": sample_record(read_sample(settings)), "settings": settings.as_dict(),
                          "cpu_lock_holder": (_read(settings.cpu_lock) or None),
                          "kernel_log": {"journal": journal_readable(), "kmsg": os.access("/dev/kmsg", os.R_OK)}},
                         indent=1))
        return 0
    command = list(args.argv)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("run: no command")

    def echo(item):
        print(json.dumps(item), file=sys.stderr, flush=True)
    try:
        outcome = run(command, settings=settings, log_path=args.log, label=args.label or Path(command[0]).name,
                      events=echo, cpu=args.cpu_lock, timeout=args.timeout)
    except GuardError as error:
        print(f"resource guard: {error}", file=sys.stderr, flush=True)
        return ABORT_EXIT
    echo({"event": "run-done", "returncode": outcome.returncode, "seconds": round(outcome.seconds, 1),
          "paused_s": outcome.paused_s, "cool_wait_s": round(outcome.cool_wait_s, 1), "timed_out": outcome.timed_out,
          **({"max_c": outcome.summary["max_c"], "gpu_hang": outcome.summary["gpu_hang"]} if outcome.summary else {})})
    if outcome.timed_out:
        return 124
    return outcome.returncode if outcome.returncode >= 0 else 128 - outcome.returncode


if __name__ == "__main__":
    sys.exit(main())
