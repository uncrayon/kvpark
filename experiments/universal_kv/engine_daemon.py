"""Persistent bridge engine: hub sides and the GPU (or BLAS) stay warm between switches.

    python engine_daemon.py --bridge-dir DIR [--device auto|cpu|gpu] [--idle-unload 120] [--idle-exit 600]
    python engine_daemon.py --check                  (this Python's numpy BLAS and GPU as JSON, then exit)

One process per bridge directory (flock on DIR/engine.lock) listens on the AF_UNIX stream
socket DIR/engine.sock (DIR 0700, socket 0600; on Linux every peer's uid is checked with
SO_PEERCRED, on macOS with LOCAL_PEERCRED). Requests and replies are one JSON object per line:

  {"op": "ping"}                                   liveness: pid, version, uptime, devices, BLAS,
                                                   features (engine.FEATURES)
  {"op": "load", "hub": DIR, "source": A, "target": B, "device": ...}
                                                   prewarm A's encoder and B's decoder sides
                                                   (of the kinds B decodes: K only into an
                                                   MLA target, K and the head decoders into
                                                   DeepSeek V4, on numpy, when "target_gguf"
                                                   names it)
                                                   ("models": [..] instead: both roles of each;
                                                   "gguf": [paths] and "cache_dir" also read
                                                   those models' GGUF profiles; with
                                                   "source_gguf" and "target_gguf" a switch's
                                                   products also run once on zero rows of its
                                                   shapes, unless "warm_up": false)
  {"op": "translate", "job": {...}}                engine.run on the inline job (no conversation
                                                   text touches the disk); the reply carries
                                                   engine.run's report and "daemon" timings
  {"op": "unload"}                                 drop every resident side and recurrent mean
  {"op": "stats"}                                  resident sides, memory, calibration, counters
  {"op": "shutdown"}                               reply, then exit

Replies are {"ok": true, ...} or {"ok": false, "error": "...", "kind": ...}. The kind tells a
client whether a one-shot engine may retry the request:

  job       the job itself failed (a bad conversation, a missing file, a model the hub lacks):
            a fresh engine would fail the same way
  template  the target server's tokens spell none of the engine's renderings of the conversation
            (engine.TemplateMismatch): a fresh engine would fail the same way; the bridge then
            translates the template's default rendering instead of the request's
  memory    the memory guard refused the job, or memory ran out: a retry outside the daemon
            would run unguarded
  busy      the daemon is shutting down (or its worker stopped): start a new one, or retry
  internal  a fault of the daemon itself (a programming error, a lost worker)
  request   a malformed, oversized or unknown request

ping and stats are answered at once (ping fails with "busy" while the daemon stops); load,
translate and unload run in order on one worker thread (the FIFO queue), so two switches never
compete for the GPU; every queued request gets its reply, even when the worker's own error
handling fails. Hub sides are an LRU per backend; before each job a memory guard compares the
job's estimated peak (calibrated from the jobs run so far: the largest peak per moved byte seen,
so it grows after a heavier direction) with MemAvailable (/proc/meminfo) and, on a GPU,
torch.cuda.mem_get_info: it first drops sides the job does not need, then moves a GPU job to
the CPU, then refuses with {"ok": false, "error": "MemoryError: memory: ...", "kind": "memory"}.
After each switch the GPU allocator's cached blocks are released (unless --keep-cache): on a
unified-memory GPU they are memory the llama-servers need. --keep-host (glibc) keeps the host
memory a switch freed in the process until the idle unload (malloc_trim then): the next switch
writes its translated cache into pages already faulted in instead of first-touching fresh ones
(with transparent huge pages that can mean direct compaction under memory pressure); measured
on kept Qwen3-4B jobs with the GPU otherwise idle, assemble 0.31 -> 0.24 and 0.36 -> 0.28 s for
repeated jobs (bench_engine.py job --heap keep). A GPU error that looks like a broken
device (engine.device_fault; the job itself is retried on numpy) moves every later job to the
CPU until the daemon exits. A DeepSeek V4 target is translated on numpy whatever the device
(translate.numpy_only), so its sides stay resident in the numpy backend. The recurrent means of
hybrid targets (engine.recurrent_mean: one group per file, read by the first job that needs it)
are kept with the sides. After --idle-unload seconds without requests (stats requests do not
count) the sides and means are dropped; after --idle-exit seconds the daemon exits. The log
(stderr) holds timings and sizes only, never conversation text; a log that cannot be written (a
full disk) is skipped.

The first products of each shape on a GPU pick and load their kernels (hipBLASLt, cuBLAS), and
the first job also runs the RoPE checks and reads the vocabularies and the chat template: the
first job after a plain load took 0.69-0.71 s (Qwen3.8 -> Muse) and 0.95-0.96 s (Muse ->
Qwen3.8) against 0.45-0.47 and 0.72-0.78 s warm. A load that names both GGUFs therefore also
runs the switch's products once on zero rows (encoder, latent, each target section's decoder,
RoPE and cache cast) and reads those inputs (0.35 s), after which the first job took
0.51-0.52 and 0.76 s.

BLAS threads (numpy, the CPU path) default to one per physical core, at most 16
(OPENBLAS_NUM_THREADS, OMP_NUM_THREADS or MKL_NUM_THREADS override it; the first one set is
copied to the others): during a switch the llama-servers sit idle (the source has saved, the
target waits for the restore), and on the 16-core Ryzen AI Max+ 395 16 threads translated
13K-token switches 24-30% faster than 8 (2.1 against 2.8 s Qwen3.8 -> Muse, 2.8 against 3.9 s
Muse -> Qwen3.8; RESULTS.md section 6).
"""

import os


def default_threads():
    """BLAS threads when the environment sets none: one per physical core (Linux topology;
    elsewhere every logical CPU), at most 16."""
    try:
        cpus = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        cpus = range(os.cpu_count() or 1)
    cores = set()
    for cpu in cpus:
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list", encoding="ascii") as siblings:
                cores.add(siblings.read().strip())
        except OSError:
            cores.add(str(cpu))
    return max(1, min(16, len(cores)))


# a thread variable the environment sets wins and is copied to the unset ones (OpenBLAS reads
# OPENBLAS_NUM_THREADS before OMP_NUM_THREADS, so a lone OMP_NUM_THREADS must not meet a default)
_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_GIVEN = next((os.environ[name] for name in _THREAD_VARIABLES if os.environ.get(name)), None)
for _name in _THREAD_VARIABLES:
    if not os.environ.get(_name):
        os.environ[_name] = _GIVEN or str(default_threads())

import argparse
import gc
import json
from pathlib import Path
import queue
import signal
import socket
import struct
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "linear_transfer"))

import numpy as np  # noqa: E402

import backends  # noqa: E402
from backends.numpy_backend import blas_info  # noqa: E402
import engine  # noqa: E402
import hub as hub_module  # noqa: E402
import translate  # noqa: E402

import chat  # noqa: E402
import kvstate  # noqa: E402

VERSION = 1
SOCKET = "engine.sock"
LOCK = "engine.lock"
MAX_REQUEST = 512 << 20            # bytes of one request line (a 262K-token job is ~25 MB)
GIB = 1 << 30
# rows of the warm-up products: a full window of the GPU backend and a partial one (GPU BLAS
# libraries pick kernels per shape); numpy only needs its weights paged in and its threads up
WARM_ROWS = {"torch": (4096, 1536), "numpy": (64,)}
# peak memory of a switch per byte of (source state + translated state), before any calibration:
# 13K-token Qwen3.8 -> Muse, 1.16 + 0.25 GB of state, peaked at 2.9 GiB on the GPU (research notes)
DEFAULT_FACTOR = {"host": 2.5, "gpu": 2.5}
RESERVE = {"host": 2 * GIB, "gpu": 1 * GIB}
JOB, MEMORY, BUSY, INTERNAL, REQUEST = "job", "memory", "busy", "internal", "request"     # error kinds
TEMPLATE = "template"
FAULTS = (AttributeError, NameError, TypeError, AssertionError, ImportError)   # programming errors


def log(event):
    """One JSON line on stderr; never raises (a full disk or a closed log must not stop the worker)."""
    try:
        print(json.dumps({"t": round(time.time(), 3), **event}, default=str), file=sys.stderr, flush=True)
    except Exception:                       # noqa: BLE001
        pass


def error_kind(error):
    """The kind of an error reply (see the module docstring): "memory" for MemoryError (the guard's
    refusal included), "template" for engine.TemplateMismatch, "internal" for programming errors,
    else "job"."""
    if isinstance(error, MemoryError):
        return MEMORY
    if isinstance(error, engine.TemplateMismatch):
        return TEMPLATE
    if isinstance(error, FAULTS):
        return INTERNAL
    return JOB


def mem_available():
    """MemAvailable in bytes (Linux), or None."""
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def rss_bytes(field="VmRSS"):
    """This process's resident (VmRSS) or peak (VmHWM) bytes: /proc on Linux, else the peak from
    getrusage, else None (Windows has no resource module; --check must still run there)."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage * (1 if sys.platform == "darwin" else 1024)


def reset_peak_rss():
    """Restart VmHWM at the current RSS (Linux /proc/self/clear_refs); False when unsupported."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as refs:
            refs.write("5")
        return True
    except OSError:
        return False


def peer_uid(connection):
    """The connecting process's uid, or None where the platform cannot tell (then the 0700 dir guards)."""
    if hasattr(socket, "SO_PEERCRED"):
        data = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        return struct.unpack("3i", data)[1]
    if sys.platform == "darwin":
        try:   # LOCAL_PEERCRED (SOL_LOCAL 0, option 1): struct xucred {u_int version; uid_t uid; ...}
            data = connection.getsockopt(0, 1, 76)
            return struct.unpack_from("=Ii", data)[1]
        except OSError:
            return None
    return None


def state_bytes(spec, cells):
    """Bytes of a translated state of `cells` cells in f16 (an upper bound for the output): every
    cell's rows (state_bytes_per_cell), or for a DeepSeek V4 target the K-only latents of its last
    n_swa cells, one pooled row per compress ratio cells and layer, and the compressor rings (f32)."""
    if spec.memory != "dsv4":
        return cells * state_bytes_per_cell(spec)
    total = min(cells, spec.n_swa) * state_bytes_per_cell(spec)
    for pool in spec.pools():
        rows, ring = cells // pool.ratio * 2 * pool.width, 2 * 4 * pool.state_size * pool.state_width
        total += len(pool.layers) * (rows + ring)
    return total


def state_bytes_per_cell(spec):
    """Bytes of one f16 cell over every attention layer's stored rows (an upper bound for the
    output): K, V where the layer stores one (an MLA layer stores K only, a DeepSeek V4 layer its
    latent), and the indexer keys of a DSA or qwen4exp model."""
    total = 0
    for layer in spec.attention_layers:
        for key in ("k", "v"):
            shape = spec.kv_shape(layer, key)
            if shape is not None:
                total += 2 * shape[0] * shape[1]
    if spec.indexer is not None:
        for layer in spec.indexer.layers:
            for key in ("k", "v"):
                shape = spec.kv_shape(layer, key, "indexer")
                if shape is not None:
                    total += 2 * shape[0] * shape[1]
    return total


def keep_freed_memory():
    """glibc: keep freed heap blocks in the process (no trim of the heap top; the mmap threshold at
    its 32 MiB maximum, so blocks up to that size come from the heap) for later allocations to
    reuse already faulted pages; False where unavailable (not glibc)."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        # M_TRIM_THRESHOLD = -1, M_MMAP_THRESHOLD = -3 (malloc.h)
        return bool(libc.mallopt(-1, ctypes.c_int(2**31 - 1))) and bool(libc.mallopt(-3, ctypes.c_int(32 << 20)))
    except (OSError, AttributeError):
        return False


def trim_memory():
    """Give the heap's free memory back to the system (glibc malloc_trim); None where unavailable."""
    try:
        import ctypes
        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (OSError, AttributeError):
        return None


def engine_info(device=None, *, gpu=True):
    """This Python's engine setup: numpy and its BLAS (a warning for the reference BLAS), the BLAS
    threads, the requested device, and with gpu whether torch sees a GPU (importing torch takes a
    few seconds)."""
    blas = blas_info()
    info = {"python": sys.executable, "numpy": np.__version__,
            "blas": {key: blas.get(key) for key in ("name", "gflops", "reference", "warning") if blas.get(key) is not None},
            "threads": os.environ.get("OPENBLAS_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS"),
            "device": backends.requested(device)}
    if gpu:
        available, detail = backends.gpu_status()
        info["gpu"] = {"available": available, "detail": str(detail)}
    return info


def warm_products(backend, encoder, decoder, rows, sections, *, spec=None, kind="k"):
    """A switch's products on `rows` zero rows: encode (stage one and the latent, centred on the
    decoder as translate.compute_cache does), each target section's decode and, with the
    target's spec, its RoPE and cache cast (backend.cache_rows, float16)."""
    if backend.name == "torch":
        import torch
        half, device = backend.half, backend.device
        if encoder.stacked:
            layers, width, _ = encoder.bases.shape
            x = torch.zeros((layers, rows, width), dtype=half, device=device)
            stage1 = torch.bmm(x, encoder.bases).transpose(0, 1).reshape(rows, -1)
        else:
            stage1 = torch.cat([torch.zeros((rows, basis.shape[0]), dtype=half, device=device) @ basis
                                for basis in encoder.bases], dim=1)
        z = torch.addmm(decoder.z_bias, stage1, encoder.weight)
    else:
        z = encoder.encode({layer: np.zeros((rows, len(mean)), dtype=np.float32)
                            for layer, mean in zip(encoder.layers, encoder.means)}) - decoder.z_mean
    for layers in sections:
        weight, mean = decoder.section(layers)
        values = (z @ weight).to(torch.float32) + mean if backend.name == "torch" else z @ weight + mean
        if spec is not None:
            backend.cache_rows(spec, layers, kind, values, np.arange(rows, dtype=np.int64), "f16")


class Daemon:
    def __init__(self, bridge_dir, *, device=None, idle_unload=120.0, idle_exit=600.0, keep_cache=False,
                 keep_host=False):
        self.dir = Path(bridge_dir)
        self.keep_cache = keep_cache
        self.keep_host = keep_host and keep_freed_memory()       # False where glibc's mallopt is missing
        self.device = backends.requested(device)
        self.idle_unload, self.idle_exit = float(idle_unload), float(idle_exit)
        self.started = time.monotonic()
        self.last_request = self.started
        self.unloaded = True
        self.jobs = self.failures = 0
        self.gpu_fault = None                   # the error that disabled the GPU, once one did
        self.factor = dict(DEFAULT_FACTOR)
        self.calibrated = {"host": False, "gpu": False}
        self.hubs = {}                          # directory -> (manifest identity, Hub)
        self.verifier = hub_module.Verifier(self.dir / "profiles" / "hub-verified.json")
        self.queue = queue.Queue()
        self.worker = None
        self.busy = False                       # the worker is running a request (never idle then)
        self.stopping = threading.Event()
        self.server = self.lock_fd = None
        self.socket_identity = None
        self.info = None                        # engine_info(), read once when serving starts

    # -- life cycle -------------------------------------------------------------------------------

    def bind(self):
        """Take the single-instance lock and the socket; False when another daemon owns the directory."""
        import fcntl
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        status = self.dir.stat()
        if status.st_uid != os.getuid():
            raise PermissionError(f"{self.dir} belongs to another user")
        if status.st_mode & 0o077:
            os.chmod(self.dir, 0o700)
        self.lock_fd = os.open(self.dir / LOCK, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.lock_fd)
            self.lock_fd = None
            return False
        path = self.dir / SOCKET
        path.unlink(missing_ok=True)            # stale: its owner no longer holds the lock
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            self.server.bind(str(path))
        finally:
            os.umask(old)
        os.chmod(path, 0o600)
        self.socket_identity = (path.stat().st_ino, path.stat().st_dev)
        self.server.listen(16)
        self.server.settimeout(1.0)
        return True

    def serve(self):
        try:                                    # a CPU daemon never imports torch
            self.info = engine_info(self.device, gpu=self.device != "cpu")
        except Exception as error:              # noqa: BLE001  (informational only)
            self.info = {"error": f"{type(error).__name__}: {error}"[:300]}
        self.worker = worker = threading.Thread(target=self._work, name="engine-worker", daemon=True)
        worker.start()
        log({"event": "start", "pid": os.getpid(), "device": self.device, "dir": str(self.dir),
             "idle_unload_s": self.idle_unload, "idle_exit_s": self.idle_exit, "threads": self.info.get("threads"),
             "blas": (self.info.get("blas") or {}).get("name")})
        checked = time.monotonic()
        try:
            while not self.stopping.is_set():
                try:
                    connection, _ = self.server.accept()
                    threading.Thread(target=self._connection, args=(connection,), daemon=True).start()
                except socket.timeout:
                    pass
                except OSError:
                    break
                if time.monotonic() - checked >= 1.0:     # also while requests keep arriving (stats polls)
                    checked = time.monotonic()
                    self._idle()
        finally:
            self.stopping.set()
            self.queue.put(None)
            worker.join(timeout=60)
            self.close()
        log({"event": "exit", "jobs": self.jobs, "failures": self.failures})

    def close(self):
        if self.server is not None:
            self.server.close()
            self.server = None
            path = self.dir / SOCKET
            try:
                status = path.stat()
                if (status.st_ino, status.st_dev) == self.socket_identity:
                    path.unlink()
            except OSError:
                pass
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None

    def _idle(self):
        idle = time.monotonic() - self.last_request
        if self.busy or not self.queue.empty():
            return
        if idle >= self.idle_exit:
            log({"event": "idle-exit", "idle_s": round(idle, 1)})
            self.stopping.set()
        elif idle >= self.idle_unload and not self.unloaded:
            self.queue.put(({"op": "unload", "reason": "idle"}, None, time.monotonic()))

    # -- requests ---------------------------------------------------------------------------------

    def _connection(self, connection):
        with connection:
            try:
                uid = peer_uid(connection)
                if uid is not None and uid != os.getuid():
                    log({"event": "refused-peer", "uid": uid})
                    return
                with connection.makefile("rwb") as stream:
                    self._requests(stream)
            except Exception:                       # noqa: BLE001  (a broken connection ends only itself)
                return

    def _requests(self, stream):
        """Answer one connection's requests, a line each, until it closes or asks for shutdown."""
        while not self.stopping.is_set():
            line = stream.readline(MAX_REQUEST + 1)
            if not line:
                return
            if len(line) > MAX_REQUEST:
                self._reply(stream, {"ok": False, "error": "request too large", "kind": REQUEST})
                return
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("a request is a JSON object")
            except ValueError as error:
                self._reply(stream, {"ok": False, "error": f"bad request: {error}", "kind": REQUEST})
                continue
            try:
                reply = self.handle(request)
            except Exception as error:              # noqa: BLE001  (a reply, never a dropped connection)
                reply = {"ok": False, "error": f"{type(error).__name__}: {error}", "kind": INTERNAL}
            self._reply(stream, reply)
            if request.get("op") == "shutdown":
                return

    @staticmethod
    def _reply(stream, payload):
        stream.write(json.dumps(payload).encode("utf-8") + b"\n")
        stream.flush()

    def handle(self, request):
        """The reply to one request (queued ops wait for the worker)."""
        op = request.get("op")
        if op != "stats":                       # a status poller must not keep the daemon from idling
            self.last_request = time.monotonic()
        if op == "ping":
            problem = self._problem()
            if problem:
                return {"ok": False, "error": problem, "kind": BUSY}
            return {"ok": True, "pid": os.getpid(), "version": VERSION, "device": self.device,
                    "gpu_fault": self.gpu_fault, "uptime_s": round(time.monotonic() - self.started, 1),
                    "engine": self.info, "features": list(engine.FEATURES)}
        if op == "stats":
            return {"ok": True, **self.stats()}
        if op == "shutdown":
            self.stopping.set()
            return {"ok": True, "pid": os.getpid()}
        if op not in ("load", "translate", "unload"):
            return {"ok": False, "error": f"unknown op {op!r}", "kind": REQUEST}
        problem = self._problem()
        if problem:
            return {"ok": False, "error": problem, "kind": BUSY}
        done = {"event": threading.Event()}
        self.queue.put((request, done, time.monotonic()))
        while not done["event"].wait(1.0):
            if not self._worker_alive():        # a worker that died answers nothing more
                self.stopping.set()
                return {"ok": False, "error": "the engine worker stopped", "kind": BUSY}
        return done["reply"]

    def _worker_alive(self):
        return self.worker is not None and self.worker.is_alive()

    def _problem(self):
        """None, or why the daemon takes no more work; a lost worker also stops the daemon, so the
        next client starts a fresh one."""
        if self.worker is not None and not self.worker.is_alive():
            self.stopping.set()
            return "the engine worker stopped"
        if self.stopping.is_set():
            return "the engine daemon is shutting down"
        return None

    def _work(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            request, done, queued = item
            reply = {"ok": False, "error": "the engine worker failed", "kind": INTERNAL}
            self.busy = True
            try:
                reply = self._run(request, done, queued)
            except Exception as error:          # noqa: BLE001  (a fault outside the op itself: keep serving)
                reply = {"ok": False, "error": f"the engine worker failed: {type(error).__name__}: {error}",
                         "kind": INTERNAL}
            finally:                            # every queued request gets its reply, whatever happens here
                self.busy = False
                if done is not None:            # a client's request (not the idle unload) ends the idle time
                    self.last_request = time.monotonic()
                    done["reply"] = reply
                    done["event"].set()

    def _run(self, request, done, queued):
        started = time.monotonic()
        try:
            if self.stopping.is_set() and done is not None:
                return {"ok": False, "error": "the engine daemon is shutting down", "kind": BUSY}
            reply = {"ok": True, **getattr(self, "op_" + request["op"])(request)}
        except Exception as error:      # noqa: BLE001  (every failure becomes an error reply)
            kind = error_kind(error)
            reply = {"ok": False, "error": f"{type(error).__name__}: {error}", "kind": kind}
            if request.get("op") == "translate":
                self.failures += 1
            log({"event": "error", "op": request.get("op"), "kind": kind, "error": reply["error"][:500]})
        if reply.get("ok") and "daemon" in reply:
            reply["daemon"].update(queue_s=round(started - queued, 3),
                                   run_s=round(time.monotonic() - started, 3))
        return reply

    # -- operations (worker thread) ---------------------------------------------------------------

    def hub(self, directory):
        """The Hub of a bundle directory, reopened when its manifest changes (stale sides dropped)."""
        directory = str(Path(directory).resolve())
        status = (Path(directory) / "manifest.json").stat()
        identity = (status.st_size, status.st_mtime_ns, status.st_ino)
        cached = self.hubs.get(directory)
        if cached is None or cached[0] != identity:
            if cached is not None:
                for backend in backends.loaded().values():
                    backend.drop([item["key"] for item in backend.resident() if item["key"][0] == directory])
            cached = self.hubs[directory] = (identity, hub_module.Hub(directory, verifier=self.verifier))
        return cached[1]

    def backend(self, device, numpy_only=False):
        want = backends.requested(device or self.device)
        if self.gpu_fault and want != "cpu":
            return backends.select("cpu")       # a device fault disabled the GPU for this daemon's life
        if numpy_only:
            return backends.select("cpu")       # a DeepSeek V4 target (engine.run would move it anyway)
        return backends.select(want)

    def op_load(self, request):
        started = time.monotonic()
        hub = self.hub(request["hub"])
        backend = self.backend(request.get("device"), self.numpy_target(request.get("target_gguf"),
                                                                         request.get("cache_dir")))
        if request.get("models"):
            sides = [(kind, name, role) for name in request["models"] for kind in hub_module.KINDS
                     for role in ("encoder", "decoder")]
        else:
            sides = self.pair_sides(request["source"], request["target"], request.get("target_gguf"),
                                    request.get("cache_dir"))
        for kind, name, role in sides:
            backend.side(hub, kind, name, role)
        backend.synchronize()
        for gguf in request.get("gguf") or ():
            engine.profile(str(gguf), request.get("cache_dir"))
        self.unloaded = False
        loaded_s = round(time.monotonic() - started, 3)
        warm = None
        if request.get("source") and request.get("target_gguf") and request.get("warm_up", True):
            warm = self.warm_up(backend, hub, request)
        log({"event": "load", "backend": backend.name, "sides": len(sides), "load_s": loaded_s,
             "warm_up_s": warm, "seconds": round(time.monotonic() - started, 3)})
        return {"loaded": [list(side) for side in sides], "backend": backend.name,
                "resident_bytes": backend.resident_bytes(), "load_s": loaded_s, "warm_up_s": warm,
                "seconds": round(time.monotonic() - started, 3)}

    @staticmethod
    def pair_kinds(target_gguf, cache_dir=None, runtime=None):
        """The hub kinds a switch into this target translates (translate.target_kinds: K only into
        an MLA target, K and the head kinds into DeepSeek V4); both without a GGUF or when its
        profile cannot be read."""
        if not target_gguf:
            return hub_module.KINDS
        try:
            engine._PROFILE_CACHE[0] = cache_dir
            return translate.target_kinds(engine.spec_of(target_gguf, runtime))
        except Exception:                       # noqa: BLE001  (sides of every kind are never wrong)
            return hub_module.KINDS

    @classmethod
    def pair_sides(cls, source, target, target_gguf, cache_dir=None, runtime=None):
        """[(kind, model, role)] a switch from source into target reads: the target's decoder of
        each pair kind, the source's encoder of each but the head kinds (they decode the K latent)."""
        kinds = cls.pair_kinds(target_gguf, cache_dir, runtime)
        return ([(kind, source, "encoder") for kind in kinds if kind not in hub_module.HEAD_KINDS]
                + [(kind, target, "decoder") for kind in kinds])

    @staticmethod
    def numpy_target(target_gguf, cache_dir=None, runtime=None):
        """Whether a switch into this target runs on numpy whatever the device (translate.numpy_only:
        DeepSeek V4); False without a GGUF or when its profile cannot be read."""
        if not target_gguf:
            return False
        try:
            engine._PROFILE_CACHE[0] = cache_dir
            return translate.numpy_only(engine.spec_of(target_gguf, runtime))
        except Exception:                       # noqa: BLE001  (the engine decides again)
            return False

    def warm_up(self, backend, hub, request):
        """Run the products of a switch from request["source"] into request["target"] once on zero
        rows (WARM_ROWS) and the backend's RoPE checks for both models; returns the seconds taken,
        or None when it failed (logged: a warm-up never fails a load)."""
        started = time.monotonic()
        try:
            engine._PROFILE_CACHE[0] = request.get("cache_dir")
            specs = {}
            for role in ("source", "target"):
                if request.get(f"{role}_gguf"):
                    specs[role] = engine.spec_of(request[f"{role}_gguf"])
                    backend.supports(specs[role])           # the GPU checks its RoPE against kvstate once
            sections = [tuple(layout.layers) for layout in specs["target"].layouts(kvstate.FLAGS_NONE)
                        if layout.kind == "kv" and layout.role == "attention" and layout.layers]
            for kind in translate.target_kinds(specs["target"]):
                decoder = backend.decoder(hub, kind, request["target"])
                if kind not in translate.encoder_kinds(specs["target"]):
                    continue                    # a head kind: pooled on numpy from the K latent's z
                encoder = backend.encoder(hub, kind, request["source"])
                own = [layers for layers in sections if set(layers) <= set(decoder.layers)]
                for rows in WARM_ROWS.get(backend.name, WARM_ROWS["numpy"]):
                    warm_products(backend, encoder, decoder, rows, own, spec=specs["target"], kind=kind)
            backend.synchronize()
            # what each job reads before the arithmetic: both vocabularies and the target's chat
            # template (compiled once per process)
            for role in specs:
                engine.vocabulary(request[f"{role}_gguf"])
            chat.compile_template(engine.profile(request["target_gguf"], request.get("cache_dir"))["chat_template"])
            if not self.keep_cache:
                backend._released()
        except Exception as error:              # noqa: BLE001
            log({"event": "warm-up-error", "error": f"{type(error).__name__}: {error}"[:300]})
            return None
        return round(time.monotonic() - started, 3)

    def op_unload(self, request):
        released = sum(backend.drop() for backend in backends.loaded().values())
        released += engine.drop_recurrent_means()
        gc.collect()
        trimmed = trim_memory() if self.keep_host else None     # the host memory --keep-host held
        self.unloaded = True
        log({"event": "unload", "reason": request.get("reason", "request"), "released_bytes": released,
             "trimmed": trimmed})
        return {"released_bytes": released}

    def op_translate(self, request):
        job = request["job"]
        engine._PROFILE_CACHE[0] = job.get("cache_dir")      # the guard's estimate reads the target profile
        hub = self.hub(job["hub"])
        backend = self.backend(job.get("device"), self.numpy_target(job["target"].get("gguf"), job.get("cache_dir"),
                                                                    job["target"].get("runtime")))
        guard = self.guard(job, hub, backend)
        if guard.get("device") == "cpu":
            backend = backends.select("cpu")
        before_rss = rss_bytes()
        peak_reset = reset_peak_rss()
        gpu = backend.name == "torch" and getattr(backend, "device", None) is not None \
            and str(backend.device).startswith("cuda")
        if gpu:
            backend.reset_peak()
            gpu_before = backend.memory().get("allocated", 0)
        started = time.monotonic()
        try:
            report = engine.run(job, hub=hub, backend=backend, log=log)
        finally:
            if not self.keep_cache:
                try:
                    backend._released()     # give the allocator's cached blocks back (GTT that llama-server shares)
                except Exception:           # noqa: BLE001  (a broken device context may refuse)
                    pass
        self.jobs += 1
        self.unloaded = False
        if report.get("device_fault") and self.gpu_fault is None:
            self.gpu_fault = report["fallback"][:300]
            log({"event": "gpu-disabled", "error": self.gpu_fault})
        measured = {}
        try:                                # bookkeeping never fails a switch that is already written
            measured["host_peak_bytes"] = max(0, rss_bytes("VmHWM") - before_rss) if peak_reset else None
            if gpu:
                memory = backend.memory()
                measured["gpu_peak_bytes"] = max(0, memory.get("peak_allocated", 0) - gpu_before)
            self.calibrate(guard, measured)
        except Exception as error:          # noqa: BLE001
            log({"event": "bookkeeping-error", "error": f"{type(error).__name__}: {error}"[:300]})
        log({"event": "translate", "backend": report.get("backend"), "t0": report.get("t0"),
             "cells": report.get("cells"), "engine_s": report.get("engine_seconds"),
             "seconds": round(time.monotonic() - started, 3), **measured})
        return {"report": report, "daemon": {"jobs": self.jobs, "memory_guard": guard, **measured}}

    # -- memory guard ------------------------------------------------------------------------------

    def estimate(self, job):
        """Bytes of (source state + translated state) a job moves: what its peak scales with."""
        source = Path(job["source"]["state"]).stat().st_size
        try:
            spec = engine.spec_of(job["target"]["gguf"], job["target"].get("runtime"))
            output = state_bytes(spec, len(job["target"]["tokens"]))
        except Exception:                       # noqa: BLE001  (the estimate must never fail a job)
            output = source
        return source + output

    def guard(self, job, hub, backend):
        moved = self.estimate(job)
        need = {"host": int(self.factor["host"] * moved) + RESERVE["host"]}
        on_gpu = backend.name == "torch"
        if on_gpu:
            need["gpu"] = int(self.factor["gpu"] * moved) + RESERVE["gpu"]
        keep = [(str(hub.directory), kind, name, role)
                for kind, name, role in self.pair_sides(job["source"]["name"], job["target"]["name"],
                                                        job["target"].get("gguf"), job.get("cache_dir"),
                                                        job["target"].get("runtime"))]
        out = {"moved_bytes": moved, "need": need, "evicted_bytes": 0}
        available = self.available(backend)
        for where in need:
            short = need[where] - available[where] if available.get(where) is not None else 0
            if short > 0:
                for other in backends.loaded().values():
                    if (where == "gpu") == (other.name == "torch"):
                        out["evicted_bytes"] += other.evict(short, keep=keep)
                available = self.available(backend)
        out["available"] = available
        host_ok = available["host"] is None or need["host"] <= available["host"]
        if on_gpu and available.get("gpu") is not None and need["gpu"] > available["gpu"]:
            if not host_ok:
                raise MemoryError(f"memory: a switch needs about {need['gpu'] >> 20} MiB on the GPU and "
                                  f"{need['host'] >> 20} MiB of RAM; {available['gpu'] >> 20} / "
                                  f"{available['host'] >> 20} MiB are free")
            out["device"] = "cpu"
            log({"event": "memory-guard", "moved_to": "cpu", "need": need, "available": available})
        elif not host_ok:
            raise MemoryError(f"memory: a switch needs about {need['host'] >> 20} MiB of RAM; "
                              f"{available['host'] >> 20} MiB are available")
        return out

    def available(self, backend):
        out = {"host": mem_available()}
        if backend.name == "torch":
            memory = backend.memory()
            out["gpu"] = memory.get("device_free")
        return out

    def calibrate(self, guard, measured):
        """Track the observed peak per moved byte (the largest seen, with a 10% margin)."""
        moved = max(guard["moved_bytes"], 1)
        for where, key in (("host", "host_peak_bytes"), ("gpu", "gpu_peak_bytes")):
            value = measured.get(key)
            if value:
                ratio = 1.1 * value / moved
                if not self.calibrated[where]:
                    self.factor[where], self.calibrated[where] = ratio, True
                else:
                    self.factor[where] = max(self.factor[where], ratio)

    def stats(self):
        out = {"pid": os.getpid(), "version": VERSION, "device": self.device,
               "uptime_s": round(time.monotonic() - self.started, 1),
               "idle_s": round(time.monotonic() - self.last_request, 1), "jobs": self.jobs,
               "failures": self.failures, "queued": self.queue.qsize(), "rss_bytes": rss_bytes(),
               "mem_available_bytes": mem_available(), "factor": self.factor, "calibrated": self.calibrated,
               "idle_unload_s": self.idle_unload, "idle_exit_s": self.idle_exit, "keep_host": self.keep_host,
               "worker_alive": self._worker_alive(), "gpu_fault": self.gpu_fault, "engine": self.info,
               "features": list(engine.FEATURES), "hubs": sorted(list(self.hubs)),
               "recurrent_means": engine.recurrent_means(), "backends": {}}
        for name, backend in backends.loaded().items():
            out["backends"][name] = {"describe": backend.describe(), "resident": backend.resident(),
                                     "resident_bytes": backend.resident_bytes(), "memory": backend.memory()}
        return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bridge-dir", type=Path, help="the bridge directory (socket, lock, log)")
    parser.add_argument("--device", default=None, help="auto | cpu | gpu (default $KVPARK_ENGINE_DEVICE or auto)")
    parser.add_argument("--idle-unload", type=float, default=120.0, help="drop hub sides after this idle time (s)")
    parser.add_argument("--idle-exit", type=float, default=600.0, help="exit after this idle time (s)")
    parser.add_argument("--keep-cache", action="store_true",
                        help="keep the GPU allocator's cached blocks between switches (faster, holds memory)")
    parser.add_argument("--keep-host", action="store_true",
                        help="keep freed host memory until the idle unload (glibc): the next switch's output pages are "
                        "already faulted in (faster, holds memory)")
    parser.add_argument("--check", action="store_true",
                        help="print this Python's engine setup (numpy BLAS, threads, GPU) as JSON and exit")
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps({**engine_info(args.device, gpu=backends.requested(args.device) != "cpu"), "version": VERSION}))
        return 0
    if args.bridge_dir is None:
        parser.error("--bridge-dir is required")
    daemon = Daemon(args.bridge_dir, device=args.device, idle_unload=args.idle_unload, idle_exit=args.idle_exit,
                    keep_cache=args.keep_cache, keep_host=args.keep_host)
    if not daemon.bind():
        log({"event": "already-running", "dir": str(args.bridge_dir)})
        return 3
    signal.signal(signal.SIGTERM, lambda *_: daemon.stopping.set())
    daemon.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
