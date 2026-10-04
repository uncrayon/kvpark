"""Run bridge translation jobs on the persistent engine daemon, or one-shot.

    report = run_job(config, job, bridge_dir=DIR, timeout=900, log=print)

The daemon (engine_daemon.py next to the configured engine script, run by the configured
engine Python) keeps the hub and the GPU warm between switches. It listens on
<bridge_dir>/engine.sock, one JSON request and one JSON reply per line. run_job connects,
starts the daemon when no one answers (detached, its log in <bridge_dir>/engine.log), sends
the job inline (the conversation never touches the disk) and returns the engine's report.

When the daemon is unavailable (it cannot start, does not answer, or reports a fault of its
own) run_job falls back to the one-shot engine of earlier releases, within what is left of the
timeout: a private job file (0600, deleted afterwards) and a fresh `python engine.py JOB`
process. run_job raises EngineError instead, like a failed one-shot engine, when the daemon
answers that the job itself failed (error kind "job": a fresh engine would fail the same way;
"template" when it is the target rendering the engine cannot reproduce, as the one-shot engine
reports with its exit status TEMPLATE_MISMATCH) or that its memory guard refused it (kind
"memory": a one-shot engine would run unguarded), and when the daemon took the job but overran
the timeout (kind "timeout"; the daemon is stopped) or died before answering (kind "lost": an
out-of-memory kill, say, which an unguarded rerun would repeat). Where AF_UNIX sockets are unavailable (Windows), with config["engine"]["mode"] ==
"oneshot" or without the daemon script, the one-shot engine is the only path.

config["engine"] may also carry "device" (auto | cpu | gpu; a job's own "device" wins),
"idle_unload" and "idle_exit" (daemon seconds) and "threads" (BLAS threads of the CPU path;
default: one per physical core, at most 16, unless the environment sets OPENBLAS_NUM_THREADS,
OMP_NUM_THREADS or MKL_NUM_THREADS). status(), stop(), prewarm() and check() serve the CLI
and stage-time warm-up; features() tells what the engine that will run a switch handles.
"""

import ast
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import uuid

SOCKET = "engine.sock"
LOG = "engine.log"
DAEMON_SCRIPT = "engine_daemon.py"
START_TIMEOUT = 60.0               # torch import and GPU initialisation take a few seconds
MAX_REPLY = 64 << 20
MAX_LOG = 8 << 20
SOCKET_PATH_LIMIT = 100            # sun_path holds 104 (macOS) or 108 (Linux) bytes
FINAL = ("job", "memory", "template")   # daemon error kinds a one-shot engine must not retry
TEMPLATE = "template"              # EngineError kind: the engine cannot reproduce the target's rendering
TEMPLATE_MISMATCH = 4              # engine.py's exit status for it (engine.TEMPLATE_MISMATCH)
LOST, TIMEOUT = "lost", "timeout"  # EngineError kinds of a daemon that took the job and never answered
ALREADY_RUNNING = 3                # the daemon's exit code when another one holds the directory
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_CHILDREN = []                     # daemons started by this process, reaped once they exit


class DaemonError(RuntimeError):
    """The daemon is unavailable (run_job then uses the one-shot engine)."""


class DaemonLost(DaemonError):
    """The daemon took a request and closed the connection without answering."""


class DaemonOverran(DaemonError):
    """The daemon took a request and did not answer it in time."""


class EngineError(RuntimeError):
    """The engine failed the job: kind "job" (the job itself), "template" (the engine's renderings
    of the conversation do not match the target server's), "memory" (the memory guard),
    "timeout" (the daemon overran) or "lost" (the daemon died during the job)."""

    def __init__(self, message, kind):
        super().__init__(message)
        self.kind = kind


def _socket_path(bridge_dir):
    return Path(bridge_dir) / SOCKET


def default_threads():
    """BLAS threads of the engine's CPU path: one per physical core (Linux topology; elsewhere
    every logical CPU), at most 16. The llama-servers are idle while a switch translates."""
    try:
        cpus = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        cpus = range(os.cpu_count() or 1)
    cores = set()
    for cpu in cpus:
        try:
            cores.add(Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip())
        except OSError:
            cores.add(str(cpu))
    return max(1, min(16, len(cores)))


def _environment(config):
    """The engine process's environment: BLAS threads from the configuration, else from this
    environment, else default_threads(). A thread variable this environment sets is copied to
    the unset ones: OpenBLAS reads OPENBLAS_NUM_THREADS before OMP_NUM_THREADS, so a lone
    OMP_NUM_THREADS would otherwise lose to the engine daemon's default."""
    threads = (config.get("engine") or {}).get("threads")
    environment = dict(os.environ)
    given = next((environment[name] for name in THREAD_VARIABLES if environment.get(name)), None)
    for name in THREAD_VARIABLES:
        if threads:
            environment[name] = str(int(threads))
        elif not environment.get(name):
            environment[name] = given or str(default_threads())
    return environment


def daemon_supported(bridge_dir):
    """Whether this platform and bridge directory can host the daemon (AF_UNIX, path length)."""
    return (os.name != "nt" and hasattr(socket, "AF_UNIX")
            and len(os.fsencode(str(_socket_path(bridge_dir)))) < SOCKET_PATH_LIMIT)


def _daemon_script(config):
    engine = config.get("engine") or {}
    script = Path(engine.get("script", "")).with_name(DAEMON_SCRIPT) if engine.get("script") else None
    return script if script is not None and script.is_file() else None


def _request(bridge_dir, payload, *, timeout):
    """One request on a fresh connection; raises DaemonError (or OSError) when no daemon answers."""
    path = _socket_path(bridge_dir)
    status = path.stat()                                        # FileNotFoundError: nobody listens
    if hasattr(os, "getuid") and status.st_uid != os.getuid():
        raise DaemonError(f"{path} belongs to another user")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(path))
        with connection.makefile("rwb") as stream:
            stream.write(json.dumps(payload).encode("utf-8") + b"\n")
            stream.flush()
            try:
                line = stream.readline(MAX_REPLY + 1)
            except ConnectionResetError:
                line = b""
            except (socket.timeout, TimeoutError):
                raise DaemonOverran(f"the engine daemon did not answer within {timeout:.0f} s") from None
    if not line:
        raise DaemonLost("the engine daemon closed the connection")
    if len(line) > MAX_REPLY:
        raise DaemonError("the engine daemon's reply is too large")
    reply = json.loads(line)
    if not isinstance(reply, dict):
        raise DaemonError("the engine daemon sent a malformed reply")
    return reply


def _ping(bridge_dir, timeout=2.0):
    reply = _request(bridge_dir, {"op": "ping"}, timeout=timeout)
    if not reply.get("ok"):
        raise DaemonError(reply.get("error") or "the engine daemon did not answer ping")
    return reply


def _open_log(bridge_dir):
    path = Path(bridge_dir) / LOG
    try:
        if path.stat().st_size > MAX_LOG:
            os.replace(path, path.with_name(LOG + ".1"))
    except OSError:
        pass
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)


def _start(config, bridge_dir, script):
    engine = config["engine"]
    command = [engine["python"], str(script), "--bridge-dir", str(bridge_dir)]
    if engine.get("device"):
        command += ["--device", str(engine["device"])]
    for key in ("idle_unload", "idle_exit"):
        if engine.get(key) is not None:
            command += [f"--{key.replace('_', '-')}", str(engine[key])]
    _CHILDREN[:] = [child for child in _CHILDREN if child.poll() is None]
    fd = _open_log(bridge_dir)
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=fd,
                                   start_new_session=True, close_fds=True, env=_environment(config))
    finally:
        os.close(fd)
    _CHILDREN.append(process)
    return process


def _log_tail(bridge_dir, lines=3):
    try:
        text = (Path(bridge_dir) / LOG).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return " | ".join(text.strip().splitlines()[-lines:])


def ensure_daemon(config, bridge_dir, *, timeout=START_TIMEOUT, log=lambda event: None):
    """The ping reply of the bridge directory's daemon, starting it when nobody answers."""
    bridge_dir = Path(bridge_dir)
    if not daemon_supported(bridge_dir):
        raise DaemonError("the engine daemon needs AF_UNIX sockets and a short bridge path")
    try:
        return _ping(bridge_dir)
    except (OSError, ValueError, DaemonError):
        pass
    script = _daemon_script(config)
    if script is None:
        raise DaemonError("no engine daemon next to the engine script")
    bridge_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.monotonic()
    process = _start(config, bridge_dir, script)
    log({"step": "engine-daemon-start", "pid": process.pid})
    exited, delay = None, 0.25
    while time.monotonic() - started < timeout:
        try:
            reply = _ping(bridge_dir)
            log({"step": "engine-daemon-ready", "pid": reply.get("pid"),
                 "seconds": round(time.monotonic() - started, 3)})
            return reply
        except (OSError, ValueError, DaemonError):
            pass
        code = process.poll()
        if code is not None and code != ALREADY_RUNNING:
            raise DaemonError(f"the engine daemon exited with {code}: {_log_tail(bridge_dir)}")
        if code == ALREADY_RUNNING:
            # another daemon holds the directory: one that won a start race answers at once; one
            # that is shutting down refuses work until it is gone, then a new one can start
            exited = exited or time.monotonic()
            if time.monotonic() - exited >= delay:
                process, exited, delay = _start(config, bridge_dir, script), None, min(2 * delay, 2.0)
                log({"step": "engine-daemon-start", "pid": process.pid, "retry": "the directory was held"})
        time.sleep(0.05)
    raise DaemonError(f"the engine daemon did not answer within {timeout:.0f} s: {_log_tail(bridge_dir)}")


def _kill(pid):
    """Stop a daemon that overran a job, so it cannot write the job's output late."""
    child = next((c for c in _CHILDREN if c.pid == pid), None)
    try:
        os.kill(pid, signal.SIGTERM)
        if child is not None:
            try:
                child.wait(timeout=2)
                return
            except subprocess.TimeoutExpired:
                pass
        else:
            for _ in range(40):
                time.sleep(0.05)
                os.kill(pid, 0)              # raises once the process is gone
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _remove_output(output):
    """Remove a job's output and the temporary files a writer stopped mid-way left beside it."""
    output = Path(output)
    output.unlink(missing_ok=True)
    try:
        for partial in output.parent.glob(f".{output.name}.partial-*"):   # state_writer.write's temporaries
            partial.unlink(missing_ok=True)
    except OSError:
        pass


def _one_shot(config, job, bridge_dir, timeout):
    jobs = Path(bridge_dir) / "jobs"
    jobs.mkdir(parents=True, exist_ok=True, mode=0o700)
    job_path = jobs / f"{uuid.uuid4().hex}.json"
    try:
        fd = os.open(job_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(job, stream)
        result = subprocess.run([config["engine"]["python"], config["engine"]["script"], str(job_path)],
                                capture_output=True, text=True, timeout=timeout, env=_environment(config))
        if result.returncode != 0:
            last = (result.stderr.strip().splitlines() or ["?"])[-1]
            if result.returncode == TEMPLATE_MISMATCH:
                raise EngineError("bridge engine failed: " + last, TEMPLATE)
            raise RuntimeError("bridge engine failed: " + last)
        return json.loads(result.stdout.strip().splitlines()[-1])
    finally:
        job_path.unlink(missing_ok=True)


def run_job(config, job, *, bridge_dir, timeout=900, log=lambda event: None):
    """The engine report for one translation job (see the module docstring); report["engine_mode"]
    says which path ran ("daemon" or "oneshot", with "daemon_error" when the daemon was unavailable).
    Raises EngineError when the daemon failed the job itself, refused it for memory, overran the
    timeout or died during it, or when either engine cannot reproduce the target's rendering
    (kind "template"), and RuntimeError when the one-shot engine fails otherwise; both read
    "bridge engine failed: ...". The call stays within `timeout` seconds (plus stopping an
    overrunning daemon); the one-shot engine's overrun raises subprocess.TimeoutExpired."""
    engine = config.get("engine") or {}
    if engine.get("device") and job.get("device") is None:
        job = {**job, "device": engine["device"]}
    daemon_error = None
    started = time.monotonic()
    if engine.get("mode", "daemon") != "oneshot" and daemon_supported(bridge_dir) and _daemon_script(config):
        pid = None
        try:
            pid = ensure_daemon(config, bridge_dir, timeout=min(START_TIMEOUT, timeout), log=log).get("pid")
            remaining = max(1.0, timeout - (time.monotonic() - started))
            reply = _request(bridge_dir, {"op": "translate", "job": job}, timeout=remaining)
            if not reply.get("ok"):
                error, kind = reply.get("error") or "the engine daemon failed", reply.get("kind")
                if kind in FINAL:
                    log({"step": "engine-failed", "kind": kind, "error": error[:300]})
                    raise EngineError(f"bridge engine failed: {error}", kind)
                raise DaemonError(f"{error} ({kind or 'no kind'})")
            report = reply["report"]
            report["engine_mode"] = "daemon"
            report["daemon"] = {**(reply.get("daemon") or {}), "client_s": round(time.monotonic() - started, 3)}
            return report
        except EngineError:
            raise
        except Exception as error:          # noqa: BLE001  (the daemon is unavailable: one-shot instead)
            daemon_error = f"{type(error).__name__}: {error}"
            overran = isinstance(error, DaemonOverran)
            if (overran or isinstance(error, (socket.timeout, TimeoutError))) and pid:
                _kill(pid)
            output = (job.get("target") or {}).get("output")
            if output:
                _remove_output(output)       # never mix a late or partial daemon write with a later one
            if isinstance(error, (DaemonOverran, DaemonLost)):
                # the daemon had the job: a rerun would wait as long again, or repeat what stopped
                # the daemon (running out of memory, say) without its memory guard
                kind = TIMEOUT if overran else LOST
                log({"step": "engine-failed", "kind": kind, "error": daemon_error[:300]})
                what = (f"the engine daemon did not finish the job within {timeout:.0f} s" if overran
                        else "the engine daemon stopped during the job")
                raise EngineError(f"bridge engine failed: {what} ({daemon_error}; its log: "
                                  f"{Path(bridge_dir) / LOG})", kind) from None
            log({"step": "engine-daemon-failed", "error": daemon_error[:300]})
    report = _one_shot(config, job, bridge_dir, max(1.0, timeout - (time.monotonic() - started)))
    report["engine_mode"] = "oneshot"
    if daemon_error:
        report["daemon_error"] = daemon_error
    return report


def status(bridge_dir, *, timeout=5.0):
    """The daemon's stats, or None when no daemon runs for this bridge directory."""
    if not daemon_supported(bridge_dir):
        return None
    try:
        reply = _request(bridge_dir, {"op": "stats"}, timeout=timeout)
    except (OSError, ValueError, DaemonError):
        return None
    return reply if reply.get("ok") else None


def stop(bridge_dir, *, timeout=10.0):
    """Ask the daemon to exit; True when one answered (and its socket is gone, or the time ran out)."""
    if not daemon_supported(bridge_dir):
        return False
    try:
        reply = _request(bridge_dir, {"op": "shutdown"}, timeout=timeout)
    except (OSError, ValueError, DaemonError):
        return False
    deadline = time.monotonic() + timeout
    while _socket_path(bridge_dir).exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    return bool(reply.get("ok"))


def prewarm(config, model_names, bridge_dir, *, timeout=120.0, warm_up=True, log=lambda event: None):
    """Best effort: start the daemon and load what a staged switch needs (hub sides, GGUF profiles
    from <bridge_dir>/profiles, the cache switch.py gives jobs); returns the daemon's reply or
    None. model_names are bridge model names: (source, target) loads the source's encoder and
    the target's decoder sides and, with warm_up, runs the switch's products once on zero rows
    (a GPU picks its kernels for those shapes then, not during the switch); any other count
    loads both roles of each."""
    engine = config.get("engine") or {}
    if (not config.get("hub") or engine.get("mode", "daemon") == "oneshot" or not daemon_supported(bridge_dir)
            or not _daemon_script(config)):
        return None
    try:
        names = [config["models"][name]["hub_name"] for name in model_names]
        request = {"op": "load", "hub": config["hub"], "device": engine.get("device"),
                   "gguf": [config["models"][name]["gguf"] for name in model_names if config["models"][name].get("gguf")],
                   "cache_dir": str(Path(bridge_dir) / "profiles")}
        if len(names) == 2:
            source, target = (config["models"][name] for name in model_names)
            request.update(source=names[0], target=names[1], source_gguf=source.get("gguf"),
                           target_gguf=target.get("gguf"), warm_up=bool(warm_up))
        else:
            request["models"] = names
        ensure_daemon(config, bridge_dir, log=log)
        reply = _request(bridge_dir, request, timeout=timeout)
        if not reply.get("ok"):
            raise DaemonError(reply.get("error") or "load failed")
        log({"step": "engine-prewarmed", "sides": len(reply.get("loaded", [])), "seconds": reply.get("seconds"),
             "warm_up_s": reply.get("warm_up_s")})
        return reply
    except Exception as error:              # noqa: BLE001  (prewarming never fails a stage)
        log({"step": "engine-prewarm-failed", "error": f"{type(error).__name__}: {error}"[:300]})
        return None


def features(config, bridge_dir, *, log=lambda event: None):
    """What the engine that will run this configuration's jobs handles beyond earlier engines
    (engine.py FEATURES: "template-options", "media-source"): the daemon's own list from its ping
    (the daemon is started when nobody answers, as the job would start it), else, where jobs run
    one-shot, the FEATURES the engine script declares (read from its source, never run). An
    empty set for an engine that declares none (one from before the list existed)."""
    engine = config.get("engine") or {}
    if engine.get("mode", "daemon") != "oneshot" and daemon_supported(bridge_dir) and _daemon_script(config):
        try:
            return frozenset(map(str, ensure_daemon(config, bridge_dir, log=log).get("features") or ()))
        except (OSError, ValueError, DaemonError):
            pass                                # the job will run one-shot: what the script declares
    return script_features(engine.get("script"))


def script_features(script):
    """The FEATURES tuple an engine script assigns at its top level (parsed with ast, not run)."""
    try:
        tree = ast.parse(Path(script).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, SyntaxError):
        return frozenset()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FEATURES" for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                return frozenset()
            return frozenset(map(str, value)) if isinstance(value, (tuple, list)) else frozenset()
    return frozenset()


def check(config, *, timeout=120.0):
    """The engine Python's setup as the daemon script reports it (`engine_daemon.py --check`:
    numpy's BLAS with a warning for the reference BLAS, BLAS threads, the GPU torch sees), or
    None without a daemon script; raises DaemonError when the check fails. Runs the engine
    Python once (importing torch takes a few seconds); a running daemon reports the same in
    status()["engine"]."""
    script = _daemon_script(config)
    if script is None:
        return None
    engine = config["engine"]
    command = [engine["python"], str(script), "--check"]
    if engine.get("device"):
        command += ["--device", str(engine["device"])]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=_environment(config),
                                stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DaemonError(f"the engine check failed: {error}") from None
    if result.returncode != 0:
        raise DaemonError("the engine check failed: " + (result.stderr.strip().splitlines() or ["?"])[-1])
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise DaemonError("the engine check printed no report") from None
