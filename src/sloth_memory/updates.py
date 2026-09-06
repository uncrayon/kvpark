"""Replace only sloth-memory, retaining a local wheel for rollback and the model process."""

import base64
import csv
import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile

import psutil
from packaging.version import Version

from . import __version__
from .cli import request
from .network import proxy_record
from .platforms import detached_options, file_lock, lock_file, process_matches
from . import releases


class ProxyStopError(RuntimeError):
    """A replacement is still alive; do not change its package underneath it."""


def same_python_environment(executable, prefix=None):
    # Resolve aliases of the *directory* too (/var vs /private/var, Windows
    # short paths), while keeping separate venvs that share one binary distinct.
    try:
        # macOS framework and Windows venv launchers can expose the underlying
        # interpreter in argv. The proxy records the environment Python loaded.
        if prefix:
            return Path(prefix).samefile(sys.prefix)
        path = Path(executable)
        return path.parent.samefile(Path(sys.executable).parent) and path.samefile(sys.executable)
    except OSError:
        return False


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as out:
        json.dump(value, out, indent=2)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def check(archive, *, url=None):
    installed = metadata.version("sloth-memory")
    record = proxy_record(Path(archive))
    address = url or (record["base_url"] if record else None)
    try:
        running = request(address, "status", timeout=5).get("version", "unknown (restart needed)") if address else "stopped"
    except (OSError, ValueError, RuntimeError):
        running = "unreachable"
    result = dict(installed=installed, loaded=__version__, proxy=running)
    state_file = Path(archive) / "update.json"
    if state_file.exists():
        result["last_update"] = json.loads(state_file.read_text())
        state = result["last_update"]
        if (state["state"] == "restart_required" and state["version"] == installed == __version__
                and running in (installed, "stopped")):
            state.update(state="complete", note="This Hermes adapter has loaded the installed release.")
    try:
        release = releases.latest(installed)
        result.update(latest=release["version"] if release else None,
                      available=bool(release and Version(release["version"]) > Version(installed)),
                      release_url=release["release_url"] if release else releases.REPOSITORY + "/releases")
    except (OSError, ValueError) as exc:
        result["check_error"] = str(exc)
    return result


def rollback_wheel(destination):
    """Repackage the installed distribution, including local patches, before pip replaces it."""
    dist = metadata.distribution("sloth-memory")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("dir_info", {}).get("editable"):
        raise ValueError("editable installations must be updated from their source checkout")
    name = f"sloth_memory-{dist.version}"
    path = destination / f"{name}-py3-none-any.whl"
    rows = []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as wheel:
        for file in dist.files or []:
            relative = file.as_posix()
            if not relative.startswith(("sloth_memory/", name + ".dist-info/")):
                continue
            if "__pycache__" in relative or file.name in {"RECORD", "direct_url.json", "INSTALLER", "REQUESTED"}:
                continue
            data = dist.locate_file(file).read_bytes()
            wheel.writestr(relative, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            rows.append((relative, "sha256=" + digest, str(len(data))))
        if not any(row[0] == "sloth_memory/__init__.py" for row in rows):
            raise ValueError("cannot back up this installation; use its original package manager")
        record = name + ".dist-info/RECORD"
        output = io.StringIO(newline="")
        csv.writer(output).writerows([*rows, (record, "", "")])
        wheel.writestr(record, output.getvalue())
    return path


def install(wheel, log):
    env = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
    env["PIP_CONFIG_FILE"] = os.devnull
    with log.open("ab") as out:
        result = subprocess.run([sys.executable, "-I", "-m", "pip", "install", "--prefix", sys.prefix, "--no-input", "--no-index",
                                 "--no-deps", "--force-reinstall", str(wheel)],
                                env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, timeout=120)
    if result.returncode:
        raise RuntimeError(f"package installation failed; details: {log}")


def stop_proxy(record):
    if not process_matches(record):
        raise RuntimeError("proxy process identity changed; refusing to stop it")
    process = psutil.Process(record["pid"])
    process.terminate()
    deadline = time.monotonic() + 15
    while process_matches(record):
        try:
            process.wait(.1)
            return
        except psutil.TimeoutExpired:
            if time.monotonic() >= deadline:
                raise RuntimeError("proxy did not stop; no force-kill was attempted") from None


def start_proxy(record, archive, env, version):
    argv = [sys.executable, "-I", "-m", "sloth_memory", "serve", "--archive-dir", str(archive),
            "--port", record["base_url"].rsplit(":", 1)[1], "--backend", record["backend"],
            "--cache-mode", record["cache_mode"], "--upstream-url", record["upstream_url"]]
    with (archive / "hermes-proxy.log").open("ab") as log:
        child = subprocess.Popen(argv, env={**env, "SLOTH_UPDATE_START": "1"}, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, **detached_options())
    try:
        for _ in range(100):
            if child.poll() is not None:
                raise RuntimeError("replacement proxy exited; inspect hermes-proxy.log")
            current = proxy_record(archive)
            if current and current["pid"] == child.pid:
                status = request(current["base_url"], "status", timeout=5)
                if current["base_url"] != record["base_url"] or status.get("version") != version:
                    raise RuntimeError("replacement proxy address or version did not match")
                if not request(current["base_url"], "doctor", timeout=10)["ok"]:
                    raise RuntimeError("replacement proxy failed its health check")
                threading.Thread(target=child.wait, daemon=True).start()
                return current
            time.sleep(.1)
        raise RuntimeError("replacement proxy did not become ready")
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(15)
            except subprocess.TimeoutExpired:
                raise ProxyStopError(f"replacement PID {child.pid} did not stop; manual recovery required") from None
        raise


def apply(archive, *, hermes_home=None, external=False):
    if external:
        raise ValueError("this proxy is externally managed; update it through its service owner")
    if os.name == "nt" and sys.argv[0].lower().endswith("sloth-memory.exe"):
        raise ValueError("on Windows use python -m sloth_memory update so the console launcher can be replaced")
    archive = Path(archive)
    archive.mkdir(parents=True, exist_ok=True, mode=0o700)
    # One package environment can serve several profiles. Refuse concurrent
    # installs even when those profiles use different archive directories.
    fd = lock_file(Path(sys.prefix) / ".sloth-memory-update.lock")
    try:
        return _apply(archive, hermes_home)
    finally:
        os.close(fd)


def _apply(archive, hermes_home):
    installed = metadata.version("sloth-memory")
    release = releases.latest(installed)
    newer = release and Version(release["version"]) > Version(installed)
    existing = proxy_record(archive)
    running = request(existing["base_url"], "status", timeout=5) if existing else {}
    if not newer and (not existing or running.get("version") == installed):
        return dict(state="current", version=installed, note="No newer compatible published release.")
    target_version = release["version"] if newer else installed
    stage = archive / "updates" / uuid.uuid4().hex
    stage.mkdir(parents=True, mode=0o700)
    wheel = releases.download(release, stage) if newer else None
    backup = rollback_wheel(stage)
    for source in [archive / "retention.json", *([Path(hermes_home) / "config.yaml"] if hermes_home else [])]:
        if source.exists():
            shutil.copyfile(source, stage / (source.name + ".backup"))
            (stage / (source.name + ".backup")).chmod(0o600)
    state = dict(state="preparing", previous=installed, version=target_version, backup=str(stage))
    state_file = archive / "update.json"
    write_json(state_file, state)
    with file_lock(archive / ".hermes-start.lock"):
        record = proxy_record(archive)
        env = None
        stopped = changed = prepared = False
        try:
            if record:
                status = request(record["base_url"], "status", timeout=5)
                if status.get("control_version", 0) < 4:
                    raise RuntimeError("the running proxy predates coordinated updates; follow the one-time upgrade in docs/updates.md")
                process = psutil.Process(record["pid"])
                argv = process.cmdline()
                if not process_matches(record) or argv[1:4] != ["-m", "sloth_memory", "serve"] and argv[1:5] != ["-I", "-m", "sloth_memory", "serve"]:
                    raise RuntimeError("proxy ownership could not be verified")
                if not same_python_environment(argv[0], record.get("python_prefix")):
                    raise RuntimeError("proxy uses a different Python environment; update it separately")
                env = process.environ()
                request(record["base_url"], "prepare-update", {}, timeout=90)
                prepared = True
                stop_proxy(record)
                stopped = True
            state["state"] = "installing"
            write_json(state_file, state)
            if wheel:
                changed = True  # A failed pip call can already have uninstalled files.
                install(wheel, stage / "install.log")
            version = subprocess.check_output([sys.executable, "-I", "-m", "sloth_memory", "--version"],
                                              text=True, timeout=15).strip()
            if version != target_version:
                raise RuntimeError("installed package failed its version check")
            if record:
                replacement = start_proxy(record, archive, env, version)
            state.update(state="restart_required" if hermes_home else "complete",
                         note="Package and proxy updated. Restart Hermes to load its new adapter." if hermes_home
                         else "Package and proxy updated; restart other agents using this Python environment.")
            write_json(state_file, state)
        except BaseException as exc:
            try:
                if isinstance(exc, ProxyStopError):
                    raise exc
                current = proxy_record(archive)
                if stopped and current:
                    # The replacement stays in maintenance until commit.
                    stop_proxy(current)
                if changed:
                    install(backup, stage / "rollback.log")
                if stopped:
                    restored = start_proxy(record, archive, env, installed)
                    request(restored["base_url"], "cancel-update", {}, timeout=5)
                elif prepared and process_matches(record):
                    request(record["base_url"], "cancel-update", {}, timeout=5)
                state.update(state="rolled_back" if changed else "failed", note=str(exc))
            except Exception as recovery:
                state.update(state="recovery_required", note=f"{exc}; recovery: {recovery}")
            write_json(state_file, state)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RuntimeError(f"{state['state']}: {state['note']}. Backup: {stage}") from None
        if record:
            # A lost acknowledgement can mean inference is already reopened.
            # Never stop a verified proxy or roll back after releasing this gate.
            try:
                request(replacement["base_url"], "cancel-update", {}, timeout=5)
            except (OSError, ValueError, RuntimeError) as exc:
                state.update(state="maintenance_pending", note=f"Release installed and verified, but reopening inference was not confirmed: {exc}")
                write_json(state_file, state)
                raise RuntimeError(state["note"] + ". The maintenance lease expires within five minutes.") from None
        write_json(state_file, state)
        return state
