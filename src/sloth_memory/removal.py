"""Stop only archive-owned services, after draining inference and saving state."""

import json
import os
from pathlib import Path

import psutil

from .cli import request
from .platforms import lock_file, process_matches
from .updates import stop_proxy


def owned_process(archive, filename):
    path = archive / filename
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if not process_matches(record):
        return None
    try:
        argv = psutil.Process(record["pid"]).cmdline()
    except psutil.NoSuchProcess:
        return None
    except psutil.Error as exc:
        raise RuntimeError("Cannot inspect the recorded service; stop it through its service owner.") from exc
    if filename == "proxy.json":
        valid = (argv[1:4] == ["-m", "sloth_memory", "serve"]
                 or argv[1:5] == ["-I", "-m", "sloth_memory", "serve"]
                 or len(argv) > 2 and Path(argv[1]).name in {"sloth-memory", "sloth-memory.exe"} and argv[2] == "serve")
        flag = "--archive-dir"
    else:
        valid = argv == record.get("argv")
        flag = "--slot-save-path"
    try:
        valid = valid and Path(argv[argv.index(flag) + 1]).resolve() == archive
    except (ValueError, IndexError):
        valid = (valid and filename == "proxy.json" and bool(record.get("archive_dir"))
                 and Path(record["archive_dir"]).resolve() == archive)
    if not valid:
        raise RuntimeError(f"Cannot verify ownership of PID {record['pid']}; stop it through its service owner.")
    return record


def plan(archive, *, external=False, managed_backend=True):
    archive = Path(archive).resolve()
    return dict(archive_dir=str(archive),
                proxy=None if external else owned_process(archive, "proxy.json"),
                backend=None if external or not managed_backend else owned_process(archive, "runtime.json"))


def stop(archive, *, external=False, managed_backend=True, disconnect=None):
    archive = Path(archive).resolve()
    # Nonblocking: a startup/update in progress should finish before uninstall.
    fd = lock_file(archive / ".hermes-start.lock")
    prepared = False
    record = None
    try:
        result = plan(archive, external=external, managed_backend=managed_backend)
        record = result["proxy"]
        if record:
            status = request(record["base_url"], "status", timeout=5)
            if (status.get("service") != "sloth-memory" or status.get("control_version", 0) < 4
                    or Path(status.get("archive_dir", "")).resolve() != archive):
                raise RuntimeError("Proxy cannot drain safely; stop clients and follow the manual uninstall guide.")
            request(record["base_url"], "prepare-update", {}, timeout=90)
            prepared = True
        # Restore the agent's route before removing the service it pointed at.
        if disconnect:
            disconnect()
        for filename, field in (("proxy.json", "proxy"), ("runtime.json", "backend")):
            if result[field]:
                current = owned_process(archive, filename)
                if current and current != result[field]:
                    raise RuntimeError("Service identity changed during uninstall; retry after stopping other clients.")
                if current:
                    try:
                        stop_proxy(current)
                    except psutil.NoSuchProcess:
                        pass
                    except (psutil.Error, RuntimeError) as exc:
                        raise RuntimeError(f"Could not stop Sloth's {field}; inspect its process before retrying: {exc}") from exc
        return result
    except BaseException:
        if prepared and record and process_matches(record):
            request(record["base_url"], "cancel-update", {}, timeout=5)
        raise
    finally:
        os.close(fd)
