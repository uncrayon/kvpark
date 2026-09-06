"""Portable local process identity, file locks, and socket allocation."""

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import socket
import sys
import time

import psutil


def data_directory():
    if os.environ.get("XDG_DATA_HOME"):
        return Path(os.environ["XDG_DATA_HOME"])
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support"
    return Path.home() / ".local/share"


def lock_file(path, *, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            while True:
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if not blocking:
                        raise
                    time.sleep(.05)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return fd
    except OSError as exc:
        os.close(fd)
        raise RuntimeError(f"another process owns {path}") from exc


@contextmanager
def file_lock(path):
    fd = lock_file(path, blocking=True)
    try:
        yield
    finally:
        os.close(fd)


def process_identity(pid):
    process = psutil.Process(pid)
    return dict(pid=pid, created_at=process.create_time())


def process_matches(data):
    try:
        process = psutil.Process(data["pid"])
        if "created_at" in data:
            return process.create_time() == data["created_at"] and process.status() != psutil.STATUS_ZOMBIE
        # Read old Linux manifests during an alpha.2 upgrade; new files are portable.
        stat = Path(f"/proc/{data['pid']}/stat").read_text().rsplit(") ", 1)[1].split()
        return stat[19] == data["start_ticks"]
    except (psutil.Error, OSError, KeyError, ValueError):
        return False


def fsync_directory(path):
    # Windows has no directory fsync via Python's file descriptors. Atomic file
    # replacement and flushing the file itself still apply there.
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def available_port(preferred, *, exclude=()):
    with socket.socket() as sock:
        if os.name == "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred if preferred not in exclude else 0))
        except OSError as exc:
            if exc.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
            sock.bind(("127.0.0.1", 0))
        selected = sock.getsockname()[1]
    if selected in exclude:
        return available_port(0, exclude=exclude)
    return selected


def detached_options():
    if os.name == "nt":
        import subprocess
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
    return {"start_new_session": True}
