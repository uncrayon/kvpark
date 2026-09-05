"""Linux process identity and a foreground launcher for the local backend."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid


def directory(value=None):
    root = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    return Path(value or os.environ.get("SLOTH_ARCHIVE_DIR", root / "sloth-memory")).expanduser().resolve()


def lock_directory(path, name):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path / name, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise RuntimeError(f"another process owns {path / name}") from None
    os.set_inheritable(fd, True)
    return fd


def file_identity(path):
    st = path.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def write_manifest(destination, pid, argv, extra_files=()):
    binary = Path(argv[0]).resolve(strict=True)
    paths = {binary, *binary.parent.glob("*.so*"), *(Path(p).resolve(strict=True) for p in extra_files)}
    # The launcher requires explicit local model paths. Include every file-valued
    # argument, split GGUF siblings, and the dynamically linked runtime libraries.
    for arg in argv[1:]:
        candidate = arg.split("=", 1)[1] if arg.startswith("--") and "=" in arg else arg
        try:
            is_file = Path(candidate).is_file()
        except OSError:
            is_file = False
        if is_file:
            path = Path(candidate).resolve()
            paths.add(path)
            match = re.match(r"(.+)-\d{5}-of-(\d{5})\.gguf$", path.name)
            if match:
                for index in range(1, int(match[2]) + 1):
                    paths.add(path.with_name(f"{match[1]}-{index:05d}-of-{match[2]}.gguf").resolve(strict=True))
    linked = subprocess.run(["ldd", str(binary)], text=True, capture_output=True)
    if linked.returncode and "not a dynamic executable" not in linked.stderr + linked.stdout:
        raise RuntimeError("cannot inspect backend libraries: " + linked.stderr.strip())
    for word in linked.stdout.split():
        if word.startswith("/") and Path(word).is_file():
            paths.add(Path(word).resolve())
    files = {str(path): file_identity(path) for path in sorted(paths)}
    # Environment affects kernels and rendering too. Store only its digest;
    # credentials and environment values must never appear in archive metadata.
    env = {k: v for k, v in os.environ.items() if k.startswith(("GGML_", "LLAMA_", "CUDA_", "HIP_", "ROCR_")) or k == "LD_LIBRARY_PATH"}
    identity = hashlib.sha256(json.dumps([argv, files, env], sort_keys=True).encode()).hexdigest()
    ticks = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    data = dict(pid=int(pid), start_ticks=ticks, run_id=uuid.uuid4().hex,
                identity=identity, files=files)
    from .proxy import atomic_json
    atomic_json(Path(destination), data)


def backend_command(server, model, archive, port, extra):
    binary = shutil.which(server) or server
    binary = str(Path(binary).expanduser().resolve(strict=True))
    model = str(Path(model).expanduser().resolve(strict=True))
    if not Path(model).is_file():
        raise ValueError("--model must be a local GGUF file")
    # This topology is part of the archive ownership contract. Never allow
    # additional arguments or llama.cpp environment defaults to override it.
    reserved = {"--model", "-m", "--model-url", "-mu", "--hf-repo", "-hf", "--hf-file", "-hff",
                "--host", "--port", "--parallel", "-np", "--slot-save-path", "--no-slots",
                "--no-cache-prompt", "--models-dir", "--models-preset", "--rpc",
                "--alias", "-a", "--mmproj-url", "--lora", "--lora-scaled"}
    for arg in extra:
        if arg.split("=", 1)[0] in reserved:
            raise ValueError(f"{arg} is managed by sloth-memory or unsupported in this alpha")
    implicit = sorted(k for k in os.environ if k.startswith("LLAMA_ARG_"))
    if implicit:
        raise ValueError("unset LLAMA_ARG_* variables and pass backend settings explicitly: " + ", ".join(implicit))
    return [binary, "--model", model, *extra, "--host", "127.0.0.1", "--port", str(port),
            "--parallel", "1", "--slot-save-path", str(archive), "--slots", "--cache-prompt",
            "--alias", "local", "--no-ui"]


def launch(server, model, archive, port, extra):
    archive = directory(archive)
    argv = backend_command(server, model, archive, port, extra)
    os.umask(0o077)
    lock_directory(archive, ".backend.lock")  # fd survives exec, released at process exit
    write_manifest(archive / "runtime.json", os.getpid(), argv)
    os.execv(argv[0], argv)
