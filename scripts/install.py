#!/usr/bin/env python3
"""Install kvpark in the environment that will actually run it."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

RELEASE = "0.3.0a2"
REPOSITORY = "https://github.com/uncrayon/kvpark"


def run(command, *, capture=False):
    result = subprocess.run([str(item) for item in command], stdin=subprocess.DEVNULL,
                            text=True, stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None)
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Command failed: {shlex.join(map(str, command))}\n{detail}")
    return result.stdout.strip() if capture else None


def probe(python):
    """Inspect an interpreter without resolving away its virtual environment."""
    try:
        result = subprocess.run([str(python), "-I", "-c", '''
import importlib.util, json, sys
from importlib.metadata import version, PackageNotFoundError
def installed(name):
    try: return version(name)
    except PackageNotFoundError: return None
print(json.dumps(dict(python=sys.executable, prefix=sys.prefix, base=sys.base_prefix,
    version=list(sys.version_info[:2]), hermes=importlib.util.find_spec("hermes_cli") is not None,
    kvpark=installed("kvpark"), legacy=installed("sloth-memory"))))
'''], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
        return json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def find_hermes(explicit=None):
    if explicit:
        python = Path(explicit).expanduser().absolute()
        info = probe(python)
        if not info or not info["hermes"]:
            raise RuntimeError(f"Hermes is not importable with {python}. Pass --hermes-python with Hermes' own Python.")
        return python, info
    candidates = []
    launcher = shutil.which("hermes")
    if launcher:
        path = Path(launcher).resolve()
        candidates.append(path.parent / "python")
        try:
            first = path.read_text().splitlines()[0]
            if first.startswith("#!"):
                interpreter = first[2:].strip()
                if "python" in Path(interpreter).name:
                    candidates.insert(0, Path(interpreter))
        except (OSError, UnicodeError, IndexError):
            pass
    if os.environ.get("HERMES_INSTALL_DIR"):
        candidates.extend(Path(os.environ["HERMES_INSTALL_DIR"]).expanduser() / name / "bin/python" for name in ("venv", ".venv"))
    homes = [Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")), Path.home() / ".hermes"]
    for home in homes:
        candidates.extend(home / "hermes-agent" / name / "bin/python" for name in ("venv", ".venv"))
    found = {}
    for python in candidates:
        info = probe(python)
        if info and info["hermes"]:
            found.setdefault(info["prefix"], (python, info))
    if len(found) > 1:
        choices = "\n".join(f"  --hermes-python {shlex.quote(str(python))}" for python, _ in found.values())
        raise RuntimeError(f"Multiple Hermes environments found. Select the one you use:\n{choices}")
    if found:
        return next(iter(found.values()))
    if launcher:
        raise RuntimeError("Found a Hermes launcher but could not identify its Python. Pass --hermes-python PATH, or --standalone.")
    return None


def fetch(url, limit):
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "kvpark-installer", "Accept": "application/vnd.github+json"})
            with urlopen(request, timeout=30) as response:
                data = response.read(limit + 1)
            if len(data) > limit:
                raise RuntimeError("Download exceeds the expected size limit.")
            return data
        except (OSError, URLError):
            if attempt == 2:
                raise RuntimeError(f"Could not download {url}. Check your connection or proxy, then rerun the installer.") from None
            time.sleep(attempt + 1)


def release_asset(version, name, destination, limit=25 * 1024**2):
    if not re.fullmatch(r"[0-9][0-9A-Za-z.\-]*", version):
        raise RuntimeError("Invalid release version; use a version such as 0.3.0a1.")
    release = json.loads(fetch(f"https://api.github.com/repos/uncrayon/kvpark/releases/tags/v{version}", 2 * 1024**2))
    url = f"{REPOSITORY}/releases/download/v{version}/{name}"
    asset = next((item for item in release.get("assets", [])
                  if item.get("name") == name and item.get("browser_download_url") == url), None)
    if not asset or not re.fullmatch(r"sha256:[0-9a-f]{64}", asset.get("digest") or ""):
        raise RuntimeError(f"Release {version} has no requested asset with a verifiable checksum. No package was installed.")
    data = fetch(url, limit)
    if "sha256:" + hashlib.sha256(data).hexdigest() != asset["digest"]:
        raise RuntimeError("Package checksum mismatch. No package was installed; rerun to download again.")
    wheel = destination / name
    wheel.write_bytes(data)
    return wheel


def build_runtime(args, root):
    destination = root / "runtimes" / args.version / args.runtime
    binary = destination / "llama-server"
    if binary.is_file():
        run([binary, "--version"])
        return binary
    missing = [tool for tool in ("git", "cmake") if not shutil.which(tool)]
    if not any(shutil.which(tool) for tool in ("c++", "g++", "clang++")):
        missing.append("C++ compiler")
    if missing:
        raise RuntimeError("Runtime build needs " + ", ".join(missing) +
                           ". Install these with your OS package manager, then rerun with --runtime " + args.runtime +
                           ". The Python package can be installed without --runtime.")
    print(f"Building the patched llama.cpp runtime ({args.runtime}); this may take several minutes.", flush=True)
    with tempfile.TemporaryDirectory(prefix="kvpark-runtime-") as tmp:
        work = Path(tmp)
        if args.package and args.package.is_dir():
            source = args.package.expanduser().resolve()
        else:
            archive = release_asset(args.version, f"kvpark-{args.version}.tar.gz", work)
            with tarfile.open(archive) as files:
                files.extractall(work / "source", filter="data")
            source = work / "source" / f"kvpark-{args.version}"
        flags = ["--cmake-arg=-DGGML_METAL=OFF", "--cmake-arg=-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"]
        if args.runtime != "cpu":
            flags.append("--cmake-arg=-DGGML_" + args.runtime.upper() + "=ON")
        run([sys.executable, source / "scripts/build_runtime.py", "--output", work / "runtime", *flags])
        built = work / "runtime/build/bin"
        run([built / "llama-server", "--version"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"Runtime destination already exists: {destination}. Use a new --install-dir.")
        with tempfile.TemporaryDirectory(prefix=".kvpark-runtime-", dir=destination.parent) as staging:
            staged = Path(staging) / "bin"
            shutil.copytree(built, staged, symlinks=True)
            run([staged / "llama-server", "--version"])
            staged.rename(destination)
    run([binary, "--version"])
    return binary


def install_directory():
    if os.environ.get("XDG_DATA_HOME"):
        return Path(os.environ["XDG_DATA_HOME"]).expanduser() / "kvpark/install"
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/kvpark/install"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "kvpark/install"


def check_launcher(bin_dir):
    launcher = bin_dir / "kvpark"
    marker = "# kvpark installer launcher"
    if launcher.exists() or launcher.is_symlink():
        if launcher.is_symlink() or marker not in launcher.read_text(errors="replace"):
            raise RuntimeError(f"{launcher} already belongs to another installation. Use --bin-dir to choose another directory.")
    return launcher


def expose_command(python, bin_dir, modify_path):
    launcher = check_launcher(bin_dir)
    marker = "# kvpark installer launcher"
    bin_dir.mkdir(parents=True, exist_ok=True)
    text = f"#!/bin/sh\n{marker}\nexec {shlex.quote(str(python))} -I -m kvpark \"$@\"\n"
    with tempfile.NamedTemporaryFile(mode="w", prefix=".kvpark-", dir=bin_dir, delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    try:
        temporary.chmod(0o755)
        temporary.replace(launcher)
    finally:
        temporary.unlink(missing_ok=True)
    if modify_path:
        shell = Path(os.environ.get("SHELL", "/bin/bash")).name
        if shell == "fish":
            profile = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "fish/conf.d/kvpark.fish"
            line = "fish_add_path -- " + shlex.quote(str(bin_dir))
        elif shell in {"bash", "zsh", "sh", "dash"}:
            if shell == "zsh":
                profile = Path(os.environ.get("ZDOTDIR", Path.home())).expanduser() / ".zshrc"
            else:
                bash_profile = ".bash_profile" if sys.platform == "darwin" else ".bashrc"
                profile = Path.home() / (bash_profile if shell == "bash" else ".profile")
            line = f"export PATH={shlex.quote(str(bin_dir))}:\"$PATH\""
        else:
            print(f"Add {bin_dir} to your shell's PATH to use kvpark by name.")
            return launcher
        current = profile.read_text() if profile.exists() else ""
        if line not in current.splitlines():
            profile.parent.mkdir(parents=True, exist_ok=True)
            with profile.open("a") as stream:
                stream.write(f"\n# kvpark\n{line}\n")
            print(f"Added the command directory to {profile}.")
    return launcher


def launch_setup(python, *, hermes):
    command = [str(python), "-I", "-m", "kvpark", "setup", "--hermes" if hermes else "--standalone"]
    available = run([python, "-I", "-c", "import importlib.util; print(bool(importlib.util.find_spec('kvpark.setup')))"], capture=True)
    if available != "True":
        print("This older release does not include the setup wizard. Install the current release for guided setup.")
        return
    try:
        terminal = open("/dev/tty", "r+b", buffering=0)
    except OSError:
        print("To choose a model later, run: " + shlex.join(command))
        return
    with terminal:
        result = subprocess.run(command, stdin=terminal, stdout=terminal, stderr=terminal)
    if result.returncode:
        print("kvpark is installed. Setup can be retried with: " + shlex.join(command))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--hermes", action="store_true")
    mode.add_argument("--standalone", action="store_true")
    parser.add_argument("--hermes-python", type=Path)
    parser.add_argument("--install-dir", type=Path, default=install_directory())
    parser.add_argument("--bin-dir", type=Path, default=Path.home() / ".local/bin")
    parser.add_argument("--runtime", choices=("cpu", "metal", "cuda", "hip", "vulkan"), help="also build the managed llama.cpp runtime (requires a C++ toolchain)")
    parser.add_argument("--no-setup", action="store_true", help="install only; open the model chooser later")
    parser.add_argument("--no-path", action="store_true")
    parser.add_argument("--version", default=RELEASE)
    parser.add_argument("--package", type=Path, help="local wheel or source checkout for development")
    parser.add_argument("--uv", default=shutil.which("uv"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9][0-9A-Za-z.\-]*", args.version):
        parser.error("--version must be a release version such as 0.3.0a1")
    if args.standalone and args.hermes_python:
        parser.error("--hermes-python cannot be combined with --standalone")
    if not args.uv:
        parser.error("Run bash install.sh to set up the required Python installation tool.")
    check_launcher(args.bin_dir.expanduser().absolute())
    hermes = None if args.standalone else find_hermes(args.hermes_python)
    if args.hermes and not hermes:
        raise RuntimeError("Hermes was not found. Install Hermes first, or specify --hermes-python PATH for a custom installation.")
    root = args.install_dir.expanduser().absolute()
    if hermes:
        python, info = hermes
        print(f"Found Hermes. Installing into {python}", flush=True)
        if info["prefix"] == info["base"]:
            raise RuntimeError("Hermes is using a system Python. Select its virtual environment with --hermes-python; system packages will not be changed.")
        if tuple(info["version"]) < (3, 10):
            raise RuntimeError("Hermes' Python is too old for kvpark (requires Python 3.10+). Update Hermes first.")
        if info["legacy"]:
            raise RuntimeError("Found sloth-memory in Hermes. Use the rename migration in docs/updates.md before enabling kvpark; existing caches and routes need to be preserved.")
    else:
        python = root / "venv/bin/python"
        info = probe(python)
        print(f"Installing standalone kvpark into {root}", flush=True)
        if not info:
            if (root / "venv").exists():
                raise RuntimeError(f"The environment at {root / 'venv'} is incomplete or was moved. Choose a new --install-dir to keep the old files and create a working install.")
            root.mkdir(parents=True, exist_ok=True)
            run([args.uv, "venv", "--quiet", "--seed", "--python", "3.12", str(root / "venv")])
            info = probe(python)
    if not info:
        raise RuntimeError("The Python environment could not be started. Rerun with a new --install-dir.")
    if info["kvpark"] and info["kvpark"] != args.version and not args.package:
        raise RuntimeError(f"kvpark {info['kvpark']} is already installed. Use its coordinated update command to preserve running services: {shlex.quote(str(python))} -m kvpark update")
    if info["kvpark"] == args.version and not args.package:
        print(f"kvpark {args.version} is already installed; checking its environment.", flush=True)
        run([args.uv, "pip", "install", "--quiet", "--python", python, "pip>=23"])
    else:
        with tempfile.TemporaryDirectory(prefix="kvpark-package-") as tmp:
            package = args.package.expanduser().resolve(strict=True) if args.package else release_asset(args.version, f"kvpark-{args.version}-py3-none-any.whl", Path(tmp))
            print("Installing kvpark and its dependencies...", flush=True)
            # An explicit interpreter avoids the active shell's unrelated environment.
            # pip is needed by kvpark's coordinated updater even in uv-created venvs.
            run([args.uv, "pip", "install", "--quiet", "--python", python, "pip>=23", str(package)])
    run([python, "-I", "-c", '''
from importlib.metadata import distribution
import kvpark, kvpark.cli, kvpark.hermes, psutil, packaging
d = distribution("kvpark")
assert any(e.name == "kvpark" and e.value == "kvpark.hermes" for e in d.entry_points if e.group == "hermes_agent.plugins"), "Hermes entry point is missing"
print("Verified kvpark " + d.version)
'''])
    if hermes:
        run([python, "-I", "-m", "hermes_cli.main", "plugins", "enable", "kvpark", "--no-allow-tool-override"])
        # Read through Hermes' config API to verify enablement in the selected profile.
        run([python, "-I", "-c", '''
from hermes_cli.config import load_config
plugins = load_config().get("plugins", {})
assert "kvpark" in plugins.get("enabled", []), "Hermes did not enable kvpark"
assert "kvpark" not in plugins.get("disabled", []), "Hermes still disables kvpark"
'''])
    launcher = expose_command(python, args.bin_dir.expanduser().absolute(), not args.no_path)
    binary = build_runtime(args, root) if args.runtime else None
    version = run([launcher, "--version"], capture=True)
    print(f"\nkvpark {version} installed successfully.", flush=True)
    print(f"Works now: {shlex.quote(str(launcher))} --help")
    if not args.no_path:
        print("In a new terminal: kvpark --help")
    if hermes:
        print("\nNext: restart Hermes, then type /kvpark setup")
        print("Setup finds running model servers and asks you to choose a model by number.")
    else:
        print("\nChoose your model with: " + shlex.quote(str(launcher)) + " setup")
    if binary:
        print("\nRuntime ready. Supply your GGUF model:")
        if hermes:
            print(f"  /kvpark setup --server {shlex.quote(str(binary))} --model /path/to/your-model.gguf")
        else:
            print(f"  {shlex.quote(str(launcher))} backend --server {shlex.quote(str(binary))} --model /path/to/your-model.gguf -- --ctx-size 8192 --cache-ram 0 --ctx-checkpoints 8 --checkpoint-min-step 128 --jinja")
            print(f"Then, in another terminal: {shlex.quote(str(launcher))} serve")
    else:
        print("Disk park/resume needs the patched llama.cpp runtime and your GGUF model.")
        print("Rerun this installer with --runtime cpu to build it, or select metal/cuda/hip/vulkan for your GPU toolchain.")
    if not args.no_setup:
        launch_setup(python, hermes=bool(hermes))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"\nkvpark installation stopped: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("\nInstallation interrupted. Rerun the same command to continue.", file=sys.stderr)
        raise SystemExit(130) from None
