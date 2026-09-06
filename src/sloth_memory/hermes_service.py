"""Start or attach to local sloth-memory services when Hermes loads its plugin."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .cli import request
from .runtime import directory


def defaults():
    return dict(base_url="http://127.0.0.1:8080", upstream_port=8090, archive_dir=str(directory()),
                autostart=True, external=False, server="", model="",
                backend_args=["--ctx-size", "8192", "--cache-ram", "0", "--ctx-checkpoints", "8",
                              "--checkpoint-min-step", "128", "--jinja"])


def validate_service(values):
    result = {**defaults(), **values}
    if set(values) - set(defaults()):
        raise ValueError("unknown service setting")
    parsed = urlsplit(result["base_url"])
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in ("", "/", "/v1", "/v1/")):
        raise ValueError("base URL must be http://127.0.0.1:<port>, optionally ending in /v1")
    port = parsed.port or 80
    if type(result["upstream_port"]) is not int or not 1 <= result["upstream_port"] <= 65535 or port == result["upstream_port"]:
        raise ValueError("upstream port must be valid and different from the proxy port")
    result["base_url"] = f"http://127.0.0.1:{port}"
    result["archive_dir"] = str(directory(result["archive_dir"]))
    for key in ("autostart", "external"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be true or false")
    for key in ("model", "server"):
        if not isinstance(result[key], str):
            raise ValueError(f"{key} must be a path")
        if result[key]:
            path = Path(result[key]).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"{key} must be a local file")
            result[key] = str(path)
    if not isinstance(result["backend_args"], list) or any(not isinstance(arg, str) for arg in result["backend_args"]):
        raise ValueError("backend_args must be a list of arguments")
    return result


class Service:
    def __init__(self, settings):
        self.settings = settings
        self.error = None
        self.worker = None
        self.lock = threading.Lock()
        self.closed = threading.Event()

    def start_async(self):
        with self.lock:
            if self.closed.is_set() or self.worker and self.worker.is_alive():
                return
            self.worker = threading.Thread(target=self._start, name="sloth-memory-start", daemon=True)
            self.worker.start()

    def _start(self):
        try:
            self.ensure()
            self.error = None
        except (OSError, ValueError, RuntimeError) as exc:
            self.error = str(exc)

    def _existing(self, config):
        try:
            status = request(config["base_url"], "status", timeout=2)
        except URLError:
            return None
        if status.get("service") != "sloth-memory" or status.get("control_version", 0) < 2:
            raise RuntimeError("this port is occupied by an incompatible service; choose another base URL or update it")
        if not config["external"] and Path(status["archive_dir"]).resolve() != Path(config["archive_dir"]):
            raise RuntimeError("the running proxy uses a different archive directory; choose a separate port")
        if status.get("upstream") != f"127.0.0.1:{config['upstream_port']}" and not config["external"]:
            raise RuntimeError("the running proxy uses a different upstream port")
        return status

    def ensure(self):
        config = validate_service(self.settings())
        if self.closed.is_set():
            raise RuntimeError("Hermes adapter has been unloaded")
        existing = self._existing(config)
        if existing and (config["external"] or self._health(config["upstream_port"])):
            return existing
        if config["external"]:
            raise RuntimeError("external proxy is unavailable; start it or configure a managed backend")
        if not config["server"] or not config["model"]:
            raise ValueError("Run /sloth-memory setup --server /path/to/llama-server --model /path/to/model.gguf")
        from .runtime import backend_command
        archive = Path(config["archive_dir"])
        backend_command(config["server"], config["model"], archive, config["upstream_port"], config["backend_args"])
        archive.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (archive / ".hermes-start.lock").open("a") as start_lock:
            # Multiple Hermes surfaces can load the same enabled plugin at once.
            fcntl.flock(start_lock, fcntl.LOCK_EX)
            existing = self._existing(config)
            if existing and self._health(config["upstream_port"]):
                return existing
            if self.closed.is_set():
                raise RuntimeError("Hermes adapter has been unloaded")
            backend_running = False
            # A surviving managed backend can be reused after a proxy crash.
            try:
                with self._health_response(config["upstream_port"]):
                    backend_running = self._owned_backend(config)
                    if not backend_running:
                        raise RuntimeError("upstream port is already in use; configure external mode or choose another port")
            except URLError as exc:
                if hasattr(exc, "code"):
                    raise RuntimeError("upstream port is already in use") from exc
            started = []
            try:
                if not backend_running:
                    backend = self._spawn(["backend", "--server", config["server"], "--model", config["model"],
                        "--archive-dir", str(archive), "--port", str(config["upstream_port"]), "--", *config["backend_args"]], archive / "hermes-backend.log")
                    started.append(backend)
                    self._wait(backend, lambda: self._health(config["upstream_port"]), archive / "hermes-backend.log")
                if not existing:
                    proxy = self._spawn(["serve", "--archive-dir", str(archive), "--port", str(urlsplit(config["base_url"]).port),
                        "--upstream-port", str(config["upstream_port"])], archive / "hermes-proxy.log")
                    started.append(proxy)
                    self._wait(proxy, lambda: self._existing(config), archive / "hermes-proxy.log")
                return self._existing(config)
            except BaseException:
                for process in reversed(started):
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                raise

    @staticmethod
    def _owned_backend(config):
        from .runtime import backend_command, file_identity
        try:
            archive = Path(config["archive_dir"])
            manifest = json.loads((archive / "runtime.json").read_text())
            proc = Path(f"/proc/{manifest['pid']}")
            ticks = (proc / "stat").read_text().rsplit(") ", 1)[1].split()[19]
            argv = (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
            return (ticks == manifest["start_ticks"] and argv == backend_command(
                config["server"], config["model"], archive, config["upstream_port"], config["backend_args"])
                and all(file_identity(Path(path)) == expected for path, expected in manifest["files"].items()))
        except (OSError, ValueError, KeyError):
            return False

    @staticmethod
    def _spawn(argv, log):
        if log.exists() and log.stat().st_size > 1024**2:
            log.replace(log.with_suffix(".previous.log"))
        with log.open("ab") as output:
            process = subprocess.Popen([sys.executable, "-m", "sloth_memory", *argv], stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        # The service survives Hermes closing so scheduled cleanup keeps running.
        threading.Thread(target=process.wait, name="sloth-memory-reaper", daemon=True).start()
        return process

    @staticmethod
    def _health_response(port):
        headers = {}
        key_file = os.environ.get("SLOTH_UPSTREAM_KEY_FILE")
        if key_file:
            headers["Authorization"] = "Bearer " + Path(key_file).read_text().strip()
        return urlopen(Request(f"http://127.0.0.1:{port}/health", headers=headers), timeout=2)

    @classmethod
    def _health(cls, port):
        try:
            with cls._health_response(port) as response:
                return json.load(response).get("status") == "ok"
        except URLError:
            return False

    def _wait(self, process, probe, log):
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline and not self.closed.is_set():
            if process.poll() is not None:
                raise RuntimeError(f"service exited; inspect {log}")
            if probe():
                return
            self.closed.wait(.2)
        raise RuntimeError(f"service startup interrupted or timed out; inspect {log}")
