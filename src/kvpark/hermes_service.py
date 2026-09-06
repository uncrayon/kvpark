"""Start or attach to local kvpark services when Hermes loads its plugin."""

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

from .backends import Backend, DEFAULT_URLS, normalize_url
from .cli import request
from .network import managed_port, proxy_record
from .platforms import detached_options, file_lock, process_matches
from .runtime import directory


def defaults():
    return dict(base_url="http://127.0.0.1:8080", upstream_port=8090, archive_dir=str(directory()),
                autostart=True, external=False, server="", model="", backend="llama.cpp",
                upstream_url="", cache_mode="native", connected=False, uninstalled=False, previous_urls=[],
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
    if type(result["upstream_port"]) is not int or not 1 <= result["upstream_port"] <= 65535:
        raise ValueError("upstream port must be valid")
    result["base_url"] = f"http://127.0.0.1:{port}"
    result["archive_dir"] = str(directory(result["archive_dir"]))
    if not isinstance(result["previous_urls"], list) or any(not isinstance(url, str) for url in result["previous_urls"]):
        raise ValueError("previous_urls must be a list of proxy addresses")
    for url in result["previous_urls"]:
        parsed_previous = urlsplit(normalize_url(url))
        if parsed_previous.scheme != "http" or parsed_previous.hostname != "127.0.0.1":
            raise ValueError("previous proxy URLs must be local")
    for key in ("autostart", "external", "connected", "uninstalled"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be true or false")
    for key in ("model", "server"):
        if not isinstance(result[key], str):
            raise ValueError(f"{key} must be a string")
        if result[key] and (key == "server" or result["backend"] == "llama.cpp" and not result["upstream_url"]):
            path = Path(result[key]).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"{key} must be a local file")
            result[key] = str(path)
    if not isinstance(result["backend_args"], list) or any(not isinstance(arg, str) for arg in result["backend_args"]):
        raise ValueError("backend_args must be a list of arguments")
    if result["upstream_url"]:
        result["upstream_url"] = normalize_url(result["upstream_url"])
    target(result)  # validate backend and native-snapshot boundary
    return result


def target(config):
    url = config["upstream_url"] or (f"http://127.0.0.1:{config['upstream_port']}" if config["backend"] == "llama.cpp"
                                      else DEFAULT_URLS.get(config["backend"], ""))
    return Backend(config["backend"], url, config["cache_mode"])


class Service:
    def __init__(self, settings, persist=None):
        self.settings = settings
        self.persist = persist
        self.error = None
        self.worker = None
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.resolved = None

    def start_async(self):
        with self.lock:
            if self.closed.is_set() or self.worker and self.worker.is_alive():
                return
            self.worker = threading.Thread(target=self._start, name="kvpark-start", daemon=True)
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
        except (URLError, RuntimeError, ValueError, TimeoutError):
            return None
        if not isinstance(status, dict) or status.get("service") != "kvpark" or status.get("control_version", 0) < 2:
            return None
        if config["external"]:
            return status
        if Path(status.get("archive_dir", "")).resolve() != Path(config["archive_dir"]).resolve():
            return None
        upstream = status.get("upstream_url", "http://" + status.get("upstream", ""))
        if upstream != target(config).url:
            return None
        caps = status.get("capabilities", {"backend": "llama.cpp", "cache_mode": "native"})
        return status if (caps.get("backend"), caps.get("cache_mode")) == (config["backend"], config["cache_mode"]) else None

    def _remember(self, config):
        previous = self.settings()["base_url"]
        if previous != config["base_url"]:
            config["previous_urls"] = list(dict.fromkeys([*config["previous_urls"], previous]))[-8:]
        self.resolved = config
        if self.persist:
            self.persist(config)
        self.error = None

    def ensure(self):
        if self.settings().get("uninstalled", False):
            raise RuntimeError("kvpark was uninstalled; follow docs/uninstall.md to reinstall")
        config = validate_service(self.settings())
        if self.closed.is_set():
            raise RuntimeError("Hermes adapter has been unloaded")
        if config["external"]:
            existing = self._existing(config)
            if not existing:
                raise RuntimeError("external proxy is unavailable or incompatible")
            self._remember(config)
            return existing
        archive = Path(config["archive_dir"])
        with file_lock(archive / ".hermes-start.lock"):
            if self.closed.is_set() or self.settings().get("uninstalled", False):
                raise RuntimeError("kvpark startup was disabled while waiting for the archive")
            # The process that won startup publishes the actual bound address.
            # Read it again under the interprocess lock, including after a reboot.
            record = proxy_record(archive)
            if record:
                expected = target(config)
                if (record["backend"], record["cache_mode"]) != (expected.name, expected.cache_mode):
                    raise RuntimeError("this archive is already serving another backend; use a separate archive directory")
                if config["upstream_url"] and record["upstream_url"] != expected.url:
                    raise RuntimeError("this archive is already serving another upstream; use a separate archive directory")
                config["base_url"] = record["base_url"]
                if config["backend"] == "llama.cpp" and not config["upstream_url"]:
                    config["upstream_port"] = urlsplit(record["upstream_url"]).port
            elif config["backend"] == "llama.cpp" and not config["upstream_url"]:
                actual = managed_port(archive)
                if actual:
                    config["upstream_port"] = actual
            existing = self._existing(config)
            if existing:
                if config["server"] and config["model"] and config["backend"] == "llama.cpp" and not config["upstream_url"]:
                    if not self._owned_backend(config):
                        raise RuntimeError("the running backend uses another model/configuration; use a separate archive directory")
                self._remember(config)
                return existing
            if record:
                raise RuntimeError("the archive's proxy is running but unavailable; inspect hermes-proxy.log")
            managed = config["backend"] == "llama.cpp" and not config["upstream_url"]
            if managed and (not config["server"] or not config["model"]):
                raise ValueError("Run /kvpark setup --server /path/to/llama-server --model /path/to/model.gguf")
            if not managed and not config["model"]:
                raise ValueError("Set --model to the model name served by your backend")
            if managed:
                from .runtime import backend_command
                backend_command(config["server"], config["model"], archive, config["upstream_port"], config["backend_args"])
            started = []
            try:
                if managed:
                    port = managed_port(archive)
                    if port:
                        config["upstream_port"] = port
                        if not self._owned_backend(config):
                            raise RuntimeError("another model/configuration owns this archive; use a separate archive directory")
                    else:
                        child = self._spawn(["backend", "--server", config["server"], "--model", config["model"],
                            "--archive-dir", str(archive), "--port", str(config["upstream_port"]), "--", *config["backend_args"]], archive / "hermes-backend.log")
                        started.append(child)
                        def ready():
                            actual = managed_port(archive)
                            if actual:
                                config["upstream_port"] = actual
                                return self._health(actual)
                            return False
                        self._wait(child, ready, archive / "hermes-backend.log")
                else:
                    self._probe(target(config))
                child = self._spawn(["serve", "--archive-dir", str(archive), "--port", str(urlsplit(config["base_url"]).port),
                    "--backend", config["backend"], "--cache-mode", config["cache_mode"],
                    "--upstream-url", target(config).url], archive / "hermes-proxy.log")
                started.append(child)
                def ready_proxy():
                    actual = proxy_record(archive)
                    if actual:
                        config["base_url"] = actual["base_url"]
                        return self._existing(config)
                    return None
                self._wait(child, ready_proxy, archive / "hermes-proxy.log")
                self._remember(config)
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
            return (process_matches(manifest) and manifest["argv"] == backend_command(
                config["server"], config["model"], archive, config["upstream_port"], config["backend_args"])
                and all(file_identity(Path(path)) == expected for path, expected in manifest["files"].items()))
        except (OSError, ValueError, KeyError):
            return False

    @staticmethod
    def _spawn(argv, log):
        if log.exists() and log.stat().st_size > 1024**2:
            log.replace(log.with_suffix(".previous.log"))
        with log.open("ab") as output:
            process = subprocess.Popen([sys.executable, "-m", "kvpark", *argv], stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, **detached_options())
        threading.Thread(target=process.wait, name="kvpark-reaper", daemon=True).start()
        return process

    @staticmethod
    def _probe(backend):
        headers = {}
        key_file = os.environ.get("KVPARK_UPSTREAM_KEY_FILE")
        if key_file:
            headers["Authorization"] = "Bearer " + Path(key_file).read_text().strip()
        with urlopen(Request(backend.url + backend.health_path, headers=headers), timeout=5) as response:
            return json.load(response)

    @classmethod
    def _health(cls, port):
        try:
            return cls._probe(Backend("llama.cpp", f"http://127.0.0.1:{port}" )).get("status") == "ok"
        except (URLError, ValueError):
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
