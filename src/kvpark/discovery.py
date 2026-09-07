"""Read-only discovery of model APIs on local listening ports."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import http.client
import ipaddress
import json
from urllib.parse import urlsplit

import psutil

from .backends import normalize_url

COMMON_PORTS = (11434, 8000, 8080, 8081, 8090, 1234, 5000)
MAX_SERVERS = 64
MAX_BODY = 256 * 1024
LABELS = {"ollama": "Ollama", "llama.cpp": "llama.cpp", "vllm": "vLLM", "mlx": "MLX", "openai": "Model server"}


@dataclass(frozen=True)
class Model:
    url: str
    model: str
    backend: str = "openai"

    @property
    def label(self):
        # Model IDs remain exact in configuration; labels cannot inject terminal controls.
        name = "".join(c if c.isprintable() else " " for c in self.model)[:160]
        return f"{name} — {LABELS[self.backend]} ({self.url})"


@dataclass
class Scan:
    models: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def local_url(url):
    try:
        parsed = urlsplit(normalize_url(url))
        return parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


def connections():
    try:
        yield from psutil.net_connections(kind="inet")
        return
    except (psutil.Error, OSError):
        pass
    # macOS can deny system-wide discovery while allowing the user's own processes.
    try:
        for process in psutil.process_iter():
            try:
                method = getattr(process, "net_connections", None) or process.connections
                yield from method(kind="inet")
            except (psutil.Error, OSError):
                continue
    except (psutil.Error, OSError):
        pass


def listening_urls(extra=()):
    # Inspect listeners instead of sweeping every port or probing a LAN.
    urls = [normalize_url(url) for url in extra if url and local_url(url)]
    urls += [f"http://127.0.0.1:{port}" for port in COMMON_PORTS]
    try:
        for connection in connections():
            if connection.status != psutil.CONN_LISTEN or not connection.laddr:
                continue
            address = ipaddress.ip_address(connection.laddr.ip)
            if address.is_loopback or address.is_unspecified:
                host = "[::1]" if address.version == 6 else "127.0.0.1"
                urls.append(f"http://{host}:{connection.laddr.port}")
    except (psutil.Error, OSError):
        # macOS and restricted containers may hide listeners; familiar ports still work.
        pass
    return list(dict.fromkeys(urls))[:MAX_SERVERS]


def get_json(url, path, timeout=.5):
    parsed = urlsplit(url)
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    try:
        # No proxy environment, credentials, redirects, generation, or model-loading requests.
        connection.request("GET", path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            return response.status, None
        body = response.read(MAX_BODY + 1)
        if len(body) > MAX_BODY:
            return response.status, None
        return response.status, json.loads(body)
    except (OSError, ValueError, http.client.HTTPException):
        return None, None
    finally:
        connection.close()


def inspect_server(url):
    url = normalize_url(url)
    status, data = get_json(url, "/v1/models")
    if status in (401, 403):
        return Scan(notes=[f"{url}: requires an API key; no models were read."])
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        status, tags = get_json(url, "/api/tags")
        if isinstance(tags, dict) and isinstance(tags.get("models"), list):
            names = [item.get("name") for item in tags["models"] if isinstance(item, dict)]
            models = [Model(url, name, "ollama") for name in names if isinstance(name, str) and name]
            return Scan(models, [] if models else [f"{url}: Ollama has no downloaded models yet."])
        return Scan()
    # A kvpark proxy advertises its upstream's models; choosing it would chain proxies.
    for path in ("/_kvpark/status", "/_sloth/status"):
        _, state = get_json(url, path)
        if isinstance(state, dict) and state.get("service") in {"kvpark", "sloth-memory"}:
            return Scan()
    owners = {str(item.get("owned_by", "")).lower() for item in data["data"] if isinstance(item, dict)}
    backend = next((name for owner, name in (("ollama", "ollama"), ("vllm", "vllm"),
                    ("llamacpp", "llama.cpp"), ("llama.cpp", "llama.cpp"), ("mlx", "mlx")) if owner in owners), "openai")
    names = list(dict.fromkeys(item["id"] for item in data["data"]
                             if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]))
    return Scan([Model(url, name, backend) for name in names],
                [] if names else [f"{url}: server found, but no model is loaded. Load one in your model app and scan again."])


def discover(extra=()):
    urls = listening_urls(extra)
    result = Scan()
    with ThreadPoolExecutor(max_workers=16) as pool:
        for server in pool.map(inspect_server, urls):
            result.models.extend(server.models)
            result.notes.extend(server.notes)
    result.models = list(dict.fromkeys(result.models))
    return result


def verify(model):
    found = inspect_server(model.url)
    if not any(item.model == model.model for item in found.models):
        raise ValueError("That model is no longer available. Start or load it in your model app, then scan again.")
