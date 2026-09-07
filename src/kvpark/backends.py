"""Backend transport capabilities; routing never implies disk-cache support."""

from dataclasses import dataclass
import http.client
from urllib.parse import urlsplit


BACKENDS = ("llama.cpp", "ollama", "vllm", "mlx", "openai")
DEFAULT_URLS = {"llama.cpp": "http://127.0.0.1:8090", "ollama": "http://127.0.0.1:11434",
                "vllm": "http://127.0.0.1:8000", "mlx": "http://127.0.0.1:8081", "openai": "http://127.0.0.1:8000"}


def normalize_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path.rstrip("/") not in ("", "/v1")):
        raise ValueError("upstream URL must be http(s)://host:port, optionally ending in /v1")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError("invalid upstream port")
    return f"{parsed.scheme}://{parsed.netloc}"


@dataclass(frozen=True)
class Backend:
    name: str
    url: str
    cache_mode: str = "routing"

    def __post_init__(self):
        if self.name not in BACKENDS:
            raise ValueError("backend must be one of " + ", ".join(BACKENDS))
        object.__setattr__(self, "url", normalize_url(self.url))
        if self.cache_mode not in ("native", "routing"):
            raise ValueError("cache mode must be native or routing")
        parsed = urlsplit(self.url)
        if self.cache_mode == "native" and (self.name != "llama.cpp" or parsed.hostname != "127.0.0.1" or parsed.scheme != "http"):
            raise ValueError("native snapshots require a local managed patched llama.cpp runtime")

    @property
    def snapshots(self):
        return self.cache_mode == "native"

    @property
    def health_path(self):
        return "/health" if self.name == "llama.cpp" else "/api/version" if self.name == "ollama" else "/v1/models"

    def connection(self, timeout):
        parsed = urlsplit(self.url)
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        return cls(parsed.hostname, parsed.port, timeout=timeout)

    def path(self, path):
        path, separator, query = path.partition("?")
        if path == "/health":
            path = self.health_path
        elif path in ("/models", "/chat/completions"):
            path = "/v1" + path
        return path + separator + query

    def capabilities(self):
        return dict(backend=self.name, chat_completions=True, streaming=True,
                    disk_snapshots=self.snapshots, cache_mode=self.cache_mode,
                    note=("Patched runtime required; validate park/resume with your model." if self.snapshots else
                          "Routing only: this adapter cannot park or restore disk snapshots. Backend-managed caches are independent."))
