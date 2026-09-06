# Backends and operating systems

This describes alpha `0.2.0a1`. Older alpha.2 installations retain their original
Linux-only scope until upgraded.

## Capability contract

| Backend | Chat proxy and Hermes routing | sloth-memory disk park/resume |
| --- | --- | --- |
| Patched llama.cpp, managed locally | Yes | Yes; validate each model/build |
| Stock llama.cpp, existing server | Yes, `--cache-mode routing` | Unavailable |
| Ollama, existing server | Yes | Unavailable through this adapter |
| vLLM, existing server | Yes | Unavailable through this adapter |
| MLX-LM, existing server | Yes | Unavailable through this adapter |

Routing preserves the served model name, messages, tool calls, and streaming.
`status` and `doctor` report `capabilities.disk_snapshots` explicitly. In routing
mode, parking returns an error instead of claiming that inference state was saved.
No transcript cache is substituted for a KV snapshot. Seven-day cleanup only
manages files owned by sloth-memory, never a backend's independent prefix cache.

The proxy and Python runtime target Linux, macOS, and Windows. CI runs the Python
suite and builds the patched llama.cpp runtime on all three systems. The backend
still has its own OS/hardware requirements; a portable proxy cannot make an
unsupported inference engine run on your hardware. A remote HTTP(S) upstream can
be used in routing mode, including a backend on another machine or in WSL.

## Connect an existing server

Start the inference server normally using its own installation instructions.
Then configure Hermes with the **exact model ID served by that server**:

```text
/sloth-memory setup --backend ollama --upstream-url http://127.0.0.1:11434 --model "your-model:tag"
/sloth-memory setup --backend vllm --upstream-url http://127.0.0.1:8000 --model "your-served-model"
/sloth-memory setup --backend mlx --upstream-url http://127.0.0.1:8081 --model "your-mlx-model"
```

Choose one backend per archive directory. To run another stack, include
`--archive-dir "/path/to/another/archive"`. Setup starts the proxy and saves its
selected URL into Hermes' model configuration. No sudo, packet interception,
firewall edits, or privileged ports are required:

```text
Hermes → sloth-memory (selected localhost port) → your inference server
```

An existing unpatched llama.cpp server can be attached explicitly:

```text
/sloth-memory setup --backend llama.cpp --cache-mode routing --upstream-url http://127.0.0.1:8090 --model "your-served-alias"
```

For native disk snapshots, build the included runtime and use the existing
`--server /path/to/llama-server --model /path/to/model.gguf` setup without
`--upstream-url`. On Windows the binary is `llama-server.exe`, often under
`runtime/build/bin/Release`; on macOS/Linux it is `runtime/build/bin/llama-server`.

Standalone example, without Hermes:

```bash
sloth-memory serve --backend ollama --upstream-url http://127.0.0.1:11434
sloth-memory doctor
```

Both commands use the same default archive directory. With a custom directory,
pass `--archive-dir` to both. `--url` remains an explicit override for controls.
The proxy accepts chat completions, model discovery, and health checks; backend
management APIs and other generation APIs are not exposed.

## Port selection and routing

The proxy tries its preferred port, then binds an OS-assigned free port if that
port is occupied or unavailable. It holds the actual listening socket while
publishing `proxy.json`, avoiding a check-then-bind race. CLI controls discover
that record; Hermes persists the selected address and reuses compatible services.
The managed llama.cpp launcher similarly selects an available backend port and
publishes it in `runtime.json`. Existing third-party servers keep their own ports.

A backend reservation must be released before launching llama.cpp, which does
not accept an inherited listening socket. If a competing bind causes startup to
exit, the launcher retries on another port, up to five attempts.

Setup selects the proxy for new Hermes sessions. For an existing session using
the configured upstream or a previously selected proxy port, middleware updates
the bound native OpenAI client's base URL before dispatch. Unrelated routes and
non-OpenAI client implementations are untouched; restart those sessions through
the configured proxy. Disabling autostart leaves already running services alive.

The proxy listens only on `127.0.0.1`. Remote upstreams can use HTTPS and an
upstream credential file (`SLOTH_UPSTREAM_KEY_FILE`); the proxy's optional
`SLOTH_API_KEY` is separate. Credentials are not stored in service metadata.

## Platform storage

Defaults follow `XDG_DATA_HOME` when set; otherwise:

- Linux: `~/.local/share/sloth-memory`
- macOS: `~/Library/Application Support/sloth-memory`
- Windows: `%LOCALAPPDATA%\sloth-memory`

Explicit `--archive-dir` keeps an existing installation in place. Portable process
identity uses psutil; Windows uses native byte-range file locks and a Job Object
for managed child lifetime. Metadata files are flushed before atomic replacement.
Windows does not provide directory fsync through Python; it has weaker power-loss
durability than the Unix directory-flush path.

## Remaining backend work

Full disk resume across all four engines is **not implemented**. The next backend
work needs real engine-side integration and model/hardware acceptance tests:

- Ollama's documented OpenAI API provides chat routing, but not sloth-memory's
  snapshot controls. `keep_alive` controls model residency, not durable snapshots.
  [Ollama API](https://docs.ollama.com/api/openai-compatibility),
  [generate parameters](https://docs.ollama.com/api/generate).
- vLLM exposes KV connector infrastructure. A connector and its storage lifecycle
  need separate integration; enabling this proxy does not configure one.
  [vLLM connectors](https://docs.vllm.ai/en/latest/features/disagg_prefill/).
- MLX-LM has prompt-cache internals, but this HTTP adapter does not own or serialize
  them. A server-side extension is needed for lifecycle control.
  [MLX-LM server source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py).

Contract tests use small HTTP fixtures for these adapters; they do not establish
real-model correctness or speedups for Ollama, vLLM, or MLX-LM.
