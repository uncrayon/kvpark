# Backends and operating systems

This describes kvpark alpha `0.3.0a1`. Existing sloth-memory users should follow
the [rename migration](updates.md#migrate-from-sloth-memory).

## Capability contract

| Backend | Chat proxy and Hermes routing | kvpark disk park/resume |
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
manages files owned by kvpark, never a backend's independent prefix cache.

The proxy and Python runtime target Linux, macOS, and Windows. CI runs the Python
suite and builds the patched llama.cpp runtime on all three systems. The backend
still has its own OS/hardware requirements; a portable proxy cannot make an
unsupported inference engine run on your hardware. A remote HTTP(S) upstream can
be used in routing mode, including a backend on another machine or in WSL.

## Connect an existing server

Start the inference server normally using its own installation instructions.
Then configure Hermes with the **exact model ID served by that server**. Use
[the model-ID lookup below](#find-your-model-id) before replacing the placeholders:

```text
/kvpark setup --backend ollama --upstream-url http://127.0.0.1:11434 --model "your-model:tag"
/kvpark setup --backend vllm --upstream-url http://127.0.0.1:8000 --model "your-served-model"
/kvpark setup --backend mlx --upstream-url http://127.0.0.1:8081 --model "your-mlx-model"
```

Choose one backend per archive directory. To run another stack, include
`--archive-dir "/path/to/another/archive"`. Setup starts the proxy and saves its
selected URL into Hermes' model configuration. No sudo, packet interception,
firewall edits, or privileged ports are required:

```text
Hermes → kvpark (selected localhost port) → your inference server
```

An existing unpatched llama.cpp server can be attached explicitly:

```text
/kvpark setup --backend llama.cpp --cache-mode routing --upstream-url http://127.0.0.1:8090 --model "your-served-alias"
```

For native disk snapshots, build the included runtime and use the existing
`--server /path/to/llama-server --model /path/to/model.gguf` setup without
`--upstream-url`. On Windows the binary is `llama-server.exe`, often under
`runtime/build/bin/Release`; on macOS/Linux it is `runtime/build/bin/llama-server`.

Standalone example, without Hermes:

```bash
kvpark serve --backend ollama --upstream-url http://127.0.0.1:11434
kvpark doctor
```

Both commands use the same default archive directory. With a custom directory,
pass `--archive-dir` to both. `--url` remains an explicit override for controls.
The proxy accepts chat completions, model discovery, and health checks; backend
management APIs and other generation APIs are not exposed.

## Find your model ID

The model ID is the string the backend expects in a chat request's `model` field.
It may be a serving alias, a repository name, a tagged name, or a full file path.
Copy it exactly, including capitalization, slashes, and tags; a display name such
as “Gemma” is not enough unless you configured that exact alias.

Start the backend, then query its **own address**, before connecting kvpark. Run
the appropriate command from the machine running Hermes:

| Backend | List models (example address) | Value to use for kvpark's `--model` |
| --- | --- | --- |
| Existing llama.cpp | `curl -fsS http://127.0.0.1:8090/v1/models` | An `id` inside `data`; usually the `--alias` value, otherwise the model file path |
| Ollama | `curl -fsS http://127.0.0.1:11434/v1/models` | An `id` inside `data`, including its tag, such as `example-model:latest` |
| vLLM | `curl -fsS http://127.0.0.1:8000/v1/models` | An `id` inside `data`; `--served-model-name` if configured, otherwise the model name/path passed to vLLM |
| MLX-LM | `curl -fsS http://127.0.0.1:8081/v1/models` | An `id` inside `data`, such as a Hugging Face repository ID or an absolute local model directory |

The ports above are examples from this guide, not guaranteed backend defaults.
Use the port shown in your server's startup output. If it runs on another machine,
replace `127.0.0.1` with its reachable address; localhost always means the machine
where you run the command. In Windows PowerShell, use `curl.exe` to avoid older
PowerShell's `curl` alias. These commands do not need `jq`.

For example, if your vLLM server returns:

```json
{
  "object": "list",
  "data": [{"id": "my-gemma", "object": "model"}]
}
```

Use the **`id` string `my-gemma`**, not the entire JSON object:

```text
/kvpark setup --backend vllm --cache-mode routing --upstream-url http://127.0.0.1:8000 --model "my-gemma"
/kvpark status
```

`my-gemma` is an example alias, not a model to download. If you choose to set that
alias when starting vLLM, add `--served-model-name my-gemma` to your working serve
command, then query `/v1/models` again to confirm it.

For Ollama, `ollama list` is another option when the CLI is connected to the same
server: copy the **NAME** column, including `:tag`. Do not copy its hexadecimal
**ID** column; that identifies the model artifact, not the API model name.

MLX-LM's list can include downloaded models that are not currently loaded. Choose
the model you intend to serve; a listing alone does not prove that it can load or
generate successfully. A locally served model can appear as an absolute directory
path, so keep the quotes around `--model` when the path contains spaces.

For kvpark's **managed native llama.cpp setup**, `--model` instead takes your local
GGUF path, as shown in the [Hermes guide](hermes.md#guided-setup). kvpark starts that
backend with the alias `local` and selects it for Hermes automatically.

If discovery fails:

- **Connection refused or timeout:** check that the server is running and that
  Hermes can reach its host and port.
- **401 or 403:** use the backend's required authentication. For a Bearer-protected
  endpoint, supply an `Authorization: Bearer …` header; keep the credential out
  of chat and committed files. See [upstream credentials](#port-selection-and-routing)
  when configuring kvpark to connect to that server.
- **404 or HTML response:** check that this is the inference API address, not a
  web UI, and that `/v1` was not added twice.
- **Empty `data` list:** check the backend's model configuration/downloads and
  logs. Do not invent an ID or assume the first model is already loaded.

Backend references: [llama.cpp model information](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility),
[vLLM serving arguments](https://docs.vllm.ai/en/latest/cli/serve/),
and [MLX-LM model listing implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py).

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
upstream credential file (`KVPARK_UPSTREAM_KEY_FILE`); the proxy's optional
`KVPARK_API_KEY` is separate. Credentials are not stored in service metadata.

## Platform storage

Defaults follow `XDG_DATA_HOME` when set; otherwise:

- Linux: `~/.local/share/kvpark`
- macOS: `~/Library/Application Support/kvpark`
- Windows: `%LOCALAPPDATA%\kvpark`

These defaults apply to new installations. The [sloth-memory migration](updates.md#migrate-from-sloth-memory)
keeps the existing archive directory, even when its name still contains `sloth-memory`.
Do not move it to match the new brand: its path is part of runtime compatibility.
Explicit `--archive-dir` keeps an existing installation in place. Portable process
identity uses psutil; Windows uses native byte-range file locks and a Job Object
for managed child lifetime. Metadata files are flushed before atomic replacement.
Windows does not provide directory fsync through Python; it has weaker power-loss
durability than the Unix directory-flush path.

## Remaining backend work

Full disk resume across all four engines is **not implemented**. The next backend
work needs real engine-side integration and model/hardware acceptance tests:

- Ollama's documented OpenAI API provides chat routing, but not kvpark's
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
