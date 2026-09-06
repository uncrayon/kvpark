# 🦥 sloth-memory

**Let your local AI nap. Resume long conversations without repeating all the work.**

sloth-memory saves a local model's conversation inference state to disk and restores
it when you return. Built for people running open-weight models on their own hardware.
MIT licensed. No cloud service or account. The portable runtime uses psutil.

**Current `main` · development version `0.2.0.dev0`.** Supports Linux, macOS, and
Windows startup, automatic free-port selection, and routing adapters for llama.cpp,
Ollama, vLLM, and MLX-LM. Disk park/resume through this adapter still requires the
managed patched llama.cpp runtime and one slot. See [backend capabilities](docs/backends.md).
The published [alpha.2](https://github.com/uncrayon/sloth-memory/releases/tag/v0.1.0-alpha.2)
is the older Linux version; install from `main` below to test the merged changes.
The service is independent of Hermes. A native Hermes adapter provides startup,
slash-command onboarding, and cleanup controls. Other agents integrate through
an HTTP proxy and explicit session metadata; a plain Python example is included.

## What it does

- **Park a conversation:** explicitly save the currently resident conversation.
- **Resume after process restarts:** send the conversation again; compatible state
  loads automatically before inference.
- **Prove reuse:** inspect cached and newly processed token counts, plus time to
  first token for streamed requests, including queueing and restoration.
- **Protect ownership:** background calls cannot claim a foreground archive.
- **Bound storage:** seven-day retention and a 32 GiB archive budget by default.

This is an inference-state cache. **Your agent must keep and replay the transcript.**
It does not add semantic memory, expand the context window, or reconstruct a chat
from a cache file. Parking does not release the model's preallocated KV buffer.

```text
Your agent → sloth-memory (local proxy) → inference server
                     ↕
              local archive files
```

## Install from main

Requirements: Linux, macOS, or Windows, Python 3.10+, and Git. Building the managed
llama.cpp runtime also needs CMake, a C/C++ toolchain, and your own supported GGUF
model. On macOS, install the Xcode Command Line Tools for the compiler. Models
and compiled runtimes are not bundled.

For Hermes, follow [installation in Hermes' Python environment](docs/hermes.md#install-alongside-hermes)
instead of creating the standalone environment below.

Standalone installation on macOS/Linux:

```bash
git clone --branch main https://github.com/uncrayon/sloth-memory.git
cd sloth-memory
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

On Windows, use `py -3 -m venv .venv` and activate it in PowerShell with
`.\.venv\Scripts\Activate.ps1`, then run the same pip command.

To connect an existing server, continue with [backend setup](docs/backends.md#connect-an-existing-server);
no runtime build is needed for routing. To test disk park/resume, build the managed
runtime (CPU first):

```bash
python scripts/build_runtime.py --cmake-arg=-DGGML_METAL=OFF
```

The build script fetches an exact llama.cpp commit and applies the included
checkpoint persistence patch. It builds in `runtime/`; it does not modify an
existing inference installation. See [runtime compatibility](docs/compatibility.md)
for GPU builds and the current scope.

Start the backend in one terminal (on Windows, use the `llama-server.exe` path
printed by the build, usually `runtime/build/bin/Release/llama-server.exe`).
The multiline shell examples below use macOS/Linux syntax; in PowerShell, put
the command on one line and omit the continuation backslashes:

```bash
sloth-memory backend \
  --server runtime/build/bin/llama-server \
  --model /absolute/path/to/your-model.gguf \
  -- --ctx-size 8192 --cache-ram 0 --ctx-checkpoints 8 --checkpoint-min-step 128 --jinja
```

In another terminal with the same virtual environment active:

```bash
sloth-memory serve
```

Keep both terminals running. Run `sloth-memory doctor` from a third terminal with
the same environment active. Ports 8080 and 8090 are preferences; occupied ports
automatically fall back to free ones. CLI controls discover the selected proxy.

Storage defaults to `~/Library/Application Support/sloth-memory` on macOS,
`~/.local/share/sloth-memory` on Linux, or `%LOCALAPPDATA%\sloth-memory` on Windows;
`XDG_DATA_HOME` overrides these. To change it, pass the **same** `--archive-dir` to
the backend, proxy, and control commands.

## Use with Hermes

Install sloth-memory in Hermes' Python environment, enable it with
`hermes plugins enable sloth-memory`, and restart Hermes. Then run:

```text
/sloth-memory setup
```

The [Hermes setup guide](docs/hermes.md) covers selecting the backend and model.
Complete setup starts the proxy and selects its actual URL for new Hermes sessions.
The managed llama.cpp backend also starts with Hermes; existing servers remain
under your control. Services keep running after Hermes closes for cleanup. No sudo
is needed. Disk controls below require native snapshot mode.

```text
/sloth-memory park
/sloth-memory delete
/sloth-memory retention 7
/sloth-memory cleanup off
/sloth-memory cleanup on
/sloth-memory base-url http://127.0.0.1:8080
```

By default, cleanup runs hourly and expires snapshots **seven days after saving**.
It deletes disk snapshots, not transcripts or live RAM. Preferences persist across
restarts; the guide lists every command and the external-service option.

## Try it without an agent framework

The example client defaults to port 8080. If the proxy selected another port, pass
`--url http://127.0.0.1:<selected-port>` to both example invocations; find the URL in
`proxy.json` inside the archive directory.

```bash
python examples/chat.py --session cli:orchid:1 --history orchid-chat.json \
  'Our project code is ORCHID-731. Help me plan a workshop.'

sloth-memory park --session cli:orchid:1
sloth-memory status
```

Stop both services with Ctrl-C, then start them again with exactly the same
commands. Continue using the same transcript and session:

```bash
python examples/chat.py --session cli:orchid:1 --history orchid-chat.json \
  'What was the project code?'

sloth-memory status
```

Short chats demonstrate the workflow but may gain little speed. Long stable
prefixes are the intended use case. Park again before leaving to save the latest
state. Use a unique transcript file and session ID for each conversation.

## Integrate your agent

Point chat completions at the selected proxy URL plus `/v1` (normally
`http://127.0.0.1:8080/v1`; consult `proxy.json` after startup). Add these fields to each
foreground request's JSON body:

```json
{
  "slot_archive_role": "foreground",
  "slot_archive_key": "my-agent:my-project:conversation-uuid"
}
```

With clients that support it, supply these fields through `extra_body`. They are
removed before forwarding to the backend. Send `slot_archive_role: "background"`
for detached work. A user/account ID is not a conversation ID.

See the [integration contract](docs/integration.md) for branching, compaction,
streaming and authentication. The optional Hermes adapter does not patch Hermes.

## Commands

| Command | Purpose |
| --- | --- |
| `sloth-memory backend --server … --model … -- …` | Launch the backend and record runtime identity |
| `sloth-memory serve` | Run the inference proxy |
| `sloth-memory doctor` | Check backend reachability, runtime identity, and the single-slot topology |
| `sloth-memory park --session …` | Save exactly that resident conversation |
| `sloth-memory status` | Inspect archives and actual completed-request reuse |
| `sloth-memory forget --session …` | Delete that saved archive; keep the transcript |
| `sloth-memory settings --ttl-days 7 --cleanup` | Persist cleanup preferences without a restart |
| `sloth-memory cleanup` | Run cleanup now |

`doctor` does not prove checkpoint support or cache reuse. Verify the complete
flow with your own model:

```bash
python scripts/acceptance.py \
  --server runtime/build/bin/llama-server \
  --model /absolute/path/to/your-model.gguf
```

This CPU test launches private services on temporary ports, disables the RAM
prompt cache, restarts both processes, and compares restored output and token
counts against resident output. Evidence is written to `evidence/acceptance.json`.
It does not stop any existing services. Read [validation](docs/validation.md) before
interpreting timing results.

## Alpha boundaries

- Compatible model, runtime, configuration, and prompt prefix are required.
  Changing weights, builds, tools, system prompts, or reasoning settings may
  require a fresh prefill. The transcript remains the source of truth.
- Only `chat/completions` inference is supported. Responses API, multiple slots,
  and model routers are outside the current scope. Remote upstreams are supported
  in routing mode only; disk snapshots require the local managed runtime.
- CI tests startup/integration and CPU runtime builds on Linux, macOS, and Windows.
  This does not establish real-model cache reuse or GPU performance on every OS.
- All generation traffic must go through the proxy. Direct backend requests
  bypass ownership tracking and can invalidate archives.
- A conversation displaced before its first park cannot be recovered from KV
  alone. Send another foreground message, then park it.
- Archives may be several GiB each. Retention is best effort under the storage
  budget, and atomic replacement temporarily needs space for both generations.
- Treat archive files as sensitive and trusted local data. They contain model
  state and prompt tokens. See [security](SECURITY.md).

## Contribute

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Useful early contributions: reproducible model compatibility reports, agent
adapters, installation feedback, and upstream checkpoint persistence work.
See [CONTRIBUTING.md](CONTRIBUTING.md). Bring a slow conversation; let the sloth help.
