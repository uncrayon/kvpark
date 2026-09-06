# 🦥 sloth-memory

**Let your local AI nap. Resume long conversations without repeating all the work.**

sloth-memory saves a local model's conversation inference state to disk and restores
it when you return. Built for people running open-weight models on their own hardware.
MIT licensed. No cloud service or account. The portable runtime uses psutil.

**In development:** Linux/macOS/Windows portability, automatic free-port selection,
and routing adapters for llama.cpp, Ollama, vLLM, and MLX-LM. Disk park/resume still
requires our patched llama.cpp runtime. See [backend capabilities](docs/backends.md).
The installation below selects the published alpha.2; these changes are not in it.

**Developer alpha · Linux · one local llama.cpp slot · pinned patched runtime.**
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
Your agent → sloth-memory :8080 → patched llama-server :8090
                     ↕
              local archive files
```

## Install the alpha

Requirements: Linux, Python 3.10+, Git, CMake, a C/C++ toolchain, and your own
supported GGUF model. Start with the CPU build to verify functionality; it is not
a GPU performance recommendation. Models and compiled runtimes are not bundled.

```bash
git clone --branch v0.1.0-alpha.2 https://github.com/uncrayon/sloth-memory.git
cd sloth-memory
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python scripts/build_runtime.py
```

The build script fetches an exact llama.cpp commit and applies the included
checkpoint persistence patch. It builds in `runtime/`; it does not modify an
existing inference installation. See [runtime compatibility](docs/compatibility.md)
for GPU builds and the current scope.

Start the backend in one terminal:

```bash
sloth-memory backend \
  --server runtime/build/bin/llama-server \
  --model /absolute/path/to/your-model.gguf \
  -- --ctx-size 8192 --cache-ram 0 --ctx-checkpoints 8 --checkpoint-min-step 128 --jinja
```

In another terminal with the same virtual environment active:

```bash
sloth-memory serve
sloth-memory doctor
```

Both processes use `~/.local/share/sloth-memory` by default, respecting
`XDG_DATA_HOME`. To change it, pass the **same** `--archive-dir` to both commands.
Keep the backend terminal running while using the proxy.

## Use with Hermes

Install sloth-memory in Hermes' Python environment, enable it with
`hermes plugins enable sloth-memory`, and restart Hermes. Then run:

```text
/sloth-memory setup
```

The [Hermes setup guide](docs/hermes.md) covers selecting the backend and model.
Once configured, services start with Hermes and keep running for cleanup.

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

```bash
python examples/chat.py --session cli:orchid:1 --history /tmp/orchid-chat.json \
  'Our project code is ORCHID-731. Help me plan a workshop.'

sloth-memory park --session cli:orchid:1
sloth-memory status
```

Stop both services with Ctrl-C, then start them again with exactly the same
commands. Continue using the same transcript and session:

```bash
python examples/chat.py --session cli:orchid:1 --history /tmp/orchid-chat.json \
  'What was the project code?'

sloth-memory status
```

Short chats demonstrate the workflow but may gain little speed. Long stable
prefixes are the intended use case. Park again before leaving to save the latest
state. Use a unique transcript file and session ID for each conversation.

## Integrate your agent

Point chat completions at `http://127.0.0.1:8080/v1`. Add these fields to each
foreground request's JSON body:

```json
{
  "slot_archive_role": "foreground",
  "slot_archive_key": "my-agent:my-project:conversation-uuid"
}
```

With clients that support it, supply these fields through `extra_body`. They are
removed before forwarding to llama.cpp. Send `slot_archive_role: "background"`
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
- Only `chat/completions` inference is supported. Responses API, remote backends,
  multiple slots, model routers, Windows, and macOS are outside this alpha.
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
