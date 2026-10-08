# OpenCode

kvpark works with [OpenCode](https://opencode.ai), the terminal coding agent. An
OpenCode user gets what a Hermes user gets: each conversation's KV cache is saved
to disk and restored when you return to it, and the long system prompt and tool list
that OpenCode sends on every request are reused instead of prefilled again. A
conversation also survives restarting the model server when it was saved first:
`kvpark stop` saves it. One command sets it up.

Tested with OpenCode 1.18.34 on Linux. Disk park/resume needs kvpark's patched
llama.cpp runtime (see [compatibility](compatibility.md)) and a GGUF model of your own.

## Set up

Build the runtime once (`python scripts/build_runtime.py`, see
[compatibility](compatibility.md#building)), then:

```bash
kvpark setup --opencode --server /absolute/path/to/runtime/build/bin/llama-server \
  --model /absolute/path/to/model.gguf
opencode -m kvpark/local
```

The first command starts kvpark's llama-server and the kvpark proxy, then adds a
`kvpark` provider and a small plugin to OpenCode's global config
(`~/.config/opencode`, or `$XDG_CONFIG_HOME/opencode`). In OpenCode, the model is
`kvpark/local`: start with `-m kvpark/local` or choose it with `/models`. Setup does
not change your default model.

The server starts with `--ctx-size 32768 --cache-ram 0 --ctx-checkpoints 8
--checkpoint-min-step 128 --jinja`. OpenCode's system prompt and tools alone take
about 7,300 tokens (Qwen3's tokenizer), so kvpark's usual 8,192-token context is too
small for it. To run
the server with other arguments (GPU layers, a larger context, a draft model), pass
them as one quoted string, which replaces that list:

```bash
kvpark setup --opencode --server … --model … \
  --backend-args "--ctx-size 65536 --n-gpu-layers 99 --cache-ram 0 --ctx-checkpoints 8 --checkpoint-min-step 128 --jinja"
```

Keep `--jinja`: OpenCode relies on tool calls.

Other ways to connect:

| Command | Uses |
| --- | --- |
| `kvpark setup --opencode --url http://127.0.0.1:8080` | A kvpark proxy that is already running, such as the one Hermes starts. OpenCode and Hermes then share it; their conversations stay apart. |
| `kvpark setup` then `kvpark setup --opencode` | A model server you already run (Ollama, vLLM, LM Studio, stock llama.cpp), picked from the numbered list. Chat only: these servers cannot be parked or resumed. |
| `--project /path/to/repo` | Write to that project's `.opencode/` directory instead of the global config. See [Project setup](#project-setup). |
| `--context 65536` | The context window, when the server does not report it (Ollama). It cannot exceed a window the server reports. |

Setup reads the context window from the server (llama.cpp's `/props`, vLLM's
`max_model_len`) and sets the model's limits: `context` to that window and `output`
to a quarter of it, at most 32,000 tokens. OpenCode compacts a conversation that
approaches the limit. Running setup again updates the provider in place.

Setup refuses a window under 16,384 tokens: after OpenCode's system prompt, tools and
the quarter kept for the answer, almost nothing is left for the conversation. Hermes
starts its model server with 8,192 tokens, so to share Hermes' proxy, first raise its
context in Hermes, for example with `/kvpark setup --backend-args "--ctx-size 32768
--cache-ram 0 --ctx-checkpoints 8 --checkpoint-min-step 128 --jinja"`.

### What setup changes

In `opencode.json` it adds only this (your other providers and settings stay):

```json
{
  "provider": {
    "kvpark": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "kvpark",
      "options": {"baseURL": "http://127.0.0.1:8080/v1", "apiKey": "{env:KVPARK_API_KEY}"},
      "models": {
        "local": {"name": "model (kvpark)", "tool_call": true, "limit": {"context": 32768, "output": 8192}}
      }
    }
  }
}
```

- The previous file is copied to `opencode.json.kvpark-<date>-<time>.bak` next to it.
  Nothing is written when nothing would change. A new file also gets `$schema`.
- Keys you add under `provider.kvpark`, such as headers or options, are kept, and so
  are models you add there. A model whose name ends in `(kvpark)` is setup's own:
  when setup connects another model and the proxy no longer serves the old one, it
  replaces it, along with a `model` or `small_model` that names it. Rename a model to
  keep it.
- If your config is `opencode.jsonc` with comments, kvpark leaves it alone and writes
  an `opencode.json` beside it, which OpenCode merges. A file kvpark cannot parse is
  never rewritten; add the provider by hand instead.
- The new file is written privately and then given the old file's permissions, so the
  API keys of other providers in it are never readable by others, not even briefly.
- The plugin is copied to `plugins/kvpark.js` in the same directory. It has no
  dependencies; OpenCode loads it at startup.
- `{env:KVPARK_API_KEY}` sends the proxy's key when you set `KVPARK_API_KEY`, and
  nothing otherwise.
- If `enabled_providers` leaves out `kvpark`, or `disabled_providers` lists it,
  OpenCode hides the provider. Setup warns and does not change those lists.

The provider package `@ai-sdk/openai-compatible` is built into OpenCode, so nothing
is downloaded. OpenCode itself installs `@opencode-ai/plugin` into each config
directory it finds, with or without kvpark.

### Project setup

`--project /path/to/repo` writes `.opencode/opencode.json` and
`.opencode/plugins/kvpark.js` in that repository, for OpenCode sessions started there.
On its next start in the repository, OpenCode adds its own files to `.opencode/`: a
`.gitignore`, `package.json`, `bun.lock` and `node_modules/` with `@opencode-ai/plugin`.
`opencode.json` names a local port, which means nothing on another machine. Keep
`.opencode/` out of version control (add it to the repository's `.gitignore`) unless
the project already shares one. `kvpark uninstall --opencode --project …` removes only
kvpark's files; delete `.opencode/` yourself if you created it only for kvpark.

## Every day

There is nothing to do. Conversations are saved when other work needs the model's
only slot and restored when you return to them. To see what is saved:

```bash
kvpark status
```

To save the conversation in the slot now, or delete a conversation's snapshot, use the
session ID that `opencode session list` shows:

```bash
kvpark save --session opencode:ses_…
kvpark forget --session opencode:ses_…
```

`kvpark save` works only for the conversation that is in the slot, or one that
background work has just displaced (it was saved before that work started). With
`--url`, OpenCode shares another proxy; pass the same `--url` to `kvpark status`, `save`
and `forget`, which otherwise ask the connection chosen with `kvpark setup`.

Snapshots follow kvpark's [storage policy](compatibility.md#storage-policy): seven
days, 32 GiB, least recently used first. `kvpark settings` changes it.

### Stop and start

```bash
kvpark stop     # save the conversation in the slot, then stop the proxy and the model server
kvpark start    # start them again; the next request restores the conversation
```

`kvpark stop` waits up to a minute for the reply in progress, saves the conversation in
the slot, and stops both processes. The setup is kept. Without it, a conversation that is in the
slot and has not been saved is lost when the processes end: a reboot, a crash, or
killing them prefills it again on the next request.

After a reboot, start kvpark before OpenCode, or run it as a boot service.
`kvpark start --supervise` keeps the services running from a systemd user service or a
launchd agent; the unit in the [Hermes guide](hermes.md#start-at-boot) works with these
two lines (`command -v kvpark` shows the path):

```ini
ExecStart=/path/to/kvpark start --supervise
ExecStop=-/path/to/kvpark stop
```

systemd runs `ExecStop` before it ends the service's processes, so
`systemctl --user stop`, a restart and a shutdown save the conversation first. Under a
supervisor, stop the service rather than running `kvpark stop` by hand: the supervisor
starts stopped services again within a minute.

## Check

```bash
kvpark doctor --opencode
```

```text
OpenCode: 1.18.34 (~/.opencode/bin/opencode)
Config: ~/.config/opencode/opencode.json — provider kvpark → http://127.0.0.1:8080/v1; models: local (context 32,768)
Plugin: ~/.config/opencode/plugins/kvpark.js (current)
Proxy: kvpark 0.3.0a4.dev0 at http://127.0.0.1:8080 → llama.cpp at http://127.0.0.1:8090; disk park/resume on
Models: local served
```

It changes nothing and exits with status 1 when OpenCode cannot reach kvpark: no
provider, a missing plugin or one from another kvpark version, a proxy that is not
running or moved to another port, a model setup wrote that the proxy does not serve, a
`model` or `small_model` naming a kvpark model that is not defined, a context larger
than the server's, or provider lists that hide `kvpark`. Each line says what to do;
most need only setup run again. A model you added by hand that the proxy does not
serve is a warning.

OpenCode does not report plugins that fail to load. When none of the proxy's recent
requests named a conversation, doctor says the plugin may not be loaded; on a proxy
shared with Hermes, Hermes' requests hide this.

## Remove

```bash
kvpark uninstall --opencode            # preview
kvpark uninstall --opencode --confirm
```

This removes `provider.kvpark`, a `model` or `small_model` that names it, and the
plugin, and backs up each file it changes. A file left with nothing but `$schema`, as
setup creates one, is deleted (its backup stays), and so is an empty `plugins/`
directory. If one file cannot be written, the ones already changed are restored.
kvpark's services and snapshots are not touched: `kvpark stop` stops the services, and
[uninstall](uninstall.md) covers the rest.

## How it works

```text
OpenCode ── plugin adds the conversation's identity ──▶ kvpark proxy ──▶ llama-server (one slot)
                                                          ↕
                                                   snapshots on disk
```

OpenCode talks to the proxy like any OpenAI-compatible server. The proxy relays
streaming responses, tool calls and reasoning byte for byte. What OpenCode needs
beyond that, and how kvpark provides it:

### Which conversation a request belongs to

kvpark only parks a request that says which conversation it belongs to and that it is
the conversation itself, not work done for it ([integration contract](integration.md#request-identity)).
OpenCode sends an `x-opencode-session-id` header with every request, and a subagent's
requests also carry `x-opencode-parent-session-id`. But its title generator,
compaction and summaries run under the same session ID as the conversation, with a
different system prompt, and no header tells them apart.

The plugin can. OpenCode 1.18.34 calls the plugin's `chat.params` hook before every
request, with the session ID and the agent's name, and passes the options the hook
sets to `@ai-sdk/openai-compatible`, which copies unknown options into the request
body. For requests to the `kvpark` provider the plugin adds:

| Request | `slot_archive_role` | `slot_archive_key` | `slot_archive_autosave` |
| --- | --- | --- | --- |
| An agent in a top-level session (build, plan, your own primary agents) | `foreground` | `opencode:<session ID>` | `true` |
| Title, summary, compaction (OpenCode's hidden agents) | `background` | — | — |
| A subagent (its session has a parent session) | `background` | — | — |

The plugin asks OpenCode once per session whether it has a parent
(`client.session.get`). If it cannot tell, it sends the request as background work,
which can never take over a conversation's snapshot. The proxy removes these fields
before the request reaches the model server. Requests to other providers are not
changed.

### Background work does not cost the conversation its cache

The model server has one slot. A title request or a subagent replaces what the slot
holds, and without kvpark the conversation is prefilled again afterwards. With
`slot_archive_autosave`, the proxy saves the conversation before background work takes
the slot, and restores it on the conversation's next request. When you switch to
another OpenCode session, the one you leave is saved too, once it holds at least
8,192 tokens (`KVPARK_MIN_TOKENS`). A smaller one is rebuilt cheaply when you return,
because the slot then holds another session's system prompt and tools, which it shares.

A saved conversation is saved again when it has grown by 4,096 tokens
(`KVPARK_RESAVE_GROWTH`) or when its snapshot no longer matches it: the system prompt
or tools changed (a new day, an edited `AGENTS.md`, another MCP server), or compaction
rewrote it. Such a snapshot cannot be restored as it is, so it is replaced on the next
save whatever the growth. A snapshot that already holds exactly what is in the slot
(the same request ran again) is not written again.

Each save writes the conversation's whole cache: 1.65 GiB for 12,000 tokens of
Qwen3-4B with an F16 cache, in 0.4 s on the test machine. A quantized cache
(`--cache-type-k q8_0 --cache-type-v q8_0`) halves that.

Subagents that run in parallel take turns in the slot and prefill each time they
take it back. If you have another model available, `"small_model"` in OpenCode's config
sends title generation to it and keeps it out of the slot entirely.

### The system prompt and tools are reused across sessions

OpenCode sends the same system prompt and tool list, about 7,300 tokens with its
built-in tools, at the start of every request. llama.cpp reuses it within a
conversation by itself. A new session is different: OpenCode generates the session's
title at the same time as the session's first answer, and the proxy runs one request
at a time. When the title request runs first, its prompt shares nothing with the
conversation's, so the slot no longer holds the shared prefix. When the first answer
runs first, the previous conversation is saved only if it holds 8,192 tokens or more
(see above).

When a new conversation arrives and the slot does not hold its system prompt and
tools, the proxy restores the smallest saved snapshot with the same system prompt,
tools and template settings (event `seeded`). llama.cpp keeps the shared prefix and
drops the rest, so only the new conversation's own messages are prefilled. The source
snapshot is not changed. On hybrid and sliding-window models llama.cpp can reuse a
seeded prefix only up to a context checkpoint it holds; past that it prefills, as it
would without kvpark.

Restoring a large snapshot can take longer than prefilling the prefix on a fast GPU
or a slow disk. The proxy measures both on your machine (restores, and the server's
own prefill timings) and skips seeding when prefilling the system prompt and tools,
estimated at four bytes per token, is faster (event `seed-skipped`). Until it has
measured both, it seeds.

OpenCode's system prompt includes the working directory and today's date, so these
prefixes are shared within one project and one day. A snapshot from an earlier day is
skipped (`restore-skipped: prompt prefix changed: system`), its conversation is
prefilled once, and the snapshot is replaced the next time the conversation is saved.

### Limits

The provider's `limit.context` comes from the server, so OpenCode compacts before a
conversation overflows the slot. Compaction rewrites the conversation; its next save
replaces the old snapshot.

### Design choices

- **A provider and a plugin, no proxy flags.** The provider is plain OpenCode
  configuration. The plugin is the only way to tell a conversation from the requests
  made for it. It relies on the `chat.params` hook and the SDK client
  (`client.session.get`) that OpenCode passes to plugins. `chat.params` is part of
  OpenCode's plugin interface in its source, but not on its plugins documentation
  page; kvpark is tested with 1.18.34.
- **OpenCode points at the proxy.** Hermes points at the model server and inserts the
  proxy per request, so it keeps working when kvpark is down. OpenCode plugins cannot
  change a request's address, so OpenCode needs the proxy running: `kvpark start`,
  or a boot service.
- **Automatic saving instead of a `/kvpark save` command.** OpenCode's custom commands
  always send a prompt to the model, and a model turn would replace the state to be
  saved. Saving before other work takes the slot needs no command; `kvpark save
  --session` remains for an explicit save. Saving after every turn (the plugin could do
  that when a session goes idle) would write the whole cache after each answer.
- **Only plain JSON is rewritten**, so comments in your config are never lost.

### `/kvpark switch` in OpenCode

Not supported yet. Hermes stages a switch with `/kvpark switch MODEL`, and its request
middleware, running in Hermes' Python process, translates the cache when the
conversation's next request reaches the target model. OpenCode would need:

1. A way to run `/kvpark switch` without a model turn. OpenCode's custom commands
   cannot; a TUI plugin that registers its own command might, and needs testing.
2. The translation run where OpenCode's request passes: in the kvpark proxy, which
   would look up a staged switch for the request's `slot_archive_key` and translate
   before forwarding, since the OpenCode plugin runs in JavaScript.
3. The target model as another model of the `kvpark` provider behind the same proxy,
   or a second provider per bridge model, so switching with `/models` reaches it.

## Validation

On October 6, 2026, OpenCode 1.18.34 ran one script of prompts twice against the same
throwaway llama-server: Qwen3-4B-Instruct-2507 Q8_0 on the pinned runtime's Vulkan
build, all layers on a Radeon 8060S, `--ctx-size 32768`. OpenCode ran with its own
scratch config, data and cache directories and auto-update off.

- **With kvpark:** set up with `kvpark setup --opencode --server … --model … --backend-args …`
  (2.5 s, including starting both services), then checked with `kvpark doctor --opencode`.
- **Without kvpark:** a plain OpenCode provider, through a kvpark proxy in routing mode
  that only forwards requests and records the server's cache counters.

Session A attached about 4,500 tokens of notes. Between A3 and A4 the model server and
the proxy were stopped and started again. The kvpark run stopped them with the drain
that `kvpark stop` runs, which saved A first; a crash or a kill would not have.

| Turn | Without kvpark | With kvpark |
| --- | --- | --- |
| A1, new session | 11,871 tokens prefilled; first output 17.8 s | 11,873 prefilled; 18.0 s (nothing saved yet) |
| A2, same session | 11,906 reused, 29 prefilled; 1.5 s | 11,910 reused, 29 prefilled; 1.4 s |
| B1, new session | 7,327 prefilled: the title request had replaced the slot; 7.3 s | A saved before the title request (12,001 tokens, 1.65 GiB, 0.42 s); B seeded from it (0.21 s); 7,308 reused, 26 prefilled; 2.2 s |
| A3, back to A | 7,306 reused (B's system prompt and tools), 4,726 prefilled; 8.9 s | A restored (0.21 s); 12,001 reused, 29 prefilled; 1.6 s |
| A4, after stopping and starting | 12,091 prefilled; 14.5 s | A restored (0.19 s); 12,063 reused, 26 prefilled; 1.6 s |

"First output" is measured by the client, from launching `opencode run` to OpenCode's
first text, and includes OpenCode's startup of about 1.3 s and the title request. The
proxy's own time to first token for B1, A3 and A4 was 0.88, 0.30 and 0.29 s with kvpark,
against 5.86, 7.61 and 13.14 s without it. Every turn completed. This run used no
tools and no reasoning model; streamed tool calls and `reasoning_content` were checked
through the proxy against a stand-in server. Every request carried the plugin's tags:
title requests went out as background work and never took a conversation's key. In a
separate run against a stand-in server, a subagent's request also went out as
background work.
