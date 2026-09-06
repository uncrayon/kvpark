# Hermes adapter and onboarding

For the next development version's portable startup, automatic port selection,
and other inference engines, see [backend setup](backends.md). The instructions
below describe the published alpha.2.

The adapter ships inside sloth-memory as a native Hermes plugin. It uses plugin
discovery, request middleware, commands, and profile settings. It does not patch
Hermes source or replace its semantic memory provider.

## Install alongside Hermes

Install into **the same Python environment that runs Hermes**:

```bash
# With your Hermes virtual environment activated:
python -m pip install https://github.com/uncrayon/sloth-memory/releases/download/v0.1.0-alpha.2/sloth_memory-0.1.0a2-py3-none-any.whl
hermes plugins enable sloth-memory
```

Restart Hermes to discover the entry point. The adapter is tested against Hermes
0.21.0, upstream commit `9dd6634c5635321cf38840cc30e9b51226689128`.
Use Python 3.11–3.13 for Hermes; the standalone service also supports 3.10/3.14.

Build the [compatible runtime](compatibility.md) and obtain your own GGUF model
before managed startup. Onboarding does not download weights or compile a backend.

## Guided setup

```text
/sloth-memory setup
```

This shows defaults and the next step. Configure the backend, check readiness,
then select the route:

```text
/sloth-memory setup --server "/absolute/path/to/llama-server" --model "/absolute/path/to/model.gguf"
/sloth-memory start
/sloth-memory connect
```

Setup begins startup automatically. `connect` uses Hermes' config writer to select
the custom model `local` at the proxy's `/v1` URL for **future sessions**. Restart
Hermes and start a new conversation. Existing sessions retain their saved route;
their prompt history is not rewritten.

After configuration, the enabled plugin starts the backend and proxy when Hermes
loads. Existing compatible services are reused; an archive-directory lock
coordinates concurrent Hermes surfaces. Services stay running after Hermes closes
so cleanup continues. Reopening Hermes does not launch another model copy.

## Defaults

| Setting | Default |
| --- | --- |
| Start with Hermes | On after configuration |
| Automatic cleanup | On |
| Snapshot lifetime | 7 days after the last successful save |
| Cleanup frequency | Proxy startup and hourly, when inference is idle |
| Saved snapshot budget | 32 GiB |
| Proxy base URL | `http://127.0.0.1:8080` |
| Backend port | `8090` |

Cleanup deletes **disk snapshots**, including expired unpublished generations
left by crashes. It keeps transcripts, weights, and live RAM state. Restoring does
not reset a snapshot's age; parking again writes a fresh snapshot. Successfully
superseded generations are deleted immediately.

The schedule runs in the proxy, not an LLM-powered Hermes cron task or an OS
crontab. It consumes no model calls. A busy inference can delay a sweep; expiry is
also checked before restoring a snapshot.

## Daily controls

```text
/sloth-memory park
/sloth-memory status
/sloth-memory slots
/sloth-memory delete
```

Park and delete target the current verified conversation and make no model call.
To delete another saved slot, copy its full key from `slots`:

```text
/sloth-memory delete k-<full-key-from-slots>
```

Resume by sending a normal message in the same compatible session.

## Preferences

```text
/sloth-memory retention 7
/sloth-memory budget 32
/sloth-memory cleanup off
/sloth-memory cleanup on
/sloth-memory cleanup now
/sloth-memory autostart off
/sloth-memory autostart on
```

Retention and budget use days and GiB. `retention 0` disables age expiry.
`cleanup off` disables scheduled deletion and automatic age expiry on restore.
The capacity budget remains enforced when saving or reducing the budget.
`cleanup now` explicitly sweeps even when scheduled cleanup is disabled.
Cleanup preferences persist in the archive's `retention.json`, without restarting
inference. Autostart off affects future launches; it does not stop running services.

Change the base address with:

```text
/sloth-memory base-url http://127.0.0.1:18080
```

Use `connect` afterward to select the new route for future sessions. A second
managed stack needs a separate archive directory and backend port:

```text
/sloth-memory setup --base-url http://127.0.0.1:18080 --upstream-port 18090 --archive-dir "/path/to/separate/archive" --server "/path/to/llama-server" --model "/path/to/model.gguf"
```

Existing services are never silently moved or killed. To attach to a separately
managed **alpha.2 or newer** sloth-memory proxy:

```text
/sloth-memory setup --external --base-url http://127.0.0.1:18080
```

External mode never launches services. `--no-external` returns to managed startup.
`--backend-args` accepts a quoted argument string; for a CPU setup, for example:

```text
/sloth-memory setup --backend-args "--device none --n-gpu-layers 0 --ctx-size 8192 --cache-ram 0 --ctx-checkpoints 8 --checkpoint-min-step 128 --jinja"
```

Backend settings apply when the backend is next launched. Setup never restarts
an active model. Plugin settings live under
`plugins.entries.sloth-memory.settings.service` in the active Hermes profile.
Secrets stay in `SLOTH_API_KEY` and `SLOTH_UPSTREAM_KEY_FILE`; onboarding does not
write or print them.

## Identity and compatibility

Only requests to the configured `/v1` route get archive metadata. The adapter
reads Hermes' turn-bound active agent to distinguish foreground work from detached
reviews sharing the same session ID. Keys are namespaced by Hermes profile.
Missing or mismatched context is unverified and cannot park; process-wide session
environment variables are never used as an identity fallback.

The compatibility surface includes
`agent.subagent_lifecycle.get_active_subagent_parent`, the bound agent's
session/persistence fields, and native plugin APIs. If Hermes changes that
surface, parking fails closed. CI exercises actual discovery, middleware, settings,
and commands against the pinned upstream version; no transport patch is required.

All archive inference must pass through the proxy. Background calls and compaction
can displace or change a prefix; park before leaving an important conversation.
The original `/qwen-slot` plugin uses a different API and is not this adapter.
