# Hermes adapter and onboarding

These instructions install alpha `0.2.0a2`, including portable startup, automatic
port selection, and release updates. See [backend setup](backends.md) for capabilities.

The adapter ships inside sloth-memory as a native Hermes plugin. It uses plugin
discovery, request middleware, commands, and profile settings. It does not patch
Hermes source or replace its semantic memory provider.

## Install alongside Hermes

Install into **the same Python environment that runs Hermes**:

```bash
# With your Hermes virtual environment activated:
git clone --branch v0.2.0a2 https://github.com/uncrayon/sloth-memory.git
cd sloth-memory
python -m pip install --upgrade .
hermes plugins enable sloth-memory
```

If Hermes asks whether the plugin may replace built-in tools, answer **No**.
sloth-memory registers commands and middleware; it needs no tool overrides.
Use `hermes plugins list --plain --no-bundled` to check that it is enabled.

For an existing installation, follow [updates and recovery](updates.md). Versions
older than `0.2.0a1` need a one-time package installation and proxy replacement;
subsequent releases can use `/sloth update`. Run `sloth-memory --version` in
Hermes' environment to confirm `0.2.0a2`. Restart Hermes after installation to
load the entry point. The adapter is tested against Hermes
0.21.0, upstream commit `9dd6634c5635321cf38840cc30e9b51226689128`.
Use Python 3.11–3.13 for Hermes; the standalone service also supports 3.10/3.14.

When activating Hermes' environment, use the interpreter that actually runs
your CLI or gateway. For a standard Linux installation this is often:

```bash
source ~/.hermes/hermes-agent/venv/bin/activate
python -c "import sys; print(sys.executable)"
```

Check your own installation path; installing in an unrelated virtual environment
will not make the plugin visible to Hermes.

For managed disk snapshots, build the [compatible runtime](compatibility.md)
and obtain your own GGUF model before setup:

```bash
python scripts/build_runtime.py --cmake-arg=-DGGML_METAL=OFF
```

This starts with a CPU build, including on macOS. Use the absolute binary path
printed by the build; Windows typically places `llama-server.exe` under
`runtime/build/bin/Release`. Onboarding does not download weights or compile a
backend. Connecting an existing server in routing mode requires no build.

## Persistence and updates

To remove the integration and return to your original model server, see
[uninstall and recovery](uninstall.md). Existing servers and model weights are
kept. The guide also includes manual removal for older versions.

The installed package, plugin enablement, profile configuration, and saved slots
persist across a computer restart. When Hermes launches, the enabled plugin
starts its managed services if autostart is on. Starting Hermes itself at boot or
login is a separate Hermes service setting. The cleanup policy still expires old
snapshots; rebooting does not reset their age.

A gateway restart reuses an existing compatible proxy so it does not interrupt
shared inference. It does not automatically replace that proxy's Python code.
Use `/sloth update check` to see all three versions and `/sloth update` for the
coordinated release update. See [updates and recovery](updates.md).

## Guided setup

`/sloth` is a shortcut for `/sloth-memory` on every platform. Use `/sloth` in
Telegram, including for the setup commands below.

Already serving a model with vLLM, Ollama, MLX-LM, or llama.cpp? First
[find its exact model ID](backends.md#find-your-model-id), then follow
[connect an existing server](backends.md#connect-an-existing-server).
That guide shows the lookup command for each backend and which response value
to copy into `--model`; no llama.cpp build is needed for routing an existing server.

```text
/sloth-memory setup
```

This shows defaults and the next step. For managed disk snapshots, configure
the included patched runtime:

```text
/sloth-memory setup --server "/absolute/path/to/llama-server" --model "/absolute/path/to/model.gguf"
/sloth-memory status
```

Complete setup starts services and saves the selected proxy `/v1` URL through
Hermes' config writer for **new sessions**. Managed llama.cpp uses the model alias
`local`; an existing backend uses the exact model ID you configure. `start` and
`connect` remain available to retry startup or explicitly select the route.
Restart Hermes and start a new conversation for the initial test. Existing
sessions using the configured upstream or a previously selected proxy can be
redirected by the native OpenAI client middleware; unrelated routes are unchanged.

### Try an existing llama.cpp server on your Mac

Keep that server running and use its listening URL and served model ID:

```text
/sloth-memory setup --backend llama.cpp --cache-mode routing --upstream-url http://127.0.0.1:8090 --model "your-served-alias"
/sloth-memory status
```

This tests `Hermes → sloth-memory → your existing llama.cpp`. It does **not**
enable disk park/resume: stock llama.cpp has save/restore APIs, but this adapter
currently requires its additional state companion for native archives. To test
persistence after this routing test, explicitly clear the existing upstream and
select native mode with a separate archive directory:

```text
/sloth-memory setup --backend llama.cpp --cache-mode native --upstream-url "" --no-external --archive-dir "/absolute/path/to/sloth-native" --server "/absolute/path/to/llama-server" --model "/absolute/path/to/model.gguf"
```

See [other backends](backends.md#connect-an-existing-server) for Ollama, vLLM, and MLX-LM.

After configuration, the enabled plugin starts the proxy and, in managed native
mode, the backend when Hermes loads. Existing upstream servers must already be
running. Existing compatible services are reused; an archive-directory lock coordinates concurrent Hermes surfaces. Services stay running after Hermes closes
so cleanup continues. Reopening Hermes does not launch another model copy.

## Defaults

| Setting | Default |
| --- | --- |
| Start with Hermes | On after configuration |
| Automatic cleanup | On |
| Snapshot lifetime | 7 days after the last successful save |
| Cleanup frequency | Proxy startup and hourly, when inference is idle |
| Saved snapshot budget | 32 GiB |
| Preferred proxy base URL | `http://127.0.0.1:8080`; falls back to a free port |
| Preferred managed backend port | `8090`; falls back to a free port |
| Archive directory | [Platform-specific default](backends.md#platform-storage) |

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

Disk park/resume requires native snapshot mode. Routing-only adapters report
`capabilities.disk_snapshots: false` in status and reject parking.

Before the first park after installation or migration, **send a normal message
in that same conversation and wait for the reply**. Then run `/sloth-memory park`.
Opening a saved transcript, `/status`, and restarting the gateway do not send its
history through the model, so they cannot create resident state to save. A chat
from the old integration may need a fresh prefill on this first message.

If the command says there is no saved or resident state, follow that sequence.
`/sloth-memory slots` lists snapshots already saved through this plugin; it cannot
recover a conversation that has never run through the new proxy.

Park and delete target the current verified conversation and make no model call.
To delete another saved slot, copy its full key from `slots`:

```text
/sloth-memory delete k-<full-key-from-slots>
```

Resume by sending a normal message in the same compatible session.

## Telegram command picker

Use `/sloth setup`, `/sloth status`, and `/sloth park` in Telegram. Telegram
command names cannot contain hyphens; the short alias also preserves the current
chat's identity through Hermes' command hooks. Avoid the automatically converted
`/sloth_memory` spelling on the tested Hermes version.

Hermes limits its Telegram menu to 60 commands by default, so a working plugin
command can be omitted from the picker. To keep `/sloth` visible, add it to the
menu priority in your Hermes profile's `config.yaml`, merging with any existing
platform settings and priority entries:

```yaml
platforms:
  telegram:
    extra:
      command_menu:
        priority: [sloth]
        priority_mode: prepend
```

Restart the Hermes gateway after installation and configuration. It publishes
the menu to Telegram for default, private-chat, and group scopes. Reopen the chat
or command menu if the client still shows cached suggestions. You can also type
`/sloth status` directly while checking menu registration.

## Discord command picker

The enabled plugin registers `/sloth-memory` with Hermes' native Discord command
list. In the picker, select it and enter `setup`, `status`, `park`, or another
subcommand in the optional **args** field. Typing the complete text command also
works.

If text commands work but the picker is missing the command, check the gateway
logs for Discord slash-command synchronization errors. Discord can rate-limit
registration independently of chat traffic. On the tested Hermes version, a
restart during the recorded cooldown skips synchronization; wait for the logged
cooldown to expire before restarting the gateway to retry. Repeated restarts
during that window do not make registration complete sooner.

After registration succeeds, close and reopen the Discord picker or refresh the
client. A functioning text command proves local dispatch, not that Discord has
accepted the application's command catalog.

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
managed stack needs a separate archive directory. Ports below are preferences;
occupied ports fall back automatically:

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

## Migrate an existing custom integration

Install and verify `/sloth-memory setup` before changing the active model route.
Keep a private backup of your Hermes profile's `config.yaml`, `.env`, old service
configuration, and plugin links. These files can contain credentials; do not
commit them. Keep existing transcripts and legacy archives in place. Use a fresh
sloth-memory archive directory: changing the runtime arguments or archive namespace
can invalidate previous snapshots, so the first request may need a fresh prefill.

Before configuring managed startup:

1. Finish active turns and pause the Hermes gateway or other clients.
2. Park the resident conversation through the old integration if it supports it.
3. Stop the old proxy and model service, and disable their automatic startup.
   Port fallback prevents address conflicts but does not prevent loading a second
   copy of the model. System-level services may need sudo to retire; sloth-memory
   itself runs as your normal user.
4. Disable the old archive plugin (`hermes plugins disable qwen-slot` for the
   original integration). Retire any custom provider override that injected its
   archive metadata. Preserve unrelated providers and model settings.
5. Configure sloth-memory, start a new Hermes session, then restart the gateway
   after the new route and disk persistence have been checked.

A model-specific GPU runtime that already includes the compatible slot-resume
extension can be supplied via `--server`; it still needs model-specific acceptance.
The standard build does not include the originating project's separate DFlash2
position or AMD hardware fixes. Preserve required fixes when migrating such a build.

Transfer model and hardware settings with `--backend-args`, including context
size, GPU device, draft model, projector, cache types, and reasoning settings.
This option **replaces** the default argument list. Include
`--ctx-checkpoints 8 --checkpoint-min-step 128`; use `--cache-ram 0` for the
restart test so RAM prompt caching cannot stand in for a disk restore.
Do not copy `--model`, `--alias`, `--host`, `--port`, `--parallel`,
`--slot-save-path`, or UI switches from the old launcher; sloth-memory owns them.
The managed model is advertised as `local`.

Custom GPU bundles may require environment variables such as `LD_LIBRARY_PATH`.
The plugin's child processes inherit Hermes' environment, not the old model
service's environment. Make the required settings available to both CLI and
gateway launches, for example in the active Hermes profile's `.env`, then restart
Hermes. Use the bundle's absolute library directory and retain its libraries.
Check the backend log and runtime manifest to confirm the intended build loaded.

If startup fails, inspect `hermes-backend.log` and `hermes-proxy.log` inside the
archive directory. Restore the saved profile and old plugin configuration before
re-enabling the old services; stop any newly managed processes first to avoid
loading duplicate models. `autostart off` alone does not stop running processes.

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
