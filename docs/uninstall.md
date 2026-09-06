# Uninstall and return to your original backend

The commands below apply to **kvpark `0.3.0a1` or newer**. Still using the former
sloth-memory package? Its `0.2.0a2` release provides `/sloth uninstall`; for
`0.2.0a1`, follow the [legacy manual procedure](#manual-procedure-for-sloth-memory-020a1) below.
The [rename migration](updates.md#migrate-from-sloth-memory) is separate from uninstall.

Uninstall has two steps: disconnect the integration, then remove the package and
any files you no longer want. Disconnect first so Hermes does not keep calling a
proxy that has been removed. No sudo is required for the standard installation.

## What was installed?

| Component | Location / owner | Removal |
| --- | --- | --- |
| Python package and Hermes entry point | The Python environment used for installation | `python -m pip uninstall kvpark` |
| Enabled plugin and model route | Active Hermes profile's `config.yaml` | Automated disconnect restores the saved route and disables the plugin |
| Proxy and hourly cleanup loop | A kvpark Python process | Automated disconnect stops the verified archive-owned proxy |
| Managed llama.cpp backend, if selected | Process started by kvpark | Automated disconnect stops the verified archive-owned backend |
| Existing llama.cpp, Ollama, vLLM, or MLX server | Your existing installation | Kept running |
| Saved slots, logs, update backups and metadata | Configured archive directory | Kept until you explicitly remove that directory |
| Optional patched llama.cpp build | `runtime/` in the source checkout, or your `--output` directory | Remove this separate build directory when no longer used |
| Model weights and Hermes transcripts | Your original locations | Kept |

The adapter does not patch Hermes source. The build script patches an exported
copy of llama.cpp inside its output directory, including when `--source` points
to an existing checkout. It does not patch that original checkout or an existing
Gemma server. There is no system cron job to remove; cleanup runs inside the proxy.
Custom modifications made outside this repository's installer need their own
rollback procedure.

## Disconnect from Hermes

Finish active turns and close other clients sharing the proxy. For a shared bot,
restrict `/kvpark` to administrators through Hermes' command
access controls. This operation changes the active profile's model route and
stops its owned services; it is not limited to the current chat.

Preview without changing anything:

```text
/kvpark uninstall
```

Apply the preview:

```text
/kvpark uninstall confirm
```

The command drains current inference, saves verified resident state in native
mode, restores the model fields saved before setup, disables the plugin and its
automatic startup, and stops its owned services. A failed drain/save aborts before
changing the profile. Process IDs and creation times are checked along with launch
arguments; unrelated processes are not terminated. A stale PID is not enough to
claim a process. There is no force-kill fallback.

An existing upstream server remains running. `--external` installations only
disconnect Hermes; their external proxy also remains under its service owner's
control. Use a separate archive per independent test stack. Other profiles sharing
an archive must be disconnected separately before removing the package or data.

Restart Hermes and **start a new conversation** afterward. Existing sessions may
retain the old proxy URL. The uninstall command does not rewrite stored sessions
or automatically restart the gateway.

Terminal equivalent, in the Python environment that runs Hermes:

```bash
python -m kvpark uninstall --hermes
python -m kvpark uninstall --hermes --confirm
```

These commands use the active Hermes profile (`HERMES_HOME` when set). They read
its archive setting, so do not add `--archive-dir`. They also work after stopping
the gateway, as long as the package and Hermes are still installed.

If setup predates route backups, or the model fields were edited afterward,
uninstall refuses to guess the previous connection. Run `hermes model` and select
your original backend first, then retry. If Hermes already points elsewhere, that
route is preserved. Unrelated profile settings are kept. Repeating disconnect is
safe; the disabled plugin's commands disappear after Hermes reloads its plugins.

## Remove the package and optional data

After disconnecting, close all Hermes processes using the installation (for a
gateway, `hermes gateway stop`). Activate that same Python environment, then run:

```bash
python -m pip uninstall kvpark
```

This removes kvpark's installed package, command launcher, and plugin entry point.
It keeps dependencies that Hermes or other packages may also use. Do not delete
Hermes' virtual environment. Restart the gateway if you want Hermes running again.
Discord and Telegram menus update when Hermes synchronizes its command catalog;
cached menus may take longer to reflect removal.

For a complete disk cleanup, remove these **only after all their services have
stopped**:

1. The exact archive directory printed by the preview. On a default Mac setup it
   is `~/Library/Application Support/kvpark`; `--archive-dir`,
   `KVPARK_ARCHIVE_DIR`, or `XDG_DATA_HOME` can change it. This deletes saved KV
   snapshots, logs, and update rollback backups. A migrated installation keeps its
   original archive path, which may still be named `sloth-memory`. Inspect a custom directory first;
   it may contain other files you placed there.
2. The optional `runtime/` build directory, or the custom build output you chose.
   Removing this isolated source/build tree removes its patched runtime. Do not
   remove your pre-existing inference server installation.
3. The `kvpark` source checkout if you no longer need it. Check for your own
   changes, downloaded weights, or other files before removing the directory.

Use Finder/Trash on macOS or your OS file manager to review these removals. The
chat command intentionally keeps them: paths can be shared or contain user data.
Model weights and conversation transcripts are not reconstructed by reinstalling
the plugin.

The disabled profile entry and original route backup remain for recovery. After
the package is removed, you may delete `plugins.entries.kvpark` and the
`kvpark` item from `plugins.disabled` in the active profile's `config.yaml`.
Preserve other entries. If you manually added `kvpark` to Telegram's command-menu
priority list, remove that one item too. Remove any kvpark-only environment
variables you added to the profile or shell; preserve variables needed by other
services. Do not restore an entire old config file over newer unrelated settings.

## Standalone service

Restore your agent's original backend URL and stop other clients first. Then:

```bash
python -m kvpark uninstall --archive-dir "/absolute/path/to/archive"
python -m kvpark uninstall --archive-dir "/absolute/path/to/archive" --confirm
python -m pip uninstall kvpark
```

Standalone uninstall has no knowledge of an agent's configuration. It previews
and stops only the proxy and managed backend recorded in that archive. If a
supervisor outside kvpark starts them automatically, disable that supervisor first.
Services started with unrecognized launch arguments must be stopped through their
original terminal or service owner. Remove optional data/builds as described above.

## Manual procedure for sloth-memory 0.2.0a1

1. Finish active turns. If using native snapshots, `/sloth park` can save the
   current conversation before shutdown.
2. Run `/sloth autostart off`, then `hermes plugins disable sloth-memory` in the
   terminal. Autostart off alone does not stop an existing process.
3. Use `hermes model` to select your original server URL and served model ID.
   Consult your pre-install profile backup if necessary. Do not select sloth-memory's
   proxy address. Then stop the gateway (`hermes gateway stop`) and close other
   interactive Hermes clients.
4. Stop the sloth-memory proxy using its original terminal (Ctrl-C), supervisor, or OS
   process manager. For a detached process, `proxy.json` inside the archive records
   its PID. Inspect the running process's command and start time before terminating
   it; a stale manifest can refer to a reused PID. Its command should be
   `python -m sloth_memory serve` with your archive path. A managed backend is
   recorded in `runtime.json`; its command must match the stored `argv` including
   `--slot-save-path`. Stop only that managed instance. Keep any server you were
   already running before installing sloth-memory.
5. Run `python -m pip uninstall sloth-memory` in Hermes’ environment. Remove
   optional files only after inspecting their actual paths and stopping their services.
   Restart Hermes and start a new conversation to check the direct backend route.

## Reinstall after disconnecting

Install the desired kvpark version in Hermes' environment, enable it again with
`hermes plugins enable kvpark`, then restart Hermes. Run `/kvpark setup` with
the backend arguments again (see [backend setup](backends.md)). Explicit setup
clears the saved uninstall flag and re-enables automatic startup. A bare
`/kvpark setup` only shows instructions. Reuse an archive only when its model and
runtime are compatible, or select a new archive for the next backend test.
