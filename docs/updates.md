# Updates and recovery

Use these commands after installing **kvpark `0.3.0a1` or newer**. If you still
have sloth-memory installed, first follow the [rename migration](#migrate-from-sloth-memory).

```text
/kvpark update check
/kvpark update
```

`check` is read-only. It reports the installed package, the adapter loaded in this
Hermes process, the running proxy, and the newest release available to the
installation's channel. An alpha installation includes prereleases; a stable
installation stays on stable releases. Only published releases with a matching
universal wheel and GitHub SHA-256 digest are offered. Installing from `main`
does not enable automatic tracking of new commits. Checks happen on command;
there is no background upgrade or notification schedule.

`update` downloads and verifies the selected release, validates its Python,
dependency, snapshot-format, and updater-protocol compatibility, and backs up the
installed package and configuration. Dependencies must already satisfy the new
release; the updater does not change Hermes' other packages. There is no sudo
prompt: run it in a writable Python environment owned by the account running
Hermes. Editable installations must be updated from their source checkout.

The proxy pauses new inference, waits up to 60 seconds for the current request,
and saves verified resident foreground state in native mode. If draining or
saving fails, the update stops before installing. During this brief maintenance
period, new inference receives HTTP 503 and should be retried after the update.
Routing mode preserves backend-managed state by retaining the backend process.

After installation, the proxy is replaced at the same address, checked for the
expected version and backend health, and reopened. A busy address causes recovery
rather than silently changing a route during an update. The model binary,
weights, backend process, saved settings, and existing compatible slots are kept.
Saving resident state starts a new seven-day snapshot lifetime under the default
policy. Updating does not rebuild a patched runtime or migrate cache formats.

In supported Hermes gateways, the command replies and then requests Hermes'
native draining restart, as `/restart` does. In an interactive Hermes CLI, or
when the gateway restart hook is unavailable, the reply tells you to restart
Hermes yourself. Reopen an interactive CLI after updating it. If several agents
or profiles share the Python environment, restart their other adapters and
update their proxies separately. `/kvpark update check` shows a pending restart
until this adapter has loaded the installed version.

From a terminal, with the same environment activated:

```bash
python -m kvpark update check
python -m kvpark update
```

Pass `--archive-dir /your/archive` to either command for a custom archive. The
module invocation also works on Windows, where an active `kvpark.exe`
launcher cannot safely replace itself. Externally managed proxies must be
updated through their service owner. On shared bots, keep `/kvpark` restricted to administrators using Hermes' slash-command access
settings: these commands can install code in the bot's Python environment.

## Migrate from sloth-memory

`kvpark` is the new repository, Python package/module, CLI, and Hermes plugin
name. **The old `/sloth update` cannot cross package names.** Installing kvpark
alongside sloth-memory does not automatically stop the old proxy or transfer its
profile settings. Use this explicit migration once per Hermes profile.

The migration preserves the original archive directory, snapshot files and keys,
cleanup policy, model settings, and original Hermes route backup. It does not
move data, rebuild a patched runtime, or stop the model backend. Keep the archive
path and runtime build unchanged: they are part of snapshot compatibility. A
migrated directory can retain `sloth-memory` in its name.

### Hermes migration sequence

1. Finish active turns. Stop the Hermes gateway and close all interactive Hermes
   sessions and other clients sharing the proxy. Leave the backend and old proxy
   running so migration can save resident state safely:

   ```bash
   hermes gateway stop
   ```

2. Activate the **same Python environment that runs Hermes** and select the same
   Hermes profile (`HERMES_HOME` when set). Install the new package alongside the
   old one:

   ```bash
   python -m pip install https://github.com/uncrayon/kvpark/releases/download/v0.3.0a1/kvpark-0.3.0a1-py3-none-any.whl
   python -m kvpark --version
   ```

   The version should be `0.3.0a1`. Keep `sloth-memory` installed until the new
   integration has been checked. Do not separately enable kvpark yet.

3. Preview the profile migration:

   ```bash
   python -m kvpark migrate --hermes
   ```

   Check the printed profile, archive directory, and proxy before applying it.
   The command reads the archive location from the old plugin entry; no
   `--archive-dir` is needed. An externally managed legacy proxy is refused;
   its service owner must coordinate replacement.

4. With clients still stopped, apply the preview:

   ```bash
   python -m kvpark migrate --hermes --confirm
   ```

   Migration backs up the profile, drains inference, saves verified resident
   foreground state in native mode, and stops only the verified old proxy. It
   copies plugin settings and route backups to the `kvpark` entry, enables
   kvpark, and disables sloth-memory. Routing mode keeps the existing backend
   process and its engine-managed cache. A failure to drain or save aborts the
   migration; follow any recovery details printed by the command.

   **This command does not launch or validate the new proxy, or restart Hermes.**
   A successful migration means the profile is ready for the next step.

5. Start Hermes and verify the new integration:

   ```bash
   hermes gateway start
   ```

   In Hermes, run:

   ```text
   /kvpark status
   /kvpark update check
   ```

   Confirm that the expected backend, archive path, saved slots, and new versions
   are shown. With autostart enabled, loading Hermes starts the new proxy and
   reuses the managed backend. If autostart was deliberately off, it stays off;
   use `/kvpark start` to start the service explicitly. In native mode, send a
   normal message in the same compatible conversation and check reuse, then use
   `/kvpark save`. A rename does not make incompatible snapshots restorable.

6. Only after those checks pass, stop Hermes processes using this environment
   again, remove the old package, and start Hermes:

   ```bash
   hermes gateway stop
   python -m pip uninstall sloth-memory
   hermes gateway start
   ```

   Close interactive clients too. Other profiles sharing this Python environment
   must be migrated and checked before removing the old package. Keep the profile
   backup until you are satisfied with the new installation.

New configuration uses `KVPARK_*`. Legacy `SLOTH_*` environment settings are
accepted during migration, including authentication settings; keep required
credentials available to both the migration command and Hermes startup. Prefer
new names when you next edit the profile or service environment. Never copy
credentials into command output, issues, or committed files.

Use `/kvpark` on Discord and Telegram. Replace any manually configured old
command-menu priority or access-control entries with the new command name,
preserving administrator restrictions. Restarting the gateway republishes its
command catalog; clients may cache the old menu temporarily.

### Earlier or custom installations

The automated migration needs an identifiable legacy profile and a proxy with
coordinated drain support (sloth-memory `0.2.0a1` or newer). If it refuses an older
proxy, do not kill a guessed port or PID. Stop clients, save through the old
integration if supported, and stop that proxy through its original terminal or
service owner. Keep a private profile backup and follow the old release's
recovery instructions before retrying. An absent `proxy.json` does not prove
that a very old proxy is stopped.

For qwen-slot or a custom integration, use the separate
[custom-integration guide](hermes.md#migrate-an-existing-custom-integration).
That integration has a different ownership contract from sloth-memory.

Standalone users must coordinate their agent's configuration themselves. With
clients stopped and the new package installed, preview and drain an owned legacy
proxy using the original archive path:

```bash
python -m kvpark migrate --archive-dir /your/existing/archive
python -m kvpark migrate --archive-dir /your/existing/archive --confirm
```

Start kvpark with the **same explicit `--archive-dir`**, backend, cache mode, and
upstream URL, then point the agent at the new proxy's selected address. Keep the
model backend running and verify status before removing the old package. If a
supervisor owns the old proxy, coordinate replacement through that owner so it
cannot restart the old process. Standalone migration does not edit agent routes;
`migrate --hermes` additionally migrates the active Hermes profile.

If new startup fails after a successful migration, keep clients stopped and
inspect `hermes-proxy.log` in the unchanged archive. Retain the old package and
migration backup for recovery. Stop any new proxy through its owner before
restoring the saved profile and restarting the old integration; never run both
against the same archive. Do not overwrite newer unrelated profile changes with
an old backup.

## If an update fails

For handled installation or replacement-startup failures, the updater attempts
to reinstall the previous wheel and restart its proxy. It reports `rolled_back`
only when recovery succeeds. `recovery_required` means manual attention is
needed; the error includes the backup directory. New inference remains paused
while a replacement is being verified; a surviving proxy's maintenance lease
expires after five minutes if the updater disappears unexpectedly.

The archive's `update.json` records the last transaction. Backups are under
`updates/<transaction>/`, including the previous wheel, configuration copies,
and installation logs. They are private and are not part of seven-day snapshot
cleanup. Keep the latest successful rollback backup; older update backups can
be removed manually when no transaction is using them.

A power loss or process kill during pip installation cannot guarantee automatic
rollback. From the same Python environment, reinstall the previous wheel named
in the backup directory, then restart Hermes:

```bash
python -m pip install --no-index --no-deps --force-reinstall /path/to/backup/kvpark-PREVIOUS-py3-none-any.whl
```

The release updater does not apply configuration or snapshot-format migrations.
The explicit sloth-memory rename migration above is separate and retains the
snapshot format. A release requiring an unsupported compatibility change is
refused with a pointer to its release notes.
