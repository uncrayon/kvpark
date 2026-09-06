# Updates and recovery

Use these commands after installing `0.2.0a1` or newer:

```text
/sloth update check
/sloth update
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
update their proxies separately. `/sloth update check` shows a pending restart
until this adapter has loaded the installed version.

From a terminal, with the same environment activated:

```bash
python -m sloth_memory update check
python -m sloth_memory update
```

Pass `--archive-dir /your/archive` to either command for a custom archive. The
module invocation also works on Windows, where an active `sloth-memory.exe`
launcher cannot safely replace itself. Externally managed proxies must be
updated through their service owner. On shared bots, keep `/sloth` and
`/sloth-memory` restricted to administrators using Hermes' slash-command access
settings: these commands can install code in the bot's Python environment.

## One-time upgrade from older alphas or development installs

Old installations do not contain an updater or a proxy drain endpoint. A
package reinstall alone leaves their old proxy process running. To bootstrap:

1. Stop Hermes and other clients of this proxy. For a managed Hermes gateway,
   run `hermes gateway stop` from a separate terminal; close interactive clients.
2. Activate the Python environment that runs Hermes and install the release:

   ```bash
   python -m pip install --upgrade https://github.com/uncrayon/sloth-memory/releases/download/v0.2.0a1/sloth_memory-0.2.0a1-py3-none-any.whl
   ```

3. With all inference clients still stopped, save any resident state and stop
   the old proxy. Set `archive` explicitly if you use a custom archive directory.
   Use the proxy's `SLOTH_API_KEY` environment setting if authentication is enabled.

   ```python
   from sloth_memory.cli import request
   from sloth_memory.network import proxy_record
   from sloth_memory.runtime import directory
   from sloth_memory.updates import stop_proxy

   archive = directory()
   record = proxy_record(archive)
   if record:
       status = request(record["base_url"], "status")
       if status["activity"]["phase"] != "idle":
           raise RuntimeError("Wait for inference to finish before upgrading")
       if status.get("resident_key"):
           request(record["base_url"], "park", {"key": status["resident_key"]})
       stop_proxy(record)
   ```

   Run this Python snippet with the same environment's interpreter. It validates
   the recorded process identity before stopping it. Do not kill arbitrary
   listeners on port 8080 or stop the model backend.

   The oldest alpha.2 proxy did not publish `proxy.json`. Stop that proxy through
   the terminal or service that launched it after parking the conversation; an
   absent record does not prove that an older proxy is stopped.

4. Start Hermes again (`hermes gateway start` for a managed gateway). The enabled
   plugin starts the new proxy and discovers its selected port. Run
   `/sloth update check`; installed, loaded, and proxy versions should agree.

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
python -m pip install --no-index --no-deps --force-reinstall /path/to/backup/sloth_memory-PREVIOUS-py3-none-any.whl
```

This alpha does not apply configuration or snapshot-format migrations. A release
requiring them is refused with a pointer to its release notes rather than
silently invalidating saved state.
