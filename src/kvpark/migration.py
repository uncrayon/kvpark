"""Explicit one-time migration from the sloth-memory distribution."""

from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import sys
import uuid

from .cli import request
from .platforms import data_directory, lock_file, process_matches
from .removal import owned_process
from .updates import stop_proxy

LEGACY = "sloth-memory"


def profile_plan(raw):
    result = copy.deepcopy(raw)
    plugins = result.setdefault("plugins", {})
    entries = plugins.setdefault("entries", {})
    if "kvpark" in entries:
        if entries["kvpark"].get("settings", {}).get("migrated_from") == LEGACY:
            return None
        raise ValueError("This profile already has kvpark settings; refusing to overwrite them.")
    if LEGACY not in entries:
        raise ValueError("No sloth-memory settings in this Hermes profile; use /kvpark setup for a fresh install.")
    entry = copy.deepcopy(entries[LEGACY])
    # Hermes falls back from settings to config per top-level setting. Mixed
    # profiles can keep service in config and newer route backups in settings.
    settings = {**copy.deepcopy(entry.get("config", {})), **entry.get("settings", {})}
    entry["settings"] = settings
    service = settings.setdefault("service", {})
    if service.get("external"):
        raise ValueError("The legacy proxy is externally managed; migrate it through its service owner first.")
    # Keep the absolute old directory: it participates in the native runtime's
    # identity. Moving it would invalidate existing KV snapshots.
    service["archive_dir"] = str(Path(service.get("archive_dir") or os.environ.get("KVPARK_ARCHIVE_DIR")
                                    or data_directory() / LEGACY).expanduser().resolve())
    settings["migrated_from"] = LEGACY
    entries["kvpark"] = entry
    old_settings = {**copy.deepcopy(entries[LEGACY].get("config", {})), **entries[LEGACY].get("settings", {})}
    entries[LEGACY]["settings"] = old_settings
    old_settings.setdefault("service", {}).update(autostart=False, connected=False, uninstalled=True)
    plugins["enabled"] = list(dict.fromkeys([name for name in plugins.get("enabled", []) if name != LEGACY] + ["kvpark"]))
    plugins["disabled"] = list(dict.fromkeys([name for name in plugins.get("disabled", []) if name != "kvpark"] + [LEGACY]))
    # A manually prioritized Telegram command should follow the rename too.
    menu = result.get("platforms", {}).get("telegram", {}).get("extra", {}).get("command_menu", {})
    if isinstance(menu.get("priority"), list):
        menu["priority"] = list(dict.fromkeys("kvpark" if name in {"sloth", LEGACY, "sloth_memory"} else name
                                            for name in menu["priority"]))
    return result


def editable():
    from hermes_cli import config, managed_scope
    paths = ("plugins.enabled", "plugins.disabled", "plugins.entries.sloth-memory", "plugins.entries.kvpark",
             "platforms.telegram.extra.command_menu.priority")
    if config.is_managed() or any(key == path or key.startswith(path + ".") or path.startswith(key + ".")
                                  for key in managed_scope.managed_config_keys() for path in paths):
        raise PermissionError("Plugin migration is managed by your administrator.")


def run(*, hermes=False, archive=None, confirm=False):
    raw = planned = None
    if hermes:
        if archive:
            raise ValueError("--hermes reads the active profile's archive; omit --archive-dir")
        from hermes_cli import config
        from hermes_cli.plugins import _locked_plugin_state
        editable()
        # Terminal migration must see the same profile credentials as the
        # stopped gateway. Import only this integration's environment settings.
        for name, value in config.load_env().items():
            if name.startswith(("SLOTH_", "KVPARK_")):
                os.environ.setdefault(name, value)
        for name, value in list(os.environ.items()):
            if name.startswith("SLOTH_"):
                os.environ.setdefault("KVPARK_" + name[6:], value)
        raw = config.read_user_config_raw()
        planned = profile_plan(raw)
        if planned is None:
            return "This Hermes profile was already migrated. Restart Hermes and check /kvpark status."
        archive = planned["plugins"]["entries"]["kvpark"]["settings"]["service"]["archive_dir"]
    archive = Path(archive or os.environ.get("KVPARK_ARCHIVE_DIR") or data_directory() / LEGACY).expanduser().resolve()
    record = owned_process(archive, "proxy.json", legacy=True)
    preview = ((f"Hermes profile: {config.get_config_path()}\n" if hermes else "")
               + f"Keep archive: {archive}\nKeep the backend process, runtime, weights, and saved snapshots.\n"
               + (f"Drain and stop legacy proxy PID {record['pid']}.\n" if record else "Legacy proxy is already stopped.\n")
               + ("Copy the active Hermes profile's settings; disable sloth-memory and enable kvpark.\n" if hermes else "Agent routes/configuration must be migrated separately.\n"))
    if not confirm:
        return "Migration preview (no changes made):\n" + preview + "Stop Hermes and other clients, then repeat with --confirm."
    with ExitStack() as locks:
        for path in (Path(sys.prefix) / ".kvpark-update.lock", Path(sys.prefix) / ".sloth-memory-update.lock",
                     archive / ".hermes-start.lock"):
            fd = lock_file(path)
            locks.callback(os.close, fd)
        current = owned_process(archive, "proxy.json", legacy=True)
        if current != record:
            raise RuntimeError("Legacy proxy changed during migration; preview and retry.")
        prepared = False
        backup = None
        written = None
        try:
            if record:
                state = request(record["base_url"], "status", timeout=5, control_prefix="/_sloth/")
                if (state.get("service") != LEGACY or state.get("control_version", 0) < 4
                        or Path(state.get("archive_dir", "")).resolve() != archive):
                    raise RuntimeError("Legacy proxy cannot be drained; follow the manual migration guide.")
                request(record["base_url"], "prepare-update", {}, timeout=90, control_prefix="/_sloth/")
                prepared = True
            if hermes:
                with _locked_plugin_state(config.get_config_path()), config._CONFIG_LOCK:
                    editable()
                    if config.read_user_config_raw() != raw:
                        raise RuntimeError("Hermes configuration changed during migration; retry.")
                    backup = config.get_config_path().with_name("config.before-kvpark-" + uuid.uuid4().hex + ".json")
                    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, "w", encoding="utf-8") as out:
                        json.dump(raw, out, indent=2)
                    config.save_config(planned)
                    written = config.read_user_config_raw()
            if record:
                stop_proxy(record)
        except BaseException:
            # A failed stop or config save keeps the old installation usable.
            if hermes and backup:
                with _locked_plugin_state(config.get_config_path()), config._CONFIG_LOCK:
                    if written is not None and config.read_user_config_raw() == written:
                        config.save_config(raw)
            if prepared and record and process_matches(record):
                request(record["base_url"], "cancel-update", {}, timeout=5, control_prefix="/_sloth/")
            raise
    return ("Migration prepared. " + preview + (f"Private profile backup: {backup}\n" if backup else "")
            + ("Start Hermes, then check /kvpark status. Keep the old package until the new proxy works."
               if hermes else "Start kvpark serve with this same --archive-dir and your previous backend settings."))
