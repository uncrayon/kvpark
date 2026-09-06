"""Record and restore the model fields selected by the Hermes adapter."""

import copy

from . import removal
from .hermes_service import defaults

FIELDS = ("provider", "default", "base_url", "api_mode")


def editable():
    from hermes_cli import config, managed_scope
    paths = ["model." + field for field in FIELDS]
    paths += ["plugins.enabled", "plugins.disabled", "plugins.entries.kvpark"]
    managed = managed_scope.managed_config_keys()
    overlaps = any(key == path or key.startswith(path + ".") or path.startswith(key + ".")
                   for key in managed for path in paths)
    if config.is_managed() or overlaps:
        raise PermissionError("model routing or plugin settings are managed by your administrator")


def save_route(settings, model):
    from hermes_cli import config
    from hermes_cli.plugins import _locked_plugin_state
    editable()
    with _locked_plugin_state(config.get_config_path()), config._CONFIG_LOCK:
        raw = config.read_user_config_raw()
        entry = raw.get("plugins", {}).get("entries", {}).get("kvpark", {})
        backup = entry.get("settings", {}).get("route_backup")
        applied = entry.get("settings", {}).get("route_applied", {})
        current = raw.get("model", {})
        if isinstance(current, str):
            current = {"default": current}
            raw["model"] = current
        urls = {url.rstrip("/") + "/v1" for url in [settings["base_url"], *settings["previous_urls"]]}
        current_url = current.get("base_url", "").rstrip("/")
        # A refused preferred port can belong to the user's original backend.
        # Only an applied route (or an already-connected legacy install) proves
        # that an address in previous_urls actually belonged to kvpark.
        owned_route = (bool(applied) and current_url == applied.get("base_url", "").rstrip("/"))
        legacy_route = not applied and settings["connected"] and current_url in urls
        if not owned_route and not legacy_route:
            backup = {field: copy.deepcopy(current[field]) for field in FIELDS if field in current}
        saved = raw.setdefault("plugins", {}).setdefault("entries", {}).setdefault("kvpark", {}).setdefault("settings", {})
        saved.update(route_backup=backup, route_applied=model)
        raw.setdefault("model", {}).update(model)
        config.save_config(raw)


def restored_config(raw):
    result = copy.deepcopy(raw)
    plugins = result.setdefault("plugins", {})
    entry = plugins.setdefault("entries", {}).setdefault("kvpark", {})
    settings = entry.setdefault("settings", {})
    service = {**defaults(), **settings.get("service", entry.get("config", {}).get("service", {}))}
    model = result.get("model", {})
    # Hermes also accepts the legacy shorthand model: "provider/model".
    route_model = {"default": model} if isinstance(model, str) else model
    applied = settings.get("route_applied", {})
    # Historical preferred ports may belong to upstream servers, not kvpark.
    urls = {applied["base_url"].rstrip("/")} if applied.get("base_url") else {service["base_url"].rstrip("/") + "/v1"}
    restore = route_model.get("base_url", "").rstrip("/") in urls
    if restore:
        backup = settings.get("route_backup")
        if not isinstance(backup, dict):
            raise ValueError("This installation has no original model-route backup. Select your original backend "
                             "in Hermes first, then retry uninstall. See docs/uninstall.md.")
        if any(model.get(field) != applied.get(field) for field in FIELDS):
            raise ValueError("Model routing changed after setup. Select your original backend in Hermes first, "
                             "then retry uninstall; your edits have been kept.")
        for field in FIELDS:
            model.pop(field, None)
            if field in backup:
                model[field] = copy.deepcopy(backup[field])
    service.update(autostart=False, connected=False, uninstalled=True)
    settings["service"] = service
    plugins["enabled"] = [name for name in plugins.get("enabled", []) if name != "kvpark"]
    plugins["disabled"] = list(dict.fromkeys([*plugins.get("disabled", []), "kvpark"]))
    return result, service, restore


def run(*, confirm=False):
    from hermes_cli import config
    from hermes_cli.plugins import _locked_plugin_state
    editable()
    raw = config.read_user_config_raw()
    _, settings, restore = restored_config(raw)
    options = dict(external=settings["external"],
                   managed_backend=settings["backend"] == "llama.cpp" and not settings["upstream_url"])
    processes = removal.plan(settings["archive_dir"], **options)
    lines = ["Restore the previous Hermes model route." if restore else "Keep the current Hermes model route.",
             "Disable the plugin and its automatic startup."]
    for field in ("proxy", "backend"):
        if processes[field]:
            lines.append(f"Stop kvpark's {field} (PID {processes[field]['pid']}).")
    if settings["external"]:
        lines.append("The externally managed proxy remains running.")
    lines += ["Keep saved snapshots, model weights, runtime builds, and transcripts.",
              "Archive: " + settings["archive_dir"]]
    if not confirm:
        return "Uninstall preview (no changes made):\n" + "\n".join(lines) + "\nFinish other turns first, then run /kvpark uninstall confirm."

    def disconnect():
        # Re-read under the same locks as Hermes' native plugin config writer.
        with _locked_plugin_state(config.get_config_path()), config._CONFIG_LOCK:
            editable()
            current = config.read_user_config_raw()
            if current != raw:
                raise RuntimeError("Hermes configuration changed during uninstall; preview and retry.")
            restored, _, _ = restored_config(current)
            config.save_config(restored)

    removal.stop(settings["archive_dir"], disconnect=disconnect, **options)
    return ("kvpark disconnected; owned services stopped and automatic startup disabled. "
            "Restart Hermes and start a new conversation to use your restored route. "
            "Existing conversations can retain the old proxy URL.\n"
            "Snapshots and runtime builds are kept. Archive: " + settings["archive_dir"] + "\n"
            "After closing every Hermes process using this Python environment, remove the package in a terminal:\n"
            "python -m pip uninstall kvpark\nUse the same Python environment that runs Hermes.")
