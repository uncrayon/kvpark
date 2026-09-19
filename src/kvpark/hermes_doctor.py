"""Explain how kvpark is installed inside Hermes and whether requests can reach its proxy."""

from importlib import metadata
import os
from pathlib import Path
import sys

from . import __version__
from .backends import route_key
from .cli import request
from .hermes_service import route_for, target

TESTED = "0.21"  # Hermes series the adapter is tested against; 0.20.1 and newer load.


def hermes_version():
    try:
        import hermes_cli
    except ImportError:
        return None
    return str(getattr(hermes_cli, "__version__", "unknown"))


def install_source():
    """("pip", version) for an installed distribution; ("directory", path) for a checkout or plugin directory."""
    import kvpark
    package = Path(kvpark.__file__).resolve().parent
    try:
        dist = metadata.distribution("kvpark")
        if Path(str(dist.locate_file("kvpark"))).resolve() == package:
            return "pip", dist.version
    except metadata.PackageNotFoundError:
        pass
    return "directory", str(package.parent)


def route_state(model, settings):
    """How Hermes' model route relates to kvpark: upstream, proxy (legacy), other, or unset."""
    current = model.get("base_url", "") if isinstance(model, dict) else ""
    if not current:
        return "unset"
    key = route_key(current)
    if settings["connected"]:
        try:
            if key == route_key(route_for(settings)["base_url"]):
                return "upstream"
        except ValueError:
            pass
    if key in {route_key(url + "/v1") for url in [settings["base_url"], *settings["previous_urls"]]}:
        return "proxy"
    return "other"


def route_line(settings):
    try:
        from hermes_cli.config import load_config
        model = load_config().get("model", {})
    except Exception as exc:  # Hermes config reading is outside kvpark's control.
        return f"Hermes route: unavailable — {exc}"
    current = model.get("base_url", "") if isinstance(model, dict) else ""
    state = route_state(model, settings)
    if state == "upstream":
        return f"Hermes route: {current} — the model server itself; kvpark inserts its proxy per request while it runs."
    if state == "proxy":
        return (f"Hermes route: {current} — points at the kvpark proxy directly (legacy). Run /kvpark connect so "
                "Hermes falls back to the model server whenever the proxy is not running.")
    if state == "other":
        return (f"Hermes route: {current} — not a kvpark connection. Run /kvpark setup, or /kvpark connect to use "
                "the configured connection by default.")
    return "Hermes route: not configured. Run /kvpark setup."


def plugin_line(adapter, source=None):
    support = getattr(adapter, "support", {})
    where = f" ({source})" if source else ""
    suggestions = ("argument suggestions on" if support.get("completion")
                   else "argument suggestions need Hermes 0.21.0 or newer")
    return f"Plugin: loaded{where}; /kvpark registered; {suggestions}"


def loaded_adapter():
    """Load plugins the way Hermes does, without starting services; find kvpark's adapter."""
    from hermes_cli.plugins import get_plugin_command_handler, get_plugin_manager
    previous = os.environ.get("KVPARK_NO_AUTOSTART")
    os.environ["KVPARK_NO_AUTOSTART"] = "1"
    try:
        manager = get_plugin_manager()
        manager.discover_and_load()
    finally:
        if previous is None:
            os.environ.pop("KVPARK_NO_AUTOSTART", None)
        else:
            os.environ["KVPARK_NO_AUTOSTART"] = previous
    info = next((entry for entry in manager.list_plugins() if entry["name"] == "kvpark"), None)
    if info is None:
        return None, ("Plugin: Hermes did not find kvpark in this Python environment. Install it into Hermes' "
                      "environment, or as a plugin directory with: hermes plugins install uncrayon/kvpark")
    if not info.get("enabled") and "enabled" in str(info.get("error") or ""):
        # Hermes reports "disabled via config" or "not enabled in config" without loading.
        return None, "Plugin: found but not enabled in this Hermes profile. Enable it with: hermes plugins enable kvpark"
    if info.get("error") or not info.get("enabled"):
        return None, (f"Plugin: found ({info['source']}) but Hermes could not load it: {info.get('error') or 'not enabled'}. "
                      "Update Hermes or kvpark; see docs/hermes.md.")
    adapter = getattr(get_plugin_command_handler("kvpark"), "__self__", None)
    if adapter is None:
        return None, "Plugin: loaded, but /kvpark is not registered. Check the Hermes log for plugin warnings."
    return adapter, plugin_line(adapter, info["source"])


def connection_line(settings):
    try:
        upstream = target(settings).url
    except ValueError as exc:
        return f"Connection: invalid settings — {exc}"
    line = (f"Connection: backend {settings['backend']} · model {settings['model'] or 'not configured'} · "
            f"upstream {upstream} · cache mode {settings['cache_mode']}")
    return line if settings["connected"] else line + " · not connected yet (run /kvpark setup)"


def proxy_line(adapter, settings, *, start):
    try:
        result = adapter.call("doctor") if start else request(settings["base_url"], "doctor", timeout=5)
    except (OSError, ValueError, RuntimeError) as exc:
        return f"Connection check: needs attention — {exc}"
    detail = result.get("reason") or result.get("note", "")
    return "Connection check: " + ("OK" if result.get("ok") else "needs attention") + (f" — {detail}" if detail else "")


def report(adapter=None):
    """Report lines for a terminal or /kvpark doctor; False when Hermes cannot use kvpark.

    With an adapter (inside Hermes) the running plugin is described and services may
    start as usual. Without one (``kvpark doctor --hermes``), plugins are loaded the
    way Hermes loads them, nothing is started, and the proxy is only probed.
    """
    version = hermes_version()
    if version is None:
        return [f"Hermes is not importable from {sys.executable}.",
                "Run this command with the Python environment that runs Hermes; see docs/hermes.md."], False
    source, detail = install_source()
    if source == "pip":
        origin = "installed package" + ("" if detail == __version__ else f" (metadata says {detail}; restart Hermes to load it)")
    else:
        origin = "directory " + detail
    lines = [f"Hermes {version} (kvpark is tested with {TESTED}.x; 0.20.1 or newer loads) · Python {sys.executable}",
             f"kvpark {__version__} from {origin}"]
    inside = adapter is not None
    if inside:
        lines.append(plugin_line(adapter))
    else:
        adapter, note = loaded_adapter()
        lines.append(note)
        if adapter is None:
            return lines, False
    settings = adapter.settings()
    if settings["uninstalled"]:
        lines.append("kvpark is disconnected in this profile. See docs/uninstall.md to reconnect or remove it.")
        return lines, True
    lines.append(connection_line(settings))
    lines.append(route_line(settings))
    if settings["connected"] or settings["external"]:
        lines.append(proxy_line(adapter, settings, start=inside))
    else:
        lines.append("Connection check: skipped until a model is chosen with /kvpark setup.")
    if adapter.service.error:
        lines.append("Startup needs attention: " + adapter.service.error)
    return lines, True
