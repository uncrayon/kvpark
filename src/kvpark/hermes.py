"""Native Hermes commands, startup, and request middleware; no core patches."""

import asyncio
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import re
import shlex

from .cli import request
from .hermes_service import Service, defaults, validate_service, target

_command_context = ContextVar("kvpark_command_context", default=None)


def session_value(name):
    # Environment fallbacks can point to a different chat in a shared gateway.
    try:
        from gateway.session_context import _VAR_MAP, _UNSET
        value = _VAR_MAP[name].get()
        return value if value is not _UNSET and value else None
    except (ImportError, KeyError):
        return None


def foreground_role(session_id):
    # Hermes binds the active agent for the duration of each turn, including
    # review forks. Read it only; never retain or modify the agent object.
    try:
        from agent.subagent_lifecycle import get_active_subagent_parent
        active = get_active_subagent_parent()
    except ImportError:
        return "unverified"
    if active is None or getattr(active, "session_id", None) != session_id:
        return "unverified"
    disabled = getattr(active, "_persist_disabled", None)
    return "foreground" if disabled is False else "background" if disabled is True else "unverified"


class Adapter:
    def __init__(self, ctx):
        self.ctx = ctx
        from hermes_constants import get_hermes_home
        self.home = Path(get_hermes_home()).resolve()
        self.namespace = "hermes:" + hashlib.sha256(str(self.home).encode()).hexdigest()[:16] + ":"
        self.service = Service(self.settings, self.persist_service)
        self.gateway = None
        from .setup import Guide
        self.setup_guide = Guide()

    def capture_gateway(self, *, gateway=None, **kwargs):
        self.gateway = gateway

    def persist_service(self, config):
        previous = self.settings()
        if previous != config:
            self.ctx.set_config("service", config)
        if config["connected"] and previous["base_url"] != config["base_url"]:
            self.write_route(config)

    def settings(self):
        return {**defaults(), **self.ctx.get_config("service", {})}

    def key(self, session_id):
        return "k-" + hashlib.sha256((self.namespace + session_id).encode()).hexdigest()

    def capture_command(self, *, command="", session_key=None, **kwargs):
        _command_context.set(session_key if command == "kvpark" else None)

    def middleware(self, *, request, session_id="", base_url="", api_mode="", **kwargs):
        config = self.settings()
        if config["uninstalled"]:
            return None
        routes = {url.rstrip("/") + "/v1" for url in [config["base_url"], *config["previous_urls"]]}
        if config["connected"]:
            routes.add(target(config).url + "/v1")
        if base_url.rstrip("/") not in routes or api_mode != "chat_completions":
            return None
        if config["autostart"]:
            self.service.ensure()
            config = self.settings()
        destination = config["base_url"] + "/v1"
        if base_url.rstrip("/") != destination:
            # Hermes' request middleware changes payloads, not transports. For
            # a bound native OpenAI client, update its documented base_url property
            # before the pending dispatch, keeping model, history and sampling.
            from agent.subagent_lifecycle import get_active_subagent_parent
            from openai import OpenAI
            active = get_active_subagent_parent()
            client = getattr(active, "client", None)
            if (getattr(active, "session_id", None) != session_id or not isinstance(client, OpenAI)
                    or str(client.base_url).rstrip("/") != base_url.rstrip("/")):
                return None
            client.base_url = destination
            active.base_url = destination
        payload = dict(request)
        extra = dict(payload.get("extra_body") or {})
        # Replace any inherited archive tags rather than trusting a fork's copy.
        for name in ("slot_archive_role", "slot_archive_key", "slot_archive_thread"):
            extra.pop(name, None)
        role = foreground_role(session_id)
        extra["slot_archive_role"] = role
        if role == "foreground" and session_id:
            extra["slot_archive_key"] = self.namespace + session_id
            thread = session_value("HERMES_SESSION_KEY")
            if thread:
                extra["slot_archive_thread"] = self.namespace + thread
        payload["extra_body"] = extra
        if os.environ.get("KVPARK_API_KEY"):
            headers = dict(payload.get("extra_headers") or {})
            headers["Authorization"] = "Bearer " + os.environ["KVPARK_API_KEY"]
            payload["extra_headers"] = headers
        return {"request": payload, "source": "kvpark", "reason": "verified conversation archive identity"}

    def call(self, action, payload=None):
        if self.settings()["autostart"]:
            self.service.ensure()
        return request(self.settings()["base_url"], action, payload)

    def onboarding(self):
        config = self.settings()
        lines = ["🅿️ kvpark setup", "Saved disk snapshots expire after 7 days by default; cleanup checks hourly.",
                 "A 32 GiB budget also limits saved snapshots. Your transcript and live RAM state are kept.",
                 f"Proxy: {config['base_url']}/v1", f"Archive: {config['archive_dir']}",
                 "Starts with Hermes: " + ("on" if config["autostart"] else "off")]
        lines += [f"Backend: {config['backend']} · cache mode: {config['cache_mode']}"]
        if config["cache_mode"] == "routing":
            lines.append("Disk resume is unavailable for this adapter; backend-managed caches remain independent.")
        if config["connected"]:
            lines.append("Connected. Start a new Hermes conversation to use this model.")
        else:
            lines.append("Run /kvpark setup to find a running model and choose it from a numbered list.")
        lines += ["Commands:", "  save | status | forget [k-…] | slots",
                  "  update check | update | uninstall [confirm]",
                  "  retention 7 | budget 32 | cleanup on|off|now",
                  "  base-url http://127.0.0.1:8080 | autostart on|off",
                  "Advanced setup and managed disk snapshots: docs/hermes.md",
                  "Restore is automatic when you return to a compatible conversation."]
        if self.service.error:
            lines.append("Startup needs attention: " + self.service.error)
        return "\n".join(lines)

    def configure(self, args):
        import argparse
        if self.settings()["uninstalled"] and self.service.closed.is_set():
            raise RuntimeError("Restart Hermes after re-enabling the plugin before running setup again")
        class Parser(argparse.ArgumentParser):
            def error(self, message):
                raise ValueError(message)
        parser = Parser(prog="/kvpark setup", add_help=False)
        for key in ("server", "model", "archive-dir", "base-url", "backend-args", "backend", "upstream-url", "cache-mode"):
            parser.add_argument("--" + key)
        parser.add_argument("--upstream-port", type=int)
        parser.add_argument("--external", action=argparse.BooleanOptionalAction, default=None)
        parsed = vars(parser.parse_args(args))
        updates = {key: value for key, value in parsed.items() if value is not None}
        # The desktop composer serializes pasted links as @url:`address`.
        # Decode only URL options; normal validation still checks the address.
        for key in ("upstream_url", "base_url"):
            value = updates.get(key, "")
            if value.startswith("@url:"):
                value = value[5:]
                if len(value) >= 2 and value[0] == value[-1] == "`":
                    value = value[1:-1]
                updates[key] = value
        if self.settings()["uninstalled"]:
            updates.update(uninstalled=False, autostart=True)
        if "backend" in updates and updates["backend"] != self.settings()["backend"]:
            # A model ID and endpoint belong to one backend. Reusing them while
            # changing engines can silently route to the previous engine.
            for key in ("model", "server", "upstream_url"):
                updates.setdefault(key, "")
        if "backend" in updates and "cache_mode" not in updates:
            updates["cache_mode"] = "native" if updates["backend"] == "llama.cpp" else "routing"
        if "backend_args" in updates:
            updates["backend_args"] = shlex.split(updates["backend_args"])
        result = validate_service({**self.settings(), **updates})
        self.ctx.set_config("service", result)
        ready = result["model"] and (result["server"] or result["upstream_url"] or result["backend"] != "llama.cpp")
        if ready and not result["external"]:
            return self.connect() + "\n" + self.onboarding()
        if result["autostart"]:
            self.service.start_async()
        return "Settings saved. " + self.onboarding()

    def current_key(self, explicit=None):
        if explicit:
            if not re.fullmatch(r"k-[0-9a-f]{64}", explicit):
                raise ValueError("use the exact k-… key from /kvpark slots")
            return explicit
        session = session_value("HERMES_SESSION_ID")
        if session:
            return self.key(session)
        thread = _command_context.get() or session_value("HERMES_SESSION_KEY")
        if thread:
            status = self.call("status")
            thread = self.namespace + thread
            if status.get("resident_thread") == thread and status.get("resident_key"):
                return status["resident_key"]
            matches = [entry["key"] for entry in status["entries"] if entry.get("thread") == thread]
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise ValueError("this conversation has no saved or resident state in kvpark yet. "
                                 "Send a normal message in this conversation, wait for the reply, then run "
                                 "/kvpark save. Restarting the gateway alone does not load its model state.")
            raise ValueError("this chat has multiple saved conversations; choose an exact key from /kvpark slots")
        raise ValueError("Hermes did not provide this command's conversation identity; use an exact key from /kvpark slots")

    def connect(self):
        self.service.ensure()
        settings = self.settings()
        self.write_route(settings)
        self.ctx.set_config("service", {**settings, "connected": True})
        return "Hermes now defaults to kvpark for new sessions. Restart Hermes and start a new conversation; existing sessions keep their saved route."

    def write_route(self, settings):
        # Explicit command selects the default for FUTURE sessions through Hermes'
        # normal config writer; live agents and prompt histories are not changed.
        from .hermes_uninstall import save_route
        model = "local" if settings["backend"] == "llama.cpp" and not settings["upstream_url"] else settings["model"]
        if not model:
            model = "local"  # legacy external llama.cpp proxy
        save_route(settings, {"provider": "custom", "default": model,
                             "base_url": settings["base_url"] + "/v1", "api_mode": "chat_completions"})

    def handle(self, raw_args):
        args = shlex.split(raw_args)
        action = args[0] if args else "setup"
        action = {"park": "save", "delete": "forget"}.get(action, action)
        rest = args[1:]
        if action == "uninstall" and rest in ([], ["confirm"]):
            from .hermes_uninstall import run
            result = run(confirm=bool(rest))
            if rest:
                self.service.closed.set()
            return result
        if self.settings()["uninstalled"] and action not in {"setup", "onboarding", "help"}:
            raise RuntimeError("kvpark is disconnected. Restart Hermes; see docs/uninstall.md to reinstall.")
        if action == "update" and rest in ([], ["check"]):
            from . import updates
            config = self.settings()
            if rest:
                result = updates.check(config["archive_dir"], url=config["base_url"] if config["external"] else None)
                lines = [f"Installed: {result['installed']} · Hermes loaded: {result['loaded']} · Proxy: {result['proxy']}"]
                if result.get("check_error"):
                    lines.append("Release check unavailable: " + result["check_error"])
                else:
                    lines.append("Available: " + result["latest"] if result["available"] else "No newer compatible published release.")
                if result.get("last_update"):
                    state = result["last_update"]
                    lines.append("Last update: " + state["state"] + ". " + state.get("note", ""))
                if result["loaded"] != result["installed"]:
                    lines.append("Restart Hermes to load the installed adapter.")
                return "\n".join(lines)
            from hermes_cli import config as hermes_config
            if hermes_config.is_managed():
                raise PermissionError("updates are managed by your administrator")
            result = updates.apply(config["archive_dir"], hermes_home=self.home, external=config["external"])
            return f"kvpark {result['version']}: {result['note']}"
        if action == "setup":
            if rest and rest[0].startswith("--"):
                return self.configure(rest)
            from .setup import connect_model, hermes_hint
            owner = _command_context.get() or session_value("HERMES_SESSION_ID") or "local"
            def apply(model):
                return connect_model(model, self.settings(), hermes=True)
            result = self.setup_guide.handle(rest, owner=owner, current=self.settings(), apply=apply, extra=[hermes_hint()])
            if result.startswith("Connected to"):
                result += "\nRestart Hermes and start a new conversation to use this model."
            return result
        if action in {"onboarding", "help"}:
            return self.onboarding()
        if action == "base-url" and len(rest) == 1:
            return self.configure(["--base-url", rest[0]])
        if action == "autostart" and rest in (["on"], ["off"]):
            config = {**self.settings(), "autostart": rest[0] == "on"}
            self.ctx.set_config("service", config)
            if config["autostart"]:
                self.service.start_async()
            return "Start with Hermes: " + rest[0] + ". Already running services are left running."
        if action == "start" and not rest:
            self.service.ensure()
            return "kvpark is running. " + json.dumps(self.call("doctor"))
        if action == "connect" and not rest:
            return self.connect()
        if action in {"retention", "budget"} and len(rest) == 1:
            value = float(rest[0])
            field = "ttl_days" if action == "retention" else "max_gib"
            result = self.call("settings", {field: value})
            return "Saved cleanup settings: " + json.dumps(result)
        if action == "cleanup" and rest in (["on"], ["off"], ["now"]):
            result = self.call("cleanup", {}) if rest == ["now"] else self.call("settings", {"cleanup_enabled": rest == ["on"]})
            return "Cleanup: " + json.dumps(result) + ". Snapshot files only; live RAM and transcripts are unchanged."
        if action in {"save", "forget"} and len(rest) <= 1:
            result = self.call(action, {"key": self.current_key(rest[0] if rest else None)})
            return (f"Parked {result['tokens']:,} tokens. " if action == "save" else "Deleted saved snapshot. ") + result["note"]
        if action in {"status", "slots"} and not rest:
            status = self.call("status")
            lines = [f"🅿️ {status['archived']} saved slots · {status['total_gib']} / {status['max_gib']} GiB",
                     f"Cleanup {'on' if status['cleanup_enabled'] else 'off'} · expires {status['ttl_days']} days after saving · checks hourly",
                     "Model: " + status["activity"]["phase"]]
            if status.get("runtime_error"):
                lines.append("Runtime needs attention: " + status["runtime_error"])
            if status.get("capabilities"):
                lines.append(status["capabilities"]["note"])
            for entry in status["entries"]:
                lines.append(f"{entry['key']} — {entry['tokens']:,} tokens, {entry['gib']} GiB, {entry['age_days']} days old")
            return "\n".join(lines)
        raise ValueError("Unknown command or arguments. Use /kvpark setup for available commands.")

    async def command(self, raw_args):
        try:
            result = await asyncio.to_thread(self.handle, raw_args)
            if shlex.split(raw_args) == ["update"]:
                from importlib.metadata import version
                from . import __version__
                if version("kvpark") != __version__ and self.gateway is not None:
                    from gateway.restart import is_container_restart_context, is_gateway_supervisor_process
                    supervised = is_gateway_supervisor_process() or is_container_restart_context()
                    # Let the command response be sent before Hermes drains and
                    # restarts through the same lifecycle used by /restart.
                    asyncio.get_running_loop().call_later(2, lambda: self.gateway.request_restart(
                        detached=not supervised, via_service=supervised))
                    result += " Hermes will restart shortly to load the new adapter."
            return result
        except (OSError, ValueError, RuntimeError) as exc:
            return "kvpark: " + str(exc)
        finally:
            _command_context.set(None)


def register(ctx):
    adapter = Adapter(ctx)
    ctx.register_command("kvpark", adapter.command,
                         description="Park model state, resume conversations, and manage saved caches",
                         args_hint="setup | save | status | forget | cleanup | retention | base-url | update | uninstall", argument_mode="mixed")
    # Hermes 0.21 exposes argument mode through register_command, but its CLI
    # completer and desktop catalog both read options from this shared registry.
    from hermes_cli.commands import SUBCOMMANDS
    options = ["setup", "save", "status", "forget", "slots", "start", "connect",
               "retention", "budget", "cleanup", "autostart", "base-url", "update", "uninstall", "help"]
    previous = SUBCOMMANDS.get("/kvpark")
    SUBCOMMANDS["/kvpark"] = options

    def remove_completions():
        if SUBCOMMANDS.get("/kvpark") is options:
            if previous is None:
                SUBCOMMANDS.pop("/kvpark", None)
            else:
                SUBCOMMANDS["/kvpark"] = previous

    ctx.on_unload(remove_completions)
    ctx.register_hook("pre_gateway_dispatch", adapter.capture_gateway)
    ctx.register_hook("pre_command", adapter.capture_command)
    ctx.register_middleware("llm_request", adapter.middleware)
    ctx.on_unload(adapter.service.closed.set)
    config = adapter.settings()
    if config["autostart"] and (config["external"] or config["model"]):
        adapter.service.start_async()
