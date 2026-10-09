"""Native Hermes commands, startup, and request middleware; no core patches."""

import asyncio
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import re
import shlex

from .backends import route_key
from .cli import request
from .hermes_service import Service, defaults, route_for, validate_service, target

_command_context = ContextVar("kvpark_command_context", default=None)


def command_args(raw):
    # Rich-text composers preserve nonbreaking spaces. Treat Unicode whitespace
    # as separators outside quotes, while keeping quoted model IDs and paths exact.
    lexer = shlex.shlex(raw, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.whitespace += "".join(char for char in set(raw) if char.isspace())
    return list(lexer)


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


def _engine_summary(report):
    """How the last switch's engine ran, for /kvpark switch status: "daemon on the GPU, 0.5 s"."""
    engine = report.get("engine") or {}
    if not engine:
        return None
    mode = report.get("engine_mode") or engine.get("engine_mode") or "engine"
    where = "the CPU" if engine.get("backend", "numpy") == "numpy" else "the GPU"
    seconds = report.get("timings", {}).get("engine_s")
    return f"{mode} on {where}" + (f", {seconds} s" if seconds is not None else "")


def _options_summary(report):
    """How the request's template options (reasoning_effort, chat_template_kwargs) were handled when
    they change the target's rendering: rendered by the engine too, or the cache cut where the
    renderings part (an older engine, or a server setting the engine cannot see)."""
    options = report.get("template_options")
    if not options:
        return None
    names = ", ".join(options.get("options") or ())
    if options.get("engine"):
        return f"rendered with the request's {names}"
    return f"the request's {names} render differently from token {options.get('shared_tokens')} on"


def _media_summary(report):
    """What happened to the conversation's media in a switch or a request rewrite, for /kvpark switch
    status: "media: 1 image as text (made by qwen38 in 22.4 s)", or None when it had none."""
    names = {"vision": "image", "audio": "audio clip", "video": "video"}
    parts = (report.get("surrogates") or {}).get("parts") or []
    out = []
    for modality in ("vision", "audio", "video"):
        mine = [part for part in parts if part.get("modality") == modality]
        if not mine:
            continue
        made = [part for part in mine if not part.get("cached") and not part.get("duplicate")
                and part.get("producer") != "conversation"]
        reused = [part for part in mine if part.get("cached") or part.get("producer") == "conversation"]
        repeated = [part for part in mine if part.get("duplicate")]
        how = []
        if made:
            producers = sorted({str(name) for part in made for name in (part.get("producer"), part.get("audio_producer"))
                                if name})
            how.append(f"made by {', '.join(producers)} in {round(sum(part.get('seconds') or 0 for part in made), 1)} s")
        if reused:
            how.append(f"{len(reused)} from the surrogate cache or the conversation")
        if repeated:
            how.append(f"{len(repeated)} repeated")
        silent = sum(1 for part in mine if part.get("sound"))
        if silent:
            how.append(f"the sound of {silent} not transcribed")
        frames = any(part.get("frames") for part in mine)
        out.append(f"{len(mine)} {names[modality]}{'s' if len(mine) > 1 else ''} as "
                   f"{'keyframes' if frames else 'text'} ({'; '.join(how)})")
    source_media = (report.get("engine") or {}).get("source_media")
    if source_media and not parts:
        # Hermes replaced the source's media by text itself (e.g. its own image description)
        out.append("the source's cache held media the request carries as text; translated up to them")
    return "media: " + ", ".join(out) if out else None


def known_route(config):
    try:
        return route_for(config)
    except ValueError:
        return None


class Adapter:
    def __init__(self, ctx):
        self.ctx = ctx
        from hermes_constants import get_hermes_home
        self.home = Path(get_hermes_home()).resolve()
        self.namespace = "hermes:" + hashlib.sha256(str(self.home).encode()).hexdigest()[:16] + ":"
        self.service = Service(self.settings, self.persist_service)
        self.gateway = None
        # Filled by register(): which optional Hermes plugin APIs this version offers.
        self.support = {"argument_mode": False, "completion": False}
        from .setup import Guide
        self.setup_guide = Guide()

    def capture_gateway(self, *, gateway=None, **kwargs):
        self.gateway = gateway

    def persist_service(self, config):
        previous = self.settings()
        if previous != config:
            self.ctx.set_config("service", config)
        if config["connected"] and known_route(previous) != known_route(config):
            self.write_route(config)

    def settings(self):
        return {**defaults(), **self.ctx.get_config("service", {})}

    def key(self, session_id):
        return "k-" + hashlib.sha256((self.namespace + session_id).encode()).hexdigest()

    def capture_command(self, *, command="", session_key=None, **kwargs):
        _command_context.set(session_key if command == "kvpark" else None)

    def rebind(self, session_id, base_url, destination):
        # Hermes' request middleware changes payloads, not transports. For a
        # bound native OpenAI client, update its documented base_url property
        # before the pending dispatch, keeping model, history and sampling.
        try:
            from agent.subagent_lifecycle import get_active_subagent_parent
            from openai import OpenAI
        except ImportError:
            return False
        active = get_active_subagent_parent()
        client = getattr(active, "client", None)
        if (getattr(active, "session_id", None) != session_id or not isinstance(client, OpenAI)
                or route_key(str(client.base_url)) != route_key(base_url)):
            return False
        client.base_url = destination
        active.base_url = destination
        return True

    def bridge_switch(self, request, session_id, base_url):
        """Run a staged `/kvpark switch` when this conversation's request reaches the target model;
        returns the source model's name when the stage was used (None otherwise).

        Never raises: on any failure the request goes out unchanged and the target prefills
        natively; the outcome is kept for `/kvpark switch status`."""
        if not session_id:
            return None
        from .bridge import registry, stage
        bridge_dir = registry.default_path().parent
        key = self.key(session_id)
        staged = stage.get(bridge_dir, key)
        if not staged:
            return None
        try:
            config = registry.load()
            model = config["models"].get(staged["target"])
            if model is None or route_key(model["url"] + "/v1") != route_key(base_url):
                return None     # not on the target yet; keep the stage until its lease ends
            stage.drop(bridge_dir, key)
            from .bridge.messages import conversation_from_openai, template_options
            from .bridge.switch import switch
            conversation = conversation_from_openai(request.get("messages"), request.get("tools"))
            report = switch(config, conversation, staged["source"], staged["target"], warm=False,
                            template_options=template_options(request))
            # a skipped switch (a model, medium or prompt the translator does not handle) is not an
            # error: the target prefills natively, and the reason is kept for the status
            self.bridge_last = {"ok": True, "source": staged["source"], "target": staged["target"],
                                "skipped": report.get("skipped"), "note": report.get("note"),
                                "restored_tokens": report.get("restored_tokens"),
                                "prompt_tokens": report.get("target_prompt_tokens"),
                                "seconds": report["timings"].get("total_s"),
                                "engine": _engine_summary(report), "options": _options_summary(report),
                                "media": _media_summary(report)}
        except Exception as exc:  # the request must go out whatever happens here
            self.bridge_last = {"ok": False, "source": staged.get("source"), "target": staged.get("target"),
                                "error": f"{type(exc).__name__}: {exc}"[:300]}
        return staged["source"]

    def bridge_media(self, request, base_url, source=None):
        """The request with text surrogates for the media its target cannot take, or None to send it
        unchanged: when it goes to a registered bridge model whose server lacks a modality its
        messages hold (an image for a text-only model), each such part becomes the surrogate the
        bridge made for it (cached by a switch, or made now by `source` or a describer), so the
        target's prompt matches what a switch translated. Never raises."""
        from .bridge import surrogates
        if not surrogates.request_media(request.get("messages")):
            return None
        target_name = None
        try:
            from .bridge import registry
            config = registry.load()
            target_name = next((name for name, model in config["models"].items()
                                if route_key(model["url"] + "/v1") == route_key(base_url)), None)
            if target_name is None:
                return None
            rewritten, summary = surrogates.rewrite_request(config, request, target_name,
                                                            bridge_dir=registry.default_path().parent, source=source)
            if summary is not None:
                self.bridge_media_last = {"ok": True, "target": target_name, "rewritten": rewritten is not None,
                                          "media": _media_summary({"surrogates": summary}),
                                          "missing": summary.get("missing"), "skipped": summary.get("skipped")}
            return rewritten
        except Exception as exc:  # the request goes out as it is
            self.bridge_media_last = {"ok": False, "target": target_name, "error": f"{type(exc).__name__}: {exc}"[:300]}
            return None

    def stage_switch(self, rest):
        from .bridge import registry, stage
        if rest == ["status"]:
            return self.switch_status()
        if len(rest) not in (1, 3) or (len(rest) == 3 and rest[1] != "from"):
            raise ValueError("use /kvpark switch MODEL [from CURRENT_MODEL], or /kvpark switch status")
        config = registry.load()
        target_name = rest[0]
        registry.model(config, target_name)
        if len(rest) == 3:
            source_name = rest[2]
            registry.model(config, source_name)
        else:
            upstream = route_key(target(self.settings()).url + "/v1")
            matches = [name for name, model in config["models"].items() if route_key(model["url"] + "/v1") == upstream]
            if len(matches) != 1:
                raise ValueError("name the model you are leaving: /kvpark switch MODEL from CURRENT_MODEL")
            source_name = matches[0]
        if source_name == target_name:
            raise ValueError("that is the current model")
        if config["models"][source_name].get("source") is False:
            raise ValueError(f"{source_name} is registered as a target only (kvpark bridge add --no-source): its cache "
                             f"is not translated, so {target_name} reads the conversation itself; switch with /model alone")
        session = session_value("HERMES_SESSION_ID")
        if not session:
            raise ValueError("run /kvpark switch inside the conversation you want to move")
        stage.put(registry.default_path().parent, self.key(session), source=source_name, target=target_name)
        # the engine loads the pair while the user moves to the target (never fails the stage)
        stage.prewarm(config, source_name, target_name, registry.default_path().parent)
        return (f"Ready: this conversation's next message to {target_name} continues from {source_name}'s cache "
                f"(valid {stage.LEASE_SECONDS // 60} minutes). Switch Hermes to {target_name} with /model, then send "
                f"your message. /kvpark switch status shows the result.")

    def switch_status(self):
        last, media = getattr(self, "bridge_last", None), getattr(self, "bridge_media_last", None)
        lines = []
        if not last:
            lines.append("No bridge switch has run in this Hermes process yet.")
        elif last["ok"] and last.get("skipped"):
            lines.append(f"Last switch {last['source']} → {last['target']} was skipped ({last['skipped']}): "
                         f"{last['note']}.")
        elif last["ok"]:
            details = [f"engine: {last['engine']}"] if last.get("engine") else []
            details += [last[key] for key in ("options", "media") if last.get(key)]
            details = f" ({'; '.join(details)})" if details else ""
            lines.append(f"Last switch {last['source']} → {last['target']}: restored {last['restored_tokens']:,} of "
                         f"{last['prompt_tokens']:,} prompt tokens in {last['seconds']} s{details}.")
        else:
            lines.append(f"Last switch {last['source']} → {last['target']} fell back to a full prefill: {last['error']}")
        if media and not media["ok"]:
            lines.append(f"Last request with media for {media['target']} went out unchanged: {media['error']}")
        elif media and media["rewritten"]:
            lines.append(f"Last request with media for {media['target']}: {media['media'] or 'media: none replaced'}.")
        elif media:
            why = (f"its {', '.join(media['skipped'])} media policy is skip" if media.get("skipped")
                   else "; ".join((media.get("missing") or {}).values()))
            lines.append(f"Last request with media for {media['target']} went out unchanged: {why}.")
        return "\n".join(lines)

    def middleware(self, *, request, session_id="", base_url="", api_mode="", **kwargs):
        config = self.settings()
        if config["uninstalled"] or api_mode != "chat_completions":
            return None
        source = self.bridge_switch(request, session_id, base_url)
        rewritten = self.bridge_media(request, base_url, source=source)
        routed = self.route(rewritten or request, session_id, base_url, config)
        if routed is None and rewritten is not None:
            # a request to a model server kvpark does not proxy still carries the surrogates
            return {"request": rewritten, "source": "kvpark", "reason": "bridge media surrogates"}
        return routed

    def route(self, request, session_id, base_url, config):
        """kvpark's proxy for this request (archive tags, API key), or None to leave its route alone."""
        routes = {route_key(url + "/v1") for url in [config["base_url"], *config["previous_urls"]]}
        upstream = target(config).url + "/v1" if config["connected"] else None
        if upstream:
            routes.add(route_key(upstream))
        if route_key(base_url) not in routes:
            return None
        if config["autostart"]:
            try:
                self.service.ensure()
            except (OSError, ValueError, RuntimeError) as exc:
                # Hermes keeps talking to the model server itself until the proxy
                # is back; parking is unavailable meanwhile. A session already
                # bound to the proxy is pointed back at the server.
                self.service.error = str(exc)
                if upstream and route_key(base_url) != route_key(upstream):
                    self.rebind(session_id, base_url, upstream)
                return None
            config = self.settings()
        elif not self.service.available(config):
            if upstream and route_key(base_url) != route_key(upstream):
                self.rebind(session_id, base_url, upstream)
            return None
        destination = config["base_url"] + "/v1"
        if route_key(base_url) != route_key(destination) and not self.rebind(session_id, base_url, destination):
            return None
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
        lines = ["🅿️ kvpark",
                 f"Proxy: {config['base_url']}/v1", f"Archive: {config['archive_dir']}",
                 "Starts with Hermes: " + ("on" if config["autostart"] else "off")]
        if config["model"]:
            lines.append("Configured model: " + config["model"])
        lines += [f"Backend: {config['backend']} · cache mode: {config['cache_mode']}"]
        if config["cache_mode"] == "routing":
            lines.append("Chat routing only. kvpark cannot save or resume disk snapshots with this backend connection. "
                         "Repeating setup will not enable disk snapshots; backend-managed caches are independent.")
        if config["connected"]:
            route = known_route(config)
            if route:
                lines.append(f"Hermes route: {route['default']} at {route['base_url']}; kvpark inserts its proxy while it runs.")
            lines.append("Connection configured. Use /kvpark doctor to check it; setup is only needed to change models.")
        else:
            lines.append("Run /kvpark setup to find a running model and choose it from a numbered list.")
        lines += ["Commands:", "  status | doctor | start | setup | help",
                  "  update check | update | uninstall [confirm]",
                  "  retention 7 | budget 32 | cleanup on|off|now",
                  "  base-url http://127.0.0.1:8080 | autostart on|off",
                  "Advanced setup and managed disk snapshots: docs/hermes.md"]
        if config["cache_mode"] == "native":
            lines += ["  save | forget [k-…] | slots",
                      "Saved disk snapshots expire after 7 days by default; cleanup checks hourly with a 32 GiB budget.",
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
        route = route_for(settings)
        return (f"Hermes now uses {route['default']} at {route['base_url']} for new sessions. kvpark inserts its "
                "proxy automatically while it is running, including sessions already on this server. "
                "Sessions on other routes are unchanged.")

    def write_route(self, settings):
        # Hermes keeps the model server as its route; kvpark is inserted per request
        # by the middleware. Written through Hermes' normal config writer for FUTURE
        # sessions; live agents and prompt histories are not changed.
        from .hermes_uninstall import save_route
        save_route(settings, route_for(settings))

    def handle(self, raw_args):
        args = command_args(raw_args)
        action = args[0] if args else "help" if self.settings()["connected"] else "setup"
        action = {"park": "save", "delete": "forget", "--help": "help", "-h": "help"}.get(action, action)
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
            from . import updates, __version__
            from .hermes_doctor import install_source
            source, detail = install_source()
            if source != "pip":
                return (f"kvpark {__version__} runs from {detail}, not an installed package. Update it with "
                        "`hermes plugins update kvpark` (or git pull in a checkout), then restart Hermes.")
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
                result += ("\nHermes uses this model for new sessions; kvpark inserts its proxy automatically while it runs. "
                           "Restart Hermes and start a new conversation to switch an existing chat.")
            return result
        if action in {"onboarding", "help"}:
            return self.onboarding()
        if action == "switch" and rest:
            return self.stage_switch(rest)
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
        if action == "doctor" and not rest:
            from .hermes_doctor import report
            return "\n".join(report(self)[0])
        if action in {"retention", "budget"} and len(rest) == 1:
            value = float(rest[0])
            field = "ttl_days" if action == "retention" else "max_gib"
            result = self.call("settings", {field: value})
            return "Saved cleanup settings: " + json.dumps(result)
        if action == "cleanup" and rest in (["on"], ["off"], ["now"]):
            result = self.call("cleanup", {}) if rest == ["now"] else self.call("settings", {"cleanup_enabled": rest == ["on"]})
            return "Cleanup: " + json.dumps(result) + ". Snapshot files only; live RAM and transcripts are unchanged."
        if action in {"save", "forget"} and len(rest) <= 1:
            if action == "save" and not target(self.settings()).snapshots:
                raise ValueError("Disk save/resume is unavailable for this backend connection. "
                                 "kvpark can route chat requests, but cannot save this server's KV state. "
                                 "Repeating setup will not enable it. Use /kvpark status to inspect the connection.")
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
        raise ValueError("Unknown command or arguments. Use /kvpark help for commands, "
                         "or /kvpark status to check your existing connection. "
                         "To choose model 1 from the setup list, use /kvpark setup 1.")

    async def command(self, raw_args):
        try:
            result = await asyncio.to_thread(self.handle, raw_args)
            if command_args(raw_args) == ["update"]:
                from importlib.metadata import PackageNotFoundError, version
                from . import __version__
                try:
                    installed = version("kvpark")
                except PackageNotFoundError:
                    installed = __version__
                if installed != __version__ and self.gateway is not None:
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
        except (ImportError, AttributeError) as exc:
            from .hermes_doctor import TESTED, hermes_version
            return (f"kvpark: {exc}. This Hermes ({hermes_version()}) differs from the {TESTED}.x series kvpark is "
                    "tested with. Run `kvpark doctor --hermes` in a terminal and see docs/hermes.md.")
        finally:
            _command_context.set(None)


def register(ctx):
    adapter = Adapter(ctx)
    command = dict(description="Park model state, resume conversations, and manage saved caches",
                   args_hint="setup | save | status | doctor | forget | cleanup | retention | base-url | switch | update | uninstall")
    try:
        ctx.register_command("kvpark", adapter.command, argument_mode="mixed", **command)
        adapter.support["argument_mode"] = True
    except TypeError:
        # Hermes before 0.21.0 has no argument modes; the command works without suggestions.
        ctx.register_command("kvpark", adapter.command, **command)
    # Hermes 0.21 exposes argument mode through register_command, but its CLI
    # completer and desktop catalog both read options from this shared registry.
    try:
        from hermes_cli.commands import SUBCOMMANDS
    except ImportError:
        SUBCOMMANDS = None
    if SUBCOMMANDS is not None:
        options = ["setup", "save", "status", "doctor", "forget", "slots", "start", "connect",
                   "retention", "budget", "cleanup", "autostart", "base-url", "update", "uninstall", "help"]
        previous = SUBCOMMANDS.get("/kvpark")
        SUBCOMMANDS["/kvpark"] = options
        adapter.support["completion"] = True

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
    # Installers and doctors load the plugin to verify it; they must not start services.
    if config["autostart"] and (config["external"] or config["model"]) and not os.environ.get("KVPARK_NO_AUTOSTART"):
        adapter.service.start_async()
