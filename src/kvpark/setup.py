"""Choose a discovered model and connect it without spelling out backend flags."""

from dataclasses import dataclass
import hashlib
import json
import os
import threading
import subprocess
import time

from . import discovery
from .hermes_service import Service, defaults, validate_service
from .runtime import directory


def saved_settings():
    try:
        return validate_service(json.loads((directory() / "setup.json").read_text()))
    except FileNotFoundError:
        return None


def clear_settings(archive):
    from .platforms import file_lock
    with file_lock(directory() / ".setup.lock"):
        config = saved_settings()
        if config and directory(config["archive_dir"]) == directory(archive):
            (directory() / "setup.json").unlink()


def hermes_settings():
    from hermes_cli.config import load_config
    raw = load_config()
    entry = raw.get("plugins", {}).get("entries", {}).get("kvpark", {})
    return {**defaults(), **entry.get("settings", entry.get("config", {})).get("service", {})}


def hermes_hint():
    from hermes_cli.config import load_config
    model = load_config().get("model", {})
    return model.get("base_url", "") if isinstance(model, dict) else ""


def connect_model(model, current, *, hermes=False):
    expected_model = None
    if hermes:
        from hermes_cli.config import read_user_config_raw
        from .hermes_uninstall import editable
        editable()
        expected_model = read_user_config_raw().get("model", {})
    discovery.verify(model)
    # Each server/model combination gets its own archive. Switching cannot take
    # over a running stack or invalidate another model's saved state.
    key = hashlib.sha256(f"{model.backend}\n{model.url}\n{model.model}".encode()).hexdigest()[:16]
    archive = directory() if os.environ.get("KVPARK_ARCHIVE_DIR") else directory() / "connections" / key
    config = validate_service({**defaults(), "backend": model.backend, "model": model.model,
                               "upstream_url": model.url, "cache_mode": "routing", "archive_dir": str(archive)})
    if (current.get("upstream_url"), current.get("model"), current.get("cache_mode")) == (model.url, model.model, "routing"):
        config = validate_service({**current, "autostart": True, "external": False, "uninstalled": False})
    def remember(updated):
        config.update(updated)
    service = Service(lambda: config, remember)
    try:
        if not service.ensure():
            raise RuntimeError("The connection stopped during setup. Start the model server and scan again.")
        config["connected"] = True
        if hermes:
            from .hermes_uninstall import save_route
            save_route(config, {"provider": "custom", "default": model.model, "base_url": config["base_url"] + "/v1",
                                "api_mode": "chat_completions"}, service=config, expected_model=expected_model)
        else:
            from .proxy import atomic_json
            from .platforms import file_lock
            with file_lock(directory() / ".setup.lock"):
                if (saved_settings() or defaults()) != current:
                    raise ValueError("Another setup changed this connection. Scan and choose again.")
                atomic_json(directory() / "setup.json", config)
    except BaseException:
        # Only clean up children started by this attempt; attached servers stay running.
        for child in reversed(service.started_processes):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        raise
    return config


@dataclass
class Pending:
    scan: discovery.Scan
    settings: dict
    created: float
    selected: object = None


class Guide:
    def __init__(self):
        self.pending = {}
        self.lock = threading.RLock()

    def handle(self, args, *, owner, current, apply, extra=(), prefix="/kvpark setup"):
        with self.lock:
            now = time.monotonic()
            self.pending = {key: value for key, value in self.pending.items() if now - value.created < 600}
            if not args or args == ["scan"] or args[:1] == ["address"]:
                if args[:1] == ["address"]:
                    if len(args) != 2:
                        return f"Paste the address from your model app after {prefix} address"
                    scan = discovery.inspect_server(args[1])
                else:
                    scan = discovery.discover([current.get("upstream_url", ""), *extra])
                if len(self.pending) >= 64:
                    self.pending.pop(next(iter(self.pending)))
                self.pending[owner] = Pending(scan, dict(current), now)
                lines = ["Choose the model you want to use:"] if scan.models else ["No available models found yet."]
                for index, model in enumerate(scan.models, 1):
                    lines.append(f"{index}. {model.label}")
                lines += ["0. None of these / set up later"]
                lines += scan.notes
                if scan.models:
                    lines += [f"Choose a number with {prefix} 1 (or {prefix} 0 to leave everything as it is)."]
                else:
                    lines += ["Open your model app, load a model, and try Scan again."]
                lines += [f"Scan again: {prefix} scan", f"Another address: {prefix} address http://host:port"]
                return "\n".join(lines)
            if args in (["0"], ["cancel"], ["no"]):
                self.pending.pop(owner, None)
                return "Setup cancelled. Your current model and settings are unchanged."
            pending = self.pending.get(owner)
            if pending is None:
                return f"Start with {prefix} to find available models. Choices expire after 10 minutes."
            if len(args) == 1 and args[0].isdigit():
                index = int(args[0]) - 1
                if not 0 <= index < len(pending.scan.models):
                    return f"Choose one of the listed numbers, or {prefix} 0 to cancel."
                pending.selected = pending.scan.models[index]
                return (f"Use {pending.selected.label}?\n"
                        "This connects future conversations through kvpark. It does not replace your model server.\n"
                        "Disk park/resume is unavailable for this existing-server connection.\n"
                        f"Connect: {prefix} yes\nCancel: {prefix} 0")
            if args in (["yes"], ["confirm"]) and pending.selected:
                if current != pending.settings:
                    self.pending.pop(owner, None)
                    return f"Your settings changed during setup. Run {prefix} again to choose using the current settings."
                model = pending.selected
                config = apply(model)
                self.pending.pop(owner, None)
                return (f"Connected to {model.label}.\n"
                        f"kvpark is ready at {config['base_url']}/v1.")
            return f"Choose a listed number, {prefix} scan to refresh, or {prefix} 0 to cancel."


def terminal(*, hermes=False, input_stream=None, output_stream=None):
    import sys
    source = input_stream or sys.stdin
    output = output_stream or sys.stdout
    if not source.isatty():
        raise ValueError("Setup needs an interactive terminal. Run kvpark setup in a terminal, or /kvpark setup inside Hermes.")
    current = hermes_settings if hermes else lambda: saved_settings() or defaults()
    guide = Guide()
    prefix = "setup"
    def apply(model):
        return connect_model(model, current(), hermes=hermes)
    def step(args):
        text = guide.handle(args, owner="terminal", current=current(), apply=apply,
                            extra=[hermes_hint()] if hermes else [], prefix=prefix)
        # The terminal takes answers directly; slash-command instructions are only for chat.
        text = text.replace("Choose a number with setup 1 (or setup 0 to leave everything as it is).", "Enter a number below.")
        text = text.replace("Scan again: setup scan", "Type scan to search again.")
        text = text.replace("Another address: setup address http://host:port", "Or paste your server's address below.")
        text = text.replace("Connect: setup yes\nCancel: setup 0", "Type yes to connect, or 0 to cancel.")
        print(text, file=output, flush=True)
    print("Looking for model servers on this computer...", file=output, flush=True)
    step([])
    while guide.pending:
        print("\nYour choice (0 to finish later): ", end="", file=output, flush=True)
        answer = source.readline()
        if not answer:
            step(["0"])
            break
        answer = answer.strip()
        if answer.lower().startswith("setup "):
            answer = answer[6:].strip()
        if not answer:
            continue
        args = ["address", answer] if answer.startswith(("http://", "https://")) else [answer.lower()]
        try:
            step(args)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Could not connect: {exc}\nType scan to try again, or 0 to finish later.", file=output, flush=True)
    if hermes:
        print("Restart Hermes and start a new conversation to use a newly selected model.", file=output)
    elif saved_settings():
        print("Check the connection with kvpark doctor. After a reboot, run kvpark start.", file=output)
