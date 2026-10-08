"""Small control surface; inference remains on the HTTP proxy."""

import argparse
import hashlib
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import __version__
from .runtime import directory
from .backends import BACKENDS

# sysexits EX_CONFIG: retrying cannot fix it. The documented service units list it in
# RestartPreventExitStatus, so a broken setup stops instead of restarting every 10 seconds.
EX_CONFIG = 78


class SetupError(ValueError):
    """kvpark start cannot run until the user changes the setup."""


def request(url, action, payload=None, *, timeout=900, control_prefix="/_kvpark/"):
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("KVPARK_API_KEY")
    if control_prefix == "/_sloth/":
        key = os.environ.get("SLOTH_API_KEY", key)
    if key:
        headers["Authorization"] = "Bearer " + key
    data = None if payload is None else json.dumps(payload).encode()
    try:
        with urlopen(Request(url.rstrip("/") + control_prefix + action, data=data, headers=headers), timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            reason = json.loads(exc.read()).get("reason", str(exc))
        except ValueError:
            reason = str(exc)
        raise RuntimeError(reason) from None


def stop_services(*, hermes=False, archive=None, configured=None):
    """Drain the proxy, save the conversation in the slot, and stop the proxy and the managed model
    server; the setup is kept, so kvpark start (or the boot service) starts them again."""
    from . import removal
    if hermes:
        try:
            from .setup import hermes_settings
            config = hermes_settings()
        except ImportError as exc:
            raise RuntimeError(f"Hermes cannot load in this Python environment ({exc}); "
                               "stop the connection with kvpark stop --archive-dir DIR instead") from None
    else:
        config = configured if configured and directory(configured["archive_dir"]) == directory(archive) else None
    archive = directory(config["archive_dir"] if config else archive)
    options = dict(external=config["external"], managed_backend=config["backend"] == "llama.cpp"
                   and not config["upstream_url"]) if config else {}
    plan = removal.plan(archive, **options)
    if not (plan["proxy"] or plan["backend"]):
        return [f"Nothing to stop: no kvpark proxy or model server of {archive} is running"
                + (" (its proxy is managed elsewhere)." if options.get("external") else ".")]
    resident = None
    if plan["proxy"]:
        try:
            resident = request(plan["proxy"]["base_url"], "status", timeout=5).get("resident_key")
        except (OSError, ValueError, RuntimeError, URLError):
            pass
    result = removal.stop(archive, **options)
    stopped = [f"{name} (PID {result[field]['pid']})" for name, field in (("proxy", "proxy"), ("model server", "backend"))
               if result[field]]
    return [f"Stopped the {' and the '.join(stopped)} for {archive}."
            + (" The conversation in the slot was saved first." if resident else ""),
            "Start again with kvpark start" + (" --hermes" if hermes else "") + "."]


def main():
    parser = argparse.ArgumentParser(description="Park your model's KV cache. Resume where you left off.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="find running models and connect using numbered choices")
    setup_mode = setup.add_mutually_exclusive_group()
    setup_mode.add_argument("--hermes", action="store_true", help="configure Hermes in this Python environment")
    setup_mode.add_argument("--standalone", action="store_true", help="configure standalone kvpark")
    setup_mode.add_argument("--opencode", action="store_true", help="point OpenCode at kvpark (docs/opencode.md)")
    opencode = setup.add_argument_group("OpenCode (with --opencode)")
    opencode.add_argument("--project", help="write to this project's .opencode directory, not the global config")
    opencode.add_argument("--server", help="kvpark's patched llama-server, to run with --model (disk park/resume)")
    opencode.add_argument("--model", help="the GGUF file with --server; otherwise the model ID the proxy serves")
    opencode.add_argument("--backend-args", help="llama-server arguments for --server, as one quoted string")
    opencode.add_argument("--url", help="a running kvpark proxy to use instead, such as Hermes' (http://127.0.0.1:8080)")
    opencode.add_argument("--context", type=int, help="the model's context window in tokens, if the server does not report it")
    start = sub.add_parser("start", help="restart the connection selected with setup")
    start.add_argument("--archive-dir")
    start.add_argument("--hermes", action="store_true", help="start the connection configured in the active Hermes profile")
    start.add_argument("--supervise", action="store_true",
                       help="keep running and relaunch stopped services; for a systemd or launchd service")
    start.add_argument("--interval", type=float, default=60, help="seconds between supervise checks (default 60)")
    stop = sub.add_parser("stop", help="save the conversation in the slot, then stop the proxy and model server; "
                                       "kvpark start starts them again")
    stop.add_argument("--archive-dir", help="the connection's archive; default: the one selected with setup")
    stop.add_argument("--hermes", action="store_true", help="stop the connection configured in the active Hermes profile")
    migrate = sub.add_parser("migrate", help="migrate sloth-memory without moving saved caches")
    migrate.add_argument("--hermes", action="store_true", help="migrate the active Hermes profile")
    migrate.add_argument("--archive-dir", help="standalone legacy archive directory")
    migrate.add_argument("--confirm", action="store_true", help="drain the old proxy and apply migration")
    uninstall = sub.add_parser("uninstall", help="preview or disconnect services; keep packages and data until explicitly removed")
    uninstall.add_argument("--confirm", action="store_true", help="apply the uninstall after previewing")
    uninstall.add_argument("--hermes", action="store_true", help="restore routing and disable the plugin in the active Hermes profile")
    uninstall.add_argument("--archive-dir", help="standalone archive; --hermes reads the profile's archive setting")
    uninstall.add_argument("--opencode", action="store_true", help="remove kvpark's provider and plugin from OpenCode's config")
    uninstall.add_argument("--project", help="with --opencode: the project whose .opencode directory to clean")
    update = sub.add_parser("update", help="check or install an official release, retaining rollback files")
    update.add_argument("action", nargs="?", choices=("check",))
    update.add_argument("--archive-dir")
    serve = sub.add_parser("serve", help="start the localhost inference proxy")
    serve.add_argument("--archive-dir")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--upstream-port", type=int, default=8090)
    serve.add_argument("--backend", choices=BACKENDS)
    serve.add_argument("--upstream-url", help="existing backend address, optionally ending in /v1")
    serve.add_argument("--cache-mode", choices=("native", "routing"))
    serve.add_argument("--ttl-days", type=float)
    serve.add_argument("--max-gib", type=float)
    serve.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=None)
    backend = sub.add_parser("backend", help="launch a compatible llama-server and track its identity")
    backend.add_argument("--server", required=True)
    backend.add_argument("--model", required=True)
    backend.add_argument("--archive-dir")
    backend.add_argument("--port", type=int, default=8090)
    backend.add_argument("server_args", nargs=argparse.REMAINDER, help="extra llama-server arguments after --")
    from .bridge.cli import add_parser as add_bridge_parser
    add_bridge_parser(sub)
    for name in ("status", "save", "forget", "doctor", "settings", "cleanup"):
        cmd = sub.add_parser(name, aliases=["park"] if name == "save" else [], help={"status": "show archives and measured reuse", "save": "park the resident conversation", "forget": "delete a saved archive", "doctor": "check runtime, backend, and topology", "settings": "view or update saved cleanup preferences", "cleanup": "delete expired archives now"}[name])
        cmd.add_argument("--url", default=os.environ.get("KVPARK_URL"))
        cmd.add_argument("--archive-dir", help="discover the running proxy in this archive")
        if name in ("save", "forget"):
            group = cmd.add_mutually_exclusive_group(required=True)
            group.add_argument("--session", help="the exact slot_archive_key supplied during inference")
            group.add_argument("--key", help="the hashed k-… key returned by status")
        if name == "settings":
            cmd.add_argument("--ttl-days", type=float)
            cmd.add_argument("--max-gib", type=float)
            cmd.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=None)
        if name == "doctor":
            cmd.add_argument("--hermes", action="store_true",
                             help="check how Hermes loads the plugin in this Python environment; starts nothing")
            cmd.add_argument("--opencode", action="store_true", help="check OpenCode's kvpark provider, plugin and proxy")
            cmd.add_argument("--project", help="with --opencode: check this project's .opencode directory")
    args = parser.parse_args()
    if args.command == "park":
        args.command = "save"
    try:
        configured = None
        explicit_backend = args.command == "serve" and (args.backend or args.upstream_url)
        hermes_archive = args.command in ("uninstall", "stop") and args.hermes  # the profile's archive
        opencode = getattr(args, "opencode", False)  # OpenCode's config, not the saved connection
        if (args.command not in {"setup", "migrate", "backend", "bridge"} and not args.archive_dir and not explicit_backend
                and not hermes_archive and not opencode):
            from .setup import saved_settings
            configured = saved_settings()
            if configured:
                args.archive_dir = configured["archive_dir"]
        if args.command == "setup" and args.opencode:
            from .opencode import setup as setup_opencode
            print("\n".join(setup_opencode(project=args.project, url=args.url, server=args.server, model=args.model,
                                           backend_args=args.backend_args, context=args.context)))
        elif args.command == "setup":
            given = [f"--{name.replace('_', '-')}" for name in ("project", "server", "model", "backend_args", "url", "context")
                     if getattr(args, name) is not None]
            if given:
                parser.error(f"{', '.join(given)} {'applies' if len(given) == 1 else 'apply'} only with --opencode")
            from importlib.util import find_spec
            from .setup import terminal
            hermes = args.hermes or (not args.standalone and find_spec("hermes_cli") is not None)
            terminal(hermes=hermes)
        elif args.command == "start":
            from .hermes_service import Service, supervise
            if args.hermes:
                if args.archive_dir:
                    parser.error("--hermes uses its profile's archive; do not pass --archive-dir")
                try:
                    from .setup import hermes_settings
                    config = hermes_settings()
                except ImportError as exc:
                    if (exc.name or "hermes_cli").split(".")[0] == "hermes_cli":
                        raise SetupError("run kvpark start --hermes with the Python environment that runs Hermes") from None
                    # Hermes is installed here but cannot import its own code: an update added a
                    # module its editable install does not map yet.
                    raise SetupError(f"Hermes cannot load in this Python environment ({exc}). Reinstall Hermes "
                                     f"into {sys.prefix}, then start kvpark again.") from None
                if config["uninstalled"]:
                    raise SetupError("kvpark is disconnected in this Hermes profile. Run /kvpark setup in Hermes first.")
                if not (config["model"] or config["external"]):
                    raise SetupError("No model is configured in this Hermes profile. Run /kvpark setup in Hermes first.")
                check = "kvpark doctor --hermes"
            else:
                from .setup import saved_settings
                config = saved_settings()
                if not config:
                    raise SetupError("Choose a model first with kvpark setup.")
                if args.archive_dir and directory(args.archive_dir) != directory(config["archive_dir"]):
                    raise SetupError("kvpark start uses the connection selected in setup; run kvpark setup to choose another.")
                check = "kvpark doctor"
            service = Service(lambda: config)
            if args.interval <= 0:
                parser.error("--interval must be positive")
            if args.supervise:
                # A boot-time service keeps retrying; the model server may not be ready yet.
                try:
                    service.ensure()
                    print(f"kvpark is ready at {config['base_url']}/v1; supervising every {args.interval:g}s.", flush=True)
                except (OSError, ValueError, RuntimeError) as exc:
                    print(f"kvpark: {exc}; retrying in {args.interval:g}s", flush=True)
                supervise(service, args.interval)
            else:
                service.ensure()
                print(f"kvpark is ready. Run {check} to check the connection.")
        elif args.command == "stop":
            if args.hermes and args.archive_dir:
                parser.error("--hermes uses its profile's archive; do not pass --archive-dir")
            print("\n".join(stop_services(hermes=args.hermes, archive=args.archive_dir, configured=configured)))
        elif args.command == "migrate":
            from .migration import run
            print(run(hermes=args.hermes, archive=args.archive_dir, confirm=args.confirm))
        elif args.command == "uninstall" and args.opencode:
            if args.hermes or args.archive_dir:
                parser.error("--opencode changes only OpenCode's config; run --hermes or --archive-dir separately")
            from .opencode import uninstall as uninstall_opencode
            print("\n".join(uninstall_opencode(project=args.project, confirm=args.confirm)))
        elif args.command == "uninstall":
            if args.project:
                parser.error("--project applies with --opencode")
            if args.hermes:
                if args.archive_dir:
                    parser.error("--hermes uses its profile's archive; do not pass --archive-dir")
                try:
                    from .hermes_uninstall import run
                    print(run(confirm=args.confirm))
                except ImportError as exc:
                    raise RuntimeError("activate Hermes' Python environment to use --hermes") from exc
            else:
                from . import removal
                archive = directory(args.archive_dir)
                result = removal.stop(archive) if args.confirm else removal.plan(archive)
                if args.confirm:
                    from .setup import clear_settings
                    clear_settings(archive)
                print(json.dumps({"applied": args.confirm, "archive_dir": str(archive),
                    "proxy_pid": result["proxy"]["pid"] if result["proxy"] else None,
                    "backend_pid": result["backend"]["pid"] if result["backend"] else None,
                    "note": "Packages, snapshots, weights and builds are kept. Restore agent routes and stop other clients before --confirm."}, indent=2))
        elif args.command == "update":
            from . import updates
            archive = directory(args.archive_dir)
            result = updates.check(archive) if args.action == "check" else updates.apply(archive)
            print(json.dumps(result, indent=2))
        elif args.command == "bridge":
            from .bridge.cli import run as run_bridge
            run_bridge(args)
        elif args.command == "doctor" and args.opencode:
            if args.hermes or args.url or args.archive_dir:
                parser.error("--opencode reads the proxy's address from OpenCode's config; "
                             "do not combine it with --hermes, --url or --archive-dir")
            from .opencode import doctor as doctor_opencode
            lines, ok = doctor_opencode(project=args.project)
            print("\n".join(lines))
            if not ok:
                raise SystemExit(1)
        elif args.command == "doctor" and args.project:
            parser.error("--project applies with --opencode")
        elif args.command == "doctor" and args.hermes:
            from .hermes_doctor import report
            lines, ok = report()
            print("\n".join(lines))
            if not ok:
                raise SystemExit(1)
        elif args.command == "serve":
            if not 1 <= args.port <= 65535 or not 1 <= args.upstream_port <= 65535:
                parser.error("ports must be between 1 and 65535")
            from .retention import validate
            from .backends import Backend, DEFAULT_URLS
            if args.backend is None:
                args.backend = configured["backend"] if configured else "llama.cpp"
                if configured and not args.upstream_url:
                    args.upstream_url = configured["upstream_url"]
                    args.cache_mode = args.cache_mode or configured["cache_mode"]
            upstream = args.upstream_url
            if not upstream and args.backend == "llama.cpp":
                from .network import managed_port
                upstream = f"http://127.0.0.1:{managed_port(directory(args.archive_dir)) or args.upstream_port}"
            target = Backend(args.backend, upstream or DEFAULT_URLS[args.backend],
                             args.cache_mode or ("native" if args.backend == "llama.cpp" else "routing"))
            overrides = validate({k: v for k, v in dict(ttl_days=args.ttl_days, max_gib=args.max_gib,
                                                      cleanup_enabled=args.cleanup).items() if v is not None})
            os.environ.update(KVPARK_ARCHIVE_DIR=str(directory(args.archive_dir)), KVPARK_PORT=str(args.port),
                              KVPARK_UPSTREAM_PORT=str(args.upstream_port), KVPARK_POLICY_OVERRIDES=json.dumps(overrides),
                              KVPARK_UPSTREAM_URL=target.url, KVPARK_BACKEND=target.name, KVPARK_CACHE_MODE=target.cache_mode)
            from .proxy import main as serve_proxy
            serve_proxy()
        elif args.command == "backend":
            from .runtime import launch
            if not 1 <= args.port <= 65535:
                parser.error("port must be between 1 and 65535")
            extra = args.server_args[1:] if args.server_args[:1] == ["--"] else args.server_args
            launch(args.server, args.model, args.archive_dir, args.port, extra)
        else:
            payload = None
            if args.command in ("save", "forget"):
                key = args.key or "k-" + hashlib.sha256(args.session.encode()).hexdigest()
                payload = {"key": key}
            elif args.command == "cleanup":
                payload = {}
            elif args.command == "settings":
                payload = {k: v for k, v in dict(ttl_days=args.ttl_days, max_gib=args.max_gib,
                                               cleanup_enabled=args.cleanup).items() if v is not None} or None
            from .network import proxy_record
            record = proxy_record(directory(args.archive_dir)) if not args.url else None
            url = args.url or (record["base_url"] if record else "http://127.0.0.1:8080")
            result = request(url, args.command, payload)
            print(json.dumps(result, indent=2))
            if args.command == "doctor" and not result["ok"]:
                raise SystemExit(1)
    except SetupError as exc:
        print(f"kvpark: {exc}", file=sys.stderr)
        raise SystemExit(EX_CONFIG) from None
    except (OSError, ValueError, RuntimeError, URLError) as exc:
        print(f"kvpark: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
