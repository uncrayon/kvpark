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


def main():
    parser = argparse.ArgumentParser(description="Park your model's KV cache. Resume where you left off.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="find running models and connect using numbered choices")
    setup_mode = setup.add_mutually_exclusive_group()
    setup_mode.add_argument("--hermes", action="store_true", help="configure Hermes in this Python environment")
    setup_mode.add_argument("--standalone", action="store_true", help="configure standalone kvpark")
    start = sub.add_parser("start", help="restart the connection selected with setup")
    start.add_argument("--archive-dir")
    migrate = sub.add_parser("migrate", help="migrate sloth-memory without moving saved caches")
    migrate.add_argument("--hermes", action="store_true", help="migrate the active Hermes profile")
    migrate.add_argument("--archive-dir", help="standalone legacy archive directory")
    migrate.add_argument("--confirm", action="store_true", help="drain the old proxy and apply migration")
    uninstall = sub.add_parser("uninstall", help="preview or disconnect services; keep packages and data until explicitly removed")
    uninstall.add_argument("--confirm", action="store_true", help="apply the uninstall after previewing")
    uninstall.add_argument("--hermes", action="store_true", help="restore routing and disable the plugin in the active Hermes profile")
    uninstall.add_argument("--archive-dir", help="standalone archive; --hermes reads the profile's archive setting")
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
    args = parser.parse_args()
    if args.command == "park":
        args.command = "save"
    try:
        configured = None
        explicit_backend = args.command == "serve" and (args.backend or args.upstream_url)
        hermes_uninstall = args.command == "uninstall" and args.hermes
        if args.command not in {"setup", "migrate", "backend"} and not args.archive_dir and not explicit_backend and not hermes_uninstall:
            from .setup import saved_settings
            configured = saved_settings()
            if configured:
                args.archive_dir = configured["archive_dir"]
        if args.command == "setup":
            from importlib.util import find_spec
            from .setup import terminal
            hermes = args.hermes or (not args.standalone and find_spec("hermes_cli") is not None)
            terminal(hermes=hermes)
        elif args.command == "start":
            from .setup import saved_settings
            from .hermes_service import Service
            config = saved_settings()
            if not config:
                raise ValueError("Choose a model first with kvpark setup.")
            if args.archive_dir and directory(args.archive_dir) != directory(config["archive_dir"]):
                raise ValueError("kvpark start uses the connection selected in setup; run kvpark setup to choose another.")
            service = Service(lambda: config)
            service.ensure()
            print("kvpark is ready. Run kvpark doctor to check the connection.")
        elif args.command == "migrate":
            from .migration import run
            print(run(hermes=args.hermes, archive=args.archive_dir, confirm=args.confirm))
        elif args.command == "uninstall":
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
    except (OSError, ValueError, RuntimeError, URLError) as exc:
        print(f"kvpark: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
