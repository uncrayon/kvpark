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


def request(url, action, payload=None, *, timeout=900):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("SLOTH_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["SLOTH_API_KEY"]
    data = None if payload is None else json.dumps(payload).encode()
    try:
        with urlopen(Request(url.rstrip("/") + "/_sloth/" + action, data=data, headers=headers), timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            reason = json.loads(exc.read()).get("reason", str(exc))
        except ValueError:
            reason = str(exc)
        raise RuntimeError(reason) from None


def main():
    parser = argparse.ArgumentParser(description="Let your local AI nap. Save its work for later.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="start the localhost inference proxy")
    serve.add_argument("--archive-dir")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--upstream-port", type=int, default=8090)
    serve.add_argument("--backend", choices=("llama.cpp", "ollama", "vllm", "mlx"), default="llama.cpp")
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
    for name in ("status", "park", "forget", "doctor", "settings", "cleanup"):
        cmd = sub.add_parser(name, help={"status": "show archives and measured reuse", "park": "save the resident conversation", "forget": "delete a saved archive", "doctor": "check runtime, backend, and topology", "settings": "view or update saved cleanup preferences", "cleanup": "delete expired archives now"}[name])
        cmd.add_argument("--url", default=os.environ.get("SLOTH_URL"))
        cmd.add_argument("--archive-dir", help="discover the running proxy in this archive")
        if name in ("park", "forget"):
            group = cmd.add_mutually_exclusive_group(required=True)
            group.add_argument("--session", help="the exact slot_archive_key supplied during inference")
            group.add_argument("--key", help="the hashed k-… key returned by status")
        if name == "settings":
            cmd.add_argument("--ttl-days", type=float)
            cmd.add_argument("--max-gib", type=float)
            cmd.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    try:
        if args.command == "serve":
            if not 1 <= args.port <= 65535 or not 1 <= args.upstream_port <= 65535:
                parser.error("ports must be between 1 and 65535")
            from .retention import validate
            from .backends import Backend, DEFAULT_URLS
            upstream = args.upstream_url
            if not upstream and args.backend == "llama.cpp":
                from .network import managed_port
                upstream = f"http://127.0.0.1:{managed_port(directory(args.archive_dir)) or args.upstream_port}"
            target = Backend(args.backend, upstream or DEFAULT_URLS[args.backend],
                             args.cache_mode or ("native" if args.backend == "llama.cpp" else "routing"))
            overrides = validate({k: v for k, v in dict(ttl_days=args.ttl_days, max_gib=args.max_gib,
                                                      cleanup_enabled=args.cleanup).items() if v is not None})
            os.environ.update(SLOTH_ARCHIVE_DIR=str(directory(args.archive_dir)), SLOTH_PORT=str(args.port),
                              SLOTH_UPSTREAM_PORT=str(args.upstream_port), SLOTH_POLICY_OVERRIDES=json.dumps(overrides),
                              SLOTH_UPSTREAM_URL=target.url, SLOTH_BACKEND=target.name, SLOTH_CACHE_MODE=target.cache_mode)
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
            if args.command in ("park", "forget"):
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
        print(f"sloth-memory: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
