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


def request(url, action, payload=None):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("SLOTH_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["SLOTH_API_KEY"]
    data = None if payload is None else json.dumps(payload).encode()
    try:
        with urlopen(Request(url.rstrip("/") + "/_sloth/" + action, data=data, headers=headers), timeout=900) as response:
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
    serve.add_argument("--ttl-days", type=float, default=7)
    serve.add_argument("--max-gib", type=float, default=32)
    backend = sub.add_parser("backend", help="launch a compatible llama-server and track its identity")
    backend.add_argument("--server", required=True)
    backend.add_argument("--model", required=True)
    backend.add_argument("--archive-dir")
    backend.add_argument("--port", type=int, default=8090)
    backend.add_argument("server_args", nargs=argparse.REMAINDER, help="extra llama-server arguments after --")
    for name in ("status", "park", "forget", "doctor"):
        cmd = sub.add_parser(name, help={"status": "show archives and measured reuse", "park": "save the resident conversation", "forget": "delete a saved archive", "doctor": "check runtime, backend, and topology"}[name])
        cmd.add_argument("--url", default=os.environ.get("SLOTH_URL", "http://127.0.0.1:8080"))
        if name in ("park", "forget"):
            group = cmd.add_mutually_exclusive_group(required=True)
            group.add_argument("--session", help="the exact slot_archive_key supplied during inference")
            group.add_argument("--key", help="the hashed k-… key returned by status")
    args = parser.parse_args()
    try:
        if args.command == "serve":
            if not 1 <= args.port <= 65535 or not 1 <= args.upstream_port <= 65535:
                parser.error("ports must be between 1 and 65535")
            if args.ttl_days < 0 or args.max_gib <= 0:
                parser.error("ttl-days must be nonnegative and max-gib must be positive")
            os.environ.update(SLOTH_ARCHIVE_DIR=str(directory(args.archive_dir)), SLOTH_PORT=str(args.port),
                              SLOTH_UPSTREAM_PORT=str(args.upstream_port), SLOTH_TTL_DAYS=str(args.ttl_days),
                              SLOTH_MAX_GB=str(args.max_gib))
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
            result = request(args.url, args.command, payload)
            print(json.dumps(result, indent=2))
            if args.command == "doctor" and not result["ok"]:
                raise SystemExit(1)
    except (OSError, ValueError, RuntimeError, URLError) as exc:
        print(f"sloth-memory: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
