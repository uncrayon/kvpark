#!/usr/bin/env python3
"""Minimal agent-independent client. Keeps the transcript in a local JSON file."""

import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="stable identity, e.g. cli:my-project:v1")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("message")
    args = parser.parse_args()
    messages = json.loads(args.history.read_text()) if args.history.exists() else [
        {"role": "system", "content": "You are a helpful assistant. Be concise."}]
    messages.append({"role": "user", "content": args.message})
    payload = dict(model="local", messages=messages, max_tokens=256,
                   slot_archive_role="foreground", slot_archive_key=args.session)
    headers = {"Content-Type": "application/json"}
    if os.environ.get("KVPARK_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["KVPARK_API_KEY"]
    with urlopen(Request(args.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(payload).encode(),
                         headers=headers), timeout=1800) as response:
        message = json.load(response)["choices"][0]["message"]
    messages.append(message)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    os.umask(0o077)
    tmp = args.history.with_suffix(args.history.suffix + ".tmp")
    tmp.write_text(json.dumps(messages, indent=2))
    tmp.replace(args.history)
    print(message.get("content") or message.get("reasoning_content") or "[empty response]")


if __name__ == "__main__":
    main()
