"""Staged switches: "the next request of this conversation to model B continues from model A".

A stage is a small private file keyed by a hashed conversation key, holding only model names
and times (no text). It expires after a lease; the Hermes middleware consumes it. Staging
also prewarms the engine (prewarm): the daemon loads the pair's hub sides and runs their
products once while the user moves to the target, so the switch itself runs warm.
"""

import hashlib
import json
import os
from pathlib import Path
import threading
import time

LEASE_SECONDS = 15 * 60


def _path(bridge_dir, key):
    return Path(bridge_dir) / "stage" / (hashlib.sha256(key.encode()).hexdigest() + ".json")


def put(bridge_dir, key, *, source, target, lease=LEASE_SECONDS, now=None):
    now = time.time() if now is None else now
    path = _path(bridge_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pending = path.with_name(path.name + ".tmp")
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"source": source, "target": target, "created": now, "expires": now + lease}, stream)
    os.replace(pending, path)
    return path


def get(bridge_dir, key, *, now=None):
    """The live stage for a conversation, or None (expired stages are removed)."""
    path = _path(bridge_dir, key)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (time.time() if now is None else now) > data.get("expires", 0):
        path.unlink(missing_ok=True)
        return None
    return data


def drop(bridge_dir, key):
    _path(bridge_dir, key).unlink(missing_ok=True)


def prewarm(config, source, target, bridge_dir, *, log=lambda event: None):
    """Start the engine daemon and load the source -> target pair in a background thread (best
    effort, never raises: engine_client.prewarm); returns the thread, or None when the bridge
    has no engine. The thread is a daemon thread, so it never holds up the caller's exit."""
    if not config.get("engine") or not config.get("hub"):
        return None
    from . import engine_client

    def run():
        engine_client.prewarm(config, [source, target], bridge_dir, log=log)
    thread = threading.Thread(target=run, name="kvpark-bridge-prewarm", daemon=True)
    thread.start()
    return thread
