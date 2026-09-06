#!/usr/bin/env python3
"""Serialize single-slot inference and restore explicitly parked conversations."""
from __future__ import annotations

import collections
import contextlib
import copy
import hashlib
import http.client
import hmac
import json
import logging
import os
import re
from pathlib import Path
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .backends import Backend
from .platforms import fsync_directory, process_matches
from .runtime import directory
from . import __version__

LISTEN_HOST = os.environ.get("SLOTH_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("SLOTH_PORT", "8080"))
UPSTREAM_HOST = os.environ.get("SLOTH_UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.environ.get("SLOTH_UPSTREAM_PORT", "8090"))
ARCHIVE = directory()
RUNTIME = Path(os.environ.get("SLOTH_RUNTIME", str(ARCHIVE / "runtime.json")))
MIN_ARCHIVE_TOKENS = int(os.environ.get("SLOTH_MIN_TOKENS", "8192"))
RESAVE_GROWTH_TOKENS = int(os.environ.get("SLOTH_RESAVE_GROWTH", "4096"))
AUTO_SAVE = os.environ.get("SLOTH_AUTO_SAVE", "0") == "1"
TTL_DAYS = float(os.environ.get("SLOTH_TTL_DAYS", "7"))
MAX_ARCHIVE_GB = float(os.environ.get("SLOTH_MAX_GB", "32"))
CHAT_PATHS = {"/v1/chat/completions", "/chat/completions"}
GENERATION_PATHS = CHAT_PATHS
CONTROL_PREFIX = "/_sloth/"
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailer", "trailers", "transfer-encoding", "upgrade"}
log = logging.getLogger("slot-proxy")
API_KEY = os.environ.get("SLOTH_API_KEY")
MAX_REQUEST_BYTES = 64 * 1024**2


def backend():
    name = os.environ.get("SLOTH_BACKEND", "llama.cpp")
    return Backend(name, os.environ.get("SLOTH_UPSTREAM_URL", f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}"),
                   os.environ.get("SLOTH_CACHE_MODE", "native" if name == "llama.cpp" else "routing"))


class UnsafeRestoreError(RuntimeError):
    """A partial restore could not be erased; inference must not continue."""


def key_for(value: str) -> str:
    return "k-" + hashlib.sha256(value.encode()).hexdigest()


def conversation_key(req: dict) -> str | None:
    # A session id can also name an ephemeral background-review fork.
    if req.get("slot_archive_role") != "foreground":
        return None
    # A user account is not a conversation; never use the OpenAI `user` field.
    for field in ("slot_archive_key", "prompt_cache_key", "session_id"):
        value = req.get(field)
        if isinstance(value, str) and value:
            return key_for(value)
    return None  # Content-based guesses can merge unrelated threads.


def prompt_signature(req):
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    messages = req.get("messages") or []
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        return None
    return {
        "role": req.get("slot_archive_role", "unverified"),
        "system": digest([m for m in messages if m.get("role") in ("system", "developer")]),
        "tools": digest(req.get("tools")),
        "render_settings": digest({k: req.get(k) for k in
            ("model", "reasoning_effort", "chat_template_kwargs", "chat_template")}),
        "reasoning_effort": req.get("reasoning_effort"),
        "messages": [digest(m) for m in messages],
        "message_count": len(messages),
    }


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(data, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def fsync_dir(path: Path) -> None:
    fsync_directory(path)


class Archive:
    def __init__(self):
        # Hold through the ENTIRE upstream response, not just switch(). Mutating controls
        # and GC share this lock; another request cannot change the slot owner.
        self.lock = threading.RLock()
        self.current_key = None
        self.current_thread = None
        self.run_id = None
        self.identity = None
        self.events = collections.deque(maxlen=40)
        self.current_prompt = None
        self.activity = {"phase": "idle", "key": None}
        self.runtime_error = None
        self.maintenance_until = time.monotonic() + 300 if os.environ.get("SLOTH_UPDATE_START") == "1" else 0
        self.displaced = {}
        self._snapshot = {}
        self.entries = {}
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        from .retention import Retention
        self.retention = Retention(ARCHIVE / "retention.json", lambda: dict(
            ttl_days=TTL_DAYS, max_gib=MAX_ARCHIVE_GB, cleanup_enabled=True))
        overrides = os.environ.get("SLOTH_POLICY_OVERRIDES")
        if overrides:
            self.retention.update(json.loads(overrides))
        self.gc_wakeup = threading.Event()
        for path in ARCHIVE.glob("k-*.meta"):
            try:
                meta = json.loads(path.read_text())
                if meta.get("format") == 2 and Path(meta["filename"]).name == meta["filename"]:
                    self.entries[path.stem] = meta
            except (OSError, ValueError, KeyError):
                log.warning("unreadable archive metadata: %s", path.name)

        self.publish_status()

    def publish_status(self):
        # The inference owner publishes a detached snapshot. Readers never take
        # the inference lock and never mutate ownership by checking runtime().
        self._snapshot = copy.deepcopy({
            "resident_key": self.current_key, "resident_thread": self.current_thread,
            "activity": self.activity, "runtime_error": self.runtime_error,
            "entries": self.entries, "recent": list(self.events)[:12],
            "retention": self.retention.settings(),
        })

    def set_activity(self, phase, key=None):
        self.activity = {"phase": phase, "key": key, "since": time.time()}
        self.publish_status()

    def record(self, action, key, **fields):
        self.events.appendleft(dict(at=time.time(), action=action, key=key, **fields))
        self.publish_status()

    def runtime(self):
        """Launcher manifest describes the process that actually loaded weights."""
        data = json.loads(RUNTIME.read_text())
        if not process_matches(data):
            raise RuntimeError("model runtime manifest is stale")
        for name, expected in data["files"].items():
            st = os.stat(name)
            if [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns] != expected:
                raise RuntimeError("model/build files changed; restart the model before archiving")
        if data["run_id"] != self.run_id:
            self.current_key = self.current_thread = self.current_prompt = None
            self.displaced.clear()
            self.run_id = data["run_id"]
        self.identity = data["identity"]
        self.runtime_error = None
        self.publish_status()
        return data

    def request(self, method, path, body=None, timeout=600):
        conn = backend().connection(timeout)
        try:
            payload = json.dumps(body).encode() if body is not None else None
            headers = {"Content-Type": "application/json", "Connection": "close"}
            key_file = os.environ.get("SLOTH_UPSTREAM_KEY_FILE")
            if key_file:
                headers["Authorization"] = "Bearer " + Path(key_file).read_text().strip()
            conn.request(method, path, payload, headers)
            resp = conn.getresponse()
            data = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"upstream {path}: HTTP {resp.status}: {data[:160]!r}")
            return json.loads(data)
        finally:
            conn.close()

    def slot(self):
        slots = self.request("GET", "/slots", timeout=5)
        if len(slots) != 1 or slots[0]["id"] != 0:
            raise RuntimeError("archive requires exactly one upstream slot (id 0)")
        if slots[0]["is_processing"]:
            raise RuntimeError("upstream slot is busy; bypass traffic is not supported")
        return slots[0]

    def remove(self, key):
        meta = self.entries.pop(key, None)
        (ARCHIVE / f"{key}.meta").unlink(missing_ok=True)
        if meta:
            for name in (meta["filename"], meta["filename"] + ".resume"):
                (ARCHIVE / name).unlink(missing_ok=True)

    def save(self, key, force=False):
        if not self.current_prompt or self.current_prompt.get("role") != "foreground":
            raise ValueError("no verified foreground prompt is resident; configure foreground session metadata and send a message first")
        tokens = int(self.slot().get("n_prompt_tokens", 0))
        if tokens <= 0:
            raise RuntimeError("no populated KV state is available to save")
        previous = self.entries.get(key)
        if not force:
            if not previous and (not AUTO_SAVE or tokens < MIN_ARCHIVE_TOKENS):
                return
            if previous and tokens - previous["tokens"] < RESAVE_GROWTH_TOKENS:
                return
        self.set_activity("saving", key)
        t0 = time.monotonic()
        filename = f"{key}.{uuid.uuid4().hex}.bin"
        files = [ARCHIVE / filename, ARCHIVE / (filename + ".resume")]
        try:
            result = self.request("POST", "/slots/0?action=save", {"filename": filename})
            # Stock /slots drops hybrid checkpoints. Do not call that a usable
            # archive: the accompanying resume file is mandatory.
            for path in files:
                with path.open("rb+") as fh:
                    os.fsync(fh.fileno())
            if sum(path.stat().st_size for path in files) > self.retention.settings()["max_gib"] * 1024**3:
                raise ValueError("snapshot exceeds max-gib; increase the limit before parking")
            meta = dict(format=2, filename=filename, identity=self.identity,
                        tokens=result["n_saved"], bytes=sum(p.stat().st_size for p in files),
                        thread=self.current_thread, prompt=self.current_prompt,
                        saved_at=time.time(), last_used=time.time())
            atomic_json(ARCHIVE / f"{key}.meta", meta)
            self.entries[key] = meta
        except Exception:
            # Rename may have succeeded before directory fsync failed. Keep
            # anything the published manifest references; deleting it would
            # turn a recoverable durability error into guaranteed data loss.
            published = False
            try:
                published = json.loads((ARCHIVE / f"{key}.meta").read_text()).get("filename") == filename
            except (OSError, ValueError):
                pass
            if published:
                self.entries[key] = meta
            else:
                for path in files:
                    path.unlink(missing_ok=True)
            raise
        if previous:
            for suffix in ("", ".resume"):
                (ARCHIVE / (previous["filename"] + suffix)).unlink(missing_ok=True)
        self.record("saved", key, tokens=meta["tokens"], gib=round(meta["bytes"] / 1024**3, 3),
                    seconds=round(time.monotonic() - t0, 3))
        log.info("saved %s: %d tokens", key, meta["tokens"])
        self.enforce_capacity()

    def restore(self, key, incoming=None):
        meta = self.entries.get(key)
        if not meta:
            return
        if meta["identity"] != self.identity:
            self.record("incompatible", key)
            return  # Keep old artifacts, but NEVER pass wrong weights to restore.
        if self.retention.expired(meta, time.time()):
            self.remove(key)
            self.record("expired", key)
            return
        saved = meta.get("prompt") or {}
        if saved.get("role") != "foreground":
            self.record("restore-skipped", key, reason="archive has no verified foreground identity; park again")
            return
        changed = [field for field in ("system", "tools", "render_settings")
                   if incoming and saved.get(field) != incoming.get(field)]
        if changed:
            self.record("restore-skipped", key, reason="prompt prefix changed: " + ", ".join(changed),
                        saved_effort=saved.get("reasoning_effort"), incoming_effort=incoming.get("reasoning_effort"))
            return
        self.set_activity("restoring", key)
        t0 = time.monotonic()
        try:
            result = self.request("POST", "/slots/0?action=restore", {"filename": meta["filename"]})
        except Exception:
            # A rejected/partial restore can have touched both contexts.
            try:
                self.request("POST", "/slots/0?action=erase", {})
            except Exception as exc:
                raise UnsafeRestoreError("restore failed and backend state could not be erased; restart the backend") from exc
            raise
        meta["last_used"] = time.time()
        atomic_json(ARCHIVE / f"{key}.meta", meta)
        self.record("restored", key, tokens=result["n_restored"],
                    gib=round(meta["bytes"] / 1024**3, 3), seconds=round(time.monotonic() - t0, 3))
        log.info("restored %s; cache reuse must be checked on the next completion", key)

    def switch(self, key, thread=None, incoming=None, role=None):
        self.runtime()
        self.displaced.pop(key, None)
        state = self.slot()
        if self.current_key and not state.get("n_prompt_tokens", 0):
            self.current_key = self.current_thread = None
        if key == self.current_key:
            return
        outgoing = self.current_key
        self.current_key = None
        if outgoing:
            try:
                protect = role == "background" and outgoing in self.entries
                self.save(outgoing, force=protect)
                if protect:
                    self.displaced[outgoing] = self.entries[outgoing]["filename"]
            except Exception as exc:
                self.record("save-failed", outgoing, reason=str(exc))
                log.exception("save failed")
        if key:
            self.restore(key, incoming)
        self.current_thread = thread
        # The relay claims current_key only after successful completion.

    def park(self, key=None, thread=None, simulate=False):
        with self.lock:
            self.runtime()
            if thread:
                key = self.current_key if thread == self.current_thread else next(
                    (k for k, v in self.entries.items() if v.get("thread") == thread), None)
            if not key:
                raise ValueError("specify this thread's key; refusing to park an arbitrary resident conversation")
            if key != self.current_key:
                meta = self.entries.get(key) or {}
                if not simulate and self.displaced.get(key) == meta.get("filename") and self.displaced.get(key):
                    return dict(parked=True, ok=True, key=key, tokens=meta["tokens"],
                                note="Foreground state was saved before background work displaced it.")
                raise ValueError("the foreground conversation is not resident; no background state was parked. Send a foreground message, then park again")
            try:
                self.save(key, force=True)  # Explicit intent bypasses the token floor.
                if simulate:
                    self.request("POST", "/slots/0?action=erase", {})
                    self.current_key = self.current_thread = self.current_prompt = None
                    self.record("gap-simulated", key)
            finally:
                self.set_activity("idle")
            return dict(parked=True, ok=True, key=key, tokens=self.entries[key]["tokens"],
                        note="Saved to disk. Resume by sending a message in this same thread.")

    def status(self):
        snapshot = copy.deepcopy(self._snapshot)
        now = time.time()
        entries = [dict(key=k, tokens=v["tokens"], gib=round(v["bytes"] / 1024**3, 3),
                        age_days=round((now - v.get("saved_at", v["last_used"])) / 86400, 3),
                        resident=k == snapshot["resident_key"], thread=v.get("thread"))
                   for k, v in snapshot["entries"].items()]
        return dict(resident_key=snapshot["resident_key"], resident_thread=snapshot["resident_thread"],
                    activity=copy.deepcopy(snapshot["activity"]), runtime_error=snapshot["runtime_error"],
                    upstream=backend().url.split("://", 1)[1], archive_dir=str(ARCHIVE),
                    archived=len(entries), entries=entries,
                    total_gib=round(sum(v["bytes"] for v in snapshot["entries"].values()) / 1024**3, 3),
                    service="sloth-memory", version=__version__, control_version=4, upstream_url=backend().url,
                    maintenance=self.maintenance_until > time.monotonic(),
                    capabilities=backend().capabilities(),
                    ttl_days=snapshot["retention"]["ttl_days"], max_gib=snapshot["retention"]["max_gib"],
                    cleanup_enabled=snapshot["retention"]["cleanup_enabled"], cleanup_interval_seconds=3600,
                    recent=[dict(e, ago_s=round(now-e["at"], 1)) for e in snapshot["recent"]])

    def prepare_update(self, timeout=60):
        # Set the gate before waiting: queued and new requests must not start
        # another generation between saving the resident state and replacement.
        if self.maintenance_until > time.monotonic():
            raise RuntimeError("an update is already preparing this proxy")
        self.maintenance_until = time.monotonic() + 300
        acquired = self.lock.acquire(timeout=timeout)
        try:
            if not acquired:
                raise RuntimeError("inference is still active; retry the update after the reply")
            if backend().snapshots and self.current_key:
                self.park(key=self.current_key)
            return dict(ok=True, maintenance=True)
        except BaseException:
            self.maintenance_until = 0
            raise
        finally:
            if acquired:
                self.lock.release()

    def doctor(self):
        # Do not queue a diagnostic behind a long generation or mutate ownership.
        if not self.lock.acquire(blocking=False):
            return dict(ok=False, reason="inference is active; retry doctor when idle")
        try:
            if not backend().snapshots:
                self.request("GET", backend().health_path, timeout=5)
                return dict(ok=True, **backend().capabilities())
            self.runtime()
            self.slot()
            return dict(ok=True, backend="reachable", slots=1,
                        note="Runtime identity valid. A real park/resume test is still required to verify checkpoint support.")
        except (OSError, ValueError, KeyError, RuntimeError, http.client.HTTPException) as exc:
            return dict(ok=False, reason=str(exc))
        finally:
            self.lock.release()

    def forget(self, key):
        with self.lock:
            if not isinstance(key, str) or key not in self.entries:
                raise ValueError("no saved archive for that key")
            self.remove(key)
            self.displaced.pop(key, None)
            self.record("forgotten", key)
        return dict(ok=True, key=key, note="Saved archive deleted; transcript and resident inference state are unchanged.")

    def update_retention(self, values):
        with self.lock:
            settings = self.retention.update(values)
            self.enforce_capacity()
            self.record("retention-updated", None, **settings)
            self.gc_wakeup.set()
            return settings

    def enforce_capacity(self):
        total = sum(m["bytes"] for m in self.entries.values())
        for key, meta in sorted(self.entries.items(), key=lambda item: item[1]["last_used"]):
            if total <= self.retention.settings()["max_gib"] * 1024**3:
                break
            self.remove(key)
            total -= meta["bytes"]
            self.record("evicted", key)

    def gc(self, *, manual=False):
        with self.lock:
            if not manual and not self.retention.settings()["cleanup_enabled"]:
                return dict(deleted=0, orphan_files=0, cleanup_enabled=False)
            before = len(self.entries)
            now = time.time()
            for key, meta in list(self.entries.items()):
                if self.retention.expired(meta, now, manual=manual):
                    self.remove(key)
                    self.record("expired", key)
            self.enforce_capacity()
            # A crash can leave an unpublished generation. Match only our exact
            # filenames, never arbitrary files or another application's backups.
            referenced = {m["filename"] + suffix for m in self.entries.values() for suffix in ("", ".resume")}
            ttl = self.retention.settings()["ttl_days"]
            orphans = 0
            for path in ARCHIVE.iterdir() if ttl > 0 else ():
                if (re.fullmatch(r"k-[0-9a-f]{64}\.[0-9a-f]{32}\.bin(?:\.resume)?", path.name)
                        and path.name not in referenced and not path.is_symlink() and path.is_file()
                        and now - path.stat().st_mtime >= ttl * 86400):
                    path.unlink()
                    orphans += 1
            if orphans:
                self.record("orphan-cleanup", None, files=orphans)
            self.publish_status()
            return dict(deleted=before - len(self.entries), orphan_files=orphans,
                        cleanup_enabled=self.retention.settings()["cleanup_enabled"])


class ResponseMetrics:
    """Observe response bytes without changing them or retaining chat history."""
    def __init__(self, streaming, started_at):
        self.streaming = streaming
        self.started_at = started_at
        self.pending = b""
        self.metrics = {}
        self.too_large = False
        self.failed = False

    def observe(self, value):
        if not isinstance(value, dict):
            return
        if value.get("error"):
            self.failed = True
        timings = value.get("timings") or {}
        if "cache_n" in timings:
            self.metrics["cached_tokens"] = timings["cache_n"]
        if "prompt_n" in timings:
            self.metrics["processed_tokens"] = timings["prompt_n"]
        usage = value.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is not None:
            self.metrics["cached_tokens"] = cached
            if "prompt_tokens" in usage:
                self.metrics.setdefault("processed_tokens", usage["prompt_tokens"] - cached)
        for choice in value.get("choices", []):
            delta = choice.get("delta") or {}
            if (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")):
                self.metrics.setdefault("ttft_seconds", round(time.monotonic() - self.started_at, 3))

    def parse(self, data):
        try:
            self.observe(json.loads(data))
        except (ValueError, TypeError, AttributeError):
            pass

    def feed(self, chunk):
        if self.too_large:
            return
        self.pending += chunk
        if len(self.pending) > 16 * 1024**2:
            self.pending = b""
            self.too_large = True
            return
        if self.streaming:
            while b"\n" in self.pending:
                line, self.pending = self.pending.split(b"\n", 1)
                if line.startswith(b"data:"):
                    self.parse(line[5:].strip())

    def finish(self):
        if not self.streaming and not self.too_large:
            self.parse(self.pending)
        self.pending = b""
        return self.metrics


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    def control(self, method, path, body):
        try:
            if (method != "GET" and ARCHIVE_STATE.maintenance_until > time.monotonic()
                    and path not in {CONTROL_PREFIX + "cancel-update", CONTROL_PREFIX + "prepare-update"}):
                raise RuntimeError("sloth-memory is updating; retry shortly")
            if method == "GET" and path == CONTROL_PREFIX + "status":
                payload = ARCHIVE_STATE.status()
            elif method == "GET" and path == CONTROL_PREFIX + "doctor":
                payload = ARCHIVE_STATE.doctor()
            elif method == "POST" and path == CONTROL_PREFIX + "prepare-update":
                payload = ARCHIVE_STATE.prepare_update()
            elif method == "POST" and path == CONTROL_PREFIX + "cancel-update":
                ARCHIVE_STATE.maintenance_until = 0
                payload = dict(ok=True)
            elif method == "GET" and path == CONTROL_PREFIX + "settings":
                payload = ARCHIVE_STATE.status()
                payload = {k: payload[k] for k in ("ttl_days", "max_gib", "cleanup_enabled")}
            elif method == "POST" and path == CONTROL_PREFIX + "settings":
                payload = ARCHIVE_STATE.update_retention(json.loads(body or b"{}"))
            elif method == "POST" and path == CONTROL_PREFIX + "cleanup":
                payload = ARCHIVE_STATE.gc(manual=True)
            elif method == "POST" and path == CONTROL_PREFIX + "forget":
                req = json.loads(body or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("control body must be a JSON object")
                payload = ARCHIVE_STATE.forget(req.get("key"))
            elif method == "POST" and path in {CONTROL_PREFIX + "park", CONTROL_PREFIX + "simulate-gap"}:
                if not backend().snapshots:
                    raise ValueError(backend().capabilities()["note"])
                req = json.loads(body or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("control body must be a JSON object")
                payload = ARCHIVE_STATE.park(key=req.get("key"), thread=req.get("thread"),
                                             simulate=path.endswith("simulate-gap"))
            else:
                self.send_error(404)
                return
            status = 200
        except (ValueError, RuntimeError, OSError) as exc:
            payload, status = {"error": str(exc), "reason": str(exc)}, 409
        blob = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def relay(self, method):
        if API_KEY and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + API_KEY):
            self.send_error(401, "bearer token required")
            self.close_connection = True
            return
        # Browser scripts must not use a user's localhost service as a control API.
        if self.headers.get("Origin"):
            self.send_error(403, "browser origins are unsupported; use a server-side agent adapter")
            self.close_connection = True
            return
        if self.headers.get("Transfer-Encoding"):
            self.send_error(400, "chunked request bodies are not supported")
            self.close_connection = True
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError()
        except ValueError:
            self.send_error(400, "invalid content length")
            return
        if length > MAX_REQUEST_BYTES:
            self.send_error(413, "request exceeds 64 MiB")
            self.close_connection = True
            return
        body = self.rfile.read(length)
        path = self.path.split("?")[0]
        if path.startswith(CONTROL_PREFIX):
            self.control(method, path, body)
            return
        # Direct slot manipulation would invalidate ownership. Admin tools must
        # use the upstream port, with inference stopped, or use this control plane.
        if method == "POST" and path.startswith("/slots/"):
            self.send_error(409, "use the archive control plane")
            return
        if not (method == "POST" and path in CHAT_PATHS or
                method == "GET" and path in {"/health", "/v1/models", "/models"}):
            self.send_error(404, "alpha supports chat completions, health, and models")
            return
        generation = method == "POST" and path in GENERATION_PATHS
        req = {}
        if generation:
            try:
                req = json.loads(body)
                if not isinstance(req, dict):
                    raise ValueError()
            except ValueError:
                self.send_error(400, "invalid JSON object")
                return
        key = conversation_key(req) if path in CHAT_PATHS else None
        signature = prompt_signature(req) if path in CHAT_PATHS and generation else None
        role = req.pop("slot_archive_role", "unverified")
        thread = req.pop("slot_archive_thread", None) if key else None
        req.pop("slot_archive_thread", None)
        req.pop("slot_archive_key", None)
        if generation:
            body = json.dumps(req).encode()
        lock = ARCHIVE_STATE.lock if generation else contextlib.nullcontext()
        started_at = time.monotonic()
        with lock:
            if generation and ARCHIVE_STATE.maintenance_until > time.monotonic():
                self.send_error(503, "sloth-memory is updating; retry shortly")
                return
            archive_ready = backend().snapshots
            if generation and archive_ready:
                try:
                    ARCHIVE_STATE.set_activity("switching", key)
                    ARCHIVE_STATE.switch(key, thread, signature, role)
                except Exception as exc:
                    ARCHIVE_STATE.current_key = ARCHIVE_STATE.current_thread = ARCHIVE_STATE.current_prompt = None
                    archive_ready = False
                    ARCHIVE_STATE.runtime_error = str(exc)
                    ARCHIVE_STATE.record("archive-unavailable", key, reason=str(exc))
                    if isinstance(exc, UnsafeRestoreError):
                        ARCHIVE_STATE.set_activity("idle")
                        self.send_error(502, str(exc))
                        return
                    log.exception("archive unavailable; inference will continue")
            if generation:
                ARCHIVE_STATE.set_activity("generating", key)
            success, metrics = self.forward(method, body, started_at)
            if generation:
                ARCHIVE_STATE.current_key = key if success and archive_ready else None
                ARCHIVE_STATE.current_thread = thread if success and archive_ready else None
                ARCHIVE_STATE.current_prompt = signature if success and archive_ready and key else None
                if success:
                    ARCHIVE_STATE.record("completed", key, **metrics)
                ARCHIVE_STATE.set_activity("idle")

    def forward(self, method, body, started_at):
        target = backend()
        conn = target.connection(1800)
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_BY_HOP | {"host"} and not (API_KEY and k.lower() == "authorization")}
        headers["Content-Length"] = str(len(body))
        headers["Connection"] = "close"
        # Proxy authentication and backend authentication are separate credentials.
        key_file = os.environ.get("SLOTH_UPSTREAM_KEY_FILE")
        if key_file:
            headers["Authorization"] = "Bearer " + Path(key_file).read_text().strip()
        started = False
        try:
            conn.request(method, target.path(self.path), body, headers)
            upstream = conn.getresponse()
            observer = ResponseMetrics("text/event-stream" in upstream.getheader("Content-Type", ""), started_at)
            self.send_response(upstream.status)
            started = True
            for k, v in upstream.getheaders():
                if k.lower() not in HOP_BY_HOP | {"content-length"}:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            client_connected = True
            while True:
                chunk = upstream.read1(8192)  # read() waits to fill 8 KiB, delaying TTFT.
                if not chunk:
                    break
                observer.feed(chunk)
                if client_connected:
                    try:
                        self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        # Drain upstream with the lock held. Its completion still
                        # owns the slot even when the downstream client is gone.
                        client_connected = False
            if client_connected:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            metrics = observer.finish()
            return upstream.status == 200 and not observer.failed, metrics
        except Exception:
            self.close_connection = True
            log.exception("upstream request failed")
            if not started:
                self.send_error(502, "upstream unavailable")
            return False, {}
        finally:
            conn.close()

    def do_GET(self): self.relay("GET")
    def do_POST(self): self.relay("POST")
    def do_DELETE(self): self.relay("DELETE")


def gc_loop():
    while True:
        try:
            ARCHIVE_STATE.gc()
        except Exception:
            log.exception("GC failed")
        ARCHIVE_STATE.gc_wakeup.wait(3600)
        ARCHIVE_STATE.gc_wakeup.clear()


def main():
    logging.basicConfig(level=os.environ.get("SLOTH_LOG", "INFO"),
                        format="%(asctime)s %(levelname)s %(message)s")
    if LISTEN_HOST != "127.0.0.1":
        raise SystemExit("the proxy listens on localhost only")
    from .runtime import lock_directory
    os.umask(0o077)
    lock_directory(ARCHIVE, ".proxy.lock")
    global ARCHIVE_STATE
    ARCHIVE_STATE = Archive()
    if backend().snapshots:
        try:
            ARCHIVE_STATE.runtime()
        except Exception as exc:
            ARCHIVE_STATE.runtime_error = str(exc)
            ARCHIVE_STATE.publish_status()
    from .network import bind_server
    from .platforms import process_identity
    server = bind_server(Handler, LISTEN_PORT, backend().url)
    atomic_json(ARCHIVE / "proxy.json", dict(**process_identity(os.getpid()),
                base_url=f"http://{LISTEN_HOST}:{server.server_port}", upstream_url=backend().url,
                backend=backend().name, cache_mode=backend().cache_mode))
    log.info("Proxy listening at http://%s:%s → %s (%s)", LISTEN_HOST, server.server_port, backend().url, backend().cache_mode)
    threading.Thread(target=gc_loop, daemon=True).start()
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
