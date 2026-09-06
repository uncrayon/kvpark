"""Migrate the actual sloth-memory 0.2.0a2 wheel in an isolated routing stack.

Usage: python scripts/test_migration_release.py dist [--legacy-wheel PATH]
Without --legacy-wheel, download the fixed release and verify its pinned SHA-256.
The synthetic archive marker tests preservation, not native KV-cache correctness.
"""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid
import venv

import psutil

LEGACY_NAME = "sloth_memory-0.2.0a2-py3-none-any.whl"
LEGACY_URL = "https://github.com/uncrayon/kvpark/releases/download/v0.2.0a2/" + LEGACY_NAME
LEGACY_SHA256 = "36a8aca441565383821e8b1321885fd025aa73f4c41ea5aad3902bf099fe3541"
MAX_WHEEL_BYTES = 25 * 1024**2


class Backend(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = b'{"status":"ok","data":[]}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def get(url, token=None):
    headers = {"Authorization": "Bearer " + token} if token else {}
    with urlopen(Request(url, headers=headers), timeout=3) as response:
        return json.load(response)


def legacy_wheel(destination, fixture):
    if fixture:
        with Path(fixture).open("rb") as source:
            data = source.read(MAX_WHEEL_BYTES + 1)
    else:
        with urlopen(Request(LEGACY_URL, headers={"User-Agent": "kvpark-migration-acceptance"}), timeout=30) as source:
            data = source.read(MAX_WHEEL_BYTES + 1)
    if len(data) > MAX_WHEEL_BYTES or hashlib.sha256(data).hexdigest() != LEGACY_SHA256:
        raise ValueError("Legacy release wheel did not match the pinned checksum")
    path = destination / LEGACY_NAME
    path.write_bytes(data)
    return path


def wait_ready(archive, launcher, service, prefix, token, log):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if launcher.poll() is not None:
            raise RuntimeError(f"{service} exited: {log.read_text(errors='replace')}")
        try:
            record = json.loads((archive / "proxy.json").read_text())
            process = psutil.Process(record["pid"])
            if process.create_time() != record["created_at"]:
                raise ValueError("stale process identity")
            status = get(record["base_url"] + prefix + "status", token)
            if status["service"] == service:
                return record, status
        except (OSError, ValueError, KeyError, psutil.Error):
            pass
        time.sleep(.1)
    raise RuntimeError(f"{service} did not become ready: {log.read_text(errors='replace')}")


def alive(record):
    try:
        process = psutil.Process(record["pid"])
        return process.create_time() == record["created_at"] and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def cleanup(launchers, records):
    # Windows venv launchers can spawn the actual Python process. Stop verified
    # published proxy identities and any descendants of our own live launchers.
    owned = []
    for record in records:
        if alive(record):
            owned.append(psutil.Process(record["pid"]))
    for launcher in launchers:
        if launcher.poll() is None:
            try:
                process = psutil.Process(launcher.pid)
                owned.extend(reversed(process.children(recursive=True)))
                owned.append(process)
            except psutil.NoSuchProcess:
                pass
    for process in owned:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(owned, timeout=10)
    for process in remaining:
        try:
            process.kill()  # Only disposable acceptance processes we launched.
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(remaining, timeout=10)
    for launcher in launchers:
        launcher.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--legacy-wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if wheel.is_dir():
        wheel, = wheel.glob("kvpark-*-py3-none-any.whl")
    with tempfile.TemporaryDirectory(prefix="kvpark-migration-test-") as tmp:
        # Match the canonical paths published by both proxies on macOS (/private)
        # and Windows (expanded short directory names).
        root = Path(tmp).resolve()
        old = legacy_wheel(root, args.legacy_wheel)
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", str(old), str(wheel)],
                       check=True, stdout=subprocess.DEVNULL, timeout=120)
        site = Path(subprocess.check_output([str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                                            text=True, timeout=15).strip())
        dependencies = {str(metadata.distribution(name).locate_file("")) for name in ("psutil", "packaging")}
        (site / "test-dependencies.pth").write_text("\n".join(sorted(dependencies)) + "\n")
        env = {key: value for key, value in os.environ.items() if not key.startswith(("SLOTH_", "KVPARK_"))}
        token = uuid.uuid4().hex
        env["SLOTH_API_KEY"] = token
        archive = root / "custom archive"
        archive.mkdir()
        key = "k-" + "a" * 64
        name = key + "." + "b" * 32 + ".bin"
        preserved = {name: b"synthetic snapshot preservation marker", name + ".resume": b"synthetic companion"}
        stamp = time.time()
        preserved[key + ".meta"] = json.dumps(dict(format=2, filename=name, identity="acceptance-marker",
            tokens=17, bytes=sum(map(len, preserved.values())), saved_at=stamp, last_used=stamp)).encode()
        preserved["retention.json"] = b'{"ttl_days":7,"max_gib":32,"cleanup_enabled":true}'
        for name, data in preserved.items():
            (archive / name).write_bytes(data)
        backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        backend.daemon_threads = True
        worker = threading.Thread(target=backend.serve_forever, daemon=True)
        worker.start()
        upstream = f"http://127.0.0.1:{backend.server_port}"
        launchers, records = [], []
        try:
            def launch(module):
                log = root / (module + ".log")
                with log.open("ab") as output:
                    child = subprocess.Popen([str(python), "-I", "-m", module, "serve", "--archive-dir", str(archive),
                        "--port", "18081", "--backend", "ollama", "--cache-mode", "routing", "--upstream-url", upstream],
                        env=env, stdout=output, stderr=output, stdin=subprocess.DEVNULL)
                launchers.append(child)
                return child, log

            old_child, old_log = launch("sloth_memory")
            original, status = wait_ready(archive, old_child, "sloth-memory", "/_sloth/", token, old_log)
            records.append(original)
            assert status["version"] == "0.2.0a2", status["version"]
            assert status["entries"][0]["key"] == key
            # Legacy startup normalizes retention JSON formatting. Compare the
            # migration against that established on-disk baseline.
            preserved["retention.json"] = (archive / "retention.json").read_bytes()
            assert json.loads(preserved["retention.json"]) == dict(ttl_days=7, max_gib=32, cleanup_enabled=True)
            command = [str(python), "-I", "-m", "kvpark", "migrate", "--archive-dir", str(archive)]
            preview = subprocess.check_output(command, env=env, text=True, timeout=30)
            assert "Migration preview" in preview, preview
            assert alive(original), "Preview stopped the legacy proxy"
            assert get(original["base_url"] + "/_sloth/status", token)["service"] == "sloth-memory"
            subprocess.run([*command, "--confirm"], env=env, check=True, stdout=subprocess.DEVNULL, timeout=120)
            old_child.wait(timeout=10)
            assert not alive(original), "Confirmed migration left the legacy proxy alive"
            assert get(upstream)["status"] == "ok", "Migration interrupted the existing backend"
            for name, data in preserved.items():
                assert (archive / name).read_bytes() == data, f"Migration changed {name}"
            child, log = launch("kvpark")
            current, status = wait_ready(archive, child, "kvpark", "/_kvpark/", token, log)
            records.append(current)
            assert status["version"] == wheel.name.split("-")[1]
            assert status["entries"][0]["key"] == key and status["entries"][0]["tokens"] == 17
            assert current["upstream_url"] == original["upstream_url"]
            assert status["archive_dir"] == str(archive)
            assert get(current["base_url"] + "/_kvpark/doctor", token)["ok"]
            try:
                get(current["base_url"] + "/_kvpark/status")
            except HTTPError as exc:
                assert exc.code == 401, exc.code
            else:
                raise AssertionError("Legacy authentication was not preserved")
            for name, data in preserved.items():
                assert (archive / name).read_bytes() == data, f"New proxy changed {name}"
            assert get(upstream)["status"] == "ok"
            print("PASS: actual legacy wheel migration, read-only preview, old proxy stopped, backend and archive retained, legacy auth accepted")
        finally:
            try:
                cleanup(launchers, records)
            finally:
                backend.shutdown()
                backend.server_close()
                worker.join(timeout=5)


if __name__ == "__main__":
    main()
