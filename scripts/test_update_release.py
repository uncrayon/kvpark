"""Exercise real pip upgrade/rollback in a disposable venv with a local HTTP backend.

Usage: python scripts/test_update_release.py dist/sloth_memory-0.2.0a1-py3-none-any.whl
Requires the build environment to have psutil and packaging installed. No network.
"""

import base64
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import venv
import zipfile

import psutil
from packaging.version import Version


def variant(original, destination, version, *, broken=False):
    rows = []
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as wheel:
        old = next(n.split("/")[0] for n in source.namelist() if n.endswith(".dist-info/METADATA"))
        old_version = old[len("sloth_memory-"):-len(".dist-info")]
        new = f"sloth_memory-{version}.dist-info"
        for name in source.namelist():
            if name.endswith("/RECORD"):
                continue
            data = source.read(name)
            if name.endswith("/METADATA") or name == "sloth_memory/__init__.py":
                data = data.replace(old_version.encode(), version.encode())
            if broken and name == "sloth_memory/__main__.py":
                data = b'import sys\nif "serve" in sys.argv: raise SystemExit(42)\n' + data
            name = name.replace(old, new)
            wheel.writestr(name, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            rows.append((name, "sha256=" + digest, str(len(data))))
        record = new + "/RECORD"
        out = io.StringIO(newline="")
        csv.writer(out).writerows([*rows, (record, "", "")])
        wheel.writestr(record, out.getvalue())


class Backend(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = b'{"status":"ok","data":[]}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    wheel = Path(sys.argv[1]).resolve()
    if wheel.is_dir():
        wheel, = wheel.glob("sloth_memory-*-py3-none-any.whl")
    current = wheel.name.split("-")[1]
    parsed = Version(current)
    future = f"{parsed.major}.{parsed.minor}.{parsed.micro + 1}a1"
    with tempfile.TemporaryDirectory(prefix="sloth-update-test-") as tmp:
        root = Path(tmp)
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        baseline = root / "sloth_memory-0.0.0-py3-none-any.whl"
        broken = root / f"sloth_memory-{future}-py3-none-any.whl"
        variant(wheel, baseline, "0.0.0")
        variant(wheel, broken, future, broken=True)
        subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(baseline)], check=True,
                       stdout=subprocess.DEVNULL)
        site = Path(subprocess.check_output([str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
        dependencies = {str(metadata.distribution(name).locate_file("")) for name in ("psutil", "packaging")}
        (site / "test-dependencies.pth").write_text("\n".join(dependencies) + "\n")
        backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        threading.Thread(target=backend.serve_forever, daemon=True).start()
        archive = root / "archive"
        archive.mkdir()
        process = None
        try:
            with (root / "proxy.log").open("ab") as log:
                process = subprocess.Popen([str(python), "-I", "-m", "sloth_memory", "serve", "--archive-dir", str(archive),
                    "--port", "18081", "--backend", "ollama", "--cache-mode", "routing",
                    "--upstream-url", f"http://127.0.0.1:{backend.server_port}"], stdout=log, stderr=log)
            for _ in range(100):
                if (archive / "proxy.json").exists():
                    break
                if process.poll() is not None:
                    raise RuntimeError((root / "proxy.log").read_text())
                time.sleep(.1)
            original = json.loads((archive / "proxy.json").read_text())
            harness = root / "apply_update.py"
            harness.write_text('''import hashlib, json, sys
from pathlib import Path
from unittest.mock import patch
from sloth_memory import releases, updates
wheel, archive, version, expected = sys.argv[1:]
wheel = Path(wheel)
release = dict(version=version, url="unused", name=wheel.name, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
with patch.object(releases, "latest", return_value=release), patch.object(releases, "fetch", return_value=wheel.read_bytes()):
    try:
        result = updates.apply(archive)
        assert expected == "complete", result
    except RuntimeError as exc:
        assert expected == "rolled_back" and "rolled_back" in str(exc), str(exc)
        result = json.loads((Path(archive) / "update.json").read_text())
print(json.dumps({"state":result["state"], "version":result["version"]}))
''')
            for candidate, version, expected in ((wheel, current, "complete"), (broken, future, "rolled_back")):
                subprocess.run([str(python), "-I", str(harness), str(candidate), str(archive), version, expected], check=True)
                actual = subprocess.check_output([str(python), "-I", "-m", "sloth_memory", "--version"], text=True).strip()
                assert actual == current, actual
                record = json.loads((archive / "proxy.json").read_text())
                assert record["base_url"] == original["base_url"]
                assert record["upstream_url"] == original["upstream_url"]
                subprocess.run([str(python), "-I", "-m", "sloth_memory", "doctor", "--archive-dir", str(archive)],
                               stdout=subprocess.DEVNULL, check=True)
            print("PASS: real package upgrade, failed-start rollback, stable proxy route, backend retained")
        finally:
            if (archive / "proxy.json").exists():
                record = json.loads((archive / "proxy.json").read_text())
                try:
                    current = psutil.Process(record["pid"])
                    if current.create_time() == record["created_at"]:
                        current.terminate()
                        current.wait(10)
                except psutil.NoSuchProcess:
                    pass
            if process:
                process.wait(10)
            backend.shutdown()
            backend.server_close()


if __name__ == "__main__":
    main()
