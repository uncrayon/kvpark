"""HTTP contract tests exercise real proxy subprocesses, without model downloads."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from kvpark.backends import Backend
from kvpark.cli import request
from kvpark.hermes_service import Service, defaults
from kvpark.network import proxy_record


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.seen.append((self.path, None, dict(self.headers)))
        body = json.dumps({"status": "ok", "version": "test", "data": [{"id": "test-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append((self.path, body, dict(self.headers)))
        if body.get("stream"):
            output = b'data: {"choices":[{"delta":{"tool_calls":[{"id":"call_1","function":{"name":"lookup","arguments":"{}"}}]}}]}\n\ndata: [DONE]\n\n'
            content_type = "text/event-stream"
        else:
            output = json.dumps({"choices": [{"message": {"role": "assistant", "content": "hello"}}],
                                 "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 75}}}).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(output)))
        self.end_headers()
        self.wfile.write(output)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.upstream.seen = []
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        self.addCleanup(self.upstream.server_close)
        self.addCleanup(self.upstream.shutdown)
        self.occupied = socket.socket()
        self.occupied.bind(("127.0.0.1", 0))
        self.occupied.listen()
        self.addCleanup(self.occupied.close)
        self.preferred = self.occupied.getsockname()[1]
        self.upstream_url = f"http://127.0.0.1:{self.upstream.server_port}"
        self.children = []
        self.addCleanup(self.stop)

    def stop(self):
        for process in self.children:
            if process.poll() is None:
                process.terminate()
                process.wait(10)

    def start(self, backend):
        archive = self.root / backend
        env = {key: value for key, value in os.environ.items() if not key.startswith("KVPARK_")}
        log = self.root / (backend + ".log")
        with log.open("wb") as output:
            process = subprocess.Popen([sys.executable, "-m", "kvpark", "serve", "--backend", backend,
                "--cache-mode", "routing", "--upstream-url", self.upstream_url, "--port", str(self.preferred),
                "--archive-dir", str(archive)], env=env, stdout=output, stderr=subprocess.STDOUT)
        self.children.append(process)
        for _ in range(100):
            self.assertIsNone(process.poll(), "proxy exited: " + log.read_text(errors="replace"))
            record = proxy_record(archive)
            if record:
                return record, archive
            time.sleep(.05)
        self.fail("proxy did not publish its bound address: " + log.read_text(errors="replace"))

    def test_uninstall_stops_each_proxy_and_keeps_existing_backend_available(self):
        from kvpark import removal
        from urllib.request import urlopen
        for name in ("llama.cpp", "ollama", "vllm", "mlx"):
            with self.subTest(backend=name):
                record, archive = self.start(name)
                snapshot = archive / "saved-test.bin"
                snapshot.write_bytes(b"preserve")
                self.assertEqual(removal.plan(archive)["proxy"]["pid"], record["pid"])
                self.assertTrue(request(record["base_url"], "doctor")["ok"])
                removal.stop(archive, managed_backend=False)
                self.assertIsNone(proxy_record(archive))
                with urlopen(self.upstream_url + "/v1/models", timeout=5) as response:
                    self.assertEqual(response.status, 200)
                self.assertEqual(snapshot.read_bytes(), b"preserve")
                self.assertIsNone(removal.stop(archive, managed_backend=False)["proxy"])

    def test_four_backends_forward_models_streaming_tools_and_report_capabilities(self):
        for name in ("llama.cpp", "ollama", "vllm", "mlx"):
            with self.subTest(backend=name):
                record, archive = self.start(name)
                url = record["base_url"]
                self.assertNotEqual(url, f"http://127.0.0.1:{self.preferred}")
                self.assertTrue(request(url, "doctor")["ok"])
                status = request(url, "status")
                self.assertFalse(status["capabilities"]["disk_snapshots"])
                self.assertIsNone(status["runtime_error"])
                with self.assertRaisesRegex(RuntimeError, "cannot park"):
                    request(url, "park", {"key": "k-" + "a"*64})
                port = int(url.rsplit(":", 1)[1])
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                for stream in (False, True):
                    payload = {"model": "actual-backend-model", "messages": [{"role": "user", "content": "hi"}],
                               "stream": stream, "tools": [{"type": "function", "function": {"name": "lookup"}}],
                               "slot_archive_key": "session", "slot_archive_role": "foreground"}
                    conn.request("POST", "/v1/chat/completions", json.dumps(payload), {"Content-Type": "application/json", "Authorization": "Bearer backend-key"})
                    response = conn.getresponse()
                    self.assertEqual(response.status, 200)
                    output = response.read()
                    self.assertIn(b"[DONE]" if stream else b"hello", output)
                    _, body, headers = self.upstream.seen[-1]
                    self.assertEqual(body["model"], "actual-backend-model")
                    self.assertEqual(body["tools"], payload["tools"])
                    self.assertNotIn("slot_archive_key", body)
                    self.assertEqual(headers["Authorization"], "Bearer backend-key")
                    self.assertEqual(headers["Host"], self.upstream_url.removeprefix("http://"))
                conn.close()
                result = subprocess.run([sys.executable, "-m", "kvpark", "status", "--archive-dir", str(archive)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["capabilities"]["backend"], name)

    def test_hermes_service_persists_auto_port_and_reuses_proxy(self):
        config = {**defaults(), "archive_dir": str(self.root / "service"), "backend": "ollama", "cache_mode": "routing",
                  "model": "test-model", "upstream_url": self.upstream_url, "base_url": f"http://127.0.0.1:{self.preferred}"}
        service = Service(lambda: config, lambda updated: config.update(updated))
        original_spawn = service._spawn
        def tracked(*args):
            child = original_spawn(*args)
            self.children.append(child)
            return child
        service._spawn = tracked
        status = service.ensure()
        self.assertEqual(status["capabilities"]["backend"], "ollama")
        self.assertNotEqual(config["base_url"], f"http://127.0.0.1:{self.preferred}")
        self.assertIn(f"http://127.0.0.1:{self.preferred}", config["previous_urls"])
        service.ensure()
        self.assertEqual(len(self.children), 1)
        second = Service(lambda: config)
        second.ensure()
        self.assertEqual(request(config["base_url"], "status")["upstream_url"], self.upstream_url)

    def test_native_snapshots_cannot_be_advertised_for_generic_backends(self):
        for name in ("ollama", "vllm", "mlx"):
            with self.assertRaisesRegex(ValueError, "patched llama.cpp"):
                Backend(name, self.upstream_url, "native")


if __name__ == "__main__":
    unittest.main()
