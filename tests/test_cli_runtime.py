import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from sloth_memory import runtime
from sloth_memory import proxy


class LauncherTests(unittest.TestCase):
    def test_managed_topology_cannot_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"model")
            with patch.dict(os.environ, {}, clear=True):
                for extra in (["--parallel=2"], ["--port", "1234"], ["-m", "other"], ["--rpc", "host"]):
                    with self.assertRaises(ValueError):
                        runtime.backend_command(sys.executable, str(model), Path(tmp), 8090, extra)
                argv = runtime.backend_command(sys.executable, str(model), Path(tmp), 8090, ["--ctx-size", "4096"])
                self.assertEqual(argv[argv.index("--parallel") + 1], "1")
                self.assertEqual(argv[argv.index("--model") + 1], str(model))
                with patch.dict(os.environ, {"LLAMA_ARG_MODEL": "another-model"}):
                    with self.assertRaisesRegex(ValueError, "LLAMA_ARG"):
                        runtime.backend_command(sys.executable, str(model), Path(tmp), 8090, [])

    def test_manifest_includes_split_models_and_detects_asset_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "model-00001-of-00002.gguf"
            second = Path(tmp) / "model-00002-of-00002.gguf"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            manifest = Path(tmp) / "runtime.json"
            argv = [sys.executable, "--model", str(first)]
            with patch("subprocess.run") as linked:
                linked.return_value = subprocess.CompletedProcess([], 0, "", "")
                runtime.write_manifest(manifest, os.getpid(), argv)
                data = json.loads(manifest.read_text())
                self.assertIn(str(second), data["files"])
                second.write_bytes(b"replacement")
                runtime.write_manifest(manifest, os.getpid(), argv)
                self.assertNotEqual(data["identity"], json.loads(manifest.read_text())["identity"])

    def test_directory_has_exclusive_process_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            fd = runtime.lock_directory(Path(tmp), ".test.lock")
            try:
                with self.assertRaises(RuntimeError):
                    runtime.lock_directory(Path(tmp), ".test.lock")
            finally:
                os.close(fd)

    def test_cli_requires_explicit_identity(self):
        result = subprocess.run([sys.executable, "-m", "sloth_memory", "park"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--session", result.stderr)


class ControlTests(unittest.TestCase):
    def test_doctor_does_not_wait_for_generation(self):
        import threading
        with tempfile.TemporaryDirectory() as tmp, patch.object(proxy, "ARCHIVE", Path(tmp)):
            state = proxy.Archive()
            held, release = threading.Event(), threading.Event()
            def hold():
                with state.lock:
                    held.set()
                    release.wait(5)
            worker = threading.Thread(target=hold)
            worker.start()
            try:
                self.assertTrue(held.wait(1))
                self.assertFalse(state.doctor()["ok"])
            finally:
                release.set()
                worker.join()

    def test_forget_requires_existing_key_and_removes_only_selected_archive(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(proxy, "ARCHIVE", Path(tmp)):
            state = proxy.Archive()
            with self.assertRaises(ValueError):
                state.forget("../../other")
            key = proxy.key_for("session")
            filename = key + ".bin"
            (Path(tmp) / filename).write_bytes(b"state")
            (Path(tmp) / (filename + ".resume")).write_bytes(b"checkpoints")
            state.entries[key] = {"filename": filename, "bytes": 16, "tokens": 100, "last_used": 0}
            state.current_key = key
            self.assertTrue(state.forget(key)["ok"])
            self.assertFalse(state.entries)
            self.assertFalse((Path(tmp) / filename).exists())
            self.assertEqual(state.current_key, key)


class HTTPBoundaryTests(unittest.TestCase):
    def setUp(self):
        import threading
        from http.server import ThreadingHTTPServer
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with patch.object(proxy, "ARCHIVE", Path(self.tmp.name)):
            self.state = proxy.Archive()
        self.state_patch = patch.object(proxy, "ARCHIVE_STATE", self.state, create=True)
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), proxy.Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def call(self, method="GET", path="/_sloth/status", body=None, headers=None):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def test_auth_protects_status_and_mutations(self):
        with patch.object(proxy, "API_KEY", "test-secret"):
            self.assertEqual(self.call()[0], 401)
            self.assertEqual(self.call("POST", "/_sloth/park", b"{}")[0], 401)
            self.assertEqual(self.call(headers={"Authorization": "Bearer test-secret"})[0], 200)

    def test_browser_origins_cannot_control_local_service(self):
        with patch.object(proxy, "API_KEY", None):
            self.assertEqual(self.call(headers={"Origin": "https://example.com"})[0], 403)

    def test_oversized_bodies_rejected_before_reading(self):
        with patch.object(proxy, "API_KEY", None):
            self.assertEqual(self.call("POST", "/v1/chat/completions", headers={"Content-Length": str(65 * 1024**2)})[0], 413)

    def test_untracked_generation_and_direct_slot_controls_are_blocked(self):
        with patch.object(proxy, "API_KEY", None):
            self.assertEqual(self.call("POST", "/v1/responses", b"{}")[0], 404)
            self.assertEqual(self.call("POST", "/slots/0?action=erase", b"{}")[0], 409)

    def test_unsafe_restore_never_forwards_generation(self):
        with patch.object(proxy, "API_KEY", None), patch.object(self.state, "switch", side_effect=proxy.UnsafeRestoreError("restart backend")), patch.object(proxy.Handler, "forward") as forward:
            self.assertEqual(self.call("POST", "/v1/chat/completions", b"{}")[0], 502)
            forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
