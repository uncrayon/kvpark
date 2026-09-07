"""Discovery and guided setup must not change routing before a verified choice."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import psutil

from kvpark import discovery, setup
from kvpark.hermes_service import defaults


class ModelAPI(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.seen.append((self.path, dict(self.headers)))
        status, body = self.server.responses.get(self.path, (404, {}))
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Location", self.server.redirect)
        self.end_headers()
        self.wfile.write(data)


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ModelAPI)
        self.server.seen = []
        self.server.responses = {"/v1/models": (200, {"data": [{"id": "my model/with spaces", "owned_by": "vllm"}]})}
        self.server.redirect = "http://127.0.0.1:1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def test_finds_models_on_a_nonstandard_listening_port(self):
        listener = SimpleNamespace(status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip="127.0.0.1", port=self.server.server_port))
        with patch.object(psutil, "net_connections", return_value=[listener]), patch.object(discovery, "COMMON_PORTS", ()):
            scan = discovery.discover()
        self.assertEqual(scan.models, [discovery.Model(self.url, "my model/with spaces", "vllm")])
        self.assertTrue(all(path not in ("/v1/chat/completions", "/api/generate") for path, _ in self.server.seen))

    def test_familiar_ports_still_work_when_listener_access_is_denied(self):
        with patch.object(psutil, "net_connections", side_effect=psutil.AccessDenied()), patch.object(psutil, "process_iter", return_value=[]), patch.object(discovery, "COMMON_PORTS", (self.server.server_port,)):
            self.assertEqual(len(discovery.discover().models), 1)

    def test_per_process_discovery_finds_custom_ports_without_system_access(self):
        listener = SimpleNamespace(status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip="127.0.0.1", port=self.server.server_port))
        process = Mock()
        process.net_connections.return_value = [listener]
        with patch.object(psutil, "net_connections", side_effect=psutil.AccessDenied()), patch.object(psutil, "process_iter", return_value=[process]), patch.object(discovery, "COMMON_PORTS", ()):
            self.assertEqual(len(discovery.discover().models), 1)

    def test_nonlocal_addresses_are_not_scanned_automatically(self):
        listener = SimpleNamespace(status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip="192.0.2.4", port=1234))
        with patch.object(psutil, "net_connections", return_value=[listener]), patch.object(discovery, "COMMON_PORTS", ()):
            self.assertEqual(discovery.listening_urls(["http://192.0.2.4:8000"]), [])

    def test_scan_never_sends_configured_credentials(self):
        with patch.dict(os.environ, KVPARK_API_KEY="private", HTTP_PROXY="http://127.0.0.1:1", KVPARK_UPSTREAM_KEY_FILE="missing-key-file"):
            self.assertTrue(discovery.inspect_server(self.url).models)
        self.assertTrue(all("Authorization" not in headers for _, headers in self.server.seen))

    def test_proxy_is_not_presented_as_an_upstream(self):
        self.server.responses["/_kvpark/status"] = (200, {"service": "kvpark"})
        self.assertEqual(discovery.inspect_server(self.url).models, [])

    def test_locked_server_explains_why_no_models_are_available(self):
        self.server.responses["/v1/models"] = (401, {})
        result = discovery.inspect_server(self.url)
        self.assertFalse(result.models)
        self.assertIn("API key", result.notes[0])

    def test_empty_server_and_unrelated_listener_are_distinguished(self):
        self.server.responses["/v1/models"] = (200, {"data": []})
        self.assertIn("no model is loaded", discovery.inspect_server(self.url).notes[0])
        self.server.responses["/v1/models"] = (200, {"status": "not a model API"})
        self.assertEqual(discovery.inspect_server(self.url), discovery.Scan())

    def test_ollama_tags_work_without_openai_models_endpoint(self):
        self.server.responses = {"/api/tags": (200, {"models": [{"name": "gemma:latest"}]})}
        self.assertEqual(discovery.inspect_server(self.url).models, [discovery.Model(self.url, "gemma:latest", "ollama")])

    def test_generic_api_is_not_mislabeled_as_a_specific_engine(self):
        self.server.responses["/v1/models"] = (200, {"data": [{"id": "custom"}]})
        self.assertEqual(discovery.inspect_server(self.url).models[0].backend, "openai")

    def test_redirects_are_not_followed(self):
        self.server.responses["/v1/models"] = (302, {})
        self.assertEqual(discovery.get_json(self.url, "/v1/models"), (302, None))
        self.assertEqual(len(self.server.seen), 1)

    def test_selected_connection_is_persisted_and_plain_cli_commands_find_it(self):
        from kvpark.network import proxy_record
        from kvpark.removal import stop
        model = discovery.inspect_server(self.url).models[0]
        with tempfile.TemporaryDirectory(prefix="kvpark selected model ") as tmp:
            env = {key: value for key, value in os.environ.items() if not key.startswith(("KVPARK_", "SLOTH_"))}
            env.update(XDG_DATA_HOME=tmp)
            with patch.dict(os.environ, env, clear=True):
                config = setup.connect_model(model, defaults())
                try:
                    self.assertEqual(setup.saved_settings()["model"], "my model/with spaces")
                    before = proxy_record(Path(config["archive_dir"]))
                    setup.connect_model(model, setup.saved_settings())
                    self.assertEqual(proxy_record(Path(config["archive_dir"]))["pid"], before["pid"])
                    self.assertFalse(discovery.inspect_server(config["base_url"]).models)
                    for command in ("doctor", "status", "start"):
                        result = subprocess.run([sys.executable, "-m", "kvpark", command], capture_output=True, text=True, timeout=15)
                        self.assertEqual(result.returncode, 0, result.stderr)
                    result = subprocess.run([sys.executable, "-m", "kvpark", "uninstall", "--confirm"], capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIsNone(setup.saved_settings())
                    self.assertTrue(discovery.inspect_server(self.url).models)
                finally:
                    stop(config["archive_dir"], managed_backend=False)

    def test_model_is_rechecked_before_connecting(self):
        model = discovery.inspect_server(self.url).models[0]
        self.server.responses["/v1/models"] = (200, {"data": []})
        with patch.object(setup.Service, "ensure") as start:
            with self.assertRaisesRegex(ValueError, "no longer available"):
                setup.connect_model(model, defaults())
        start.assert_not_called()


class GuideTests(unittest.TestCase):
    def setUp(self):
        self.guide = setup.Guide()
        self.models = [discovery.Model("http://127.0.0.1:11434", "gemma:latest", "ollama"),
                       discovery.Model("http://127.0.0.1:9321", "custom/model ID")]
        scan = patch.object(discovery, "discover", return_value=discovery.Scan(self.models))
        scan.start()
        self.addCleanup(scan.stop)
        self.apply = Mock(return_value={"base_url": "http://127.0.0.1:12345"})
        self.current = {"model": "original"}

    def step(self, args, owner="chat-a", current=None):
        return self.guide.handle(args, owner=owner, current=current or self.current, apply=self.apply)

    def test_number_then_confirmation_is_enough_to_select_the_exact_model(self):
        self.assertIn("1. gemma:latest", self.step([]))
        self.assertIn("Use custom/model ID", self.step(["2"]))
        self.apply.assert_not_called()
        self.assertIn("Connected to custom/model ID", self.step(["yes"]))
        self.apply.assert_called_once_with(self.models[1])

    def test_none_of_these_makes_no_changes(self):
        self.step([])
        self.step(["1"])
        self.assertIn("unchanged", self.step(["0"]))
        self.apply.assert_not_called()
        self.assertFalse(self.guide.pending)

    def test_choices_and_confirmation_are_scoped_to_the_conversation(self):
        self.step([])
        self.step(["1"])
        self.assertIn("Start with", self.step(["yes"], owner="chat-b"))
        self.apply.assert_not_called()
        self.step(["yes"])
        self.apply.assert_called_once_with(self.models[0])

    def test_settings_changed_after_preview_are_preserved(self):
        self.step([])
        self.step(["1"])
        self.assertIn("settings changed", self.step(["yes"], current={"model": "changed elsewhere"}))
        self.apply.assert_not_called()

    def test_expired_choices_require_a_new_scan(self):
        self.step([])
        self.step(["1"])
        self.guide.pending["chat-a"].created -= 601
        self.assertIn("expire", self.step(["yes"]))
        self.apply.assert_not_called()

    def test_no_models_gives_plain_next_steps_and_a_cancel_choice(self):
        with patch.object(discovery, "discover", return_value=discovery.Scan()):
            result = self.step([])
        self.assertIn("Open your model app", result)
        self.assertIn("0. None", result)
        self.assertNotIn("--model", result)

    def test_terminal_reads_simple_answers(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        output = io.StringIO()
        with patch.object(setup, "saved_settings", return_value=None), patch.object(setup, "connect_model", self.apply):
            setup.terminal(input_stream=Terminal("2\nyes\n"), output_stream=output)
        self.assertIn("Connected to custom/model ID", output.getvalue())
        self.assertNotIn("--model", output.getvalue())
        self.assertEqual(self.apply.call_args.args[0], self.models[1])


class ConnectionTests(unittest.TestCase):
    def test_connection_failure_leaves_saved_selection_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(setup, "directory", return_value=Path(tmp)), patch.object(discovery, "verify"), patch.object(setup.Service, "ensure", side_effect=RuntimeError("offline")):
                with self.assertRaisesRegex(RuntimeError, "offline"):
                    setup.connect_model(discovery.Model("http://127.0.0.1:8000", "model"), defaults())
                self.assertFalse((Path(tmp) / "setup.json").exists())

    def test_explicit_archive_environment_is_used_by_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, KVPARK_ARCHIVE_DIR=tmp), patch.object(discovery, "verify"), patch.object(setup.Service, "ensure", return_value={"service": "kvpark"}):
                config = setup.connect_model(discovery.Model("http://127.0.0.1:8000", "model"), defaults())
                self.assertEqual(Path(config["archive_dir"]), Path(tmp).resolve())
                self.assertEqual(setup.saved_settings()["model"], "model")

    def test_failed_config_save_stops_only_children_created_by_this_attempt(self):
        child = Mock()
        child.poll.return_value = None
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(setup, "directory", return_value=Path(tmp)), patch.object(discovery, "verify"), patch.object(setup, "Service") as service, patch.object(setup, "saved_settings", return_value=None), patch("kvpark.proxy.atomic_json", side_effect=OSError("disk full")):
                service.return_value.ensure.return_value = {"service": "kvpark"}
                service.return_value.started_processes = [child]
                with self.assertRaisesRegex(OSError, "disk full"):
                    setup.connect_model(discovery.Model("http://127.0.0.1:8000", "model"), defaults())
        child.terminate.assert_called_once()
        child.wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
