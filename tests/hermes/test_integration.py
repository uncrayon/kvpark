"""Run with Hermes' Python environment and its source importable; all profiles are temporary."""

import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sloth_memory import proxy


class HermesIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        env = patch.dict(os.environ, {"HERMES_HOME": str(self.home), "HERMES_BUNDLED_PLUGINS": str(self.home / "empty")})
        env.start()
        self.addCleanup(env.stop)
        with patch.object(proxy, "ARCHIVE", self.home / "archive"):
            self.archive = proxy.Archive()
        self.archive_patch = patch.object(proxy, "ARCHIVE", self.home / "archive")
        self.archive_patch.start()
        self.addCleanup(self.archive_patch.stop)
        state = patch.object(proxy, "ARCHIVE_STATE", self.archive, create=True)
        state.start()
        self.addCleanup(state.stop)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), proxy.Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        (self.home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["sloth-memory"],
            "entries": {"sloth-memory": {"settings": {"service": {"autostart": False, "external": True, "base_url": self.url}}}}},
            "agent": {"max_turns": 12}, "model": {"provider": "custom", "default": "old", "base_url": self.url + "/v1"}}))
        from hermes_cli.plugins import get_plugin_manager, get_plugin_command_handler
        self.manager = get_plugin_manager()
        self.manager.discover_and_load()
        self.addCleanup(self.manager.unload)
        self.command = get_plugin_command_handler("sloth-memory")
        self.assertIsNotNone(self.command, "installed entry point did not register the slash command")

    def command_text(self, text):
        return asyncio.run(self.command(text))

    def test_real_discovery_onboarding_and_persistent_settings_over_http(self):
        self.assertIn("setup --server", self.command_text("setup --no-external"))
        self.assertIn("ttl_days", self.command_text("retention 14"))
        self.assertEqual(self.archive.retention.settings()["ttl_days"], 14)
        self.assertIn("Cleanup:", self.command_text("cleanup off"))
        self.assertFalse(self.archive.retention.settings()["cleanup_enabled"])
        reboot = proxy.Archive()
        self.assertFalse(reboot.retention.settings()["cleanup_enabled"])
        self.assertEqual(reboot.retention.settings()["ttl_days"], 14)
        self.assertIn("Cleanup:", self.command_text("cleanup on"))
        self.assertTrue(self.archive.retention.settings()["cleanup_enabled"])
        from hermes_cli.config import read_user_config_raw
        self.assertEqual(read_user_config_raw()["agent"]["max_turns"], 12)

    def test_real_middleware_distinguishes_same_id_background_and_scopes_routes(self):
        from agent.subagent_lifecycle import bind_subagent_parent
        from hermes_cli.middleware import apply_llm_request_middleware
        from gateway.session_context import set_session_vars, clear_session_vars
        tokens = set_session_vars(session_id="session-a", session_key="discord:thread-a")
        self.addCleanup(clear_session_vars, tokens)
        original = {"model": "local", "messages": [{"role": "system", "content": "stable"}],
                    "extra_body": {"slot_archive_key": "inherited-wrong-key"}}
        for disabled, expected in ((False, "foreground"), (True, "background")):
            with bind_subagent_parent(SimpleNamespace(session_id="session-a", _persist_disabled=disabled)):
                result = apply_llm_request_middleware(original, session_id="session-a", base_url=self.url + "/v1", api_mode="chat_completions")
            extra = result.payload["extra_body"]
            self.assertEqual(extra["slot_archive_role"], expected)
            if disabled:
                self.assertNotIn("slot_archive_key", extra)
            else:
                self.assertTrue(extra["slot_archive_key"].endswith(":session-a"))
                self.assertEqual(result.payload["messages"], original["messages"])
        self.assertNotIn("slot_archive_role", original["extra_body"])
        result = apply_llm_request_middleware(original, session_id="session-a", base_url="https://example.com/v1", api_mode="chat_completions")
        self.assertEqual(result.payload, original)

    def test_unknown_active_context_is_unverified(self):
        from hermes_cli.middleware import apply_llm_request_middleware
        result = apply_llm_request_middleware({"messages": []}, session_id="unknown", base_url=self.url + "/v1", api_mode="chat_completions")
        self.assertEqual(result.payload["extra_body"]["slot_archive_role"], "unverified")

    def test_delete_command_removes_exact_snapshot_without_model_turn(self):
        key = proxy.key_for("selected")
        filename = key + "." + "a"*32 + ".bin"
        for suffix in ("", ".resume"):
            (proxy.ARCHIVE / (filename + suffix)).write_bytes(b"state")
        self.archive.entries[key] = dict(filename=filename, bytes=10, tokens=100, saved_at=0, last_used=0)
        self.archive.publish_status()
        self.assertIn("Deleted saved snapshot", self.command_text("delete " + key))
        self.assertFalse(self.archive.entries)
        self.assertFalse((proxy.ARCHIVE / filename).exists())

    def test_autostart_setting_persists_and_discovery_invokes_startup(self):
        self.command_text("autostart off")
        from sloth_memory.hermes_service import Service
        with patch.object(Service, "start_async") as start:
            self.command_text("autostart on")
            start.assert_called_once()
            start.reset_mock()
            self.manager.discover_and_load(force=True)
            start.assert_called_once()

    def test_connect_changes_future_route_without_dropping_other_settings(self):
        self.assertIn("new sessions", self.command_text("connect"))
        from hermes_cli.config import read_user_config_raw
        config = read_user_config_raw()
        self.assertEqual(config["model"]["base_url"], self.url + "/v1")
        self.assertEqual(config["model"]["default"], "local")
        self.assertEqual(config["agent"]["max_turns"], 12)

    def test_matching_live_openai_client_follows_new_proxy_port(self):
        from agent.subagent_lifecycle import bind_subagent_parent
        from hermes_cli.middleware import apply_llm_request_middleware
        from openai import OpenAI
        adapter = self.command.__self__
        old_url = "http://127.0.0.1:12345"
        adapter.ctx.set_config("service", {**adapter.settings(), "connected": True, "previous_urls": [old_url]})
        with OpenAI(base_url=old_url + "/v1", api_key="local") as client:
            active = SimpleNamespace(session_id="route-test", _persist_disabled=False, client=client, base_url=old_url + "/v1")
            with bind_subagent_parent(active):
                result = apply_llm_request_middleware({"model": "local", "messages": []}, session_id="route-test",
                    base_url=old_url + "/v1", api_mode="chat_completions")
            self.assertEqual(str(client.base_url).rstrip("/"), self.url + "/v1")
            self.assertEqual(active.base_url, self.url + "/v1")
            self.assertEqual(result.payload["extra_body"]["slot_archive_role"], "foreground")

    def test_generic_backend_model_name_is_preserved_when_connecting(self):
        adapter = self.command.__self__
        adapter.ctx.set_config("service", {**adapter.settings(), "backend": "ollama", "cache_mode": "routing",
                                          "model": "my-model:latest"})
        self.assertIn("new sessions", self.command_text("connect"))
        from hermes_cli.config import read_user_config_raw
        self.assertEqual(read_user_config_raw()["model"]["default"], "my-model:latest")

    def test_changing_backend_clears_previous_model_and_endpoint(self):
        adapter = self.command.__self__
        adapter.ctx.set_config("service", {**adapter.settings(), "backend": "ollama", "cache_mode": "routing",
            "model": "old-model", "upstream_url": "http://127.0.0.1:11434"})
        self.command_text("setup --backend vllm")
        self.assertEqual(adapter.settings()["model"], "")
        self.assertEqual(adapter.settings()["upstream_url"], "")


@unittest.skipUnless(os.environ.get("SLOTH_TEST_SERVER") and os.environ.get("SLOTH_TEST_MODEL"), "optional real-model startup test")
class RealStartupTests(unittest.TestCase):
    def test_hermes_discovery_launches_services_and_native_park_delete_work(self):
        from sloth_memory.hermes_service import Service
        from sloth_memory.cli import request
        from hermes_cli.plugins import get_plugin_manager, get_plugin_command_handler
        from hermes_cli.middleware import apply_llm_request_middleware
        from agent.subagent_lifecycle import bind_subagent_parent
        from gateway.session_context import set_session_vars, clear_session_vars
        from openai import OpenAI

        with tempfile.TemporaryDirectory(prefix="sloth-hermes-real-") as tmp:
            home = Path(tmp)
            occupied = []
            for _ in range(2):
                sock = socket.socket()
                sock.bind(("127.0.0.1", 0))
                sock.listen()
                occupied.append(sock)
                self.addCleanup(sock.close)
            port, upstream = (sock.getsockname()[1] for sock in occupied)
            url = f"http://127.0.0.1:{port}"
            config = dict(autostart=True, base_url=url, upstream_port=upstream, archive_dir=str(home / "archive"),
                          server=os.environ["SLOTH_TEST_SERVER"], model=os.environ["SLOTH_TEST_MODEL"],
                          backend_args=["--device", "none", "--n-gpu-layers", "0", "--ctx-size", "2048",
                                        "--threads", "4", "--threads-batch", "4", "--cache-ram", "0",
                                        "--ctx-checkpoints", "8", "--checkpoint-min-step", "128", "--jinja"])
            (home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["sloth-memory"],
                "entries": {"sloth-memory": {"settings": {"service": config}}}}}))
            children = []
            spawn = Service._spawn
            def tracked_spawn(*args):
                child = spawn(*args)
                children.append(child)
                return child
            with patch.dict(os.environ, {"HERMES_HOME": str(home), "HERMES_BUNDLED_PLUGINS": str(home / "empty")}), patch.object(Service, "_spawn", side_effect=tracked_spawn):
                manager = get_plugin_manager()
                try:
                    manager.discover_and_load()
                    command = get_plugin_command_handler("sloth-memory")
                    # No explicit start command: discovery itself must launch.
                    deadline = time.monotonic() + 120
                    while time.monotonic() < deadline:
                        try:
                            from sloth_memory.network import proxy_record
                            record = proxy_record(home / "archive")
                            if record:
                                url = record["base_url"]
                                status = request(url, "status", timeout=2)
                                break
                        except OSError:
                            pass
                        time.sleep(.2)
                    else:
                        self.fail("Hermes autostart did not launch the proxy")
                    self.assertEqual(len(children), 2)
                    command.__self__.service.worker.join(10)
                    actual = command.__self__.settings()
                    self.assertNotEqual(actual["upstream_port"], upstream)
                    self.assertNotEqual(url, f"http://127.0.0.1:{port}")
                    self.assertEqual(actual["base_url"], url)
                    self.assertTrue(status["cleanup_enabled"])
                    self.assertEqual(status["ttl_days"], 7)
                    tokens = set_session_vars(session_id="hermes-real", session_key="cli:real")
                    try:
                        with bind_subagent_parent(SimpleNamespace(session_id="hermes-real", _persist_disabled=False)):
                            payload = apply_llm_request_middleware(dict(model="local", max_tokens=16, temperature=0,
                                messages=[dict(role="user", content="Reply with HELLO only.")]),
                                session_id="hermes-real", base_url=url + "/v1", api_mode="chat_completions").payload
                        with OpenAI(base_url=url + "/v1", api_key="local", timeout=120) as client:
                            response = client.chat.completions.create(**payload)
                        self.assertTrue(response.choices[0].message.content)
                        self.assertIn("Parked", asyncio.run(command("park")))
                        self.assertEqual(request(url, "status")["archived"], 1)
                        self.assertIn("Deleted saved snapshot", asyncio.run(command("delete")))
                        self.assertEqual(request(url, "status")["archived"], 0)
                    finally:
                        clear_session_vars(tokens)
                    manager.unload()
                    self.assertTrue(all(child.poll() is None for child in children), "cleanup services must survive closing Hermes")
                    manager.discover_and_load(force=True)
                    get_plugin_command_handler("sloth-memory").__self__.service.worker.join(10)
                    self.assertEqual(len(children), 2, "reload started duplicate services")
                finally:
                    manager.unload()
                    for child in reversed(children):
                        if child.poll() is None:
                            child.terminate()
                            try:
                                child.wait(10)
                            except Exception:
                                child.kill()
                                child.wait()


if __name__ == "__main__":
    unittest.main()
