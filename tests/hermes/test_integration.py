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

from kvpark import proxy


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
        (self.home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["kvpark"],
            "entries": {"kvpark": {"settings": {"service": {"autostart": False, "external": True, "base_url": self.url}}}}},
            "agent": {"max_turns": 12}, "model": {"provider": "custom", "default": "old", "base_url": self.url + "/v1"}}))
        from hermes_cli.plugins import get_plugin_manager, get_plugin_command_handler
        self.manager = get_plugin_manager()
        self.manager.discover_and_load()
        self.addCleanup(self.manager.unload)
        self.command = get_plugin_command_handler("kvpark")
        self.assertIsNotNone(self.command, "installed entry point did not register the slash command")

    def command_text(self, text):
        return asyncio.run(self.command(text))

    async def gateway_command(self, text, thread="discord:acceptance", *, platform="discord", command="kvpark"):
        from unittest.mock import AsyncMock
        from gateway.run_inbound import GatewayInboundMixin
        from gateway.session import SessionSource
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent

        class Runner(GatewayInboundMixin):
            config = {}
            _draining = False
            hooks = SimpleNamespace(emit_collect=AsyncMock(return_value=[]))

            def _check_slash_access(self, source, command):
                return None

            def _gateway_plain_command_handlers(self):
                return {}

            def _gateway_idle_command_handlers(self):
                return {}

        source = SessionSource(platform=Platform(platform), chat_id="test-channel", user_id="test-user")
        event = MessageEvent(text="/" + command + " " + text, source=source)
        handled, result = await Runner()._hm_dispatch_idle_commands(event, source, thread)
        self.assertTrue(handled)
        return result

    def test_telegram_short_command_preserves_chat_identity(self):
        adapter = self.command.__self__
        self.archive.current_key = adapter.key("telegram-session")
        self.archive.current_thread = adapter.namespace + "telegram:acceptance"
        self.archive.publish_status()
        with patch.object(self.archive, "park", return_value={"tokens": 456, "note": "Saved."}) as park:
            result = asyncio.run(self.gateway_command("save", "telegram:acceptance",
                                                      platform="telegram", command="kvpark"))
        self.assertIn("Parked 456 tokens", result)
        self.assertEqual(park.call_args.kwargs["key"], adapter.key("telegram-session"))

    def test_telegram_menu_priority_keeps_short_command_visible(self):
        from hermes_cli.config import save_config
        from hermes_cli.commands_platforms import telegram_menu_commands
        save_config({"platforms": {"telegram": {"extra": {"command_menu": {
            "priority": ["kvpark"], "priority_mode": "prepend"}}}}}, merge_existing=True)
        menu, _ = telegram_menu_commands(max_commands=5)
        self.assertIn("kvpark", [name for name, _ in menu])
        self.assertEqual(sum(name == "kvpark" for name, _ in menu), 1)

    def test_update_check_dispatches_without_installing_or_restarting(self):
        from kvpark import updates
        result = dict(installed="0.2.0a1", loaded="0.2.0a1", proxy="0.2.0a1", latest="0.2.0a2", available=True)
        with patch.object(updates, "check", return_value=result), patch.object(updates, "apply") as install:
            text = asyncio.run(self.gateway_command("update check", platform="telegram", command="kvpark"))
        self.assertIn("Available: 0.2.0a2", text)
        install.assert_not_called()

    def test_uninstall_restores_route_disables_plugin_and_does_not_restart_it(self):
        from hermes_cli.config import read_user_config_raw, save_config
        adapter = self.command.__self__
        # The original server occupied kvpark's preferred port, forcing fallback.
        original = dict(provider="custom", default="gemma", base_url="http://127.0.0.1:8080/v1", context_length=12345)
        adapter.ctx.set_config("service", {**adapter.settings(), "previous_urls": ["http://127.0.0.1:8080"]})
        save_config({"model": original}, merge_existing=True)
        self.command_text("connect")
        self.command_text("connect")  # Reconnect must not overwrite the original route.
        before = read_user_config_raw()
        preview = asyncio.run(self.gateway_command("uninstall", command="kvpark", platform="telegram"))
        self.assertIn("no changes", preview)
        self.assertEqual(before, read_user_config_raw())
        result = asyncio.run(self.gateway_command("uninstall confirm", command="kvpark", platform="telegram"))
        self.assertIn("kvpark disconnected", result)
        current = read_user_config_raw()
        self.assertEqual(current["model"], original)
        self.assertEqual(current["agent"]["max_turns"], 12)
        self.assertIn("kvpark", current["plugins"]["disabled"])
        self.assertNotIn("kvpark", current["plugins"]["enabled"])
        self.assertTrue(adapter.service.closed.is_set())
        self.assertIsNone(adapter.middleware(request={}, base_url=self.url + "/v1", api_mode="chat_completions"))
        self.assertIn("kvpark is disconnected", self.command_text("status"))
        self.assertIn("kvpark disconnected", self.command_text("uninstall confirm"))
        self.manager.unload()
        self.manager.discover_and_load(force=True)
        from hermes_cli.plugins import get_plugin_command_handler
        self.assertIsNone(get_plugin_command_handler("kvpark"))

    def test_uninstall_without_route_backup_does_not_change_profile(self):
        from hermes_cli.config import read_user_config_raw
        before = read_user_config_raw()
        result = self.command_text("uninstall confirm")
        self.assertIn("no original model-route backup", result)
        self.assertEqual(before, read_user_config_raw())

    def test_legacy_profile_migration_keeps_model_route_and_archive(self):
        from hermes_cli.config import read_user_config_raw, save_config
        from kvpark.migration import run
        raw = read_user_config_raw()
        old = raw["plugins"]["entries"].pop("kvpark")
        old["settings"]["service"].update(external=False, archive_dir=str(self.home / "legacy-archive"))
        old["settings"]["route_backup"] = {"provider": "custom", "base_url": "http://127.0.0.1:9999/v1"}
        raw["plugins"]["entries"]["sloth-memory"] = old
        raw["plugins"]["enabled"] = ["sloth-memory"]
        save_config(raw)
        before = read_user_config_raw()
        self.assertIn("no changes", run(hermes=True))
        self.assertEqual(read_user_config_raw(), before)
        self.assertIn("Migration prepared", run(hermes=True, confirm=True))
        after = read_user_config_raw()
        self.assertEqual(after["model"], before["model"])
        settings = after["plugins"]["entries"]["kvpark"]["settings"]
        self.assertEqual(settings["service"]["archive_dir"], str((self.home / "legacy-archive").resolve()))
        self.assertEqual(settings["route_backup"], old["settings"]["route_backup"])
        self.assertIn("already migrated", run(hermes=True, confirm=True))
        backup, = self.home.glob("config.before-kvpark-*.json")
        self.assertEqual(json.loads(backup.read_text()), before)
        if os.name != "nt":
            self.assertEqual(backup.stat().st_mode & 0o777, 0o600)

    def test_failed_legacy_proxy_stop_restores_profile_and_reopens_gate(self):
        from hermes_cli.config import read_user_config_raw, save_config
        from kvpark import migration
        raw = read_user_config_raw()
        old = raw["plugins"]["entries"].pop("kvpark")
        archive = str(self.home / "legacy-archive")
        old["settings"]["service"].update(external=False, archive_dir=archive)
        raw["plugins"]["entries"]["sloth-memory"] = old
        raw["plugins"]["enabled"] = ["sloth-memory"]
        save_config(raw)
        before = read_user_config_raw()
        record = dict(pid=123, base_url=self.url)
        status = dict(service="sloth-memory", control_version=4, archive_dir=archive)
        with patch.object(migration, "owned_process", return_value=record), \
                patch.object(migration, "request", side_effect=[status, {"ok": True}, {"ok": True}]) as request, \
                patch.object(migration, "stop_proxy", side_effect=RuntimeError("stop failed")), \
                patch.object(migration, "process_matches", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "stop failed"):
                migration.run(hermes=True, confirm=True)
            self.assertEqual(request.call_args.args[1], "cancel-update")
            self.assertEqual(read_user_config_raw(), before)

    def test_uninstall_respects_managed_plugin_leaf_settings(self):
        from hermes_cli.config import read_user_config_raw
        before = read_user_config_raw()
        for key in ("plugins.entries.kvpark.settings.service.autostart",
                    "plugins.entries.kvpark.settings.route_backup.base_url"):
            with patch("hermes_cli.managed_scope.managed_config_keys", return_value={key}):
                result = self.command_text("uninstall confirm")
                self.assertIn("managed by your administrator", result)
                self.assertEqual(before, read_user_config_raw())

    def test_successful_update_requests_native_gateway_restart_after_reply(self):
        from unittest.mock import Mock
        from hermes_cli.lifecycle import invoke_hook
        from kvpark import updates
        gateway = Mock()
        invoke_hook("pre_gateway_dispatch", gateway=gateway)
        async def command():
            loop = asyncio.get_running_loop()
            with patch.object(loop, "call_later", wraps=loop.call_later) as later:
                result = await self.gateway_command("update", command="kvpark")
                self.assertIn("restart shortly", result)
                gateway.request_restart.assert_not_called()
                callback = next(call.args[1] for call in later.call_args_list if call.args[0] == 2)
                callback()
                gateway.request_restart.assert_called_once_with(detached=False, via_service=True)
        with patch.object(updates, "apply", return_value=dict(version="0.2.0a2", note="Updated.")), \
                patch("importlib.metadata.version", return_value="0.2.0a2"), \
                patch("kvpark.__version__", "0.2.0a1"), \
                patch("gateway.restart.is_gateway_supervisor_process", return_value=True):
            asyncio.run(command())

    def test_failed_update_never_requests_gateway_restart(self):
        from unittest.mock import Mock
        from hermes_cli.lifecycle import invoke_hook
        from kvpark import updates
        gateway = Mock()
        invoke_hook("pre_gateway_dispatch", gateway=gateway)
        with patch.object(updates, "apply", side_effect=RuntimeError("rolled_back")):
            result = self.command_text("update")
        self.assertIn("rolled_back", result)
        gateway.request_restart.assert_not_called()

    def test_discord_park_before_first_turn_explains_how_to_load_state(self):
        with patch.object(self.archive, "park") as park:
            result = asyncio.run(self.gateway_command("save"))
        self.assertIn("Send a normal message", result)
        park.assert_not_called()

    def test_discord_park_uses_hook_thread_without_a_bound_session_id(self):
        adapter = self.command.__self__
        self.archive.current_key = adapter.key("discord-session")
        self.archive.current_thread = adapter.namespace + "discord:acceptance"
        self.archive.publish_status()
        with patch.dict(os.environ, {"HERMES_SESSION_ID": "unrelated-process-session"}):
            with patch.object(self.archive, "park", return_value={"tokens": 123, "note": "Saved."}) as park:
                result = asyncio.run(self.gateway_command("save"))
        self.assertIn("Parked 123 tokens", result)
        self.assertEqual(park.call_args.kwargs["key"], adapter.key("discord-session"))

    def test_discord_park_never_selects_another_threads_resident_state(self):
        adapter = self.command.__self__
        self.archive.current_key = adapter.key("someone-else")
        self.archive.current_thread = adapter.namespace + "discord:someone-else"
        self.archive.publish_status()
        with patch.object(self.archive, "park") as park:
            result = asyncio.run(self.gateway_command("save"))
        self.assertIn("Send a normal message", result)
        park.assert_not_called()

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
        from kvpark.hermes_service import Service
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


@unittest.skipUnless(os.environ.get("KVPARK_TEST_SERVER") and os.environ.get("KVPARK_TEST_MODEL"), "optional real-model startup test")
class RealStartupTests(unittest.TestCase):
    def test_hermes_discovery_launches_services_and_native_park_delete_work(self):
        from kvpark.hermes_service import Service
        from kvpark.cli import request
        from hermes_cli.plugins import get_plugin_manager, get_plugin_command_handler
        from hermes_cli.middleware import apply_llm_request_middleware
        from agent.subagent_lifecycle import bind_subagent_parent
        from gateway.session_context import set_session_vars, clear_session_vars
        from openai import OpenAI

        with tempfile.TemporaryDirectory(prefix="kvpark-hermes-real-") as tmp:
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
                          server=os.environ["KVPARK_TEST_SERVER"], model=os.environ["KVPARK_TEST_MODEL"],
                          backend_args=["--device", "none", "--n-gpu-layers", "0", "--ctx-size", "2048",
                                        "--threads", "4", "--threads-batch", "4", "--cache-ram", "0",
                                        "--ctx-checkpoints", "8", "--checkpoint-min-step", "128", "--jinja"])
            (home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["kvpark"],
                "entries": {"kvpark": {"settings": {"service": config}}}}}))
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
                    command = get_plugin_command_handler("kvpark")
                    # No explicit start command: discovery itself must launch.
                    deadline = time.monotonic() + 120
                    while time.monotonic() < deadline:
                        try:
                            from kvpark.network import proxy_record
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
                        self.assertIn("Parked", asyncio.run(command("save")))
                        self.assertEqual(request(url, "status")["archived"], 1)
                        self.assertIn("Deleted saved snapshot", asyncio.run(command("delete")))
                        self.assertEqual(request(url, "status")["archived"], 0)
                    finally:
                        clear_session_vars(tokens)
                    manager.unload()
                    self.assertTrue(all(child.poll() is None for child in children), "cleanup services must survive closing Hermes")
                    manager.discover_and_load(force=True)
                    get_plugin_command_handler("kvpark").__self__.service.worker.join(10)
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
