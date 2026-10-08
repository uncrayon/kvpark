"""OpenCode integration: config merge and removal, the plugin's conversation identity, the proxy's
automatic saving around background work, and the CLI commands."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from kvpark import cli, opencode
from kvpark import proxy as p
from kvpark.hermes_service import defaults

STATUS = {"service": "kvpark", "version": "test", "control_version": 4, "upstream_url": "http://127.0.0.1:18731",
          "capabilities": {"backend": "llama.cpp", "cache_mode": "native", "disk_snapshots": True}}


class Isolated(unittest.TestCase):
    """OpenCode's and kvpark's directories inside a temporary home."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        env = patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.root / "config"),
                                      "XDG_DATA_HOME": str(self.root / "data"), "KVPARK_API_KEY": ""})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("KVPARK_ARCHIVE_DIR", None)
        self.dir = self.root / "config" / "opencode"
        # Never run the user's OpenCode: doctor would start it with their real state and cache.
        found = patch.object(opencode, "find_opencode", return_value=None)
        found.start()
        self.addCleanup(found.stop)

    def connected(self, *, served=("local",), context=32768, status=STATUS, model="local"):
        """The proxy, its models and the server's context, as setup sees them."""
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(opencode, "_connection", return_value=("http://127.0.0.1:18740", model, "Qwen3-4B")))
        stack.enter_context(patch("kvpark.cli.request", return_value=status))
        stack.enter_context(patch.object(opencode, "served_models", return_value=list(served)))
        stack.enter_context(patch.object(opencode, "server_context", return_value=context))
        return stack

    def setup_opencode(self, connection=None, **kwargs):
        with self.connected(**(connection or {})):
            return opencode.setup(**kwargs)

    def config(self, name="opencode.json"):
        return json.loads((self.dir / name).read_text())

    def backups(self):
        return sorted(self.dir.glob("*.kvpark-*.bak"))

    def run_cli(self, *argv):
        output = io.StringIO()
        with patch.object(sys, "argv", ["kvpark", *argv]), contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(output):
            try:
                cli.main()
                code = 0
            except SystemExit as exc:
                code = exc.code
        return code, output.getvalue()


class ConfigTests(Isolated):
    def test_new_config_gets_the_provider_and_the_plugin(self):
        lines = self.setup_opencode()
        config = self.config()
        self.assertEqual(config["$schema"], opencode.SCHEMA)
        provider = config["provider"]["kvpark"]
        self.assertEqual(provider["npm"], "@ai-sdk/openai-compatible")
        self.assertEqual(provider["options"], {"baseURL": "http://127.0.0.1:18740/v1", "apiKey": "{env:KVPARK_API_KEY}"})
        self.assertEqual(provider["models"]["local"]["limit"], {"context": 32768, "output": 8192})
        self.assertNotIn("model", config)  # choosing the default model stays the user's call
        plugin = self.dir / "plugins" / "kvpark.js"
        self.assertEqual(plugin.read_bytes(), opencode.plugin_source().read_bytes())
        self.assertFalse(self.backups())
        self.assertIn("opencode -m kvpark/local", "\n".join(lines))

    def test_existing_keys_survive_and_the_old_file_is_backed_up(self):
        self.dir.mkdir(parents=True)
        original = {"$schema": opencode.SCHEMA, "model": "anthropic/claude-sonnet-4-5", "plugin": ["opencode-helicone-session"],
                    "provider": {"ollama": {"npm": "@ai-sdk/openai-compatible", "options": {"baseURL": "http://127.0.0.1:11434/v1"}},
                                 "kvpark": {"options": {"headers": {"X-Mine": "1"}, "baseURL": "http://127.0.0.1:9999/v1"},
                                            "models": {"old": {"name": "kept"}, "local": {"options": {"temperature": 0.2}}}}}}
        text = json.dumps(original, indent=4)
        (self.dir / "opencode.json").write_text(text)
        os.chmod(self.dir / "opencode.json", 0o600)
        self.setup_opencode()
        config = self.config()
        self.assertEqual(config["model"], "anthropic/claude-sonnet-4-5")
        self.assertEqual(config["plugin"], ["opencode-helicone-session"])
        self.assertEqual(config["provider"]["ollama"], original["provider"]["ollama"])
        kvpark = config["provider"]["kvpark"]
        self.assertEqual(kvpark["options"]["headers"], {"X-Mine": "1"})
        self.assertEqual(kvpark["options"]["baseURL"], "http://127.0.0.1:18740/v1")
        self.assertEqual(kvpark["models"]["old"], {"name": "kept"})
        self.assertEqual(kvpark["models"]["local"]["options"], {"temperature": 0.2})
        self.assertEqual(kvpark["models"]["local"]["limit"]["context"], 32768)
        [backup] = self.backups()
        self.assertEqual(backup.read_text(), text)
        self.assertEqual((self.dir / "opencode.json").stat().st_mode & 0o777, 0o600)

    def test_setup_is_idempotent(self):
        self.setup_opencode()
        before = (self.dir / "opencode.json").read_bytes()
        lines = self.setup_opencode()
        self.assertEqual((self.dir / "opencode.json").read_bytes(), before)
        self.assertFalse(self.backups())
        self.assertIn("already current", lines[0])
        self.assertTrue(any("plugins/kvpark.js (already current)" in line for line in lines))

    def test_commented_jsonc_is_left_alone_and_a_json_file_is_added_beside_it(self):
        self.dir.mkdir(parents=True)
        jsonc = '{\n  // my settings\n  "theme": "opencode",\n}\n'
        (self.dir / "opencode.jsonc").write_text(jsonc)
        self.setup_opencode()
        self.assertEqual((self.dir / "opencode.jsonc").read_text(), jsonc)
        self.assertIn("kvpark", self.config()["provider"])

    def test_plain_jsonc_is_edited_in_place(self):
        self.dir.mkdir(parents=True)
        (self.dir / "opencode.jsonc").write_text('{"$schema": "https://opencode.ai/config.json"}')
        self.setup_opencode()
        self.assertIn("kvpark", self.config("opencode.jsonc")["provider"])
        self.assertFalse((self.dir / "opencode.json").exists())

    def test_broken_config_is_refused_without_changes(self):
        self.dir.mkdir(parents=True)
        (self.dir / "opencode.json").write_text('{"provider": {')
        with self.assertRaisesRegex(ValueError, "not plain JSON"):
            self.setup_opencode()
        self.assertEqual((self.dir / "opencode.json").read_text(), '{"provider": {')
        self.assertFalse((self.dir / "plugins").exists())

    def test_a_foreign_plugin_named_kvpark_is_not_overwritten(self):
        (self.dir / "plugins").mkdir(parents=True)
        (self.dir / "plugins" / "kvpark.js").write_text("export const Mine = async () => ({})\n")
        with self.assertRaisesRegex(ValueError, "not kvpark's plugin"):
            self.setup_opencode()
        self.assertIn("Mine", (self.dir / "plugins" / "kvpark.js").read_text())
        self.assertFalse((self.dir / "opencode.json").exists())

    def test_conflicting_or_missing_targets_are_refused_before_anything_starts(self):
        with patch.object(opencode, "_connection") as connection:
            with self.assertRaisesRegex(ValueError, "not both"):
                opencode.setup(server="/bin/llama-server", model="m.gguf", url="http://127.0.0.1:8080")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                opencode.setup(project=str(self.root / "missing"))
        connection.assert_not_called()

    def test_symlinked_config_stays_a_link(self):
        real = self.root / "dotfiles" / "opencode.json"
        real.parent.mkdir()
        real.write_text('{"theme": "system"}')
        self.dir.mkdir(parents=True)
        (self.dir / "opencode.json").symlink_to(real)
        self.setup_opencode()
        self.assertTrue((self.dir / "opencode.json").is_symlink())
        self.assertIn("kvpark", json.loads(real.read_text())["provider"])

    def test_project_setup_writes_the_project_opencode_directory(self):
        project = self.root / "repo"
        project.mkdir()
        self.setup_opencode(project=str(project))
        self.assertIn("kvpark", json.loads((project / ".opencode" / "opencode.json").read_text())["provider"])
        self.assertTrue((project / ".opencode" / "plugins" / "kvpark.js").exists())
        self.assertFalse(self.dir.exists())

    def test_context_comes_from_the_server_or_the_option(self):
        with self.connected(context=None), self.assertRaisesRegex(ValueError, "--context"):
            opencode.setup()
        with self.connected(context=None):
            opencode.setup(context=131072)
        self.assertEqual(self.config()["provider"]["kvpark"]["models"]["local"]["limit"], {"context": 131072, "output": 32000})

    def test_a_model_the_proxy_does_not_serve_is_refused(self):
        with self.connected(served=("other",)), self.assertRaisesRegex(ValueError, "does not serve local"):
            opencode.setup()
        self.assertFalse(self.dir.exists())

    def test_url_mode_takes_the_only_served_model(self):
        with patch("kvpark.cli.request", return_value=STATUS), \
                patch.object(opencode, "served_models", return_value=["qwen38"]), \
                patch.object(opencode, "server_context", return_value=65536):
            lines = opencode.setup(url="http://localhost:8080/v1")
        self.assertIn("Pass --url http://127.0.0.1:8080 to kvpark status", "\n".join(lines))
        provider = self.config()["provider"]["kvpark"]
        self.assertEqual(provider["options"]["baseURL"], "http://127.0.0.1:8080/v1")
        self.assertEqual(list(provider["models"]), ["qwen38"])
        with self.assertRaisesRegex(ValueError, "local kvpark proxy"):
            opencode.setup(url="http://example.com:8080")

    def test_not_a_kvpark_proxy_is_refused(self):
        with self.connected(status={"service": "something-else"}), self.assertRaisesRegex(RuntimeError, "not a kvpark proxy"):
            opencode.setup()


    def test_schema_is_added_only_to_a_new_file(self):
        self.dir.mkdir(parents=True)
        (self.dir / "opencode.json").write_text('{"theme": "system"}')
        self.setup_opencode()
        self.assertEqual(list(self.config()), ["theme", "provider"])

    def test_a_new_model_replaces_the_one_setup_wrote_but_not_the_users(self):
        self.setup_opencode()
        config = self.config()
        config["model"] = "kvpark/local"
        config["provider"]["kvpark"]["models"]["mine"] = {"name": "My draft model"}
        (self.dir / "opencode.json").write_text(json.dumps(config))
        lines = self.setup_opencode({"model": "qwen38", "served": ("qwen38",)})
        config = self.config()
        self.assertEqual(list(config["provider"]["kvpark"]["models"]), ["mine", "qwen38"])
        self.assertEqual(config["model"], "kvpark/qwen38")
        self.assertIn("replaced local", "\n".join(lines))
        # A model the proxy still serves stays, whoever wrote it.
        self.setup_opencode({"model": "other", "served": ("qwen38", "other")})
        self.assertEqual(list(self.config()["provider"]["kvpark"]["models"]), ["mine", "qwen38", "other"])

    def test_contexts_opencode_cannot_use_are_refused(self):
        with self.connected(context=8192), self.assertRaisesRegex(ValueError, "at least 16,384"):
            opencode.setup()
        with self.connected(context=32768), self.assertRaisesRegex(ValueError, "larger than the 32,768"):
            opencode.setup(context=65536)
        with self.connected(context=None), self.assertRaisesRegex(ValueError, "at least 16,384"):
            opencode.setup(context=8192)
        self.assertFalse(self.dir.exists())
        with patch("kvpark.setup.connect_managed") as connect:
            with self.assertRaisesRegex(ValueError, "at least 16,384"):
                opencode.setup(server="/bin/llama-server", model="m.gguf", backend_args="--ctx-size 8192 --jinja")
            with self.assertRaisesRegex(ValueError, "at least 16,384"):
                opencode.setup(server="/bin/llama-server", model="m.gguf", backend_args="-c 4096")
        connect.assert_not_called()
        self.assertEqual(opencode.ctx_size(["--jinja", "--ctx-size=65536"]), 65536)
        self.assertIsNone(opencode.ctx_size(["--jinja"]))

    def test_the_temporary_file_is_private_until_it_is_complete(self):
        self.dir.mkdir(parents=True)
        path = self.dir / "opencode.json"
        path.write_text('{"provider": {"anthropic": {"options": {"apiKey": "sk-test"}}}}')
        os.chmod(path, 0o644)
        seen = []
        chmod = os.chmod
        def spy(target, mode, **kwargs):
            if str(target).endswith("opencode.json.kvpark-tmp"):
                seen.append((os.stat(target).st_mode & 0o777, json.loads(Path(target).read_text())["provider"]["anthropic"]))
            return chmod(target, mode, **kwargs)
        with patch.object(opencode.os, "chmod", spy):
            self.setup_opencode()
        self.assertEqual(seen, [(0o600, {"options": {"apiKey": "sk-test"}})])
        self.assertEqual(path.stat().st_mode & 0o777, 0o644)
        before = path.read_text()
        with patch.object(opencode.os, "replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
            opencode.write(path, {"theme": "x"}, before)
        self.assertEqual(path.read_text(), before)
        self.assertFalse(list(self.dir.glob("*.kvpark-tmp")))

    def test_provider_lists_that_hide_kvpark_are_reported(self):
        self.dir.mkdir(parents=True)
        (self.dir / "opencode.json").write_text('{"enabled_providers": ["anthropic"]}')
        lines = self.setup_opencode()
        self.assertIn("enabled_providers without kvpark", "\n".join(lines))
        project = self.root / "repo"
        project.mkdir()
        (project / "opencode.json").write_text('{"disabled_providers": ["kvpark"]}')
        self.assertIn("disabled_providers", opencode.hidden_by(opencode.filter_dirs(project)))
        (self.dir / "opencode.json").write_text('{"enabled_providers": ["anthropic", "kvpark"]}')
        self.assertIsNone(opencode.hidden_by(opencode.filter_dirs()))


class UninstallTests(Isolated):
    def test_preview_then_remove_only_what_kvpark_added(self):
        self.dir.mkdir(parents=True)
        (self.dir / "opencode.json").write_text(json.dumps({"theme": "system", "provider": {"ollama": {"name": "Ollama"}}}))
        self.setup_opencode()
        config = self.config()
        config["model"] = "kvpark/local"
        (self.dir / "opencode.json").write_text(json.dumps(config))
        mine = self.dir / "plugins" / "mine.js"
        mine.write_text("export const Mine = async () => ({})\n")
        before = (self.dir / "opencode.json").read_text()
        preview = opencode.uninstall()
        self.assertIn("Preview", preview[0])
        self.assertEqual((self.dir / "opencode.json").read_text(), before)
        self.assertTrue((self.dir / "plugins" / "kvpark.js").exists())
        opencode.uninstall(confirm=True)
        self.assertEqual(self.config(), {"theme": "system", "provider": {"ollama": {"name": "Ollama"}}})
        self.assertFalse((self.dir / "plugins" / "kvpark.js").exists())
        self.assertTrue(mine.exists())
        self.assertTrue(self.backups())
        self.assertIn("nothing changed", opencode.uninstall(confirm=True)[0])

    def test_what_setup_created_is_removed_again(self):
        self.setup_opencode()
        opencode.uninstall(confirm=True)
        self.assertFalse((self.dir / "opencode.json").exists())
        self.assertFalse((self.dir / "plugins").exists())
        [backup] = self.backups()
        self.assertIn("kvpark", json.loads(backup.read_text())["provider"])
        original = {"theme": "system"}
        (self.dir / "opencode.json").write_text(json.dumps(original))
        (self.dir / "plugins").mkdir()
        (self.dir / "plugins" / "mine.js").write_text("export const Mine = async () => ({})\n")
        self.setup_opencode()
        opencode.uninstall(confirm=True)
        self.assertEqual(self.config(), original)
        self.assertTrue((self.dir / "plugins" / "mine.js").exists())

    def test_a_failed_write_restores_every_file(self):
        self.dir.mkdir(parents=True)
        (self.dir / "opencode.json").write_text('{"theme": "system"}')
        self.setup_opencode()
        provider = self.config()["provider"]
        (self.dir / "config.json").write_text(json.dumps({"theme": "system", "provider": provider}))
        before = {name: (self.dir / name).read_text() for name in ("config.json", "opencode.json")}
        write = opencode.write
        calls = []
        def failing(path, document, original):
            calls.append(path.name)
            if len(calls) == 2:
                raise OSError("disk full")
            return write(path, document, original)
        with patch.object(opencode, "write", failing), self.assertRaises(OSError):
            opencode.uninstall(confirm=True)
        self.assertEqual(calls, ["config.json", "opencode.json"])
        self.assertEqual({name: (self.dir / name).read_text() for name in before}, before)
        self.assertTrue((self.dir / "plugins" / "kvpark.js").exists())

    def test_commented_files_are_reported_not_rewritten(self):
        self.dir.mkdir(parents=True)
        text = '{\n  // added by hand\n  "provider": {"kvpark": {}}\n}\n'
        (self.dir / "opencode.jsonc").write_text(text)
        lines = opencode.uninstall(confirm=True)
        self.assertIn("by hand", "\n".join(lines))
        self.assertEqual((self.dir / "opencode.jsonc").read_text(), text)


class DoctorTests(Isolated):
    def doctor(self, status=STATUS, served=("local",), context=32768):
        with patch("kvpark.cli.request", return_value=status) if not isinstance(status, Exception) else \
                patch("kvpark.cli.request", side_effect=status), \
                patch.object(opencode, "served_models", return_value=list(served)), \
                patch.object(opencode, "server_context", return_value=context):
            return opencode.doctor()

    def test_a_working_setup_passes(self):
        self.setup_opencode()
        lines, ok = self.doctor()
        self.assertTrue(ok, lines)
        self.assertIn("disk park/resume on", "\n".join(lines))

    def test_missing_config_outdated_plugin_and_stopped_proxy_fail(self):
        lines, ok = self.doctor()
        self.assertFalse(ok)
        self.assertIn("kvpark setup --opencode", lines[-1])
        self.setup_opencode()
        (self.dir / "plugins" / "kvpark.js").write_text(opencode.MARKER + " 0\n")
        lines, ok = self.doctor()
        self.assertFalse(ok)
        self.assertIn("another kvpark version", "\n".join(lines))
        self.setup_opencode()
        from urllib.error import URLError
        lines, ok = self.doctor(status=URLError("connection refused"))
        self.assertFalse(ok)
        self.assertIn("kvpark start", lines[-1])

    def test_an_edited_plugin_of_this_version_passes(self):
        self.setup_opencode()
        plugin = self.dir / "plugins" / "kvpark.js"
        plugin.write_text(plugin.read_text() + "// my note\n")
        lines, ok = self.doctor()
        self.assertTrue(ok, lines)

    def test_models_the_user_added_only_warn_and_a_default_kvpark_model_must_exist(self):
        self.setup_opencode()
        config = self.config()
        config["provider"]["kvpark"]["models"]["mine"] = {"name": "My draft model"}
        (self.dir / "opencode.json").write_text(json.dumps(config))
        lines, ok = self.doctor()
        self.assertTrue(ok, lines)
        self.assertIn("does not serve mine, which you added", "\n".join(lines))
        config["small_model"] = "kvpark/gone"
        (self.dir / "opencode.json").write_text(json.dumps(config))
        lines, ok = self.doctor()
        self.assertFalse(ok)
        self.assertIn("small_model is kvpark/gone", "\n".join(lines))

    def test_requests_without_a_conversation_point_at_an_unloaded_plugin(self):
        self.setup_opencode()
        recent = [{"action": "completed", "key": None}, {"action": "saved", "key": "k-1"}]
        lines, ok = self.doctor(status={**STATUS, "recent": recent})
        self.assertTrue(ok)
        self.assertIn("plugin is not loaded", lines[-1])
        lines, ok = self.doctor(status={**STATUS, "recent": recent + [{"action": "completed", "key": "k-1"}]})
        self.assertNotIn("plugin is not loaded", "\n".join(lines))

    def test_a_provider_list_that_hides_kvpark_fails(self):
        self.setup_opencode()
        config = self.config()
        config["disabled_providers"] = ["kvpark"]
        (self.dir / "opencode.json").write_text(json.dumps(config))
        lines, ok = self.doctor()
        self.assertFalse(ok)
        self.assertIn("disabled_providers", "\n".join(lines))

    def test_a_smaller_server_context_or_a_missing_model_fails(self):
        self.setup_opencode()
        lines, ok = self.doctor(context=8192)
        self.assertFalse(ok)
        self.assertIn("runs 8,192", "\n".join(lines))
        lines, ok = self.doctor(served=("other",))
        self.assertFalse(ok)
        self.assertIn("does not serve local", "\n".join(lines))


class CliTests(Isolated):
    def test_setup_doctor_and_uninstall_commands(self):
        with self.connected():
            code, output = self.run_cli("setup", "--opencode")
        self.assertEqual(code, 0, output)
        self.assertIn("provider kvpark", output)
        with self.connected():
            code, output = self.run_cli("doctor", "--opencode")
        self.assertEqual(code, 0, output)
        code, output = self.run_cli("uninstall", "--opencode", "--confirm")
        self.assertEqual(code, 0, output)
        self.assertFalse((self.dir / "opencode.json").exists())  # setup created it; a backup remains
        code, output = self.run_cli("doctor", "--opencode")
        self.assertEqual(code, 1, output)

    def test_opencode_commands_ignore_a_saved_standalone_connection(self):
        from kvpark.runtime import directory
        directory().mkdir(parents=True)
        (directory() / "setup.json").write_text(json.dumps({**defaults(), "backend": "ollama", "cache_mode": "routing",
                                                            "model": "gemma:latest", "upstream_url": "http://127.0.0.1:11434"}))
        self.setup_opencode()
        code, output = self.run_cli("uninstall", "--opencode", "--confirm")
        self.assertEqual(code, 0, output)
        self.assertIn("Removed", output)

    def test_opencode_options_need_opencode(self):
        code, output = self.run_cli("setup", "--server", "/bin/true")
        self.assertEqual(code, 2)
        self.assertIn("only with --opencode", output)
        code, output = self.run_cli("uninstall", "--opencode", "--hermes")
        self.assertEqual(code, 2)

    def test_doctor_options_that_do_not_combine(self):
        for argv in (("doctor", "--opencode", "--hermes"), ("doctor", "--opencode", "--url", "http://127.0.0.1:8080"),
                     ("doctor", "--project", str(self.root))):
            code, output = self.run_cli(*argv)
            self.assertEqual(code, 2, argv)

    def test_without_a_connection_setup_explains_the_choices(self):
        code, output = self.run_cli("setup", "--opencode")
        self.assertEqual(code, 1)
        self.assertIn("--server", output)
        self.assertIn("--url", output)


class StopTests(Isolated):
    """kvpark stop: drain and save through the proxy, stop both processes, keep the setup."""

    def save_connection(self, **values):
        from kvpark.runtime import directory
        directory().mkdir(parents=True, exist_ok=True)
        config = {**defaults(), "backend": "ollama", "cache_mode": "routing", "model": "gemma:latest",
                  "upstream_url": "http://127.0.0.1:11434", **values}
        (directory() / "setup.json").write_text(json.dumps(config))
        return config

    def test_stop_drains_the_selected_connection_and_keeps_it(self):
        from kvpark import removal
        from kvpark.runtime import directory
        archive = self.root / "archive"
        self.save_connection(archive_dir=str(archive))
        record = {"pid": 4242, "base_url": "http://127.0.0.1:18740"}
        with patch.object(removal, "plan", return_value={"proxy": record, "backend": None}), \
                patch.object(removal, "stop", return_value={"proxy": record, "backend": None}) as stop, \
                patch("kvpark.cli.request", return_value={"resident_key": "k-1"}):
            code, output = self.run_cli("stop")
        self.assertEqual(code, 0, output)
        stop.assert_called_once_with(archive.resolve(), external=False, managed_backend=False)
        self.assertIn("proxy (PID 4242)", output)
        self.assertIn("saved first", output)
        self.assertTrue((directory() / "setup.json").exists())

    def test_stop_with_nothing_running_and_the_hermes_profile(self):
        from kvpark import removal
        with patch.object(removal, "plan", return_value={"proxy": None, "backend": None}), \
                patch.object(removal, "stop") as stop:
            code, output = self.run_cli("stop", "--archive-dir", str(self.root / "elsewhere"))
            self.assertEqual(code, 0, output)
            self.assertIn("Nothing to stop", output)
            profile = {**defaults(), "archive_dir": str(self.root / "hermes"), "connected": True}
            with patch("kvpark.setup.hermes_settings", return_value=profile):
                code, output = self.run_cli("stop", "--hermes")
            self.assertEqual(code, 0, output)
            missing = ModuleNotFoundError("No module named 'hermes_yaml'", name="hermes_yaml")
            with patch("kvpark.setup.hermes_settings", side_effect=missing):
                code, output = self.run_cli("stop", "--hermes")
            self.assertEqual(code, 1)
            self.assertIn("--archive-dir", output)
        stop.assert_not_called()
        code, output = self.run_cli("stop", "--hermes", "--archive-dir", str(self.root))
        self.assertEqual(code, 2)


class ManagedConnectionTests(Isolated):
    def test_server_and_model_become_the_standalone_connection(self):
        from kvpark import setup
        server, model = self.root / "llama-server", self.root / "model.gguf"
        server.write_text("")
        model.write_text("")
        with patch.object(setup.Service, "ensure", return_value={"service": "kvpark"}) as ensure:
            config = setup.connect_managed(str(server), str(model), opencode.BACKEND_ARGS)
            again = setup.connect_managed(str(server), str(model), opencode.BACKEND_ARGS)
        self.assertEqual(ensure.call_count, 2)
        saved = setup.saved_settings()
        self.assertEqual(saved, config)
        self.assertEqual(again["archive_dir"], config["archive_dir"])
        self.assertEqual((saved["backend"], saved["cache_mode"], saved["upstream_url"]), ("llama.cpp", "native", ""))
        self.assertEqual(saved["backend_args"], opencode.BACKEND_ARGS)
        self.assertEqual(Path(saved["archive_dir"]).parent, (self.root / "data" / "kvpark" / "connections").resolve())
        with patch.object(setup.Service, "ensure", return_value={"service": "kvpark"}):
            bigger = setup.connect_managed(str(server), str(model), ["--ctx-size", "65536", "--jinja"])
        self.assertNotEqual(bigger["archive_dir"], config["archive_dir"])

    def test_a_failed_start_saves_nothing(self):
        from kvpark import setup
        server, model = self.root / "llama-server", self.root / "model.gguf"
        server.write_text("")
        model.write_text("")
        with patch.object(setup.Service, "ensure", side_effect=RuntimeError("service exited")):
            with self.assertRaises(RuntimeError):
                setup.connect_managed(str(server), str(model))
        self.assertIsNone(setup.saved_settings())


@unittest.skipUnless(shutil.which("node"), "the plugin test runs it with Node.js")
class PluginTests(unittest.TestCase):
    """Run the shipped plugin's chat.params hook with a stand-in OpenCode client."""

    SCRIPT = """
const { KvparkPlugin } = await import(process.argv[1]);
const lookups = [];
const sessions = { ses_main: { parentID: undefined }, ses_child: { parentID: "ses_main" } };
const client = { session: { get: async ({ path }) => {
  lookups.push(path.id);
  if (path.id === "ses_broken") throw new Error("server unavailable");
  return { data: sessions[path.id] };
} } };
const hooks = await KvparkPlugin({ client, directory: "/work" });
async function call(providerID, agent, sessionID) {
  const output = { temperature: 0, options: { reasoningEffort: "low" } };
  await hooks["chat.params"]({ sessionID, agent, model: { providerID }, provider: {}, message: {} }, output);
  return output.options;
}
const results = {
  main: await call("kvpark", "build", "ses_main"),
  again: await call("kvpark", "plan", "ses_main"),
  title: await call("kvpark", "title", "ses_main"),
  compaction: await call("kvpark", "compaction", "ses_main"),
  child: await call("kvpark", "general", "ses_child"),
  unknown: await call("kvpark", "build", "ses_gone"),
  broken: await call("kvpark", "build", "ses_broken"),
  other: await call("anthropic", "build", "ses_main"),
};
console.log(JSON.stringify({ results, lookups }));
"""

    def test_roles_and_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = Path(tmp) / "kvpark.mjs"
            shutil.copyfile(opencode.plugin_source(), module)
            result = subprocess.run(["node", "--input-type=module", "-e", self.SCRIPT, str(module)],
                                    capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        results = data["results"]
        foreground = {"reasoningEffort": "low", "slot_archive_role": "foreground",
                      "slot_archive_key": "opencode:ses_main", "slot_archive_autosave": True}
        self.assertEqual(results["main"], foreground)
        self.assertEqual(results["again"], foreground)
        for name in ("title", "compaction", "child", "unknown", "broken"):
            self.assertEqual(results[name], {"reasoningEffort": "low", "slot_archive_role": "background"}, name)
        self.assertEqual(results["other"], {"reasoningEffort": "low"})
        # One lookup per session; a failed lookup is not cached.
        self.assertEqual(data["lookups"].count("ses_main"), 1)


class AutosaveTests(unittest.TestCase):
    """The proxy saves an autosave conversation before background work takes the slot."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.multiple(p, ARCHIVE=Path(tmp.name), AUTO_SAVE=False, MIN_ARCHIVE_TOKENS=8192)
        env.start()
        self.addCleanup(env.stop)
        self.a = p.Archive()
        self.a.identity = "weights-a"
        self.a.runtime = lambda: None
        self.a.request = self.upstream
        self.tokens = 100
        self.calls = []
        self.prompt = {"slot_archive_role": "foreground", "messages": [{"role": "system", "content": "opencode"}]}

    def upstream(self, method, path, body=None, timeout=600):
        self.calls.append(path)
        if path == "/slots":
            return [dict(id=0, is_processing=False, n_prompt_tokens=self.tokens)]
        if "action=save" in path:
            for suffix in ("", ".resume"):
                (p.ARCHIVE / (body["filename"] + suffix)).write_bytes(b"state")
            return dict(n_saved=self.tokens)
        if "action=restore" in path:
            return dict(n_restored=self.tokens)
        return {}

    def resident(self, session, autosave=True):
        signature = p.prompt_signature(self.prompt)
        if autosave:
            signature["autosave"] = True
        self.a.current_key, self.a.current_prompt = p.key_for(session), signature

    def test_background_work_saves_an_autosave_conversation_and_it_resumes(self):
        self.resident("opencode:ses_a")
        key = self.a.current_key
        self.a.switch(None, role="background")  # a title request, for example
        self.assertIn(key, self.a.entries)
        self.assertEqual(self.a.entries[key]["tokens"], 100)  # far below the 8,192-token floor
        self.a.switch(key, incoming=p.prompt_signature(self.prompt), role="foreground")
        self.assertEqual(self.calls.count("/slots/0?action=restore"), 1)

    def test_without_autosave_background_work_still_discards_an_unparked_conversation(self):
        self.resident("hermes:session", autosave=False)
        self.a.switch(None, role="background")
        self.assertFalse(self.a.entries)

    def test_another_conversation_saves_it_only_above_the_token_floor(self):
        self.resident("opencode:ses_a")
        self.a.switch(p.key_for("opencode:ses_b"), role="foreground")
        self.assertFalse(self.a.entries)
        self.resident("opencode:ses_a")
        self.tokens = 9000
        self.a.switch(p.key_for("opencode:ses_b"), role="foreground")
        self.assertIn(p.key_for("opencode:ses_a"), self.a.entries)

    def incoming(self, system="opencode", autosave=True):
        signature = p.prompt_signature(dict(self.prompt, messages=[{"role": "system", "content": system},
                                                                   {"role": "user", "content": "new task"}]))
        if autosave:
            signature["autosave"] = True
        return signature

    def restored_files(self):
        return [body["filename"] for path, body in self.bodies if "action=restore" in path]

    def park_conversation(self, session, tokens, system="opencode"):
        self.tokens = tokens
        self.resident(session)
        self.a.current_prompt = dict(self.incoming(system), autosave=True)
        self.a.park(key=p.key_for(session))
        return self.a.entries[p.key_for(session)]["filename"]

    def track_bodies(self):
        self.bodies = []
        upstream = self.a.request
        def recording(method, path, body=None, timeout=600):
            self.bodies.append((path, body))
            return upstream(method, path, body, timeout)
        self.a.request = recording

    def test_a_new_conversation_after_background_work_starts_from_a_compatible_snapshot(self):
        self.track_bodies()
        small = self.park_conversation("opencode:ses_a", 9000)
        self.park_conversation("opencode:ses_c", 20000)
        self.resident("opencode:ses_c")
        self.a.switch(None, role="background")  # the new session's title request
        self.a.current_prompt = None  # as the relay records a completed keyless request
        self.a.switch(p.key_for("opencode:ses_b"), incoming=self.incoming(), role="foreground")
        self.assertEqual(self.restored_files(), [small])  # the smallest compatible snapshot
        event = self.a.events[0]
        self.assertEqual((event["action"], event["key"], event["source"]), ("seeded", p.key_for("opencode:ses_b"), p.key_for("opencode:ses_a")))
        self.assertNotIn(p.key_for("opencode:ses_b"), self.a.entries)

    def test_no_seed_when_the_slot_holds_the_prefix_or_nothing_matches(self):
        self.track_bodies()
        self.park_conversation("opencode:ses_a", 9000)
        self.resident("opencode:ses_c")  # same system prompt and tools, resident in the slot
        self.a.current_prompt = self.incoming()
        self.tokens = 100
        self.a.switch(p.key_for("opencode:ses_b"), incoming=self.incoming(), role="foreground")
        self.assertEqual(self.restored_files(), [])
        self.a.current_key = self.a.current_prompt = None  # after background work
        self.a.switch(p.key_for("opencode:ses_d"), incoming=self.incoming(system="another agent"), role="foreground")
        self.a.switch(p.key_for("opencode:ses_e"), incoming=self.incoming(autosave=False), role="foreground")
        self.assertEqual(self.restored_files(), [])

    def test_a_conversation_with_its_own_snapshot_resumes_it_instead(self):
        self.track_bodies()
        self.park_conversation("opencode:ses_a", 9000)
        own = self.park_conversation("opencode:ses_b", 12000)
        self.a.current_key = self.a.current_prompt = None
        self.a.switch(p.key_for("opencode:ses_b"), incoming=self.incoming(), role="foreground")
        self.assertEqual(self.restored_files(), [own])
        self.assertEqual(self.a.events[0]["action"], "restored")

    def complete(self, session, signature, tokens):
        """What the relay records after a successful completion."""
        self.tokens = tokens
        self.a.current_key, self.a.current_prompt = p.key_for(session), signature

    def saves(self):
        return [path for path in self.calls if "action=save" in path]

    def test_a_snapshot_from_another_day_is_replaced_on_the_next_switch(self):
        a, b = p.key_for("opencode:ses_a"), p.key_for("opencode:ses_b")
        self.complete("opencode:ses_a", self.incoming("Today is Monday"), 12000)
        self.a.switch(b, incoming=self.incoming("Today is Monday"), role="foreground")
        monday = self.a.entries[a]["filename"]
        self.complete("opencode:ses_b", self.incoming("Today is Monday"), 7400)
        self.a.switch(a, incoming=self.incoming("Today is Tuesday"), role="foreground")
        self.assertEqual(self.a.events[0]["action"], "restore-skipped")
        self.complete("opencode:ses_a", self.incoming("Today is Tuesday"), 12100)  # grew by only 100 tokens
        self.a.switch(b, incoming=self.incoming("Today is Tuesday"), role="foreground")
        self.assertNotEqual(self.a.entries[a]["filename"], monday)
        self.complete("opencode:ses_b", self.incoming("Today is Tuesday"), 7500)
        self.a.switch(a, incoming=self.incoming("Today is Tuesday"), role="foreground")
        self.assertEqual(self.a.events[0]["action"], "restored")

    def test_a_compacted_conversation_replaces_its_longer_snapshot(self):
        a = p.key_for("opencode:ses_a")
        history = [{"role": "system", "content": "opencode"}] + [{"role": "user", "content": f"turn {i}"} for i in range(3)]
        def signature(messages):
            return dict(p.prompt_signature(dict(self.prompt, messages=messages)), autosave=True)
        self.complete("opencode:ses_a", signature(history), 20000)
        self.a.switch(None, role="background")  # the compaction request
        self.assertEqual(self.a.entries[a]["tokens"], 20000)
        self.a.current_prompt = None
        compacted = [history[0], {"role": "user", "content": "summary"}, {"role": "user", "content": "turn 3"}]
        self.complete("opencode:ses_a", signature(compacted), 5000)  # below the token floor, too
        self.a.switch(p.key_for("opencode:ses_b"), incoming=self.incoming(), role="foreground")
        self.assertEqual(self.a.entries[a]["tokens"], 5000)
        # A conversation that only grew keeps the growth rule.
        self.complete("opencode:ses_a", signature(compacted + [{"role": "user", "content": "turn 4"}]), 5200)
        self.a.switch(p.key_for("opencode:ses_b"), incoming=self.incoming(), role="foreground")
        self.assertEqual(self.a.entries[a]["tokens"], 5000)

    def test_a_repeated_request_does_not_save_the_same_state_again(self):
        signature = self.incoming()
        for _ in range(3):  # the turn, a subagent's background request, the same turn again
            self.complete("opencode:ses_a", signature, 12000)
            self.a.switch(None, role="background")
            self.assertIn(p.key_for("opencode:ses_a"), self.a.displaced)  # `kvpark save` still answers
            self.a.current_key = self.a.current_prompt = None
            self.a.switch(p.key_for("opencode:ses_a"), incoming=signature, role="foreground")
        self.assertEqual(len(self.saves()), 1)
        self.assertEqual(self.calls.count("/slots/0?action=restore"), 3)

    def seed_after_background(self):
        self.track_bodies()
        source = self.park_conversation("opencode:ses_a", 9000)
        self.a.switch(None, role="background")  # nothing ran since the park: no second save
        self.a.current_key = self.a.current_prompt = None
        self.a.entries[p.key_for("opencode:ses_a")]["bytes"] = 4 * 1024**3
        incoming = dict(self.incoming(), prefix_bytes=30000)  # about 7,500 tokens
        self.a.switch(p.key_for("opencode:ses_b"), incoming=incoming, role="foreground")
        return source

    def test_seeding_is_skipped_where_prefill_was_measured_faster_than_restoring(self):
        self.a.rates = {"restore": 2 * 1024**3, "prefill": 10000}  # 2 s to restore, 0.75 s to prefill
        self.seed_after_background()
        self.assertEqual(self.restored_files(), [])
        event = self.a.events[0]
        self.assertEqual((event["action"], event["restore_seconds"], event["prefill_seconds"]), ("seed-skipped", 2.0, 0.75))

    def test_seeding_uses_measured_speeds_or_seeds_when_none_are_known(self):
        source = self.seed_after_background()
        self.assertEqual(self.restored_files(), [source])
        self.a.rates = {"restore": 8 * 1024**3, "prefill": 1250}  # 0.5 s to restore, 6 s to prefill
        self.a.current_key = self.a.current_prompt = None
        self.a.switch(p.key_for("opencode:ses_c"), incoming=dict(self.incoming(), prefix_bytes=30000), role="foreground")
        self.assertEqual(self.a.events[0]["action"], "seeded")

    def test_speeds_come_from_restores_and_the_servers_timings(self):
        metrics = p.ResponseMetrics(True, 0)
        metrics.feed(b'data: {"choices": [], "timings": {"cache_n": 7300, "prompt_n": 4700, "prompt_ms": 3760.0}}\n')
        result = metrics.finish()
        self.assertEqual(result["prefill_ms"], 3760.0)
        self.a.measured(result)
        self.assertAlmostEqual(self.a.rates["prefill"], 1250)
        self.a.measured({"processed_tokens": 30, "prefill_ms": 40})  # too few tokens to measure
        self.assertAlmostEqual(self.a.rates["prefill"], 1250)
        self.track_bodies()
        self.park_conversation("opencode:ses_a", 9000)
        meta = dict(self.a.entries[p.key_for("opencode:ses_a")], bytes=1024**3)
        with patch.object(p.time, "monotonic", side_effect=[0.0, 0.5]):
            self.a.load(meta)
        self.assertEqual(self.a.rates["restore"], 2 * 1024**3)

    def test_relay_strips_the_flag_and_honours_it_only_with_a_foreground_key(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import http.client
        import threading
        bodies = []

        class Upstream(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                data = b'{"choices": []}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        servers = [ThreadingHTTPServer(("127.0.0.1", 0), Upstream), ThreadingHTTPServer(("127.0.0.1", 0), p.Handler)]
        for server in servers:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
        switches = []
        self.a.switch = lambda key, thread, incoming, role: switches.append((key, incoming, role))
        with patch.multiple(p, UPSTREAM_PORT=servers[0].server_port), patch.object(p, "ARCHIVE_STATE", self.a, create=True), \
                patch.dict(os.environ, {"KVPARK_UPSTREAM_URL": f"http://127.0.0.1:{servers[0].server_port}"}):
            for role, key in (("foreground", "opencode:ses_a"), ("background", None)):
                body = dict(self.prompt, slot_archive_role=role, slot_archive_autosave=True)
                if key:
                    body["slot_archive_key"] = key
                conn = http.client.HTTPConnection("127.0.0.1", servers[1].server_port, timeout=5)
                conn.request("POST", "/v1/chat/completions", json.dumps(body))
                response = conn.getresponse()
                response.read()
                self.assertEqual(response.status, 200)
                conn.close()
            with self.a.lock:
                pass
        self.assertTrue(all(not any(name.startswith("slot_archive_") for name in body) for body in bodies), bodies)
        (key_a, incoming_a, _), (key_b, incoming_b, role_b) = switches
        self.assertEqual(key_a, p.key_for("opencode:ses_a"))
        self.assertTrue(incoming_a["autosave"])
        self.assertIsNone(key_b)
        self.assertNotIn("autosave", incoming_b)
        self.assertEqual(role_b, "background")


if __name__ == "__main__":
    unittest.main()
