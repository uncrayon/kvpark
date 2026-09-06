import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import psutil

from sloth_memory import removal
from sloth_memory.hermes_uninstall import restored_config
from sloth_memory.platforms import process_identity


class RestoreTests(unittest.TestCase):
    def profile(self):
        applied = dict(provider="custom", default="gemma", base_url="http://127.0.0.1:8080/v1", api_mode="chat_completions")
        return dict(model={**applied, "context_length": 12345}, agent={"max_turns": 8}, plugins=dict(
            enabled=["other", "sloth-memory"], entries={"sloth-memory": {"settings": {
                "service": {"connected": True}, "route_applied": applied,
                "route_backup": {"provider": "custom", "default": "gemma", "base_url": "http://127.0.0.1:11434/v1"}}}}))

    def test_restore_removes_added_fields_and_preserves_other_settings(self):
        original = self.profile()
        result, settings, restored = restored_config(original)
        self.assertTrue(restored)
        self.assertEqual(result["model"]["base_url"], "http://127.0.0.1:11434/v1")
        self.assertNotIn("api_mode", result["model"])
        self.assertEqual(result["model"]["context_length"], 12345)
        self.assertEqual(result["agent"], original["agent"])
        self.assertEqual(result["plugins"]["enabled"], ["other"])
        self.assertIn("sloth-memory", result["plugins"]["disabled"])
        self.assertFalse(settings["autostart"])
        self.assertTrue(settings["uninstalled"])
        self.assertEqual(original["model"]["base_url"], "http://127.0.0.1:8080/v1")
        again, _, restored = restored_config(result)
        self.assertFalse(restored)
        self.assertEqual(result, again)

    def test_changed_route_is_preserved_and_missing_backup_fails_closed(self):
        profile = self.profile()
        profile["plugins"]["entries"]["sloth-memory"]["settings"].pop("route_backup")
        with self.assertRaisesRegex(ValueError, "no original"):
            restored_config(profile)
        profile["model"]["base_url"] = "http://127.0.0.1:9999/v1"
        result, _, restored = restored_config(profile)
        self.assertFalse(restored)
        self.assertEqual(result["model"], profile["model"])

    def test_user_edits_on_proxy_route_are_not_overwritten(self):
        profile = self.profile()
        profile["model"]["default"] = "another-model"
        with self.assertRaisesRegex(ValueError, "routing changed"):
            restored_config(profile)

    def test_legacy_model_string_is_preserved_when_already_disconnected(self):
        profile = self.profile()
        profile["model"] = "provider/original-model"
        result, _, restored = restored_config(profile)
        self.assertFalse(restored)
        self.assertEqual(result["model"], profile["model"])


class RemovalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = Path(self.tmp.name).resolve()

    def test_preview_does_not_create_archive_and_external_never_inspects_processes(self):
        missing = self.archive / "missing"
        self.assertIsNone(removal.plan(missing)["proxy"])
        self.assertFalse(missing.exists())
        with patch.object(removal, "owned_process") as owned:
            removal.stop(self.archive, external=True)
            owned.assert_not_called()

    def test_live_unrelated_pid_and_reused_pid_are_not_stopped(self):
        record = process_identity(os.getpid())
        path = self.archive / "proxy.json"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            removal.plan(self.archive)
        record["created_at"] -= 1
        path.write_text(json.dumps(record))
        self.assertIsNone(removal.plan(self.archive)["proxy"])

    def test_real_owned_backend_stops_and_unrelated_process_survives(self):
        script = self.archive / "fake_backend.py"
        script.write_text("import pathlib, sys, time; pathlib.Path(sys.argv[1]).touch(); time.sleep(60)")
        ready = self.archive / "ready"
        argv = [sys.executable, str(script), str(ready), "--slot-save-path", str(self.archive)]
        children = [subprocess.Popen(argv), subprocess.Popen([sys.executable, str(script), str(self.archive / "other-ready")])]
        def cleanup():
            for child in children:
                if child.poll() is None:
                    child.terminate()
                child.wait(10)
        self.addCleanup(cleanup)
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            self.assertIsNone(children[0].poll())
            time.sleep(.01)
        self.assertTrue(ready.exists(), "fake backend did not start")
        # macOS framework Python can re-exec with a different argv[0]. This
        # fixture records the ready Python stand-in, not its transient launcher.
        actual = psutil.Process(children[0].pid).cmdline()
        self.assertEqual(actual[1:], argv[1:])
        record = {**process_identity(children[0].pid), "argv": actual}
        (self.archive / "runtime.json").write_text(json.dumps(record))
        (self.archive / "snapshot.bin").write_bytes(b"keep")
        # The externally launched upstream mode does not own this backend.
        self.assertIsNone(removal.plan(self.archive, managed_backend=False)["backend"])
        removal.stop(self.archive)
        self.assertIsNotNone(children[0].poll())
        self.assertIsNone(children[1].poll())
        self.assertEqual((self.archive / "snapshot.bin").read_bytes(), b"keep")
        self.assertIsNone(removal.stop(self.archive)["backend"])

    def test_failed_drain_never_disconnects_or_stops_processes(self):
        record = dict(pid=123, base_url="http://127.0.0.1:8080")
        result = dict(proxy=record, backend=None)
        status = dict(service="sloth-memory", control_version=4, archive_dir=str(self.archive))
        with patch.object(removal, "plan", return_value=result), \
                patch.object(removal, "request", side_effect=[status, RuntimeError("busy")]), \
                patch.object(removal, "stop_proxy") as stop, \
                patch.object(removal, "process_matches", return_value=True):
            from unittest.mock import Mock
            disconnect = Mock()
            with self.assertRaisesRegex(RuntimeError, "busy"):
                removal.stop(self.archive, disconnect=disconnect)
            disconnect.assert_not_called()
            stop.assert_not_called()

    def test_config_failure_reopens_proxy_without_stopping_it(self):
        record = dict(pid=123, base_url="http://127.0.0.1:8080")
        status = dict(service="sloth-memory", control_version=4, archive_dir=str(self.archive))
        def disconnect():
            raise RuntimeError("configuration changed")
        with patch.object(removal, "plan", return_value=dict(proxy=record, backend=None)), \
                patch.object(removal, "request", side_effect=[status, {"ok": True}, {"ok": True}]) as request, \
                patch.object(removal, "stop_proxy") as stop, \
                patch.object(removal, "process_matches", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "configuration changed"):
                removal.stop(self.archive, disconnect=disconnect)
            self.assertEqual([call.args[1] for call in request.call_args_list], ["status", "prepare-update", "cancel-update"])
            stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
