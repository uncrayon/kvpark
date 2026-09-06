import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch

from sloth_memory import releases, updates


def release(version, **overrides):
    name = f"sloth_memory-{version}-py3-none-any.whl"
    return dict(tag_name="v" + version, draft=False, assets=[dict(name=name,
        browser_download_url=f"{releases.REPOSITORY}/releases/download/v{version}/{name}",
        digest="sha256:" + "a" * 64)], **overrides)


class ReleaseTests(unittest.TestCase):
    def test_incompatible_snapshot_format_requires_manual_upgrade(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "package.whl"
            with zipfile.ZipFile(wheel, "w") as out:
                out.writestr("sloth_memory-0.2.0a2.dist-info/METADATA", "Name: sloth-memory\nVersion: 0.2.0a2\n")
                out.writestr("sloth_memory/update_compat.json", json.dumps(dict(updater_protocol=1, snapshot_format=3, control_version=4)))
            with self.assertRaisesRegex(ValueError, "manual compatibility"):
                releases.validate_wheel(wheel, "0.2.0a2")

    def test_alpha_channel_and_stable_channel_do_not_downgrade(self):
        data = [release("0.2.0a2"), release("0.2.0a10"), release("0.1.0")]
        with patch.object(releases, "fetch", return_value=json.dumps(data).encode()):
            self.assertEqual(releases.latest("0.2.0a1")["version"], "0.2.0a10")
            self.assertEqual(releases.latest("0.1.0")["version"], "0.1.0")

    def test_drafts_unhashed_assets_and_foreign_sources_are_ignored(self):
        draft = release("0.2.0a3"); draft["draft"] = True
        unhashed = release("0.2.0a4"); unhashed["assets"][0]["digest"] = None
        foreign = release("0.2.0a5"); foreign["assets"][0]["browser_download_url"] = "https://example.com/wheel"
        with patch.object(releases, "fetch", return_value=json.dumps([draft, unhashed, foreign]).encode()):
            self.assertIsNone(releases.latest("0.2.0a1"))

    def test_bad_checksum_never_reaches_wheel_validation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(releases, "fetch", return_value=b"bad"), patch.object(releases, "validate_wheel") as validate:
            with self.assertRaisesRegex(ValueError, "checksum"):
                releases.download(dict(url="unused", sha256="a" * 64, name="x.whl"), tmp)
            validate.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = Path(self.tmp.name)
        self.events = []
        self.record = dict(pid=123, created_at=1, base_url="http://127.0.0.1:8123",
                           upstream_url="http://127.0.0.1:9123", backend="llama.cpp", cache_mode="native")
        self.live = self.record
        self.install_calls = 0
        self.install_error = None
        self.start_error = None
        self.new = dict(version="0.2.0a2")
        self.mocks = {}
        values = dict(
            **{"metadata.version": lambda _: "0.2.0a1", "releases.latest": lambda _: self.new,
               "releases.download": lambda r, stage: stage / "new.whl",
               "rollback_wheel": lambda stage: stage / "old.whl",
               "proxy_record": lambda _: self.live, "process_matches": lambda _: True,
               "request": self.request, "install": self.install, "stop_proxy": self.stop,
               "start_proxy": self.start, "subprocess.check_output": lambda *a, **kw: "0.2.0a2\n"})
        for name, value in values.items():
            mocked = patch("sloth_memory.updates." + name, side_effect=value).start()
            self.addCleanup(patch.stopall)
            self.mocks[name] = mocked
        process = MagicMock()
        process.cmdline.return_value = [sys.executable, "-m", "sloth_memory", "serve"]
        process.environ.return_value = {}
        self.process = patch("sloth_memory.updates.psutil.Process", return_value=process).start()

    def request(self, url, action, *args, **kwargs):
        self.events.append(action)
        return dict(control_version=4, version="0.2.0a1", ok=True)

    def install(self, wheel, log):
        self.events.append(wheel.name)
        self.install_calls += 1
        if self.install_calls == 1 and self.install_error:
            raise self.install_error

    def stop(self, record):
        self.events.append("stop")
        self.live = None

    def start(self, record, archive, env, version):
        self.events.append("start:" + version)
        if version == "0.2.0a2" and self.start_error:
            raise self.start_error
        self.live = dict(record, pid=456)
        return self.live

    def test_success_saves_before_stopping_and_keeps_backend(self):
        result = updates._apply(self.archive, None)
        self.assertEqual(result["state"], "complete")
        self.assertLess(self.events.index("prepare-update"), self.events.index("stop"))
        self.assertLess(self.events.index("stop"), self.events.index("new.whl"))
        self.assertEqual(self.events[-1], "cancel-update")
        self.assertEqual(self.mocks["stop_proxy"].call_count, 1)
        self.assertEqual(self.mocks["start_proxy"].call_args.args[0]["upstream_url"], self.record["upstream_url"])

    def test_failed_install_restores_package_and_proxy(self):
        self.install_error = RuntimeError("pip failed")
        with self.assertRaisesRegex(RuntimeError, "rolled_back"):
            updates._apply(self.archive, None)
        self.assertIn("old.whl", self.events)
        self.assertIn("start:0.2.0a1", self.events)
        self.assertEqual(self.events[-1], "cancel-update")

    def test_bad_new_proxy_restores_package_and_proxy(self):
        self.start_error = RuntimeError("health failed")
        with self.assertRaisesRegex(RuntimeError, "rolled_back"):
            updates._apply(self.archive, None)
        self.assertIn("old.whl", self.events)
        self.assertIn("start:0.2.0a1", self.events)

    def test_keyboard_interrupt_recovers_before_propagating(self):
        self.install_error = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            updates._apply(self.archive, None)
        self.assertIn("old.whl", self.events)
        self.assertIn("start:0.2.0a1", self.events)

    def test_other_virtual_environment_is_not_stopped_or_installed(self):
        self.process.return_value.cmdline.return_value[0] = str(self.archive / "other-python")
        with self.assertRaisesRegex(RuntimeError, "different Python environment"):
            updates._apply(self.archive, None)
        self.mocks["stop_proxy"].assert_not_called()
        self.mocks["install"].assert_not_called()

    def test_old_control_protocol_needs_bootstrap_without_touching_proxy(self):
        self.mocks["request"].side_effect = lambda *a, **kw: dict(control_version=3)
        with self.assertRaisesRegex(RuntimeError, "one-time upgrade"):
            updates._apply(self.archive, None)
        self.mocks["stop_proxy"].assert_not_called()
        self.mocks["install"].assert_not_called()

    def test_no_update_does_not_reinstall_or_restart(self):
        self.new = dict(version="0.2.0a1")
        result = updates._apply(self.archive, None)
        self.assertEqual(result["state"], "current")
        self.mocks["stop_proxy"].assert_not_called()
        self.mocks["install"].assert_not_called()

    def test_current_package_can_replace_an_older_running_proxy(self):
        self.new = dict(version="0.2.0a1")
        self.mocks["request"].side_effect = lambda *a, **kw: dict(control_version=4, version="0.2.0a0", ok=True)
        self.mocks["subprocess.check_output"].side_effect = lambda *a, **kw: "0.2.0a1\n"
        result = updates._apply(self.archive, None)
        self.assertEqual(result["state"], "complete")
        self.assertIn("start:0.2.0a1", self.events)
        self.mocks["install"].assert_not_called()

    def test_surviving_replacement_blocks_package_rollback(self):
        def start(*args):
            self.live = dict(self.record, pid=456)
            raise RuntimeError("replacement could not be stopped")
        def stop(record):
            if record["pid"] == 456:
                raise RuntimeError("still alive")
            self.stop(record)
        self.mocks["start_proxy"].side_effect = start
        self.mocks["stop_proxy"].side_effect = stop
        with self.assertRaisesRegex(RuntimeError, "recovery_required"):
            updates._apply(self.archive, None)
        self.assertNotIn("old.whl", self.events)

    def test_unpublished_surviving_replacement_blocks_package_rollback(self):
        self.start_error = updates.ProxyStopError("unpublished replacement still alive")
        with self.assertRaisesRegex(RuntimeError, "recovery_required"):
            updates._apply(self.archive, None)
        self.assertNotIn("old.whl", self.events)

    def test_lost_reopen_ack_never_kills_a_verified_replacement(self):
        def request(*args, **kwargs):
            if args[1] == "cancel-update":
                raise OSError("lost acknowledgement")
            return self.request(*args, **kwargs)
        self.mocks["request"].side_effect = request
        with self.assertRaisesRegex(RuntimeError, "reopening inference"):
            updates._apply(self.archive, None)
        self.assertEqual(self.mocks["stop_proxy"].call_count, 1)
        self.assertNotIn("old.whl", self.events)

    def test_pip_settings_cannot_redirect_install(self):
        # Undo the transaction mock to exercise the subprocess boundary itself.
        patch.stopall()
        with patch.dict(os.environ, {"PIP_TARGET": "elsewhere", "PIP_USER": "1"}), patch.object(updates.subprocess, "run") as run:
            run.return_value.returncode = 0
            updates.install(self.archive / "package.whl", self.archive / "install.log")
            argv = run.call_args.args[0]
            env = run.call_args.kwargs["env"]
            self.assertEqual(argv[:4], [sys.executable, "-I", "-m", "pip"])
            self.assertEqual(argv[argv.index("--prefix") + 1], sys.prefix)
            self.assertNotIn("PIP_TARGET", env)
            self.assertNotIn("PIP_USER", env)
            self.assertEqual(env["PIP_CONFIG_FILE"], os.devnull)


if __name__ == "__main__":
    unittest.main()
