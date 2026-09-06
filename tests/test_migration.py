import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kvpark.migration import profile_plan


class MigrationTests(unittest.TestCase):
    def profile(self):
        return {"model": {"base_url": "http://127.0.0.1:8080/v1", "default": "local"},
                "agent": {"max_turns": 20}, "plugins": {"enabled": ["other", "sloth-memory"],
                "disabled": ["kvpark", "unrelated"], "entries": {"sloth-memory": {"settings": {
                    "service": {"archive_dir": str(Path('/old/archive').resolve()), "connected": True, "autostart": True,
                                "previous_urls": ["http://127.0.0.1:9090"], "backend": "llama.cpp"},
                    "route_backup": {"provider": "old"}, "route_applied": {"default": "local"}}}}},
                "platforms": {"telegram": {"extra": {"command_menu": {"priority": ["status", "sloth", "kvpark"]}}}}}

    def test_preserves_cache_identity_route_and_original_backup(self):
        source = self.profile()
        before = copy.deepcopy(source)
        result = profile_plan(source)
        self.assertEqual(source, before)
        settings = result["plugins"]["entries"]["kvpark"]["settings"]
        old = source["plugins"]["entries"]["sloth-memory"]["settings"]
        for key in old:
            self.assertEqual(settings[key], old[key])
        self.assertEqual(result["model"], source["model"])
        self.assertEqual(result["agent"], source["agent"])
        self.assertEqual(result["plugins"]["enabled"], ["other", "kvpark"])
        self.assertIn("sloth-memory", result["plugins"]["disabled"])
        self.assertTrue(result["plugins"]["entries"]["sloth-memory"]["settings"]["service"]["uninstalled"])
        self.assertEqual(result["platforms"]["telegram"]["extra"]["command_menu"]["priority"], ["status", "kvpark"])
        self.assertIsNone(profile_plan(result))

    def test_legacy_config_shape_migrates_and_existing_new_settings_are_protected(self):
        source = self.profile()
        old = source["plugins"]["entries"]["sloth-memory"]
        old["config"] = old.pop("settings")
        self.assertEqual(profile_plan(source)["plugins"]["entries"]["kvpark"]["settings"]["route_backup"], {"provider": "old"})
        source["plugins"]["entries"]["kvpark"] = {"settings": {"service": {"model": "new"}}}
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            profile_plan(source)

    def test_external_service_cannot_be_migrated_by_another_owner(self):
        source = self.profile()
        source["plugins"]["entries"]["sloth-memory"]["settings"]["service"]["external"] = True
        with self.assertRaisesRegex(ValueError, "externally managed"):
            profile_plan(source)

    def test_mixed_legacy_config_and_new_settings_keep_exact_service(self):
        source = self.profile()
        old = source["plugins"]["entries"]["sloth-memory"]
        service = old["settings"].pop("service")
        old["config"] = {"service": service}
        result = profile_plan(source)
        settings = result["plugins"]["entries"]["kvpark"]["settings"]
        self.assertEqual(settings["service"], service)
        self.assertEqual(settings["route_backup"], old["settings"]["route_backup"])
        # A complete settings.service overrides, rather than deep-merges, config.service.
        old["settings"]["service"] = {"archive_dir": str(Path('/another/archive').resolve())}
        result = profile_plan(source)
        settings = result["plugins"]["entries"]["kvpark"]["settings"]
        self.assertEqual(settings["service"], old["settings"]["service"])

    def test_legacy_environment_fallback_and_new_name_precedence(self):
        env = {key: value for key, value in os.environ.items() if not key.startswith(("SLOTH_", "KVPARK_"))}
        env.update(SLOTH_API_KEY="old-test-key", SLOTH_ARCHIVE_DIR="old-dir", KVPARK_API_KEY="new-test-key")
        command = [sys.executable, "-c", "import kvpark, os; assert os.environ['KVPARK_API_KEY']=='new-test-key'; assert os.environ['KVPARK_ARCHIVE_DIR']=='old-dir'"]
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    unittest.main()
