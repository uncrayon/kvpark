"""Load kvpark as a Hermes plugin directory (hermes plugins install uncrayon/kvpark) with no pip package."""

import asyncio
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]


class PluginDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        home = Path(self.tmp.name)
        # What `hermes plugins install` leaves under ~/.hermes/plugins/kvpark, minus docs and tests.
        plugin = home / "plugins/kvpark"
        plugin.mkdir(parents=True)
        for name in ("plugin.yaml", "__init__.py"):
            shutil.copy(REPO / name, plugin / name)
        shutil.copytree(REPO / "src/kvpark", plugin / "src/kvpark", ignore=shutil.ignore_patterns("__pycache__"))
        (home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["kvpark"]}}))
        env = patch.dict(os.environ, {"HERMES_HOME": str(home), "HERMES_BUNDLED_PLUGINS": str(home / "empty"),
                                      "KVPARK_NO_AUTOSTART": "1"})
        env.start()
        self.addCleanup(env.stop)

    def test_directory_plugin_registers_the_command_without_a_package(self):
        from hermes_cli.plugins import PluginManager, get_plugin_command_handler, get_plugin_manager
        with patch.object(PluginManager, "_scan_entry_points", return_value=[]):
            manager = get_plugin_manager()
            manager.discover_and_load(force=True)
            self.addCleanup(manager.unload)
            info = next(entry for entry in manager.list_plugins() if entry["name"] == "kvpark")
            self.assertEqual(info["source"], "user")
            self.assertTrue(info["enabled"], info["error"])
            self.assertIsNone(info["error"])
            handler = get_plugin_command_handler("kvpark")
            self.assertIsNotNone(handler, "directory plugin did not register /kvpark")
            self.assertIn("kvpark", asyncio.run(handler("help")))
            self.assertIn("Plugin: loaded", asyncio.run(handler("doctor")))


if __name__ == "__main__":
    unittest.main()
