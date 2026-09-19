"""The repository root doubles as a Hermes plugin directory."""

import importlib.util
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PluginDirectoryLayoutTests(unittest.TestCase):
    def test_manifest_matches_the_package_version(self):
        from kvpark import __version__
        text = (ROOT / "plugin.yaml").read_text()
        self.assertEqual(re.search(r"^name: (.+)$", text, re.M).group(1), "kvpark")
        self.assertEqual(re.search(r"^version: (.+)$", text, re.M).group(1), __version__)

    def test_root_init_exposes_register_the_way_hermes_loads_it(self):
        spec = importlib.util.spec_from_file_location("hermes_plugins.kvpark", ROOT / "__init__.py",
                                                      submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from kvpark.hermes import register
        self.assertIs(module.register, register)
        self.assertIn(str(ROOT / "src"), sys.path)


if __name__ == "__main__":
    unittest.main()
