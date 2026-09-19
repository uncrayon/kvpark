"""Route classification and installation reporting that need no Hermes."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from kvpark.backends import route_key
from kvpark.hermes_doctor import install_source, report, route_state
from kvpark.hermes_service import defaults, route_for


class DoctorHelperTests(unittest.TestCase):
    def test_route_key_treats_loopback_aliases_and_trailing_slashes_alike(self):
        pairs = (("http://localhost:8080/v1/", "http://127.0.0.1:8080/v1"),
                 ("http://[::1]:8080", "http://127.0.0.1:8080/"),
                 ("https://example.com/v1", "https://example.com:443/v1"))
        for left, right in pairs:
            self.assertEqual(route_key(left), route_key(right))
        self.assertNotEqual(route_key("http://127.0.0.1:8080/v1"), route_key("http://127.0.0.1:8081/v1"))
        self.assertNotEqual(route_key("http://127.0.0.1:8080"), route_key("http://127.0.0.1:8080/v1"))

    def test_route_state_distinguishes_upstream_legacy_proxy_and_other(self):
        settings = {**defaults(), "connected": True, "backend": "vllm", "cache_mode": "routing",
                    "upstream_url": "http://127.0.0.1:8081", "model": "m", "previous_urls": ["http://127.0.0.1:9000"]}
        self.assertEqual(route_state({"base_url": "http://localhost:8081/v1"}, settings), "upstream")
        self.assertEqual(route_state({"base_url": "http://127.0.0.1:8080/v1"}, settings), "proxy")
        self.assertEqual(route_state({"base_url": "http://127.0.0.1:9000/v1"}, settings), "proxy")
        self.assertEqual(route_state({"base_url": "https://api.example.com/v1"}, settings), "other")
        self.assertEqual(route_state({}, settings), "unset")
        self.assertEqual(route_state("provider/model", settings), "unset")

    def test_route_for_points_hermes_at_the_model_server(self):
        self.assertEqual(route_for(defaults()), {"provider": "custom", "default": "local",
                                                 "base_url": "http://127.0.0.1:8090/v1", "api_mode": "chat_completions"})
        existing = {**defaults(), "backend": "ollama", "cache_mode": "routing",
                    "upstream_url": "http://127.0.0.1:11434", "model": "gemma:latest"}
        self.assertEqual(route_for(existing)["base_url"], "http://127.0.0.1:11434/v1")
        self.assertEqual(route_for(existing)["default"], "gemma:latest")

    def test_install_source_reports_the_running_package(self):
        source, detail = install_source()
        self.assertIn(source, ("pip", "directory"))
        if source == "directory":
            self.assertTrue((Path(detail) / "kvpark").is_dir())

    def test_report_without_hermes_explains_which_python_to_use(self):
        with patch.dict(sys.modules, {"hermes_cli": None}):
            lines, ok = report()
        self.assertFalse(ok)
        self.assertIn("not importable", lines[0])


if __name__ == "__main__":
    unittest.main()
