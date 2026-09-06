from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from sloth_memory.hermes_service import Service, defaults, validate_service


class ServiceTests(unittest.TestCase):
    def test_base_url_normalization_and_local_only_boundary(self):
        self.assertEqual(validate_service({"base_url": "http://127.0.0.1:9000/v1/"})["base_url"], "http://127.0.0.1:9000")
        for url in ("https://example.com", "http://127.0.0.1:8080/x", "http://user@127.0.0.1:8080", "http://127.0.0.1:8090"):
            with self.assertRaises(ValueError):
                validate_service({"base_url": url})

    def test_existing_service_is_attached_without_spawning(self):
        config = defaults()
        state = {"service": "sloth-memory", "control_version": 2, "archive_dir": config["archive_dir"], "upstream": "127.0.0.1:8090"}
        service = Service(lambda: config)
        with patch("sloth_memory.hermes_service.request", return_value=state), patch.object(service, "_health", return_value=True), patch.object(service, "_spawn") as spawn:
            self.assertEqual(service.ensure(), state)
            spawn.assert_not_called()

    def test_unrelated_service_is_never_taken_over(self):
        service = Service(defaults)
        with patch("sloth_memory.hermes_service.request", return_value={"service": "other"}), patch.object(service, "_spawn") as spawn:
            with self.assertRaisesRegex(RuntimeError, "incompatible service"):
                service.ensure()
            spawn.assert_not_called()

    def test_missing_configuration_guides_onboarding_without_spawning(self):
        service = Service(defaults)
        with patch("sloth_memory.hermes_service.request", side_effect=URLError("offline")), patch.object(service, "_spawn") as spawn:
            with self.assertRaisesRegex(ValueError, "/sloth-memory setup"):
                service.ensure()
            spawn.assert_not_called()

    def test_backend_args_cannot_override_archive_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"model")
            config = {**defaults(), "server": sys.executable, "model": str(model), "backend_args": ["--parallel", "8"]}
            service = Service(lambda: config)
            with patch("sloth_memory.hermes_service.request", side_effect=URLError("offline")), patch.object(service, "_spawn") as spawn:
                with self.assertRaises(ValueError):
                    service.ensure()
                spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
