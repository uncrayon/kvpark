"""Boot-time startup of a Hermes-configured connection without opening Hermes."""

import contextlib
import io
import sys
import threading
import unittest
from unittest.mock import patch

from kvpark import cli
from kvpark.hermes_service import Service, defaults, supervise


class SuperviseTests(unittest.TestCase):
    def test_supervise_retries_until_stopped_and_logs_failures(self):
        stop = threading.Event()
        calls = []
        lines = []

        class Flaky:
            def ensure(self):
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError("model server not ready")
                if len(calls) == 3:
                    stop.set()
                return {"service": "kvpark"}

        supervise(Flaky(), 0.01, stop=stop, log=lambda text, **_: lines.append(text))
        self.assertEqual(len(calls), 3)
        self.assertEqual(lines, ["kvpark: model server not ready; retrying in 0.01s"])


class StartHermesTests(unittest.TestCase):
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

    def test_start_hermes_uses_the_profile_connection(self):
        config = {**defaults(), "backend": "ollama", "cache_mode": "routing", "model": "gemma:latest",
                  "upstream_url": "http://127.0.0.1:11434", "connected": True}
        with patch("kvpark.setup.hermes_settings", return_value=config), \
                patch.object(Service, "ensure", return_value={"service": "kvpark"}) as ensure:
            code, output = self.run_cli("start", "--hermes")
        self.assertEqual(code, 0, output)
        ensure.assert_called_once()
        self.assertIn("kvpark doctor --hermes", output)

    def test_start_hermes_supervises_after_a_failed_first_attempt(self):
        config = {**defaults(), "backend": "ollama", "cache_mode": "routing", "model": "gemma:latest",
                  "upstream_url": "http://127.0.0.1:11434", "connected": True}
        with patch("kvpark.setup.hermes_settings", return_value=config), \
                patch.object(Service, "ensure", side_effect=RuntimeError("not yet")), \
                patch("kvpark.hermes_service.supervise") as loop:
            code, output = self.run_cli("start", "--hermes", "--supervise", "--interval", "5")
        self.assertEqual(code, 0, output)
        self.assertIn("retrying in 5s", output)
        loop.assert_called_once()
        self.assertEqual(loop.call_args.args[1], 5)

    def test_start_hermes_refuses_unconfigured_or_disconnected_profiles(self):
        for config, expected in (({**defaults()}, "No model is configured"),
                                 ({**defaults(), "uninstalled": True}, "disconnected")):
            with self.subTest(expected=expected), patch("kvpark.setup.hermes_settings", return_value=config), \
                    patch.object(Service, "ensure") as ensure:
                code, output = self.run_cli("start", "--hermes")
            self.assertEqual(code, 1)
            self.assertIn(expected, output)
            ensure.assert_not_called()

    def test_start_hermes_needs_hermes_python(self):
        with patch("kvpark.setup.hermes_settings", side_effect=ImportError("No module named hermes_cli")):
            code, output = self.run_cli("start", "--hermes")
        self.assertEqual(code, 1)
        self.assertIn("Python environment that runs Hermes", output)


if __name__ == "__main__":
    unittest.main()
