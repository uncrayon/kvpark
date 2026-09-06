import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from kvpark.platforms import available_port, process_identity, process_matches
from kvpark.network import bind_server
from kvpark.proxy import Handler, atomic_json
from kvpark.runtime import lock_directory


class PlatformTests(unittest.TestCase):
    def test_lock_excludes_other_process_and_releases_after_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            fd = lock_directory(Path(tmp), "lock")
            command = [sys.executable, "-c", "from kvpark.runtime import lock_directory; from pathlib import Path; import sys; lock_directory(Path(sys.argv[1]), 'lock')", tmp]
            try:
                self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            finally:
                os.close(fd)
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_process_identity_rejects_dead_pid_and_pid_reuse(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            identity = process_identity(process.pid)
            self.assertTrue(process_matches(identity))
            self.assertFalse(process_matches({**identity, "created_at": 0}))
        finally:
            process.terminate()
            process.wait(10)
        self.assertFalse(process_matches(identity))

    def test_listener_reserves_selected_port_until_closed(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            preferred = occupied.getsockname()[1]
            with patch("socket.getfqdn", side_effect=AssertionError("localhost startup must not use DNS")):
                server = bind_server(Handler, preferred, f"http://127.0.0.1:{preferred}")
            try:
                selected = server.server_port
                self.assertNotEqual(selected, preferred)
                self.assertNotEqual(available_port(selected), selected)
            finally:
                server.server_close()

    def test_atomic_metadata_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "state.json"
            atomic_json(file, {"version": 1})
            atomic_json(file, {"version": 2})
            self.assertEqual(json.loads(file.read_text()), {"version": 2})

    @unittest.skipUnless(os.name == "nt", "Windows process lifetime")
    def test_job_stops_child_when_launcher_releases_handle(self):
        from kvpark.windows_job import Job
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            job = Job(process.pid)
            job.close()
            process.wait(10)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


if __name__ == "__main__":
    unittest.main()
