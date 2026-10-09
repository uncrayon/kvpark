"""live_bridge.py under a resource guard's Watch, without a model: its throwaway servers stay in the run's process
group (so the guard pauses and ends them with it) and stop() ends them; the engine daemon it starts there gets the
configuration's idle unload and an idle exit the run outlives, and every switch first checks that the daemon
answering is still that one; chunked_prefill sums its requests' prompt times.

live_bridge imports kvpark.bridge, whose registry needs psutil, which the engine venvs lack: the checks run in a
python3 that imports psutil (as test_dsv4's EngineBridgeTests do) and skip without one."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = Path(__file__).resolve().parent

# a stand-in llama-server: answers GET /health with 200 on its --port until it is ended
FAKE_SERVER = """#!{python}
import http.server, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""

CHECKS = r"""
import json, os, signal, socket, subprocess, sys, time, types
from pathlib import Path
from unittest import mock
sys.path.insert(0, sys.argv[1])
import live_bridge, resource_guard
work = Path(sys.argv[2])
fake = str(work / "fake-server")
out = {}

def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

def gone(pid, wait=10.0):
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            if open(f"/proc/{pid}/stat").read().split(") ")[1][0] == "Z":
                return True
        except OSError:
            return True
        time.sleep(0.05)
    return False

# under a Watch (KVPARK_GUARD_WATCHED=1) the server shares this process group, and stop() ends it
for watched in ("1", None):
    env = dict(os.environ)
    env.pop(resource_guard.WATCHED, None)
    if watched:
        env[resource_guard.WATCHED] = watched
    with mock.patch.dict(os.environ, env, clear=True):
        server = live_bridge.Server("fake", fake, "/m.gguf", free_port(), 4096, work, work / "server.log", args=[])
        server.start(timeout=60)
    pid = server.process.pid
    key = "watched" if watched else "own"
    out[key] = {"same_group": os.getpgid(pid) == os.getpgid(0), "own_session": os.getsid(pid) == pid}
    server.stop()
    out[key]["stopped"] = gone(pid)

# the engine daemon started in the run's group: the configuration's idle unload, an idle exit the run outlives
started = []
class Process:
    pid = 4242
    returncode = None
    def poll(self):
        return self.returncode
def popen(command, **options):
    started.append(command)
    return Process()
config = {"engine": {"python": "/py", "script": "/x/engine.py", "device": "vulkan", "idle_unload": 120}}
bridge = work / "bridge"
bridge.mkdir()
with mock.patch.object(live_bridge.subprocess, "Popen", popen), \
        mock.patch.object(live_bridge.engine_client, "_environment", lambda config: {}), \
        mock.patch.object(live_bridge.engine_client, "_ping", lambda directory, timeout=2.0: {"ok": True, "pid": 4242}):
    process = live_bridge.start_engine_daemon(config, bridge / "bridge.json")
    command = started[0]
    out["command"] = {"idle_unload": command[command.index("--idle-unload") + 1],
                      "idle_exit": float(command[command.index("--idle-exit") + 1]),
                      "device": command[command.index("--device") + 1]}
    live_bridge.check_daemon(bridge / "bridge.json")              # the same daemon answers: fine
    out["same_daemon"] = "ok"
    live_bridge.check_daemon(None)                                # no configuration path: nothing to check
with mock.patch.object(live_bridge.engine_client, "_ping", lambda directory, timeout=2.0: {"ok": True, "pid": 77}):
    try:
        live_bridge.switch(config, {"messages": []}, "a", "b", config_path=bridge / "bridge.json")
        out["other_daemon"] = "switched"
    except RuntimeError as error:
        out["other_daemon"] = str(error)
process.returncode = 0
try:
    live_bridge.check_daemon(bridge / "bridge.json")
    out["exited"] = "not noticed"
except RuntimeError as error:
    out["exited"] = str(error)

# chunked_prefill: requests of `chunk` tokens after a cooling wait each; their prompt times summed
asked = []
class Target:
    def slot_action(self, action):
        asked.append(action)
    def complete(self, tokens, n_predict=1):
        asked.append(len(tokens))
        cached = 0 if len(asked) == 2 else len(tokens) - (len(tokens) % 1000 or 1000) - 4
        return {"timings": {"cache_n": cached, "prompt_n": len(tokens) - cached, "prompt_ms": 100.0}}, 0.1
result = live_bridge.chunked_prefill(Target(), list(range(2500)), 1000, types.SimpleNamespace(cool_c=None))
out["chunked"] = {"asked": asked, "prompt_ms": result["prompt_ms"], "prompt_n": result["prompt_n"],
                  "ends": [item["end"] for item in result["chunks"]]}
print("RESULT " + json.dumps(out))
"""


def bridge_python():
    """A Python that imports kvpark.bridge.registry (psutil), or None."""
    python = shutil.which("python3")
    if python is None or subprocess.run([python, "-c", "import psutil"], capture_output=True).returncode:
        return None
    return python


class LiveBridgeGuardTest(unittest.TestCase):
    def test_servers_and_the_engine_daemon_stay_under_the_guard(self):
        python = bridge_python()
        if python is None:
            self.skipTest("no python3 with psutil (kvpark.bridge's registry)")
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "fake-server"
            fake.write_text(FAKE_SERVER.format(python=python))
            fake.chmod(0o755)
            result = subprocess.run([python, "-c", textwrap.dedent(CHECKS), str(HERE), directory], capture_output=True,
                                    text=True, timeout=180, env={k: v for k, v in os.environ.items()})
            lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
            self.assertTrue(lines, result.stderr[-3000:])
            out = json.loads(lines[-1][len("RESULT "):])
        # under a Watch: the run's group, so the guard's SIGSTOP and SIGTERM reach it; else a session of its own
        self.assertEqual(out["watched"], {"same_group": True, "own_session": False, "stopped": True})
        self.assertEqual(out["own"], {"same_group": False, "own_session": True, "stopped": True})
        self.assertEqual(out["command"]["idle_unload"], "120")
        self.assertGreaterEqual(out["command"]["idle_exit"], 24 * 3600)    # the run stops it, not its idle time
        self.assertEqual(out["command"]["device"], "vulkan")
        self.assertEqual(out["same_daemon"], "ok")
        self.assertIn("not the one started in the run's process group (4242)", out["other_daemon"])
        self.assertIn("exited", out["exited"])
        self.assertEqual(out["chunked"]["ends"], [1000, 2000, 2500])
        self.assertEqual(out["chunked"]["asked"], ["erase", 1000, 2000, 2500, "erase"])
        self.assertEqual(out["chunked"]["prompt_ms"], 300.0)


if __name__ == "__main__":
    unittest.main()
