import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from kvpark.bridge import engine_client


ONE_SHOT = textwrap.dedent("""\
    import json, os, stat, sys
    path = sys.argv[1]
    job = json.load(open(path))
    marker = os.path.join(os.path.dirname(os.path.dirname(path)), "oneshot-seen.json")
    json.dump({"mode": stat.S_IMODE(os.stat(path).st_mode), "job": job}, open(marker, "w"))
    if job.get("explode"):
        sys.exit("engine exploded")
    if job.get("mismatch"):                  # engine.py's TemplateMismatch exit status
        print("TemplateMismatch: the target server's tokens do not spell the target rendering", file=sys.stderr)
        sys.exit(4)
    print(json.dumps({"t0": 5, "engine_seconds": 0.01, "device": job.get("device")}))
""")

# a stand-in daemon (stdlib only) that run_job may start: answers ping, translate and shutdown
DAEMON = textwrap.dedent("""\
    import argparse, json, os, socket, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-dir")
    parser.add_argument("--device")
    args, _ = parser.parse_known_args()
    if os.path.exists(os.path.join(args.bridge_dir, "held")):      # another daemon holds the directory
        sys.exit(3)
    print("daemon starting", " ".join(sys.argv[1:]), "threads=" + os.environ.get("OPENBLAS_NUM_THREADS", "?"),
          file=sys.stderr, flush=True)
    path = os.path.join(args.bridge_dir, "engine.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(4)
    while True:
        connection, _ = server.accept()
        stream = connection.makefile("rwb")
        request = json.loads(stream.readline())
        op = request["op"]
        if op == "ping":
            reply = {"ok": True, "pid": os.getpid(), "device": args.device}
        elif op == "translate":
            reply = {"ok": True, "report": {"t0": 9, "engine_seconds": 0.02}, "daemon": {"queue_s": 0.0}}
        else:
            reply = {"ok": True}
        stream.write(json.dumps(reply).encode() + b"\\n")
        stream.flush()
        connection.close()
        if op == "shutdown":
            os.unlink(path)
            break
""")


class FakeDaemon:
    """An in-process daemon on <bridge_dir>/engine.sock with scripted replies per op."""

    def __init__(self, bridge_dir, replies=None, *, pid=None, hang=(), drop=()):
        self.path = Path(bridge_dir) / "engine.sock"
        self.requests, self.replies, self.hang, self.drop, self.pid = [], dict(replies or {}), set(hang), set(drop), pid
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.path))
        self.server.listen(8)
        self.closed = threading.Event()
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        while not self.closed.is_set():
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self.handle, args=(connection,), daemon=True).start()

    def handle(self, connection):
        with connection, connection.makefile("rwb") as stream:
            for line in stream:
                request = json.loads(line)
                self.requests.append(request)
                op = request["op"]
                if op in self.hang:
                    self.closed.wait(30)
                    return
                if op in self.drop:            # dies with the request in hand
                    return
                if op == "ping":
                    reply = self.replies.get("ping", {"ok": True, "pid": self.pid or 0})
                else:
                    reply = self.replies.get(op, {"ok": True})
                stream.write(json.dumps(reply).encode() + b"\n")
                stream.flush()

    def close(self):
        self.closed.set()
        self.server.close()
        self.path.unlink(missing_ok=True)


class EngineClientTest(unittest.TestCase):
    def setUp(self):
        # a short directory: AF_UNIX socket paths are limited to about 100 bytes
        self.tmp = tempfile.TemporaryDirectory(dir="/tmp" if os.path.isdir("/tmp") else None)
        self.root = Path(self.tmp.name)
        self.bridge = self.root / "b"
        self.bridge.mkdir(mode=0o700)
        self.engine = self.root / "engine.py"
        self.engine.write_text(ONE_SHOT)
        self.config = {"engine": {"python": sys.executable, "script": str(self.engine)}, "hub": str(self.root / "hub"),
                       "models": {"a": {"hub_name": "qwen38"}, "b": {"hub_name": "muse"}}}
        self.output = self.root / "out.bin"
        self.job = {"hub": "H", "conversation": {"messages": [{"role": "user", "content": "secret"}]},
                    "target": {"output": str(self.output)}}
        self.fakes = []

    def tearDown(self):
        for fake in self.fakes:
            fake.close()
        for child in engine_client._CHILDREN:          # daemons a test started
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        engine_client._CHILDREN.clear()
        self.tmp.cleanup()

    def daemon(self, *args, **kwargs):
        (self.root / "engine_daemon.py").write_text("raise SystemExit('not used')")
        fake = FakeDaemon(self.bridge, *args, **kwargs)
        self.fakes.append(fake)
        return fake

    def seen(self):
        path = self.bridge / "oneshot-seen.json"
        return json.loads(path.read_text()) if path.exists() else None

    @unittest.skipUnless(engine_client.daemon_supported(Path("/tmp/x")), "no AF_UNIX sockets")
    def test_a_running_daemon_takes_the_job_inline(self):
        fake = self.daemon({"translate": {"ok": True, "report": {"t0": 11, "engine_seconds": 0.1},
                                          "daemon": {"queue_s": 0.0, "run_s": 0.1}}})
        self.config["engine"]["device"] = "gpu"
        report = engine_client.run_job(self.config, self.job, bridge_dir=self.bridge)
        self.assertEqual((report["t0"], report["engine_mode"]), (11, "daemon"))
        self.assertIn("client_s", report["daemon"])
        translate = next(r for r in fake.requests if r["op"] == "translate")
        self.assertEqual(translate["job"]["conversation"], self.job["conversation"])
        self.assertEqual(translate["job"]["device"], "gpu")                     # the engine's default device
        self.assertIsNone(self.seen())                                            # no one-shot, no job file
        self.assertFalse((self.bridge / "jobs").exists() and any((self.bridge / "jobs").iterdir()))
        report = engine_client.run_job(self.config, {**self.job, "device": "cpu"}, bridge_dir=self.bridge)
        self.assertEqual(fake.requests[-1]["job"]["device"], "cpu")               # a job's own device wins

    @unittest.skipUnless(engine_client.daemon_supported(Path("/tmp/x")), "no AF_UNIX sockets")
    def test_daemon_faults_fall_back_to_the_one_shot_engine(self):
        fake = self.daemon()
        for reply in ({"ok": False, "error": "AttributeError: boom", "kind": "internal"},
                      {"ok": False, "error": "the engine daemon is shutting down", "kind": "busy"},
                      {"ok": False, "error": "an older daemon without kinds"}):
            fake.replies["translate"] = reply
            self.output.write_bytes(b"half written by the daemon")
            partial = self.root / f".{self.output.name}.partial-x1"
            partial.write_bytes(b"a writer's temporary")
            events = []
            report = engine_client.run_job(self.config, self.job, bridge_dir=self.bridge, log=events.append)
            self.assertEqual((report["t0"], report["engine_mode"]), (5, "oneshot"))
            self.assertIn(reply["error"], report["daemon_error"])
            self.assertFalse(self.output.exists() or partial.exists())   # a daemon's write never mixes with the fallback
            seen = self.seen()
            self.assertEqual(seen["mode"], 0o600)
            self.assertEqual(seen["job"]["conversation"], self.job["conversation"])
            self.assertEqual(list((self.bridge / "jobs").iterdir()), [])            # job file deleted
            self.assertTrue(any(e.get("step") == "engine-daemon-failed" for e in events))
            (self.bridge / "oneshot-seen.json").unlink()

    @unittest.skipUnless(engine_client.daemon_supported(Path("/tmp/x")), "no AF_UNIX sockets")
    def test_job_errors_and_memory_refusals_are_not_rerun_one_shot(self):
        fake = self.daemon()
        for kind, error in (("memory", "MemoryError: memory: a switch needs about 9000 MiB of RAM"),
                            ("job", "ValueError: conversation too short to hand over (T0 = 3)")):
            fake.replies["translate"] = {"ok": False, "error": error, "kind": kind}
            events = []
            with self.assertRaises(engine_client.EngineError) as caught:
                engine_client.run_job(self.config, self.job, bridge_dir=self.bridge, log=events.append)
            self.assertEqual(caught.exception.kind, kind)
            self.assertEqual(str(caught.exception), "bridge engine failed: " + error)   # as a one-shot failure reads
            self.assertIsInstance(caught.exception, RuntimeError)
            self.assertIsNone(self.seen())                                            # no unguarded rerun
            self.assertEqual([e["step"] for e in events if e["step"].startswith("engine-")], ["engine-failed"])

    @unittest.skipUnless(engine_client.daemon_supported(Path("/tmp/x")), "no AF_UNIX sockets")
    def test_a_daemon_that_overruns_is_stopped_and_the_job_not_rerun(self):
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            self.daemon(pid=sleeper.pid, hang={"translate"})
            self.output.write_bytes(b"half written by the daemon")
            events = []
            started = time.monotonic()
            with self.assertRaises(engine_client.EngineError) as caught:
                engine_client.run_job(self.config, self.job, bridge_dir=self.bridge, timeout=1.5, log=events.append)
            self.assertLess(time.monotonic() - started, 5)          # one timeout, not one per path
            self.assertEqual(caught.exception.kind, "timeout")
            self.assertIn("did not finish the job within 2 s", str(caught.exception))
            self.assertIsNotNone(sleeper.wait(timeout=5))          # the overrunning daemon was killed
            self.assertIsNone(self.seen())                           # a rerun would only wait as long again
            self.assertFalse(self.output.exists())
            self.assertEqual(events[-1]["step"], "engine-failed")
        finally:
            if sleeper.poll() is None:
                sleeper.kill()

    @unittest.skipUnless(engine_client.daemon_supported(Path("/tmp/x")), "no AF_UNIX sockets")
    def test_a_daemon_that_dies_during_the_job_is_not_rerun_unguarded(self):
        self.daemon(drop={"translate"})
        with self.assertRaises(engine_client.EngineError) as caught:
            engine_client.run_job(self.config, self.job, bridge_dir=self.bridge)
        self.assertEqual(caught.exception.kind, "lost")
        self.assertIn("stopped during the job", str(caught.exception))
        self.assertIsNone(self.seen())                               # e.g. killed for memory: no unguarded rerun

    def test_the_fallback_gets_what_is_left_of_the_timeout(self):
        (self.root / "engine_daemon.py").write_text("raise SystemExit('not used')")

        def slow_start(*args, **kwargs):
            time.sleep(0.5)
            raise engine_client.DaemonError("the engine daemon did not answer")
        with mock.patch.object(engine_client, "ensure_daemon", side_effect=slow_start) as ensure, \
                mock.patch.object(engine_client, "_one_shot", return_value={"t0": 5}) as one_shot:
            report = engine_client.run_job(self.config, self.job, bridge_dir=self.bridge, timeout=30)
        self.assertEqual((report["engine_mode"], ensure.call_args.kwargs["timeout"]), ("oneshot", 30))
        self.assertLessEqual(one_shot.call_args[0][3], 29.5)

    def test_one_shot_without_a_daemon_script_or_by_configuration(self):
        report = engine_client.run_job(self.config, self.job, bridge_dir=self.bridge)
        self.assertEqual(report["engine_mode"], "oneshot")
        self.assertNotIn("daemon_error", report)
        (self.root / "engine_daemon.py").write_text("raise SystemExit('never started')")
        self.config["engine"]["mode"] = "oneshot"
        self.assertEqual(engine_client.run_job(self.config, self.job, bridge_dir=self.bridge)["engine_mode"],
                         "oneshot")
        del self.config["engine"]["mode"]
        with mock.patch.object(engine_client, "daemon_supported", return_value=False):    # e.g. Windows
            report = engine_client.run_job(self.config, self.job, bridge_dir=self.bridge)
        self.assertEqual((report["engine_mode"], report.get("daemon_error")), ("oneshot", None))

    def test_one_shot_failures_carry_the_engine_error(self):
        with self.assertRaisesRegex(RuntimeError, "bridge engine failed: engine exploded"):
            engine_client.run_job(self.config, {**self.job, "explode": True}, bridge_dir=self.bridge)
        self.assertEqual(list((self.bridge / "jobs").iterdir()), [])
        # a rendering the engine cannot reproduce has its own exit status and EngineError kind
        with self.assertRaises(engine_client.EngineError) as caught:
            engine_client.run_job(self.config, {**self.job, "mismatch": True}, bridge_dir=self.bridge)
        self.assertEqual(caught.exception.kind, engine_client.TEMPLATE)
        self.assertIn("TemplateMismatch", str(caught.exception))

    @unittest.skipUnless(engine_client.daemon_supported(Path("/tmp/x")), "no AF_UNIX sockets")
    def test_template_mismatches_on_the_daemon_are_not_rerun_one_shot(self):
        fake = self.daemon()
        fake.replies["translate"] = {"ok": False, "error": "TemplateMismatch: no match", "kind": "template"}
        with self.assertRaises(engine_client.EngineError) as caught:
            engine_client.run_job(self.config, self.job, bridge_dir=self.bridge)
        self.assertEqual(caught.exception.kind, "template")
        self.assertIsNone(self.seen())

    def test_features_come_from_the_daemon_or_the_engine_script(self):
        # an engine script from before the list: none
        self.assertEqual(engine_client.features(self.config, self.bridge), frozenset())
        self.engine.write_text('FEATURES = ("template-options", "media-source")\n' + ONE_SHOT)
        self.assertEqual(engine_client.features(self.config, self.bridge), {"template-options", "media-source"})
        for broken in ("FEATURES = compute()\n", "def f(:\n"):
            self.engine.write_text(broken)
            self.assertEqual(engine_client.script_features(self.engine), frozenset())
        self.assertEqual(engine_client.script_features(self.root / "missing.py"), frozenset())
        if engine_client.daemon_supported(self.bridge):
            # a running daemon answers for itself (it may be older than the script on disk)
            self.daemon({"ping": {"ok": True, "pid": 1, "features": ["template-options"]}})
            self.assertEqual(engine_client.features(self.config, self.bridge), {"template-options"})
            self.fakes[-1].replies["ping"] = {"ok": True, "pid": 1}
            self.assertEqual(engine_client.features(self.config, self.bridge), frozenset())

    def test_long_bridge_paths_skip_the_daemon(self):
        deep = self.root / ("d" * 120)
        self.assertFalse(engine_client.daemon_supported(deep))
        self.assertIsNone(engine_client.status(deep))

    def test_blas_threads(self):
        self.assertTrue(1 <= engine_client.default_threads() <= 16)
        with mock.patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "3", "OMP_NUM_THREADS": ""}):
            self.assertEqual(engine_client._environment(self.config)["OPENBLAS_NUM_THREADS"], "3")   # the user's
            self.config["engine"]["threads"] = 5                                                 # the configured
            self.assertEqual({engine_client._environment(self.config)[name] for name in engine_client.THREAD_VARIABLES},
                             {"5"})
        del self.config["engine"]["threads"]
        with mock.patch.dict(os.environ, {name: "" for name in engine_client.THREAD_VARIABLES}):
            self.assertEqual(engine_client._environment(self.config)["MKL_NUM_THREADS"],
                             str(engine_client.default_threads()))
        # a lone OMP_NUM_THREADS reaches OpenBLAS too, which reads its own variable first
        with mock.patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": ""}):
            self.assertEqual({engine_client._environment(self.config)[name] for name in engine_client.THREAD_VARIABLES},
                             {"4"})

    def test_check_asks_the_engine_python(self):
        self.assertIsNone(engine_client.check(self.config))                  # no daemon script, nothing to ask
        (self.root / "engine_daemon.py").write_text(textwrap.dedent("""\
            import json, os, sys
            assert sys.argv[1:] == ["--check", "--device", "gpu"], sys.argv
            print("loading", file=sys.stderr)
            print(json.dumps({"blas": {"name": "scipy-openblas", "reference": False},
                              "threads": os.environ.get("OPENBLAS_NUM_THREADS")}))
        """))
        self.config["engine"].update(device="gpu", threads=4)
        self.assertEqual(engine_client.check(self.config),
                         {"blas": {"name": "scipy-openblas", "reference": False}, "threads": "4"})
        (self.root / "engine_daemon.py").write_text("import sys; sys.exit('numpy is missing')")
        with self.assertRaisesRegex(engine_client.DaemonError, "numpy is missing"):
            engine_client.check(self.config)
        (self.root / "engine_daemon.py").write_text("print('no json')")
        with self.assertRaisesRegex(engine_client.DaemonError, "no report"):
            engine_client.check(self.config)

    @unittest.skipUnless(engine_client.daemon_supported(Path("/tmp/x")), "no AF_UNIX sockets")
    def test_starts_a_detached_daemon_then_status_and_stop(self):
        (self.root / "engine_daemon.py").write_text(DAEMON)
        self.config["engine"]["device"] = "cpu"
        events = []
        report = engine_client.run_job(self.config, self.job, bridge_dir=self.bridge, log=events.append)
        self.assertEqual((report["t0"], report["engine_mode"]), (9, "daemon"))
        steps = [e["step"] for e in events]
        self.assertEqual(steps[:2], ["engine-daemon-start", "engine-daemon-ready"])
        log = (self.bridge / "engine.log").read_text()
        self.assertIn("--device cpu", log)
        self.assertIn(f"threads={engine_client.default_threads()}", log)     # unless the environment sets them
        self.assertEqual((self.bridge / "engine.log").stat().st_mode & 0o777, 0o600)
        pid = engine_client._ping(self.bridge)["pid"]
        self.assertEqual(os.getsid(pid), pid)                    # its own session: survives the caller
        self.assertTrue(engine_client.status(self.bridge)["ok"])
        self.assertTrue(engine_client.stop(self.bridge))
        self.assertFalse((self.bridge / "engine.sock").exists())
        self.assertIsNone(engine_client.status(self.bridge))
        self.assertFalse(engine_client.stop(self.bridge))

    @unittest.skipUnless(engine_client.daemon_supported(Path("/tmp/x")), "no AF_UNIX sockets")
    def test_a_daemon_shutting_down_is_replaced(self):
        (self.root / "engine_daemon.py").write_text(DAEMON)
        old = FakeDaemon(self.bridge, {"ping": {"ok": False, "error": "shutting down", "kind": "busy"}})
        self.fakes.append(old)
        (self.bridge / "held").write_text("")        # its lock: new daemons exit 3 until it is gone

        def finish():
            time.sleep(1.0)
            (self.bridge / "held").unlink()
            old.close()
        threading.Thread(target=finish, daemon=True).start()
        events = []
        report = engine_client.run_job(self.config, self.job, bridge_dir=self.bridge, log=events.append)
        self.assertEqual((report["t0"], report["engine_mode"]), (9, "daemon"))
        self.assertEqual(events[-1]["step"], "engine-daemon-ready")
        self.assertGreater([e["step"] for e in events].count("engine-daemon-start"), 1)     # exited 3, started again
        self.assertTrue(engine_client.stop(self.bridge))

    @unittest.skipUnless(engine_client.daemon_supported(Path("/tmp/x")), "no AF_UNIX sockets")
    def test_a_daemon_that_never_answers_is_reported(self):
        (self.root / "engine_daemon.py").write_text("import sys; print('broken daemon', file=sys.stderr); sys.exit(2)")
        report = engine_client.run_job(self.config, self.job, bridge_dir=self.bridge)
        self.assertEqual(report["engine_mode"], "oneshot")
        self.assertIn("exited with 2", report["daemon_error"])
        self.assertIn("broken daemon", report["daemon_error"])

    @unittest.skipUnless(engine_client.daemon_supported(Path("/tmp/x")), "no AF_UNIX sockets")
    def test_prewarm_loads_the_pair_and_never_raises(self):
        fake = self.daemon({"load": {"ok": True, "loaded": [["k", "qwen38", "encoder"]], "seconds": 0.2}})
        self.config["models"]["b"]["gguf"] = "/models/muse.gguf"
        reply = engine_client.prewarm(self.config, ["a", "b"], self.bridge)
        self.assertTrue(reply["ok"])
        load = fake.requests[-1]
        self.assertEqual((load["op"], load["source"], load["target"], load["hub"]),
                         ("load", "qwen38", "muse", self.config["hub"]))
        self.assertEqual((load["gguf"], load["cache_dir"]), (["/models/muse.gguf"], str(self.bridge / "profiles")))
        # a pair is also warmed up: the daemon runs its products once, knowing both models' layouts
        self.assertEqual((load["source_gguf"], load["target_gguf"], load["warm_up"]), (None, "/models/muse.gguf", True))
        engine_client.prewarm(self.config, ["a", "b"], self.bridge, warm_up=False)
        self.assertFalse(fake.requests[-1]["warm_up"])
        engine_client.prewarm(self.config, ["a", "b", "a"], self.bridge)
        self.assertEqual(fake.requests[-1]["models"], ["qwen38", "muse", "qwen38"])
        self.assertNotIn("target_gguf", fake.requests[-1])
        fake.replies["load"] = {"ok": False, "error": "no adapter for nobody"}
        events = []
        self.assertIsNone(engine_client.prewarm(self.config, ["a", "b"], self.bridge, log=events.append))
        self.assertEqual(events[-1]["step"], "engine-prewarm-failed")
        self.assertIsNone(engine_client.prewarm(self.config, ["a", "missing"], self.bridge))
        self.config["engine"]["mode"] = "oneshot"
        self.assertIsNone(engine_client.prewarm(self.config, ["a", "b"], self.bridge))


if __name__ == "__main__":
    unittest.main()
