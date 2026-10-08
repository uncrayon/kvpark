"""resource_guard.py on fake sensor trees and fake children: readings, settings, thread caps, the
decisions, Watch (pause, resume, abort, sensor loss, a pause that never clears, the guard off), the GPU-hang
stop (a fake kernel log), run()
and wait_cool, Guard.check and its background thread, the CPU lock (exclusive, inherited, lock order),
the campaign's commands under the guard and fit_hub.py's memory estimate and refusal."""

import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import resource_guard

HERE = Path(__file__).resolve().parent
GIB = resource_guard.GIB
_ENVIRONMENT = []


def setUpModule():
    """These tests set the guard up themselves: none of the caller's (a suite run under
    `resource_guard.py run` is watched; KVPARK_GUARD_* would change the defaults tested here)."""
    patcher = mock.patch.dict(os.environ)
    patcher.start()
    _ENVIRONMENT.append(patcher)
    for name in list(os.environ):
        if name.startswith("KVPARK_GUARD") or name in ("KVPARK_HWMON_ROOT", resource_guard.CPU_LOCK_FD):
            del os.environ[name]


def tearDownModule():
    _ENVIRONMENT.pop().stop()


class FakeMachine:
    """A fake /sys/class/hwmon (k10temp Tctl and optionally Tccd1, amdgpu edge and PPT, one NVMe) and
    /proc/meminfo; set() changes them, tctl=None removes the CPU sensors (sensor loss)."""

    def __init__(self, root, *, tctl=45.0, ccd=None, edge=40.0, ppt=20.0, nvme=35.0, available_gib=100.0,
                 swap_total_gib=32.0, swap_used_gib=0.0):
        self.root = Path(root)
        self.hwmon = self.root / "hwmon"
        self.meminfo = self.root / "meminfo"
        self.cpu_lock = self.root / "cpu.lock"
        self.values = {}
        for index, name in enumerate(("k10temp", "amdgpu", "nvme")):
            directory = self.hwmon / f"hwmon{index}"
            directory.mkdir(parents=True)
            (directory / "name").write_text(name + "\n")
        (self.hwmon / "hwmon0" / "temp1_label").write_text("Tctl\n")
        (self.hwmon / "hwmon1" / "temp1_label").write_text("edge\n")
        (self.hwmon / "hwmon2" / "temp1_label").write_text("Composite\n")
        self.set(tctl=tctl, ccd=ccd, edge=edge, ppt=ppt, nvme=nvme, available_gib=available_gib,
                 swap_total_gib=swap_total_gib, swap_used_gib=swap_used_gib)

    def set(self, **values):
        self.values.update(values)
        v = self.values
        k10 = self.hwmon / "hwmon0"
        for item in ("temp1_input", "temp3_input", "temp3_label"):
            (k10 / item).unlink(missing_ok=True)
        if v["tctl"] is not None:
            (k10 / "temp1_input").write_text(f"{int(v['tctl'] * 1000)}\n")
            if v["ccd"] is not None:
                (k10 / "temp3_label").write_text("Tccd1\n")
                (k10 / "temp3_input").write_text(f"{int(v['ccd'] * 1000)}\n")
        gpu = self.hwmon / "hwmon1"
        (gpu / "temp1_input").write_text(f"{int(v['edge'] * 1000)}\n") if v["edge"] is not None else \
            (gpu / "temp1_input").unlink(missing_ok=True)
        (gpu / "power1_average").write_text(f"{int(v['ppt'] * 1e6)}\n")
        (self.hwmon / "hwmon2" / "temp1_input").write_text(f"{int(v['nvme'] * 1000)}\n")
        kib = lambda gib: int(gib * 1024 * 1024)  # noqa: E731
        self.meminfo.write_text(f"MemTotal: {kib(128)} kB\nMemFree: {kib(10)} kB\n"
                                f"MemAvailable: {kib(v['available_gib'])} kB\nSwapTotal: {kib(v['swap_total_gib'])} kB\n"
                                f"SwapFree: {kib(v['swap_total_gib'] - v['swap_used_gib'])} kB\n")

    def settings(self, **overrides):
        values = dict(hwmon_root=str(self.hwmon), meminfo=str(self.meminfo), cpu_lock=str(self.cpu_lock), poll_s=0.05,
                      sensor_loss_s=0.5, stop_grace_s=2.0, max_pause_s=0.0, gpu_watch="off",
                      gpu_lock=str(self.root / "gpu.lock"))
        values.update(overrides)
        return resource_guard.Settings(**values)

    def environ(self, **extra):
        return {"KVPARK_HWMON_ROOT": str(self.hwmon), "KVPARK_GUARD_MEMINFO": str(self.meminfo),
                "KVPARK_GUARD_CPU_LOCK": str(self.cpu_lock), "KVPARK_GUARD_GPU_WATCH": "off",
                "KVPARK_GUARD_GPU_LOCK": str(self.root / "gpu.lock"), **extra}


def guard_arguments(root):
    """fit_hub.py's --guard settings for a fit a test runs in-process: a cool fake machine and a CPU lock
    under `root`, so that the test neither waits for this machine nor meets a real job's lock."""
    fake = FakeMachine(Path(root) / "guard-machine")
    return ["--guard", f"hwmon_root={fake.hwmon}", "--guard", f"meminfo={fake.meminfo}",
            "--guard", f"cpu_lock={fake.cpu_lock}", "--guard", "gpu_watch=off", "--guard", f"gpu_lock={fake.root / 'gpu.lock'}"]


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Stopwatch:
    """Notices whether this process was stopped while a timing test ran (an outer guard's SIGSTOP when
    the suite runs under `campaign.py exec` or `resource_guard.py run` on a hot machine): a gap of more
    than `gap` seconds between the ticks of a thread. The commands such a test starts run in their own
    sessions and go on meanwhile, so its timings mean nothing then."""

    def __init__(self, gap=0.5):
        self.gap, self.longest, self._stop = gap, 0.0, threading.Event()

    def _tick(self):
        last = time.monotonic()
        while not self._stop.wait(0.02):
            now = time.monotonic()
            self.longest, last = max(self.longest, now - last), now

    def __enter__(self):
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        return False

    def skip_if_stopped(self, test):
        if self.longest > self.gap:
            test.skipTest(f"this process was stopped for {self.longest:.1f} s while timing (an outer guard's "
                          "pause): rerun the test outside one")


def machine(test, **values):
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    return FakeMachine(directory.name, **values)


def read_log(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def state_of(pid):
    """The process state letter of /proc/PID/stat (T: stopped)."""
    stat = Path(f"/proc/{pid}/stat").read_text()
    return stat[stat.rindex(")") + 2]


def stopped(pid, wait=3.0):
    """Whether process `pid` is (or within `wait` seconds becomes) stopped: signals arrive asynchronously."""
    deadline = time.monotonic() + wait
    while state_of(pid) != "T" and time.monotonic() < deadline:
        time.sleep(0.01)
    return state_of(pid) == "T"


def running(pid, wait=3.0):
    deadline = time.monotonic() + wait
    while state_of(pid) == "T" and time.monotonic() < deadline:
        time.sleep(0.01)
    return state_of(pid) != "T"


def gone(pid, wait=5.0):
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            if state_of(pid) == "Z":
                return True
        except OSError:
            return True
        time.sleep(0.05)
    return False


def sleeper(seconds=30.0, grandchild_file=None):
    """argv of a light child (a sleeping Python); with grandchild_file it starts a sleeping grandchild first
    and writes its pid there."""
    code = "import time\n"
    if grandchild_file:
        code = (f"import subprocess, sys, time\nchild = subprocess.Popen(['sleep', '60'])\n"
                f"open({str(grandchild_file)!r}, 'w').write(str(child.pid))\n")
    return [sys.executable, "-c", code + f"time.sleep({seconds})"]


class SensorTest(unittest.TestCase):
    def test_reads_cpu_gpu_power_nvme_and_memory(self):
        fake = machine(self, tctl=61.5, ccd=66.0, edge=58.0, ppt=95.5, nvme=41.0, available_gib=70.0,
                       swap_total_gib=32.0, swap_used_gib=8.0)
        sample = resource_guard.read_sample(fake.settings())
        self.assertEqual(sample["cpu_c"], {"Tctl": 61.5, "Tccd1": 66.0})
        self.assertEqual((sample["edge_c"], sample["ppt_w"], sample["nvme_c"]), (58.0, 95.5, [41.0]))
        self.assertEqual(sample["package_c"], 66.0)                     # the hottest CPU sensor or the iGPU edge
        fake.set(edge=70.0)
        self.assertEqual(resource_guard.read_sample(fake.settings())["package_c"], 70.0)
        self.assertEqual(sample["mem_available"], 70 * GIB)
        self.assertAlmostEqual(sample["swap_fraction"], 0.25)
        record = resource_guard.sample_record(sample)
        self.assertEqual((record["mem_available_gib"], record["swap_used_gib"], record["swap_pct"]), (70.0, 8.0, 25.0))

    def test_without_a_cpu_sensor_there_is_no_package_temperature(self):
        fake = machine(self, tctl=None, edge=50.0)
        self.assertIsNone(resource_guard.read_sample(fake.settings())["package_c"])     # the iGPU alone does not count
        empty = resource_guard.read_sample(fake.settings(hwmon_root=str(fake.root / "missing"),
                                                         meminfo=str(fake.root / "missing")))
        self.assertIsNone(empty["package_c"])
        self.assertIsNone(empty["mem_available"])
        (fake.hwmon / "hwmon0" / "temp1_label").unlink()                 # an unlabelled k10temp temp1 is Tctl
        fake.set(tctl=50.0)
        self.assertEqual(resource_guard.read_temperatures(fake.hwmon)["cpu"], {"Tctl": 50.0})

    def test_this_machine_has_readable_sensors(self):
        sample = resource_guard.read_sample(resource_guard.Settings())
        if not Path("/sys/class/hwmon").is_dir() or sample["package_c"] is None:
            self.skipTest("no k10temp sensor here")
        self.assertGreater(sample["package_c"], 0)
        self.assertGreater(sample["mem_available"], 0)


class SettingsTest(unittest.TestCase):
    def test_defaults(self):
        s = resource_guard.Settings.from_env({})
        self.assertEqual((s.start_c, s.pause_c, s.resume_c, s.abort_c), (75, 90, 80, 95))
        self.assertEqual((s.swap_pause, s.swap_resume, s.mem_floor_gib, s.mem_resume_gib, s.fit_margin_gib),
                         (0.85, 0.80, 8, 12, 25))
        self.assertEqual((s.poll_s, s.sensor_loss_s, s.stop_grace_s, s.threads, s.enabled, s.abort_samples),
                         (2, 60, 30, 8, True, 2))
        self.assertEqual((s.cpu_lock, s.hwmon_root), ("/tmp/kvpark-campaign-cpu.lock", "/sys/class/hwmon"))
        self.assertEqual(s.gpu_watch, "auto")                           # every Watch reads the kernel log
        self.assertEqual(resource_guard.Settings.from_env({"KVPARK_GUARD_GPU_WATCH": "off"}).gpu_watch, "off")
        with self.assertRaises(ValueError):
            resource_guard.Settings.from_env({}, ["gpu_watch=dmesg"])

    def test_environment_and_command_line_overrides(self):
        s = resource_guard.Settings.from_env({"KVPARK_GUARD_PAUSE_C": "88", "KVPARK_GUARD_THREADS": "4",
                                              "KVPARK_HWMON_ROOT": "/x", "KVPARK_GUARD_MEM_FLOOR_GIB": "6"})
        self.assertEqual((s.pause_c, s.threads, s.hwmon_root, s.mem_floor), (88, 4, "/x", 6 * GIB))
        self.assertFalse(resource_guard.Settings.from_env({"KVPARK_GUARD": "off"}).enabled)
        import argparse
        parser = argparse.ArgumentParser()
        resource_guard.add_arguments(parser)
        args = parser.parse_args(["--guard", "pause_c=85", "--guard", "resume_c=70", "--guard", "off"])
        s = resource_guard.Settings.from_args(args, environ={"KVPARK_GUARD_PAUSE_C": "88"})
        self.assertEqual((s.pause_c, s.resume_c, s.enabled), (85, 70, False))             # the command line wins
        for bad in (["pause_c=79"], ["nonsense=1"], ["pause_c"], ["pause_c=hot"], ["swap_resume=0.9"]):
            with self.assertRaises(ValueError, msg=bad):
                resource_guard.Settings.from_env({}, bad)
        with self.assertRaises(ValueError):
            resource_guard.Settings.from_env({"KVPARK_GUARD": "maybe"})


class ThreadTest(unittest.TestCase):
    def test_heavy_children_get_eight_threads_unless_set(self):
        env = resource_guard.thread_env({"PATH": "/bin", "MKL_NUM_THREADS": "2"})
        self.assertEqual([env[name] for name in resource_guard.THREAD_VARIABLES], ["8", "8", "2", "8", "8"])
        self.assertEqual(resource_guard.thread_env({}, 4)["OPENBLAS_NUM_THREADS"], "4")
        self.assertEqual(resource_guard.thread_env({"KVPARK_GUARD_THREADS": "6"})["OMP_NUM_THREADS"], "6")
        environ = {"OPENBLAS_NUM_THREADS": "16"}
        resource_guard.cap_threads(environ)
        self.assertEqual((environ["OPENBLAS_NUM_THREADS"], environ["OMP_NUM_THREADS"]), ("16", "8"))

    def test_an_explicit_thread_count_caps_what_a_parent_set(self):
        """campaign.py caps its own BLAS at 8 before numpy; a GPU evaluation's children get no more than 4."""
        env = resource_guard.thread_env({"OPENBLAS_NUM_THREADS": "8", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "x"}, 4)
        self.assertEqual((env["OPENBLAS_NUM_THREADS"], env["OMP_NUM_THREADS"], env["MKL_NUM_THREADS"],
                          env["BLIS_NUM_THREADS"]), ("4", "2", "x", "4"))
        environ = {"OPENBLAS_NUM_THREADS": "8"}
        resource_guard.cap_threads(environ, 4)
        self.assertEqual(environ["OPENBLAS_NUM_THREADS"], "4")

    def test_gpu_evaluation_defaults_yield_to_the_environment_and_overrides(self):
        settings = resource_guard.Settings.from_env({}, defaults=resource_guard.GPU_EVALUATION)
        self.assertEqual((settings.pause_c, settings.resume_c, settings.threads, settings.abort_c), (85.0, 65.0, 4, 95.0))
        settings = resource_guard.Settings.from_env({"KVPARK_GUARD_PAUSE_C": "88", "KVPARK_GUARD_THREADS": "6"},
                                                    ["resume_c=70"], defaults=resource_guard.GPU_EVALUATION)
        self.assertEqual((settings.pause_c, settings.resume_c, settings.threads), (88.0, 70.0, 6))
        self.assertEqual(resource_guard.Settings.from_env({}).pause_c, 90.0)        # other jobs keep the defaults

    def test_scripts_cap_their_own_blas_before_numpy(self):
        """fit_hub.py, campaign.py, pair_eval.py and r0_experiments.py started without thread variables
        run numpy's BLAS on 8 threads (fit_hub.py defaulted to 16, OpenBLAS alone to all 32)."""
        environ = {key: value for key, value in os.environ.items() if key not in resource_guard.THREAD_VARIABLES}
        code = ("import runpy, sys, os; sys.argv = [{script!r}, '--help']\n"
                "try:\n    runpy.run_path({script!r}, run_name='__main__')\nexcept SystemExit:\n    pass\n"
                "print('THREADS', os.environ.get('OPENBLAS_NUM_THREADS'), os.environ.get('OMP_NUM_THREADS'))")
        # pair_eval.py is a GPU evaluation: resource_guard.GPU_EVALUATION's 4 threads
        for script, threads in (("fit_hub.py", 8), ("campaign.py", 8), ("pair_eval.py", 4), ("r0_experiments.py", 8)):
            result = subprocess.run([sys.executable, "-c", code.format(script=str(HERE / script))], cwd=HERE,
                                    env=environ, capture_output=True, text=True, timeout=120)
            self.assertIn(f"THREADS {threads} {threads}", result.stdout, (script, result.stderr[-2000:]))

    def test_probe_runs_use_the_guard_thread_count(self):
        import campaign
        import pair_eval
        import r0_experiments
        self.assertEqual(campaign.CAPTURE_THREADS, 8)
        ran = []

        def fake_run(command, **options):
            ran.append(command)
            return types.SimpleNamespace(returncode=0, seconds=1.0, paused_s=0.0, cool_wait_s=0.0)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(resource_guard, "run", fake_run):
            out = Path(directory) / "states"
            pair_eval.probe(Path("/probe"), [Path("/lib/libllama.so")], "save-state", "/m.gguf", [], out, ctx=4096)
            pair_eval.probe(Path("/probe"), [], "save-state", "/m.gguf", [], out, ctx=4096,
                            guard=resource_guard.Settings(threads=4))
            out.mkdir()
            (out / "report.json").write_text("{}")
            model = Path(directory) / "small.gguf"
            model.write_bytes(b"GGUF")
            args = types.SimpleNamespace(probe=Path("/probe"), ctx=4096, device="vulkan", log=Path(directory) / "log",
                                         guard=["threads=6"])
            with contextlib.redirect_stdout(io.StringIO()):
                r0_experiments.probe(args, [], "save-state", model, [], out)
        threads = [command[command.index("--threads") + 1] for command in ran]
        self.assertEqual(threads, ["4", "4", "6"])        # pair_eval: the GPU-evaluation default


class AssessTest(unittest.TestCase):
    def test_decisions(self):
        s = resource_guard.Settings(swap_pause=0.85, swap_resume=0.8)

        def sample(c, available=100, swap=0.1):
            return {"package_c": c, "mem_available": available * GIB, "swap_total": 32 * GIB, "swap_fraction": swap}
        cases = [(sample(70), False, "run"), (sample(89.9), False, "run"), (sample(90), False, "pause"),
                 (sample(95), False, "abort"), (sample(96), True, "abort"), (sample(85), True, "hold"),
                 (sample(80), True, "resume"), (sample(None), False, "lost"), (sample(70, available=7), False, "pause"),
                 (sample(70, available=10), True, "hold"), (sample(70, available=12), True, "resume"),
                 (sample(70, available=20, swap=0.86), False, "pause"), (sample(70, available=20, swap=0.82), True, "hold"),
                 (sample(70, available=20, swap=0.79), True, "resume"),
                 # with MemAvailable at swap_ignore_gib (24) or more, swap use neither stops nor holds a job: it holds
                 # pages swapped out earlier, mostly the stopped job's own, which come back only when it runs
                 (sample(70, available=40, swap=0.99), False, "run"), (sample(70, available=40, swap=0.99), True, "resume"),
                 (sample(70, available=24, swap=0.99), True, "resume"), (sample(70, available=23, swap=0.99), True, "hold"),
                 (sample(85, available=40, swap=0.99), True, "hold")]
        for current, paused, expected in cases:
            self.assertEqual(resource_guard.assess(current, s, paused)[0], expected, (current, paused))
        always = resource_guard.Settings(swap_pause=0.85, swap_resume=0.8, swap_ignore_gib=0)   # as before 2026-10-07
        self.assertEqual(resource_guard.assess(sample(70, swap=0.86), always, False)[0], "pause")
        self.assertEqual(resource_guard.assess(sample(70, swap=0.82), always, True)[0], "hold")
        with self.assertRaises(ValueError):
            resource_guard.Settings(swap_ignore_gib=10)                       # below mem_resume_gib
        self.assertIn("MemAvailable 7.0 GiB < floor 8 GiB", resource_guard.assess(sample(70, available=7), s, False)[1])
        no_swap = {"package_c": 70, "mem_available": 100 * GIB, "swap_total": 0, "swap_fraction": 0.0}
        self.assertEqual(resource_guard.assess(no_swap, s, False)[0], "run")


class WatchTest(unittest.TestCase):
    def start(self, argv):
        child = subprocess.Popen(argv, start_new_session=True)
        self.addCleanup(lambda: gone(child.pid, 0) or resource_guard.stop_group(child, 2))
        return child

    def test_pause_on_heat_and_resume_once_cool(self):
        fake = machine(self)
        log = fake.root / "watch.jsonl"
        child = self.start(sleeper())
        clock = FakeClock()
        watch = resource_guard.Watch(fake.settings(), log_path=log, label="t", clock=clock).attach(child.pid)
        self.assertNotEqual(state_of(child.pid), "T")
        fake.set(tctl=91.0)
        clock.sleep(1)
        watch.step()
        self.assertTrue(stopped(child.pid))                              # the whole group stopped
        fake.set(tctl=85.0)
        clock.sleep(1)
        watch.step()
        self.assertTrue(stopped(child.pid, wait=0.2))                    # still above resume_c
        fake.set(tctl=79.0)
        clock.sleep(1)
        watch.step()
        self.assertTrue(running(child.pid))
        self.assertEqual(watch.paused_seconds(), 2.0)
        clock.sleep(0.01)
        watch.step()                                                     # within poll_s: no sample
        summary = watch.close()
        self.assertEqual((summary["pauses"], summary["paused_s"], summary["max_c"], summary["samples"]), (1, 2.0, 91.0, 4))
        events = read_log(log)
        self.assertEqual([e["event"] for e in events],
                         ["watch", "sample", "sample", "pause", "sample", "sample", "resume", "watch-done"])
        self.assertIn("91.0 C >= pause 90 C", events[3]["reason"])
        self.assertIn("child_rss_gib", events[1])
        self.assertEqual(events[4]["state"], "paused")

    def test_memory_floor_and_swap_pause_the_group(self):
        fake = machine(self)
        child = self.start(sleeper())
        clock = FakeClock()
        watch = resource_guard.Watch(fake.settings(), clock=clock).attach(child.pid)
        for pressure, relief in (({"available_gib": 6.0}, {"available_gib": 20.0}),
                                 ({"swap_used_gib": 28.0, "available_gib": 20.0}, {"swap_used_gib": 20.0}),
                                 ({"swap_used_gib": 28.0, "available_gib": 20.0}, {"available_gib": 30.0})):
            fake.set(**pressure)
            clock.sleep(1)
            watch.step()
            self.assertTrue(stopped(child.pid), pressure)
            fake.set(**relief)
            clock.sleep(1)
            watch.step()
            self.assertTrue(running(child.pid), relief)
            fake.set(available_gib=100.0, swap_used_gib=0.0)
        self.assertEqual(watch.pauses, 3)

    def test_abort_at_the_critical_temperature_ends_the_whole_group(self):
        fake = machine(self)
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "grandchild"
            child = self.start(sleeper(grandchild_file=pidfile))
            deadline = time.monotonic() + 10
            while not (pidfile.is_file() and pidfile.read_text()) and time.monotonic() < deadline:
                time.sleep(0.05)
            watch = resource_guard.Watch(fake.settings(), log_path=Path(directory) / "w.jsonl").attach(child.pid)
            fake.set(tctl=91.0)
            watch.step(force=True)                                        # stopped first ...
            fake.set(tctl=96.0)
            watch.step(force=True)                                        # one critical sample: still stopped
            self.assertTrue(stopped(child.pid, wait=0.2))
            with self.assertRaisesRegex(resource_guard.GuardAbort, "96.0 C >= abort 95 C"):
                watch.step(force=True)                                    # the second in a row ends it
            resource_guard.stop_group(child, 2)                           # ... still ended: SIGTERM + SIGCONT
            self.assertTrue(gone(int(pidfile.read_text())))
            self.assertEqual(read_log(Path(directory) / "w.jsonl")[-1]["event"], "abort")
            self.assertIn("abort", watch.summary()["aborted"])

    def test_a_critical_spike_stops_the_group_and_ends_it_only_if_it_persists(self):
        fake = machine(self)
        child = self.start(sleeper())
        clock = FakeClock()
        watch = resource_guard.Watch(fake.settings(), clock=clock).attach(child.pid)
        fake.set(tctl=96.5)                                               # a spike while running
        clock.sleep(2)
        watch.step()
        self.assertTrue(stopped(child.pid))
        fake.set(tctl=62.0)                                               # gone once the job stopped
        clock.sleep(2)
        watch.step()
        self.assertTrue(running(child.pid))
        self.assertEqual((watch.pauses, watch.aborted), (1, None))
        once = resource_guard.Watch(fake.settings(abort_samples=1), clock=clock).attach(child.pid)
        fake.set(tctl=95.0)
        clock.sleep(2)
        with self.assertRaises(resource_guard.GuardAbort):                # abort_samples=1: at once
            once.step()

    def test_lost_sensors_abort_after_the_limit(self):
        fake = machine(self)
        child = self.start(sleeper())
        clock = FakeClock()
        watch = resource_guard.Watch(fake.settings(sensor_loss_s=60), clock=clock).attach(child.pid)
        fake.set(tctl=None)
        for _ in range(30):                                               # 58 s without a CPU temperature
            clock.sleep(2)
            watch.step()
        fake.set(tctl=50.0)
        clock.sleep(2)
        watch.step()                                                      # back in time: the count starts over
        fake.set(tctl=None)
        clock.sleep(2)
        watch.step()
        clock.sleep(59)
        watch.step()
        clock.sleep(2)
        with self.assertRaisesRegex(resource_guard.GuardAbort, "no CPU temperature readable for 61 s"):
            watch.step()

    def test_a_pause_that_never_clears_ends_the_job(self):
        fake = machine(self)
        child = self.start(sleeper())
        clock = FakeClock()
        watch = resource_guard.Watch(fake.settings(max_pause_s=1800), clock=clock).attach(child.pid)
        fake.set(available_gib=5.0)
        clock.sleep(2)
        watch.step()
        clock.sleep(1799)
        watch.step()
        clock.sleep(2)
        with self.assertRaisesRegex(resource_guard.GuardAbort, "paused for 1801 s and MemAvailable 5.0 GiB"):
            watch.step()

    def test_with_the_guard_off_it_logs_and_warns_but_never_acts(self):
        fake = machine(self, tctl=99.0)
        child = self.start(sleeper())
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            watch = resource_guard.Watch(fake.settings(enabled=False), log_path=fake.root / "off.jsonl").attach(child.pid)
        watch.step(force=True)
        self.assertFalse(stopped(child.pid, wait=0.3))
        self.assertIn("WARNING: the resource guard is OFF", stderr.getvalue())
        self.assertIn("guard-off", [e["event"] for e in read_log(fake.root / "off.jsonl")])


HANG = ("{epoch:.6f} laplace-evo-x2 kernel: amdgpu 0000:c6:00.0: ring gfx_0.0.0 timeout, signaled seq=6167215, "
        "emitted seq=6167217")
RESET = "{epoch:.6f} laplace-evo-x2 kernel: amdgpu 0000:c6:00.0: GPU reset begin!. Source:  1"
QUIET = "{epoch:.6f} laplace-evo-x2 kernel: input: BT+2.4G Mouse as /devices/virtual/misc/uhid/0005:25A7:FAA0.0010"


class FakeKernelLog:
    """A kernel log a test writes lines to (journalctl -o short-unix format): iterating it blocks for
    the next line, yielding "" while idle so that its reader notices close()."""

    def __init__(self):
        import queue
        self.queue = queue.Queue()

    def write(self, template, epoch=None):
        self.queue.put(template.format(epoch=time.time() if epoch is None else epoch))

    def __iter__(self):
        import queue
        while True:
            try:
                yield self.queue.get(timeout=0.02)
            except queue.Empty:
                yield ""

    def kernel_log(self, since=None):
        return resource_guard.KernelLog(since, source=self)


def until(predicate, wait=5.0):
    deadline = time.monotonic() + wait
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


class GpuHangTest(unittest.TestCase):
    """The GPU-hang stop: amdgpu ring timeouts or resets in the kernel log from a job's start end it."""

    def start(self, argv):
        child = subprocess.Popen(argv, start_new_session=True)
        self.addCleanup(lambda: gone(child.pid, 0) or resource_guard.stop_group(child, 2))
        return child

    def test_what_counts_as_a_gpu_hang(self):
        for line in (HANG, RESET, "amdgpu 0000:c6:00.0: Starting gfx_0.0.0 ring reset",
                     "amdgpu 0000:c6:00.0: Ring gfx_0.0.0 reset succeeded",
                     "amdgpu 0000:c6:00.0: [drm] device wedged, but recovered through reset",
                     "amdgpu 0000:c6:00.0: MES failed to respond to msg=RESET"):
            self.assertTrue(resource_guard.gpu_hang_line(line.format(epoch=1.0)), line)
        for line in (QUIET, "amdgpu 0000:c6:00.0: Dumping IP State", "xhci_hcd 0000:c4:00.4: ring expansion succeeded",
                     "usb 1-2: device descriptor read timeout, GPU reset not involved"):
            self.assertFalse(resource_guard.gpu_hang_line(line.format(epoch=1.0)), line)

    def test_the_kernel_log_is_read_from_the_jobs_start(self):
        fake = FakeKernelLog()
        since = time.time()
        fake.write(HANG, epoch=since - 30)                                # an earlier hang: not this job's
        fake.write(QUIET)
        log = fake.kernel_log(since).start()
        try:
            self.assertTrue(until(lambda: fake.queue.empty()))
            time.sleep(0.05)
            self.assertEqual(log.poll(), [])
            fake.write(RESET)
            fake.write(HANG)
            self.assertTrue(until(lambda: len(log.hits) == 2))
            hits = log.poll()
            self.assertEqual([("GPU reset" in h["line"], "timeout" in h["line"]) for h in hits],
                             [(True, False), (False, True)])
            self.assertEqual(log.poll(), [])                              # each hit is returned once
            self.assertEqual(log.describe()["source"], "lines")
        finally:
            log.close()

    def test_a_gpu_hang_ends_the_watched_group_like_an_abort(self):
        fake_machine = machine(self)
        log = fake_machine.root / "hang.jsonl"
        fake = FakeKernelLog()
        child = self.start(sleeper())
        watch = resource_guard.Watch(fake_machine.settings(), log_path=log, label="gpu",
                                     kernel_log=fake.kernel_log()).attach(child.pid)
        watch.step(force=True)
        fake.write(QUIET)
        fake.write(HANG)
        with self.assertRaisesRegex(resource_guard.GuardAbort, "GPU hang in the kernel log: .*ring gfx_0.0.0 timeout"):
            until(lambda: watch.kernel.hits)
            watch.step()                                                  # every step reads the log, not every poll_s
        resource_guard.stop_group(child, 2)
        self.assertTrue(gone(child.pid))
        summary = watch.close()
        self.assertIn("ring gfx_0.0.0 timeout", summary["gpu_hang"])
        self.assertIn("GPU hang", summary["aborted"])
        events = read_log(log)
        self.assertEqual([e["event"] for e in events][:2], ["watch", "gpu-watch"])
        hang = next(e for e in events if e["event"] == "gpu-hang")
        self.assertTrue(hang["acted"])
        abort = next(e for e in events if e["event"] == "abort")
        self.assertIn("ring gfx_0.0.0 timeout", abort["gpu_hang"])
        self.assertIsNone(watch.kernel._process)                          # nothing left following the journal

    def test_with_the_guard_off_a_hang_is_logged_only(self):
        fake_machine = machine(self)
        log = fake_machine.root / "off.jsonl"
        fake = FakeKernelLog()
        child = self.start(sleeper())
        with contextlib.redirect_stderr(io.StringIO()):
            watch = resource_guard.Watch(fake_machine.settings(enabled=False), log_path=log,
                                         kernel_log=fake.kernel_log()).attach(child.pid)
        fake.write(HANG)
        until(lambda: watch.kernel.hits)
        watch.step(force=True)
        self.assertFalse(stopped(child.pid, wait=0.2))
        watch.close()
        hang = next(e for e in read_log(log) if e["event"] == "gpu-hang")
        self.assertFalse(hang["acted"])

    def test_run_stops_a_command_at_a_gpu_hang(self):
        fake_machine = machine(self)
        fake = FakeKernelLog()
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "grandchild"
            threading.Thread(target=lambda: (until(pidfile.is_file), time.sleep(0.2), fake.write(HANG)),
                             daemon=True).start()
            with self.assertRaisesRegex(resource_guard.GuardAbort, "GPU hang"):
                resource_guard.run(sleeper(30, pidfile), settings=fake_machine.settings(), kernel_log=fake.kernel_log(),
                                   log_path=Path(directory) / "run.jsonl")
            self.assertTrue(gone(int(pidfile.read_text())))
            self.assertIn("gpu-hang", [e["event"] for e in read_log(Path(directory) / "run.jsonl")])

    def test_an_in_process_gpu_fit_stops_at_check(self):
        fake_machine = machine(self)
        log = fake_machine.root / "guard.jsonl"
        fake = FakeKernelLog()
        clock = FakeClock()
        with self.assertRaisesRegex(resource_guard.GuardAbort, "GPU hang"):
            with resource_guard.Guard(fake_machine.settings(), log_path=log, clock=clock, sleep=clock.sleep,
                                      background=False, environ={}, gpu=True, kernel_log=fake.kernel_log()) as guard:
                guard.check("chunk")
                fake.write(RESET)
                until(lambda: guard.kernel.hits)
                guard.check("chunk")                                     # within poll_s: still read
        events = read_log(log)
        self.assertIn("gpu-hang", [e["event"] for e in events])
        self.assertIn("GPU reset", events[-1]["gpu_hang"])                # guard-done records it
        self.assertEqual(next(e for e in events if e["event"] == "gpu-hang")["where"], "chunk")

    def test_a_gpu_call_that_never_returns_is_ended_by_the_background_thread(self):
        fake_machine = machine(self)
        fake = FakeKernelLog()
        critical = []
        with resource_guard.Guard(fake_machine.settings(), on_critical=critical.append, environ={}, gpu=True,
                                  kernel_log=fake.kernel_log()):
            fake.write(HANG)                                              # no check() comes
            self.assertTrue(until(lambda: critical))
        self.assertIn("GPU hang in the kernel log outside check()", critical[0])

    def test_under_an_outer_watch_the_watch_reads_the_log(self):
        fake_machine = machine(self)
        with resource_guard.Guard(fake_machine.settings(gpu_watch="auto"), environ={resource_guard.WATCHED: "1"},
                                  gpu=True) as guard:
            self.assertIsNone(guard.kernel)
        with resource_guard.Guard(fake_machine.settings(gpu_watch="auto"), environ={}, background=False) as guard:
            self.assertIsNone(guard.kernel)                               # a CPU fit: no kernel log

    def test_an_unreadable_kernel_log_is_logged_and_warned_about(self):
        fake_machine = machine(self)
        stderr = io.StringIO()
        with mock.patch.object(resource_guard, "journal_readable", lambda: False), \
                mock.patch.object(resource_guard.os, "access", lambda path, mode: False), \
                contextlib.redirect_stderr(stderr):
            child = self.start(sleeper())
            watch = resource_guard.Watch(fake_machine.settings(gpu_watch="auto"), log_path=fake_machine.root / "w.jsonl",
                                         label="blind").attach(child.pid)
            watch.close()
        self.assertIn("WARNING: no GPU-hang stop for blind", stderr.getvalue())
        event = next(e for e in read_log(fake_machine.root / "w.jsonl") if e["event"] == "gpu-watch")
        self.assertEqual(event["source"], "unavailable")
        self.assertIn("cannot read the kernel log", event["error"])

    def test_this_machine_follows_its_kernel_log(self):
        if not resource_guard.journal_readable():
            self.skipTest("journalctl -k shows nothing to this user")
        log = resource_guard.KernelLog().start()
        try:
            self.assertEqual(log.source, "journal")
            self.assertIsNotNone(log._process)
            time.sleep(0.5)
            self.assertIsNone(log._process.poll())                        # following
        finally:
            log.close()
        self.assertIsNotNone(log._process.poll())


class RunTest(unittest.TestCase):
    def heat(self, fake, steps):
        """Change the fake machine on a timer: [(delay seconds, values)]."""
        def run():
            for delay, values in steps:
                time.sleep(delay)
                fake.set(**values)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def test_a_paused_command_finishes_and_its_pause_is_not_counted(self):
        fake = machine(self)
        log = fake.root / "run.jsonl"
        self.heat(fake, [(0.2, {"tctl": 92.0}), (1.0, {"tctl": 70.0})])
        started = time.monotonic()
        outcome = resource_guard.run(sleeper(0.8), settings=fake.settings(), log_path=log, label="t", timeout=1.6)
        wall = time.monotonic() - started
        self.assertEqual((outcome.returncode, outcome.timed_out), (0, False))
        self.assertGreater(outcome.paused_s, 0.6)
        self.assertLess(outcome.seconds, wall - 0.5)
        events = [e["event"] for e in read_log(log)]
        self.assertEqual(events.count("pause"), 1)
        self.assertEqual(events.count("resume"), 1)

    def test_a_critical_temperature_stops_the_command_and_raises(self):
        fake = machine(self)
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "grandchild"
            self.heat(fake, [(0.6, {"tctl": 97.0})])
            with self.assertRaises(resource_guard.GuardAbort):
                resource_guard.run(sleeper(30, pidfile), settings=fake.settings())
            self.assertTrue(gone(int(pidfile.read_text())))

    def test_failures_timeouts_and_the_cpu_lock(self):
        fake = machine(self)
        with self.assertRaises(subprocess.CalledProcessError):
            resource_guard.run([sys.executable, "-c", "raise SystemExit(3)"], settings=fake.settings(), check=True)
        outcome = resource_guard.run(sleeper(30), settings=fake.settings(), timeout=0.3)
        self.assertTrue(outcome.timed_out)
        code = ("import fcntl, os; fd = int(os.environ['KVPARK_CPU_LOCK_FD']); fcntl.flock(fd, fcntl.LOCK_EX | "
                "fcntl.LOCK_NB); print(os.environ['OPENBLAS_NUM_THREADS'], os.environ['KVPARK_GUARD_WATCHED'])")
        environ = {key: value for key, value in os.environ.items() if key not in resource_guard.THREAD_VARIABLES}
        with tempfile.TemporaryFile("w+") as out:
            outcome = resource_guard.run([sys.executable, "-c", code], settings=fake.settings(threads=3), cpu=True,
                                         stdout=out, env=environ)
            out.seek(0)
            self.assertEqual((outcome.returncode, out.read().split()), (0, ["3", "1"]))

    def test_under_an_outer_watch_the_command_stays_in_the_process_group(self):
        fake = machine(self, tctl=99.0)                                    # no wait_cool, no Watch of its own
        with tempfile.TemporaryFile("w+") as out, mock.patch.dict(os.environ, {resource_guard.WATCHED: "1"}):
            outcome = resource_guard.run([sys.executable, "-c", "import os; print(os.getpgid(0))"],
                                         settings=fake.settings(), stdout=out)
            out.seek(0)
            self.assertEqual(int(out.read()), os.getpgid(0))
        self.assertEqual((outcome.returncode, outcome.paused_s, outcome.summary), (0, 0.0, None))

    def test_wait_cool(self):
        fake = machine(self, tctl=82.0)
        clock = FakeClock()
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            clock.sleep(seconds)
            if len(sleeps) == 3:
                fake.set(tctl=74.0)
        log = resource_guard.JsonLog(fake.root / "cool.jsonl")
        waited = resource_guard.wait_cool(fake.settings(poll_s=2.0), log=log, clock=clock, sleep=sleep)
        self.assertEqual(waited, 6.0)
        self.assertEqual([e["event"] for e in read_log(fake.root / "cool.jsonl")], ["cool-wait", "cool"])
        fake.set(tctl=60.0, available_gib=10.0)                             # memory short: waits too
        with self.assertRaisesRegex(resource_guard.GuardError, "did not cool"):
            resource_guard.wait_cool(fake.settings(max_wait_s=10), clock=clock, sleep=clock.sleep)
        fake.set(tctl=None, available_gib=100.0)
        with self.assertRaisesRegex(resource_guard.GuardError, "no CPU temperature"):
            resource_guard.wait_cool(fake.settings(sensor_loss_s=60), clock=clock, sleep=clock.sleep)
        fake.set(tctl=99.0)
        self.assertEqual(resource_guard.wait_cool(fake.settings(enabled=False), clock=clock, sleep=clock.sleep), 0.0)

    def test_command_line(self):
        fake = machine(self)
        environ = {**os.environ, **fake.environ(KVPARK_GUARD_POLL_S="0.05")}
        log = fake.root / "cli.jsonl"
        result = subprocess.run([sys.executable, str(HERE / "resource_guard.py"), "run", "--log", str(log), "--",
                                 sys.executable, "-c", "raise SystemExit(4)"], env=environ, capture_output=True,
                                text=True, timeout=60)
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertIn("watch-done", [e["event"] for e in read_log(log)])
        threading.Thread(target=lambda: (time.sleep(1.0), fake.set(tctl=97.0)), daemon=True).start()
        result = subprocess.run([sys.executable, str(HERE / "resource_guard.py"), "run", "--", "sleep", "30"],
                                env=environ, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, resource_guard.ABORT_EXIT, result.stderr)
        result = subprocess.run([sys.executable, str(HERE / "resource_guard.py"), "status"], env=environ,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(json.loads(result.stdout)["sample"]["package_c"], 97.0)


class GuardTest(unittest.TestCase):
    def test_check_sleeps_while_hot_and_returns_once_cool(self):
        fake = machine(self)
        clock = FakeClock()
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            clock.sleep(seconds)
            if len(sleeps) == 4:
                fake.set(tctl=78.0)
        log = fake.root / "guard.jsonl"
        with resource_guard.Guard(fake.settings(poll_s=2.0), log_path=log, clock=clock, sleep=sleep, background=False,
                                  environ={}) as guard:
            guard.check("a")
            fake.set(tctl=90.5)
            guard.check("b")                                             # within poll_s: not even sampled
            self.assertEqual(sleeps, [])
            clock.sleep(2)
            guard.check("b")
            self.assertEqual(len(sleeps), 4)
            self.assertEqual((guard.pauses, guard.paused_total), (1, 8.0))
        events = read_log(log)
        self.assertEqual([e["event"] for e in events].count("sample"), 6)
        pause = next(e for e in events if e["event"] == "pause")
        self.assertEqual((pause["where"], pause["package_c"]), ("b", 90.5))
        self.assertIn("rss_gib", events[1])
        self.assertEqual(events[-1]["event"], "guard-done")

    def test_check_raises_when_critical_or_blind(self):
        fake = machine(self, tctl=95.5)
        clock = FakeClock()
        with resource_guard.Guard(fake.settings(), clock=clock, sleep=clock.sleep, background=False, environ={}) as guard:
            with self.assertRaisesRegex(resource_guard.GuardAbort, "95.5 C >= abort"):
                guard.check()
            fake.set(tctl=91.0)
            clock.sleep(1)
            with self.assertRaisesRegex(resource_guard.GuardAbort, "abort"):
                def heat(seconds):
                    clock.sleep(seconds)
                    fake.set(tctl=96.0)
                guard._sleep = heat
                guard.check()                                            # paused, then critical
            fake.set(tctl=None)
            clock.sleep(1)
            guard.check()
            clock.sleep(1)
            with self.assertRaisesRegex(resource_guard.GuardAbort, "no CPU temperature"):
                guard.check()

    def test_under_an_outer_watch_check_only_aborts(self):
        fake = machine(self, tctl=93.0)
        clock = FakeClock()
        log = fake.root / "watched.jsonl"
        with resource_guard.Guard(fake.settings(), log_path=log, clock=clock, sleep=self.fail,
                                  environ={resource_guard.WATCHED: "1"}) as guard:
            self.assertEqual(guard.wait_cool(), 0.0)
            guard.check()                                                # no sleep: the outer Watch stops us
            self.assertIsNone(guard._thread)
            fake.set(tctl=97.0)
            clock.sleep(1)
            guard.check()                                                # one critical sample
            clock.sleep(1)
            with self.assertRaises(resource_guard.GuardAbort):
                guard.check()                                            # two in a row
        self.assertNotIn("sample", [e["event"] for e in read_log(log)])

    def test_the_background_thread_logs_and_ends_the_job_outside_check(self):
        fake = machine(self)
        log = fake.root / "background.jsonl"
        critical = []
        with resource_guard.Guard(fake.settings(), log_path=log, on_critical=critical.append, environ={}):
            time.sleep(0.3)
            fake.set(tctl=96.0)                                           # as in a long LAPACK call: no check()
            deadline = time.monotonic() + 5
            while not critical and time.monotonic() < deadline:
                time.sleep(0.02)
        self.assertEqual(len(critical), 1)
        self.assertIn("outside check()", critical[0])
        events = read_log(log)
        self.assertGreaterEqual([e["event"] for e in events].count("sample"), 3)
        self.assertGreater(next(e for e in events if e["event"] == "sample")["rss_gib"], 0)
        self.assertIn("abort", [e["event"] for e in events])

    def test_hub_fit_loops_call_the_hook(self):
        import hub
        from test_hub import make_views
        views, spans, _ = make_views(n=1200)
        calls = []
        hub.guard_check = calls.append
        try:
            fitted = hub.fit_hub_kind(views[:2], spans, "k", layer_rank=8, view_dim=12, hub_dim=6, chunk=300,
                                      log=lambda line: None)
            fitted_calls = list(calls)
            hub.add_model_kind(fitted, views[2], views[:2], spans, "k", layer_rank=8, view_dim=12, chunk=300,
                               log=lambda line: None)
            hub.evaluate(fitted, views, spans, "k", chunk=300)
        finally:
            hub.guard_check = hub._no_check
        for where in ("view", "gcca", "decoder"):
            self.assertIn(where, fitted_calls)
        self.assertIn("onboard", calls)
        self.assertIn("evaluate", calls)


class LockTest(unittest.TestCase):
    def test_the_cpu_lock_is_exclusive_and_waits_name_the_holder(self):
        fake = machine(self)
        holder = subprocess.Popen([sys.executable, "-c", (
            "import resource_guard, sys, time\n"
            f"with resource_guard.cpu_lock({str(fake.cpu_lock)!r}, label='other job'):\n"
            "    print('held', flush=True); time.sleep(0.8)")], cwd=HERE, stdout=subprocess.PIPE, text=True)
        self.assertEqual(holder.stdout.readline().strip(), "held")
        events = []
        with resource_guard.cpu_lock(fake.cpu_lock, log=events.append, environ={}) as held:
            self.assertGreater(held.waited, 0.3)
            with open(fake.cpu_lock, "a") as other, self.assertRaises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue(resource_guard.holds_cpu_lock())
            with resource_guard.cpu_lock(fake.cpu_lock, environ={}) as again:  # re-entrant in a process
                self.assertTrue(again.inherited)
        holder.wait()
        holder.stdout.close()
        self.assertFalse(resource_guard.holds_cpu_lock())
        self.assertEqual([e["event"] for e in events], ["cpu-lock-wait", "cpu-lock"])
        self.assertIn("other job", events[0]["holder"])
        with open(fake.cpu_lock, "a") as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)                   # released

    def test_a_child_handed_the_descriptor_holds_it_without_waiting(self):
        fake = machine(self)
        code = ("import resource_guard, sys\n"
                f"with resource_guard.cpu_lock({str(fake.cpu_lock)!r}) as held:\n    print(held.inherited)\n")
        with resource_guard.cpu_lock(fake.cpu_lock, environ={}) as held:
            result = subprocess.run([sys.executable, "-c", code], cwd=HERE, pass_fds=(held.fd,), capture_output=True,
                                    text=True, timeout=20, env={**os.environ, resource_guard.CPU_LOCK_FD: str(held.fd)})
            self.assertEqual(result.stdout.strip(), "True", result.stderr)
            with open(fake.cpu_lock, "a") as other, self.assertRaises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)            # the child did not unlock ours
            # a descriptor of another file is not the lock
            with open(fake.root / "other", "w") as stray:
                self.assertIsNone(resource_guard._inherited(str(fake.cpu_lock), {resource_guard.CPU_LOCK_FD:
                                                                                str(stray.fileno())}))

    def test_the_guards_gpu_lock_is_exclusive_and_lent_to_a_child_handed_its_descriptor(self):
        fake = machine(self)
        gpu = fake.root / "gpu.lock"
        with resource_guard.cpu_lock(fake.cpu_lock, environ={}):
            with self.assertRaises(resource_guard.LockOrderError):
                with resource_guard.gpu_lock(gpu, environ={}):
                    pass
        events = []
        with resource_guard.gpu_lock(gpu, environ={}, log=events.append) as held:
            self.assertFalse(held.inherited)
            with open(gpu, "a") as other, self.assertRaises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            code = ("import sys, resource_guard\nwith resource_guard.gpu_lock(sys.argv[1], environ={}) as h:\n"
                    "    print(h.inherited)")
            environ = {k: v for k, v in os.environ.items() if not k.startswith("KVPARK_GUARD")}
            lent = subprocess.run([sys.executable, "-c", code, str(gpu)], pass_fds=(held.fd,), cwd=HERE, env=environ,
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(lent.stdout.strip(), "True", lent.stderr)
            with self.assertRaises(subprocess.TimeoutExpired):            # without the descriptor: it waits
                subprocess.run([sys.executable, "-c", code, str(gpu)], cwd=HERE, env=environ, capture_output=True,
                               text=True, timeout=2)
        self.assertEqual(events[0]["event"], "gpu-lock")
        with open(gpu, "a") as other:                                    # released
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_the_gpu_lock_comes_first(self):
        import campaign
        fake = machine(self)
        gpu = fake.root / "gpu.lock"
        with resource_guard.cpu_lock(fake.cpu_lock, environ={}):
            with self.assertRaises(resource_guard.LockOrderError):
                with campaign.gpu_lock(0, lambda item: None, path=gpu, available=lambda: 100 * GIB):
                    pass
        with mock.patch.object(campaign, "GPU_LOCK", gpu), mock.patch.object(campaign, "CPU_LOCK", fake.cpu_lock), \
                mock.patch.object(campaign, "mem_available", lambda: 100 * GIB):
            with campaign.guarded_locks(0, lambda item: None, cpu=True) as (held, fds, extra):
                self.assertEqual(len(fds), 2)
                self.assertEqual(extra, {resource_guard.CPU_LOCK_FD: str(fds[1])})
                for path in (gpu, fake.cpu_lock):
                    with open(path, "a") as other, self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


class CampaignGuardTest(unittest.TestCase):
    """campaign.py's commands under the guard: locks, thread caps, the Watch, pauses and aborts."""

    def setUp(self):
        import campaign
        self.campaign = campaign
        self.fake = machine(self)
        patches = [mock.patch.dict(os.environ, self.fake.environ(KVPARK_GUARD_POLL_S="0.05")),
                   mock.patch.object(campaign, "GPU_LOCK", self.fake.root / "gpu.lock"),
                   mock.patch.object(campaign, "CPU_LOCK", self.fake.cpu_lock),
                   mock.patch.object(campaign, "mem_available", lambda: 100 * GIB),
                   contextlib.redirect_stdout(io.StringIO())]
        for patcher in patches:
            patcher.__enter__()
            self.addCleanup(patcher.__exit__, None, None, None)
        gguf = self.fake.root / "model.gguf"
        gguf.write_bytes(b"GGUF")
        self.camp = campaign.Campaign("m", directory=self.fake.root / "m", gguf=gguf, catalog=None)
        self.camp.save()

    def test_a_cpu_heavy_command_holds_both_locks_with_capped_threads_under_a_watch(self):
        code = ("import fcntl, os, time; fd = int(os.environ['KVPARK_CPU_LOCK_FD']); "
                "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); time.sleep(0.3); "
                "print(os.environ['OPENBLAS_NUM_THREADS'], os.environ['KVPARK_GUARD_WATCHED'], os.environ['MKL_NUM_THREADS'])")
        with mock.patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "", "MKL_NUM_THREADS": "3"}), Stopwatch() as watch:
            self.camp.run([sys.executable, "-c", code], stage="onboard", label="fit", memory_bytes=10, cpu=True,
                          cwd=self.fake.root)
        watch.skip_if_stopped(self)
        self.assertEqual(self.camp.path("logs", "fit.log").read_text().split(), ["8", "1", "3"])
        entry = self.camp.state["stages"]["onboard"]["commands"][0]
        self.assertEqual((entry["cpu_lock"], entry["gpu_lock"], entry["threads"], entry["paused_s"],
                          entry["guard_abort"]), (True, True, "8", 0.0, None))
        events = read_log(entry["guard_log"])
        self.assertGreaterEqual([e["event"] for e in events].count("sample"), 2)
        self.assertEqual(events[-1]["event"], "watch-done")
        self.assertIn("exec --memory-bytes 10 --cpu-lock --guard-log", self.camp.path("commands.sh").read_text())
        for path in (self.fake.root / "gpu.lock", self.fake.cpu_lock):         # both released afterwards
            with open(path, "a") as other:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_pauses_count_neither_as_command_time_nor_toward_the_timeout(self):
        fake = self.fake

        def heat():
            time.sleep(0.3)
            fake.set(tctl=91.0)
            time.sleep(1.2)
            fake.set(tctl=60.0)
        heating = threading.Thread(target=heat, daemon=True)
        heating.start()
        self.addCleanup(heating.join)                                     # before the fake machine goes
        started = time.monotonic()
        # 1.2 s: shorter than the wall time, longer than the command ran unpaused
        with Stopwatch() as watch:
            seconds = self.camp.run(sleeper(1.0), stage="tables", label="slow", cwd=fake.root, timeout=1.2, cpu=True)
        wall = time.monotonic() - started
        watch.skip_if_stopped(self)
        entry = self.camp.state["stages"]["tables"]["commands"][0]
        self.assertFalse(entry["timed_out"])
        self.assertGreater(entry["paused_s"], 0.9)
        self.assertLess(seconds, wall - 0.9)
        self.assertEqual(entry["max_c"], 91.0)
        events = [json.loads(line)["event"] for line in self.camp.path("log.jsonl").read_text().splitlines()]
        self.assertIn("pause", events)
        self.assertIn("resume", events)

    def test_the_guard_ends_a_command_and_fails_its_stage(self):
        self.fake.set(tctl=70.0)
        threading.Thread(target=lambda: (time.sleep(0.5), self.fake.set(tctl=96.0)), daemon=True).start()
        with self.assertRaisesRegex(self.campaign.GuardStop, "ended by the resource guard: guard: package 96.0 C"):
            self.camp.run(sleeper(30), stage="evaluate", label="pair", memory_bytes=10, cwd=self.fake.root, cpu=True)
        entry = self.camp.state["stages"]["evaluate"]["commands"][0]
        self.assertIn("abort", entry["guard_abort"])

    def test_a_hot_machine_delays_the_start(self):
        self.fake.set(tctl=80.0)
        threading.Thread(target=lambda: (time.sleep(0.4), self.fake.set(tctl=70.0)), daemon=True).start()
        self.camp.run([sys.executable, "-c", "pass"], stage="render", label="cooled", cwd=self.fake.root)
        entry = self.camp.state["stages"]["render"]["commands"][0]
        self.assertGreater(entry["cool_wait_s"], 0.2)
        self.assertLess(entry["seconds"], entry["cool_wait_s"] + 5)

    def test_exec_runs_under_both_locks_and_the_guard(self):
        log = self.fake.root / "exec.jsonl"
        code = "import fcntl, os; fcntl.flock(int(os.environ['KVPARK_CPU_LOCK_FD']), fcntl.LOCK_EX | fcntl.LOCK_NB)"
        with mock.patch.object(self.campaign.signal, "signal"), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.campaign.main(["exec", "--memory-bytes", "0", "--cpu-lock", "--guard-log", str(log),
                                                 "--", sys.executable, "-c", code]), 0)
            threading.Thread(target=lambda: (time.sleep(0.5), self.fake.set(tctl=97.0)), daemon=True).start()
            self.assertEqual(self.campaign.main(["exec", "--memory-bytes", "0", "--", "sleep", "30"]),
                             resource_guard.ABORT_EXIT)
        self.assertIn("watch-done", [e["event"] for e in read_log(log)])

    def test_cpu_heavy_stages_ask_for_the_cpu_lock(self):
        asked = []

        def fake_run(camp, argv, **options):
            asked.append((options["label"], options.get("cpu", False)))
            output = Path(argv[argv.index("--output") + 1])
            output.mkdir(parents=True)
            (output / "report.json").write_text("{}")
        self.camp.path("eval").mkdir()
        with mock.patch.object(self.campaign.Campaign, "run", fake_run):
            self.campaign.run_evaluation(self.camp, "a-to-b", ["python", "pair_eval.py", "--output",
                                                               str(self.camp.path("eval", "a-to-b"))],
                                         self.camp.path("eval", "a-to-b"), 10)
        self.assertEqual(asked, [("pair_eval-a-to-b", True)])
        source = (HERE / "campaign.py").read_text()
        for label in ('label="fit_hub", cpu=True', 'label="token_tables", memory_bytes=TABLES_MEMORY, cpu=True',
                      'label="calibrate", memory_bytes=campaign.gguf.stat().st_size, cpu=True',
                      'cpu=device == "cpu"'):         # a capture on the CPU device is CPU-heavy (Track B, 2026-10-06)
            self.assertIn(label, source)


class FitMemoryTest(unittest.TestCase):
    def test_onboarding_estimate_is_the_campaigns_and_covers_the_recorded_peaks(self):
        import campaign
        import fit_hub
        self.assertEqual((fit_hub.ONBOARD_BASE, fit_hub.ONBOARD_COVARIANCE_BYTES, fit_hub.ONBOARD_DECODER_BYTES,
                          fit_hub.ONBOARD_DEV_LATENT_BYTES, fit_hub.ONBOARD_DEV_COLUMN_BYTES),
                         (campaign.FIT_HUB_BASE, campaign.FIT_HUB_COVARIANCE_BYTES, campaign.FIT_HUB_DECODER_BYTES,
                          campaign.FIT_HUB_DEV_LATENT_BYTES, campaign.FIT_HUB_DEV_COLUMN_BYTES))
        base = int(4.58 * GIB)
        pivot = fit_hub.view_shape([1024] * 16, 512, 4096)
        # (per-layer widths of K, dev spans of the onboarding, the recorded peak in GiB)
        for widths, dev, measured in (([512] * 8, 9945, 11.0),                # LFM2-2.6B: 8 layers of 8 x 64
                                      ([576] * 27, 8010, 16.7),               # DeepSeek-V2-Lite: MLA K rows
                                      ([512] * 10, 17355, 12.96)):            # Qwen3.6-35B-A3B: 10 of 2 x 256
            shapes = {"k": [fit_hub.view_shape(widths, 512, 4096)] + [pivot] * 4}
            estimate = fit_hub.memory_estimate(shapes, hub_dim=4096, dev_spans=dev, base_bytes=base)
            self.assertEqual(estimate["mode"], "onboard")
            self.assertGreater(estimate["bytes"], measured * GIB)
            self.assertLess(estimate["bytes"], (measured + 1.5) * GIB)

    def test_a_joint_refit_estimate_follows_top_eigs_path(self):
        """Seven views of 4,096 (P = 28,672). The GCCA holds C and W C W (EIGH_HELD P x P float64) plus what
        fit_backend.top_eig allocates on the path it takes: with scipy (the ROCm venv the fits run in) the
        float32 subset solver, about one more P x P float64, so the decoders' phase is the peak (the estimate
        until 2026-10-07 counted numpy's float64 eigh, 4.1 more: 46 GiB); without scipy numpy's eigh of the
        float32 copy. The fitted K kind stays held while V is fitted."""
        import fit_hub
        widths = {"qwen38": [1024] * 16, "muse": [256] * 52, "qwen3-4b": [1024] * 36, "ornith": [512] * 8,
                  "lfm2": [512] * 8, "dsv2": [576] * 27, "qwen36": [512] * 10}
        shapes = {kind: [fit_hub.view_shape(w, 512, 4096) for w in widths.values()] for kind in ("k", "v")}
        P, D = 28672, 4096
        with_scipy = fit_hub.memory_estimate(shapes, hub_dim=D, dev_spans=7965, scipy=True)
        self.assertEqual(with_scipy["mode"], "joint")
        gcca = (fit_hub.EIGH_HELD + fit_hub.EIGH_SUBSET32_COPIES) * 8 * P * P + 12 * P * D
        self.assertAlmostEqual(with_scipy["kinds"]["k"]["gcca"], gcca / GIB, places=1)
        self.assertEqual(with_scipy["phase"], "v decoder")
        self.assertGreater(with_scipy["kinds"]["k"]["decoder"], with_scipy["kinds"]["k"]["gcca"])
        self.assertLess(with_scipy["bytes"], 40 * GIB)
        without = fit_hub.memory_estimate(shapes, hub_dim=D, dev_spans=7965, scipy=False)
        self.assertAlmostEqual(without["kinds"]["k"]["gcca"],
                               ((fit_hub.EIGH_HELD + fit_hub.EIGH_FULL32_COPIES) * 8 * P * P + 12 * P * D) / GIB, places=1)
        self.assertGreater(without["bytes"], with_scipy["bytes"])
        # the paths of fit_backend.top_eig: a small or a non-subset problem is numpy's float64 eigh
        self.assertEqual(fit_hub.eigh_bytes(2048, 512, scipy=True), int(fit_hub.EIGH_FULL_COPIES * 8 * 2048 ** 2
                                                                         + 12 * 2048 * 512))
        self.assertEqual(fit_hub.eigh_bytes(8192, 8192, scipy=True), int(fit_hub.EIGH_FULL_COPIES * 8 * 8192 ** 2
                                                                         + 12 * 8192 ** 2))
        self.assertEqual(fit_hub.eigh_bytes(P, D, scipy=False, precision="float64"),
                         int(fit_hub.EIGH_FULL_COPIES * 8 * P * P + 12 * P * D))
        self.assertEqual(fit_hub.eigh_bytes(P, D, scipy=True, precision="float64"),
                         int(fit_hub.EIGH_SUBSET64_COPIES * 8 * P * P + 12 * P * D))
        self.assertEqual(fit_hub.EIGH_COPIES, fit_hub.EIGH_HELD + fit_hub.EIGH_FULL_COPIES)

    @unittest.skipUnless(__import__("fit_backend")._scipy_linalg() is not None, "scipy: the subset solver's path")
    def test_the_float32_subset_eigh_holds_what_the_estimate_counts(self):
        """EIGH_SUBSET32_COPIES: fit_backend.top_eig of a 4,096-square float64 matrix (the smallest size its subset
        path takes), top 512, measured in a fresh process on 2 threads: its allocations beyond the input."""
        import fit_hub
        code = ("import sys\nsys.path.insert(0, %r)\nimport numpy as np\nimport fit_backend\n"
                "def field(name):\n"
                "    return next(int(l.split()[1]) * 1024 for l in open('/proc/self/status') if l.startswith(name))\n"
                "n, k = 4096, 512\nx = np.random.default_rng(0).standard_normal((n, n // 2))\ncov = x @ x.T / n\n"
                "del x\nopen('/proc/self/clear_refs', 'w').write('5')\nbefore = field('VmRSS:')\n"
                "fit_backend.top_eig(cov, k, precision='float32')\nprint((field('VmHWM:') - before) / (n * n * 8))") % str(HERE)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                                env={**os.environ, "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"})
        copies = float(result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "nan")
        self.assertGreater(copies, 0.4, result.stderr)                  # at least the float32 copy
        self.assertLessEqual(copies * 8 * 4096 ** 2, fit_hub.eigh_bytes(4096, 512, scipy=True, precision="float32")
                             + 0.05 * 8 * 4096 ** 2)

    def test_the_float32_path_without_scipy_holds_what_the_estimate_counts(self):
        """EIGH_FULL32_COPIES: without scipy, top_eig's float32 path is numpy's eigh of the float32 copy, which
        numpy.linalg computes in float64: more than numpy's float64 eigh (2 threads, n 4,096, top 512)."""
        import fit_hub
        code = ("import sys\nsys.path.insert(0, %r)\nimport numpy as np\nimport fit_backend\n"
                "fit_backend._scipy_linalg = lambda: None\n"
                "def field(name):\n"
                "    return next(int(l.split()[1]) * 1024 for l in open('/proc/self/status') if l.startswith(name))\n"
                "n, k = 4096, 512\nx = np.random.default_rng(0).standard_normal((n, n // 2))\ncov = x @ x.T / n\n"
                "del x\nopen('/proc/self/clear_refs', 'w').write('5')\nbefore = field('VmRSS:')\n"
                "fit_backend.top_eig(cov, k, precision='float32')\nprint((field('VmHWM:') - before) / (n * n * 8))") % str(HERE)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                                env={**os.environ, "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"})
        copies = float(result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "nan")
        self.assertGreater(copies, fit_hub.EIGH_FULL_COPIES, result.stderr)
        self.assertLessEqual(copies * 8 * 4096 ** 2, fit_hub.eigh_bytes(4096, 512, scipy=False, precision="float32")
                             + 0.05 * 8 * 4096 ** 2)

    def test_numpy_eigh_holds_what_the_estimate_counts(self):
        """EIGH_COPIES: the covariance and its whitened copy held, then numpy's eigh (input copy, dsyevd's
        2 n^2 workspace, eigenvectors), measured in a fresh process on 2 threads."""
        import fit_hub
        code = ("import numpy as np\n"
                "def field(name):\n"
                "    return next(int(l.split()[1]) * 1024 for l in open('/proc/self/status') if l.startswith(name))\n"
                "n = 1500\nx = np.random.default_rng(0).standard_normal((n, n))\ncov = x @ x.T / n\ndel x\n"
                "w = cov.copy()\nopen('/proc/self/clear_refs', 'w').write('5')\nbefore = field('VmRSS:')\n"
                "np.linalg.eigh(w)\nprint(2 + (field('VmHWM:') - before) / (n * n * 8))")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                                env={**os.environ, "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"})
        copies = float(result.stdout.strip() or "nan")
        self.assertGreater(copies, 5.5, result.stderr)
        self.assertLessEqual(copies, fit_hub.EIGH_COPIES + 0.1)

    def fit_main(self, fake, argv_extra=(), available_gib=100.0):
        """fit_hub.main on three small in-memory views (test_hub.make_views) with the fake machine."""
        import fit_hub
        from test_hub import make_views
        views, spans, _ = make_views(n=1200)
        for view in views:
            view.captures = types.SimpleNamespace(rows_total=1200)
        by_name = {view.name: view for view in views}
        fake.set(available_gib=available_gib)
        output = fake.root / "hub"
        argv = [arg for view in views for arg in ("--model", f"{view.name}=/cap/{view.name}@/records.jsonl")]
        argv += ["--output", str(output), "--hub-dim", "6", "--view-dim", "12", "--layer-rank", "8", "--chunk", "300",
                 "--fit-device", "cpu", *argv_extra]          # GPU fits: test_fit_backend.FitHubDeviceTest
        checks = []

        def check(where=None):
            with open(fake.cpu_lock, "a") as other:
                try:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    checks.append("unlocked")
                except BlockingIOError:
                    checks.append(where)
        original = resource_guard.Guard.check

        def counting(guard, where=None):
            check(where)
            return original(guard, where)
        with mock.patch.dict(os.environ, fake.environ(KVPARK_GUARD_POLL_S="0.05")), \
                mock.patch.object(fit_hub.hub, "CaptureView", lambda name, captures, records: by_name[name]), \
                mock.patch.object(fit_hub.hub, "joint_spans", lambda views_: spans), \
                mock.patch.object(resource_guard.Guard, "check", counting), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            result = fit_hub.main(argv)
        return result, output, checks, printed.getvalue()

    def test_fit_hub_refuses_a_fit_that_leaves_too_little_memory(self):
        fake = machine(self)
        with self.assertRaisesRegex(SystemExit, "refused"):
            self.fit_main(fake, available_gib=28.0)                       # 4.5 GiB of base + the fit > 28 - 25
        self.assertFalse((fake.root / "hub").exists())
        logged = [json.loads(line) for line in (fake.root / "hub.log.jsonl").read_text().splitlines()]
        self.assertEqual([item["stage"] for item in logged], ["spans", "memory", "guard", "memory-check"])
        self.assertEqual(logged[2]["event"], "cpu-lock")
        self.assertFalse(logged[-1]["fits"])
        result, output, checks, _ = self.fit_main(fake, ["--guard", "fit_margin_gib=20"], available_gib=28.0)
        self.assertEqual(result, 0)

    def test_fit_hub_fits_under_the_cpu_lock_and_the_guard(self):
        import hub
        fake = machine(self)
        result, output, checks, printed = self.fit_main(fake)
        self.assertEqual(result, 0)
        self.assertTrue((output / "manifest.json").is_file())
        self.assertTrue(checks)
        self.assertNotIn("unlocked", checks)                              # the CPU lock held through the fit
        self.assertIn("gcca", checks)
        self.assertIs(hub.guard_check, hub._no_check)                     # restored
        events = read_log(fake.root / "hub.guard.jsonl")
        self.assertEqual(events[0]["event"], "guard")
        self.assertEqual(events[-1]["event"], "guard-done")
        result, _, _, printed = self.fit_main(machine(self), ["--estimate"])
        self.assertEqual(result, 0)
        self.assertIn('"stage": "memory"', printed)


if __name__ == "__main__":
    unittest.main()
