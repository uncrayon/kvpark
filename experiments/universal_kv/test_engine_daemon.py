"""engine.run and engine_daemon on tiny GGUF fixtures (a dense source, a Muse-like target) and a
random hub saved as a format 1 bundle and converted to format 2."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import weakref
from pathlib import Path
from unittest import mock

import numpy as np

import backends
from backends.numpy_backend import NumpyBackend
import convert_hub
import engine
import engine_daemon
import hub
import recurrent_init
import translate
from test_translate import StubSpans, StubView, attention_widths, kind_widths, layout_state, random_hub

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "linear_transfer"))
import kvstate  # noqa: E402
from tiny_fixture import create_fixture  # noqa: E402

REAL_LOG = engine_daemon.log          # DaemonTest replaces it with a list

TEMPLATE = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>{% endif %}")
MESSAGES = [{"role": "system", "content": "You answer briefly and stay on topic."},
            {"role": "user", "content": "Tell me about the tides along the northern coast this week."},
            {"role": "assistant", "content": "The tides run high at dawn and low near noon, with a strong ebb."},
            {"role": "user", "content": "And what about next week, when the moon is full again?"}]


def request(directory, payload, timeout=60):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(Path(directory) / engine_daemon.SOCKET))
        with connection.makefile("rwb") as stream:
            stream.write((payload if isinstance(payload, bytes) else json.dumps(payload).encode()) + b"\n")
            stream.flush()
            return json.loads(stream.readline())


class Fixture:
    """GGUFs, states, hub bundles and a job builder under one (short) temporary directory."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/tmp" if os.path.isdir("/tmp") else None)
        self.root = Path(self.tmp.name)
        self.gguf = {"src": self.root / "src.gguf", "tgt": self.root / "tgt.gguf"}
        create_fixture(self.gguf["src"], "qwen3", chat_template=TEMPLATE)
        create_fixture(self.gguf["tgt"], "muse-glimmer", chat_template=TEMPLATE, seed=5)
        self.spec = {name: kvstate.ModelSpec.from_gguf(path) for name, path in self.gguf.items()}
        rng = np.random.default_rng(0)
        # the source slot: the conversation before the newest message, rendered by the same template
        history = TEMPLATE_RENDER(MESSAGES[:-1], generation=False).encode()
        tokens = np.frombuffer(history, dtype=np.uint8).astype(np.int32)
        spec = self.spec["src"]
        layers = spec.layers_of("full")
        k = {l: rng.standard_normal((len(tokens),) + spec.head_shape(l, "k")).astype(np.float32) for l in layers}
        v = {l: rng.standard_normal((len(tokens),) + spec.head_shape(l, "v")).astype(np.float32) for l in layers}
        self.source_state = self.root / "source.bin"
        kvstate.write_new(self.source_state, kvstate.build_dense_state(
            spec, tokens, np.arange(len(tokens)), k, v, server_tokens=True).to_bytes())
        spec = self.spec["tgt"]
        layers = spec.layers_of("full", "sliding")
        k = {l: rng.standard_normal((2,) + spec.head_shape(l, "k")).astype(np.float32) for l in layers}
        v = {l: rng.standard_normal((2,) + spec.head_shape(l, "v")).astype(np.float32) for l in layers}
        self.template = self.root / "template.bin"
        kvstate.write_new(self.template, kvstate.build_muse_state(spec, [256, 72], np.arange(2), k, v,
                                                                  server_tokens=True).to_bytes())
        hubs = random_hub({"src": attention_widths(self.spec["src"]), "tgt": attention_widths(self.spec["tgt"])})
        views = [StubView(name, hubs, self.spec[name].architecture) for name in ("src", "tgt")]
        self.hub1 = self.root / "hub1"
        hub.save_hub(self.hub1, hubs, views, StubSpans(), {"note": "test"})
        self.hub2 = self.root / "hub2"
        convert_hub.convert(self.hub1, self.hub2, log=lambda line: None)
        self.outputs = 0

    def job(self, **target):
        self.outputs += 1
        text = TEMPLATE_RENDER(MESSAGES, generation=True)
        return {"hub": str(self.hub2), "tail": 8, "fill": "mean", "device": "cpu",
                "cache_dir": str(self.root / "profiles"),
                "conversation": {"messages": MESSAGES, "tools": None},
                "source": {"name": "src", "gguf": str(self.gguf["src"]), "state": str(self.source_state)},
                "target": {"name": "tgt", "gguf": str(self.gguf["tgt"]), "template_state": str(self.template),
                           "output": str(self.root / f"out-{self.outputs}.bin"),
                           "tokens": [[ord(c), c] for c in text], **target}}

    def close(self):
        self.tmp.cleanup()


def TEMPLATE_RENDER(messages, generation):  # noqa: N802  (the template, by hand)
    text = "".join(f"<|{m['role']}|>{m['content']}<|end|>\n" for m in messages)
    return text + ("<|assistant|>" if generation else "")


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = Fixture()

    @classmethod
    def tearDownClass(cls):
        cls.fx.close()

    def test_run_writes_the_target_state_and_reports_its_parts(self):
        job = self.fx.job()
        report = engine.run(job, backend=backends.select("cpu"))
        state = kvstate.read_file(job["target"]["output"], self.fx.spec["tgt"])
        P = len(job["target"]["tokens"])
        last = TEMPLATE_RENDER(MESSAGES[:-1], generation=False).encode()
        self.assertEqual(report["t0"], min(P - 8, len(last) + len("<|user|>")))
        self.assertEqual(len(state.tokens), report["t0"])
        self.assertEqual(report["sections"], {"base": report["t0"], "swa": 17})
        self.assertEqual(report["backend"], "numpy")
        self.assertEqual(report["output_bytes"], Path(job["target"]["output"]).stat().st_size)
        for key in ("hub_load_s", "profiles_s", "prepare_s", "compute_s", "assemble_s", "write_s"):
            self.assertIn(key, report["engine_timings"])
        self.assertIn("blas", report)
        self.assertEqual(Path(job["target"]["output"]).stat().st_mode & 0o777, 0o600)
        # format 1 and format 2 bundles give the same bytes
        again = self.fx.job()
        engine.run({**again, "hub": str(self.fx.hub1)}, backend=backends.select("cpu"))
        self.assertEqual(Path(again["target"]["output"]).read_bytes(), Path(job["target"]["output"]).read_bytes())

    def test_max_t0_message_stops_before_that_message(self):
        job = self.fx.job(max_t0_message=2)
        report = engine.run(job, backend=backends.select("cpu"))
        before = TEMPLATE_RENDER(MESSAGES[:2], generation=False).encode()
        # the first token past the content of messages 0-1: "<|end|>\n<|assistant|>" stays native
        self.assertEqual(report["t0"], len(before) - len("<|end|>\n"))
        self.assertLess(report["t0"], engine.run(self.fx.job(), backend=backends.select("cpu"))["t0"])
        with self.assertRaisesRegex(ValueError, "max_t0_message"):
            engine.run(self.fx.job(max_t0_message=9), backend=backends.select("cpu"))
        with self.assertRaisesRegex(ValueError, "too short"):
            engine.run(self.fx.job(max_t0_message=0), backend=backends.select("cpu"))

    def test_media_after_the_cap_is_left_to_the_target(self):
        # the newest message holds an image: chat.py renders text only, so the engine renders and
        # checks messages [0, 3) and the server's tokens of the whole prompt need only start with them
        image = {"role": "user", "content": [{"type": "text", "text": "And what is in this picture?"},
                                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
        prompt = TEMPLATE_RENDER(MESSAGES[:3], generation=False) + "<|user|><__media__>And what is in this " \
            "picture?<|end|>\n<|assistant|>"
        job = self.fx.job(max_t0_message=3)
        job["conversation"]["messages"] = MESSAGES[:3] + [image]
        job["target"]["tokens"] = [[ord(c), c] for c in prompt]
        report = engine.run(job, backend=backends.select("cpu"))
        self.assertEqual(report["t0"], len(TEMPLATE_RENDER(MESSAGES[:3], generation=False)) - len("<|end|>\n"))
        self.assertEqual(report["media_after"], 3)
        # the same cells as the text conversation capped at the same message
        text = self.fx.job(max_t0_message=3)
        self.assertEqual(engine.run(text, backend=backends.select("cpu"))["t0"], report["t0"])
        self.assertEqual(Path(job["target"]["output"]).read_bytes(), Path(text["target"]["output"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "do not spell the rendering of the messages before"):
            engine.run({**job, "target": {**job["target"], "output": job["target"]["output"] + ".x",
                                          "tokens": [[ord(c), c] for c in prompt.replace("tides", "waves")]}},
                       backend=backends.select("cpu"))
        for cap in (None, 3):                       # media before the cap, or no cap: chat.py refuses
            bad = self.fx.job(max_t0_message=cap)
            bad["conversation"]["messages"] = [MESSAGES[0], image] + MESSAGES[2:]
            with self.assertRaisesRegex(ValueError, "content must be a string"):
                engine.run(bad, backend=backends.select("cpu"))
        with self.assertRaisesRegex(ValueError, "too short"):
            engine.run({**job, "target": {**job["target"], "max_t0_message": 0}}, backend=backends.select("cpu"))

    def test_gpu_failures_fall_back_to_numpy(self):
        class Broken(NumpyBackend):
            name = "broken-gpu"
            error = FloatingPointError("float16 overflow")
            released = 0

            def encode(self, *args, **kwargs):
                raise type(self.error)(*self.error.args)

            def _released(self):
                type(self).released += 1
        events, arrays, live = [], [], []
        real_arrays = translate._output_arrays

        def output_arrays(plan):
            # the retry starts once the failed attempt's traceback, and the arrays it holds, are gone
            live.append(sum(ref() is not None for ref in arrays))
            out = real_arrays(plan)
            arrays.extend(weakref.ref(a) for kinds in out.values() for rows in kinds for a in rows.values())
            return out
        job = self.fx.job()
        with mock.patch.object(translate, "_output_arrays", output_arrays):
            report = engine.run(job, backend=Broken(), log=events.append)
        self.assertEqual(live, [0, 0])
        self.assertEqual(report["backend"], "numpy")
        self.assertIn("float16 overflow", report["fallback"])
        self.assertFalse(report["device_fault"])
        self.assertEqual(Broken.released, 1)           # the failed attempt's device cache, before the retry
        self.assertTrue(any("translated on the CPU" in w for w in report["warnings"]))
        self.assertEqual(events[0]["event"], "fallback")
        reference = self.fx.job()
        engine.run(reference, backend=backends.select("cpu"))
        self.assertEqual(Path(job["target"]["output"]).read_bytes(), Path(reference["target"]["output"]).read_bytes())
        for error, fault in ((RuntimeError("HIP error: an illegal memory access was encountered"), True),
                             (RuntimeError("HIP out of memory. Tried to allocate 2.00 GiB"), False),
                             (NotImplementedError("no device RoPE for mode 'spiral'"), False)):
            Broken.error = error
            self.assertEqual(engine.run(self.fx.job(), backend=Broken())["device_fault"], fault, error)

    def test_profiles_follow_a_gguf_replaced_at_the_same_path(self):
        path = self.fx.root / "swapped.gguf"
        path.write_bytes(self.fx.gguf["src"].read_bytes())
        self.assertEqual(engine.spec_of(path).architecture, "qwen3")
        before = engine.vocabulary(path)
        replacement = self.fx.root / "swapped.tmp"
        replacement.write_bytes(self.fx.gguf["tgt"].read_bytes())
        os.replace(replacement, path)                   # same path, another file
        self.assertEqual(engine.spec_of(path).architecture, "muse-glimmer")
        self.assertIsNot(engine.vocabulary(path), before)

    def test_one_shot_command_line(self):
        job = self.fx.job()
        path = self.fx.root / "job.json"
        path.write_text(json.dumps(job))
        result = subprocess.run([sys.executable, str(HERE / "engine.py"), str(path)], capture_output=True, text=True,
                                timeout=300, env={**os.environ, backends.ENV: "cpu"})
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report["output"], job["target"]["output"])
        self.assertTrue(Path(job["target"]["output"]).is_file())


class HybridTargetTest(unittest.TestCase):
    """A hybrid target's recurrent cell at T0 - 1: zero by default, the target's mean npz with
    target["recurrent_mean"], its group by the conversation (RESULTS.md section 7), loads kept."""

    @classmethod
    def setUpClass(cls):
        cls.fx = Fixture()
        root = cls.fx.root
        cls.gguf = root / "hybrid.gguf"
        create_fixture(cls.gguf, "qwen35", chat_template=TEMPLATE, seed=9)
        cls.spec = kvstate.ModelSpec.from_gguf(cls.gguf)
        rng = np.random.default_rng(4)
        tokens = [ord(c) for c in "Hi"]
        rows = {kind: {l: rng.standard_normal((2,) + cls.spec.head_shape(l, kind)).astype(np.float32)
                       for l in cls.spec.layers_of("full")} for kind in ("k", "v")}
        cls.template = root / "hybrid-template.bin"
        kvstate.write_new(cls.template, layout_state(cls.spec, tokens, rows["k"], rows["v"], rng).to_bytes())
        hubs = random_hub({"src": attention_widths(cls.fx.spec["src"]), "hyb": attention_widths(cls.spec)})
        cls.hub = root / "hybrid-hub"
        hub.save_hub(cls.hub, hubs, [StubView("src", hubs, "qwen3"), StubView("hyb", hubs, "qwen35")], StubSpans(),
                     {"note": "test"})
        # a mean of the template's layout: chat and tools groups (and the pooled one)
        layout = recurrent_init.recurrent_part(kvstate.read_file(cls.template, cls.spec))

        def cell(shift):
            return recurrent_init.section(layout, 0, {key: {l: rng.standard_normal(size // 4) + shift
                                                            for l, size in recurrent_init.row_sizes(
                                                                recurrent_init.Layout.of(layout), key)}
                                                      for key in ("r", "s")})
        cls.mean = recurrent_init.build_mean([(cell(1.0), "chat"), (cell(1.0), "chat"), (cell(-1.0), "tools")])
        cls.npz = root / "mean.npz"
        cls.mean.save(cls.npz)
        cls.chat_only = root / "mean-chat.npz"
        recurrent_init.RecurrentMean.load(cls.npz, groups=["chat"]).save(cls.chat_only)

    @classmethod
    def tearDownClass(cls):
        cls.fx.close()

    def setUp(self):
        engine.drop_recurrent_means()

    def job(self, tools=None, **target):
        job = self.fx.job(name="hyb", gguf=str(self.gguf), template_state=str(self.template), **target)
        job["hub"] = str(self.hub)
        job["conversation"]["tools"] = tools
        return job

    def cell(self, job):
        state = kvstate.read_file(job["target"]["output"], self.spec)
        return state.tokens, recurrent_init.cell_rows(recurrent_init.recurrent_part(state))

    def attention(self, job):
        state = kvstate.read_file(job["target"]["output"], self.spec)
        stream = next(part for part in state.state.parts if isinstance(part, kvstate.KVCache)).stream()
        return [np.asarray(stream.positions)] + [np.asarray(rows[layer]) for rows in (stream.k, stream.v)
                                                 for layer in sorted(rows)]

    def assert_cell(self, rows, group):
        for key in ("r", "s"):
            for layer, values in self.mean.rows(group)[key].items():
                np.testing.assert_allclose(rows[key][layer], values, rtol=1e-3, atol=1e-3)   # stored in float16

    def test_the_recurrent_cell_starts_from_the_mean_of_the_conversations_group(self):
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
        job = self.job()
        report = engine.run(job, backend=backends.select("cpu"))
        self.assertEqual(report["recurrent"], {"strategy": "zero"})
        tokens, rows = self.cell(job)
        self.assertFalse(any(values.any() for kind in rows.values() for values in kind.values()))
        attention = self.attention(job)
        for tools_, flag, group in ((None, False, "chat"), (tools, False, "all"), (tools, True, "tools"),
                                    (None, True, "chat")):
            with self.subTest(tools=bool(tools_), recurrent_tools=flag):
                job = self.job(tools_, recurrent_mean=str(self.npz), recurrent_tools=flag)
                report = engine.run(job, backend=backends.select("cpu"))
                self.assertEqual(report["recurrent"], {"strategy": "mean", "group": group, "file": "mean.npz"})
                self.assertIn("recurrent_s", report["engine_timings"])
                state_tokens, rows = self.cell(job)
                self.assert_cell(rows, group)
                # the attention cells are those of the zero cell's translation
                self.assertEqual(len(state_tokens), report["t0"])
                for x, y in zip(self.attention(job), attention, strict=True):
                    np.testing.assert_array_equal(x, y)
        # a file without the conversation's group nor the pooled one: zero, said so
        report = engine.run(self.job(tools, recurrent_mean=str(self.chat_only)), backend=backends.select("cpu"))
        self.assertEqual((report["recurrent"]["strategy"], report["recurrent"]["requested"]), ("zero", "mean"))
        self.assertIn("neither the all group", report["recurrent"]["note"])
        # a registered file since moved or damaged: zero, said so (the switch still saves the prefill)
        damaged = self.fx.root / "damaged.npz"
        damaged.write_bytes(self.npz.read_bytes()[:100])
        for path, error in ((self.fx.root / "moved.npz", "FileNotFoundError"), (damaged, "BadZipFile")):
            with self.subTest(error=error):
                job = self.job(recurrent_mean=str(path))
                report = engine.run(job, backend=backends.select("cpu"))
                self.assertEqual((report["recurrent"]["strategy"], report["recurrent"]["requested"]), ("zero", "mean"))
                self.assertIn(f"the mean cannot be read ({error}", report["recurrent"]["note"])
                _, rows = self.cell(job)
                self.assertFalse(any(values.any() for kind in rows.values() for values in kind.values()))
        # an explicit strategy; one that needs the mean without it
        report = engine.run(self.job(recurrent_mean=str(self.npz), recurrent="mean_conv"),
                            backend=backends.select("cpu"))
        self.assertEqual(report["recurrent"]["strategy"], "mean_conv")
        with self.assertRaisesRegex(ValueError, "needs the target's recurrent_mean"):
            engine.run(self.job(recurrent="mean"), backend=backends.select("cpu"))
        # a dense target ignores it
        report = engine.run(self.fx.job(recurrent_mean=str(self.npz)), backend=backends.select("cpu"))
        self.assertNotIn("recurrent", report)

    def test_a_mean_of_another_layout_is_refused(self):
        other = self.fx.root / "other.npz"
        if not other.exists():
            part = kvstate.RecurrentState("recurrent", (0,), np.asarray([0], np.int32), ((),), 4,
                                          {0: kvstate.OpaqueRows(0, 16, bytes(16))},
                                          {0: kvstate.OpaqueRows(0, 32, bytes(32))})
            recurrent_init.build_mean([part]).save(other)
        with self.assertRaisesRegex(ValueError, "another recurrent layout"):
            engine.run(self.job(recurrent_mean=str(other)), backend=backends.select("cpu"))

    def test_loaded_groups_are_kept_per_file_identity(self):
        path = self.fx.root / "kept.npz"
        path.unlink(missing_ok=True)
        self.mean.save(path)
        loads = []
        real = recurrent_init.RecurrentMean.load

        def load(cls, *args, **kwargs):
            loads.append(kwargs.get("groups"))
            return real.__func__(cls, *args, **kwargs)
        with mock.patch.object(recurrent_init.RecurrentMean, "load", classmethod(load)):
            for _ in range(2):
                engine.run(self.job(recurrent_mean=str(path)), backend=backends.select("cpu"))
            self.assertEqual(loads, [["chat"]])
            self.assertEqual([(item["file"], item["group"]) for item in engine.recurrent_means()],
                             [("kept.npz", "chat")])
            replacement = self.fx.root / "kept.tmp.npz"
            replacement.unlink(missing_ok=True)
            self.mean.save(replacement)
            os.replace(replacement, path)           # the same path, another file: read again
            engine.run(self.job(recurrent_mean=str(path)), backend=backends.select("cpu"))
            self.assertEqual(loads, [["chat"], ["chat"]])
        self.assertGreater(engine.drop_recurrent_means(), 0)
        self.assertEqual(engine.recurrent_means(), [])

    def test_the_daemon_keeps_the_mean_until_the_unload(self):
        patcher = mock.patch.object(engine_daemon, "log", lambda event: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        directory = Path(tempfile.mkdtemp(dir=self.fx.root, prefix="r"))
        daemon = engine_daemon.Daemon(directory, device="cpu")
        self.assertTrue(daemon.bind())
        thread = threading.Thread(target=daemon.serve, daemon=True)
        thread.start()
        self.addCleanup(lambda: (daemon.stopping.set(), thread.join(10)))
        reply = request(directory, {"op": "translate", "job": self.job(recurrent_mean=str(self.npz))})
        self.assertTrue(reply["ok"], reply)
        self.assertEqual(reply["report"]["recurrent"]["group"], "chat")
        means = request(directory, {"op": "stats"})["recurrent_means"]
        self.assertEqual([(item["file"], item["group"]) for item in means], [("mean.npz", "chat")])
        self.assertGreaterEqual(request(directory, {"op": "unload"})["released_bytes"], means[0]["bytes"])
        self.assertEqual(request(directory, {"op": "stats"})["recurrent_means"], [])


class DaemonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = Fixture()

    @classmethod
    def tearDownClass(cls):
        cls.fx.close()

    def setUp(self):
        self.events = []
        patcher = mock.patch.object(engine_daemon, "log", self.events.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def start(self, device="cpu", **options):
        directory = Path(tempfile.mkdtemp(dir=self.fx.root, prefix="d"))
        daemon = engine_daemon.Daemon(directory, device=device, **options)
        self.assertTrue(daemon.bind())
        thread = threading.Thread(target=daemon.serve, daemon=True)
        thread.start()
        self.addCleanup(lambda: (daemon.stopping.set(), thread.join(10)))
        return daemon, directory, thread

    def test_protocol_ping_load_translate_stats_unload_shutdown(self):
        daemon, directory, thread = self.start()
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
        self.assertEqual((directory / "engine.sock").stat().st_mode & 0o777, 0o600)
        ping = request(directory, {"op": "ping"})
        self.assertEqual((ping["ok"], ping["pid"], ping["version"]), (True, os.getpid(), engine_daemon.VERSION))
        profiles = directory / "profiles"
        loaded = request(directory, {"op": "load", "hub": str(self.fx.hub2), "source": "src", "target": "tgt",
                                     "gguf": [str(self.fx.gguf["tgt"])], "cache_dir": str(profiles)})
        self.assertTrue(loaded["ok"])
        self.assertEqual(len(loaded["loaded"]), 4)
        self.assertEqual(len(list(profiles.glob("tgt.gguf-*.json"))), 1)       # the GGUF profile, parsed ahead
        self.assertGreater(loaded["resident_bytes"], 0)
        job = self.fx.job()
        reply = request(directory, {"op": "translate", "job": job})
        self.assertTrue(reply["ok"], reply)
        self.assertEqual(reply["report"]["engine_timings"]["hub_load_s"] < 0.5, True)    # sides were resident
        for key in ("queue_s", "run_s", "memory_guard", "jobs"):
            self.assertIn(key, reply["daemon"])
        reference = self.fx.job()
        engine.run(reference, backend=backends.select("cpu"))
        self.assertEqual(Path(job["target"]["output"]).read_bytes(), Path(reference["target"]["output"]).read_bytes())
        stats = request(directory, {"op": "stats"})
        self.assertEqual((stats["jobs"], stats["failures"]), (1, 0))
        self.assertTrue(stats["backends"]["numpy"]["resident"])
        self.assertTrue(stats["calibrated"]["host"] or not engine_daemon.reset_peak_rss())
        self.assertGreater(request(directory, {"op": "unload"})["released_bytes"], 0)
        self.assertEqual(request(directory, {"op": "stats"})["backends"]["numpy"]["resident"], [])
        self.assertTrue(request(directory, {"op": "shutdown"})["ok"])
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertFalse((directory / "engine.sock").exists())

    def load_pair(self, directory, **extra):
        return request(directory, {"op": "load", "hub": str(self.fx.hub2), "source": "src", "target": "tgt",
                                   "source_gguf": str(self.fx.gguf["src"]), "target_gguf": str(self.fx.gguf["tgt"]),
                                   "cache_dir": str(directory / "profiles"), **extra})

    def test_load_warms_up_the_pair(self):
        daemon, directory, _ = self.start()
        loaded = self.load_pair(directory)
        self.assertTrue(loaded["ok"], loaded)
        self.assertIsInstance(loaded["warm_up_s"], float)
        self.assertGreaterEqual(loaded["seconds"], loaded["load_s"])
        # the target's sections are built ahead (what the first switch would otherwise do)
        hub_object = daemon.hub(str(self.fx.hub2))
        layouts = self.fx.spec["tgt"].layouts(kvstate.FLAGS_NONE)
        expected = {tuple(layout.layers) for layout in layouts if layout.kind == "kv" and layout.layers}
        self.assertEqual(len(expected), 2)                          # Muse-like: global and sliding layers
        for kind in hub.KINDS:
            self.assertEqual(set(backends.select("cpu").decoder(hub_object, kind, "tgt")._sections), expected)
        # the switch itself is unchanged by the warm-up
        job, reference = self.fx.job(), self.fx.job()
        self.assertTrue(request(directory, {"op": "translate", "job": job})["ok"])
        engine.run(reference, backend=backends.select("cpu"))
        self.assertEqual(Path(job["target"]["output"]).read_bytes(), Path(reference["target"]["output"]).read_bytes())
        self.assertIsNone(self.load_pair(directory, warm_up=False)["warm_up_s"])
        # a failing warm-up is logged and never fails the load
        with mock.patch.object(engine_daemon, "warm_products", side_effect=RuntimeError("no kernel for this shape")):
            loaded = self.load_pair(directory)
        self.assertEqual((loaded["ok"], loaded["warm_up_s"]), (True, None))
        self.assertIn("no kernel", next(e["error"] for e in self.events if e["event"] == "warm-up-error"))

    def test_load_warms_up_the_pair_on_the_gpu(self):
        available, detail = backends.gpu_status()
        if not available:
            self.skipTest(f"no GPU backend: {detail}")
        _, directory, _ = self.start(device="gpu")
        loaded = self.load_pair(directory, device="gpu")
        self.assertEqual((loaded["ok"], loaded["backend"]), (True, "torch"))
        self.assertIsInstance(loaded["warm_up_s"], float)
        self.assertNotIn("warm-up-error", [e["event"] for e in self.events])
        reply = request(directory, {"op": "translate", "job": {**self.fx.job(), "device": "gpu"}})
        self.assertEqual((reply["ok"], reply["report"]["backend"]), (True, "torch"))

    def test_ping_and_stats_report_the_engine_setup(self):
        _, directory, _ = self.start()
        ping = request(directory, {"op": "ping"})
        self.assertIn("gflops", ping["engine"]["blas"])
        self.assertNotIn("gpu", ping["engine"])                     # a CPU daemon never imports torch
        self.assertEqual((ping["engine"]["threads"], ping["engine"]["device"]),
                         (os.environ["OPENBLAS_NUM_THREADS"], "cpu"))
        self.assertEqual(request(directory, {"op": "stats"})["engine"], ping["engine"])
        # what this engine handles beyond older ones: the bridge drops those skips for it
        self.assertEqual(ping["features"], ["media-source", "template-options", "recurrent-mean", "vocabularies"])
        stats = request(directory, {"op": "stats"})
        self.assertEqual((stats["features"], stats["recurrent_means"]), (ping["features"], []))

    def test_check_command_line_and_thread_default(self):
        self.assertTrue(1 <= engine_daemon.default_threads() <= 16)
        environment = {**os.environ, "OPENBLAS_NUM_THREADS": "3", "OMP_NUM_THREADS": "3"}
        result = subprocess.run([sys.executable, str(HERE / "engine_daemon.py"), "--check", "--device", "cpu"],
                                capture_output=True, text=True, timeout=120, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        info = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual((info["device"], info["threads"], info["version"]), ("cpu", "3", engine_daemon.VERSION))
        self.assertIn("reference", info["blas"])
        self.assertNotIn("gpu", info)
        environment = {name: value for name, value in os.environ.items()
                       if name not in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
        result = subprocess.run([sys.executable, str(HERE / "engine_daemon.py"), "--check", "--device", "cpu"],
                                capture_output=True, text=True, timeout=120, env=environment)
        self.assertEqual(json.loads(result.stdout)["threads"], str(engine_daemon.default_threads()))
        # a lone OMP_NUM_THREADS wins over the default too (OpenBLAS reads its own variable first)
        result = subprocess.run([sys.executable, str(HERE / "engine_daemon.py"), "--check", "--device", "cpu"],
                                capture_output=True, text=True, timeout=120, env={**environment, "OMP_NUM_THREADS": "2"})
        self.assertEqual(json.loads(result.stdout)["threads"], "2")

    def test_check_runs_without_the_resource_module(self):
        # Windows has no resource module; `kvpark bridge status` runs --check there too
        code = ("import runpy, sys; sys.modules['resource'] = None; sys.argv = [sys.argv[1], '--check', '--device', "
                "'cpu']; runpy.run_path(sys.argv[0], run_name='__main__')")
        result = subprocess.run([sys.executable, "-c", code, str(HERE / "engine_daemon.py")], capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1])["device"], "cpu")

    def test_errors_are_replies_and_the_daemon_keeps_serving(self):
        _, directory, _ = self.start()
        self.assertIn("bad request", request(directory, b"{not json")["error"])
        self.assertIn("unknown op", request(directory, {"op": "dance"})["error"])
        broken = self.fx.job()
        broken["source"]["state"] = str(self.fx.root / "missing.bin")
        reply = request(directory, {"op": "translate", "job": broken})
        self.assertFalse(reply["ok"])
        self.assertIn("FileNotFoundError", reply["error"])
        self.assertTrue(request(directory, {"op": "ping"})["ok"])
        self.assertEqual(request(directory, {"op": "stats"})["failures"], 1)

    def test_error_replies_carry_a_kind(self):
        daemon, directory, _ = self.start()
        self.assertEqual(request(directory, b"{not json")["kind"], "request")
        self.assertEqual(request(directory, {"op": "dance"})["kind"], "request")
        broken = self.fx.job()
        broken["source"]["state"] = str(self.fx.root / "missing.bin")
        self.assertEqual(request(directory, {"op": "translate", "job": broken})["kind"], "job")
        short = self.fx.job(max_t0_message=0)
        reply = request(directory, {"op": "translate", "job": short})
        self.assertEqual((reply["kind"], "too short" in reply["error"]), ("job", True))
        with mock.patch.object(engine_daemon, "mem_available", return_value=10 ** 6):
            reply = request(directory, {"op": "translate", "job": self.fx.job()})
        self.assertEqual(reply["kind"], "memory")
        # a rendering the engine cannot reproduce: its own kind (the bridge then translates the default one)
        mismatch = self.fx.job(tokens=[[ord(c), c] for c in TEMPLATE_RENDER(MESSAGES, generation=True) + "!"])
        reply = request(directory, {"op": "translate", "job": mismatch})
        self.assertEqual((reply["kind"], reply["error"].split(":")[0]), ("template", "TemplateMismatch"))
        self.assertEqual(engine_daemon.error_kind(AttributeError("x")), "internal")
        with mock.patch.object(daemon, "stats", side_effect=RuntimeError("dictionary changed size during iteration")):
            reply = request(directory, {"op": "stats"})        # a reply, not a dropped connection
        self.assertEqual((reply["ok"], reply["kind"]), (False, "internal"))
        self.assertTrue(request(directory, {"op": "ping"})["ok"])

    def test_the_worker_survives_a_failing_log(self):
        # any exception in the worker's own error handling (here a log write) still gives a reply
        daemon, directory, _ = self.start()
        self.assertTrue(request(directory, {"op": "ping"})["ok"])          # serving, start logged
        patcher = mock.patch.object(engine_daemon, "log", side_effect=OSError(28, "No space left on device"))
        patcher.start()
        try:
            broken = self.fx.job()
            broken["source"]["state"] = str(self.fx.root / "missing.bin")
            first = request(directory, {"op": "translate", "job": broken}, timeout=30)
            second = request(directory, {"op": "translate", "job": self.fx.job()}, timeout=30)
        finally:
            patcher.stop()
        self.assertEqual([first["ok"], second["ok"]], [False, False])         # replies, no hang
        self.assertTrue(daemon.worker.is_alive())
        self.assertTrue(request(directory, {"op": "translate", "job": self.fx.job()})["ok"])
        # and the real log does not raise on a full disk in the first place
        with mock.patch.object(sys, "stderr", mock.Mock(write=mock.Mock(side_effect=OSError(28, "full")))):
            REAL_LOG({"event": "translate", "seconds": 0.4})

    def test_a_lost_worker_stops_the_daemon(self):
        daemon, directory, thread = self.start()
        self.assertTrue(request(directory, {"op": "ping"})["ok"])          # serving: the worker runs
        daemon.queue.put(None)                          # the worker ends (as if it died)
        daemon.worker.join(10)
        reply = request(directory, {"op": "translate", "job": self.fx.job()})
        self.assertEqual((reply["ok"], reply["kind"]), (False, "busy"))
        thread.join(10)
        self.assertFalse(thread.is_alive())             # stopped: the next client starts a fresh daemon

    def test_a_device_fault_moves_later_jobs_to_the_cpu(self):
        class Faulty(NumpyBackend):
            name = "faulty-gpu"

            def encode(self, *args, **kwargs):
                raise RuntimeError("HIP error: an illegal memory access was encountered")
        faulty, numpy = Faulty(), backends.select("cpu")
        daemon, directory, _ = self.start()
        with mock.patch.object(engine_daemon.backends, "select",
                               side_effect=lambda device=None: numpy if device == "cpu" else faulty):
            first = request(directory, {"op": "translate", "job": {**self.fx.job(), "device": "gpu"}})
            self.assertTrue(first["ok"], first)
            self.assertTrue(first["report"]["device_fault"])
            self.assertIn("illegal memory access", request(directory, {"op": "ping"})["gpu_fault"])
            second = request(directory, {"op": "translate", "job": {**self.fx.job(), "device": "gpu"}})
        self.assertEqual((second["report"]["backend"], second["report"].get("fallback")), ("numpy", None))
        self.assertIn("illegal memory access", request(directory, {"op": "stats"})["gpu_fault"])

    def test_one_instance_per_directory(self):
        daemon, directory, _ = self.start()
        second = engine_daemon.Daemon(directory, device="cpu")
        self.assertFalse(second.bind())
        self.assertTrue(request(directory, {"op": "ping"})["ok"])          # the first one still serves

    def test_idle_unload_then_exit(self):
        daemon, directory, thread = self.start(idle_unload=0.3, idle_exit=2.5)
        request(directory, {"op": "load", "hub": str(self.fx.hub2), "source": "src", "target": "tgt"})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not daemon.unloaded:
            request(directory, {"op": "stats"})          # a status poller does not keep it busy
            time.sleep(0.1)
        self.assertTrue(daemon.unloaded)
        self.assertEqual(backends.select("cpu").resident(), [])
        self.assertIn("idle", [e.get("reason") for e in self.events if e["event"] == "unload"])
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertFalse((directory / "engine.sock").exists())

    def test_memory_guard_evicts_moves_to_the_cpu_or_refuses(self):
        daemon, directory, _ = self.start()
        job = self.fx.job()
        hub_object = daemon.hub(job["hub"])
        numpy = backends.select("cpu")
        moved = daemon.estimate(job)
        with mock.patch.object(engine_daemon, "mem_available", return_value=10 * moved + 4 * engine_daemon.GIB):
            self.assertEqual(daemon.guard(job, hub_object, numpy)["need"]["host"],
                             int(daemon.factor["host"] * moved) + engine_daemon.RESERVE["host"])

        class FakeGpu:
            name = "torch"

            def __init__(self, free):
                self.free = free

            def memory(self):
                return {"device_free": self.free}
        with mock.patch.object(engine_daemon, "mem_available", return_value=10 ** 12):
            self.assertEqual(daemon.guard(job, hub_object, FakeGpu(10 ** 6)).get("device"), "cpu")
            self.assertNotIn("device", daemon.guard(job, hub_object, FakeGpu(10 ** 12)))
        with mock.patch.object(engine_daemon, "mem_available", return_value=10 ** 6):
            reply = request(directory, {"op": "translate", "job": job})
        self.assertFalse(reply["ok"])
        self.assertIn("memory:", reply["error"])

    def test_memory_estimate_counts_the_rows_each_layer_stores(self):
        dense, muse = self.fx.spec["src"], self.fx.spec["tgt"]
        self.assertEqual(engine_daemon.state_bytes_per_cell(dense),
                         sum(2 * 2 * np.prod(dense.head_shape(l, "k")) for l in dense.layers_of("full")))
        self.assertEqual(engine_daemon.state_bytes_per_cell(muse),
                         sum(2 * 2 * np.prod(muse.head_shape(l, "k")) for l in muse.layers_of("full", "sliding")))
        # MLA: one K row per layer ([c_kv | k_pe]), no V; DSA adds the indexer keys
        from test_state_writer import mla_spec
        mla, dsa = mla_spec(), mla_spec("deepseek32", indexer=True)
        self.assertEqual(engine_daemon.state_bytes_per_cell(mla), 2 * 3 * 32)
        self.assertEqual(engine_daemon.state_bytes_per_cell(dsa), 2 * 3 * 32 + 2 * 3 * 64)

    def test_an_mla_target_loads_and_warms_only_its_key_sides(self):
        root = Path(tempfile.mkdtemp(dir=self.fx.root, prefix="m"))
        sys.path.insert(0, str(HERE.parent / "linear_transfer"))
        import tiny_fixture
        tiny_fixture.create_mla_fixture(root / "plain.gguf", "deepseek2")
        tiny_fixture.copy_with_metadata(root / "plain.gguf", root / "mla.gguf",
                                        values={"tokenizer.chat_template": TEMPLATE})
        self.assertEqual(engine_daemon.Daemon.pair_kinds(str(root / "mla.gguf"), str(root / "profiles")), ("k",))
        self.assertEqual(engine_daemon.Daemon.pair_kinds(str(self.fx.gguf["tgt"])), ("k", "v"))
        self.assertEqual(engine_daemon.Daemon.pair_kinds(None), hub.KINDS)
        self.assertEqual(engine_daemon.Daemon.pair_kinds(str(root / "missing.gguf")), hub.KINDS)
        # a hub holding both kinds of both models: the load and its warm-up touch only the K sides
        mla = kvstate.ModelSpec.from_gguf(root / "mla.gguf")
        hubs = random_hub({"src": kind_widths(self.fx.spec["src"]), "mla": kind_widths(mla)})
        bundle = root / "hub"
        hub.save_hub(bundle, hubs, [StubView("src", hubs, "qwen3"), StubView("mla", hubs, mla.architecture)],
                     StubSpans(), {"note": "test"})
        _, directory, _ = self.start()
        loaded = request(directory, {"op": "load", "hub": str(bundle), "source": "src", "target": "mla",
                                     "source_gguf": str(self.fx.gguf["src"]), "target_gguf": str(root / "mla.gguf"),
                                     "cache_dir": str(root / "profiles")})
        self.assertTrue(loaded["ok"], loaded)
        self.assertEqual(sorted(map(tuple, loaded["loaded"])), [("k", "mla", "decoder"), ("k", "src", "encoder")])
        self.assertIsInstance(loaded["warm_up_s"], float)
        self.assertNotIn("warm-up-error", [e["event"] for e in self.events])
        resident = {tuple(item["key"][1:]) for item in backends.select("cpu").resident()
                    if item["key"][0] == str(bundle.resolve())}
        self.assertEqual(resident, {("k", "mla", "decoder"), ("k", "src", "encoder")})

    def test_keep_host_holds_freed_memory_until_the_idle_unload(self):
        with mock.patch.object(engine_daemon, "keep_freed_memory", return_value=True) as keep, \
                mock.patch.object(engine_daemon, "trim_memory", return_value=True) as trim:
            daemon, directory, _ = self.start(keep_host=True)
            keep.assert_called_once_with()
            self.assertTrue(request(directory, {"op": "stats"})["keep_host"])
            self.assertTrue(request(directory, {"op": "translate", "job": self.fx.job()})["ok"])
            trim.assert_not_called()                    # kept between switches
            request(directory, {"op": "unload"})
            trim.assert_called_once_with()
        self.assertIn(True, [e.get("trimmed") for e in self.events if e["event"] == "unload"])
        # without the option nothing is changed or trimmed
        with mock.patch.object(engine_daemon, "keep_freed_memory") as keep, \
                mock.patch.object(engine_daemon, "trim_memory") as trim:
            daemon, directory, _ = self.start()
            request(directory, {"op": "unload"})
        keep.assert_not_called()
        trim.assert_not_called()
        self.assertFalse(daemon.keep_host)
        # the real calls work on this platform (glibc) or say they cannot
        self.assertIn(engine_daemon.trim_memory(), (True, False, None))

    def test_peer_credentials(self):
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with left, right:
            uid = engine_daemon.peer_uid(left)
            self.assertIn(uid, (os.getuid(), None))

    def test_command_line_daemon(self):
        directory = Path(tempfile.mkdtemp(dir=self.fx.root, prefix="c"))
        process = subprocess.Popen([sys.executable, str(HERE / "engine_daemon.py"), "--bridge-dir", str(directory),
                                    "--device", "cpu"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        try:
            for _ in range(200):
                if (directory / "engine.sock").exists():
                    break
                time.sleep(0.05)
            self.assertTrue(request(directory, {"op": "ping"})["ok"])
            second = subprocess.run([sys.executable, str(HERE / "engine_daemon.py"), "--bridge-dir", str(directory)],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(second.returncode, 3)                     # already running
            reply = request(directory, {"op": "translate", "job": self.fx.job()})
            self.assertTrue(reply["ok"], reply)
            request(directory, {"op": "shutdown"})
            _, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0)
            log = [json.loads(line) for line in stderr.splitlines() if line.startswith("{")]
            self.assertEqual([e["event"] for e in log if e["event"] in ("start", "translate", "exit")],
                             ["start", "translate", "exit"])
            self.assertNotIn("tides", json.dumps(log))                  # no conversation text in the log
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()


if __name__ == "__main__":
    unittest.main()
