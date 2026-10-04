import ast
import json
import os
import struct
import sys
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from kvpark.bridge import capabilities, engine_client, registry, resume
from kvpark.bridge import switch as bridge_switch
from kvpark.bridge import cli as bridge_cli
from kvpark.bridge.client import LlamaServer, ServerError

KVSTATE = Path(__file__).resolve().parents[1] / "experiments" / "linear_transfer" / "kvstate.py"


def write_gguf(path, arch, **counts):
    """A GGUF v3 header without tensors: a few keys of other types, then general.architecture and
    `counts` (uint32 values; a keyword's double underscores stand for dots)."""
    def string(text):
        data = text.encode()
        return struct.pack("<Q", len(data)) + data
    kvs = [string("general.name") + struct.pack("<I", 8) + string("fixture"),
           string("x.count") + struct.pack("<I", 4) + struct.pack("<I", 7),
           string("x.tokens") + struct.pack("<IIQ", 9, 8, 2) + string("a") + string("bc"),
           string("x.flags") + struct.pack("<IIQ", 9, 7, 3) + bytes(3),
           string("general.architecture") + struct.pack("<I", 8) + string(arch)]
    kvs += [string(key.replace("__", ".")) + struct.pack("<II", 4, value) for key, value in counts.items()]
    Path(path).write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs))
    return path


def module_constants(path):
    """Module-level constants of a Python file (literals, names and +), without importing it."""
    env = {}

    def value(node):
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return value(node.left) + value(node.right)
        return ast.literal_eval(node)
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                env[node.targets[0].id] = value(node.value)
            except (ValueError, KeyError, TypeError):
                pass
    return env


def write_recurrent_mean(path, meta):
    """An npz as recurrent_init.RecurrentMean.save writes it, as far as the bridge reads it: meta.npy
    holding one JSON string (a little-endian unicode NPY array), next to a stand-in row array."""
    text = json.dumps(meta, sort_keys=True)
    header = f"{{'descr': '<U{len(text)}', 'fortran_order': False, 'shape': (), }}"
    header += " " * (-(10 + len(header) + 1) % 64) + "\n"
    data = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin-1") + text.encode("utf-32-le")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("meta.npy", data)
        archive.writestr("all/r/0.npy", b"rows")
    return path


def qwen35_mean_meta(**changes):
    """The meta of a recurrent mean for the hybrid GGUF of BridgeTest.hybrid_gguf (8 layers, every
    4th attention; GDN rows of conv kernel 4, inner 64, state 16, 2 groups)."""
    layers = [0, 1, 2, 4, 5, 6]
    meta = {"format_version": 1, "dtype": "float16", "groups": {"all": {"count": 3}, "chat": {"count": 2}},
            "layout": {"name": "recurrent", "layers": layers, "n_layer": 8, "r_bytes": [4 * 3 * 128] * 6,
                       "s_bytes": [4 * 16 * 64] * 6, "p_layers": [], "p_bytes": 0},
            "model": {"architecture": "qwen35", "name": "hybrid"}}
    for key, value in changes.items():
        (meta["layout"] if key in meta["layout"] else meta)[key] = value
    return meta


def dflash_template_draft(n, *, ext=False):
    """A DFlash-like draft blob of a native save of n tokens: iSWA, base without layers, one F16 SWA layer."""
    def section(layers):
        cells = b"".join(struct.pack("<iI", p, 1) + (struct.pack("<iii", p, p, -1) if ext else b"") + struct.pack("<i", 0)
                         for p in range(n))
        rows = [struct.pack("<iQ", 1, 8) + b"\x01" * 8 * n for _ in range(2 * layers)]
        return struct.pack("<II", 1, n) + cells + struct.pack("<II", 0, layers) + b"".join(rows)
    return struct.pack("<Ii", resume.IO_MAGIC, 0) + section(0) + section(1)


def mtp_template_draft(n):
    cells = b"".join(struct.pack("<iI", p, 1) + struct.pack("<iii", p, p, 65) + struct.pack("<i", 0) for p in range(n))
    rows = struct.pack("<iQ", 1, 8) + bytes(8 * n)
    return struct.pack("<IiII", resume.IO_MAGIC, 0, 1, n) + cells + struct.pack("<II", 0, 1) + rows + rows


class FakeLlama:
    """Just enough of llama-server's slot, template, tokenize and completion API.

    patched: like a kvpark-patched server, write <slot>.resume on save and require it on restore;
    draft: the draft blob a save of n tokens writes (a function of n), or None without a draft."""

    def __init__(self, slot_dir, name):
        self.slot_dir, self.name = Path(slot_dir), name
        self.calls, self.resident, self.bodies = [], 0, []
        self.patched, self.draft, self.speculative, self.tasks = False, None, "none", 0
        self.thinking = "start"            # where enable_thinking: false changes the rendering
        self.modalities, self.marker, self.text_join = {"vision": False, "audio": False}, "<__media_x__>", ""
        self.restored_resume = self.restored_payload = None
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                fake.calls.append(("GET", self.path))
                if self.path == "/props":
                    return self.reply(200, {"build_info": "b1-test", "total_slots": 1, "modalities": fake.modalities,
                                            "media_marker": fake.marker})
                if self.path == "/slots":
                    slot = {"id": 0, "speculative": fake.speculative != "none"}
                    if fake.tasks:
                        slot["params"] = {"speculative.types": fake.speculative}
                    return self.reply(200, [slot])
                return self.reply(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                fake.calls.append(("POST", self.path))
                fake.bodies.append((self.path, body))
                if self.path.startswith("/slots/0?action="):
                    action = self.path.split("=", 1)[1]
                    if action == "erase":
                        fake.resident = 0
                        return self.reply(200, {"id_slot": 0})
                    if action == "save":
                        n = fake.resident or 7
                        (fake.slot_dir / body["filename"]).write_text(json.dumps({"tokens": n}))
                        if fake.patched:
                            draft = fake.draft(n) if fake.draft else b""
                            (fake.slot_dir / (body["filename"] + ".resume")).write_bytes(resume.Resume(draft).to_bytes())
                        return self.reply(200, {"n_saved": n, "n_written": 10})
                    if action == "restore":
                        path = fake.slot_dir / body["filename"]
                        if not path.exists():
                            return self.reply(400, {"error": "missing"})
                        if fake.patched:
                            sidecar = Path(str(path) + ".resume")
                            if not sidecar.exists():
                                return self.reply(400, {"error": "Missing slot resume file"})
                            state = resume.parse_resume(sidecar.read_bytes())
                            if bool(state.draft) != (fake.draft is not None):
                                return self.reply(400, {"error": "Unable to restore draft state"})
                            fake.restored_resume = state
                        fake.restored_payload = json.loads(path.read_text())
                        fake.resident = fake.restored_payload["tokens"]
                        return self.reply(200, {"n_restored": fake.resident})
                if self.path == "/apply-template":
                    def content(value):
                        # text-only parts reach the template as a list; this one joins them like a typed template
                        return fake.text_join.join(p["text"] for p in value) if isinstance(value, list) else value
                    # like llama-server (prefill_assistant): a final assistant message is continued, and
                    # one with tool calls is refused
                    last = body["messages"][-1] if body["messages"] else {}
                    if last.get("role") == "assistant" and last.get("tool_calls"):
                        return self.reply(400, {"error": "Cannot continue an assistant message that contains "
                                                         "tool calls."})
                    generation = body.get("add_generation_prompt", True) and last.get("role") != "assistant"
                    parts = [f"<{m['role']}>{content(m['content'])}" for m in body["messages"]]
                    suffix = "<assistant>" if generation else ""
                    if (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False:
                        if fake.thinking == "start":         # like Qwen3.8's: in the system prompt
                            parts.insert(0, "<no-think>")
                        elif fake.thinking == "newest":      # before the newest message
                            parts.insert(len(parts) - 1, "<no-think>")
                        elif generation:                     # like classic Qwen3's: after the generation prompt
                            suffix += "<think></think>"
                    return self.reply(200, {"prompt": "".join(parts) + suffix})
                if self.path == "/tokenize":
                    return self.reply(200, {"tokens": [{"id": ord(c), "piece": c} for c in body["content"]]})
                if self.path == "/completion":
                    fake.tasks += 1
                    cached = min(fake.resident, len(body["prompt"]) - 1)
                    fake.resident = len(body["prompt"]) + body.get("n_predict", 0)
                    return self.reply(200, {"tokens": [1], "timings": {"cache_n": cached,
                                                                       "prompt_n": len(body["prompt"]) - cached,
                                                                       "prompt_ms": 1.0}})
                return self.reply(404, {})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


FAKE_ENGINE = r'''
import json, sys
job = json.load(open(sys.argv[1]))
if job.get("fail"):
    sys.exit("engine exploded")
assert open(job["source"]["state"]).read()
t0 = len(job["target"]["tokens"]) - job["tail"]
target = {k: v for k, v in job["target"].items() if k != "tokens"}
open(job["target"]["output"], "w").write(json.dumps({"tokens": t0, "job": {"tail": job["tail"], "target": target,
                                                                          "source": job["source"],
                                                                          "conversation": job["conversation"]}}))
print(json.dumps({"t0": t0, "engine_seconds": 0.01, "P": len(job["target"]["tokens"])}))
'''


# an engine of this release: it declares what it handles beyond older engines (engine.py
# FEATURES) and, like engine.py, exits with TEMPLATE_MISMATCH when it cannot reproduce the target's
# rendering (here: when KVPARK_FAKE_MISMATCH says so: "options", for jobs with template options, or "all")
FEATURED_ENGINE = 'FEATURES = ("media-source", "template-options")\n' + FAKE_ENGINE.replace(
    "assert open(", """import os
mismatch = os.environ.get("KVPARK_FAKE_MISMATCH")
if mismatch == "all" or mismatch == "options" and job["conversation"].get("template_options"):
    print("TemplateMismatch: the target server's tokens do not spell the target rendering", file=sys.stderr)
    sys.exit(4)
assert open(""", 1)


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.config_path = root / "bridge" / "bridge.json"
        self.slots = {name: root / f"slots-{name}" for name in ("a", "b")}
        for path in self.slots.values():
            path.mkdir()
        self.servers = {name: FakeLlama(self.slots[name], name) for name in ("a", "b")}
        self.gguf = write_gguf(root / "model.gguf", "qwen3")
        hub = root / "hub"
        hub.mkdir()
        (hub / "manifest.json").write_text(json.dumps({"format_version": 1, "models": {"a": {}, "b": {}}}))
        engine = root / "engine.py"
        engine.write_text(FAKE_ENGINE)
        config = registry.empty()
        for name in ("a", "b"):
            registry.add_model(config, name, url=self.servers[name].url, gguf=self.gguf, slot_dir=self.slots[name])
        registry.set_engine(config, python=sys.executable, script=engine, hub=hub)
        config["tail"] = 5
        config["min_handover"] = 0          # the fake conversations are tiny
        registry.save(config, self.config_path)
        self.config = registry.load(self.config_path)
        self.conversation = {"messages": [{"role": "system", "content": "be brief"},
                                          {"role": "user", "content": "tell me about parking caches"}]}

    def tearDown(self):
        for server in self.servers.values():
            server.close()
        self.tmp.cleanup()

    def switch(self, conversation=None, **kwargs):
        return bridge_switch.switch(self.config, conversation or self.conversation, "a", "b",
                                    config_path=self.config_path, **kwargs)

    def assert_no_bridge_files(self):
        for path in self.slots.values():
            self.assertEqual(sorted(p.name for p in path.iterdir()), [])
        self.assertEqual(list((self.config_path.parent / "jobs").glob("*")), [])

    def test_switch_saves_translates_restores_and_cleans_up(self):
        events = []
        self.assertNotIn("native_prefix_tokens", self.switch())      # opt-in per target
        self.config["models"]["b"]["native_prefix"] = True
        report = self.switch(log=events.append)
        prompt_tokens = len("<system>be brief<user>tell me about parking caches<assistant>")
        self.assertEqual(report["target_prompt_tokens"], prompt_tokens)
        self.assertEqual(report["engine"]["t0"], prompt_tokens - 5)
        self.assertEqual(report["restored_tokens"], prompt_tokens - 5)
        self.assertTrue(report["reuse_ok"])
        self.assertEqual(report["warm"]["prompt_n"], 5)
        self.assertEqual(report["draft"]["kind"], "none")
        self.assertFalse(report["resume_sidecar"])        # an upstream server: no .resume
        # no engine_daemon.py beside the fake engine: the one-shot engine ran, with no daemon error
        self.assertEqual(report["engine_mode"], "oneshot")
        self.assertNotIn("daemon_error", report)
        self.assertEqual([e["step"] for e in events], ["source-saved", "translated"])
        # nothing of the bridge is left in either slot directory or the job directory
        self.assert_no_bridge_files()
        # the native prefix covers the system message: the common start of the system-only and the
        # full rendering, "<system>be brief<" (17 single-character tokens), minus a margin of 2
        self.assertEqual(report["native_prefix_tokens"], 15)
        # the layout template and the native prefix are cached and reused
        templates = list((self.config_path.parent / "templates").iterdir())
        prefixes = list((self.config_path.parent / "prefix").iterdir())
        self.assertEqual((len(templates), len(prefixes)), (1, 1))
        completions = sum(1 for method, path in self.servers["b"].calls if path == "/completion")
        self.switch()
        self.assertEqual(list((self.config_path.parent / "templates").iterdir()), templates)
        self.assertEqual(list((self.config_path.parent / "prefix").iterdir()), prefixes)
        # only the warm request ran on the target the second time
        self.assertEqual(sum(1 for method, path in self.servers["b"].calls if path == "/completion"), completions + 1)
        # call order on the target: erase before restore, completion after
        posts = [path for method, path in self.servers["b"].calls if method == "POST"]
        self.assertEqual(posts[-3:], ["/slots/0?action=erase", "/slots/0?action=restore", "/completion"])

    def test_short_target_prompt_is_skipped_not_translated(self):
        self.config.pop("min_handover")
        report = self.switch()
        self.assertEqual(report["skipped"], "short")
        self.assertIn("prefills the conversation natively", report["note"])
        self.assertFalse(any("restore" in path or "save" in path for _, path in self.servers["b"].calls))
        self.assertFalse(any("save" in path for _, path in self.servers["a"].calls))      # no source save either
        for path in self.slots.values():
            self.assertEqual([p.name for p in path.iterdir()], [])

    def test_per_target_minimum_handover(self):
        self.config["models"]["b"]["min_handover"] = 10_000
        self.assertEqual(self.switch()["skipped"], "short")

    def test_per_target_tail_overrides_the_default(self):
        self.config["models"]["b"]["tail"] = 7
        report = self.switch()
        prompt_tokens = len("<system>be brief<user>tell me about parking caches<assistant>")
        self.assertEqual(report["engine"]["t0"], prompt_tokens - 7)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            registry.add_model(registry.empty(), "x", url="http://127.0.0.1:1", gguf=self.gguf,
                               slot_dir=self.slots["a"], tail=-1)

    def test_engine_failure_leaves_no_files_and_target_untouched(self):
        self.config["engine"]["script"] = str(Path(self.tmp.name) / "broken.py")
        Path(self.config["engine"]["script"]).write_text("import sys; sys.exit('engine exploded')")
        with self.assertRaisesRegex(RuntimeError, "engine exploded"):
            self.switch()
        self.assertFalse(any("restore" in path for _, path in self.servers["b"].calls))
        for path in self.slots.values():
            self.assertEqual([p.name for p in path.iterdir()], [])

    def fake_run_job(self, *, mode="daemon", calls=None):
        """An engine_client.run_job stand-in that writes the output as FAKE_ENGINE does."""
        def run_job(config, job, *, bridge_dir, timeout, log):
            if calls is not None:
                calls.append({"job": job, "bridge_dir": bridge_dir, "timeout": timeout})
            t0 = len(job["target"]["tokens"]) - job["tail"]
            Path(job["target"]["output"]).write_text(json.dumps({"tokens": t0, "job": {"tail": job["tail"]}}))
            return {"t0": t0, "engine_seconds": 0.01, "engine_mode": mode, "backend": "torch", "device": "cuda:0",
                    **({"daemon_error": "DaemonError: no daemon"} if mode == "oneshot" else {})}
        return run_job

    def test_the_engine_runs_on_the_daemon_with_the_job_inline(self):
        calls = []
        with mock.patch.object(bridge_switch.engine_client, "run_job", self.fake_run_job(calls=calls)):
            report = self.switch(engine_timeout=77)
        self.assertTrue(report["reuse_ok"])
        self.assertEqual((report["engine_mode"], report["engine"]["backend"]), ("daemon", "torch"))
        (call,) = calls
        self.assertEqual((call["bridge_dir"], call["timeout"]), (self.config_path.parent, 77))
        # the conversation travels inside the job; no job file is written for the daemon
        self.assertEqual(call["job"]["conversation"]["messages"][1]["content"], "tell me about parking caches")
        self.assertFalse((self.config_path.parent / "jobs").exists())
        self.assertNotIn("runtime", call["job"]["target"])      # no RoPE settings registered
        # a fallback to the one-shot engine is recorded with the daemon's error
        with mock.patch.object(bridge_switch.engine_client, "run_job", self.fake_run_job(mode="oneshot")):
            report = self.switch()
        self.assertEqual((report["engine_mode"], report["daemon_error"]), ("oneshot", "DaemonError: no daemon"))
        self.assert_no_bridge_files()

    def test_a_memory_refusal_is_a_skip_and_a_job_error_is_raised(self):
        def refuse(kind, message):
            def run_job(*args, **kwargs):
                raise engine_client.EngineError(f"bridge engine failed: {message}", kind)
            return run_job
        with mock.patch.object(bridge_switch.engine_client, "run_job",
                               refuse("memory", "MemoryError: memory: a switch needs about 9000 MiB of RAM")):
            report = self.switch()
        self.assertEqual(report["skipped"], "engine-memory")
        self.assertIn("memory guard", report["note"])
        self.assertIn("9000 MiB", report["note"])
        self.assertFalse(any("restore" in path for _, path in self.servers["b"].calls))
        self.assert_no_bridge_files()
        with mock.patch.object(bridge_switch.engine_client, "run_job", refuse("job", "ValueError: templates differ")):
            with self.assertRaisesRegex(RuntimeError, "bridge engine failed: ValueError: templates differ"):
                self.switch()
        self.assertFalse(any("restore" in path for _, path in self.servers["b"].calls))
        self.assert_no_bridge_files()

    def test_keep_job_copies_the_job_and_its_inputs(self):
        keep = Path(self.tmp.name) / "kept"
        with mock.patch.dict(os.environ, {"KVPARK_BRIDGE_KEEP_JOB": str(keep)}):
            report = self.switch()
        self.assertTrue(report["reuse_ok"])
        (job,) = keep.glob("job-*.json")
        if os.name == "posix":
            self.assertEqual((keep.stat().st_mode & 0o777, job.stat().st_mode & 0o777), (0o700, 0o600))
        kept = json.loads(job.read_text())
        self.assertEqual(kept["target"]["name"], "b")
        self.assertEqual(kept["conversation"]["messages"][0]["content"], "be brief")
        names = sorted(p.name for p in keep.iterdir())
        self.assertTrue(any(name.startswith("kvbridge-") for name in names))     # the source slot
        self.assertEqual(len(names), 3)                                        # job, source slot, template
        self.assert_no_bridge_files()                                          # the originals are gone

    def test_rope_settings_reach_the_engine_job(self):
        runtime = {"rope_scaling_type": "yarn", "rope_freq_scale": 0.25, "yarn_orig_ctx": 32768, "yarn_beta_fast": None}
        # the codec of today's translated architectures takes RoPE from the GGUF: refused up front
        with self.assertRaisesRegex(ValueError, "qwen3 caches with the GGUF's own RoPE"):
            registry.add_model(self.config, "b", url=self.servers["b"].url, gguf=self.gguf, slot_dir=self.slots["b"],
                               runtime=runtime)
        mla = write_gguf(Path(self.tmp.name) / "mla.gguf", "deepseek2")
        registry.add_model(self.config, "b", url=self.servers["b"].url, gguf=mla, slot_dir=self.slots["b"],
                           runtime=runtime)
        self.assertEqual(self.config["models"]["b"]["runtime"],
                         {"rope_scaling_type": "yarn", "rope_freq_scale": 0.25, "yarn_orig_ctx": 32768})
        with mock.patch.object(bridge_switch.capabilities, "check", return_value=None):   # the hub has no MLA yet
            report = self.switch()
        self.assertTrue(report["reuse_ok"])
        job = self.servers["b"].restored_payload["job"]
        self.assertEqual(job["target"]["runtime"], self.config["models"]["b"]["runtime"])
        self.assertNotIn("runtime", job["source"])
        for bad, error in (({"rope_scale": 2}, "unknown"), ({"rope_scaling_type": "longrope"}, "rope_scaling_type"),
                           ({"rope_freq_base": float("nan")}, "finite"), ({"yarn_orig_ctx": -1}, "nonnegative"),
                           ({"yarn_ext_factor": True}, "finite")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, error):
                registry.check_runtime(bad)

    def test_engine_settings(self):
        engine = Path(self.config["engine"]["script"])
        hub = Path(self.config["hub"])
        with self.assertRaisesRegex(ValueError, "first engine setting"):
            registry.set_engine(registry.empty(), python=sys.executable, script=engine)
        config = registry.set_engine(registry.empty(), python=sys.executable, script=engine, hub=hub, device="gpu",
                                     mode="oneshot", idle_unload=30, idle_exit=300.5, threads=12)
        self.assertEqual({k: config["engine"][k] for k in ("device", "mode", "idle_unload", "idle_exit", "threads")},
                         {"device": "gpu", "mode": "oneshot", "idle_unload": 30, "idle_exit": 300.5, "threads": 12})
        # later settings keep what they do not name
        registry.set_engine(config, device="cpu", threads=0)
        self.assertEqual((config["engine"]["device"], config["engine"]["mode"]), ("cpu", "oneshot"))
        self.assertNotIn("threads", config["engine"])
        self.assertEqual(config["engine"]["python"], sys.executable)
        for kwargs, error in (({"device": "tpu"}, "device"), ({"mode": "fast"}, "mode"),
                              ({"idle_exit": 0}, "positive"), ({"idle_unload": float("inf")}, "positive"),
                              ({"threads": -2}, "threads")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, error):
                registry.set_engine(config, **kwargs)
        # hub manifests of format 1 (fit_hub.py) and 2 (convert_hub.py); anything else is refused
        v2 = Path(self.tmp.name) / "hub-v2"
        v2.mkdir()
        (v2 / "manifest.json").write_text(json.dumps({"format_version": 2, "models": {"a": {}, "c": {}}}))
        registry.set_engine(config, hub=v2)
        self.assertEqual((registry.hub_format(v2), registry.hub_models(config)), (2, {"a", "c"}))
        (v2 / "manifest.json").write_text(json.dumps({"format_version": 3, "models": {}}))
        with self.assertRaisesRegex(ValueError, "format 3"):
            registry.set_engine(config, hub=v2)
        with self.assertRaisesRegex(ValueError, "no manifest"):
            registry.set_engine(config, hub=Path(self.tmp.name))

    def run_cli(self, *argv):
        import argparse
        parser = argparse.ArgumentParser()
        bridge_cli.add_parser(parser.add_subparsers(dest="command"))
        with mock.patch("builtins.print") as printed:
            bridge_cli.run(parser.parse_args(["bridge", *argv, "--config", str(self.config_path)]))
        return printed

    def test_cli_engine_options_status_and_stop(self):
        printed = self.run_cli("engine", "--device", "cpu", "--mode", "oneshot", "--idle-unload", "30",
                               "--threads", "6")
        out = json.loads(printed.call_args[0][0])
        self.assertEqual((out["engine"]["device"], out["engine"]["mode"], out["engine"]["threads"]), ("cpu", "oneshot", 6))
        self.assertFalse(out["daemon_stopped"])
        stored = registry.load(self.config_path)["engine"]
        self.assertEqual((stored["idle_unload"], stored["python"]), (30.0, sys.executable))
        self.assertEqual(json.loads(self.run_cli("engine-status").call_args[0][0])["running"], False)
        self.assertEqual(self.run_cli("engine-stop").call_args[0][0], "no engine daemon is running")
        # a running daemon is stopped when the engine settings change (it keeps its old ones)
        with mock.patch.object(bridge_cli.engine_client, "stop", return_value=True) as stop:
            out = json.loads(self.run_cli("engine", "--device", "auto").call_args[0][0])
            self.assertEqual(self.run_cli("engine-stop").call_args[0][0], "stopped the engine daemon")
        self.assertTrue(out["daemon_stopped"])
        self.assertEqual(stop.call_args[0][0], self.config_path.parent)
        with mock.patch.object(bridge_cli.engine_client, "status", return_value={"ok": True, "pid": 42, "jobs": 3}):
            self.assertEqual(json.loads(self.run_cli("engine-status").call_args[0][0])["pid"], 42)

    def test_cli_add_rope_flags(self):
        mla = write_gguf(Path(self.tmp.name) / "mla.gguf", "deepseek2")
        self.run_cli("add", "c", "--url", "http://127.0.0.1:9", "--gguf", str(mla), "--slot-dir",
                     str(self.slots["a"]), "--rope-scaling", "yarn", "--rope-scale", "4", "--yarn-orig-ctx", "32768",
                     "--rope-freq-base", "1000000")
        self.assertEqual(registry.load(self.config_path)["models"]["c"]["runtime"],
                         {"rope_scaling_type": "yarn", "rope_freq_base": 1000000.0, "rope_freq_scale": 0.25,
                          "yarn_orig_ctx": 32768})
        self.run_cli("add", "d", "--url", "http://127.0.0.1:9", "--gguf", str(self.gguf), "--slot-dir",
                     str(self.slots["a"]))
        self.assertNotIn("runtime", registry.load(self.config_path)["models"]["d"])
        with self.assertRaisesRegex(ValueError, "not both"):
            self.run_cli("add", "e", "--url", "http://127.0.0.1:9", "--gguf", str(self.gguf), "--slot-dir",
                         str(self.slots["a"]), "--rope-scale", "4", "--rope-freq-scale", "0.25")

    def hybrid_gguf(self):
        return write_gguf(Path(self.tmp.name) / "hybrid.gguf", "qwen35", qwen35__block_count=9,
                          qwen35__nextn_predict_layers=1, qwen35__full_attention_interval=4,
                          qwen35__ssm__conv_kernel=4, qwen35__ssm__inner_size=64, qwen35__ssm__state_size=16,
                          qwen35__ssm__group_count=2)

    def test_recurrent_mean_registration(self):
        root = Path(self.tmp.name)
        hybrid, mean = self.hybrid_gguf(), write_recurrent_mean(root / "mean.npz", qwen35_mean_meta())

        def add(name, gguf=hybrid, **options):
            return registry.add_model(registry.empty(), name, url="http://127.0.0.1:9", gguf=gguf,
                                      slot_dir=self.slots["a"], **options)["models"][name]
        entry = add("h", recurrent_mean=mean)
        self.assertEqual((entry["recurrent_mean"], "recurrent_tools" in entry), (str(mean.resolve()), False))
        self.assertEqual(registry.recurrent_mean_meta(mean)["groups"]["chat"], {"count": 2})
        tools = write_recurrent_mean(root / "tools.npz", qwen35_mean_meta(groups={"all": {}, "chat": {}, "tools": {}}))
        self.assertTrue(add("h", recurrent_mean=tools, recurrent_tools=True)["recurrent_tools"])
        refusals = [({"gguf": self.gguf, "recurrent_mean": mean}, "only hybrid models"),
                    ({"recurrent_mean": root / "missing.npz"}, "is not a file"),
                    ({"recurrent_mean": self.gguf}, "not a recurrent mean file"),
                    ({"recurrent_tools": True}, "needs a recurrent mean"),
                    ({"recurrent_mean": mean, "recurrent_tools": True}, "no tools group"),
                    ({"recurrent_mean": write_recurrent_mean(root / "v2.npz", qwen35_mean_meta(format_version=2))},
                     "of format 1"),
                    ({"recurrent_mean": write_recurrent_mean(root / "muse.npz", qwen35_mean_meta(
                        model={"architecture": "muse-glimmer"}))}, "of a muse-glimmer model"),
                    ({"recurrent_mean": write_recurrent_mean(root / "deep.npz", qwen35_mean_meta(n_layer=64))},
                     "64-layer model; this one has 8"),
                    ({"recurrent_mean": write_recurrent_mean(root / "layers.npz", qwen35_mean_meta(
                        layers=[0, 1, 2, 3, 4, 5]))}, "layout's layers differ"),
                    ({"recurrent_mean": write_recurrent_mean(root / "rows.npz", qwen35_mean_meta(
                        s_bytes=[4] * 6))}, "layout's s_bytes differ")]
        for options, message in refusals:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                add("h", **options)
        self.run_cli("add", "c", "--url", "http://127.0.0.1:9", "--gguf", str(hybrid), "--slot-dir",
                     str(self.slots["a"]), "--recurrent-mean", str(tools), "--recurrent-tools")
        stored = registry.load(self.config_path)["models"]["c"]
        self.assertEqual((stored["recurrent_mean"], stored["recurrent_tools"]), (str(tools.resolve()), True))

    def test_the_recurrent_mean_reaches_hybrid_targets_only(self):
        mean = write_recurrent_mean(Path(self.tmp.name) / "mean.npz", qwen35_mean_meta())
        self.config["models"]["b"].update(arch="qwen35", recurrent_mean=str(mean), recurrent_tools=True)
        report = self.switch()
        target = self.servers["b"].restored_payload["job"]["target"]
        self.assertEqual((target["recurrent_mean"], target["recurrent_tools"]), (str(mean), True))
        # this engine reports nothing about it: an older engine, which writes zeros as it always did
        self.assertEqual(report["recurrent"], {"strategy": "zero", "note": "the engine takes no recurrent mean"})
        engine = Path(self.config["engine"]["script"])
        engine.write_text(FAKE_ENGINE.replace('print(json.dumps({"t0": t0,', 'print(json.dumps({"recurrent": '
                                              '{"strategy": "mean", "group": "all"}, "t0": t0,'))
        self.assertEqual(self.switch()["recurrent"], {"strategy": "mean", "group": "all"})
        # a target that keeps no recurrent state never gets it (an entry from before its GGUF changed)
        self.config["models"]["b"]["arch"] = "qwen3"
        report = self.switch()
        self.assertNotIn("recurrent_mean", self.servers["b"].restored_payload["job"]["target"])
        self.assertNotIn("recurrent", report)

    def test_server_template_flags_reach_the_engine_job(self):
        # the target server's own template flags, as kvpark bridge add takes them (llama-server's names)
        self.run_cli("add", "b", "--url", self.servers["b"].url, "--gguf", str(self.gguf), "--slot-dir",
                     str(self.slots["b"]), "--chat-template-kwargs", '{"reasoning_effort": "max", "x": [1]}',
                     "--reasoning", "off", "--reasoning-effort", "low", "--no-reasoning-preserve")
        flags = {"reasoning_effort": "low", "x": [1], "enable_thinking": False, "preserve_reasoning": False}
        self.config = registry.load(self.config_path)
        self.config["min_handover"] = 0
        self.assertEqual(self.config["models"]["b"]["template_kwargs"], flags)
        self.assertTrue(self.switch()["reuse_ok"])
        self.assertEqual(self.servers["b"].restored_payload["job"]["target"]["template_kwargs"], flags)
        # a server without such flags: nothing is passed (the engine tries llama-server's defaults)
        self.run_cli("add", "b", "--url", self.servers["b"].url, "--gguf", str(self.gguf), "--slot-dir",
                     str(self.slots["b"]))
        self.config = registry.load(self.config_path)
        self.config["min_handover"] = 0
        self.assertNotIn("template_kwargs", self.config["models"]["b"])
        self.assertTrue(self.switch()["reuse_ok"])
        self.assertNotIn("template_kwargs", self.servers["b"].restored_payload["job"]["target"])

    def test_server_template_kwargs_mirror_llama_server_flags(self):
        from kvpark.bridge import messages
        kwargs = messages.server_template_kwargs
        self.assertIsNone(kwargs())
        self.assertEqual(kwargs(reasoning="auto"), {"preserve_reasoning": True})
        self.assertEqual(kwargs(reasoning="on", reasoning_effort="xhigh", reasoning_preserve=True),
                         {"enable_thinking": True, "reasoning_effort": "xhigh", "preserve_reasoning": True})
        # --reasoning-effort default drops an effort --chat-template-kwargs set; flags after the kwargs win
        self.assertEqual(kwargs(chat_template_kwargs={"reasoning_effort": "low", "preserve_reasoning": False},
                                reasoning_effort="default"), {"preserve_reasoning": False})
        self.assertEqual(kwargs(chat_template_kwargs='{"enable_thinking": true}', reasoning="disabled"),
                         {"enable_thinking": False, "preserve_reasoning": True})
        for bad, error in (({"chat_template_kwargs": "[1]"}, "JSON object"), ({"chat_template_kwargs": "{"}, "JSON"),
                           ({"reasoning": "maybe"}, "--reasoning"), ({"reasoning_effort": ""}, "level"),
                           ({"chat_template_kwargs": {"enable_thinking": "false"}}, "boolean")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, error):
                kwargs(**bad)
        with self.assertRaisesRegex(ValueError, "JSON values"):
            registry.add_model(self.config, "c", url="http://127.0.0.1:9", gguf=self.gguf, slot_dir=self.slots["a"],
                               template_kwargs={"x": object()})

    def test_status_shows_the_daemon_and_warns_about_the_engine(self):
        stats = {"ok": True, "pid": 7, "device": "auto", "uptime_s": 5.0, "idle_s": 1.0, "jobs": 2, "failures": 0,
                 "gpu_fault": "HIP error: illegal address", "features": ["media-source", "template-options"],
                 "backends": {"torch": {"resident_bytes": 1000}, "numpy": {"resident_bytes": 24}},
                 "engine": {"blas": {"name": "blas", "reference": True}, "gpu": {"available": True}}}
        with mock.patch.object(bridge_cli.engine_client, "status", return_value=stats), \
                mock.patch.object(bridge_cli.engine_client, "check") as check:
            printed = self.run_cli("status")
        out = json.loads(printed.call_args[0][0])
        check.assert_not_called()                    # the running daemon answered for the engine Python
        self.assertEqual(out["engine_daemon"], {"running": True, "pid": 7, "device": "auto",
                                                "backends": ["numpy", "torch"], "uptime_s": 5.0, "idle_s": 1.0,
                                                "jobs": 2, "failures": 0, "resident_bytes": 1024,
                                                "gpu_fault": "HIP error: illegal address",
                                                "features": ["media-source", "template-options"]})
        warnings = " | ".join(out["warnings"])
        for text in ("reference BLAS", "stopped using the GPU", "format 1"):
            self.assertIn(text, warnings)
        # also on stderr, before the JSON
        self.assertTrue(any("reference BLAS" in c.args[0] and c.kwargs.get("file") is sys.stderr
                            for c in printed.call_args_list))
        # no daemon: the engine Python is asked directly
        self.config["engine"]["device"] = "gpu"
        registry.save(self.config, self.config_path)
        report = {"blas": {"name": "scipy-openblas", "reference": False}, "gpu": {"available": False,
                                                                                 "detail": "torch unavailable"}}
        with mock.patch.object(bridge_cli.engine_client, "check", return_value=report):
            out = json.loads(self.run_cli("status").call_args[0][0])
        self.assertEqual((out["engine_daemon"], out["engine_check"]), ({"running": False}, report))
        self.assertTrue(any("has no GPU" in w for w in out["warnings"]))
        self.assertFalse(any("BLAS" in w for w in out["warnings"]))
        with mock.patch.object(bridge_cli.engine_client, "check",
                               side_effect=engine_client.DaemonError("the engine check failed: boom")):
            out = json.loads(self.run_cli("status").call_args[0][0])
        self.assertEqual(out["engine_check"], {"error": "the engine check failed: boom"})
        self.assertTrue(any(w.startswith("the engine check failed: boom") for w in out["warnings"]))
        # a daemon that cannot run makes every switch one-shot: no daemon script, or a socket path too long
        self.assertTrue(any("no engine_daemon.py" in w for w in out["warnings"]) or os.name == "nt")
        Path(self.config["engine"]["script"]).with_name("engine_daemon.py").write_text("")
        with mock.patch.object(bridge_cli.engine_client, "check", return_value=report):
            out = json.loads(self.run_cli("status").call_args[0][0])
            self.assertFalse(any("one-shot" in w for w in out["warnings"]))
            with mock.patch.object(bridge_cli.engine_client, "daemon_supported", return_value=False):
                out = json.loads(self.run_cli("status").call_args[0][0])
            self.assertTrue(any("socket path" in w for w in out["warnings"]) or os.name == "nt")
            self.config["engine"]["mode"] = "oneshot"                 # chosen: nothing to warn about
            registry.save(self.config, self.config_path)
            with mock.patch.object(bridge_cli.engine_client, "daemon_supported", return_value=False):
                out = json.loads(self.run_cli("status").call_args[0][0])
            self.assertFalse(any("one-shot" in w for w in out["warnings"]))

    def test_request_options_that_change_the_rendering_skip_the_switch(self):
        from kvpark.bridge.messages import template_options
        request = {"messages": [], "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                   "reasoning_effort": "high", "temperature": 0}
        options = template_options(request)
        self.assertEqual(options, {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "high"})
        self.assertEqual(template_options({"messages": [], "extra_body": None}), {})
        # the real request renders differently from its first tokens: nothing translated would be reused
        self.config["min_handover"] = 4
        report = self.switch(template_options=options)
        self.assertEqual(report["skipped"], "template-options")
        self.assertIn("chat_template_kwargs", report["note"])
        self.assertEqual(report["template_options"]["shared_tokens"], 1)          # "<"
        self.assertFalse(any("save" in path for _, path in self.servers["a"].calls))
        self.config["min_handover"] = 0
        rendered = [body for path, body in self.servers["b"].bodies if path == "/apply-template"]
        self.assertEqual(rendered[-1]["reasoning_effort"], "high")              # passed as the request has them
        # options that render the same (here: an effort this template ignores) switch as usual
        self.assertTrue(self.switch(template_options={"reasoning_effort": "high"})["reuse_ok"])
        self.assert_no_bridge_files()
        # the CLI reads them from the conversation file
        self.config["min_handover"] = 4
        registry.save(self.config, self.config_path)
        path = Path(self.tmp.name) / "conversation.json"
        path.write_text(json.dumps({**self.conversation, "chat_template_kwargs": {"enable_thinking": False}}))
        printed = self.run_cli("switch", "--from", "a", "--to", "b", "--conversation", str(path))
        self.assertEqual(json.loads(printed.call_args[0][0])["skipped"], "template-options")

    def test_request_options_that_change_the_rendering_late_cap_t0(self):
        options = {"chat_template_kwargs": {"enable_thinking": False}}
        prompt = "<system>be brief<user>tell me about parking caches<assistant>"
        # a classic Qwen3 template: only the generation prompt changes, after every message
        self.servers["b"].thinking = "suffix"
        report = self.switch(template_options=options)
        self.assertEqual(report["template_options"], {"options": ["chat_template_kwargs"], "shared_tokens": len(prompt),
                                                      "engine": False})
        self.assertEqual(report["engine_features"], [])         # FAKE_ENGINE declares none: an older engine
        self.assertEqual(report["restored_tokens"], len(prompt) - 5)               # P - tail, as without options
        self.assertTrue(report["reuse_ok"])
        # the warm request is the real request's rendering, which reuses the translated cells
        self.assertEqual(self.servers["b"].bodies[-1][1]["prompt"][-len("<think></think>"):],
                         [ord(c) for c in "<think></think>"])
        # a setting written before the newest message: T0 stops where the renderings part
        self.servers["b"].thinking = "newest"
        report = self.switch(template_options=options)
        shared = len("<system>be brief<")                                       # "<no-think>" vs "<user>"
        self.assertEqual((report["template_options"]["shared_tokens"], report["restored_tokens"]), (shared, shared))
        self.assertEqual(self.servers["b"].restored_payload["job"]["tail"], len(prompt) - shared)
        self.assertTrue(report["reuse_ok"])
        self.assert_no_bridge_files()

    def featured(self):
        """Use an engine of this release (FEATURED_ENGINE: it declares its features)."""
        script = Path(self.tmp.name) / "featured" / "engine.py"
        script.parent.mkdir(exist_ok=True)
        script.write_text(FEATURED_ENGINE)
        self.config["engine"]["script"] = str(script)

    def test_an_engine_that_renders_template_options_translates_the_request(self):
        self.featured()
        self.assertEqual(engine_client.script_features(self.config["engine"]["script"]),
                         {"media-source", "template-options"})
        options = {"chat_template_kwargs": {"enable_thinking": False}}
        request = "<no-think><system>be brief<user>tell me about parking caches<assistant>"
        # Qwen3.8-like: the options change the system prompt, which an older engine had to skip
        report = self.switch(template_options=options)
        self.assertNotIn("skipped", report)
        self.assertEqual(report["template_options"], {"options": ["chat_template_kwargs"], "shared_tokens": 1,
                                                      "engine": True})
        self.assertEqual(report["engine_features"], ["media-source", "template-options"])
        job = self.servers["b"].restored_payload["job"]
        self.assertEqual(job["conversation"]["template_options"], options)       # the engine renders with them
        self.assertEqual((job["tail"], report["restored_tokens"]), (5, len(request) - 5))
        self.assertEqual(report["target_prompt_tokens"], len(request))
        self.assertTrue(report["reuse_ok"])                     # the warm request is the request's rendering
        # options that render as the defaults do are not sent to the engine at all
        report = self.switch(template_options={"reasoning_effort": "high"})
        self.assertNotIn("template_options", report)
        self.assertNotIn("template_options", self.servers["b"].restored_payload["job"]["conversation"])
        self.assert_no_bridge_files()

    def test_a_template_mismatch_falls_back_to_the_default_rendering_or_skips(self):
        self.featured()
        options = {"chat_template_kwargs": {"enable_thinking": False}}
        events = []
        with mock.patch.dict(os.environ, {"KVPARK_FAKE_MISMATCH": "options"}):
            # the engine cannot reproduce the request's rendering: the default rendering is translated
            # up to the tokens both share, as with an older engine
            self.servers["b"].thinking = "newest"
            report = self.switch(template_options=options, log=events.append)
            shared = len("<system>be brief<")
            self.assertEqual(report["restored_tokens"], shared)
            self.assertFalse(report["template_options"]["engine"])
            self.assertIn("TemplateMismatch", report["template_options"]["engine_error"])
            self.assertIn("template-fallback", [e["step"] for e in events])
            self.assertNotIn("template_options", self.servers["b"].restored_payload["job"]["conversation"])
            self.assertTrue(report["reuse_ok"])
            # here nothing would be left to translate: skipped as before
            self.servers["b"].thinking = "start"
            self.config["min_handover"] = 4
            self.assertEqual(self.switch(template_options=options)["skipped"], "template-options")
        self.config["min_handover"] = 0
        # the default rendering itself does not match: skipped, the target untouched
        restores = sum(1 for _, path in self.servers["b"].calls if "restore" in path)
        with mock.patch.dict(os.environ, {"KVPARK_FAKE_MISMATCH": "all"}):
            report = self.switch()
        self.assertEqual(report["skipped"], "template-mismatch")
        self.assertIn("TemplateMismatch", report["note"])
        self.assertEqual(sum(1 for _, path in self.servers["b"].calls if "restore" in path), restores)
        # the daemon reports the same as an EngineError of kind "template"
        def mismatch(*args, **kwargs):
            raise engine_client.EngineError("bridge engine failed: TemplateMismatch: no match", "template")
        with mock.patch.object(bridge_switch.engine_client, "run_job", mismatch):
            self.assertEqual(self.switch()["skipped"], "template-mismatch")
        self.assert_no_bridge_files()

    def test_dsa_targets_need_indexer_keys_for_long_prompts(self):
        dsa = write_gguf(Path(self.tmp.name) / "dsa.gguf", "deepseek32", deepseek32__attention__indexer__top_k=8)
        self.config["models"]["b"].update(gguf=str(dsa), arch="deepseek32")
        report = self.switch()
        self.assertEqual(report["skipped"], "no-indexer-keys")
        self.assertIn("beyond 8 cells", report["note"])
        self.assertFalse(any("save" in path for _, path in self.servers["a"].calls))     # before the source's save
        # with the hub's indexer ridge for the target the switch goes ahead
        manifest = Path(self.config["hub"]) / "manifest.json"
        manifest.write_text(json.dumps({"format_version": 1, "models": {"a": {}, "b": {}},
                                        "indexer_ridges": {"b": {"layers": [0]}}}))
        self.assertEqual(registry.hub_indexer_ridges(self.config), {"b"})
        self.assertTrue(self.switch()["reuse_ok"])
        # the engine's own refusal (its rule is the final word) is the same skip
        def refuse(*args, **kwargs):
            raise engine_client.EngineError("bridge engine failed: IndexerKeysUnavailable: no indexer keys for b: "
                                            "T0 = 40 > top_k (8)", "job")
        with mock.patch.object(bridge_switch.engine_client, "run_job", refuse):
            report = self.switch()
        self.assertEqual(report["skipped"], "no-indexer-keys")
        self.assertIn("no indexer keys for b", report["note"])
        self.assert_no_bridge_files()
        # and the one-shot engine's: a traceback whose last line is the refusal (a RuntimeError)
        Path(self.config["engine"]["script"]).write_text(
            "import sys\nprint('Traceback (most recent call last):', file=sys.stderr)\n"
            "sys.exit('translate.IndexerKeysUnavailable: no indexer keys for b: T0 = 40 > top_k (8)')\n")
        report = self.switch()
        self.assertEqual((report["skipped"], report.get("engine_mode")), ("no-indexer-keys", None))
        self.assertTrue(report["note"].startswith("translate.IndexerKeysUnavailable: no indexer keys for b"))
        self.assert_no_bridge_files()
        # a DeepSeek V4 target keeps its indexer keys in pooled LID rows, which its head decoders fill
        dsv4 = write_gguf(Path(self.tmp.name) / "dsv4.gguf", "deepseek4", deepseek4__attention__indexer__top_k=8)
        self.assertIsNone(capabilities.indexer("b", dsv4, set(), 4096))
        self.assertEqual(capabilities.indexer("b", dsa, set(), 4096)[0], "no-indexer-keys")

    def test_engine_interpreter_symlink_is_kept(self):
        # a venv's python is a symlink; resolving it would run the base interpreter without the venv
        link = Path(self.tmp.name) / "venv-python"
        link.symlink_to(sys.executable)
        hub = Path(self.config["hub"])
        config = registry.set_engine(registry.empty(), python=link, script=Path(self.config["engine"]["script"]),
                                     hub=hub)
        self.assertEqual(config["engine"]["python"], str(link))

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "same model"):
            bridge_switch.switch(self.config, self.conversation, "a", "a", config_path=self.config_path)
        with self.assertRaisesRegex(ValueError, "unknown bridge model"):
            bridge_switch.switch(self.config, self.conversation, "a", "zz", config_path=self.config_path)
        self.config["models"]["b"]["hub_name"] = "other"
        with self.assertRaisesRegex(ValueError, "no adapter"):
            self.switch()
        self.config["models"]["b"]["hub_name"] = "b"
        self.config["models"]["b"]["url"] = self.config["models"]["a"]["url"]
        with self.assertRaisesRegex(ValueError, "same server"):
            self.switch()
        with self.assertRaisesRegex(ValueError, "local http"):
            registry.check_url("http://example.com:8080")
        with self.assertRaisesRegex(ValueError, "slot-save-path"):
            registry.add_model(registry.empty(), "x", url="http://127.0.0.1:1", gguf=self.gguf,
                               slot_dir=Path(self.tmp.name) / "missing")
        not_gguf = Path(self.tmp.name) / "notes.txt"
        not_gguf.write_text("hello")
        with self.assertRaisesRegex(ValueError, "not a readable GGUF"):
            registry.add_model(registry.empty(), "x", url="http://127.0.0.1:1", gguf=not_gguf, slot_dir=self.slots["a"])
        with self.assertRaisesRegex(ValueError, "draft_window"):
            registry.add_model(registry.empty(), "x", url="http://127.0.0.1:1", gguf=self.gguf,
                               slot_dir=self.slots["a"], draft_window=0)

    def test_cli_round_trip(self):
        import argparse
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        bridge_cli.add_parser(sub)
        conversation = Path(self.tmp.name) / "conversation.json"
        conversation.write_text(json.dumps(self.conversation))
        with mock.patch("builtins.print") as printed:
            bridge_cli.run(parser.parse_args(["bridge", "status", "--config", str(self.config_path)]))
            status = json.loads(printed.call_args[0][0])
        self.assertEqual(sorted(status["models"]), ["a", "b"])
        self.assertTrue(status["models"]["a"]["in_hub"])
        self.assertEqual(status["models"]["a"]["arch"], "qwen3")
        with mock.patch("builtins.print") as printed:
            bridge_cli.run(parser.parse_args(["bridge", "switch", "--from", "a", "--to", "b", "--conversation",
                                              str(conversation), "--config", str(self.config_path)]))
        self.assertTrue(json.loads(printed.call_args[0][0])["reuse_ok"])
        with mock.patch("builtins.print") as printed:
            bridge_cli.run(parser.parse_args(["bridge", "add", "c", "--url", "http://127.0.0.1:9", "--gguf",
                                              str(self.gguf), "--slot-dir", str(self.slots["a"]), "--draft-window",
                                              "4096", "--config", str(self.config_path)]))
        self.assertEqual(printed.call_args[0][0], "registered c (qwen3)")
        self.assertEqual(registry.load(self.config_path)["models"]["c"]["draft_window"], 4096)
        with mock.patch("builtins.print"):
            bridge_cli.run(parser.parse_args(["bridge", "remove", "a", "--config", str(self.config_path)]))
        self.assertEqual(sorted(registry.load(self.config_path)["models"]), ["b", "c"])

    # -- capabilities ------------------------------------------------------------------------

    def test_architectures_the_engine_cannot_handle_are_skipped(self):
        root = Path(self.tmp.name)
        for side, arch, code in (("b", "mamba2", "no-per-token-cache"), ("a", "rwkv7", "no-per-token-cache"),
                                 ("a", "deepseek4", "dsv4-source"), ("b", "deepseek4", "no-head-decoders"),
                                 ("b", "llama", "unsupported-arch")):
            with self.subTest(side=side, arch=arch):
                self.config["models"][side]["gguf"] = str(write_gguf(root / f"{arch}.gguf", arch))
                self.config["models"][side]["arch"] = arch
                # such a model has no hub adapter either: the skip comes first, not the missing adapter
                self.config["models"][side]["hub_name"] = f"{arch}-model"
                report = self.switch()
                self.assertEqual(report["skipped"], code)
                self.assertIn(arch, report["note"])
                self.config["models"][side].update(gguf=str(self.gguf), arch="qwen3", hub_name=side)
        self.assertEqual(self.servers["a"].calls, [])          # decided before any server is touched
        self.assertEqual(self.servers["b"].calls, [])

    def test_translatable_architectures_are_read_by_the_state_codec(self):
        if not KVSTATE.is_file():
            self.skipTest("no experiments/linear_transfer/kvstate.py in this checkout")
        constants = module_constants(KVSTATE)
        codec = set(constants["ARCHITECTURES"])
        self.assertTrue({"qwen35", "muse-glimmer", "qwen3"} <= codec)
        # the gate never lets through a model whose state the engine cannot even parse
        self.assertLessEqual(set(capabilities.TRANSLATABLE), codec)
        # kvpark bridge add refuses RoPE settings for exactly the codec's dense architectures
        # (DeepSeek V4's pooled-cache codec takes the server's settings, like MLA and the hybrids)
        dense = codec - set(constants["MLA_ARCHITECTURES"] + constants["HYBRID_ARCHITECTURES"]
                            + constants["RECURRENT_ARCHITECTURES"]) - {"deepseek4"}
        self.assertEqual(set(registry.FIXED_ROPE_ARCHS), dense)
        # the hybrids (a recurrent cell beside attention: kvpark bridge add --recurrent-mean) are the codec's
        # GQA hybrids, Qwen3.5 and the MLA family's K-only hybrids
        self.assertLessEqual(set(constants["HYBRID_ARCHITECTURES"]), capabilities.HYBRID)
        self.assertEqual(capabilities.HYBRID - set(constants["HYBRID_ARCHITECTURES"] + constants["MLA_ARCHITECTURES"]),
                         {"qwen35"})
        self.assertLessEqual(capabilities.HYBRID, capabilities.TRANSLATABLE)

    def test_architecture_is_read_lazily_for_older_entries(self):
        stored = registry.load(self.config_path)
        for name in ("a", "b"):
            del stored["models"][name]["arch"]
        registry.save(stored, self.config_path)
        self.config = registry.load(self.config_path)
        self.assertTrue(self.switch()["reuse_ok"])
        self.assertEqual(registry.load(self.config_path)["models"]["b"]["arch"], "qwen3")    # written back
        self.config["models"]["b"].pop("arch")
        self.config["models"]["b"]["gguf"] = str(Path(self.tmp.name) / "gone.gguf")
        report = self.switch()
        self.assertEqual(report["skipped"], "unsupported-arch")
        self.assertIn("cannot read", report["note"])

    def test_gguf_reader(self):
        path = write_gguf(Path(self.tmp.name) / "m.gguf", "muse-glimmer")
        self.assertEqual(capabilities.gguf_architecture(path), "muse-glimmer")
        self.assertEqual(capabilities.gguf_value(path, "x.count"), 7)
        self.assertIsNone(capabilities.gguf_value(path, "x.missing"))
        self.assertIsNone(capabilities.check(("a", "qwen35"), ("b", "muse-glimmer")))

    # -- drafts and .resume sidecars ------------------------------------------------------------

    def test_patched_target_with_a_dflash_draft_gets_an_aligned_draft(self):
        target = self.servers["b"]
        target.patched, target.speculative = True, "draft-dflash"
        target.draft = lambda n: dflash_template_draft(n, ext=True)
        self.config["models"]["b"]["draft_window"] = 9
        report = self.switch()
        prompt_tokens = len("<system>be brief<user>tell me about parking caches<assistant>")
        # the native tail covers the draft's window minus one
        self.assertEqual(report["engine"]["t0"], prompt_tokens - 8)
        self.assertEqual(target.restored_payload["job"]["tail"], 8)
        self.assertTrue(report["reuse_ok"] and report["resume_sidecar"])
        self.assertEqual((report["draft"]["kind"], report["draft"]["window"]), ("dflash", 9))
        self.assertEqual(report["draft"]["speculative"], "draft-dflash")
        restored = resume.parse_state_blob(target.restored_resume.draft)
        t0 = report["engine"]["t0"]
        for section in restored.sections:
            self.assertEqual(section.streams[0].positions, (t0 - 2, t0 - 1))
            self.assertEqual(section.streams[0].ext, ((t0 - 2, t0 - 2, -1), (t0 - 1, t0 - 1, -1)))
        self.assertEqual((target.restored_resume.speculative, target.restored_resume.checkpoints), (b"", []))
        self.assertEqual(report["draft"]["blob_bytes"], len(target.restored_resume.draft))
        # the template keeps its .resume; everything else is gone
        templates = sorted(p.name for p in (self.config_path.parent / "templates").iterdir())
        self.assertEqual([name.endswith(".resume") for name in templates], [False, True])
        self.assert_no_bridge_files()
        # a short conversation once the DFlash tail applies is skipped
        self.config["models"]["b"]["draft_window"] = 2048
        report = self.switch()
        self.assertEqual(report["skipped"], "short")
        self.assertIn("DFlash", report["note"])

    def test_patched_target_with_mtp_gets_an_empty_marker(self):
        target = self.servers["b"]
        target.patched, target.speculative, target.draft = True, "none,draft-mtp", mtp_template_draft
        report = self.switch()
        self.assertEqual(report["draft"]["kind"], "mtp")
        self.assertEqual(target.restored_resume.draft, struct.pack("<IiII", resume.IO_MAGIC, 0, 1, 0))
        self.assertEqual(report["engine"]["t0"], report["target_prompt_tokens"] - 5)     # no tail rule

    def test_patched_target_without_a_draft_gets_an_empty_draft_blob(self):
        self.servers["b"].patched = True
        report = self.switch()
        self.assertTrue(report["reuse_ok"])
        self.assertEqual((report["draft"]["kind"], self.servers["b"].restored_resume.draft), ("none", b""))

    def test_upstream_dflash_server_still_gets_the_draft_tail(self):
        self.servers["b"].speculative = "draft-dflash"
        self.config["models"]["b"]["draft_window"] = 10
        report = self.switch()
        self.assertEqual((report["draft"]["kind"], report["resume_sidecar"]), ("dflash", False))
        self.assertEqual(report["engine"]["t0"], report["target_prompt_tokens"] - 9)
        # the speculative setup was read from /slots after a one-token task on the untouched slot
        self.assertEqual(report["draft"]["speculative"], "draft-dflash")

    def test_unsupported_draft_is_skipped(self):
        target = self.servers["b"]
        target.patched, target.speculative = True, "draft-simple"
        target.draft = lambda n: struct.pack("<IiII", resume.IO_MAGIC, 0, 1, 0) + b"junk"
        report = self.switch()
        self.assertEqual(report["skipped"], "draft-unsupported")
        self.assertFalse(any("save" in path for _, path in self.servers["a"].calls))

    def test_one_layer_draft_that_is_not_mtp_is_skipped(self):
        # an EAGLE3 (or plain one-layer) draft saves a state shaped like MTP's; /slots tells them apart
        target = self.servers["b"]
        target.patched, target.speculative, target.draft = True, "draft-eagle3", mtp_template_draft
        report = self.switch()
        self.assertEqual((report["skipped"], report["draft"]["kind"]), ("draft-unsupported", "other"))
        self.assertIn("draft-eagle3", report["note"])
        self.assertFalse(any("save" in path for _, path in self.servers["a"].calls))

    def test_upstream_deepseek4_target_with_a_draft_is_skipped(self):
        # an upstream server keeps no .resume: its draft layout comes from the speculative types, and a
        # DeepSeek V4 target restores only without a draft (capabilities.draft)
        manifest = json.loads((Path(self.config["hub"]) / "manifest.json").read_text())
        manifest["heads"] = {kind: {"base": "k", "models": {"b": {"layers": [2]}}} for kind in capabilities.DSV4_HEADS}
        (Path(self.config["hub"]) / "manifest.json").write_text(json.dumps(manifest))
        self.config["models"]["b"]["arch"] = "deepseek4"
        self.servers["b"].speculative = "draft-mtp"
        report = self.switch()
        self.assertEqual((report["skipped"], report["draft"]["kind"]), ("draft-unsupported", "mtp"))
        self.assertIn("speculative types draft-mtp", report["note"])
        self.assertFalse(any("save" in path for _, path in self.servers["a"].calls))
        self.servers["b"].speculative = "none"
        self.assertTrue(self.switch()["reuse_ok"])          # without a draft it translates

    def test_deepseek4_target_takes_no_native_prefix(self):
        # the engine synthesizes a DeepSeek V4 cache whole and refuses a native prefix: registration
        # refuses it, and an entry from before (or prefix=True) switches without one
        dsv4 = write_gguf(Path(self.tmp.name) / "dsv4.gguf", "deepseek4", deepseek4__attention__indexer__top_k=8)

        def add(**options):
            return registry.add_model(registry.empty(), "d", url="http://127.0.0.1:9", gguf=dsv4,
                                      slot_dir=self.slots["b"], **options)["models"]["d"]
        with self.assertRaisesRegex(ValueError, "no native prefix into a deepseek4 cache"):
            add(native_prefix=True)
        self.assertEqual(add()["arch"], "deepseek4")
        manifest = json.loads((Path(self.config["hub"]) / "manifest.json").read_text())
        manifest["heads"] = {kind: {"base": "k", "models": {"b": {"layers": [2]}}} for kind in capabilities.DSV4_HEADS}
        (Path(self.config["hub"]) / "manifest.json").write_text(json.dumps(manifest))
        self.config["models"]["b"].update(arch="deepseek4", native_prefix=True)
        for options in ({}, {"prefix": True}):
            with self.subTest(**options):
                report = self.switch(**options)
                self.assertTrue(report["reuse_ok"])
                self.assertEqual(report["native_prefix_tokens"], 0)
                target = self.servers["b"].restored_payload["job"]["target"]
                self.assertEqual((target["prefix_state"], target["prefix_tokens"]), (None, 0))
                self.assertFalse((self.config_path.parent / "prefix").exists())
        self.assert_no_bridge_files()

    def test_unreadable_template_resume_is_dropped(self):
        self.servers["b"].patched = True
        self.switch()
        templates = self.config_path.parent / "templates"
        (cached,) = templates.glob("*.resume")
        cached.write_bytes(cached.read_bytes()[:-1])         # a save cut short
        with self.assertRaisesRegex(RuntimeError, "unreadable"):
            self.switch()
        self.assertEqual(list(templates.iterdir()), [])
        self.assertTrue(self.switch()["reuse_ok"])              # the next switch makes a new one
        self.assert_no_bridge_files()

    def test_template_cache_follows_the_speculative_setup_and_drops_a_stale_template(self):
        target = self.servers["b"]
        self.switch()
        target.speculative = "draft-dflash"            # the server restarted with a draft model
        self.switch()
        templates = self.config_path.parent / "templates"
        self.assertEqual(len(list(templates.glob("*.bin"))), 2)
        # a template that no longer matches the server (here: a patched server whose template has
        # no .resume) fails the restore, which drops it so the next switch makes a new one
        target.speculative = "none"
        target.patched = True
        with self.assertRaisesRegex(Exception, "Missing slot resume file"):
            self.switch()
        self.assertEqual(len(list(templates.glob("*.bin"))), 1)
        self.assertTrue(self.switch()["resume_sidecar"])
        self.assert_no_bridge_files()

    # -- media -----------------------------------------------------------------------------------

    def media_conversation(self, *, early=False):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        from kvpark.bridge.messages import conversation_from_openai
        messages = [{"role": "system", "content": "be brief"},
                    {"role": "user", "content": [{"type": "text", "text": "what is this"}, image] if early
                     else "tell me about parking caches"},
                    {"role": "assistant", "content": "a cache"},
                    {"role": "user", "content": [{"type": "text", "text": "and"}, {"type": "text", "text": "this?"},
                                                 image, {"type": "text", "text": "thanks"}]}]
        return conversation_from_openai(messages)

    def test_media_needs_the_modality_on_the_target(self):
        report = self.switch(self.media_conversation())
        self.assertEqual(report["skipped"], "target-no-vision")
        self.assertIn("mmproj", report["note"])

    @unittest.skipUnless(os.name == "posix", "POSIX permissions")
    def test_the_configuration_is_private(self):
        self.assertEqual(self.config_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.config_path.parent.stat().st_mode & 0o777, 0o700)

    def test_media_into_a_deepseek_v4_target_is_prefilled_natively(self):
        from kvpark.bridge import capabilities
        rule = capabilities.media({"vision": 3}, source="a", target="b", source_modalities={},
                                  target_modalities={"vision": True}, source_saw=2, target_arch="deepseek4")
        self.assertEqual(rule[0], "dsv4-media")
        self.assertIsNone(capabilities.media({"vision": 3}, source="a", target="b", source_modalities={},
                                             target_modalities={"vision": True}, source_saw=2,
                                             target_arch="qwen35"))

    def test_media_in_the_newest_message_is_translated_up_to_it(self):
        self.servers["b"].modalities = {"vision": True}
        self.servers["a"].modalities = {"vision": True}       # the source never saw the newest message
        report = self.switch(self.media_conversation())
        self.assertNotIn("skipped", report)
        job = self.servers["b"].restored_payload["job"]
        self.assertEqual(job["target"]["max_t0_message"], 3)
        # /tokenize reads the marker as text: its tokens are added to the native tail
        self.assertEqual(job["tail"], 5 + len("<__media_x__>"))
        # rendered as llama-server renders the request: text parts joined by newlines, none next to a marker
        self.assertEqual(job["conversation"]["messages"][3]["content"], "and\nthis?<__media_x__>thanks")
        templates = [body for path, body in self.servers["b"].bodies if path == "/apply-template"]
        self.assertIn("<__media_x__>", templates[0]["messages"][3]["content"])
        self.assertNotIn("warm", report)                      # never with media
        self.assertEqual(report["media"]["modalities"], ["vision"])
        # the rendering up to the media message's content (its stand-in)
        self.assertEqual(report["media"]["tokens_before"],
                         len("<system>be brief<user>tell me about parking caches<assistant>a cache<user>"))

    def test_media_the_source_holds_is_skipped(self):
        self.servers["b"].modalities = {"vision": True}
        self.servers["a"].modalities = {"vision": True}
        # an engine older than "media-source" cannot read a cache holding media
        report = self.switch(self.media_conversation(early=True))
        self.assertEqual((report["skipped"], report["engine_features"]), ("media-source", []))
        # a text-only source never processed the image itself: translated up to it (here: too little)
        self.servers["a"].modalities = {"vision": False}
        self.config["min_handover"] = 30
        report = self.switch(self.media_conversation(early=True))
        self.assertEqual(report["skipped"], "short")
        self.assertEqual(report["media"]["tokens_before"], len("<system>be brief<user>"))

    def test_media_the_source_holds_is_read_up_to_it_by_a_featured_engine(self):
        self.featured()
        self.servers["b"].modalities = {"vision": True}
        self.servers["a"].modalities = {"vision": True}
        report = self.switch(self.media_conversation(early=True))
        self.assertNotIn("skipped", report)
        self.assertTrue(report["media"]["source_media"])
        job = self.servers["b"].restored_payload["job"]
        # T0 stops before the first message with media; the engine reads the source up to its first medium
        self.assertEqual(job["target"]["max_t0_message"], 1)
        self.assertEqual(report["media"]["tokens_before"], len("<system>be brief<user>"))
        self.assert_no_bridge_files()

    def test_media_in_a_tool_result_after_tool_calls(self):
        # a screenshot tool: the server refuses to continue the assistant message with the tool
        # calls, so the media start is never rendered as a conversation ending there
        from kvpark.bridge.messages import conversation_from_openai
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        conversation = conversation_from_openai([
            {"role": "user", "content": "take a screenshot of the cache dashboard"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "screenshot", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": [image]}])
        self.servers["b"].modalities = {"vision": True}
        self.servers["a"].modalities = {"vision": True}       # the source has not seen the tool result
        report = self.switch(conversation)
        self.assertNotIn("skipped", report)
        self.assertEqual(report["media"]["tokens_before"],
                         len("<user>take a screenshot of the cache dashboard<assistant><tool>"))
        self.assertEqual(self.servers["b"].restored_payload["job"]["target"]["max_t0_message"], 2)
        # a template that refuses the stand-in: the text before the first media marker
        real = LlamaServer.apply_template

        def refuse_stand_in(server, messages, *args, **kwargs):
            if messages[-1]["content"] == bridge_switch.MEDIA_STAND_IN:
                raise ServerError("HTTP 400: template refuses")
            return real(server, messages, *args, **kwargs)
        with mock.patch.object(LlamaServer, "apply_template", refuse_stand_in):
            report = self.switch(conversation)
        self.assertEqual(report["media"]["tokens_before"],
                         len("<user>take a screenshot of the cache dashboard<assistant><tool>"))

    def test_raw_openai_parts_are_refused_until_converted(self):
        from kvpark.bridge import messages
        raw = {"messages": [{"role": "user", "content": [{"type": "text", "text": "what is this"},
                                                         {"type": "image_url", "image_url": {"url": "x"}}]}]}
        for helper in (messages.media, messages.media_count, messages.flatten):
            with self.assertRaisesRegex(ValueError, "conversation_from_openai"):
                helper(raw)
        # the CLI converts the file's OpenAI messages, so the image is seen (and b has no vision)
        import argparse
        parser = argparse.ArgumentParser()
        bridge_cli.add_parser(parser.add_subparsers(dest="command"))
        path = Path(self.tmp.name) / "conversation.json"
        path.write_text(json.dumps(raw))
        with mock.patch("builtins.print") as printed:
            bridge_cli.run(parser.parse_args(["bridge", "switch", "--from", "a", "--to", "b", "--conversation",
                                              str(path), "--config", str(self.config_path)]))
        self.assertEqual(json.loads(printed.call_args[0][0])["skipped"], "target-no-vision")

    def test_text_parts_follow_the_target_template(self):
        from kvpark.bridge.messages import conversation_from_openai
        conversation = conversation_from_openai([{"role": "user", "content": [{"type": "text", "text": "tell me "},
                                                                              {"type": "text", "text": "everything"}]}])
        self.servers["b"].text_join = "\n"
        report = self.switch(conversation)
        self.assertEqual(self.servers["b"].restored_payload["job"]["conversation"]["messages"][0]["content"],
                         "tell me \neverything")
        self.assertTrue(report["reuse_ok"])
        self.servers["b"].text_join = " | "       # a separator the probe finds, and plain text reproduces
        self.assertTrue(self.switch(conversation)["reuse_ok"])
        with mock.patch.object(bridge_switch, "_text_join", return_value=None):
            self.assertEqual(self.switch(conversation)["skipped"], "text-parts")


class MessagesAndStageTest(unittest.TestCase):
    def test_openai_messages_become_engine_conversations(self):
        from kvpark.bridge.messages import conversation_from_openai, flatten, media
        messages = [
            {"role": "system", "content": "be brief\n"},
            {"role": "user", "content": [{"type": "text", "text": "find "}, {"type": "text", "text": "caches"}]},
            {"role": "assistant", "content": None, "reasoning_content": "search first",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "search", "arguments": "{\"q\": \"kv\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result", "extra": 1},
            {"role": "user", "content": "thanks"},
        ]
        conversation = conversation_from_openai(messages, tools=[{"type": "function"}])
        out = conversation["messages"]
        self.assertEqual(out[0], {"role": "system", "content": "be brief\n"})
        self.assertEqual(out[1]["content"], [{"type": "text", "text": "find "}, {"type": "text", "text": "caches"}])
        self.assertEqual(flatten(conversation)["messages"][1]["content"], "find caches")
        self.assertEqual(out[2]["tool_calls"][0]["function"], {"name": "search", "arguments": {"q": "kv"}})
        self.assertEqual(out[2]["reasoning_content"], "search first")
        self.assertEqual(out[2]["content"], "")
        self.assertEqual(out[3], {"role": "tool", "content": "result", "tool_call_id": "c1", "name": "search"})
        self.assertEqual(conversation["tools"], [{"type": "function"}])
        self.assertEqual(media(conversation), {})
        for bad in ([{"role": "user", "content": [{"type": "file"}]}],
                    [{"role": "assistant", "content": "", "tool_calls": [{"id": "x", "function": {"name": "f",
                                                                                                    "arguments": "{"}}]}],
                    [{"role": "tool", "content": "r"}], [{"role": "critic", "content": "?"}], []):
            with self.assertRaises(ValueError):
                conversation_from_openai(bad)

    def test_media_parts_become_placeholders_and_markers(self):
        from kvpark.bridge.messages import conversation_from_openai, flatten, media
        conversation = conversation_from_openai([
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
                                         {"type": "text", "text": "transcribe"}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "x"}},
                                         {"type": "image_url", "image_url": {"url": "y"}}, {"type": "text", "text": "b"},
                                         {"type": "text", "text": "c"}]}])
        self.assertEqual(conversation["messages"][0]["content"][0], {"type": "media", "modality": "audio"})
        self.assertNotIn("AAAA", json.dumps(conversation))          # no media bytes are kept
        self.assertEqual(media(conversation), {"audio": 0, "vision": 2})
        flat = flatten(conversation, marker="<M>")
        self.assertEqual(flat["messages"][0]["content"], "<M>transcribe")
        self.assertEqual(flat["messages"][2]["content"], "a<M><M>b\nc")
        with self.assertRaisesRegex(ValueError, "media marker"):
            flatten(conversation)

    def test_staging_prewarms_the_engine_in_the_background(self):
        from kvpark.bridge import stage
        config = {"engine": {"python": "p", "script": "s"}, "hub": "h", "models": {}}
        started = threading.Event()
        with mock.patch.object(engine_client, "prewarm", side_effect=lambda *a, **k: started.set()) as prewarm:
            thread = stage.prewarm(config, "a", "b", "/bridge")
            thread.join(10)
        self.assertTrue(started.is_set() and thread.daemon)
        self.assertEqual(prewarm.call_args[0], (config, ["a", "b"], "/bridge"))
        self.assertIsNone(stage.prewarm({"engine": None, "hub": None}, "a", "b", "/bridge"))     # no engine yet

    def test_stage_lease_and_drop(self):
        from kvpark.bridge import stage
        with tempfile.TemporaryDirectory() as directory:
            path = stage.put(directory, "k-1", source="a", target="b", now=1000)
            if os.name == "posix":
                self.assertEqual(oct(path.stat().st_mode & 0o777), "0o600")
            self.assertEqual(stage.get(directory, "k-1", now=1000 + 60)["target"], "b")
            self.assertIsNone(stage.get(directory, "k-2", now=1000))
            self.assertIsNone(stage.get(directory, "k-1", now=1000 + stage.LEASE_SECONDS + 1))
            self.assertFalse(path.exists())                  # an expired stage is removed
            stage.put(directory, "k-1", source="a", target="b")
            stage.drop(directory, "k-1")
            self.assertIsNone(stage.get(directory, "k-1"))


if __name__ == "__main__":
    unittest.main()
