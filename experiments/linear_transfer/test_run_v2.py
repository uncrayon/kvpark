"""run_v2.py: unit checks, and every stage end to end on tiny fixtures (KVPARK_KV_PROBE=/path/binary)."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

import run_v2


PROBE = os.environ.get("KVPARK_KV_PROBE")

QWEN_CONTROL = ["<|im_start|>", "<|im_end|>"]
QWEN_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message.role == 'system' %}<|im_start|>system\n{{ message.content }}"
    "{% if tools %}\n\n# Tools{% for tool in tools %}\n{{ tool | tojson }}{% endfor %}{% endif %}<|im_end|>\n"
    "{% elif message.role == 'user' %}<|im_start|>user\n{{ message.content }}<|im_end|>\n"
    "{% elif message.role == 'assistant' %}<|im_start|>assistant\n"
    "{% if message.reasoning_content %}<think>\n{{ message.reasoning_content | trim }}\n</think>\n\n{% endif %}"
    "{{ message.content | trim }}"
    "{% if message.tool_calls %}{% for call in message.tool_calls %}\n<tool_call>\n"
    "{\"name\": \"{{ call.function.name }}\", \"arguments\": {{ call.function.arguments | tojson }}}\n"
    "</tool_call>{% endfor %}{% endif %}<|im_end|>\n"
    "{% elif message.role == 'tool' %}<|im_start|>user\n<tool_response>\n{{ message.content }}\n"
    "</tool_response><|im_end|>\n"
    "{% endif %}{% endfor %}")
MUSE_BOS = "<|begin_of_text|>"
MUSE_CONTROL = ["<|start|>", "<|message|>", "<|eot|>", "<|channel|>", "<|end|>", "<|call|>"]
MUSE_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if message.role == 'assistant' %}<|start|>assistant"
    "{% if message.reasoning_content %}<|channel|>analysis<|message|>{{ message.reasoning_content }}"
    "<|end|><|start|>assistant{% endif %}"
    "{% if message.tool_calls %}{% for call in message.tool_calls %}<|channel|>commentary to="
    "{{ call.function.name }}<|message|>{% for name, value in call.function.arguments.items() %}"
    "{{ name }}={{ value }};{% endfor %}<|call|>{% endfor %}"
    "{% else %}<|message|>{{ message.content }}<|eot|>{% endif %}"
    "{% elif message.role == 'tool' %}<|start|>tool:{{ message.name }}<|message|>{{ message.content }}<|eot|>"
    "{% else %}<|start|>{{ message.role }}<|message|>{{ message.content }}"
    "{% if loop.first and tools %}\nTools:{% for tool in tools %} {{ tool.function.name }}{% endfor %}{% endif %}"
    "<|eot|>{% endif %}\n{% endfor %}")
WORDS = ["alpha", "river", "café", "naïve", "日本語", "données", "Zürich",
         "λόγος", "stone", "light", "garden", "мир", "quiet", "north"]
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather forecast for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]
SPLITS = ["train"] * 8 + ["dev"] * 3 + ["test"] * 3
SUBSETS = ["chat", "reasoning", "tools"]


def sentence(rng, count):
    return " ".join(rng.choice(WORDS, count)).capitalize() + "."


def conversation(number):
    rng = np.random.default_rng(number)
    subset = SUBSETS[number % 3]
    record = {"id": f"{subset}:{number}", "source": subset, "split": SPLITS[number], "tools": None,
              "messages": [{"role": "system", "content": f"You are assistant {number}. {sentence(rng, 4)}"},
                           {"role": "user", "content": sentence(rng, 6 + number % 4)}]}
    final = " ".join(sentence(rng, 7) for _ in range(5))
    if subset == "chat":
        record["messages"] += [{"role": "assistant", "content": sentence(rng, 8)},
                               {"role": "user", "content": sentence(rng, 5)},
                               {"role": "assistant", "content": final}]
    elif subset == "reasoning":
        record["messages"].append({"role": "assistant", "reasoning_content": sentence(rng, 12), "content": final})
    else:
        record["tools"] = TOOLS
        call = {"id": f"call_{number}", "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": str(rng.choice(WORDS)), "days": 3}}}
        record["messages"] += [
            {"role": "assistant", "content": sentence(rng, 3) if number % 2 else "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "name": "get_weather", "content": sentence(rng, 6)},
            {"role": "assistant", "content": final}]
    return record


def run_quiet(argv):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        run_v2.main(argv)
    return [json.loads(line) for line in output.getvalue().splitlines()]


class UnitTests(unittest.TestCase):
    def test_auto_eval_layers_reproduce_the_real_run_spec(self):
        muse = ["sliding" if layer % 4 < 3 else "full" for layer in range(52)]
        evaluated, attn_out = run_v2.auto_eval_layers(muse)
        self.assertEqual(evaluated, [0, 3, 7, 11, 15, 19, 23, 26, 27, 31, 35, 39, 43, 47, 50, 51])
        self.assertEqual(attn_out, [26, 27])
        self.assertEqual(run_v2.middle_full_layer(muse), 27)
        self.assertEqual(run_v2.auto_eval_layers(["sliding"] * 3 + ["full"]), ([0, 2, 3], [2, 3]))
        with self.assertRaises(ValueError):
            run_v2.auto_eval_layers(["full"] * 4)

    def test_inject_selection_is_round_robin_by_patched_prefix_and_deterministic(self):
        class Plan:
            def __init__(self, document_id, n_tokens, patched):
                self.document_id, self.n_tokens = document_id, n_tokens
                self.positions = np.arange(patched)
                self.continuation = np.arange(n_tokens - 20, n_tokens)
        sizes = [(700, 10), (800, 50), (500, 99), (3000, 20), (900, 5), (4000, 7), (1200, 30), (1000, 5)]
        plans = [Plan(f"{i:064x}", n, patched) for i, (n, patched) in enumerate(sizes)]
        records = {plan.document_id: {"source": "ab"[i % 2] if i < 4 else "c"} for i, plan in enumerate(plans)}
        chosen = run_v2.select_inject_documents(plans, records, 4, 600, 3500)
        self.assertEqual([p.document_id[-1] for p in chosen], ["0", "1", "6", "3"])
        self.assertEqual(chosen, run_v2.select_inject_documents(list(reversed(plans)), records, 4, 600, 3500))
        everything = run_v2.select_inject_documents(plans, records, 50, 600, 3500)
        self.assertEqual([p.document_id[-1] for p in everything], ["0", "1", "6", "3", "4", "7"])

    def test_decoded_tokens_counts_whole_ubatches(self):
        class Plan:
            n_tokens = 1000
            continuation = np.arange(600, 1000)
        self.assertEqual(run_v2.decoded_tokens(Plan, 256, 512, 300), 1000)
        self.assertEqual(run_v2.decoded_tokens(Plan, 10, 512, 300), 1000)
        self.assertEqual(run_v2.decoded_tokens(Plan, 10, 128, 300), 640)
        self.assertEqual(run_v2.decoded_tokens(Plan, 10, 128, 700), 768)

    def test_fingerprints_hash_small_files_and_stat_large_ones(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "d").mkdir()
            (root / "d" / "small").write_bytes(b"abc")
            (root / "d" / "large").write_bytes(b"x" * 64)
            with mock.patch.object(run_v2, "HASH_LIMIT", 16):
                before = run_v2.fingerprint(root, ["d"])
                self.assertIn("sha256", before["d/small"])
                self.assertEqual(set(before["d/large"]), {"size", "mtime_ns"})
                (root / "d" / "small").write_bytes(b"abd")
                self.assertNotEqual(run_v2.fingerprint(root, ["d"])["d/small"], before["d/small"])
                os.utime(root / "d" / "large", ns=(1, 1))
                self.assertNotEqual(run_v2.fingerprint(root, ["d"])["d/large"], before["d/large"])
            with self.assertRaises(ValueError):
                run_v2.fingerprint(root, ["missing"])

    def test_attention_skill_is_relative_to_the_floor(self):
        self.assertAlmostEqual(run_v2.skill(0.99, 0.96), 0.75)
        self.assertEqual(run_v2.skill(0.96, 0.96), 0.0)
        self.assertIsNone(run_v2.skill(0.99, 1.0))
        self.assertIsNone(run_v2.skill(None, 0.5))

    def test_positive_control(self):
        def report(kl, top1):
            conditions = [{"name": name, "kl": {"max_token": value}} for name, value in
                          (("native", 0.0), ("sanity_native_all", 0.0), ("all_kv", kl), ("mean_all", kl))]
            return {"conditions": conditions, "per_document": [
                {"conditions": {"mean_all": {"kl_mean": kl, "top1_agree": top1}}}]}
        self.assertEqual(run_v2.positive_control(report(0.2, 0.5))["documents_top1_below_1"], 1)
        self.assertFalse(run_v2.positive_control(report(0.2, 1.0))["top1_changed"])
        with self.assertRaisesRegex(ValueError, "positive control failed"):
            run_v2.positive_control(report(0.0, 1.0))

    def test_failure_stops_the_whole_child_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "child.py"
            script.write_text("import subprocess, sys, time\n"
                              "child = subprocess.Popen(['sleep', '60'])\n"
                              "print(child.pid, flush=True)\n"
                              "time.sleep(60)\n")
            args = run_v2.build_parser().parse_args(
                ["--output", str(root), "--probe", __file__, "--corpus", __file__, "--source-model", __file__,
                 "--source-model-id", "s", "--target-model", __file__, "--target-model-id", "t"])
            run = run_v2.Run(args)
            seen = []

            def on_line(line):
                seen.append(int(line))
                raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                run.command("test", [script], "test.log", threads=1, on_line=on_line)
            for _ in range(100):
                try:
                    os.kill(seen[0], 0)
                except ProcessLookupError:
                    break
                state = Path(f"/proc/{seen[0]}/stat").read_text().split(")")[-1].split()[0]
                if state == "Z":
                    break
                time.sleep(0.05)
            else:
                self.fail("the grandchild survived")

    def test_ignored_hangup_stays_ignored(self):
        previous = run_v2.signal.signal(run_v2.signal.SIGHUP, run_v2.signal.SIG_IGN)
        try:
            with run_v2._TerminationSignals():
                self.assertIs(run_v2.signal.getsignal(run_v2.signal.SIGHUP), run_v2.signal.SIG_IGN)
                self.assertIs(run_v2.signal.getsignal(run_v2.signal.SIGTERM), run_v2._TerminationSignals._raise)
        finally:
            run_v2.signal.signal(run_v2.signal.SIGHUP, previous)

    def test_argument_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = ["--output", str(Path(temporary) / "run"), "--probe", __file__, "--corpus", __file__,
                    "--source-model", __file__, "--source-model-id", "s", "--target-model", __file__,
                    "--target-model-id", "t"]
            for extra, fragment in ((["--adopt"], "--adopt requires --resume"),
                                    (["--from", "fit", "--until", "align"], "--from"),
                                    (["--batch", "9000", "--ctx", "5120"], "batch <= ctx"),
                                    (["--resume"], "not an existing run directory")):
                with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, fragment):
                    run_quiet(base + extra)


@unittest.skipUnless(PROBE, "set KVPARK_KV_PROBE to run the pipeline on tiny fixtures")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tiny_fixture import create_fixture
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-run-v2-")
        root = cls.root = Path(cls.temp.name)
        create_fixture(root / "qwen.gguf", "qwen35", control_tokens=QWEN_CONTROL, chat_template=QWEN_TEMPLATE)
        create_fixture(root / "muse.gguf", "muse-glimmer", control_tokens=MUSE_CONTROL, chat_template=MUSE_TEMPLATE,
                       bos_token=MUSE_BOS, add_bos=True, sliding_window=16)
        cls.corpus = root / "conversations.jsonl"
        cls.corpus.write_text("".join(json.dumps(conversation(n), ensure_ascii=False) + "\n"
                                      for n in range(len(SPLITS))), encoding="utf-8")
        cls.run_dir = root / "run"
        cls.events = run_quiet(cls.argv(cls.run_dir))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def argv(cls, output, *extra):
        return ["--output", str(output), "--probe", PROBE, "--corpus", str(cls.corpus),
                "--source-model", str(cls.root / "qwen.gguf"), "--source-model-id", "tiny/qwen",
                "--target-model", str(cls.root / "muse.gguf"), "--target-model-id", "tiny/muse",
                "--ctx", "1024", "--batch", "64", "--threads", "2", "--fit-threads", "2",
                "--inject-docs", "3", "--inject-min-tokens", "50", "--inject-max-tokens", "1024",
                "--inject-docs-per-run", "2", "--max-scored", "16", "--inject-tail", "32",
                "--skip-memory-check", *extra]

    def copy_run(self, name):
        destination = self.root / name
        shutil.copytree(self.run_dir, destination, symlinks=True)
        return destination

    def record(self, run_dir, stage):
        return json.loads((run_dir / "stages" / f"{stage}.json").read_text(encoding="utf-8"))

    def snapshot(self, run_dir):
        return {str(path.relative_to(run_dir)): (path.stat().st_mtime_ns, path.read_bytes())
                for path in sorted(run_dir.rglob("*")) if path.is_file()}

    def test_every_stage_completes_with_the_expected_layout(self):
        done = [event["stage"] for event in self.events if event["event"] == "stage_done"]
        self.assertEqual(done, list(run_v2.STAGES))
        self.assertEqual(self.events[-1]["event"], "run_done")
        for rel in ("rendered-source.jsonl", "rendered-target.jsonl", "rendered-target-train.jsonl",
                    "rendered-target-eval.jsonl", "templates/qwen.jinja", "templates/muse.jinja", "spec-kv.json",
                    "spec-target-eval.json", "splits.json", "model-hashes.json", "smoke/source", "smoke/target",
                    "captures/source", "captures/target-train", "captures/target-eval", "logs/collect-source.log",
                    "logs/collect-target-eval.log", "alignment/alignment.npz", "alignment/coverage.json",
                    "mapper/manifest.json", "attention/attention-report.json", "injection/injection-report.json",
                    "summary.json", "summary.md"):
            self.assertTrue((self.run_dir / rel).exists(), rel)
        self.assertEqual([p.name for p in self.run_dir.iterdir() if p.name.startswith(".")], [])
        self.assertEqual(json.loads((self.run_dir / "spec-target-eval.json").read_text()),
                         {"k": "all", "v": "all", "q": [0, 2, 3], "gate": [0, 2, 3], "attn_out": [2, 3]})
        records = [json.loads(line) for line in (self.run_dir / "rendered-target.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["text"].startswith(MUSE_BOS) for r in records))
        self.assertEqual({r["model"] for r in records}, {"muse"})
        eta = [event for event in self.events if event["event"] == "eta"]
        self.assertEqual({event["stage"] for event in eta} >= {"collect-source", "collect-target", "inject"}, True)
        self.assertTrue(any(event["event"] == "progress" and event["stage"] == "inject" for event in self.events))

    def test_attention_is_validated_on_both_attention_types(self):
        smoke = self.record(self.run_dir, "smoke")["metrics"]["attn_out_relative_error"]
        self.assertEqual(sorted(smoke), ["2", "3"])
        self.assertTrue(all(value < 1e-4 for value in smoke.values()))
        report = json.loads((self.run_dir / "attention/attention-report.json").read_text())
        self.assertEqual(report["unvalidated_attention_types"], [])
        self.assertEqual(sorted(item["attention"] for item in report["validation"]), ["full", "sliding"])
        self.assertTrue(all(item["max_relative_error"] < 1e-4 for item in report["validation"]))

    def test_injection_sanity_is_exact_and_decoding_stops_early(self):
        summary = json.loads((self.run_dir / "summary.json").read_text())
        injection = summary["injection"]
        self.assertEqual(injection["documents"], 3)
        self.assertEqual(injection["sanity"]["max_token_kl"], 0.0)
        self.assertTrue(injection["sanity"]["passed"])
        conditions = {item["name"]: item for item in injection["conditions"]}
        self.assertEqual(conditions["sanity_native_all"]["nll_delta"]["mean"], 0.0)
        self.assertIn("L3_kv", conditions)
        self.assertGreater(conditions["all_kv"]["kl"]["mean"], 0)
        self.assertTrue(injection["positive_control"]["passed"])
        contrasts = {(item["name"], item["baseline"]) for item in injection["contrasts"]}
        self.assertIn(("all_kv", "mean_all"), contrasts)
        self.assertEqual(set(injection["groups"]), {"long_prefix", "short_prefix"})
        self.assertEqual(sum(group["documents"] for group in injection["groups"].values()), 3)
        tail = conditions["all_kv_tail32"]
        self.assertLess(tail["mapped_rows"], conditions["all_kv"]["mapped_rows"])
        self.assertEqual(conditions["sanity_native_all"]["mapped_rows"], 0)
        metrics = self.record(self.run_dir, "inject")["metrics"]
        self.assertEqual(metrics["decoded_tokens_this_invocation"], metrics["decoded_tokens"])
        self.assertEqual(metrics["decoded_tokens"], metrics["decoded_tokens_planned"])
        self.assertLess(metrics["decoded_tokens"], metrics["full_decode_tokens"])
        documents = (self.run_dir / "stages/inject-documents.txt").read_text().split()
        self.assertEqual(len(documents), 3)
        sources = {json.loads(line)["document_id"]: json.loads(line)["source"]
                   for line in (self.run_dir / "rendered-target.jsonl").read_text().splitlines()}
        self.assertEqual(len({sources[d] for d in documents}), 3)

    def test_summary_has_every_section(self):
        summary = json.loads((self.run_dir / "summary.json").read_text())
        for key in ("models", "corpus", "coverage", "fit", "attention", "injection", "timings", "caveats"):
            self.assertIn(key, summary)
        self.assertEqual(set(summary["coverage"]["by_split"]), {"train", "dev", "test"})
        self.assertEqual(set(summary["coverage"]["by_subset"]), set(SUBSETS))
        self.assertTrue(all(0 < item["aligned_fraction_of_content"] <= 1
                            for item in summary["coverage"]["by_split"].values()))
        self.assertEqual(summary["fit"]["ks"], [1])
        self.assertEqual(sorted({(m["kind"], m["target"]) for m in summary["fit"]["maps"]}),
                         [(kind, layer) for kind in "kv" for layer in range(4)])
        self.assertTrue(all(m["dev_r2_by_k"] and m["test_r2"] is not None for m in summary["fit"]["maps"]))
        self.assertEqual(summary["attention"]["headline_split"], "test")
        self.assertTrue(any(item["split"] == "test" and item["query_set"] == "continuation"
                            for item in summary["attention"]["layers"]))
        self.assertEqual(summary["attention"]["floor_variant"], "mean_kv")
        scored = [item for item in summary["attention"]["layers"]
                  if item["query_set"] == "scored" and item["variant"] == "kv"]
        self.assertTrue(scored and all("skill_contribution_cosine_mean" in item for item in scored))
        self.assertEqual([item["stage"] for item in summary["timings"]], list(run_v2.STAGES[:-1]))
        markdown = (self.run_dir / "summary.md").read_text()
        for heading in ("## Coverage", "## Fit", "## Attention output", "## Prefix injection", "## Timings",
                        "## Caveats"):
            self.assertIn(heading, markdown)

    def test_resume_is_a_no_op(self):
        run_dir = self.copy_run("resumed")
        before = self.snapshot(run_dir)
        events = run_quiet(self.argv(run_dir, "--resume"))
        self.assertEqual([event["stage"] for event in events if event["event"] == "stage_skip"], list(run_v2.STAGES))
        self.assertFalse(any(event["event"] in ("stage_start", "warning") for event in events))
        self.assertEqual(self.snapshot(run_dir), before)
        with self.assertRaisesRegex(ValueError, "already exists; pass --resume"):
            run_quiet(self.argv(run_dir))
        with self.assertRaisesRegex(ValueError, "different settings"):
            run_quiet(self.argv(run_dir, "--resume", "--max-scored", "8"))

    def test_corrupted_stage_outputs_are_detected_on_resume(self):
        for rel, stage in (("captures/target-eval/v-2.npy", "collect-target"), ("mapper/manifest.json", "fit"),
                           ("injection/injection-report.json", "inject")):
            with self.subTest(rel=rel):
                run_dir = self.copy_run(f"corrupt-{stage}")
                path = run_dir / rel
                data = bytearray(path.read_bytes())
                data[-2] ^= 1
                path.write_bytes(bytes(data))
                with self.assertRaisesRegex(ValueError, f"stage {stage} outputs changed.*{rel}"):
                    run_quiet(self.argv(run_dir, "--resume"))
        run_dir = self.copy_run("corrupt-record")
        (run_dir / "stages/align.json").write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "align.json is corrupt"):
            run_quiet(self.argv(run_dir, "--resume"))
        run_dir = self.copy_run("missing-output")
        shutil.rmtree(run_dir / "attention")
        with self.assertRaisesRegex(ValueError, "stage attention output is missing"):
            run_quiet(self.argv(run_dir, "--resume"))

    def test_output_of_an_interrupted_attempt_is_reused(self):
        run_dir = self.copy_run("interrupted")
        for rel in ("attention", "summary.json", "summary.md", "stages/attention.json", "stages/report.json"):
            path = run_dir / rel
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        original = run_v2.Run.write_record

        def crash(run, stage, *args, **kwargs):
            if stage == "attention":
                raise TypeError("simulated crash after publishing")
            return original(run, stage, *args, **kwargs)

        output = io.StringIO()
        with mock.patch.object(run_v2.Run, "write_record", crash), contextlib.redirect_stdout(output), \
                self.assertRaises(TypeError):
            run_v2.main(self.argv(run_dir, "--resume"))
        error = json.loads(output.getvalue().splitlines()[-1])
        self.assertEqual((error["event"], error["stage"], error["type"]), ("error", "attention", "TypeError"))
        report = (run_dir / "attention/attention-report.json").read_bytes()
        marker = run_dir / "stages/attention.started.json"
        self.assertTrue(marker.is_file())
        stale = marker.read_text(encoding="utf-8")
        marker.write_text(stale.replace('"dev"', '"train"', 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "was not published by an attempt with these settings"):
            run_quiet(self.argv(run_dir, "--resume"))
        marker.write_text(stale, encoding="utf-8")
        events = run_quiet(self.argv(run_dir, "--resume"))
        self.assertEqual([e["stage"] for e in events if e["event"] == "stage_reuse"], ["attention"])
        self.assertEqual([e["stage"] for e in events if e["event"] == "stage_done"], ["attention", "report"])
        self.assertEqual((run_dir / "attention/attention-report.json").read_bytes(), report)
        self.assertFalse(marker.exists())

    def test_adopt_records_existing_collection_and_continues(self):
        run_dir = self.copy_run("adopted")
        for rel in ("alignment", "mapper", "attention", "injection", "stages"):
            shutil.rmtree(run_dir / rel)
        for rel in ("summary.json", "summary.md", "splits.json"):
            (run_dir / rel).unlink()
        (run_dir / "logs/collection-steps.jsonl").write_text("".join(json.dumps(item) + "\n" for item in [
            {"step": "source", "event": "start", "time": "2026-09-28T02:38:01-06:00"},
            {"step": "source", "event": "done", "time": "2026-09-28T02:48:01-06:00"}]))
        before = {rel: (run_dir / rel).read_bytes() for rel in ("rendered-source.jsonl", "spec-target-eval.json")}
        with self.assertRaisesRegex(ValueError, "--from align: stage prepare has no record.*--adopt"):
            run_quiet(self.argv(run_dir, "--resume", "--from", "align"))
        with self.assertRaisesRegex(ValueError, "exists but the stage has no record"):
            run_quiet(self.argv(run_dir, "--resume"))
        events = run_quiet(self.argv(run_dir, "--resume", "--adopt", "--from", "align", "--until", "fit"))
        done = {event["stage"]: event for event in events if event["event"] == "stage_done"}
        self.assertEqual(list(done), ["prepare", "smoke", "collect-source", "collect-target", "align", "fit"])
        for stage in run_v2.ADOPTABLE:
            self.assertTrue(self.record(run_dir, stage)["adopted"], stage)
        self.assertFalse(self.record(run_dir, "align")["adopted"])
        collect = self.record(run_dir, "collect-source")
        self.assertEqual(collect["wall_seconds"], 600.0)
        self.assertEqual(collect["metrics"]["parts"]["source"]["seconds"], 600.0)
        self.assertEqual(self.record(run_dir, "prepare")["metrics"]["written"], ["splits.json"])
        self.assertEqual({rel: (run_dir / rel).read_bytes() for rel in before}, before)
        for rel in ("alignment/alignment.npz", "mapper/W-v-3.npy", "mapper/W-k-0.npy"):
            self.assertEqual((run_dir / rel).read_bytes(), (self.run_dir / rel).read_bytes(), rel)
        events = run_quiet(self.argv(run_dir, "--resume", "--until", "fit"))
        self.assertEqual([e["stage"] for e in events if e["event"] == "stage_skip"], list(run_v2.STAGES[:6]))

    def test_adopt_rejects_outputs_that_differ(self):
        run_dir = self.copy_run("adopt-bad")
        shutil.rmtree(run_dir / "stages")
        path = run_dir / "rendered-target-eval.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("assistant", "assistent", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "rendered-target-eval.jsonl differs"):
            run_quiet(self.argv(run_dir, "--resume", "--adopt", "--from", "align", "--until", "align"))
        run_dir = self.copy_run("adopt-missing")
        shutil.rmtree(run_dir / "stages")
        shutil.rmtree(run_dir / "captures/target-eval")
        with self.assertRaisesRegex(ValueError, "would have to run collections \\['target-eval'\\]"):
            run_quiet(self.argv(run_dir, "--resume", "--adopt", "--from", "align", "--until", "align"))


if __name__ == "__main__":
    unittest.main()
