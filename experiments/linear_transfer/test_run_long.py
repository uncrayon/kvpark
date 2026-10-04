"""run_long.py: unit checks, and every stage end to end on tiny fixtures (KVPARK_KV_PROBE=/path/binary;
KVPARK_KV_PROBE_VULKAN=/path/gpu-binary adds the same run on the GPU)."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

import chat
import handoff
import kvstate
import run_long
import run_v2
from test_run_v2 import MUSE_BOS, MUSE_CONTROL, MUSE_TEMPLATE, QWEN_CONTROL, QWEN_TEMPLATE, SPLITS, WORDS, conversation


PROBE = os.environ.get("KVPARK_KV_PROBE")
VULKAN_PROBE = os.environ.get("KVPARK_KV_PROBE_VULKAN")
BATCH = 16


def sentence(rng, count):
    return " ".join(rng.choice(WORDS, count)).capitalize() + "."


def long_conversation(number):
    """A multi-turn test conversation whose final assistant message carries reasoning."""
    rng = np.random.default_rng(1000 + number)
    messages = [{"role": "system", "content": f"You are long assistant {number}. {sentence(rng, 4)}"}]
    for _ in range(2 + number % 4):
        messages += [{"role": "user", "content": sentence(rng, 6)}, {"role": "assistant", "content": sentence(rng, 7)}]
    messages += [{"role": "user", "content": sentence(rng, 8)},
                 {"role": "assistant", "reasoning_content": sentence(rng, 10), "content": sentence(rng, 9)}]
    return {"id": f"long:{number}", "source": "chat" if number % 2 else "reading", "split": "test", "tools": None,
            "messages": messages}


def train_conversation(number):
    record = conversation(number % 8)
    record = json.loads(json.dumps(record))
    record["id"] = f"extra:{number}"
    record["split"] = "train"
    record["messages"][0]["content"] = f"You are extra helper {number}. " + record["messages"][0]["content"]
    return record


def run_quiet(main, argv):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main(argv)
    return [json.loads(line) for line in output.getvalue().splitlines()]


class Workspace:
    """Tiny fixtures, a v2 run through fit, a long corpus with a manifest and an extra train corpus (once)."""

    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        from tiny_fixture import create_fixture
        self.temp = tempfile.TemporaryDirectory(prefix="kvpark-run-long-")
        root = self.root = Path(self.temp.name)
        self.qwen, self.muse = root / "qwen.gguf", root / "muse.gguf"
        create_fixture(self.qwen, "qwen35", control_tokens=QWEN_CONTROL, chat_template=QWEN_TEMPLATE)
        create_fixture(self.muse, "muse-glimmer", control_tokens=MUSE_CONTROL, chat_template=MUSE_TEMPLATE,
                       bos_token=MUSE_BOS, add_bos=True, sliding_window=16)
        corpus = root / "v2-corpus.jsonl"
        corpus.write_text("".join(json.dumps(conversation(n), ensure_ascii=False) + "\n" for n in range(len(SPLITS))),
                          encoding="utf-8")
        self.v2 = root / "v2"
        run_quiet(run_v2.main, ["--output", str(self.v2), "--probe", PROBE, "--corpus", str(corpus),
                                "--source-model", str(self.qwen), "--source-model-id", "tiny/qwen",
                                "--target-model", str(self.muse), "--target-model-id", "tiny/muse",
                                "--ctx", "1024", "--batch", "64", "--threads", "2", "--fit-threads", "2",
                                "--fit-ks", "1", "--skip-memory-check", "--until", "fit"])
        conversations = [long_conversation(n) for n in range(6)]
        self.long_corpus = root / "long" / "conversations.jsonl"
        self.long_corpus.parent.mkdir()
        self.long_corpus.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in conversations),
                                    encoding="utf-8")
        documents = [{"id": c["id"], "document_id": chat.document_id(c), "source": c["source"],
                      "bucket": [0, 400] if n < 3 else [400, 1000], "stratum": None,
                      "role": "reserve" if n == 5 else "primary"} for n, c in enumerate(conversations)]
        (self.long_corpus.parent / "manifest.json").write_text(json.dumps({"long_eval": {"documents": documents}}),
                                                               encoding="utf-8")
        self.train_corpus = root / "train.jsonl"
        self.train_corpus.write_text("".join(json.dumps(train_conversation(n), ensure_ascii=False) + "\n"
                                             for n in range(5)), encoding="utf-8")

    def argv(self, output, probe, *extra, device="cpu"):
        return ["--output", str(output), "--probe", probe, "--device", device, "--corpus", str(self.long_corpus),
                "--v2-run", str(self.v2), "--source-model", str(self.qwen), "--source-model-id", "tiny/qwen",
                "--target-model", str(self.muse), "--target-model-id", "tiny/muse", "--tails", "0,48,96",
                "--ctx", "1024", "--batch", str(BATCH), "--threads", "2", "--build-threads", "2",
                "--docs-per-run", "2", "--max-scored", "16", "--min-handoff-tokens", "32", "--chunk-rows", "16",
                "--train-ctx", "1024", "--train-batch", "64", "--fit-ks", "1", "--bootstrap", "200",
                "--skip-memory-check", *extra]


def tearDownModule():
    if Workspace._instance is not None:
        Workspace._instance.temp.cleanup()
        Workspace._instance = None


class UnitTests(unittest.TestCase):
    def test_t0_rounds_down_to_the_batch_grid_and_skips_small_handoffs(self):
        self.assertEqual(run_long.plan_t0(4287, 2048, 512, True, 512), 2048)
        self.assertIsNone(run_long.plan_t0(4287, 4096, 512, True, 512))
        self.assertEqual(run_long.plan_t0(4909, 4096, 512, True, 512), 512)
        self.assertEqual(run_long.plan_t0(4909, 4096, 512, False, 512), 813)
        self.assertEqual(run_long.plan_t0(4909, 0, 512, True, 512), 4908)
        conditions = run_long.document_conditions(4909, list(run_long.CONDITION_KINDS), [0, 2048], ["v2", "v3"],
                                                  512, True, 512)
        names = [c["name"] for c in conditions]
        self.assertEqual(names[:3], ["table_fill_0", "table_fill_0_v3", "mean_old_0"])
        self.assertIn("oracle_fill_2048_v3", names)
        self.assertNotIn("mean_old_2048_v3", names)
        self.assertEqual({c["t0"] for c in conditions if c["tail"] == 2048}, {2560})
        self.assertTrue(all(c["native_state"] == (c["kind"] in ("sanity", "oracle", "oracle_fill", "mean_oracle"))
                            for c in conditions))

    def test_contrasts_pair_conditions_with_floors_upper_bounds_families_and_tails(self):
        conditions = {c["name"]: c for c in run_long.document_conditions(
            5000, list(run_long.CONDITION_KINDS), [0, 2048, 4096], ["v2", "v3"], 512, True, 512)}
        pairs = run_long.report_contrasts(conditions, ["v2", "v3"])
        for pair in (("table_fill_2048", "mean_old_2048"), ("table_fill_2048_v3", "mean_old_2048"),
                     ("table_fill_2048", "oracle_fill_2048"), ("table_fill_2048_v3", "oracle_fill_2048_v3"),
                     ("mean_oracle_2048", "oracle_fill_2048"), ("table_fill_2048_v3", "table_fill_2048"),
                     ("oracle_fill_4096_v3", "oracle_fill_4096"), ("table_fill_0", "table_fill_2048"),
                     ("table_fill_4096_v3", "table_fill_2048_v3"), ("table_fill_0", "mean_old_0"),
                     ("table_fill_2048", "mean_table_2048"), ("table_fill_0_v3", "mean_table_0"),
                     ("mean_table_4096", "mean_old_4096")):
            self.assertIn(pair, pairs)
        self.assertFalse(any(name.startswith(("sanity", "oracle_2", "oracle_4")) for name, _ in pairs))
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_extra_conditions_need_no_native_state(self):
        self.assertEqual(run_long.parse_extra_kinds("none"), [])
        self.assertEqual(run_long.parse_extra_kinds("mean_old,mean_table"), ["mean_old", "mean_table"])
        for text in ("sanity", "oracle_fill", "mean_oracle", "bogus", "mean_table,mean_table"):
            with self.subTest(text=text), self.assertRaises(Exception):
                run_long.parse_extra_kinds(text)
        self.assertEqual(run_long.build_parser().get_default("conditions"), list(run_long.PLAN_KINDS))

    def test_sanity_gates_check_every_statistic(self):
        class Args:
            sanity_kl, sanity_mean_kl, sanity_nll = 1e-2, 1e-3, 5e-3

        class Run:
            args = Args

        def check(max_token_kl=0.0, mean_kl=0.0, nll_delta=0.0):
            return {"max_token_kl": max_token_kl, "mean_kl": mean_kl, "nll_delta": nll_delta, "top1_agree": 1.0}

        self.assertIsNone(run_long.sanity_verdict(Run, {}))
        exact = run_long.sanity_verdict(Run, {"a/sanity_2048": check(), "a/oracle_2048": check()})
        self.assertTrue(exact["passed"] and exact["exact"])
        noisy = run_long.sanity_verdict(Run, {"a/sanity_2048": check(5e-3, 2e-5, -1e-4), "b/oracle_2048": check()})
        self.assertTrue(noisy["passed"])
        self.assertFalse(noisy["exact"])
        self.assertEqual((noisy["max_token_kl"], noisy["nll_delta"]["worst"]), (5e-3, "a/sanity_2048"))
        for bad, statistic in ((check(max_token_kl=0.2), "max_token_kl"), (check(mean_kl=2e-3), "mean_kl"),
                               (check(nll_delta=-0.01), "nll_delta")):
            with self.subTest(statistic=statistic):
                verdict = run_long.sanity_verdict(Run, {"a/sanity_2048": check(), "b/sanity_4096": bad})
                self.assertFalse(verdict["passed"])
                self.assertEqual([(f["check"], f["statistic"]) for f in verdict["failures"]],
                                 [("b/sanity_4096", statistic)])
                self.assertIn("b/sanity_4096 " + statistic, run_long.sanity_error(verdict, "somewhere"))

    def test_verdict_uses_the_confidence_interval(self):
        def item(mean, ci):
            return {"name": "table_fill_2048", "documents": 10, "nll_delta": {"mean": mean, "ci95": ci}}
        self.assertEqual(run_long.verdict(item(0.03, [0.01, 0.045]), 0.05)["status"], "meets")
        self.assertEqual(run_long.verdict(item(0.08, [0.06, 0.1]), 0.05)["status"], "misses")
        straddle = run_long.verdict(item(0.049, [0.02, 0.09]), 0.05)
        self.assertEqual(straddle["status"], "inconclusive")
        self.assertAlmostEqual(straddle["gap"], -0.001)

    def test_caching_mapper_memoizes_by_row_content(self):
        class Fake:
            target_shape = {"k": (2, 4)}
            calls = 0
            cleared = False

            def predict(self, kind, layer, rows, shape):
                Fake.calls += 1
                return np.full((len(rows[3]),) + tuple(shape), float(rows[3].sum() + layer), dtype=np.float32)

            def clear(self):
                Fake.cleared = True

        mapper = run_long.CachingMapper(Fake())
        rng = np.random.default_rng(0)
        rows = {3: rng.normal(size=(5, 1, 8)).astype(np.float32), 7: rng.normal(size=(5, 1, 8)).astype(np.float32)}
        first = mapper.predict("k", 0, rows, (2, 4))
        self.assertEqual(mapper.target_shape, {"k": (2, 4)})
        again = mapper.predict("k", 0, {3: rows[3].copy(), 7: rows[7].copy()}, (2, 4))
        np.testing.assert_array_equal(first, again)
        self.assertEqual((Fake.calls, mapper.hits, mapper.misses), (1, 1, 1))
        mapper.predict("k", 1, rows, (2, 4))
        changed = {3: rows[3], 7: rows[7] + 1}
        mapper.predict("k", 0, changed, (2, 4))
        self.assertEqual(Fake.calls, 3)
        mapper.clear()
        self.assertFalse(Fake.cleared)
        mapper.predict("k", 0, rows, (2, 4))
        self.assertEqual(Fake.calls, 3)
        mapper.release()
        self.assertFalse(Fake.cleared)
        mapper.predict("k", 0, rows, (2, 4))
        self.assertEqual(Fake.calls, 4)
        mapper.unload()
        self.assertTrue(Fake.cleared)

    def test_gpu_reports_need_full_offload_and_the_expected_graph_splits(self):
        class Args:
            device, expect_graph_splits = "vulkan", "2"
        good = {"device": {"backend": "vulkan", "offloaded_layers": "53/53", "graph_splits": "2"}}
        self.assertEqual(run_long.check_device(good, Args, "score"), good["device"])
        for device, fragment in (({"backend": "vulkan", "offloaded_layers": "53/53", "graph_splits": "3"},
                                  "graph splits"),
                                 ({"backend": "vulkan", "offloaded_layers": "50/53", "graph_splits": "2"},
                                  "offloaded"), (None, "device section")):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, fragment):
                run_long.check_device({"device": device}, Args, "score")
        Args.expect_graph_splits = "any"
        run_long.check_device({"device": dict(good["device"], graph_splits="7")}, Args, "score")
        Args.device = "cpu"
        self.assertIsNone(run_long.check_device({}, Args, "score"))
        with self.assertRaisesRegex(ValueError, "CPU run"):
            run_long.check_device(good, Args, "score")

    def test_kv_cache_bytes_count_every_attention_layer(self):
        spec = kvstate.ModelSpec("muse-glimmer", ("sliding", "sliding", "sliding", "full") * 13, (2,) * 52,
                                 (128,) * 52, (128,) * 52, n_swa=2048, rope_type="norm", n_rot=128)
        self.assertEqual(run_long.kv_cache_bytes(spec, 32768, "f16"), 52 * 2 * 256 * 2 * 32768)
        self.assertEqual(run_long.state_bytes_per_token(spec, "f16", spec.layers_of("full")), 13 * 1024)

    def test_scored_positions_follow_the_probe_rule(self):
        tokens = {"byte_end": np.asarray([2, 5, 7, 9, 12, 15])}
        positions, eligible = run_long.scored_positions(tokens, 5, 15, 3)
        self.assertEqual((positions.tolist(), eligible), ([2, 3, 4], 4))
        positions, _ = run_long.scored_positions(tokens, 6, 15, 10)
        self.assertEqual(positions.tolist(), [3, 4, 5])

    def test_argument_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v2 = root / "v2"
            for rel in ("rendered-source.jsonl", "rendered-target.jsonl", "mapper/manifest.json",
                        "alignment/alignment.npz", "spec-kv.json"):
                (v2 / rel).parent.mkdir(parents=True, exist_ok=True)
                (v2 / rel).write_text("{}")
            for rel in ("captures/source", "captures/target-train", "captures/target-eval"):
                (v2 / rel).mkdir(parents=True)
            base = ["--output", str(root / "run"), "--probe", __file__, "--corpus", __file__, "--v2-run", str(v2),
                    "--source-model", __file__, "--source-model-id", "s", "--target-model", __file__,
                    "--target-model-id", "t"]
            for extra, fragment in ((["--grow-train"], "--train-corpus"), (["--until", "grow-train"], "--grow-train"),
                                    (["--batch", "900", "--ctx", "512"], "batch <= ctx"),
                                    (["--expect-graph-splits", "two"], "graph-splits"),
                                    (["--headline", "mean_old_2048"], "--headline"),
                                    (["--sanity-mean-kl", "-1"], "sanity-mean-kl"),
                                    (["--resume"], "not an existing run directory")):
                with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, fragment):
                    run_quiet(run_long.main, base + extra)
            shutil.rmtree(v2 / "captures/target-eval")
            with self.assertRaisesRegex(ValueError, "lacks captures/target-eval"):
                run_quiet(run_long.main, base)


class EndToEndBase:
    """Every stage on tiny fixtures; subclasses pick the probe and device."""

    probe = None
    device = "cpu"
    extra = ()
    headline = None

    @classmethod
    def setUpClass(cls):
        cls.work = Workspace.get()
        cls.run_dir = cls.work.root / f"run-{cls.device}"
        cls.events = run_quiet(run_long.main, cls.argv(cls.run_dir))
        cls.report = json.loads((cls.run_dir / "long/report.json").read_text(encoding="utf-8"))
        cls.plan = json.loads((cls.run_dir / "long/plan.json").read_text(encoding="utf-8"))

    @classmethod
    def argv(cls, output, *extra):
        return cls.work.argv(output, cls.probe, *cls.extra, *extra, device=cls.device)

    def copy_run(self, name):
        destination = self.work.root / f"{self.device}-{name}"
        shutil.copytree(self.run_dir, destination, symlinks=True)
        return destination

    def snapshot(self, run_dir):
        return {str(path.relative_to(run_dir)): (path.stat().st_mtime_ns, path.read_bytes())
                for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != ".lock"}

    def documents(self):
        """The plan's documents with the --extra-conditions ones (mean_table by default)."""
        run = run_long.LongRun(run_long.build_parser().parse_args(self.argv(self.run_dir, "--resume")))
        return run.documents()

    def conditions(self, group="all"):
        return {item["name"]: item for item in self.report["groups"][group]["conditions"]}

    def test_every_stage_completes(self):
        done = [event["stage"] for event in self.events if event["event"] == "stage_done"]
        self.assertEqual(done, [stage for stage in run_long.STAGES if stage != "grow-train" or self.grow])
        self.assertEqual(self.events[-1]["event"], "run_done")
        self.assertEqual(list((self.run_dir / "long/.staging").iterdir()), [])
        for stage in ("qwen-states", "muse-states", "score"):
            self.assertTrue(any(e["event"] == "eta" and e["stage"] == stage for e in self.events), stage)
            self.assertTrue(any(e["event"] == "progress" and e["stage"] == stage for e in self.events), stage)
        self.assertTrue(any(e["event"] == "progress" and e["stage"] == "build" for e in self.events))
        self.assertTrue(any(e["event"] == "verdict" for e in self.events))

    def test_plan_uses_primaries_on_the_batch_grid(self):
        documents = self.plan["documents"]
        self.assertEqual(len(documents), 5)
        self.assertEqual({d["bucket"] for d in documents}, {"0-400", "400-1000"})
        for document in documents:
            for condition in document["conditions"]:
                if condition["tail"]:
                    self.assertEqual(condition["t0"] % BATCH, 0)
                    self.assertLessEqual(condition["t0"], document["prefix"] - condition["tail"])
                else:
                    self.assertEqual(condition["t0"], document["prefix"] - 1)
            self.assertEqual(document["first_scored"], document["prefix"])
            self.assertEqual(document["n_scored"], 16)

    def test_native_state_checks_reproduce_native_scores(self):
        sanity = self.report["sanity"]
        self.assertTrue(sanity["passed"])
        self.assertLessEqual(sanity["max_token_kl"], self.sanity_kl)
        conditions = self.conditions()
        for tail in (48, 96):
            for kind in ("sanity", "oracle"):
                item = conditions[f"{kind}_{tail}"]
                self.assertEqual(item["documents"], 5)
                self.assertLessEqual(item["kl"]["max_token"], self.sanity_kl)
        documents = self.documents()
        self.assertEqual(self.report["timings"][-1]["stage"], "score")
        self.assertEqual(sum(len(json.loads((self.run_dir / f"long/handoff/{d['document_id']}/document.json")
                                            .read_text())["conditions"]) for d in documents),
                         sum(1 for d in documents for c in d["conditions"] if c["kind"] != "sanity"))
        self.assertTrue(all(c["kind"] != "mean_table" for d in self.plan["documents"] for c in d["conditions"]))
        self.assertEqual(sum(1 for d in documents for c in d["conditions"] if c["kind"] == "mean_table"),
                         sum(len({c["tail"] for c in d["conditions"]}) for d in self.plan["documents"]))

    def test_report_has_conditions_contrasts_buckets_and_a_verdict(self):
        conditions = self.conditions()
        families = [None, "v3"] if self.grow else [None]
        for family in families:
            for tail in (0, 48, 96):
                self.assertIn(run_long.condition_name("table_fill", tail, family), conditions)
        for name in ("mean_old_0", "mean_old_48", "mean_oracle_96", "oracle_fill_48", "mean_table_0", "mean_table_96"):
            self.assertIn(name, conditions)
        self.assertGreater(conditions["mean_old_48"]["kl"]["mean"], 0)
        self.assertGreater(conditions["mean_table_48"]["kl"]["mean"], 0)
        self.assertTrue(self.report["positive_control"]["passed"])
        self.assertEqual(self.report["definitions"]["conditions"]["mean_table"]["planned_by"], "--extra-conditions")
        self.assertIn("not information-free", conditions["mean_oracle_48"]["role"])
        for item in conditions.values():
            self.assertTrue(0 < item["handoff_fraction"] < 1, item["name"])
        self.assertLess(conditions["table_fill_96"]["handoff_fraction"], conditions["table_fill_0"]["handoff_fraction"])
        pairs = {(item["name"], item["baseline"]) for item in self.report["groups"]["all"]["contrasts"]}
        self.assertIn(("table_fill_48", "mean_old_48"), pairs)
        self.assertIn(("table_fill_48", "mean_table_48"), pairs)
        self.assertIn(("table_fill_48", "oracle_fill_48"), pairs)
        self.assertIsNotNone(self.report["build"]["table_fill_48"]["table_unseen_share"])
        self.assertIsNone(self.report["build"]["oracle_fill_48"]["table_unseen_share"])
        if self.grow:
            self.assertIn(("table_fill_48_v3", "table_fill_48"), pairs)
        self.assertEqual(set(self.report["groups"]), {"all", "bucket 0-400", "bucket 400-1000"})
        verdict = self.report["verdict"]
        self.assertEqual(len(verdict["deployable"]), 3 * len(families))
        self.assertIn(verdict["headline"]["status"], ("meets", "misses", "inconclusive"))
        self.assertEqual(verdict["headline"]["target"], 0.05)
        self.assertIn("post hoc", verdict["best_post_hoc"]["selection"])
        if self.headline:
            self.assertEqual((verdict["headline"]["condition"], verdict["headline"]["selection"]),
                             (self.headline, "pre-registered"))
        else:
            self.assertIn("table_fill_2048 was not scored", verdict["headline"]["selection"])
            self.assertEqual(verdict["headline"]["condition"], verdict["best_post_hoc"]["condition"])
        self.assertEqual(set(verdict["headline_by_bucket"]), {"0-400", "400-1000"})
        for item in conditions.values():
            self.assertIsNotNone(item["nll_delta"]["ci95"])
        self.assertTrue(self.report["prefix_covariate"])
        self.assertEqual(len(self.report["per_document"]), 5)
        markdown = (self.run_dir / "long/report.md").read_text(encoding="utf-8")
        for heading in ("## Verdict", "## Checks", "## All:", "## Bucket 0-400", "## Per document", "## Build",
                        "## Timings", "## Caveats"):
            self.assertIn(heading, markdown)

    def test_resume_is_a_no_op_and_changes_are_refused(self):
        run_dir = self.copy_run("resumed")
        before = self.snapshot(run_dir)
        events = run_quiet(run_long.main, self.argv(run_dir, "--resume"))
        skipped = [event["stage"] for event in events if event["event"] == "stage_skip"
                   and event.get("reason") == "complete and verified"]
        self.assertEqual(skipped, [stage for stage in run_long.STAGES if stage != "grow-train" or self.grow])
        self.assertFalse(any(event["event"] in ("stage_start", "warning") for event in events))
        self.assertEqual(self.snapshot(run_dir), before)
        with self.assertRaisesRegex(ValueError, "already exists; pass --resume"):
            run_quiet(run_long.main, self.argv(run_dir))
        with self.assertRaisesRegex(ValueError, "different settings"):
            run_quiet(run_long.main, self.argv(run_dir, "--resume", "--max-scored", "8"))


@unittest.skipUnless(PROBE, "set KVPARK_KV_PROBE to run the long evaluation on tiny fixtures")
class CpuEndToEndTests(EndToEndBase, unittest.TestCase):
    """CPU probe with the v3 family (grow-train collects, aligns and refits); sanity is exactly 0 on the grid."""

    probe = PROBE
    device = "cpu"
    grow = True
    sanity_kl = 0.0

    @classmethod
    def setUpClass(cls):
        cls.headline = "table_fill_48_v3"
        cls.extra = ("--grow-train", "--train-corpus", str(Workspace.get().train_corpus), "--sanity-kl", "0",
                     "--headline", cls.headline)
        super().setUpClass()

    def test_grow_train_refits_on_v2_plus_extra_train_with_v2_dev_test(self):
        record = json.loads((self.run_dir / "stages/grow-train.json").read_text())
        metrics = record["metrics"]
        self.assertFalse(record["adopted"])
        v2 = json.loads((self.work.v2 / "mapper/manifest.json").read_text())
        v3 = json.loads((self.run_dir / "refit/mapper/manifest.json").read_text())
        self.assertGreater(v3["rows"]["train"]["pairs"], v2["rows"]["train"]["pairs"])
        self.assertEqual(v3["rows"]["test"], v2["rows"]["test"])
        self.assertEqual(v3["split_documents"]["dev"], v2["split_documents"]["dev"])
        self.assertEqual(set(metrics["r2"]["means"]), {f"{k}/{a}" for k in "kv" for a in ("full", "sliding", "all")})
        self.assertEqual(metrics["alignment"]["documents"], len(SPLITS) + 5)

    def test_grow_train_adopts_existing_outputs(self):
        run_dir = self.copy_run("adopt")
        for rel in ("stages", "long"):
            shutil.rmtree(run_dir / rel)
        before = {rel: (run_dir / rel / "manifest.json").read_bytes() for rel in ("refit/mapper",
                                                                                  "captures/train-extra-target")}
        events = run_quiet(run_long.main, self.argv(run_dir, "--resume", "--until", "grow-train"))
        record = json.loads((run_dir / "stages/grow-train.json").read_text())
        self.assertTrue(record["adopted"])
        self.assertEqual(len(record["metrics"]["adopted"]), 8)
        self.assertEqual([e["stage"] for e in events if e["event"] == "stage_done"], ["prepare", "grow-train"])
        self.assertEqual({rel: (run_dir / rel / "manifest.json").read_bytes() for rel in before}, before)
        run_dir = self.copy_run("adopt-bad")
        shutil.rmtree(run_dir / "stages")
        shutil.rmtree(run_dir / "refit/mapper")
        (run_dir / "refit/.pending-mapper-x").mkdir()
        with self.assertRaisesRegex(ValueError, "a refit is still running"):
            run_quiet(run_long.main, self.argv(run_dir, "--resume", "--until", "grow-train"))

    def test_cached_build_matches_an_uncached_build(self):
        import attention_eval
        document = self.plan["documents"][2]
        condition = next(c for c in document["conditions"] if c["name"] == "table_fill_48")
        run = run_long.LongRun(run_long.build_parser().parse_args(self.argv(self.run_dir, "--resume")))
        qwen_path = next(p for p in (self.run_dir / "long/states/source").rglob("*.state")
                         if p.parent.name == document["document_id"])
        result = handoff.build_handoff(
            kvstate.read_file(qwen_path, run.spec("source")), run.rendered("source")[document["document_id"]],
            run.tokens("source", document["document_id"]), run.rendered("target")[document["document_id"]],
            run.tokens("target", document["document_id"]), attention_eval.Mapper(self.work.v2 / "mapper"),
            t0=condition["t0"], muse_spec=run.spec("target"), content_policy="mapped", fill_policy="table",
            table=self.run_dir / "long/token-table.npz", dtype="f16", v_trans=False, chunk_rows=16)
        built = self.run_dir / f"long/handoff/{document['document_id']}/table_fill_48.state"
        self.assertEqual(result.state.to_bytes(), built.read_bytes())
        cache = json.loads((self.run_dir / "stages/build.json").read_text())["metrics"]["prediction_cache"]
        self.assertGreater(cache["hits"], cache["misses"])

    def test_oracle_rebuilds_are_the_native_state_bytes(self):
        self.assertEqual(self.report["build"]["oracle_48"]["identical_to_native_state"], "5/5")
        self.assertEqual(self.report["sanity"]["max_token_kl"], 0.0)
        self.assertTrue(self.report["sanity"]["exact"])

    def test_corrupted_or_stale_outputs_are_detected(self):
        document = self.plan["documents"][0]["document_id"]
        for rel, stage in ((f"long/handoff/{document}/table_fill_48.state", "build"),
                           ("long/scores/group-000/results.json", "score")):
            with self.subTest(rel=rel):
                run_dir = self.copy_run(f"corrupt-{stage}")
                path = run_dir / rel
                data = bytearray(path.read_bytes())
                data[-2] ^= 1
                path.write_bytes(bytes(data))
                with self.assertRaisesRegex(ValueError, f"stage {stage} outputs changed.*{rel}"):
                    run_quiet(run_long.main, self.argv(run_dir, "--resume"))
        run_dir = self.copy_run("stale-group")
        for rel in ("stages/score.json", "stages/report.json"):
            (run_dir / rel).unlink()
        group = run_dir / "long/scores/group-000/group.json"
        group.write_text(json.dumps(dict(json.loads(group.read_text()), key="0" * 64)))
        with self.assertRaisesRegex(ValueError, "group-000 was scored with other settings"):
            run_quiet(run_long.main, self.argv(run_dir, "--resume"))

    def test_sanity_gates_apply_to_reused_groups_and_never_force_re_scoring(self):
        run_dir = self.copy_run("sanity-gates")
        for rel in ("stages/score.json", "stages/report.json", "long/report.json", "long/report.md"):
            (run_dir / rel).unlink()
        document = self.plan["documents"][3]["document_id"]
        path = run_dir / f"long/scores/group-001/scores/{document}/sanity_96.npy"
        scores = np.load(path)
        scores[0, 1] = 0.5
        np.save(path, scores)
        with self.assertRaisesRegex(ValueError, rf"{document[:12]}/sanity_96 max_token_kl 0.5 > --sanity-kl 0"):
            run_quiet(run_long.main, self.argv(run_dir, "--resume"))
        self.assertFalse((run_dir / "stages/score.json").exists())
        events = run_quiet(run_long.main, self.argv(run_dir, "--resume", "--sanity-kl", "0.9", "--sanity-mean-kl",
                                                    "0.9"))
        self.assertEqual([e["group"] for e in events if e["event"] == "group_reuse"],
                         [f"long/scores/group-{index:03d}" for index in range(3)])
        self.assertFalse(any(e["event"] == "group_done" for e in events))
        report = json.loads((run_dir / "long/report.json").read_text())
        self.assertEqual(report["sanity"]["max_token_kl"], 0.5)
        self.assertTrue(report["sanity"]["passed"])

    def test_an_interrupted_stage_continues_with_the_missing_groups(self):
        run_dir = self.copy_run("interrupted")
        for rel in ("stages/score.json", "stages/report.json", "long/report.json", "long/report.md"):
            (run_dir / rel).unlink()
        shutil.rmtree(run_dir / "long/scores/group-001")
        leftover = run_dir / "long/.staging/score-killed/raw"
        leftover.mkdir(parents=True)
        (leftover / "results.json").write_text("{}")
        events = run_quiet(run_long.main, self.argv(run_dir, "--resume"))
        self.assertEqual(list((run_dir / "long/.staging").iterdir()), [])
        self.assertEqual([e["removed"] for e in events if e["event"] == "cleanup"], [["score-killed"]])
        self.assertEqual([e["group"] for e in events if e["event"] == "group_reuse"],
                         ["long/scores/group-000", "long/scores/group-002"])
        self.assertEqual([e["group"] for e in events if e["event"] == "group_done"], ["long/scores/group-001"])
        self.assertEqual([e["stage"] for e in events if e["event"] == "stage_done"], ["score", "report"])
        report = json.loads((run_dir / "long/report.json").read_text())
        self.assertEqual(report["groups"]["all"]["conditions"], self.report["groups"]["all"]["conditions"])


@unittest.skipUnless(PROBE and VULKAN_PROBE, "set KVPARK_KV_PROBE and KVPARK_KV_PROBE_VULKAN for the GPU run")
class VulkanEndToEndTests(EndToEndBase, unittest.TestCase):
    """The same run with --device vulkan (every probe stage on the GPU, graph splits checked)."""

    probe = VULKAN_PROBE
    device = "vulkan"
    grow = False
    sanity_kl = 1e-6

    def test_gpu_reports_are_recorded(self):
        for stage in ("qwen-states", "muse-states", "score"):
            device = json.loads((self.run_dir / f"stages/{stage}.json").read_text())["metrics"]["device"]
            self.assertEqual((device["backend"], device["graph_splits"]), ("vulkan", "2"), stage)


if __name__ == "__main__":
    unittest.main()
