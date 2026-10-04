"""Patch-directory building, condition definitions and injection summaries with fakes."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import attention_eval as ae
import inject
from align import align
from captures import CaptureSet
from test_attention_eval import TGT_TYPES, doc_id, make_scenario, render, write_fit_mapper


FOCUS, TAIL = 3, 4


def fake_patch_score(docs_path, output, *, sanity_kl=0.0, n_scored=6, seed=0, drop_array=False):
    """Write a patch-score directory the way kvpark-kv-probe does, with synthetic scores.

    Condition i shifts the native NLL by 0.1 * i; KL is 0.05 * i; top-1 agreement drops
    for the first i tokens. applied_rows mirrors the arrays found in each patch dir.
    """
    records = [json.loads(line) for line in Path(docs_path).read_text(encoding="utf-8").splitlines()]
    output = Path(output)
    output.mkdir()
    rng = np.random.default_rng(seed)
    documents = []
    for record in records:
        directory = output / "scores" / record["document_id"]
        directory.mkdir(parents=True)
        native = rng.uniform(1, 3, n_scored)
        conditions = []
        for index, condition in enumerate(record["conditions"]):
            scores = np.zeros((n_scored, 3), dtype=np.float32)
            scores[:, 0] = native + 0.1 * index
            scores[:, 1] = 0.05 * index
            scores[:, 2] = (np.arange(n_scored) >= index).astype(np.float32)
            applied = {}
            if condition["patch_dir"] is not None:
                patch_dir = Path(condition["patch_dir"])
                count = len(np.load(patch_dir / "positions.npy"))
                applied = {path.stem: count for path in sorted(patch_dir.glob("[kv]-*.npy"))}
                if drop_array and applied:
                    applied.pop(sorted(applied)[0])
            if condition["name"] == inject.SANITY:
                scores[:, 0], scores[:, 1], scores[:, 2] = native, sanity_kl, 1
            np.save(directory / f"{condition['name']}.npy", scores)
            conditions.append({"name": condition["name"], "nll_mean": float(np.mean(scores[:, 0], dtype=np.float64)),
                               "kl_mean": float(np.mean(scores[:, 1])), "top1_agree": float(np.mean(scores[:, 2])),
                               "applied_rows": applied})
        documents.append({"document_id": record["document_id"], "n_tokens": 100, "n_scored": n_scored,
                          "n_eligible": n_scored + 2, "scored_truncated": True, "conditions": conditions})
    (output / "results.json").write_text(json.dumps({"format_version": 2, "mode": "patch-score",
                                                     "documents": documents, "settings": {}}))
    (output / "manifest.json").write_text(json.dumps({"model_sha256": "a" * 64}))


def no_rescore(docs, output):
    raise AssertionError("a kept group was rescored")


class ConditionTests(unittest.TestCase):
    def test_catalog_order_names_and_sources(self):
        conditions = inject.define_conditions(TGT_TYPES, focus_layer=FOCUS, tail=TAIL, prune_noop_native=False)
        self.assertEqual([c.name for c in conditions],
                         ["native", "sanity_native_all", "L3_v", "L3_kv", "full_kv", "all_kv", "all_kv_tail4",
                          "all_kv_system_native", "mean_all"])
        by_name = {c.name: c for c in conditions}
        self.assertEqual(by_name["native"].sources, {})
        self.assertEqual(len(by_name["sanity_native_all"].sources), 16)
        self.assertEqual({pair for pair, s in by_name["L3_v"].sources.items() if s == "mapped"}, {("v", 3)})
        self.assertEqual({pair for pair, s in by_name["L3_kv"].sources.items() if s == "mapped"},
                         {("k", 3), ("v", 3)})
        self.assertEqual({pair for pair, s in by_name["full_kv"].sources.items() if s == "mapped"},
                         {(kind, layer) for kind in "kv" for layer in (3, 7)})
        self.assertEqual(set(by_name["mean_all"].sources.values()), {"mean"})
        self.assertEqual((by_name["all_kv_tail4"].override, by_name["all_kv_system_native"].override),
                         ("tail", "system"))

    def test_noop_native_arrays_are_pruned_except_in_sanity(self):
        conditions = {c.name: c for c in inject.define_conditions(TGT_TYPES, focus_layer=FOCUS, tail=TAIL)}
        self.assertEqual(len(conditions["sanity_native_all"].sources), 16)
        l3v = conditions["L3_v"]
        self.assertEqual(sorted(l3v.sources), [("k", 4), ("k", 5), ("k", 6), ("k", 7), ("v", 3),
                                               ("v", 4), ("v", 5), ("v", 6), ("v", 7)])
        self.assertIn(("k", 3), l3v.omitted)
        self.assertEqual(sorted(conditions["full_kv"].omitted),
                         [(kind, layer) for kind in "kv" for layer in (0, 1, 2)])
        self.assertEqual(conditions["all_kv"].omitted, [])
        self.assertEqual(conditions["full_kv"].describe()["arrays"], {"native": 6, "mapped": 4, "mean": 0})

    def test_subset_and_invalid_requests(self):
        names = [c.name for c in inject.define_conditions(TGT_TYPES, focus_layer=FOCUS, names=["mean_all", "L3_v"])]
        self.assertEqual(names, ["native", "sanity_native_all", "L3_v", "mean_all"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            inject.define_conditions(TGT_TYPES, focus_layer=FOCUS, names=["L27_v"])
        with self.assertRaisesRegex(ValueError, "not an attention layer"):
            inject.define_conditions(["recurrent"] * 3 + ["full"], focus_layer=1)


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scenario").mkdir()
        self.s = make_scenario(self.root / "scenario")
        self.plans, _ = ae.plan_documents(self.s["alignment"], self.s["tgt_records"], self.s["tgt_set"],
                                          self.s["src_records"], self.s["src_set"], {"test"})
        self.mapper = ae.Mapper(self.s["mapper"])

    def values(self, plan):
        return inject.PlanValues(plan, self.s["tgt_set"], self.s["src_set"], self.mapper, (2, 8))

    def test_condition_arrays_use_native_mapped_mean_and_overrides(self):
        plan = self.plans[1]
        values = self.values(plan)
        conditions = {c.name: c for c in inject.define_conditions(TGT_TYPES, focus_layer=FOCUS, tail=TAIL)}
        rows = plan.tgt_start + plan.positions
        native = lambda kind, layer: self.s["tgt_arrays"][kind, layer][rows]  # noqa: E731
        sanity = inject.condition_arrays(conditions["sanity_native_all"], plan, values, TAIL)
        for (kind, layer), array in sanity.items():
            np.testing.assert_array_equal(array, native(kind, layer))
        l3v = inject.condition_arrays(conditions["L3_v"], plan, values, TAIL)
        src_rows = plan.src_start + plan.src_positions
        features = np.concatenate([self.s["src_arrays"]["v", layer][src_rows].reshape(len(rows), -1)
                                   for layer in (3, 7)], axis=1)
        expected = features @ np.load(self.s["mapper"] / "W-v-3.npy") + np.load(self.s["mapper"] / "b-v-3.npy")
        np.testing.assert_allclose(l3v["v", 3], expected.reshape(-1, 2, 8), rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(l3v["k", 5], native("k", 5))
        all_kv = inject.condition_arrays(conditions["all_kv"], plan, values, TAIL)
        tail_condition = conditions["all_kv_tail4"]
        tail = inject.condition_arrays(tail_condition, plan, values, TAIL)
        mask = plan.positions >= plan.prefix_tokens - TAIL
        self.assertTrue(mask.any() and not mask.all())
        # Overridden positions are left out of the patch (recomputed), not written with oracle values.
        np.testing.assert_array_equal(inject.condition_rows(tail_condition, plan, TAIL), np.flatnonzero(~mask))
        np.testing.assert_array_equal(tail["k", 2], all_kv["k", 2][~mask])
        self.assertEqual(inject.mapped_row_count(tail_condition, plan, TAIL), int(np.sum(~mask)))
        system = inject.condition_arrays(conditions["all_kv_system_native"], plan, values, TAIL)
        self.assertTrue(plan.system.any() and not plan.system.all())
        np.testing.assert_array_equal(system["v", 6], all_kv["v", 6][~plan.system])
        self.assertEqual(inject.mapped_row_count(conditions["sanity_native_all"], plan, TAIL), 0)
        self.assertEqual(inject.mapped_row_count(conditions["all_kv"], plan, TAIL), len(plan.positions))
        mean = inject.condition_arrays(conditions["mean_all"], plan, values, TAIL)
        mean_k = np.load(self.s["mapper"] / "mean-k-0.npy").reshape(1, 2, 8).astype(np.float64)
        mean_k /= np.sqrt(np.mean(mean_k ** 2, axis=-1, keepdims=True))
        np.testing.assert_allclose(mean["k", 0], np.broadcast_to(mean_k[0], (len(rows), 2, 8)), rtol=1e-6)
        np.testing.assert_array_equal(mean["v", 0], np.broadcast_to(
            np.load(self.s["mapper"] / "mean-v-0.npy").reshape(2, 8), (len(rows), 2, 8)))

    def test_override_covering_every_position_writes_native_rows(self):
        plan = self.plans[1]
        values = self.values(plan)
        wide = len(plan.positions) + plan.prefix_tokens
        condition = {c.name: c for c in inject.define_conditions(TGT_TYPES, focus_layer=FOCUS, tail=wide)}[
            f"all_kv_tail{wide}"]
        np.testing.assert_array_equal(inject.condition_rows(condition, plan, wide), np.arange(len(plan.positions)))
        self.assertEqual(inject.mapped_row_count(condition, plan, wide), 0)
        arrays = inject.condition_arrays(condition, plan, values, wide)
        rows = plan.tgt_start + plan.positions
        np.testing.assert_array_equal(arrays["k", 2], self.s["tgt_arrays"]["k", 2][rows])

    def test_patch_dir_format_and_refusals(self):
        positions = np.array([1, 4, 9])
        arrays = {("k", 2): np.ones((3, 2, 8), np.float32), ("v", 10): np.zeros((3, 2, 8), np.float32)}
        path = inject.write_patch_dir(self.root / "patch", positions, arrays)
        self.assertEqual(sorted(p.name for p in path.iterdir()), ["k-2.npy", "positions.npy", "v-10.npy"])
        loaded = np.load(path / "positions.npy", mmap_mode="r")
        self.assertEqual((loaded.dtype.str, loaded.tolist()), ("<i4", [1, 4, 9]))
        with (path / "k-2.npy").open("rb") as stream:
            self.assertEqual(np.lib.format.read_magic(stream), (1, 0))
        self.assertEqual(np.load(path / "v-10.npy").dtype.str, "<f4")
        with self.assertRaises(FileExistsError):
            inject.write_patch_dir(path, positions, arrays)
        cases = [(np.array([1, 1, 2]), arrays), (np.array([], dtype=np.int64), {}),
                 (np.array([-1, 0, 1]), arrays), (positions, {("k", 2): np.full((3, 2, 8), np.nan, np.float32)}),
                 (positions, {("k", 2): np.ones((2, 2, 8), np.float32)}),
                 (positions, {("q", 2): np.ones((3, 2, 8), np.float32)}),
                 (positions, {("k", 2): np.ones((3, 2, 8), np.float64)}), (positions, {})]
        for number, (bad_positions, bad_arrays) in enumerate(cases):
            with self.subTest(case=number), self.assertRaises(ValueError):
                inject.write_patch_dir(self.root / f"bad-{number}", bad_positions, bad_arrays)
            self.assertFalse((self.root / f"bad-{number}").exists())
        self.assertEqual([p.name for p in self.root.iterdir() if p.name.startswith(".kvpark")], [])

    def test_document_patches_record_and_expected_applied_rows(self):
        plan = self.plans[0]
        conditions = inject.define_conditions(TGT_TYPES, focus_layer=FOCUS, tail=TAIL)
        record = dict(self.s["tgt_records"][0], template_sha256="c" * 64, bos_token="<S>")
        patch_record, expected = inject.build_document_patches(
            self.root / "doc", plan, record, conditions, self.values(plan), self.s["tgt_set"], TAIL)
        self.assertEqual([c["name"] for c in patch_record["conditions"]], [c.name for c in conditions])
        self.assertEqual(patch_record["conditions"][0], {"name": "native", "patch_dir": None})
        self.assertEqual(patch_record["score_ranges"], [record["segments"][2][1:]])
        self.assertEqual((patch_record["template_sha256"], patch_record["bos_token"]), ("c" * 64, "<S>"))
        tokens = self.s["tgt_set"].token_id[self.s["tgt_set"].rows(plan.document_id)]
        self.assertEqual(patch_record["expected_token_sha256"], inject.token_sha256(tokens))
        by_name = {condition.name: condition for condition in conditions}
        for condition in patch_record["conditions"][1:]:
            directory = Path(condition["patch_dir"])
            self.assertTrue(directory.is_absolute())
            rows = inject.condition_rows(by_name[condition["name"]], plan, TAIL)
            np.testing.assert_array_equal(np.load(directory / "positions.npy"), plan.positions[rows])
            files = {path.stem for path in directory.glob("[kv]-*.npy")}
            self.assertEqual(files, set(expected[condition["name"]]))
            self.assertTrue(all(count == len(rows) for count in expected[condition["name"]].values()))
        self.assertLess(len(np.load(Path(patch_record["conditions"][6]["patch_dir"]) / "positions.npy")),
                        len(plan.positions))
        self.assertEqual(expected["native"], {})
        self.assertEqual(len(expected["sanity_native_all"]), 16)


class RunAndSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scenario").mkdir()
        self.s = make_scenario(self.root / "scenario", splits=("test", "test", "test"))
        self.plans, _ = ae.plan_documents(self.s["alignment"], self.s["tgt_records"], self.s["tgt_set"],
                                          self.s["src_records"], self.s["src_set"], {"test"})
        self.conditions = inject.define_conditions(TGT_TYPES, focus_layer=FOCUS, tail=TAIL)

    def run_groups(self, work, **fake):
        return inject.run(self.plans, self.s["tgt_records"], self.s["tgt_set"], self.s["src_set"],
                          ae.Mapper(self.s["mapper"]), self.conditions, work, tail=TAIL, docs_per_run=2,
                          score_fn=lambda docs, output: fake_patch_score(docs, output, **fake), log=lambda _: None)

    def test_run_groups_documents_checks_and_cleans_patches(self):
        runs = self.run_groups(self.root / "work")
        self.assertEqual([path.name for path in runs], ["group-000", "group-001"])
        self.assertFalse((self.root / "work" / "patches" / "group-000").exists())
        report = inject.summarize(runs, seed=3, resamples=500)
        self.assertEqual(report["documents"], 3)
        self.assertEqual(report["scored_tokens"], 18)
        self.assertTrue(report["sanity"]["passed"])
        by_name = {item["name"]: item for item in report["conditions"]}
        self.assertEqual(list(by_name), [c.name for c in self.conditions])
        for index, condition in enumerate(self.conditions):
            item = by_name[condition.name]
            if condition.name == inject.SANITY:
                self.assertEqual((item["nll_delta"]["mean"], item["kl"]["mean"]), (0.0, 0.0))
                continue
            self.assertAlmostEqual(item["nll_delta"]["mean"], 0.1 * index, places=5)
            self.assertAlmostEqual(item["kl"]["mean"], 0.05 * index, places=6)
            self.assertAlmostEqual(item["top1_agree"]["mean"], max(0, 6 - index) / 6, places=6)
            low, high = item["nll_delta"]["ci95"]
            self.assertLessEqual(low, item["nll_delta"]["mean"] + 1e-6)
            self.assertGreaterEqual(high, item["nll_delta"]["mean"] - 1e-6)
        json.dumps(report, allow_nan=False)

    def test_contrasts_and_groups_share_resamples(self):
        runs = self.run_groups(self.root / "work")
        ids = [plan.document_id for plan in self.plans]
        report = inject.summarize(runs, seed=5, resamples=300, groups={"first": ids[:1], "rest": ids[1:]})
        order = [c.name for c in self.conditions]
        contrasts = {(item["name"], item["baseline"]): item for item in report["contrasts"]}
        self.assertNotIn(("native", "mean_all"), contrasts)
        self.assertNotIn(("mean_all", "all_kv"), contrasts)
        item = contrasts["all_kv", "mean_all"]
        expected = 0.1 * (order.index("all_kv") - order.index("mean_all"))
        self.assertAlmostEqual(item["nll"]["mean"], expected, places=5)
        self.assertAlmostEqual(item["kl"]["mean"], expected / 2, places=6)
        by_name = {entry["name"]: entry for entry in report["conditions"]}
        # One seed for every condition: identical per-document values give identical intervals.
        self.assertEqual(by_name["native"]["nll"]["ci95"], by_name[inject.SANITY]["nll"]["ci95"])
        self.assertEqual((report["groups"]["first"]["documents"], report["groups"]["rest"]["documents"]), (1, 2))
        self.assertEqual(report["groups"]["rest"]["scored_tokens"], 12)
        self.assertIsNone(report["groups"]["first"]["conditions"][0]["nll"]["ci95"])
        with self.assertRaisesRegex(ValueError, "unscored documents"):
            inject.summarize(runs, groups={"bad": ["0" * 64]})

    def test_bootstrap_intervals_reflect_document_spread(self):
        runs = self.run_groups(self.root / "work")
        report = inject.summarize(runs, seed=0, resamples=2000)
        nll = {item["name"]: item["nll"] for item in report["conditions"]}["native"]
        documents = inject.load_scores(runs)
        sums = np.array([d["conditions"]["native"][:, 0].sum() for d in documents.values()])
        self.assertAlmostEqual(nll["mean"], sums.sum() / 18, places=6)
        low, high = nll["ci95"]
        self.assertLess(low, high)
        self.assertGreaterEqual(low, sums.min() / 6 - 1e-9)
        self.assertLessEqual(high, sums.max() / 6 + 1e-9)
        again = inject.summarize(runs, seed=0, resamples=2000)
        self.assertEqual(report, again)

    def test_sanity_failure_aborts(self):
        with self.assertRaisesRegex(ValueError, "not a no-op"):
            self.run_groups(self.root / "work", sanity_kl=2e-6)

    def test_missing_applied_rows_abort(self):
        with self.assertRaisesRegex(ValueError, "applied rows differ"):
            self.run_groups(self.root / "work", drop_array=True)

    def test_summaries_reject_inconsistent_results(self):
        runs = self.run_groups(self.root / "work")
        with self.assertRaisesRegex(ValueError, "two patch-score runs"):
            inject.summarize([runs[0], runs[0]])
        document = json.loads((runs[1] / "results.json").read_text())["documents"][0]["document_id"]
        path = runs[1] / "scores" / document / "all_kv.npy"
        scores = np.load(path)
        scores[0, 0] += 5
        path.unlink()
        np.save(path, scores)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            inject.summarize(runs)

    def cli_argv(self, result, *extra):
        directory = self.root / "records"
        if not directory.exists():
            directory.mkdir()
            for name in ("tgt", "src"):
                (directory / f"{name}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in self.s[f"{name}_records"]))
            self.s["alignment"].save(directory / "alignment.npz")
            (self.root / "probe").write_bytes(b"probe")
        return ["run", "--probe", str(self.root / "probe"), "--model", str(self.root / "muse.gguf"),
                "--model-id", "muse-test", "--tgt-docs", str(directory / "tgt.jsonl"), "--tgt-capture",
                str(self.s["tgt"]), "--src-docs", str(directory / "src.jsonl"), "--src-capture", str(self.s["src"]),
                "--alignment", str(directory / "alignment.npz"), "--mapper", str(self.s["mapper"]),
                "--output", str(result), "--focus-layer", "3", "--tail", "4", "--bootstrap", "200", *extra]

    def test_cli_summarize_and_run_publish_new_outputs(self):
        runs = self.run_groups(self.root / "work")
        output = self.root / "report.json"
        with contextlib.redirect_stdout(io.StringIO()):
            inject.main(["summarize", *map(str, runs), "--output", str(output), "--bootstrap", "200"])
        self.assertEqual(json.loads(output.read_text())["documents"], 3)
        with self.assertRaises(FileExistsError):
            inject.main(["summarize", str(runs[0]), "--output", str(output)])

        def fake_score_group(probe, model, model_id, docs, output, **options):
            self.assertEqual(options["max_scored"], 256)
            fake_patch_score(docs, output)

        result = self.root / "injection"
        argv = self.cli_argv(result, "--docs-per-run", "2", "--conditions", "L3_kv,mean_all", "--limit-docs", "2")
        with patch.object(inject, "score_group", fake_score_group), contextlib.redirect_stdout(io.StringIO()):
            inject.main(argv)
        report = json.loads((result / "injection-report.json").read_text())
        self.assertEqual([item["name"] for item in report["conditions"]],
                         ["native", "sanity_native_all", "L3_kv", "mean_all"])
        self.assertEqual((report["planned_documents"], report["documents"]), (3, 2))
        self.assertEqual([d["document_id"] for d in report["scored_documents"]], [doc_id(0), doc_id(1)])
        self.assertEqual(report["truncated_documents"], 2)
        manifest = json.loads((result / "manifest.json").read_text())
        self.assertEqual(manifest["runs"], ["group-000"])
        self.assertEqual(manifest["inputs"]["target_model_sha256"], "a" * 64)
        self.assertFalse(manifest["settings"]["resumed"])
        self.assertTrue((result / "runs" / "group-000" / "results.json").is_file())
        self.assertEqual(sorted(p.name for p in self.root.iterdir() if p.name.startswith(".kvpark")
                                or p.name.endswith(".partial")), [])
        with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
            inject.main(argv)

    def test_cli_keeps_scored_groups_on_failure_and_resumes(self):
        calls = []

        def flaky_score_group(probe, model, model_id, docs, output, **options):
            calls.append(Path(docs).name)
            if len(calls) == 2:
                raise OSError("disk full")
            fake_patch_score(docs, output)

        result = self.root / "injection"
        partial = self.root / "injection.partial"
        argv = self.cli_argv(result, "--docs-per-run", "1", "--conditions", "all_kv")
        quiet = contextlib.ExitStack()
        quiet.enter_context(contextlib.redirect_stdout(io.StringIO()))
        quiet.enter_context(contextlib.redirect_stderr(io.StringIO()))
        with quiet, patch.object(inject, "score_group", flaky_score_group):
            with self.assertRaisesRegex(OSError, "disk full"):
                inject.main(argv)
            self.assertFalse(result.exists())
            self.assertTrue((partial / "out" / "runs" / "group-000" / "results.json").is_file())
            self.assertEqual(list((partial / "patches").iterdir()), [])
            with self.assertRaisesRegex(FileExistsError, "--resume"):
                inject.main(argv)
            with self.assertRaisesRegex(ValueError, r"differing: .*'settings.tail'"):
                inject.main(self.cli_argv(result, "--docs-per-run", "1", "--conditions", "all_kv", "--resume")
                            [:-3] + ["--tail", "5", "--resume"])
            # A killed collect_v2 leaves its staging directory behind; resume clears it.
            (partial / "out" / "runs" / ".kvpark-probe-stale").mkdir()
            inject.main(argv + ["--resume"])
        self.assertEqual(calls, ["group-000.jsonl", "group-001.jsonl", "group-001.jsonl", "group-002.jsonl"])
        self.assertFalse(partial.exists())
        report = json.loads((result / "injection-report.json").read_text())
        self.assertEqual(report["documents"], 3)
        manifest = json.loads((result / "manifest.json").read_text())
        self.assertEqual(manifest["runs"], ["group-000", "group-001", "group-002"])
        self.assertTrue(manifest["settings"]["resumed"])
        self.assertEqual(sorted(p.name for p in (result / "runs").iterdir()), manifest["runs"])
        with self.assertRaisesRegex(ValueError, "nothing to resume"), contextlib.redirect_stdout(io.StringIO()):
            inject.main(self.cli_argv(self.root / "other", "--resume"))

    def test_resume_revalidates_kept_groups(self):
        work = self.root / "work"
        self.run_groups(work)
        with self.assertRaises(FileExistsError):
            inject.run(self.plans, self.s["tgt_records"], self.s["tgt_set"], self.s["src_set"],
                       ae.Mapper(self.s["mapper"]), self.conditions, work, tail=TAIL, docs_per_run=2,
                       score_fn=fake_patch_score, log=lambda _: None)
        path = work / "out" / "runs" / "group-001" / "scores" / doc_id(2) / "sanity_native_all.npy"
        scores = np.load(path)
        scores[:, 1] = 1e-3
        path.unlink()
        np.save(path, scores)
        with self.assertRaisesRegex(ValueError, "not a no-op"):
            inject.run(self.plans, self.s["tgt_records"], self.s["tgt_set"], self.s["src_set"],
                       ae.Mapper(self.s["mapper"]), self.conditions, work, tail=TAIL, docs_per_run=2,
                       score_fn=no_rescore, resume=True, log=lambda _: None)

    def test_documents_file_selects_and_orders(self):
        chosen = self.root / "documents.txt"
        chosen.write_text(f"{doc_id(2)}\n\n{doc_id(0)}\n")
        result = self.root / "injection"
        with patch.object(inject, "score_group", lambda *a, **k: fake_patch_score(a[3], a[4])), \
                contextlib.redirect_stdout(io.StringIO()):
            inject.main(self.cli_argv(result, "--conditions", "L3_v", "--documents", str(chosen)))
        report = json.loads((result / "injection-report.json").read_text())
        self.assertEqual([d["document_id"] for d in report["scored_documents"]], [doc_id(2), doc_id(0)])
        manifest = json.loads((result / "manifest.json").read_text())
        self.assertEqual(manifest["inputs"]["documents"], [doc_id(2), doc_id(0)])
        for text, pattern in ((f"{doc_id(0)}\n{doc_id(0)}\n", "distinct"), ("", "distinct"),
                              (f"{doc_id(9)}\n", "not a planned document")):
            chosen.write_text(text)
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, pattern):
                inject.main(self.cli_argv(self.root / "other", "--documents", str(chosen)))
        self.assertFalse((self.root / "other.partial").exists())

    def test_scored_model_must_match_the_target_capture(self):
        def other_model(probe, model, model_id, docs, output, **options):
            fake_patch_score(docs, output)
            (output / "manifest.json").write_text(json.dumps({"model_sha256": "f" * 64}))

        quiet = contextlib.ExitStack()
        quiet.enter_context(contextlib.redirect_stdout(io.StringIO()))
        quiet.enter_context(contextlib.redirect_stderr(io.StringIO()))
        with quiet, patch.object(inject, "score_group", other_model), \
                self.assertRaisesRegex(ValueError, "differs from the model of the target capture"):
            inject.main(self.cli_argv(self.root / "injection", "--conditions", "L3_v"))
        self.assertFalse((self.root / "injection").exists())

    def test_sanity_condition_is_required(self):
        conditions = [c for c in self.conditions if c.name != inject.SANITY]
        with self.assertRaisesRegex(ValueError, "no-op check is required"):
            inject.run(self.plans, self.s["tgt_records"], self.s["tgt_set"], self.s["src_set"],
                       ae.Mapper(self.s["mapper"]), conditions, self.root / "work", tail=TAIL, docs_per_run=2,
                       score_fn=fake_patch_score, log=lambda _: None)

    def test_missing_condition_in_results_aborts(self):
        def drop_last(docs, output):
            path = Path(docs)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for record in records:
                record["conditions"] = record["conditions"][:-1]
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            fake_patch_score(path, output)

        with self.assertRaisesRegex(ValueError, "have conditions"):
            inject.run(self.plans, self.s["tgt_records"], self.s["tgt_set"], self.s["src_set"],
                       ae.Mapper(self.s["mapper"]), self.conditions, self.root / "work", tail=TAIL,
                       docs_per_run=2, score_fn=drop_last, log=lambda _: None)

    def test_row_caches_and_maps_are_released_before_scoring(self):
        mapper = ae.Mapper(self.s["mapper"])
        seen = []

        def score(docs, output):
            seen.append(dict(mapper._maps))
            fake_patch_score(docs, output)

        inject.run(self.plans, self.s["tgt_records"], self.s["tgt_set"], self.s["src_set"], mapper,
                   self.conditions, self.root / "work", tail=TAIL, docs_per_run=2, score_fn=score,
                   log=lambda _: None)
        self.assertEqual(seen, [{}, {}])

    def test_score_group_invokes_collect_v2_patch_score(self):
        calls = []
        with patch("collect_v2.main", lambda argv: calls.append(argv)):
            inject.score_group(Path("/p/probe"), Path("/m/muse.gguf"), "muse", Path("/d/docs.jsonl"),
                               Path("/o/out"), ctx=5120, batch=512, threads=8, max_scored=128,
                               hash_cache=Path("/h/cache.json"), skip_memory_check=True)
        argv = calls[0]
        self.assertEqual(argv[argv.index("--mode") + 1], "patch-score")
        self.assertEqual(argv[argv.index("--max-scored") + 1], "128")
        self.assertIn("--skip-memory-check", argv)
        self.assertEqual(argv[argv.index("--hash-cache") + 1], "/h/cache.json")


PROBE = os.environ.get("KVPARK_KV_PROBE")


@contextlib.contextmanager
def quiet_fds(log_path):
    """Send fd 1 and 2 (the probe's inherited output) to a log file while inside."""
    saved = [os.dup(1), os.dup(2)]
    with open(log_path, "wb") as log:
        try:
            for fd in (1, 2):
                os.dup2(log.fileno(), fd)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yield
        finally:
            for fd, copy in zip((1, 2), saved):
                os.dup2(copy, fd)
                os.close(copy)


@unittest.skipUnless(PROBE, "set KVPARK_KV_PROBE to score patches with the real kv_probe")
class RuntimeInjectionTests(unittest.TestCase):
    """Tiny Muse/Qwen fixtures: real collect, align, a random fit_layers-format mapper, and
    inject through collect_v2 and kvpark-kv-probe --mode patch-score."""

    def collect(self, root, model, records, spec, name):
        directory = root / name
        directory.mkdir()
        (directory / "docs.jsonl").write_text("".join(
            json.dumps({"document_id": r["document_id"], "text": r["text"]}) + "\n" for r in records))
        (directory / "spec.json").write_text(json.dumps(spec))
        result = subprocess.run([PROBE, "--mode", "collect", "--model", str(model), "--input",
                                 str(directory / "docs.jsonl"), "--output", str(directory / "capture"),
                                 "--ctx", "1024", "--batch", "512", "--threads", "2",
                                 "--capture-spec", str(directory / "spec.json")],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return directory / "capture"

    def test_sanity_is_exact_pruning_is_a_no_op_and_mapping_changes_predictions(self):
        from tiny_fixture import create_fixture
        target_template = ("<S>", "<E>", "<U>", "<E>", "<A>", "<E>")
        source_template = ("[sys]", "[/]", "[usr]", "[/]", "[bot]", "[/]")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_fixture(root / "muse.gguf", "muse-glimmer", sliding_window=16,
                           control_tokens=["<S>", "<E>", "<U>", "<A>"])
            create_fixture(root / "qwen.gguf", "qwen35", control_tokens=["[sys]", "[/]", "[usr]", "[bot]"])
            rng = np.random.default_rng(1)
            words = ["alpha", "beta", "gamma", "delta", "kappa", "omega", "sigma", "caf\u00e9"]
            tgt_records, src_records = [], []
            for number in range(5):
                messages = [f"You are helpful {number}.", " ".join(rng.choice(words, 6 + number)),
                            " ".join(rng.choice(words, 5))]
                document = hashlib.sha256(f"d{number}".encode()).hexdigest()
                for template, records in ((target_template, tgt_records), (source_template, src_records)):
                    text, segments = render(template, messages)
                    records.append({"document_id": document, "text": text, "segments": segments,
                                    "segment_keys": ["0:content", "1:content", "2:content"], "split": "test",
                                    "final_message_segments": [2], "source": "synthetic"})
            tgt = self.collect(root, root / "muse.gguf", tgt_records,
                               {"k": "all", "v": "all", "q": "all", "gate": "all", "attn_out": "all"}, "tgt")
            src = self.collect(root, root / "qwen.gguf", src_records, {"k": "all", "v": "all"}, "src")
            tgt_set, src_set = CaptureSet([tgt]), CaptureSet([src])
            align(src_records, src_set, tgt_records, tgt_set).save(root / "alignment.npz")
            for name, records in (("tgt", tgt_records), ("src", src_records)):
                (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            source_layer = src_set.layers("k")[-1]
            source_shape = tuple(src_set.captures[0].array("k", source_layer).shape[1:])
            target_shape = tuple(tgt_set.captures[0].array("k", 0).shape[1:])
            width, outputs = int(np.prod(source_shape)), int(np.prod(target_shape))
            maps = [(kind, layer, [source_layer], kind == "k",
                     (0.3 * rng.normal(size=(width, outputs))).astype(np.float32),
                     rng.normal(size=outputs).astype(np.float32), rng.normal(size=outputs).astype(np.float32))
                    for layer in tgt_set.layers("k") for kind in ("k", "v")]
            mapper = write_fit_mapper(root / "mapper", maps, source_shape=source_shape, target_shape=target_shape)
            common = ["--tgt-docs", str(root / "tgt.jsonl"), "--tgt-capture", str(tgt), "--src-docs",
                      str(root / "src.jsonl"), "--src-capture", str(src), "--alignment", str(root / "alignment.npz"),
                      "--mapper", str(mapper)]
            reports = {}
            for label, extra in (("pruned", []), ("literal", ["--write-noop-native"])):
                argv = ["run", "--probe", PROBE, "--model", str(root / "muse.gguf"), "--model-id", "tiny",
                        *common, "--output", str(root / label), "--focus-layer", "1", "--tail", "8",
                        "--docs-per-run", "2", "--ctx", "1024", "--threads", "2", "--skip-memory-check",
                        "--hash-cache", str(root / "hashes.json"), "--bootstrap", "200", *extra]
                with quiet_fds(root / f"{label}.log"):
                    inject.main(argv)
                reports[label] = json.loads((root / label / "injection-report.json").read_text())
            pruned = {item["name"]: item for item in reports["pruned"]["conditions"]}
            literal = {item["name"]: item for item in reports["literal"]["conditions"]}
            self.assertEqual(reports["pruned"]["documents"], 5)
            self.assertEqual(list(pruned), ["native", "sanity_native_all", "L1_v", "L1_kv", "full_kv", "all_kv",
                                            "all_kv_tail8", "all_kv_system_native", "mean_all"])
            self.assertEqual(pruned["sanity_native_all"]["kl"]["max_token"], 0.0)
            self.assertTrue(reports["pruned"]["sanity"]["passed"])
            for name in pruned:
                with self.subTest(condition=name):
                    self.assertEqual(pruned[name]["nll"]["mean"], literal[name]["nll"]["mean"])
                    self.assertEqual(pruned[name]["kl"]["mean"], literal[name]["kl"]["mean"])
            for name in ("all_kv", "mean_all", "L1_kv"):
                self.assertGreater(pruned[name]["kl"]["mean"], 1e-4)
            applied = reports["pruned"]["per_document"][0]["conditions"]
            self.assertIn("v-1", applied["L1_v"]["applied_rows"])
            self.assertNotIn("k-1", applied["L1_v"]["applied_rows"])
            self.assertIn("k-1", reports["literal"]["per_document"][0]["conditions"]["L1_v"]["applied_rows"])

            with contextlib.redirect_stdout(io.StringIO()):
                ae.main(["--model", str(root / "muse.gguf"), *common, "--output", str(root / "attention.json"),
                         "--splits", "test", "--variants", "kv,mean_kv"])
            attention = json.loads((root / "attention.json").read_text())
            self.assertEqual(attention["unvalidated_attention_types"], [])
            self.assertTrue(all(item["max_relative_error"] < ae.VALIDATION_TOLERANCE
                                for item in attention["validation"]))


if __name__ == "__main__":
    unittest.main()
