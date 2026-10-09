"""campaign.py without a model: stage planning and resume, the capture forecast and its chunks
(speed, size, peak, timeout, adoption), template parity rules, the GPU/memory lock and the commands
under it (process groups, inherited lock, waits, provenance), disk accounting, the evaluation
statistics and reports, assembled-hub bookkeeping and the stage runner with stand-in stages."""

import contextlib
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np

import campaign
from test_resource_guard import FakeMachine

_MODULE_PATCHES = []


def setUpModule():
    """Every command campaign.run starts runs under the resource guard: give it a cool fake machine
    and a CPU lock of its own, so these tests neither wait for this machine to cool nor meet a real
    job's lock."""
    directory = tempfile.TemporaryDirectory()
    fake = FakeMachine(directory.name)
    _MODULE_PATCHES.extend([directory, mock.patch.dict(os.environ, fake.environ(KVPARK_GUARD_POLL_S="0.1")),
                            mock.patch.object(campaign, "CPU_LOCK", fake.cpu_lock)])
    for patcher in _MODULE_PATCHES[1:]:
        patcher.start()


def tearDownModule():
    for patcher in reversed(_MODULE_PATCHES[1:]):
        patcher.stop()
    _MODULE_PATCHES.pop(0).cleanup()
    _MODULE_PATCHES.clear()


def statuses(**items):
    return {stage: items.get(stage) for stage in campaign.STAGES}


def quiet(test):
    """Silence Campaign.log's echo for the rest of a test."""
    stream = contextlib.redirect_stdout(io.StringIO())
    stream.__enter__()
    test.addCleanup(stream.__exit__, None, None, None)


def draft_gguf(path, window):
    """A GGUF holding only a DFlash draft's sliding window (what dflash_drafts reads)."""
    import tiny_fixture
    gguf = tiny_fixture._gguf_module()
    writer = gguf.GGUFWriter(str(path), "dflash")
    writer.add_sliding_window(window)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return Path(path)


class PlanTest(unittest.TestCase):
    def test_fresh_model_runs_every_applicable_stage(self):
        self.assertEqual(campaign.plan_run(statuses(), recurrent=True), list(campaign.STAGES))
        self.assertEqual(campaign.plan_run(statuses(), recurrent=False),
                         [s for s in campaign.STAGES if s != "recurrent"])

    def test_done_stages_are_skipped_and_unfinished_ones_rerun(self):
        state = statuses(render="done", capture="done", onboard="failed", tables="running", recurrent="done",
                         evaluate="stale")
        self.assertEqual(campaign.plan_run(state, recurrent=True), ["onboard", "tables", "evaluate", "report"])

    def test_redo_takes_every_dependent_stage_along(self):
        done = statuses(**{stage: "done" for stage in campaign.STAGES})
        self.assertEqual(campaign.plan_run(done, recurrent=True, redo=["capture"]),
                         ["capture", "onboard", "tables", "evaluate", "report"])
        self.assertEqual(campaign.plan_run(done, recurrent=True, redo=["recurrent"]), ["recurrent", "evaluate", "report"])
        self.assertEqual(campaign.plan_run(done, recurrent=False, redo=["render"]),
                         ["render", "capture", "onboard", "tables", "evaluate", "report"])
        self.assertEqual(campaign.stale_stages(["tables"], recurrent=True), ["tables", "evaluate", "report"])
        self.assertEqual(campaign.downstream("render"), {"capture", "onboard", "tables", "evaluate", "report"})

    def test_only_runs_the_named_stages_with_their_requirements_met(self):
        state = statuses(render="done", capture="done")
        self.assertEqual(campaign.plan_run(state, recurrent=True, only=["onboard", "tables"]), ["onboard", "tables"])
        self.assertEqual(campaign.plan_run(state, recurrent=True, only=["recurrent"]), ["recurrent"])
        self.assertEqual(campaign.plan_run(statuses(**{s: "done" for s in campaign.STAGES}), recurrent=True,
                                           only=["capture"]), [])
        with self.assertRaisesRegex(ValueError, "evaluate needs onboard"):
            campaign.plan_run(state, recurrent=True, only=["evaluate"])
        with self.assertRaisesRegex(ValueError, "capture needs render"):
            campaign.plan_run(statuses(), recurrent=True, only=["capture"])
        with self.assertRaisesRegex(ValueError, "onboard needs capture"):     # a redone requirement is not done
            campaign.plan_run(statuses(**{s: "done" for s in campaign.STAGES}), recurrent=True, only=["onboard"],
                              redo=["capture"])

    def test_a_model_without_a_recurrent_state_needs_no_mean(self):
        state = statuses(render="done", capture="done", onboard="done", tables="done")
        self.assertEqual(campaign.plan_run(state, recurrent=False, only=["evaluate"]), ["evaluate"])
        with self.assertRaisesRegex(ValueError, "evaluate needs recurrent"):
            campaign.plan_run(state, recurrent=True, only=["evaluate"])

    def test_stage_lists(self):
        self.assertEqual(campaign.parse_stages("report, render"), ["render", "report"])
        self.assertEqual(campaign.parse_stages(None), [])
        with self.assertRaisesRegex(ValueError, "unknown stages"):
            campaign.parse_stages("render,fit")


def document(number, split, source, tokens):
    return {"document_id": f"{number:064x}", "split": split, "source": source, "tokens": tokens}


class CapturePlanTest(unittest.TestCase):
    def corpus(self):
        docs = [document(1, "dev", "a", 100), document(2, "test", "b", 100)]
        docs += [document(10 + i, "train", "a", 100) for i in range(6)]
        docs += [document(30 + i, "train", "b", 300) for i in range(3)]
        return docs

    def test_everything_fits(self):
        plan = campaign.capture_plan(self.corpus(), bytes_per_token=10, tokens_per_second=1000,
                                     budget_bytes=10 ** 6, budget_seconds=3600)
        self.assertEqual(plan["limit"], "corpus")
        self.assertEqual(plan["documents"], [d["document_id"] for d in self.corpus()])
        self.assertEqual(plan["tokens"], 1700)
        self.assertEqual(plan["bytes"], 17000)
        self.assertEqual(plan["train_fraction"], 1.0)
        self.assertEqual(plan["by_split"]["train"], {"documents": 9, "tokens": 1500})
        self.assertEqual([int(d, 16) for d in plan["held_out"]], [1, 2])
        # the priority order alternates the sources, each in id order
        self.assertEqual([int(d, 16) for d in plan["order"]], [10, 30, 11, 31, 12, 32, 13, 14, 15])

    def test_size_budget_keeps_held_out_documents_and_alternates_sources(self):
        # 900 tokens: dev + test (200), then train round-robin a, b, a, b, ...: a10 (100), b30 (300),
        # a11 (100), b31 (would reach 1000: skipped), a12 (100), b32 skipped, a13 (100) -> 900
        plan = campaign.capture_plan(self.corpus(), bytes_per_token=10, tokens_per_second=10 ** 6,
                                     budget_bytes=9000, budget_seconds=3600)
        chosen = {int(d, 16) for d in plan["documents"]}
        self.assertEqual(chosen, {1, 2, 10, 30, 11, 12, 13})
        self.assertEqual(plan["tokens"], 900)
        self.assertEqual(plan["limit"], "bytes")
        self.assertEqual(plan["budget_tokens"], 900)
        self.assertEqual(plan["documents"], [d["document_id"] for d in self.corpus() if int(d["document_id"], 16)
                                             in chosen])               # render order kept
        self.assertAlmostEqual(plan["train_fraction"], 700 / 1500, places=4)
        # room reserved for a chunk's peak comes off the size budget
        reserved = campaign.capture_plan(self.corpus(), bytes_per_token=10, tokens_per_second=10 ** 6,
                                         budget_bytes=9000, budget_seconds=3600, reserve_tokens=300)
        self.assertEqual(reserved["budget_tokens"], 600)

    def test_time_budget(self):
        plan = campaign.capture_plan(self.corpus(), bytes_per_token=10, tokens_per_second=1,
                                     budget_bytes=10 ** 9, budget_seconds=500, load_seconds=100)
        self.assertEqual(plan["limit"], "time")
        self.assertEqual(plan["budget_tokens"], 400)
        self.assertEqual(plan["tokens"], 400)
        self.assertEqual(plan["seconds"], 500.0)

    def test_held_out_documents_must_fit(self):
        with self.assertRaisesRegex(ValueError, "dev and test documents alone"):
            campaign.capture_plan(self.corpus(), bytes_per_token=10, tokens_per_second=1000, budget_bytes=1000)
        with self.assertRaisesRegex(ValueError, "positive"):
            campaign.capture_plan(self.corpus(), bytes_per_token=0, tokens_per_second=1)

    def test_plan_is_deterministic_whatever_the_input_order(self):
        docs = self.corpus()
        plan = campaign.capture_plan(docs, bytes_per_token=10, tokens_per_second=10 ** 6, budget_bytes=9000)
        again = campaign.capture_plan(docs[::-1], bytes_per_token=10, tokens_per_second=10 ** 6, budget_bytes=9000)
        self.assertEqual(sorted(plan["documents"]), sorted(again["documents"]))
        self.assertEqual(plan["order"], again["order"])

    def test_next_chunk(self):
        tokens = {"a": 100, "b": 300, "c": 100, "d": 50, "e": 1000}
        pending = ["a", "b", "c", "d", "e"]
        self.assertEqual(campaign.next_chunk(pending, tokens, chunk_tokens=450, budget_tokens=10 ** 6), (["a", "b"], []))
        self.assertEqual(pending, ["c", "d", "e"])
        # a document past the remaining budget is dropped, a smaller one after it still taken
        self.assertEqual(campaign.next_chunk(pending, tokens, chunk_tokens=450, budget_tokens=160), (["c", "d"], ["e"]))
        self.assertEqual(pending, [])
        # a document larger than a chunk makes a chunk of its own
        self.assertEqual(campaign.next_chunk(["e", "a"], tokens, chunk_tokens=450, budget_tokens=10 ** 6), (["e"], []))
        self.assertEqual(campaign.chunk_tokens_for(60), campaign.CHUNK_MIN_TOKENS)
        self.assertEqual(campaign.chunk_tokens_for(6 * 10 ** 6), 10 ** 6)


class FakeSpec:
    def __init__(self, layer_types, shapes, indexer=None, key_slots=1, architecture="lfm2", recurrent_row_bytes=None):
        self.layer_types = layer_types
        self.attention_layers = tuple(i for i, kind in enumerate(layer_types) if kind in ("full", "sliding", "nextn"))
        self.recurrent_layers = tuple(i for i, kind in enumerate(layer_types) if kind == "recurrent")
        self.shapes, self.indexer, self.indexer_key_slots = shapes, indexer, key_slots
        self.architecture, self.recurrent_row_bytes = architecture, recurrent_row_bytes

    def head_shape(self, layer, key):
        return self.shapes[key]


class TypedSpec(FakeSpec):
    """A FakeSpec whose K and V head shape depends on the layer's type (Gemma 4's sliding and full layers)."""

    def __init__(self, layer_types, shapes, **kwargs):
        super().__init__(layer_types, shapes, architecture="gemma4", **kwargs)

    def head_shape(self, layer, key):
        return self.shapes[self.layer_types[layer]]


class ModelFactsTest(unittest.TestCase):
    def test_bytes_per_token(self):
        gqa = FakeSpec(("recurrent", "full", "recurrent", "full", "nextn"), {"k": (8, 64), "v": (8, 64)})
        self.assertEqual(campaign.bytes_per_token(gqa), 2 * 2 * 512 * 4 + 13)       # nextn layers are not captured
        dsa = FakeSpec(("full", "full"), {"k": (1, 576), "v": (1, 512)},
                       indexer=types.SimpleNamespace(layers=(0, 1), head_size=128))
        self.assertEqual(campaign.bytes_per_token(dsa), 2 * (576 + 512) * 4 + 2 * 128 * 4 + 13)

    def test_fit_hub_memory_grows_with_the_model_width(self):
        # LFM2-2.6B: 8 attention layers of 8 x 64; DeepSeek-V2-Lite: 27 MLA layers, K 576 and V 512 wide;
        # Qwen3.6-35B-A3B: 10 attention layers of 2 x 256
        lfm2 = FakeSpec(("recurrent", "full") * 8, {"k": (8, 64), "v": (8, 64)})
        dsv2 = FakeSpec(("full",) * 27, {"k": (1, 576), "v": (1, 512)}, architecture="deepseek2")
        qwen36 = FakeSpec(("recurrent", "recurrent", "recurrent", "full") * 10, {"k": (2, 256), "v": (2, 256)},
                          architecture="qwen35moe")

        def extra(spec, dev, gpu=True):
            return campaign.fit_hub_extra(spec, 512, dev_spans=dev, pivots=4, hub_dim=4096, gpu=gpu)

        def formula(stage1, columns, dev, gpu=True):
            return (campaign.FIT_HUB_BASE + (campaign.FIT_HUB_GPU if gpu else 0) + 28 * stage1 ** 2
                    + 10 * 4096 * columns + 12 * 4096 * columns + dev * 4 * 12 * 4096)
        self.assertEqual(extra(lfm2, 9945), formula(4096, 4096, 9945))
        self.assertEqual(extra(dsv2, 8010, gpu=False), formula(13824, 27 * 576, 8010, gpu=False))   # K is the wider kind
        base_hub = 4.58 * campaign.GIB
        # what the three CPU fits measured with the 4.6 GiB base hub (anonymous memory) and their dev spans: the
        # estimate is above it and not far above
        for spec, dev, measured in ((lfm2, 9945, 11.0), (dsv2, 8010, 16.72), (qwen36, 17355, 12.96)):
            self.assertGreater(base_hub + extra(spec, dev, gpu=False), measured * campaign.GIB)
            self.assertLess(base_hub + extra(spec, dev, gpu=False), (measured + 2.5) * campaign.GIB)
            # the base hub's dev spans (17,355) bound every model's
            self.assertGreaterEqual(extra(spec, 17355), extra(spec, dev))
        # Gemma 4: five sliding layers to one full; 26B-A4B 8 x 256 and 2 x 512, 31B 16 x 256 and 4 x 512
        pattern = ("sliding",) * 5 + ("full",)
        gemma26 = TypedSpec(pattern * 5, {"sliding": (8, 256), "full": (2, 512)})
        gemma31 = TypedSpec(pattern * 10, {"sliding": (16, 256), "full": (4, 512)})
        self.assertEqual(extra(gemma26, 16930), formula(15360, 25 * 2048 + 5 * 1024, 16930))
        # recorded on the GPU (2026-10-08/09): anonymous peak + torch's reserved peak on the device (GTT)
        self.assertGreater(base_hub + extra(gemma26, 16930), (15.74 + 8.43) * campaign.GIB)
        self.assertGreater(base_hub + extra(gemma31, 16930), (32.7 + 23.67) * campaign.GIB)
        # the dev rows tiled over the pivots (counted until 2026-10-09) asked 98 GiB for Gemma-4-31B: the lock
        # (25 GiB margin) could never be granted with 114 GiB free of the machine's 123
        self.assertLess(base_hub + extra(gemma31, 17355) + campaign.GPU_MARGIN, 114 * campaign.GIB)
        nextn = FakeSpec(("full", "nextn"), {"k": (1, 576), "v": (1, 512)})
        self.assertEqual(extra(nextn, 100), formula(512, 576, 100))                 # nextn layers are not captured
        # a fit on the GPU unless fit_hub runs in the experiment venv (no torch)
        self.assertFalse(campaign.fits_on_gpu(campaign.PYTHON))
        self.assertTrue(campaign.fits_on_gpu(campaign.FIT_PYTHON))

    def test_capture_mode_and_refusals(self):
        self.assertEqual(campaign.capture_mode("lfm2"), "collect")
        self.assertEqual(campaign.capture_mode("bailingmoe3"), "states")
        with self.assertRaisesRegex(ValueError, "deepseek4"):
            campaign.capture_mode("deepseek4")
        # where: the Vulkan probe, except states captures and Gemma 4 (its Vulkan capture run differs
        # from a plain run), which collect on the CPU probe
        self.assertEqual(campaign.capture_device("lfm2"), "vulkan")
        self.assertEqual(campaign.capture_device("bailingmoe3"), "cpu")
        self.assertEqual((campaign.capture_mode("gemma4"), campaign.capture_device("gemma4")), ("collect", "cpu"))
        self.assertTrue(all((mode, campaign.capture_device(arch)) in campaign.CAPTURE_SPEED
                            for arch in ("lfm2", "gemma4", "bailingmoe3") for mode in [campaign.capture_mode(arch)]))
        self.assertIsNone(campaign.refusal("qwen35"))
        self.assertEqual(campaign.capture_ctx(2955), 5120)
        self.assertEqual(campaign.capture_ctx(5100), 5632)

    def test_recurrent_cell(self):
        conv = FakeSpec(("recurrent", "full"), {"k": (8, 64), "v": (8, 64)}, recurrent_row_bytes=(16384, 0))
        cell = campaign.recurrent_cell(conv)
        self.assertFalse(cell["mean"])
        self.assertIn("convolution taps only", cell["reason"])
        delta = FakeSpec(("recurrent", "full"), {"k": (8, 64), "v": (8, 64)}, recurrent_row_bytes=(6144, 524288))
        self.assertTrue(campaign.recurrent_cell(delta)["mean"])
        unknown = FakeSpec(("recurrent", "full"), {"k": (8, 64), "v": (8, 64)})
        self.assertTrue(campaign.recurrent_cell(unknown)["mean"])                 # unknown sizes: the mean, to be safe
        dense = FakeSpec(("full", "full"), {"k": (8, 64), "v": (8, 64)})
        self.assertEqual(campaign.recurrent_cell(dense)["layers"], 0)
        self.assertFalse(campaign.recurrent_cell(dense)["mean"])

    def test_parity_sample_and_prompt_shape(self):
        conversations = [{"id": f"{source}{n}", "source": source, "messages": []}
                         for source in "abc" for n in (3, 1, 2)]
        sample = campaign.parity_sample(conversations, per_source=2)
        self.assertEqual([c["id"] for c in sample], ["a1", "a2", "b1", "b2", "c1", "c2"])
        conversation = {"id": "x", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                                                {"role": "tool", "content": "t"}, {"role": "assistant", "content": "b"}]}
        self.assertEqual([m["role"] for m in campaign.prompt_shape(conversation)["messages"]],
                         ["user", "assistant", "tool"])
        self.assertEqual(len(conversation["messages"]), 4)

    def test_model_file_comes_from_the_catalog_and_refused_architectures_stop_early(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.json"
            catalog.write_text(json.dumps({"models": [
                {"id": "one", "arch": "lfm2", "variants": [{"parts": ["/models/one.gguf"]}]},
                {"id": "split", "arch": "qwen35", "variants": [{"parts": ["/m/a-1.gguf", "/m/a-2.gguf"]}]},
                {"id": "v4", "arch": "deepseek4", "variants": [{"parts": ["/m/v4-1.gguf", "/m/v4-2.gguf"]}]}]}))
            self.assertEqual(campaign.model_file(campaign.catalog_entry("one", catalog), "one"), Path("/models/one.gguf"))
            with self.assertRaisesRegex(ValueError, "single GGUF"):
                campaign.model_file(campaign.catalog_entry("split", catalog), "split")
            with self.assertRaisesRegex(ValueError, "not in the catalog"):
                campaign.model_file(campaign.catalog_entry("none", catalog), "none")
            with self.assertRaisesRegex(ValueError, "v4 is refused: deepseek4: DeepSeek V4"):
                campaign.Campaign("v4", directory=Path(directory) / "v4", catalog=catalog)
            # the CLI reports it in one line instead of a traceback
            with mock.patch.object(campaign, "CAMPAIGN_DIR", Path(directory)), \
                    mock.patch.dict(campaign.Campaign.__init__.__kwdefaults__, {"catalog": catalog}), \
                    mock.patch.object(campaign.signal, "signal"):
                with self.assertRaisesRegex(SystemExit, "v4: v4 is refused"):
                    campaign.main(["plan", "--name", "v4"])

    def test_a_hub_name_other_than_the_catalog_id(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.json"
            catalog.write_text(json.dumps({"models": [
                {"id": "one-chat", "arch": "deepseek2", "smoke": {"prefill_tps": 1600.0},
                 "variants": [{"parts": ["/models/one.gguf"]}]}]}))
            camp = campaign.Campaign("one", directory=Path(directory) / "one", catalog=catalog, catalog_id="one-chat")
            self.assertEqual((camp.gguf, camp.state["catalog"]["id"]), (Path("/models/one.gguf"), "one-chat"))
            with self.assertRaisesRegex(ValueError, "none is not in the catalog"):
                campaign.Campaign("one", directory=Path(directory) / "x", catalog=catalog, catalog_id="none")
            camp.save()
            again = campaign.Campaign("one", directory=Path(directory) / "one", catalog=None)   # the state remembers it
            self.assertEqual(again.state["catalog"]["id"], "one-chat")
            with self.assertRaisesRegex(ValueError, "started with catalog entry one-chat, not other"):
                campaign.Campaign("one", directory=Path(directory) / "one", catalog=catalog, catalog_id="other")

    def test_dflash_drafts_and_their_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            draft = draft_gguf(Path(directory) / "draft.gguf", 4096)
            missing = Path(directory) / "missing.gguf"
            entry = {"companions": [{"kind": "dflash", "paths": [str(draft), str(missing)]},
                                    {"kind": "mmproj", "paths": [str(draft)]}]}
            self.assertEqual(campaign.dflash_drafts(entry), [{"path": str(draft), "window": 4096},
                                                             {"path": str(missing), "window": None}])
        self.assertEqual(campaign.dflash_drafts(None), [])
        self.assertEqual(campaign.dflash_drafts({"companions": None}), [])


class ParityTest(unittest.TestCase):
    def report(self, cases):
        counts = {"cases": len(cases), "matched": 0, "server_errors": 0, "both_errors": 0, "mismatch": 0,
                  "chat_py_errors": 0}
        for case in cases:
            counts[case.pop("outcome")] += 1
        return {"counts": counts, "cases": cases}

    def test_every_whole_conversation_and_every_prompt_must_match(self):
        whole = self.report([{"options": {}, "outcome": "matched"}, {"options": {}, "outcome": "matched"}])
        prompts = self.report([{"options": None, "outcome": "matched"},
                               {"options": {"reasoning_effort": "low"}, "outcome": "matched"}])
        self.assertEqual(campaign.parity_problems(prompts, whole), [])
        broken = self.report([{"options": {}, "outcome": "matched"},
                              {"options": {}, "outcome": "server_errors", "server_error": "HTTP 500"}])
        self.assertIn("whole conversations: 1 of 2", campaign.parity_problems(prompts, broken)[0])
        self.assertTrue(campaign.parity_problems(prompts, self.report([])))            # nothing checked: no parity

    def test_a_twin_that_refuses_everything_fails(self):
        whole = self.report([{"options": {}, "outcome": "matched"}])
        refused = self.report([{"options": None, "outcome": "server_errors", "server_error": "HTTP 500"},
                               {"options": {"reasoning_effort": "none"}, "outcome": "server_errors",
                                "server_error": "HTTP 400"}])
        problems = campaign.parity_problems(refused, whole)
        self.assertIn("prompts: no case matches", problems)
        self.assertTrue(any("not tolerated" in item for item in problems))

    def test_refusals_are_accepted_only_for_named_option_sets(self):
        whole = self.report([{"options": {}, "outcome": "matched"}])
        prompts = self.report([{"options": None, "outcome": "matched"},
                               {"options": {"reasoning_effort": "none"}, "outcome": "server_errors",
                                "server_error": "HTTP 400"}])
        label = campaign.option_label({"reasoning_effort": "none"})
        self.assertEqual(label, '{"reasoning_effort":"none"}')
        self.assertEqual(campaign.option_label(None), "default")
        self.assertTrue(campaign.parity_problems(prompts, whole))
        self.assertEqual(campaign.parity_problems(prompts, whole, tolerate=[label]), [])
        mismatch = self.report([{"options": None, "outcome": "matched"}, {"options": {}, "outcome": "mismatch"}])
        self.assertIn("1 mismatches", campaign.parity_problems(mismatch, whole, tolerate=["default"])[0])

    def test_a_record_that_differs_from_the_server_is_explained(self):
        template = ("{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}"
                    "{% if preserve_thinking %}[kept]{% endif %}")
        conversation = {"id": "c1", "messages": [{"role": "user", "content": "hi"},
                                                 {"role": "assistant", "content": "yo"}]}
        record = "<user>hi</user><assistant>yo</assistant>"
        server = record + "[kept]"                       # llama-server's context keeps earlier reasoning
        whole = {"cases": [{"conversation": "c1", "context": "server",
                            "prompt_sha256": hashlib.sha256(server.encode()).hexdigest()}],
                 "bos_token": "", "eos_token": "", "add_bos": False, "add_eos": False}
        found = campaign.record_differences([{"conversation_id": "c1", "text": record}], [conversation], whole,
                                            template)
        self.assertEqual(len(found), 1)
        item = found[0]
        self.assertEqual((item["context"], item["reproduced"], item["at"]), ("server", True, len(record)))
        self.assertIn("preserve_thinking", item["context_differs"])
        self.assertEqual(item["server"][-6:], "[kept]")
        same = {**whole, "cases": [{"conversation": "c1", "context": "template",
                                    "prompt_sha256": hashlib.sha256(record.encode()).hexdigest()}]}
        self.assertEqual(campaign.record_differences([{"conversation_id": "c1", "text": record}], [conversation], same,
                                                     template), [])
        self.assertIsNone(campaign.text_difference("abc", "abc"))
        self.assertEqual(campaign.text_difference("abc", "abd")["at"], 2)


class LockTest(unittest.TestCase):
    def test_waits_for_memory_and_logs_every_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            path, events, sleeps = Path(directory) / "gpu.lock", [], []
            readings = iter([10 * campaign.GIB, 40 * campaign.GIB])
            with campaign.gpu_lock(10 * campaign.GIB, events.append, path=path, available=lambda: next(readings),
                                   sleep=sleeps.append) as held:
                self.assertEqual(held.free, 40 * campaign.GIB)
                with open(path, "a") as other, self.assertRaises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)        # held for the whole block
            self.assertEqual([e["event"] for e in events], ["memory-wait", "gpu-lock"])
            self.assertEqual(sleeps, [campaign.GPU_POLL_S])
            with open(path, "a") as other:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)            # released afterwards

    def test_lock_is_released_when_the_command_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.lock"
            with self.assertRaises(RuntimeError):
                with campaign.gpu_lock(0, lambda item: None, path=path, available=lambda: 30 * campaign.GIB):
                    raise RuntimeError("probe failed")
            with open(path, "a") as other:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_a_command_given_the_descriptor_keeps_the_lock_after_campaign_lets_go(self):
        """What happens when campaign.py dies while its command runs: the command still holds it."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.lock"
            with campaign.gpu_lock(0, lambda item: None, path=path, available=lambda: 30 * campaign.GIB) as held:
                child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"],
                                         pass_fds=(held.stream.fileno(),), start_new_session=True)
            with open(path, "a") as other:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                child.wait()
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_disk_usage_tells_own_from_linked_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base, out = Path(directory) / "base", Path(directory) / "out"
            base.mkdir()
            out.mkdir()
            (base / "shared.npy").write_bytes(b"x" * 10000)
            os.link(base / "shared.npy", out / "shared.npy")
            (out / "own.npy").write_bytes(b"y" * 20000)
            usage = campaign.disk_usage([out, Path(directory) / "missing"])
            self.assertEqual(usage["files"], 2)
            self.assertGreaterEqual(usage["bytes"], 20000)
            self.assertGreaterEqual(usage["linked_bytes"], 10000)
            self.assertLess(usage["bytes"], 30000)
            self.assertEqual(campaign.file_bytes([out]), 30000)


def gone(pid, wait=5.0):
    """Whether process `pid` has ended (waiting up to `wait` seconds)."""
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


class ProcessTest(unittest.TestCase):
    def test_a_timeout_stops_the_whole_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "pid"
            started = time.monotonic()
            outcome = campaign.run_child(["sh", "-c", f"sleep 30 & echo $! > {pidfile}; sleep 30"], timeout=0.5)
            self.assertTrue(outcome.timed_out)
            self.assertLess(time.monotonic() - started, 20)
            self.assertTrue(gone(int(pidfile.read_text())))                  # the grandchild too

    def test_an_interrupt_stops_the_group_and_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "pid"
            real_sleep, calls = time.sleep, []

            def interrupted(seconds):
                calls.append(seconds)
                if len(calls) == 5:
                    raise KeyboardInterrupt
                real_sleep(seconds)
            with mock.patch.object(campaign.time, "sleep", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    campaign.run_child(["sh", "-c", f"sleep 30 & echo $! > {pidfile}; sleep 30"], poll=0.1)
            self.assertTrue(gone(int(pidfile.read_text())))

    def test_exit_status_and_peak_memory(self):
        outcome = campaign.run_child([sys.executable, "-c", "import sys; x = bytearray(64 << 20); sys.exit(3)"])
        self.assertEqual((outcome.returncode, outcome.timed_out), (3, False))
        self.assertGreater(outcome.max_rss, 64 << 20)
        # the anonymous memory of the whole group, sampled while it runs (a grandchild's included)
        script = "import time; x = bytearray(96 << 20); time.sleep(1.5)"
        outcome = campaign.run_child(["sh", "-c", f"{sys.executable} -c '{script}'; true"], sample=0.2)
        self.assertGreater(outcome.max_anon, 96 << 20)
        self.assertLess(outcome.max_anon, 1 << 30)

    def test_what_a_command_leaves_running_in_its_group_is_stopped_before_its_locks_go(self):
        """live_bridge.py keeps its llama-servers and engine daemon in the command's group under a Watch: anything
        still running there when the command exits (it crashed, it forgot one) would hold GPU memory after the
        locks are released, so run_child ends it."""
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "pid"
            outcome = campaign.run_child(["sh", "-c", f"sleep 60 & echo $! > {pidfile}; exit 0"], poll=0.05)
            self.assertEqual(outcome.returncode, 0)
            self.assertTrue(gone(int(pidfile.read_text())))
            self.assertEqual(campaign.group_members(999_999_999), [])

    def test_gpu_memory_is_the_rise_of_gtt_while_the_command_runs(self):
        """On this APU a GPU job's memory is GTT (no RssAnon shows it): run_child records its rise over the value
        at the start, and the largest anonymous + GPU sum of one sample (what came off MemAvailable)."""
        readings = iter([6 * campaign.GIB] + [10 * campaign.GIB] * 1000)
        outcome = campaign.run_child([sys.executable, "-c", "import time; time.sleep(0.6)"], sample=0.1,
                                     gpu_memory=lambda: next(readings))
        self.assertEqual(outcome.max_gpu, 4 * campaign.GIB)
        self.assertGreaterEqual(outcome.max_total, 4 * campaign.GIB)
        blind = campaign.run_child([sys.executable, "-c", "pass"], gpu_memory=lambda: None)
        self.assertIsNone(blind.max_gpu)
        self.assertEqual(blind.max_total, blind.max_anon)

    def test_exec_runs_one_command_under_the_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(campaign, "GPU_LOCK", Path(directory) / "gpu.lock"), \
                    mock.patch.object(campaign, "mem_available", lambda: 100 * campaign.GIB), \
                    mock.patch.object(campaign.signal, "signal"), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(campaign.main(["exec", "--memory-bytes", "0", "--", sys.executable, "-c",
                                                "import sys; sys.exit(3)"]), 3)
                self.assertEqual(campaign.main(["exec", "--memory-bytes", "0", "--timeout", "0.3", "--", "sleep", "10"]),
                                 124)


class EvaluationTablesTest(unittest.TestCase):
    def rows(self):
        rows = []
        for number, (n_scored, hub, floor) in enumerate(((100, 0.1, 0.5), (300, 0.3, 0.7))):
            rows.append({"document_id": f"{number:064x}", "n_scored": n_scored, "P": 9000, "t0": 6144,
                         "mapped_cells": 3000, "fill_cells": 1000,
                         "hub": {"nll_delta": hub, "kl": hub / 2, "top1": 0.9},
                         "floor": {"nll_delta": floor, "kl": floor / 2, "top1": 0.8},
                         "hub_prefix": {"nll_delta": hub + 0.1, "kl": hub, "top1": 0.85},
                         "floor_prefix": {"nll_delta": floor, "kl": floor, "top1": 0.8}})
        return rows

    def test_token_weighted_conditions_with_intervals(self):
        table = campaign.condition_table({"documents": self.rows()})
        self.assertEqual(set(table), {"hub", "floor", "hub_prefix", "floor_prefix"})
        self.assertAlmostEqual(table["hub"]["nll_delta"], (0.1 * 100 + 0.3 * 300) / 400)
        self.assertAlmostEqual(table["hub"]["kl"], (0.05 * 100 + 0.15 * 300) / 400)
        low, high = table["hub"]["nll_ci95"]
        self.assertTrue(0.1 <= low <= table["hub"]["nll_delta"] <= high <= 0.3)
        self.assertEqual(table["floor"]["top1_ci95"], [0.8, 0.8])

    def test_paired_differences_and_recovered_share(self):
        rows = self.rows()
        pair = campaign.paired(rows, "hub", "floor", "kl")
        self.assertAlmostEqual(pair["diff"], -0.2)                 # every chat 0.2 lower: a tight interval
        self.assertEqual((pair["lower"], pair["documents"]), (2, 2))
        self.assertAlmostEqual(pair["ci95"][0], -0.2)
        self.assertAlmostEqual(pair["ci95"][1], -0.2)
        self.assertEqual(campaign.verdict(pair["ci95"]), "lower")
        self.assertEqual(campaign.verdict([-0.1, 0.2]), "no measurable difference")
        self.assertEqual(campaign.verdict([0.1, 0.2]), "higher")
        self.assertEqual(set(campaign.paired_table(rows)), {"hub-floor", "hub_prefix-floor_prefix", "hub_prefix-hub"})
        # the exact sign-flip test: with 2 chats both lower, 2 of the 4 sign patterns are as extreme
        self.assertEqual(pair["p"], 0.5)
        eight = np.ones(8)
        self.assertEqual(campaign.sign_flip_p(eight, eight), 2 / 256)            # the smallest p 8 chats allow
        self.assertEqual(campaign.sign_flip_p([1.0, 2, 3, -0.5, 1, 1, 1, 1], eight), 4 / 256)
        self.assertLess(campaign.sign_flip_p(np.ones(20), np.ones(20)), 1e-4)     # random patterns past 16 chats
        self.assertIn("(p 0.500)", campaign._diff(pair))
        share = campaign.recovered_share(rows, "hub", "floor")
        self.assertAlmostEqual(share["share"], 1 - 0.125 / 0.325)
        largest = share["largest"]
        self.assertEqual(largest["document_id"], rows[1]["document_id"])
        self.assertAlmostEqual(largest["floor_share"], 0.35 * 300 / (0.25 * 100 + 0.35 * 300))
        self.assertAlmostEqual(largest["share_without"], 1 - 0.05 / 0.25)
        self.assertAlmostEqual(largest["hub_without"], 0.05)

    def test_headline_and_switch_setting(self):
        rows = self.rows()
        head = campaign.headline(rows, prefix=False)
        self.assertEqual((head["condition"], head["floor"], head["verdict"]), ("hub", "floor", "lower"))
        self.assertAlmostEqual(head["recovered_nll"], 1 - 0.25 / 0.65)
        self.assertEqual(campaign.headline(rows, prefix=True)["floor"], "floor_prefix")
        # a floor whose ΔNLL interval reaches 0 gives no ΔNLL share (it would mean nothing)
        rows[0]["floor"]["nll_delta"] = -0.9
        self.assertIsNone(campaign.headline(rows, prefix=False)["recovered_nll"])
        rows = self.rows()
        self.assertEqual(campaign.switch_setting(rows, True), (True, "RESULTS.md section 2 (a Qwen-family target)"))
        prefix, rule = campaign.switch_setting(rows, False)       # hub_prefix higher on every chat: hub
        self.assertFalse(prefix)
        self.assertIn("paired 95% CI above 0", rule)
        rows[0]["hub_prefix"]["kl"] = 0.0                          # one chat each way: the CI spans 0
        prefix, rule = campaign.switch_setting(rows, False)
        self.assertEqual(prefix, campaign.NATIVE_PREFIX_DEFAULT)
        self.assertIn("spans 0", rule)

    def test_pooling_and_the_baseline_comparison(self):
        rows = self.rows()
        again = json.loads(json.dumps(rows[1]))
        extra = {**json.loads(json.dumps(rows[0])), "document_id": "e" * 64}
        pooled, gap = campaign.pool_rows([{"documents": rows}, {"documents": [again, extra]}])
        self.assertEqual([row["document_id"] for row in pooled], [rows[0]["document_id"], rows[1]["document_id"],
                                                                   "e" * 64])
        self.assertEqual(gap, 0.0)
        self.assertIsNone(campaign.pool_rows([{"documents": rows}])[1])
        baseline = json.loads(json.dumps(rows))
        for row in baseline:
            row["hub_prefix"]["kl"] += 0.05                        # the baseline source does worse, same floors
        comparison = campaign.baseline_comparison(rows, baseline)
        self.assertEqual((comparison["documents"], comparison["floor_max_abs_diff"]), (2, 0.0))
        self.assertAlmostEqual(comparison["kl"]["diff"], -0.05)
        self.assertEqual(comparison["kl"]["lower"], 2)
        self.assertGreater(comparison["recovered_kl"], comparison["baseline_recovered_kl"])
        self.assertEqual(comparison["per_chat"][0]["mapped_fraction"], 0.75)
        self.assertEqual(campaign.baseline_comparison(rows, [])["documents"], 0)
        summary = campaign.evaluation_summary({"documents": rows, "source": "qwen38", "target": "m",
                                               "selection": "ids", "left_out": {"x" * 64: {"P": 100}},
                                               "recurrent": "mean", "summary": {}})
        self.assertEqual((summary["documents"], summary["mapped_fraction"], summary["left_out"]), (2, 0.75, ["x" * 64]))


def two_sides_bundle(directory):
    """A format 2 manifest with sides of qwen38, m and other (file sizes in bytes as their names say)."""
    directory.mkdir(parents=True)
    sides = {}
    for name, sizes in (("qwen38", (1, 2, 10, 20)), ("m", (100, 200, 30, 40)), ("other", (5000, 5000, 5000, 5000))):
        for kind, (encoder, decoder) in (("k", sizes[0::2]), ("v", sizes[1::2])):
            for role, size in (("encoder", encoder), ("decoder", decoder)):
                files = {}
                for dtype in ("f32", "f16"):
                    file = f"{name}.{kind}.{role}.{dtype}.npy"
                    (directory / file).write_bytes(b"x" * (size if dtype == "f32" else 7))
                    files[dtype] = {"file": file, "sha256": "s"}
                sides[f"{name}.{kind}.{role}"] = {"files": files}
    (directory / "manifest.json").write_text(json.dumps({
        "format_version": 2, "purpose": "universal_kv_hub", "models": {"qwen38": {}, "m": {}, "other": {}},
        "sides": sides}))


class RunnerTest(unittest.TestCase):
    """run_stages with stand-in stage functions: statuses, failures, resume, redo and cleaning."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.gguf = self.root / "model.gguf"
        self.gguf.write_bytes(b"GGUF")
        self.calls = []
        self.fail = set()

        def stand_in(stage):
            def run(camp, args):
                self.calls.append(stage)
                for name in campaign.OUTPUTS[stage]:
                    path = camp.path(name)
                    if "." in name:
                        path.write_text("{}")
                    else:
                        path.mkdir(exist_ok=True)
                        (path / f"{stage}.out").write_text(stage)
                if stage in self.fail:
                    raise campaign.StageError(f"{stage} failed")
                return {"stage": stage}
            return run
        patcher = mock.patch.dict(campaign.STAGE_FUNCTIONS, {s: stand_in(s) for s in campaign.STAGES})
        patcher.start()
        self.addCleanup(patcher.stop)
        quiet(self)
        self.addCleanup(self.temporary.cleanup)

    def campaign(self, recurrent=True, hybrid=True):
        camp = campaign.Campaign("m", directory=self.root / "m", gguf=self.gguf, catalog=None)
        camp.state.setdefault("hybrid", hybrid)
        camp.state.setdefault("architecture", "qwen35" if recurrent else ("lfm2" if hybrid else "qwen3"))
        camp.state.setdefault("recurrent_mean", recurrent)
        camp.state.setdefault("recurrent_cell", {"mean": recurrent, "reason": "a stand-in cell"})
        return camp

    def args(self, stages=None, redo=None):
        return types.SimpleNamespace(stages=stages, redo=redo)

    def test_command_time_counts_every_attempt_since_a_stage_started_over(self):
        events = [{"event": "run", "redo": []},
                  {"event": "command-done", "stage": "capture", "seconds": 10.0},
                  {"event": "command-done", "stage": "evaluate", "seconds": 600.0},
                  {"event": "stage-failed", "stage": "evaluate"},
                  {"event": "run", "redo": []},                     # resumed: the finished evaluation is kept
                  {"event": "command-done", "stage": "evaluate", "seconds": 900.0},
                  {"event": "command-done", "label": "a command outside the stages", "seconds": 5.0}]
        self.assertEqual(campaign.command_seconds(events, recurrent=True), {"capture": 10.0, "evaluate": 1500.0})
        events += [{"event": "run", "redo": ["onboard"]},           # onboard and every stage after it start over
                   {"event": "command-done", "stage": "onboard", "seconds": 50.0},
                   {"event": "command-done", "stage": "evaluate", "seconds": 30.0}]
        self.assertEqual(campaign.command_seconds(events, recurrent=True),
                         {"capture": 10.0, "onboard": 50.0, "evaluate": 30.0})

    def test_a_failed_stage_resumes_where_it_stopped(self):
        self.fail = {"onboard"}
        with self.assertRaises(campaign.StageError):
            campaign.run_stages(self.campaign(), self.args())
        camp = self.campaign()
        self.assertEqual(camp.statuses()["onboard"], "failed")
        self.assertIn("onboard failed", camp.state["stages"]["onboard"]["error"])
        self.assertEqual(self.calls, ["render", "capture", "onboard"])
        self.fail = set()
        self.calls = []
        ran = campaign.run_stages(camp, self.args())
        self.assertEqual(ran, ["onboard", "tables", "recurrent", "evaluate", "report"])
        self.assertEqual(self.calls, ran)
        state = json.loads((self.root / "m" / "state.json").read_text())
        self.assertTrue(all(state["stages"][s]["status"] == "done" for s in campaign.STAGES))
        self.assertEqual(state["stages"]["render"]["result"], {"stage": "render"})
        self.assertGreater(state["stages"]["capture"]["disk"]["files"], 0)
        self.assertEqual(state["stages"]["capture"]["waited_s"], 0)
        self.assertEqual(campaign.run_stages(self.campaign(), self.args()), [])     # nothing left to do

    def test_a_cell_without_state_skips_the_recurrent_stage_with_its_reason(self):
        camp = self.campaign(recurrent=False)
        campaign.run_stages(camp, self.args())
        self.assertNotIn("recurrent", self.calls)
        self.assertEqual(camp.state["stages"]["recurrent"], {"status": "skipped", "reason": "a stand-in cell"})

    def test_a_stage_that_stops_applying_loses_its_outputs(self):
        camp = self.campaign(recurrent=True)
        campaign.run_stages(camp, self.args())
        self.assertTrue(camp.path("recurrent").is_dir())
        later = self.campaign()
        later.state.update(recurrent_mean=False)
        campaign.run_stages(later, self.args())
        self.assertFalse(later.path("recurrent").exists())
        self.assertEqual(later.statuses()["recurrent"], "skipped")

    def test_interrupted_stage_is_cleaned_and_rerun(self):
        camp = self.campaign()
        campaign.run_stages(camp, self.args(stages="render,capture"))
        camp.state["stages"]["onboard"] = {"status": "running", "pid": 1}
        camp.save()
        (camp.path("hub")).mkdir()
        (camp.path("hub", "partial.npz")).write_text("half")
        (camp.path(".kvpark-probe-x")).mkdir()
        (camp.path("capture", ".kvpark-probe-y")).mkdir()
        self.calls = []
        campaign.run_stages(self.campaign(), self.args(stages="onboard"))
        self.assertEqual(self.calls, ["onboard"])
        self.assertFalse(camp.path("hub", "partial.npz").exists())
        self.assertFalse(camp.path(".kvpark-probe-x").exists())
        self.assertFalse(camp.path("capture", ".kvpark-probe-y").exists())

    def test_redo_marks_dependents_stale_before_running(self):
        camp = self.campaign()
        campaign.run_stages(camp, self.args())
        self.calls = []
        self.fail = {"capture"}
        with self.assertRaises(campaign.StageError):
            campaign.run_stages(self.campaign(), self.args(redo="capture"))
        statuses_now = self.campaign().statuses()
        self.assertEqual(statuses_now["capture"], "failed")
        self.assertEqual([statuses_now[s] for s in ("onboard", "tables", "evaluate", "report")], ["stale"] * 4)
        self.assertEqual(statuses_now["render"], "done")
        self.assertEqual(statuses_now["recurrent"], "done")
        self.fail = set()
        self.calls = []
        campaign.run_stages(self.campaign(), self.args())
        self.assertEqual(self.calls, ["capture", "onboard", "tables", "evaluate", "report"])

    def test_resume_keeps_published_chunks_and_finished_evaluations(self):
        camp = self.campaign()
        for name in ("capture/chunk-00", "eval/a-to-b", "eval/b-to-a", "hub"):
            camp.path(name).mkdir(parents=True)
        (camp.path("eval", "a-to-b", "report.json")).write_text("{}")
        (camp.path("eval", "b-to-a", "scores.log")).write_text("half")
        (camp.path("capture-plan.json")).write_text("{}")
        campaign.clean_outputs(camp, "capture", redo=False)
        self.assertTrue(camp.path("capture", "chunk-00").is_dir())
        self.assertFalse(camp.path("capture-plan.json").exists())
        campaign.clean_outputs(camp, "evaluate", redo=False)
        self.assertTrue(camp.path("eval", "a-to-b", "report.json").is_file())
        self.assertTrue(camp.path("eval", "b-to-a", "scores.log").is_file())      # unfinished: run_evaluation decides
        campaign.clean_outputs(camp, "evaluate", redo=True)
        self.assertFalse(camp.path("eval").exists())
        campaign.clean_outputs(camp, "capture", redo=True)
        self.assertFalse(camp.path("capture").exists())

    def test_the_onboarding_fits_on_the_fit_python_with_the_chosen_encoder(self):
        camp = self.campaign(recurrent=False, hybrid=False)
        base = self.root / "base"
        base.mkdir()
        (base / "manifest.json").write_text(json.dumps({"settings": {"view_dim": 64, "layer_rank": 8, "fit_rows": 100,
                                                                     "hub_dim": 16}, "spans": {"dev": 10}}))
        ran = []

        def run(argv, **options):
            ran.append((list(argv), options))
            if options["label"] == "fit_hub":
                camp.path("hub").mkdir(parents=True)
                camp.path("hub", "evaluation.json").write_text(json.dumps({"k": {}, "v": {}}))
                camp.path("hub.log.jsonl").write_text("\n".join(json.dumps(item) for item in (
                    {"stage": "spans", "spans": {"train": 1}, "documents": 1, "rejected": 0},
                    {"stage": "onboard", "kind": "k", "alpha": 0.1, "dev_r2": {}},
                    {"stage": "fit-backend-done", "device": "gpu"})) + "\n")
            return 1.0
        rocm = self.root / "rocm-python"
        rocm.write_text("")
        spec = FakeSpec(["full"], {"k": (2, 4), "v": (2, 4)}, architecture="qwen3")
        with mock.patch.object(campaign, "BASE_HUB", base), mock.patch.object(campaign, "FIT_PYTHON", rocm), \
                mock.patch.object(campaign, "base_models", lambda: [("qwen38", [self.root / "c"], self.root / "r")]), \
                mock.patch.object(campaign, "hub_bytes", lambda directory: 0), \
                mock.patch.object(campaign, "own_files", lambda directory, name: {}), \
                mock.patch.object(camp, "spec", lambda: spec), mock.patch.object(camp, "run", run), \
                mock.patch.dict(os.environ, {"KVPARK_FIT_PYTHON": ""}):
            result = campaign.stage_onboard(camp, self.args())
        argv = ran[0][0]
        self.assertEqual(argv[0], rocm)
        import hub
        self.assertEqual(campaign.ONBOARD_ENCODER, hub.DEFAULT_ONBOARD_ENCODER)
        self.assertEqual(argv[argv.index("--onboard-encoder") + 1], campaign.ONBOARD_ENCODER)
        self.assertEqual(argv[argv.index("--fit-device") + 1], "auto")
        self.assertTrue(ran[0][1]["cpu"])
        self.assertEqual((result["encoder"], result["fit_python"], result["fit_device"]),
                         (campaign.ONBOARD_ENCODER, str(rocm), "gpu"))
        with mock.patch.object(campaign, "FIT_PYTHON", self.root / "missing"):
            self.assertEqual(campaign.fit_python({}), campaign.PYTHON)               # no torch venv: the CPU
            self.assertEqual(campaign.fit_python({"KVPARK_FIT_PYTHON": "/x/python"}), Path("/x/python"))

    def test_reencode_swaps_the_bundles_and_redoes_only_the_runs_out_of_the_model(self):
        camp = self.campaign(recurrent=False, hybrid=False)
        for stage in campaign.STAGES:
            camp.stage(stage)["status"] = "skipped" if stage == "recurrent" else "done"
        camp.stage("onboard")["result"] = {"encoder": "ridge", "alphas": {"k": {"alpha": 0.1}}, "spans": {}}
        for name in ("hub", "hub-v2"):
            camp.path(name).mkdir(parents=True)
            (camp.path(name) / "old").write_text("old")
        camp.path("hub.log.jsonl").write_text("{}\n")
        for run in ("qwen38-to-m", "m-to-qwen38", "m-to-qwen38-server"):
            camp.path("eval", run).mkdir(parents=True)
            (camp.path("eval", run) / "report.json").write_text("{}")
            camp.path("eval", f"{run}.command.json").write_text("{}")
        camp.save()
        ran = []

        def run(argv, **options):
            ran.append((list(argv), options))
            output = Path(argv[-1]) if options["label"].startswith("fit_hub") else Path(argv[3])
            output.mkdir()
            if options["label"].startswith("fit_hub"):
                (output / "evaluation.json").write_text(json.dumps({"k": {}, "v": {}}))
                Path(str(output) + ".log.jsonl").write_text("\n".join(json.dumps(item) for item in (
                    {"stage": "spans", "spans": {"train": 1}, "documents": 1, "rejected": 0},
                    {"stage": "reencode", "kind": "k"}, {"stage": "fit-backend-done", "device": "cpu"})) + "\n")
            return 2.0
        with mock.patch.object(campaign, "base_models", lambda: [("qwen38", [self.root / "c"], self.root / "r")]), \
                mock.patch.object(campaign, "hub_bytes", lambda directory: 0), \
                mock.patch.object(campaign, "own_files", lambda directory, name: {"new": directory.name}), \
                mock.patch.object(camp, "run", run):
            result = campaign.reencode(camp, "consensus", stamp="t0")
            with self.assertRaisesRegex(SystemExit, "already consensus"):
                campaign.reencode(camp, "consensus")
        argv, options = ran[0]
        self.assertEqual(argv[argv.index("--base") + 1], camp.path("hub"))
        self.assertIn("--reencode", argv)
        self.assertEqual(argv[argv.index("--onboard-encoder") + 1], "consensus")
        self.assertTrue(options["cpu"])
        self.assertEqual(options["stage"], "onboard")
        history = camp.path("history", "onboard-ridge-t0")
        self.assertEqual((history / "hub" / "old").read_text(), "old")                 # the previous bundles kept
        self.assertTrue((history / "eval" / "m-to-qwen38" / "report.json").is_file())
        self.assertTrue((history / "eval" / "m-to-qwen38-server.command.json").is_file())
        self.assertTrue(camp.path("eval", "qwen38-to-m", "report.json").is_file())   # into the model: kept
        self.assertFalse(camp.path("eval", "m-to-qwen38").exists())
        self.assertTrue(camp.path("hub").is_dir() and not camp.path("hub.reencode").exists())
        self.assertEqual((result["encoder"], result["alphas"]), ("consensus", {"k": {"alpha": 0.1}}))
        statuses = camp.statuses()
        self.assertEqual((statuses["onboard"], statuses["tables"], statuses["evaluate"], statuses["report"]),
                         ("done", "done", campaign.OUTDATED, campaign.OUTDATED))
        self.assertEqual(campaign.plan_run(statuses, recurrent=False), ["evaluate", "report"])
        campaign.clean_outputs(camp, "evaluate", redo=False)                            # how run_stages cleans it
        self.assertTrue(camp.path("eval", "qwen38-to-m", "report.json").is_file())
        # a reverse encoder fitted before the variance cap (none recorded) may be fitted again with it, once
        camp.stage("onboard")["result"] = {**camp.stage("onboard")["result"], "encoder": "reverse", "variance_cap": None}
        with mock.patch.object(campaign, "base_models", lambda: [("qwen38", [self.root / "c"], self.root / "r")]), \
                mock.patch.object(campaign, "hub_bytes", lambda directory: 0), \
                mock.patch.object(campaign, "own_files", lambda directory, name: {"new": directory.name}), \
                mock.patch.object(camp, "run", run):
            campaign.reencode(camp, "reverse", stamp="t1")
            self.assertIn("--reencode", ran[-2][0])
            camp.stage("onboard")["result"]["variance_cap"] = __import__("hub").ONBOARD_VARIANCE_CAP
            with self.assertRaisesRegex(SystemExit, "already reverse"):
                campaign.reencode(camp, "reverse")
        camp.stage("onboard")["status"] = "failed"
        with self.assertRaisesRegex(SystemExit, "is not onboarded"):
            campaign.reencode(camp, "match")

    def test_an_evaluation_asks_for_the_larger_model_and_the_sides_it_reads(self):
        camp = self.campaign(recurrent=False, hybrid=False)
        reference = self.root / "reference.gguf"
        reference.write_bytes(b"G" * 1000)
        two_sides_bundle(camp.path("hub-v2"))
        asked = {}

        def evaluation(camp_, label, argv, output, memory_bytes):
            asked[label] = memory_bytes
            Path(output).mkdir(parents=True)
            (Path(output) / "report.json").write_text(json.dumps({"documents": [], "selection": "ids"}))
        with mock.patch.object(campaign, "REFERENCE_GGUF", reference), \
                mock.patch.object(campaign, "run_evaluation", evaluation), \
                mock.patch.object(campaign, "run_baseline", lambda camp_: self.root / "baseline"):
            campaign.stage_evaluate(camp, self.args())
        # qwen38's encoder sides (1 + 2) and m's decoder sides (30 + 40); m's encoders (100 + 200) and qwen38's
        # decoders (10 + 20), the float16 copies and the other model are not read; and the probe's buffers and
        # pair_eval's states on top (EVALUATION_EXTRA)
        extra = campaign.EVALUATION_EXTRA
        self.assertEqual(asked, {"qwen38-to-m": 1000 + 73 + extra, "m-to-qwen38": 1000 + 300 + 30 + extra})

    def test_a_model_whose_server_renders_otherwise_is_evaluated_both_ways(self):
        camp = self.campaign(recurrent=False, hybrid=False)
        reference = self.root / "reference.gguf"
        reference.write_bytes(b"G" * 1000)
        two_sides_bundle(camp.path("hub-v2"))
        differing = [{"conversation": "c1", "context": "server", "reproduced": True}]
        camp.state["stages"]["render"] = {"status": "done", "result": {"parity": {"records_differing": differing}}}
        commands = {}

        def evaluation(camp_, label, argv, output, memory_bytes):
            commands[label] = argv
            Path(output).mkdir(parents=True)
            left_out = {"x" * 64: {"P": 3000}} if label == "qwen38-to-m" else {}
            (Path(output) / "report.json").write_text(json.dumps({"documents": [], "left_out": left_out}))
        with mock.patch.object(campaign, "REFERENCE_GGUF", reference), \
                mock.patch.object(campaign, "run_evaluation", evaluation), \
                mock.patch.object(campaign, "run_baseline", lambda camp_: self.root / "baseline"):
            result = campaign.stage_evaluate(camp, self.args())
        # the templates' rendering first (its longer set only where the fixed set renders short), then the servers'
        self.assertEqual(list(commands), ["qwen38-to-m", "qwen38-to-m-long", "m-to-qwen38", "qwen38-to-m-server",
                                          "m-to-qwen38-server"])
        self.assertEqual([label for label, argv in commands.items() if "--server-render" in argv],
                         ["qwen38-to-m-server", "m-to-qwen38-server"])
        self.assertTrue(result["server_rendering"])
        # a record the server context does not reproduce says nothing about how the server renders
        self.assertFalse(campaign.server_renders_differently(
            {"parity": {"records_differing": [{"context": "server", "reproduced": False}]}}))
        self.assertFalse(campaign.server_renders_differently({"parity": {"records_differing": []}}))
        self.assertFalse(campaign.server_renders_differently(None))

    def test_an_evaluation_is_kept_only_for_the_same_command(self):
        camp = self.campaign()
        output = camp.path("eval", "a-to-b")
        argv = ["python", "pair_eval.py", "--output", str(output)]
        self.assertFalse(campaign.kept_run(output, argv))
        output.mkdir(parents=True)
        (output / "report.json").write_text("{}")
        self.assertFalse(campaign.kept_run(output, argv))                         # no command record
        campaign.write_json(output.with_name("a-to-b.command.json"), {"argv": argv,
                                                                      "sha256": campaign.command_digest(argv)})
        self.assertTrue(campaign.kept_run(output, argv))
        self.assertFalse(campaign.kept_run(output, argv + ["--recurrent", "zero"]))
        self.assertTrue(campaign.kept_run(output, argv + ["--resume"]))     # a flag that changes no result

    def test_an_unfinished_evaluation_resumes_only_for_the_same_command(self):
        camp = self.campaign()
        output = camp.path("eval", "a-to-b")
        argv = ["python", "pair_eval.py", "--output", str(output), "--resume"]
        self.assertEqual(campaign.eval_argv(("a", "a.gguf"), ("b", "b.gguf"), output, hub="h", tables="t",
                                            recurrent=[])[-1], "--resume")
        ran = []

        def run(argv_, **options):
            ran.append(sorted(p.name for p in output.iterdir()) if output.exists() else None)
            return 1.0
        output.mkdir(parents=True)
        (output / "states-t").mkdir()                                          # a run the guard stopped
        campaign.write_json(output.with_name("a-to-b.command.json"), {"argv": argv,
                                                                      "sha256": campaign.command_digest(argv)})
        with mock.patch.object(camp, "run", run):
            campaign.run_evaluation(camp, "a-to-b", argv, output, 0)
            self.assertEqual(ran[-1], ["states-t"])                            # kept for pair_eval --resume
            campaign.run_evaluation(camp, "a-to-b", argv + ["--server-render"], output, 0)
            self.assertIsNone(ran[-1])                                         # another command: removed

    def test_commands_are_logged_runnable_and_failures_raise(self):
        camp = self.campaign()
        camp.save()
        script = self.root / "hello.py"
        script.write_text("print('hello')\n")
        seconds = camp.run([sys.executable, script], stage="tables", label="hello", cwd=self.root)
        self.assertGreaterEqual(seconds, 0)
        self.assertIn("hello", (camp.path("logs", "hello.log")).read_text())
        self.assertIn(f"{sys.executable} {script}", camp.path("commands.sh").read_text())
        entry = camp.state["stages"]["tables"]["commands"][0]
        self.assertEqual((entry["label"], entry["returncode"], entry["gpu_lock"]), ("hello", 0, False))
        self.assertEqual(entry["scripts"], {str(script): hashlib.sha256(script.read_bytes()).hexdigest()})
        self.assertIn("head", entry["git"])
        with self.assertRaisesRegex(campaign.StageError, "exited with 3"):
            camp.run([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], stage="tables", label="boom",
                     cwd=self.root)
        with self.assertRaises(campaign.CommandTimeout):
            camp.run(["sleep", "10"], stage="tables", label="slow", cwd=self.root, timeout=0.3)

    def test_a_gpu_evaluation_runs_with_the_gpu_evaluation_guard_defaults(self):
        """pair_eval runs (evaluate stage, baseline) stop at 85 C, continue at 65 C, 4 threads, unless
        KVPARK_GUARD_* say otherwise; commands.sh reproduces them (exec --gpu-evaluation)."""
        camp = self.campaign()
        camp.save()
        script = "import os; print('THREADS', os.environ['OPENBLAS_NUM_THREADS'])"
        unset = {"KVPARK_GUARD_PAUSE_C": "", "KVPARK_GUARD_RESUME_C": "", "KVPARK_GUARD_THREADS": "",
                 "OPENBLAS_NUM_THREADS": "8"}
        with mock.patch.object(campaign, "GPU_LOCK", self.root / "gpu.lock"), \
                mock.patch.object(campaign, "mem_available", lambda: 100 * campaign.GIB), \
                mock.patch.dict(os.environ, unset):
            camp.run([sys.executable, "-c", script], stage="evaluate", label="pair_eval-x", cwd=self.root,
                     memory_bytes=10, cpu=True, gpu_evaluation=True)
            camp.run([sys.executable, "-c", script], stage="tables", label="tables", cwd=self.root, memory_bytes=10)
        self.assertIn("THREADS 4", camp.path("logs", "pair_eval-x.log").read_text())     # capped below the parent's 8
        self.assertIn("THREADS 8", camp.path("logs", "tables.log").read_text())
        commands = camp.path("commands.sh").read_text()
        self.assertIn("campaign.py exec --memory-bytes 10 --cpu-lock --gpu-evaluation --guard-log", commands)
        self.assertEqual(commands.count("--gpu-evaluation"), 1)
        settings = {}
        for label in ("pair_eval-x", "tables"):
            for line in camp.path("logs", f"{label}.guard.jsonl").read_text().splitlines():
                item = json.loads(line)
                if item.get("event") == "watch":
                    settings[label] = item["settings"]
        self.assertEqual((settings["pair_eval-x"]["pause_c"], settings["pair_eval-x"]["resume_c"],
                          settings["pair_eval-x"]["threads"]), (90.0, 88.0, 4))
        self.assertEqual((settings["tables"]["pause_c"], settings["tables"]["threads"]), (90.0, 8))
        ran = []
        camp.path("eval").mkdir()
        with mock.patch.object(camp, "run", lambda argv, **options: ran.append(options) or 1.0):
            campaign.run_evaluation(camp, "a-to-b", ["python", "pair_eval.py"], camp.path("eval", "a-to-b"), 0)
        self.assertTrue(ran[0]["gpu_evaluation"] and ran[0]["cpu"])

    def test_lock_waits_are_not_command_time(self):
        camp = self.campaign()
        camp.save()
        readings = iter([1, 100 * campaign.GIB])
        with mock.patch.object(campaign, "GPU_LOCK", self.root / "gpu.lock"), \
                mock.patch.object(campaign, "GPU_POLL_S", 0.4), \
                mock.patch.object(campaign, "mem_available", lambda: next(readings)):
            seconds = camp.run([sys.executable, "-c", "pass"], stage="capture", label="quick", cwd=self.root,
                               memory_bytes=10)
        entry = camp.state["stages"]["capture"]["commands"][0]
        self.assertGreaterEqual(entry["waited_s"], 0.4)
        self.assertLess(seconds, entry["waited_s"] + 5)
        self.assertEqual(entry["seconds"], round(seconds, 1))
        self.assertIn("campaign.py exec --memory-bytes 10 --", camp.path("commands.sh").read_text())
        events = [json.loads(line)["event"] for line in camp.path("log.jsonl").read_text().splitlines()]
        self.assertIn("memory-wait", events)

    def test_one_run_per_model(self):
        camp = self.campaign()
        with camp.exclusive():
            with self.assertRaisesRegex(SystemExit, "another campaign run"):
                with self.campaign().exclusive():
                    pass
        with self.campaign().exclusive():
            pass

    def test_started_campaign_keeps_its_model(self):
        camp = self.campaign()
        camp.save()
        other = self.root / "other.gguf"
        other.write_bytes(b"GGUF")
        with self.assertRaisesRegex(ValueError, "was started with"):
            campaign.Campaign("m", directory=self.root / "m", gguf=other, catalog=None)


class CaptureStageTest(unittest.TestCase):
    """stage_capture with a stand-in collect_v2 run: chunks, measured speed, size and peak, timeouts,
    adoption of published chunks."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        quiet(self)
        root = Path(self.temporary.name)
        gguf = root / "model.gguf"
        gguf.write_bytes(b"GGUF")
        self.camp = campaign.Campaign("m", directory=root / "m", gguf=gguf, catalog=None)
        self.camp.state.update(architecture="lfm2", hybrid=True, recurrent_mean=False,
                               catalog={"smoke": {"prefill_tps": 1000}})
        render = self.camp.path("render")
        render.mkdir(parents=True)
        docs = [document(1, "dev", "a", 100), document(2, "test", "b", 200)]
        docs += [document(10 + i, "train", "ab"[i % 2], 100 + 50 * (i % 3)) for i in range(10)]
        self.docs = docs
        campaign.write_json(render / "tokens.json", {"documents": docs})
        (render / "records.jsonl").write_text("".join(json.dumps({"document_id": d["document_id"], "text": "x"}) + "\n"
                                                      for d in docs))
        (render / "template.jinja").write_text("{{ x }}")
        self.identity = {"mode": "collect", "runtime_revision": "r1"}
        self.speed, self.load, self.timeout_chunk, self.interrupt_chunk = 10.0, 0.0, None, None
        self.ran, self.devices = [], set()
        spec = FakeSpec(("recurrent", "full"), {"k": (1, 2), "v": (1, 2)}, architecture="lfm2")    # 29 bytes per token
        self.per_token = campaign.bytes_per_token(spec)
        test = self

        def fake_run(camp, argv, *, stage, label, memory_bytes=None, env=None, cwd=None, timeout=None, cpu=False):
            # a capture on the CPU probe (--mode states, CPU_CAPTURE_ARCHITECTURES) takes the CPU lock; a
            # Vulkan capture does not
            on_cpu = "--device" not in argv
            self.assertEqual(cpu, on_cpu)
            self.assertEqual(Path(argv[argv.index("--probe") + 1]), campaign.PROBE_CPU if on_cpu else campaign.PROBE_VULKAN)
            self.assertEqual("--capture-spec" in argv, argv[argv.index("--mode") + 1] == "collect")
            test.devices.add("cpu" if on_cpu else "vulkan")
            self.assertEqual(argv[argv.index("--threads") + 1], 8)  # the guard's thread cap
            docs_path, output = Path(argv[argv.index("--docs") + 1]), Path(argv[argv.index("--output") + 1])
            name = output.name
            if name == test.interrupt_chunk:
                raise KeyboardInterrupt
            if name == test.timeout_chunk:
                output.mkdir()
                raise campaign.CommandTimeout(f"{label} ran past its timeout")
            ids = [json.loads(line)["document_id"] for line in docs_path.read_text().splitlines()]
            tokens = sum(d["tokens"] for d in test.docs if d["document_id"] in ids)
            output.mkdir()
            campaign.write_json(output / "manifest.json", {"rows": tokens, **test.identity,
                                                           "rendered_docs_sha256": campaign.sha256_file(docs_path)})
            camp.stage(stage).setdefault("commands", []).append({"label": label, "waited_s": 2.0, "max_rss_gib": 1.0,
                                                                 "timeout": timeout, "memory_bytes": memory_bytes})
            test.ran.append((name, ids, str(argv[argv.index("--ctx") + 1])))
            return test.load + tokens / test.speed
        patches = [mock.patch.object(campaign.Campaign, "run", fake_run),
                   mock.patch.object(campaign.Campaign, "spec", lambda camp, probe=None: spec),
                   mock.patch.object(campaign, "capture_identity", lambda *a: dict(test.identity)),
                   mock.patch.object(campaign, "chunk_tokens_for", lambda budget: 300)]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def args(self, minutes=60.0, gigabytes=60.0):
        return types.SimpleNamespace(capture_tps=None, budget_gb=gigabytes, budget_minutes=minutes)

    def test_everything_fits_in_chunks(self):
        result = campaign.stage_capture(self.camp, self.args())
        self.assertEqual(result["limit"], "corpus")
        self.assertEqual(result["documents"], 12)
        self.assertEqual(result["tokens"], sum(d["tokens"] for d in self.docs))
        self.assertEqual(self.ran[0][1], [self.docs[0]["document_id"], self.docs[1]["document_id"]])   # held out first
        self.assertTrue(all(sum(d["tokens"] for d in self.docs if d["document_id"] in ids) <= 300
                            for _, ids, _ in self.ran[1:]))
        self.assertEqual(result["captures"], [str(self.camp.path("capture", name)) for name, _, _ in self.ran])
        self.assertEqual(result["waited_s"], 2.0 * len(self.ran))
        self.assertEqual(result["tokens_per_second"], 10.0)                  # lock waits excluded
        self.assertEqual(len(self.camp.path("capture-docs.jsonl").read_text().splitlines()), 12)
        plan = json.loads(self.camp.path("capture-plan.json").read_text())
        self.assertEqual(plan["limit"], "corpus")
        self.assertEqual(self.ran[0][2], "5120")
        self.assertEqual((result["device"], plan["device"], self.devices), ("vulkan", "vulkan", {"vulkan"}))

    def test_gemma4_collects_on_the_cpu_probe(self):
        spec = FakeSpec(("sliding", "full"), {"k": (1, 2), "v": (1, 2)}, architecture="gemma4")
        self.camp.state["architecture"] = "gemma4"
        with mock.patch.object(campaign.Campaign, "spec", lambda camp, probe=None: spec):
            result = campaign.stage_capture(self.camp, self.args())
        plan = json.loads(self.camp.path("capture-plan.json").read_text())
        self.assertEqual((result["mode"], result["device"], plan["device"]), ("collect", "cpu", "cpu"))
        self.assertEqual(result["probe"], str(campaign.PROBE_CPU))
        self.assertEqual(self.devices, {"cpu"})
        self.assertEqual(plan["tokens_per_second"], round(1000 * campaign.CAPTURE_SPEED["collect", "cpu"], 1))
        self.assertEqual(result["documents"], 12)

    def test_the_measured_speed_ends_the_capture_when_time_runs_out(self):
        # the forecast (1000 x 0.4 tokens/s, 120 s of load) says everything fits in 130 s; measured at
        # 10 tokens/s, the held-out documents take 30 s and train chunks follow while time is left:
        # 250, 300, 150 and 300 tokens, then the last three documents no longer fit
        self.speed = 10.0
        result = campaign.stage_capture(self.camp, self.args(minutes=130 / 60))
        self.assertEqual(json.loads(self.camp.path("capture-plan.json").read_text())["limit"], "corpus")
        self.assertEqual(result["limit"], "time")
        self.assertEqual(result["dropped"], 3)
        self.assertEqual(result["probe_seconds"], 130.0)
        self.assertEqual([c["tokens"] for c in result["chunks"]], [300, 250, 300, 150, 300])

    def test_size_budget_counts_a_states_chunk_twice(self):
        self.camp.state["architecture"] = "bailingmoe3"
        spec = FakeSpec(("full",), {"k": (1, 2), "v": (1, 2)}, architecture="bailingmoe3")
        self.identity = {"mode": "states"}
        with mock.patch.object(campaign.Campaign, "spec", lambda camp, probe=None: spec):
            budget = 1200 * self.per_token                     # 1,200 tokens of rows, 300 of them held out
            result = campaign.stage_capture(self.camp, self.args(gigabytes=budget / 10 ** 9))
        self.assertEqual(result["mode"], "states")
        self.assertEqual(result["limit"], "bytes")
        # each chunk was started only with room for its rows and its state files together
        captured = 0
        for _, ids, _ in self.ran:
            tokens = sum(d["tokens"] for d in self.docs if d["document_id"] in ids)
            if captured:
                self.assertLessEqual(captured + 2 * tokens, 1200)
            captured += tokens
        self.assertLessEqual(result["tokens"], 1200)

    def test_a_budget_below_the_held_out_documents_fails_the_stage_with_what_they_take(self):
        # 300 held-out tokens at 29 bytes each and 1000 x 0.4 tokens/s: 8,700 bytes, 120 s of load + 0.75 s
        with self.assertRaisesRegex(campaign.StageError, r"dev and test documents alone .*take 0\.0 GB and 2 minutes"
                                                         r" at 400\.0 tokens/s: give --budget-gb"):
            campaign.stage_capture(self.camp, self.args(gigabytes=200 * self.per_token / 10 ** 9))
        self.assertEqual(self.ran, [])
        # 300 tokens of 10 MB at 1 token/s: 3 GB, (120 + 300) s
        self.assertTrue(campaign.held_out_need(self.docs, bytes_per_token=10 ** 7, tokens_per_second=1).startswith(
            "the held-out documents alone take 3.0 GB and 7 minutes at 1.0 tokens/s"))
        # `plan` shows it instead of a traceback
        args = types.SimpleNamespace(**vars(self.args(gigabytes=200 * self.per_token / 10 ** 9)), stages=None, redo=None)
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            campaign.show_plan(self.camp, args)
        shown = json.loads(stdout.getvalue())
        self.assertNotIn("capture_plan", shown)
        self.assertIn("take 0.0 GB and 2 minutes at 400.0 tokens/s: give --budget-gb", shown["capture_plan_error"])
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            campaign.show_plan(self.camp, types.SimpleNamespace(**vars(self.args()), stages=None, redo=None))
        self.assertEqual(json.loads(stdout.getvalue())["capture_plan"]["limit"], "corpus")

    def test_a_train_chunk_past_the_hard_limit_ends_the_capture(self):
        self.timeout_chunk = "chunk-02"
        result = campaign.stage_capture(self.camp, self.args())
        self.assertEqual(result["limit"], "timeout")
        self.assertEqual([c["name"] for c in result["chunks"]], ["chunk-00", "chunk-01"])
        self.assertFalse(self.camp.path("capture", "chunk-02").exists())
        self.assertGreater(result["dropped"], 0)
        self.timeout_chunk = "chunk-00"
        self.camp.path("capture").rename(self.camp.path("capture-old"))
        with self.assertRaisesRegex(campaign.StageError, "dev and test documents were not captured"):
            campaign.stage_capture(self.camp, self.args())

    def test_an_interrupted_capture_adopts_its_published_chunks(self):
        self.interrupt_chunk = "chunk-02"
        with self.assertRaises(KeyboardInterrupt):
            campaign.stage_capture(self.camp, self.args())
        first = list(self.ran)
        self.ran, self.interrupt_chunk = [], None
        result = campaign.stage_capture(self.camp, self.args())
        self.assertEqual([c["adopted"] for c in result["chunks"][:2]], [True, True])
        self.assertEqual(self.ran[0][0], "chunk-02")
        self.assertEqual(result["documents"], 12)
        self.assertEqual(result["captures"][:2], [str(self.camp.path("capture", name)) for name, _, _ in first])
        # another probe build: every chunk is captured again
        self.identity = {"mode": "collect", "runtime_revision": "r2"}
        self.ran = []
        result = campaign.stage_capture(self.camp, self.args())
        self.assertFalse(any(c["adopted"] for c in result["chunks"]))
        self.assertEqual(self.ran[0][0], "chunk-00")

    def test_adopt_chunk_checks_documents_and_identity(self):
        out = self.camp.path("capture")
        (out / "chunk-00").mkdir(parents=True)
        docs = out / "chunk-00.docs.jsonl"
        docs.write_text('{"document_id": "a"}\n')
        self.assertEqual(campaign.adopt_chunk(out / "chunk-00", docs, {}), (False, "no published capture"))
        manifest = {"rendered_docs_sha256": campaign.sha256_file(docs), "mode": "collect",
                    "settings": {"ctx": 5120, "batch": 512, "device": "vulkan"}}
        campaign.write_json(out / "chunk-00" / "manifest.json", manifest)
        self.assertEqual(campaign.adopt_chunk(out / "chunk-00", docs, {"mode": "collect",
                                                                       "settings": {"ctx": 5120}}), (True, None))
        self.assertEqual(campaign.adopt_chunk(out / "chunk-00", docs, {"mode": "states"}), (False, "another mode"))
        self.assertIn("ctx", campaign.adopt_chunk(out / "chunk-00", docs, {"settings": {"ctx": 6144}})[1])
        docs.write_text('{"document_id": "b"}\n')
        self.assertEqual(campaign.adopt_chunk(out / "chunk-00", docs, {}), (False, "other documents"))


def fake_manifest(directory, files):
    directory.mkdir(parents=True, exist_ok=True)
    campaign.write_json(directory / "manifest.json", {
        "format_version": 1, "purpose": "universal_kv_hub", "models": {"base": {}, "m.2": {}},
        "files": {key: {"file": f"{key}.npz", "sha256": sha} for key, sha in files.items()}})


class AssembledHubsTest(unittest.TestCase):
    """Which assembled hubs still hold a model's onboarding (by sha256), and what assemble refuses."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        quiet(self)
        self.root = Path(self.temporary.name)
        gguf = self.root / "model.gguf"
        gguf.write_bytes(b"GGUF")
        self.camp = campaign.Campaign("m.2", directory=self.root / "campaign" / "m.2", gguf=gguf, catalog=None)
        for bundle in ("hub", "hub-v2"):
            fake_manifest(self.camp.path(bundle), {"base.k": "b0", "m.2.k": "n1", "m.2.v": "n2"})
        files = {name: campaign.own_files(self.camp.path(name), "m.2") for name in ("hub", "hub-v2")}
        self.assertEqual(files["hub"], {"m.2.k.npz": "n1", "m.2.v.npz": "n2"})
        self.camp.state["stages"]["onboard"] = {"status": "done", "result": {"files": files}}
        self.camp.save()

    def test_a_hub_assembled_from_an_older_onboarding_is_flagged(self):
        hub5 = self.root / "hub5"
        fake_manifest(hub5, {"base.k": "b0", "m.2.k": "n1", "m.2.v": "n2"})
        campaign.record_assembly(self.camp, hub5, "hub")
        self.assertEqual(campaign.stale_hubs(self.camp), [])
        self.assertEqual(len(campaign.stale_hubs(self.camp, assume_redo=True)), 1)
        # the model onboarded again: other files
        self.camp.state["stages"]["onboard"]["result"]["files"]["hub"] = {"m.2.k.npz": "x1", "m.2.v.npz": "x2"}
        self.assertEqual(campaign.stale_hubs(self.camp), [(hub5, "built from an older onboarding of this model")])
        self.camp.state["stages"]["onboard"]["status"] = "running"
        self.assertEqual(campaign.stale_hubs(self.camp)[0][1], "the onboarding is running")
        campaign.record_assembly(self.camp, hub5, "hub")                       # recorded once
        self.assertEqual(len(campaign.assembled_hubs(self.camp)), 1)

    def test_assemble_refuses_busy_unfinished_or_changed_members(self):
        def attempt():
            return campaign.assemble(self.root / "hub6", ["m.2"], base=self.root / "base", base_v2=self.root / "base-v2",
                                     campaign_dir=self.root / "campaign")
        with self.camp.exclusive():
            with self.assertRaisesRegex(SystemExit, "another campaign run of m.2"):
                attempt()
        fake_manifest(self.camp.path("hub"), {"base.k": "b0", "m.2.k": "changed", "m.2.v": "n2"})
        with self.assertRaisesRegex(SystemExit, "does not hold the files"):
            attempt()
        self.camp.state["stages"]["onboard"]["status"] = "failed"
        self.camp.save()
        with self.assertRaisesRegex(SystemExit, "is not onboarded"):
            attempt()
        with self.assertRaisesRegex(SystemExit, "has no campaign"):
            campaign.assemble(self.root / "hub6", ["other"], campaign_dir=self.root / "campaign")
        self.assertFalse((self.root / "hub6").exists())


class AssembleEvaluateTest(unittest.TestCase):
    """assemble on real (small) bundles: two models onboarded separately against one base end up in
    one hub, recorded in each campaign, and --evaluate measures the pair between them."""

    def test_assemble_and_evaluate_the_onboarded_pair(self):
        import convert_hub
        import hub
        import test_hub
        quiet(self)
        views, spans, latent = test_hub.make_views(n=1200)
        extra = {"d": test_hub.FakeView("d", latent, [10, 10], 4), "e": test_hub.FakeView("e", latent, [6, 6, 6], 5)}
        spans.rows.update({name: np.arange(1200) for name in extra})
        silent = lambda line: None                                      # noqa: E731
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {kind: hub.fit_hub_kind(views[:2], spans, kind, layer_rank=8, view_dim=12, hub_dim=6, log=silent)
                    for kind in hub.KINDS}
            hub.save_hub(root / "base", base, views[:2], spans, {"note": "base"})
            convert_hub.convert(root / "base", root / "base-v2", log=silent)
            gguf = root / "m.gguf"
            gguf.write_bytes(b"GGUF")
            for name, view in extra.items():
                camp = campaign.Campaign(name, directory=root / "campaign" / name, gguf=gguf, catalog=None)
                hubs, manifest = hub.load_hub(root / "base")
                for kind in hub.KINDS:
                    hub.add_model_kind(hubs[kind], view, views[:2], spans, kind, layer_rank=8, view_dim=12, log=silent)
                hub.save_hub(camp.path("hub"), hubs, [view] + views[:2], spans, {"note": name},
                             base_models=manifest["models"], link_base=root / "base")
                convert_hub.convert(camp.path("hub"), camp.path("hub-v2"), log=silent, reuse=[root / "base-v2"])
                camp.state["stages"] = {
                    "capture": {"status": "done", "result": {"captures": [str(root / "captures" / name)]}},
                    "onboard": {"status": "done", "result": {"files": {
                        bundle: campaign.own_files(camp.path(bundle), name) for bundle in ("hub", "hub-v2")}}}}
                camp.save()
            children = []

            def child(argv, *, settings, guard_log, label, events, environment, **options):
                """assemble --evaluate's watched child, run in this process (the mocks hold) on the CPU backend"""
                children.append({"argv": argv, "settings": settings, "environment": environment, **options})
                argv = [item if item != "auto" else "cpu" for item in argv]
                with mock.patch.dict(os.environ, {campaign.resource_guard.WATCHED: "1"}):
                    code = campaign.main(argv[2:])
                return types.SimpleNamespace(returncode=code, aborted=None, seconds=0.0, paused_s=0.0)
            with mock.patch.object(hub, "CaptureView", lambda name, captures, records: extra[name]), \
                    mock.patch.object(hub, "joint_spans", lambda chosen: spans), \
                    mock.patch.object(campaign, "GPU_LOCK", root / "gpu.lock"), \
                    mock.patch.object(campaign, "CPU_LOCK", root / "cpu.lock"), \
                    mock.patch.object(campaign, "guarded_child", child), \
                    mock.patch.object(campaign, "mem_available", lambda: 10 ** 12), \
                    mock.patch.dict(os.environ, {"KVPARK_GUARD_PAUSE_C": "", "KVPARK_GUARD_RESUME_C": "",
                                                 "KVPARK_GUARD_THREADS": ""}):
                result = campaign.assemble(root / "hub5", ["d", "e"], evaluate=True, base=root / "base",
                                           base_v2=root / "base-v2", campaign_dir=root / "campaign", log=silent)
            # a child process on the fit Python's GPU backend, watched with the GPU-evaluation defaults
            [call] = children
            self.assertEqual(call["argv"][1:4], [str(campaign.HERE / "campaign.py"), "evaluate-onboarded", "--hub"])
            self.assertEqual(call["argv"][0], str(campaign.fit_python()))
            self.assertIn("auto", call["argv"])
            self.assertEqual((call["settings"].pause_c, call["settings"].resume_c, call["settings"].threads),
                             (90.0, 88.0, 4))
            self.assertLessEqual(int(call["environment"]["OPENBLAS_NUM_THREADS"]), 4)
            self.assertIn(campaign.resource_guard.CPU_LOCK_FD, call["environment"])
            self.assertEqual(sorted(result["models"]), ["a", "b", "d", "e"])
            evaluation = json.loads((root / "hub5" / "evaluation-onboarded.json").read_text())
            self.assertEqual(evaluation["pairs"], ["d->e", "e->d"])
            for kind in hub.KINDS:
                self.assertEqual(set(evaluation[kind]), {"d->e", "e->d"})
                self.assertGreater(min(item["r2"] for item in evaluation[kind].values()), 0.9)
            for name in extra:
                camp = campaign.Campaign(name, directory=root / "campaign" / name, catalog=None)
                self.assertEqual([item["directory"] for item in campaign.assembled_hubs(camp)],
                                 [str(root / "hub5"), str(root / "hub5-v2")])
                self.assertEqual(campaign.stale_hubs(camp), [])
            self.assertEqual((root / "base" / "a.k.npz").stat().st_mode & 0o777, 0o444)
            with self.assertRaisesRegex(SystemExit, "exists"):
                campaign.assemble(root / "hub5", ["d"], base=root / "base", base_v2=root / "base-v2",
                                  campaign_dir=root / "campaign")


def pair_rows(target_kl, floor_kl, ids, n_scored=256):
    rows = []
    for number, (hub, floor) in enumerate(zip(target_kl, floor_kl)):
        conditions = {"sanity": (0.0, 0.0), "hub": (hub + 0.01, hub * 1.2), "floor": (floor, floor * 1.5),
                      "hub_prefix": (hub, hub), "floor_prefix": (floor * 0.8, floor), "hub_table_prefix": (hub, hub)}
        rows.append({"document_id": ids[number], "source": "s", "P": 6000 + 1000 * number, "t0": 4096, "H": 20,
                     "n_scored": n_scored, "mapped_cells": 900, "fill_cells": 100,
                     **{name: {"nll_delta": nll, "kl": kl, "top1": 0.9} for name, (nll, kl) in conditions.items()}})
    return rows


class ReportTest(unittest.TestCase):
    """stage_report on a campaign directory holding stand-in stage results and pair_eval reports."""

    def test_the_encoder_diagnostics_and_warnings_reach_the_report(self):
        stats = {"k": {"full": {"min_r2": 0.012, "median_r2": 0.4, "min_ratio": 0.9, "max_ratio": 4.0, "capped": 3},
                       "sub": {"min_r2": 0.2, "median_r2": 0.5, "min_ratio": 1.0, "max_ratio": 2.5, "capped": 0}},
                 "v": {"full": {"min_r2": 0.1, "median_r2": 0.45, "min_ratio": 1.0, "max_ratio": 3.0, "capped": 0}}}
        text = campaign.encoder_stats_markdown(stats, ["the variance cap (4 x the pivots') limited 3 dimension(s)"])
        self.assertIn("K: per-dimension train R² from 0.012", text)
        self.assertIn("3 dimension(s) capped", text)
        self.assertIn("K (layer subset)", text)
        self.assertIn("**Warnings:** the variance cap", text)
        self.assertEqual(campaign.encoder_stats_markdown(None), "")

    def test_report_holds_every_section(self):
        quiet(self)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gguf = root / "m.gguf"
            gguf.write_bytes(b"GGUF")
            camp = campaign.Campaign("m", directory=root / "m", gguf=gguf, catalog=None)
            ids = [f"{n:064x}" for n in range(1, 12)]

            def report(source, target, rows, left_out=None, selection="ids"):
                return {"source": source, "target": target, "recurrent": "zero" if target == "m" else "mean",
                        "recurrent_mean": None, "selection": selection, "left_out": left_out or {}, "documents": rows,
                        "summary": {}}
            runs = {"qwen38-to-m": report("qwen38", "m", pair_rows([0.01] * 6, [0.02] * 6, ids[:6]),
                                          left_out={ids[6]: {"P": 3000}, ids[7]: {"P": 1600}}),
                    "qwen38-to-m-long": report("qwen38", "m", pair_rows([0.01] * 4, [0.02] * 4, ids[3:7]),
                                               selection="corpus"),
                    "m-to-qwen38": report("m", "qwen38", pair_rows([0.06] * 8, [0.07] * 8, ids[:8])),
                    # as the servers render: every fixed chat long enough, more cells mapped
                    "qwen38-to-m-server": report("qwen38", "m", pair_rows([0.01] * 8, [0.03] * 8, ids[:8])),
                    "m-to-qwen38-server": report("m", "qwen38", pair_rows([0.05] * 8, [0.07] * 8, ids[:8]))}
            for row in runs["m-to-qwen38-server"]["documents"]:
                row["mapped_cells"], row["fill_cells"] = 990, 10
            for item in runs.values():
                item["rendering"] = "server" if item is runs["qwen38-to-m-server"] or item is runs["m-to-qwen38-server"] \
                    else "template"
            paths = {}
            for label, item in runs.items():
                (camp.path("eval", label)).mkdir(parents=True)
                campaign.write_json(camp.path("eval", label, "report.json"), item)
                paths[label] = {"report": str(camp.path("eval", label, "report.json"))}
            baseline = report("qwen3-4b", "qwen38", pair_rows([0.065] * 8, [0.07] * 8, ids[:8]))
            campaign.write_json(root / "baseline.json", baseline)
            campaign.write_json(camp.path("capture-plan.json"), {
                "mode": "collect", "bytes_per_token": 32781, "tokens_per_second": 1386.0, "chunk_tokens": 305000,
                "documents": ["a"], "order": [], "held_out": [], "tokens": 181445, "bytes": 5947951545,
                "limit": "corpus", "budget_tokens": 1830330, "train_fraction": 1.0,
                "by_split": {"train": {"documents": 204, "tokens": 154353}}})
            camp.state.update(architecture="lfm2", hybrid=True, recurrent_mean=False,
                              recurrent_cell={"mean": False, "reason": "convolution taps only"},
                              catalog={"companions": [{"kind": "dflash",
                                                       "paths": [str(draft_gguf(root / "draft.gguf", 4096))]}]})
            camp.state["stages"] = {
                "render": {"status": "done", "seconds": 6.0, "disk": {"bytes": 10 ** 6, "linked_bytes": 0},
                           "result": {"documents": 248, "tokens": 181445, "max_tokens": 2955,
                                      "parity": {"build_info": "b1", "prompts": {"matched": 96, "cases": 96,
                                                                                 "server_errors": 0},
                                                 "records": {"matched": 12, "cases": 12, "server_errors": 0},
                                                 "tolerated": [], "records_identical": False, "contexts": ["template"],
                                                 "records_differing": [{"conversation": "c1", "context": "server",
                                                                        "context_differs": {"preserve_thinking": True},
                                                                        "at": 120}]}}},
                "capture": {"status": "done", "seconds": 80.0, "waited_s": 0.0, "max_rss_gib": 3.1,
                            "result": {"rows": 181445, "documents": 248, "bytes": 5947951545, "chunks": [{}, {}],
                                       "limit": "corpus", "dropped": 0, "probe_seconds": 75.0, "waited_s": 0.0,
                                       "tokens_per_second": 2419.3, "speed_share_of_smoke": 0.698}},
                "onboard": {"status": "done", "result": {"pivots": ["qwen38"], "spans": {"train": 1, "dev": 2, "test": 3},
                                                         "documents": 248, "rejected": 0, "encoder": "reverse",
                                                         "r2": {"k": {"m->qwen38": 0.7}, "v": {"m->qwen38": 0.5}}},
                            "reencoded": [{"from": "ridge", "to": "reverse", "at": "2026-10-07T02:10:35-06:00"}]},
                "tables": {"status": "done", "result": {}},
                "recurrent": {"status": "skipped", "reason": "convolution taps only"},
                "evaluate": {"status": "done", "result": {"runs": paths, "baseline": str(root / "baseline.json")}}}
            # the evaluation was interrupted once and resumed: both attempts' commands count
            camp.path("log.jsonl").write_text("".join(json.dumps(item) + "\n" for item in (
                {"event": "run", "redo": []}, {"event": "command-done", "stage": "evaluate", "seconds": 600.0},
                {"event": "run", "redo": []}, {"event": "command-done", "stage": "evaluate", "seconds": 900.0})))
            campaign.stage_report(camp, types.SimpleNamespace())
            data = json.loads(camp.path("report.json").read_text())
            text = camp.path("report.md").read_text()
        self.assertIn("Encoder `reverse`; fitted again from `ridge` on 2026-10-07", text)
        self.assertEqual(data["onboard"]["reencoded"][0]["from"], "ridge")
        into = data["directions"]["qwen38-to-m"]
        self.assertEqual((into["runs"], into["documents"], into["duplicate_max_abs_diff"]),
                         (["qwen38-to-m", "qwen38-to-m-long"], 7, 0.0))
        self.assertEqual(into["verdict"], "lower")
        self.assertTrue(into["native_prefix"])                    # hub_prefix lower on every chat
        self.assertIsNone(data["bridge"]["recurrent_mean"])
        self.assertEqual(data["bridge"]["draft_window"], 4096)
        server = data["directions"]["m-to-qwen38-server"]
        self.assertEqual((server["rendering"], server["runs"], server["documents"]),
                         ("server", ["m-to-qwen38-server"], 8))
        self.assertEqual(server["against_template"][0]["mapped_fraction"], [0.9, 0.99])
        self.assertEqual(data["bridge"]["native_prefix_from"], "qwen38-to-m-server")
        self.assertEqual(data["baseline"]["comparison"]["documents"], 8)          # against the templates' rendering
        self.assertEqual(data["baseline"]["comparison"]["floor_max_abs_diff"], 0.0)
        self.assertAlmostEqual(data["baseline"]["comparison"]["kl"]["diff"], -0.005)
        for needle in ("# Campaign: m", "| render | done | 0.1 min |", "96/96", "`c1` differs at character 120",
                       "preserve_thinking", "limit `corpus`", "5.95 GB", "| m->qwen38 | 0.700 | 0.500 |",
                       "### qwen38 → m", "2 of the set render too short", "the longer set", "7 distinct chats",
                       "Paired difference", "hub_prefix − floor_prefix", "## m → qwen38 against qwen3-4b → qwen38",
                       "floors are identical", "No recurrent mean (skipped): convolution taps only",
                       "--hub-name m --native-prefix`", "not comparable", "add `--draft-window 4096`",
                       "native tail of at least 4,095 tokens", "the bridge assumes 2,048",
                       "### m → qwen38, as the servers render", "Runs labelled `-server`",
                       "The same chats under the templates' rendering (`m-to-qwen38`)", "| 90% / 99% |",
                       "The native prefix follows `qwen38-to-m-server`", "| evaluate | done | — | 25.0 min |"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
