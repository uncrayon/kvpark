"""check_model_support: helpers, and end-to-end runs on tiny fixtures (KVPARK_KV_PROBE)."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

import chat
import check_model_support as cms
import kvstate
from test_chat import QWEN_LIKE, conversation
from tiny_fixture import create_fixture, create_mla_fixture


PROBE = os.environ.get("KVPARK_KV_PROBE")


def rendered_test_record():
    """A test-split record whose final assistant message starts after more than 16 tokens."""
    conv = conversation("test")
    conv["tools"] = None
    conv["messages"] = [m for m in conv["messages"] if m["role"] != "tool"]
    for message in conv["messages"]:
        message.pop("tool_calls", None)
    conv["messages"][-1]["content"] = "Expect about 12 degrees and clouds, so bring a jacket and an umbrella."
    return chat.rendered_record(conv, "fixture", chat.render(conv, QWEN_LIKE))


class HelperTests(unittest.TestCase):
    def test_score_ranges_and_first_scored_token(self):
        record = {"segments": [[0, 0, 4], [1, 10, 20], [2, 25, 30]], "final_message_segments": [1, 2]}
        ranges = cms.score_ranges(record)
        self.assertEqual(ranges, [[10, 20], [25, 30]])
        # token p is scored when its bytes [byte_end[p-1], byte_end[p]) lie inside a range
        self.assertEqual(cms.first_scored([3, 9, 12, 15, 21], ranges), 3)
        self.assertEqual(cms.first_scored([3, 10, 12], ranges), 2)
        self.assertIsNone(cms.first_scored([3, 9, 21, 31], ranges))

    def test_pick_document_wants_a_test_split_final_message_after_one_batch(self):
        records = [{"document_id": name, "split": split, "segments": [[0, start, start + 5]],
                    "final_message_segments": [0]} for name, split, start in
                   (("a", "test", 4), ("b", "test", 40), ("c", "train", 40), ("d", "test", 60))]
        tokens = {name: {"token_id": np.arange(n), "byte_end": np.arange(1, n + 1)}
                  for name, n in (("a", 10), ("b", 50), ("c", 45), ("d", 70))}
        self.assertEqual(cms.pick_document(records, tokens, 16, 4096), "b")
        self.assertEqual(cms.pick_document(records, tokens, 16, 60), "b")
        with self.assertRaisesRegex(ValueError, "no test document"):
            cms.pick_document(records, tokens, 64, 4096)

    def test_pair_norms_follow_the_rope_pairing(self):
        k = np.zeros((1, 1, 8), dtype=np.float32)
        k[0, 0, :6] = [3, 0, 0, 4, 0, 0]
        k[0, 0, 7] = -2
        neox = kvstate.ModelSpec("qwen3", ("full",), (1,), (8,), (8,), rope_type="neox", n_rot=6)
        # NEOX pairs (i, i + 3) over the first 6 dims: (3, 4) -> 5; dims past n_rot keep |x|
        np.testing.assert_array_equal(cms.pair_norms(neox, 0, k)[0, 0], [5, 0, 0, 5, 0, 0, 0, 2])
        norm = kvstate.ModelSpec("muse-glimmer", ("sliding", "full"), (1, 1), (8, 8), (8, 8), n_swa=4,
                                 rope_type="norm", n_rot=6)
        # NORMAL pairs (2i, 2i + 1): (3, 0) -> 3, (0, 4) -> 4
        np.testing.assert_array_equal(cms.pair_norms(norm, 0, k)[0, 0], [3, 3, 4, 4, 0, 0, 0, 2])


@unittest.skipUnless(PROBE, "set KVPARK_KV_PROBE to run check_model_support on tiny fixtures")
class FixtureRunTests(unittest.TestCase):
    def run_check(self, architecture, *extra, mla=None):
        """summary.json of a run on a fixture (create_mla_fixture with options `mla`, else create_fixture)."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / f"{architecture}.gguf"
            if mla is not None:
                create_mla_fixture(model, architecture, **mla)
            else:
                create_fixture(model, architecture, control_tokens=["<|im_start|>", "<|im_end|>"])
            docs = root / "rendered.jsonl"
            docs.write_text(json.dumps(rendered_test_record(), ensure_ascii=False) + "\n", encoding="utf-8")
            output = root / "check"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cms.main(["--probe", PROBE, "--model", str(model), "--docs", str(docs),
                                           "--output", str(output), "--device", "cpu", "--batch", "16",
                                           "--threads", "2", "--skip-memory-check", *extra]), 0)
            return json.loads((output / "summary.json").read_text(encoding="utf-8"))

    def test_dense_and_hybrid_fixtures_on_the_cpu_are_exact(self):
        for architecture, sections in (("qwen3", ["kv"]), ("qwen35", ["attn", "recurrent"])):
            with self.subTest(architecture=architecture):
                summary = self.run_check(architecture)
                self.assertEqual((summary["architecture"], summary["settings"]["cache_type"]), (architecture, "f16"))
                self.assertTrue(summary["tokenize_matches_collect"] and summary["tokenize_matches_wrapper"])
                state = summary["state"]
                self.assertTrue(state["round_trip_bytes_equal"] and state["tokens_equal"])
                self.assertEqual([s["name"] for s in state["sections"]], sections)
                self.assertEqual(state["worst"]["v_max_abs"], 0)
                self.assertLessEqual(state["worst"]["k_derot_max_ulps"], 1)
                seed = summary["document"]["seed_tokens"]
                self.assertEqual(seed % 16, 0)
                self.assertLess(seed, summary["document"]["first_scored_position"])
                # same device, seed on the --batch grid: identical scores; the pinned CPU RoPE is
                # reproduced bit for bit, so the rebuilt state scores identically as well
                for name in ("seed", "rebuilt"):
                    self.assertTrue(summary["seed_score"][name]["scores_identical_to_native"], name)
                    self.assertEqual(summary["seed_score"][name]["kl_max"], 0)
                    self.assertEqual(summary["seed_score"][name]["restored_pos_max"], seed - 1)

    def test_mla_and_dsa_fixtures_on_the_cpu_are_exact(self):
        # K-only rows, the nextn layer (zero rows on 85d5703a3, none on a7b94df2c), DSA indexer keys
        # (top_k 8: the document is longer, so the rebuilt keys decide which cells attention reads)
        # and DSA-iSWA sections
        from test_kv_probe_integration import probe_supports
        with tempfile.TemporaryDirectory() as directory:
            create_mla_fixture(Path(directory) / "m.gguf", "deepseek2")
            if not probe_supports(Path(directory) / "m.gguf", Path(directory)):
                self.skipTest("this kvpark-kv-probe predates the MLA/DSA table rows; rebuild it")
        for architecture, options, sections in (("deepseek2", {"nextn": 1}, ["kv"]),
                                                ("deepseek32", {}, ["kv", "indexer"]),
                                                ("dots3note", {"layers": 4}, ["base", "indexer", "swa"])):
            with self.subTest(architecture=architecture):
                summary = self.run_check(architecture, mla=options)
                state = summary["state"]
                self.assertTrue(state["round_trip_bytes_equal"] and state["tokens_equal"])
                self.assertEqual([s["name"] for s in state["sections"]], sections)
                self.assertTrue(all(s["k_only"] == s["layers"] for s in state["sections"] if s["kind"] == "kv"))
                self.assertEqual(state["worst"]["v_max_abs"], 0)                 # V is the stored c_kv view
                self.assertLessEqual(state["worst"]["k_derot_max_ulps"], 1)
                self.assertLessEqual(state["worst"].get("ik_derot_max_ulps", 0), 1.5)
                if architecture == "deepseek2":
                    # only 85d5703a3's plain cache keeps the nextn layer (zero rows); a7b94df2c filters it out
                    kept = state["llama_revision"] == "85d5703a3"
                    self.assertEqual(state["layers"].get("2"), {"nextn_zero": True} if kept else None)
                    self.assertEqual(state["sections"][0]["layers"], [0, 1, 2] if kept else [0, 1])
                    self.assertEqual(summary["spec"]["llama_revision"], state["llama_revision"])
                    if kvstate.llama_revision_of(PROBE) is not None:
                        self.assertEqual(state["llama_revision"], kvstate.llama_revision_of(PROBE))
                for name in ("seed", "rebuilt"):
                    self.assertTrue(summary["seed_score"][name]["scores_identical_to_native"], name)

    def test_existing_output_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                cms.main(["--probe", PROBE, "--model", "x.gguf", "--docs", "x.jsonl", "--output", directory])


if __name__ == "__main__":
    unittest.main()
