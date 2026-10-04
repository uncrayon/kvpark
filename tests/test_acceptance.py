"""Checks that model acceptance evidence requires real restoration and reuse."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("acceptance", Path(__file__).parents[1] / "scripts" / "acceptance.py")
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_snapshot_requires_both_published_files_and_matching_metadata(self):
        parked = {"key": "test", "tokens": 2400}
        metadata = {"filename": "test.generation.bin", "tokens": 2400, "format": 2, "bytes": 12}
        path = self.root / "test.meta"
        path.write_text(json.dumps(metadata))
        (self.root / metadata["filename"]).write_bytes(b"snapshot")
        companion = self.root / (metadata["filename"] + ".resume")
        with self.assertRaises(RuntimeError):
            acceptance.snapshot_evidence(self.root, parked)
        companion.write_bytes(b"full")
        result = acceptance.snapshot_evidence(self.root, parked)
        self.assertEqual(result["files"], {".bin": 8, ".bin.resume": 4})
        companion.write_bytes(b"")
        with self.assertRaises(RuntimeError):
            acceptance.snapshot_evidence(self.root, parked)
        companion.write_bytes(b"changed")
        with self.assertRaises(RuntimeError):
            acceptance.snapshot_evidence(self.root, parked)

    def result(self, cached, processed, content="ORCHID-731"):
        return {"cached_tokens": cached, "processed_tokens": processed,
                "message": {"role": "assistant", "content": content}}

    def checks(self, *, restored=None, events=None, cold_tokens=2300, replay=None):
        return acceptance.acceptance_checks(
            self.result(0, cold_tokens), self.result(2305, 22), replay,
            self.result(2305, 22) if restored is None else restored,
            [{"action": "restored", "tokens": 2305}] if events is None else events,
            2305, min_prompt_tokens=2049)

    def test_direct_resume_requires_disk_event_counts_and_matching_continuation(self):
        checks = self.checks()
        self.assertTrue(all(checks.values()))
        self.assertNotIn("replay_matches", checks)
        self.assertFalse(self.checks(events=[])["disk_restore"])
        self.assertFalse(self.checks(events=[{"action": "restored", "tokens": 100}])["disk_restore"])
        self.assertFalse(self.checks(restored=self.result(2305, 22, "wrong"))["continuation_matches"])
        for cached, processed in ((2305, None), (None, 22), (22, 2305), (0, 0), (2305, -1)):
            with self.subTest(cached=cached, processed=processed):
                self.assertFalse(self.checks(restored=self.result(cached, processed))["continuation_reused_majority"])

    def test_window_threshold_uses_measured_tokens_and_replay_is_checked_when_present(self):
        self.assertFalse(self.checks(cold_tokens=2048)["minimum_prompt_tokens"])
        self.assertTrue(self.checks(cold_tokens=2049)["minimum_prompt_tokens"])
        self.assertFalse(self.checks(replay=self.result(0, 2300))["replay_reused_majority"])
        self.assertFalse(self.checks(replay=self.result(2299, 1, "wrong"))["replay_matches"])

    def test_reasoning_only_continuations_require_identical_generated_messages(self):
        cold, warm = self.result(0, 2300, ""), self.result(2305, 22, "")
        cold["message"]["reasoning_content"] = "Initial reasoning"
        warm["message"]["reasoning_content"] = "Continuation reasoning"
        replay = {**self.result(2299, 1), "message": dict(cold["message"])}
        restored = {**self.result(2305, 22), "message": dict(warm["message"])}
        events = [{"action": "restored", "tokens": 2305}]
        checks = acceptance.acceptance_checks(cold, warm, replay, restored, events, 2305)
        self.assertTrue(all(checks.values()))
        replay["message"]["reasoning_content"] = "Different initial reasoning"
        restored["message"]["reasoning_content"] = "Different continuation reasoning"
        checks = acceptance.acceptance_checks(cold, warm, replay, restored, events, 2305)
        self.assertFalse(checks["replay_matches"])
        self.assertFalse(checks["continuation_matches"])

    def test_empty_generation_fails_and_all_message_fields_must_match(self):
        for message in ({"role": "assistant", "content": ""},
                        {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}):
            self.assertFalse(acceptance.generated_messages_match(message, dict(message)))
        message = {"role": "assistant", "content": "answer", "reasoning_content": "reasoning"}
        self.assertFalse(acceptance.generated_messages_match(message, {**message, "reasoning_content": "different"}))
        tool_message = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "test", "arguments": "{}"}}]}
        self.assertTrue(acceptance.generated_messages_match(tool_message, dict(tool_message)))

    def test_custom_runtime_flags_preserve_private_cpu_and_disabled_ram_cache(self):
        args = acceptance.arguments(["--server", "server", "--model", "model.gguf",
                                     "--output", str(self.root / "evidence.json"),
                                     "--ctx-size", "6144", "--prompt-repetitions", "220",
                                     "--max-tokens", "12", "--threads", "2", "--direct-continuation",
                                     "--min-prompt-tokens", "2049", "--batch-size", "128", "--ubatch-size", "64",
                                     "--cache-type-k", "f32", "--cache-type-v", "f32", "--flash-attn", "off",
                                     "--load-mode", "mmap", "--no-repack"])
        options = acceptance.backend_arguments(args)
        for flag, value in (("--device", "none"), ("--n-gpu-layers", "0"), ("--cache-ram", "0"),
                            ("--ctx-size", "6144"), ("--threads-batch", "2"), ("--load-mode", "mmap"),
                            ("--cache-type-v", "f32"), ("--ubatch-size", "64")):
            self.assertEqual(options[options.index(flag) + 1], value)
        self.assertIn("--no-repack", options)
        self.assertTrue(args.direct_continuation)


if __name__ == "__main__":
    unittest.main()
