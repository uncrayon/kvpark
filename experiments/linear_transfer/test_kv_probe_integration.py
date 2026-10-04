"""Real graph checks for kvpark-kv-probe.

CPU checks: KVPARK_KV_PROBE=/path/to/binary. Vulkan checks (tiny fixtures on the GPU):
KVPARK_KV_PROBE_VULKAN=/path/to/a binary built against a GGML_VULKAN=ON runtime. The
MLA/DSA and hybrid captures also run the attention hybrids test-llama-archs writes when
KVPARK_LLAMA_ARCHS=/path/to/test-llama-archs is set (given a byte vocabulary so the probe
can tokenize text for them).
"""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

import kvstate
from captures import Capture
from tiny_fixture import YARN_V2, _gguf_module, copy_with_metadata, create_fixture, create_mla_fixture


PROBE = os.environ.get("KVPARK_KV_PROBE")
PROBE_VULKAN = os.environ.get("KVPARK_KV_PROBE_VULKAN")
LLAMA_ARCHS = os.environ.get("KVPARK_LLAMA_ARCHS")
CONTROL = ["<|im_start|>", "<|im_end|>"]
CONTROL_IDS = {"<|im_start|>": 258, "<|im_end|>": 259}
CHAT = ("<|im_start|>system\nYou answer briefly.<|im_end|>\n"
        "<|im_start|>user\nWhat is a café in 日本? Explain.<|im_end|>\n"
        "<|im_start|>assistant\nA small place that serves coffee, tea and snacks.<|im_end|>\n")
PLAIN = "Second document: tabs\tand\nnewlines; no special tokens at all."


def document_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def expected_tokens(text):
    """Byte-level fixture tokenization with control tokens parsed as single ids."""
    ids, ends, special, offset, rest = [], [], [], 0, text
    while rest:
        control = next((token for token in CONTROL if rest.startswith(token)), None)
        if control:
            ids.append(CONTROL_IDS[control])
            offset += len(control.encode("utf-8"))
            ends.append(offset)
            special.append(1)
            rest = rest[len(control):]
            continue
        for byte in rest[0].encode("utf-8"):
            ids.append(byte)
            offset += 1
            ends.append(offset)
            special.append(0)
        rest = rest[1:]
    return ids, ends, special


def copy_with_byte_vocabulary(source, path):
    """A copy of a test-llama-archs GGUF (tokenizer "no_vocab", n_vocab ids) whose tokenizer is gpt2 with
    one token per byte 0..n_vocab-1 and no merges, so kv_probe can tokenize ASCII text for it; tensors
    and every other key unchanged (as tiny_fixture.copy_with_metadata, with array-valued keys)."""
    gguf = _gguf_module()
    from gguf.vocab import bytes_to_unicode
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    reader = gguf.GGUFReader(source)
    writer = gguf.GGUFWriter(path, reader.get_field(gguf.Keys.General.ARCHITECTURE).contents())
    n_vocab = len(reader.get_field("tokenizer.ggml.tokens").data)
    if n_vocab > 256:
        raise ValueError("a byte vocabulary holds at most 256 tokens")
    symbols = bytes_to_unicode()
    replaced = {"tokenizer.ggml.model": ("gpt2", gguf.GGUFValueType.STRING, None),
                "tokenizer.ggml.pre": ("gpt-2", gguf.GGUFValueType.STRING, None),
                "tokenizer.ggml.tokens": ([symbols[b] for b in range(n_vocab)], gguf.GGUFValueType.ARRAY,
                                          gguf.GGUFValueType.STRING),
                "tokenizer.ggml.token_type": ([int(gguf.TokenType.NORMAL)] * n_vocab, gguf.GGUFValueType.ARRAY,
                                              gguf.GGUFValueType.INT32),
                "tokenizer.ggml.merges": (["<unused-left> <unused-right>"], gguf.GGUFValueType.ARRAY,
                                          gguf.GGUFValueType.STRING)}
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF.") or field.name in replaced:
            continue
        value_type = field.types[0]
        if value_type == gguf.GGUFValueType.ARRAY and not field.data:
            continue                    # gguf-py cannot write empty arrays
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)
    for name, (value, value_type, sub_type) in replaced.items():
        writer.add_key_value(name, value, value_type, sub_type=sub_type)
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes,
                               tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    return path


def probe_supports(model, workdir):
    """Whether PROBE knows the model's architecture (binaries built before a table row was added refuse it)."""
    docs = Path(tempfile.mkdtemp(dir=workdir)) / "docs.jsonl"
    docs.write_text(json.dumps({"document_id": document_id("probe"), "text": "probe"}) + "\n", encoding="utf-8")
    result = subprocess.run([PROBE, "--mode", "tokenize", "--model", str(model), "--input", str(docs), "--output",
                             str(docs.with_name("out"))], capture_output=True, text=True, timeout=120)
    return "kv probe supports only" not in result.stderr


def rms_norm(value, eps=1e-6):
    return value / np.sqrt(np.mean(value ** 2, axis=-1, keepdims=True) + eps)


@unittest.skipUnless(PROBE, "set KVPARK_KV_PROBE to enable real llama.cpp kv-probe checks")
class KvProbeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-kv-probe-")
        cls.root = Path(cls.temp.name)
        cls.models, cls.weights = {}, {}
        for architecture in ("muse-glimmer", "qwen35", "qwen3"):
            path = cls.root / f"{architecture}.gguf"
            cls.weights[architecture] = create_fixture(path, architecture, control_tokens=CONTROL)
            cls.models[architecture] = path
        cls.fused = {}
        for architecture in ("muse-glimmer", "qwen3"):
            cls.fused[architecture] = cls.root / f"{architecture}-fused.gguf"
            create_fixture(cls.fused[architecture], architecture, fused_qkv=True, control_tokens=CONTROL)
        # probes older than --device (kv-probe-build-s1) stay usable for the rest of this class
        usage = subprocess.run([PROBE, "--help"], capture_output=True, text=True, timeout=60).stdout
        cls.has_device_option = "--device" in usage

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def workdir(self):
        return Path(tempfile.mkdtemp(dir=self.root))

    def write_docs(self, directory, records):
        path = directory / "docs.jsonl"
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                        encoding="utf-8")
        return path

    def run_probe(self, mode, model, docs, output, *extra, ctx=1024):
        command = [PROBE, "--mode", mode, "--model", str(model), "--input", str(docs),
                   "--output", str(output), "--ctx", str(ctx), "--threads", "2", *map(str, extra)]
        return subprocess.run(command, capture_output=True, text=True, timeout=120)

    def assert_fails(self, result, output, fragment):
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn(fragment, result.stderr.lower())
        self.assertFalse(output.exists())
        self.assertEqual([p for p in output.parent.iterdir() if ".partial-" in p.name], [])

    def collect(self, architecture, texts, *, batch=128, spec=None, fused=False):
        directory = self.workdir()
        docs = self.write_docs(directory, [{"document_id": document_id(t), "text": t} for t in texts])
        extra = ["--batch", batch]
        if spec is not None:
            (directory / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
            extra += ["--capture-spec", directory / "spec.json"]
        model = self.fused[architecture] if fused else self.models[architecture]
        output = directory / "capture"
        result = self.run_probe("collect", model, docs, output, *extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual((report["format_version"], report["mode"]), (2, "collect"))
        self.assertEqual(report["settings"]["cache_type"], "f32")
        rows = report["rows"]
        self.assertEqual(rows, sum(len(expected_tokens(t)[0]) for t in texts))
        arrays = {}
        for capture in report["captures"]:
            value = np.load(output / capture["file"], mmap_mode="r")
            self.assertEqual(value.dtype, np.dtype("<f4"))
            self.assertEqual(value.shape, (rows, capture["heads"], capture["dim"]))
            self.assertTrue(np.isfinite(value).all())
            arrays[(capture["key"], capture["layer"])] = np.array(value)
        return report, arrays, output

    # -- tokenize ---------------------------------------------------------

    def test_tokenize_parses_control_tokens_and_exact_byte_ends(self):
        texts = [CHAT, PLAIN]
        for architecture, attention in (("muse-glimmer", ["sliding"] * 3 + ["full"]),
                                        ("qwen35", ["recurrent"] * 3 + ["full"]), ("qwen3", ["full"] * 4)):
            with self.subTest(architecture=architecture):
                directory = self.workdir()
                docs = self.write_docs(directory, [{"document_id": document_id(t), "text": t, "extra": 1}
                                                   for t in texts])
                output = directory / "tokens"
                result = self.run_probe("tokenize", self.models[architecture], docs, output)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads((output / "report.json").read_text(encoding="utf-8"))
                self.assertEqual((report["format_version"], report["mode"]), (2, "tokenize"))
                self.assertEqual(report["architecture"], architecture)
                self.assertEqual([layer["attention"] for layer in report["layers"]], attention)
                self.assertNotIn("captures", report)
                token_id = np.load(output / "token_id.npy")
                byte_end = np.load(output / "byte_end.npy")
                special = np.load(output / "special.npy")
                self.assertEqual((token_id.dtype, byte_end.dtype, special.dtype),
                                 (np.dtype("<i4"), np.dtype("<i8"), np.dtype("u1")))
                first = 0
                for text, document in zip(texts, report["documents"], strict=True):
                    ids, ends, flags = expected_tokens(text)
                    self.assertEqual(document, {"document_id": document_id(text), "n_tokens": len(ids),
                                                "first_row": first})
                    rows = slice(first, first + len(ids))
                    self.assertEqual(token_id[rows].tolist(), ids)
                    self.assertEqual(byte_end[rows].tolist(), ends)
                    self.assertEqual(special[rows].tolist(), flags)
                    first += len(ids)
                self.assertEqual(report["rows"], first)
                self.assertEqual(int(special.sum()), 6)

    def test_tokenize_rejects_context_overflow_and_invalid_documents(self):
        directory = self.workdir()
        docs = self.write_docs(directory, [{"document_id": document_id(PLAIN), "text": PLAIN}])
        output = directory / "tokens"
        result = self.run_probe("tokenize", self.models["muse-glimmer"], docs, output, "--batch", 16, ctx=16)
        self.assert_fails(result, output, "exceeding --ctx")
        for records in ([{"document_id": document_id("a"), "text": ""}],
                        [{"document_id": "ABC", "text": "a"}],
                        [{"document_id": document_id("a"), "text": "a"}] * 2):
            directory = self.workdir()
            output = directory / "tokens"
            result = self.run_probe("tokenize", self.models["muse-glimmer"],
                                    self.write_docs(directory, records), output)
            self.assert_fails(result, output, "line")
        directory = self.workdir()
        docs = directory / "docs.jsonl"
        docs.write_text('{"document_id": "%s", "text": "a", "text": "b"}\n' % document_id("b"), encoding="utf-8")
        output = directory / "tokens"
        result = self.run_probe("tokenize", self.models["muse-glimmer"], docs, output)
        self.assert_fails(result, output, 'duplicate json key "text"')

    def test_command_line_rejects_unknown_and_repeated_options(self):
        directory = self.workdir()
        docs = self.write_docs(directory, [{"document_id": document_id(PLAIN), "text": PLAIN}])
        output = directory / "tokens"
        cases = [(["--capture_spec", "x.json"], "unknown option --capture_spec"),
                 (["--output-dir", str(directory / "x")], "unknown option --output-dir"),
                 (["--batch", "8", "--batch", "16"], "--batch given more than once"),
                 (["--batch", "8x"], "invalid integer for --batch")]
        if self.has_device_option:
            cases.append((["--device", "gpu"], "--device must be cpu or vulkan"))
        for extra, fragment in cases:
            with self.subTest(extra=extra):
                result = self.run_probe("tokenize", self.models["muse-glimmer"], docs, output, *extra)
                self.assert_fails(result, output, fragment)
        command = [PROBE, "--mode", "tokenize", "--mode", "collect", "--model", str(self.models["muse-glimmer"]),
                   "--input", str(docs), "--output", str(output)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assert_fails(result, output, "--mode given more than once")

    def test_leftover_staging_directories_are_reported_and_not_reused(self):
        directory = self.workdir()
        docs = self.write_docs(directory, [{"document_id": document_id(PLAIN), "text": PLAIN}])
        stale = directory / f".tokens.partial-{os.getpid()}"
        stale.mkdir()
        (stale / "junk").write_bytes(b"x")
        output = directory / "tokens"
        result = self.run_probe("tokenize", self.models["muse-glimmer"], docs, output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("leftover staging directory", result.stderr)
        self.assertIn(str(stale), result.stderr)
        self.assertEqual(sorted(p.name for p in directory.iterdir()), sorted(["docs.jsonl", stale.name, "tokens"]))
        self.assertEqual((stale / "junk").read_bytes(), b"x")
        self.assertEqual(sorted(p.name for p in output.iterdir()),
                         ["byte_end.npy", "report.json", "special.npy", "token_id.npy"])

    # -- collect ----------------------------------------------------------

    def test_collect_all_attention_layers_chunked_equals_whole_and_documents_are_isolated(self):
        texts = [CHAT, PLAIN]
        for architecture, layers in (("muse-glimmer", [0, 1, 2, 3]), ("qwen35", [3]), ("qwen3", [0, 1, 2, 3])):
            with self.subTest(architecture=architecture):
                report, whole, _ = self.collect(architecture, texts)
                self.assertEqual(sorted(whole), sorted((key, layer) for key in "kv" for layer in layers))
                _, chunked, _ = self.collect(architecture, texts, batch=7)
                _, isolated, _ = self.collect(architecture, [PLAIN])
                second = report["documents"][1]
                rows = slice(second["first_row"], second["first_row"] + second["n_tokens"])
                for name, expected in whole.items():
                    np.testing.assert_allclose(chunked[name], expected, atol=2e-5, rtol=2e-4)
                    np.testing.assert_allclose(isolated[name], expected[rows], atol=2e-6, rtol=2e-5)
                    self.assertGreater(float(np.std(expected)), 0)

    def test_muse_first_layer_matches_independent_projections(self):
        _, arrays, _ = self.collect("muse-glimmer", [CHAT], spec={"k": [0], "v": [0], "q": [0], "gate": [0]})
        weights = self.weights["muse-glimmer"]
        hidden = rms_norm(rms_norm(weights["token_embd.weight"][expected_tokens(CHAT)[0]]))
        n = len(hidden)
        value = hidden @ weights["blk.0.attn_v.weight"].T
        np.testing.assert_allclose(arrays[("v", 0)].reshape(n, -1), value, atol=2e-6, rtol=2e-5)
        key = rms_norm((hidden @ weights["blk.0.attn_k.weight"].T).reshape(n, 2, 32))
        np.testing.assert_allclose(arrays[("k", 0)], key, atol=2e-5, rtol=2e-4)
        query = rms_norm((hidden @ weights["blk.0.attn_q.weight"].T).reshape(n, 2, 32))
        np.testing.assert_allclose(arrays[("q", 0)], query, atol=2e-5, rtol=2e-4)
        gate = 1 / (1 + np.exp(-(hidden @ weights["blk.0.attn_gate.weight"].T)))
        np.testing.assert_allclose(arrays[("gate", 0)].reshape(n, -1), gate, atol=2e-6, rtol=2e-5)

    def test_qwen3_first_layer_matches_independent_projections_with_gqa(self):
        # Dense Qwen3: q/k RMSNorm per head before RoPE (4 query heads share 2 KV heads)
        report, arrays, _ = self.collect("qwen3", [CHAT], spec={"k": [0], "v": [0], "q": [0]})
        self.assertEqual([c["tensor"] for c in report["captures"]], ["Kcur_normed-0", "Vcur-0", "Qcur_normed-0"])
        weights = self.weights["qwen3"]
        hidden = rms_norm(weights["token_embd.weight"][expected_tokens(CHAT)[0]])
        n = len(hidden)
        value = hidden @ weights["blk.0.attn_v.weight"].T
        self.assertEqual(arrays[("v", 0)].shape, (n, 2, 32))
        np.testing.assert_allclose(arrays[("v", 0)].reshape(n, -1), value, atol=2e-6, rtol=2e-5)
        key = rms_norm((hidden @ weights["blk.0.attn_k.weight"].T).reshape(n, 2, 32))
        np.testing.assert_allclose(arrays[("k", 0)], key, atol=2e-5, rtol=2e-4)
        query = rms_norm((hidden @ weights["blk.0.attn_q.weight"].T).reshape(n, 4, 32))
        self.assertEqual(arrays[("q", 0)].shape, (n, 4, 32))
        np.testing.assert_allclose(arrays[("q", 0)], query, atol=2e-5, rtol=2e-4)

    def test_muse_full_layer_key_rms_and_attention_output_match_nope_attention(self):
        spec = {"k": "all", "v": [3], "q": [3], "gate": [3], "attn_out": [3]}
        report, arrays, _ = self.collect("muse-glimmer", [CHAT], spec=spec, batch=9)
        n = report["rows"]
        for name in [("q", 3), ("gate", 3), ("attn_out", 3), ("k", 3), ("v", 3)]:
            self.assertEqual(arrays[name].shape, (n, 2, 32))
        rms = np.sqrt(np.mean(arrays[("k", 3)].astype(np.float64) ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1, atol=1e-4)
        q, k, v = (arrays[(key, 3)].astype(np.float64) for key in "qkv")
        scores = np.einsum("ihd,jhd->hij", q, k) / np.sqrt(32)
        scores[:, np.triu(np.ones((n, n), dtype=bool), 1)] = -np.inf
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        expected = np.einsum("hij,jhd->ihd", weights, v)
        np.testing.assert_allclose(arrays[("attn_out", 3)], expected, atol=2e-5, rtol=1e-4)

    def test_qwen_two_dimensional_value_name_collision_is_not_captured(self):
        # One-token ubatches with one KV head give the 2-D Vcur MUL_MAT the same ne as the
        # 3-D reshape; only the RESHAPE may be captured (a second match is an error).
        report, single, _ = self.collect("qwen35", [PLAIN], batch=1)
        _, whole, _ = self.collect("qwen35", [PLAIN])
        self.assertEqual(report["captures"][1]["tensor"], "Vcur-3")
        for name, expected in whole.items():
            np.testing.assert_allclose(single[name], expected, atol=2e-5, rtol=2e-4)

    def test_fused_qkv_matches_separate_projections(self):
        spec = {"k": "all", "v": "all", "q": "all"}
        for architecture in ("muse-glimmer", "qwen3"):
            with self.subTest(architecture=architecture):
                _, separate, _ = self.collect(architecture, [CHAT], spec=spec)
                _, fused, _ = self.collect(architecture, [CHAT], spec=spec, fused=True, batch=7)
                self.assertEqual(sorted(fused), sorted(separate))
                for name, expected in separate.items():
                    np.testing.assert_allclose(fused[name], expected, atol=2e-5, rtol=2e-4)

    def test_collect_rejects_invalid_capture_specs(self):
        cases = [("qwen35", {"gate": [3]}, "only defined"), ("qwen35", {"k": [0]}, "not an attention"),
                 ("qwen3", {"attn_out": [0]}, "only defined for muse-glimmer"), ("qwen3", {"v": [4]}, "not an attention"),
                 ("muse-glimmer", {"k": [4]}, "not an attention"), ("muse-glimmer", {"x": "all"}, "unknown"),
                 ("muse-glimmer", {"k": []}, "nonempty"), ("muse-glimmer", {"k": [1, 1]}, "duplicate"),
                 ("muse-glimmer", '{"k": [3], "k": [0]}', 'duplicate json key "k"')]
        for architecture, spec, fragment in cases:
            with self.subTest(spec=spec):
                directory = self.workdir()
                docs = self.write_docs(directory, [{"document_id": document_id(PLAIN), "text": PLAIN}])
                text = spec if isinstance(spec, str) else json.dumps(spec)
                (directory / "spec.json").write_text(text, encoding="utf-8")
                output = directory / "capture"
                result = self.run_probe("collect", self.models[architecture], docs, output,
                                        "--capture-spec", directory / "spec.json")
                self.assert_fails(result, output, fragment)

    # -- patch-score ------------------------------------------------------

    def patch_dir(self, directory, name, positions, arrays):
        path = directory / name
        path.mkdir()
        np.save(path / "positions.npy", np.asarray(positions, dtype=np.int32))
        for (key, layer), value in arrays.items():
            np.save(path / f"{key}-{layer}.npy", np.ascontiguousarray(value, dtype=np.float32))
        return path

    def score_record(self, text, token_ids, conditions, sha=None):
        start = text.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
        start_byte = len(text[:start].encode("utf-8"))
        end_byte = len(text[:text.rindex("<|im_end|>")].encode("utf-8"))
        return {"document_id": document_id(text), "text": text,
                "expected_token_sha256": sha or hashlib.sha256(
                    np.asarray(token_ids, dtype="<i4").tobytes()).hexdigest(),
                "score_ranges": [[start_byte, end_byte]],
                "conditions": [{"name": name, "patch_dir": None if path is None else str(path)}
                               for name, path in conditions]}

    def test_patch_score_sanity_is_exact_and_zeroed_cache_rows_change_predictions(self):
        # batch 8 reaches later ubatches through the cache; batch 128 holds the whole
        # document in one ubatch, so the patched rows must reach the same-ubatch attention.
        cases = [(architecture, layers, 3, batch) for batch in (8, 128)
                 for architecture, layers in (("muse-glimmer", [0, 1, 2, 3]), ("qwen35", [3]),
                                              ("qwen3", [0, 1, 2, 3]))]
        for architecture, layers, full, batch in cases:
            with self.subTest(architecture=architecture, batch=batch):
                report, native, capture = self.collect(architecture, [CHAT], batch=batch)
                token_ids = np.load(capture / "token_id.npy").tolist()
                prefix = CHAT.index("<|im_start|>assistant")
                n_prefix = sum(1 for end in expected_tokens(CHAT)[1] if end <= len(CHAT[:prefix].encode()))
                positions = np.arange(1, n_prefix, 3)
                directory = self.workdir()
                sanity = self.patch_dir(directory, "sanity", positions,
                                        {(key, layer): native[(key, layer)][positions]
                                         for key in "kv" for layer in layers})
                zero_v = self.patch_dir(directory, "zero_v", positions,
                                        {("v", full): np.zeros_like(native[("v", full)][positions])})
                zero_k = self.patch_dir(directory, "zero_k", positions,
                                        {("k", full): np.zeros_like(native[("k", full)][positions])})
                conditions = [("native", None), ("sanity", sanity), ("zero_v", zero_v),
                              ("zero_k", zero_k), ("native_again", None)]
                docs = self.write_docs(directory, [self.score_record(CHAT, token_ids, conditions)])
                output = directory / "scores"
                result = self.run_probe("patch-score", self.models[architecture], docs, output,
                                        "--batch", batch)
                self.assertEqual(result.returncode, 0, result.stderr)
                results = json.loads((output / "results.json").read_text(encoding="utf-8"))
                self.assertEqual((results["format_version"], results["mode"]), (2, "patch-score"))
                document = results["documents"][0]
                self.assertEqual(document["n_tokens"], len(token_ids))
                scored = document["scored_positions"]
                self.assertEqual(document["n_scored"], len(scored))
                self.assertGreater(len(scored), 8)
                self.assertTrue(all(p > n_prefix for p in scored))
                conditions = {c["name"]: c for c in document["conditions"]}
                scores = {name: np.load(output / "scores" / document["document_id"] / f"{name}.npy")
                          for name in conditions}
                for value in scores.values():
                    self.assertEqual(value.shape, (len(scored), 3))
                self.assertTrue((scores["native"][:, 1:] == [0, 1]).all())
                self.assertEqual(conditions["native"]["applied_rows"], {})
                np.testing.assert_array_equal(scores["sanity"][:, 0], scores["native"][:, 0])
                np.testing.assert_array_equal(scores["native_again"], scores["native"])
                self.assertLess(abs(conditions["sanity"]["kl_mean"]), 1e-9)
                self.assertEqual(conditions["sanity"]["top1_agree"], 1)
                self.assertEqual(conditions["sanity"]["applied_rows"],
                                 {f"{key}-{layer}": len(positions) for key in "kv" for layer in layers})
                for name in ("zero_v", "zero_k"):
                    self.assertEqual(conditions[name]["applied_rows"],
                                     {f"{name[-1]}-{full}": len(positions)})
                    self.assertGreater(conditions[name]["kl_mean"], 0)
                    self.assertFalse(np.array_equal(scores[name][:, 0], scores["native"][:, 0]))
                    self.assertTrue((scores[name][:, 1] >= -1e-12).all())

    def score(self, architecture, records, directory, name, batch):
        output = directory / name
        docs = self.write_docs(self.workdir(), records)
        result = self.run_probe("patch-score", self.models[architecture], docs, output, "--batch", batch)
        self.assertEqual(result.returncode, 0, result.stderr)
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))
        loaded = []
        for document in results["documents"]:
            scores = {c["name"]: np.load(output / "scores" / document["document_id"] / f"{c['name']}.npy")
                      for c in document["conditions"]}
            loaded.append((document, scores))
        return loaded

    def test_patch_score_second_document_matches_solo_run(self):
        # The cache is cleared and the native reference reset between documents.
        other = CHAT.replace("coffee, tea and snacks", "bread")
        for architecture, full in (("muse-glimmer", 3), ("qwen35", 3), ("qwen3", 3)):
            with self.subTest(architecture=architecture):
                _, native, capture = self.collect(architecture, [CHAT], batch=16)
                token_ids = np.load(capture / "token_id.npy").tolist()
                positions = np.arange(2, 40, 4)
                directory = self.workdir()
                zero_v = self.patch_dir(directory, "zero_v", positions,
                                        {("v", full): np.zeros_like(native[("v", full)][positions])})
                conditions = [("native", None), ("zero_v", zero_v)]
                other_ids = expected_tokens(other)[0]
                first = self.score_record(other, other_ids, [("native", None)])
                second = self.score_record(CHAT, token_ids, conditions)
                paired = self.score(architecture, [first, second], directory, "paired", 16)
                solo = self.score(architecture, [second], directory, "solo", 16)
                self.assertEqual(len(paired), 2)
                self.assertEqual(paired[1][0], solo[0][0])
                for name in ("native", "zero_v"):
                    np.testing.assert_array_equal(paired[1][1][name], solo[0][1][name])
                self.assertGreater(solo[0][0]["conditions"][1]["kl_mean"], 0)

    def test_patch_score_native_scores_do_not_depend_on_batch_size(self):
        # Checks positions across ubatch boundaries (Qwen K/V captures are pre-RoPE).
        token_ids = expected_tokens(CHAT)[0]
        record = self.score_record(CHAT, token_ids, [("native", None)])
        for architecture in ("muse-glimmer", "qwen35", "qwen3"):
            with self.subTest(architecture=architecture):
                directory = self.workdir()
                runs = {batch: self.score(architecture, [record], directory, f"b{batch}", batch)[0]
                        for batch in (128, 7, 1)}
                reference = runs[128][1]["native"]
                self.assertGreater(float(np.std(reference[:, 0])), 0)
                np.testing.assert_array_equal(runs[7][1]["native"], reference)
                np.testing.assert_allclose(runs[1][1]["native"], reference, atol=1e-5, rtol=0)

    def test_patch_score_caps_scored_tokens(self):
        directory = self.workdir()
        token_ids = expected_tokens(CHAT)[0]
        docs = self.write_docs(directory, [self.score_record(CHAT, token_ids, [("native", None)])])
        output = directory / "scores"
        result = self.run_probe("patch-score", self.models["muse-glimmer"], docs, output,
                                "--batch", 5, "--max-scored", 3)
        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads((output / "results.json").read_text(encoding="utf-8"))["documents"][0]
        self.assertEqual(document["n_scored"], 3)
        self.assertEqual(document["scored_positions"], list(range(document["scored_positions"][0],
                                                                  document["scored_positions"][0] + 3)))
        start = CHAT.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
        eligible = len(CHAT[start:CHAT.rindex("<|im_end|>")].encode("utf-8"))
        self.assertEqual((document["n_eligible"], document["scored_truncated"]), (eligible, True))
        output = directory / "uncapped"
        result = self.run_probe("patch-score", self.models["muse-glimmer"], docs, output,
                                "--batch", 5, "--max-scored", eligible)
        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads((output / "results.json").read_text(encoding="utf-8"))["documents"][0]
        self.assertEqual((document["n_scored"], document["n_eligible"], document["scored_truncated"]),
                         (eligible, eligible, False))

    def test_patch_score_stops_after_the_last_needed_ubatch_with_identical_scores(self):
        tail = "<|im_start|>user\n" + "Tokens after the scored range cannot change it. " * 6 + "<|im_end|>\n"
        text = CHAT + tail
        token_ids = expected_tokens(text)[0]
        _, native, capture = self.collect("muse-glimmer", [text], batch=5)
        start = text.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
        end = text.index("<|im_end|>", start)
        record = self.score_record(text, token_ids, [], sha=None)
        record["score_ranges"] = [[len(text[:start].encode()), len(text[:end].encode())]]
        directory = self.workdir()
        early_patch = np.arange(1, 20, 3)
        late_patch = np.asarray([2, len(token_ids) - 30])
        zero_v = self.patch_dir(directory, "zero_v", early_patch,
                                {("v", 3): np.zeros_like(native[("v", 3)][early_patch])})
        late = self.patch_dir(directory, "late", late_patch,
                              {("k", 1): native[("k", 1)][late_patch] * 2})
        record["conditions"] = [{"name": "native", "patch_dir": None},
                                {"name": "zero_v", "patch_dir": str(zero_v)},
                                {"name": "late", "patch_dir": str(late)}]
        docs = self.write_docs(directory, [record])
        runs = {}
        for label, extra in (("early", []), ("full", ["--full-decode"])):
            output = directory / label
            result = self.run_probe("patch-score", self.models["muse-glimmer"], docs, output,
                                    "--batch", 5, "--max-scored", 4, *extra)
            self.assertEqual(result.returncode, 0, result.stderr)
            results = json.loads((output / "results.json").read_text(encoding="utf-8"))
            document = results["documents"][0]
            runs[label] = (results, document, {c["name"]: np.load(output / "scores" / document["document_id"]
                                                                   / f"{c['name']}.npy")
                                               for c in document["conditions"]})
        early, full = runs["early"], runs["full"]
        self.assertEqual(early[0]["settings"]["decode"], "through_last_needed_ubatch")
        self.assertEqual(full[0]["settings"]["decode"], "full")
        last = early[1]["scored_positions"][-1]
        self.assertEqual(len(early[1]["scored_positions"]), 4)
        decoded = {c["name"]: c["decoded_tokens"] for c in early[1]["conditions"]}
        self.assertEqual(decoded["native"], -(-last // 5) * 5)
        self.assertEqual(decoded["zero_v"], decoded["native"])
        self.assertEqual(decoded["late"], min(len(token_ids), -(-(int(late_patch[-1]) + 1) // 5) * 5))
        self.assertLess(decoded["native"], len(token_ids) - 100)
        self.assertEqual({c["decoded_tokens"] for c in full[1]["conditions"]}, {len(token_ids)})
        for name in ("native", "zero_v", "late"):
            np.testing.assert_array_equal(early[2][name], full[2][name])
        self.assertEqual([c["applied_rows"] for c in early[1]["conditions"]],
                         [c["applied_rows"] for c in full[1]["conditions"]])
        self.assertGreater(early[2]["zero_v"][:, 1].max(), 0)
        wrong_mode = self.run_probe("collect", self.models["muse-glimmer"], docs, directory / "x", "--full-decode")
        self.assert_fails(wrong_mode, directory / "x", "only valid with --mode patch-score")

    def test_patch_score_rejects_invalid_requests(self):
        token_ids = expected_tokens(CHAT)[0]
        v = np.zeros((2, 2, 32), dtype=np.float32)
        directory = self.workdir()
        good = self.patch_dir(directory, "good", [1, 2], {("v", 3): v})
        outside = self.patch_dir(directory, "outside", [1, len(token_ids)], {("v", 3): v})
        unordered = self.patch_dir(directory, "unordered", [2, 1], {("v", 3): v})
        wrong_shape = self.patch_dir(directory, "wrong_shape", [1, 2], {("v", 3): v[:, :1]})
        unexpected = self.patch_dir(directory, "unexpected", [1, 2], {("q", 3): v})
        cases = [
            (self.score_record(CHAT, token_ids, [("native", None)], sha="0" * 64), "expected_token_sha256"),
            (self.score_record(CHAT, token_ids, [("good", good)]), "first condition"),
            (self.score_record(CHAT, token_ids, [("native", None), ("x", outside)]), "strictly increasing"),
            (self.score_record(CHAT, token_ids, [("native", None), ("x", unordered)]), "strictly increasing"),
            (self.score_record(CHAT, token_ids, [("native", None), ("x", wrong_shape)]), "must be float32"),
            (self.score_record(CHAT, token_ids, [("native", None), ("x", unexpected)]), "unexpected patch"),
            (self.score_record(CHAT, token_ids, [("native", None), ("x", good), ("x", good)]), "duplicate"),
            (self.score_record(CHAT, token_ids, [("native", None), ("../x", good)]), "invalid"),
        ]
        for record, fragment in cases:
            with self.subTest(fragment=fragment):
                work = self.workdir()
                output = work / "scores"
                result = self.run_probe("patch-score", self.models["muse-glimmer"],
                                        self.write_docs(work, [record]), output, "--batch", 8)
                self.assert_fails(result, output, fragment.lower())



LAYOUTS = [(cache, flash) for cache in ("f32", "f16") for flash in ("off", "on")]
LONG = ("<|im_start|>system\nYou answer briefly.<|im_end|>\n<|im_start|>user\nRemember these words: "
        + "alpha beta gamma delta " * 15 + "<|im_end|>\n<|im_start|>assistant\n"
        + "They were alpha, beta, gamma and delta, repeated many times.<|im_end|>\n")


def layout_options(cache, flash, swa_full="on"):
    return ["--cache-type", cache, "--flash-attn", flash, "--swa-full", swa_full]


def assistant_range(text):
    start = text.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    return len(text[:start].encode("utf-8")), len(text[:text.rindex("<|im_end|>")].encode("utf-8"))


@unittest.skipUnless(PROBE, "set KVPARK_KV_PROBE to enable real llama.cpp kv-probe checks")
class KvStateIntegrationTests(unittest.TestCase):
    """save-state / seed-score on tiny fixtures, parsed and rebuilt with kvstate."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-kv-state-")
        cls.root = Path(cls.temp.name)
        cls.models, cls.specs = {}, {}
        fixtures = {"muse-glimmer": {}, "qwen35": {}, "qwen3": {},
                    # base 1e7 with 16 rotated dims: numpy's float32 power is 1 ulp off libm powf there
                    "qwen35-partial": {"rope_dimension_count": 16, "rope_sections": [3, 3, 2, 0],
                                       "rope_freq_base": 1e7}}
        for name, extra in fixtures.items():
            path = cls.root / f"{name}.gguf"
            create_fixture(path, name.split("-partial")[0], control_tokens=CONTROL, **extra)
            cls.models[name] = path
            cls.specs[name] = kvstate.ModelSpec.from_gguf(path)
        cls.cache = {}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def workdir(self):
        return Path(tempfile.mkdtemp(dir=self.root))

    def run_probe(self, mode, model, records, output, *extra, ctx=1024, batch=16):
        directory = output.parent
        docs = directory / f"{output.name}.jsonl"
        docs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        command = [PROBE, "--mode", mode, "--model", str(self.models[model]), "--input", str(docs),
                   "--output", str(output), "--ctx", str(ctx), "--threads", "2", "--batch", str(batch),
                   *map(str, extra)]
        return subprocess.run(command, capture_output=True, text=True, timeout=120)

    def ok(self, result):
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])

    def assert_fails(self, result, output, fragment):
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn(fragment.lower(), result.stderr.lower())
        self.assertFalse(output.exists())
        self.assertEqual([p for p in output.parent.iterdir() if ".partial-" in p.name], [])

    def prepared(self, model, cache, flash, text=CHAT, save_at=(64, 75), swa_full="on", batch=16):
        """(captures, save-state report, output dir) for one layout; reused across tests."""
        key = (model, cache, flash, text, tuple(save_at), swa_full, batch)
        if key not in self.cache:
            directory = self.workdir()
            options = layout_options(cache, flash, swa_full)
            capture = directory / "capture"
            self.ok(self.run_probe("collect", model, [{"document_id": document_id(text), "text": text}], capture,
                                   *options, batch=batch))
            record = {"document_id": document_id(text), "text": text, "save_at": list(save_at)}
            states = directory / "states"
            self.ok(self.run_probe("save-state", model, [record], states, *options, batch=batch))
            report = json.loads((states / "report.json").read_text(encoding="utf-8"))
            arrays = {(name[0], int(name[2:-4])): np.load(capture / name)
                      for name in os.listdir(capture) if name[:2] in ("k-", "v-")}
            self.cache[key] = (np.load(capture / "token_id.npy"), arrays, report, states)
        return self.cache[key]

    def seed_record(self, text, tokens, conditions):
        start, end = assistant_range(text)
        return {"document_id": document_id(text), "text": text,
                "expected_token_sha256": hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest(),
                "score_ranges": [[start, end]],
                "conditions": [{"name": name, "state_file": None if path is None else str(path)}
                               for name, path in conditions]}

    def seed_score(self, model, record, cache, flash, swa_full="on", batch=16):
        output = self.workdir() / "seed"
        self.ok(self.run_probe("seed-score", model, [record], output, *layout_options(cache, flash, swa_full),
                               batch=batch))
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))
        self.assertEqual((results["format_version"], results["mode"]), (2, "seed-score"))
        document = results["documents"][0]
        scores = {c["name"]: np.load(output / "scores" / document["document_id"] / f"{c['name']}.npy")
                  for c in document["conditions"]}
        return results, document, {c["name"]: c for c in document["conditions"]}, scores

    # -- save-state ---------------------------------------------------------

    def test_save_state_files_round_trip_and_hold_the_captured_cache(self):
        for model in ("muse-glimmer", "qwen35", "qwen3"):
            spec = self.specs[model]
            for cache, flash in LAYOUTS:
                with self.subTest(model=model, cache=cache, flash=flash):
                    tokens, arrays, report, states = self.prepared(model, cache, flash)
                    n = len(tokens)
                    self.assertEqual((report["format_version"], report["mode"]), (2, "save-state"))
                    self.assertEqual((report["settings"]["cache_type"], report["settings"]["flash_attention"]),
                                     (cache, flash == "on"))
                    document = report["documents"][0]
                    self.assertEqual(document["prefill_tokens"], 75)
                    self.assertEqual(document["ubatch_ends"], [16, 32, 48, 64, 75])
                    self.assertEqual([s["n_tokens"] for s in document["states"]], [64, 75])
                    dtype = np.float16 if cache == "f16" else np.float32
                    atol = 1e-3 if cache == "f16" else 1e-6
                    for entry in document["states"]:
                        size = entry["n_tokens"]
                        data = (states / entry["file"]).read_bytes()
                        self.assertEqual(len(data), entry["bytes"])
                        self.assertEqual(entry["pos_max"], size - 1)
                        parsed = kvstate.parse_file(data, spec)
                        self.assertEqual(parsed.to_bytes(), data)
                        np.testing.assert_array_equal(parsed.tokens, tokens[:size])
                        self.assertFalse(parsed.server_packed)
                        for part in parsed.state.parts:
                            if isinstance(part, kvstate.RecurrentState):
                                self.assertEqual((part.layers, part.positions.tolist(), part.n_layer),
                                                 ((0, 1, 2), [size - 1], 4))
                                continue
                            stream = part.stream()
                            self.assertEqual(stream.v_trans, flash == "off")
                            expected = np.arange(size - 16, size) if part.name == "swa" else np.arange(size)
                            np.testing.assert_array_equal(stream.positions, expected)
                            if spec.cell_ext:
                                np.testing.assert_array_equal(
                                    stream.ext, np.stack([expected, expected, tokens[expected]], axis=1))
                            for layer in part.layers:
                                captured_k = arrays[("k", layer)][expected]
                                self.assertEqual(stream.k[layer].dtype, dtype)
                                np.testing.assert_array_equal(stream.v[layer], arrays[("v", layer)][expected].astype(dtype))
                                stored = stream.k[layer].astype(np.float32)
                                if spec.layer_types[layer] == "full" and model == "muse-glimmer":
                                    np.testing.assert_array_equal(stream.k[layer], captured_k.astype(dtype))
                                    continue
                                # bit-exact on the pinned CPU build (libm powf/cosf/sinf, fused multiply-adds)
                                rotated = kvstate.cache_keys(spec, layer, captured_k, expected)
                                np.testing.assert_array_equal(stream.k[layer], rotated.astype(dtype))
                                self.assertGreater(np.abs(stored - captured_k).max(), 0.01)
                                np.testing.assert_allclose(kvstate.pre_rope_keys(spec, layer, stored, expected),
                                                           captured_k, atol=2 * atol, rtol=0)

    def test_qwen_partial_rotary_state_keys_derotate_to_the_captured_keys(self):
        spec = self.specs["qwen35-partial"]
        self.assertEqual((spec.n_rot, spec.rope_sections, spec.rope_freq_base), (16, (3, 3, 2, 0), 1e7))
        tokens, arrays, report, states = self.prepared("qwen35-partial", "f32", "on", save_at=(80,))
        parsed = kvstate.read_file(states / report["documents"][0]["states"][0]["file"], spec)
        stream = parsed.state.part("attn").stream()
        positions = stream.positions
        stored, captured = stream.k[3], arrays[("k", 3)][positions]
        np.testing.assert_array_equal(stored[:, :, 16:], captured[:, :, 16:])
        self.assertGreater(np.abs(stored[:, :, :16] - captured[:, :, :16]).max(), 0.01)
        np.testing.assert_array_equal(kvstate.cache_keys(spec, 3, captured, positions), stored)
        np.testing.assert_allclose(kvstate.pre_rope_keys(spec, 3, stored, positions), captured, atol=1e-6, rtol=0)

    def test_swa_ring_cells_hold_the_last_window_and_restore_without_swa_full(self):
        # swa_full off: the SWA cache has 256 cells, so a 400-token prefill wraps it.
        tokens = expected_tokens(LONG)[0]
        start_position = next(i for i, end in enumerate(expected_tokens(LONG)[1])
                              if end > assistant_range(LONG)[0])
        seed = start_position // 16 * 16
        self.assertGreater(seed, 256 + 16)
        spec = self.specs["muse-glimmer"]
        _, arrays, report, states = self.prepared("muse-glimmer", "f16", "on", text=LONG, save_at=(seed,),
                                                  swa_full="off")
        path = states / report["documents"][0]["states"][0]["file"]
        parsed = kvstate.read_file(path, spec)
        swa = parsed.state.part("swa").stream()
        self.assertEqual(sorted(swa.positions.tolist()), list(range(seed - 16, seed)))
        for layer in spec.layers_of("sliding"):
            np.testing.assert_array_equal(swa.v[layer], arrays[("v", layer)][swa.positions].astype(np.float16))
        np.testing.assert_array_equal(parsed.state.part("base").stream().positions, np.arange(seed))
        record = self.seed_record(LONG, tokens, [("native", None), ("seed", path)])
        _, document, conditions, scores = self.seed_score("muse-glimmer", record, "f16", "on", swa_full="off")
        self.assertEqual(conditions["seed"]["restored_pos_min"], seed - 16)
        self.assertEqual(conditions["seed"]["seed_tokens"], seed)
        # restored cells sit in other ring slots than in the native pass; allow summation-order noise
        self.assertLess(float(scores["seed"][:, 1].max()), 1e-9)
        np.testing.assert_allclose(scores["seed"][:, 0], scores["native"][:, 0], atol=1e-5, rtol=0)
        self.assertEqual(conditions["seed"]["top1_agree"], 1)

    def test_save_state_rejects_invalid_requests(self):
        n = len(expected_tokens(CHAT)[0])
        cases = [({"save_at": [0]}, [], "save_at entries"), ({"save_at": [n + 1]}, [], "save_at entries"),
                 ({"save_at": [5, 5]}, [], "duplicate save_at"), ({"save_at": 5}, [], "nonempty list"),
                 ({"expected_token_sha256": "0" * 64}, [], "expected_token_sha256"),
                 ({}, ["--cache-type", "q8_0"], "--cache-type must be"),
                 ({}, ["--flash-attn", "auto"], "--flash-attn must be on or off"),
                 ({}, ["--max-scored", "3"], "only valid with --mode patch-score or seed-score")]
        for fields, extra, fragment in cases:
            with self.subTest(fragment=fragment, fields=fields):
                output = self.workdir() / "states"
                record = {"document_id": document_id(CHAT), "text": CHAT, **fields}
                self.assert_fails(self.run_probe("save-state", "muse-glimmer", [record], output, *extra),
                                  output, fragment)
        output = self.workdir() / "tokens"
        self.assert_fails(self.run_probe("tokenize", "muse-glimmer", [{"document_id": document_id(CHAT), "text": CHAT}],
                                         output, "--cache-type", "f16"), output, "not valid with --mode tokenize")

    # -- seed-score ---------------------------------------------------------

    def test_seed_score_native_rebuilt_and_perturbed_states(self):
        for model in ("muse-glimmer", "qwen35", "qwen3"):
            spec = self.specs[model]
            for cache, flash in LAYOUTS:
                with self.subTest(model=model, cache=cache, flash=flash):
                    tokens, arrays, report, states = self.prepared(model, cache, flash)
                    files = {s["n_tokens"]: states / s["file"] for s in report["documents"][0]["states"]}
                    directory = self.workdir()
                    conditions = [("native", None), ("seed64", files[64]), ("seed75", files[75])]
                    native64 = kvstate.read_file(files[64], spec)
                    if model in ("muse-glimmer", "qwen3"):
                        layers = spec.layers_of("full", "sliding")
                        options = {"dtype": cache, "v_trans": flash == "off"}
                        build = kvstate.build_muse_state if model == "muse-glimmer" else kvstate.build_dense_state
                        rebuilt = build(spec, tokens[:64], np.arange(64), {il: arrays[("k", il)][:64] for il in layers},
                                        {il: arrays[("v", il)][:64] for il in layers}, **options)
                        # the builder reproduces the native file byte for byte (pinned CPU RoPE numerics)
                        self.assertEqual(rebuilt.to_bytes(), files[64].read_bytes())
                        kvstate.write_file(directory / "rebuilt.state", rebuilt)
                        packed = kvstate.StateFile(kvstate.token_words(tokens[:64], server_packed=True), rebuilt.state)
                        kvstate.write_file(directory / "packed.state", packed)
                        conditions += [("rebuilt", directory / "rebuilt.state"), ("packed", directory / "packed.state")]
                    else:
                        recurrent = native64.state.part("recurrent")
                        rows = recurrent.s[0]
                        halved = (np.frombuffer(rows.data, "<f4") * np.float32(0.5)).astype("<f4").tobytes()
                        recurrent.s[0] = kvstate.OpaqueRows(rows.ggml_type, rows.row_bytes, halved)
                        kvstate.write_file(directory / "recurrent.state", native64)
                        conditions.append(("recurrent_halved", directory / "recurrent.state"))
                        native64 = kvstate.read_file(files[64], spec)
                    part = native64.state.parts[0]
                    stream, layer = part.stream(), part.layers[0]
                    stream.v[layer] = stream.v[layer].copy()
                    stream.v[layer][5:20] = 0
                    kvstate.write_file(directory / "perturbed.state", native64)
                    conditions.append(("perturbed", directory / "perturbed.state"))
                    record = self.seed_record(CHAT, tokens, conditions)
                    results, document, by_name, scores = self.seed_score(model, record, cache, flash)
                    self.assertEqual(results["settings"]["cache_type"], cache)
                    last = document["scored_positions"][-1]
                    self.assertGreater(document["scored_positions"][0], 75)
                    decoded_end = min(len(tokens), -(-last // 16) * 16)
                    self.assertEqual(by_name["native"]["decoded_tokens"], decoded_end)
                    self.assertEqual((by_name["native"]["seed_tokens"], by_name["native"]["state_file"]), (0, None))
                    for name, seed in (("seed64", 64), ("seed75", 75)):
                        self.assertEqual(by_name[name]["seed_tokens"], seed)
                        self.assertEqual(by_name[name]["restored_pos_max"], seed - 1)
                        # iSWA: the SWA window; hybrid: the recurrent cell; plain KV: every cell from 0
                        self.assertEqual(by_name[name]["restored_pos_min"],
                                         {"muse-glimmer": seed - 16, "qwen35": seed - 1, "qwen3": 0}[model])
                        self.assertEqual(by_name[name]["decoded_tokens"], decoded_end - seed)
                        self.assertEqual(by_name[name]["applied_rows"], {})
                    # a seed on the --batch grid repeats the native ubatches exactly
                    np.testing.assert_array_equal(scores["seed64"], scores["native"])
                    # 75 splits a ubatch in save-state; only CPU summation-order noise may remain
                    self.assertLess(float(scores["seed75"][:, 1].max()), 1e-12)
                    self.assertEqual(by_name["seed75"]["top1_agree"], 1)
                    if model in ("muse-glimmer", "qwen3"):
                        for name in ("rebuilt", "packed"):
                            np.testing.assert_array_equal(scores[name], scores["native"])
                        self.assertEqual((by_name["rebuilt"]["server_packed_tokens"],
                                          by_name["packed"]["server_packed_tokens"]), (False, True))
                    else:
                        self.assertGreater(by_name["recurrent_halved"]["kl_mean"], 0)
                    self.assertGreater(by_name["perturbed"]["kl_mean"], 0)
                    self.assertFalse(np.array_equal(scores["perturbed"][:, 0], scores["native"][:, 0]))

    def test_seed_score_rejects_mismatched_states(self):
        spec = self.specs["muse-glimmer"]
        tokens, _, report, states = self.prepared("muse-glimmer", "f32", "off")
        path64 = states / report["documents"][0]["states"][0]["file"]
        directory = self.workdir()
        other_tokens, _, other_report, other_states = self.prepared("muse-glimmer", "f32", "off",
                                                                    text=CHAT.replace("briefly", "slowly"))
        foreign = other_states / other_report["documents"][0]["states"][0]["file"]
        state = kvstate.read_file(path64, spec)
        edited = kvstate.StateFile(state.token_words.copy(), state.state)
        edited.token_words[10] += 1
        kvstate.write_file(directory / "edited.state", edited)
        short = kvstate.StateFile(state.token_words[:63].copy(), state.state)
        kvstate.write_file(directory / "short.state", short, check_positions=False)
        (directory / "trailing.state").write_bytes(path64.read_bytes() + b"\0")
        (directory / "junk.state").write_bytes(b"not a state file at all")
        full = self.prepared("muse-glimmer", "f32", "off", save_at=(64, len(tokens)))
        full_path = full[3] / full[2]["documents"][0]["states"][1]["file"]
        f16_path = self.prepared("muse-glimmer", "f16", "off")[3] / "states" / document_id(CHAT) / "64.state"
        cases = [
            ([("native", None), ("x", foreign)], "f32", "not a prefix of the document tokens"),
            ([("native", None), ("x", directory / "edited.state")], "f32", "not a prefix"),
            ([("native", None), ("x", full_path)], "f32", "needs logits from a decoded token"),
            ([("native", None), ("x", directory / "trailing.state")], "f32", "trailing bytes"),
            ([("native", None), ("x", directory / "junk.state")], "f32", "magic"),
            ([("native", None), ("x", directory / "short.state")], "f32", "ends at position 63"),
            ([("native", None), ("x", f16_path)], "f32", "could not restore"),
            ([("x", path64)], "f32", "first condition must be native"),
            ([("native", None), ("x", Path("relative.state"))], "f32", "existing absolute file"),
        ]
        self.assertNotEqual(other_tokens[:64].tolist(), tokens[:64].tolist())
        for conditions, cache, fragment in cases:
            with self.subTest(fragment=fragment):
                output = self.workdir() / "seed"
                record = self.seed_record(CHAT, tokens, conditions)
                self.assert_fails(self.run_probe("seed-score", "muse-glimmer", [record], output,
                                                 *layout_options(cache, "off")), output, fragment)


# MLA/DSA fixtures (gguf-py, byte vocabulary): name -> (architecture, create_mla_fixture options)
MLA_FIXTURES = {"deepseek2": ("deepseek2", {}), "deepseek2-yarn": ("deepseek2", {"yarn": YARN_V2}),
                "deepseek2-nextn": ("deepseek2", {"nextn": 1}), "mistral4-yarn": ("mistral4", {"yarn": YARN_V2}),
                "deepseek32": ("deepseek32", {}), "glm-dsa": ("glm-dsa", {"layers": 4, "indexer_types": [1, 0, 1, 0]}),
                "dots3note": ("dots3note", {"layers": 4})}
# test-llama-archs hybrids whose attention layers the probe captures
HYBRID_ARCHS = ("kimi-linear", "qwen3next", "qwen35moe", "qwen4exp", "plamo2", "jamba", "falcon-h1", "granitehybrid",
                "nemotron_h", "nemotron_h_moe", "lfm2", "lfm2moe", "minimax-01")
ASCII = ("Plain ASCII only: the archs fixtures know bytes 0-127. " * 3).strip()


@unittest.skipUnless(PROBE, "set KVPARK_KV_PROBE to enable real llama.cpp kv-probe checks")
class KvProbeFamilyTests(unittest.TestCase):
    """Captures of the MLA/DSA family (the [c_kv | k_pe] row from two graph nodes, its c_kv view, the
    indexer keys) and of the attention layers of the new hybrids, against the probe's own saved states
    parsed with kvstate: K is the de-rotated cache row, V the stored V (an MLA layer's c_kv view)."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-kv-probe-families-")
        cls.root = Path(cls.temp.name)
        cls.models = {}
        for name, (architecture, options) in MLA_FIXTURES.items():
            cls.models[name] = cls.root / f"{name}.gguf"
            create_mla_fixture(cls.models[name], architecture, **options)
        if not probe_supports(cls.models["deepseek2"], cls.root):
            cls.temp.cleanup()
            raise unittest.SkipTest("this kvpark-kv-probe predates the MLA/DSA and hybrid tables; rebuild it")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def workdir(self):
        return Path(tempfile.mkdtemp(dir=self.root))

    def probe(self, mode, model, records, *extra, spec=None):
        directory = self.workdir()
        docs = directory / "docs.jsonl"
        docs.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        options = list(extra)
        if spec is not None:
            (directory / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
            options += ["--capture-spec", directory / "spec.json"]
        output = directory / "out"
        result = subprocess.run([PROBE, "--mode", mode, "--model", str(model), "--input", str(docs), "--output",
                                 str(output), "--ctx", "256", "--batch", "32", "--threads", "2", *map(str, options)],
                                capture_output=True, text=True, timeout=300)
        return result, output

    def capture_and_state(self, model, spec, text):
        """(Capture, kvstate.StateFile at the document's end) for one document."""
        record = {"document_id": document_id(text), "text": text}
        # the DSA lightning-indexer keys (qwen4exp's pooled-block indexer has no capture key)
        keys = {"k": "all", "v": "all", **({"ik": "all"} if spec.memory in ("dsa", "dsa_iswa") else {})}
        result, output = self.probe("collect", model, [record], spec=keys)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        capture = Capture(output)
        record["save_at"] = [capture.rows_total]
        result, saved = self.probe("save-state", model, [record])
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        report = json.loads((saved / "report.json").read_text(encoding="utf-8"))
        return capture, kvstate.read_file(saved / report["documents"][0]["states"][0]["file"], spec)

    def check_against_state(self, spec, capture, state):
        self.assertEqual([kind for kind in spec.layer_types if kind != "nextn"], capture.layer_types)
        checked = set()
        for part, layout in zip(state.state.parts, spec.layouts(state.state.flags)):
            if layout.kind != "kv" or (layout.role == "indexer" and "ik" not in capture.keys()):
                continue
            stream = part.stream()
            positions = np.asarray(stream.positions, dtype=np.int64)
            for layer in part.layers:
                if spec.layer_types[layer] == "nextn":
                    self.assertNotIn(layer, capture.layers("k"))         # not a trunk layer: not captured
                    continue
                key = "k" if layout.role == "attention" else "ik"
                pre = kvstate.pre_rope_keys(spec, layer, stream.k[layer], positions, role=layout.role)
                captured = np.asarray(capture.array(key, layer))[positions]
                scale = float(np.abs(captured).max())
                # f32 cache: de-rotation (float64) and, for indexer keys, the Hadamard mul_mat's summation order
                np.testing.assert_allclose(pre, captured, rtol=0, atol=(2e-6 if key == "ik" else 1e-6) * scale,
                                           err_msg=f"{key}-{layer}")
                if layout.role == "attention":
                    np.testing.assert_array_equal(kvstate.value_rows(spec, stream, layer),
                                                  np.asarray(capture.array("v", layer))[positions])
                checked.add((key, layer))
        self.assertEqual(checked | {("v", layer) for key, layer in checked if key == "k"},
                         {(key, layer) for key in capture.keys() for layer in capture.layers(key)})

    def test_mla_rows_and_indexer_keys_match_the_saved_cache(self):
        text = "Latent caches: one row of [c_kv | k_pe] per token and layer. " * 2
        for name, model in self.models.items():
            with self.subTest(model=name):
                spec = kvstate.ModelSpec.from_gguf(model)
                capture, state = self.capture_and_state(model, spec, text)
                self.check_against_state(spec, capture, state)
                for layer in capture.layers("k"):
                    # the K row is assembled from the latent and the pre-RoPE k_pe nodes
                    self.assertEqual(capture.array("k", layer).shape[1:], spec.head_shape(layer, "k"))
                    self.assertEqual(capture.array("v", layer).shape[1:], (1, spec.kv_lora[layer]))
                entries = {(c["key"], c["layer"]): c["tensor"] for c in capture.report["captures"]}
                self.assertEqual(entries["k", 0], "kv_cmpr-0+k_pe-0")
                self.assertEqual(capture.indexer_layers(), list(spec.indexer.layers) if spec.indexer else [])

    def test_indexer_keys_without_the_k_norm_bias(self):
        # glm-dsa's indexer k_norm weight and bias are optional (glm-dsa.cpp:144-145): without the bias the
        # LayerNorm ends in its weight MUL (layer 0), without either in the NORM itself (layer 2)
        model = self.root / "glm-dsa-bare-k-norm.gguf"
        copy_with_metadata(self.models["glm-dsa"], model, drop_tensors=[
            "blk.0.indexer.k_norm.bias", "blk.2.indexer.k_norm.weight", "blk.2.indexer.k_norm.bias"])
        spec = kvstate.ModelSpec.from_gguf(model)
        capture, state = self.capture_and_state(model, spec, ASCII)
        self.check_against_state(spec, capture, state)
        self.assertEqual(capture.layers("ik"), [0, 2])

    def test_mla_capture_requests_are_checked(self):
        record = {"document_id": document_id(ASCII), "text": ASCII}
        for model, request, fragment in (
                (self.models["deepseek2"], {"ik": "all"}, "capture key ik is only defined for deepseek32, glm-dsa and "
                                                          "dots3note"),
                (self.models["glm-dsa"], {"ik": [1]}, "layer 1 stores no indexer keys"),
                (self.models["deepseek2"], {"q": "all"}, "capture key q is only defined for")):
            with self.subTest(request=request):
                result, output = self.probe("collect", model, [record], spec=request)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(fragment, result.stderr)
                self.assertFalse(output.exists())
        # the MLA K row comes from two nodes: it can be collected, not patched
        directory = self.workdir()
        np.save(directory / "positions.npy", np.asarray([1], dtype=np.int32))
        np.save(directory / "k-0.npy", np.zeros((1, 1, 48), dtype=np.float32))
        text = ASCII
        scored = dict(record, expected_token_sha256=hashlib.sha256(
            np.frombuffer(text.encode(), dtype=np.uint8).astype("<i4").tobytes()).hexdigest(),
                      score_ranges=[[0, len(text)]], conditions=[{"name": "native", "patch_dir": None},
                                                                  {"name": "patched", "patch_dir": str(directory)}])
        result, output = self.probe("patch-score", self.models["deepseek2"], [scored])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("assembled from two graph nodes", result.stderr)

    def collect_v2(self, model, mode, text, *extra):
        """A published collect_v2.py capture directory of one document."""
        import collect_v2
        directory = self.workdir()
        docs = directory / "docs.jsonl"
        docs.write_text(json.dumps({"document_id": document_id(text), "text": text}) + "\n", encoding="utf-8")
        output = directory / mode
        with contextlib.redirect_stdout(io.StringIO()):
            collect_v2.main(["--probe", PROBE, "--mode", mode, "--model", str(model), "--model-id", "fixture",
                             "--docs", str(docs), "--output", str(output), "--ctx", "256", "--batch", "32",
                             "--threads", "2", "--skip-memory-check", *map(str, extra)])
        return Capture(output)

    def test_state_captures_equal_graph_captures(self):
        # collect_v2.py --mode states: the same views from the saved cache instead of graph nodes
        text = "Views from the cache: K de-rotated, V the latent, indexer keys too. " * 2
        for name in ("deepseek2-yarn", "deepseek32", "dots3note"):
            with self.subTest(model=name):
                model = self.models[name]
                spec = kvstate.ModelSpec.from_gguf(model)
                if name == "dots3note":         # sliding layers keep only their last n_swa cells in a state
                    with self.assertRaisesRegex(ValueError, "sliding layers"):
                        self.collect_v2(model, "states", text)
                    continue
                keys = {"k": "all", "v": "all", **({"ik": "all"} if spec.indexer else {})}
                spec_path = self.workdir() / "spec.json"
                spec_path.write_text(json.dumps(keys), encoding="utf-8")
                graph = self.collect_v2(model, "collect", text, "--capture-spec", spec_path)
                states = self.collect_v2(model, "states", text)
                self.assertEqual(states.layout(), graph.layout())
                self.assertEqual(states.indexer_layers(), graph.indexer_layers())
                np.testing.assert_array_equal(states.token_id, graph.token_id)
                for key, layer in graph.layout():
                    a, b = np.asarray(states.array(key, layer)), np.asarray(graph.array(key, layer))
                    if key == "v":
                        np.testing.assert_array_equal(a, b)
                    else:
                        np.testing.assert_allclose(a, b, rtol=0, atol=2e-6 * np.abs(b).max(), err_msg=f"{key}-{layer}")
                self.assertEqual(json.loads((Path(states.path) / "manifest.json").read_text())["mode"], "states")

    @unittest.skipUnless(LLAMA_ARCHS, "set KVPARK_LLAMA_ARCHS for the test-llama-archs hybrids")
    def test_state_captures_of_graphs_without_names(self):
        out = self.workdir()
        for architecture in ("kimi-k3", "bailingmoe3", "hy_v4"):
            with self.subTest(architecture=architecture):
                result = subprocess.run([LLAMA_ARCHS, "-a", architecture, "-s", "7", "-o", str(out)],
                                        capture_output=True, text=True, timeout=600)
                self.assertEqual(result.returncode, 0, result.stderr[-400:])
                model = copy_with_byte_vocabulary(next(out.glob(f"{architecture}-*.gguf")),
                                                  out / f"{architecture}.vocab.gguf")
                spec = kvstate.ModelSpec.from_gguf(model)
                capture = self.collect_v2(model, "states", ASCII)
                attention = [il for il in spec.attention_layers if spec.layer_types[il] != "nextn"]
                self.assertEqual(capture.layers("k"), attention)
                for layer in attention:
                    self.assertEqual(capture.array("k", layer).shape[1:], (1, spec.head_dim_k[layer]))
                    self.assertEqual(capture.array("v", layer).shape[1:], (1, spec.kv_lora[layer]))
                    np.testing.assert_array_equal(capture.array("v", layer),
                                                  capture.array("k", layer)[..., :spec.kv_lora[layer]])
                self.assertEqual(capture.layers("ik"), list(spec.indexer.layers) if spec.indexer else [])

    @unittest.skipUnless(LLAMA_ARCHS, "set KVPARK_LLAMA_ARCHS for the test-llama-archs hybrids")
    def test_only_lfm2_reads_the_sliding_window(self):
        # attention.sliding_window makes lfm2's attention layers slide (lfm2.cpp:26, covered below); the other
        # hybrids classified by their KV heads never read it, so their attention layers stay full
        out = self.workdir()
        for architecture in ("jamba", "kimi-linear"):
            with self.subTest(architecture=architecture):
                result = subprocess.run([LLAMA_ARCHS, "-a", architecture, "-s", "7", "-o", str(out)],
                                        capture_output=True, text=True, timeout=600)
                self.assertEqual(result.returncode, 0, result.stderr[-400:])
                vocab = copy_with_byte_vocabulary(next(out.glob(f"{architecture}-*.gguf")),
                                                  out / f"{architecture}.vocab.gguf")
                model = out / f"{architecture}.window.gguf"
                copy_with_metadata(vocab, model, values={f"{architecture}.attention.sliding_window": 16})
                spec = kvstate.ModelSpec.from_gguf(model)
                self.assertNotIn("sliding", spec.layer_types)
                capture, state = self.capture_and_state(model, spec, ASCII)
                self.check_against_state(spec, capture, state)

    @unittest.skipUnless(LLAMA_ARCHS, "set KVPARK_LLAMA_ARCHS for the test-llama-archs hybrids")
    def test_hybrid_attention_captures_match_the_saved_cache(self):
        out = self.workdir()
        for architecture in HYBRID_ARCHS + ("kimi-k3", "bailingmoe3", "hy_v4"):
            with self.subTest(architecture=architecture):
                result = subprocess.run([LLAMA_ARCHS, "-a", architecture, "-s", "7", "-o", str(out)],
                                        capture_output=True, text=True, timeout=600)
                self.assertEqual(result.returncode, 0, result.stderr[-400:])
                model = copy_with_byte_vocabulary(next(out.glob(f"{architecture}-*.gguf")),
                                                  out / f"{architecture}.vocab.gguf")
                spec = kvstate.ModelSpec.from_gguf(model)
                if architecture in HYBRID_ARCHS:
                    capture, state = self.capture_and_state(model, spec, ASCII)
                    self.check_against_state(spec, capture, state)
                    continue
                # kimi-k3, bailingmoe3 and hy_v4 name neither the latent row nor k_pe before RoPE
                result, _ = self.probe("collect", model, [{"document_id": document_id(ASCII), "text": ASCII}])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("capture key k is only defined for", result.stderr)


@unittest.skipUnless(PROBE and LLAMA_ARCHS, "set KVPARK_KV_PROBE and KVPARK_LLAMA_ARCHS for the DeepSeek V4 captures")
class KvProbeDsv4Tests(unittest.TestCase):
    """DeepSeek V4 (deepseek4) captures of test-llama-archs' model given a byte vocabulary: K is the raw
    latent before RoPE (kv_norm), and each compressed layer's compressor row [a | s] (the score before
    APE) of its CSA/LID (ratio 4) or HCA (ratio 128) pools, against the probe's own saved state: the
    window rows are the latents RoPE'd (kvstate), the rings the last tokens' a and s + APE; then the whole
    check_model_support run. The report's architecture is deepseek4, which captures.py does not list
    yet, so the files are read directly (and the check_model_support run extends its tables)."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-kv-probe-dsv4-")
        cls.root = Path(cls.temp.name)
        out = cls.root / "archs"
        out.mkdir()
        result = subprocess.run([LLAMA_ARCHS, "-a", "deepseek4", "-s", "7", "-o", str(out)], capture_output=True,
                                text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"test-llama-archs deepseek4: {result.stderr[-400:]}")
        cls.model = copy_with_byte_vocabulary(out / "deepseek4-moe.gguf", cls.root / "deepseek4-bytes.gguf")
        if not probe_supports(cls.model, cls.root):
            cls.temp.cleanup()
            raise unittest.SkipTest("this kvpark-kv-probe predates the DeepSeek V4 row; rebuild it")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def probe(self, mode, records, *extra):
        directory = Path(tempfile.mkdtemp(dir=self.root))
        docs = directory / "docs.jsonl"
        docs.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        result = subprocess.run([PROBE, "--mode", mode, "--model", str(self.model), "--input", str(docs), "--output",
                                 str(directory / "out"), "--ctx", "512", "--batch", "64", "--threads", "2",
                                 *map(str, extra)], capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return directory / "out"

    def test_latents_and_compressor_rows_match_the_saved_cache(self):
        text = "Pooled caches keep one row per block of four or of one hundred twenty-eight tokens. " * 3
        record = {"document_id": document_id(text), "text": text}
        output = self.probe("collect", [record])
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        n = report["rows"]
        self.assertGreater(n, 128)
        self.assertEqual([layer.get("compress", 0) for layer in report["layers"]], [0, 0, 4, 128])
        self.assertEqual({layer["attention"] for layer in report["layers"]}, {"sliding"})
        entries = {(c["key"], c["layer"]): c for c in report["captures"]}
        self.assertEqual(sorted(entries), [("csa", 2), ("hca", 3), ("k", 0), ("k", 1), ("k", 2), ("k", 3), ("lid", 2)])
        self.assertEqual(entries["csa", 2]["tensor"], "csa_state_kv-2+csa_state_score-2")
        self.assertEqual([(entries[key]["heads"], entries[key]["dim"]) for key in (("k", 0), ("csa", 2), ("hca", 3),
                                                                                  ("lid", 2))],
                         [(1, 64), (1, 256), (1, 128), (1, 512)])
        rows = {key: np.load(output / entry["file"]).reshape(n, -1) for key, entry in entries.items()}
        record["save_at"] = [n]
        saved = self.probe("save-state", [record])
        state_report = json.loads((saved / "report.json").read_text(encoding="utf-8"))
        spec = kvstate.ModelSpec.from_gguf(self.model)
        state = kvstate.read_file(saved / state_report["documents"][0]["states"][0]["file"], spec)
        swa = state.state.part("swa").stream()
        positions = np.asarray(swa.positions, dtype=np.int64)
        for layer in range(4):
            cached = spec.rope(layer).apply(rows["k", layer][positions].reshape(-1, 1, 64), positions)
            np.testing.assert_array_equal(swa.k[layer], cached.astype(swa.k[layer].dtype))   # the probe's f32 cache
        import gguf_tensors
        for pool in spec.pools():
            ring = state.state.part(pool.name + "_state")
            for layer in pool.layers:
                a, s = np.split(rows[pool.name, layer], 2, axis=1)
                ape_name = "indexer_compressor_ape" if pool.name == "lid" else "attn_compressor_ape"
                ape = gguf_tensors.read_tensor(self.model, f"blk.{layer}.{ape_name}.weight")
                last = np.arange(n - pool.state_size, n)
                np.testing.assert_array_equal(ring.kv[layer][0][last % pool.state_size], a[last])
                np.testing.assert_array_equal(ring.score[layer][0][last % pool.state_size],
                                              (s[last] + ape[last % pool.ratio]).astype(np.float32))
        # check_model_support's DeepSeek V4 checks on the same runs
        import check_model_support
        wanted = check_model_support.captured_layers(spec)
        self.assertEqual(sorted((key, layer) for key, layers in wanted.items() for layer in layers), sorted(entries))
        arrays = {key: np.load(output / entry["file"]) for key, entry in entries.items()}
        for key, array in arrays.items():
            self.assertEqual(array.shape, (n,) + check_model_support.captured_shape(spec, *key))
        data = (saved / state_report["documents"][0]["states"][0]["file"]).read_bytes()
        _, checks = check_model_support.check_state(spec, data, np.load(output / "token_id.npy"), arrays, "f32",
                                                    self.model)
        self.assertTrue(checks["round_trip_bytes_equal"] and checks["tokens_equal"] and checks["worst"]["rings_exact"])
        self.assertLessEqual(checks["worst"]["k_derot_max_ulps"], 1.0)
        # requests for a pool a layer does not have are refused
        directory = Path(tempfile.mkdtemp(dir=self.root))
        (directory / "spec.json").write_text(json.dumps({"hca": [2]}), encoding="utf-8")
        docs = directory / "docs.jsonl"
        docs.write_text(json.dumps(record) + "\n", encoding="utf-8")
        result = subprocess.run([PROBE, "--mode", "collect", "--model", str(self.model), "--input", str(docs),
                                 "--output", str(directory / "out"), "--capture-spec", str(directory / "spec.json")],
                                capture_output=True, text=True, timeout=300)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("has no hca compressor", result.stderr)

    def test_check_model_support_runs_end_to_end(self):
        # a 200-byte two-message document scored from its second message (byte tokens: 200 tokens)
        import check_model_support
        text = "".join(chr(33 + (7 * i) % 90) for i in range(200))
        record = {"document_id": document_id(text), "text": text, "segments": [[0, 0, 100], [1, 100, 200]],
                  "segment_keys": ["0:content", "1:content"], "split": "test", "final_message_segments": [1]}
        docs = self.root / "cms-docs.jsonl"
        docs.write_text(json.dumps(record) + "\n", encoding="utf-8")
        output = self.root / "cms"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            check_model_support.main(["--probe", PROBE, "--model", str(self.model), "--docs", str(docs), "--output",
                                      str(output), "--device", "cpu", "--batch", "64", "--threads", "2",
                                      "--skip-memory-check"])
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["document"]["seed_tokens"], summary["spec"]["memory"]), (64, "dsv4"))
        self.assertTrue(summary["tokenize_matches_collect"] and summary["state"]["round_trip_bytes_equal"])
        self.assertTrue(summary["state"]["worst"]["rings_exact"])
        self.assertLessEqual(summary["state"]["worst"]["k_derot_max_ulps"], 1.0)
        # the rebuilt raw window (RoPE'd collected latents) continues exactly as the saved state
        scores = output / "seed-score" / "scores" / record["document_id"]
        np.testing.assert_array_equal(np.load(scores / "rebuilt.npy"), np.load(scores / "seed.npy"))
        self.assertLess(summary["seed_score"]["seed"]["kl_max"], 1e-9)


@unittest.skipUnless(PROBE_VULKAN, "set KVPARK_KV_PROBE_VULKAN to enable Vulkan kv-probe checks")
class KvProbeVulkanTests(unittest.TestCase):
    """--device vulkan on tiny fixtures: every mode, GPU vs CPU values, exactness of same-device comparisons."""

    # Vulkan F32 matrix products stage operands in half precision; the fixture K/V differ from the CPU by
    # about 2e-3 of their largest magnitude (measured 1.6e-3 Muse, 2.0e-3 Qwen on RADV Strix Halo).
    GPU_CPU_REL = 1e-2

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-kv-probe-vulkan-")
        cls.root = Path(cls.temp.name)
        cls.models, cls.specs = {}, {}
        # muse-wide: a sliding window longer than the documents, so restored SWA cells keep their ring slots;
        # muse-fused: V is a strided view of the fused QKV output (device-memory patching of views)
        for name, extra in (("muse-glimmer", {}), ("qwen35", {}), ("qwen3", {}), ("muse-wide", {"sliding_window": 1024}),
                            ("muse-fused", {"fused_qkv": True})):
            path = cls.root / f"{name}.gguf"
            architecture = name if name in ("qwen35", "qwen3") else "muse-glimmer"
            create_fixture(path, architecture, control_tokens=CONTROL, **extra)
            cls.models[name] = path
            cls.specs[name] = kvstate.ModelSpec.from_gguf(path)
        cls.cache = {}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def workdir(self):
        return Path(tempfile.mkdtemp(dir=self.root))

    def run_probe(self, mode, model, records, output, *extra, device="vulkan", batch=16, binary=None):
        docs = output.parent / f"{output.name}.jsonl"
        docs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        command = [binary or PROBE_VULKAN, "--mode", mode, "--model", str(self.models[model]), "--input", str(docs),
                   "--output", str(output), "--ctx", "1024", "--threads", "2", "--batch", str(batch),
                   "--device", device, *map(str, extra)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        name = "results.json" if mode.endswith("score") else "report.json"
        return json.loads((output / name).read_text(encoding="utf-8"))

    def assert_on_gpu(self, report):
        self.assertEqual(report["settings"]["device"], "vulkan")
        device = report["device"]
        self.assertEqual((device["backend"], device["offload_kqv"]), ("vulkan", True))
        layers, total = device["offloaded_layers"].split("/")
        self.assertEqual(layers, total)
        # one CPU split (token embeddings) and one GPU split: no graph op fell back to the CPU,
        # including Qwen's GATED_DELTA_NET and SSM_CONV (fixture head size S_v = 32)
        self.assertEqual(device["graph_splits"], "2")

    def collect(self, model, text=CHAT, *, device="vulkan", batch=16, extra=(), binary=None):
        key = (model, text, device, batch, tuple(extra), binary)
        if key not in self.cache:
            output = self.workdir() / "capture"
            report = self.run_probe("collect", model, [{"document_id": document_id(text), "text": text}], output,
                                    *extra, device=device, batch=batch, binary=binary)
            arrays = {(c["key"], c["layer"]): np.load(output / c["file"]) for c in report["captures"]}
            self.cache[key] = (report, arrays, np.load(output / "token_id.npy"))
        return self.cache[key]

    def test_tokenize_needs_no_device_and_matches_cpu(self):
        outputs = {}
        for device in ("vulkan", "cpu"):
            output = self.workdir() / "tokens"
            report = self.run_probe("tokenize", "muse-glimmer", [{"document_id": document_id(CHAT), "text": CHAT}],
                                    output, device=device)
            self.assertNotIn("device", report)
            outputs[device] = {name: (output / name).read_bytes() for name in os.listdir(output)}
        self.assertEqual(outputs["vulkan"], outputs["cpu"])

    def test_collect_on_gpu_matches_cpu_within_tolerance(self):
        for model in ("muse-glimmer", "qwen35", "qwen3"):
            with self.subTest(model=model):
                report, gpu, tokens = self.collect(model)
                self.assert_on_gpu(report)
                cpu_report, cpu, cpu_tokens = self.collect(model, device="cpu")
                self.assertNotIn("device", cpu_report)
                self.assertNotIn("device", cpu_report["settings"])
                np.testing.assert_array_equal(tokens, cpu_tokens)
                self.assertEqual(sorted(gpu), sorted(cpu))
                worst = 0.0
                for name, expected in cpu.items():
                    scale = float(np.abs(expected).max())
                    error = float(np.abs(gpu[name] - expected).max()) / scale
                    worst = max(worst, error)
                    self.assertLess(error, self.GPU_CPU_REL, name)
                    self.assertTrue(np.isfinite(gpu[name]).all())
                print(f"\n[vulkan] {model} collect max |gpu - cpu| / max|cpu| = {worst:.3g}")
                # CPU runs of a Vulkan-enabled build equal the CPU-only build bit for bit
                if PROBE:
                    _, reference, _ = self.collect(model, device="cpu", binary=PROBE)
                    for name, expected in reference.items():
                        np.testing.assert_array_equal(cpu[name], expected)
                # chunking stays within GPU noise and every ubatch is captured on the device
                _, chunked, _ = self.collect(model, batch=7)
                for name, expected in gpu.items():
                    np.testing.assert_allclose(chunked[name], expected, atol=5e-3 * np.abs(expected).max(), rtol=0)

    def test_patch_score_sanity_is_exact_on_gpu(self):
        for model, layers in (("muse-glimmer", [0, 1, 2, 3]), ("muse-fused", [0, 1, 2, 3]), ("qwen35", [3]),
                              ("qwen3", [0, 1, 2, 3])):
            for batch in (8, 128):
                with self.subTest(model=model, batch=batch):
                    _, native, tokens = self.collect(model, batch=batch)
                    prefix = CHAT.index("<|im_start|>assistant")
                    n_prefix = sum(1 for end in expected_tokens(CHAT)[1] if end <= len(CHAT[:prefix].encode()))
                    positions = np.arange(1, n_prefix, 3)
                    directory = self.workdir()
                    dirs = {}
                    for name, arrays in (("sanity", {(key, layer): native[(key, layer)][positions]
                                                     for key in "kv" for layer in layers}),
                                         ("zero_v", {("v", 3): np.zeros_like(native[("v", 3)][positions])})):
                        path = directory / name
                        path.mkdir()
                        np.save(path / "positions.npy", positions.astype(np.int32))
                        for (key, layer), value in arrays.items():
                            np.save(path / f"{key}-{layer}.npy", np.ascontiguousarray(value, dtype=np.float32))
                        dirs[name] = str(path)
                    start, end = assistant_range(CHAT)
                    record = {"document_id": document_id(CHAT), "text": CHAT,
                              "expected_token_sha256": hashlib.sha256(tokens.astype("<i4").tobytes()).hexdigest(),
                              "score_ranges": [[start, end]],
                              "conditions": [{"name": "native", "patch_dir": None},
                                             {"name": "sanity", "patch_dir": dirs["sanity"]},
                                             {"name": "zero_v", "patch_dir": dirs["zero_v"]},
                                             {"name": "native_again", "patch_dir": None}]}
                    output = directory / "scores"
                    results = self.run_probe("patch-score", model, [record], output, batch=batch)
                    self.assert_on_gpu(results)
                    self.assertEqual(results["settings"]["callback_split_points"], "k,v of every attention layer")
                    document = results["documents"][0]
                    conditions = {c["name"]: c for c in document["conditions"]}
                    scores = {name: np.load(output / "scores" / document["document_id"] / f"{name}.npy")
                              for name in conditions}
                    self.assertGreater(len(document["scored_positions"]), 8)
                    np.testing.assert_array_equal(scores["native_again"], scores["native"])
                    np.testing.assert_array_equal(scores["sanity"], scores["native"])
                    self.assertEqual(conditions["sanity"]["kl_mean"], 0)
                    self.assertEqual(conditions["native"]["applied_rows"], {})
                    self.assertEqual(conditions["sanity"]["applied_rows"],
                                     {f"{key}-{layer}": len(positions) for key in "kv" for layer in layers})
                    self.assertEqual(conditions["zero_v"]["applied_rows"], {"v-3": len(positions)})
                    self.assertGreater(conditions["zero_v"]["kl_mean"], 0)

    def save_states(self, model, cache, flash, save_at=(64, 75)):
        record = {"document_id": document_id(CHAT), "text": CHAT, "save_at": list(save_at)}
        output = self.workdir() / "states"
        report = self.run_probe("save-state", model, [record], output, *layout_options(cache, flash))
        self.assert_on_gpu(report)
        return report, {s["n_tokens"]: output / s["file"] for s in report["documents"][0]["states"]}

    def test_save_state_f16_flash_attention_parses_and_matches_gpu_collect(self):
        for model in ("muse-glimmer", "qwen35", "qwen3"):
            with self.subTest(model=model):
                spec = self.specs[model]
                _, arrays, tokens = self.collect(model, extra=layout_options("f16", "on"))
                _, files = self.save_states(model, "f16", "on")
                data = files[75].read_bytes()
                parsed = kvstate.parse_file(data, spec)
                self.assertEqual(parsed.to_bytes(), data)
                np.testing.assert_array_equal(parsed.tokens, tokens[:75])
                for part in parsed.state.parts:
                    if isinstance(part, kvstate.RecurrentState):
                        self.assertEqual(part.positions.tolist(), [74])
                        continue
                    stream = part.stream()
                    self.assertFalse(stream.v_trans)
                    expected = np.arange(75 - 16, 75) if part.name == "swa" else np.arange(75)
                    np.testing.assert_array_equal(stream.positions, expected)
                    for layer in part.layers:
                        self.assertEqual(stream.k[layer].dtype, np.float16)
                        # the cache stores the GPU's own K/V: V bit-exact, K after the GPU RoPE
                        np.testing.assert_array_equal(stream.v[layer], arrays[("v", layer)][expected].astype(np.float16))
                        captured_k = arrays[("k", layer)][expected]
                        if model == "muse-glimmer" and spec.layer_types[layer] == "full":
                            np.testing.assert_array_equal(stream.k[layer], captured_k.astype(np.float16))
                            continue
                        derotated = kvstate.pre_rope_keys(spec, layer, stream.k[layer].astype(np.float32), expected)
                        np.testing.assert_allclose(derotated, captured_k, atol=4e-3, rtol=0)

    def test_seed_score_native_states_are_exact_on_gpu(self):
        for model in ("muse-glimmer", "muse-wide", "qwen35", "qwen3"):
            for cache, flash in (("f16", "on"), ("f32", "off")):
                with self.subTest(model=model, cache=cache, flash=flash):
                    _, files = self.save_states(model, cache, flash)
                    tokens = np.asarray(expected_tokens(CHAT)[0])
                    start, end = assistant_range(CHAT)
                    record = {"document_id": document_id(CHAT), "text": CHAT,
                              "expected_token_sha256": hashlib.sha256(tokens.astype("<i4").tobytes()).hexdigest(),
                              "score_ranges": [[start, end]],
                              "conditions": [{"name": "native", "state_file": None},
                                             {"name": "seed64", "state_file": str(files[64])},
                                             {"name": "seed75", "state_file": str(files[75])}]}
                    output = self.workdir() / "seed"
                    results = self.run_probe("seed-score", model, [record], output, *layout_options(cache, flash))
                    self.assert_on_gpu(results)
                    document = results["documents"][0]
                    conditions = {c["name"]: c for c in document["conditions"]}
                    scores = {name: np.load(output / "scores" / document["document_id"] / f"{name}.npy")
                              for name in conditions}
                    self.assertEqual(conditions["seed64"]["restored_pos_max"], 63)
                    self.assertEqual(conditions["seed64"]["restored_pos_min"],
                                     {"muse-glimmer": 48, "muse-wide": 0, "qwen35": 63, "qwen3": 0}[model])
                    if model == "muse-glimmer" and flash == "on":
                        # The state holds only the last window of SWA cells and restores them into ring slots
                        # 0..15 (native pass: 48..63). The Vulkan flash-attention kernel reduces over KV blocks, so
                        # other slots mean another summation order: KL ~1e-7, not 0 (muse-wide below is exact).
                        self.assertLess(float(scores["seed64"][:, 1].max()), 1e-6)
                        np.testing.assert_allclose(scores["seed64"][:, 0], scores["native"][:, 0], atol=2e-3, rtol=0)
                        self.assertEqual(conditions["seed64"]["top1_agree"], 1)
                    else:
                        # a seed on the --batch grid with cells in their native slots repeats the native
                        # ubatches: KL exactly 0
                        np.testing.assert_array_equal(scores["seed64"], scores["native"])
                        self.assertEqual(conditions["seed64"]["kl_mean"], 0)
                    # 75 splits a save-state ubatch, so the GPU sums in another order; random fixture logits
                    # have near ties, so top-1 agreement is not asserted
                    self.assertLess(float(scores["seed75"][:, 1].max()), 1e-6)
                    np.testing.assert_allclose(scores["seed75"][:, 0], scores["native"][:, 0], atol=2e-3, rtol=0)


if __name__ == "__main__":
    unittest.main()
