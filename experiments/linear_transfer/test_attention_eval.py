"""Numerical checks for the offline attention evaluation (RoPE, GQA, window, mapper, plans)."""

import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import attention_eval as ae
import fit_layers
from align import align
from captures import CaptureSet, write_synthetic_capture


# -- direct transcription of ggml-cpu/ops.cpp rope (GGML_ROPE_TYPE_NORMAL, forward) ------------

def _f32(value):
    return np.float32(value)


def ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow):
    def corr_dim(n_rot):
        return n_dims * math.log(n_ctx_orig / (n_rot * 2 * math.pi)) / (2 * math.log(freq_base))
    start = math.floor(corr_dim(beta_fast))
    end = math.ceil(corr_dim(beta_slow))
    return [max(0, start), min(n_dims - 1, end)]


def ggml_rope_yarn(theta_extrap, freq_scale, corr_dims, i0, ext_factor, mscale):
    theta_interp = _f32(freq_scale * theta_extrap)
    theta = theta_interp
    if ext_factor != 0.0:
        y = (i0 // 2 - corr_dims[0]) / max(0.001, corr_dims[1] - corr_dims[0])
        ramp_mix = (1 - min(1, max(0, y))) * ext_factor
        theta = _f32(theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix)
        mscale *= 1.0 + 0.1 * math.log(1.0 / freq_scale)
    return _f32(np.cos(np.float64(theta)) * mscale), _f32(np.sin(np.float64(theta)) * mscale)


def ggml_rope_cache_init(theta_base, freq_scale, corr_dims, ne0, ext_factor, mscale, sin_sign, theta_scale):
    cache = np.zeros(ne0, dtype=np.float32)
    theta = _f32(theta_base)
    for i0 in range(0, ne0, 2):
        cache[i0], cache[i0 + 1] = ggml_rope_yarn(theta, freq_scale, corr_dims, i0, ext_factor, mscale)
        cache[i0 + 1] *= sin_sign
        theta = _f32(theta * theta_scale)
    return cache


def ggml_rope_normal(src, pos, n_dims, freq_base, freq_scale=1.0, ext_factor=0.0, attn_factor=1.0,
                     beta_fast=32.0, beta_slow=1.0, n_ctx_orig=0, forward=True):
    """src [ne2 tokens, ne1 heads, ne0 dims] like a contiguous ggml tensor; loops as in ops.cpp."""
    ne2, ne1, ne0 = src.shape
    dst = np.zeros_like(src, dtype=np.float32)
    theta_scale = np.power(_f32(freq_base), _f32(-2.0) / _f32(n_dims), dtype=np.float32)
    corr_dims = ggml_rope_yarn_corr_dims(n_dims, max(n_ctx_orig, 1), freq_base, beta_fast, beta_slow)
    sin_sign = 1.0 if forward else -1.0
    for i2 in range(ne2):
        cache = ggml_rope_cache_init(pos[i2], freq_scale, corr_dims, ne0, ext_factor, attn_factor,
                                     sin_sign, theta_scale)
        for i1 in range(ne1):
            row, out = src[i2, i1], dst[i2, i1]
            # rotate_pairs<T>(n_dims, 1, cache, src + n_offs, dst + n_offs, scale=1): ic = i0
            for i0 in range(0, n_dims, 2):
                cos_theta, sin_theta = cache[i0], cache[i0 + 1]
                x0, x1 = _f32(row[i0]), _f32(row[i0 + 1])
                out[i0] = x0 * cos_theta - x1 * sin_theta
                out[i0 + 1] = x0 * sin_theta + x1 * cos_theta
            # fill the remaining channels from src (n_offs = 0)
            i0 = 0
            while i0 < ne0:
                if i0 == 0:
                    i0 += n_dims - 2
                else:
                    out[i0], out[i0 + 1] = row[i0], row[i0 + 1]
                i0 += 2
    return dst


# -- independent float64 references ---------------------------------------------------------

def reference_attention(q, k, v, window=None):
    """Dense float64 causal GQA attention; q head h uses kv head h // (Hq/Hkv)."""
    n, heads, _ = q.shape
    group = heads // k.shape[1]
    out = np.zeros(q.shape)
    for h in range(heads):
        kv = h // group
        scores = q[:, h].astype(np.float64) @ k[:, kv].astype(np.float64).T / math.sqrt(q.shape[2])
        i, j = np.indices((n, n))
        mask = j > i
        if window is not None:
            mask |= i - j >= window
        scores[mask] = -np.inf
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        out[:, h] = weights @ v[:, kv]
    return out


def reference_contribution(sdpa, gate, w_o, norm, eps=1e-8):
    x = (np.asarray(sdpa, np.float64) * gate).reshape(len(sdpa), -1) @ np.asarray(w_o, np.float64).T
    return x / np.sqrt(np.mean(x ** 2, axis=1, keepdims=True) + eps) * norm


# -- synthetic scenario shared with test_inject --------------------------------------------

PARAMS = ae.ModelParams(n_head=4, n_head_kv=2, head_dim=8, n_rot=8, rope_base=10000.0, window=6)
TGT_LAYERS = 8
TGT_TYPES = ["sliding" if layer % 4 < 3 else "full" for layer in range(TGT_LAYERS)]
SRC_LAYERS = (3, 7)
EVAL_LAYERS = (2, 3)
EMBED = 12
TARGET_TEMPLATE = ("<S>", "<E>", "<U>", "<E>", "<A>", "<E>")
SOURCE_TEMPLATE = ("[sys]", "[/]", "[usr]", "[/]", "[bot]", "[/]")


def doc_id(number):
    return hashlib.sha256(f"doc-{number}".encode()).hexdigest()


def render(template, messages):
    text, segments, offset = "", [], 0
    for index, message in enumerate(messages):
        opening, closing = template[2 * index], template[2 * index + 1]
        text += opening
        start = len(text.encode("utf-8"))
        text += message
        segments.append([index, start, len(text.encode("utf-8"))])
        text += closing
    return text, segments


def tokenize(text, markers, width):
    ends, special, index = [], [], 0
    markers = sorted(set(markers), key=len, reverse=True)
    while index < len(text):
        marker = next((m for m in markers if text.startswith(m, index)), None)
        if marker:
            index += len(marker)
            special.append(1)
        else:
            stop = index + 1
            while stop < len(text) and stop - index < width and not any(text.startswith(m, stop) for m in markers):
                stop += 1
            index = stop
            special.append(0)
        ends.append(len(text[:index].encode("utf-8")))
    return ends, special


def conversation(number):
    rng = np.random.default_rng(number)
    words = ["alpha", "beta", "gamma", "delta", "kappa", "omega", "sigma"]
    pick = lambda count: " ".join(rng.choice(words, count))  # noqa: E731
    return [f"You are helpful {number}.", pick(3 + number), pick(2 + number)]


def write_fit_mapper(directory, maps, *, source_shape, target_shape, source_architecture="qwen35",
                     target_architecture="muse-glimmer"):
    """Mapper directory in fit_layers' manifest schema.

    maps: [(kind, target, source_layers, k_renorm, weight, bias, train_mean)].
    """
    directory = Path(directory)
    directory.mkdir()
    entries = []
    for kind, target, sources, renorm, weight, bias, mean in maps:
        files = {"weight": f"W-{kind}-{target}.npy", "bias": f"b-{kind}-{target}.npy",
                 "mean": f"mean-{kind}-{target}.npy"}
        hashes = {}
        for name, array in (("weight", weight), ("bias", bias), ("mean", mean)):
            np.save(directory / files[name], np.asarray(array, dtype=np.float32))
            hashes[name] = hashlib.sha256((directory / files[name]).read_bytes()).hexdigest()
        entries.append({"kind": kind, "target": target, "source_layers": list(sources), "k": len(sources),
                        "alpha": 0.01, "lambda": 1.0, "k_renorm": bool(renorm), "files": files, "sha256": hashes})
    manifest = {"format_version": 1, "purpose": "kv_linear_transfer_mapper",
                "source": {"architecture": source_architecture,
                           "head_shape": {kind: list(source_shape) for kind in ("k", "v")}},
                "target": {"architecture": target_architecture,
                           "head_shape": {kind: list(target_shape) for kind in ("k", "v")}},
                "maps": entries}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def gguf_tensors_available():
    return Path(__file__).resolve().parents[2].joinpath("runtime/source/gguf-py").is_dir()


def make_scenario(root, splits=("test", "test", "dev"), seed=5, attn_out_error=0.0):
    """Synthetic Muse/Qwen captures, rendered records, alignment and mapper directory."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    tgt_records, src_records, tgt_docs, src_docs = [], [], [], []
    for number, split in enumerate(splits):
        messages = conversation(number)
        document = doc_id(number)
        for template, records, docs, width in ((TARGET_TEMPLATE, tgt_records, tgt_docs, 1),
                                               (SOURCE_TEMPLATE, src_records, src_docs, 2)):
            text, segments = render(template, messages)
            records.append({"document_id": document, "text": text, "segments": segments,
                            "segment_keys": ["0:content", "1:content", "2:content"], "split": split,
                            "final_message_segments": [2], "source": "synthetic"})
            ends, special = tokenize(text, template, width)
            docs.append({"document_id": document, "byte_end": ends, "special": special,
                         "token_id": [int(value) for value in rng.integers(0, 200, len(ends))]})
    n_tgt = sum(len(d["byte_end"]) for d in tgt_docs)
    n_src = sum(len(d["byte_end"]) for d in src_docs)
    tgt_arrays = {}
    for layer in range(TGT_LAYERS):
        for key in ("k", "v"):
            tgt_arrays[key, layer] = rng.normal(size=(n_tgt, 2, 8)).astype(np.float32)
    for layer in EVAL_LAYERS:
        tgt_arrays["q", layer] = rng.normal(size=(n_tgt, 4, 8)).astype(np.float32)
        tgt_arrays["gate", layer] = (1 / (1 + np.exp(-rng.normal(size=(n_tgt, 4, 8))))).astype(np.float32)
        out, first = np.zeros((n_tgt, 4, 8), dtype=np.float32), 0
        for document in tgt_docs:
            n = len(document["byte_end"])
            rows = slice(first, first + n)
            q, k, v = (tgt_arrays[key, layer][rows] for key in "qkv")
            window = None
            if TGT_TYPES[layer] == "sliding":
                q, k = (ggml_rope_normal(x, np.arange(n), PARAMS.n_rot, PARAMS.rope_base) for x in (q, k))
                window = PARAMS.window
            out[rows] = reference_attention(q, k, v, window)
            first += n
        out[0] += attn_out_error
        tgt_arrays["attn_out", layer] = out
    src_arrays = {(key, layer): rng.normal(size=(n_src, 1, 16)).astype(np.float32)
                  for key in ("k", "v") for layer in SRC_LAYERS}
    tgt = write_synthetic_capture(root / "tgt", tgt_docs, tgt_arrays, n_layer=TGT_LAYERS,
                                  manifest={"model_sha256": "a" * 64})
    src = write_synthetic_capture(root / "src", src_docs, src_arrays, architecture="qwen35",
                                  n_layer=TGT_LAYERS, manifest={"model_sha256": "b" * 64})
    maps = []
    for layer in range(TGT_LAYERS):
        for kind in ("k", "v"):
            sources = list(SRC_LAYERS) if layer % 2 else [7]
            maps.append((kind, layer, sources, kind == "k",
                         rng.normal(size=(16 * len(sources), 16)).astype(np.float32),
                         rng.normal(size=16).astype(np.float32), rng.normal(size=16).astype(np.float32)))
    mapper = write_fit_mapper(root / "mapper", maps, source_shape=(1, 16), target_shape=(2, 8))
    tgt_set, src_set = CaptureSet([tgt]), CaptureSet([src])
    alignment = align(src_records, src_set, tgt_records, tgt_set)
    weights = {layer: (rng.normal(size=(EMBED, 32)).astype(np.float32),
                       rng.uniform(.5, 1.5, EMBED).astype(np.float32)) for layer in EVAL_LAYERS}
    return {"tgt": tgt, "src": src, "tgt_set": tgt_set, "src_set": src_set, "tgt_records": tgt_records,
            "src_records": src_records, "alignment": alignment, "mapper": mapper, "weights": weights,
            "tgt_arrays": tgt_arrays, "src_arrays": src_arrays}


class RopeTests(unittest.TestCase):
    def test_rope_matches_ggml_cpu_transcription(self):
        rng = np.random.default_rng(3)
        for n_rot, dim, base in ((8, 8, 10000.0), (16, 16, 5e5), (8, 12, 5e5)):
            with self.subTest(n_rot=n_rot, dim=dim):
                x = rng.normal(size=(7, 3, dim)).astype(np.float32)
                positions = np.array([0, 1, 2, 17, 511, 2047, 4999])
                expected = ggml_rope_normal(x, positions, n_rot, base)
                actual = ae.rope_norm(x, positions, n_rot=n_rot, base=base)
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
                np.testing.assert_array_equal(actual[:, :, n_rot:], x[:, :, n_rot:])

    def test_muse_sized_rope_matches_transcription_at_long_positions(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(4, 2, 128)).astype(np.float32)
        positions = np.array([3, 2048, 4095, 5119])
        expected = ggml_rope_normal(x, positions, 128, 5e5)
        np.testing.assert_allclose(ae.rope_norm(x, positions, n_rot=128, base=5e5), expected,
                                   rtol=2e-6, atol=2e-6)

    def test_angles_follow_ggml_float32_recurrence_and_the_closed_form(self):
        positions = np.array([0, 5, 4000])
        angles = ae.rope_angles(positions, 128, 5e5)
        theta_scale = np.power(np.float32(5e5), np.float32(-2.0) / np.float32(128), dtype=np.float32)
        for row, position in enumerate(positions):
            theta = np.float32(position)
            for index in range(64):
                self.assertEqual(angles[row, index], theta)
                theta = np.float32(theta * theta_scale)
        closed = positions[:, None] * 5e5 ** (-2 * np.arange(64) / 128)
        np.testing.assert_allclose(angles, closed, rtol=2e-5, atol=1e-6)

    def test_adjacent_pairs_are_rotated_and_scores_depend_on_relative_position(self):
        rng = np.random.default_rng(5)
        q, k = rng.normal(size=(1, 1, 8)), rng.normal(size=(1, 1, 8))
        one = ae.rope_norm(np.zeros((1, 1, 8)) + [[[1, 0, 0, 0, 0, 0, 0, 0]]], [1], n_rot=8, base=10000.0)
        np.testing.assert_allclose(one[0, 0, :2], [math.cos(1), math.sin(1)], atol=1e-6)
        dots = [float(np.sum(ae.rope_norm(q, [p + 3], n_rot=8, base=100.0)
                             * ae.rope_norm(k, [p], n_rot=8, base=100.0))) for p in (0, 10, 40)]
        np.testing.assert_allclose(dots, dots[0], rtol=1e-4)

    def test_invalid_rope_arguments(self):
        with self.assertRaises(ValueError):
            ae.rope_norm(np.zeros((2, 1, 4)), [0, 1], n_rot=6, base=10.0)
        with self.assertRaises(ValueError):
            ae.rope_angles([0], 3, 10.0)


class AttentionTests(unittest.TestCase):
    def test_gqa_window_and_query_chunks_match_dense_reference(self):
        rng = np.random.default_rng(6)
        q = rng.normal(size=(23, 6, 5)).astype(np.float32)
        k, v = (rng.normal(size=(23, 2, 5)).astype(np.float32) for _ in range(2))
        for window in (None, 1, 4, 30):
            expected = reference_attention(q, k, v, window)
            for chunk in (1, 5, 64):
                with self.subTest(window=window, chunk=chunk):
                    actual = ae.causal_attention(q, k, v, scale=1 / math.sqrt(5), window=window, chunk=chunk)
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            subset = np.array([0, 3, 9, 22])
            actual = ae.causal_attention(q, k, v, scale=1 / math.sqrt(5), window=window,
                                         query_index=subset, chunk=3, dtype=np.float64)
            np.testing.assert_allclose(actual, expected[subset], rtol=1e-6, atol=1e-6)

    def test_query_head_group_mapping(self):
        q = np.zeros((2, 4, 2), dtype=np.float32)
        k = np.zeros((2, 2, 2), dtype=np.float32)
        v = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 2.0)], axis=1).astype(np.float32)
        out = ae.causal_attention(q, k, v, scale=1.0)
        np.testing.assert_array_equal(out[:, :, 0], [[1, 1, 2, 2], [1, 1, 2, 2]])
        with self.assertRaises(ValueError):
            ae.causal_attention(np.zeros((2, 3, 2)), k, v, scale=1.0)

    def test_window_masks_keys_at_distance_window(self):
        q = np.zeros((5, 1, 1), dtype=np.float32)
        k = np.zeros((5, 1, 1), dtype=np.float32)
        v = np.arange(5, dtype=np.float32).reshape(5, 1, 1)
        out = ae.causal_attention(q, k, v, scale=1.0, window=2)
        np.testing.assert_allclose(out[:, 0, 0], [0, 0.5, 1.5, 2.5, 3.5])

    def test_contribution_matches_reference(self):
        rng = np.random.default_rng(7)
        sdpa, gate = rng.normal(size=(9, 4, 8)), rng.uniform(size=(9, 4, 8))
        w_o, norm = rng.normal(size=(12, 32)), rng.uniform(.5, 1.5, 12)
        actual = ae.attention_contribution(sdpa.astype(np.float32), gate.astype(np.float32),
                                           w_o.astype(np.float32), norm.astype(np.float32), chunk=4)
        np.testing.assert_allclose(actual, reference_contribution(sdpa, gate, w_o, norm), rtol=1e-4, atol=1e-5)
        with self.assertRaises(ValueError):
            ae.attention_contribution(sdpa, gate, w_o[:, :30], norm)

    def test_affected_queries(self):
        np.testing.assert_array_equal(ae.affected_queries(10, np.array([2, 4]), None), np.arange(2, 10))
        np.testing.assert_array_equal(ae.affected_queries(10, np.array([2, 4]), 3), [2, 3, 4, 5, 6])
        self.assertEqual(len(ae.affected_queries(10, np.array([], dtype=np.int64), 3)), 0)


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_plans_cover_prefix_aligned_tokens_and_final_message(self):
        s = make_scenario(self.root)
        plans, skipped = ae.plan_documents(s["alignment"], s["tgt_records"], s["tgt_set"],
                                           s["src_records"], s["src_set"], {"test"})
        self.assertEqual(skipped, [])
        self.assertEqual([plan.document_id for plan in plans], [doc_id(0), doc_id(1)])
        for plan, record in zip(plans, s["tgt_records"]):
            final_start = record["segments"][2][1]
            ends = s["tgt_set"].byte_end[s["tgt_set"].rows(plan.document_id)]
            self.assertEqual(plan.prefix_tokens, int(np.sum(ends <= final_start)))
            self.assertTrue(np.all(plan.positions < plan.prefix_tokens))
            self.assertTrue(np.all(np.diff(plan.positions) > 0))
            # Char-level target vs two-char source: aligned tokens end at even segment offsets
            # or at a segment end.
            self.assertGreater(len(plan.positions), 5)
            self.assertEqual(plan.system.sum(), np.sum(plan.segments == 0))
            self.assertEqual(plan.score_ranges, [record["segments"][2][1:]])
            self.assertEqual(len(plan.continuation), len(conversation(int(plan.document_id == doc_id(1)))[2]))
        dev, _ = ae.plan_documents(s["alignment"], s["tgt_records"], s["tgt_set"],
                                   s["src_records"], s["src_set"], {"dev"})
        self.assertEqual([plan.document_id for plan in dev], [doc_id(2)])

    def test_alignment_from_other_captures_is_rejected(self):
        s = make_scenario(self.root)
        s["alignment"].rel_end[0] += 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            ae.plan_documents(s["alignment"], s["tgt_records"], s["tgt_set"], s["src_records"],
                              s["src_set"], {"test", "dev"})

    def test_mapper_prediction_renormalization_and_validation(self):
        s = make_scenario(self.root)
        mapper = ae.Mapper(s["mapper"])
        rows = {3: np.ones((2, 1, 16), np.float32), 7: np.full((2, 1, 16), 2, np.float32)}
        w = np.load(s["mapper"] / "W-v-1.npy")
        b = np.load(s["mapper"] / "b-v-1.npy")
        expected = (np.concatenate([rows[3].reshape(2, 16), rows[7].reshape(2, 16)], axis=1) @ w + b)
        np.testing.assert_allclose(mapper.predict("v", 1, rows, (2, 8)), expected.reshape(2, 2, 8), rtol=1e-6)
        k = mapper.predict("k", 1, rows, (2, 8))
        np.testing.assert_allclose(np.sqrt(np.mean(k.astype(np.float64) ** 2, axis=-1)), 1, rtol=1e-6)
        np.testing.assert_array_equal(mapper.mean("v", 1, (2, 8)),
                                      np.load(s["mapper"] / "mean-v-1.npy").reshape(2, 8))
        np.testing.assert_array_equal(k, fit_layers.apply(mapper.layer_map("k", 1), rows))
        self.assertEqual(mapper.source_layers("v", 2), ("v", [7]))
        with self.assertRaises(KeyError):
            mapper.predict("v", 1, {3: rows[3]}, (2, 8))
        with self.assertRaises(KeyError):
            mapper.entry("v", 99)
        with self.assertRaisesRegex(ValueError, "head shape"):
            mapper.predict("v", 1, rows, (1, 16))
        mapper.clear()
        self.assertEqual(mapper._maps, {})

        original = (s["mapper"] / "manifest.json").read_text()

        def rejected(edit, pattern):
            manifest = json.loads(original)
            edit(manifest)
            (s["mapper"] / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, pattern):
                ae.Mapper(s["mapper"])

        rejected(lambda m: m["maps"].append(dict(m["maps"][0])), "duplicate")
        rejected(lambda m: m.update(purpose="other"), "not a fit_layers mapper")
        rejected(lambda m: m.update(mappers=m.pop("maps")), "nonempty 'maps'")
        rejected(lambda m: m["maps"][0]["files"].update(weight="../W-k-0.npy"), "invalid mapper entry")
        rejected(lambda m: m["maps"][1].update(k_renorm=True), "invalid mapper entry")
        rejected(lambda m: m["target"]["head_shape"].update(k=[2]), "head shape")
        (s["mapper"] / "manifest.json").write_text(original)
        # A file verified once is not re-hashed on reload, but a replaced file is.
        mapper = ae.Mapper(s["mapper"])
        mapper.predict("v", 1, rows, (2, 8))
        mapper.clear()
        np.testing.assert_allclose(mapper.predict("v", 1, rows, (2, 8)), expected.reshape(2, 2, 8), rtol=1e-6)
        mapper.clear()
        np.save(s["mapper"] / "b-v-1.npy", np.zeros(16, np.float32))
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            mapper.predict("v", 1, rows, (2, 8))
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            ae.Mapper(s["mapper"]).predict("v", 1, rows, (2, 8))

    def test_reads_the_mapper_fit_layers_writes(self):
        s = make_scenario(self.root, splits=("train", "train", "train", "dev", "test"))
        s["alignment"].save(self.root / "alignment.npz")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fit_layers.main(["--src-capture", str(s["src"]), "--tgt-capture", str(s["tgt"]), "--alignment",
                             str(self.root / "alignment.npz"), "--output", str(self.root / "fit"), "--ks", "1,2",
                             "--threads", "1"])
        mapper, reference = ae.Mapper(self.root / "fit"), fit_layers.load_mapper(self.root / "fit")
        self.assertEqual(set(mapper.entries), set(reference.maps))
        self.assertEqual((mapper.source_architecture, mapper.target_architecture), ("qwen35", "muse-glimmer"))
        plans, _ = ae.plan_documents(s["alignment"], s["tgt_records"], s["tgt_set"], s["src_records"],
                                     s["src_set"], {"test"})
        for (kind, layer), layer_map in reference.maps.items():
            predicted = ae.mapped_rows(mapper, kind, layer, plans[0], s["src_set"], (2, 8))
            rows = plans[0].src_start + plans[0].src_positions
            expected = fit_layers.apply(layer_map, {source: s["src_set"].gather(kind, source, rows)
                                                    for source in layer_map.source_layers})
            np.testing.assert_array_equal(predicted, expected)
            np.testing.assert_array_equal(mapper.mean(kind, layer, (2, 8)), layer_map.train_mean.reshape(2, 8))
        ae.check_mapper(mapper, PARAMS, list(EVAL_LAYERS), s["src_set"])
        with self.assertRaisesRegex(ValueError, "head shape"):
            ae.check_mapper(mapper, dataclasses.replace(PARAMS, head_dim=4), [2], s["src_set"])

    def test_substituting_native_values_changes_nothing(self):
        s = make_scenario(self.root)
        plans, _ = ae.plan_documents(s["alignment"], s["tgt_records"], s["tgt_set"],
                                     s["src_records"], s["src_set"], {"test"})
        plan = plans[0]
        rows = np.arange(plan.tgt_start, plan.tgt_start + plan.n_tokens)
        for layer in EVAL_LAYERS:
            q, k, v, gate = (s["tgt_set"].gather(key, layer, rows) for key in ("q", "k", "v", "gate"))
            same = (k[plan.positions], v[plan.positions])
            error, results = ae.compare_layer(
                q, k, v, plan.positions, {"kv": same}, params=PARAMS, sliding=TGT_TYPES[layer] == "sliding",
                query_sets={"all": np.arange(plan.n_tokens), "continuation": plan.continuation},
                gate=gate, weights=s["weights"][layer], attn_out=s["tgt_set"].gather("attn_out", layer, rows))
            self.assertLess(error, 1e-5)
            for acc in results["kv"].values():
                result = acc.result()
                # Only query-block boundaries differ from the native pass (last-bit BLAS effects).
                self.assertAlmostEqual(result["sdpa_head_cosine_mean"], 1.0, places=12)
                self.assertAlmostEqual(result["contribution_cosine_mean"], 1.0, places=12)
                self.assertLess(result["contribution_relative_error_pooled"], 1e-6)

    def test_substitution_metrics_match_direct_computation(self):
        s = make_scenario(self.root)
        plans, _ = ae.plan_documents(s["alignment"], s["tgt_records"], s["tgt_set"],
                                     s["src_records"], s["src_set"], {"test"})
        plan = plans[1]
        rows = np.arange(plan.tgt_start, plan.tgt_start + plan.n_tokens)
        rng = np.random.default_rng(9)
        for layer in EVAL_LAYERS:
            sliding = TGT_TYPES[layer] == "sliding"
            q, k, v, gate = (s["tgt_set"].gather(key, layer, rows).astype(np.float64)
                             for key in ("q", "k", "v", "gate"))
            new_k = rng.normal(size=(len(plan.positions), 2, 8))
            new_v = rng.normal(size=(len(plan.positions), 2, 8))
            k2, v2 = k.copy(), v.copy()
            k2[plan.positions], v2[plan.positions] = new_k, new_v
            n = plan.n_tokens

            def run(kk, vv):
                qq = q
                if sliding:
                    qq = ggml_rope_normal(q.astype(np.float32), np.arange(n), 8, PARAMS.rope_base)
                    kk = ggml_rope_normal(kk.astype(np.float32), np.arange(n), 8, PARAMS.rope_base)
                return reference_attention(qq, kk, vv, PARAMS.window if sliding else None)

            native, mapped = run(k, v), run(k2, v2)
            head_cos = np.sum(native * mapped, -1) / np.linalg.norm(native, axis=-1) / np.linalg.norm(mapped, axis=-1)
            w_o, norm = s["weights"][layer]
            a, b = reference_contribution(native, gate, w_o, norm), reference_contribution(mapped, gate, w_o, norm)
            cont = plan.continuation
            _, results = ae.compare_layer(
                q, k, v, plan.positions, {"kv": (new_k, new_v)}, params=PARAMS, sliding=sliding,
                query_sets={"all": np.arange(n), "continuation": cont}, gate=gate.astype(np.float32),
                weights=s["weights"][layer], chunk=7)
            for name, index in (("all", np.arange(n)), ("continuation", cont)):
                result = results["kv"][name].result()
                self.assertEqual(result["queries"], len(index))
                self.assertAlmostEqual(result["sdpa_head_cosine_mean"], head_cos[index].mean(), places=5)
                cos = np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
                self.assertAlmostEqual(result["contribution_cosine_mean"], cos[index].mean(), places=5)
                rel = np.linalg.norm(b - a, axis=1) / np.linalg.norm(a, axis=1)
                self.assertAlmostEqual(result["contribution_relative_error_mean"], rel[index].mean(), places=4)
                pooled = np.linalg.norm(b[index] - a[index]) / np.linalg.norm(a[index])
                self.assertAlmostEqual(result["contribution_relative_error_pooled"], pooled, places=4)
            if sliding:
                # Queries beyond the window of the last patched key are untouched.
                unaffected = np.setdiff1d(np.arange(n), ae.affected_queries(n, plan.positions, PARAMS.window))
                np.testing.assert_allclose(head_cos[unaffected], 1, atol=1e-12)

    def test_query_sets_and_floor(self):
        s = make_scenario(self.root)
        plans, _ = ae.plan_documents(s["alignment"], s["tgt_records"], s["tgt_set"],
                                     s["src_records"], s["src_set"], {"test"})
        plan = plans[0]
        sets = ae.query_sets(plan, PARAMS.window, 3)
        self.assertEqual(sorted(sets), sorted(ae.QUERY_SETS))
        np.testing.assert_array_equal(sets["scored"], plan.continuation[:3] - 1)
        np.testing.assert_array_equal(sets["continuation"], plan.continuation)
        self.assertTrue(set(sets["affected"]) <= set(plan.continuation))
        visible = ae.affected_queries(plan.n_tokens, plan.positions, PARAMS.window)
        np.testing.assert_array_equal(sets["affected"], np.intersect1d(plan.continuation, visible))
        far = dataclasses.replace(plan, positions=plan.positions[:1])
        self.assertEqual(len(ae.query_sets(far, 1, 3)["affected"]), 0)
        mapper = ae.Mapper(s["mapper"])
        floor = mapper.floor("k", 1, (2, 8))
        np.testing.assert_allclose(np.sqrt(np.mean(floor.astype(np.float64) ** 2, axis=-1)), 1, rtol=1e-6)
        np.testing.assert_array_equal(mapper.floor("v", 1, (2, 8)), mapper.mean("v", 1, (2, 8)))

    def test_evaluate_validates_before_reporting(self):
        s = make_scenario(self.root)
        plans, _ = ae.plan_documents(s["alignment"], s["tgt_records"], s["tgt_set"],
                                     s["src_records"], s["src_set"], {"test", "dev"})
        mapper = ae.Mapper(s["mapper"])
        sections = ae.evaluate(plans, s["tgt_set"], s["src_set"], mapper, PARAMS, list(EVAL_LAYERS), TGT_TYPES,
                               ["kv", "k", "v", "mean_kv"], load_weights=lambda layer: s["weights"][layer])
        self.assertEqual([item["layer"] for item in sections["validation"]], [2, 3])
        self.assertTrue(all(item["max_relative_error"] < 1e-5 for item in sections["validation"]))
        keys = {(item["split"], item["layer"], item["variant"], item["query_set"]) for item in sections["layers"]}
        self.assertIn(("test", 3, "mean_kv", "continuation"), keys)
        self.assertEqual({key[3] for key in keys}, set(ae.QUERY_SETS))
        self.assertIn(("dev", 2, "k", "all"), keys)
        self.assertEqual({key[0] for key in keys}, {"test", "dev"})
        self.assertEqual(sections["layers"][0]["split"], "test")
        for item in sections["layers"]:
            self.assertLess(item["sdpa_head_cosine_mean"], 1.0)
            self.assertGreater(item["contribution_relative_error_pooled"], 0)
        json.dumps(sections, allow_nan=False)

        (self.root / "broken").mkdir()
        broken = make_scenario(self.root / "broken", attn_out_error=1.0)
        plans, _ = ae.plan_documents(broken["alignment"], broken["tgt_records"], broken["tgt_set"],
                                     broken["src_records"], broken["src_set"], {"test"})
        with self.assertRaisesRegex(ValueError, "does not reproduce captured attn_out"):
            ae.evaluate(plans, broken["tgt_set"], broken["src_set"], ae.Mapper(broken["mapper"]), PARAMS,
                        [3], TGT_TYPES, ["kv"])


class CommandLineTests(unittest.TestCase):
    def test_main_writes_a_new_report_with_validation_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            s = make_scenario(root)
            for name in ("tgt", "src"):
                (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in s[f"{name}_records"]))
            s["alignment"].save(root / "alignment.npz")
            (root / "muse.gguf").write_bytes(b"not a model")

            class Hashes:
                def __init__(self, path):
                    pass

                def sha256(self, path):
                    return "a" * 64

            argv = ["--model", str(root / "muse.gguf"), "--tgt-docs", str(root / "tgt.jsonl"), "--tgt-capture",
                    str(s["tgt"]), "--src-docs", str(root / "src.jsonl"), "--src-capture", str(s["src"]),
                    "--alignment", str(root / "alignment.npz"), "--mapper", str(s["mapper"]),
                    "--output", str(root / "report.json"), "--variants", "kv,mean_kv", "--chunk", "5"]
            with patch.object(ae, "load_model_params", lambda model: PARAMS), \
                    patch.object(ae, "load_output_weights", lambda model, layer: s["weights"][layer]), \
                    patch("collect_v2.ModelHashCache", Hashes), contextlib.redirect_stdout(io.StringIO()):
                ae.main(argv)
                report = json.loads((root / "report.json").read_text())
                self.assertEqual(report["settings"]["layers"], [2, 3])
                self.assertEqual([item["layer"] for item in report["validation"]], [2, 3])
                self.assertEqual(report["unvalidated_attention_types"], [])
                self.assertEqual(report["headline_split"], "test")
                self.assertEqual(report["inputs"]["model_sha256"], "a" * 64)
                self.assertEqual(len(report["documents"]), 3)
                with self.assertRaises(FileExistsError):
                    ae.main(argv)
                argv[argv.index(str(root / "report.json"))] = str(root / "other.json")
                Hashes.sha256 = lambda self, path: "f" * 64
                with self.assertRaisesRegex(ValueError, "differs from the model"):
                    ae.main(argv)
                Hashes.sha256 = lambda self, path: "a" * 64
                with self.assertRaisesRegex(ValueError, "needs captured q"):
                    ae.main(argv + ["--layers", "4"])
            self.assertFalse((root / "other.json").exists())


@unittest.skipUnless(gguf_tensors_available(), "pinned gguf-py is not available")
class GgufTests(unittest.TestCase):
    def test_model_params_and_output_weights_from_tiny_fixture(self):
        from tiny_fixture import create_fixture
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "muse.gguf"
            tensors = create_fixture(path, "muse-glimmer", sliding_window=16)
            params = ae.load_model_params(path)
            self.assertEqual((params.n_head, params.n_head_kv, params.head_dim, params.n_rot, params.window),
                             (2, 2, 32, 32, 16))
            self.assertEqual(params.rope_base, 500000.0)
            w_o, norm = ae.load_output_weights(path, 1)
            np.testing.assert_array_equal(w_o, tensors["blk.1.attn_output.weight"])
            np.testing.assert_array_equal(norm, tensors["blk.1.post_attention_norm.weight"])
            qwen = Path(temporary) / "qwen.gguf"
            create_fixture(qwen, "qwen35")
            with self.assertRaisesRegex(ValueError, "not muse-glimmer"):
                ae.load_model_params(qwen)


PROBE = os.environ.get("KVPARK_KV_PROBE")


@unittest.skipUnless(PROBE, "set KVPARK_KV_PROBE to check numpy attention against the real Muse graph")
class RuntimeAttentionTests(unittest.TestCase):
    def test_numpy_attention_reproduces_captured_attn_out_on_sliding_and_full_layers(self):
        from tiny_fixture import create_fixture
        text = ("A longer document so that the sixteen token sliding window is crossed several times; "
                "RoPE NORM pairs adjacent dims and the full layer uses no positions at all.")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "muse.gguf"
            create_fixture(model, "muse-glimmer", sliding_window=16)
            document = hashlib.sha256(text.encode()).hexdigest()
            (root / "docs.jsonl").write_text(json.dumps({"document_id": document, "text": text}) + "\n")
            spec = {"k": "all", "v": "all", "q": "all", "gate": "all", "attn_out": "all"}
            (root / "spec.json").write_text(json.dumps(spec))
            result = subprocess.run([PROBE, "--mode", "collect", "--model", str(model), "--input",
                                     str(root / "docs.jsonl"), "--output", str(root / "capture"),
                                     "--ctx", "1024", "--batch", "37", "--threads", "2",
                                     "--capture-spec", str(root / "spec.json")],
                                    capture_output=True, text=True, timeout=300)
            self.assertEqual(result.returncode, 0, result.stderr)
            capture = CaptureSet([root / "capture"])
            params = ae.load_model_params(model)
            n = capture.rows_total
            self.assertGreater(n, 3 * params.window)
            for layer in range(4):
                sliding = capture.captures[0].layer_types[layer] == "sliding"
                q, k, v, out = (np.array(capture.captures[0].array(key, layer))
                                for key in ("q", "k", "v", "attn_out"))
                error, _ = ae.compare_layer(q, k, v, np.zeros(0, dtype=np.int64), {}, params=params,
                                            sliding=sliding, query_sets={}, attn_out=out)
                with self.subTest(layer=layer, sliding=sliding):
                    self.assertLess(error, ae.VALIDATION_TOLERANCE)
                if sliding:
                    for wrong in (dataclasses.replace(params, window=n + 1),
                                  dataclasses.replace(params, rope_base=1e4)):
                        error, _ = ae.compare_layer(q, k, v, np.zeros(0, dtype=np.int64), {}, params=wrong,
                                                    sliding=True, query_sets={}, attn_out=out)
                        self.assertGreater(error, 1e-2)


if __name__ == "__main__":
    unittest.main()
