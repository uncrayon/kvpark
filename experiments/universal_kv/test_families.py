"""The MLA/DSA family and the K-only hybrids end to end on tiny fixtures, through llama.cpp itself.

  IdentityTranslationTests  a model translated into itself through an exact identity hub
                            (fixtures_families.identity_hub) writes exactly the state kvstate's
                            own operations make of the native one (de-rotate, latent shell,
                            re-rotate, indexer keys, R0), and it restores (the state driver's
                            restore-decode) to the native logits: deepseek2 (YaRN on and off,
                            nextn), mistral4 (YaRN), deepseek32 and glm-dsa (indexer keys zero
                            while the prompt fits in top_k cells; above that the indexer ridge),
                            and amplified test-llama-archs models: hy_v4, the KDA hybrids
                            (kimi-linear, kimi-k3, bailingmoe3) and the GQA hybrids, whose
                            translated attention rows restore to the native logits next to the
                            native recurrent rows (R0 aside; the logits of the qwen4exp
                            fixture, and on 85d5703a3 those of qwen3next and qwen35moe, barely
                            hear attention even amplified, so only their states are checked),
                            qwen4exp's block indexer above top_k (the ridge); kvstate's RoPE against the rows llama.cpp
                            stores, for every model whose first layer is attention;
  CrossFamilyTests          a hub fitted (fit_hub.py) on kv_probe captures of a qwen3 fixture
                            and MLA/DSA fixtures translates between the families: shapes,
                            positions, restore and decoding (random weights: no quality);
  EngineJobTests            engine.py's one-shot path (a job file) into an MLA target, with the
                            server's RoPE settings, and the refusal of a DSA target without
                            indexer keys.

Needs KVPARK_STATE_DRIVER (kvpark-state-driver), KVPARK_LLAMA_ARCHS (test-llama-archs) for the
archs models and KVPARK_KV_PROBE (kvpark-kv-probe) for the captures; tests skip without them.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import convert_hub
import fit_hub
import fixtures_families as ff
import hub as hub_module
import state_writer
import translate
from test_resource_guard import guard_arguments
from test_translate import StubSpans, StubView, kind_widths, random_hub

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "linear_transfer"))
import kvstate  # noqa: E402
from tiny_fixture import copy_with_metadata, create_fixture  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNTIME_YARN = {"rope_scaling_type": "yarn", "rope_freq_scale": 0.125, "yarn_orig_ctx": 32}
RUNTIME_YARN_FLAGS = ["--rope-scaling", "yarn", "--rope-freq-scale", "0.125", "--yarn-orig-ctx", "32"]


def torch_backend():
    """The torch backend when torch imports (on its GPU, else the CPU device), else None."""
    try:
        from backends.torch_backend import TorchBackend, default_device
    except ImportError:
        return None
    return TorchBackend(default_device() or "cpu")


def reference_state(spec, native, t0, norms, *, ridge=None):
    """What kvstate's own operations make of a native state holding [0, t0): K rows de-rotated, MLA
    latents put on their RMS shell, re-rotated at the same positions; V copied; indexer keys zero or
    the ridge's (from the shelled K rows); recurrent rows reset at t0 - 1 (R0)."""
    parts = []
    shelled = {}
    for part, layout in zip(native.state.parts, spec.layouts(native.state.flags)):
        if layout.kind != "kv":
            parts.append(kvstate.recurrent_reset(part, t0 - 1))
            continue
        stream = part.stream()
        positions = np.asarray(stream.positions, dtype=np.int64)
        k = {}
        for layer in part.layers:
            if spec.layer_types[layer] == "nextn":
                k[layer] = np.zeros_like(stream.k[layer])
            elif layout.role == "attention":
                pre = kvstate.pre_rope_keys(spec, layer, stream.k[layer], positions)
                if layer in norms:
                    lora, (weight, eps) = spec.kv_lora[layer], norms[layer]
                    pre[..., :lora] = kvstate.mla_renormalize_latent(pre[..., :lora], weight, eps)
                shelled[layer] = pre
                k[layer] = kvstate.cache_keys(spec, layer, pre, positions).astype(stream.k[layer].dtype)
            elif ridge is None:
                k[layer] = np.zeros_like(stream.k[layer])
            else:
                keys = ridge.predict(layer, shelled[layer].reshape(len(positions), -1))[:, None, :]
                keys = kvstate.cache_keys(spec, layer, keys, positions, role="indexer")
                k[layer] = keys.astype(stream.k[layer].dtype)
        v = {layer: np.asarray(stream.v[layer]) for layer in stream.v}
        rebuilt = kvstate.KVStream(stream.positions, stream.seq_ids, stream.ext, stream.v_trans, k, v,
                                   dict(stream.k_types), dict(stream.v_types))
        parts.append(kvstate.KVCache(part.name, part.layers, [rebuilt if s is not None else None
                                                              for s in part.streams]))
    return kvstate.StateFile(native.token_words, kvstate.SequenceState(spec, native.state.flags, parts))


def _resliced(state_file, positions_of):
    """state_file with each sliding KV section holding the cells at positions_of(stream): its own rows
    where it has a cell there, zero rows elsewhere."""
    spec = state_file.state.spec
    parts = []
    for part, layout in zip(state_file.state.parts, spec.layouts(state_file.state.flags)):
        if layout.kind != "kv" or not layout.sliding:
            parts.append(part)
            continue
        streams = []
        for stream in part.streams:
            if stream is None:
                streams.append(None)
                continue
            if stream.ext is not None:
                raise ValueError("cells with an ext (IMROPE, PLE) are not resliced")
            positions = np.asarray(positions_of(stream), dtype=np.int32)
            index = {int(p): i for i, p in enumerate(stream.positions)}
            rows = [(at, index[int(p)]) for at, p in enumerate(positions) if int(p) in index]

            def moved(table):
                out = {}
                for layer, values in table.items():
                    out[layer] = np.zeros((len(positions),) + values.shape[1:], values.dtype)
                    for at, cell in rows:
                        out[layer][at] = values[cell]
                return out
            streams.append(kvstate.KVStream(positions, (stream.seq_ids[0],) * len(positions), None, stream.v_trans,
                                            moved(stream.k), moved(stream.v), dict(stream.k_types),
                                            dict(stream.v_types)))
        parts.append(kvstate.KVCache(part.name, part.layers, streams))
    return kvstate.StateFile(state_file.token_words, kvstate.SequenceState(spec, state_file.state.flags, parts))


def sliding_history(native, n):
    """A native state of [0, n) whose sliding sections hold every position: their own cells, and zero rows
    where a cell left the window. The identity hub reads every layer at every source position; the zero
    rows reach only target cells outside the window (a translated sliding section keeps the last n_swa + 1,
    one more than a native save may hold, and no query reads the first of them)."""
    return _resliced(native, lambda stream: np.arange(n))


def sliding_window(state_file, start, end):
    """state_file with every sliding section holding [start, end): zero rows where it had no cell (a cell
    that left the window, translated from sliding_history's zero rows)."""
    return _resliced(state_file, lambda stream: np.arange(start, end))


def assert_states_equal(case, got, expected):
    """Two state files hold the same tokens, sections, cells and values (+0 and -0 alike: the identity
    hub's products turn a stored -0 into +0) and identical recurrent rows."""
    np.testing.assert_array_equal(got.token_words, expected.token_words)
    case.assertEqual([p.name for p in got.state.parts], [p.name for p in expected.state.parts])
    for a, b in zip(got.state.parts, expected.state.parts):
        if isinstance(b, kvstate.RecurrentState):
            case.assertEqual(kvstate._write_recurrent(expected.state.spec, a),
                             kvstate._write_recurrent(expected.state.spec, b))
            continue
        sa, sb = a.stream(), b.stream()
        np.testing.assert_array_equal(sa.positions, sb.positions)
        np.testing.assert_array_equal(sa.ext if sa.ext is not None else [], sb.ext if sb.ext is not None else [])
        case.assertEqual((sa.v_trans, sorted(sa.k), sorted(sa.v)), (sb.v_trans, sorted(sb.k), sorted(sb.v)))
        for layer in sb.k:
            case.assertEqual(sa.k[layer].dtype, sb.k[layer].dtype)
            np.testing.assert_array_equal(sa.k[layer], sb.k[layer], err_msg=f"{a.name} K {layer}")
        for layer in sb.v:
            np.testing.assert_array_equal(sa.v[layer], sb.v[layer], err_msg=f"{a.name} V {layer}")


@unittest.skipUnless(ff.DRIVER, "set KVPARK_STATE_DRIVER to run the MLA/DSA family end to end")
class IdentityTranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-families-")
        cls.root = Path(cls.temp.name)
        cls.models = ff.Models(cls.root)
        cls.rng = np.random.default_rng(2026)
        cls.ridges = {}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def translate_into_itself(self, name, n, *, cache="f32", fa="off", ridge=None, backend=None, n_next=3, out=None):
        """(native logits, translated logits, report, translated state, reference state, spec) for a
        prompt of n tokens translated into itself through the identity hub; `out` (a dict) receives
        the driver, the native state, its path and the next token ids."""
        driver = ff.Driver(self.models, self.root, cache=cache, fa=fa)
        spec = self.models.spec(name)
        vocab = self.models.vocab[name]
        prompt, next_ids = ff.ascii_ids(self.rng, n, vocab), ff.ascii_ids(self.rng, n_next, vocab)
        native_path, native_logits = driver.save(name, prompt, next_ids)
        template, _ = driver.save(name, prompt[:1])
        native = kvstate.read_file(native_path, spec)
        record, tokens = ff.byte_side(prompt)
        side = translate.Side(name, spec, record, tokens, gguf=self.models.path(name))
        state, report = translate.translate(sliding_history(native, n), side, side, ff.identity_hub(name, spec), t0=n,
                                            template=kvstate.read_file(template, spec), server_tokens=False,
                                            indexer_ridge=ridge, backend=backend)
        path = driver.fresh(f"{name}-translated").with_suffix(".state")
        state_writer.write(path, state)
        logits, restored = driver.restore(name, path, next_ids)
        self.assertEqual(restored["next_pos"], n)
        expected = reference_state(spec, native, n, translate.latent_norms(side), ridge=ridge)
        if spec.n_swa:
            expected = sliding_window(expected, max(0, n - spec.n_swa - 1), n)
        if out is not None:
            out.update(driver=driver, native=native, native_path=native_path, next_ids=next_ids)
        return native_logits, logits, report, state, expected, spec

    @staticmethod
    def restored_logits(driver, name, state, next_ids):
        """The logits of next_ids after restoring a state file (kvstate.StateFile) into `name`."""
        path = driver.fresh(f"{name}-restored").with_suffix(".state")
        state_writer.write(path, state)
        return driver.restore(name, path, next_ids)[0]

    def test_mla_models_restore_to_their_native_logits(self):
        for name in ("deepseek2", "deepseek2-yarn", "deepseek2-nextn", "mistral4-yarn"):
            for cache, fa in (("f32", "off"), ("f16", "on"), ("f16", "off")):
                with self.subTest(model=name, cache=cache, flash_attn=fa):
                    native, logits, report, state, expected, spec = self.translate_into_itself(name, 30, cache=cache,
                                                                                               fa=fa)
                    self.assertEqual((report["kinds"], report["mapped_cells"], report["fill_cells"]), (["k"], 30, 0))
                    # exactly what kvstate's own operations make of the native state
                    assert_states_equal(self, state, expected)
                    if cache == "f16":
                        # the latent shell moves c_kv by eps-sized amounts: below the f16 cache's step
                        np.testing.assert_array_equal(logits, native)
                    else:
                        # float32: the shell (|RMS(c_kv / g) - 1| ~ eps / mean square) and the RoPE round trip
                        np.testing.assert_allclose(logits, native, rtol=0, atol=5e-6 * max(1.0, np.abs(native).max()))
                    if name == "deepseek2-nextn":
                        # the nextn layer: zero rows in 85d5703a3's plain cache, left out of a7b94df2c's
                        stream = state.state.part("kv").stream()
                        if spec.nextn_cached:
                            self.assertFalse(stream.k[2].any())
                        self.assertEqual((sorted(stream.k), spec.llama_revision),
                                         ([0, 1, 2] if spec.nextn_cached else [0, 1], self.models.revision))

    def test_dsa_indexer_keys_below_and_above_top_k(self):
        for name in ("deepseek32", "glm-dsa"):
            spec = self.models.spec(name)
            self.assertEqual(spec.indexer.top_k, 8)
            # 5 cells + 3 decoded: every query sees at most top_k = 8 cells, so zero keys are exact
            for cache, fa in (("f32", "off"), ("f16", "on")):
                with self.subTest(model=name, cells=5, cache=cache, flash_attn=fa):
                    native, logits, report, state, expected, _ = self.translate_into_itself(name, 5, cache=cache,
                                                                                            fa=fa)
                    self.assertEqual(report["indexer"], "zeros")
                    assert_states_equal(self, state, expected)
                    tolerance = 0 if cache == "f16" else 5e-6 * max(1.0, np.abs(native).max())
                    np.testing.assert_allclose(logits, native, rtol=0, atol=tolerance)
            with self.subTest(model=name, cells=30, indexer="refused"):
                with self.assertRaisesRegex(translate.IndexerKeysUnavailable, "no indexer keys"):
                    self.translate_into_itself(name, 30)
            # T0 = 5 within top_k but a prompt of 27: the native tail's queries see more than top_k cells and
            # would rank the translated cells by their keys, so zeros are refused there too; the ridge fills them
            with self.subTest(model=name, cells=5, prompt=27):
                driver = ff.Driver(self.models, self.root)
                prompt = ff.ascii_ids(self.rng, 27, self.models.vocab[name])
                source = kvstate.read_file(driver.save(name, prompt[:5])[0], spec)
                record, tokens = ff.byte_side(prompt)
                side = translate.Side(name, spec, record, tokens, gguf=self.models.path(name))
                options = dict(t0=5, template=kvstate.read_file(driver.save(name, prompt[:1])[0], spec),
                               server_tokens=False)
                with self.assertRaisesRegex(translate.IndexerKeysUnavailable, "prompt of 27 tokens \\(T0 = 5\\)"):
                    translate.translate(source, side, side, ff.identity_hub(name, spec), **options)
                state, report = translate.translate(source, side, side, ff.identity_hub(name, spec),
                                                    indexer_ridge=self.ridge(name), **options)
                self.assertEqual(report["indexer"], "ridge")
                logits, restored = driver.restore(name, self.written(driver, name, state), prompt[5:])
                self.assertEqual(restored["next_pos"], 5)
                self.assertTrue(np.isfinite(logits).all())
            # above top_k: the indexer ridge (fitted on the model's own native states) fills the keys
            with self.subTest(model=name, cells=30, indexer="ridge"):
                native, logits, report, state, expected, _ = self.translate_into_itself(name, 30,
                                                                                        ridge=self.ridge(name))
                self.assertEqual(report["indexer"], "ridge")
                assert_states_equal(self, state, expected)
                self.assertTrue(state.state.part("indexer").stream().k[0].any())
                self.assertTrue(np.isfinite(logits).all())

    @staticmethod
    def written(driver, name, state):
        """A new state file holding `state`."""
        path = driver.fresh(f"{name}-written").with_suffix(".state")
        state_writer.write(path, state)
        return path

    def ridge(self, name):
        """An IndexerRidge of a DSA fixture from its own native f32 states (12 prompts of 60 tokens)."""
        if name not in self.ridges:
            spec = self.models.spec(name)
            driver = ff.Driver(self.models, self.root)
            rng = np.random.default_rng(11)                     # the same ridge whichever tests run before
            states = [kvstate.read_file(driver.save(name, ff.ascii_ids(rng, 60, 256))[0], spec) for _ in range(12)]
            self.ridges[name] = hub_module.fit_indexer_ridge(
                ff.StateRowsView(name, spec, states, ["train"] * 10 + ["dev"] * 2), log=lambda line: None)
        return self.ridges[name]

    def test_ridge_beats_zero_indexer_keys_above_top_k(self):
        # 30 cells, top_k 8: the indexer keys choose the cells; neither fill is exact, the ridge's keys
        # (R^2 about 0.7-0.9 on these fixtures) land closer to native than zeros
        for name in ("deepseek32", "glm-dsa"):
            with self.subTest(model=name):
                spec, ridge = self.models.spec(name), self.ridge(name)
                driver = ff.Driver(self.models, self.root)
                errors, rng = {"ridge": [], "zeros": []}, np.random.default_rng(7)
                for _ in range(3):
                    prompt, next_ids = ff.ascii_ids(rng, 30, 256), ff.ascii_ids(rng, 3, 256)
                    native_path, native = driver.save(name, prompt, next_ids)
                    template, _ = driver.save(name, prompt[:1])
                    record, tokens = ff.byte_side(prompt)
                    side = translate.Side(name, spec, record, tokens, gguf=self.models.path(name))
                    ridged, _ = translate.translate(kvstate.read_file(native_path, spec), side, side,
                                                    ff.identity_hub(name, spec), t0=30,
                                                    template=kvstate.read_file(template, spec), server_tokens=False,
                                                    indexer_ridge=ridge)
                    zeroed = kvstate.read_file(native_path, spec)
                    stream = zeroed.state.part("indexer").stream()
                    stream.k = {layer: np.zeros_like(rows) for layer, rows in stream.k.items()}
                    for fill, state in (("ridge", ridged), ("zeros", zeroed)):
                        path = driver.fresh(f"{name}-{fill}").with_suffix(".state")
                        state_writer.write(path, state)
                        logits, _ = driver.restore(name, path, next_ids)
                        errors[fill].append(float(np.abs(logits - native).mean()))
                # the mean error, prompt by prompt (the largest single-logit error need not be smaller)
                for ridged_error, zeroed_error in zip(errors["ridge"], errors["zeros"]):
                    self.assertLess(ridged_error, zeroed_error, errors)

    def test_hybrids_and_hy_v4(self):
        # the K-only MLA hybrids (KDA), hy_v4 (MLA with a [nope | rope] indexer) and the hybrids with GQA
        # attention next to GatedDeltaNet, Mamba, short-conv or lightning layers (iSWA attention for lfm2,
        # an indexer (V rows on 85d5703a3, [key | pooled] rows on a7b94df2c) and PLE rows for qwen4exp, layers
        # in both caches for falcon-h1), amplified
        # (ff.Models.amplified): latents on the RMS shell, so the identity translation is exact up to
        # rounding, and attention loud enough in the logits for the restore check below to fail on bad rows
        if not ff.LLAMA_ARCHS:
            self.skipTest("set KVPARK_LLAMA_ARCHS for the test-llama-archs models")
        insensitive = set()
        for arch in ff.ARCHS:
            name = self.models.amplified("archs-" + arch)
            for cache, fa in (("f32", "off"), ("f16", "on")):
                with self.subTest(model=arch, cache=cache, flash_attn=fa):
                    n = 5 if self.models.spec(name).indexer is not None else 30
                    run = {}
                    native, logits, report, state, expected, spec = self.translate_into_itself(name, n, cache=cache,
                                                                                               fa=fa, out=run)
                    assert_states_equal(self, state, expected)              # exactly kvstate's construction
                    self.assertTrue(np.isfinite(logits).all())
                    if spec.indexer is not None:
                        self.assertEqual(report["indexer"], "zeros")
                    restored = logits
                    if "recurrent" in [layout.name for layout in spec.layouts(0)]:
                        # R0: the recurrent rows restart from zeros at t0 - 1; decoding continues from there
                        self.assertTrue(report["recurrent_reset"])
                        recurrent = state.state.part("recurrent")
                        self.assertEqual(recurrent.positions.tolist(), [n - 1])
                        self.assertFalse(any(any(rows.data) for rows in recurrent.s.values()))
                        # the translated attention rows next to the native recurrent rows: R0 aside
                        parts = [ours if isinstance(ours, kvstate.KVCache) else theirs
                                 for ours, theirs in zip(state.state.parts, run["native"].state.parts)]
                        restored = self.restored_logits(run["driver"], name, kvstate.StateFile(
                            state.token_words, kvstate.SequenceState(spec, state.state.flags, parts)), run["next_ids"])
                    tolerance = 1e-5 * np.abs(native).max()
                    np.testing.assert_allclose(restored, native, rtol=0, atol=tolerance)
                    # the native state with every attention row zeroed: these logits must move, else the
                    # check above could not fail on bad attention rows
                    zeroed = kvstate.read_file(run["native_path"], spec)
                    for part, layout in zip(zeroed.state.parts, spec.layouts(zeroed.state.flags)):
                        if layout.kind == "kv" and layout.role == "attention":
                            stream = part.stream()
                            stream.k = {layer: np.zeros_like(rows) for layer, rows in stream.k.items()}
                            stream.v = {layer: np.zeros_like(rows) for layer, rows in stream.v.items()}
                    moved = np.abs(self.restored_logits(run["driver"], name, zeroed, run["next_ids"]) - native).max()
                    if moved < 100 * tolerance:
                        insensitive.add(arch)
        # even amplified, attention barely reaches the logits of these fixtures on 85d5703a3 (moves of 3e-8 to
        # 2.4e-7 at a logit scale of 1.4e-3 to 4.9e-3): their attention rows are checked against kvstate's own
        # construction only (and their RoPE against kv_probe captures, test_kv_probe_integration). With
        # a7b94df2c (its test-llama-archs writes ssm_a negative, and GatedDeltaNet normalizes q/k as
        # x * rsqrt(sum x^2 + eps): STATE_FORMAT_a7b94df.md) the qwen3next and qwen35moe logits were measured
        # to hear attention, so the restore checks their rows as well
        expected = {"qwen3next", "qwen35moe", "qwen4exp"} if self.models.revision == "85d5703a3" else {"qwen4exp"}
        self.assertEqual(insensitive, expected)

    def test_gqa_models_restore_to_their_native_logits(self):
        # plain and sliding-window GQA caches (per-layer KV heads and head sizes, gemma4's proportional RoPE
        # and shared KV layers, laguna's plain RoPE on its sliding layers, llama's frequency factors) and
        # GLM5-Next (NoPE MLA latents, KDA, [key | gate | pooled] indexer rows): a model translated into itself
        # through the identity hub writes exactly kvstate's construction of the native state and restores to
        # the native logits. The test-llama-archs models are amplified (their attention is a whisper in the
        # logits otherwise); the tiny Gemma models hear attention as they are, through a GELU that rounds its
        # input to float16 on the CPU, so a re-rotated key moves their logits by up to ~3e-4 of the scale
        # (linear_transfer/test_layouts.py GELU_TOLERANCE). A sliding-window model's prompt is longer than its
        # window, so its translated sliding section keeps the last n_swa + 1 cells only (what llama-server needs
        # to reuse a restored iSWA cache) and still restores to the native logits.
        names = list(ff.GEMMA)
        if ff.LLAMA_ARCHS:
            names += ["archs-" + arch for arch in ff.GQA_ARCHS + ff.LATER_ARCHS]
        checked, insensitive = [], set()
        for base in names:
            try:
                name = base if base in ff.GEMMA else self.models.amplified(base)
                self.models.spec(name)
            except (unittest.SkipTest, kvstate.StateFormatError):
                continue                        # not written by this test-llama-archs, or not on this revision
            checked.append(base)
            for cache, fa in (("f32", "off"), ("f16", "on")):
                with self.subTest(model=base, cache=cache, flash_attn=fa):
                    model_spec = self.models.spec(name)
                    n = 5 if model_spec.indexer is not None else model_spec.n_swa + 9 if model_spec.n_swa else 30
                    run = {}
                    native, logits, report, state, expected, spec = self.translate_into_itself(name, n, cache=cache,
                                                                                               fa=fa, out=run)
                    assert_states_equal(self, state, expected)              # exactly kvstate's construction
                    self.assertTrue(np.isfinite(logits).all())
                    restored = logits
                    if any(isinstance(part, kvstate.RecurrentState) for part in state.state.parts):
                        # the translated attention rows next to the native recurrent rows: R0 aside
                        parts = [ours if isinstance(ours, kvstate.KVCache) else theirs
                                 for ours, theirs in zip(state.state.parts, run["native"].state.parts)]
                        restored = self.restored_logits(run["driver"], name, kvstate.StateFile(
                            state.token_words, kvstate.SequenceState(spec, state.state.flags, parts)), run["next_ids"])
                    if spec.memory == "iswa":
                        sliding = state.state.part("swa").stream()
                        self.assertEqual(sliding.positions.tolist(), list(range(max(0, n - spec.n_swa - 1), n)))
                    tolerance = (2e-3 if base in ff.GEMMA and cache == "f32" else 1e-5) * np.abs(native).max()
                    np.testing.assert_allclose(restored, native, rtol=0, atol=tolerance)
                    # every attention row zeroed: the logits must move, else the check could not fail
                    zeroed = kvstate.read_file(run["native_path"], spec)
                    for part, layout in zip(zeroed.state.parts, spec.layouts(zeroed.state.flags)):
                        if layout.kind == "kv" and layout.role == "attention":
                            stream = part.stream()
                            stream.k = {layer: np.zeros_like(rows) for layer, rows in stream.k.items()}
                            stream.v = {layer: np.zeros_like(rows) for layer, rows in stream.v.items()}
                    moved = np.abs(self.restored_logits(run["driver"], name, zeroed, run["next_ids"]) - native).max()
                    if moved < 100 * tolerance:
                        insensitive.add(base)
        self.assertIn("gemma4", checked)
        # the Gemma models and the amplified test-llama-archs ones hear their attention rows
        self.assertEqual(insensitive, set())

    def test_pooled_indexer_ridge(self):
        # qwen4exp's block indexer caches raw per-token keys (no RoPE; pooled into blocks when read) and zero
        # V rows (85d5703a3) or the pooled keys, written zero and re-pooled on restore (a7b94df2c); GLM5-Next's
        # rows hold a per-token pool gate beside the key ([key | gate | pooled]), which its ridge predicts with
        # the key: above top_k they need the ridge (fitted on their own states' keys, like the DSA models')
        if not ff.LLAMA_ARCHS:
            self.skipTest("set KVPARK_LLAMA_ARCHS for the test-llama-archs models")
        for name in ("archs-qwen4exp", "archs-glm5-next"):
            try:
                spec = self.models.spec(name)
            except kvstate.StateFormatError:
                continue                        # GLM5-Next: a7b94df2c only
            with self.subTest(model=name):
                with self.assertRaisesRegex(translate.IndexerKeysUnavailable, "no indexer keys"):
                    self.translate_into_itself(name, 30)
                ridge = self.ridge(name)
                self.assertEqual(ridge.weights[0].shape[1], spec.indexer.head_size * spec.indexer_key_slots)
                native, logits, report, state, expected, spec = self.translate_into_itself(name, 30, ridge=ridge)
                self.assertEqual(report["indexer"], "ridge")
                assert_states_equal(self, state, expected)
                self.assertTrue(all(keys.any() for keys in state.state.part("indexer").stream().k.values()))
                self.assertTrue(np.isfinite(logits).all())

    def test_rope_conventions_match_llama_cpp(self):
        # A first layer's pre-RoPE K row (and indexer key) depends on its token alone, so the cache rows one
        # token leaves at different positions de-rotate (kvstate's RoPE) to one row, and a wrong mode, base,
        # scaling, offset or rotary width leaves them apart. This checks kvstate's RoPE against the rows
        # llama.cpp stores, also for the graphs without a pre-RoPE capture (kimi-k3 and kimi-linear: NoPE;
        # bailingmoe3, hy_v4 and its indexer keys).
        names = list(ff.TINY) + list(ff.GEMMA)
        if ff.LLAMA_ARCHS:
            names += ["archs-" + arch for arch in ff.ARCHS + ff.GQA_ARCHS + ff.LATER_ARCHS]
        rng = np.random.default_rng(5)
        checked = set()
        for name in names:
            try:
                spec = self.models.spec(name)
            except (unittest.SkipTest, kvstate.StateFormatError):
                continue                        # not written by this test-llama-archs, or not on this revision
            if 0 not in spec.attention_layers:
                continue
            prompt = (ff.ascii_ids(rng, 6, self.models.vocab[name]) * 7)[:40]     # six tokens at 6-7 positions each
            native = kvstate.read_file(ff.Driver(self.models, self.root).save(name, prompt)[0], spec)
            for part, layout in zip(native.state.parts, spec.layouts(native.state.flags)):
                if layout.kind != "kv" or 0 not in part.layers:
                    continue
                with self.subTest(model=name, section=part.name):
                    stream = part.stream()
                    positions = np.asarray(stream.positions, dtype=np.int64)
                    tokens = np.asarray(prompt)[positions]
                    rope = spec.rope(0, layout.role)
                    columns = slice(None) if rope is None else slice(rope.offset, rope.offset + rope.n_dims)

                    def spread(at):
                        """(largest gap between the rows of one token, largest value): rows de-rotated at `at`."""
                        rows = kvstate.pre_rope_keys(spec, 0, stream.k[0], at, role=layout.role)[..., columns]
                        rows = rows.reshape(len(at), -1).astype(np.float64)
                        gap = max(np.abs(rows[tokens == t] - rows[tokens == t][0]).max() for t in set(tokens.tolist()))
                        return gap, np.abs(rows).max()

                    gap, scale = spread(positions)
                    self.assertGreater(scale, 0)
                    self.assertLess(gap, 1e-5 * scale)                  # f32 cache: rounding only
                    if rope is not None:
                        # de-rotated at other positions, the rows stay apart: the check can fail
                        self.assertGreater(spread(2 * positions)[0], 0.1 * scale)
                    checked.add((spec.architecture, layout.role, rope is not None))
        if ff.LLAMA_ARCHS:
            self.assertLessEqual({("kimi-k3", "attention", False), ("kimi-linear", "attention", False),
                                  ("bailingmoe3", "attention", True), ("hy_v4", "attention", True),
                                  ("hy_v4", "indexer", True)}, checked)

    def test_the_gpu_backend_restores_like_numpy(self):
        backend = torch_backend()
        if backend is None:
            self.skipTest("torch is not installed")
        for name in ("deepseek2-yarn", "deepseek32"):
            with self.subTest(model=name):
                native, logits, report, state, expected, spec = self.translate_into_itself(
                    name, 5, cache="f16", fa="on", backend=backend)
                self.assertEqual(report["backend"], "torch")
                # float16 activations hold an f16 cache exactly; RoPE and the shell in float64
                for a, b in zip(state.state.parts, expected.state.parts):
                    for layer in a.layers:
                        x, y = a.stream().k[layer].astype(np.float32), b.stream().k[layer].astype(np.float32)
                        np.testing.assert_allclose(x, y, rtol=0, atol=2e-3 * max(1.0, np.abs(y).max()))
                np.testing.assert_allclose(logits, native, rtol=0, atol=1e-3 * max(1.0, np.abs(native).max()))


@unittest.skipUnless(ff.DRIVER and ff.PROBE, "set KVPARK_STATE_DRIVER and KVPARK_KV_PROBE for the cross-family hub")
class CrossFamilyTests(unittest.TestCase):
    """qwen3 (GQA) <-> deepseek2 (MLA), deepseek32 (DSA: the hub's indexer ridge), dots3note (DSA-iSWA:
    a sliding-window source encodes its global layers) and gemma4 with a 16-token window (iSWA with
    per-layer head sizes and proportional RoPE: as a target its sliding layers hold the last 17 cells,
    as a source its global layers alone are encoded) through a hub fitted on kv_probe captures of the
    five fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-families-hub-")
        cls.root = Path(cls.temp.name)
        cls.models = ff.Models(cls.root)
        if not ff.probe_supports(cls.models.path("deepseek2"), cls.root):
            cls.temp.cleanup()
            raise unittest.SkipTest("this kvpark-kv-probe predates the MLA/DSA table rows; rebuild it")
        cls.models.paths["qwen3"], cls.models.vocab["qwen3"] = cls.root / "qwen3.gguf", 256
        create_fixture(cls.models.paths["qwen3"], "qwen3", chat_template=ff.TEMPLATE)
        conversations = ff.conversations(16, seed=3, splits=["train"] * 12 + ["dev"] * 2 + ["test"] * 2)
        records = ff.rendered_records(conversations)
        (cls.root / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        arguments = []
        names = ("qwen3", "deepseek2", "deepseek32", "dots3note")
        # probes built before the GQA rows refuse gemma4: the hub then leaves it out
        names += ("gemma4-w16",) if ff.probe_supports(cls.models.path("gemma4-w16"), cls.root) else ()
        for name in names:
            keys = ("k", "v", "ik") if name in ("deepseek32", "dots3note") else None
            capture = ff.probe_collect(cls.models.path(name), records, cls.root / f"{name}-capture", keys=keys)
            arguments += ["--model", f"{name}={capture}@{cls.root / 'records.jsonl'}"]
        if "gemma4-w16" in names:
            arguments += ["--source-global", "gemma4-w16"]
        cls.gemma = "gemma4-w16" in names
        cls.bundle = cls.root / "hub"
        with open(os.devnull, "w") as quiet:
            stdout, sys.stdout = sys.stdout, quiet
            try:
                fit_hub.main(arguments + ["--output", str(cls.bundle), "--hub-dim", "12", "--view-dim", "24",
                                          "--layer-rank", "16", "--indexer-ridge", "deepseek32",
                                          "--indexer-ridge", "dots3note", "--source-global", "dots3note",
                                          *guard_arguments(cls.root)])
            finally:
                sys.stdout = stdout
        cls.bundle2 = cls.root / "hub-v2"
        convert_hub.convert(cls.bundle, cls.bundle2, log=lambda line: None)
        cls.test_records = [r for r in records if r["split"] == "test"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def switch(self, source, target, bundle, *, tail=16):
        """(translated state, report, its restore report, logits) for the first test conversation."""
        record = self.test_records[0]
        ids = list(record["text"].encode("ascii"))
        tokens = {"token_id": np.asarray(ids, dtype=np.int32), "byte_end": np.arange(1, len(ids) + 1),
                  "special": np.zeros(len(ids), dtype=np.uint8)}
        driver = ff.Driver(self.models, self.root, cache="f16", fa="on")
        source_spec, target_spec = self.models.spec(source), self.models.spec(target)
        source_state = kvstate.read_file(driver.save(source, ids)[0], source_spec)
        template = kvstate.read_file(driver.save(target, ids[:1])[0], target_spec)
        t0 = len(ids) - tail
        state, report = translate.translate(
            source_state, translate.Side(source, source_spec, record, tokens, gguf=self.models.path(source)),
            translate.Side(target, target_spec, record, tokens, gguf=self.models.path(target)),
            hub_module.open_hub(bundle), t0=t0, template=template, server_tokens=False)
        path = driver.fresh(f"{source}-{target}").with_suffix(".state")
        state_writer.write(path, state)
        logits, restored = driver.restore(target, path, ids[t0:] + [10, 10])
        return state, report, restored, logits, t0

    def test_gqa_and_mla_translate_both_ways(self):
        pairs = [("qwen3", "deepseek2"), ("deepseek2", "qwen3"), ("qwen3", "deepseek32"), ("deepseek2", "deepseek32"),
                 ("qwen3", "dots3note"), ("dots3note", "qwen3"), ("dots3note", "deepseek2")]
        if self.gemma:
            pairs += [("qwen3", "gemma4-w16"), ("gemma4-w16", "qwen3"), ("gemma4-w16", "deepseek32"),
                      ("dots3note", "gemma4-w16")]
        for source, target in pairs:
            with self.subTest(source=source, target=target):
                state, report, restored, logits, t0 = self.switch(source, target, self.bundle)
                spec = self.models.spec(target)
                self.assertEqual(restored["next_pos"], t0)
                self.assertTrue(np.isfinite(logits).all())
                self.assertGreater(report["mapped_cells"], t0 // 2)
                for part, layout in zip(state.state.parts, spec.layouts(0)):
                    stream = part.stream()
                    # sliding sections: the last n_swa + 1 cells (what llama-server needs to reuse the cache)
                    first = t0 - spec.n_swa - 1 if layout.sliding else 0
                    np.testing.assert_array_equal(stream.positions, np.arange(first, t0))
                    for layer in part.layers:
                        self.assertEqual(stream.k[layer].shape[1:], spec.kv_shape(layer, "k", layout.role))
                        self.assertEqual(layer in stream.v, spec.kv_shape(layer, "v", layout.role) is not None)
                if spec.indexer is not None:
                    self.assertEqual(report["indexer"], "ridge")          # the bundle's ridge (prompts > top_k)
                    self.assertTrue(state.state.part("indexer").stream().k[0].any())
                if spec.kv_lora is not None:
                    self.assertEqual(report["kinds"], ["k"])

    def test_format_2_bundles_translate_like_format_1(self):
        for source, target in (("qwen3", "deepseek32"), ("deepseek32", "qwen3"), ("dots3note", "dots3note")):
            with self.subTest(source=source, target=target):
                first = self.switch(source, target, self.bundle)[0].to_bytes()
                self.assertEqual(self.switch(source, target, self.bundle2)[0].to_bytes(), first)


@unittest.skipUnless(ff.DRIVER, "set KVPARK_STATE_DRIVER to run engine jobs into MLA targets")
class EngineJobTests(unittest.TestCase):
    """engine.py JOB.json (the one-shot path) from a qwen3 fixture into MLA and DSA fixtures."""

    MESSAGES = [{"role": "system", "content": "You answer briefly and stay on topic."},
                {"role": "user", "content": "Tell me about the tides along the northern coast this week."},
                {"role": "assistant", "content": "The tides run high at dawn and low near noon, with a strong ebb."},
                {"role": "user", "content": "And what about next week, when the moon is full again?"}]

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-families-engine-")
        cls.root = Path(cls.temp.name)
        cls.models = ff.Models(cls.root)
        cls.models.paths["src"], cls.models.vocab["src"] = cls.root / "src.gguf", 256
        create_fixture(cls.models.paths["src"], "qwen3", chat_template=ff.TEMPLATE)
        for name in ("deepseek2-yarn", "deepseek32"):
            # the MLA fixture with the same chat template (the engine renders with the target's)
            copy_with_metadata(cls.models.path(name), cls.root / f"{name}-chat.gguf",
                               values={"tokenizer.chat_template": ff.TEMPLATE})
            cls.models.paths[name + "-chat"], cls.models.vocab[name + "-chat"] = cls.root / f"{name}-chat.gguf", 256
        cls.driver = ff.Driver(cls.models, cls.root)
        import chat
        history = chat.render({"messages": cls.MESSAGES[:-1], "tools": None}, ff.TEMPLATE, live=True).text
        cls.source_state = cls.driver.save("src", list(history.encode("ascii")))[0]
        cls.text = chat.render({"messages": cls.MESSAGES, "tools": None}, ff.TEMPLATE, add_generation_prompt=True,
                               live=True).text

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def job(self, target, *, runtime=None, extra=()):
        driver = ff.Driver(self.models, self.root, extra=extra)
        spec = self.models.spec(target, runtime)
        hubs = random_hub({"src": kind_widths(self.models.spec("src")), "tgt": kind_widths(spec)})
        bundle = driver.fresh("hub")
        views = [StubView(name, hubs, arch) for name, arch in (("src", "qwen3"), ("tgt", spec.architecture))]
        hub_module.save_hub(bundle, hubs, views, StubSpans(), {"note": "test"})
        output = driver.fresh("out").with_suffix(".state")
        return {"hub": str(bundle), "tail": 8, "fill": "mean", "device": "cpu",
                "conversation": {"messages": self.MESSAGES, "tools": None},
                "source": {"name": "src", "gguf": str(self.models.path("src")), "state": str(self.source_state)},
                "target": {"name": "tgt", "gguf": str(self.models.path(target)), "runtime": runtime,
                           "template_state": str(driver.save(target, [ord(self.text[0])])[0]),
                           "output": str(output), "tokens": [[ord(c), c] for c in self.text]}}, driver

    def run_engine(self, job):
        path = Path(job["target"]["output"]).with_suffix(".job.json")
        path.write_text(json.dumps(job), encoding="utf-8")
        return subprocess.run([sys.executable, str(HERE / "engine.py"), str(path)], capture_output=True, text=True,
                              timeout=600, cwd=HERE)

    def test_one_shot_job_into_mla_targets(self):
        target = "deepseek2-yarn-chat"
        outputs = {}
        for runtime, extra in ((None, ()), (RUNTIME_YARN, RUNTIME_YARN_FLAGS)):
            with self.subTest(runtime=runtime):
                # the whole tail is decoded as one ubatch after the restore
                job, driver = self.job(target, runtime=runtime, extra=["--ctx", "512", "--batch", "256", *extra])
                result = self.run_engine(job)
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                report = json.loads(result.stdout.strip().splitlines()[-1])
                self.assertEqual((report["kinds"], report["latent_layers"], report["backend"]), (["k"], 2, "numpy"))
                spec = self.models.spec(target, runtime)
                state = kvstate.read_file(job["target"]["output"], spec)
                t0 = report["t0"]
                stream = state.state.part("kv").stream()
                self.assertEqual(stream.positions.tolist(), list(range(t0)))
                outputs[runtime is None] = (spec, stream)
                # restore into the target and prefill the tail natively, as llama-server would
                logits, restored = driver.restore(target, Path(job["target"]["output"]),
                                                  [ord(c) for c in self.text[t0:]])
                self.assertEqual(restored["next_pos"], t0)
                self.assertTrue(np.isfinite(logits).all())
        # the same decoded rows, rotated with the RoPE each server applies (the job's runtime settings:
        # YaRN over an original context of 32 at scale 1/8 instead of the GGUF's 64 at 1/4)
        (plain_spec, plain), (yarn_spec, yarn) = outputs[True], outputs[False]
        positions = np.asarray(plain.positions, dtype=np.int64)
        self.assertNotEqual((plain_spec.rope(0).freq_scale, plain_spec.rope(0).n_ctx_orig),
                            (yarn_spec.rope(0).freq_scale, yarn_spec.rope(0).n_ctx_orig))
        for layer in (0, 1):
            self.assertGreater(np.abs(plain.k[layer][..., 32:] - yarn.k[layer][..., 32:]).max(), 1e-3)
            expected = kvstate.pre_rope_keys(plain_spec, layer, plain.k[layer], positions)
            np.testing.assert_allclose(kvstate.pre_rope_keys(yarn_spec, layer, yarn.k[layer], positions), expected,
                                       rtol=0, atol=1e-6 * np.abs(expected).max())

    def test_dsa_target_without_indexer_keys_is_refused(self):
        job, _ = self.job("deepseek32-chat")
        result = self.run_engine(job)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no indexer keys for tgt", result.stderr)
        self.assertFalse(Path(job["target"]["output"]).exists())
        # the bridge's pre-check (stdlib, GGUF header only) decides the same before any engine runs
        sys.path.insert(0, str(HERE.parent.parent / "src"))
        from kvpark.bridge import capabilities
        gguf = job["target"]["gguf"]
        prompt = len(job["target"]["tokens"])                       # P, the native tail included
        code, note = capabilities.indexer("tgt", gguf, [], prompt)
        self.assertEqual(code, "no-indexer-keys")
        self.assertIn(f"a prompt of {prompt} tokens", note)
        self.assertIn("beyond 8 cells", note)
        self.assertIsNone(capabilities.indexer("tgt", gguf, ["tgt"], prompt))            # the hub has a ridge
        self.assertIsNone(capabilities.indexer("tgt", gguf, [], 8))                      # within top_k
        self.assertEqual(capabilities.indexer("tgt", gguf, [], 9)[0], "no-indexer-keys")
        self.assertIsNone(capabilities.indexer("tgt", str(self.models.path("deepseek2-yarn-chat")), [], prompt))


if __name__ == "__main__":
    unittest.main()
