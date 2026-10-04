"""Compute backends: selection, numpy BLAS checks, and the torch backend against the numpy reference.

The torch tests run where torch imports (on its GPU, else on the CPU device); the experiment
venv has no torch, so run them also with a torch Python, e.g. ~/venvs/rocm/bin/python -m
unittest test_backends.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import backends
from backends import numpy_backend
import hub
import test_translate
from test_translate import (DEEPSEEK2, DOTS, DSA_TOP4, HYBRID, KDA, MUSE, SOURCE, TARGET, StubSpans, StubView,
                            attention_widths, latent_norm_weights, random_hub)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "linear_transfer"))
import kvstate  # noqa: E402

try:
    import torch
    from backends import torch_backend
except ImportError:
    torch = None


class SelectionTest(unittest.TestCase):
    def test_requested_device(self):
        with mock.patch.dict(os.environ, {backends.ENV: ""}):
            self.assertEqual(backends.requested(None), "auto")
            self.assertEqual(backends.requested("GPU"), "gpu")
        with mock.patch.dict(os.environ, {backends.ENV: "cpu"}):
            self.assertEqual(backends.requested(None), "cpu")
            self.assertEqual(backends.requested("gpu"), "gpu")        # the job wins over the environment
        with self.assertRaisesRegex(ValueError, "engine device"):
            backends.requested("tpu")

    def test_auto_falls_back_to_numpy_and_gpu_refuses(self):
        self.assertEqual(backends.select("cpu").name, "numpy")
        with mock.patch.object(backends, "gpu_status", return_value=(False, "no torch")):
            self.assertEqual(backends.select("auto").name, "numpy")
            with self.assertRaisesRegex(RuntimeError, "no GPU backend: no torch"):
                backends.select("gpu")
        self.assertIs(backends.select("cpu"), backends.select("cpu"))   # one instance per process

    def test_reference_blas_is_flagged(self):
        reference = numpy_backend.classify_blas({"name": "blas", "gflops": 30.0, "libraries": [
            "/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.12.1"]})
        self.assertTrue(reference["reference"])
        self.assertIn("reference BLAS", reference["warning"])
        fast = numpy_backend.classify_blas({"name": "scipy-openblas", "gflops": 300.0,
                                            "libraries": ["/x/numpy.libs/libscipy_openblas64_-1.so"]})
        self.assertFalse(fast["reference"])
        self.assertNotIn("warning", fast)
        # Debian's alternatives can point "blas" at OpenBLAS
        alternative = numpy_backend.classify_blas({"name": "blas", "gflops": 400.0, "libraries": [
            "/usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3"]})
        self.assertFalse(alternative["reference"])
        slow = numpy_backend.classify_blas({"name": "accelerate", "gflops": 12.0, "libraries": []})
        self.assertIn("slow", slow["warning"])
        info = numpy_backend.blas_info()
        self.assertGreater(info["gflops"], 0)
        self.assertIn("blas", backends.select("cpu").describe())


class NumpyRopeTest(unittest.TestCase):
    def test_in_place_rotation_is_bit_identical_to_kvstate(self):
        positions = np.asarray([0, 3, 77, 4097, 65535, 262143], dtype=np.int64)
        rng = np.random.default_rng(2)
        # MLA rows: YaRN on the k_pe tail of [c_kv | k_pe] (DEEPSEEK2), the DSA-iSWA sliding base (DOTS)
        for spec, layer in ((TARGET, 0), (MUSE, 0), (HYBRID, 1), (DEEPSEEK2, 1), (DOTS, 1), (DOTS, 2)):
            x = (rng.standard_normal((len(positions),) + spec.head_shape(layer, "k")) * 3).astype(np.float32)
            rope = spec.rope(layer)
            for inverse, reference in ((False, kvstate.cache_keys), (True, kvstate.pre_rope_keys)):
                got = numpy_backend.rotate_inplace(x.copy(), positions, rope, inverse=inverse)
                np.testing.assert_array_equal(got, reference(spec, layer, x, positions))
        x = rng.standard_normal((len(positions), 1, 80)).astype(np.float32)
        for rope in (kvstate.Rope("norm", 64, 10000.0, offset=16, freq_scale=0.025, ext_factor=1.0, attn_factor=0.9,
                                  n_ctx_orig=4096), kvstate.Rope("neox", 32, 1e6, offset=48, freq_scale=0.5)):
            np.testing.assert_array_equal(numpy_backend.rotate_inplace(x.copy(), positions, rope, inverse=False),
                                          rope.apply(x, positions))
            np.testing.assert_array_equal(numpy_backend.rotate_inplace(x.copy(), positions, rope, inverse=True),
                                          rope.invert(x, positions))


class LatentShellTest(unittest.TestCase):
    def test_numpy_renormalization_is_kvstate_itself(self):
        rng = np.random.default_rng(5)
        norms = latent_norm_weights(DOTS)
        layers = [0, 1, 2, 3]                                   # rows of 12, 10, 12, 10: c_kv 8, 6, 8, 6
        values = (rng.standard_normal((7, 44)) * 3).astype(np.float32)
        expected, offset = values.copy(), 0
        for layer in layers:
            lora, width = DOTS.kv_lora[layer], DOTS.head_dim_k[layer]
            weight, eps = norms[layer]
            expected[:, offset:offset + lora] = kvstate.mla_renormalize_latent(values[:, offset:offset + lora], weight,
                                                                               eps)
            offset += width
        got = numpy_backend.NumpyBackend().renormalize_latents(DOTS, layers, values.copy(), norms)
        np.testing.assert_array_equal(got, expected)
        # only the latent columns change; layers without a norm are left alone
        np.testing.assert_array_equal(got[:, 8:12], values[:, 8:12])
        untouched = numpy_backend.NumpyBackend().renormalize_latents(DOTS, layers, values.copy(), {})
        np.testing.assert_array_equal(untouched, values)


class ResidentSidesTest(unittest.TestCase):
    def test_sides_are_kept_until_dropped(self):
        backend = numpy_backend.NumpyBackend()
        hubs = random_hub({"src": attention_widths(SOURCE), "tgt": attention_widths(TARGET)})
        first = backend.encoder(hubs, "k", "src")
        self.assertIs(backend.encoder(hubs, "k", "src"), first)
        backend.decoder(hubs, "k", "tgt")
        backend.decoder(hubs, "v", "tgt")
        keys = [tuple(item["key"]) for item in backend.resident()]
        self.assertEqual([key[1:] for key in keys], [("k", "src", "encoder"), ("k", "tgt", "decoder"),
                                                     ("v", "tgt", "decoder")])
        total = backend.resident_bytes()
        self.assertGreater(total, 0)
        released = backend.evict(1, keep=[keys[0]])                  # least recently used first, keep honoured
        self.assertEqual([tuple(i["key"]) for i in backend.resident()], [keys[0], keys[2]])
        self.assertGreater(released, 0)
        self.assertEqual(backend.drop(), total - released)
        self.assertEqual(backend.resident(), [])
        other = random_hub({"src": attention_widths(SOURCE), "tgt": attention_widths(TARGET)}, seed=1)
        self.assertIsNot(backend.encoder(other, "k", "src"), first)  # another hub object, another side


@unittest.skipIf(torch is None, "torch is not installed")
class TorchBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = torch_backend.TorchBackend(torch_backend.default_device() or "cpu")
        cls.positions = np.asarray([0, 1, 2, 63, 1000, 4097, 65535, 131071, 262143], dtype=np.int64)

    def assert_rope_matches(self, spec, layer, *, atol=1e-6):
        rng = np.random.default_rng(layer)
        x = (rng.standard_normal((len(self.positions),) + spec.head_shape(layer, "k")) * 4).astype(np.float32)
        rope = torch_backend.layer_rope(spec, layer)
        pos = torch.from_numpy(self.positions).to(self.backend.device)
        for inverse, reference in ((False, kvstate.cache_keys), (True, kvstate.pre_rope_keys)):
            expected = reference(spec, layer, x, self.positions)
            tensor = torch.from_numpy(x.copy()).to(self.backend.device)
            got = tensor if rope is None else self.backend.rotate(tensor, pos, rope, inverse=inverse)
            np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=0, atol=atol * np.abs(x).max(),
                                       err_msg=f"{spec.architecture} layer {layer} inverse={inverse}")

    def test_rope_matches_kvstate_for_neox_norm_and_imrope(self):
        self.assert_rope_matches(TARGET, 0)                       # NEOX over the whole head
        self.assert_rope_matches(MUSE, 0)                         # NORM, sliding base
        self.assert_rope_matches(HYBRID, 1)                       # IMROPE over 8 of 16 dims, one theta_e pair
        self.assertIsNone(torch_backend.layer_rope(MUSE, 2))      # Muse global layers are NoPE
        real = kvstate.ModelSpec("qwen35", ("recurrent",) * 3 + ("full",), (0, 0, 0, 4), (0, 0, 0, 256),
                                 (0, 0, 0, 256), rope_type="imrope", n_rot=64, rope_freq_base=1e7,
                                 rope_sections=(11, 11, 10, 0), recurrent_row_bytes=(16, 32))
        self.assert_rope_matches(real, 3)                         # Qwen3.8's geometry
        muse = kvstate.ModelSpec("muse-glimmer", ("sliding", "full"), (2, 2), (128, 128), (128, 128), n_swa=2048,
                                 rope_type="norm", n_rot=128, rope_freq_base=5e5, rope_freq_base_swa=5e5)
        self.assert_rope_matches(muse, 0)                         # Muse's geometry

    def test_mla_ropes_and_latent_shell_match_kvstate(self):
        self.assert_rope_matches(DEEPSEEK2, 0)                    # YaRN on the k_pe tail of [c_kv | k_pe]
        self.assert_rope_matches(DOTS, 1)                         # a sliding MLA layer: base 500
        self.assertIsNone(torch_backend.layer_rope(KDA, 0))       # kimi rows are NoPE
        rng = np.random.default_rng(6)
        norms = latent_norm_weights(DOTS)
        values = (rng.standard_normal((9, 44)) * 3).astype(np.float32)
        expected = numpy_backend.NumpyBackend().renormalize_latents(DOTS, [0, 1, 2, 3], values.copy(), norms)
        device_values = torch.from_numpy(values.copy()).to(self.backend.device)
        got = self.backend.renormalize_latents(DOTS, [0, 1, 2, 3], device_values, norms)
        np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=0, atol=1e-6 * np.abs(expected).max())

    def test_yarn_and_offset_ropes_match_kvstate(self):
        pos = torch.from_numpy(self.positions).to(self.backend.device)
        x = np.random.default_rng(1).standard_normal((len(self.positions), 1, 80)).astype(np.float32)
        for rope in (kvstate.Rope("norm", 64, 10000.0, offset=16, freq_scale=0.025, ext_factor=1.0,
                                  attn_factor=0.9, n_ctx_orig=4096),
                     kvstate.Rope("neox", 32, 1e6, offset=48, freq_scale=0.5)):
            device = torch_backend.device_rope(rope)
            for inverse, expected in ((False, rope.apply(x, self.positions)), (True, rope.invert(x, self.positions))):
                got = self.backend.rotate(torch.from_numpy(x.copy()).to(self.backend.device), pos, device,
                                          inverse=inverse)
                np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=0, atol=2e-6 * np.abs(x).max())

    def test_mrope_components_reduce_to_text_positions(self):
        rope = torch_backend.layer_rope(HYBRID, 1)
        text = torch.from_numpy(self.positions).to(self.backend.device)
        components = torch.stack([text, text, text, torch.zeros_like(text)], dim=1)
        for a, b in zip(self.backend.cos_sin(text, rope), self.backend.cos_sin(components, rope)):
            self.assertTrue(torch.equal(a, b))
        shifted = components.clone()
        shifted[:, 1] += 7                                        # an image cell: h differs from t
        cos, _ = self.backend.cos_sin(shifted, rope)
        self.assertFalse(torch.equal(cos, self.backend.cos_sin(text, rope)[0]))

    def test_unmirrored_rope_is_refused(self):
        class Odd:
            mode = "spiral"
        with self.assertRaises(NotImplementedError):
            torch_backend.device_rope(Odd())

    def translate_both(self, source_spec, target_spec, subsets=None, hub_override=None, window_rows=None, **options):
        case = test_translate.SectionsTest()   # by module: not collected here a second time
        case.setUp()
        out = []
        for backend in (backends.select("cpu"), self.backend):
            case.rng = np.random.default_rng(11)
            state, report, hubs, _ = case.build(source_spec, target_spec, subsets=subsets, dtype="f16", backend=backend,
                                                hub_override=hub_override, window_rows=window_rows, **options)
            out.append((kvstate.parse_file(state.to_bytes(), target_spec), report))
        return out

    def test_translations_match_numpy_for_every_family(self):
        ridge = test_translate.random_indexer_ridge(DSA_TOP4)
        cases = [(SOURCE, TARGET, None, {}), (SOURCE, MUSE, None, {}), (SOURCE, HYBRID, None, {}),
                 (MUSE, HYBRID, {"src": list(MUSE.layers_of("full"))}, {}), (HYBRID, MUSE, None, {}),
                 # MLA, DSA (indexer ridge), DSA-iSWA and K-only hybrid targets, MLA sources
                 (SOURCE, DEEPSEEK2, None, {}), (DEEPSEEK2, TARGET, None, {}), (SOURCE, DSA_TOP4, None,
                                                                                 {"indexer_ridge": ridge}),
                 (SOURCE, DOTS, None, {}), (HYBRID, KDA, None, {}), (KDA, DEEPSEEK2, None, {})]
        for (source_spec, target_spec, subsets, options), window in [(case, None) for case in cases] + [(cases[1], 3)]:
            with self.subTest(source=source_spec.architecture, target=target_spec.architecture, window=window):
                (ref, ref_report), (got, got_report) = self.translate_both(source_spec, target_spec, subsets,
                                                                           window_rows=window, **options)
                self.assertEqual((got_report["backend"], ref_report["backend"]), ("torch", "numpy"))
                for key in ("t0", "cells", "mapped_cells", "fill_cells", "sections", "decoded_rows", "recurrent_reset"):
                    self.assertEqual(got_report[key], ref_report[key], key)
                self.assertEqual(len(got.to_bytes()), len(ref.to_bytes()))
                for a, b in zip(got.state.parts, ref.state.parts):
                    if isinstance(a, kvstate.RecurrentState):
                        self.assertEqual(kvstate._write_recurrent(target_spec, a),
                                         kvstate._write_recurrent(target_spec, b))
                        continue
                    sa, sb = a.stream(), b.stream()
                    np.testing.assert_array_equal(sa.positions, sb.positions)
                    self.assertEqual(sorted(sa.v), sorted(sb.v))
                    for layer in a.layers:
                        for x, y in [(sa.k[layer], sb.k[layer])] + [(sa.v[l], sb.v[l]) for l in [layer] if l in sb.v]:
                            x, y = x.astype(np.float64), y.astype(np.float64)
                            if not y.any():                                 # nextn layers, zero indexer keys
                                self.assertFalse(x.any())
                                continue
                            self.assertLess(np.linalg.norm(x - y) / np.linalg.norm(y), 2e-3)

    def test_packed_float16_bundle_gives_the_same_state(self):
        import convert_hub
        case = test_translate.SectionsTest()
        case.setUp()
        _, _, hubs, _ = case.build(SOURCE, MUSE)
        with tempfile.TemporaryDirectory() as directory:
            views = [StubView(name, hubs, "x") for name in ("src", "tgt")]
            hub.save_hub(Path(directory) / "v1", hubs, views, StubSpans(), {})
            convert_hub.convert(Path(directory) / "v1", Path(directory) / "v2", log=lambda line: None)
            states = []
            for bundle in ("v1", "v2"):
                case.rng = np.random.default_rng(3)
                state, _, _, _ = case.build(SOURCE, MUSE, backend=self.backend,
                                            hub_override=hub.open_hub(Path(directory) / bundle))
                states.append(state.to_bytes())
            self.assertEqual(states[0], states[1])     # float16 file == float32 file cast on the device
            self.backend.drop()

    def test_resident_sides_report_device_bytes_and_release(self):
        hubs = random_hub({"src": attention_widths(SOURCE), "tgt": attention_widths(MUSE)})
        backend = torch_backend.TorchBackend(self.backend.device)
        decoder = backend.decoder(hubs, "k", "tgt")
        before = backend.resident_bytes()
        self.assertGreater(before, 0)
        weight, mean = decoder.section(list(MUSE.layers_of("sliding")))
        self.assertEqual(tuple(weight.shape), (6, 4 * 8))
        decoder.section(list(MUSE.layers_of("full")))            # every layer covered: per-layer copy released
        self.assertEqual(decoder.decoders, {})
        torch.testing.assert_close(decoder.section([0])[0], weight[:, :8])
        self.assertGreater(backend.drop(), 0)


if __name__ == "__main__":
    unittest.main()
