"""The ggml backend (libkvbridge on Vulkan, or ggml's CPU backend) against kvstate and the numpy reference.

Skipped when libkvbridge is not built (ggml/README.md; $KVPARK_GGML_BRIDGE locates another build),
except the selection tests. The tests run on the library's default device (the first Vulkan GPU,
else ggml's CPU backend); CpuFallbackTest translates on ggml's CPU backend, and the Blocked tests
repeat the products and translations with every device array split into row blocks (a backend
whose largest buffer is lowered to a few KiB).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import backends
from backends import ggml_backend, numpy_backend
import engine
import hub
import test_translate
from test_translate import (DEEPSEEK2, DOTS, DSA_TOP4, HYBRID, KDA, MUSE, SOURCE, TARGET, StubSpans, StubView,
                            kind_widths, latent_norm_weights, native_state)
import translate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "linear_transfer"))
import kvstate  # noqa: E402

try:
    ggml_backend.library()
    MISSING = None
except OSError as error:
    MISSING = str(error)

POSITIONS = np.asarray([0, 1, 2, 63, 1000, 4097, 65535, 131071, 262143], dtype=np.int64)
# Qwen3.8's attention geometry (IMROPE 64 of 256 dims, sections [11, 11, 10, 0]) and Muse's (NORM 128)
QWEN38 = kvstate.ModelSpec("qwen35", ("recurrent",) * 3 + ("full",), (0, 0, 0, 4), (0, 0, 0, 256), (0, 0, 0, 256),
                           rope_type="imrope", n_rot=64, rope_freq_base=1e7, rope_sections=(11, 11, 10, 0),
                           recurrent_row_bytes=(16, 32))
MUSE_REAL = kvstate.ModelSpec("muse-glimmer", ("sliding", "full"), (2, 2), (128, 128), (128, 128), n_swa=2048,
                              rope_type="norm", n_rot=128, rope_freq_base=5e5, rope_freq_base_swa=5e5)
# wider fixtures for the products: a dense NEOX source and a Muse-like iSWA target, heads of 64
WIDE_SOURCE = kvstate.ModelSpec("qwen3", ("full",) * 3, (2, 2, 2), (64, 64, 64), (64, 64, 64), rope_type="neox",
                                n_rot=64, rope_freq_base=1e6)
WIDE_TARGET = kvstate.ModelSpec("muse-glimmer", ("sliding", "sliding", "full", "sliding"), (2,) * 4, (64,) * 4,
                                (64,) * 4, n_swa=8, rope_type="norm", n_rot=64, rope_freq_base=5e5,
                                rope_freq_base_swa=1e4)
_SHARED = []


def shared_backend():
    """One GgmlBackend on the default device for the whole module."""
    if not _SHARED:
        _SHARED.append(ggml_backend.GgmlBackend())
    return _SHARED[0]


def wide_hub(models, *, dim=96, rank=40, view=64, seed=0):
    """{kind: HubKind} like test_translate.random_hub, of real-sized products: models {name: {kind: {layer:
    width}}} (kind_widths), stage-one ranks up to `rank`, a view of `view` and hub latents of `dim`."""
    rng = np.random.default_rng(seed)
    hubs = {}
    for kind in hub.KINDS:
        widths = {name: w[kind] for name, w in models.items()}
        transforms, encoders, decoders, means, z_mean = {}, {}, {}, {}, {}
        for name, w in widths.items():
            layers = sorted(w)
            ranks = [min(w[layer], rank) for layer in layers]
            bases = [np.asfortranarray(rng.standard_normal((w[l], r)).astype(np.float32) / np.sqrt(w[l]))
                     for l, r in zip(layers, ranks)]
            basis2 = np.asfortranarray(rng.standard_normal((sum(ranks), view)).astype(np.float32) / np.sqrt(sum(ranks)))
            transforms[name] = hub.ViewTransform(layers, [rng.standard_normal(w[l]).astype(np.float32) for l in layers],
                                                 bases, basis2, rng.uniform(0.5, 2.0, view).astype(np.float32))
            encoders[name] = (rng.standard_normal((view, dim)) / np.sqrt(view)).astype(np.float32)
            decoders[name] = [(rng.standard_normal((dim, w[l])) / np.sqrt(dim)).astype(np.float32) for l in layers]
            means[name] = [rng.standard_normal(w[l]).astype(np.float32) for l in layers]
            z_mean[name] = (rng.standard_normal(dim) / 10).astype(np.float32)
        names = list(widths)
        hubs[kind] = hub.HubKind(names, transforms, encoders, np.ones(dim), means, z_mean, decoders,
                                 {name: {"alpha": 0.01} for name in names})
    return hubs


def max_rel(got, ref):
    """max |got - ref| / max |ref| (bench_engine's measure, over the whole array)."""
    got, ref = np.asarray(got, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    return float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))


def rel_frobenius(got, ref):
    got, ref = np.asarray(got, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))


class SelectionTest(unittest.TestCase):
    """How backends.select reaches the ggml backend (runs without the library)."""

    def torch_absent(self, backend="torch"):
        return (False, "no torch") if backend == "torch" else ggml_backend.status()

    def test_vulkan_is_a_device_and_refused_without_the_library(self):
        self.assertEqual(backends.requested("Vulkan"), "vulkan")
        with mock.patch.dict(os.environ, {backends.ENV: "vulkan"}):
            self.assertEqual(backends.requested(None), "vulkan")
        with mock.patch.dict(os.environ, {ggml_backend.LIBRARY_ENV: "/nonexistent/libkvbridge.so"}):
            available, detail = backends.gpu_status("vulkan")
            self.assertFalse(available)
            self.assertIn("no libkvbridge at /nonexistent", detail)
            with self.assertRaisesRegex(RuntimeError, "no Vulkan backend: no libkvbridge"):
                backends.select("vulkan")
            with mock.patch.object(backends, "gpu_status", side_effect=self.torch_absent):
                self.assertEqual(backends.select("auto").name, "numpy")
                with self.assertRaisesRegex(RuntimeError, "no GPU backend: no torch; no libkvbridge"):
                    backends.select("gpu")

    def test_auto_takes_torch_then_vulkan_then_numpy(self):
        statuses = {"torch": (True, "cuda"), "vulkan": (True, "Vulkan0")}
        with mock.patch.object(backends, "gpu_status", side_effect=lambda backend="torch", **_: statuses[backend]), \
                mock.patch.object(backends, "_instance", side_effect=lambda name: name):
            self.assertEqual([backends.select(d) for d in ("auto", "gpu", "vulkan", "cpu")],
                             ["torch", "torch", "ggml", "numpy"])
            statuses["torch"] = (False, "no torch")
            self.assertEqual([backends.select(d) for d in ("auto", "gpu", "vulkan")], ["ggml", "ggml", "ggml"])
            statuses["vulkan"] = (False, "ggml sees no Vulkan GPU")
            self.assertEqual(backends.select("auto"), "numpy")
            with self.assertRaisesRegex(RuntimeError, "no Vulkan backend: ggml sees no Vulkan GPU"):
                backends.select("vulkan")

    @unittest.skipIf(MISSING, f"libkvbridge is not built ({MISSING})")
    def test_status_names_the_device(self):
        found = ggml_backend.devices()
        cpu = next(d for d in found if d["kind"] == "cpu")          # ggml's CPU backend is always there
        for named in ("cpu", str(cpu["index"])):
            with mock.patch.dict(os.environ, {ggml_backend.DEVICE_ENV: named}):
                # asked for by name or index: "vulkan" takes it, "auto" and "gpu" do not (not a GPU)
                available, detail = ggml_backend.status()
                self.assertFalse(available)
                self.assertIn("is ggml's cpu device, not a GPU", detail)
                self.assertEqual(backends.gpu_status("vulkan"), (available, detail))
                self.assertTrue(ggml_backend.status(gpu=False)[0])
                self.assertTrue(backends.gpu_status("vulkan", gpu=False)[0])
        backend = ggml_backend.GgmlBackend(str(cpu["index"]), threads=2)       # a device index
        try:
            self.assertEqual((backend.device, backend.on_gpu), ("CPU", False))
        finally:
            backend.close()
        with mock.patch.dict(os.environ, {ggml_backend.DEVICE_ENV: "Vulkan99"}):
            available, detail = ggml_backend.status()
            self.assertFalse(available)
            self.assertIn("no ggml device Vulkan99", detail)
        gpus = [d for d in found if d["kind"] in ("gpu", "igpu")]
        self.assertEqual(ggml_backend.status()[0], bool(gpus))


@unittest.skipIf(MISSING, f"libkvbridge is not built ({MISSING})")
class GgmlTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = shared_backend()


class RopeTest(GgmlTestCase):
    def assert_rotation_matches(self, spec, layer, *, positions=POSITIONS, atol=2e-6):
        rng = np.random.default_rng(layer)
        x = (rng.standard_normal((len(positions),) + spec.head_shape(layer, "k")) * 4).astype(np.float32)
        rope = spec.rope(layer)
        for inverse, reference in ((False, kvstate.cache_keys), (True, kvstate.pre_rope_keys)):
            got = self.backend.rotate(x, positions, rope, inverse=inverse)
            np.testing.assert_allclose(got, reference(spec, layer, x, positions), rtol=0, atol=atol * np.abs(x).max(),
                                       err_msg=f"{spec.architecture} layer {layer} inverse={inverse}")

    def test_rope_matches_kvstate_for_neox_norm_and_imrope(self):
        self.assert_rotation_matches(TARGET, 0)                       # NEOX over the whole head
        self.assert_rotation_matches(MUSE, 0)                         # NORM, sliding base
        self.assert_rotation_matches(HYBRID, 1)                       # IMROPE over 8 of 16 dims, one theta_e pair
        self.assert_rotation_matches(QWEN38, 3)                       # Qwen3.8's geometry
        self.assert_rotation_matches(MUSE_REAL, 0)                    # Muse's geometry
        self.assertIsNone(MUSE.rope(2))                               # Muse global layers are NoPE

    def test_yarn_offset_and_frequency_factors_match_kvstate(self):
        self.assert_rotation_matches(DEEPSEEK2, 0)                    # YaRN on the k_pe tail of [c_kv | k_pe]
        self.assert_rotation_matches(DOTS, 1)                         # a sliding MLA layer: base 500
        x = np.random.default_rng(1).standard_normal((len(POSITIONS), 1, 80)).astype(np.float32)
        factors = tuple(float(f) for f in np.linspace(1.0, 8.0, 16))
        for rope in (kvstate.Rope("norm", 64, 10000.0, offset=16, freq_scale=0.025, ext_factor=1.0, attn_factor=0.9,
                                  n_ctx_orig=4096),
                     kvstate.Rope("norm", 32, 10000.0, offset=3),       # an odd offset of NORM pairs
                     kvstate.Rope("neox", 32, 1e6, offset=48, freq_scale=0.5),
                     kvstate.Rope("neox", 32, 1e6, offset=8, freq_factors=factors)):
            for inverse, expected in ((False, rope.apply(x, POSITIONS)), (True, rope.invert(x, POSITIONS))):
                np.testing.assert_allclose(self.backend.rotate(x, POSITIONS, rope, inverse=inverse), expected, rtol=0,
                                           atol=2e-6 * np.abs(x).max(), err_msg=f"{rope} inverse={inverse}")

    def test_tables_are_kvstates_cos_sin(self):
        positions = np.concatenate([POSITIONS, np.arange(200000, 200512)])
        for rope in (TARGET.rope(0), MUSE_REAL.rope(0), QWEN38.rope(3), DEEPSEEK2.rope(0),
                     kvstate.Rope("norm", 64, 10000.0, offset=16, freq_scale=0.025, ext_factor=1.0, attn_factor=0.9,
                                  n_ctx_orig=4096)):
            cos, sin = ggml_backend.rope_tables(rope, positions)
            expected_cos, expected_sin = rope.cos_sin(positions)
            # libm cosf/sinf (not correctly rounded: 1-3% of these values) against float64 cos/sin
            # rounded to float32: one unit in the last place apart at most
            ulp = 2.0 ** -23 * max(1.0, float(rope.magnitude()))
            np.testing.assert_allclose(cos, expected_cos, rtol=0, atol=ulp)
            np.testing.assert_allclose(sin, expected_sin, rtol=0, atol=ulp)
            self.assertGreater(np.mean((cos == expected_cos) & (sin == expected_sin)), 0.95)
            c, s = ggml_backend.rope_tables(rope, positions, inverse=True)
            norm = cos.astype(np.float64) ** 2 + sin.astype(np.float64) ** 2
            np.testing.assert_array_equal(c, (cos / norm).astype(np.float32))
            np.testing.assert_array_equal(s, (-sin / norm).astype(np.float32))

    def test_mrope_components_reduce_to_text_positions(self):
        rope = HYBRID.rope(1)
        components = np.stack([POSITIONS, POSITIONS, POSITIONS, np.zeros_like(POSITIONS)], axis=1)
        for a, b in zip(ggml_backend.rope_tables(rope, POSITIONS), ggml_backend.rope_tables(rope, components)):
            np.testing.assert_array_equal(a, b)
        shifted = components.copy()
        shifted[:, 1] += 7                                            # an image cell: h differs from t
        self.assertFalse(np.array_equal(ggml_backend.rope_tables(rope, shifted)[0],
                                        ggml_backend.rope_tables(rope, POSITIONS)[0]))
        with self.assertRaises(NotImplementedError):
            ggml_backend.rope_tables(TARGET.rope(0), components)      # M-RoPE positions need IMROPE

    def test_supports_checks_each_rope_once_and_refuses_a_disagreeing_one(self):
        for spec in (SOURCE, TARGET, MUSE, HYBRID, DEEPSEEK2, DSA_TOP4, DOTS, KDA, QWEN38, MUSE_REAL):
            self.backend.supports(spec)
        fresh = ggml_backend.GgmlBackend()
        tables = ggml_backend.rope_tables
        try:
            def wrong(rope, positions, inverse=False):
                cos, sin = tables(rope, positions, inverse=inverse)
                return cos, -sin
            with mock.patch.object(ggml_backend, "rope_tables", side_effect=wrong):
                with self.assertRaisesRegex(NotImplementedError, "disagrees with kvstate"):
                    fresh.supports(TARGET)
        finally:
            fresh.close()

        class Odd:
            mode = "spiral"
        with self.assertRaises(NotImplementedError):
            ggml_backend._Ropes(POSITIONS, False).of(Odd())


def blocked_backend(largest):
    """A GgmlBackend on the default device whose device arrays go in row blocks of at most `largest`
    bytes (and the library's graphs in rows within it), as arrays beyond ggml-vulkan's 1 GiB do."""
    backend = ggml_backend.GgmlBackend()
    backend.max_buffer = int(largest)
    return backend


class ProductsTest(GgmlTestCase):
    """Every backend method on real-sized products, against NumpyBackend's."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.numpy = numpy_backend.NumpyBackend()
        cls.hubs = wide_hub({"src": kind_widths(WIDE_SOURCE), "tgt": kind_widths(WIDE_TARGET)})
        tokens = np.arange(300, dtype=np.int32) + 10
        state, _, _ = native_state(WIDE_SOURCE, tokens, np.random.default_rng(4), dtype="f16")
        cls.cache = translate.SourceCache(state, WIDE_SOURCE)
        cls.positions = np.random.default_rng(5).permutation(300)[:150]

    def sides(self, kind, backend):
        return backend.encoder(self.hubs, kind, "src"), backend.decoder(self.hubs, kind, "tgt")

    def test_encode_matches_numpy(self):
        for kind in hub.KINDS:
            numpy_encoder, numpy_decoder = self.sides(kind, self.numpy)
            reference = self.numpy.encode(numpy_encoder, self.cache, kind, self.positions, chunk_rows=64,
                                          center=numpy_decoder)
            encoder, decoder = self.sides(kind, self.backend)
            got = self.backend.host(self.backend.encode(encoder, self.cache, kind, self.positions, chunk_rows=64,
                                                        center=decoder))
            self.assertEqual(got.shape, reference.shape)
            self.assertLess(max_rel(got, reference), 1e-3, kind)
            self.assertLess(rel_frobenius(got, reference), 1e-3, kind)
            # uncentred, from the rows themselves (hub.EncoderSide.encode)
            side = self.hubs[kind].encoder_side("src")
            rows = {layer: self.cache.rows(kind, layer, self.positions) for layer in side.layers}
            plain = self.backend.host(self.backend.encode_rows(encoder, rows))
            self.assertLess(max_rel(plain, side.encode(rows)), 1e-3, kind)

    def test_section_matches_numpy_with_mapped_and_fill_rows(self):
        rng = np.random.default_rng(6)
        for kind in hub.KINDS:
            encoder, decoder = self.sides(kind, self.backend)
            reference_decoder = self.sides(kind, self.numpy)[1]
            z = self.backend.encode(encoder, self.cache, kind, self.positions, center=decoder)
            z_host = self.backend.host(z)
            for layers in ([0, 1, 3], [2], [0, 1, 2, 3]):
                width = sum(int(np.prod(WIDE_TARGET.head_shape(l, kind))) for l in layers)
                n_rows = 120
                mapped = rng.choice(len(z_host), 90, replace=False)
                rows = rng.permutation(n_rows)
                mapped_rows, fill_rows = np.sort(rows[:90]), np.sort(rows[90:])
                table = rng.standard_normal((len(fill_rows), width)).astype(np.float32)
                for fill in (None, table):
                    got = self.backend.host(self.backend.section(decoder, z, layers, n_rows, mapped, mapped_rows,
                                                                 fill_rows, fill, centered=True))
                    expected = self.numpy.section(reference_decoder, z_host, layers, n_rows, mapped, mapped_rows,
                                                  fill_rows, fill, centered=True)
                    self.assertLess(max_rel(got[mapped_rows], expected[mapped_rows]), 1e-3, (kind, layers))
                    np.testing.assert_array_equal(got[fill_rows], expected[fill_rows])     # copied, not computed
                # uncentred latents and every row in order (no gather, no scatter)
                plain = self.numpy.encode(self.sides(kind, self.numpy)[0], self.cache, kind, self.positions)
                got = self.backend.host(self.backend.section(decoder, self.backend.from_host(plain), layers,
                                                             len(plain), np.arange(len(plain)), np.arange(len(plain)),
                                                             np.zeros(0, dtype=np.int64)))
                expected = self.numpy.section(reference_decoder, plain, layers, len(plain), np.arange(len(plain)),
                                              np.arange(len(plain)), np.zeros(0, dtype=np.int64))
                self.assertLess(max_rel(got, expected), 1e-3, (kind, layers))

    def cache_rows_both(self, kind, layers, values, positions, dtype):
        expected = self.numpy.cache_rows(WIDE_TARGET, layers, kind, values.copy(), positions, dtype)
        got = self.backend.cache_rows(WIDE_TARGET, layers, kind, self.backend.from_host(values), positions, dtype)
        return got, expected

    def test_cache_rows_rotate_and_cast_like_numpy(self):
        rng = np.random.default_rng(7)
        layers = [0, 1, 2, 3]                                         # NORM sliding layers and a NoPE global one
        positions = np.arange(262143 - 299, 262144, dtype=np.int64)
        values = (rng.standard_normal((300, 4 * 128)) * 3).astype(np.float32)
        got, expected = self.cache_rows_both("k", layers, values, positions, "f32")
        for layer in layers:
            np.testing.assert_allclose(got[layer], expected[layer], rtol=0, atol=2e-6 * np.abs(values).max())
        np.testing.assert_array_equal(got[2], expected[2])            # NoPE: copied
        got, expected = self.cache_rows_both("v", layers, values, positions, "f32")
        for layer in layers:
            np.testing.assert_array_equal(got[layer], expected[layer])
        # the float16 cast of identical float32 values is numpy's, bit for bit: ties to even, subnormals,
        # the largest finite value and -0
        special = np.asarray([0.0, -0.0, 1.0 + 2.0 ** -11, 1.0 + 3 * 2.0 ** -11, 2.0 ** -24, 2.0 ** -25,
                              1.5 * 2.0 ** -25, 3e-6, -6e-8, 65504.0, 65519.99, -65519.0, 1e-3, 1 / 3], dtype=np.float32)
        values = (rng.standard_normal((300, 4 * 128)) * 100).astype(np.float32)
        values.reshape(-1)[:len(special)] = special
        got, expected = self.cache_rows_both("v", layers, values, positions, "f16")
        for layer in layers:
            self.assertEqual(got[layer].dtype, np.float16)
            np.testing.assert_array_equal(got[layer].view(np.uint16), expected[layer].view(np.uint16))
        values = (rng.standard_normal((300, 4 * 128)) * 100).astype(np.float32)
        got, expected = self.cache_rows_both("k", layers, values, positions, "f16")
        for layer in layers:
            # rotated in float32 (a few units of float32 rounding of the largest term apart), then cast: one
            # float16 rounding apart, or less than the float32 rounding where the rotation cancels
            a, b = got[layer].astype(np.float32), expected[layer].astype(np.float32)
            np.testing.assert_allclose(a, b, rtol=2.0 ** -10, atol=4e-6 * np.abs(values).max())
            self.assertGreater(float(np.mean(got[layer].view(np.uint16) == expected[layer].view(np.uint16))), 0.99)

    def test_cache_rows_refuse_values_not_finite_in_the_cache_dtype(self):
        values = np.ones((40, 4 * 128), dtype=np.float32)
        values[17, 300] = 70000.0                                     # layer 2 overflows float16
        with self.assertRaisesRegex(FloatingPointError, "layer 2 are not finite in f16"):
            self.backend.cache_rows(WIDE_TARGET, [0, 1, 2, 3], "v", values, np.arange(40), "f16")
        with self.assertRaises(kvstate.StateFormatError):              # what numpy, the fallback, then says
            self.numpy.cache_rows(WIDE_TARGET, [0, 1, 2, 3], "v", values, np.arange(40), "f16")
        self.backend.cache_rows(WIDE_TARGET, [0, 1, 2, 3], "v", values, np.arange(40), "f32")
        values[3, 5] = np.nan
        with self.assertRaisesRegex(FloatingPointError, "layer 0 are not finite in f32"):
            self.backend.cache_rows(WIDE_TARGET, [0, 1, 2, 3], "k", values, np.arange(40), "f32")

    def test_cache_rows_write_into_given_arrays_in_chunks(self):
        rng = np.random.default_rng(8)
        values = rng.standard_normal((50, 2 * 128)).astype(np.float32)
        held = {layer: np.zeros((80, 2, 64), dtype=np.float16) for layer in (0, 1)}
        out = {layer: held[layer][20:70] for layer in (0, 1)}
        result = self.backend.cache_rows(WIDE_TARGET, [0, 1], "k", self.backend.from_host(values), np.arange(50) + 9,
                                         "f16", out=out, chunk_rows=16)
        expected = self.numpy.cache_rows(WIDE_TARGET, [0, 1], "k", values.copy(), np.arange(50) + 9, "f16")
        for layer in (0, 1):
            self.assertIs(result[layer], out[layer])
            np.testing.assert_allclose(held[layer][20:70].astype(np.float32), expected[layer].astype(np.float32),
                                       rtol=2e-3, atol=1e-3)
            self.assertFalse(held[layer][:20].any() or held[layer][70:].any())
        with self.assertRaisesRegex(ValueError, "C-contiguous"):
            self.backend.cache_rows(WIDE_TARGET, [0], "k", values[:, :128], np.arange(50), "f16",
                                    out={0: np.zeros((50, 64, 2), dtype=np.float16).transpose(0, 2, 1)})

    def test_latent_shell_matches_numpy(self):
        rng = np.random.default_rng(6)
        norms = latent_norm_weights(DOTS)
        values = (rng.standard_normal((9, 44)) * 3).astype(np.float32)
        expected = self.numpy.renormalize_latents(DOTS, [0, 1, 2, 3], values.copy(), norms)
        device = self.backend.from_host(values)
        self.assertIs(self.backend.renormalize_latents(DOTS, [0, 1, 2, 3], device, norms), device)
        np.testing.assert_array_equal(self.backend.host(device), expected)      # kvstate's own arithmetic
        np.testing.assert_array_equal(self.backend.host(self.backend.from_host(values)), values)

    def test_resident_sides_report_device_bytes_and_release(self):
        backend = ggml_backend.GgmlBackend()
        try:
            before = backend.memory()["weights"]
            decoder = backend.decoder(self.hubs, "k", "tgt")
            backend.encoder(self.hubs, "k", "src")
            self.assertEqual(backend.memory()["weights"] - before, backend.resident_bytes())
            self.assertGreater(backend.resident_bytes(), 0)
            weight, mean = decoder.section([0, 1, 3])
            self.assertEqual((weight.rows, weight.cols, mean.cols), (3 * 128, 96, 3 * 128))
            decoder.section([2])                                      # every layer covered: per-layer copy released
            self.assertEqual(decoder.decoders, {})
            weight, mean = decoder.section([1])                       # rebuilt from a built section
            self.assertEqual(weight.rows, 128)
            keys = [tuple(item["key"]) for item in backend.resident()]
            del decoder, weight, mean                                 # the sides' last references go with them
            self.assertGreater(backend.evict(1, keep=[keys[1]]), 0)
            self.assertEqual([tuple(item["key"]) for item in backend.resident()], [keys[1]])
            self.assertGreater(backend.drop(), 0)
            self.assertEqual(backend.memory()["weights"], before)
            info = backend.describe()
            self.assertEqual((info["backend"], info["device"]), ("ggml", backend.device))
            self.assertIn("GGML_PREC_F32", info["compute"])
            memory = backend.memory()
            self.assertGreaterEqual(memory["peak_allocated"], memory["allocated"])
            backend._released()
            self.assertEqual((backend.memory()["pooled"], backend.memory()["graph"]), (0, 0))
        finally:
            backend.close()

    def test_warm_products_run_the_switch_shapes(self):
        encoder, decoder = self.sides("k", self.backend)
        self.backend.warm_products(encoder, decoder, 64, [(0, 1, 3), (2,)], spec=WIDE_TARGET, kind="k")

    def test_decoded_values_match_numpy(self):
        """translate.compute on one plan: every section's decoded rows within 1e-3 (max |err| / max |ref|)."""
        case = test_translate.SectionsTest()
        case.setUp()
        source_state, _, _ = native_state(WIDE_SOURCE, case.s_tok["token_id"], np.random.default_rng(1), dtype="f16")
        template, _, _ = native_state(WIDE_TARGET, case.t_tok["token_id"], np.random.default_rng(2), dtype="f16")
        layers = {kind: translate.source_layers(self.hubs, kind, "src") for kind in translate.KINDS}
        plan = translate.prepare(source_state, translate.Side("src", WIDE_SOURCE, case.s_rec, case.s_tok),
                                 translate.Side("tgt", WIDE_TARGET, case.t_rec, case.t_tok), layers,
                                 t0=len(case.t_tok["token_id"]) - 1, template=template)
        reference = translate.compute(plan, self.hubs, self.numpy)
        got = translate.compute(plan, self.hubs, self.backend)
        self.assertEqual(sorted(got), sorted(reference))
        for key, value in reference.items():
            self.assertLess(max_rel(self.backend.host(got[key]), value), 1e-3, key)


    def test_products_are_checked_against_numpy_once(self):
        self.backend.supports(WIDE_TARGET)
        self.assertEqual(self.backend._products, "")
        fresh = ggml_backend.GgmlBackend()
        try:
            fresh.max_buffer = self.backend.max_buffer
            with mock.patch.object(fresh, "host", side_effect=lambda x: np.zeros(x.shape, dtype=np.float32)):
                with self.assertRaisesRegex(NotImplementedError, "products on .* disagree with numpy"):
                    fresh.supports(WIDE_TARGET)
            with self.assertRaisesRegex(NotImplementedError, "disagree with numpy"):     # refused for good
                fresh.supports(WIDE_TARGET)
        finally:
            fresh.close()

    def test_latents_not_finite_are_refused(self):
        encoder, decoder = self.sides("k", self.backend)
        rows = {layer: np.full((6, width), 1e5, dtype=np.float32) for layer, width in encoder.widths.items()}
        # a float16 overflow of the centred rows: inf in z (numpy, the fallback, stays in float32)
        with self.assertRaisesRegex(FloatingPointError, "non-finite hub latents"):
            self.backend.encode_rows(encoder, rows, center=decoder)
        rows = {layer: np.ones((6, width), dtype=np.float32) for layer, width in encoder.widths.items()}
        self.assertTrue(np.isfinite(self.backend.host(self.backend.encode_rows(encoder, rows, center=decoder))).all())

    def test_empty_inputs_give_empty_arrays(self):
        encoder, decoder = self.sides("k", self.backend)
        none = np.zeros(0, dtype=np.int64)
        z = self.backend.encode(encoder, self.cache, "k", none, center=decoder)
        self.assertEqual((z.shape, self.backend.host(z).shape), ((0, 96), (0, 96)))
        values = self.backend.section(decoder, z, [0, 1], 0, none, none, none, centered=True)
        self.assertEqual(self.backend.host(values).shape, (0, 256))
        for given in (values, np.zeros((0, 256), dtype=np.float32)):
            got = self.backend.cache_rows(WIDE_TARGET, [0, 1], "k", given, none, "f16")
            self.assertEqual({layer: a.shape for layer, a in got.items()}, {0: (0, 2, 64), 1: (0, 2, 64)})
        with self.assertRaises(IndexError):
            self.backend.section(decoder, z, [0, 1], 4, none, none, np.asarray([4]))

    def test_an_array_beyond_the_largest_buffer_is_out_of_memory(self):
        backend = self.backend
        cols = 4096
        rows = backend.max_buffer // (4 * cols) + 1
        if rows * cols * 4 > 1 << 40:
            self.skipTest(f"{backend.device} has no largest buffer")
        with self.assertRaisesRegex(MemoryError, "exceeds the device's largest buffer") as caught:
            ggml_backend._Tensor(backend, backend.lib.kvb_new(backend.ctx, ggml_backend.F32, rows, cols, 0), "array")
        self.assertFalse(engine.device_fault(caught.exception))          # numpy retries; the GPU stays in use
        self.assertTrue(engine._fallback_worthy(caught.exception))
        # a DeviceArray of the same size is two blocks (refused before anything is allocated: only checked)
        self.assertEqual(-(-rows // backend.block_rows(cols)), 2)

    def test_uploads_keep_no_graph_buffer(self):
        backend = ggml_backend.GgmlBackend()
        try:
            backend._released()
            matrices = [np.random.default_rng(i).standard_normal((512, 1024)).astype(np.float32) for i in range(4)]
            for tensor in backend.matrices(matrices, transposed=True) + [backend.matrix(matrices[0], transposed=True)]:
                self.assertEqual(tensor.shape, (1024, 512))
            memory = backend.memory()
            self.assertEqual(memory["graph"], 0)                         # the uploads' allocator is gone
            self.assertGreaterEqual(memory["peak_allocated"], memory["weights"] + (2 << 20))   # the sources counted
        finally:
            backend.close()


@unittest.skipIf(MISSING, f"libkvbridge is not built ({MISSING})")
class BlockedProductsTest(ProductsTest):
    """ProductsTest with device arrays in row blocks: z and section rows of 8-170 rows per block."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.backend = blocked_backend(16 << 10)

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()

    def test_arrays_are_blocked(self):
        z = self.backend.from_host(np.arange(150 * 96, dtype=np.float32).reshape(150, 96))
        self.assertEqual([row0 for row0, _ in z.blocks], [0, 42, 84, 126])
        np.testing.assert_array_equal(self.backend.host(z), np.arange(150 * 96, dtype=np.float32).reshape(150, 96))
        self.assertEqual(z.nbytes, sum(tensor.nbytes for _, tensor in z.blocks))

    test_an_array_beyond_the_largest_buffer_is_out_of_memory = None
    test_uploads_keep_no_graph_buffer = None
    test_resident_sides_report_device_bytes_and_release = None


class TranslateTest(GgmlTestCase):
    def translate_both(self, source_spec, target_spec, subsets=None, hub_override=None, window_rows=None,
                       backend=None, **options):
        case = test_translate.SectionsTest()   # by module: not collected here a second time
        case.setUp()
        out = []
        for each in (backends.select("cpu"), backend or self.backend):
            case.rng = np.random.default_rng(11)
            state, report, _, _ = case.build(source_spec, target_spec, subsets=subsets, dtype="f16", backend=each,
                                             hub_override=hub_override, window_rows=window_rows, **options)
            out.append((kvstate.parse_file(state.to_bytes(), target_spec), report))
        return out

    def assert_same_state(self, target_spec, ref, ref_report, got, got_report):
        self.assertEqual((got_report["backend"], ref_report["backend"]), ("ggml", "numpy"))
        for key in ("t0", "cells", "mapped_cells", "fill_cells", "sections", "decoded_rows", "recurrent_reset"):
            self.assertEqual(got_report[key], ref_report[key], key)
        self.assertEqual(len(got.to_bytes()), len(ref.to_bytes()))
        for a, b in zip(got.state.parts, ref.state.parts):
            if isinstance(a, kvstate.RecurrentState):
                self.assertEqual(kvstate._write_recurrent(target_spec, a), kvstate._write_recurrent(target_spec, b))
                continue
            sa, sb = a.stream(), b.stream()
            np.testing.assert_array_equal(sa.positions, sb.positions)
            self.assertEqual(sorted(sa.v), sorted(sb.v))
            for layer in a.layers:
                for x, y in [(sa.k[layer], sb.k[layer])] + [(sa.v[l], sb.v[l]) for l in [layer] if l in sb.v]:
                    if not np.asarray(y).any():                       # nextn layers, zero indexer keys
                        self.assertFalse(np.asarray(x).any())
                        continue
                    self.assertLess(rel_frobenius(x, y), 2e-3)

    def test_translations_match_numpy_for_every_family(self):
        ridge = test_translate.random_indexer_ridge(DSA_TOP4)
        cases = [(SOURCE, TARGET, None, {}), (SOURCE, MUSE, None, {}), (SOURCE, HYBRID, None, {}),
                 (MUSE, HYBRID, {"src": list(MUSE.layers_of("full"))}, {}), (HYBRID, MUSE, None, {}),
                 # MLA, DSA (indexer ridge), DSA-iSWA and K-only hybrid targets, MLA sources
                 (SOURCE, DEEPSEEK2, None, {}), (DEEPSEEK2, TARGET, None, {}),
                 (SOURCE, DSA_TOP4, None, {"indexer_ridge": ridge}), (SOURCE, DOTS, None, {}), (HYBRID, KDA, None, {}),
                 (KDA, DEEPSEEK2, None, {})]
        for (source_spec, target_spec, subsets, options), window in [(case, None) for case in cases] + [(cases[1], 3)]:
            with self.subTest(source=source_spec.architecture, target=target_spec.architecture, window=window):
                (ref, ref_report), (got, got_report) = self.translate_both(source_spec, target_spec, subsets,
                                                                           window_rows=window, **options)
                self.assert_same_state(target_spec, ref, ref_report, got, got_report)

    def test_whole_section_stages_match_the_windowed_translation(self):
        case = test_translate.SectionsTest()
        case.setUp()
        states = []
        for module in (translate, test_translate.WholeSections):
            case.rng = np.random.default_rng(11)
            state, _, _, _ = case.build(SOURCE, MUSE, dtype="f16", backend=self.backend, module=module)
            states.append(state.to_bytes())
        self.assertEqual(states[0], states[1])

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


@unittest.skipIf(MISSING, f"libkvbridge is not built ({MISSING})")
class BlockedTranslateTest(TranslateTest):
    """The translations with every device array in row blocks of 64 bytes (1-2 rows of the fixtures)."""

    @classmethod
    def setUpClass(cls):
        cls.backend = blocked_backend(64)

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()

    def test_translations_use_several_blocks(self):
        made, array = [], self.backend.array

        def counted(rows, cols):
            out = array(rows, cols)
            made.append(len(out.blocks))
            return out
        with mock.patch.object(self.backend, "array", side_effect=counted):
            (ref, ref_report), (got, got_report) = self.translate_both(SOURCE, MUSE)
        self.assert_same_state(MUSE, ref, ref_report, got, got_report)
        self.assertGreater(max(made), 2)


@unittest.skipIf(MISSING, f"libkvbridge is not built ({MISSING})")
class CpuFallbackTest(TranslateTest):
    """ggml's CPU backend: what the library falls back to without a Vulkan device."""

    @classmethod
    def setUpClass(cls):
        cls.backend = ggml_backend.GgmlBackend("cpu", threads=4)

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()

    def test_translations_match_numpy_for_every_family(self):
        self.assertEqual((self.backend.device, self.backend.on_gpu), ("CPU", False))
        for source_spec, target_spec in ((SOURCE, MUSE), (HYBRID, MUSE), (SOURCE, DEEPSEEK2)):
            with self.subTest(source=source_spec.architecture, target=target_spec.architecture):
                (ref, ref_report), (got, got_report) = self.translate_both(source_spec, target_spec)
                self.assert_same_state(target_spec, ref, ref_report, got, got_report)
        self.backend.supports(QWEN38)

    test_packed_float16_bundle_gives_the_same_state = None
    test_whole_section_stages_match_the_windowed_translation = None


if __name__ == "__main__":
    unittest.main()
