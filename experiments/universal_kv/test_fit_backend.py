"""fit_backend.py: the fits' products on the CPU and (when torch sees a GPU) on the GPU, on small synthetic
data. The CPU fits are the algebra hub.py did before the backend (pooled decoder statistics, the dev R^2 in
closed form); a GPU fit matches the CPU fit to float16 rounding; the GPU products stay within the launch
budget (GPU RULE). GPU tests skip without torch or a GPU; run them under the GPU lock and the guard's
GPU-hang stop (campaign.py exec --memory-bytes N -- ~/venvs/rocm/bin/python -m unittest test_fit_backend)."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import fit_backend
import hub
from test_hub import make_views

HERE = Path(__file__).resolve().parent
GPU, WHY = fit_backend.gpu_available()


def quiet(line):
    return None


def relative(got, reference):
    got, reference = np.asarray(got, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    return float(np.abs(got - reference).max() / max(np.abs(reference).max(), 1e-30))


def columns(n, d, seed=0):
    """float32 [n, d] with columns of very different scales, an outlier column, a zero column and a column
    of 1e-26 (as K rows hold: squared in float32 on a GPU, which flushes subnormals, it vanishes)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32) * np.exp(rng.standard_normal(d)).astype(np.float32)
    x[:, 1] *= 3e4
    x[:, 2] = 0
    x[:, 3] += 500.0
    x[:, 4] *= 1e-26
    x[7, 5] = 1e-30
    return x


def gram_error(got, reference):
    """max |got - ref| / sqrt(ref_ii ref_jj): the error relative to each entry's natural scale."""
    scale = np.sqrt(np.outer(np.diag(reference), np.diag(reference))) + 1e-30
    return float(np.max(np.abs(got - reference) / scale))


class EigTest(unittest.TestCase):
    def test_top_eig_is_the_top_of_numpy_eigh(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((200, 120))
        cov = a @ a.T / 120 + np.diag(np.linspace(0, 1, 200))
        values, vectors = np.linalg.eigh(cov)
        for k, minimum in ((5, 100), (100, 100), (150, 100), (200, 100), (5, 4096)):   # subset (scipy) and full
            with mock.patch.object(fit_backend, "SUBSET_MIN", minimum):
                got_values, got_vectors = fit_backend.top_eig(cov, k, precision="float64")
            np.testing.assert_allclose(got_values, np.clip(values[::-1][:k], 0, None), rtol=1e-10, atol=1e-12)
            # the same eigenvectors up to sign
            overlap = np.abs(np.sum(got_vectors * vectors[:, ::-1][:, :k], axis=0))
            np.testing.assert_allclose(overlap, 1.0, atol=1e-8)
        self.assertEqual(hub._top_eig(cov, 7)[0].shape, (7,))
        with mock.patch.object(fit_backend, "SUBSET_MIN", 100):
            got_values, got_vectors = fit_backend.top_eig(cov, 50, precision="float32")
        self.assertEqual((got_values.dtype, got_vectors.dtype), (np.float64, np.float64))
        np.testing.assert_allclose(got_values, values[::-1][:50], rtol=1e-5)
        cosines = np.linalg.svd(vectors[:, ::-1][:, :50].T @ got_vectors, compute_uv=False)
        self.assertGreater(cosines.min(), 1 - 1e-4)                      # the same subspace
        with self.assertRaises(ValueError):
            fit_backend.top_eig(cov, 5, precision="float16")
        self.assertEqual(fit_backend.LARGE_EIGH, os.environ.get("KVPARK_FIT_LARGE_EIGH", "float32"))
        with mock.patch.object(fit_backend, "SUBSET_MIN", 100):        # the default: float32 for large subsets
            np.testing.assert_allclose(fit_backend.top_eig(cov, 50)[0], values[::-1][:50], rtol=1e-5)


class SelectionTest(unittest.TestCase):
    def test_devices(self):
        self.assertEqual(fit_backend.resolve("cpu"), "cpu")
        self.assertEqual(fit_backend.resolve("auto"), "gpu" if GPU else "cpu")
        with self.assertRaises(ValueError):
            fit_backend.resolve("tpu")
        with mock.patch.object(fit_backend, "gpu_available", lambda: (False, "no GPU here")):
            self.assertEqual(fit_backend.resolve("auto"), "cpu")
            with self.assertRaisesRegex(RuntimeError, "no GPU here"):
                fit_backend.resolve("gpu")
        self.assertIsInstance(fit_backend.make("cpu"), fit_backend.NumpyBackend)

    def test_using_installs_and_restores_the_backend(self):
        self.assertIs(fit_backend.current(), fit_backend.CPU)
        backend = fit_backend.NumpyBackend()
        with fit_backend.using(backend):
            self.assertIs(fit_backend.current(), backend)
            self.assertIs(hub.Moments(3).backend, backend)
        self.assertIs(fit_backend.current(), fit_backend.CPU)


class CpuBackendTest(unittest.TestCase):
    def test_moments_and_products(self):
        x = columns(500, 9)
        m = hub.Moments(9)
        for part in hub.chunks(500, 120):
            m.add(x[part])
        np.testing.assert_allclose(m.mean, x.astype(np.float64).mean(axis=0), rtol=1e-6, atol=1e-6)
        reference = np.cov(x.astype(np.float64), rowvar=False, bias=True)
        self.assertLess(gram_error(m.covariance(), reference), 1e-6)
        with self.assertRaises(RuntimeError):
            m.add(x[:10])                                # the sums were read
        pooled = hub._pool([m, m])
        self.assertEqual(pooled[0], 1000)
        np.testing.assert_allclose(pooled[2], 2 * 500 * m.covariance())

    def test_pooled_decoder_statistics_and_the_closed_form_dev_r2(self):
        """_DecoderStats against the direct computation: tiled rows, every alpha's dev predictions."""
        rng = np.random.default_rng(3)
        D, widths, repeat, n, n_dev = 6, [4, 3], 3, 900, 200
        latent = rng.standard_normal((n + n_dev, D))
        zs = [(latent + 0.3 * rng.standard_normal(latent.shape)).astype(np.float32) for _ in range(repeat)]
        ys = [(latent @ rng.standard_normal((D, w)) + 0.5 * rng.standard_normal((n + n_dev, w)) + 2).astype(np.float32)
              for w in widths]
        backend = fit_backend.CPU
        stats = hub._DecoderStats(backend, D, widths, repeat)
        moments = [hub.Moments(D) for _ in range(repeat)]
        dev_moments = [hub.Moments(D) for _ in range(repeat)]
        for part in hub.chunks(n, 250):
            for m, z in zip(moments, zs):
                m.add(z[part])
            stats.add(backend.total([z[part] for z in zs]), sum(z[part].sum(axis=0, dtype=np.float64) for z in zs),
                      [y[part] for y in ys])
        stats.finish(moments)
        for part in hub.chunks(n_dev, 70):
            rows = slice(n + part.start, n + part.stop)
            for m, z in zip(dev_moments, zs):
                m.add(z[rows])
            stats.add_dev(backend.total([z[rows] for z in zs]), [y[rows] for y in ys])
        alphas = (1e-4, 1e-2, 1.0, 10.0)
        weights, means, z_mean, best, results = stats.solve(alphas, dev_moments)
        # the direct computation, as hub.py did it before
        z_train = np.concatenate([z[:n] for z in zs]).astype(np.float64)
        y_train = [np.tile(y[:n], (repeat, 1)).astype(np.float64) for y in ys]
        mu, y_mu = z_train.mean(axis=0), [y.mean(axis=0) for y in y_train]
        gram = (z_train - mu).T @ (z_train - mu)
        evals, evecs = np.linalg.eigh(gram)
        evals = np.clip(evals, 0, None)
        z_dev = np.concatenate([z[n:] for z in zs]).astype(np.float64) - mu
        y_dev = [np.tile(y[n:], (repeat, 1)).astype(np.float64) for y in ys]
        total = sum(float(np.sum(np.square(y - y.mean(axis=0)))) for y in y_dev)
        direct = {}
        for alpha in alphas:
            lam = alpha * float(evals.mean())
            ws = [np.linalg.solve(gram + lam * np.eye(D), (z_train - mu).T @ (y - m)) for y, m in zip(y_train, y_mu)]
            sse = sum(float(np.sum(np.square(y - (z_dev @ w + m)))) for y, w, m in zip(y_dev, ws, y_mu))
            direct[alpha] = 1 - sse / total
            if alpha == best:
                for w, got in zip(ws, weights):
                    self.assertLess(relative(got, w), 1e-5)
        for alpha in alphas:
            self.assertAlmostEqual(results[alpha], direct[alpha], places=6)
        self.assertEqual(best, max(direct, key=direct.get))
        np.testing.assert_allclose(z_mean, mu, atol=1e-5)
        for m, reference in zip(means, y_mu):
            np.testing.assert_allclose(m, reference, rtol=1e-5)

    def test_prefetching_reads_every_chunk_once_and_changes_nothing(self):
        class Prefetching(fit_backend.NumpyBackend):
            prefetch = True                             # as the GPU backend: the next chunk read on a thread
        loaded = []

        def load(part):
            loaded.append(part.start)
            return part.start
        with fit_backend.using(Prefetching()):
            self.assertEqual([(p.start, d) for p, d in hub._loaded_chunks(1000, 300, load)],
                             [(0, 0), (300, 300), (600, 600), (900, 900)])
        self.assertEqual(sorted(loaded), [0, 300, 600, 900])
        views, spans, _ = make_views(n=1200)
        fits = []
        for backend in (fit_backend.CPU, Prefetching()):
            with fit_backend.using(backend):
                fits.append(hub.fit_hub_kind(views, spans, "k", layer_rank=8, view_dim=12, hub_dim=6, chunk=300,
                                             log=quiet))
        for name in fits[0].names:
            for a, b in zip(fits[0].decoders[name], fits[1].decoders[name]):
                np.testing.assert_array_equal(a, b)

    def test_evaluate_reads_each_chunk_once(self):
        views, spans, _ = make_views(n=1200)
        fitted = hub.fit_hub_kind(views, spans, "k", layer_rank=8, view_dim=12, hub_dim=6, chunk=300, log=quiet)
        reads = []
        for view in views:
            original = view.rows

            def counted(kind, layer, rows, original=original, name=view.name):
                reads.append(name)
                return original(kind, layer, rows)
            view.rows = counted
        results = hub.evaluate(fitted, views, spans, "k", chunk=300)
        self.assertEqual(len(results), 9)
        test_chunks = -(-len(spans.select("test")) // 300)
        per_chunk = sum(len(v.layers("k")) for v in views)
        self.assertEqual(len(reads), test_chunks * per_chunk)     # not once per pair


@unittest.skipUnless(GPU, f"no GPU for the torch fit backend: {WHY}")
class GpuBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lines = []
        cls.gpu = fit_backend.TorchBackend(log=cls.lines.append)

    def test_calibration_sets_a_launch_budget(self):
        item = json.loads(self.lines[0])
        self.assertEqual(item["stage"], "fit-backend")
        self.assertLessEqual(self.gpu.budget, fit_backend.MAX_LAUNCH_FLOPS)
        self.assertGreater(self.gpu.rate["accumulate"], 1e12)       # float16 products, not the 0.07 TFLOPS float32 path
        self.assertLess(self.gpu.longest_s, 0.25)

    def test_grams_and_cross_products_at_the_float16_floor(self):
        x, y = columns(3000, 70), columns(3000, 33, seed=5)
        for symmetric in (True, False):
            acc = self.gpu.zeros(70, 70, symmetric=symmetric)
            for part in hub.chunks(3000, 1100):
                self.gpu.add_gram(acc, x[part])
            got = self.gpu.fetch(acc)
            reference = x.astype(np.float64).T @ x.astype(np.float64)
            self.assertLess(gram_error(got, reference), 2e-4)
            np.testing.assert_array_equal(got, got.T) if symmetric else None
        acc = self.gpu.zeros(70, 33)
        self.gpu.add_cross(acc, x, y)
        reference = x.astype(np.float64).T @ y.astype(np.float64)
        scale = np.sqrt(np.outer((x.astype(np.float64) ** 2).sum(0), (y.astype(np.float64) ** 2).sum(0))) + 1e-30
        self.assertLess(float(np.max(np.abs(self.gpu.fetch(acc) - reference) / scale)), 2e-4)

    def test_tiles_mirror_and_launches_split_to_the_budget(self):
        x = columns(2000, 150)
        reference = x.astype(np.float64).T @ x.astype(np.float64)
        with mock.patch.object(fit_backend, "TILE", 64), mock.patch.object(self.gpu, "budget", 2.0 * 512 * 64 * 64):
            before = self.gpu.launches
            acc = self.gpu.zeros(150, 150, symmetric=True)
            self.gpu.add_gram(acc, x)
            got = self.gpu.fetch(acc)
            # 3 tiles per side: 6 upper-triangle launches per 512-row step, 4 steps
            self.assertEqual(self.gpu.launches - before, 6 * 4)
        self.assertLess(gram_error(got, reference), 2e-4)
        np.testing.assert_array_equal(got, got.T)

    def test_scaled_operands_are_reused_only_while_unchanged(self):
        x = self.gpu.put(columns(500, 20))
        y = self.gpu.put(columns(500, 7, seed=4))
        first = self.gpu._scaled16(x)
        self.assertIs(self.gpu._scaled16(x), first)                  # the same estimates for every layer
        for other in (y, self.gpu.put(columns(500, 7, seed=6)), y):
            self.gpu._scaled16(other)
        self.assertIs(self.gpu._scaled16(x), first)                  # recently used: kept
        x.mul_(2.0)                                                  # changed in place: scaled again
        acc = self.gpu.zeros(20, 7)
        self.gpu.add_cross(acc, x, y)
        reference = self.gpu.host(x).astype(np.float64).T @ self.gpu.host(y).astype(np.float64)
        self.assertLess(relative(self.gpu.fetch(acc), reference), 1e-3)

    def test_projections_do_not_overflow_float16(self):
        x = columns(500, 300) * 1e3                    # rows with norms far past float16's range
        x[9] = 1e-27                                   # and a row of 1e-27
        rng = np.random.default_rng(2)
        weight = (rng.standard_normal((300, 40)) * 1e3).astype(np.float32)
        got = self.gpu.host(self.gpu.mm(x, weight))
        reference = x.astype(np.float64) @ weight.astype(np.float64)
        self.assertTrue(np.isfinite(got).all())
        row_scale = np.linalg.norm(x.astype(np.float64), axis=1)[:, None] * np.linalg.norm(weight, axis=0)[None, :]
        self.assertLess(float(np.max(np.abs(got - reference) / (row_scale + 1e-30))), 2e-3)

    def test_a_gpu_fit_matches_the_cpu_fit(self):
        views, spans, _ = make_views(n=3000)
        fits = {}
        for name, backend in (("cpu", fit_backend.CPU), ("gpu", self.gpu)):
            with fit_backend.using(backend):
                fitted = hub.fit_hub_kind(views[:2], spans, "k", layer_rank=8, view_dim=12, hub_dim=6, chunk=300,
                                          log=quiet, source_layers={"b": [0, 2, 4]})
                hub.add_model_kind(fitted, views[2], views[:2], spans, "k", layer_rank=8, view_dim=12, chunk=300,
                                   log=quiet, encoder="consensus", consensus=["a", "b"])
                fits[name] = (fitted, hub.evaluate(fitted, views, spans, "k", chunk=300))
        (cpu, cpu_r2), (gpu, gpu_r2) = fits["cpu"], fits["gpu"]
        for pair in cpu_r2:
            self.assertLess(abs(cpu_r2[pair]["r2"] - gpu_r2[pair]["r2"]), 2e-3, pair)
        # the hub latent's basis is arbitrary within its (here nearly degenerate) eigenspaces, so compare
        # translations, not adapters
        rows = spans.select("test")
        by_name = {v.name: v for v in views}
        for source in cpu.names:
            sent = {layer: by_name[source].rows("k", layer, rows) for layer in cpu.source_layers(source)}
            for target in cpu.names:
                self.assertEqual(cpu.alphas[target]["alpha"], gpu.alphas[target]["alpha"])
                for index in range(len(by_name[target].layers("k"))):
                    expected = cpu.decode(target, cpu.encode(source, sent), index)
                    self.assertLess(relative(gpu.decode(target, gpu.encode(source, sent), index), expected), 1e-2,
                                    (source, target, index))

    def test_heads_and_mla_views_on_the_gpu(self):
        import test_hub
        views, spans, latent = make_views(n=3000)
        mla = test_hub.FakeMlaView("m", latent, 2, 5)
        spans.rows["m"] = np.arange(3000)
        target = test_hub.FakeView("t", latent, [10], 7)          # a model with a head kind ("csa") besides k, v
        rng = np.random.default_rng(8)
        target.data["csa", 0] = (latent["k"] @ rng.standard_normal((6, 9)) + 0.05 * rng.standard_normal((3000, 9))
                                 ).astype(np.float32)
        target._layers["csa"] = [0]
        spans.rows["t"] = np.arange(3000)
        out = {}
        for name, backend in (("cpu", fit_backend.CPU), ("gpu", self.gpu)):
            with fit_backend.using(backend):
                fitted = hub.fit_hub_kind(views[:2] + [mla], spans, "k", layer_rank=8, view_dim=10, hub_dim=6,
                                          chunk=256, log=quiet)
                head = hub.fit_head(fitted, views[:2] + [mla], target, spans, "csa", chunk=256, log=quiet)
                out[name] = (hub.evaluate(fitted, views[:2] + [mla], spans, "k", chunk=256), head)
        (cpu_r2, cpu_head), (gpu_r2, gpu_head) = out["cpu"], out["gpu"]
        for pair in cpu_r2:
            self.assertGreater(cpu_r2[pair]["r2"], 0.9, pair)
            self.assertLess(abs(cpu_r2[pair]["r2"] - gpu_r2[pair]["r2"]), 2e-3, pair)
        for pivot, value in cpu_head.alphas["t"]["test_r2"].items():
            self.assertGreater(value, 0.9)
            self.assertLess(abs(value - gpu_head.alphas["t"]["test_r2"][pivot]), 2e-3, pivot)
        self.assertEqual(cpu_head.alphas["t"]["alpha"], gpu_head.alphas["t"]["alpha"])


class FitHubDeviceTest(unittest.TestCase):
    """fit_hub.py --fit-device on three small in-memory views with a fake machine (test_resource_guard)."""

    def fit_main(self, device):
        import contextlib
        import fcntl
        import io
        import types
        import fit_hub
        import resource_guard
        from test_resource_guard import FakeMachine
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fake = FakeMachine(directory.name)
        views, spans, _ = make_views(n=1200)
        for view in views:
            view.captures = types.SimpleNamespace(rows_total=1200)
        by_name = {view.name: view for view in views}
        output = fake.root / "hub"
        argv = [arg for view in views for arg in ("--model", f"{view.name}=/cap/{view.name}@/records.jsonl")]
        argv += ["--output", str(output), "--hub-dim", "6", "--view-dim", "12", "--layer-rank", "8", "--chunk", "300",
                 "--fit-device", device]
        seen = {}
        original = fit_hub.fit

        def fit(args, views_, spans_, log, started):
            seen["backend"] = fit_backend.current()
            with open(fake.root / "gpu.lock", "a") as other:
                try:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    seen["gpu_lock"] = False
                except BlockingIOError:
                    seen["gpu_lock"] = True
            return original(args, views_, spans_, log, started)
        with mock.patch.dict(os.environ, fake.environ(KVPARK_GUARD_POLL_S="0.05")), \
                mock.patch.object(fit_hub.hub, "CaptureView", lambda name, captures, records: by_name[name]), \
                mock.patch.object(fit_hub.hub, "joint_spans", lambda views_: spans), \
                mock.patch.object(fit_hub, "fit", fit), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fit_hub.main(argv), 0)
        manifest = json.loads((output / "manifest.json").read_text())
        logged = [json.loads(line) for line in (fake.root / "hub.log.jsonl").read_text().splitlines()]
        return seen, manifest, logged, json.loads((output / "evaluation.json").read_text())

    def test_a_cpu_fit_takes_no_gpu_lock(self):
        seen, manifest, logged, _ = self.fit_main("cpu")
        self.assertIsInstance(seen["backend"], fit_backend.NumpyBackend)
        self.assertFalse(seen["gpu_lock"])
        self.assertEqual(manifest["settings"]["fit_device"], "cpu")
        self.assertEqual(manifest["settings"]["large_eigh"], fit_backend.LARGE_EIGH)
        self.assertIn("fit-backend-done", [item["stage"] for item in logged])
        self.assertIs(fit_backend.current(), fit_backend.CPU)

    @unittest.skipUnless(GPU, f"no GPU for the torch fit backend: {WHY}")
    def test_a_gpu_fit_holds_the_gpu_lock_and_matches_the_cpu_fit(self):
        seen, manifest, logged, gpu_evaluation = self.fit_main("gpu")
        self.assertIsInstance(seen["backend"], fit_backend.TorchBackend)
        self.assertTrue(seen["gpu_lock"])
        self.assertEqual(manifest["settings"]["fit_device"], "gpu")
        stages = [item["stage"] for item in logged]
        self.assertLess(stages.index("guard"), stages.index("fit-backend"))      # the GPU lock, then calibration
        self.assertEqual(logged[stages.index("guard")]["event"], "gpu-lock")
        _, _, _, cpu_evaluation = self.fit_main("cpu")
        for kind in hub.KINDS:
            for pair, item in cpu_evaluation[kind].items():
                self.assertLess(abs(item["r2"] - gpu_evaluation[kind][pair]["r2"]), 2e-3, (kind, pair))


class ReencodeMainTest(unittest.TestCase):
    """fit_hub.py --base BUNDLE --reencode: only the onboarded model's encoder is fitted again; the bundle grows
    from the base the onboarding grew from (its files linked), so it assembles like an onboarding."""

    def test_reencode_keeps_transforms_and_decoders_and_links_the_base(self):
        import contextlib
        import io
        import types
        import fit_hub
        from test_resource_guard import FakeMachine
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fake = FakeMachine(directory.name)
        views, spans, _ = make_views(n=1200)
        for view in views:
            view.captures = types.SimpleNamespace(rows_total=1200)
        by_name = {view.name: view for view in views}
        root = fake.root

        def run(argv, sizes=("--view-dim", "12", "--layer-rank", "8")):
            with mock.patch.dict(os.environ, fake.environ(KVPARK_GUARD_POLL_S="0.05")), \
                    mock.patch.object(fit_hub.hub, "CaptureView", lambda name, captures, records: by_name[name]), \
                    mock.patch.object(fit_hub.hub, "joint_spans", lambda views_: spans), \
                    contextlib.redirect_stdout(io.StringIO()):
                return fit_hub.main(argv + ["--chunk", "300", "--fit-device", "cpu", *sizes])
        model = {name: ["--model", f"{name}=/cap/{name}@/records.jsonl"] for name in by_name}
        self.assertEqual(run(model["a"] + model["b"] + ["--output", str(root / "base"), "--hub-dim", "6"]), 0)
        onboard = model["c"] + model["a"] + model["b"]
        self.assertEqual(run(onboard + ["--base", str(root / "base"), "--link-base", "--evaluate", "new",
                                        "--output", str(root / "onboarded")]), 0)
        onboarded_settings = json.loads((root / "onboarded" / "manifest.json").read_text())["settings"]
        self.assertEqual((onboarded_settings["onboard_encoder"], onboarded_settings["onboard_variance_cap"]),
                         (hub.DEFAULT_ONBOARD_ENCODER, hub.ONBOARD_VARIANCE_CAP))
        # a re-encode as campaign.py runs it: no --view-dim, --layer-rank or --fit-rows (their defaults differ from
        # the onboarding's), which the bundle records as its transforms were fitted
        self.assertEqual(run(onboard + ["--base", str(root / "onboarded"), "--reencode", "--onboard-encoder", "match",
                                        "--link-base", "--evaluate", "new", "--output", str(root / "match")],
                             sizes=()), 0)
        manifest = json.loads((root / "match" / "manifest.json").read_text())
        settings = manifest["settings"]
        self.assertEqual((settings["base"], settings["reencoded_from"], settings["onboard_encoder"]),
                         (str((root / "base").resolve()), str((root / "onboarded").resolve()), "match"))
        self.assertEqual(settings["consensus_views"], ["a", "b"])
        self.assertEqual(settings["hub_dim"], 6)
        for key in fit_hub.REENCODE_KEPT:
            self.assertEqual(settings[key], onboarded_settings[key], key)
        self.assertEqual((settings["view_dim"], settings["layer_rank"]), (12, 8))
        logged = [json.loads(line) for line in (root / "match.log.jsonl").read_text().splitlines()]
        diagnostics = [item for item in logged if item["stage"] == "onboard-encoder"]
        self.assertEqual(sorted(item["kind"] for item in diagnostics), sorted(hub.KINDS))
        self.assertTrue(all(item["method"] == "match" and 0 < item["min_r2"] <= 1 for item in diagnostics))
        for name in ("a", "b"):                          # the base's own files, linked
            for kind in hub.KINDS:
                entry = manifest["files"][f"{name}.{kind}"]["file"]
                self.assertEqual((root / "match" / entry).stat().st_ino, (root / "base" / entry).stat().st_ino)
        onboarded, _ = hub.load_hub(root / "onboarded")
        reencoded, _ = hub.load_hub(root / "match")
        for kind in hub.KINDS:
            for before, after in zip(onboarded[kind].decoders["c"], reencoded[kind].decoders["c"]):
                np.testing.assert_array_equal(before, after)
            np.testing.assert_array_equal(onboarded[kind].transforms["c"].basis2, reencoded[kind].transforms["c"].basis2)
            self.assertFalse(np.array_equal(reencoded[kind].encoders["c"], onboarded[kind].encoders["c"]))
        merged = hub.merge_bundles(root / "assembled", [root / "base", root / "match"])   # grown from the base
        self.assertEqual(sorted(merged["models"]), ["a", "b", "c"])
        # a base model or a bundle without a recorded base cannot be re-encoded
        with self.assertRaisesRegex(SystemExit, "a model of the base"):
            run(model["a"] + model["b"] + ["--base", str(root / "onboarded"), "--reencode", "--output", str(root / "x")])
        with self.assertRaisesRegex(SystemExit, "holds no c"):
            run(onboard + ["--base", str(root / "base"), "--reencode", "--output", str(root / "y")])
        self.assertEqual(run(onboard + ["--output", str(root / "joint"), "--hub-dim", "6"]), 0)
        with self.assertRaisesRegex(SystemExit, "records no base"):
            run(onboard + ["--base", str(root / "joint"), "--reencode", "--output", str(root / "y")])
        with self.assertRaisesRegex(SystemExit, "need --base"):
            run(onboard + ["--reencode", "--output", str(root / "z")])

if __name__ == "__main__":
    unittest.main()
