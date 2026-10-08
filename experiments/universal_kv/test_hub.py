import errno
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import hub


class FakeView:
    """In-memory view: per (kind, layer) rows = latent @ mixing + noise."""

    def __init__(self, name, latent, layer_dims, seed, noise=0.05, architecture="fake"):
        rng = np.random.default_rng(seed)
        self.name, self.architecture = name, architecture
        self.data, self._layers = {}, {}
        for kind in hub.KINDS:
            self._layers[kind] = list(range(len(layer_dims)))
            for layer, dim in enumerate(layer_dims):
                mixing = rng.standard_normal((latent[kind].shape[1], dim)).astype(np.float32)
                rows = latent[kind] @ mixing + noise * rng.standard_normal((len(latent[kind]), dim))
                self.data[kind, layer] = (rows + rng.standard_normal(dim)).astype(np.float32)

    def layers(self, kind):
        return self._layers[kind]

    def shape(self, kind, layer):
        return (1, self.data[kind, layer].shape[1])

    def width(self, kind, layer):
        return self.data[kind, layer].shape[1]

    def rows(self, kind, layer, rows):
        return self.data[kind, layer][rows]


class FakeMlaView(FakeView):
    """In-memory view of an MLA model: per layer the K row [c_kv | k_pe] and V = its c_kv view, both
    from the shared K latent; optionally DSA indexer keys (key "ik"), a noisy linear map of the K row."""

    def __init__(self, name, latent, layers, seed, *, kv_lora=12, rope=4, indexer=None, splits=None, noise=0.05):
        rng = np.random.default_rng(seed)
        self.name, self.architecture = name, "deepseek32" if indexer else "deepseek2"
        self.data, self._layers = {}, {"k": list(range(layers)), "v": list(range(layers)), "ik": []}
        n = len(latent["k"])
        for layer in range(layers):
            mixing = rng.standard_normal((latent["k"].shape[1], kv_lora + rope)).astype(np.float32)
            row = latent["k"] @ mixing + noise * rng.standard_normal((n, kv_lora + rope)) + rng.standard_normal(
                kv_lora + rope)
            self.data["k", layer] = row.astype(np.float32)
            self.data["v", layer] = self.data["k", layer][:, :kv_lora].copy()
            if indexer:
                self.data["ik", layer] = (row @ rng.standard_normal((kv_lora + rope, indexer)) / 3
                                          + noise * rng.standard_normal((n, indexer))).astype(np.float32)
                self._layers["ik"].append(layer)
        self.splits = splits

    def split_rows(self, split):
        return np.flatnonzero(self.splits == split)


def make_views(n=3000, latent_dim=6, seed=0):
    rng = np.random.default_rng(seed)
    latent = {kind: rng.standard_normal((n, latent_dim)).astype(np.float32) for kind in hub.KINDS}
    views = [FakeView("a", latent, [16, 16, 16], 1), FakeView("b", latent, [8] * 5, 2),
             FakeView("c", latent, [24, 12], 3)]
    documents = n // 100
    spans = hub.Spans([f"d{i}" for i in range(documents)],
                      ["train"] * (documents - 6) + ["dev"] * 3 + ["test"] * 3,
                      np.repeat(np.arange(documents), 100), {v.name: np.arange(n) for v in views})
    return views, spans, latent


class TokenCaptures:
    """The capture-set interface joint_spans reads: one document's byte ends and special flags."""

    def __init__(self, document_id, byte_end, special):
        self.document_ids = [document_id]
        self.byte_end = np.asarray(byte_end, dtype=np.int64)
        self.byte_start = np.concatenate([[0], self.byte_end[:-1]]).astype(np.int64)
        self.special = np.asarray(special, dtype=np.uint8)

    def __contains__(self, document_id):
        return document_id in self.document_ids

    def rows(self, document_id):
        return slice(0, len(self.byte_end))


class JointSpansTest(unittest.TestCase):
    def test_tokens_that_spell_nothing_are_no_spans(self):
        # a SentencePiece token kv_probe leaves empty (after a special token) repeats the byte end before it
        record = {"document_id": "d0", "text": "<s>ab cd ef", "segments": [[0, 3, 11]], "segment_keys": ["0:content"],
                  "split": "train"}

        class View:
            def __init__(self, name, byte_end, special):
                self.name, self.records = name, {"d0": record}
                self.captures = TokenCaptures("d0", byte_end, special)
        views = [View("a", [3, 5, 8, 11], [1, 0, 0, 0]), View("b", [3, 3, 5, 5, 8, 11], [1, 0, 0, 0, 0, 0])]
        spans = hub.joint_spans(views)
        self.assertEqual(spans.document_ids, ["d0"])
        np.testing.assert_array_equal(spans.rows["a"], [1, 2, 3])
        np.testing.assert_array_equal(spans.rows["b"], [2, 4, 5])


class HubTest(unittest.TestCase):
    def test_shared_latent_translates_between_every_pair(self):
        views, spans, _ = make_views()
        logs = []
        hubs = {kind: hub.fit_hub_kind(views, spans, kind, layer_rank=8, view_dim=12, hub_dim=6, chunk=700,
                                       log=logs.append) for kind in hub.KINDS}
        for kind in hub.KINDS:
            results = hub.evaluate(hubs[kind], views, spans, kind, chunk=500)
            for (source, target), item in results.items():
                self.assertGreater(item["r2"], 0.97, (kind, source, target, item["r2"]))
                self.assertEqual(item["rows"], 300)
            self.assertTrue(np.all(hubs[kind].eigenvalues[:6] > 2.5))   # 3 views, near-perfect correlation
            self.assertTrue(np.all(hubs[kind].eigenvalues <= 3 + 1e-6))  # sums of canonical correlations
        self.assertTrue(any(json.loads(line)["stage"] == "gcca" for line in logs))

    def test_view_specific_variance_is_not_shared(self):
        # whitening on a subsample can leave view-specific directions with variance > 1; the
        # generalized eigenproblem must still rank the truly shared directions first
        views, spans, _ = make_views()
        fitted = hub.fit_hub_kind(views, spans, "v", layer_rank=8, view_dim=12, hub_dim=6, chunk=1000,
                                  log=lambda line: None, fit_rows=400)
        self.assertTrue(np.all(fitted.eigenvalues <= 3 + 1e-6))
        self.assertGreater(hub.evaluate(fitted, views, spans, "v")["a", "c"]["r2"], 0.95)

    def test_unrelated_view_does_not_translate(self):
        views, spans, latent = make_views()
        other = {kind: np.random.default_rng(9).standard_normal(latent[kind].shape).astype(np.float32)
                 for kind in hub.KINDS}
        views.append(FakeView("noise", other, [10, 10], 4))
        spans.rows["noise"] = np.arange(3000)
        fitted = hub.fit_hub_kind(views, spans, "k", layer_rank=8, view_dim=12, hub_dim=6, chunk=1000,
                                  log=lambda line: None)
        results = hub.evaluate(fitted, views, spans, "k")
        self.assertLess(results["noise", "a"]["r2"], 0.1)
        self.assertGreater(results["b", "a"]["r2"], 0.95)

    def test_layer_subset_source_encoder(self):
        views, spans, _ = make_views()
        fitted = hub.fit_hub_kind(views, spans, "k", layer_rank=8, view_dim=12, hub_dim=6, chunk=1000,
                                  log=lambda line: None, source_layers={"b": [1, 3]})
        self.assertEqual(fitted.source_layers("b"), [1, 3])
        self.assertEqual(fitted.source_layers("a"), [0, 1, 2])
        results = hub.evaluate(fitted, views, spans, "k")
        self.assertGreater(results["b", "a"]["r2"], 0.95)
        self.assertGreater(results["b", "c"]["r2"], 0.95)

    def test_onboarding_a_model_without_refitting(self):
        views, spans, _ = make_views()
        base = hub.fit_hub_kind(views[:2], spans, "k", layer_rank=8, view_dim=12, hub_dim=6, chunk=1000,
                                log=lambda line: None)
        before = {name: base.decoders[name][0].copy() for name in base.names}
        hub.add_model_kind(base, views[2], views[:2], spans, "k", layer_rank=8, view_dim=12, chunk=1000,
                           log=lambda line: None)
        self.assertEqual(base.names, ["a", "b", "c"])
        for name, decoder in before.items():
            np.testing.assert_array_equal(base.decoders[name][0], decoder)   # existing adapters untouched
        results = hub.evaluate(base, views, spans, "k")
        for pair in (("c", "a"), ("a", "c"), ("c", "b"), ("b", "c")):
            self.assertGreater(results[pair]["r2"], 0.95, pair)
        new_only = hub.evaluate(base, views, spans, "k", involving="c")          # fit_hub.py --evaluate new
        self.assertEqual(set(new_only), {pair for pair in results if "c" in pair})
        self.assertEqual(new_only["a", "c"], results["a", "c"])
        chosen = hub.evaluate(base, views, spans, "k", pairs=[("c", "b"), ("b", "c")])   # campaign.py assemble --evaluate
        self.assertEqual(set(chosen), {("c", "b"), ("b", "c")})
        self.assertEqual(chosen["c", "b"], results["c", "b"])
        with self.assertRaisesRegex(ValueError, "already in the hub"):
            hub.add_model_kind(base, views[2], views[:2], spans, "k", log=lambda line: None)

    def test_onboarding_encoders_and_the_shrinkage_of_the_ridge(self):
        # four noisy base views (canonical correlations well below 1: eigenvalues lambda < M) and a fifth to onboard
        rng = np.random.default_rng(5)
        n = 6000
        latent = {kind: rng.standard_normal((n, 6)).astype(np.float32) for kind in hub.KINDS}
        views = [FakeView(name, latent, dims, seed, noise=3.0) for name, dims, seed in
                 (("a", [16, 16], 1), ("b", [8] * 4, 2), ("c", [24], 3), ("d", [12, 12], 4), ("e", [10, 10], 6))]
        documents = n // 100
        spans = hub.Spans([f"d{i}" for i in range(documents)], ["train"] * (documents - 10) + ["dev"] * 5 + ["test"] * 5,
                          np.repeat(np.arange(documents), 100), {v.name: np.arange(n) for v in views})
        quiet = lambda line: None  # noqa: E731
        fitted = {}
        for encoder in ("ridge", "consensus", "match", "cca"):
            kind = hub.fit_hub_kind(views[:4], spans, "k", layer_rank=8, view_dim=12, hub_dim=6, chunk=1000, log=quiet)
            hub.add_model_kind(kind, views[4], views[:4], spans, "k", layer_rank=8, view_dim=12, chunk=1000, log=quiet,
                               encoder=encoder, consensus=["a", "b", "c", "d"])
            self.assertEqual(kind.alphas["e"]["encoder"], encoder)
            fitted[encoder] = kind
        lam = fitted["ridge"].eigenvalues
        self.assertTrue(np.all(lam < 3.9) and np.all(lam > 1.0))
        rows = spans.select("test")
        blocks = {layer: views[4].rows("k", layer, rows) for layer in views[4].layers("k")}
        z = {encoder: kind.encode("e", blocks).astype(np.float64) for encoder, kind in fitted.items()}
        var = {encoder: values.var(axis=0) for encoder, values in z.items()}
        # the ridge regresses onto the mean of the pivots' estimates: in a MAXVAR hub the estimates of its M views
        # sum to lambda Z, so (every pivot giving its full view) the ridge is the consensus encoder times lambda / M
        ridge, consensus = fitted["ridge"].encoders["e"], fitted["consensus"].encoders["e"]
        self.assertLess(np.abs(ridge - consensus * (lam / 4)).max() / np.abs(ridge).max(), 1e-3)
        np.testing.assert_allclose(var["ridge"].sum() / var["consensus"].sum(), np.mean((lam / 4) ** 2), rtol=0.1)
        pivot = fitted["ridge"].encode("a", {layer: views[0].rows("k", layer, rows) for layer in views[0].layers("k")})
        self.assertGreater(var["match"].sum() / pivot.var(axis=0).sum(), 0.8)             # spread like a pivot
        self.assertLess(var["ridge"].sum() / pivot.var(axis=0).sum(), 0.6)
        self.assertGreater(var["cca"].sum(), var["consensus"].sum())                       # unit gain on every pair
        r2 = {encoder: hub.evaluate(kind, views, spans, "k", pairs=[("e", "a"), ("a", "e")])
              for encoder, kind in fitted.items()}
        for encoder in fitted:
            self.assertGreater(r2[encoder]["e", "a"]["r2"], 0.1, encoder)
            self.assertEqual(r2[encoder]["a", "e"], r2["ridge"]["a", "e"])               # the decoder is the same
        with self.assertRaisesRegex(ValueError, "every model the hub latent was fitted on"):
            hub.add_model_kind(fitted["ridge"], FakeView("f", latent, [8], 7), views[:3], spans, "k", log=quiet,
                               encoder="consensus", consensus=["a", "b", "c", "d"])
        with self.assertRaisesRegex(ValueError, "needs the models the hub latent was fitted on"):
            hub.add_model_kind(fitted["ridge"], FakeView("f", latent, [8], 7), views[:4], spans, "k", log=quiet,
                               encoder="cca")
        with self.assertRaisesRegex(ValueError, "unknown onboarding encoder"):
            hub.add_model_kind(fitted["ridge"], FakeView("f", latent, [8], 7), views[:4], spans, "k", log=quiet,
                               encoder="nope")

    def test_reencoding_changes_only_the_encoders(self):
        views, spans, _ = make_views()
        quiet = lambda line: None  # noqa: E731
        fits = {}
        for encoder in ("ridge", "match"):
            kind = hub.fit_hub_kind(views[:2], spans, "k", layer_rank=8, view_dim=12, hub_dim=6, chunk=1000, log=quiet)
            hub.add_model_kind(kind, views[2], views[:2], spans, "k", layer_rank=8, view_dim=12, chunk=700, log=quiet,
                               encoder=encoder, source_layers=[0])
            fits[encoder] = kind
        kind = fits["ridge"]
        decoders = {name: [d.copy() for d in kind.decoders[name]] for name in kind.names}
        transform, sub = kind.transforms["c"], kind.source_transforms["c"]
        logs = []
        hub.reencode_model_kind(kind, views[2], views[:2], spans, "k", encoder="match", chunk=700, log=logs.append)
        for name in kind.names:                                          # every decoder exactly as it was
            for before, after in zip(decoders[name], kind.decoders[name]):
                np.testing.assert_array_equal(before, after)
        self.assertIs(kind.transforms["c"], transform)
        self.assertIs(kind.source_transforms["c"], sub)
        # the same encoders as onboarding with that method (the same transforms and statistics)
        np.testing.assert_allclose(kind.encoders["c"], fits["match"].encoders["c"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(kind.source_encoders["c"], fits["match"].source_encoders["c"], rtol=1e-4, atol=1e-5)
        self.assertEqual((kind.alphas["c"]["encoder"], kind.alphas["c"]["reencoded_from"]), ("match", "ridge"))
        self.assertEqual(json.loads(logs[-1])["stage"], "reencode")
        with self.assertRaisesRegex(ValueError, "not in the hub"):
            stranger = FakeView("x", {k: np.zeros((3000, 6)) for k in hub.KINDS}, [4], 1)
            hub.reencode_model_kind(kind, stranger, views[:2], spans, "k",
                                    encoder="match")
        with self.assertRaisesRegex(ValueError, "not other models"):
            hub.reencode_model_kind(kind, views[2], views, spans, "k", encoder="match")

    def test_onboarding_encoder_algebra(self):
        rng = np.random.default_rng(3)
        n, p, d = 20000, 8, 5
        u = rng.standard_normal((n, p)) @ rng.standard_normal((p, p))
        target = u @ rng.standard_normal((p, d)) * 0.3 + rng.standard_normal((n, d)) * np.arange(1, d + 1)
        u, target = u - u.mean(axis=0), target - target.mean(axis=0)
        cov_u, cross, target_cov = u.T @ u / n, u.T @ target / n, target.T @ target / n
        pooled_var = np.linspace(0.5, 2.0, d)
        solve = lambda method: hub.solve_onboard_encoder(method, cov_u, cross, target_cov, pooled_var,  # noqa: E731
                                                         ridge=1e-9, variance_cap=0).astype(np.float64)
        ridge = solve("ridge")
        np.testing.assert_allclose(ridge, np.linalg.lstsq(u, target, rcond=None)[0], rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(solve("consensus"), ridge)          # the same solve; its target differs
        variance = lambda weight: (u @ weight).var(axis=0)  # noqa: E731
        np.testing.assert_allclose(variance(solve("match")), pooled_var, rtol=1e-4)
        np.testing.assert_allclose(variance(solve("tls")), np.diag(target_cov), rtol=1e-4)
        encoded = u @ ridge
        slope = np.diag(target_cov) / np.einsum("ij,ij->j", encoded, target) * n
        np.testing.assert_allclose(solve("reverse"), ridge * slope, rtol=1e-4)
        np.testing.assert_array_equal(solve("reverse_z"), solve("reverse"))   # the same solve onto the consensus
        # cca: every canonical pair at unit gain, so the encodings carry the target's covariance
        cca = u @ solve("cca")
        np.testing.assert_allclose(cca.T @ cca / n, target_cov, rtol=1e-3, atol=1e-3)
        with self.assertRaisesRegex(ValueError, "unknown onboarding encoder"):
            solve("nope")

    def test_the_reverse_variance_cap(self):
        """reverse scales dimension d by about 1 / R^2_d: a dimension the view barely predicts would be inflated
        ~200x; the cap holds its variance at variance_cap x the pooled pivot estimates' (sqrt(cap) x match's scale)
        and leaves the others exactly as they were; the diagnostics say so."""
        rng = np.random.default_rng(11)
        n, p, d = 40000, 8, 4
        u = rng.standard_normal((n, p))
        signal = u @ rng.standard_normal((p, d))
        target = signal * np.array([1.0, 1.0, 1.0, 0.07]) + rng.standard_normal((n, d))    # R^2 of d=3 ~ 0.03
        u, target = u - u.mean(axis=0), target - target.mean(axis=0)
        cov_u, cross, target_cov = u.T @ u / n, u.T @ target / n, target.T @ target / n
        pooled_var = np.diag(target_cov) * 1.4
        variance = lambda weight: (u @ weight.astype(np.float64)).var(axis=0)  # noqa: E731
        infos = {}

        def solve(method, cap):
            infos[method, cap] = {}
            return hub.solve_onboard_encoder(method, cov_u, cross, target_cov, pooled_var, ridge=1e-9,
                                             variance_cap=cap, info=infos[method, cap])
        free, capped = solve("reverse", 0), solve("reverse", 4.0)
        ratio = variance(free) / pooled_var
        self.assertGreater(ratio[3], 10)                                 # 1 / R^2 inflation of the weak dimension
        self.assertTrue(np.all(ratio[:3] < 4))
        np.testing.assert_array_equal(capped[:, :3], free[:, :3])        # the others: bit-identical
        np.testing.assert_allclose(variance(capped)[3], 4.0 * pooled_var[3], rtol=1e-4)
        match = solve("match", 4.0)
        np.testing.assert_allclose(capped[:, 3], 2.0 * match[:, 3], rtol=1e-5)   # sqrt(4) x match's scale
        info = infos["reverse", 4.0]
        self.assertEqual((info["capped"], info["cap"]), (1, 4.0))
        self.assertAlmostEqual(info["max_ratio"], 4.0, places=3)
        self.assertGreater(info["max_ratio_uncapped"], 10)
        self.assertLess(info["min_r2"], 0.05)
        self.assertEqual(infos["reverse", 0]["capped"], 0)
        warnings = hub.encoder_warnings(info)
        self.assertEqual(len(warnings), 2)
        self.assertIn("limited 1 dimension", warnings[0])
        self.assertIn("barely predicts", warnings[1])
        np.testing.assert_array_equal(solve("reverse_z", 4.0), capped)    # the same solve onto its target
        for method in ("ridge", "consensus", "match", "tls", "cca"):     # the cap is the reverse family's
            np.testing.assert_array_equal(solve(method, 4.0), solve(method, 0))
            self.assertIsNone(infos[method, 4.0]["cap"])
        self.assertEqual(hub.encoder_warnings({"min_r2": 0.5, "capped": 0}), [])

    def test_onboarding_through_every_encoder_and_reencoding_into_each(self):
        """reverse, reverse_z, tls (and the rest) through add_model_kind and reencode_model_kind, a pivot that
        encodes from a layer subset (as Muse does, --source-global) among the pivots: the same encoders both
        ways; reverse spreads at least as far as match; the consensus target reads every pivot's full view."""
        rng = np.random.default_rng(5)
        n = 6000
        latent = {kind: rng.standard_normal((n, 6)).astype(np.float32) for kind in hub.KINDS}
        views = [FakeView(name, latent, dims, seed, noise=3.0) for name, dims, seed in
                 (("a", [16, 16], 1), ("b", [8] * 4, 2), ("c", [24], 3), ("e", [10, 10], 6))]
        documents = n // 100
        spans = hub.Spans([f"d{i}" for i in range(documents)], ["train"] * (documents - 10) + ["dev"] * 5 + ["test"] * 5,
                          np.repeat(np.arange(documents), 100), {v.name: np.arange(n) for v in views})
        logs = []
        quiet = lambda line: logs.append(line)  # noqa: E731

        def base(subset):
            return hub.fit_hub_kind(views[:3], spans, "k", layer_rank=8, view_dim=12, hub_dim=6, chunk=1000, log=quiet,
                                    source_layers={"b": [1, 3]} if subset else None)
        onboarded = {}
        for encoder in hub.ONBOARD_ENCODERS:
            kind = base(subset=True)
            hub.add_model_kind(kind, views[3], views[:3], spans, "k", layer_rank=8, view_dim=12, chunk=1000, log=quiet,
                               encoder=encoder, consensus=["a", "b", "c"])
            stats = kind.alphas["e"]["encoder_stats"]["full"]
            self.assertEqual((stats["method"], kind.alphas["e"]["encoder"]), (encoder, encoder))
            onboarded[encoder] = kind
        self.assertTrue(any(json.loads(line)["stage"] == "onboard-encoder" for line in logs))
        reencoded = base(subset=True)               # onboarded with ridge, then fitted again with every other
        hub.add_model_kind(reencoded, views[3], views[:3], spans, "k", layer_rank=8, view_dim=12, chunk=1000,
                           log=quiet, encoder="ridge")
        for encoder in hub.ONBOARD_ENCODERS:
            if encoder == "ridge":
                continue
            hub.reencode_model_kind(reencoded, views[3], views[:3], spans, "k", encoder=encoder, chunk=700, log=quiet,
                                    consensus=["a", "b", "c"])
            np.testing.assert_allclose(reencoded.encoders["e"], onboarded[encoder].encoders["e"], rtol=2e-3, atol=2e-4,
                                       err_msg=encoder)
            self.assertEqual(reencoded.alphas["e"]["encoder_stats"]["full"]["method"], encoder)
        rows = spans.select("test")
        blocks = {layer: views[3].rows("k", layer, rows) for layer in views[3].layers("k")}
        spread = {encoder: kind.encode("e", blocks).astype(np.float64).var(axis=0).sum()
                  for encoder, kind in onboarded.items()}
        self.assertGreater(spread["reverse"], spread["match"])
        self.assertGreater(spread["match"], spread["ridge"])
        self.assertGreater(spread["reverse_z"], spread["consensus"])
        # the consensus target is the latent the base was fitted as: the full views of every pivot, so a pivot
        # that also encodes from a layer subset changes nothing (the mean-of-estimates target does change)
        plain = base(subset=False)
        hub.add_model_kind(plain, views[3], views[:3], spans, "k", layer_rank=8, view_dim=12, chunk=1000, log=quiet,
                           encoder="consensus", consensus=["a", "b", "c"])
        np.testing.assert_allclose(plain.encoders["e"], onboarded["consensus"].encoders["e"], rtol=1e-4, atol=1e-5)
        ridge_plain = base(subset=False)
        hub.add_model_kind(ridge_plain, views[3], views[:3], spans, "k", layer_rank=8, view_dim=12, chunk=1000,
                           log=quiet, encoder="ridge")
        self.assertGreater(np.abs(ridge_plain.encoders["e"] - onboarded["ridge"].encoders["e"]).max(), 1e-3)

    def test_consensus_views_of_a_bundle(self):
        manifests = {"/base": {"models": {"a": {}, "b": {}}, "settings": {"base": None}},
                     "/onboarded": {"models": {"a": {}, "b": {}, "c": {}}, "settings": {"base": "/base"}},
                     "/assembled": {"models": {"a": {}, "b": {}, "c": {}, "e": {}}, "settings": {"base": None},
                                    "assembled_base": "/base"},
                     "/recorded": {"models": {"a": {}, "b": {}, "c": {}}, "settings": {"consensus_views": ["a", "b"]}}}
        read = lambda path: manifests[str(path)]  # noqa: E731
        with mock.patch.object(Path, "is_file", lambda self: str(self.parent) in manifests):
            self.assertEqual(hub.consensus_views(manifests["/base"], read=read), ["a", "b"])
            self.assertEqual(hub.consensus_views(manifests["/onboarded"], read=read), ["a", "b"])
            self.assertEqual(hub.consensus_views(manifests["/assembled"], read=read), ["a", "b"])
        self.assertEqual(hub.consensus_views(manifests["/recorded"]), ["a", "b"])
        self.assertIsNone(hub.consensus_views({"models": {"c": {}}, "settings": {"base": "/missing/base"}}))

    def test_save_and_load_round_trip(self):
        views, spans, _ = make_views(n=1200)
        hubs = {kind: hub.fit_hub_kind(views, spans, kind, layer_rank=8, view_dim=12, hub_dim=6,
                                       log=lambda line: None, source_layers={"a": [0, 2]}) for kind in hub.KINDS}
        with tempfile.TemporaryDirectory() as directory:
            hub.save_hub(Path(directory) / "hub", hubs, views, spans, {"note": "test"})
            loaded, manifest = hub.load_hub(Path(directory) / "hub")
            self.assertEqual(manifest["spans"]["test"], 300)
            blocks = {layer: views[0].rows("v", layer, np.arange(50)) for layer in views[0].layers("v")}
            for index in range(len(views[1].layers("v"))):
                np.testing.assert_allclose(loaded["v"].decode("b", loaded["v"].encode("a", blocks), index),
                                           hubs["v"].decode("b", hubs["v"].encode("a", blocks), index), rtol=1e-5,
                                           atol=1e-5)
            entry = manifest["files"]["a.k"]["file"]
            data = bytearray((Path(directory) / "hub" / entry).read_bytes())
            data[-1] ^= 1
            (Path(directory) / "hub" / entry).write_bytes(bytes(data))
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                hub.load_hub(Path(directory) / "hub")


class MixedFamiliesTest(unittest.TestCase):
    """GQA-like views and MLA views (K = [c_kv | k_pe], V = c_kv) in one hub; an MLA model onboarded
    onto a fitted hub; a DSA model's indexer ridge, persisted in format 1 and 2 bundles."""

    @classmethod
    def setUpClass(cls):
        # one hidden state behind both kinds, as in a model (an MLA row's c_kv is K and V at once)
        shared = np.random.default_rng(0).standard_normal((3000, 6)).astype(np.float32)
        cls.latent = {kind: shared for kind in hub.KINDS}
        cls.views = [FakeView("a", cls.latent, [16, 16, 16], 1), FakeView("b", cls.latent, [8] * 5, 2),
                     FakeView("c", cls.latent, [24, 12], 3)]
        _, cls.spans, _ = make_views()
        splits = np.repeat(np.asarray(cls.spans.splits), 100)
        cls.mla = FakeMlaView("ds2", cls.latent, 2, 5, splits=splits)
        cls.dsa = FakeMlaView("dsa", cls.latent, 2, 6, indexer=16, splits=splits)
        for view in (cls.mla, cls.dsa):
            cls.spans.rows[view.name] = np.arange(3000)

    def test_mla_views_translate_both_ways(self):
        views = self.views[:2] + [self.mla]
        for kind in hub.KINDS:
            fitted = hub.fit_hub_kind(views, self.spans, kind, layer_rank=8, view_dim=12, hub_dim=6, chunk=1000,
                                      log=lambda line: None)
            results = hub.evaluate(fitted, views, self.spans, kind)
            for pair in (("a", "ds2"), ("ds2", "a"), ("ds2", "b")):
                self.assertGreater(results[pair]["r2"], 0.95, (kind, pair))
            # an MLA decoder's rows are the full K row (k) or c_kv (v)
            self.assertEqual([d.shape[1] for d in fitted.decoders["ds2"]], [16, 16] if kind == "k" else [12, 12])

    def test_onboarding_an_mla_model(self):
        base = {kind: hub.fit_hub_kind(self.views, self.spans, kind, layer_rank=8, view_dim=12, hub_dim=6,
                                       chunk=1000, log=lambda line: None) for kind in hub.KINDS}
        for kind in hub.KINDS:
            hub.add_model_kind(base[kind], self.mla, self.views[:2], self.spans, kind, layer_rank=8, view_dim=12,
                               chunk=1000, log=lambda line: None)
            results = hub.evaluate(base[kind], self.views + [self.mla], self.spans, kind)
            for pair in (("ds2", "c"), ("c", "ds2"), ("a", "ds2")):
                self.assertGreater(results[pair]["r2"], 0.95, (kind, pair))

    def test_indexer_ridge_fit_and_bundles(self):
        logs = []
        ridge = hub.fit_indexer_ridge(self.dsa, log=logs.append)
        self.assertEqual((ridge.name, ridge.layers), ("dsa", [0, 1]))
        test = self.dsa.split_rows("test")
        for layer in ridge.layers:
            y = self.dsa.rows("ik", layer, test)
            predicted = ridge.predict(layer, self.dsa.rows("k", layer, test))
            r2 = 1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean(axis=0)) ** 2)
            self.assertGreater(r2, 0.99)
            self.assertEqual(predicted.dtype, np.float32)
        self.assertTrue(all(json.loads(line)["stage"] == "indexer_ridge" for line in logs))
        with self.assertRaisesRegex(ValueError, "no indexer keys"):
            hub.fit_indexer_ridge(self.mla)
        views = self.views[:1] + [self.dsa]
        hubs = {kind: hub.fit_hub_kind(views, self.spans, kind, layer_rank=8, view_dim=12, hub_dim=6,
                                       log=lambda line: None) for kind in hub.KINDS}
        self.assertIs(hub.indexer_ridge({**hubs, "indexer": {"dsa": ridge}}, "dsa"), ridge)
        self.assertIsNone(hub.indexer_ridge(hubs, "dsa"))
        import convert_hub
        with tempfile.TemporaryDirectory() as directory:
            v1 = Path(directory) / "v1"
            hub.save_hub(v1, hubs, views, self.spans, {"note": "test"}, indexer_ridges={"dsa": ridge})
            convert_hub.convert(v1, Path(directory) / "v2", log=lambda line: None)
            for bundle in ("v1", "v2"):
                opened = hub.open_hub(Path(directory) / bundle)
                self.assertIsNone(opened.indexer_ridge("a"))
                loaded = opened.indexer_ridge("dsa")
                self.assertIs(opened.indexer_ridge("dsa"), loaded)                   # read once
                self.assertEqual(loaded.layers, ridge.layers)
                rows = self.dsa.rows("k", 1, np.arange(20))
                np.testing.assert_array_equal(loaded.predict(1, rows), ridge.predict(1, rows))
                self.assertEqual(set(hub.load_indexer_ridges(Path(directory) / bundle)), {"dsa"})
            # what fit_hub.py --base does: the base bundle's ridges carried into the new bundle
            hub.save_hub(Path(directory) / "again", hubs, views, self.spans, {},
                         indexer_ridges=hub.load_indexer_ridges(v1))
            again = hub.open_hub(Path(directory) / "again").indexer_ridge("dsa")
            np.testing.assert_array_equal(again.weights[0], ridge.weights[0])
            self.assertEqual(again.fit[0]["alpha"], ridge.fit[0]["alpha"])
            with self.assertRaisesRegex(ValueError, "not in the hub"):
                hub.save_hub(Path(directory) / "bad", hubs, views, self.spans, {}, indexer_ridges={"zz": ridge})


class SidesAndBundlesTest(unittest.TestCase):
    """Fused encoders, translation sides, format 2 bundles (convert_hub.py) and lazy, stamp-verified reads."""

    @classmethod
    def setUpClass(cls):
        cls.views, cls.spans, _ = make_views(n=1200)
        cls.hubs = {kind: hub.fit_hub_kind(cls.views, cls.spans, kind, layer_rank=8, view_dim=12, hub_dim=6,
                                           log=lambda line: None, source_layers={"a": [0, 2]}) for kind in hub.KINDS}
        cls.blocks = {name: {kind: {layer: view.rows(kind, layer, np.arange(40)) for layer in view.layers(kind)}
                             for kind in hub.KINDS} for name, view in ((v.name, v) for v in cls.views)}

    def unfused(self, hub_kind, name, blocks):
        t = hub_kind.source_transforms.get(name) or hub_kind.transforms[name]
        encoder = hub_kind.source_encoders.get(name, hub_kind.encoders[name])
        return t([blocks[layer] for layer in t.layers]) @ encoder

    def test_fused_encoder_is_the_same_algebra(self):
        for kind in hub.KINDS:
            for name in ("a", "b"):            # a encodes from a layer subset, b from every layer
                z = self.hubs[kind].encode(name, self.blocks[name][kind])
                expected = self.unfused(self.hubs[kind], name, self.blocks[name][kind])
                np.testing.assert_allclose(z, expected, rtol=1e-5, atol=1e-5 * np.abs(expected).max())
                weight = self.hubs[kind].encoder_weight(name)
                self.assertIs(self.hubs[kind].encoder_weight(name), weight)          # computed once
                self.assertTrue(weight.flags["C_CONTIGUOUS"])

    def test_fused_encoder_follows_replaced_arrays(self):
        import dataclasses
        hub_kind = dataclasses.replace(self.hubs["k"], encoders=dict(self.hubs["k"].encoders), _fused={})
        before = hub_kind.encoder_weight("b")
        hub_kind.encoders["b"] = hub_kind.encoders["b"] * 2        # a new array: the old one may be freed
        np.testing.assert_allclose(hub_kind.encoder_weight("b"), before * 2, rtol=1e-6)
        self.assertIs(hub_kind.encoder_weight("b"), hub_kind.encoder_weight("b"))

    def test_sides_encode_and_decode_like_the_hub_kind(self):
        for kind in hub.KINDS:
            encoder = self.hubs[kind].encoder_side("a")
            self.assertEqual(encoder.layers, [0, 2])
            z = encoder.encode(self.blocks["a"][kind])
            np.testing.assert_array_equal(z, self.hubs[kind].encode("a", self.blocks["a"][kind]))
            decoder = self.hubs[kind].decoder_side("c")
            layers = decoder.layers[::-1]                                          # any order, one product
            got = decoder.decode(z, layers)
            expected = np.concatenate([self.hubs[kind].decode("c", z, decoder.layers.index(l)) for l in layers], axis=1)
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    def save(self, directory):
        hub.save_hub(Path(directory) / "v1", self.hubs, self.views, self.spans, {"note": "test"})
        return Path(directory) / "v1"

    def test_format_2_bundle_holds_the_same_sides(self):
        import convert_hub
        with tempfile.TemporaryDirectory() as directory:
            v1 = self.save(directory)
            manifest = convert_hub.convert(v1, Path(directory) / "v2", log=lambda line: None)
            self.assertEqual(manifest["format_version"], hub.FORMAT_V2)
            self.assertEqual(set(manifest["models"]), {"a", "b", "c"})
            lazy1, lazy2 = hub.open_hub(v1), hub.open_hub(Path(directory) / "v2")
            for kind in hub.KINDS:
                for name in ("a", "b", "c"):
                    self.assertEqual(lazy2.source_layers(kind, name), lazy1.source_layers(kind, name))
                    e1, e2 = lazy1.encoder_side(kind, name), lazy2.encoder_side(kind, name)
                    np.testing.assert_array_equal(e2.weight, e1.weight)
                    np.testing.assert_array_equal(e2.weight, self.hubs[kind].encoder_weight(name))
                    for b1, b2 in zip(e1.bases, e2.bases):
                        np.testing.assert_array_equal(b1, b2)
                        self.assertTrue(b2.flags["C_CONTIGUOUS"])
                    d1, d2 = lazy1.decoder_side(kind, name), lazy2.decoder_side(kind, name)
                    np.testing.assert_array_equal(d2.decode(e2.encode(self.blocks[name][kind]), d2.layers),
                                                  d1.decode(e1.encode(self.blocks[name][kind]), d1.layers))
                    entry = manifest["sides"][f"{name}.{kind}.encoder"]
                    self.assertEqual(set(entry["files"]), {"f32", "f16"})
                    _, half = lazy2.side_flat(kind, name, "encoder", dtype="f16")
                    _, full = lazy2.side_flat(kind, name, "encoder")
                    np.testing.assert_array_equal(half, full.astype(np.float16))
            # load_hub rebuilds equivalent HubKinds from format 2
            loaded, _ = hub.load_hub(Path(directory) / "v2")
            for kind in hub.KINDS:
                self.assertEqual(loaded[kind].source_layers("a"), [0, 2])
                for name in ("a", "b"):
                    np.testing.assert_allclose(self.unfused(loaded[kind], name, self.blocks[name][kind]),
                                               self.unfused(self.hubs[kind], name, self.blocks[name][kind]),
                                               rtol=1e-6, atol=1e-6)
                    np.testing.assert_array_equal(loaded[kind].encode(name, self.blocks[name][kind]),
                                                  self.hubs[kind].encode(name, self.blocks[name][kind]))
            with self.assertRaisesRegex(ValueError, "not a format 1"):
                convert_hub.convert(Path(directory) / "v2", Path(directory) / "v3", log=lambda line: None)

    def test_files_are_hashed_once_per_identity(self):
        import convert_hub
        with tempfile.TemporaryDirectory() as directory:
            v2 = Path(directory) / "v2"
            convert_hub.convert(self.save(directory), v2, log=lambda line: None)
            stamps = Path(directory) / "cache" / "hub-verified.json"
            first = hub.open_hub(v2, cache_dir=stamps.parent)
            first.encoder_side("k", "b")
            self.assertEqual(first.verifier.hashed, 1)
            first.encoder_side("k", "b")
            self.assertEqual(first.verifier.hashed, 1)                 # in-process stamp
            self.assertEqual(stamps.stat().st_mode & 0o777, 0o600)
            second = hub.open_hub(v2, cache_dir=stamps.parent)
            second.encoder_side("k", "b")
            self.assertEqual(second.verifier.hashed, 0)                # stamp shared through the cache file
            # a rewritten file has a new identity and is hashed again
            path = v2 / json.loads((v2 / "manifest.json").read_text())["sides"]["b.k.encoder"]["files"]["f32"]["file"]
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(bytes(data))
            third = hub.open_hub(v2, cache_dir=stamps.parent)
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                third.encoder_side("k", "b")

    def test_onboarded_bundles_link_the_base_and_assemble_into_one_hub(self):
        """fit_hub.py --base --link-base, convert_hub.py --reuse and campaign.py assemble on a small
        hub: two models onboarded separately against the same base end up in one bundle of hard
        links, in both formats, translating as their own bundles do."""
        import convert_hub
        views, spans, latent = make_views(n=1200)
        # two more models of the same latent, onboarded separately with a and b as pivots
        extra = {"d": FakeView("d", latent, [10, 10], 4), "e": FakeView("e", latent, [6, 6, 6], 5)}
        spans.rows.update({name: np.arange(1200) for name in extra})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_hubs = {kind: hub.fit_hub_kind(views[:2], spans, kind, layer_rank=8, view_dim=12, hub_dim=6,
                                                log=lambda line: None) for kind in hub.KINDS}
            hub.save_hub(root / "base", base_hubs, views[:2], spans, {"note": "base"})
            convert_hub.convert(root / "base", root / "base-v2", log=lambda line: None)
            base_manifest = hub.read_manifest(root / "base")
            for name in extra:
                hubs, manifest = hub.load_hub(root / "base")
                for kind in hub.KINDS:
                    hub.add_model_kind(hubs[kind], extra[name], views[:2], spans, kind, layer_rank=8, view_dim=12,
                                       log=lambda line: None)
                hub.save_hub(root / name, hubs, [extra[name]] + views[:2], spans, {"note": name},
                             base_models=manifest["models"], link_base=root / "base")
                logs = []
                convert_hub.convert(root / name, root / f"{name}-v2", log=logs.append, reuse=[root / "base-v2"])
                self.assertEqual([json.loads(line).get("linked") for line in logs if "linked" in line], ["a", "b"])
                for kind in hub.KINDS:
                    for model in ("a", "b"):                         # the base's files, linked, read-only
                        entry = base_manifest["files"][f"{model}.{kind}"]
                        self.assertEqual((root / name / entry["file"]).stat().st_ino,
                                         (root / "base" / entry["file"]).stat().st_ino)
                        self.assertEqual((root / "base" / entry["file"]).stat().st_mode & 0o777, 0o444)
                    self.assertEqual((root / name / f"{name}.{kind}.npz").stat().st_nlink, 1)
            # a base model whose arrays changed in memory is never linked
            hubs, manifest = hub.load_hub(root / "base")
            hubs["v"].encoders["a"] = hubs["v"].encoders["a"] + 1e-3
            with self.assertRaisesRegex(ValueError, "a base model changed"):
                hub.save_hub(root / "changed", hubs, views[:2], spans, {}, base_models=manifest["models"],
                             link_base=root / "base")
            # links never cross a filesystem (they would be copies)
            with mock.patch("os.link", side_effect=OSError(errno.EXDEV, "cross-device link")):
                with self.assertRaisesRegex(ValueError, "different filesystems"):
                    hub.link_file(root / "base" / "manifest.json", root / "copied.json")
            self.assertFalse((root / "copied.json").exists())
            with mock.patch.object(hub, "_device", side_effect=lambda path: 1 if "elsewhere" in str(path) else 0):
                with self.assertRaisesRegex(ValueError, "different filesystems"):
                    hub.merge_bundles(root / "elsewhere", [root / "base", root / "d"])
            self.assertFalse((root / "elsewhere").exists())
            with self.assertRaisesRegex(ValueError, "not a format 1"):
                hub.save_hub(root / "bad", base_hubs, views[:2], spans, {}, link_base=root / "base-v2")
            merged = hub.merge_bundles(root / "hub5", [root / "base", root / "d", root / "e"])
            merged_v2 = hub.merge_bundles(root / "hub5-v2", [root / "base-v2", root / "d-v2", root / "e-v2"])
            self.assertEqual(sorted(merged["models"]), ["a", "b", "d", "e"])
            self.assertEqual(sorted(merged_v2["models"]), ["a", "b", "d", "e"])
            self.assertEqual(sorted(merged["assembled"]), sorted(str((root / n).resolve()) for n in ("d", "e")))
            self.assertEqual(set(merged["k_alphas"]), {"a", "b", "d", "e"})
            for directory_name in ("hub5", "hub5-v2"):
                for path in (root / directory_name).iterdir():
                    if path.name != "manifest.json":
                        self.assertGreater(path.stat().st_nlink, 1, path.name)
            # an assembled hub translates d -> e as e's own decoder after d's own encoder
            lazy, lazy_v2 = hub.open_hub(root / "hub5"), hub.open_hub(root / "hub5-v2")
            own_d, own_e = hub.open_hub(root / "d"), hub.open_hub(root / "e")
            for kind in hub.KINDS:
                blocks = {layer: extra["d"].rows(kind, layer, np.arange(30)) for layer in extra["d"].layers(kind)}
                expected = own_e.decoder_side(kind, "e").decode(own_d.encoder_side(kind, "d").encode(blocks), [0, 1, 2])
                for assembled in (lazy, lazy_v2):
                    got = assembled.decoder_side(kind, "e").decode(assembled.encoder_side(kind, "d").encode(blocks),
                                                                   [0, 1, 2])
                    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
            loaded, _ = hub.load_hub(root / "hub5-v2")
            self.assertEqual(loaded["k"].names, ["a", "b", "d", "e"])
            # members that were not grown from the base, or that add the same model twice, are refused
            with self.assertRaisesRegex(ValueError, "added twice"):
                hub.merge_bundles(root / "twice", [root / "base", root / "d", root / "d"])
            with self.assertRaisesRegex(ValueError, "different formats"):
                hub.merge_bundles(root / "mixed", [root / "base", root / "d-v2"])
            other = {kind: hub.fit_hub_kind(views[:2], spans, kind, layer_rank=8, view_dim=12, hub_dim=5,
                                            log=lambda line: None) for kind in hub.KINDS}
            hub.save_hub(root / "other", other, views[:2], spans, {})
            with self.assertRaisesRegex(ValueError, "not grown from"):
                hub.merge_bundles(root / "foreign", [root / "base", root / "other"])
            self.assertFalse((root / "twice").exists() or (root / "foreign").exists())

    def test_lazy_format_1_reads_only_the_sides_asked_for(self):
        with tempfile.TemporaryDirectory() as directory:
            lazy = hub.open_hub(self.save(directory))
            self.assertEqual(lazy.source_layers("v", "a"), [0, 2])
            self.assertEqual(lazy.source_layers("v", "b"), [0, 1, 2, 3, 4])
            encoder = lazy.encoder_side("v", "a")
            np.testing.assert_array_equal(encoder.weight, self.hubs["v"].encoder_weight("a"))
            self.assertEqual(lazy.verifier.hashed, 2)                  # a.v.npz and b.v.npz, once each
            lazy.decoder_side("v", "a")
            self.assertEqual(lazy.verifier.hashed, 2)
            with self.assertRaisesRegex(ValueError, "no adapter"):
                lazy.decoder_side("v", "nobody")


if __name__ == "__main__":
    unittest.main()
