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
