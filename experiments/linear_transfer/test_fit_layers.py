"""Layer screening, top-k ridge maps and the mapper format on synthetic captures."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest

import numpy as np

from align import ALIGNED, TEMPLATE, Alignment
from captures import CaptureSet, write_synthetic_capture
import fit_layers
import ridge


SRC_LAYERS = (3, 7, 11, 15)
TGT_LAYERS = (0, 1, 2, 3)
SRC_SHAPE = (2, 4)
TGT_SHAPE = (2, 3)
# Target layer -> the source layers its K and V depend on.
TRUTH = {0: (7,), 1: (3, 11), 2: (3, 7, 11, 15), 3: (11,)}
RENORM_TARGET = 3


def doc_id(number):
    return hashlib.sha256(f"fit-{number}".encode()).hexdigest()


def heads_rms(values):
    return np.sqrt(np.mean(np.square(values.reshape(len(values), *TGT_SHAPE)), axis=-1))


class Dataset:
    """Source/target captures whose target K/V are known linear maps of chosen source layers."""

    def __init__(self, root, seed=0, doc_rows=40, splits=("train",) * 16 + ("dev",) * 5 + ("test",) * 5,
                 noise=0.5, offset=2.0, split_target=False, constant=None):
        """split_target: write the target as a {k, v} train capture and a {k, v, q} capture with
        the remaining documents. constant: (kind, target, split) whose target rows are constant."""
        Path(root).mkdir(parents=True)
        rng = np.random.default_rng(seed)
        self.splits = list(splits)
        sizes = [doc_rows + (number % 3) for number in range(len(splits))]
        rows = sum(sizes)
        row_split = np.repeat(np.asarray(self.splits), sizes)
        width, outputs = int(np.prod(SRC_SHAPE)), int(np.prod(TGT_SHAPE))
        self.source, self.target, self.truth = {}, {}, {}
        for kind in fit_layers.KINDS:
            latent = rng.standard_normal((rows, width))
            for layer in SRC_LAYERS:
                mixing = 0.3 * rng.standard_normal((width, width))
                values = latent @ mixing + rng.standard_normal((rows, width)) + offset
                self.source[kind, layer] = values.astype(np.float32)
            for target in TGT_LAYERS:
                layers = TRUTH[target]
                total = rng.standard_normal(outputs) * 0.5
                for layer in layers:
                    weight = rng.standard_normal((width, outputs)) / np.sqrt(width * len(layers))
                    self.truth[kind, target, layer] = weight
                    total = total + (self.source[kind, layer] - offset) @ weight
                if kind == "k" and target == RENORM_TARGET:
                    values = fit_layers.renormalize_heads(total + 0.05 * rng.standard_normal(total.shape),
                                                          *TGT_SHAPE)
                else:
                    values = total + noise * rng.standard_normal(total.shape)
                if constant is not None and constant[:2] == (kind, target):
                    values[row_split == constant[2]] = 1.5
                self.target[kind, target] = values.astype(np.float32)
        src_docs, tgt_docs, src_arrays, tgt_arrays = [], [], {}, {}
        src_row, tgt_row, doc = [], [], []
        tgt_class, first, tgt_first = [], 0, 0
        tgt_index = []
        for number, size in enumerate(sizes):
            src_docs.append({"document_id": doc_id(number), "byte_end": list(range(1, size + 1))})
            # The target has an extra leading template token per document (never aligned).
            tgt_docs.append({"document_id": doc_id(number), "byte_end": list(range(1, size + 2)),
                             "special": [1] + [0] * size})
            src_row.extend(range(first, first + size))
            tgt_row.extend(range(tgt_first + 1, tgt_first + size + 1))
            doc.extend([number] * size)
            tgt_class.extend([TEMPLATE] + [ALIGNED] * size)
            tgt_index.extend([None] + list(range(first, first + size)))
            first += size
            tgt_first += size + 1
        for key, values in self.source.items():
            src_arrays[key] = values.reshape(rows, *SRC_SHAPE)
        for key, values in self.target.items():
            padded = np.zeros((len(tgt_index), outputs), dtype=np.float32)
            chosen = [i for i, row in enumerate(tgt_index) if row is not None]
            padded[chosen] = values
            padded[[i for i, row in enumerate(tgt_index) if row is None]] = 7.0
            tgt_arrays[key] = padded.reshape(len(tgt_index), *TGT_SHAPE)
        self.src_path = write_synthetic_capture(Path(root) / "src", src_docs, src_arrays,
                                                architecture="qwen35", n_layer=16)
        if split_target:
            documents = self.splits.index("dev")
            if not set(self.splits[documents:]) <= {"dev", "test"}:
                raise ValueError("split_target needs the train documents first")
            boundary = sum(size + 1 for size in sizes[:documents])
            eval_arrays = {key: values[boundary:] for key, values in tgt_arrays.items()}
            eval_arrays["q", 0] = np.ones((len(tgt_index) - boundary, 4, 3), dtype=np.float32)
            eval_arrays["q", 3] = np.ones((len(tgt_index) - boundary, 4, 3), dtype=np.float32)
            self.tgt_paths = [
                write_synthetic_capture(Path(root) / "tgt-train", tgt_docs[:documents],
                                        {key: values[:boundary] for key, values in tgt_arrays.items()},
                                        architecture="muse-glimmer", n_layer=4),
                write_synthetic_capture(Path(root) / "tgt-eval", tgt_docs[documents:], eval_arrays,
                                        architecture="muse-glimmer", n_layer=4)]
        else:
            self.tgt_paths = [write_synthetic_capture(Path(root) / "tgt", tgt_docs, tgt_arrays,
                                                      architecture="muse-glimmer", n_layer=4)]
        self.tgt_path = self.tgt_paths[0]
        count = len(src_row)
        zeros = np.zeros(count, dtype=np.int64)
        self.alignment = Alignment(
            document_ids=[doc_id(n) for n in range(len(sizes))], splits=self.splits,
            src_row=np.asarray(src_row, dtype=np.int64), tgt_row=np.asarray(tgt_row, dtype=np.int64),
            doc=np.asarray(doc, dtype=np.int64), segment=zeros, rel_end=zeros + 1,
            src_position=zeros, tgt_position=zeros,
            src_class=np.full(rows, ALIGNED, dtype=np.int8), tgt_class=np.asarray(tgt_class, dtype=np.int8))
        self.doc = np.asarray(doc)

    def problem(self):
        if len(self.tgt_paths) > 1:
            target = fit_layers.open_capture_set(self.tgt_paths)
        else:
            target = CaptureSet(self.tgt_paths)
        return fit_layers.Problem(CaptureSet([self.src_path]), target, self.alignment)

    def rows(self, split):
        chosen = np.asarray([self.splits[d] == split for d in self.doc])
        return np.flatnonzero(chosen)


class LinearAlgebraTests(unittest.TestCase):
    def test_blocked_cholesky_solve_matches_dense_solvers(self):
        rng = np.random.default_rng(1)
        for size, block in ((37, 8), (16, 16), (5, 8), (64, 16)):
            x = rng.standard_normal((size * 2, size))
            gram = x.T @ x
            rhs = rng.standard_normal((size, 7))
            lam = 0.3
            system = gram.copy()
            system.flat[::size + 1] += lam
            system[np.triu_indices(size, 1)] = np.nan
            inverses = fit_layers.cholesky_in_place(system, block)
            solved = fit_layers.cholesky_solve_in_place(system, inverses, rhs.copy(), block)
            np.testing.assert_allclose(solved, np.linalg.solve(gram + lam * np.eye(size), rhs), rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(solved, ridge.solve_cholesky(gram, rhs, lam), rtol=1e-9, atol=1e-10)
            lower = np.tril(system)
            np.testing.assert_allclose(lower @ lower.T, gram + lam * np.eye(size), rtol=1e-10, atol=1e-9)

    def test_non_positive_definite_system_is_rejected(self):
        system = np.array([[1.0, 2.0], [2.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "not positive definite"):
            fit_layers.cholesky_in_place(system, 1)

    def test_head_renormalization(self):
        values = np.random.default_rng(2).standard_normal((9, 6)) * 5 + 1
        normalized = fit_layers.renormalize_heads(values, *TGT_SHAPE)
        np.testing.assert_allclose(heads_rms(normalized), 1, rtol=1e-6)
        np.testing.assert_allclose(fit_layers.renormalize_heads(normalized, *TGT_SHAPE), normalized, rtol=1e-6)
        cosine = np.sum(values.reshape(9, 2, 3) * normalized.reshape(9, 2, 3), axis=-1)
        self.assertTrue(np.all(cosine > 0))


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data = Dataset(cls.root / "data")
        cls.problem = cls.data.problem()
        cls.pending = cls.root / "pending"
        cls.pending.mkdir()
        work = cls.root / "work"
        work.mkdir()
        cls.maps, cls.report = fit_layers.fit(cls.problem, alphas=fit_layers.DEFAULT_ALPHAS, ks=(1, 2, 4),
                                              chunk_rows=97, work_dir=work, pending=cls.pending)
        cls.by_key = {(m["kind"], m["target"]): m for m in cls.maps}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_streaming_statistics_equal_direct_computation(self):
        rows = self.data.rows("train")
        for kind in fit_layers.KINDS:
            x = np.concatenate([self.data.source[kind, layer][rows] for layer in SRC_LAYERS], axis=1)
            y = np.concatenate([self.data.target[kind, target][rows] for target in TGT_LAYERS], axis=1)
            gram, cross, x_mean, y_mean = ridge.centered_gram(x, y)
            for chunk in (1, 7, len(rows), len(rows) + 5):
                stats = fit_layers.accumulate(self.problem, kind, chunk, block=5)
                np.testing.assert_allclose(stats.gram, gram, rtol=1e-10, atol=1e-8)
                np.testing.assert_array_equal(stats.gram, stats.gram.T)
                np.testing.assert_allclose(stats.cross, cross, rtol=1e-10, atol=1e-8)
                np.testing.assert_allclose(stats.x_mean, x_mean, rtol=1e-12)
                np.testing.assert_allclose(stats.y_mean, y_mean, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(stats.y_centered_squares,
                                           np.sum(np.square(y - y.mean(axis=0, dtype=np.float64)), axis=0),
                                           rtol=1e-10)
                self.assertEqual(stats.rows, len(rows))

    def test_screening_matches_direct_single_layer_ridge(self):
        rows, dev = self.data.rows("train"), self.data.rows("dev")
        entries = {(e["kind"], e["target"], e["source"]): e for e in self.report["screening"]}
        self.assertEqual(len(entries), 2 * len(TGT_LAYERS) * len(SRC_LAYERS))
        for kind, target, source, alpha_index in (("v", 1, 3, 2), ("k", 0, 15, 6), ("k", 3, 11, 0)):
            x = self.data.source[kind, source]
            y = self.data.target[kind, target]
            gram = ridge.centered_gram(x[rows], y[rows])[0]
            lam = fit_layers.DEFAULT_ALPHAS[alpha_index] * np.mean(np.diag(gram))
            weight, bias = ridge.fit(x[rows], y[rows], lam)
            predicted = x[dev] @ weight + bias
            actual = y[dev].astype(np.float64)
            expected = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean(axis=0)) ** 2)
            entry = entries[kind, target, source]
            self.assertAlmostEqual(entry["lambda"][alpha_index], lam, delta=1e-9 * lam)
            self.assertAlmostEqual(entry["dev_r2"][alpha_index], expected, delta=1e-5)
            self.assertEqual(entry["best_dev_r2"], max(entry["dev_r2"]))

    def test_ranking_puts_true_layers_first_and_is_shared_by_kinds(self):
        ranking = {item["target"]: item for item in self.report["ranking"]}
        for target, layers in TRUTH.items():
            order = ranking[target]["order"]
            self.assertEqual(sorted(order), list(SRC_LAYERS))
            self.assertEqual(set(order[:len(layers)]), set(layers), (target, order))
            self.assertEqual(ranking[target]["score"], sorted(ranking[target]["score"], reverse=True))
            for kind in fit_layers.KINDS:
                self.assertEqual(self.by_key[kind, target]["source_layers"], order[:self.by_key[kind, target]["k"]])

    def test_dev_selection_prefers_the_true_number_of_layers(self):
        for (kind, target), record in self.by_key.items():
            self.assertEqual(record["k"], len(TRUTH[target]), (kind, target))
            rows = [e for e in self.report["top_k"] if e["kind"] == kind and e["target"] == target]
            self.assertEqual(len(rows), 3 * len(fit_layers.DEFAULT_ALPHAS))
            best = max(rows, key=fit_layers.selection_r2)
            self.assertEqual((best["k"], best["alpha"]), (record["k"], record["alpha"]))
            self.assertEqual(record["dev"]["raw"], best["dev"])
            applied = record["dev"][record["dev"]["applied"]]["r2"]
            self.assertEqual(applied, max(fit_layers.selection_r2(e) for e in rows))
            if kind == "v":
                self.assertEqual(applied, max(e["dev"]["r2"] for e in rows))
            if (kind, target) != ("k", RENORM_TARGET):
                self.assertGreater(record["dev"]["raw"]["r2"], 0.75)
                self.assertGreater(record["test"]["raw"]["r2"], 0.75)
                self.assertGreater(record["train_r2"], 0.75)

    def test_top_k_recovers_the_generating_map(self):
        width = int(np.prod(SRC_SHAPE))
        for kind in fit_layers.KINDS:
            for target in (0, 1, 2):
                record = self.by_key[kind, target]
                weight = np.load(self.pending / record["files"]["weight"])
                self.assertEqual(weight.shape, (record["k"] * width, int(np.prod(TGT_SHAPE))))
                for index, layer in enumerate(record["source_layers"]):
                    block = weight[index * width:(index + 1) * width]
                    np.testing.assert_allclose(block, self.data.truth[kind, target, layer], atol=0.2)
                    error = np.linalg.norm(block - self.data.truth[kind, target, layer])
                    self.assertLess(error / np.linalg.norm(self.data.truth[kind, target, layer]), 0.25)

    def test_k_renormalization_is_chosen_only_where_it_helps(self):
        for (kind, target), record in self.by_key.items():
            if kind == "v":
                self.assertFalse(record["k_renorm"])
                self.assertNotIn("renorm", record["dev"])
                continue
            dev = record["dev"]
            self.assertEqual(record["k_renorm"], dev["renorm"]["r2"] > dev["raw"]["r2"])
            self.assertEqual(dev["applied"], "renorm" if record["k_renorm"] else "raw")
        renorm = self.by_key["k", RENORM_TARGET]
        self.assertTrue(renorm["k_renorm"])
        self.assertGreater(renorm["dev"]["renorm"]["r2"], renorm["dev"]["raw"]["r2"] + 0.01)
        self.assertGreater(renorm["test"]["renorm"]["r2"], renorm["test"]["raw"]["r2"])
        self.assertFalse(self.by_key["k", 1]["k_renorm"])

    def test_invalid_problems_are_rejected(self):
        data = Dataset(self.root / "invalid", splits=("train",) * 6 + ("dev",) * 2)
        with self.assertRaisesRegex(ValueError, "split test"):
            data.problem()
        data = Dataset(self.root / "invalid2", splits=("train",) * 6 + ("dev",) * 2 + ("eval",))
        with self.assertRaisesRegex(ValueError, "split train, dev or test"):
            data.problem()
        data = Dataset(self.root / "invalid3", splits=("train",) * 6 + ("dev", "test"))
        problem = data.problem()
        with self.assertRaisesRegex(ValueError, "exceeds"):
            fit_layers.fit(problem, ks=(1, 8), work_dir=self.root, pending=self.root)
        with self.assertRaisesRegex(ValueError, "distinct positive"):
            fit_layers.fit(problem, alphas=(0.0, 1.0), ks=(1,), work_dir=self.root, pending=self.root)
        for split in ("dev", "test"):
            data = Dataset(self.root / f"constant-{split}", doc_rows=10, constant=("v", 2, split),
                           splits=("train",) * 6 + ("dev", "test"))
            with self.assertRaisesRegex(ValueError, f"target v-2 is constant on the {split} rows"):
                work = Path(tempfile.mkdtemp(dir=self.root))
                fit_layers.fit(data.problem(), ks=(1,), work_dir=work, pending=work)
        data = Dataset(self.root / "invalid4", splits=("train",) * 6 + ("dev", "test"))
        alignment = data.alignment
        alignment.tgt_class = alignment.tgt_class.copy()
        alignment.tgt_class[alignment.tgt_row[0]] = TEMPLATE
        with self.assertRaisesRegex(ValueError, "not all classified as aligned"):
            data.problem()
        alignment.tgt_class = alignment.tgt_class[:-1]
        with self.assertRaisesRegex(ValueError, "different row counts"):
            data.problem()


class SplitTargetTests(unittest.TestCase):
    def test_target_split_across_differently_keyed_captures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = ("train",) * 10 + ("dev",) * 3 + ("test",) * 3
            whole = Dataset(root / "whole", seed=5, doc_rows=20, splits=splits)
            split = Dataset(root / "split", seed=5, doc_rows=20, splits=splits, split_target=True)
            problem = split.problem()
            self.assertEqual(problem.tgt_layers, list(TGT_LAYERS))
            self.assertEqual(len(problem.tgt_set.captures), 2)
            results = []
            for name, data in (("whole", whole), ("split", split)):
                work, pending = root / f"work-{name}", root / f"pending-{name}"
                work.mkdir()
                pending.mkdir()
                maps, report = fit_layers.fit(data.problem(), ks=(1, 2), chunk_rows=64, work_dir=work,
                                              pending=pending)
                results.append((maps, report, pending))
            (whole_maps, whole_report, whole_pending), (split_maps, split_report, split_pending) = results
            self.assertEqual(whole_maps, split_maps)
            self.assertEqual(whole_report["top_k"], split_report["top_k"])
            for record in split_maps:
                np.testing.assert_array_equal(np.load(whole_pending / record["files"]["weight"]),
                                              np.load(split_pending / record["files"]["weight"]))

            split.alignment.save(root / "alignment.npz")
            output = root / "mapper"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(fit_layers.main(
                    ["--src-capture", str(split.src_path), "--tgt-capture", str(split.tgt_paths[0]),
                     "--tgt-capture", str(split.tgt_paths[1]), "--alignment", str(root / "alignment.npz"),
                     "--output", str(output), "--ks", "1,2", "--threads", "1"]), 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual([Path(c["path"]).name for c in manifest["target"]["captures"]],
                             ["tgt-train", "tgt-eval"])
            self.assertEqual([m["test"] for m in manifest["maps"]], [m["test"] for m in split_maps])

    def test_captures_must_agree_on_k_and_v(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = [{"document_id": doc_id(n), "byte_end": [1, 2, 3]} for n in range(2)]
            arrays = {(kind, layer): np.ones((3, 2, 3), np.float32) for kind in "kv" for layer in (0, 1)}
            first = write_synthetic_capture(root / "a", docs[:1], arrays)
            fewer = write_synthetic_capture(root / "b", docs[1:], {key: value for key, value in arrays.items()
                                                                     if key != ("v", 1)})
            wider = write_synthetic_capture(root / "c", docs[1:], {**arrays, ("k", 1): np.ones((3, 1, 6), np.float32)})
            for other in (fewer, wider):
                with self.assertRaisesRegex(ValueError, "layout"):
                    fit_layers.open_capture_set([first, other])


class CommandLineTests(unittest.TestCase):
    def test_cli_publishes_mapper_that_apply_reproduces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = Dataset(root / "data", seed=3, doc_rows=25,
                           splits=("train",) * 10 + ("dev",) * 3 + ("test",) * 3)
            alignment_path = root / "alignment.npz"
            data.alignment.save(alignment_path)
            output = root / "mapper"
            arguments = ["--src-capture", str(data.src_path), "--tgt-capture", str(data.tgt_path),
                         "--alignment", str(alignment_path), "--output", str(output),
                         "--ks", "4,1,2", "--chunk-rows", "50", "--threads", "1"]
            with contextlib.redirect_stdout(io.StringIO()) as stdout, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(fit_layers.main(arguments), 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(len(summary["maps"]), 2 * len(TGT_LAYERS))
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["alignment.npz", "data", "mapper"])
            manifest = json.loads((output / "manifest.json").read_text())
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(manifest["settings"]["ks"], [1, 2, 4])
            self.assertEqual(manifest["source"]["layers"], list(SRC_LAYERS))
            self.assertEqual(manifest["target"]["layers"], list(TGT_LAYERS))
            self.assertEqual(manifest["rows"]["dev"]["documents"], 3)
            self.assertEqual(manifest["alignment"]["sha256"],
                             hashlib.sha256(alignment_path.read_bytes()).hexdigest())
            self.assertEqual(set(manifest["code_sha256"]), set(fit_layers.CODE_FILES))
            self.assertEqual(len(report["top_k"]), 2 * len(TGT_LAYERS) * 3 * len(fit_layers.DEFAULT_ALPHAS))

            mapper = fit_layers.load_mapper(output)
            test_rows = data.rows("test")
            train_rows = data.rows("train")
            for record in manifest["maps"]:
                kind, target = record["kind"], record["target"]
                layer_map = mapper.get(kind, target)
                self.assertEqual(layer_map.source_layers, record["source_layers"])
                sources = {layer: data.source[kind, layer][test_rows].reshape(-1, *SRC_SHAPE)
                           for layer in SRC_LAYERS}
                predicted = fit_layers.apply(layer_map, sources)
                self.assertEqual(predicted.shape, (len(test_rows),) + TGT_SHAPE)
                np.testing.assert_array_equal(predicted, mapper.apply(kind, target, sources))
                flat = predicted.reshape(len(test_rows), -1)
                actual = data.target[kind, target][test_rows].astype(np.float64)
                applied = record["test"][record["test"]["applied"]]
                sse = np.sum((actual - flat) ** 2)
                self.assertAlmostEqual(applied["r2"], 1 - sse / np.sum((actual - actual.mean(0)) ** 2), places=9)
                self.assertAlmostEqual(applied["mse"], sse / actual.size, places=12)
                mean = data.target[kind, target][train_rows].mean(axis=0, dtype=np.float64)
                np.testing.assert_allclose(np.load(output / record["files"]["mean"]), mean, rtol=1e-6, atol=1e-7)
                weight = np.load(output / record["files"]["weight"]).astype(np.float64)
                bias = np.load(output / record["files"]["bias"]).astype(np.float64)
                direct = np.concatenate([data.source[kind, layer][test_rows] for layer in record["source_layers"]],
                                        axis=1) @ weight + bias
                if record["k_renorm"]:
                    direct = fit_layers.renormalize_heads(direct, *TGT_SHAPE)
                    np.testing.assert_allclose(heads_rms(flat), 1, rtol=1e-5)
                np.testing.assert_allclose(flat, direct, rtol=1e-5, atol=1e-5)
            with self.assertRaises(KeyError):
                fit_layers.apply(mapper.get("v", 2), {3: np.zeros((1,) + SRC_SHAPE, np.float32)})
            with self.assertRaisesRegex(ValueError, "wrong shape"):
                fit_layers.apply(mapper.get("v", 0), {7: np.zeros((2, 3), np.float32)})
            for kind, target in (("v", 0), ("k", RENORM_TARGET)):
                layer_map = mapper.get(kind, target)
                for shape in ((0,) + SRC_SHAPE, (0, int(np.prod(SRC_SHAPE)))):
                    empty = {layer: np.zeros(shape, np.float32) for layer in SRC_LAYERS}
                    result = fit_layers.apply(layer_map, empty)
                    self.assertEqual((result.shape, result.dtype), ((0,) + TGT_SHAPE, np.float32))

            with self.assertRaises(FileExistsError), contextlib.redirect_stderr(io.StringIO()):
                fit_layers.main(arguments)
            weight_path = output / manifest["maps"][0]["files"]["weight"]
            weight_path.chmod(0o644)
            tampered = np.load(weight_path)
            tampered[0, 0] += 1
            weight_path.unlink()
            np.save(weight_path, tampered)
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                fit_layers.load_mapper(output)

    def test_failed_fit_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = Dataset(root / "data", seed=4, doc_rows=12, splits=("train",) * 4 + ("dev", "test"))
            data.alignment.save(root / "alignment.npz")
            arguments = ["--src-capture", str(data.src_path), "--tgt-capture", str(data.tgt_path),
                         "--alignment", str(root / "alignment.npz"), "--output", str(root / "mapper")]
            original = fit_layers.top_k

            def failing(*args, **kwargs):
                raise ValueError("injected failure")

            fit_layers.top_k = failing
            try:
                with self.assertRaisesRegex(ValueError, "injected"), contextlib.redirect_stderr(io.StringIO()):
                    fit_layers.main(arguments + ["--ks", "1,2"])
            finally:
                fit_layers.top_k = original
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["alignment.npz", "data"])
            with self.assertRaisesRegex(ValueError, "exceeds"):
                fit_layers.main(arguments)
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                fit_layers.main(arguments + ["--alphas", "1,1"])

    def test_sigterm_removes_work_and_pending_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = Dataset(root / "data", seed=4, doc_rows=12, splits=("train",) * 4 + ("dev", "test"))
            data.alignment.save(root / "alignment.npz")
            arguments = ["--src-capture", str(data.src_path), "--tgt-capture", str(data.tgt_path),
                         "--alignment", str(root / "alignment.npz"), "--output", str(root / "mapper"),
                         "--ks", "1"]
            original, before = fit_layers.accumulate, signal.getsignal(signal.SIGTERM)
            seen = []

            def terminated(*args, **kwargs):
                seen.append(sorted(prefix for prefix in (".kvpark-fit-work-", ".pending-mapper-")
                                   for p in root.iterdir() if p.name.startswith(prefix)))
                os.kill(os.getpid(), signal.SIGTERM)
                return original(*args, **kwargs)

            fit_layers.accumulate = terminated
            try:
                with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()):
                    fit_layers.main(arguments)
            finally:
                fit_layers.accumulate = original
            self.assertEqual(caught.exception.code, 128 + signal.SIGTERM)
            self.assertEqual(seen, [[".kvpark-fit-work-", ".pending-mapper-"]])
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["alignment.npz", "data"])
            self.assertIs(signal.getsignal(signal.SIGTERM), before)


if __name__ == "__main__":
    unittest.main()
