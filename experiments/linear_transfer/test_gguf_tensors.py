"""Dequantization and row slicing checks against gguf-py on tiny generated GGUF files."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

import gguf_tensors


@unittest.skipUnless(gguf_tensors.GGUF_PY.is_dir(), "pinned gguf-py is not available")
class GGUFTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.gguf = gguf_tensors._gguf()
        cls.path = Path(cls.temp.name) / "tiny.gguf"
        rng = np.random.default_rng(7)
        types = cls.gguf.GGMLQuantizationType
        cls.source = {
            "f32.weight": (rng.normal(size=(5, 64)).astype(np.float32), types.F32),
            "f16.weight": (rng.normal(size=(6, 64)).astype(np.float32), types.F16),
            "bf16.weight": (rng.normal(size=(4, 32)).astype(np.float32), types.BF16),
            "q8.weight": (rng.normal(size=(9, 96)).astype(np.float32) * 3, types.Q8_0),
            "q8_3d.weight": (rng.normal(size=(2, 3, 64)).astype(np.float32), types.Q8_0),
            "norm.weight": (rng.normal(size=(64,)).astype(np.float32), types.F32),
        }
        writer = cls.gguf.GGUFWriter(cls.path, "muse-glimmer")
        writer.add_block_count(1)
        writer.add_head_count(4)
        writer.add_token_list(["a", "bé", "<|x|>"])
        writer.add_chat_template("{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}")
        cls.stored = {}
        for name, (values, kind) in cls.source.items():
            if kind in (types.F32, types.F16):
                raw = cls.gguf.quants.quantize(values, kind)
                writer.add_tensor(name, raw)
            else:
                raw = cls.gguf.quants.quantize(values, kind)
                writer.add_tensor(name, raw, raw_dtype=kind)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        reader = cls.gguf.GGUFReader(cls.path)
        cls.reference = {tensor.name: cls.gguf.quants.dequantize(tensor.data, tensor.tensor_type)
                         for tensor in reader.tensors}

    @classmethod
    def tearDownClass(cls):
        gguf_tensors._READERS.clear()
        cls.temp.cleanup()

    def test_full_tensors_match_gguf_dequantize(self):
        for name, (values, _) in self.source.items():
            with self.subTest(name=name):
                result = gguf_tensors.read_tensor(self.path, name)
                self.assertEqual(result.dtype, np.float32)
                self.assertEqual(result.shape, values.shape)
                np.testing.assert_array_equal(result, np.asarray(self.reference[name], dtype=np.float32))
                scale = np.abs(values).max()
                np.testing.assert_allclose(result, values, atol=scale * 0.02)

    def test_row_selection_matches_reference_rows(self):
        for name, rows in (("q8.weight", [8, 0, 3, 3]), ("f16.weight", slice(1, 5, 2)),
                           ("f32.weight", np.array([4], dtype=np.int32)), ("bf16.weight", []),
                           ("q8_3d.weight", [1])):
            with self.subTest(name=name, rows=rows):
                result = gguf_tensors.read_tensor(self.path, name, rows)
                expected = np.asarray(self.reference[name], dtype=np.float32)[rows]
                self.assertEqual(result.shape, expected.shape)
                np.testing.assert_array_equal(result, expected)

    def test_chunked_rows_equal_single_pass(self):
        rows = [5, 1, 7, 2, 8]
        np.testing.assert_array_equal(gguf_tensors.read_tensor(self.path, "q8.weight", rows, chunk_rows=2),
                                      gguf_tensors.read_tensor(self.path, "q8.weight", rows))

    def test_rejects_bad_row_requests(self):
        for name, rows in (("q8.weight", [9]), ("q8.weight", [-1]), ("q8.weight", [0.5]),
                           ("norm.weight", [0]), ("q8.weight", [[0]])):
            with self.subTest(name=name, rows=rows), self.assertRaises(ValueError):
                gguf_tensors.read_tensor(self.path, name, rows)
        with self.assertRaises(KeyError):
            gguf_tensors.read_tensor(self.path, "missing.weight")

    def test_metadata_template_tokens_and_info(self):
        metadata = gguf_tensors.read_metadata(self.path)
        self.assertEqual(metadata["general.architecture"], "muse-glimmer")
        self.assertEqual(metadata["muse-glimmer.attention.head_count"], 4)
        self.assertEqual(metadata["tokenizer.ggml.tokens"], ["a", "bé", "<|x|>"])
        short = gguf_tensors.read_metadata(self.path, max_array=2)
        self.assertEqual(short["tokenizer.ggml.tokens"]["array_length"], 3)
        self.assertIn("bos_token", gguf_tensors.chat_template(self.path))
        self.assertEqual(gguf_tensors.token_text(self.path, 1), "bé")
        with self.assertRaises(ValueError):
            gguf_tensors.token_text(self.path, 3)
        info = gguf_tensors.tensor_info(self.path, "q8.weight")
        self.assertEqual((info["type"], info["ne"], info["shape"]), ("Q8_0", [96, 9], [9, 96]))

    def test_independent_q8_0_decoder(self):
        scale = np.float16(0.5).tobytes()
        block = np.frombuffer(scale + np.arange(-16, 16, dtype=np.int8).tobytes(), dtype=np.uint8)
        np.testing.assert_array_equal(gguf_tensors.dequantize_rows(block[None], "Q8_0")[0],
                                      np.arange(-16, 16, dtype=np.float32) * 0.5)
        with self.assertRaises(ValueError):
            gguf_tensors.dequantize_rows(block[None, :33], "Q8_0")
        with self.assertRaises(ValueError):
            gguf_tensors.dequantize_rows(block[None], "Q4_K")


if __name__ == "__main__":
    unittest.main()
