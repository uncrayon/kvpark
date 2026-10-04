import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from apply_mapper import main, predict
from probe import Capture


class MapperApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "mapper.npz"
        self.source = Capture(np.arange(24, dtype=np.float32).reshape(4, 2, 3),
                              np.array(["a" * 64] * 4), np.arange(1, 5),
                              {"model_id": "source", "model_revision": "fixed", "layer": 3})
        self.metadata = {"format_version": 1, "purpose": "value_reconstruction_probe_only",
                         "source": self.source.metadata.copy(), "target": {"model_id": "target"},
                         "source_head_shape": [2, 3], "source_capture_dtype": "float32",
                         "target_head_shape": [1, 2]}
        self.weight = np.arange(12, dtype=np.float64).reshape(6, 2) / 10
        self.bias = np.array([2., -3.])

    def save(self):
        np.savez(self.path, weight=self.weight, bias=self.bias,
                 metadata_json=np.array(json.dumps(self.metadata)))

    def test_applies_saved_mapper_with_different_source_and_target_widths(self):
        self.save()
        actual, metadata, elapsed = predict(self.path, self.source)
        expected = self.source.values.reshape(4, 6) @ self.weight + self.bias
        np.testing.assert_allclose(actual[:, 0, :], expected)
        self.assertEqual(metadata["target"]["model_id"], "target")
        self.assertGreaterEqual(elapsed, 0)

    def test_rejects_source_revision_or_direction_mismatch(self):
        self.save()
        for field, value in (("model_id", "target"), ("model_revision", "other"), ("layer", 7)):
            original = self.source.metadata[field]
            self.source.metadata[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity"):
                predict(self.path, self.source)
            self.source.metadata[field] = original

    def test_rejects_invalid_weights_and_layout(self):
        self.weight[0, 0] = np.nan
        self.save()
        with self.assertRaisesRegex(ValueError, "weights"):
            predict(self.path, self.source)
        self.weight[0, 0] = 0
        self.metadata["source_head_shape"] = [3, 2]
        self.save()
        with self.assertRaisesRegex(ValueError, "layout"):
            predict(self.path, self.source)

    def test_failed_save_and_existing_output_do_not_publish_or_overwrite(self):
        self.save()
        output = Path(self.temp.name) / "predictions.npz"
        arguments = ["--mapper", str(self.path), "--source", "unused.npz", "--output", str(output)]
        with patch("apply_mapper.load_capture", return_value=self.source):
            with patch("apply_mapper.np.savez", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    main(arguments)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(self.temp.name).glob(".predictions-*")), [])
            output.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                main(arguments)
            self.assertEqual(output.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
