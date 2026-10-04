"""Apply an offline value mapper on source token ends, without loading either LLM."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from probe import load_capture


def predict(mapper_path, source):
    with np.load(mapper_path, allow_pickle=False) as data:
        weight, bias = data["weight"], data["bias"]
        metadata = json.loads(data["metadata_json"].item())
    if (metadata.get("format_version") != 1
            or metadata.get("purpose") != "value_reconstruction_probe_only"):
        raise ValueError("unsupported mapper format or purpose")
    if metadata.get("source") != source.metadata:
        raise ValueError("source capture identity does not match the mapper direction")
    if (metadata.get("source_head_shape") != list(source.values.shape[1:])
            or metadata.get("source_capture_dtype") != str(source.values.dtype)):
        raise ValueError("source capture layout or dtype does not match the mapper")
    target_shape = metadata.get("target_head_shape")
    if (not isinstance(target_shape, list) or len(target_shape) != 2
            or any(type(size) is not int or size <= 0 for size in target_shape)):
        raise ValueError("invalid target head shape")
    features = int(np.prod(source.values.shape[1:]))
    outputs = int(np.prod(target_shape))
    if (weight.shape != (features, outputs) or bias.shape != (outputs,)
            or weight.dtype.kind != "f" or bias.dtype.kind != "f"
            or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
        raise ValueError("invalid mapper weights or bias")
    values = source.values.reshape(len(source.values), features).astype(np.float64)
    started = time.perf_counter()
    predicted = (values @ weight + bias).reshape(len(values), *target_shape)
    elapsed = time.perf_counter() - started
    if not np.isfinite(predicted).all():
        raise ValueError("mapping produced nonfinite values")
    return predicted, metadata, elapsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapper", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new predictions NPZ, not a native cache")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    source = load_capture(args.source)
    predicted, mapper_metadata, elapsed = predict(args.mapper, source)
    metadata = {
        "format_version": 1,
        "purpose": "predicted_values_on_source_token_ends",
        "complete_target_cache": False,
        "mapper_sha256": hashlib.sha256(args.mapper.read_bytes()).hexdigest(),
        "source": source.metadata,
        "target": mapper_metadata["target"],
        "matrix_application_seconds": elapsed,
        "timing_scope": "matrix multiply and bias only; excludes capture/load/save and is not handoff latency",
    }
    # The distinct field name prevents predictions being mistaken for measured captures.
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".predictions-") as temporary:
        pending = Path(temporary) / "predictions.npz"
        np.savez(pending, predicted_values=predicted, document_sha256=source.documents,
                 source_byte_end=source.byte_end,
                 metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
        os.link(pending, args.output)
    print(json.dumps({"output": str(args.output), "shape": list(predicted.shape),
                      "matrix_application_seconds": elapsed, "complete_target_cache": False}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
