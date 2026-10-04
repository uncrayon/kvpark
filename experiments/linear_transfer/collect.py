"""Collect one model's value projections offline; never contacts a running server."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def read_corpus(path):
    documents, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid corpus JSON on line {number}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"corpus line {number} must contain a text string")
            text = record["text"]
            if not text:
                raise ValueError(f"empty text on corpus line {number}")
            try:
                encoded = text.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(f"invalid UTF-8 text on corpus line {number}") from exc
            if len(encoded) > 1024 * 1024:
                raise ValueError(f"text exceeds 1 MiB on corpus line {number}")
            digest = hashlib.sha256(encoded).hexdigest()
            if digest in seen:
                raise ValueError(f"duplicate document on corpus line {number}")
            seen.add(digest)
            documents.append({"document_sha256": digest, "text": text})
    if not documents:
        raise ValueError("corpus must contain at least one nonempty document")
    return documents


def verify_runtime(collector):
    """Verify the built extractor and record identity without machine-local paths."""
    manifest_path = collector.with_name(collector.name + ".build.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or type(manifest.get("format_version")) is not int
            or manifest["format_version"] != 1):
        raise ValueError("unsupported collector build manifest")
    revision = manifest.get("runtime_source_revision")
    source_sha = manifest.get("collector_source_sha256")
    artifacts = manifest.get("artifacts")
    if (not isinstance(revision, str) or not revision.strip()
            or not isinstance(source_sha, str) or not SHA256.fullmatch(source_sha)
            or not isinstance(artifacts, list) or not artifacts):
        raise ValueError("collector manifest is missing build identity")
    verified, names, paths = [], set(), []
    for artifact in artifacts:
        if (not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str)
                or not isinstance(artifact.get("sha256"), str)
                or not SHA256.fullmatch(artifact["sha256"])):
            raise ValueError("invalid collector artifact identity")
        path = Path(artifact["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("collector artifacts must be existing absolute file paths")
        path = path.resolve()
        if path.name in names:
            raise ValueError("duplicate collector artifact filename")
        names.add(path.name)
        before = file_identity(path)
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"collector artifact hash mismatch: {path.name}")
        if file_identity(path) != before:
            raise ValueError(f"collector artifact changed while hashing: {path.name}")
        paths.append(path)
        verified.append({"name": path.name, "sha256": artifact["sha256"]})
    if collector.resolve() not in paths:
        raise ValueError("collector executable is absent from its build manifest")
    identity = {
        "format_version": 1,
        "runtime_source_revision": revision,
        "collector_source_sha256": source_sha,
        "artifacts": sorted(verified, key=lambda item: item["name"]),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), identity, paths


def read_raw_capture(directory, documents, layer):
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if (not isinstance(report, dict) or type(report.get("format_version")) is not int
            or report["format_version"] != 1 or type(report.get("layer")) is not int
            or report["layer"] != layer
            or report.get("attention_type") not in ("full", "sliding")
            or report.get("cache_dtype") != "f32"
            or report.get("extraction") != "Vcur_pre_cache_f32"
            or not isinstance(report.get("architecture"), str) or not report["architecture"]
            or not isinstance(report.get("file_type"), str)
            or not report["file_type"].isdigit()):
        raise ValueError("unsupported or mismatched collector report")
    captured = report.get("documents")
    if not isinstance(captured, list) or len(captured) != len(documents):
        raise ValueError("collector returned the wrong document count")
    arrays, hashes, ends, tokens = [], [], [], []
    head_shape = None
    for expected, actual in zip(documents, captured):
        digest = expected["document_sha256"]
        if not isinstance(actual, dict) or actual.get("document_sha256") != digest:
            raise ValueError("collector document hashes/order differ from the corpus")
        shape = actual.get("shape")
        offsets, token_ids = actual.get("byte_end"), actual.get("token_ids")
        if (not isinstance(shape, list) or len(shape) != 3
                or any(type(size) is not int or size <= 0 for size in shape)
                or not isinstance(offsets, list) or len(offsets) != shape[0]
                or not isinstance(token_ids, list) or len(token_ids) != shape[0]):
            raise ValueError("invalid collector value shape or row count")
        if (any(type(token) is not int or not 0 <= token <= np.iinfo(np.int64).max
                for token in token_ids)
                or any(type(end) is not int or end <= previous
                       for previous, end in zip([0, *offsets[:-1]], offsets))
                or offsets[-1] != len(expected["text"].encode("utf-8"))):
            raise ValueError("invalid collector token IDs or UTF-8 end offsets")
        if head_shape is not None and shape[1:] != head_shape:
            raise ValueError("collector head layout differs between documents")
        head_shape = shape[1:]
        if actual.get("values_file") != digest + ".f32":
            raise ValueError("collector values filename must be the document hash plus .f32")
        path = directory / actual["values_file"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != math.prod(shape) * 4:
            raise ValueError("collector values file has incorrect size or is not a regular file")
        values = np.fromfile(path, dtype="<f4").reshape(shape)
        if not np.isfinite(values).all():
            raise ValueError("collector values contain nonfinite numbers")
        arrays.append(values)
        hashes.extend([digest] * shape[0])
        ends.extend(offsets)
        tokens.extend(token_ids)
    return report, {
        "values": np.concatenate(arrays),
        "document_sha256": np.asarray(hashes),
        "byte_end": np.asarray(ends, dtype=np.int64),
        "token_id": np.asarray(tokens, dtype=np.int64),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="single GGUF model file")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--corpus", required=True, type=Path, help='JSONL documents: {"text": "..."}')
    parser.add_argument("--layer", required=True, type=int, help="zero-based attention layer")
    parser.add_argument("--output", required=True, type=Path, help="new NPZ capture; never overwritten")
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    if (args.layer < 0 or min(args.ctx, args.batch, args.threads) <= 0
            or args.ctx > 32768 or args.batch > args.ctx or not args.model_id.strip()):
        raise ValueError("layer must be nonnegative; model ID and positive ctx/batch/threads required; batch <= ctx")
    if not args.model.is_file():
        raise ValueError("model must be a single GGUF file")
    documents = read_corpus(args.corpus)
    collector = args.collector.resolve()
    runtime_revision, runtime_identity, artifacts = verify_runtime(collector)
    artifact_stats = {path: file_identity(path) for path in artifacts}
    model = args.model.resolve()
    model_stat = file_identity(model)
    model_revision = sha256_file(model)
    if file_identity(model) != model_stat:
        raise ValueError("model file changed while computing its SHA-256")
    settings = {
        "ctx": args.ctx, "batch": args.batch, "threads": args.threads, "device": "cpu",
        "weight_loading": "mmap", "weight_repacking": False,
        "flash_attention": False, "k_cache_dtype": "f32", "v_cache_dtype": "f32",
        "add_special": False, "parse_special": False, "mtp": False,
    }
    with tempfile.TemporaryDirectory(prefix=".kvpark-capture-", dir=args.output.parent) as temporary:
        temporary = Path(temporary)
        input_path, raw_dir = temporary / "input.jsonl", temporary / "raw"
        with input_path.open("w", encoding="utf-8") as stream:
            for document in documents:
                stream.write(json.dumps(document, ensure_ascii=False) + "\n")
        command = [str(collector), "--model", str(model), "--input", str(input_path),
                   "--output", str(raw_dir), "--layer", str(args.layer)]
        for option in ("ctx", "batch", "threads"):
            command.extend(["--" + option, str(settings[option])])
        env = os.environ.copy()
        env.pop("LD_PRELOAD", None)
        for key in list(env):
            if key.startswith("GGML_"):
                env.pop(key)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(sorted({str(path.parent) for path in artifacts}))
        subprocess.run(command, check=True, env=env)
        if file_identity(model) != model_stat:
            raise ValueError("model file changed during collection")
        if any(file_identity(path) != before for path, before in artifact_stats.items()):
            raise ValueError("collector runtime artifact changed during collection")
        report, arrays = read_raw_capture(raw_dir, documents, args.layer)
        metadata = {
            "format_version": 1,
            "model_id": args.model_id,
            "model_revision": model_revision,
            "runtime_revision": runtime_revision,
            "weight_quantization": report["file_type"],
            "architecture": report["architecture"],
            "layer": args.layer,
            "attention_type": report["attention_type"],
            "cache_kind": "value",
            "cache_dtype": report["cache_dtype"],
            "text_mode": "raw_utf8_no_special_tokens",
            "extraction": report["extraction"],
            "collector_build": runtime_identity,
            "collector_wrapper_sha256": sha256_file(Path(__file__).resolve()),
            "settings": settings,
        }
        pending = temporary / "capture.npz"
        with pending.open("xb") as stream:
            np.savez(stream, **arrays, metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
        # Hard-link publication is atomic and fails if a competing writer created output.
        os.link(pending, args.output)
    print(json.dumps({"output": str(args.output), "documents": len(documents),
                      "rows": len(arrays["values"]), "shape": list(arrays["values"].shape)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
