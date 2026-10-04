"""Collect sequentially, split whole documents, and fit an offline value mapper."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

import collect
import probe


HERE = Path(__file__).resolve().parent


def read_split_corpus(path):
    documents = collect.read_corpus(path)
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    identifiers, counts = set(), {"train": 0, "eval": 0}
    for document, record in zip(documents, records):
        split = record.get("split")
        if split not in counts:
            raise ValueError("every corpus document must have split 'train' or 'eval'")
        identifier = record.get("id")
        if "id" in record:
            if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
                raise ValueError("supplied corpus IDs must be unique nonempty strings")
            identifiers.add(identifier)
        document["split"] = split
        counts[split] += 1
    if counts["train"] < 2 or counts["eval"] < 1:
        raise ValueError("corpus needs at least two training documents and one evaluation document")
    return documents


def stable_file_hash(path):
    before = collect.file_identity(path)
    digest = collect.sha256_file(path)
    if collect.file_identity(path) != before:
        raise ValueError(f"input changed while hashing: {path.name}")
    return digest


def publish_json(path, value):
    """Publish complete files without replacing existing data, including on resume."""
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    publish_text(path, text)


def publish_text(path, text):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".pending-", delete=False) as stream:
        pending = Path(stream.name)
        stream.write(text)
    try:
        os.link(pending, path)
    finally:
        pending.unlink()


def validate_capture(path, documents, identity, side):
    capture = probe.load_capture(path)
    expected = identity[side]
    metadata = capture.metadata
    if (metadata["model_id"] != expected["model_id"]
            or metadata["model_revision"] != expected["model_sha256"]
            or metadata["runtime_revision"] != identity["runtime_revision"]
            or metadata["layer"] != expected["layer"]
            or metadata.get("collector_wrapper_sha256") != identity["code_sha256"]["collect.py"]
            or metadata.get("cache_dtype") != "f32"
            or metadata.get("extraction") != "Vcur_pre_cache_f32"
            or capture.values.dtype != np.dtype("<f4")
            or any(metadata.get("settings", {}).get(key) != value
                   for key, value in identity["settings"].items())):
        raise ValueError(f"{side} capture identity differs from this run")
    if set(capture.documents) != {item["document_sha256"] for item in documents}:
        raise ValueError(f"{side} capture documents differ from the corpus")
    for document in documents:
        offsets = capture.byte_end[capture.documents == document["document_sha256"]]
        if np.any(np.diff(offsets) <= 0) or offsets[-1] != len(document["text"].encode("utf-8")):
            raise ValueError(f"{side} capture has incomplete or unordered document offsets")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in
                  ("values", "document_sha256", "byte_end", "token_id", "metadata_json")}
    if (arrays["token_id"].shape != capture.byte_end.shape
            or arrays["token_id"].dtype.kind not in "iu" or np.any(arrays["token_id"] < 0)):
        raise ValueError(f"{side} capture lacks valid token provenance")
    return arrays


def publish_split(path, arrays, document_hashes):
    selected = np.isin(arrays["document_sha256"], list(document_hashes))
    split = {key: value if key == "metadata_json" else value[selected]
             for key, value in arrays.items()}
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(split) or any(
                    not np.array_equal(existing[key], value) for key, value in split.items()):
                raise ValueError(f"existing split capture differs from its source: {path.name}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
        pending = Path(stream.name)
        np.savez(stream, **split)
    try:
        os.link(pending, path)
    finally:
        pending.unlink()


def collect_one(args, side, corpus_path, output):
    command = [sys.executable, "-u", str(HERE / "collect.py"), "--collector", str(args.collector),
               "--model", str(getattr(args, side + "_model")), "--model-id",
               getattr(args, side + "_model_id"), "--corpus", str(corpus_path), "--layer",
               str(getattr(args, side + "_layer")), "--output", str(output)]
    for setting in ("ctx", "batch", "threads"):
        command.extend(["--" + setting, str(getattr(args, setting))])
    log_path = args.output / (side + ".log")
    print(json.dumps({"stage": "collect", "model": side, "log": str(log_path)}), flush=True)
    with log_path.open("ab") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def fit_captures(output):
    inputs = {f"{side}-{split}.npz": collect.sha256_file(output / f"{side}-{split}.npz")
              for side in ("source", "target") for split in ("train", "eval")}
    for name in ("probe.py", "ridge.py"):
        inputs[name] = collect.sha256_file(HERE / name)
    fit = output / "fit"
    if fit.exists():
        stage = json.loads((fit / "stage.json").read_text(encoding="utf-8"))
        if stage.get("inputs") != inputs or set(stage.get("outputs", {})) != {"mapper.npz", "report.json"}:
            raise ValueError("existing fit does not match the current captures")
        if any(collect.sha256_file(fit / name) != digest for name, digest in stage["outputs"].items()):
            raise ValueError("existing fit artifacts have changed")
    else:
        with tempfile.TemporaryDirectory(dir=output, prefix=".fit-") as temporary:
            pending = Path(temporary) / "fit"
            arguments = []
            for side in ("source", "target"):
                for split in ("train", "eval"):
                    arguments.extend([f"--{split}-{side}", str(output / f"{side}-{split}.npz")])
            probe.main(arguments + ["--output", str(pending)])
            stage = {"inputs": inputs, "outputs": {
                name: collect.sha256_file(pending / name) for name in ("mapper.npz", "report.json")}}
            publish_json(pending / "stage.json", stage)
            pending.rename(fit)
    return json.loads((fit / "report.json").read_text(encoding="utf-8"))


def run(args, documents, identity):
    corpus_path = args.output / "collection.jsonl"
    corpus_text = "".join(json.dumps({"text": document["text"]}, ensure_ascii=False) + "\n"
                          for document in documents)
    if corpus_path.exists():
        if corpus_path.read_text(encoding="utf-8") != corpus_text:
            raise ValueError("saved collection corpus differs from this run")
    else:
        publish_text(corpus_path, corpus_text)
    for side in ("source", "target"):
        path = args.output / f"{side}-all.npz"
        if not path.exists():
            collect_one(args, side, corpus_path, path)
        arrays = validate_capture(path, documents, identity, side)
        for split in ("train", "eval"):
            document_hashes = {document["document_sha256"] for document in documents
                               if document["split"] == split}
            publish_split(args.output / f"{side}-{split}.npz", arrays, document_hashes)
        del arrays
        print(json.dumps({"stage": "captured", "model": side}), flush=True)
    report = fit_captures(args.output)
    summary = {
        "format_version": 1,
        "purpose": "offline_value_reconstruction_prototype",
        "documents": {split: sum(document["split"] == split for document in documents)
                      for split in ("train", "eval")},
        "source": identity["source"],
        "target": identity["target"],
        "train_alignment": report["train_alignment"],
        "eval_alignment": report["eval_alignment"],
        "held_out_values": report["held_out_values"],
        "limitations": report["limitations"],
        "mapper": "fit/mapper.npz",
        "report": "fit/report.json",
    }
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
            raise ValueError("existing summary differs from the validated fit")
    else:
        publish_json(summary_path, summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector", required=True, type=Path)
    for side, layer in (("source", 31), ("target", 27)):
        parser.add_argument(f"--{side}-model", required=True, type=Path)
        parser.add_argument(f"--{side}-model-id", required=True)
        parser.add_argument(f"--{side}-layer", type=int, default=layer)
    parser.add_argument("--corpus", required=True, type=Path,
                        help="JSONL records with text, split (train/eval), and optional unique id")
    parser.add_argument("--output", required=True, type=Path, help="new checkpoint directory")
    parser.add_argument("--resume", action="store_true", help="resume a matching existing checkpoint directory")
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if (min(args.ctx, args.batch, args.threads) <= 0 or args.ctx > 32768 or args.batch > args.ctx
            or min(args.source_layer, args.target_layer) < 0
            or not args.source_model_id.strip() or not args.target_model_id.strip()):
        raise ValueError("invalid model identity, layer, context, batch, or thread count")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output already exists; use --resume with identical inputs: {args.output}")
    if args.resume and not args.output.is_dir():
        raise ValueError("--resume requires an existing checkpoint directory")
    corpus_sha256 = stable_file_hash(args.corpus)
    documents = read_split_corpus(args.corpus)
    if stable_file_hash(args.corpus) != corpus_sha256:
        raise ValueError("corpus changed while reading its documents")
    runtime_revision, _, _ = collect.verify_runtime(args.collector.resolve())
    print(json.dumps({"stage": "fingerprinting_inputs"}), flush=True)
    identity = {
        "format_version": 1,
        "corpus_sha256": corpus_sha256,
        "runtime_revision": runtime_revision,
        "settings": {"ctx": args.ctx, "batch": args.batch, "threads": args.threads, "device": "cpu"},
        "code_sha256": {name: stable_file_hash(HERE / name)
                        for name in ("collect.py", "probe.py", "ridge.py", "run_prototype.py")},
    }
    for side in ("source", "target"):
        identity[side] = {"model_id": getattr(args, side + "_model_id"),
                          "layer": getattr(args, side + "_layer"),
                          "model_sha256": stable_file_hash(getattr(args, side + "_model"))}
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another process is already running this prototype directory") from exc
        manifest = args.output / "run.json"
        if args.resume:
            if json.loads(manifest.read_text(encoding="utf-8")) != identity:
                raise ValueError("resume input fingerprints differ; use a new output directory")
        else:
            publish_json(manifest, identity)
        run(args, documents, identity)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
