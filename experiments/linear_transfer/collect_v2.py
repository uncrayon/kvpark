"""Run kvpark-kv-probe offline on rendered documents and publish a validated capture directory.

--mode states builds the capture from the model's own cache instead of graph nodes: the probe
saves every document's sequence state at its end (f32 cache) and capture_from_states turns the
cached rows into k/v (and DSA indexer key) arrays with kvstate. That gives hub views of models
whose graphs name no pre-RoPE key (kimi-k3, bailingmoe3, hy_v4); sliding-window layers, whose
states keep only their last n_swa cells, are refused.
"""

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

from align import segment_table
from captures import KEYS, Capture
from chat import gguf_tokenizer_info
from collect import file_identity, sha256_file, verify_runtime


SHA256 = re.compile(r"[0-9a-f]{64}")
CONDITION_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")
MODES = ("collect", "tokenize", "patch-score", "states")
MAX_TEXT_BYTES = 4 * 1024 * 1024
MEMORY_HEADROOM = 6 * 1024 ** 3
TINY_MODEL_BYTES = 1024 ** 3
BINARY_CONSTANTS = {"device": "cpu", "weight_loading": "mmap", "weight_repacking": False, "mtp": False}


def expected_settings(mode, *, ctx, batch, threads, max_scored=None):
    """Settings the binary must report for a mode (tokenize loads the vocabulary only; states runs
    save-state with collect's cache layout)."""
    if mode == "tokenize":
        return {"ctx": ctx, "add_special": False, "parse_special": True}
    settings = {"ctx": ctx, "batch": batch, "threads": threads, "cache_type": "f32",
                "flash_attention": False, "swa_full": True, "add_special": False, "parse_special": True}
    if mode == "patch-score":
        settings["max_scored"] = max_scored
    return settings


def check_settings(reported, expected):
    if not isinstance(reported, dict) or any(
            name not in reported or type(reported[name]) is not type(value) or reported[name] != value
            for name, value in expected.items()):
        raise ValueError(f"kv_probe report settings differ from the request: {reported!r}")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


class ModelHashCache:
    """SHA-256 of model files, cached by file identity so a run hashes each model once."""

    def __init__(self, path=None):
        self.path = None if path is None else Path(path)
        self.entries = self._read()

    def _read(self):
        if self.path is None or not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("format_version") != 1
                or not isinstance(data.get("entries"), dict)):
            raise ValueError(f"unsupported model hash cache: {self.path}")
        return data["entries"]

    @staticmethod
    def key(path):
        return json.dumps([str(Path(path).resolve()), *file_identity(Path(path).resolve())])

    def sha256(self, path):
        path = Path(path).resolve()
        key = self.key(path)
        cached = self.entries.get(key)
        if isinstance(cached, str) and SHA256.fullmatch(cached):
            return cached
        digest = sha256_file(path)
        if self.key(path) != key:
            raise ValueError(f"model file changed while computing its SHA-256: {path.name}")
        self.entries[key] = digest
        if self.path is not None:
            # Concurrent writers merge under a sidecar lock instead of dropping each other's entries.
            lock = self.path.with_name(self.path.name + ".lock")
            with lock.open("a") as held:
                fcntl.flock(held, fcntl.LOCK_EX)
                self.entries = {**self._read(), key: digest}
                with tempfile.NamedTemporaryFile("w", dir=self.path.parent, prefix=".kvpark-hash-",
                                                 delete=False, encoding="utf-8") as stream:
                    json.dump({"format_version": 1, "entries": self.entries}, stream, indent=2)
                os.replace(stream.name, self.path)
        return digest


def mem_available(meminfo=Path("/proc/meminfo")):
    for line in meminfo.read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            value, unit = line.split()[1:3]
            if unit != "kB":
                raise ValueError("unexpected MemAvailable unit")
            return int(value) * 1024
    raise ValueError("MemAvailable is not reported")


def memory_check(model, skip=False):
    size = Path(model).stat().st_size
    if skip:
        if size >= TINY_MODEL_BYTES:
            raise ValueError("--skip-memory-check is only allowed for tiny fixture models (< 1 GiB)")
        return
    available = mem_available()
    if available <= size + MEMORY_HEADROOM:
        raise ValueError(f"MemAvailable {available / 1024 ** 3:.1f} GiB does not exceed model size "
                         f"{size / 1024 ** 3:.1f} GiB + 6 GiB; refusing to load weights")


def probe_environment(artifacts):
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    for key in list(env):
        if key.startswith("GGML_"):
            env.pop(key)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(sorted({str(path.parent) for path in artifacts}))
    return env


def publish_directory(source, destination):
    """Rename a finished directory into place; never replaces an existing path."""
    destination = Path(destination)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (OSError, AttributeError):
        renameat2 = None
    if renameat2 is not None:
        at_fdcwd, rename_noreplace = -100, 1
        if renameat2(at_fdcwd, os.fsencode(str(source)), at_fdcwd,
                     os.fsencode(str(destination)), rename_noreplace) == 0:
            return
        code = ctypes.get_errno()
        if code not in (errno.ENOSYS, errno.EINVAL):
            raise OSError(code, os.strerror(code), str(destination))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    os.rename(source, destination)


def read_documents(path, mode):
    records, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid document JSON on line {number}") from exc
            if (not isinstance(record, dict) or not isinstance(record.get("document_id"), str)
                    or not SHA256.fullmatch(record["document_id"])
                    or not isinstance(record.get("text"), str) or not record["text"]):
                raise ValueError(f"line {number} needs a 64-hex document_id and nonempty text")
            try:
                size = len(record["text"].encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise ValueError(f"invalid UTF-8 text on line {number}") from exc
            if size > MAX_TEXT_BYTES:
                raise ValueError(f"text exceeds 4 MiB on line {number}")
            if record["document_id"] in seen:
                raise ValueError(f"duplicate document_id on line {number}")
            seen.add(record["document_id"])
            if "segments" in record:
                segment_table(record)
            if mode == "patch-score":
                validate_patch_document(record, number)
            records.append(record)
    if not records:
        raise ValueError("documents file is empty")
    return records


def validate_patch_document(record, number):
    ranges, conditions = record.get("score_ranges"), record.get("conditions")
    size = len(record["text"].encode("utf-8"))
    if not isinstance(record.get("expected_token_sha256"), str) or not SHA256.fullmatch(
            record["expected_token_sha256"]):
        raise ValueError(f"line {number} needs expected_token_sha256")
    if (not isinstance(ranges, list) or not ranges
            or any(not isinstance(item, list) or len(item) != 2
                   or any(type(value) is not int for value in item)
                   or not 0 <= item[0] < item[1] <= size for item in ranges)):
        raise ValueError(f"line {number} has invalid score_ranges")
    if (not isinstance(conditions, list) or not conditions
            or conditions[0] != {"name": "native", "patch_dir": None}):
        raise ValueError(f"line {number}: the first condition must be native with a null patch_dir")
    names = set()
    for condition in conditions:
        if (not isinstance(condition, dict) or set(condition) != {"name", "patch_dir"}
                or not isinstance(condition["name"], str)
                or not CONDITION_NAME.fullmatch(condition["name"]) or condition["name"] in names):
            raise ValueError(f"line {number} has an invalid or duplicate condition name")
        names.add(condition["name"])
        directory = condition["patch_dir"]
        if condition is not conditions[0] and (not isinstance(directory, str)
                                               or not Path(directory).is_absolute()
                                               or not Path(directory).is_dir()):
            raise ValueError(f"line {number}: patch_dir must be an existing absolute directory")


def render_provenance(records):
    """(template_sha256, bos_token) shared by all records, or (None, None) for unrendered text."""
    found = {(record.get("template_sha256"), record.get("bos_token")) for record in records}
    if any(("template_sha256" in record) != ("bos_token" in record)
           or ("segments" in record and "template_sha256" not in record) for record in records):
        raise ValueError("rendered records must carry template_sha256 and bos_token (render with chat.py)")
    if len(found) != 1:
        raise ValueError("documents were rendered with different templates or BOS texts")
    template_sha, bos = found.pop()
    if template_sha is None:
        return None, None
    if not isinstance(template_sha, str) or not SHA256.fullmatch(template_sha) or not isinstance(bos, str):
        raise ValueError("invalid template_sha256 or bos_token in rendered records")
    return template_sha, bos


def check_bos(records, bos, info):
    """The template's BOS must be the model's; with add_bos each text starts with exactly one BOS."""
    if bos != info["bos_token"]:
        raise ValueError(f"documents were rendered with BOS {bos!r} but the model's BOS is "
                         f"{info['bos_token']!r}")
    if info["add_bos"] and bos:
        for record in records:
            if not record["text"].startswith(bos) or record["text"].startswith(bos * 2):
                raise ValueError(f"model adds a BOS but document {record['document_id']} does not "
                                 f"start with exactly one {bos!r}")


def check_capture_bos(capture, records, bos, info):
    if not (info["add_bos"] and bos):
        return
    size = len(bos.encode("utf-8"))
    for record in records:
        first = capture.rows(record["document_id"]).start
        if capture.special[first] != 1 or capture.byte_end[first] != size:
            raise ValueError(f"document {record['document_id']} does not begin with a special BOS token")


def read_capture_spec(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not spec or set(spec) - set(KEYS):
        raise ValueError(f"capture spec keys must be a nonempty subset of {list(KEYS)}")
    for key, value in spec.items():
        if value == "all":
            continue
        if (not isinstance(value, list) or not value
                or any(type(layer) is not int or layer < 0 for layer in value)
                or len(set(value)) != len(value)):
            raise ValueError(f"capture spec {key!r} must be 'all' or unique nonnegative layers")
    return spec


def check_capture_spec(capture, spec):
    attention = capture.attention_layers()
    every = {key: capture.indexer_layers() if key == "ik" else attention for key in spec}
    expected = {(key, layer) for key, value in spec.items()
                for layer in (every[key] if value == "all" else value)}
    actual = {(key, layer) for key in capture.keys() for layer in capture.layers(key)}
    if expected != actual:
        raise ValueError(f"captured tensors differ from the capture spec: missing "
                         f"{sorted(expected - actual)}, unexpected {sorted(actual - expected)}")


def patch_inputs_sha256(records):
    digest = hashlib.sha256()
    for record in records:
        for condition in record["conditions"][1:]:
            directory = Path(condition["patch_dir"])
            for path in sorted(directory.iterdir()):
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"patch directory entries must be regular files: {path.name}")
                digest.update(f"{record['document_id']}/{condition['name']}/{path.name}\0".encode())
                digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def validate_patch_results(directory, records):
    results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    if (not isinstance(results, dict) or results.get("format_version") != 2
            or results.get("mode") != "patch-score" or not isinstance(results.get("documents"), list)):
        raise ValueError("unsupported patch-score results")
    if [item.get("document_id") if isinstance(item, dict) else None
            for item in results["documents"]] != [record["document_id"] for record in records]:
        raise ValueError("patch-score documents differ from the input or its order")
    for record, document in zip(records, results["documents"]):
        names = [condition["name"] for condition in record["conditions"]]
        conditions = document.get("conditions")
        if (not isinstance(conditions, list) or [item.get("name") for item in conditions] != names
                or type(document.get("n_scored")) is not int or document["n_scored"] < 0):
            raise ValueError(f"patch-score conditions differ for {record['document_id']}")
        for name in names:
            path = directory / "scores" / record["document_id"] / f"{name}.npy"
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"missing score file {path.name} for {record['document_id']}")
            scores = np.load(path, allow_pickle=False)
            if scores.dtype.str != "<f4" or scores.shape != (document["n_scored"], 3):
                raise ValueError(f"score file {name} for {record['document_id']} has the wrong shape")
            if not np.isfinite(scores).all():
                raise ValueError(f"score file {name} for {record['document_id']} is nonfinite")
    return results


def run_probe(probe, artifacts, mode, model, records, output, *, ctx, batch, threads,
              capture_spec=None, max_scored=None, device="cpu"):
    """Run the binary into output (must not exist) and return the validated result."""
    input_path = output.parent / (output.name + ".input.jsonl")
    fields = None if mode == "patch-score" else ("document_id", "text")
    with input_path.open("x", encoding="utf-8") as stream:
        for record in records:
            if fields is not None:
                record = {name: record[name] for name in fields}
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    command = [str(probe), "--mode", mode, "--model", str(model), "--input", str(input_path),
               "--output", str(output), "--ctx", str(ctx), "--batch", str(batch),
               "--threads", str(threads)]
    if capture_spec is not None:
        spec_path = output.parent / (output.name + ".capture-spec.json")
        spec_path.write_text(json.dumps(capture_spec), encoding="utf-8")
        command.extend(["--capture-spec", str(spec_path)])
    if max_scored is not None:
        command.extend(["--max-scored", str(max_scored)])
    if device != "cpu":
        command.extend(["--device", device])
    subprocess.run(command, check=True, env=probe_environment(artifacts))
    expected = expected_settings(mode, ctx=ctx, batch=batch, threads=threads, max_scored=max_scored)
    if device != "cpu" and mode != "tokenize":
        expected["device"] = device
    if mode == "patch-score":
        results = validate_patch_results(output, records)
        check_settings(results.get("settings"), expected)
        return results
    capture = Capture(output)
    if capture.report["mode"] != mode:
        raise ValueError("kv_probe report mode differs from the requested mode")
    check_settings(capture.report.get("settings"), expected)
    capture.check_documents(records)
    if mode == "collect":
        check_capture_spec(capture, capture_spec)
        capture.check_finite()
    return capture


def capture_from_states(saved, spec, output):
    """Write a collect-format capture directory `output` (new) from a kv_probe save-state directory
    holding each document's state at its end (f32 cache, flash attention off): per trunk attention
    layer k = the cached K rows de-rotated (kvstate.pre_rope_keys), v = the stored V (an MLA layer's
    c_kv view, kvstate.value_rows) and, for a model with an indexer section, ik = the de-rotated
    indexer keys. The values
    are the cache's own, up to the de-rotation's rounding. Returns the report."""
    import kvstate
    saved, output = Path(saved), Path(output)
    report = json.loads((saved / "report.json").read_text(encoding="utf-8"))
    settings = report.get("settings", {})
    if report.get("mode") != "save-state" or settings.get("cache_type") != "f32" or settings.get("flash_attention"):
        raise ValueError("states captures need a save-state run with an f32 cache and flash attention off")
    trunk = [kind for kind in spec.layer_types if kind != "nextn"]
    if [item["attention"] for item in report["layers"]] != trunk:
        raise ValueError("probe layer table differs from kvstate.ModelSpec")
    if "sliding" in trunk:
        raise ValueError("a state keeps only the last n_swa cells of sliding layers: capture this model with collect")
    attention = [il for il in spec.attention_layers if spec.layer_types[il] != "nextn"]
    wanted = [("k", il) for il in attention] + [("v", il) for il in attention]
    if spec.indexer is not None:
        wanted += [("ik", il) for il in spec.indexer.layers]
    rows = report["rows"]
    sections = {layer: layout.name for layout in spec.layouts(0) for layer in layout.layers}
    output.mkdir()
    arrays, captures = {}, []
    for key, layer in wanted:
        heads, dim = (1, spec.indexer.head_size) if key == "ik" else spec.head_shape(layer, key)
        name = f"{key}-{layer}.npy"
        arrays[key, layer] = np.lib.format.open_memmap(output / name, mode="w+", dtype="<f4",
                                                       shape=(rows, heads, dim))
        captures.append({"key": key, "layer": layer, "tensor": f"state:{'indexer' if key == 'ik' else sections[layer]}",
                         "heads": heads, "dim": dim, "file": name, "dtype": "<f4"})
    documents = []
    for document in report["documents"]:
        n, first = document["n_tokens"], document["first_row"]
        state_entry = [s for s in document["states"] if s["n_tokens"] == n]
        if len(state_entry) != 1:
            raise ValueError(f"document {document['document_id']} has no state at its end")
        state = kvstate.read_file(saved / state_entry[0]["file"], spec)
        for part, layout in zip(state.state.parts, spec.layouts(state.state.flags)):
            if layout.kind != "kv":
                continue
            stream = part.stream()
            positions = np.asarray(stream.positions, dtype=np.int64)
            if not np.array_equal(positions, np.arange(n)):
                raise ValueError(f"{part.name} cells of document {document['document_id']} are not "
                                 f"positions 0..{n - 1}")
            for layer in part.layers:
                if spec.layer_types[layer] == "nextn":
                    continue
                key = "ik" if layout.role == "indexer" else "k"
                arrays[key, layer][first:first + n] = kvstate.pre_rope_keys(spec, layer, stream.k[layer], positions,
                                                                            role=layout.role)
                if layout.role == "attention":
                    arrays["v", layer][first:first + n] = kvstate.value_rows(spec, stream, layer)
        documents.append({"document_id": document["document_id"], "n_tokens": n, "first_row": first})
    for array in arrays.values():
        array.flush()
    del arrays
    for name in ("token_id.npy", "byte_end.npy", "special.npy"):
        (output / name).write_bytes((saved / name).read_bytes())
    layers = [dict(item) for item in report["layers"]]
    if spec.indexer is not None:
        for il in spec.indexer.layers:
            layers[il]["indexer"] = spec.indexer.head_size
    out = {"format_version": 2, "mode": "collect", "architecture": report["architecture"],
           "file_type": report["file_type"], "n_layer": report["n_layer"], "layers": layers, "settings": settings,
           "captures": captures, "documents": documents, "rows": rows, "source": "save-state"}
    (output / "report.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def run_states(probe, artifacts, model, records, output, *, ctx, batch, threads, workdir):
    """kv_probe tokenize and save-state (each document at its end), then capture_from_states into output."""
    import kvstate
    tokens = tokenize(probe, model, records, ctx=ctx, batch=batch, threads=threads, workdir=workdir)
    saves = [{"document_id": r["document_id"], "text": r["text"],
              "save_at": [len(tokens[r["document_id"]]["token_id"])]} for r in records]
    with tempfile.TemporaryDirectory(prefix=".kvpark-states-", dir=workdir) as temporary:
        saved = Path(temporary) / "saved"
        input_path = Path(temporary) / "saved.input.jsonl"
        input_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in saves), encoding="utf-8")
        subprocess.run([str(probe), "--mode", "save-state", "--model", str(model), "--input", str(input_path),
                        "--output", str(saved), "--ctx", str(ctx), "--batch", str(batch), "--threads", str(threads)],
                       check=True, env=probe_environment(artifacts))
        report = json.loads((saved / "report.json").read_text(encoding="utf-8"))
        check_settings(report.get("settings"), expected_settings("states", ctx=ctx, batch=batch, threads=threads))
        capture_from_states(saved, kvstate.ModelSpec.from_gguf(model), output)
    capture = Capture(output)
    capture.check_documents(records)
    capture.check_finite()
    return capture


def tokenize(probe, model, records, *, ctx=5120, batch=512, threads=4, workdir=None):
    """Vocab-only tokenization; returns {document_id: {token_id, byte_end, special}} in input order."""
    probe = Path(probe).resolve()
    _, _, artifacts = verify_runtime(probe)
    records = [{"document_id": record["document_id"], "text": record["text"]} for record in records]
    with tempfile.TemporaryDirectory(prefix=".kvpark-tokenize-", dir=workdir) as temporary:
        capture = run_probe(probe, artifacts, "tokenize", Path(model).resolve(), records,
                            Path(temporary) / "raw", ctx=ctx, batch=batch, threads=threads)
        return {record["document_id"]: {name: np.array(getattr(capture, name)[capture.rows(record["document_id"])])
                                        for name in ("token_id", "byte_end", "special")}
                for record in records}


def model_tokenizer_info(model):
    return gguf_tokenizer_info(model)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path, help="kvpark-kv-probe binary")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--model", required=True, type=Path, help="single GGUF model file")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--docs", required=True, type=Path, help="rendered documents JSONL")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--ctx", type=int, default=5120)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--capture-spec", type=Path, help="collect: capture spec JSON")
    parser.add_argument("--max-scored", type=int, help="patch-score: scored tokens per document")
    parser.add_argument("--chat-template", type=Path, help="template used to render --docs")
    parser.add_argument("--hash-cache", type=Path, help="model SHA-256 cache shared across a run")
    parser.add_argument("--skip-memory-check", action="store_true", help="tiny fixtures only")
    parser.add_argument("--device", choices=("cpu", "vulkan"), default="cpu",
                        help="probe device; vulkan needs a probe built against a Vulkan runtime")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    if (not 0 < args.ctx <= 65536 or not 0 < args.batch <= args.ctx or not 0 < args.threads <= 256
            or not args.model_id.strip()):
        raise ValueError("model ID and positive ctx <= 65536, batch <= ctx, threads <= 256 required")
    if (args.capture_spec is None) != (args.mode != "collect"):
        raise ValueError("--capture-spec is required for collect and not accepted otherwise")
    if args.mode == "states" and args.device != "cpu":
        raise ValueError("--mode states runs on the CPU")
    if args.max_scored is not None and (args.mode != "patch-score" or args.max_scored <= 0):
        raise ValueError("--max-scored must be positive and is only valid for patch-score")
    if args.mode == "patch-score" and args.max_scored is None:
        args.max_scored = 256
    if not args.model.is_file():
        raise ValueError("model must be a single GGUF file")
    docs_bytes = args.docs.read_bytes()
    records = read_documents(args.docs, args.mode)
    capture_spec = read_capture_spec(args.capture_spec) if args.capture_spec else None
    template_sha = sha256_file(args.chat_template) if args.chat_template else None
    render_template_sha, render_bos = render_provenance(records)
    if template_sha is not None and render_template_sha not in (None, template_sha):
        raise ValueError("--chat-template differs from the template the documents were rendered with")
    probe = args.probe.resolve()
    runtime_revision, runtime_identity, artifacts = verify_runtime(probe)
    artifact_stats = {path: file_identity(path) for path in artifacts}
    model = args.model.resolve()
    if args.mode != "tokenize":
        memory_check(model, args.skip_memory_check)
    tokenizer_info = None
    if render_bos is not None:
        tokenizer_info = model_tokenizer_info(model)
        check_bos(records, render_bos, tokenizer_info)
    model_stat = file_identity(model)
    model_sha = ModelHashCache(args.hash_cache).sha256(model)
    patch_sha = patch_inputs_sha256(records) if args.mode == "patch-score" else None
    settings = expected_settings(args.mode, ctx=args.ctx, batch=args.batch, threads=args.threads,
                                 max_scored=args.max_scored)
    settings.update({"vocab_only": True} if args.mode == "tokenize"
                    else {**BINARY_CONSTANTS, "device": args.device})
    with tempfile.TemporaryDirectory(prefix=".kvpark-probe-", dir=args.output.parent) as temporary:
        raw = Path(temporary) / "raw"
        if args.mode == "states":
            result = run_states(probe, artifacts, model, records, raw, ctx=args.ctx, batch=args.batch,
                                threads=args.threads, workdir=temporary)
        else:
            result = run_probe(probe, artifacts, args.mode, model, records, raw, ctx=args.ctx,
                               batch=args.batch, threads=args.threads, capture_spec=capture_spec,
                               max_scored=args.max_scored, device=args.device)
        if file_identity(model) != model_stat:
            raise ValueError("model file changed during the run")
        if args.mode != "patch-score" and tokenizer_info is not None:
            check_capture_bos(result, records, render_bos, tokenizer_info)
        if any(file_identity(path) != before for path, before in artifact_stats.items()):
            raise ValueError("kv_probe runtime artifact changed during the run")
        manifest = {
            "format_version": 2,
            "mode": args.mode,
            "model_id": args.model_id,
            "model_sha256": model_sha,
            "runtime_revision": runtime_revision,
            "collector_build": runtime_identity,
            "wrapper_sha256": sha256_file(Path(__file__).resolve()),
            "validator_sha256": sha256_file(Path(__file__).resolve().with_name("captures.py")),
            "settings": settings,
            "rendered_docs_sha256": sha256_bytes(docs_bytes),
            "documents": len(records),
            "capture_spec": capture_spec,
            "chat_template_sha256": template_sha,
            "render_template_sha256": render_template_sha,
            "render_bos_token": render_bos,
            "model_add_bos": None if tokenizer_info is None else tokenizer_info["add_bos"],
        }
        if args.mode == "patch-score":
            manifest["patch_inputs_sha256"] = patch_sha
        else:
            manifest.update(architecture=result.architecture, file_type=result.report["file_type"],
                            rows=result.rows_total)
        (raw / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                                      allow_nan=False) + "\n", encoding="utf-8")
        publish_directory(raw, args.output)
    print(json.dumps({"output": str(args.output), "mode": args.mode, "documents": len(records),
                      "rows": manifest.get("rows")}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
