"""Check one model end to end with kvpark-kv-probe and kvstate on one chat.py-rendered document.

Probe runs (each into a new directory under --output):
  tokenize    vocabulary only; its token ids must equal those of every later run
  collect     k/v of every attention layer (Kcur_normed and Vcur, the MLA row [c_kv | k_pe] and its
              c_kv view, ...) and a DSA model's indexer keys (ik), same cache layout as below
  save-state  a sequence state after --seed-tokens tokens (default: the largest multiple of
              --batch before the first scored token, so the seeded run repeats the native ubatches)
  seed-score  native vs the saved state (on one device this must give KL exactly 0) and vs a
              "rebuilt" state whose K/V were replaced by the collected values (K re-rotated with
              kvstate.cache_keys, V cast to the cache type); recurrent sections stay native
kvstate checks on the saved file: parse_file/to_bytes byte round trip, section layout and
positions, V == collected V (cast to the cache type; an MLA layer's V is the c_kv view of its K
row, kvstate.value_rows), and kvstate.pre_rope_keys(stored K) vs the collected pre-RoPE K (and
indexer keys) in units of the cache type's spacing at each rotated pair's norm (Hadamard-rotated
indexer keys: at the row's RMS); nextn layers a section keeps must hold zero rows. DeepSeek V4
(deepseek4): K is the raw latent (no V key), the collected compressor rows [a | s] must equal
the rings' last tokens (a, s + APE) exactly, and the rebuilt state replaces the raw window only
(its pooled rows stay native; pooling is checked in experiments/universal_kv/test_dsv4.py).

Scored tokens are those of the final message (record["final_message_segments"]); without
--document-id the shortest test-split document whose final message starts after --batch
tokens is used. Writes <output>/summary.json.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from collect import verify_runtime
import collect_v2
import kvstate


def run_probe(probe, environment, mode, model, records, output, options, timeout):
    docs = output.parent / f"{output.name}.input.jsonl"
    docs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    command = [str(probe), "--mode", mode, "--model", str(model), "--input", str(docs), "--output", str(output),
               *map(str, options)]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=timeout)
    (output.parent / f"{output.name}.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{mode} failed:\n{result.stderr[-3000:]}")
    name = "results.json" if mode.endswith("score") else "report.json"
    return json.loads((output / name).read_text(encoding="utf-8")), time.monotonic() - started


def score_ranges(record):
    final = set(record["final_message_segments"])
    return [[start, end] for index, start, end in record["segments"] if index in final]


def first_scored(byte_end, ranges):
    """First token position p >= 1 whose bytes lie inside a score range (the probe's rule)."""
    for p in range(1, len(byte_end)):
        start, end = byte_end[p - 1], byte_end[p]
        if any(lo <= start and end <= hi for lo, hi in ranges):
            return p
    return None


def pick_document(records, tokens, batch, max_tokens):
    candidates = []
    for record in records:
        if record.get("split") != "test" or not record.get("final_message_segments"):
            continue
        ids = tokens[record["document_id"]]
        first = first_scored(ids["byte_end"].tolist(), score_ranges(record))
        if first is not None and first > batch and len(ids["token_id"]) <= max_tokens:
            candidates.append((len(ids["token_id"]), record["document_id"]))
    if not candidates:
        raise ValueError("no test document has a final message after --batch tokens")
    return min(candidates)[1]


def pair_norms(spec, layer, k, role="attention"):
    """Norm of the rotated pair each element belongs to (elements outside the RoPE window, and every
    element of a NoPE layer: their own magnitude)."""
    k = np.asarray(k, dtype=np.float64)
    out = np.abs(k)
    rope = spec.rope(layer, role)
    if rope is None:
        return out
    o, n = rope.offset, rope.n_dims
    if rope.mode == "norm":
        a, b = slice(o, o + n, 2), slice(o + 1, o + n, 2)
    else:
        a, b = slice(o, o + n // 2), slice(o + n // 2, o + n)
    norm = np.hypot(k[..., a], k[..., b])
    out[..., a], out[..., b] = norm, norm
    return out


def trunk_layer_types(spec):
    """The probe's layer table: the kind of every trunk layer (nextn layers are not trunk layers)."""
    return [kind for kind in spec.layer_types if kind != "nextn"]


def captured_layers(spec):
    """{key: layers} kv_probe's collect must return: k/v of every trunk attention layer, the DSA indexer
    keys of every indexer layer (qwen4exp's pooled-block indexer has no capture key); DeepSeek V4: k and
    the compressor rows of its CSA, HCA and LID layers."""
    attention = [il for il in spec.attention_layers if spec.layer_types[il] != "nextn"]
    if spec.memory == "dsv4":
        return {"k": attention, **{pool.name: list(pool.layers) for pool in spec.pools() if pool.layers}}
    out = {"k": attention, "v": attention}
    if spec.memory in ("dsa", "dsa_iswa"):
        out["ik"] = list(spec.indexer.layers)
    return out


def captured_shape(spec, key, layer):
    if spec.memory == "dsv4" and key != "k":
        return 1, 2 * spec.pool(key).state_width             # [a | s]
    return (1, spec.indexer.head_size) if key == "ik" else spec.head_shape(layer, key)


def check_rings(spec, part, arrays, model, n):
    """A DeepSeek V4 ring section against the collected compressor rows: its last tokens' a and s + APE."""
    import gguf_tensors
    pool = spec.pool(part.name[:-len("_state")])
    item = {"name": part.name, "kind": "ring", "layers": list(part.layers), "exact": True}
    last = np.arange(max(0, n - pool.state_size), n)
    for layer in part.layers:
        rows = arrays[(pool.name, layer)].reshape(len(arrays[(pool.name, layer)]), -1)
        a, s = rows[:, :pool.state_width], rows[:, pool.state_width:]
        name = "indexer_compressor_ape" if pool.name == "lid" else "attn_compressor_ape"
        ape = gguf_tensors.read_tensor(model, f"blk.{layer}.{name}.weight")
        item["exact"] &= bool(np.array_equal(part.kv[layer][0][last % pool.state_size], a[last])
                              and np.array_equal(part.score[layer][0][last % pool.state_size],
                                                 (s[last] + ape[last % pool.ratio]).astype(np.float32)))
    return item


def check_state(spec, data, tokens, arrays, dtype, model=None):
    """kvstate checks of one saved state against the collected arrays (a DeepSeek V4 model's rings also
    need its GGUF, `model`, for the APE)."""
    parsed = kvstate.parse_file(data, spec)
    n = len(parsed.tokens)
    out = {"round_trip_bytes_equal": parsed.to_bytes() == data, "n_tokens": n,
           "tokens_equal": bool(np.array_equal(parsed.tokens, tokens[:n])), "sections": [], "layers": {}}
    parsed.check_positions()
    np_dtype = np.float16 if dtype == "f16" else np.float32
    worst = {"v_max_abs": 0.0, "k_derot_max_abs": 0.0, "k_derot_max_ulps": 0.0}
    for part, layout in zip(parsed.state.parts, spec.layouts(parsed.state.flags)):
        if isinstance(part, kvstate.RecurrentState):
            out["sections"].append({"name": part.name, "kind": "recurrent", "layers": list(part.layers),
                                    "positions": part.positions.tolist(), "n_layer": part.n_layer,
                                    "bytes": part.nbytes})
            continue
        if isinstance(part, kvstate.PooledRows):
            out["sections"].append({"name": part.name, "kind": "rows", "layers": list(part.layers),
                                    "rows": part.n_rows})
            continue
        if isinstance(part, kvstate.CompressorRing):
            out["sections"].append(check_rings(spec, part, arrays, model, n))
            worst["rings_exact"] = worst.get("rings_exact", True) and out["sections"][-1]["exact"]
            continue
        stream = part.stream()
        positions = np.asarray(stream.positions)
        if layout.role == "indexer" and not any(key == "ik" for key, _ in arrays):
            continue                    # indexer keys without a capture key (qwen4exp)
        out["sections"].append({"name": part.name, "kind": "kv", "role": layout.role, "layers": list(part.layers),
                                "cells": int(len(positions)), "pos_min": int(positions.min()),
                                "pos_max": int(positions.max()), "v_trans": stream.v_trans,
                                "cell_ext": stream.ext is not None, "k_only": sorted(set(stream.k) - set(stream.v))})
        for layer in part.layers:
            if spec.layer_types[layer] == "nextn":
                # the trunk never writes a nextn layer the cache keeps (src/llama-kv-cache.cpp clears it)
                out["layers"][str(layer)] = {"nextn_zero": not np.asarray(stream.k[layer]).any()}
                continue
            indexer = layout.role == "indexer"
            name, key, prefix = (f"indexer {layer}", "ik", "ik") if indexer else (str(layer), "k", "k")
            item = {}
            if not indexer and ("v", layer) in arrays:
                v_ref = arrays[("v", layer)][positions].astype(np_dtype)
                v_diff = np.abs(kvstate.value_rows(spec, stream, layer).astype(np.float32) - v_ref.astype(np.float32))
                item.update(v_exact_fraction=float(np.mean(v_diff == 0)), v_max_abs=float(v_diff.max()))
                worst["v_max_abs"] = max(worst["v_max_abs"], float(v_diff.max()))
            k_ref = arrays[(key, layer)][positions]
            stored = stream.k[layer].astype(np.float32)
            derot = kvstate.pre_rope_keys(spec, layer, stored, positions, role=layout.role)
            error = np.abs(derot.astype(np.float64) - k_ref)
            norms = pair_norms(spec, layer, k_ref, layout.role)
            if indexer and spec.indexer.hadamard:
                # H spreads every stored element's rounding over the whole row: units of the row's RMS
                norms = np.broadcast_to(np.sqrt(np.mean(np.square(k_ref.astype(np.float64)), axis=-1, keepdims=True)),
                                        k_ref.shape)
            norms = np.maximum(norms, np.finfo(np_dtype).tiny)
            spacing = np.spacing(norms.astype(np_dtype)).astype(np.float64)
            ulps = float((error / spacing).max())
            item.update({f"{prefix}_derot_max_abs": float(error.max()), f"{prefix}_max_abs": float(np.abs(k_ref).max()),
                         f"{prefix}_derot_max_ulps": ulps})
            out["layers"][name] = item
            for metric, value in ((f"{prefix}_derot_max_abs", float(error.max())), (f"{prefix}_derot_max_ulps", ulps)):
                worst[metric] = max(worst.get(metric, 0.0), value)
    out["worst"] = worst
    return parsed, out


def rebuild(spec, parsed, arrays, dtype):
    """The parsed state with every KV layer's K/V (and indexer keys) replaced by collected values (K via
    cache_keys); MLA layers store no V, nextn layers keep their zero rows."""
    np_dtype = np.float16 if dtype == "f16" else np.float32
    for part, layout in zip(parsed.state.parts, spec.layouts(parsed.state.flags)):
        if not isinstance(part, kvstate.KVCache):
            continue                    # recurrent rows, DeepSeek V4 pooled rows and rings stay native
        stream = part.stream()
        positions = np.asarray(stream.positions)
        key = "ik" if layout.role == "indexer" else "k"
        if layout.role == "indexer" and not any(name == "ik" for name, _ in arrays):
            continue
        for layer in part.layers:
            if spec.layer_types[layer] == "nextn":
                continue
            stream.k[layer] = kvstate.cache_keys(spec, layer, arrays[(key, layer)][positions], positions,
                                                 role=layout.role).astype(np_dtype)
            if layer in stream.v:
                stream.v[layer] = arrays[("v", layer)][positions].astype(np_dtype)
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", required=True, type=Path, help="kvpark-kv-probe with a .build.json manifest")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--docs", required=True, type=Path, help="chat.py rendered JSONL for this model")
    parser.add_argument("--document-id", help="document to use (default: see above)")
    parser.add_argument("--output", required=True, type=Path, help="new directory")
    parser.add_argument("--device", choices=("cpu", "vulkan"), default="vulkan")
    parser.add_argument("--cache-type", choices=("f32", "f16"), default="f16")
    parser.add_argument("--flash-attn", choices=("on", "off"), default="on")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed-tokens", type=int, help="state size (default: see above)")
    parser.add_argument("--max-scored", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=4096, help="largest document considered")
    parser.add_argument("--timeout", type=int, default=900, help="seconds per probe run")
    parser.add_argument("--skip-memory-check", action="store_true", help="tiny fixtures only")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    probe, model = args.probe.resolve(), args.model.resolve()
    _, _, artifacts = verify_runtime(probe)
    environment = collect_v2.probe_environment(artifacts)
    collect_v2.memory_check(model, args.skip_memory_check)
    spec = kvstate.ModelSpec.from_gguf(model)
    records = collect_v2.read_documents(args.docs, "tokenize")
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    tokens = collect_v2.tokenize(probe, model, records, ctx=32768, batch=512, threads=args.threads, workdir=output)
    document_id = args.document_id or pick_document(records, tokens, args.batch, args.max_tokens)
    record = next(r for r in records if r["document_id"] == document_id)
    ids, byte_end = tokens[document_id]["token_id"], tokens[document_id]["byte_end"]
    n = len(ids)
    ranges = score_ranges(record)
    first = first_scored(byte_end.tolist(), ranges)
    if first is None:
        raise ValueError("the document's final message has no scored tokens")
    seed = args.seed_tokens or (first - 1) // args.batch * args.batch
    if not 0 < seed < first:
        raise ValueError(f"seed {seed} must lie in [1, {first - 1}] (first scored token {first})")
    ctx = -(-n // 256) * 256
    layout = ["--ctx", ctx, "--batch", args.batch, "--threads", args.threads, "--cache-type", args.cache_type,
              "--flash-attn", args.flash_attn, "--device", args.device]
    base = {"document_id": document_id, "text": record["text"]}
    timings = {}

    # the probe's default batch (512) may exceed a short document's ctx; n > --batch (pick_document)
    report, timings["tokenize"] = run_probe(probe, environment, "tokenize", model, [base], output / "tokenize",
                                            ["--ctx", ctx, "--batch", min(args.batch, ctx)], args.timeout)
    token_ids = np.load(output / "tokenize" / "token_id.npy")
    wanted = captured_layers(spec)
    capture_options = []
    if "ik" in wanted or spec.memory == "dsv4":
        (output / "capture-spec.json").write_text(json.dumps({key: "all" for key in wanted}), encoding="utf-8")
        capture_options = ["--capture-spec", output / "capture-spec.json"]
    report, timings["collect"] = run_probe(probe, environment, "collect", model, [base], output / "collect",
                                           layout + capture_options, args.timeout)
    device = report.get("device")
    arrays = {(c["key"], c["layer"]): np.load(output / "collect" / c["file"]) for c in report["captures"]}
    attention = wanted["k"]
    if sorted(arrays) != sorted((key, il) for key, layers in wanted.items() for il in layers):
        raise ValueError("collect did not capture k/v (and indexer keys) of exactly the layers of the spec")
    trunk = trunk_layer_types(spec)
    if report["n_layer"] != len(trunk) or [x["attention"] for x in report["layers"]] != trunk:
        raise ValueError("probe layer table differs from kvstate.ModelSpec")
    for key, il in arrays:
        expected = captured_shape(spec, key, il)
        if arrays[key, il].shape != (n,) + expected:
            raise ValueError(f"{key}-{il} has shape {arrays[key, il].shape}, spec says {(n,) + expected}")

    save = dict(base, save_at=[seed])
    report, timings["save-state"] = run_probe(probe, environment, "save-state", model, [save], output / "save-state",
                                              layout, args.timeout)
    entry = report["documents"][0]["states"][0]
    state_path = output / "save-state" / entry["file"]
    data = state_path.read_bytes()
    parsed, state_checks = check_state(spec, data, token_ids, arrays, args.cache_type, model)
    state_checks.update(bytes=len(data), pos_min=entry["pos_min"], pos_max=entry["pos_max"])
    rebuilt_path = output / "rebuilt.state"
    kvstate.write_file(rebuilt_path, rebuild(spec, parsed, arrays, args.cache_type))

    sha = hashlib.sha256(np.asarray(token_ids, dtype="<i4").tobytes()).hexdigest()
    score = dict(base, expected_token_sha256=sha, score_ranges=ranges,
                 conditions=[{"name": "native", "state_file": None},
                             {"name": "seed", "state_file": str(state_path)},
                             {"name": "rebuilt", "state_file": str(rebuilt_path)}])
    results, timings["seed-score"] = run_probe(probe, environment, "seed-score", model, [score],
                                               output / "seed-score", layout + ["--max-scored", args.max_scored],
                                               args.timeout)
    document = results["documents"][0]
    scores = {c["name"]: np.load(output / "seed-score" / "scores" / document_id / f"{c['name']}.npy")
              for c in document["conditions"]}
    conditions = {}
    for item in document["conditions"]:
        name = item["name"]
        conditions[name] = {key: item[key] for key in ("nll_mean", "kl_mean", "top1_agree", "seed_tokens",
                                                         "decoded_tokens")}
        conditions[name]["kl_max"] = float(scores[name][:, 1].max())
        conditions[name]["nll_max_abs_diff_vs_native"] = float(np.abs(scores[name][:, 0] - scores["native"][:, 0]).max())
        conditions[name]["scores_identical_to_native"] = bool(np.array_equal(scores[name], scores["native"]))
        if name != "native":
            conditions[name]["restored_pos_min"] = item["restored_pos_min"]
            conditions[name]["restored_pos_max"] = item["restored_pos_max"]

    summary = {
        "model": str(model), "architecture": spec.architecture,
        "spec": {"n_layer": len(trunk), "attention_layers": attention,
                 "memory": spec.memory, "n_head_kv": sorted({spec.n_head_kv[il] for il in attention}),
                 "head_dim_k": sorted({spec.head_dim_k[il] for il in attention}),
                 "kv_lora": sorted({spec.kv_lora[il] for il in attention}) if spec.kv_lora else None,
                 "indexer_layers": wanted.get("ik"),
                 "rope_type": spec.rope_type, "n_rot": spec.n_rot, "rope_freq_base": spec.rope_freq_base,
                 "rope_sections": list(spec.rope_sections)},
        "probe": str(probe), "device": device,
        "settings": {"cache_type": args.cache_type, "flash_attn": args.flash_attn, "batch": args.batch,
                     "ctx": ctx, "threads": args.threads, "max_scored": args.max_scored},
        "document": {"document_id": document_id, "conversation_id": record.get("conversation_id"),
                     "n_tokens": n, "first_scored_position": first, "seed_tokens": seed,
                     "n_scored": document["n_scored"]},
        "tokenize_matches_collect": bool(np.array_equal(token_ids, np.load(output / "collect" / "token_id.npy"))),
        "tokenize_matches_wrapper": bool(np.array_equal(token_ids, ids)),
        "state": state_checks,
        "seed_score": conditions,
        "seconds": {key: round(value, 2) for key, value in timings.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "architecture": spec.architecture, "document": summary["document"],
                      "state_round_trip": state_checks["round_trip_bytes_equal"], "worst": state_checks["worst"],
                      "seed_kl_mean": conditions["seed"]["kl_mean"], "seed_identical": conditions["seed"]["scores_identical_to_native"],
                      "rebuilt_kl_mean": conditions["rebuilt"]["kl_mean"], "rebuilt_kl_max": conditions["rebuilt"]["kl_max"],
                      "seconds": summary["seconds"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, RuntimeError, kvstate.StateFormatError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc)) from exc
