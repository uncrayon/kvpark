"""Score extra handoff conditions on documents of a finished run_long.py run (no new saves).

Builds states from the run's saved Qwen states and native Muse states at T0, then
seed-scores them with kv_probe exactly like run_long's score stage (same probe settings,
native full prefill as the reference in the same invocation). Conditions:

  sanity        the native Muse state at T0 (check: KL 0)
  table_fill    mapped content, table fill (the run's headline, re-scored here)
  oracle_fill   mapped content, Muse's native values in every fill cell
                (template, special, unmatched, straddling)
  prefix        native values for the system message and the rendered tool block
                (everything before message 1), the rest as table_fill
  tool_obs      native values for every tool-result message's content, the rest as
                table_fill
  tool_all      prefix + tool_obs native, the rest as table_fill

Writes <output>/states (built, deleted after scoring unless --keep-states),
<output>/raw (probe output), <output>/report.json and a printed table.
"""

import os

if __name__ == "__main__":
    for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, "8")

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attention_eval  # noqa: E402
from collect import verify_runtime  # noqa: E402
import collect_v2  # noqa: E402
import handoff  # noqa: E402
import kvstate  # noqa: E402
from live_handoff import RunData  # noqa: E402

CONDITIONS = ("sanity", "table_fill", "oracle_fill", "prefix", "tool_obs", "tool_all")


def segment_mask(record, tokens, wanted):
    """bool per Muse token: tokens whose bytes end inside a segment of a wanted message index."""
    ends = tokens["byte_end"]
    starts = np.concatenate([[0], ends[:-1]])
    mask = np.zeros(len(ends), dtype=bool)
    for (index, start, end), key in zip(record["segments"], record["segment_keys"]):
        message = int(key.split(":", 1)[0])
        if message in wanted:
            mask |= (ends > start) & (starts < end)
    return mask


def prefix_mask(record, tokens):
    """bool per Muse token: everything before the first segment of message 1 (system + tool block)."""
    first = min(start for (_, start, _), key in zip(record["segments"], record["segment_keys"])
                if key.split(":", 1)[0] == "1")
    mask = np.zeros(len(tokens["byte_end"]), dtype=bool)
    mask[:int(np.searchsorted(tokens["byte_end"], first, side="right"))] = True
    return mask


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True, help="conversations.jsonl of the run (message roles)")
    parser.add_argument("--source", action="append", required=True, help="plan document source to include")
    parser.add_argument("--tail", type=int, default=2048)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--mapper", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--muse-model", type=Path,
                        default=Path("/srv/models/muse-glimmer-30b-q8/Muse-Glimmer-30B-Q8_0.gguf"))
    parser.add_argument("--qwen-model", type=Path, default=Path("/srv/models/qwen3.8-27b-q8/Qwen3.8-27B-Q8_0.gguf"))
    parser.add_argument("--device", default="vulkan")
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-scored", type=int, default=256)
    parser.add_argument("--build-only", action="store_true", help="build the states and stop (no GPU)")
    parser.add_argument("--keep-states", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    conditions = [c for c in args.conditions.split(",") if c]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown conditions: {sorted(unknown)}")
    run = RunData(args.run)
    roles = {}
    with open(args.corpus, encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            roles[item["id"]] = [message["role"] for message in item["messages"]]
    documents = [d for d in run.plan["documents"] if d["source"] in args.source and d.get("role") == "primary"]
    if not documents:
        raise SystemExit("no documents match")
    muse_spec = kvstate.ModelSpec.from_gguf(args.muse_model)
    qwen_spec = kvstate.ModelSpec.from_gguf(args.qwen_model)
    states_dir = args.output / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    mapper = attention_eval.Mapper(args.mapper)
    table = handoff.TokenTable(run.path / "long" / "token-table.npz")
    inputs, built = [], {}
    for doc in documents:
        document_id = doc["document_id"]
        t0 = next(c["t0"] for c in doc["conditions"] if c["name"] == f"sanity_{args.tail}")
        native_path = run.state("target", document_id, t0)
        record_q, record_m = run.records("source")[document_id], run.records("target")[document_id]
        q_tokens, m_tokens = run.tokens("source", document_id), run.tokens("target", document_id)
        message_roles = roles[doc["conversation_id"]]
        tool_messages = {i for i, role in enumerate(message_roles) if role == "tool"}
        masks = {"prefix": prefix_mask(record_m, m_tokens),
                 "tool_obs": segment_mask(record_m, m_tokens, tool_messages)}
        masks["tool_all"] = masks["prefix"] | masks["tool_obs"]
        out = states_dir / document_id
        out.mkdir(exist_ok=True)
        entries = [{"name": "native", "state_file": None}]
        info = {"t0": t0, "prefix_tokens": int(masks["prefix"][:t0].sum()),
                "tool_obs_tokens": int(masks["tool_obs"][:t0].sum()), "tool_messages": len(tool_messages),
                "conditions": {}}
        qwen_state = native_state = None
        for name in conditions:
            path = native_path if name == "sanity" else out / f"{name}.state"
            if name != "sanity" and not path.exists():
                if qwen_state is None:
                    qwen_state = kvstate.read_file(run.state("source", document_id, doc["qwen_prefix"]), qwen_spec)
                    native_state = kvstate.read_file(native_path, muse_spec)
                options = {"fill_policy": "oracle" if name == "oracle_fill" else "table"}
                if name in masks:
                    options["native_mask"] = masks[name]
                started = time.perf_counter()
                result = handoff.build_handoff(qwen_state, record_q, q_tokens, record_m, m_tokens, mapper, t0=t0,
                                               muse_spec=muse_spec, content_policy="mapped", table=table,
                                               oracle_state=native_state, **options)
                kvstate.write_new(path, result.state.to_bytes())
                info["conditions"][name] = {"seconds": round(time.perf_counter() - started, 2),
                                            "native_cells": result.report["native_cells"],
                                            "sections": result.report["sections"]}
                print(json.dumps({"built": name, "document": document_id[:12], "t0": t0,
                                  "native_cells": result.report["native_cells"],
                                  "seconds": info["conditions"][name]["seconds"]}), flush=True)
            entries.append({"name": name, "state_file": str(path.resolve())})
        built[document_id] = info
        inputs.append({"document_id": document_id, "text": record_m["text"],
                       "expected_token_sha256": doc["muse_token_sha256"],
                       "score_ranges": [[doc["final_start_byte"], doc["text_bytes"]]], "conditions": entries})
    built_path = args.output / "built.json"
    if built_path.exists():  # keep the build details of states built by an earlier invocation
        earlier = json.loads(built_path.read_text(encoding="utf-8"))
        for document_id, info in built.items():
            info["conditions"] = {**earlier.get(document_id, {}).get("conditions", {}), **info["conditions"]}
    built_path.write_text(json.dumps(built, indent=1) + "\n", encoding="utf-8")
    if args.build_only:
        return 0

    raw = args.output / "raw"
    if raw.exists():
        raise SystemExit(f"{raw} exists; remove it to re-score")
    input_path = args.output / "score.input.jsonl"
    input_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in inputs), encoding="utf-8")
    _, _, artifacts = verify_runtime(args.probe.resolve())
    command = [str(args.probe.resolve()), "--mode", "seed-score", "--model", str(args.muse_model), "--input",
               str(input_path), "--output", str(raw), "--ctx", str(args.ctx), "--batch", str(args.batch),
               "--threads", str(args.threads), "--cache-type", "f16", "--flash-attn", "on",
               "--max-scored", str(args.max_scored)]
    if args.device != "cpu":
        command += ["--device", args.device]
    started = time.monotonic()
    with open(args.output / "score.log", "w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                       env=collect_v2.probe_environment(artifacts))
    seconds = time.monotonic() - started
    results = json.loads((raw / "results.json").read_text(encoding="utf-8"))
    by_doc = {doc["document_id"]: doc for doc in documents}
    rows = []
    for item in results["documents"]:
        document_id = item["document_id"]
        native = np.load(raw / "scores" / document_id / "native.npy").astype(np.float64)
        row = {"document_id": document_id, "bucket": by_doc[document_id]["bucket"],
               "prefix": by_doc[document_id]["prefix"], **{k: built[document_id][k] for k in
                                                           ("t0", "prefix_tokens", "tool_obs_tokens")},
               "native_nll": float(native[:, 0].mean()), "conditions": {}}
        for name in conditions:
            scores = np.load(raw / "scores" / document_id / f"{name}.npy").astype(np.float64)
            row["conditions"][name] = {"nll_delta": float(np.mean(scores[:, 0] - native[:, 0])),
                                       "kl": float(scores[:, 1].mean()), "top1": float(scores[:, 2].mean())}
        rows.append(row)
    report = {"run": str(args.run.resolve()), "sources": args.source, "tail": args.tail, "conditions": conditions,
              "mapper": str(args.mapper.resolve()), "probe": str(args.probe.resolve()), "seconds": seconds,
              "device": results.get("device"), "documents": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"{'doc':>12} {'P':>6} {'T0':>6} {'pre':>5} {'tool':>6}  " + "  ".join(f"{c:>11}" for c in conditions))
    for row in rows:
        print(f"{row['document_id'][:12]:>12} {row['prefix']:>6} {row['t0']:>6} {row['prefix_tokens']:>5} "
              f"{row['tool_obs_tokens']:>6}  " + "  ".join(f"{row['conditions'][c]['nll_delta']:>+11.3f}"
                                                          for c in conditions))
    means = {c: float(np.mean([row["conditions"][c]["nll_delta"] for row in rows])) for c in conditions}
    print(f"{'mean':>12} {'':>6} {'':>6} {'':>5} {'':>6}  " + "  ".join(f"{means[c]:>+11.3f}" for c in conditions))
    if not args.keep_states:
        shutil.rmtree(states_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
