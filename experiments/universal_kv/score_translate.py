"""Score hub translations end to end: seed-score delta NLL on the target model.

Uses the documents of a run_long.py run (rendered records, tokens, saved source states at the
source prefix, native target states at T0 as layout templates and sanity checks). Per document
and tail W, conditions:

  sanity      the native target state at T0 (KL 0 check)
  hub         translate.translate(): source state -> hub -> target, fill = --fill
  direct      handoff.build_handoff() with a pairwise fit_layers mapper (--direct-mapper), the
              baseline the hub must match (source qwen35 -> target muse-glimmer only)

A hybrid target's recurrent cell at T0 - 1 starts from --recurrent (recurrent_init: zero, the
default, or mean with --recurrent-mean and --recurrent-group, by default the pooled mean).

Output: <output>/report.json and a printed table (token-weighted delta NLL vs native).
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

import hub as hub_module
import recurrent_init
import translate

sys.path.insert(0, str(hub_module.HERE.parent / "linear_transfer"))
import attention_eval  # noqa: E402
from collect import verify_runtime  # noqa: E402
import collect_v2  # noqa: E402
import handoff  # noqa: E402
import kvstate  # noqa: E402
from live_handoff import RunData  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--select", action="append", required=True, metavar="RUN_DIR:BUCKET")
    parser.add_argument("--per-bucket", type=int, default=10)
    parser.add_argument("--source-filter", action="append", default=None, help="plan document sources to keep")
    parser.add_argument("--tail", type=int, default=2048)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--fill", choices=("mean", "table"), default="table")
    parser.add_argument("--table", type=Path, help="target token table (default: the run's long/token-table.npz)")
    parser.add_argument("--direct-mapper", type=Path)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--keep-states", action="store_true")
    parser.add_argument("--recurrent", default="zero", help="hybrid targets: the recurrent cell at T0 - 1 "
                        "(recurrent_init strategies: zero, mean, mean_conv, mean_s<x>)")
    parser.add_argument("--recurrent-mean", type=Path, help="the target's recurrent mean npz, for strategies "
                        "that read it")
    parser.add_argument("--recurrent-group", default=recurrent_init.ALL, help="the mean's group (default: all)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    hubs, manifest = hub_module.load_hub(args.hub)
    hubs["indexer"] = hub_module.load_indexer_ridges(args.hub)       # DSA targets: their indexer keys
    source_spec = kvstate.ModelSpec.from_gguf(args.source_model)
    target_spec = kvstate.ModelSpec.from_gguf(args.target_model)
    documents = []
    for item in args.select:
        path, _, bucket = item.rpartition(":")
        run = RunData(path)
        docs = [d for d in run.plan["documents"] if d["bucket"] == bucket and d.get("role") == "primary"
                and (args.source_filter is None or d["source"] in args.source_filter)]
        docs.sort(key=lambda d: d["prefix"])
        picks = np.unique(np.round(np.linspace(0, len(docs) - 1, min(args.per_bucket, len(docs)))).astype(int))
        documents += [(run, docs[i]) for i in picks]
    mapper = attention_eval.Mapper(args.direct_mapper) if args.direct_mapper else None
    tables = {}
    states_dir = args.output / "states"
    inputs, built = [], {}
    recurrent = recurrent_init.translate_options(args.recurrent, args.recurrent_mean, args.recurrent_group)
    for run, doc in documents:
        document_id = doc["document_id"]
        t0 = next(c["t0"] for c in doc["conditions"] if c["name"] == f"sanity_{args.tail}")
        native_path = run.state("target", document_id, t0)
        template = kvstate.read_file(native_path, target_spec)
        # the run's states may come from either pinned runtime: what follows uses their own reading
        # (translate adopts the template's and the source state's; the direct mapper and the window
        # take the template's)
        reading = template.state.spec
        source = translate.Side(args.source_name, source_spec, run.records("source")[document_id],
                                run.tokens("source", document_id), gguf=args.source_model)
        target = translate.Side(args.target_name, target_spec, run.records("target")[document_id],
                                run.tokens("target", document_id), gguf=args.target_model)
        source_state = kvstate.read_file(run.state("source", document_id, doc["qwen_prefix"]), source_spec)
        table_path = args.table or run.path / "long" / "token-table.npz"
        if args.fill == "table" and table_path not in tables:
            tables[table_path] = handoff.TokenTable(table_path)
        out = states_dir / document_id
        out.mkdir(parents=True)
        conditions = [{"name": "native", "state_file": None}, {"name": "sanity", "state_file": str(native_path)}]
        info = {"t0": t0}
        state, report = translate.translate(source_state, source, target, hubs, t0=t0, template=template,
                                            fill=args.fill, table=tables.get(table_path),
                                            swa_cells=reading.n_swa or None, **recurrent)
        kvstate.write_new(out / "hub.state", state.to_bytes())
        conditions.append({"name": "hub", "state_file": str(out / "hub.state")})
        info["hub"] = report
        if mapper is not None:
            started = time.perf_counter()
            result = handoff.build_handoff(source_state, source.record, source.tokens, target.record, target.tokens,
                                           mapper, t0=t0, muse_spec=reading, content_policy="mapped",
                                           fill_policy=args.fill, table=tables.get(table_path))
            kvstate.write_new(out / "direct.state", result.state.to_bytes())
            conditions.append({"name": "direct", "state_file": str(out / "direct.state")})
            info["direct"] = {"seconds": round(time.perf_counter() - started, 3)}
        built[document_id] = info
        print(json.dumps({"built": document_id[:12], "t0": t0, "hub_seconds": info["hub"]["seconds"],
                          "mapped": info["hub"]["mapped_cells"], "fill": info["hub"]["fill_cells"]}), flush=True)
        inputs.append({"document_id": document_id, "text": target.record["text"],
                       "expected_token_sha256": doc["muse_token_sha256"],
                       "score_ranges": [[doc["final_start_byte"], doc["text_bytes"]]], "conditions": conditions})
    input_path = args.output / "score.input.jsonl"
    input_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in inputs), encoding="utf-8")
    _, _, artifacts = verify_runtime(args.probe.resolve())
    raw = args.output / "raw"
    command = [str(args.probe.resolve()), "--mode", "seed-score", "--model", str(args.target_model), "--input",
               str(input_path), "--output", str(raw), "--ctx", str(args.ctx), "--batch", "512", "--threads", "16",
               "--cache-type", "f16", "--flash-attn", "on", "--max-scored", "256", "--device", "vulkan"]
    with open(args.output / "score.log", "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                       env=collect_v2.probe_environment(artifacts))
    names = [c["name"] for c in inputs[0]["conditions"][1:]]
    rows, sums, counts = [], {n: [] for n in names}, []
    for run, doc in documents:
        base = raw / "scores" / doc["document_id"]
        native = np.load(base / "native.npy").astype(np.float64)
        row = {"document_id": doc["document_id"], "source": doc["source"], "bucket": doc["bucket"],
               "prefix": doc["prefix"], "n_scored": len(native)}
        for name in names:
            scores = np.load(base / f"{name}.npy").astype(np.float64)
            row[name] = {"nll_delta": float(np.mean(scores[:, 0] - native[:, 0])), "kl": float(scores[:, 1].mean()),
                         "top1": float(scores[:, 2].mean())}
            sums[name].append(float(np.sum(scores[:, 0] - native[:, 0])))
        counts.append(len(native))
        rows.append(row)
    counts = np.asarray(counts, dtype=np.float64)
    rng = np.random.default_rng(0)
    index = rng.integers(0, len(counts), (10000, len(counts)))
    summary = {}
    for name in names:
        s = np.asarray(sums[name])
        boot = s[index].sum(axis=1) / counts[index].sum(axis=1)
        summary[name] = {"nll_delta": float(s.sum() / counts.sum()),
                         "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}
    if "direct" in names:
        d = np.asarray(sums["hub"]) - np.asarray(sums["direct"])
        boot = d[index].sum(axis=1) / counts[index].sum(axis=1)
        summary["hub_minus_direct"] = {"nll_delta": float(d.sum() / counts.sum()),
                                       "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}
    report = {"hub": str(args.hub.resolve()), "source": args.source_name, "target": args.target_name,
              "tail": args.tail, "fill": args.fill, "recurrent": args.recurrent, "documents": rows, "built": built,
              "summary": summary}
    (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['document_id'][:12]} {row['source'][:14]:>14} {row['prefix']:>6} "
              + " ".join(f"{n}={row[n]['nll_delta']:+.3f}" for n in names))
    print(json.dumps(summary, indent=1))
    if not args.keep_states:
        shutil.rmtree(states_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
