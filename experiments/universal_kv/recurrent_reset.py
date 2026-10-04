"""How much does a hybrid model lose if its recurrent state is reset at the handoff point?

For a hybrid target (Qwen3.5/3.8-family gated DeltaNet), the per-token attention cache can be
translated, but the recurrent state (conv + delta-rule matrices) summarizes the whole history
and has no per-token form. Strategy R0 resets it to zero at T0 and lets the native tail
[T0, P) rebuild it. This measures R0 alone: the attention cache stays native, so the loss is
the price of the reset.

Per document: save-state at T0 = P - tail (rounded down to the batch grid) with kv_probe,
write a copy with every recurrent row zeroed, seed-score both against a native full prefill.
"""

import os

if __name__ == "__main__":
    for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, "8")

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "linear_transfer"))

from collect import verify_runtime  # noqa: E402
import collect_v2  # noqa: E402
import handoff  # noqa: E402
import kvstate  # noqa: E402
from live_handoff import RunData  # noqa: E402


def zero_recurrent(state_file):
    parts = []
    for part in state_file.state.parts:
        if isinstance(part, kvstate.RecurrentState):
            def zeroed(rows):
                return {layer: kvstate.OpaqueRows(**{**vars(r), "data": bytes(len(r.data))}) for layer, r in rows.items()}
            part = kvstate.RecurrentState(part.name, part.layers, part.positions, part.seq_ids, part.n_layer,
                                          zeroed(part.r), zeroed(part.s))
        parts.append(part)
    state = kvstate.SequenceState(state_file.state.spec, state_file.state.flags, parts)
    return kvstate.StateFile(state_file.token_words, state)


def probe(args, artifacts, mode, records, output, extra=()):
    path = output.parent / (output.name + ".input.jsonl")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    command = [str(args.probe), "--mode", mode, "--model", str(args.model), "--input", str(path), "--output",
               str(output), "--ctx", str(args.ctx), "--batch", "512", "--threads", "16", "--cache-type", "f16",
               "--flash-attn", "on", "--device", "vulkan", *extra]
    with open(output.parent / (output.name + ".log"), "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                       env=collect_v2.probe_environment(artifacts))
    return json.loads((output / ("report.json" if mode == "save-state" else "results.json")).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="append", required=True, metavar="RUN_DIR:BUCKET")
    parser.add_argument("--per-bucket", type=int, default=3)
    parser.add_argument("--tails", default="2048,4096")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/srv/models/qwen3.8-27b-q8/Qwen3.8-27B-Q8_0.gguf"))
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    _, _, artifacts = verify_runtime(args.probe.resolve())
    spec = kvstate.ModelSpec.from_gguf(args.model)
    tails = [int(t) for t in args.tails.split(",")]
    documents = []
    for item in args.select:
        path, _, bucket = item.rpartition(":")
        run = RunData(path)
        docs = sorted([d for d in run.plan["documents"] if d["bucket"] == bucket and d.get("role") == "primary"],
                      key=lambda d: d["prefix"])
        picks = np.unique(np.round(np.linspace(0, len(docs) - 1, min(args.per_bucket, len(docs)))).astype(int))
        documents += [(run, docs[i]) for i in picks]
    records, saves = [], []
    for run, doc in documents:
        record = run.records("source")[doc["document_id"]]
        P = doc["qwen_prefix"]
        t0s = sorted({max(512, (P - tail) // 512 * 512) for tail in tails})
        records.append({"document_id": doc["document_id"], "text": record["text"], "save_at": t0s,
                        "expected_token_sha256": doc["qwen_token_sha256"]})
        saves.append((run, doc, record, t0s))
    probe(args, artifacts, "save-state", records, args.output / "states")
    score_inputs = []
    for run, doc, record, t0s in saves:
        conditions = [{"name": "native", "state_file": None}]
        for t0 in t0s:
            native = next((args.output / "states").rglob(f"{doc['document_id']}/{t0}.state"))
            reset = native.with_name(f"{t0}.reset.state")
            kvstate.write_new(reset, zero_recurrent(kvstate.read_file(native, spec)).to_bytes())
            conditions += [{"name": f"native_{t0}", "state_file": str(native)},
                           {"name": f"reset_{t0}", "state_file": str(reset)}]
        start = handoff.final_message_start(record)
        score_inputs.append({"document_id": doc["document_id"], "text": record["text"],
                             "expected_token_sha256": doc["qwen_token_sha256"],
                             "score_ranges": [[start, len(record["text"].encode("utf-8"))]], "conditions": conditions})
    probe(args, artifacts, "seed-score", score_inputs, args.output / "scores", extra=("--max-scored", "256"))
    rows = []
    for (run, doc, record, t0s) in saves:
        base = args.output / "scores" / "scores" / doc["document_id"]
        native = np.load(base / "native.npy").astype(np.float64)
        row = {"document_id": doc["document_id"], "source": doc["source"], "prefix": doc["qwen_prefix"],
               "n_scored": len(native)}
        for t0 in t0s:
            for name in (f"native_{t0}", f"reset_{t0}"):
                scores = np.load(base / f"{name}.npy").astype(np.float64)
                row[name] = {"nll_delta": float(np.mean(scores[:, 0] - native[:, 0])), "kl": float(scores[:, 1].mean()),
                             "top1": float(scores[:, 2].mean())}
            row[f"tail_{t0}"] = doc["qwen_prefix"] - t0
        rows.append(row)
        print(json.dumps({k: (v if not isinstance(v, dict) else round(v["nll_delta"], 4)) for k, v in row.items()}))
    (args.output / "report.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
