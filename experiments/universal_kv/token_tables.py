"""Per-model token tables for the hub: mean native K (pre-RoPE) / V per token id over the rows the
hub does not cover (template, special and unmatched tokens) in the train split.

The translator fills those cells from the target's table (`fill="table"`) instead of one mean
row, so every template token gets its own typical values. Same file format as
linear_transfer/handoff.py (TokenTable reads it).

    token_tables.py --model NAME=CAPTURES@RECORDS.jsonl [...] --output DIR [--only NAME ...]

The hub spans are those of every --model given (as the hub was fitted, or onboarded with its
pivots); --only builds the tables of the named models alone (an onboarded model: give it and its
pivots, and --only it).
"""

import os

if __name__ == "__main__":
    for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, "16")

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

import hub
from fit_hub import parse_model

sys.path.insert(0, str(hub.HERE.parent / "linear_transfer"))
import handoff  # noqa: E402


def uncovered_rows(view, spans, split="train"):
    """Global capture rows of `split` documents that are not hub spans for this model."""
    chosen = [d for d, s in zip(spans.document_ids, spans.splits) if s == split]
    rows = np.concatenate([np.arange(view.captures.rows(d).start, view.captures.rows(d).stop) for d in chosen])
    return np.setdiff1d(rows, spans.rows[view.name], assume_unique=True)


def build(view, rows, chunk_rows=16384, log=print):
    token_ids, inverse = np.unique(view.captures.token_id[rows], return_inverse=True)
    inverse = inverse.reshape(-1)
    count = np.bincount(inverse, minlength=len(token_ids)).astype(np.int64)
    arrays, shapes = {"token_id": token_ids.astype(np.int32), "count": count}, {}
    started = time.perf_counter()
    for layer in view.layers("k"):
        for kind in hub.KINDS:
            sums = total = shape = None
            for start in range(0, len(rows), chunk_rows):
                block = view.captures.gather(kind, layer, rows[start:start + chunk_rows])
                if sums is None:
                    shape = block.shape[1:]
                    sums = np.zeros((len(token_ids), int(np.prod(shape))))
                    total = np.zeros(int(np.prod(shape)))
                flat = block.reshape(len(block), -1).astype(np.float64)
                local = inverse[start:start + chunk_rows]
                order = np.argsort(local, kind="stable")
                groups, first = np.unique(local[order], return_index=True)
                sums[groups] += np.add.reduceat(flat[order], first, axis=0)
                total += flat.sum(axis=0)
            arrays[f"{kind}-{layer}"] = (sums / count[:, None]).astype(np.float32).reshape((-1,) + shape)
            arrays[f"mean-{kind}-{layer}"] = (total / len(rows)).astype(np.float32).reshape(shape)
            shapes[kind] = list(shape)
    log(json.dumps({"model": view.name, "rows": int(len(rows)), "unique_tokens": int(len(token_ids)),
                    "seconds": round(time.perf_counter() - started, 1)}))
    metadata = {"format_version": handoff.TABLE_FORMAT, "purpose": handoff.TABLE_PURPOSE, "split": "train",
                "architecture": view.architecture, "layers": list(view.layers("k")), "head_shape": shapes,
                "rows": int(len(rows)), "unique_tokens": int(len(token_ids)),
                "definition": "mean native K (pre-RoPE) and V per token id over train rows that are not hub "
                              "spans (template/special, unmatched, straddling tokens)"}
    return {"arrays": arrays, "metadata": metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=parse_model, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True, help="directory for <NAME>.npz tables")
    parser.add_argument("--only", action="append", metavar="NAME", help="build only this model's table "
                        "(repeatable; default: every model)")
    args = parser.parse_args(argv)
    names = [name for name, _, _ in args.model]
    if args.only and set(args.only) - set(names):
        raise SystemExit(f"--only names a model without --model: {sorted(set(args.only) - set(names))}")
    args.output.mkdir(parents=True, exist_ok=True)
    views = [hub.CaptureView(name, captures, records) for name, captures, records in args.model]
    spans = hub.joint_spans(views)
    for view in views:
        if args.only and view.name not in args.only:
            continue
        handoff.write_token_table(args.output / f"{view.name}.npz", build(view, uncovered_rows(view, spans)),
                                  {"hub_models": [v.name for v in views]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
