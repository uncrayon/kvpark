"""Fit a universal KV hub on captured models, evaluate every ordered pair, optionally compare a direct mapper.

  fit_hub.py --model qwen38=CAP1,CAP2@rendered.jsonl --model muse=CAP3,CAP4@rendered.jsonl \
      --output DIR [--hub-dim 1024] [--view-dim 2048] [--layer-rank 512] [--fit-rows 60000] \
      [--direct qwen38:muse=MAPPER_DIR] [--indexer-ridge NAME] [--heads NAME]

Models of any cache family mix: the captures hold each attention layer's pre-RoPE K and V (an
MLA model's K is the [c_kv | k_pe] row, its V the c_kv view). --indexer-ridge NAME fits a DSA
model's indexer ridge (hub.fit_indexer_ridge: its captured indexer keys, kv_probe key "ik", from
its K rows) into the bundle; with --base the base bundle's ridges are kept. --heads NAME fits a
DeepSeek V4 model's head decoders (hub.fit_head: its captured compressor inputs, keys csa, hca and
lid, out of the K latent of the other models) into the bundle; with --base the base bundle's
heads are kept.
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


def parse_model(text):
    name, _, rest = text.partition("=")
    captures, _, records = rest.rpartition("@")
    if not name or not captures or not records:
        raise argparse.ArgumentTypeError("--model NAME=CAPTURE[,CAPTURE...]@RECORDS.jsonl")
    return name, [Path(p) for p in captures.split(",")], Path(records)


def direct_r2(mapper_dir, source, target, spans, kind, split="test", chunk=4096):
    """Variance-weighted R^2 of a fit_layers mapper on the hub's test spans (same rows as the hub)."""
    sys.path.insert(0, str(hub.HERE.parent / "linear_transfer"))
    import attention_eval
    mapper = attention_eval.Mapper(mapper_dir)
    selected = spans.select(split)
    layers = target.layers(kind)
    sse, y_sum, y_sq, n = np.zeros(len(layers)), [None] * len(layers), np.zeros(len(layers)), 0
    for part in hub.chunks(len(selected), chunk):
        rows = selected[part]
        source_rows = {layer: source.captures.gather(kind, layer, spans.rows[source.name][rows])
                       for layer in source.layers(kind)}
        for index, layer in enumerate(layers):
            _, wanted = mapper.source_layers(kind, layer)
            prediction = mapper.predict(kind, layer, {l: source_rows[l] for l in wanted}, target.shape(kind, layer))
            y = target.rows(kind, layer, spans.rows[target.name][rows])
            sse[index] += float(np.sum(np.square(y - prediction.reshape(len(rows), -1)), dtype=np.float64))
            y_sum[index] = y.sum(axis=0, dtype=np.float64) + (0 if y_sum[index] is None else y_sum[index])
            y_sq[index] += float(np.sum(np.square(y.astype(np.float64))))
        n += len(rows)
        mapper.clear()
    total = np.asarray([y_sq[i] - np.sum(np.square(y_sum[i])) / n for i in range(len(layers))])
    return {"r2": float(1 - sse.sum() / total.sum()), "rows": int(n),
            "per_layer": [float(1 - s / t) for s, t in zip(sse, total)]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=parse_model, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hub-dim", type=int, default=1024)
    parser.add_argument("--view-dim", type=int, default=2048)
    parser.add_argument("--layer-rank", type=int, default=512)
    parser.add_argument("--fit-rows", type=int, default=60000, help="train spans used for stages 1-2")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--direct", action="append", default=[], metavar="SOURCE:TARGET=MAPPER_DIR")
    parser.add_argument("--base", type=Path, help="onboard: an existing hub bundle; --model names the new model "
                        "(first) and pivots already in the hub (the rest)")
    parser.add_argument("--source-global", action="append", default=[], metavar="NAME",
                        help="as a source, this model only provides its full-attention layers (sliding-window models)")
    parser.add_argument("--indexer-ridge", action="append", default=[], metavar="NAME",
                        help="fit this DSA model's indexer ridge from its captured indexer keys (key ik)")
    parser.add_argument("--heads", action="append", default=[], metavar="NAME",
                        help="fit this DeepSeek V4 model's head decoders from its captured compressor inputs "
                        "(keys csa, hca, lid)")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"output exists: {args.output}")
    started = time.perf_counter()
    log_path = args.output.with_name(args.output.name + ".log.jsonl")

    def log(line):
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    views = [hub.CaptureView(name, captures, records) for name, captures, records in args.model]
    spans = hub.joint_spans(views)
    log(json.dumps({"stage": "spans", "documents": len(spans.document_ids), "rejected": len(spans.rejected),
                    "spans": {split: int(len(spans.select(split))) for split in ("train", "dev", "test")},
                    "tokens": {v.name: int(v.captures.rows_total) for v in views}}))
    by_name = {v.name: v for v in views}
    source_layers = {}
    for name in args.source_global:
        types = by_name[name].captures.captures[0].layer_types
        source_layers[name] = {kind: [l for l in by_name[name].layers(kind) if types[l] == "full"] for kind in hub.KINDS}
    hubs, evaluation, base_models, ridges = {}, {}, None, {}
    if args.base:
        hubs, base_manifest = hub.load_hub(args.base)
        base_models = base_manifest["models"]
        ridges = hub.load_indexer_ridges(args.base)
    for kind in hub.KINDS:
        if args.base:
            new, pivots = views[0], views[1:]
            hub.add_model_kind(hubs[kind], new, pivots, spans, kind, layer_rank=args.layer_rank,
                               view_dim=args.view_dim, fit_rows=args.fit_rows, chunk=args.chunk, log=log,
                               source_layers=source_layers.get(new.name, {}).get(kind))
        else:
            hubs[kind] = hub.fit_hub_kind(views, spans, kind, layer_rank=args.layer_rank, view_dim=args.view_dim,
                                          hub_dim=args.hub_dim, fit_rows=args.fit_rows, chunk=args.chunk, log=log,
                                          source_layers={n: layers[kind] for n, layers in source_layers.items()})
        evaluation[kind] = {f"{s}->{t}": item for (s, t), item in
                            hub.evaluate(hubs[kind], views, spans, kind, chunk=args.chunk).items()}
        log(json.dumps({"stage": "test", "kind": kind,
                        "r2": {pair: round(item["r2"], 4) for pair, item in evaluation[kind].items()}}))
        for item in args.direct:
            pair, _, mapper_dir = item.partition("=")
            source, _, target = pair.partition(":")
            result = direct_r2(Path(mapper_dir), by_name[source], by_name[target], spans, kind, chunk=args.chunk)
            evaluation[kind][f"direct {source}->{target}"] = result
            log(json.dumps({"stage": "direct", "kind": kind, "pair": pair, "r2": round(result["r2"], 4)}))
    for name in args.indexer_ridge:
        ridges[name] = hub.fit_indexer_ridge(by_name[name], fit_rows=args.fit_rows, chunk=args.chunk, log=log)
        evaluation.setdefault("indexer", {})[name] = {str(layer): item for layer, item in ridges[name].fit.items()}
    heads = []
    for name in args.heads:
        pivots = [view for view in views if view.name != name]
        for kind in hub.HEAD_KINDS:
            if by_name[name].layers(kind):
                heads.append(hub.fit_head(hubs[hub.HEAD_BASE], pivots, by_name[name], spans, kind, chunk=args.chunk,
                                          log=log))
                evaluation.setdefault("heads", {})[f"{name}.{kind}"] = heads[-1].alphas[name]
        if not any(head.names == [name] for head in heads):
            raise SystemExit(f"{name}: its captures hold no compressor inputs ({', '.join(hub.HEAD_KINDS)})")
    hubs = hub.with_heads(hubs, heads)
    settings = {key: getattr(args, key) for key in ("hub_dim", "view_dim", "layer_rank", "fit_rows", "source_global",
                                                    "indexer_ridge", "heads")}
    settings["models"] = {name: {"captures": [str(c.resolve()) for c in caps], "records": str(rec.resolve())}
                          for name, caps, rec in args.model}
    settings["base"] = str(args.base.resolve()) if args.base else None
    hub.save_hub(args.output, hubs, views, spans, settings, base_models=base_models, indexer_ridges=ridges)
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, indent=1) + "\n", encoding="utf-8")
    log(json.dumps({"stage": "done", "seconds": round(time.perf_counter() - started, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
