"""End-to-end evaluation of hub translation for any (source, target) model pair.

For each held-out conversation (corpus JSONL in chat.py's format, ending with the assistant
message to score):

1. render it with both models' own chat templates (chat.py, GGUF metadata) and tokenize with
   each model (kv_probe tokenize);
2. P = the target's first token of the final message; T0 = P - tail rounded down to the batch
   grid; the source prefix P_s = the source's first token of the final message;
3. save-state: the source at P_s, the target at T0 (sanity check and layout template);
4. build: `hub` (translate.translate) and `floor` (no mapped cell: every cell the decoder's
   train mean, the information-free reference with the same native tail);
   `hub_prefix` and `floor_prefix`: the same with the target's own values for its prompt
   start (every token before the first non-system message: the attention-sink tokens and the
   system prompt), which a short native prefill gives exactly in a live switch;
5. seed-score on the target (GPU, f16 cache, flash attention on): native full prefill vs
   sanity / hub / floor / hub_prefix / floor_prefix, the first 256 tokens of the final message.

Both models render with their templates' own defaults (chat.py corpus mode), or with
--server-render as llama-server renders a chat (chat.server_context: its default template kwargs,
preserve_reasoning on, which a template such as Qwen3.6's reads as preserve_thinking and then keeps
the reasoning of earlier turns; the thinking switch; the clock fixed at SERVER_RENDER_NOW so the
renderings are reproducible): what a live switch holds when a server renders a conversation other
than its template's defaults do.

A hybrid target's recurrent cell at T0 - 1 starts from --recurrent (recurrent_init: zero, the
default, or mean with --recurrent-mean, its group by the conversation's tools unless
--recurrent-group) in every built condition.

--ids evaluates a fixed set of conversations (document ids, e.g. RESULTS.md section 2's) where they
fit the prefix range; when fewer than --min-documents of them do (a template that renders them
shorter), the documents are selected from the whole corpus as without --ids. The report records
the selection ("ids" or "fallback") and why each named conversation was left out.

Writes <output>/report.json and prints token-weighted delta NLL with document-bootstrap CIs.
"""

import os

if __name__ == "__main__":
    for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, "8")

import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

import hub as hub_module
import recurrent_init
import translate

sys.path.insert(0, str(hub_module.HERE.parent / "linear_transfer"))
import chat  # noqa: E402
from collect import verify_runtime  # noqa: E402
import collect_v2  # noqa: E402
import handoff  # noqa: E402
import kvstate  # noqa: E402
from run_long import token_sha256  # noqa: E402


def probe(probe_path, artifacts, mode, model, records, output, *, ctx, extra=()):
    path = output.parent / (output.name + ".input.jsonl")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    command = [str(probe_path), "--mode", mode, "--model", str(model), "--input", str(path), "--output", str(output),
               "--ctx", str(ctx), "--batch", "512", "--threads", "16", "--cache-type", "f16", "--flash-attn", "on",
               "--device", "vulkan", *extra]
    with open(output.parent / (output.name + ".log"), "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                       env=collect_v2.probe_environment(artifacts))


# --server-render: the clock llama-server's datetime, date_string and strftime_now read
SERVER_RENDER_NOW = datetime.datetime(2026, 1, 1, 12, 0)


def render(conversation, gguf, server=False):
    """The conversation's record as the model's template renders it (chat.py corpus mode), with the
    EOS text llama.cpp gives every template (DeepSeek-V2's closes each answer with it); server: as
    llama-server renders it (chat.server_context with no request options, at SERVER_RENDER_NOW)."""
    info = chat.gguf_tokenizer_info(gguf)
    if info["chat_template"] is None:
        raise ValueError("model has no tokenizer.chat_template metadata")
    context = (chat.server_context({}, server=True, eos_token=info["eos_token"], now=SERVER_RENDER_NOW) if server
               else {"eos_token": info["eos_token"]})
    rendered = chat.render(conversation, info["chat_template"], bos_token=info["bos_token"], extra_context=context)
    return chat.rendered_record({**conversation, "split": "test", "source": conversation.get("source", "eval")},
                                "model", rendered)


def wanted_ids(text, document_ids):
    """The document ids --ids names (full ids, or prefixes of 8+ hex digits matching one id of the
    corpus; @FILE: one per line); ids not in the corpus are kept as given."""
    if not text:
        return set()
    if text.startswith("@"):
        items = [line.strip() for line in Path(text[1:]).read_text(encoding="utf-8").splitlines()]
    else:
        items = [item.strip() for item in text.split(",")]
    out = set()
    for item in filter(None, items):
        if len(item) < 8:
            raise SystemExit(f"--ids: {item!r} is shorter than 8 hex digits")
        found = [did for did in document_ids if did.startswith(item)]
        if len(found) > 1:
            raise SystemExit(f"--ids: {item!r} matches {len(found)} documents")
        out.add(found[0] if found else item)
    return out


def boot(sums, counts, seed=0):
    sums, counts = np.asarray(sums, float), np.asarray(counts, float)
    index = np.random.default_rng(seed).integers(0, len(sums), (10000, len(sums)))
    stats = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    return float(sums.sum() / counts.sum()), [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="conversations JSONL (chat.py format)")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--min-prefix", type=int, default=4096, help="target tokens before the final message")
    parser.add_argument("--max-prefix", type=int, default=30000)
    parser.add_argument("--sources", help="comma list of conversation 'source' values to keep")
    parser.add_argument("--ids", help="comma list of document ids (or unique prefixes of 8+ hex digits) to evaluate, "
                        "or @FILE with one per line; --limit does not apply to them")
    parser.add_argument("--min-documents", type=int, default=1, help="with --ids: fewer of them fitting the prefix "
                        "range select from the whole corpus instead")
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--source", required=True, help="NAME=GGUF (NAME as in the hub)")
    parser.add_argument("--target", required=True, help="NAME=GGUF")
    parser.add_argument("--tail", type=int, default=2048)
    parser.add_argument("--tables", type=Path, help="token tables directory (<target>.npz; default: token-tables "
                        "next to the hub); adds hub_table_prefix (table fill + native prefix)")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--recurrent", default="zero", help="hybrid targets: the recurrent cell at T0 - 1 "
                        "(recurrent_init strategies: zero, mean, mean_conv, mean_s<x>)")
    parser.add_argument("--recurrent-mean", type=Path, help="the target's recurrent mean npz (r0_experiments.py "
                        "calibrate), for strategies that read it")
    parser.add_argument("--recurrent-group", help="the mean's group (default: chat, or all for conversations "
                        "with tools, as the engine picks it)")
    parser.add_argument("--server-render", action="store_true",
                        help="render both models as llama-server does (its default template kwargs and thinking "
                        "switch, the clock at SERVER_RENDER_NOW) instead of with their templates' defaults")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    (s_name, s_gguf), (t_name, t_gguf) = (item.split("=", 1) for item in (args.source, args.target))
    hubs, _ = hub_module.load_hub(args.hub)
    hubs["indexer"] = hub_module.load_indexer_ridges(args.hub)       # DSA targets: their indexer keys
    table_path = (args.tables or args.hub.parent / "token-tables") / f"{args.target.split('=', 1)[0]}.npz"
    table = handoff.TokenTable(table_path) if table_path.is_file() else None
    revision, _, artifacts = verify_runtime(args.probe.resolve())
    # the probe's runtime saves the states: its revision's layouts, which they are read strictly with
    s_spec, t_spec = (kvstate.ModelSpec.from_gguf(gguf, llama_revision=kvstate.llama_revision_of(revision))
                      for gguf in (s_gguf, t_gguf))
    keep = set(args.sources.split(",")) if args.sources else None

    conversations = [json.loads(line) for line in open(args.corpus, encoding="utf-8") if line.strip()]
    candidates = []
    for conversation in conversations:
        if keep and conversation.get("source") not in keep:
            continue
        try:
            records = {"s": render(conversation, s_gguf, args.server_render),
                       "t": render(conversation, t_gguf, args.server_render)}
        except ValueError as error:
            print(json.dumps({"skip": conversation.get("id"), "reason": str(error)[:120]}))
            continue
        candidates.append((conversation, records))
    tokens = {}
    for side, gguf in (("s", s_gguf), ("t", t_gguf)):
        tokens[side] = collect_v2.tokenize(args.probe, gguf, [r[side] for _, r in candidates], ctx=args.ctx,
                                           workdir=args.output)
    wanted = wanted_ids(args.ids, [records["t"]["document_id"] for _, records in candidates])
    documents, left_out = [], {}
    for conversation, records in candidates:
        did = records["t"]["document_id"]
        P = handoff.handoff_position(records["t"], tokens["t"][did])
        P_s = handoff.handoff_position(records["s"], tokens["s"][did])
        t0 = (P - args.tail) // 512 * 512
        total = max(len(tokens["t"][did]["token_id"]), len(tokens["s"][did]["token_id"]))
        if not args.min_prefix <= P <= args.max_prefix or total + 64 > args.ctx or t0 <= (t_spec.n_swa or 0) + 1:
            if did in wanted:
                left_out[did] = {"P": P, "tokens": total, "t0": t0}
            continue
        documents.append({"conversation": conversation, "records": records, "id": did, "P": P, "P_s": P_s, "t0": t0})
    documents.sort(key=lambda d: d["P"])
    chosen = [d for d in documents if d["id"] in wanted]
    selection = "ids" if wanted and len(chosen) >= args.min_documents else ("fallback" if wanted else "corpus")
    if selection == "ids":
        documents = chosen
    elif len(documents) > args.limit:
        picks = np.unique(np.round(np.linspace(0, len(documents) - 1, args.limit)).astype(int))
        documents = [documents[i] for i in picks]
    if not documents:
        raise SystemExit("no conversation fits the prefix range")
    print(json.dumps({"documents": len(documents), "selection": selection, "prefix": [d["P"] for d in documents],
                      "left_out": {did[:12]: item for did, item in left_out.items()}}), flush=True)

    for side, gguf, at in (("s", s_gguf, "P_s"), ("t", t_gguf, "t0")):
        probe(args.probe, artifacts, "save-state", gguf,
              [{"document_id": d["id"], "text": d["records"][side]["text"], "save_at": [d[at]],
                "expected_token_sha256": token_sha256(tokens[side][d["id"]]["token_id"])} for d in documents],
              args.output / f"states-{side}", ctx=args.ctx)
    built = args.output / "built"
    inputs, means = [], {}
    for d in documents:
        did = d["id"]
        source_state = kvstate.read_file(next((args.output / "states-s").rglob(f"{did}/{d['P_s']}.state")), s_spec,
                                         strict=True)
        native_path = next((args.output / "states-t").rglob(f"{did}/{d['t0']}.state"))
        template = kvstate.read_file(native_path, t_spec, strict=True)
        source = translate.Side(s_name, s_spec, d["records"]["s"], tokens["s"][did], gguf=s_gguf)
        target = translate.Side(t_name, t_spec, d["records"]["t"], tokens["t"][did], gguf=t_gguf)
        out = built / did
        out.mkdir(parents=True)
        recurrent = recurrent_init.translate_options(args.recurrent, args.recurrent_mean, args.recurrent_group,
                                                     tools=d["conversation"].get("tools"), means=means)
        state, report = translate.translate(source_state, source, target, hubs, t0=d["t0"], template=template,
                                            **recurrent)
        kvstate.write_new(out / "hub.state", state.to_bytes())
        empty = translate.Side(s_name, s_spec, {**d["records"]["s"], "segments": []}, tokens["s"][did], gguf=s_gguf)
        floor, _ = translate.translate(source_state, empty, target, hubs, t0=d["t0"], template=template, **recurrent)
        kvstate.write_new(out / "floor.state", floor.to_bytes())
        first = min(start for (number, start, _) in d["records"]["t"]["segments"]
                    if d["records"]["t"]["segment_keys"][number].split(":", 1)[0] != "0")
        H = int(np.searchsorted(tokens["t"][did]["byte_end"], first, side="right"))
        d["H"] = H
        for name, side in (("hub_prefix", source), ("floor_prefix", empty)):
            state, _ = translate.translate(source_state, side, target, hubs, t0=d["t0"], template=template,
                                           native_state=template, native_prefix=H, **recurrent)
            kvstate.write_new(out / f"{name}.state", state.to_bytes())
        if table is not None:
            state, _ = translate.translate(source_state, source, target, hubs, t0=d["t0"], template=template,
                                           native_state=template, native_prefix=H, fill="table", table=table,
                                           **recurrent)
            kvstate.write_new(out / "hub_table_prefix.state", state.to_bytes())
        d["build"] = report
        print(json.dumps({"built": did[:12], "P": d["P"], "t0": d["t0"], "mapped": report["mapped_cells"],
                          "fill": report["fill_cells"], "seconds": report["seconds"]}), flush=True)
        start = handoff.final_message_start(d["records"]["t"])
        inputs.append({"document_id": did, "text": d["records"]["t"]["text"],
                       "expected_token_sha256": token_sha256(tokens["t"][did]["token_id"]),
                       "score_ranges": [[start, len(d["records"]["t"]["text"].encode("utf-8"))]],
                       "conditions": [{"name": "native", "state_file": None},
                                      {"name": "sanity", "state_file": str(native_path)},
                                      {"name": "hub", "state_file": str(out / "hub.state")},
                                      {"name": "floor", "state_file": str(out / "floor.state")},
                                      {"name": "hub_prefix", "state_file": str(out / "hub_prefix.state")},
                                      {"name": "floor_prefix", "state_file": str(out / "floor_prefix.state")}]
                       + ([{"name": "hub_table_prefix", "state_file": str(out / "hub_table_prefix.state")}]
                          if table is not None else [])})
    probe(args.probe, artifacts, "seed-score", t_gguf, inputs, args.output / "scores", ctx=args.ctx,
          extra=("--max-scored", "256"))
    names = ("sanity", "hub", "floor", "hub_prefix", "floor_prefix") + (("hub_table_prefix",) if table is not None else ())
    rows, sums, counts = [], {n: [] for n in names}, []
    for d in documents:
        base = args.output / "scores" / "scores" / d["id"]
        native = np.load(base / "native.npy").astype(np.float64)
        row = {"document_id": d["id"], "source": d["conversation"].get("source"), "P": d["P"], "t0": d["t0"], "H": d["H"],
               "n_scored": len(native), "mapped_cells": d["build"]["mapped_cells"], "fill_cells": d["build"]["fill_cells"]}
        for name in names:
            scores = np.load(base / f"{name}.npy").astype(np.float64)
            row[name] = {"nll_delta": float(np.mean(scores[:, 0] - native[:, 0])), "kl": float(scores[:, 1].mean()),
                         "top1": float(scores[:, 2].mean())}
            sums[name].append(float(np.sum(scores[:, 0] - native[:, 0])))
        counts.append(len(native))
        rows.append(row)
    summary = {name: dict(zip(("nll_delta", "ci95"), boot(sums[name], counts))) for name in names}
    for hub_name, floor_name in (("hub", "floor"), ("hub_prefix", "floor_prefix"), ("hub_table_prefix", "floor_prefix")):
        if hub_name not in summary:
            continue
        hf, fl = summary[hub_name]["nll_delta"], summary[floor_name]["nll_delta"]
        summary[f"recovered_{hub_name}"] = 1 - hf / fl if fl > 0 else None
    for name in names:
        summary[name]["kl"] = float(np.mean([row[name]["kl"] for row in rows]))
        summary[name]["top1"] = float(np.mean([row[name]["top1"] for row in rows]))
    report = {"source": s_name, "target": t_name, "tail": args.tail, "hub": str(args.hub.resolve()),
              "corpus": str(args.corpus.resolve()), "recurrent": args.recurrent,
              "rendering": "server" if args.server_render else "template",
              "recurrent_mean": None if args.recurrent_mean is None else str(args.recurrent_mean.resolve()),
              "selection": selection, "ids": sorted(wanted), "left_out": left_out,
              "missing_ids": sorted(set(wanted) - {r["t"]["document_id"] for _, r in candidates}),
              "documents": rows, "summary": summary}
    (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['document_id'][:12]} {str(row['source'])[:14]:>14} P={row['P']:>6} "
              + " ".join(f"{n}={row[n]['nll_delta']:+.3f}" for n in names))
    print(json.dumps(summary))
    shutil.rmtree(built)
    for side in ("s", "t"):
        shutil.rmtree(args.output / f"states-{side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
