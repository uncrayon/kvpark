"""Recurrent-state strategies for hybrid targets, measured on real models (recurrent_init.py).

recurrent_reset.py measured R0 alone: the target keeps its *native* attention cache at T0 and
its recurrent cell is zeroed. This script does the same for every strategy of recurrent_init
(zero, mean, mean_conv, mean_s<x>, prefix), on chat and agent conversations, per native tail,
and with *translated* attention (the hub) as the switch writes it. Subcommands:

  calibrate  native saves of training-split conversations at many depths (chat conversations of
             one source concatenated into long sessions, tool conversations as they are) ->
             <output>/recurrent-mean.npz (groups all / chat / tools) and report.json (how much of
             the rows' energy the mean explains, per depth bin; timings; file size)
  run        per held-out conversation (run_long plans, --select RUN_DIR:BUCKET) and tail W:
             T0 = P - W rounded down to the --grid (P = the target's first token of the final
             message); the native save at T0 with its recurrent cell replaced by each strategy,
             seed-scored against the native full prefill on the first 256 tokens of the final
             message; plus a sanity condition (the native save itself)
  combined   the same with translated attention: the source model's save at its own P_s, the
             hub translation into the target at T0 with the native prompt start (H tokens, the
             bridge's rule) as the bridge switches into Qwen targets, then each strategy
  summarize  tables from report.json files: token-weighted mean delta NLL with document-bootstrap
             95% CIs per (kind, tail, strategy), the paired difference to zero, KL and top-1

T0 on the batch grid (--grid 512, the default) keeps every ubatch of the restored conditions
where the native prefill has it, so the sanity condition scores exactly 0 (checked at the middle
tail). --grid 1 puts T0 at exactly P - W, as the engine does: the save at T0 and the restored
tail then split one ubatch of the native prefill in two, and a sanity condition at every tail
measures what that alone costs. Every native save is also rewritten through the strategy write
path (section, replace_recurrent, state_writer) with its own cell, which must give its bytes back.

Prefix H (the prefix strategy's native state and the combined case's native prompt start)
follows kvpark.bridge.switch.native_prefix: the tokens the system message rendered alone shares
with the prompt, less 2, at least 4. Its save is a separate probe record (save_records): a save
at H in the same pass would split the first ubatch there, and the saves at T0 would then come
from another ubatch partition than the native reference (the sanity condition would not be 0).
Documents run in chunks (--chunk), each with its own probe runs, so the slot files on disk stay
bounded; a finished chunk is skipped by --resume.

Memory guard: before each probe run of a model file over 8 GB, MemAvailable must reach the
model's size + 25 GB (polled every 60 s for up to 90 min; waits are logged). Probe runs are
sequential, so one model is loaded at a time.

Resource guard (resource_guard.py): BLAS threads and the probe's --threads are capped at 8; run by
hand, every probe run waits until the CPU is at most 88 C and runs under a Watch (paused at 90 C, continued at
88 C, ended at 98 C on two samples in a row; samples in <output>/.../<step>.guard.jsonl); inside a campaign
command (calibrate) the campaign's Watch covers the whole process group instead. --guard KEY=VALUE
changes a setting.
"""

import os

import resource_guard

if __name__ == "__main__":
    resource_guard.cap_threads(os.environ)

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

import hub as hub_module
import pair_eval
import recurrent_init
import state_writer
import translate

sys.path.insert(0, str(hub_module.HERE.parent / "linear_transfer"))
import chat  # noqa: E402
from collect import verify_runtime  # noqa: E402
import collect_v2  # noqa: E402
import handoff  # noqa: E402
import kvstate  # noqa: E402
from run_long import token_sha256  # noqa: E402

GIB = 1024 ** 3
GUARD_MARGIN = 25 * GIB
GUARD_MODEL = 8 * GIB
GUARD_POLL_S = 60
GUARD_LIMIT_S = 90 * 60
BATCH = 512
DEPTH_BINS = (1024, 2048, 4096, 8192)
DEFAULT_STRATEGIES = "zero,mean,mean_conv,mean_s0.5,prefix"


def log_line(path, item):
    item = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **item}
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(item) + "\n")
    print(json.dumps(item), flush=True)


def mem_available():
    with open("/proc/meminfo", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def wait_for_memory(model, log, *, margin=GUARD_MARGIN, poll=GUARD_POLL_S, limit=GUARD_LIMIT_S):
    """Block until MemAvailable >= model size + margin (models over 8 GB only); log every wait."""
    size = Path(model).stat().st_size
    if size <= GUARD_MODEL:
        return 0.0
    need, started = size + margin, time.monotonic()
    while True:
        available = mem_available()
        if available >= need:
            waited = time.monotonic() - started
            if waited >= 1:
                log_line(log, {"event": "memory-ok", "model": Path(model).name, "waited_s": round(waited),
                               "available_gib": round(available / GIB, 1)})
            return waited
        waited = time.monotonic() - started
        if waited >= limit:
            raise SystemExit(f"memory guard: MemAvailable {available / GIB:.1f} GiB < {need / GIB:.1f} GiB "
                             f"for {Path(model).name} after {waited / 60:.0f} min")
        log_line(log, {"event": "memory-wait", "model": Path(model).name, "available_gib": round(available / GIB, 1),
                       "need_gib": round(need / GIB, 1), "waited_s": round(waited)})
        time.sleep(poll)


def probe(args, artifacts, mode, model, records, output, *, extra=()):
    """One kv_probe run (Vulkan, f16 cache, flash attention, batch 512, --threads the resource guard's
    cap) behind the memory guard and under the resource guard (resource_guard.run: samples in
    <output>.guard.jsonl); returns (parsed report or results JSON, seconds: guard pauses excluded)."""
    guard = resource_guard.Settings.from_args(args)
    path = output.parent / (output.name + ".input.jsonl")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    wait_for_memory(model, args.log)
    command = [str(args.probe), "--mode", mode, "--model", str(model), "--input", str(path), "--output", str(output),
               "--ctx", str(args.ctx), "--batch", str(BATCH), "--threads", str(guard.threads), "--cache-type", "f16",
               "--flash-attn", "on", "--device", args.device, *extra]
    log_line(args.log, {"event": "probe", "mode": mode, "model": Path(model).name, "documents": len(records),
                        "available_gib": round(mem_available() / GIB, 1), "load": os.getloadavg()[0]})
    with open(output.parent / (output.name + ".log"), "w") as stream:
        outcome = resource_guard.run(command, settings=guard, log_path=output.parent / (output.name + ".guard.jsonl"),
                                     label=f"probe {mode} {Path(model).name}", stdout=stream,
                                     stderr=subprocess.STDOUT, env=collect_v2.probe_environment(artifacts), check=True)
    seconds = outcome.seconds
    log_line(args.log, {"event": "probe-done", "mode": mode, "seconds": round(seconds, 1),
                        "paused_s": outcome.paused_s, "cool_wait_s": round(outcome.cool_wait_s, 1)})
    return json.loads((output / ("report.json" if mode == "save-state" else "results.json")).read_text()), seconds


def state_path(directory, document_id, n_tokens):
    found = sorted(Path(directory).rglob(f"{document_id}/{n_tokens}.state"))
    if len(found) != 1:
        raise FileNotFoundError(f"{directory}: {len(found)} states {document_id[:12]}@{n_tokens}")
    return found[0]


def common_prefix_bytes(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0
    x, y = np.frombuffer(a[:n], dtype=np.uint8), np.frombuffer(b[:n], dtype=np.uint8)
    differ = np.flatnonzero(x != y)
    return int(differ[0]) if len(differ) else n


def prefix_length(conversation, record, tokens, template, bos, P, *, minimum=4):
    """H of kvpark.bridge.switch.native_prefix, in bytes: the target tokens wholly inside the bytes
    the system message rendered alone (with the generation prompt, as /apply-template renders it)
    shares with the prompt, less 2; at least `minimum`, below P."""
    messages, H = conversation["messages"], 0
    if messages and messages[0]["role"] == "system" and len(messages) > 1:
        head = chat.render({"messages": messages[:1], "tools": conversation.get("tools")}, template,
                           bos_token=bos, add_generation_prompt=True, live=True).text.encode("utf-8")
        shared = common_prefix_bytes(head, record["text"].encode("utf-8"))
        H = max(0, int(np.searchsorted(tokens["byte_end"], shared, side="right")) - 2)
    return int(min(max(H, minimum), P - 1))


def kind_of(conversation):
    return "agent" if conversation.get("tools") or "toolcalling" in str(conversation.get("source")) else "chat"


def group_of(conversation):
    return "tools" if conversation.get("tools") else "chat"


def t0_for(P, tail, grid=BATCH):
    """T0 for a native tail of W = tail tokens: P - W rounded down to the grid (1: exactly)."""
    return max(grid, (P - tail) // grid * grid)


def strategy_list(text):
    """[(label, strategy, group)] from 'zero,mean,mean:tools,mean:match,...' (group 'match': the
    document's own group, chat or tools)."""
    out = []
    for item in text.split(","):
        name, _, group = item.strip().partition(":")
        recurrent_init.parse_strategy(name)
        out.append((name + (f"-{group}" if group else ""), name, group or recurrent_init.ALL))
    return out


# -- documents -----------------------------------------------------------------------------

def plan_conversations(selections, per_bucket, sources):
    """[(conversation, plan document)] of run_long plans: RUN_DIR:BUCKET, primary documents."""
    out = []
    for item in selections:
        path, _, bucket = item.rpartition(":")
        plan = json.loads((Path(path) / "long" / "plan.json").read_text(encoding="utf-8"))
        corpus = {c["id"]: c for c in chat.read_conversations(Path(plan["corpus"]["path"]))}
        docs = sorted([d for d in plan["documents"] if d["bucket"] == bucket and d.get("role") == "primary"
                       and (not sources or d["source"] in sources)], key=lambda d: (d["source"], d["qwen_prefix"]))
        if per_bucket:
            picks = np.unique(np.round(np.linspace(0, len(docs) - 1, min(per_bucket, len(docs)))).astype(int))
            docs = [docs[i] for i in picks]
        out += [(corpus[d["conversation_id"]], d) for d in docs]
    return out


def render_documents(args, conversations, gguf, *, workdir, prefix=True):
    """Per conversation: record (rendered with the model's own template), tokens, P and (prefix)
    the native prompt start H."""
    template, bos = chat.gguf_chat_context(str(gguf))
    records = [pair_eval.render(conversation, str(gguf)) for conversation in conversations]
    tokens = collect_v2.tokenize(args.probe, gguf, records, ctx=args.ctx, workdir=workdir)
    out = []
    for conversation, record in zip(conversations, records):
        did = record["document_id"]
        P = handoff.handoff_position(record, tokens[did])
        out.append({"conversation": conversation, "record": record, "tokens": tokens[did], "P": P,
                    "H": prefix_length(conversation, record, tokens[did], template, bos, P) if prefix else None,
                    "kind": kind_of(conversation), "group": group_of(conversation)})
    return out


def prefix_id(document_id):
    """The probe record id of a document's prefix save (64 lowercase hex, like a document id)."""
    return hashlib.sha256(f"{document_id}:prefix".encode()).hexdigest()


def save_records(docs):
    """save-state records: per document its T0s (on the batch grid), and a second record of the
    same text saving only H (see the module docstring)."""
    out = []
    for doc in docs:
        record, sha = doc["record"], token_sha256(doc["tokens"]["token_id"])
        out.append({"document_id": record["document_id"], "text": record["text"],
                    "save_at": sorted(set(doc["t0s"].values())), "expected_token_sha256": sha})
        out.append({"document_id": prefix_id(record["document_id"]), "text": record["text"], "save_at": [doc["H"]],
                    "expected_token_sha256": sha})
    return out


def score_input(doc, conditions):
    record = doc["record"]
    start = handoff.final_message_start(record)
    return {"document_id": record["document_id"], "text": record["text"],
            "expected_token_sha256": token_sha256(doc["tokens"]["token_id"]),
            "score_ranges": [[start, len(record["text"].encode("utf-8"))]], "conditions": conditions}


def score_rows(scores_dir, doc, names):
    """{condition: {nll_delta, kl, top1, nll_sum_delta}} of one document's seed-score output."""
    base = Path(scores_dir) / "scores" / doc["record"]["document_id"]
    native = np.load(base / "native.npy").astype(np.float64)
    out = {"n_scored": int(len(native)), "native_nll": float(native[:, 0].mean())}
    for name in names:
        scores = np.load(base / f"{name}.npy").astype(np.float64)
        out[name] = {"nll_delta": float(np.mean(scores[:, 0] - native[:, 0])),
                     "nll_sum_delta": float(np.sum(scores[:, 0] - native[:, 0])),
                     "kl": float(scores[:, 1].mean()), "top1": float(scores[:, 2].mean())}
    return out


def chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def prepare_output(args):
    if args.output.exists() and not args.resume:
        raise SystemExit(f"{args.output} exists (use --resume to continue it)")
    args.output.mkdir(parents=True, exist_ok=True)
    args.log = args.output / "log.jsonl"
    _, _, artifacts = verify_runtime(args.probe.resolve())
    return artifacts


def distances(native, part):
    distance = recurrent_init.state_distance(native, part)
    return {key: round(value["relative"], 6) for key, value in distance.items() if value["relative"] is not None}


# -- calibrate -------------------------------------------------------------------------------

def calibration_conversations(paths, *, concat_chars, exclude):
    """Training-split conversations, joined per (source, tool list) into sessions of about
    concat_chars characters: the first one's system message (if it has one), then every
    conversation's turns after its own system message (tool call ids made unique). A
    conversation whose tool list no other shares stays alone."""
    seen, groups = set(), {}
    for path in paths:
        for conversation in chat.read_conversations(Path(path)):
            if conversation.get("split") != "train" or conversation["id"] in seen or conversation["id"] in exclude:
                continue
            seen.add(conversation["id"])
            key = (conversation["source"], json.dumps(conversation.get("tools"), sort_keys=True))
            groups.setdefault(key, []).append(conversation)

    def size(conversation):
        return sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "")
                   for m in conversation["messages"])

    def system(conversation):
        messages = conversation["messages"]
        return messages[:1] if messages and messages[0]["role"] == "system" else []
    sessions = []
    for key in sorted(groups):
        group, total = [], 0
        for conversation in sorted(groups[key], key=lambda c: c["id"]):
            group.append(conversation)
            total += size(conversation)
            if total >= concat_chars:
                sessions.append(group)
                group, total = [], 0
        if group:
            sessions.append(group)
    out = []
    for group in sessions:
        if len(group) == 1:
            out.append(group[0])
            continue
        messages = list(system(group[0]))
        for number, conversation in enumerate(group):
            for message in conversation["messages"][len(system(conversation)):]:
                message = json.loads(json.dumps(message))
                for call in message.get("tool_calls") or ():
                    call["id"] = f"c{number}-{call['id']}"
                if message["role"] == "tool":
                    message["tool_call_id"] = f"c{number}-{message['tool_call_id']}"
                messages.append(message)
        out.append({"id": "concat:" + "+".join(c["id"] for c in group), "source": group[0]["source"],
                    "split": "train", "tools": group[0].get("tools"), "messages": messages})
    return out


def probe_spec(args, gguf):
    """The model's spec at the revision of args.probe's runtime (its build manifest): the layouts and RoPE
    of the states it saves, which are read strictly with it (kvstate.parse_state)."""
    return kvstate.ModelSpec.from_gguf(gguf, llama_revision=kvstate.llama_revision_of(Path(args.probe)))


def calibrate(args):
    artifacts = prepare_output(args)
    spec = probe_spec(args, args.model)
    exclude = set()
    for item in args.exclude or ():
        exclude |= {c["id"] for c in chat.read_conversations(Path(item))}
    conversations = calibration_conversations(args.corpus, concat_chars=args.concat_chars, exclude=exclude)
    work = args.output / "work"
    work.mkdir(exist_ok=True)
    docs = render_documents(args, conversations, args.model, workdir=work, prefix=False)
    plan = []
    for doc in docs:
        n = len(doc["tokens"]["token_id"])
        cuts = list(range(args.min_depth, n, args.every))
        if cuts and n + 64 <= args.ctx:
            plan.append((doc, cuts))
    log_line(args.log, {"event": "calibration-plan", "conversations": len(conversations), "documents": len(plan),
                        "saves": sum(len(c) for _, c in plan), "tokens": sum(len(d["tokens"]["token_id"]) for d, _ in plan),
                        "chat_sessions": sum(d["group"] == "chat" for d, _ in plan)})
    accumulator, bins = recurrent_init.MeanAccumulator(), recurrent_init.MeanAccumulator()
    seconds, read_seconds = 0.0, 0.0
    for number, chunk in enumerate(chunks(plan, args.chunk)):
        states = work / f"states-{number:03d}"
        records = [{"document_id": d["record"]["document_id"], "text": d["record"]["text"], "save_at": cuts,
                    "expected_token_sha256": token_sha256(d["tokens"]["token_id"])} for d, cuts in chunk]
        _, spent = probe(args, artifacts, "save-state", args.model, records, states)
        seconds += spent
        started = time.perf_counter()
        for doc, cuts in chunk:
            for cut in cuts:
                path = state_path(states, doc["record"]["document_id"], cut)
                state = kvstate.read_file(path, spec, strict=True)
                accumulator.add(state, group=doc["group"], depth=cut)
                bins.add(state, group=f"depth-{max(b for b in DEPTH_BINS if b <= cut)}", depth=cut)
                del state
                path.unlink()
        read_seconds += time.perf_counter() - started
        shutil.rmtree(states)
    model = {"gguf": Path(args.model).name, "bytes": Path(args.model).stat().st_size,
             "architecture": spec.architecture, "name": args.name}
    started = time.perf_counter()
    mean = accumulator.result(model=model)
    output = mean.save(args.output / "recurrent-mean.npz", dtype=args.dtype)
    save_seconds = time.perf_counter() - started
    overall = mean.rows()
    by_depth = {}
    for label in sorted(bins.groups, key=lambda name: int(name.split("-")[1])):
        stats = bins.stats(label)
        bin_mean = bins.mean(label)
        shift = {}
        for key in ("r", "s"):
            moved = sum(float(np.sum((bin_mean[key][l].astype(np.float64) - overall[key][l]) ** 2))
                        for l in bin_mean[key])
            energy = stats[key]["energy"] if key in stats else 0.0
            # how far this depth's mean lies from the overall mean, against the rows' energy here
            shift[key] = moved / energy if energy else None
        by_depth[label] = {"count": stats["count"], "depth": stats["depth"],
                           "explained_by_own_mean": {k: stats[k]["explained"] for k in ("r", "s") if k in stats},
                           "overall_mean_shift": shift}
    groups = {name: accumulator.stats(name) for name in mean.groups}
    report = {"model": model, "corpora": [str(p) for p in args.corpus], "concat_chars": args.concat_chars,
              "min_depth": args.min_depth, "every": args.every, "documents": len(plan),
              "saves": sum(len(c) for _, c in plan), "groups": {
                  name: {"count": item["count"], "depth": item["depth"],
                         "explained": {k: item[k]["explained"] for k in ("r", "s", "p") if k in item},
                         "energy": {k: item[k]["energy"] for k in ("r", "s", "p") if k in item},
                         "s_explained_per_layer": {str(l): v["explained"] for l, v in item["s"]["layers"].items()}}
                  for name, item in groups.items()},
              "depth_bins": by_depth, "probe_seconds": round(seconds, 1), "read_seconds": round(read_seconds, 1),
              "save_seconds": round(save_seconds, 2), "npz_bytes": output.stat().st_size, "npz": str(output),
              "dtype": args.dtype}
    (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(work)
    print(json.dumps({k: report[k] for k in ("documents", "saves", "probe_seconds", "npz_bytes")}))
    return 0


# -- run (native attention) ------------------------------------------------------------------

def same_bytes(path, state_file):
    """Whether state_writer writes state_file as exactly the bytes of the file at path (streamed)."""
    with open(path, "rb") as stream:
        for chunk in state_writer.chunks(state_file):
            data = bytes(memoryview(chunk))
            if stream.read(len(data)) != data:
                return False
        return stream.read(1) == b""


def check_rewrite(native_path, native, t0):
    """The strategy write path (section, replace_recurrent, state_writer) applied to the native
    save with its own cell must give the native file's bytes back: what every condition differs
    by is then the recurrent cell alone."""
    part = recurrent_init.recurrent_part(native)
    if part.positions.tolist() != [t0 - 1]:
        raise SystemExit(f"{native_path}: recurrent cells at {part.positions.tolist()}, expected [{t0 - 1}]")
    rewritten = recurrent_init.section(part, t0 - 1, recurrent_init.cell_rows(part))
    if not same_bytes(native_path, recurrent_init.replace_recurrent(native, rewritten)):
        raise SystemExit(f"{native_path}: the recurrent write path does not reproduce the native save")


def sanity_tails(t0s):
    """Tails with a sanity condition: the middle one when every T0 is on the batch grid (it then
    scores exactly 0), else every tail (each split ubatch has its own cost)."""
    if all(t0 % BATCH == 0 for t0 in t0s.values()):
        return {sorted(t0s)[len(t0s) // 2]}
    return set(t0s)


def build_conditions(doc, t0s, states, built, strategies, mean, *, translated=None):
    """Write one state file per (tail, strategy) and return (seed-score conditions, diagnostics).

    translated: {t0: StateFile} of translated states (combined case); default: the native saves."""
    did = doc["record"]["document_id"]
    prefix = kvstate.read_file(state_path(states, prefix_id(did), doc["H"]), doc["spec"], strict=True)
    conditions, diagnostics, timings = [], {}, []
    sanity = sanity_tails(t0s)
    out = Path(built) / did
    out.mkdir(parents=True, exist_ok=True)
    for tail, t0 in sorted(t0s.items()):
        native_path = state_path(states, did, t0)
        native = kvstate.read_file(native_path, doc["spec"], strict=True)
        check_rewrite(native_path, native, t0)
        base = translated[t0] if translated is not None else native
        if tail in sanity:
            conditions.append({"name": f"sanity_{tail}", "state_file": str(native_path)})
        for label, name, group in strategies:
            group = doc["group"] if group == "match" else group
            started = time.perf_counter()
            part = recurrent_init.initial_state(native, t0 - 1, name, mean=mean, prefix=prefix, group=group)
            timings.append(time.perf_counter() - started)
            path = out / f"{label}_{tail}.state"
            state_writer.write(path, recurrent_init.replace_recurrent(base, part))
            conditions.append({"name": f"{label}_{tail}", "state_file": str(path)})
            diagnostics[f"{label}_{tail}"] = distances(native, part)
        del native, base
    return conditions, diagnostics, timings


def run(args):
    artifacts = prepare_output(args)
    spec = probe_spec(args, args.model)
    mean = recurrent_init.RecurrentMean.load(args.mean) if args.mean else None
    strategies = strategy_list(args.strategies)
    tails = [int(t) for t in args.tails.split(",")]
    pairs = plan_conversations(args.select, args.per_bucket, set(args.sources.split(",")) if args.sources else None)
    work = args.output / "work"
    work.mkdir(exist_ok=True)
    docs = render_documents(args, [c for c, _ in pairs], args.model, workdir=work)
    for doc in docs:
        doc["spec"] = spec
        doc["t0s"] = {tail: t0_for(doc["P"], tail, args.grid) for tail in tails}
        if len(set(doc["t0s"].values())) != len(tails) or doc["H"] >= min(doc["t0s"].values()):
            raise SystemExit(f"{doc['record']['document_id'][:12]}: P {doc['P']} too short for tails {tails}")
    log_line(args.log, {"event": "run-plan", "model": Path(args.model).name, "documents": len(docs),
                        "kinds": {k: sum(d["kind"] == k for d in docs) for k in ("chat", "agent")},
                        "P": [d["P"] for d in docs], "H": [d["H"] for d in docs], "tails": tails, "grid": args.grid,
                        "strategies": [label for label, _, _ in strategies]})
    rows = []
    for number, chunk in enumerate(chunks(docs, args.chunk)):
        directory = args.output / f"chunk-{number:03d}"
        if (directory / "rows.json").is_file():
            rows += json.loads((directory / "rows.json").read_text())
            continue
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()
        _, save_s = probe(args, artifacts, "save-state", args.model, save_records(chunk), directory / "states")
        inputs, names, timings = [], {}, []
        for doc in chunk:
            conditions, doc["distance"], spent = build_conditions(doc, doc["t0s"], directory / "states",
                                                                  directory / "built", strategies, mean)
            timings += spent
            names[doc["record"]["document_id"]] = [c["name"] for c in conditions]
            inputs.append(score_input(doc, [{"name": "native", "state_file": None}] + conditions))
        _, score_s = probe(args, artifacts, "seed-score", args.model, inputs, directory / "scores",
                           extra=("--max-scored", str(args.max_scored)))
        chunk_rows = []
        for doc in chunk:
            did = doc["record"]["document_id"]
            chunk_rows.append({"document_id": did, "conversation_id": doc["conversation"]["id"],
                               "source": doc["conversation"].get("source"), "kind": doc["kind"],
                               "group": doc["group"], "P": doc["P"], "H": doc["H"],
                               "t0": {str(k): v for k, v in doc["t0s"].items()},
                               "tail": {str(k): doc["P"] - v for k, v in doc["t0s"].items()},
                               "scores": score_rows(directory / "scores", doc, names[did]),
                               "state_distance": doc["distance"]})
        (directory / "timings.json").write_text(json.dumps(
            {"save_state_s": round(save_s, 1), "seed_score_s": round(score_s, 1),
             "initial_state_s": [round(t, 4) for t in timings]}) + "\n")
        (directory / "rows.json").write_text(json.dumps(chunk_rows, indent=1) + "\n")
        shutil.rmtree(directory / "built")
        shutil.rmtree(directory / "states")
        rows += chunk_rows
        log_line(args.log, {"event": "chunk-done", "chunk": number, "documents": len(chunk)})
    report = {"model": Path(args.model).name, "mean": str(args.mean) if args.mean else None, "tails": tails,
              "grid": args.grid,
              "strategies": [label for label, _, _ in strategies], "select": args.select, "documents": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(summary_text(summarize_reports([report])))
    return 0


# -- combined (translated attention) -----------------------------------------------------------

def combined(args):
    artifacts = prepare_output(args)
    (s_name, s_gguf), (t_name, t_gguf) = (item.split("=", 1) for item in (args.source, args.target))
    s_spec, t_spec = probe_spec(args, s_gguf), probe_spec(args, t_gguf)
    mean = recurrent_init.RecurrentMean.load(args.mean) if args.mean else None
    strategies = strategy_list(args.strategies)
    tails = [int(t) for t in args.tails.split(",")]
    hub = hub_module.open_hub(args.hub)
    backend = translate.backends.select(args.engine_device)
    pairs = plan_conversations(args.select, args.per_bucket, set(args.sources.split(",")) if args.sources else None)
    work = args.output / "work"
    work.mkdir(exist_ok=True)
    conversations = [c for c, _ in pairs]
    docs = render_documents(args, conversations, t_gguf, workdir=work)
    sources = render_documents(args, conversations, s_gguf, workdir=work, prefix=False)
    for doc, source in zip(docs, sources):
        doc.update(spec=t_spec, source=source, t0s={tail: t0_for(doc["P"], tail, args.grid) for tail in tails})
        if len(set(doc["t0s"].values())) != len(tails) or doc["H"] >= min(doc["t0s"].values()):
            raise SystemExit(f"{doc['record']['document_id'][:12]}: P {doc['P']} too short for tails {tails}")
    log_line(args.log, {"event": "combined-plan", "source": s_name, "target": t_name, "documents": len(docs),
                        "P": [d["P"] for d in docs], "P_s": [d["source"]["P"] for d in docs],
                        "H": [d["H"] for d in docs], "tails": tails, "grid": args.grid,
                        "strategies": [label for label, _, _ in strategies]})
    rows = []
    for number, chunk in enumerate(chunks(docs, args.chunk)):
        directory = args.output / f"chunk-{number:03d}"
        if (directory / "rows.json").is_file():
            rows += json.loads((directory / "rows.json").read_text())
            continue
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()
        _, source_s = probe(args, artifacts, "save-state", s_gguf, [
            {"document_id": d["source"]["record"]["document_id"], "text": d["source"]["record"]["text"],
             "save_at": [d["source"]["P"]], "expected_token_sha256": token_sha256(d["source"]["tokens"]["token_id"])}
            for d in chunk], directory / "states-s")
        _, target_s = probe(args, artifacts, "save-state", t_gguf, save_records(chunk), directory / "states")
        inputs, names, builds = [], {}, []
        for doc in chunk:
            did, source = doc["record"]["document_id"], doc["source"]
            source_state = kvstate.read_file(state_path(directory / "states-s", source["record"]["document_id"],
                                                        source["P"]), s_spec, strict=True)
            prefix = kvstate.read_file(state_path(directory / "states", prefix_id(did), doc["H"]), t_spec,
                                       strict=True)
            source_side = translate.Side(s_name, s_spec, source["record"], source["tokens"], gguf=s_gguf)
            target_side = translate.Side(t_name, t_spec, doc["record"], doc["tokens"], gguf=t_gguf)
            translated, reports = {}, {}
            for tail, t0 in doc["t0s"].items():
                template = kvstate.read_file(state_path(directory / "states", did, t0), t_spec, strict=True)
                translated[t0], report = translate.translate(source_state, source_side, target_side, hub, t0=t0,
                                                             template=template, native_state=prefix,
                                                             native_prefix=doc["H"], backend=backend)
                reports[str(tail)] = {k: report[k] for k in ("mapped_cells", "fill_cells", "native_prefix", "seconds")}
            del source_state
            conditions, doc["distance"], _ = build_conditions(doc, doc["t0s"], directory / "states",
                                                              directory / "built", strategies, mean,
                                                              translated=translated)
            del translated
            doc["translate"] = reports
            builds.append(reports)
            names[did] = [c["name"] for c in conditions]
            inputs.append(score_input(doc, [{"name": "native", "state_file": None}] + conditions))
            log_line(args.log, {"event": "built", "document": did[:12], "translate": reports})
        shutil.rmtree(directory / "states-s")
        _, score_s = probe(args, artifacts, "seed-score", t_gguf, inputs, directory / "scores",
                           extra=("--max-scored", str(args.max_scored)))
        chunk_rows = []
        for doc in chunk:
            did = doc["record"]["document_id"]
            chunk_rows.append({"document_id": did, "conversation_id": doc["conversation"]["id"],
                               "source": doc["conversation"].get("source"), "kind": doc["kind"],
                               "group": doc["group"], "P": doc["P"], "P_s": doc["source"]["P"], "H": doc["H"],
                               "t0": {str(k): v for k, v in doc["t0s"].items()},
                               "tail": {str(k): doc["P"] - v for k, v in doc["t0s"].items()},
                               "translate": doc["translate"], "scores": score_rows(directory / "scores", doc, names[did]),
                               "state_distance": doc["distance"]})
        (directory / "timings.json").write_text(json.dumps(
            {"source_save_s": round(source_s, 1), "target_save_s": round(target_s, 1),
             "seed_score_s": round(score_s, 1)}) + "\n")
        (directory / "rows.json").write_text(json.dumps(chunk_rows, indent=1) + "\n")
        shutil.rmtree(directory / "built")
        shutil.rmtree(directory / "states")
        rows += chunk_rows
        log_line(args.log, {"event": "chunk-done", "chunk": number, "documents": len(chunk)})
    report = {"model": Path(t_gguf).name, "source": s_name, "target": t_name, "hub": str(args.hub),
              "mean": str(args.mean) if args.mean else None, "tails": tails, "grid": args.grid, "translated": True,
              "strategies": [label for label, _, _ in strategies], "select": args.select, "documents": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(summary_text(summarize_reports([report])))
    return 0


# -- summarize -------------------------------------------------------------------------------

def paired_boot(differences, counts, seed=0):
    return pair_eval.boot(differences, counts, seed=seed)


def weighted(values, counts):
    """The token-weighted mean of per-document means."""
    return float(np.dot(values, counts) / np.sum(counts))


def summarize_reports(reports, *, baseline="zero"):
    """{kind: {tail: {label: {nll_delta, ci95, vs_zero, vs_zero_ci95, kl, kl_ci95, kl_doc_mean,
    kl_vs_zero, kl_vs_zero_ci95, kl_wins, top1, docs, tail_mean, tail_range, s_error, r_error}}}}
    over the documents of the reports (kind: chat, agent, all), and per tail a "sanity" entry.
    Means are token-weighted, CIs document-bootstrap (pair_eval.boot); kl_doc_mean is the plain
    mean over documents (a long document weighs no more than a short one); vs_zero: the paired
    difference to the baseline strategy on the same documents."""
    out = {}
    for kind in ("chat", "agent", "all"):
        docs = [d for report in reports for d in report["documents"] if kind == "all" or d["kind"] == kind]
        if not docs:
            continue
        tails = reports[0]["tails"]
        labels = reports[0]["strategies"]
        out[kind] = {}
        for tail in tails:
            table = {}
            counts = [d["scores"]["n_scored"] for d in docs]
            base = [d["scores"][f"{baseline}_{tail}"]["nll_sum_delta"] for d in docs] \
                if f"{baseline}_{tail}" in docs[0]["scores"] else None
            base_kl = [d["scores"][f"{baseline}_{tail}"]["kl"] * d["scores"]["n_scored"] for d in docs] \
                if base is not None else None
            for label in labels:
                name = f"{label}_{tail}"
                sums = [d["scores"][name]["nll_sum_delta"] for d in docs]
                value, ci = pair_eval.boot(sums, counts)
                kl_sums = [d["scores"][name]["kl"] * d["scores"]["n_scored"] for d in docs]
                kl, kl_ci = pair_eval.boot(kl_sums, counts)
                tails_of = [d["tail"][str(tail)] for d in docs]
                entry = {"nll_delta": value, "ci95": ci, "docs": len(docs), "kl": kl, "kl_ci95": kl_ci,
                         "kl_doc_mean": float(np.mean([d["scores"][name]["kl"] for d in docs])),
                         "top1": weighted([d["scores"][name]["top1"] for d in docs], counts),
                         "tail_mean": float(np.mean(tails_of)), "tail_range": [min(tails_of), max(tails_of)]}
                if base is not None and label != baseline:
                    entry["vs_zero"], entry["vs_zero_ci95"] = paired_boot(np.subtract(sums, base), counts)
                    entry["kl_vs_zero"], entry["kl_vs_zero_ci95"] = paired_boot(np.subtract(kl_sums, base_kl), counts)
                    entry["kl_wins"] = int(np.sum(np.asarray(kl_sums) < np.asarray(base_kl)))
                distance = [d["state_distance"].get(name, {}) for d in docs]
                for key in ("s", "r"):
                    values = [item[key] for item in distance if key in item]
                    if values:
                        entry[f"{key}_error"] = float(np.mean(values))
                table[label] = entry
            sanity = [(d["scores"][f"sanity_{tail}"], d["scores"]["n_scored"]) for d in docs
                      if f"sanity_{tail}" in d["scores"]]
            if sanity:
                scores, n = [item for item, _ in sanity], [count for _, count in sanity]
                table["sanity"] = {"max_abs": float(np.max(np.abs([item["nll_delta"] for item in scores]))),
                                   "nll_delta": weighted([item["nll_delta"] for item in scores], n),
                                   "kl": weighted([item["kl"] for item in scores], n),
                                   "top1": weighted([item["top1"] for item in scores], n), "docs": len(sanity)}
            out[kind][str(tail)] = table
    return out


def summary_text(summary):
    lines = []
    for kind, tails in summary.items():
        for tail, table in tails.items():
            sanity = table.get("sanity")
            lines.append(f"## {kind}, tail {tail}" + (
                f" (sanity over {sanity['docs']} docs: dNLL {sanity['nll_delta']:+.4f}, max |dNLL| "
                f"{sanity['max_abs']:.2g}, KL {sanity['kl']:.4f}, top-1 {sanity['top1']:.3f})" if sanity else ""))
            lines.append("| strategy | docs | mean tail (range) | dNLL [95% CI] | vs zero [95% CI] | KL [95% CI] | KL doc "
                         "mean | KL vs zero [95% CI] (docs below zero's) | top-1 | S rel. err | R rel. err |")
            lines.append("| --- | ---: | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |")
            for label, entry in table.items():
                if label == "sanity":
                    continue
                vs = (f"{entry['vs_zero']:+.4f} [{entry['vs_zero_ci95'][0]:+.4f}, {entry['vs_zero_ci95'][1]:+.4f}]"
                      if "vs_zero" in entry else "—")
                kl_vs = (f"{entry['kl_vs_zero']:+.4f} [{entry['kl_vs_zero_ci95'][0]:+.4f}, "
                         f"{entry['kl_vs_zero_ci95'][1]:+.4f}] ({entry['kl_wins']}/{entry['docs']})"
                         if "kl_vs_zero" in entry else "—")
                low, high = entry["tail_range"]
                lines.append(f"| {label} | {entry['docs']} | {entry['tail_mean']:.0f} ({low}-{high}) | "
                             f"{entry['nll_delta']:+.4f} [{entry['ci95'][0]:+.4f}, {entry['ci95'][1]:+.4f}] | {vs} | "
                             f"{entry['kl']:.4f} [{entry['kl_ci95'][0]:.4f}, {entry['kl_ci95'][1]:.4f}] | "
                             f"{entry['kl_doc_mean']:.4f} | {kl_vs} | "
                             f"{entry['top1']:.3f} | {entry.get('s_error', float('nan')):.3f} | "
                             f"{entry.get('r_error', float('nan')):.3f} |")
            lines.append("")
    return "\n".join(lines)


def read_report(path):
    """A report.json, or the report so far of an unfinished run directory (its finished chunks)."""
    path = Path(path)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if (path / "report.json").is_file():
        return json.loads((path / "report.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    plan = next(e for e in events if e["event"] in ("run-plan", "combined-plan"))
    rows = [row for chunk in sorted(path.glob("chunk-*/rows.json")) for row in json.loads(chunk.read_text())]
    return {"tails": plan["tails"], "strategies": plan["strategies"], "documents": rows, "partial": True}


def summarize(args):
    reports = [read_report(path) for path in args.reports]
    summary = summarize_reports(reports)
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(summary_text(summary))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub, *, resumable=True):
        sub.add_argument("--probe", type=Path, required=True, help="kvpark-kv-probe build (Vulkan for real models)")
        sub.add_argument("--device", default="vulkan", choices=("vulkan", "cpu"))
        sub.add_argument("--ctx", type=int, default=32768)
        sub.add_argument("--chunk", type=int, default=3, help="documents per probe run")
        sub.add_argument("--output", type=Path, required=True)
        resource_guard.add_arguments(sub)
        if resumable:
            sub.add_argument("--resume", action="store_true",
                             help="continue an output directory (finished chunks kept)")
        else:
            sub.set_defaults(resume=False)

    sub = commands.add_parser("calibrate", help="build the recurrent mean of a model (not resumable: the sums "
                                                "live in memory; start again in a new --output)")
    common(sub, resumable=False)
    sub.add_argument("--model", type=Path, required=True)
    sub.add_argument("--name", required=True, help="model name (the hub's)")
    sub.add_argument("--corpus", type=Path, action="append", required=True, help="conversations JSONL (train split used)")
    sub.add_argument("--exclude", type=Path, action="append", help="conversations JSONL whose ids are left out")
    sub.add_argument("--concat-chars", type=int, default=48000, help="characters per concatenated chat session")
    sub.add_argument("--min-depth", type=int, default=1024)
    sub.add_argument("--every", type=int, default=1024, help="tokens between saves")
    sub.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    sub.set_defaults(func=calibrate)

    for name, func, help_text in (("run", run, "native attention + each strategy"),
                                  ("combined", combined, "translated attention + each strategy")):
        sub = commands.add_parser(name, help=help_text)
        common(sub)
        if name == "run":
            sub.add_argument("--model", type=Path, required=True)
        else:
            sub.add_argument("--source", required=True, help="NAME=GGUF (NAME as in the hub)")
            sub.add_argument("--target", required=True, help="NAME=GGUF")
            sub.add_argument("--hub", type=Path, required=True)
            sub.add_argument("--engine-device", default="cpu", help="translate backend (cpu: numpy)")
        sub.add_argument("--mean", type=Path, help="recurrent-mean.npz of the target (calibrate)")
        sub.add_argument("--select", action="append", required=True, metavar="RUN_DIR:BUCKET")
        sub.add_argument("--per-bucket", type=int, help="documents per (run, bucket), spread over the prefix range")
        sub.add_argument("--sources", help="comma list of conversation sources to keep")
        sub.add_argument("--tails", default="1024,2048,4096")
        sub.add_argument("--grid", type=int, default=BATCH,
                         help=f"T0 = P - tail rounded down to this (default {BATCH}, the batch: sanity exactly 0; "
                              "1: exactly P - tail, as the engine does)")
        sub.add_argument("--strategies", default=DEFAULT_STRATEGIES,
                         help="comma list: recurrent_init strategies, NAME:GROUP for a mean group (match: the "
                              "document's own)")
        sub.add_argument("--max-scored", type=int, default=256)
        sub.set_defaults(func=func)

    sub = commands.add_parser("summarize", help="tables from report.json files")
    sub.add_argument("reports", nargs="+", help="report.json files or run directories (finished chunks)")
    sub.add_argument("--json", help="also write the summary JSON here")
    sub.set_defaults(func=summarize)
    args = parser.parse_args(argv)
    try:
        resource_guard.Settings.from_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except resource_guard.GuardError as error:
        sys.exit(f"r0_experiments: {error}")
