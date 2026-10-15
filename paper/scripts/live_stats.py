"""Summarize live agent hand-offs (live_bridge.py agent: agent.jsonl).

    python3 paper/scripts/live_stats.py RUN_DIR [...] [--json OUT]

Per run: how often the next tool call after a switch matches the one after a native prefill (same tool; same
tool and the same target: the file a read/edit/write touches or the command run; identical arguments), how often
each matches the call the session actually made, the reuse (cache_n == T0), and the time to the first generated
token: native prefill against the switch (bridge total, the target's native prompt start included) plus the
target's prefill of the native tail, with the speed-up per session. With the run's guard log beside it
(RUN.guard.jsonl), the same times without the guard's pauses that fall inside each step ("net").
"""
import argparse
import datetime
import json
import statistics
from pathlib import Path

TARGET_KEYS = ("filepath", "path", "file", "command", "node", "query")


def target(call):
    """The tool and what it acts on (the first of TARGET_KEYS among its arguments), else just the tool."""
    arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    key = next((k for k in TARGET_KEYS if k in arguments), None)
    value = arguments.get(key) if key else None
    if isinstance(value, str) and key == "command":
        value = value.strip().split("\n")[0].split("|")[0].strip()       # the command's first stage
    return (call.get("name"), key, value)


def compare(a, b):
    names = [c["name"] for c in a] == [c["name"] for c in b]
    targets = names and [target(c) for c in a] == [target(c) for c in b]
    same = names and [c["arguments"] for c in a] == [c["arguments"] for c in b]
    return names, targets, same


def pauses(path):
    """[(start, end)] epoch seconds of the guard's pauses (its resume events carry paused_s)."""
    out = []
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "resume" and event.get("paused_s"):
            end = datetime.datetime.fromisoformat(event["time"]).timestamp()
            out.append((end - event["paused_s"], end))
    return out


def paused_in(intervals, start, end):
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals)


def grown(intervals, start, seconds):
    """The pause time inside a step of `seconds` (pauses included) that began at `start`."""
    return paused_in(intervals, start, start + seconds)


def summarize(rows, intervals=()):
    ok = [r for r in rows if "error" not in r and r.get("bridged")]
    out = {"sessions": len(rows), "errors": len(rows) - len(ok),
           "reuse_ok": sum(bool(r["bridged"].get("reuse_ok")) for r in ok)}
    for label, a, b in (("bridged_vs_native", "bridged", "native"),):
        trio = [compare(r[a]["tool_calls"], r[b]["tool_calls"]) for r in ok]
        out[label] = {"same_tool": sum(t[0] for t in trio), "same_target": sum(t[1] for t in trio),
                      "identical": sum(t[2] for t in trio)}
    for case in ("native", "bridged"):
        trio = [compare(r[case]["tool_calls"], r["gold"]) for r in ok]
        out[f"{case}_vs_session"] = {"same_tool": sum(t[0] for t in trio), "same_target": sum(t[1] for t in trio),
                                     "identical": sum(t[2] for t in trio)}
    native = [r["native"]["prompt_ms"] / 1000 for r in ok]
    switched = [r["switch"]["timings"]["total_s"] + r["bridged"]["prompt_ms"] / 1000 for r in ok]
    engine = [r["switch"]["timings"]["engine_s"] for r in ok]
    cold = [r["switch"]["timings"].get("native_prefix_s") or 0 for r in ok]
    speedups = [n / s for n, s in zip(native, switched)]
    if intervals:
        native_net = [r["native"]["prompt_ms"] / 1000 - grown(intervals, r["native"]["clock"][0], r["native"]["prompt_ms"] / 1000)
                      for r in ok]
        switch_net = [r["switch"]["timings"]["total_s"]
                      - paused_in(intervals, *r["switch"]["clock"])
                      + r["bridged"]["prompt_ms"] / 1000
                      - grown(intervals, r["bridged"]["clock"][0], r["bridged"]["prompt_ms"] / 1000) for r in ok]
        net_speedups = [n / s for n, s in zip(native_net, switch_net)]
        out["timing_net"] = {"native_prefill_s": [round(min(native_net), 1), round(statistics.median(native_net), 1),
                                                  round(max(native_net), 1)],
                             "switch_plus_tail_s": [round(min(switch_net), 1), round(statistics.median(switch_net), 1),
                                                    round(max(switch_net), 1)],
                             "speedup": [round(min(net_speedups), 2), round(statistics.median(net_speedups), 2),
                                         round(max(net_speedups), 2)]}
    out["timing"] = {"native_prefill_s": [round(min(native), 1), round(statistics.median(native), 1), round(max(native), 1)],
                     "switch_plus_tail_s": [round(min(switched), 1), round(statistics.median(switched), 1),
                                            round(max(switched), 1)],
                     "engine_s": [round(min(engine), 2), round(statistics.median(engine), 2), round(max(engine), 2)],
                     "speedup": [round(min(speedups), 2), round(statistics.median(speedups), 2), round(max(speedups), 2)],
                     "native_prefix_prefills": sum(1 for c in cold if c > 1.0),
                     "tokens": [min(r["tokens"] for r in ok), max(r["tokens"] for r in ok)],
                     "t0": [min(r["switch"]["t0"] for r in ok), max(r["switch"]["t0"] for r in ok)]}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    results = {}
    for run in args.runs:
        rows = [json.loads(line) for line in (run / "agent.jsonl").read_text().splitlines() if line.strip()]
        results[run.name] = summarize(rows, pauses(run.parent / f"{run.name}.guard.jsonl"))
        print(run.name, json.dumps(results[run.name], indent=1))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
