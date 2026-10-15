"""Chat-fitted against chat+agent-fitted two-model hubs (evidence/universal-kv/swe-agent-hub/eval/*/report.json:
pair_eval with --hub hub-chat-v2 --extra-hub agent=hub-chatagent-v2).

    python3 paper/scripts/agent_hub_stats.py EVAL_DIR [--json OUT]

Per evaluation and condition (hub, hub_prefix): token-weighted KL of each hub and its own floor, the share of the
floor's KL each removes with a 95% document-bootstrap interval, the change in KL from adding agent sessions with its
interval, and an exact paired sign-flip p (all 2^n sign patterns, or 200,000 random ones above n = 17) on the
per-document KL differences.
"""
import argparse
import itertools
import json
import random
from pathlib import Path


def weighted(rows, name):
    return sum(r[name]["kl"] * r["n_scored"] for r in rows) / sum(r["n_scored"] for r in rows)


def share(rows, hub, floor):
    return 1 - weighted(rows, hub) / weighted(rows, floor)


def boot(rows, fn, draws=10000, seed=0):
    rng = random.Random(seed)
    values = sorted(fn([rng.choice(rows) for _ in rows]) for _ in range(draws))
    return [round(values[int(0.025 * draws)], 3), round(values[int(0.975 * draws) - 1], 3)]


def sign_flip(diffs, seed=0):
    """Two-sided p of mean(diffs) under random sign flips."""
    observed = abs(sum(diffs))
    n = len(diffs)
    if n <= 17:
        patterns = itertools.product((1, -1), repeat=n)
        total = 2 ** n
    else:
        rng = random.Random(seed)
        total = 200000
        patterns = ([rng.choice((1, -1)) for _ in range(n)] for _ in range(total))
    hits = sum(abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12 for signs in patterns)
    return hits / total


def summarize(rows):
    out = {"documents": len(rows), "scored_tokens": sum(r["n_scored"] for r in rows)}
    for base in ("hub", "hub_prefix"):
        floor = base.replace("hub", "floor")
        chat, agent = (base, floor), (f"{base}.agent", f"{floor}.agent")
        diffs = [(r[base]["kl"] - r[f"{base}.agent"]["kl"]) * r["n_scored"] for r in rows]
        out[base] = {
            "chat": {"kl": round(weighted(rows, chat[0]), 3), "floor_kl": round(weighted(rows, chat[1]), 3),
                     "removed": round(share(rows, *chat), 3), "ci95": boot(rows, lambda s: share(s, *chat))},
            "agent": {"kl": round(weighted(rows, agent[0]), 3), "floor_kl": round(weighted(rows, agent[1]), 3),
                      "removed": round(share(rows, *agent), 3), "ci95": boot(rows, lambda s: share(s, *agent))},
            "kl_change": round(1 - weighted(rows, agent[0]) / weighted(rows, chat[0]), 3),
            "kl_change_ci95": boot(rows, lambda s: 1 - weighted(s, agent[0]) / weighted(s, chat[0])),
            "agent_better_documents": sum(d > 0 for d in diffs),
            "sign_flip_p": round(sign_flip(diffs), 4)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    results = {}
    for report in sorted(args.eval_dir.glob("*/report.json")):
        results[report.parent.name] = summarize(json.loads(report.read_text())["documents"])
    print(json.dumps(results, indent=1))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
