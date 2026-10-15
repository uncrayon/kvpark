"""Where the agent-session loss lives: summarize pair_eval --native-classes (report.json).

    python3 paper/scripts/decompose_stats.py RUN_DIR [--json OUT]

Per class (pair_eval.token_classes over the target's tokens in [H, T0): fill, the cells the hub has no estimate
for; tool_output, tool_call, reasoning, the mapped cells of that kind; recent, the 2,048 tokens before T0):
its share of the translated tokens, and the share of the remaining KL that giving the target its own native
values for that class removes, 1 - KL(<base>-CLASS) / KL(<base>), token-weighted over the scored tokens, with a
95% document-bootstrap interval; the same against the floor.
"""
import argparse
import json
import random
from pathlib import Path

CLASSES = ("fill", "tool_output", "tool_call", "reasoning", "recent")


def weighted(rows, name):
    return sum(r[name]["kl"] * r["n_scored"] for r in rows) / sum(r["n_scored"] for r in rows)


def share(rows, base, c):
    return 1 - weighted(rows, f"{base}-{c}") / weighted(rows, base)


def boot(rows, fn, draws=10000, seed=0):
    rng = random.Random(seed)
    values = sorted(fn([rng.choice(rows) for _ in rows]) for _ in range(draws))
    return [values[int(0.025 * draws)], values[int(0.975 * draws) - 1]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    report = json.loads((args.run / "report.json").read_text())
    rows = report["documents"]
    translated = sum(r["t0"] - r["H"] for r in rows)
    out = {"documents": len(rows), "scored_tokens": sum(r["n_scored"] for r in rows),
           "kl": {n: round(weighted(rows, n), 3) for n in ("hub", "floor", "hub_prefix", "floor_prefix")},
           "classes": {}}
    for c in CLASSES:
        tokens = sum(r["classes"][c] for r in rows)
        entry = {"tokens_share": round(tokens / translated, 3)}
        for base in ("hub", "floor"):
            entry[base] = {"kl": round(weighted(rows, f"{base}-{c}"), 3), "removed": round(share(rows, base, c), 3),
                           "ci95": [round(x, 3) for x in boot(rows, lambda s, b=base: share(s, b, c))]}
        out["classes"][c] = entry
    print(json.dumps(out, indent=1))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
