"""Summarize pair_eval reports with the campaign's statistics (campaign.py: token-weighted means with
document-bootstrap 95% intervals, the share of the floor's KL removed, exact sign-flip p).

    experiments/linear_transfer/.venv/bin/python paper/scripts/agent_stats.py REPORT_DIR [...] [--json OUT]

Each REPORT_DIR holds a pair_eval report.json. Prints one block per report; --json writes them all.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/universal_kv"))
import campaign  # noqa: E402

# (translated condition, its information-free floor)
SHARES = (("hub", "floor"), ("hub_prefix", "floor_prefix"), ("hub_table_prefix", "floor_prefix"))


def summarize(report):
    rows = report["documents"]
    names = set(campaign.condition_names(rows))
    out = {"documents": len(rows), "scored_tokens": sum(row["n_scored"] for row in rows),
           "conditions": campaign.conditions_of(rows), "shares": {}, "paired": campaign.paired_table(rows)}
    for hub, floor in SHARES:
        if hub in names and floor in names:
            share = campaign.recovered_share(rows, hub, floor)
            share["p"] = campaign.paired(rows, hub, floor, "kl")["p"]
            share["lower"] = campaign.paired(rows, hub, floor, "kl")["lower"]
            out["shares"][hub] = share
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    results = {}
    for directory in args.reports:
        report = json.loads((directory / "report.json").read_text())
        summary = results[directory.name] = summarize(report)
        print(f"== {directory.name}: {summary['documents']} documents, {summary['scored_tokens']} scored tokens")
        for name, item in summary["conditions"].items():
            print(f"  {name:18s} dNLL {item['nll_delta']:+.3f} [{item['nll_ci95'][0]:+.3f}, {item['nll_ci95'][1]:+.3f}]"
                  f"  KL {item['kl']:.3f} [{item['kl_ci95'][0]:.3f}, {item['kl_ci95'][1]:.3f}]  top1 {item['top1']:.3f}")
        for name, share in summary["shares"].items():
            if share["share"] is None:
                continue
            largest = share["largest"] or {}
            print(f"  share removed by {name:16s} {100 * share['share']:6.1f}% "
                  f"[{100 * share['ci95'][0]:.1f}, {100 * share['ci95'][1]:.1f}]  p {share['p']:.3g}  "
                  f"lower on {share['lower']}/{summary['documents']}"
                  + (f"  (largest doc {100 * largest['floor_share']:.0f}% of floor; without it "
                     f"{100 * largest['share_without']:.1f}%)" if largest.get("share_without") is not None else ""))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
