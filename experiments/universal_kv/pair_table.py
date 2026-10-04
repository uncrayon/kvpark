"""Markdown table of pair_eval.py reports: token-weighted delta NLL [95% CI], KL and top-1 per condition."""

import json
import sys
from pathlib import Path

CONDITIONS = ("hub", "floor", "hub_prefix", "floor_prefix", "hub_table_prefix")


def cell(summary, name):
    if name not in summary:
        return "—"
    item = summary[name]
    ci = item.get("ci95") or [None, None]
    return f"{item['nll_delta']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] · KL {item.get('kl', float('nan')):.2f}"


def main(paths):
    print("| Source → target | Docs | " + " | ".join(CONDITIONS) + " |")
    print("| --- | ---: | " + " | ".join("---" for _ in CONDITIONS) + " |")
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        summary = report["summary"]
        print(f"| {report['source']} → {report['target']} | {len(report['documents'])} | "
              + " | ".join(cell(summary, name) for name in CONDITIONS) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
