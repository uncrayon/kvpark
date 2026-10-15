"""Figures for the hub bundle's model card and release notes (PNG, light surface).

    ~/venvs/rocm/bin/python paper/scripts/fig_cards.py   (from the repository root)

Writes paper/figures/cards/{speed,models,agent,loss}.png from:
  speed   evidence/universal-kv/swe-agent-live/{qwen38-to-gemma-4-31b,gemma-4-31b-to-qwen38}/agent.jsonl with their
          guard logs (live_stats.py: native prefill against switch plus native tail, guard pauses removed)
  models  paper/tables/chat.tex (into each model from Qwen3.8, long held-out chats)
  agent   paper/data/pilot.json (hub6) and paper/data/agent-hub-qg.json (two-model hubs), into Gemma-4-31B
  loss    paper/data/decompose.json
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
import live_stats  # noqa: E402

OUT = ROOT / "paper" / "figures" / "cards"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated together, light)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURFACE,
                     "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})


def clean(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def speed():
    live = ROOT / "evidence/universal-kv/swe-agent-live"
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, (run, title) in zip(axes, (("qwen38-to-gemma-4-31b", "Into Gemma-4-31B (from Qwen3.8)"),
                                       ("gemma-4-31b-to-qwen38", "Into Qwen3.8-27B (from Gemma-4-31B)"))):
        rows = [json.loads(line) for line in (live / run / "agent.jsonl").read_text().splitlines() if line.strip()]
        rows = [r for r in rows if "error" not in r and r.get("bridged")]
        pauses = live_stats.pauses(live / f"{run}.guard.jsonl")
        tokens = [r["tokens"] / 1000 for r in rows]
        native = [r["native"]["prompt_ms"] / 1000 - live_stats.grown(pauses, r["native"]["clock"][0],
                                                                     r["native"]["prompt_ms"] / 1000) for r in rows]
        switch = [r["switch"]["timings"]["total_s"] - live_stats.paused_in(pauses, *r["switch"]["clock"])
                  + r["bridged"]["prompt_ms"] / 1000
                  - live_stats.grown(pauses, r["bridged"]["clock"][0], r["bridged"]["prompt_ms"] / 1000) for r in rows]
        ax.scatter(tokens, native, s=26, color=ORANGE, edgecolor=SURFACE, linewidth=1.2, label="Native prefill", zorder=3)
        ax.scatter(tokens, switch, s=26, color=BLUE, edgecolor=SURFACE, linewidth=1.2, label="Switch + native tail",
                   zorder=3)
        ratio = sorted(n / s for n, s in zip(native, switch))[len(native) // 2]
        ax.set_title(f"{title}\nmedian {ratio:.1f}× faster, 18 agent sessions", fontsize=9)
        ax.set_xlabel("Session length (thousand tokens)")
        clean(ax, "both")
    axes[0].set_ylabel("Seconds until the model can answer")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Switching models mid-session: time to resume", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "speed")


def models():
    rows = []
    for line in (ROOT / "paper/tables/chat.tex").read_text().splitlines():
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 4 or "\\%" not in cells[2]:
            continue
        head = cells[2].split("\\%")[0]
        lo, hi = cells[2].split("[")[1].split("]")[0].replace("$", "").split(",")
        name = cells[0].replace("~", " ")
        if name.startswith("Jamba"):
            continue                                    # not in the hub
        rows.append((name, cells[1], float(head), float(lo), float(hi)))
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    y = range(len(rows))
    ax.barh(list(y), [r[2] for r in rows], height=0.62, color=BLUE, edgecolor=SURFACE, linewidth=2)
    ax.errorbar([r[2] for r in rows], list(y), xerr=[[r[2] - r[3] for r in rows], [r[4] - r[2] for r in rows]],
                fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2)
    for i, r in enumerate(rows):
        ax.text(max(r[4], r[2]) + 1.5, i, f"{r[2]:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y), [f"{r[0]}  ·  {r[1]}" for r in rows], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the information-free cache's loss that translation removes (%)")
    ax.set_title("How much a translated cache helps depends on the cache family\n"
                 "Qwen3.8 → each model, long chats; 95% intervals (Mamba hybrids gain least)", fontsize=9.5)
    clean(ax, "x")
    fig.tight_layout()
    save(fig, "models")


def agent():
    pilot = json.loads((ROOT / "paper/data/pilot.json").read_text())
    hubs = json.loads((ROOT / "paper/data/agent-hub-qg.json").read_text())["qwen38-to-gemma-4-31b-agent"]["hub_prefix"]
    chat = pilot["qwen38-to-gemma-4-31b-server"]["shares"]["hub_prefix"]
    agent_hub6 = pilot["qwen38-to-gemma-4-31b-gemma31-dev40"]["shares"]["hub_prefix"]
    bars = [("Chat (6 long chats)\n15-model hub", chat["share"], chat["ci95"]),
            ("Coding-agent sessions (18)\n15-model hub", agent_hub6["share"], agent_hub6["ci95"]),
            ("Coding-agent sessions (18)\ntwo-model hub, fitted on chat", hubs["chat"]["removed"], hubs["chat"]["ci95"]),
            ("Coding-agent sessions (18)\ntwo-model hub, + 11 agent sessions", hubs["agent"]["removed"],
             hubs["agent"]["ci95"])]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    y = list(range(len(bars)))[::-1]
    values = [100 * b[1] for b in bars]
    ax.barh(y, values, height=0.6, color=BLUE, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(values, y, xerr=[[100 * (b[1] - b[2][0]) for b in bars], [100 * (b[2][1] - b[1]) for b in bars]],
                fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2)
    for yy, b in zip(y, bars):
        ax.text(100 * b[2][1] + 1.5, yy, f"{100 * b[1]:.0f}%", va="center", fontsize=8.5)
    ax.set_yticks(y, [b[0] for b in bars], fontsize=8)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Share of the information-free cache's loss that translation removes (%)")
    ax.set_title("Agent sessions are harder; fitting the hub on them helps\n"
                 "Qwen3.8 → Gemma-4-31B, as deployed (Gemma prefills its own system prompt); 95% intervals",
                 fontsize=9.5)
    clean(ax, "x")
    fig.tight_layout()
    save(fig, "agent")


def loss():
    d = json.loads((ROOT / "paper/data/decompose.json").read_text())["classes"]
    names = {"fill": "Template markup and\nspecial tokens (no hub estimate)", "reasoning": "Reasoning",
             "tool_output": "Tool output", "tool_call": "Tool-call arguments"}
    keys = list(names)[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    y = range(len(keys))
    tokens = [100 * d[k]["tokens_share"] for k in keys]
    removed = [100 * d[k]["hub"]["removed"] for k in keys]
    ax.barh([i + 0.2 for i in y], removed, height=0.38, color=BLUE, edgecolor=SURFACE, linewidth=2,
            label="Loss removed by restoring the class")
    ax.barh([i - 0.2 for i in y], tokens, height=0.38, color=ORANGE, edgecolor=SURFACE, linewidth=2,
            label="Share of the translated tokens")
    for i, (t, r) in enumerate(zip(tokens, removed)):
        ax.text(r + 1, i + 0.2, f"{r:.0f}%", va="center", fontsize=8)
        ax.text(t + 1, i - 0.2, f"{t:.0f}%", va="center", fontsize=8, color=INK2)
    ax.set_yticks(list(y), [names[k] for k in keys], fontsize=8)
    ax.set_xlim(0, 60)
    ax.set_xlabel("%")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("Where the loss of a switch into Gemma-4-31B lives\n"
                 "12 agent sessions; an oracle restores one class of tokens at a time", fontsize=9.5)
    clean(ax, "x")
    fig.tight_layout()
    save(fig, "loss")


if __name__ == "__main__":
    speed()
    models()
    agent()
    loss()
    print(sorted(p.name for p in OUT.glob("*.png")))
