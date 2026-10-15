"""Figure: held-out R^2 of translated K and V between every pair of hub6 models.

    ~/venvs/rocm/bin/python paper/scripts/fig_r2.py   (from the repository root)

Reads evidence/universal-kv/hub6-20261009-summary/r2-summary.json, writes paper/figures/r2.pdf.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "evidence/universal-kv/hub6-20261009-summary/r2-summary.json"
OUT = ROOT / "paper/figures/r2.pdf"
LABELS = {"qwen38": "Qwen3.8-27B", "muse": "Muse-30B", "qwen3-4b": "Qwen3-4B", "ornith": "Ornith-9B",
          "lfm2-2.6b": "LFM2-2.6B", "deepseek-v2-lite": "DeepSeek-V2-Lite", "qwen3.6-35b-a3b": "Qwen3.6-35B",
          "glm-4.7-flash": "GLM-4.7-Flash", "granite-4.0-h-tiny": "Granite-4.0-H", "falcon-h1-3b": "Falcon-H1-3B",
          "nemotron-3.5-lightning-30b-a3b": "Nemotron-3.5", "kimi-linear-48b-a3b": "Kimi-Linear-48B",
          "ling-3.0-tiny": "Ling-3.0", "gemma-4-26b-a4b": "Gemma-4-26B", "gemma-4-31b": "Gemma-4-31B"}
# the reference palette's sequential blue, steps 100 -> 700
RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def main():
    data = json.loads(SUMMARY.read_text())
    models = data["models"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), constrained_layout=True)
    for ax, kind in zip(axes, ("k", "v")):
        matrix = np.array([[data[kind]["matrix"][f"{a}->{b}"] for b in models] for a in models])
        image = ax.imshow(matrix, cmap=RAMP, vmin=0.2, vmax=0.95)
        for i in range(len(models)):
            for j in range(len(models)):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.2f}"[1:] if value < 1 else "1", ha="center", va="center", fontsize=5.5,
                        color="white" if value > 0.62 else "#1f1f1f")
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#1f1f1f", linewidth=0.8))
        ax.axhline(3.5, color="white", linewidth=1.5)
        ax.axvline(3.5, color="white", linewidth=1.5)
        names = [LABELS[m] for m in models]
        ax.set_xticks(range(len(models)), names, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(models)), names if kind == "k" else [""] * len(models), fontsize=7)
        ax.set_title({"k": "Keys", "v": "Values"}[kind], fontsize=10)
        ax.set_xlabel("target", fontsize=8)
        if kind == "k":
            ax.set_ylabel("source", fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = fig.colorbar(image, ax=axes, shrink=0.7, pad=0.01)
    bar.set_label("held-out $R^2$", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=160)
    print(OUT)


if __name__ == "__main__":
    main()
