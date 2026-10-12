"""Draw the v1 figures (80 prompts) from experiments/data/results.json.

    uv run --group experiments python paper/make_figures_v1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = json.loads((ROOT.parent / "experiments/data/results.json").read_text())
OUT = ROOT / "figures" / "v1"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#14171c", "#4a5160", "#7d838f", "#e2e5ea"
LABS = ["Meta", "Google", "Alibaba", "Microsoft", "Mistral"]
CONDITIONS = ["direct", "persona_fwd", "z_mode", "nodes_1", "nodes_2", "nodes_3", "para_1"]
HOPS = ["persona_fwd", "nodes_1", "nodes_2", "nodes_3"]
HOP_LABELS = ["Z forward", "exit", "middle", "guard"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "font.size": 9, "axes.titlesize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def row(cond: str, feat: str, att: str) -> dict:
    """Accuracy row; for `direct` the static attacker is the adaptive one."""
    if cond == "direct":
        att = "adaptive"
    return next(r for r in RESULTS["rows"]
                if r["condition"] == cond and r["features"] == feat and r["attacker"] == att)


def pct(v: float) -> float:
    return 100 * v


def fig_accuracy() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    ys = np.arange(len(CONDITIONS))[::-1]
    for ax, att in zip(axes, ["adaptive", "static"]):
        ax.axvline(20, color=MUTED, lw=0.9, zorder=1)
        for feat, color, off in [("tfidf", BLUE, 0.14), ("embed", ORANGE, -0.14)]:
            for y, c in zip(ys, CONDITIONS):
                r = row(c, feat, att)
                ax.plot([pct(r["ci"][0]), pct(r["ci"][1])], [y + off] * 2, color=color, lw=1.6, alpha=0.55,
                        solid_capstyle="round", zorder=2)
                ax.plot(pct(r["acc"]), y + off, "o", ms=4.2, color=color, mec="white", mew=0.8, zorder=3)
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{att.capitalize()} attacker", color=INK, loc="left")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys, CONDITIONS, family="monospace", fontsize=8)
    axes[0].set_ylim(-0.6, len(CONDITIONS) - 0.4)
    handles = [plt.Line2D([], [], marker="o", ls="-", color=BLUE, mec="white", label="style (TF-IDF)"),
               plt.Line2D([], [], marker="o", ls="-", color=ORANGE, mec="white", label="meaning (embeddings)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.06), fontsize=8)
    fig.savefig(OUT / "accuracy.pdf")
    plt.close(fig)


def fig_hops() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.4))
    x = np.arange(len(HOPS))
    series = [("tfidf", "adaptive", BLUE, "-", "style, adaptive"), ("tfidf", "static", BLUE, "--", "style, static"),
              ("embed", "adaptive", ORANGE, "-", "meaning, adaptive"), ("embed", "static", ORANGE, "--", "meaning, static")]
    for feat, att, color, ls, label in series:
        acc = [pct(row(h, feat, att)["acc"]) for h in HOPS]
        guard = [float(row(h, feat, att)["to_guard"].rstrip("%")) for h in HOPS]
        face = color if ls == "-" else "white"
        a.plot(x, acc, ls=ls, color=color, lw=1.5, marker="o", ms=4, mfc=face, mec=color, label=label)
        b.plot(x, guard, ls=ls, color=color, lw=1.5, marker="o", ms=4, mfc=face, mec=color)
    for ax, top, ref_label, title in [(a, 100, "chance", "(a) accuracy at identifying Z"),
                                      (b, 40, "uniform guess", "(b) predictions naming the guard")]:
        ax.axhline(20, color=MUTED, lw=0.9, zorder=0)
        below = top > 50
        ax.text(1.5, 20 + (-1 if below else 1) * top * 0.025, ref_label, color=MUTED, fontsize=7.5,
                ha="center", va="top" if below else "bottom")
        ax.set_ylim(0, top)
        ticks = range(0, top + 1, 20 if top > 50 else 10)
        ax.set_yticks(ticks, [f"{t}%" for t in ticks])
        ax.set_xticks(x, HOP_LABELS)
        ax.set_xlim(-0.25, len(HOPS) - 0.75)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, color=INK, loc="left")
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=7.5, handlelength=2.2)
    fig.tight_layout()
    fig.savefig(OUT / "hops.pdf")
    plt.close(fig)


def fig_confusion() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.75))
    for ax, (cond, title) in zip(axes, [("direct", "(a) unprotected"), ("nodes_3", "(b) after three hops")]):
        m = np.array(RESULTS["confusion"][cond])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=80)
        for i in range(5):
            for j in range(5):
                v = m[i, j]
                ax.text(j, i, v, ha="center", va="center", fontsize=8,
                        color="white" if v > 40 else INK, fontweight="bold" if i == j else "normal")
        for i in range(5):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=0.7, alpha=0.6))
        ax.set_xticks(range(5), LABS, rotation=30, ha="right", fontsize=7.5)
        ax.set_yticks(range(5), LABS, fontsize=7.5)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_title(title, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("true Z", fontsize=8)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT / "confusion.pdf")
    plt.close(fig)


def fig_utility() -> None:
    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    floor, ceil = RESULTS["floor"], RESULTS["ceiling"]
    for v, s in [(floor, "floor"), (ceil, "ceiling")]:
        ax.axvline(v, color=MUTED, lw=0.9, zorder=0)
        ax.text(v - 0.001, 98, s, color=MUTED, fontsize=7.5, ha="right", va="top")
    direct = pct(row("direct", "tfidf", "adaptive")["acc"])
    for v, s in [(20, "chance"), (direct, f"direct ({direct:.1f}%)")]:
        ax.axhline(v, color=MUTED, lw=0.9, zorder=0)
        if v == 20:
            ax.text(0.9635, v + 1.2, s, color=MUTED, fontsize=7.5, ha="right", va="bottom")
        else:
            ax.text(0.8465, v - 1.2, s, color=MUTED, fontsize=7.5, ha="left", va="top")
    place = {"persona_fwd": (6, 0, "left"), "z_mode": (6, 0, "left"), "nodes_1": (6, 2, "left"),
             "nodes_2": (6, -3, "left"), "nodes_3": (-6, -2, "right"), "para_1": (-6, 0, "right")}
    for c, u in RESULTS["utility"].items():
        r = row(c, "tfidf", "adaptive")
        ax.plot([u, u], [pct(r["ci"][0]), pct(r["ci"][1])], color=BLUE, lw=1.6, alpha=0.55, solid_capstyle="round")
        ax.plot(u, pct(r["acc"]), "o", ms=5, color=BLUE, mec="white", mew=0.8)
        dx, dy, ha = place[c]
        ax.annotate(c, (u, pct(r["acc"])), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    family="monospace", fontsize=7.5, color=INK)
    ax.set_xlim(0.845, 0.965)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_xlabel("cosine similarity to Z's unprotected answer")
    ax.set_ylabel("attribution accuracy (style, adaptive)")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "utility.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_accuracy()
    fig_hops()
    fig_confusion()
    fig_utility()
    print("figures written to", OUT.relative_to(ROOT.parent))
