"""Draw the paper's figures from experiments/data/results_v2.json.

    uv run --group experiments python paper/make_figures.py [path/to/results_v2.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "experiments/data/results_v2.json"
R = json.loads(SRC.read_text())
OUT = ROOT / "figures"

# attacker colours: categorical slots 1-3, validated as a set (all pairs)
ATT = {"A1": ("#2a78d6", "TF-IDF"), "A2": ("#eb6834", "embeddings"), "A3": ("#1baf7a", "fine-tuned")}
INK, INK2, MUTED, GRID = "#14171c", "#4a5160", "#7d838f", "#e2e5ea"
LABS = ["Meta", "Google", "Alibaba", "Microsoft", "Mistral"]
CONDITIONS = ["direct", "persona_fwd", "nodes_1", "nodes_2", "nodes_3",
              "plain_1", "plain_2", "plain_3", "fixed_1", "plain_1_hot"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "font.size": 9, "axes.titlesize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def row(cond: str, att: str, mode: str = "adaptive") -> dict:
    return next(r for r in R["rows"] if r["condition"] == cond and r["attacker"] == att and r["mode"] == mode)


def pct(v: float) -> float:
    return 100 * v


def pct_axis(ax, axis="y", top=100, step=20):
    ticks = range(0, top + 1, step)
    (ax.set_yticks if axis == "y" else ax.set_xticks)(ticks, [f"{t}%" for t in ticks])


def mono(name: str) -> str:
    """Condition name in typewriter type (cmr10 has no underscore glyph)."""
    return r"$\mathtt{" + name.replace("_", r"\_") + "}$"


def attacker_legend(fig, y):
    handles = [plt.Line2D([], [], marker="o", ls="-", color=ATT[a][0], mec="white", label=ATT[a][1]) for a in ATT]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.55, y), fontsize=8)


def fig_accuracy() -> None:
    fig, ax = plt.subplots(figsize=(6.3, 3.7))
    ys = np.arange(len(CONDITIONS))[::-1].astype(float)
    ax.axvline(20, color=MUTED, lw=0.9, zorder=1)
    for sep in (1.5, 4.5, 7.5):  # between groups: baseline | persona chain | plain chain | single-rewrite baselines
        ax.axhline(len(CONDITIONS) - 1 - sep, color=GRID, lw=0.8, zorder=0)
    for att, off in [("A1", 0.2), ("A2", 0.0), ("A3", -0.2)]:
        color = ATT[att][0]
        for y, c in zip(ys, CONDITIONS):
            r = row(c, att)
            ax.plot([pct(r["ci"][0]), pct(r["ci"][1])], [y + off] * 2, color=color, lw=1.5, alpha=0.55,
                    solid_capstyle="round", zorder=2)
            ax.plot(pct(r["acc"]), y + off, "o", ms=4, color=color, mec="white", mew=0.7, zorder=3)
    ax.set_xlim(0, 100)
    pct_axis(ax, "x")
    ax.set_yticks(ys, [mono(c) for c in CONDITIONS], fontsize=8.5)
    ax.set_ylim(-0.6, len(CONDITIONS) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("attribution accuracy, adaptive attacker (chance 20%)")
    attacker_legend(fig, 1.03)
    fig.savefig(OUT / "accuracy.pdf")
    plt.close(fig)


def fig_hops() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for ax, (chain, start, title) in zip(axes, [("nodes", "persona_fwd", "(a) persona prompt, then hops"),
                                                ("plain", "direct", "(b) hops only")]):
        conds = [start] + [f"{chain}_{k}" for k in (1, 2, 3)]
        x = np.arange(4)
        for att in ATT:
            rs = [row(c, att) for c in conds]
            color = ATT[att][0]
            ax.fill_between(x, [pct(r["ci"][0]) for r in rs], [pct(r["ci"][1]) for r in rs], color=color, alpha=0.12, lw=0)
            ax.plot(x, [pct(r["acc"]) for r in rs], color=color, lw=1.6, marker="o", ms=4, mec="white", mew=0.7)
        ax.axhline(20, color=MUTED, lw=0.9, zorder=0)
        ax.text(2.5, 17.5, "chance", color=MUTED, fontsize=7.5, ha="center", va="top")
        ax.set_xticks(x, [mono(start), "exit", "middle", "guard"], fontsize=8)
        ax.set_xlim(-0.2, 3.2)
        ax.set_ylim(0, 100)
        pct_axis(ax)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, color=INK, loc="left")
    attacker_legend(fig, 1.08)
    fig.tight_layout()
    fig.savefig(OUT / "hops.pdf")
    plt.close(fig)


def fig_curve() -> None:
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    n_train = 640
    styles = {"direct": ":", "plain_3": "--", "nodes_3": "-"}
    series: dict[tuple, list] = {}
    for c in R["curves"]:
        series.setdefault((c["attacker"], c["condition"]), []).append(c)
    for (att, cond), pts in sorted(series.items()):
        pts = sorted(pts, key=lambda p: p["frac"])
        x = [p["frac"] * n_train for p in pts]
        y = [pct(p["acc"]) for p in pts]
        color = ATT[att][0]
        ax.fill_between(x, [pct(p["ci"][0]) for p in pts], [pct(p["ci"][1]) for p in pts], color=color, alpha=0.10, lw=0)
        ax.plot(x, y, ls=styles[cond], color=color, lw=1.6, marker="o", ms=3.5, mec="white", mew=0.6)
        ax.annotate(f"{ATT[att][1]}, " + mono(cond), (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)
    ax.axhline(20, color=MUTED, lw=0.9, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([160, 320, 640], ["160", "320", "640"])
    ax.minorticks_off()
    ax.set_xlim(140, 1500)
    ax.set_ylim(0, 100)
    pct_axis(ax)
    ax.set_xlabel("training texts per fold")
    ax.set_ylabel("accuracy")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "curve.pdf")
    plt.close(fig)


def fig_forest() -> None:
    hyp = R["hypotheses"]
    order = ["H7", "H8", "H9_1", "H9_2", "H9_3", "H10", "H11", "H12"]
    label = {"H7": "H7   fine-tuned − TF-IDF on " + mono("nodes_3"),
             "H8": "H8   100% − 50% of training data, " + mono("nodes_3"),
             "H9_1": "H9   " + mono("nodes_1") + " − " + mono("plain_1"),
             "H9_2": "H9   " + mono("nodes_2") + " − " + mono("plain_2"),
             "H9_3": "H9   " + mono("nodes_3") + " − " + mono("plain_3"),
             "H10": "H10  " + mono("plain_3") + " − " + mono("plain_1"),
             "H11": "H11  " + mono("fixed_1") + " − " + mono("plain_1"),
             "H12": "H12  " + mono("plain_1_hot") + " − " + mono("plain_1")}
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    ys = np.arange(len(order))[::-1]
    ax.axvline(0, color=MUTED, lw=0.9)
    for y, h in zip(ys, order):
        d, lo, hi = pct(hyp[h]["diff"]), pct(hyp[h]["ci"][0]), pct(hyp[h]["ci"][1])
        ok = bool(R["verdicts"][h])
        ax.plot([lo, hi], [y, y], color=INK, lw=1.6, alpha=0.5, solid_capstyle="round")
        ax.plot(d, y, "o", ms=5, color=INK if ok else "white", mec=INK, mew=1.2)
    ax.set_yticks(ys, [label[h] for h in order], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("difference in accuracy (percentage points), paired bootstrap 95% CI")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=INK, label="supported"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="not supported")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.6, 1.04), fontsize=7.5)
    fig.savefig(OUT / "forest.pdf")
    plt.close(fig)


def fig_confusion() -> None:
    conds = ["direct", "nodes_3", "plain_3", "fixed_1"]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.1))
    n = R["n_units"] // 5
    for k, (ax, cond) in enumerate(zip(axes, conds)):
        m = np.array(R["confusion"][cond])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=n)
        for i in range(5):
            for j in range(5):
                v = m[i, j]
                ax.text(j, i, v, ha="center", va="center", fontsize=6.3,
                        color="white" if v > n / 2 else INK, fontweight="bold" if i == j else "normal")
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=0.6, alpha=0.6))
        ax.set_xticks(range(5), [l[:4] for l in LABS], rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(5), LABS if k == 0 else [], fontsize=6.5)
        ax.set_title(f"({'abcd'[k]}) " + mono(cond), color=INK, loc="left", fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("true Z", fontsize=7.5)
    fig.supxlabel("predicted", fontsize=7.5, y=-0.04)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "confusion.pdf")
    plt.close(fig)


def fig_utility() -> None:
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for v, s in [(R["floor"], "floor"), (R["ceiling"], "ceiling")]:
        ax.axvline(v, color=MUTED, lw=0.9, zorder=0)
        ax.text(v - 0.001, 98, s, color=MUTED, fontsize=7.5, ha="right", va="top")
    ax.axhline(20, color=MUTED, lw=0.9, zorder=0)
    ax.text(0.9635, 21.2, "chance", color=MUTED, fontsize=7.5, ha="right", va="bottom")
    for c in CONDITIONS[1:]:
        r = row(c, R["best"][c])
        u = R["utility"][c]["cos"]
        persona = c.startswith(("nodes", "persona"))
        ax.plot([u, u], [pct(r["ci"][0]), pct(r["ci"][1])], color=INK, lw=1.4, alpha=0.45, solid_capstyle="round")
        ax.plot(u, pct(r["acc"]), "o", ms=5, color=INK if persona else "white", mec=INK, mew=1.1)
        ax.annotate(mono(c), (u, pct(r["acc"])), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=INK)
    ax.set_xlim(0.845, 0.965)
    ax.set_ylim(0, 100)
    pct_axis(ax)
    ax.set_xlabel("cosine similarity to Z's unprotected answer")
    ax.set_ylabel("accuracy, strongest attacker")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=INK, label="with persona prompt"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="without")]
    ax.legend(handles=handles, loc="upper left", fontsize=7.5)
    fig.savefig(OUT / "utility.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_accuracy()
    fig_hops()
    fig_curve()
    fig_forest()
    fig_confusion()
    fig_utility()
    print("figures written to", OUT.relative_to(ROOT.parent), "from", SRC.name)
