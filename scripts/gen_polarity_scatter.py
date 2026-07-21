"""
fig:polarity — codebook mean yes-rate vs. behavioral baseline yes-rate.

Five-point scatter: VILA-U, Liquid, Chameleon (plotted), plus FSQ v1 and FSQ v2
annotated on the y-axis only (no codebook bias measurement, so no x-coordinate).

Run from repo root:
    source activate sae && python scripts/gen_polarity_scatter.py

Outputs:
    figures/polarity/polarity_scatter.png  (200 dpi, for paper)
    figures/polarity/polarity_scatter.pdf  (vector, for camera-ready)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIG_OUT = ROOT / "figures" / "polarity"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# ── palette (matches gen_overview_figure.py) ─────────────────────────────────────
C_YES  = "#2166ac"   # yes-biased natural specimens
C_NO   = "#b2182b"   # no-biased natural specimen (Chameleon)
C_FSQ  = "#4dac26"   # FSQ annotations (yes-suppressing)
C_DIAG = "#999999"   # diagonal reference line

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# ── data ─────────────────────────────────────────────────────────────────────
# x = codebook mean yes-rate (from codebook_bias_*.json)
# y = POPE baseline yes-rate (from bench_pope_full_*_baseline.jsonl)
scatter_models = [
    dict(label="VILA-U",    x=0.7224, y=0.8100, color=C_YES, marker="o"),
    dict(label="Liquid",    x=0.7460, y=0.7817, color=C_YES, marker="s"),
    dict(label="Chameleon", x=0.2439, y=0.2230, color=C_NO,  marker="^"),
]

# FSQ recipes: y-axis annotation only — no codebook bias measurement
fsq_recipes = [
    dict(label="FSQ-v1\n(64k codes)", y=0.0263),
    dict(label="FSQ-v2\n(22k codes)", y=0.0137),
]

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.8, 3.5))

# diagonal y=x reference
ax.plot([0, 1], [0, 1], color=C_DIAG, lw=0.8, ls="--", zorder=0,
        label="$y = x$")

# scatter points
for m in scatter_models:
    ax.scatter(m["x"], m["y"], color=m["color"], marker=m["marker"],
               s=70, zorder=3, linewidths=0.5, edgecolors="white")
    # label offset: nudge each point to avoid overlap
    # default: right of point, slightly above
    xoff, yoff, ha, va = 0.020, 0.022, "left", "bottom"
    if m["label"] == "VILA-U":
        # right of marker, above — Liquid is below-right so no collision
        xoff, yoff, ha, va = 0.022, 0.018, "left", "bottom"
    elif m["label"] == "Liquid":
        # right of marker, below VILA-U label
        xoff, yoff, ha, va = 0.022, -0.055, "left", "top"
    elif m["label"] == "Chameleon":
        # place above-right; below collides with legend
        xoff, yoff, ha, va = 0.022, 0.022, "left", "bottom"
    ax.text(m["x"] + xoff, m["y"] + yoff, m["label"],
            fontsize=8, ha=ha, va=va, color=m["color"], fontweight="bold")

# FSQ annotations on the y-axis only.
# v1 and v2 are very close (y=0.026, 0.014) so we label them together as a
# bracketed annotation rather than two separate rules.
y1, y2 = fsq_recipes[0]["y"], fsq_recipes[1]["y"]

# draw a single bracket on the y-axis spanning both FSQ values
ax.axhline(y1, xmin=0, xmax=1, color=C_FSQ, lw=0.7, ls=":", zorder=1, alpha=0.8)
ax.axhline(y2, xmin=0, xmax=1, color=C_FSQ, lw=0.7, ls=":", zorder=1, alpha=0.8)

# vertical bracket line in the right margin
ymid = (y1 + y2) / 2
ax.annotate(
    "",
    xy=(1.04, y2), xycoords=("axes fraction", "data"),
    xytext=(1.04, y1), textcoords=("axes fraction", "data"),
    arrowprops=dict(arrowstyle="-", color=C_FSQ, lw=0.8),
)
# single combined label at the midpoint
ax.text(1.06, ymid,
        "FSQ-v1 (2.6%)\nFSQ-v2 (1.4%)",
        transform=ax.get_yaxis_transform(),
        fontsize=7, va="center", ha="left", color=C_FSQ, linespacing=1.4)

# axes
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("Codebook mean yes-rate", fontsize=9)
ax.set_ylabel("Behavioral baseline yes-rate\n(POPE-full, $n$=3000)", fontsize=9)
ax.set_title("Codebook polarity vs. behavioural yes-rate", fontsize=10)

# tick marks at clean intervals
ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])

# legend for marker shapes
handles = [
    mlines.Line2D([], [], color=C_YES, marker="o", ls="", ms=6, label="VILA-U"),
    mlines.Line2D([], [], color=C_YES, marker="s", ls="", ms=6, label="Liquid"),
    mlines.Line2D([], [], color=C_NO,  marker="^", ls="", ms=6, label="Chameleon"),
    mlines.Line2D([], [], color=C_DIAG, ls="--", lw=0.8, label="$y = x$"),
]
# upper-left quadrant is empty — park the legend there
ax.legend(handles=handles, fontsize=7, loc="upper left",
          bbox_to_anchor=(0.02, 0.98),
          frameon=True, framealpha=0.85, edgecolor="none")

fig.tight_layout(rect=[0, 0, 0.88, 1])   # leave right margin for FSQ labels

for ext in ("png", "pdf"):
    out = FIG_OUT / f"polarity_scatter.{ext}"
    fig.savefig(out)
    print(f"saved {out}")

plt.close(fig)
