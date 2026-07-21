"""
Figure 3 — Codebook structure → circuit polarity (two panels).

Panel (a): Paired bar chart — induced variants (capacity sweep + FSQ).
           Base vs. L0-ablation yes-rate, colored by polarity class.
Panel (b): Scatter — natural specimens, codebook mean yes-rate (x) vs.
           POPE behavioral baseline yes-rate (y). Colored by the same
           polarity palette. Diagonal y=x shows codebook-tracks-behavior.

Run from repo root:
    source activate sae && python scripts/gen_codebook_polarity_bars.py

Outputs:
    figures/k_sweep/codebook_polarity_bars.png  (200 dpi, for paper)
    figures/k_sweep/codebook_polarity_bars.pdf  (vector, for camera-ready)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIG_OUT = ROOT / "figures" / "k_sweep"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
C_NEUTRAL     = "#969696"
C_SUPPRESS    = "#4575b4"
C_PROMOTE     = "#d73027"
C_NEUTRAL_L0  = "#cccccc"
C_SUPPRESS_L0 = "#abd9e9"
C_PROMOTE_L0  = "#f4a582"
C_DIAG        = "#999999"

POLARITY_COLORS = {
    "neutral":     (C_NEUTRAL,  C_NEUTRAL_L0),
    "suppressing": (C_SUPPRESS, C_SUPPRESS_L0),
    "promoting":   (C_PROMOTE,  C_PROMOTE_L0),
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── data ──────────────────────────────────────────────────────────────────────
variants = [
    dict(label="$K{=}1024$",            base=32.4, l0=32.9, polarity="neutral"),
    dict(label="$K{=}4096$",            base=79.8, l0=93.2, polarity="promoting"),
    dict(label="$K{=}16384$",           base=85.5, l0=14.1, polarity="suppressing"),
    dict(label="$K{=}65536$",           base=87.2, l0=64.4, polarity="suppressing"),
    dict(label="FSQ v1\n[8,8,8,5,5,5]", base=2.6,  l0=61.0, polarity="promoting"),
    dict(label="FSQ v2\n[7,7,7,6,5,5]", base=1.4,  l0=54.7, polarity="promoting"),
]

# natural specimens: (codebook mean yes-rate, behavioral POPE yes-rate, polarity)
# polarity matches codebook bias direction (same convention as panel a)
scatter_models = [
    dict(label="VILA-U",    x=0.7224, y=0.8100, polarity="promoting",  marker="o"),
    dict(label="Liquid",    x=0.7460, y=0.7817, polarity="promoting",  marker="s"),
    dict(label="Chameleon", x=0.2439, y=0.2230, polarity="suppressing", marker="^"),
]

# ── figure: 2 panels, width ratio ~5:3 ───────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(
    1, 2,
    figsize=(12.5, 4.2),
    gridspec_kw={"width_ratios": [5, 3], "wspace": 0.32},
)

# ════════════════════════════════════════════════════════════════════════════
# Panel (a) — induced variants bar chart
# ════════════════════════════════════════════════════════════════════════════
n       = len(variants)
group_w = 0.70
bar_w   = group_w / 2
x_ctrs  = np.arange(n)

for i, v in enumerate(variants):
    c_base, c_l0 = POLARITY_COLORS[v["polarity"]]
    xb = x_ctrs[i] - bar_w / 2
    xl = x_ctrs[i] + bar_w / 2

    ax_a.bar(xb, v["base"], width=bar_w, color=c_base,
             align="edge", zorder=2, linewidth=0)
    ax_a.bar(xl, v["l0"],  width=bar_w, color=c_l0,
             align="edge", zorder=2, linewidth=0.6,
             edgecolor="white", hatch="///")

    delta = v["l0"] - v["base"]
    sign  = "+" if delta > 0 else ""
    top   = max(v["base"], v["l0"])
    ax_a.text(x_ctrs[i], top + 1.5, f"{sign}{delta:.1f}",
              ha="center", va="bottom", fontsize=7,
              color=c_base if abs(delta) > 2 else "#666666")

ax_a.axvline(3.5, color="#cccccc", lw=0.8, ls="--", zorder=1)
ax_a.text(1.5, 100, "VQ + Linear  (capacity sweep)",
          ha="center", fontsize=7.5, color="#555555", style="italic")
ax_a.text(4.5, 100, "FSQ  (no collapse)",
          ha="center", fontsize=7.5, color="#555555", style="italic")

ax_a.set_xlim(-0.55, n - 0.45)
ax_a.set_ylim(0, 103)
ax_a.set_xticks(x_ctrs)
ax_a.set_xticklabels([v["label"] for v in variants], fontsize=8.5)
ax_a.set_ylabel("POPE yes-rate (%)", fontsize=9)
ax_a.set_yticks([0, 25, 50, 75, 100])
ax_a.yaxis.grid(True, lw=0.4, color="#dddddd", zorder=0)
ax_a.set_axisbelow(True)
ax_a.set_title("(a)  Induced variants", fontsize=10, pad=6)

# ════════════════════════════════════════════════════════════════════════════
# Panel (b) — natural specimens scatter
# ════════════════════════════════════════════════════════════════════════════
ax_b.plot([0, 1], [0, 1], color=C_DIAG, lw=0.8, ls="--", zorder=0)

label_offsets = {
    "VILA-U":    ( 0.025,  0.018, "left",  "bottom"),
    "Liquid":    ( 0.025, -0.055, "left",  "top"),
    "Chameleon": ( 0.025,  0.018, "left",  "bottom"),
}

for m in scatter_models:
    c = POLARITY_COLORS[m["polarity"]][0]
    ax_b.scatter(m["x"], m["y"], color=c, marker=m["marker"],
                 s=65, zorder=3, linewidths=0.5, edgecolors="white")
    xo, yo, ha, va = label_offsets[m["label"]]
    ax_b.text(m["x"] + xo, m["y"] + yo, m["label"],
              fontsize=8, ha=ha, va=va, color=c, fontweight="bold")

ax_b.set_xlim(-0.03, 1.03)
ax_b.set_ylim(-0.03, 1.03)
ax_b.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
ax_b.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
ax_b.set_xlabel("Codebook mean yes-rate", fontsize=9)
ax_b.set_ylabel("Behavioral baseline yes-rate\n(POPE-full, $n$=3000)", fontsize=9)
ax_b.set_title("(b)  Natural specimens", fontsize=10, pad=6)

# ── shared legend below both panels ──────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor="#888888",  label="Base"),
    mpatches.Patch(facecolor="#cccccc",  hatch="///", edgecolor="white",
                   label="$L_0$ ablation"),
    mpatches.Patch(facecolor=C_NEUTRAL,  label="Neutral"),
    mpatches.Patch(facecolor=C_SUPPRESS, label="Suppressing"),
    mpatches.Patch(facecolor=C_PROMOTE,  label="Promoting"),
    mlines.Line2D([], [], color=C_DIAG, ls="--", lw=0.8, label="$y = x$"),
]
fig.legend(handles=legend_handles, fontsize=8, ncol=6,
           loc="lower center", bbox_to_anchor=(0.5, -0.07),
           frameon=False, columnspacing=1.0, handlelength=1.2)

fig.tight_layout(rect=[0, 0.05, 1, 1])

for ext in ("png", "pdf"):
    out = FIG_OUT / f"codebook_polarity_bars.{ext}"
    fig.savefig(out)
    print(f"saved {out}")

plt.close(fig)
