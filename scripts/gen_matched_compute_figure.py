"""
3-panel bar chart for the matched-compute contrast (Table 2 → Figure).

Panels:
  (a) A₁ peak   — VQ installs the routing bottleneck; MLP does not
  (b) A₂ L₀ mass — both conditions equal (matched compute)
  (c) POPE yes-rate (base and L₀ ablation) — VQ circuit is causal

Run from repo root:
    source activate sae && python scripts/gen_matched_compute_figure.py

Output:
    figures/matched_compute/matched_compute_3panel.png
    figures/matched_compute/matched_compute_3panel.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).parent.parent
FIG_OUT = ROOT / "figures" / "matched_compute"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# ── colour palette (matches gen_hero_figure.py) ──────────────────────────────
C_VQ  = "#762a83"   # induced VQ
C_MLP = "#969696"   # MLP / continuous control
C_BASE = "#d73027"  # base (pre-ablation) bar
C_L0   = "#4575b4"  # L0-ablation bar

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

# ── data from Table 2 ─────────────────────────────────────────────────────────
CONDITIONS  = ["VQ+Linear\n(K=16 384)", "MLP\n(matched-compute)"]
A1_VALS     = [1.28, 0.21]
A2_VALS     = [26.0, 25.5]
YR_BASE     = [85.5, 13.7]   # POPE yes-rate at baseline
YR_L0       = [14.1,  8.7]   # POPE yes-rate after L0 ablation

X = np.array([0.0, 1.0])
W = 0.34   # bar width for the grouped yes-rate panel

COND_COLORS = [C_VQ, C_MLP]


def _value_label(ax, x, y, txt, color, fs=8.5, dy=0.03, fmt_bold=True):
    ax.text(x, y + dy, txt, ha="center", va="bottom",
            fontsize=fs, color=color,
            fontweight="bold" if fmt_bold else "normal")


# ─────────────────────────────────────────────────────────────────────────────
# Panel A  —  A₁ peak
# ─────────────────────────────────────────────────────────────────────────────
def draw_a1(ax):
    bars = ax.bar(X, A1_VALS, color=COND_COLORS, width=0.50,
                  alpha=0.88, edgecolor="white", linewidth=0.5, zorder=3)

    # threshold line at A₁ = 0.8 (gate pass threshold)
    thresh = 0.80
    ax.axhline(thresh, color="#aaaaaa", ls="--", lw=1.0, zorder=2)
    ax.text(1.29, thresh + 0.04, "pass threshold (0.80)",
            ha="right", va="bottom", fontsize=7.0, color="#888888")

    # checkmarks / cross
    MARKS = ["✓", "✗"]
    MARK_COLORS = ["#2ca02c", "#d62728"]
    for xi, val, col, mk, mc in zip(X, A1_VALS, COND_COLORS, MARKS, MARK_COLORS):
        _value_label(ax, xi, val, f"{val:.2f}", col)
        ax.text(xi, val + 0.22, mk, ha="center", va="bottom",
                fontsize=13, color=mc, fontweight="bold")

    ax.set_xticks(X)
    ax.set_xticklabels(CONDITIONS, fontsize=8.5)
    ax.set_ylabel("A₁ peak  (relative divergence)", fontsize=8.5)
    ax.set_ylim(0, 1.85)
    ax.set_xlim(-0.45, 1.45)
    ax.set_title("(a)  A₁ peak\n"
                 r"$\it{VQ\ installs\ the\ routing\ bottleneck}$",
                 fontsize=9.5, pad=6, fontweight="bold")

    ax.text(0.5, A1_VALS[0] + 0.38, "6× larger", ha="center",
            fontsize=8.0, color="#333333", fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# Panel B  —  A₂ L₀ mass  (the "control" panel — nearly equal)
# ─────────────────────────────────────────────────────────────────────────────
def draw_a2(ax):
    bars = ax.bar(X, A2_VALS, color=COND_COLORS, width=0.50,
                  alpha=0.88, edgecolor="white", linewidth=0.5, zorder=3)

    for xi, val, col in zip(X, A2_VALS, COND_COLORS):
        _value_label(ax, xi, val, f"{val:.1f}%", col)

    ax.set_xticks(X)
    ax.set_xticklabels(CONDITIONS, fontsize=8.5)
    ax.set_ylabel("A₂ L₀ attention mass (%)", fontsize=8.5)
    ax.set_ylim(0, 35)
    ax.set_xlim(-0.45, 1.45)
    ax.set_title("(b)  A₂ L₀ mass\n"
                 r"$\it{matched\ — single\ variable\ contrast}$",
                 fontsize=9.5, pad=6, fontweight="bold")

    ax.text(0.5, max(A2_VALS) + 3.5, "≈ equal  (matched compute)",
            ha="center", fontsize=7.5, color="#777777", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# Panel C  —  POPE yes-rate: base and L₀ ablation grouped bars
# ─────────────────────────────────────────────────────────────────────────────
def draw_yesrate(ax):
    x_vq  = X[0]
    x_mlp = X[1]

    # VQ bars
    ax.bar(x_vq  - W/2, YR_BASE[0], width=W, color=C_BASE,  alpha=0.85,
           edgecolor="white", linewidth=0.5, zorder=3, label="Baseline")
    ax.bar(x_vq  + W/2, YR_L0[0],  width=W, color=C_L0,   alpha=0.85,
           edgecolor="white", linewidth=0.5, zorder=3, label="L₀ ablation")

    # MLP bars
    ax.bar(x_mlp - W/2, YR_BASE[1], width=W, color=C_BASE,  alpha=0.85,
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x_mlp + W/2, YR_L0[1],  width=W, color=C_L0,   alpha=0.85,
           edgecolor="white", linewidth=0.5, zorder=3)

    # value labels
    for xi, yb, yl in zip(X, YR_BASE, YR_L0):
        _value_label(ax, xi - W/2, yb, f"{yb:.1f}%", C_BASE, fs=8.0, dy=1.0)
        _value_label(ax, xi + W/2, yl,  f"{yl:.1f}%",  C_L0,  fs=8.0, dy=1.0)

    delta_vq  = YR_BASE[0] - YR_L0[0]
    delta_mlp = YR_BASE[1] - YR_L0[1]
    ax.text(x_vq + 0.05, (YR_BASE[0] + YR_L0[0]) / 2 + 4,
            f"−{delta_vq:.1f} pp",
            ha="center", fontsize=8.5, color=C_L0, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=C_L0, alpha=0.92, lw=1.1))
    ax.text(x_mlp + 0.05, 30,
            f"−{delta_mlp:.1f} pp",
            ha="center", fontsize=8.5, color="#666666",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#aaaaaa", alpha=0.92, lw=1.0))

    ax.set_xticks(X)
    ax.set_xticklabels(CONDITIONS, fontsize=8.5)
    ax.set_ylabel("POPE yes-rate (%)", fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.55, 1.55)
    ax.legend(fontsize=8.0, loc="upper right", framealpha=0.90)
    ax.set_title("(c)  POPE yes-rate  (base → L₀ ablation)\n"
                 r"$\it{VQ\ circuit\ is\ causal}$",
                 fontsize=9.5, pad=6, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# Compose
# ─────────────────────────────────────────────────────────────────────────────
def make_figure():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4),
                             gridspec_kw={"wspace": 0.40,
                                          "left": 0.07, "right": 0.97,
                                          "top": 0.84,  "bottom": 0.16})

    draw_a1(axes[0])
    draw_a2(axes[1])
    draw_yesrate(axes[2])

    fig.suptitle(
        "Matched-compute contrast: same data, steps, LR, frozen LM — only projector type differs",
        fontsize=10.5, fontweight="bold", y=0.98)

    for suffix in (".png", ".pdf"):
        out = FIG_OUT / f"matched_compute_3panel{suffix}"
        fig.savefig(out, dpi=200 if suffix == ".png" else None,
                    bbox_inches="tight")
        print(f"  saved → {out}")

    plt.close(fig)


if __name__ == "__main__":
    make_figure()
