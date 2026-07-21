"""
Three-panel hero figure for the paper (teaser / page-1 overview).

Panel A (left):   L0 routing circuit — conceptual schematic
Panel B (middle): Gate-status colored grid (~22 models × 3 gates)
Panel C (right):  CHAIRi dissociation — VILA-U, 4 methods

Run from repo root:
    source activate sae && python scripts/gen_overview_figure.py

Output:
    figures/hero/hero_3panel.png   (200 dpi, for paper)
    figures/hero/hero_3panel.pdf   (vector, for camera-ready)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIG_OUT = ROOT / "figures" / "hero"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# ── colour palette (matches gen_all_figures.py) ──────────────────────────────
C_VQ_NAT  = "#2166ac"   # natural VQ specimens
C_INDUCED = "#762a83"   # induced VQ
C_CTRL    = "#969696"   # continuous controls
C_PASS    = "#2ca02c"   # gate pass (green)
C_FAIL    = "#d62728"   # gate fail (red)
C_BASE    = "#d73027"   # baseline bar
C_L0      = "#4575b4"   # L0 ablation
C_DOLA    = "#f46d43"   # DoLA
C_VCD     = "#fdae61"   # VCD

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   9,
    "axes.titlesize": 10,
    "figure.dpi":  150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

CLASS_COLOR = {"vq_spec": C_VQ_NAT, "induced": C_INDUCED, "ctrl": C_CTRL}


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def _box(ax, x, y, w, h, fc, ec, text, fs=8, fw="normal", lw=1.0, tc=None):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08", linewidth=lw,
        edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fs,
            color=tc or ec, fontweight=fw, zorder=3)


def _arrow(ax, x0, y0, x1, y1, color, lw=1.2, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->", color=color, lw=lw,
                    connectionstyle=f"arc3,rad={rad:.2f}"),
                zorder=4)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL A — L0 routing circuit schematic
# ═══════════════════════════════════════════════════════════════════════════

def draw_circuit_schematic(ax):
    """
    Top-down flow with L0 at the TOP of the transformer (nearest input):

        [Image]
           ↓
      [VQ Codebook]  (collapsed, biased codes)
           ↓  snapping bias
     v₁  v₂  v₃ … vₙ   │   ?  ← prompt-last
         │  │  │  └──── ↘ ↗  attention
     ┌── L0 ══════════════════ visual→prompt_last sink ─┐
     │   L1                                              │
     │   L2                                              │
     │   …  (32 layers total)                            │
     └───────────────────────────────────────────────────┘
           ↓
       [→ YES  (biased output)]
    """
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 13.5)
    ax.axis("off")

    # ── image ────────────────────────────────────────────────────────────
    _box(ax, 1.4, 12.65, 3.2, 0.70, "#eeeeee", "#888888",
         "Image  (visual input)", fs=8, tc="#444444")
    _arrow(ax, 3.0, 12.65, 3.0, 12.10, "#888888")

    # ── VQ codebook ───────────────────────────────────────────────────────
    _box(ax, 0.55, 11.35, 4.90, 0.72, "#f2e0f7", "#762a83",
         "VQ Codebook  (collapsed, ~80 active codes)",
         fs=7.5, fw="bold", lw=1.8, tc="#5a1680")
    _arrow(ax, 3.0, 11.35, 3.0, 10.75, "#762a83")
    ax.text(4.1, 11.05, "snapping bias →", ha="left", va="center",
            fontsize=6.5, color="#762a83", style="italic")

    # ── token row ─────────────────────────────────────────────────────────
    tok_y  = 10.35
    tok_xs = [0.50, 1.25, 2.00, 2.75]
    for xi in tok_xs:
        _box(ax, xi - 0.30, tok_y - 0.26, 0.60, 0.52,
             "#d4e8f7", C_VQ_NAT, "v", fs=7.5, fw="bold", tc="white")
    _box(ax, 3.20, tok_y - 0.26, 0.52, 0.52,
         "white", "#aaaaaa", "…", fs=9, tc="#888888")
    # prompt-last token (outside transformer, at right)
    _box(ax, 4.35, tok_y - 0.26, 1.50, 0.52,
         "#fff8d6", "#d73027", "?  prompt-last", fs=6.5, fw="bold", lw=1.5)
    ax.text(1.65, tok_y + 0.42, "visual tokens",
            ha="center", fontsize=7, color="#555555")
    ax.text(5.10, tok_y + 0.42, "answer\nposition",
            ha="center", fontsize=6.0, color="#d73027")

    # short arrows from tokens into transformer (downward)
    for xi in tok_xs:
        _arrow(ax, xi, tok_y - 0.26, xi, tok_y - 0.60, C_VQ_NAT, lw=0.8)

    # ── transformer block ─────────────────────────────────────────────────
    tf_x, tf_w = 0.20, 5.60
    tf_top = tok_y - 0.65        # just below token row
    tf_bot = 1.65                # higher floor → more space for YES box
    tf_h   = tf_top - tf_bot

    ax.add_patch(mpatches.FancyBboxPatch(
        (tf_x, tf_bot), tf_w, tf_h,
        boxstyle="round,pad=0.12", linewidth=0.9,
        edgecolor="#cccccc", facecolor="#f8f8f8", alpha=0.50, zorder=0))
    ax.text(tf_x + tf_w / 2, tf_top + 0.20,
            "Shared Transformer  (32 layers)",
            ha="center", fontsize=7, color="#666666", style="italic")

    # layer bands — L0 at TOP (i=0 → highest y), descending
    n_show = 6
    band_h = 0.52
    usable = tf_h - 0.30
    gap    = (usable - n_show * band_h) / max(n_show - 1, 1)

    layer_centers = []
    for i in range(n_show):
        # top-down: i=0 is L0 near the input
        ly = tf_top - 0.16 - (i + 1) * band_h - i * gap
        cy = ly + band_h / 2
        layer_centers.append(cy)

        is_l0   = (i == 0)
        is_last = (i == n_show - 1)
        fc = "#fff3b0" if is_l0 else "#f0f0f0"
        ec = "#d73027" if is_l0 else "#cccccc"
        lw = 2.4 if is_l0 else 0.6
        ax.add_patch(mpatches.FancyBboxPatch(
            (tf_x + 0.25, ly), tf_w - 0.50, band_h,
            boxstyle="round,pad=0.04", linewidth=lw,
            edgecolor=ec, facecolor=fc, zorder=1, alpha=0.92))

        if is_l0:
            ax.text(tf_x + 0.46, cy,
                    "L0  ← routing layer",
                    va="center", fontsize=8.5, color="#d73027",
                    fontweight="bold", zorder=2)
        elif is_last:
            ax.text(tf_x + 0.46, cy,
                    "L5  ⋯  L31",
                    va="center", fontsize=7.5, color="#aaaaaa", zorder=2)
        else:
            ax.text(tf_x + 0.46, cy,
                    f"L{i}",
                    va="center", fontsize=7.5, color="#cccccc", zorder=2)

    # ── attention arrows inside L0: visual token columns → prompt-last ────
    l0_cy  = layer_centers[0]        # center-y of L0 band
    l0_ly  = l0_cy - band_h / 2     # bottom of L0 band
    pl_x   = tf_x + tf_w - 0.28     # prompt-last column inside transformer

    for k, xi in enumerate(tok_xs):
        rad = 0.10 + 0.05 * k
        ax.annotate("",
                    xy   =(pl_x, l0_cy),
                    xytext=(xi, l0_cy),
                    arrowprops=dict(
                        arrowstyle="->", color=C_L0, lw=1.2,
                        connectionstyle=f"arc3,rad={rad:.2f}"),
                    zorder=5)

    # "sink" label inside L0 at prompt-last column
    ax.text(pl_x + 0.05, l0_cy + 0.18,
            "sink\n(all 32 heads)", ha="left", fontsize=6.2,
            color=C_L0, va="center", zorder=6)

    # connecting line: prompt-last token (above) ↕ sink (inside L0)
    ax.annotate("",
                xy   =(pl_x, l0_cy + band_h / 2),
                xytext=(5.10, tok_y - 0.26),
                arrowprops=dict(arrowstyle="->", color="#d73027",
                                lw=1.1, linestyle="dashed",
                                connectionstyle="arc3,rad=0.0"),
                zorder=4)

    # ── answer token output ───────────────────────────────────────────────
    _arrow(ax, 3.0, tf_bot, 3.0, 0.85, "#d73027", lw=1.4)
    _box(ax, 1.4, 0.20, 3.2, 0.62, "#fde8e8", "#d73027",
         "→  YES  (biased output)", fs=9, fw="bold", lw=1.8)

    ax.set_title("(a)  L₀ routing circuit", fontsize=9.5, pad=5, fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════════
# PANEL B — gate-status colored cell grid
# ═══════════════════════════════════════════════════════════════════════════

# (model, A1, A2, A3, yes_rate, class)
# Sorted: PASS first (by class), then FAIL
GATE_DATA = [
    # ── natural VQ — all 3 gates PASS ──────────────────────────────────────
    ("VILA-U",          True,  True,  True,  0.810, "vq_spec"),
    ("Liquid",          True,  True,  True,  0.782, "vq_spec"),
    ("Chameleon",       True,  True,  True,  0.223, "vq_spec"),
    ("Anole",           True,  True,  True,  0.753, "vq_spec"),
    ("Lumina-mGPT",     True,  True,  True,  0.001, "vq_spec"),
    # ── induced VQ — all 3 gates PASS ──────────────────────────────────────
    ("LLaVA-VQ K16384", True,  True,  True,  0.855, "induced"),
    ("LLaVA-VQ K65536", True,  True,  True,  0.872, "induced"),
    ("LLaVA-VQ K4096",  True,  True,  True,  0.798, "induced"),   # YES-suppress anomaly
    ("LLaVA-VQ-FSQ v1", True,  True,  True,  0.026, "induced"),
    ("LLaVA-VQ-FSQ v2", True,  True,  True,  0.014, "induced"),
    # ── FAIL models — at least one gate fails ──────────────────────────────
    ("LLaVA-MLP",  False, True,  True,  0.000, "ctrl"),      # A1 FAIL (VQ bottleneck absent)
    ("LLaVA-1.6",       False, False, False, 0.573, "ctrl"),
    ("Janus-Pro",       True,  True,  False, 0.444, "ctrl"),
    ("Emu3",            False, True,  True,  0.327, "ctrl"),
    ("UniTok",          False, False, True,  0.475, "ctrl"),
    ("Show-o",          False, False, True,  0.984, "ctrl"),
    ("SEED",            False, False, True,  0.396, "ctrl"),
    ("HaploOmni",       False, False, False, 0.148, "ctrl"),
    ("LaVIT",           False, False, False, 0.973, "ctrl"),
    ("Qwen2.5-VL",      False, False, False, 0.242, "ctrl"),
    ("Qwen-VQ",         False, True,  True,  0.003, "induced"),   # collapse without circuit
    ("LLaVA-VQ K1024",  False, True,  True,  0.324, "induced"),
]

# Gate labels shown in column headers
GATE_LABELS = ["A₁\nrel_div", "A₂\nattn mass", "A₃\noff-manifold"]


def draw_gate_grid(ax):
    n = len(GATE_DATA)

    # coordinate system: x ∈ [−0.6, 4.5], y ∈ [−0.6, n + 0.8]
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(-0.65, n + 0.85)
    ax.axis("off")

    # column headers
    for ci, lbl in enumerate(GATE_LABELS):
        ax.text(ci, n + 0.55, lbl, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="#333333")
    ax.text(3.55, n + 0.55, "yes-rate", ha="left", va="center",
            fontsize=7.0, color="#666666")

    # PASS / FAIL separator — label in the yes-rate column (x≈3.9) to avoid cell overlap
    n_pass = sum(1 for r in GATE_DATA if all(r[1:4]))
    sep_y  = n - n_pass - 0.5
    ax.axhline(sep_y, color="#aaaaaa", lw=1.2, ls="--",
               xmin=0.00, xmax=0.98)
    # "8 PASS" label above the line, "14 FAIL" below — placed in the yes-rate margin
    ax.text(4.55, sep_y + 0.40, f"← {n_pass} PASS", fontsize=6.5,
            color="#2ca02c", fontweight="bold", va="center", ha="right")
    ax.text(4.55, sep_y - 0.40, f"← {n - n_pass} FAIL", fontsize=6.5,
            color="#d62728", fontweight="bold", va="center", ha="right")

    for ri, (model, a1, a2, a3, yr, cls) in enumerate(GATE_DATA):
        y        = n - 1 - ri
        all_pass = a1 and a2 and a3
        cls_col  = CLASS_COLOR[cls]

        # model label
        fw = "bold" if all_pass else "normal"
        ax.text(-0.52, y, model, ha="right", va="center",
                fontsize=7.0, color=cls_col, fontweight=fw)

        # gate cells (colored rectangles)
        for ci, gate_pass in enumerate([a1, a2, a3]):
            fc = "#d5f5d5" if gate_pass else "#fadadd"
            ec = C_PASS if gate_pass else C_FAIL
            ax.add_patch(plt.Rectangle(
                (ci - 0.44, y - 0.40), 0.88, 0.80,
                linewidth=0.9, edgecolor=ec, facecolor=fc, zorder=2))
            sym = "✓" if gate_pass else "✗"
            ax.text(ci, y, sym, ha="center", va="center",
                    fontsize=10, color=ec, fontweight="bold", zorder=3)

        # yes-rate mini-bar
        bar_w = yr * 0.85
        bar_fc = cls_col if all_pass else "#dddddd"
        ax.add_patch(plt.Rectangle(
            (3.05, y - 0.20), bar_w, 0.40,
            facecolor=bar_fc, alpha=0.55, linewidth=0))
        ax.text(3.07 + bar_w, y, f" {yr:.0%}", ha="left", va="center",
                fontsize=6.2, color="#555555")

        # subtle row highlight for PASS models
        if all_pass:
            ax.axhspan(y - 0.44, y + 0.44, color=cls_col, alpha=0.04, zorder=0)

    # legend — placed inside the axes at bottom-right of data area
    leg = [
        mpatches.Patch(color=C_VQ_NAT,  label="Natural VQ specimen"),
        mpatches.Patch(color=C_INDUCED, label="Induced VQ"),
        mpatches.Patch(color=C_CTRL,    label="Continuous / control"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=6.5, framealpha=0.90)

    ax.set_title("(b)  3-gate diagnostic — full cohort",
                 fontsize=9.5, pad=5, fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════════
# PANEL C — CHAIRi dissociation on VILA-U
# ═══════════════════════════════════════════════════════════════════════════

# Data from the baseline, L0, and tuned-baseline runs
CHAIR_DATA = {
    "Baseline":    10.3,
    "L0\nablation":  7.1,
    "Tuned\nDoLA":   9.9,
    "Tuned\nVCD":   10.7,
}
CHAIR_COLORS = [C_BASE, C_L0, C_DOLA, C_VCD]


def draw_chair_bars(ax):
    methods = list(CHAIR_DATA.keys())
    vals    = list(CHAIR_DATA.values())
    x       = np.arange(len(methods))

    bars = ax.bar(x, vals, color=CHAIR_COLORS, alpha=0.88,
                  edgecolor="white", linewidth=0.5, width=0.60, zorder=3)

    # baseline reference dashed line
    ax.axhline(10.3, color=C_BASE, ls="--", lw=0.9, alpha=0.45, zorder=2)

    # value labels on bars
    for xi, val, col in zip(x, vals, CHAIR_COLORS):
        ax.text(xi, val + 0.20, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8.5,
                color=col, fontweight="bold")

    # ── headline annotation: L0 is the only method that helps ────────────
    # bracket arrow from baseline to L0
    ax.annotate("",
                xy   =(x[1], vals[1] + 0.15),
                xytext=(x[0], vals[0] + 0.15),
                arrowprops=dict(
                    arrowstyle="->", color=C_L0, lw=1.8,
                    connectionstyle="arc3,rad=-0.30"),
                zorder=5)
    ax.text(0.5, 8.05,
            "−31 % relative\n(only method that\nreduces CHAIRi)",
            ha="center", va="center", fontsize=7.8, color=C_L0,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.30", fc="white",
                      ec=C_L0, alpha=0.92, lw=1.2))

    # annotation for DoLA / VCD
    ax.text(2.5, 11.55,
            "win binary (POPE/AMBER)\nbut ✗ generation CHAIRi",
            ha="center", fontsize=7.0, color="#888888", style="italic",
            bbox=dict(boxstyle="round,pad=0.20", fc="white",
                      ec="#cccccc", alpha=0.85))
    ax.annotate("", xy=(x[2], vals[2] + 0.20), xytext=(2.5, 11.45),
                arrowprops=dict(arrowstyle="->", color="#bbbbbb", lw=0.9))
    ax.annotate("", xy=(x[3], vals[3] + 0.20), xytext=(2.5, 11.45),
                arrowprops=dict(arrowstyle="->", color="#bbbbbb", lw=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8.5)
    ax.set_ylabel("CHAIRi (%)  ↓  lower = better", fontsize=8.5)
    ax.set_ylim(0, 14.5)
    ax.set_xlim(-0.55, 3.75)

    ax.set_title(
        "(c)  VILA-U CHAIRi — binary vs generation dissociation",
        fontsize=9.5, pad=5, fontweight="bold")

    # # source note
    # ax.text(3.72, 0.25, ",",
    #         ha="right", fontsize=6.0, color="#aaaaaa")


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSE
# ═══════════════════════════════════════════════════════════════════════════

def make_hero():
    n_models = len(GATE_DATA)

    # Height driven by gate-grid row count (≈ 0.34 in per row)
    fig_h = max(8.0, n_models * 0.34 + 2.0)
    fig   = plt.figure(figsize=(17.5, fig_h))

    gs = GridSpec(
        1, 3, figure=fig,
        width_ratios=[3.0, 3.5, 3.2],
        wspace=0.28,
        left=0.05, right=0.97,
        top=0.94,  bottom=0.08,
    )

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    draw_circuit_schematic(ax_a)
    draw_gate_grid(ax_b)
    draw_chair_bars(ax_c)

    fig.suptitle(
        "A cross-architecture L₀ routing circuit in VQ-tokenized vision-language models",
        fontsize=11.5, fontweight="bold", y=0.99)

    for suffix in (".png", ".pdf"):
        out = FIG_OUT / f"hero_3panel{suffix}"
        fig.savefig(out, dpi=200 if suffix == ".png" else None,
                    bbox_inches="tight")
        print(f"  saved → {out}")

    plt.close(fig)


if __name__ == "__main__":
    make_hero()
