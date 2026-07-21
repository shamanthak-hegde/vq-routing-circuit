"""
Generate all paper figures from on-disk results.
Run from repo root: python scripts/gen_all_figures.py
Requires: matplotlib, numpy, pandas (all available in sae env)
"""

import json
import csv
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import seaborn as sns

# ── output root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"

# Publication style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colour palette
C_VQ_NAT   = "#2166ac"   # natural VQ specimens (YES-promoting)
C_VQ_NO    = "#4dac26"   # natural VQ NO-promoting (Chameleon)
C_INDUCED  = "#762a83"   # induced VQ collapsed
C_FSQ      = "#1a9850"   # FSQ (YES-suppressing)
C_CTRL     = "#969696"   # continuous controls
C_BASE     = "#d73027"   # baseline bars
C_L0       = "#4575b4"   # L0 ablation
C_L1       = "#313695"
C_VTI      = "#fee090"
C_DOLA     = "#f46d43"
C_VCD      = "#fdae61"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_meta(path):
    with open(RES / path) as f:
        return json.load(f)["metrics"]


def load_attn(name):
    with open(RES / f"attn_head_weights_{name}.json") as f:
        return json.load(f)


def load_reldiv(name):
    with open(RES / f"residual_divergence_{name}.json") as f:
        return json.load(f)


def savefig(path, fig=None):
    out = FIG / path
    out.parent.mkdir(parents=True, exist_ok=True)
    (fig or plt).savefig(out)
    plt.close("all")
    print(f"  saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Codebook-bias scatter: 4 codebook conditions
#             Collapsed YES-promoting / Collapsed NO-promoting / FSQ v1 / FSQ v2
#             FSQ v1 + v2 are a structural replication pair (same backbone, both diverse)
# ═══════════════════════════════════════════════════════════════════════════

def fig1_codebook_scatter():
    def cb_mean(name):
        with open(RES / f"codebook_bias_{name}.json") as f:
            d = json.load(f)
        rates = list(d["entry_yes_rate"].values())
        return float(np.mean(rates))

    # Collapsed VQ natural specimens — have a measurable codebook bias
    nat = {
        "VILA-U":    (cb_mean("vilau"),    load_meta("bench_pope_full_vilau_baseline.meta.json")["yes_rate"]),
        "Liquid":    (cb_mean("liquid"),   load_meta("bench_pope_full_liquid_baseline.meta.json")["yes_rate"]),
        "Chameleon": (cb_mean("chameleon"),load_meta("bench_pope_full_chameleon_baseline.meta.json")["yes_rate"]),
    }
    # FSQ: diverse codebook, NO measurable collapse bias → anchored to y-axis (x=0)
    fsq_v1_y = load_meta("bench_pope_full_llava_vq_fsq_baseline.meta.json")["yes_rate"]
    fsq_v2_y = load_meta("bench_pope_full_llava_vq_fsq_v2_baseline.meta.json")["yes_rate"]

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    fig.subplots_adjust(top=0.82)

    # reference diagonal (y = x) — draw first so points sit on top
    xs = np.linspace(0, 1, 100)
    ax.plot(xs, xs, "--", color="#aaaaaa", lw=0.8, alpha=0.6)
    ax.text(0.94, 0.96, "y = x", fontsize=7, color="#aaaaaa", ha="right")

    # collapsed VQ specimens — per-model label offsets to avoid collision
    label_offsets = {
        "VILA-U":    (-0.04,  0.04),   # above-left (Liquid is to the right and below)
        "Liquid":    ( 0.03, -0.06),   # below-right
        "Chameleon": (-0.10,  0.03),   # left of point
    }
    for label, (x, y) in nat.items():
        c = C_VQ_NO if "Chameleon" in label else C_VQ_NAT
        ax.scatter(x, y, s=90, color=c, zorder=5)
        dx, dy = label_offsets[label]
        ax.annotate(label, (x, y), xytext=(x + dx, y + dy),
                    fontsize=8.5, color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6,
                                    shrinkA=5, shrinkB=4))

    # FSQ replication pair — no x value; anchored to y-axis
    for label, yr, jit in [("FSQ v1", fsq_v1_y, +0.012), ("FSQ v2", fsq_v2_y, -0.012)]:
        ax.scatter(0, yr + jit, s=90, marker="^", color=C_FSQ, zorder=5,
                   clip_on=False)
        ax.annotate(label, (0, yr + jit),
                    xytext=(0.04, yr + jit + 0.01),
                    fontsize=8, color=C_FSQ)

    # Brace linking the two FSQ points on the y-axis
    mid_y = (fsq_v1_y + fsq_v2_y) / 2
    ax.annotate("", xy=(0, fsq_v2_y - 0.012), xytext=(0, fsq_v1_y + 0.012),
                arrowprops=dict(arrowstyle="-", color=C_FSQ, lw=1.0,
                                connectionstyle="arc3,rad=0.0"))
    ax.text(-0.07, mid_y, "replication\npair", ha="center", va="center",
            fontsize=6.5, color=C_FSQ, style="italic",
            rotation=90, clip_on=False)

    # Light shading in x ∈ [−0.12, 0] to mark the "no collapse" zone
    ax.axvspan(-0.12, 0, color=C_FSQ, alpha=0.06, zorder=0)
    ax.text(-0.06, 0.55, "no\ncollapse", ha="center", fontsize=6.5,
            color=C_FSQ, style="italic", clip_on=False)

    ax.set_xlabel("Codebook mean yes-rate  (structural axis)")
    ax.set_ylabel("Behavioral baseline yes-rate  (POPE)")
    ax.set_xlim(-0.14, 1.02)
    ax.set_ylim(-0.04, 1.04)
    leg = [
        mpatches.Patch(color=C_VQ_NAT, label="Collapsed VQ — YES-promoting (VILA-U, Liquid)"),
        mpatches.Patch(color=C_VQ_NO,  label="Collapsed VQ — NO-promoting (Chameleon)"),
        mpatches.Patch(color=C_FSQ,    label="Diverse FSQ — no structural value (v1 + v2 replication)"),
    ]
    ax.legend(handles=leg, loc="lower right", bbox_to_anchor=(1.0, 1.01),
              borderaxespad=0, framealpha=0.9, fontsize=7.5)
    savefig("codebook_bias/scatter_4conditions.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — CHAIR two-model bar chart (4 metrics × VILA-U + LLaVA-VQ K65536)
# ═══════════════════════════════════════════════════════════════════════════

def fig2_chair_bar():
    # hardcoded (the project log verified)
    data = {
        "VILA-U":           {"CHAIRs": (35.8, 17.6), "CHAIRi": (10.3, 7.1),
                             "Recall": (58.5, 51.6), "AvgLen": (158, 78)},
        "LLaVA-VQ K65536":  {"CHAIRs": (31.2, 19.6), "CHAIRi": (52.5, 42.7),
                             "Recall": (11.1, 9.5),  "AvgLen": (158, 145)},
    }
    metrics = ["CHAIRs", "CHAIRi", "Recall", "AvgLen"]
    ylabels = ["CHAIRs (%)", "CHAIRi (%)", "Recall (%)", "Avg caption length (tokens)"]
    lower_is_better = [True, True, False, False]

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.2))

    models = list(data.keys())
    colors_base = [C_BASE, "#e08214"]
    colors_l0   = [C_L0,  "#542788"]

    for col, (ax, metric, ylabel, lib) in enumerate(zip(axes, metrics, ylabels, lower_is_better)):
        x = np.arange(len(models))
        w = 0.35
        for i, (model, cb, cl) in enumerate(zip(models, colors_base, colors_l0)):
            base_val, l0_val = data[model][metric]
            ax.bar(i - w/2, base_val, w, color=cb, alpha=0.85, label="Baseline" if i == 0 else "")
            ax.bar(i + w/2, l0_val,   w, color=cl, alpha=0.85, label="L0 ablation" if i == 0 else "")
            # delta label
            delta = l0_val - base_val
            sign = "−" if delta < 0 else "+"
            ax.text(i + w/2, max(base_val, l0_val) + (2 if metric != "AvgLen" else 4),
                    f"{delta:+.1f}", ha="center", fontsize=7, color=cl)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=7)
        ax.set_ylabel(ylabel)
        direction = " ↓ (lower=better)" if lib else " ↑"
        ax.set_title(metric + direction)

        if col == 0:
            ax.legend(loc="upper right", fontsize=7)

    fig.suptitle("CHAIR evaluation: VILA-U vs LLaVA-VQ K65536 (baseline vs L0 ablation)",
                 fontsize=10, y=1.02)
    savefig("chair/chair_bar.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — C2→C6 inset: same backbone, induced collapse → 5× CHAIRi
# ═══════════════════════════════════════════════════════════════════════════

def fig3_c2_c6_inset():
    models = ["VILA-U\n(natural VQ)", "LLaVA-VQ K65536\n(induced collapse)"]
    chairi_base = [10.3, 52.5]
    chairi_l0   = [7.1,  42.7]

    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    x = np.arange(2)
    w = 0.32
    ax.bar(x - w/2, chairi_base, w, color=C_BASE, label="Baseline", alpha=0.85)
    ax.bar(x + w/2, chairi_l0,   w, color=C_L0,   label="L0 ablation", alpha=0.85)

    # annotation: 5× inflation
    ax.annotate("", xy=(1 - w/2, 52.5), xytext=(0 - w/2, 10.3),
                arrowprops=dict(arrowstyle="->", color="#d73027", lw=1.4,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(0.5, 35, "same backbone\ninduced collapse\n→ 5× CHAIRi ↑",
            ha="center", fontsize=7.5, color="#d73027",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d73027", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("CHAIRi (%) ↓")
    ax.set_title("Induced collapse inflates hallucination 5×\n(C2 → C6 link)")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 65)
    savefig("chair/c2_c6_inset.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Binary-vs-CHAIR dissociation
# ═══════════════════════════════════════════════════════════════════════════

def fig4_dissociation():
    # VILA-U POPE acc (from meta files)
    pope_acc = {
        "Baseline":   load_meta("bench_pope_full_vilau_baseline.meta.json")["accuracy"],
        "VCD-tuned":  0.7290,
        "DoLA-tuned": 0.7937,
        "L0":         load_meta("bench_pope_full_vilau_L0.meta.json")["accuracy"],
        "L1":         load_meta("bench_pope_full_vilau_L1.meta.json")["accuracy"],
        "VTI":        load_meta("bench_pope_full_vilau_vti_a0.005.meta.json")["accuracy"],
    }
    amber_acc = {
        "Baseline":   load_meta("bench_amber_vilau_baseline.meta.json")["accuracy"],
        "VCD-tuned":  0.7070,
        "DoLA-tuned": 0.7952,
        "L0":         load_meta("bench_amber_vilau_intervention.meta.json")["accuracy"],
        "L1":         load_meta("bench_amber_vilau_L1.meta.json")["accuracy"],
        "VTI":        load_meta("bench_amber_vilau_vti_a0.005.meta.json")["accuracy"],
    }
    # CHAIRi only available for: Baseline, L0, DoLA-tuned, VCD-tuned
    chairi = {
        "Baseline":   10.3,
        "VCD-tuned":  10.7,
        "DoLA-tuned": 9.9,
        "L0":         7.1,
    }

    methods_all = ["Baseline", "VCD-tuned", "DoLA-tuned", "L0", "L1", "VTI"]
    methods_chair = ["Baseline", "VCD-tuned", "DoLA-tuned", "L0"]

    method_colors = {
        "Baseline":   C_CTRL,
        "VCD-tuned":  C_VCD,
        "DoLA-tuned": C_DOLA,
        "L0":         C_L0,
        "L1":         C_L1,
        "VTI":        "#9e0142",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # Left: POPE + AMBER acc grouped bars
    x = np.arange(len(methods_all))
    w = 0.38
    bars_pope  = [pope_acc[m] * 100 for m in methods_all]
    bars_amber = [amber_acc[m] * 100 for m in methods_all]

    for i, m in enumerate(methods_all):
        c = method_colors[m]
        ax1.bar(i - w/2, bars_pope[i],  w, color=c, alpha=0.85)
        ax1.bar(i + w/2, bars_amber[i], w, color=c, alpha=0.5, hatch="//")

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_all, rotation=30, ha="right")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Binary benchmarks (POPE solid / AMBER hatched)\n↑ DoLA-tuned dominates", fontsize=9)
    ax1.set_ylim(30, 90)

    leg1 = [mpatches.Patch(fc="gray",           label="POPE acc"),
            mpatches.Patch(fc="gray", hatch="//", label="AMBER acc")]
    ax1.legend(handles=leg1, fontsize=7, loc="lower right")

    # Right: CHAIRi
    x2 = np.arange(len(methods_chair))
    for i, m in enumerate(methods_chair):
        c = method_colors[m]
        bar = ax2.bar(i, chairi[m], color=c, alpha=0.85)
        ax2.text(i, chairi[m] + 0.15, f"{chairi[m]:.1f}%",
                 ha="center", va="bottom", fontsize=7.5)

    ax2.axhline(10.3, color=C_CTRL, ls="--", lw=0.9, alpha=0.7)
    ax2.text(3.4, 10.5, "baseline", fontsize=6.5, color=C_CTRL)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(methods_chair, rotation=20, ha="right")
    ax2.set_ylabel("CHAIRi (%) ↓")
    ax2.set_title("Generation hallucination (CHAIRi)\n↓ Only L0 ablation helps", fontsize=9)
    ax2.set_ylim(0, 14)

    fig.suptitle("Binary-vs-CHAIR dissociation on VILA-U  (§5.6)",
                 fontsize=10, y=1.02)
    savefig("intervention/dissociation.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Intervention table with 95 % CI error bars
# ═══════════════════════════════════════════════════════════════════════════

def fig5_ci_errorbars():
    df = pd.read_csv(RES / "paired_bootstrap_summary.tsv", sep="\t")
    df_acc = df[df["metric"] == "acc"].copy()

    # select headline rows
    keep = [
        ("vilau_pope_L0",   "POPE", "L0"),
        ("vilau_pope_L1",   "POPE", "L1"),
        ("vilau_amber_L1",  "AMBER","L1"),
        ("vilau_amber_vti", "AMBER","VTI"),
        ("liquid_pope_L0",  "POPE", "Liquid L0†"),
        ("chameleon_pope_L0","POPE","Chameleon L0"),
    ]

    # Also add tuned baselines (hardcoded, no bootstrap — just point estimates)
    extra = [
        ("DoLA-tuned (POPE)", "POPE", "DoLA-tuned", 0.7937 - 0.6193, None, None),
        ("DoLA-tuned (AMBER)","AMBER","DoLA-tuned", 0.7952 - 0.3967, None, None),
        ("VCD-tuned (POPE)",  "POPE", "VCD-tuned",  0.7290 - 0.6193, None, None),
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    y_pos = 0
    yticks, ylabels = [], []

    # mechanism-targeted (with CIs)
    for label, bench, short in keep:
        row = df_acc[df_acc["label"] == label]
        if row.empty:
            continue
        r = row.iloc[0]
        delta = float(r["mean_delta"])
        lo    = float(r["ci_lo"])
        hi    = float(r["ci_hi"])
        c = C_L0 if "L0" in short else (C_L1 if "L1" in short else "#9e0142")
        marker = "o" if bench == "POPE" else "s"
        ax.errorbar(delta * 100, y_pos, xerr=[[( delta - lo)*100], [(hi - delta)*100]],
                    fmt=marker, color=c, ms=6, capsize=3, lw=1.5)
        yticks.append(y_pos)
        ylabels.append(f"{short} ({bench})")
        y_pos += 1

    y_pos += 0.5  # gap before tuned baselines

    for (lbl, bench, short, delta, lo, hi) in extra:
        c = C_DOLA if "DoLA" in short else C_VCD
        marker = "o" if bench == "POPE" else "s"
        if lo is None:
            ax.plot(delta * 100, y_pos, marker, color=c, ms=7, alpha=0.6)
            ax.annotate("(no CI — point est.)", (delta * 100, y_pos),
                        xytext=(2, 0), textcoords="offset points",
                        fontsize=6.5, color=c, va="center")
        yticks.append(y_pos)
        ylabels.append(f"{short} ({bench})")
        y_pos += 1

    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Δ Accuracy (pp) vs. VILA-U baseline  [95% CI]")
    ax.set_title("Intervention accuracy with paired-bootstrap 95% CIs  (§5.6)")

    leg = [
        mpatches.Patch(color=C_L0,    label="L0 ablation"),
        mpatches.Patch(color=C_L1,    label="L1 ablation"),
        mpatches.Patch(color="#9e0142",label="VTI"),
        mpatches.Patch(color=C_DOLA,  label="DoLA-tuned"),
        mpatches.Patch(color=C_VCD,   label="VCD-tuned"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=7)
    savefig("intervention/ci_errorbars.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — 5-panel L0 head-weight heatmaps
#            VILA-U (reference) / Liquid / Chameleon / Anole / Lumina-mGPT
# ═══════════════════════════════════════════════════════════════════════════

def fig6_heatmaps_4panel():
    models = [
        ("vilau",       "VILA-U\n(reference, YES-promoting)",   True),
        ("liquid",      "Liquid (Gemma-7B)\nNatural, YES-promoting", False),
        ("chameleon",   "Chameleon-7B\nNatural, NO-promoting",  False),
        ("anole",       "Anole-7b\nFine-tuned descendant",      False),
        ("lumina_mgpt", "Lumina-mGPT\nHeld-out, struct+/beh-",  False),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.0))

    im = None
    for ax, (name, title, is_ref) in zip(axes, models):
        d = load_attn(name)
        mat = np.array(d["mean_head_visual_sum"])
        n_layers, n_heads = mat.shape
        vmax = mat.max()
        im = ax.imshow(mat.T, aspect="auto", origin="lower",
                       cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Head" if name == "vilau" else "")
        title_str = title
        ax.set_title(title_str, fontsize=8,
                     fontweight="bold" if is_ref else "normal",
                     color="#2166ac" if is_ref else "black")
        ax.axvline(-0.5, color="white", lw=1.4, ls="--", alpha=0.8)
        ax.text(0, n_heads + 0.3, "L0", ha="center", fontsize=6.5, color="white")
        if is_ref:
            for spine in ax.spines.values():
                spine.set_edgecolor("#2166ac")
                spine.set_linewidth(1.5)

    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.55, pad=0.02)
    cb.set_label("Visual→prompt_last attn mass\n(per-model max normalisation)", fontsize=7)
    fig.suptitle("L0 head-weight heatmaps — VILA-U reference + 4 new natural specimens  (§5.1 / Gate A.2)",
                 fontsize=10)
    savefig("attn_heads/heatmap_5panel_new_specimens.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7 — A.2 cross-model bar chart annotated by criterion tier
# ═══════════════════════════════════════════════════════════════════════════

def fig7_a2_bar():
    # (name, display label, metric to use, tier)
    # tier: "proj"=projector-amplified, "vocab"=unified-vocab
    entries = [
        # --- projector-amplified (use L0 mass) ---
        ("vilau",            "VILA-U",          "proj"),
        ("llava_vq",         "LLaVA-VQ K16384", "proj"),
        ("llava_vq_fsq_v2",  "LLaVA-VQ-FSQ v2","proj"),
        ("llava_mlp_control","LLaVA-MLP ctrl",  "proj"),
        # --- unified-vocab (use [0,8) window) ---
        ("chameleon",        "Chameleon-7B",    "vocab"),
        ("liquid",           "Liquid",          "vocab"),
        ("anole",            "Anole-7b",        "vocab"),
        ("lumina_mgpt",      "Lumina-mGPT",     "vocab"),
        # --- others (not in final gate-pass cohort) ---
        ("unitok",           "UniTok",          "other"),
        ("emu3",             "Emu3",            "other"),
        ("qwen_vq",          "Qwen-VQ",         "other"),
        ("janus",            "Janus-Pro",       "other"),
    ]

    names, labels, values, tiers = [], [], [], []
    for (name, label, tier) in entries:
        d = load_attn(name)
        if tier == "proj":
            val = d["mean_layer_visual_sum"][0]       # L0 mass
        else:
            val = sum(d["mean_layer_visual_sum"][:8]) # [0,8) window
        names.append(name)
        labels.append(label)
        values.append(val)
        tiers.append(tier)

    tier_color = {"proj": C_VQ_NAT, "vocab": "#74add1", "other": C_CTRL}
    colors = [tier_color[t] for t in tiers]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, alpha=0.85)

    # threshold line
    ax.axhline(15, color="#d73027", lw=1.0, ls="--", alpha=0.8, label="Pass threshold (≥ 15)")

    # annotate PASS / FAIL
    for i, (v, t) in enumerate(zip(values, tiers)):
        if t != "other":
            status = "PASS" if v >= 15 else "FAIL"
            col = "#1a9641" if status == "PASS" else "#d7191c"
            ax.text(i, v + 1.5, status, ha="center", fontsize=6.5, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("Gate A.2 score\n(L0 mass for proj-amp; [0,8) for unified-vocab)")
    ax.set_title("Cross-model gate A.2 scores  (§5.1 / §5.5)", fontsize=10)
    ax.legend(fontsize=8)

    leg = [
        mpatches.Patch(color=C_VQ_NAT, label="Projector-amplified (L0 mass)"),
        mpatches.Patch(color="#74add1", label="Unified-vocab ([0,8) window)"),
        mpatches.Patch(color=C_CTRL,   label="Other / negative controls"),
    ]
    ax.legend(handles=leg, fontsize=7, loc="upper right")
    savefig("attn_heads/cross_model_a2_bar.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8 — A.1 curves: VILA-U vs LLaVA-VQ vs LLaVA-MLP
# ═══════════════════════════════════════════════════════════════════════════

def fig8_a1_curves():
    specs = [
        ("vilau",             "VILA-U (natural VQ)",       C_VQ_NAT, "-"),
        ("llava_vq",          "LLaVA-VQ (induced, VQ)",    C_INDUCED, "-"),
        ("llava_mlp_control", "LLaVA-MLP (induced, MLP)",  C_CTRL,   "--"),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for name, label, color, ls in specs:
        d = load_reldiv(name)
        rd = d["mean_rel_div"]
        ax.plot(rd, label=label, color=color, ls=ls, lw=1.8, marker="o", ms=3)

    ax.axhline(0.4, color="#aaaaaa", ls=":", lw=0.8, label="Gate A.1 threshold (0.4)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean relative divergence")
    ax.set_title("Gate A.1: residual divergence curves  (§5.5)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.45)
    savefig("mechanism/a1_curves.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 10 — 4-panel gate comparison (2-row × 3-col)
#             LLaVA-MLP / LLaVA-VQ / LLaVA-VQ-FSQ-v2
# ═══════════════════════════════════════════════════════════════════════════

def fig10_4panel_gate():
    cols = [
        ("llava_mlp_control", "LLaVA-MLP\n(matched-compute ctrl)", C_CTRL),
        ("llava_vq",          "LLaVA-VQ (collapsed)\nInduced YES-promoting", C_INDUCED),
        ("llava_vq_fsq_v2",   "LLaVA-VQ-FSQ v2\nInduced YES-suppressing", C_FSQ),
    ]

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0 — A.1 curves (MLP and VQ have data; FSQ does not)
    ref_reldiv = load_reldiv("vilau")["mean_rel_div"]

    for col_i, (name, title, color) in enumerate(cols):
        ax = fig.add_subplot(gs[0, col_i])
        if "fsq" in name:
            ax.text(0.5, 0.5, "A.1 data\nnot collected", ha="center", va="center",
                    transform=ax.transAxes, color="#aaaaaa", fontsize=9)
        else:
            d = load_reldiv(name)
            rd = d["mean_rel_div"]
            ax.plot(rd, color=color, lw=1.6, marker="o", ms=2.5)
            ax.plot(ref_reldiv, color=C_VQ_NAT, lw=1.0, ls="--", alpha=0.5, label="VILA-U (ref)")
            ax.axhline(0.4, color="#aaaaaa", ls=":", lw=0.7)
            peak = max(rd)
            ax.set_title(f"Peak={peak:.3f} → {'PASS' if peak >= 0.4 else 'FAIL'}", fontsize=8)
        ax.set_xlabel("Layer" if col_i == 1 else "")
        ax.set_ylabel("Rel. divergence" if col_i == 0 else "")
        if col_i == 0:
            ax.set_title(f"Gate A.1 — {title}", fontsize=8)
        elif col_i == 1:
            pass  # title set above or in loop

    # Row 1 — A.2 heatmaps
    ref_attn = load_attn("vilau")
    ref_mat  = np.array(ref_attn["mean_head_visual_sum"])
    vmax = ref_mat.max()

    for col_i, (name, title, color) in enumerate(cols):
        ax = fig.add_subplot(gs[1, col_i])
        d = load_attn(name)
        mat = np.array(d["mean_head_visual_sum"])
        im = ax.imshow(mat.T, aspect="auto", origin="lower",
                       cmap="YlOrRd", vmin=0, vmax=vmax)
        l0_mass = d["mean_layer_visual_sum"][0]
        pass_str = "PASS" if l0_mass >= 15 else "FAIL"
        ax.set_title(f"Gate A.2 — L0 mass={l0_mass:.1f} → {pass_str}", fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Head" if col_i == 0 else "")
        ax.axvline(0, color="white", lw=1.0, ls="--", alpha=0.6)

    cb_ax = fig.add_axes([0.92, 0.12, 0.015, 0.35])
    fig.colorbar(im, cax=cb_ax).set_label("Attn mass\n(per-model max)", fontsize=7)

    fig.suptitle("Single-variable comparison: projector type determines gate outcomes  (§5.2)",
                 fontsize=10)
    savefig("mechanism/4panel_gate_comparison.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 11 — K-sweep polarity figure
# ═══════════════════════════════════════════════════════════════════════════

def fig11_k_sweep():
    # K=16384 uses the original llava_vq files
    k_specs = [
        (1024,  "bench_pope_full_llava_vq_K1024_baseline.meta.json",
                "bench_pope_full_llava_vq_K1024_L0.meta.json"),
        (4096,  "bench_pope_full_llava_vq_K4096_baseline.meta.json",
                "bench_pope_full_llava_vq_K4096_L0.meta.json"),
        (16384, "bench_pope_full_llava_vq_baseline.meta.json",
                "bench_pope_full_llava_vq_intervention.meta.json"),
        (65536, "bench_pope_full_llava_vq_K65536_baseline.meta.json",
                "bench_pope_full_llava_vq_K65536_L0.meta.json"),
    ]

    Ks, bases, l0s, deltas = [], [], [], []
    for K, bf, lf in k_specs:
        bm = load_meta(bf)
        lm = load_meta(lf)
        Ks.append(K)
        bases.append(bm["yes_rate"])
        l0s.append(lm["yes_rate"])
        deltas.append(lm["yes_rate"] - bm["yes_rate"])

    # utilisation from the project log
    util = {1024: 15.4, 4096: 2.3, 16384: 0.55, 65536: 0.13}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left: yes_rate baseline + L0 per K
    x = np.arange(len(Ks))
    w = 0.35
    axes[0].bar(x - w/2, [b * 100 for b in bases], w, color=C_BASE, alpha=0.85, label="Baseline")
    axes[0].bar(x + w/2, [l * 100 for l in l0s],   w, color=C_L0,   alpha=0.85, label="L0 ablation")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"K={K:,}" for K in Ks])
    axes[0].set_ylabel("POPE yes-rate (%)")
    axes[0].set_title("Baseline vs L0 yes-rate per K")
    axes[0].legend()

    # Right: Δyes_rate vs util, colour by polarity
    pol_colors = []
    for d in deltas:
        if abs(d) < 0.02:
            pol_colors.append(C_CTRL)
        elif d > 0:
            pol_colors.append(C_FSQ)     # YES-suppressing (L0 KO raises yes)
        else:
            pol_colors.append(C_INDUCED)  # YES-promoting (L0 KO lowers yes)

    for K, d, u, c in zip(Ks, deltas, [util[k] for k in Ks], pol_colors):
        axes[1].scatter(u, d * 100, s=90, color=c, zorder=5)
        axes[1].annotate(f"K={K:,}", (u, d * 100),
                         xytext=(2, 4), textcoords="offset points", fontsize=7)

    axes[1].axhline(0, color="#555555", lw=0.8, ls="--")
    axes[1].set_xlabel("Codebook utilisation (%)")
    axes[1].set_ylabel("Δ yes-rate (L0 − baseline, pp)")
    axes[1].set_title("Polarity vs utilisation\n(K=4096 anomaly: YES-suppressing despite collapse)")

    leg = [
        mpatches.Patch(color=C_INDUCED, label="YES-promoting (↓ yes under L0 KO)"),
        mpatches.Patch(color=C_FSQ,     label="YES-suppressing (↑ yes under L0 KO)"),
        mpatches.Patch(color=C_CTRL,    label="Neutral"),
    ]
    axes[1].legend(handles=leg, fontsize=7)

    fig.suptitle("K-sweep: codebook size vs polarity  (§5.2 / §5.3)", fontsize=10)
    savefig("k_sweep/k_sweep_polarity.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 12 — Induction / reversal / matched-control asymmetry panel
# ═══════════════════════════════════════════════════════════════════════════

def fig12_induction_reversal():
    # Three rows (one subplot each)
    rows = [
        ("Induction (C2)",
         "LLaVA-1.6\n(healthy baseline)",
         "LLaVA-VQ\n(VQ projector added)",
         load_meta("bench_pope_full_llava_baseline.meta.json")["yes_rate"],
         load_meta("bench_pope_full_llava_vq_baseline.meta.json")["yes_rate"],
         "→  pathology induced",   C_INDUCED),
        ("Reversal",
         "VILA-U\n(VQ-trained LLM)",
         "VILA-U-MLP\n(MLP projector swapped in)",
         load_meta("bench_pope_full_vilau_baseline.meta.json")["yes_rate"],
         load_meta("bench_pope_full_vilau_mlp_baseline.meta.json")["yes_rate"],
         "→  pathology WORSENS",   "#d73027"),
        ("Matched control",
         "LLaVA-VQ\n(VQ projector)",
         "LLaVA-MLP\n(same steps, MLP)",
         load_meta("bench_pope_full_llava_vq_baseline.meta.json")["yes_rate"],
         load_meta("bench_pope_full_llava_mlp_baseline.meta.json")["yes_rate"],
         "→  no pathology",        C_CTRL),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 5.5), sharex=False)

    for ax, (title, label_a, label_b, yr_a, yr_b, annotation, color) in zip(axes, rows):
        ax.barh([label_b, label_a], [yr_b * 100, yr_a * 100],
                color=[color, "#aaaaaa"], alpha=0.85)
        ax.axvline(50, color="#555555", lw=0.6, ls="--")
        ax.set_xlim(0, 105)
        ax.set_xlabel("POPE yes-rate (%)" if ax is axes[-1] else "")
        ax.set_title(title, fontsize=9, pad=3)
        ax.text(102, 0.5, annotation, ha="right", va="center",
                fontsize=8, color=color, transform=ax.get_yaxis_transform())

    fig.suptitle("Induction ✓ / Reversal ✗ / Null-control ✓  (§5.2 / §5.4)",
                 fontsize=10, y=1.01)
    savefig("mechanism/induction_reversal_control.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 13 — L1 cross-architecture polarity panel
# ═══════════════════════════════════════════════════════════════════════════

def fig13_l1_polarity():
    df = pd.read_csv(RES / "paired_bootstrap_summary.tsv", sep="\t")

    # VILA-U L1 POPE acc
    def get_row(label, metric="acc"):
        rows = df[(df["label"] == label) & (df["metric"] == metric)]
        if rows.empty:
            return None
        r = rows.iloc[0]
        return float(r["mean_delta"]), float(r["ci_lo"]), float(r["ci_hi"])

    # Chameleon L1 yes_rate delta (POPE)
    specs = [
        ("VILA-U L1 — POPE acc",      get_row("vilau_pope_L1",      "acc"),     C_VQ_NAT, "YES-amplifier (L1 ↑ acc)\n→ L1 reads & amplifies YES-routing"),
        ("Chameleon L1 — POPE yes",   get_row("chameleon_pope_L1",  "yes_rate"),C_VQ_NO,  "NO-writer (L1 ablation inverts to YES)\n→ L1 writes codebook NO-bias"),
        ("Liquid L1 — POPE acc",      get_row("liquid_pope_L1",     "acc"),     C_VQ_NAT, "Partial YES-amplifier (logit-valid)\n→ L1 ablation gen-degenerate†"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    for ax, (title, row_data, color, annotation) in zip(axes, specs):
        if row_data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        delta, lo, hi = row_data
        ax.barh([0], [delta * 100], xerr=[[( delta - lo)*100], [(hi - delta)*100]],
                color=color, alpha=0.85, capsize=4)
        ax.axvline(0, color="#555555", lw=0.8, ls="--")
        ax.set_yticks([])
        ax.set_xlabel("Δ metric vs baseline (pp)")
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.text(0.5, -0.22, annotation, ha="center", fontsize=7.5, color=color,
                transform=ax.transAxes, wrap=True)

    fig.suptitle("L1 ablation effect by model class  (§5.6 / §7.3)",
                 fontsize=10, y=1.06)
    savefig("mechanism/l1_cross_arch_polarity.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 14 — Architecture-specificity bar chart (~20 models)
# ═══════════════════════════════════════════════════════════════════════════

def fig14_arch_specificity():
    model_specs = [
        # (display, base_meta, l0_meta, class)
        # class: "vq_spec"=VQ specimen, "ctrl"=continuous, "induced"=induced
        ("VILA-U",          "bench_pope_full_vilau_baseline.meta.json",
                            "bench_pope_full_vilau_L0.meta.json",               "vq_spec"),
        ("Liquid",          "bench_pope_full_liquid_baseline.meta.json",
                            "bench_pope_full_liquid_L0.meta.json",              "vq_spec"),
        ("Chameleon",       "bench_pope_full_chameleon_baseline.meta.json",
                            "bench_pope_full_chameleon_L0.meta.json",           "vq_spec"),
        ("Anole",           "bench_pope_full_anole_baseline.meta.json",
                            "bench_pope_full_anole_L0.meta.json",               "vq_spec"),
        ("Lumina-mGPT",     "bench_pope_full_lumina_mgpt_baseline.meta.json",
                            "bench_pope_full_lumina_mgpt_L0.meta.json",         "vq_spec"),
        ("LLaVA-VQ K16384", "bench_pope_full_llava_vq_baseline.meta.json",
                            "bench_pope_full_llava_vq_intervention.meta.json",  "induced"),
        ("LLaVA-VQ K1024",  "bench_pope_full_llava_vq_K1024_baseline.meta.json",
                            "bench_pope_full_llava_vq_K1024_L0.meta.json",      "induced"),
        ("LLaVA-VQ K4096",  "bench_pope_full_llava_vq_K4096_baseline.meta.json",
                            "bench_pope_full_llava_vq_K4096_L0.meta.json",      "induced"),
        ("LLaVA-VQ K65536", "bench_pope_full_llava_vq_K65536_baseline.meta.json",
                            "bench_pope_full_llava_vq_K65536_L0.meta.json",     "induced"),
        ("LLaVA-VQ-FSQ v1", "bench_pope_full_llava_vq_fsq_baseline.meta.json",
                            "bench_pope_full_llava_vq_fsq_L0.meta.json",        "induced"),
        ("LLaVA-VQ-FSQ v2", "bench_pope_full_llava_vq_fsq_v2_baseline.meta.json",
                            "bench_pope_full_llava_vq_fsq_v2_L0.meta.json",     "induced"),
        ("LLaVA-MLP ctrl",  "bench_pope_full_llava_mlp_baseline.meta.json",
                            "bench_pope_full_llava_mlp_L0.meta.json",           "induced"),
        ("LLaVA-1.6",       "bench_pope_full_llava_baseline.meta.json",
                            "bench_pope_full_llava_intervention.meta.json",     "ctrl"),
        ("Qwen2.5-VL",      "bench_pope_full_qwen3vl_baseline.meta.json",
                            "bench_pope_full_qwen3vl_intervention.meta.json",   "ctrl"),
        ("Emu3",            "bench_pope_full_emu3_baseline.meta.json",
                            "bench_pope_full_emu3_L0.meta.json",                "ctrl"),
        ("UniTok",          "bench_pope_full_unitok_baseline.meta.json",
                            "bench_pope_full_unitok_intervention.meta.json",    "ctrl"),
        ("Show-o",          "bench_pope_full_showo_baseline.meta.json",
                            "bench_pope_full_showo_L4.meta.json",               "ctrl"),
        ("Janus-Pro",       "bench_pope_full_janus_baseline.meta.json",
                            "bench_pope_full_janus_L0.meta.json",               "ctrl"),
        ("HaploOmni",       "bench_pope_full_haplo_baseline.meta.json",
                            "bench_pope_full_haplo_L0.meta.json",               "ctrl"),
        ("LaVIT",           "bench_pope_full_lavit_baseline.meta.json",
                            "bench_pope_full_lavit_L0.meta.json",               "ctrl"),
        ("Qwen-VQ",         "bench_pope_full_qwen_vq_baseline.meta.json",
                            "bench_pope_full_qwen_vq_L0.meta.json",             "induced"),
    ]

    class_color = {"vq_spec": C_VQ_NAT, "ctrl": C_CTRL, "induced": C_INDUCED}
    labels, deltas, colors = [], [], []

    for (disp, bf, lf, cls) in model_specs:
        try:
            bm = load_meta(bf)
            lm = load_meta(lf)
            delta = (lm["accuracy"] - bm["accuracy"]) * 100
        except Exception:
            continue
        labels.append(disp)
        deltas.append(delta)
        colors.append(class_color[cls])

    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(labels))
    ax.bar(x, deltas, color=colors, alpha=0.85)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("POPE Δ accuracy (pp)\nL0 ablation vs baseline")
    ax.set_title("Architecture-specificity: POPE accuracy change under L0 ablation  (§5.6)", fontsize=10)

    leg = [
        mpatches.Patch(color=C_VQ_NAT,  label="VQ specimen (natural)"),
        mpatches.Patch(color=C_INDUCED, label="Induced / VQ-swap"),
        mpatches.Patch(color=C_CTRL,    label="Continuous control"),
    ]
    ax.legend(handles=leg, loc="lower left", fontsize=8)
    savefig("yes_bias/arch_specificity_bar.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# REGEN — gate_status.png  (full current cohort)
# ═══════════════════════════════════════════════════════════════════════════

def regen_gate_status():
    # Gate results from R-D (2026-05-22) + prior results
    # (A1 pass, A2 pass, A3 pass, behavioral_yes_rate, circuit class)
    gate_data = [
        # ---- PASS models ----
        ("VILA-U",          True,  True,  True,  0.810, "vq_spec"),
        ("Liquid",          True,  True,  True,  0.782, "vq_spec"),
        ("Chameleon",       True,  True,  True,  0.223, "vq_spec"),
        ("Anole",           True,  True,  True,  0.753, "vq_spec"),
        ("Lumina-mGPT",     True,  True,  True,  0.001, "vq_spec"),
        ("LLaVA-VQ K16384", True,  True,  True,  0.855, "induced"),
        ("LLaVA-VQ K65536", True,  True,  True,  0.872, "induced"),
        ("LLaVA-VQ-FSQ v1", True,  True,  True,  0.026, "induced"),
        ("LLaVA-VQ-FSQ v2", True,  True,  True,  0.014, "induced"),
        # --- MLP ctrl: A1 FAIL ---
        ("LLaVA-MLP ctrl",  False, True,  True,  0.000, "ctrl"),
        # ---- FAIL models (various gates) ----
        ("LLaVA-1.6",       False, False, False, 0.573, "ctrl"),
        ("Janus-Pro",       True,  True,  False, 0.444, "ctrl"),
        ("Emu3",            False, True,  True,  0.327, "ctrl"),
        ("UniTok",          False, False, True,  0.475, "ctrl"),
        ("Show-o",          False, False, True,  0.984, "ctrl"),
        ("SEED",            False, False, True,  0.396, "ctrl"),
        ("HaploOmni",       False, False, False, 0.148, "ctrl"),
        ("LaVIT",           False, False, False, 0.973, "ctrl"),
        ("Qwen2.5-VL",      False, False, False, 0.242, "ctrl"),
        ("Qwen-VQ",         False, True,  True,  0.003, "induced"),
        ("LLaVA-VQ K1024",  False, True,  True,  0.324, "induced"),
        ("LLaVA-VQ K4096",  True,  True,  True,  0.798, "induced"),
    ]

    # Sort: PASS first, then by class
    def sort_key(row):
        all_pass = all(row[1:4])
        return (0 if all_pass else 1, row[5], row[0])

    gate_data.sort(key=sort_key)

    n = len(gate_data)
    fig, ax = plt.subplots(figsize=(10, n * 0.38 + 1.5))

    gate_names = ["A.1\nrel_div", "A.2\nattn mass", "A.3\noff-manifold"]
    col_x = [1, 2, 3]

    for row_i, (model, a1, a2, a3, yr, cls) in enumerate(gate_data):
        y = n - row_i
        for col_i, (gate_pass, cx) in enumerate(zip([a1, a2, a3], col_x)):
            c = "#2ca02c" if gate_pass else "#d62728"
            marker = "✓" if gate_pass else "✗"
            ax.text(cx, y, marker, ha="center", va="center",
                    fontsize=12, color=c, fontweight="bold")

        all_pass = a1 and a2 and a3
        model_color = class_color_for(cls)
        ax.text(0, y, model, ha="right", va="center", fontsize=8, color=model_color)
        ax.text(4, y, f"{yr:.0%}", ha="left", va="center", fontsize=7.5, color="#555555")
        if all_pass:
            ax.axhspan(y - 0.4, y + 0.4, alpha=0.07, color=model_color)

    ax.set_xlim(-0.5, 5)
    ax.set_ylim(0.2, n + 0.8)
    ax.set_xticks(col_x + [4])
    ax.set_xticklabels(gate_names + ["Baseline\nyes-rate"], fontsize=8)
    ax.set_yticks([])
    ax.set_title("3-gate diagnostic chain — full cohort  (§4 / §5.5)", fontsize=10)

    leg = [
        mpatches.Patch(color=class_color_for("vq_spec"),  label="Natural VQ specimen"),
        mpatches.Patch(color=class_color_for("induced"),  label="Induced VQ"),
        mpatches.Patch(color=class_color_for("ctrl"),     label="Continuous control"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=7)
    savefig("yes_bias/gate_status.png", fig)


def class_color_for(cls):
    return {
        "vq_spec": C_VQ_NAT,
        "induced": C_INDUCED,
        "ctrl":    C_CTRL,
    }.get(cls, "#555555")


# ═══════════════════════════════════════════════════════════════════════════
# REGEN — benchmark_summary.png  (tuned baselines + mechanism interventions)
# ═══════════════════════════════════════════════════════════════════════════

def regen_benchmark_summary():
    methods = ["Baseline", "L0", "L1", "VTI", "VCD-tuned", "DoLA-tuned"]
    pope_acc = [
        load_meta("bench_pope_full_vilau_baseline.meta.json")["accuracy"],
        load_meta("bench_pope_full_vilau_L0.meta.json")["accuracy"],
        load_meta("bench_pope_full_vilau_L1.meta.json")["accuracy"],
        load_meta("bench_pope_full_vilau_vti_a0.005.meta.json")["accuracy"],
        0.7290,  # VCD best
        0.7937,  # DoLA best
    ]
    amber_acc = [
        load_meta("bench_amber_vilau_baseline.meta.json")["accuracy"],
        load_meta("bench_amber_vilau_intervention.meta.json")["accuracy"],
        load_meta("bench_amber_vilau_L1.meta.json")["accuracy"],
        load_meta("bench_amber_vilau_vti_a0.005.meta.json")["accuracy"],
        0.7070,  # VCD best
        0.7952,  # DoLA best
    ]

    method_colors = [C_CTRL, C_L0, C_L1, "#9e0142", C_VCD, C_DOLA]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=False)

    for ax, accs, bench in zip(axes, [pope_acc, amber_acc], ["POPE", "AMBER"]):
        x = np.arange(len(methods))
        bars = ax.bar(x, [a * 100 for a in accs], color=method_colors, alpha=0.85)
        for b, v in zip(bars, accs):
            ax.text(b.get_x() + b.get_width()/2, v * 100 + 0.3,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{bench} (VILA-U)")
        ax.set_ylim(30, 90 if bench == "POPE" else 90)
        # divider between mechanism-targeted and tuned baselines
        ax.axvline(3.5, color="#aaaaaa", ls="--", lw=0.8)
        ax.text(1.5, 87, "mechanism-targeted", fontsize=7, color="#555555", ha="center")
        ax.text(4.5, 87, "tuned baselines", fontsize=7, color="#555555", ha="center")

    fig.suptitle("VILA-U intervention summary: mechanism-targeted vs tuned baselines  (§5.6)",
                 fontsize=10)
    savefig("intervention/benchmark_summary.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# REGEN — comparison figures (causal restoration heatmaps, current cohort)
#         Uses WARN filter: only records where logit_clean > logit_corrupt
# ═══════════════════════════════════════════════════════════════════════════

_SWEEP_GROUP_ORDER = ["visual", "question", "prompt_last"]
_SWEEP_CMAP = "RdYlGn"


def _load_sweep_filtered(name: str) -> pd.DataFrame:
    """Load a sweep JSONL and apply WARN filter (logit_clean > logit_corrupt)."""
    rows = []
    with open(RES / f"sweep_{name}.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["warn"] = df["logit_clean"] <= df["logit_corrupt"]
    return df[~df["warn"]]


def _sweep_pivot(df: pd.DataFrame, source: str) -> pd.DataFrame:
    sub = df[df["source"] == source]
    if sub.empty:
        return pd.DataFrame()
    piv = (
        sub.groupby(["layer_start", "token_group"])["score"]
        .mean()
        .unstack()
        .reindex(columns=_SWEEP_GROUP_ORDER)
    )
    return piv


def _draw_sweep_heatmap(ax, pivot, title, n_records, vmin=-0.3, vmax=1.0):
    if pivot.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="#aaaaaa")
        ax.set_title(title, fontsize=9)
        return
    layer_ends = [r + 4 for r in pivot.index]
    ylabels = [f"[{ls},{le})" for ls, le in zip(pivot.index, layer_ends)]
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f",
                vmin=vmin, vmax=vmax, cmap=_SWEEP_CMAP,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 8},
                cbar_kws={"shrink": 0.75})
    ax.set_yticklabels(ylabels, rotation=0, fontsize=8)
    ax.set_xticklabels(pivot.columns, rotation=15, ha="right", fontsize=8)
    ax.set_xlabel("Token group", fontsize=9)
    ax.set_ylabel("Layer window", fontsize=9)
    ax.set_title(f"{title}\n(n={n_records})", fontsize=9)


def _draw_sweep_diff(ax, pivot, title, vmax=0.5):
    if pivot.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="#aaaaaa")
        ax.set_title(title, fontsize=9)
        return
    layer_ends = [r + 4 for r in pivot.index]
    ylabels = [f"[{ls},{le})" for ls, le in zip(pivot.index, layer_ends)]
    sns.heatmap(pivot, ax=ax, annot=True, fmt="+.2f",
                vmin=-vmax, vmax=vmax, cmap="RdBu_r", center=0,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 8},
                cbar_kws={"shrink": 0.75})
    ax.set_yticklabels(ylabels, rotation=0, fontsize=8)
    ax.set_xticklabels(pivot.columns, rotation=15, ha="right", fontsize=8)
    ax.set_xlabel("Token group", fontsize=9)
    ax.set_ylabel("", fontsize=9)
    ax.set_title(title, fontsize=9)


def _n_records(df: pd.DataFrame, source: str) -> int:
    return df[df["source"] == source]["record_id"].nunique() if not df.empty else 0


def regen_comparison_figures():
    """Regenerate all three comparison figure directories with current sweep data.

    figures/comparison/        — classic 3-model: LLaVA | VILA | VILA-U
    figures/comparison_4model/ — 5-model:  LLaVA | VILA | Qwen | VILA-U | Chameleon
    figures/comparison_filtered/ — VQ-focus: LLaVA | Liquid | Chameleon | LLaVA-VQ | VILA-U
    """
    print("  Loading sweep data (WARN-filtered)...")
    dfs = {}
    sweep_map = {
        "llava":     "llava",
        "vila":      "vila",
        "vilau":     "vilau",
        "qwen":      "qwen3vl",
        "chameleon": "chameleon",
        "liquid":    "liquid",
        "llava_vq":  "llava_vq",
        "lumina":    "lumina_mgpt",
    }
    for key, fname in sweep_map.items():
        try:
            dfs[key] = _load_sweep_filtered(fname)
            print(f"    {key}: loaded")
        except FileNotFoundError:
            dfs[key] = pd.DataFrame()
            print(f"    {key}: MISSING")

    for source in ("pope", "naturalbench"):
        # ── Set 1: classic 3-model (same models as original comparison/) ──
        ref_piv = _sweep_pivot(dfs["llava"],  source)
        vi_piv  = _sweep_pivot(dfs["vila"],   source)
        vu_piv  = _sweep_pivot(dfs["vilau"],  source)
        common = ref_piv.index.intersection(vi_piv.index).intersection(vu_piv.index)
        if len(common) > 0:
            ref_p = ref_piv.reindex(index=common, columns=_SWEEP_GROUP_ORDER)
            vi_p  = vi_piv.reindex(index=common, columns=_SWEEP_GROUP_ORDER)
            vu_p  = vu_piv.reindex(index=common, columns=_SWEEP_GROUP_ORDER)
            fig, axes = plt.subplots(1, 5, figsize=(27, 5))
            _draw_sweep_heatmap(axes[0], ref_p, f"LLaVA-1.6 — {source}",
                                _n_records(dfs["llava"], source))
            _draw_sweep_heatmap(axes[1], vi_p,  f"VILA v1.0 — {source}",
                                _n_records(dfs["vila"], source))
            _draw_sweep_heatmap(axes[2], vu_p,  f"VILA-U — {source}",
                                _n_records(dfs["vilau"], source))
            _draw_sweep_diff(axes[3], vi_p  - ref_p, "Δ  (VILA − LLaVA)")
            _draw_sweep_diff(axes[4], vu_p  - ref_p, "Δ  (VILA-U − LLaVA)")
            fig.suptitle(f"3-model causal restoration — {source}", fontsize=13)
            fig.tight_layout()
            savefig(f"comparison/compare_{source}.png", fig)

        # ── Set 2: 5-model extended (comparison_4model/ dir) ──
        q_piv  = _sweep_pivot(dfs["qwen"],      source)
        ch_piv = _sweep_pivot(dfs["chameleon"], source)
        common2 = ref_piv.index.intersection(vi_piv.index).intersection(vu_piv.index)
        if not q_piv.empty:
            common2 = common2.intersection(q_piv.index)
        if not ch_piv.empty:
            common2 = common2.intersection(ch_piv.index)
        if len(common2) > 0:
            ref_p = ref_piv.reindex(index=common2, columns=_SWEEP_GROUP_ORDER)
            vi_p  = vi_piv.reindex(index=common2, columns=_SWEEP_GROUP_ORDER)
            q_p   = q_piv.reindex(index=common2, columns=_SWEEP_GROUP_ORDER) if not q_piv.empty else pd.DataFrame()
            ch_p  = ch_piv.reindex(index=common2, columns=_SWEEP_GROUP_ORDER) if not ch_piv.empty else pd.DataFrame()
            vu_p  = vu_piv.reindex(index=common2, columns=_SWEEP_GROUP_ORDER)
            fig, axes = plt.subplots(1, 7, figsize=(38, 5))
            _draw_sweep_heatmap(axes[0], ref_p, f"LLaVA-1.6 — {source}",
                                _n_records(dfs["llava"], source))
            _draw_sweep_heatmap(axes[1], vi_p,  f"VILA v1.0 — {source}",
                                _n_records(dfs["vila"], source))
            _draw_sweep_heatmap(axes[2], q_p,   f"Qwen2.5-VL — {source}",
                                _n_records(dfs["qwen"], source))
            _draw_sweep_heatmap(axes[3], ch_p,  f"Chameleon — {source}",
                                _n_records(dfs["chameleon"], source))
            _draw_sweep_heatmap(axes[4], vu_p,  f"VILA-U — {source}",
                                _n_records(dfs["vilau"], source))
            _draw_sweep_diff(axes[5], vu_p  - ref_p, "Δ  (VILA-U − LLaVA)")
            _draw_sweep_diff(axes[6], ch_p  - ref_p, "Δ  (Chameleon − LLaVA)")
            fig.suptitle(f"5-model causal restoration — {source}", fontsize=13)
            fig.tight_layout()
            savefig(f"comparison_4model/compare_{source}.png", fig)

        # ── Set 3: VQ-focus comparison (comparison_filtered/) ──
        liq_piv = _sweep_pivot(dfs["liquid"],   source)
        vq_piv  = _sweep_pivot(dfs["llava_vq"], source)
        common3 = ref_piv.index.intersection(vu_piv.index)
        for p in [ch_piv, liq_piv, vq_piv]:
            if not p.empty:
                common3 = common3.intersection(p.index)
        if len(common3) > 0:
            ref_p = ref_piv.reindex(index=common3, columns=_SWEEP_GROUP_ORDER)
            ch_p  = ch_piv.reindex(index=common3, columns=_SWEEP_GROUP_ORDER) if not ch_piv.empty else pd.DataFrame()
            liq_p = liq_piv.reindex(index=common3, columns=_SWEEP_GROUP_ORDER) if not liq_piv.empty else pd.DataFrame()
            vq_p  = vq_piv.reindex(index=common3, columns=_SWEEP_GROUP_ORDER) if not vq_piv.empty else pd.DataFrame()
            vu_p  = vu_piv.reindex(index=common3, columns=_SWEEP_GROUP_ORDER)
            fig, axes = plt.subplots(1, 7, figsize=(38, 5))
            _draw_sweep_heatmap(axes[0], ref_p,  f"LLaVA-1.6 — {source}",
                                _n_records(dfs["llava"], source))
            _draw_sweep_heatmap(axes[1], ch_p,   f"Chameleon — {source}",
                                _n_records(dfs["chameleon"], source))
            _draw_sweep_heatmap(axes[2], liq_p,  f"Liquid — {source}",
                                _n_records(dfs["liquid"], source))
            _draw_sweep_heatmap(axes[3], vq_p,   f"LLaVA-VQ — {source}",
                                _n_records(dfs["llava_vq"], source))
            _draw_sweep_heatmap(axes[4], vu_p,   f"VILA-U — {source}",
                                _n_records(dfs["vilau"], source))
            _draw_sweep_diff(axes[5], vu_p  - ref_p, "Δ  (VILA-U − LLaVA)")
            _draw_sweep_diff(axes[6], liq_p - ref_p, "Δ  (Liquid − LLaVA)")
            fig.suptitle(f"VQ-focus causal restoration — {source}", fontsize=13)
            fig.tight_layout()
            savefig(f"comparison_filtered/compare_{source}.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import traceback

    tasks = [
        ("Fig  1 — Codebook-bias scatter (4 conditions)",  fig1_codebook_scatter),
        ("Fig  2 — CHAIR bar chart",                       fig2_chair_bar),
        ("Fig  3 — C2→C6 inset",                          fig3_c2_c6_inset),
        ("Fig  4 — Binary-vs-CHAIR dissociation",          fig4_dissociation),
        ("Fig  5 — CI error bars",                         fig5_ci_errorbars),
        ("Fig  6 — Head-weight 5-panel heatmaps",          fig6_heatmaps_4panel),
        ("Fig  7 — A.2 cross-model bar chart",             fig7_a2_bar),
        ("Fig  8 — A.1 residual-div curves",               fig8_a1_curves),
        ("Fig 10 — 4-panel gate comparison",               fig10_4panel_gate),
        ("Fig 11 — K-sweep polarity",                      fig11_k_sweep),
        ("Fig 12 — Induction/reversal/control",            fig12_induction_reversal),
        ("Fig 13 — L1 cross-arch polarity",                fig13_l1_polarity),
        ("Fig 14 — Architecture-specificity bar",          fig14_arch_specificity),
        ("Regen  — gate_status.png",                       regen_gate_status),
        ("Regen  — benchmark_summary.png",                 regen_benchmark_summary),
        ("Regen  — comparison figures (current cohort)",   regen_comparison_figures),
    ]

    # optionally run only subset via args: python gen_all_figures.py 1 3 5
    run_indices = set(int(a) for a in sys.argv[1:]) if len(sys.argv) > 1 else None

    errors = []
    for idx, (desc, fn) in enumerate(tasks, 1):
        if run_indices and idx not in run_indices:
            continue
        print(f"\n[{idx:02d}/{len(tasks)}] {desc}")
        try:
            fn()
        except Exception:
            print(f"  ERROR:")
            traceback.print_exc()
            errors.append(desc)

    print(f"\n{'='*60}")
    print(f"Done. {len(tasks) - len(errors)}/{len(tasks)} figures generated.")
    if errors:
        print("Failed:")
        for e in errors:
            print(f"  - {e}")
    print("Note: Figure 9 (3-gate flowchart) is an authorial schematic — not generated here.")
    print("Fig 1 → codebook_bias/scatter_4conditions.png  (4-condition, FSQ replication pair)")
    print("Fig 6 → attn_heads/heatmap_5panel_new_specimens.png  (now includes VILA-U reference)")
