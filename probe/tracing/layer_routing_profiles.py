"""
Per-layer routing profiles across model families (gate A.2 justification).

For each PASS-ALL and FAIL model, plots head-weight mass on visual tokens
versus layer.  Two-panel top row (per-layer profiles, split by architecture
type) + three-panel bottom row (representative head × layer heatmaps) shows:

  - PASS-ALL projector-amplified models spike at L0 (VILA-U, LLaVA-VQ)
  - PASS-ALL unified-vocab models are concentrated in [0, 8) (Chameleon,
    Liquid, Anole) — architectural reason for the two-tier criterion
  - FAIL models either fall below the L0 ≥ 15 threshold (Janus-Pro, Emu3)
    or distribute mass uniformly across all layers (Qwen-VQ, LLaVA-MLP)
  - The threshold of 15 cleanly separates PASS from FAIL for proj-amp models

Usage
-----
    python -m probe.tracing.layer_routing_profiles
    python -m probe.tracing.layer_routing_profiles --results results --out figures/mechanism
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Cohort definition
# ---------------------------------------------------------------------------
# (result_stem, display_label, verdict, arch_type)
# verdict: "pass" | "fail"
# arch_type: "proj_amp" | "unified_vocab"
PASS_PROJ = [
    ("vilau",    "VILA-U",    "pass", "proj_amp"),
    ("llava_vq", "LLaVA-VQ", "pass", "proj_amp"),
]
FAIL_PROJ = [
    ("janus",             "Janus-Pro",  "fail", "proj_amp"),
    ("qwen_vq",           "Qwen-VQ",    "fail", "proj_amp"),
    ("llava_mlp_control", "LLaVA-MLP\n(control)", "fail", "proj_amp"),
]
PASS_UNIV = [
    ("anole",     "Anole",     "pass", "unified_vocab"),
    ("chameleon", "Chameleon", "pass", "unified_vocab"),
    ("liquid",    "Liquid",    "pass", "unified_vocab"),
]
FAIL_UNIV = [
    ("emu3",    "Emu3",    "fail", "unified_vocab"),
]

ALL_MODELS = PASS_PROJ + FAIL_PROJ + PASS_UNIV + FAIL_UNIV

# Heatmap exemplars: (stem, label, verdict)
HEATMAP_MODELS = [
    ("vilau",    "VILA-U\n(PASS, proj-amp)",    "pass"),
    ("liquid",   "Liquid\n(PASS, unified-vocab)",  "pass"),
    ("janus",    "Janus-Pro\n(FAIL, L0 < 15)",  "fail"),
    ("qwen_vq",  "Qwen-VQ\n(FAIL, no circuit)", "fail"),
]

GATE_A2_THRESHOLD = 15.0
WINDOW = 8  # [0, 8) for unified-vocab

# Style
PASS_CMAP = plt.cm.tab10
FAIL_COLOR = "#888888"
FAIL_ALPHA = 0.75
FAIL_LS = "--"
THRESHOLD_COLOR = "#d62728"

PROJ_PASS_COLORS  = [PASS_CMAP(0), PASS_CMAP(1)]           # blue, orange
UNIV_PASS_COLORS  = [PASS_CMAP(3), PASS_CMAP(2), PASS_CMAP(4)]  # purple, green, red
FAIL_PROJ_COLORS  = ["#999999", "#bbbbbb", "#cccccc"]
FAIL_UNIV_COLORS  = ["#aaaaaa"]


def _load(results_dir: Path, stem: str) -> dict | None:
    path = results_dir / f"attn_head_weights_{stem}.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _layer_profile(data: dict) -> np.ndarray:
    return np.asarray(data["mean_layer_visual_sum"], dtype=float)


def _head_matrix(data: dict) -> np.ndarray:
    return np.asarray(data["mean_head_visual_sum"], dtype=float)  # (n_layers, n_heads)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def _draw_profile_panel(
    ax: plt.Axes,
    entries: list[tuple],          # (stem, label, verdict, arch) pairs
    colors: list,
    results_dir: Path,
    *,
    title: str,
    show_threshold: bool = True,
    show_window_shade: bool = False,
    max_layer: int = 16,
    threshold: float = GATE_A2_THRESHOLD,
) -> None:
    """Draw per-layer routing profiles on ax."""
    loaded: list[tuple[dict, str, str, str]] = []
    for stem, label, verdict, _arch in entries:
        data = _load(results_dir, stem)
        if data is not None:
            loaded.append((data, label, verdict, stem))

    if not loaded:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Determine y-axis cap: clip LLaVA-MLP which dominates (backbone routing)
    YMAX_CLIP = 60.0

    pass_idx = 0
    fail_idx = 0
    for (data, label, verdict, stem), color in zip(loaded, colors):
        profile = _layer_profile(data)[:max_layer]
        layers = np.arange(len(profile))
        profile_clipped = np.minimum(profile, YMAX_CLIP)

        if verdict == "pass":
            ax.plot(
                layers, profile_clipped,
                color=color, linewidth=2.0, marker="o", markersize=3,
                label=label, zorder=3,
            )
            pass_idx += 1
        else:
            # Dashed line for FAIL, label with note if clipped
            display_label = label
            if profile.max() > YMAX_CLIP and stem == "llava_mlp_control":
                display_label = label + "\n(backbone routing)"
            ax.plot(
                layers, profile_clipped,
                color=color, linewidth=1.5, linestyle=FAIL_LS,
                alpha=FAIL_ALPHA, label=display_label, zorder=2,
            )
            fail_idx += 1

    # [0, 8) window shading
    if show_window_shade:
        ax.axvspan(-0.5, WINDOW - 0.5, alpha=0.08, color="#1f77b4", zorder=0,
                   label=f"[0, {WINDOW}) window")

    # L0 column shading
    ax.axvspan(-0.5, 0.5, alpha=0.12, color="#d62728", zorder=0)

    # Threshold line
    if show_threshold:
        ax.axhline(threshold, color=THRESHOLD_COLOR, linewidth=1.5,
                   linestyle=":", zorder=4, label=f"Gate A.2 threshold ({threshold:.0f})")

    ax.set_xlim(-0.5, max_layer - 0.5)
    ymax_display = min(YMAX_CLIP * 1.05, max(
        _layer_profile(_load(results_dir, s))[:max_layer].max()
        for s, *_ in entries
        if _load(results_dir, s) is not None
    ) * 1.1)
    ax.set_ylim(0, ymax_display)

    ax.set_xlabel("Layer index", fontsize=10)
    ax.set_ylabel("Visual attention mass\n(sum over heads)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(range(0, max_layer, 2))
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, framealpha=0.85, ncol=1, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # L0 annotation
    ax.text(0, ax.get_ylim()[1] * 0.98, "L0", ha="center", va="top",
            fontsize=8, color=THRESHOLD_COLOR, fontweight="bold")


def _draw_heatmap_panel(
    ax: plt.Axes,
    data: dict,
    *,
    title: str,
    verdict: str,
    max_layer: int = 16,
    normalize: bool = True,
):
    """Draw head × layer heatmap for one model (per-model normalized by default)."""
    mat = _head_matrix(data)[:max_layer, :]  # (layer, head)

    vmax = float(mat.max()) if normalize else None
    vmax = max(vmax or 1e-6, 1e-6)

    im = ax.imshow(
        mat,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0,
        vmax=vmax,
    )

    n_layers, n_heads = mat.shape
    ax.set_xlabel("Head index", fontsize=9)
    ax.set_ylabel("Layer", fontsize=9)
    ax.set_xticks(range(0, n_heads, max(1, n_heads // 8)))
    ax.set_yticks(range(0, n_layers, 2))
    ax.tick_params(labelsize=8)

    border_color = "#2ca02c" if verdict == "pass" else "#d62728"
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2.5)

    verdict_label = "PASS" if verdict == "pass" else "FAIL"
    verdict_bg = "#2ca02c" if verdict == "pass" else "#d62728"
    ax.text(
        0.02, 0.97, verdict_label,
        transform=ax.transAxes, fontsize=9, fontweight="bold",
        va="top", ha="left", color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=verdict_bg, alpha=0.9),
    )

    ax.set_title(title, fontsize=9, pad=4)
    return im


def _make_profiles_figure(
    all_data: dict[str, dict | None],
    results_dir: Path,
    out_dir: Path,
) -> None:
    """Figure 1: per-layer routing profile line plots (panels A + B)."""
    MAX_LAYER = 16

    fig, (ax_proj, ax_univ) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Per-layer routing profiles across model families\n"
        "Head-weight mass on visual tokens by layer — gate A.2 threshold = 15",
        fontsize=12, fontweight="bold",
    )
    fig.subplots_adjust(wspace=0.30, top=0.86, bottom=0.13)

    # Panel A: Projector-amplified
    _draw_profile_panel(
        ax_proj,
        PASS_PROJ + FAIL_PROJ,
        PROJ_PASS_COLORS + FAIL_PROJ_COLORS,
        results_dir,
        title="(A)  Projector-amplified VQ models",
        show_threshold=True,
        show_window_shade=False,
        max_layer=MAX_LAYER,
    )
    ax_proj.text(
        MAX_LAYER - 1, GATE_A2_THRESHOLD + 0.8,
        "L0 ≥ 15: PASS →",
        ha="right", va="bottom",
        fontsize=8, color=THRESHOLD_COLOR, fontweight="bold",
    )

    # Panel B: Unified-vocab
    _draw_profile_panel(
        ax_univ,
        PASS_UNIV + FAIL_UNIV,
        UNIV_PASS_COLORS + FAIL_UNIV_COLORS,
        results_dir,
        title="(B)  Unified-vocab VQ models",
        show_threshold=True,
        show_window_shade=True,
        max_layer=MAX_LAYER,
    )
    ax_univ.set_ylim(0, 35)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_dir / f"layer_routing_profiles.{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"Figure written to {p}")
    plt.close(fig)


def _make_heatmaps_figure(
    heat_data: dict[str, dict | None],
    out_dir: Path,
) -> None:
    """Figure 2: head × layer attention heatmaps for representative models."""
    MAX_LAYER = 16
    n = len(HEATMAP_MODELS)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    fig.suptitle(
        "Visual-to-answer routing heatmaps (head × layer)\n"
        "Each cell = mean attention mass from last prompt token to visual tokens",
        fontsize=11, fontweight="bold",
    )
    fig.subplots_adjust(wspace=0.38, top=0.84, bottom=0.10)

    last_im = None
    for ax, (stem, label, verdict) in zip(axes, HEATMAP_MODELS):
        data = heat_data.get(stem)
        if data is None:
            ax.text(0.5, 0.5, f"{stem}\nnot found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            continue
        panel_letter = chr(ord("A") + HEATMAP_MODELS.index((stem, label, verdict)))
        im = _draw_heatmap_panel(
            ax, data,
            title=f"({panel_letter})  {label}",
            verdict=verdict,
            max_layer=MAX_LAYER,
            normalize=True,
        )
        n_heads = _head_matrix(data).shape[1]
        ax.axhline(0.5, color="cyan", linewidth=0.8, alpha=0.6, linestyle=":")
        ax.text(n_heads - 0.5, 0, "L0", color="cyan", fontsize=7, ha="right", va="bottom")
        last_im = im

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes[-1], shrink=0.85, pad=0.04)
        cbar.set_label("Norm. attention mass\n(per-model max = 1)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_dir / f"routing_heatmaps.{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"Figure written to {p}")
    plt.close(fig)


def make_figure(results_dir: Path, out_dir: Path) -> None:
    all_data: dict[str, dict | None] = {
        stem: _load(results_dir, stem)
        for stem, *_ in ALL_MODELS
    }
    heat_data: dict[str, dict | None] = {
        stem: _load(results_dir, stem)
        for stem, *_ in HEATMAP_MODELS
    }

    if not any(d is not None for d in all_data.values()):
        print("No attn_head_weights_*.json files found. Run attn_head_weights.py first.")
        return

    _make_profiles_figure(all_data, results_dir, out_dir)
    _make_heatmaps_figure(heat_data, out_dir)

    # Summary table
    print("\n" + "=" * 72)
    print("GATE A.2 ROUTING SUMMARY")
    print("=" * 72)
    print(f"{'Model':<22} {'Verdict':6} {'Arch':14} {'L0 mass':>9}  {'[0,8) total':>11}  {'≥ 15':>6}")
    print("-" * 72)
    for stem, label, verdict, arch in ALL_MODELS:
        data = all_data.get(stem)
        if data is None:
            continue
        prof = _layer_profile(data)
        l0 = prof[0]
        w8 = prof[:8].sum()
        check = "✓" if (l0 >= 15 and arch == "proj_amp") or (w8 >= 15 and arch == "unified_vocab") else "✗"
        print(f"{label.replace(chr(10), ' '):<22} {verdict.upper():6} {arch:14} {l0:>9.2f}  {w8:>11.2f}  {check:>6}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-layer routing profiles across model families (gate A.2)"
    )
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures/mechanism")
    args = parser.parse_args()
    make_figure(Path(args.results), Path(args.out))


if __name__ == "__main__":
    main()
