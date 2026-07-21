"""
Figures for per-head visual-to-prompt_last attention weights.

Renders per-backend heatmaps plus a VILA-U vs UniTok comparison. --include_noisy
adds a clean-vs-noisy VILA-U side-by-side panel.

Usage
-----
    # clean only
    python -m probe.tracing.attn_head_weights_figure

    # clean + noisy (reads attn_head_weights_vilau_clean_noisy.json)
    python -m probe.tracing.attn_head_weights_figure --include_noisy
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


_BACKENDS = {
    "vilau": {
        "title": "VILA-U: visual -> prompt_last attention",
        "target": [0, 8],
    },
    "unitok": {
        "title": "UniTok: visual -> prompt_last attention",
        "target": [12, 14],
    },
}


def _load(results_dir: Path, backend: str) -> dict | None:
    path = results_dir / f"attn_head_weights_{backend}.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _matrix(data: dict, noisy: bool = False):
    import numpy as np

    key = "noisy_mean_head_visual_sum" if noisy else "mean_head_visual_sum"
    return np.asarray(data[key], dtype=float)


def _target_window(data: dict, backend: str) -> tuple[int, int]:
    if "target_window" in data:
        lo, hi = data["target_window"]
        return int(lo), int(hi)
    lo, hi = _BACKENDS[backend]["target"]
    return int(lo), int(hi)


def _draw_heatmap(ax, data: dict, backend: str, vmax: float, noisy: bool = False):
    import matplotlib.patches as patches

    mat = _matrix(data, noisy=noisy)
    label = "noisy" if noisy else "clean"
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
    title = _BACKENDS[backend]["title"]
    if noisy:
        title = title.replace("attention", "attention [noisy]")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xlim(-0.5, n_heads - 0.5)
    ax.set_ylim(-0.5, n_layers - 0.5)
    ax.set_xticks(list(range(0, n_heads, 4)))
    ax.set_yticks(list(range(0, n_layers, 4)))

    lo, hi = _target_window(data, backend)
    rect = patches.Rectangle(
        (-0.5, lo - 0.5),
        n_heads,
        hi - lo,
        linewidth=1.5,
        edgecolor="#5fe3ff",
        facecolor="none",
    )
    ax.add_patch(rect)

    for row in data.get("candidates", [])[:3]:
        layer = int(row["layer"])
        head = int(row["head"])
        ax.scatter(head, layer, s=80, facecolors="none", edgecolors="white", linewidths=1.2)
        ax.text(
            head,
            layer,
            str(row["rank"]),
            color="white",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
        )

    ax.text(
        0.01,
        1.02,
        f"n={data['n_records']}  target=L{lo}-{hi - 1}",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
    )
    return im


def _print_table(datasets: dict[str, dict]) -> None:
    print("\n" + "=" * 72)
    print("VISUAL -> PROMPT_LAST ATTENTION CANDIDATES")
    print("=" * 72)
    for backend, data in datasets.items():
        print(f"\n{backend.upper()}  n={data['n_records']}  target_window={data['target_window']}")
        print(f"  {'rank':>4}  {'layer':>5}  {'head':>4}  {'mean_sum':>10}  {'mean_max':>10}")
        print("  " + "-" * 43)
        for row in data.get("candidates", [])[:10]:
            print(
                f"  {row['rank']:>4}  {row['layer']:>5}  {row['head']:>4}  "
                f"{row['mean_sum']:>10.6f}  {row['mean_max']:>10.6f}"
            )
    print("=" * 72)


def _load_clean_noisy(results_dir: Path, backend: str) -> dict | None:
    path = results_dir / f"attn_head_weights_{backend}_clean_noisy.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def make_figures(results_dir: Path, out_dir: Path, include_noisy: bool = False) -> None:
    datasets = {
        backend: data
        for backend in _BACKENDS
        if (data := _load(results_dir, backend)) is not None
    }
    if not datasets:
        print("No attn_head_weights_*.json files found. Run attn_head_weights.py first.")
        return

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figures.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    vmax = max(float(_matrix(data).max()) for data in datasets.values())
    vmax = max(vmax, 1e-6)

    _print_table(datasets)

    for backend, data in datasets.items():
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        im = _draw_heatmap(ax, data, backend, vmax)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("Mean attention mass to visual block")
        fig.tight_layout()
        out_path = out_dir / f"{backend}_visual_to_prompt_last.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"Figure written to {out_path}")
        plt.close(fig)

    if {"vilau", "unitok"}.issubset(datasets):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        im = None
        for ax, backend in zip(axes, ("vilau", "unitok")):
            im = _draw_heatmap(ax, datasets[backend], backend, vmax)
        assert im is not None
        cbar = fig.colorbar(im, ax=axes, shrink=0.85)
        cbar.set_label("Mean attention mass to visual block")
        fig.suptitle(
            "Per-head visual routing to prompt_last",
            fontsize=11,
        )
        out_path = out_dir / "comparison_vilau_vs_unitok.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"Figure written to {out_path}")
        plt.close(fig)

    # clean-vs-noisy panel for VILA-U
    if include_noisy:
        cn_data = _load_clean_noisy(results_dir, "vilau")
        if cn_data is None:
            print(
                "Note: attn_head_weights_vilau_clean_noisy.json not found; "
                "run attn_head_weights.py --include_noisy first."
            )
        elif not cn_data.get("include_noisy"):
            print("Note: loaded VILA-U data does not have noisy measurements.")
        else:
            sigma = cn_data.get("noisy_sigma", "?")
            corr = cn_data.get("l0_clean_noisy_pearson_r", float("nan"))
            vmax_cn = max(
                float(_matrix(cn_data, noisy=False).max()),
                float(_matrix(cn_data, noisy=True).max()),
                1e-6,
            )
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            for ax, noisy in zip(axes, [False, True]):
                im = _draw_heatmap(ax, cn_data, "vilau", vmax_cn, noisy=noisy)
            cbar = fig.colorbar(im, ax=axes, shrink=0.85)
            cbar.set_label("Mean attention mass to visual block")
            fig.suptitle(
                f"VILA-U: clean vs noisy (σ={sigma}) L0 routing  "
                f"(L0 Pearson r={corr:.3f})",
                fontsize=11,
            )
            out_path = out_dir / "clean_vs_noisy_vilau.png"
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"Figure written to {out_path}")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render per-head visual-attention heatmaps"
    )
    parser.add_argument("--results", default="results",
                        help="Directory with attn_head_weights_*.json files")
    parser.add_argument("--out", default="figures/attn_heads",
                        help="Output directory for heatmaps")
    parser.add_argument(
        "--include_noisy",
        action="store_true",
        help="Also render clean-vs-noisy VILA-U panel",
    )
    args = parser.parse_args()

    make_figures(Path(args.results), Path(args.out), include_noisy=args.include_noisy)


if __name__ == "__main__":
    main()
