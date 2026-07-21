"""
Gate A.1 divergence-curve panel: three architectural signatures.

Plots per-layer relative residual divergence (mean_rel_div) for three models
that exhibit qualitatively distinct depth profiles:

  - VILA-U   (VQ):   peak-and-decline  — early peak, late collapse
  - SEED-LLaMA:      monotonic ramp    — slow climb to a sustained plateau
  - Show-o:          flat              — never crosses the A.1 threshold

Gate A.1 criterion: rel_div >= 0.4 in >= 4 of the first 8 layers (dashed line).
x-axis is normalized network depth so the 24-layer Show-o and 32-layer models
are shape-comparable.

Usage
-----
    python -m probe.tracing.a1_divergence_panel \\
        [--results results/] [--out figures/residual_divergence/]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_MODELS = {
    "vilau": {
        "label": "VILA-U (VQ): peak and decline",
        "color": "#e6194b",
        "marker": "^",
    },
    "seed": {
        "label": "SEED-LLaMA: monotonic ramp",
        "color": "#3cb44b",
        "marker": "o",
    },
    "showo": {
        "label": "Show-o (MAGVIT-v2): flat",
        "color": "#4363d8",
        "marker": "s",
    },
}

_THRESHOLD = 0.4


def _load(results_dir: Path, backend: str) -> dict | None:
    p = results_dir / f"residual_divergence_{backend}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def make_figure(results_dir: Path, out_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figure.")
        return None

    datasets = {b: _load(results_dir, b) for b in _MODELS}
    available = {b: d for b, d in datasets.items() if d is not None}
    if not available:
        print("No residual_divergence_*.json files found.")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.axhline(_THRESHOLD, color="#888888", ls="--", lw=1.2, zorder=1)
    ax.text(
        0.015, _THRESHOLD + 0.012,
        "Gate A.1 threshold (rel_div = 0.4)",
        ha="left", va="bottom", fontsize=8.5, color="#555555",
    )
    # Early-layer window [0, 8) where the gate is evaluated.
    ax.axvspan(0.0, 8.0 / 32.0, color="#fffacd", zorder=0, label="Early window [0, 8)")

    for backend, cfg in _MODELS.items():
        d = available.get(backend)
        if d is None:
            continue
        rel = d["mean_rel_div"]
        n = len(rel)
        depth = [i / (n - 1) for i in range(n)]
        ax.plot(
            depth, rel,
            color=cfg["color"], marker=cfg["marker"], ms=4.5, lw=1.8,
            label=f"{cfg['label']}  (n={n}, σ={d['sigma']})",
            zorder=3,
        )

    ax.set_xlabel("Normalized network depth (layer index / final layer)")
    ax.set_ylabel(r"Relative residual divergence at prompt_last"
                  "\n"
                  r"$\;\|\Delta\,\mathrm{residual}\|_2 \,/\, \|\mathrm{residual}\|_2$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    fig.tight_layout()
    out_path = out_dir / "a1_divergence_panel.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures/residual_divergence"))
    args = ap.parse_args()
    make_figure(args.results, args.out)


if __name__ == "__main__":
    main()
