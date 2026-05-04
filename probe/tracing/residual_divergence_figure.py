"""
Comparative figure: per-layer residual divergence at prompt_last (N-036).

Reads results/residual_divergence_{vilau,unitok}.json and produces a
two-panel figure:
  Panel A: mean absolute divergence per layer (both models, shared y-axis)
  Panel B: divergence normalized by layer-0 value (amplification vs attenuation)

Highlights layers [0, 8) in both panels.

Usage
-----
    python -m probe.tracing.residual_divergence_figure \\
        [--results results/] [--out figures/residual_divergence/]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_BACKENDS = {
    "vilau":  {"label": "VILA-U (VQ, σ=1.0)", "color": "#e6194b", "marker": "^"},
    "unitok": {"label": "UniTok (VQ, σ=0.2)", "color": "#4363d8", "marker": "D"},
}


def _load(results_dir: Path, backend: str) -> dict | None:
    p = results_dir / f"residual_divergence_{backend}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def make_figure(results_dir: Path, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available — skipping figure.")
        return

    datasets = {b: _load(results_dir, b) for b in _BACKENDS}
    available = {b: d for b, d in datasets.items() if d is not None}
    if not available:
        print("No residual_divergence_*.json files found. Run residual_divergence.py first.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    n_layers = max(d["n_layers"] for d in available.values())
    layers = list(range(n_layers))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Per-layer residual divergence at prompt_last position  [N-036]\n"
        "Stage-1 vs Stage-2: does VILA-U amplify the substituted visual signal?",
        fontsize=11,
    )

    highlight = dict(color="#fffacd", zorder=0)

    for ax_idx, (ax, key, ylabel, title) in enumerate(zip(
        axes,
        ["mean_abs_div", "norm_abs_div"],
        [
            "Mean ‖residual_clean − residual_noisy‖₂ at prompt_last",
            "Divergence normalized by layer-0 value\n(> 1 = amplification, < 1 = attenuation)",
        ],
        [
            "Absolute residual divergence",
            "Normalized divergence (amplification / attenuation)",
        ],
    )):
        ax.axvspan(-0.5, 7.5, **highlight, label="Layers [0, 8)")
        ax.axhline(1.0 if key == "norm_abs_div" else 0, color="gray",
                   lw=0.8, ls="--", alpha=0.5)

        for backend, data in available.items():
            meta = _BACKENDS[backend]
            vals = data[key][:n_layers]
            ax.plot(layers[:len(vals)], vals,
                    color=meta["color"], marker=meta["marker"],
                    markersize=4, linewidth=1.5, label=meta["label"])
            # Annotate σ and n at each model
            ax.annotate(
                f"σ={data['sigma']}  n={data['n_records']}",
                xy=(n_layers - 1, vals[-1]),
                xytext=(-60, 5), textcoords="offset points",
                fontsize=7, color=meta["color"],
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-1, n_layers)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "vilau_vs_unitok_divergence.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out_path}")
    plt.close(fig)


def print_table(results_dir: Path) -> None:
    print("\n" + "=" * 72)
    print("RESIDUAL DIVERGENCE COMPARISON — prompt_last position")
    print("=" * 72)
    print(f"  {'backend':>8}  {'sigma':>6}  {'n':>4}  "
          f"{'norm[0,4)':>10}  {'norm[4,8)':>10}  {'norm[24,32)':>12}  verdict")
    print("  " + "-" * 68)
    for backend in ("vilau", "unitok"):
        d = _load(results_dir, backend)
        if d is None:
            print(f"  {backend:>8}  — not found —")
            continue
        na = d["norm_abs_div"]
        early04 = sum(na[:4]) / 4
        early48 = sum(na[4:8]) / 4
        late    = sum(na[24:32]) / 8 if len(na) >= 32 else float("nan")
        early   = (early04 + early48) / 2
        verdict = ("AMPLIF." if early > 1.05 else
                   "ATTEN."  if early < 0.95 else "FLAT")
        print(f"  {backend:>8}  {d['sigma']:>6}  {d['n_records']:>4}  "
              f"{early04:>10.4f}  {early48:>10.4f}  {late:>12.4f}  {verdict}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparative residual-divergence figure (N-036)"
    )
    parser.add_argument("--results", default="results",
                        help="Directory with residual_divergence_*.json files")
    parser.add_argument("--out", default="figures/residual_divergence",
                        help="Output directory for figures")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)

    print_table(results_dir)
    make_figure(results_dir, out_dir)


if __name__ == "__main__":
    main()
