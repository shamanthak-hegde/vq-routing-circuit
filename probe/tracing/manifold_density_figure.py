"""
Cross-model manifold-density figure.

Reads per-backend manifold_density_<backend>.json summaries and existing
sweep_<backend>.jsonl files, then produces the headline scatter + regression
figure: intrinsic dimensionality (x) vs POPE WARN rate (y), coloured by
early-visual circuit score.

Outputs
-------
  results/manifold_summary.tsv          — tab-separated summary table
  figures/manifold/density_vs_warn.png  — headline scatter + regression
  figures/manifold/density_panels.png   — supplementary side panels

Usage
─────
    python -m probe.tracing.manifold_density_figure \\
        --results results/ --out figures/manifold/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ── sweep JSONL parsing ───────────────────────────────────────────────────────

_SWEEP_FILES = {
    "llava":  ("sweep_llava.jsonl", "sweep_pope_multiseed.jsonl"),
    "vila":   ("sweep_vila.jsonl",),
    "vilau":  ("sweep_vilau.jsonl",),
    "unitok": ("sweep_unitok.jsonl",),
}


def _load_sweep_stats(backend: str, results_dir: Path) -> dict:
    """Compute WARN rate and early-visual score from sweep JSONL(s)."""
    paths = [results_dir / p for p in _SWEEP_FILES.get(backend, [])]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return {}

    # Collect per-record POPE data
    pope_warn: dict[str, bool] = {}
    nb_warn: dict[str, bool]   = {}

    # Layer causal weights for circuit-type scalar
    # early_visual_score = mean visual score in [0,8) minus mean prompt_last in [20,32)
    vis_early: list[float]    = []  # scores for visual tokens, layers 0-7
    pl_late:   list[float]    = []  # scores for prompt_last tokens, layers 20+

    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rid    = row["record_id"]
                source = row.get("source", "")
                is_warn = row["logit_clean"] <= row["logit_corrupt"]

                if source == "pope":
                    if rid not in pope_warn:
                        pope_warn[rid] = is_warn
                    else:
                        pope_warn[rid] = pope_warn[rid] or is_warn
                elif source == "naturalbench":
                    if rid not in nb_warn:
                        nb_warn[rid] = is_warn
                    else:
                        nb_warn[rid] = nb_warn[rid] or is_warn

                # Circuit-type accumulation
                ls = row.get("layer_start", -1)
                le = row.get("layer_end", -1)
                grp = row.get("token_group", "")
                score = row.get("score", 0.0)

                if grp == "visual" and ls < 8 and not is_warn:
                    vis_early.append(score)
                if grp == "prompt_last" and ls >= 20 and not is_warn:
                    pl_late.append(score)

    warn_rate_pope = (
        sum(pope_warn.values()) / len(pope_warn) if pope_warn else float("nan")
    )
    warn_rate_nb = (
        sum(nb_warn.values()) / len(nb_warn) if nb_warn else float("nan")
    )
    early_visual_score = (
        (sum(vis_early) / len(vis_early) if vis_early else 0.0)
        - (sum(pl_late)  / len(pl_late)  if pl_late  else 0.0)
    )

    return {
        "warn_rate_pope":    round(warn_rate_pope, 4),
        "warn_rate_nb":      round(warn_rate_nb, 4),
        "early_visual_score": round(early_visual_score, 4),
        "n_pope_records":    len(pope_warn),
        "n_nb_records":      len(nb_warn),
    }


# ── aggregate ─────────────────────────────────────────────────────────────────

def collect_rows(results_dir: Path) -> list[dict]:
    rows = []
    for backend in ("llava", "vila", "vilau", "unitok"):
        density_json = results_dir / f"manifold_density_{backend}.json"
        if not density_json.exists():
            continue

        with open(density_json) as f:
            dens = json.load(f)

        sweep_stats = _load_sweep_stats(backend, results_dir)
        vq = dens.get("vq", {})
        row = {
            "backend":                backend,
            "intrinsic_dim_twonn":    dens.get("intrinsic_dim_twonn"),
            "intrinsic_dim_pca95":    dens.get("intrinsic_dim_pca95"),
            "mean_cross_nn_cos_dist": dens.get("mean_cross_nn_cos_dist"),
            "vq_util_mean":           vq.get("mean_utilization"),
            "vq_entropy_mean":        vq.get("mean_entropy_nats"),
            "warn_rate_pope":         sweep_stats.get("warn_rate_pope"),
            "warn_rate_nb":           sweep_stats.get("warn_rate_nb"),
            "early_visual_score":     sweep_stats.get("early_visual_score"),
        }
        rows.append(row)

    return rows


def write_tsv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(out_path, "w") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join("" if row[h] is None else str(row[h]) for h in headers) + "\n")
    print(f"Summary TSV written to {out_path}")


# ── OLS regression ────────────────────────────────────────────────────────────

def _ols(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, R²) for OLS fit of y ~ x."""
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if len(xa) < 2:
        return float("nan"), float("nan"), float("nan")
    xm, ym = xa.mean(), ya.mean()
    slope = float(((xa - xm) * (ya - ym)).sum() / ((xa - xm) ** 2).sum())
    intercept = float(ym - slope * xm)
    y_pred = slope * xa + intercept
    ss_res = float(((ya - y_pred) ** 2).sum())
    ss_tot = float(((ya - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


# ── figure ────────────────────────────────────────────────────────────────────

_BACKEND_LABELS = {
    "llava":  "LLaVA-1.6",
    "vila":   "VILA",
    "vilau":  "VILA-U",
    "unitok": "UniTok",
}

_BACKEND_MARKERS = {
    "llava":  "o",
    "vila":   "s",
    "vilau":  "^",
    "unitok": "D",
}


def _make_figure(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not available — skipping figure generation.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    valid_rows = [r for r in rows if r["intrinsic_dim_twonn"] is not None
                  and r["warn_rate_pope"] is not None]
    if len(valid_rows) < 2:
        print(f"Only {len(valid_rows)} rows with intrinsic_dim_twonn + warn_rate_pope — "
              "skipping figure.")
        return

    # ── Colour by early_visual_score ──────────────────────────────────────────
    ev_scores = [r["early_visual_score"] for r in valid_rows
                 if r["early_visual_score"] is not None]
    if ev_scores:
        ev_min, ev_max = min(ev_scores), max(ev_scores)
        ev_range = ev_max - ev_min or 1.0
        norm = mcolors.Normalize(vmin=ev_min - 0.1 * ev_range,
                                 vmax=ev_max + 0.1 * ev_range)
        cmap = cm.RdBu_r
    else:
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = cm.viridis

    # ── Main scatter + regression ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def _scatter_ax(ax, x_key, y_key, x_label, y_label, title):
        xs, ys, cs, labels, markers = [], [], [], [], []
        for r in rows:
            xv = r.get(x_key)
            yv = r.get(y_key)
            ev = r.get("early_visual_score")
            if xv is None or yv is None or math.isnan(float(xv)) or math.isnan(float(yv)):
                continue
            xs.append(float(xv))
            ys.append(float(yv))
            cs.append(float(ev) if ev is not None else 0.0)
            labels.append(_BACKEND_LABELS.get(r["backend"], r["backend"]))
            markers.append(_BACKEND_MARKERS.get(r["backend"], "o"))

        if len(xs) < 2:
            ax.set_title(title + " (insufficient data)")
            return

        slope, intercept, r2 = _ols(xs, ys)

        # Regression line
        x_sorted = sorted(xs)
        x_lo, x_hi = min(xs), max(xs)
        margin = max((x_hi - x_lo) * 0.1, 0.5)
        xfit = np.linspace(x_lo - margin, x_hi + margin, 100)
        yfit = slope * xfit + intercept
        ax.plot(xfit, yfit, "k--", lw=1.2, alpha=0.6, zorder=1)
        ax.text(
            0.98, 0.05,
            f"slope={slope:.3f}\n$R^2$={r2:.2f}\n(n={len(xs)}, descriptive)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

        # Scatter
        for x, y, c, lbl, mk in zip(xs, ys, cs, labels, markers):
            color = cmap(norm(c))
            ax.scatter(x, y, c=[color], marker=mk, s=120, zorder=3, edgecolors="k", lw=0.5)
            ax.annotate(
                lbl, (x, y),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    _scatter_ax(
        axes[0],
        x_key="intrinsic_dim_twonn",
        y_key="warn_rate_pope",
        x_label="Intrinsic dimensionality (TwoNN)",
        y_label="POPE WARN rate",
        title="Manifold richness vs susceptibility",
    )
    _scatter_ax(
        axes[1],
        x_key="intrinsic_dim_twonn",
        y_key="mean_cross_nn_cos_dist",
        x_label="Intrinsic dimensionality (TwoNN)",
        y_label="Mean cross-record NN cosine distance",
        title="ID vs cross-record NN distance",
    )
    _scatter_ax(
        axes[2],
        x_key="intrinsic_dim_twonn",
        y_key="warn_rate_nb",
        x_label="Intrinsic dimensionality (TwoNN)",
        y_label="NaturalBench WARN rate",
        title="Manifold richness vs NB WARN rate",
    )

    # Colourbar for early_visual_score
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.01)
    cbar.set_label("Early-visual circuit score\n(visual[0,8) − prompt_last[20,32))", fontsize=8)

    fig.suptitle(
        "Cross-model manifold-density vs circuit susceptibility",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    out_path = out_dir / "density_vs_warn.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out_path}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-model manifold-density figure"
    )
    parser.add_argument("--results", default="results",
                        help="Directory containing manifold_density_*.json and sweep_*.jsonl")
    parser.add_argument("--out", default="figures/manifold",
                        help="Output directory for figures (default: figures/manifold/)")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)

    rows = collect_rows(results_dir)
    if not rows:
        print("No manifold_density_*.json files found in results/. Run manifold_density.py first.")
        return

    print(f"\nFound {len(rows)} backends: {[r['backend'] for r in rows]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(rows, results_dir / "manifold_summary.tsv")
    _make_figure(rows, out_dir)

    # Print table
    sep = "=" * 80
    print(f"\n{sep}")
    print("MANIFOLD DENSITY SUMMARY TABLE")
    print(sep)
    headers = [
        "backend", "id_twonn", "id_pca95", "cross_nn_dist",
        "vq_util", "vq_ent", "warn_pope", "warn_nb", "ev_score",
    ]
    print("  ".join(f"{h:>12}" for h in headers))
    print("  ".join("-" * 12 for _ in headers))
    for r in rows:
        def _fmt(v):
            if v is None:
                return f"{'—':>12}"
            try:
                return f"{float(v):>12.4f}"
            except Exception:
                return f"{str(v):>12}"

        print("  ".join([
            f"{r['backend']:>12}",
            _fmt(r["intrinsic_dim_twonn"]),
            _fmt(r["intrinsic_dim_pca95"]),
            _fmt(r["mean_cross_nn_cos_dist"]),
            _fmt(r["vq_util_mean"]),
            _fmt(r["vq_entropy_mean"]),
            _fmt(r["warn_rate_pope"]),
            _fmt(r["warn_rate_nb"]),
            _fmt(r["early_visual_score"]),
        ]))
    print(sep)

    # OLS on all backends with both metrics
    xs = [r["intrinsic_dim_twonn"] for r in rows if r["intrinsic_dim_twonn"] is not None
          and r["warn_rate_pope"] is not None]
    ys = [r["warn_rate_pope"]     for r in rows if r["intrinsic_dim_twonn"] is not None
          and r["warn_rate_pope"] is not None]
    slope, intercept, r2 = _ols(xs, ys)
    print(f"\nOLS (ID vs POPE WARN rate):  slope={slope:.4f}  R²={r2:.4f}  (n={len(xs)}, descriptive)")
    print("Note: n=4 maximum — regression is descriptive, not inferential.")


if __name__ == "__main__":
    main()
