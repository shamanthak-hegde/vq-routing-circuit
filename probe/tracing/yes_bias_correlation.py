"""N-060E: Yes-bias and drift correlation analysis across the full model cohort.

Aggregates per-model metrics from existing sweep, accuracy, bench, and
codebook-probe results. Computes cross-model correlations supporting C4
(yes-bias is the extreme-drift limiting case) and C1 (drift severity
separates VQ-class from continuous-projector class).

Usage
-----
    source activate sae
    python -m probe.tracing.yes_bias_correlation \
        --results_dir results \
        --out results/yes_bias_correlation.json \
        --fig_dir figures/yes_bias
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ── model catalogue ────────────────────────────────────────────────────────────

# Each entry: {sweep, accuracy, bench_pope_baseline, bench_amber_baseline,
#              codebook_probe, model_class, gate_a1, gate_a2}
# gate values: "pass", "fail", "partial", None (not tested)
_MODELS: dict[str, dict[str, Any]] = {
    "vilau": {
        "sweep": "sweep_vilau.jsonl",
        "accuracy": "accuracy_vilau.jsonl",
        "bench_pope_baseline": "bench_pope_full_vilau_baseline.meta.json",
        "bench_amber_baseline": "bench_amber_vilau_baseline.meta.json",
        "codebook_probe": "codebook_probe_vilau.jsonl",
        "model_class": "vq",
        "gate_a1": "pass",   # rel_div 0.93 @ L2 (N-036)
        "gate_a2": "pass",   # L0 mass 17.50 (N-038)
    },
    "unitok": {
        "sweep": "sweep_unitok.jsonl",
        "accuracy": "accuracy_unitok.jsonl",
        "bench_pope_baseline": "bench_pope_full_unitok_baseline.meta.json",
        "bench_amber_baseline": "bench_amber_unitok_baseline.meta.json",
        "codebook_probe": "codebook_probe_unitok.jsonl",
        "model_class": "vq",
        "gate_a1": "pass",   # rel_div 0.49 @ L13 (N-055/R-030)
        "gate_a2": "fail",   # head-mass 1.37 vs threshold 10 (N-055/R-030)
    },
    "seed": {
        "sweep": "sweep_seed.jsonl",
        "accuracy": None,
        "bench_pope_baseline": "bench_pope_seed_baseline.meta.json",
        "bench_amber_baseline": None,
        "codebook_probe": None,
        "model_class": "vq",
        "gate_a1": "partial",  # peaks L14 @ 0.41, misses [8,12) target (R-029)
        "gate_a2": None,
    },
    "showo": {
        "sweep": "sweep_showo.jsonl",
        "accuracy": None,
        "bench_pope_baseline": None,
        "bench_amber_baseline": None,
        "codebook_probe": None,
        "model_class": "vq",
        "gate_a1": "fail",   # flat rel_div 0.09-0.14 (R-028)
        "gate_a2": None,
    },
    "llava_vq": {
        "sweep": "sweep_llava_vq.jsonl",
        "accuracy": "accuracy_llava_vq_baseline.jsonl",
        "bench_pope_baseline": None,
        "bench_amber_baseline": None,
        "codebook_probe": None,
        "model_class": "vq",
        "gate_a1": "pass",   # rel_div peak 1.276 @ L5 (R-031)
        "gate_a2": "pass",   # L0 mass 26.025 (R-031)
    },
    "emu3": {
        "sweep": "sweep_emu3.jsonl",
        "accuracy": None,
        "bench_pope_baseline": None,
        "bench_amber_baseline": None,
        "codebook_probe": None,
        "model_class": "vq",
        "gate_a1": None,
        "gate_a2": None,
    },
    "llava": {
        "sweep": "sweep_llava.jsonl",
        "accuracy": "accuracy_llava.jsonl",
        "bench_pope_baseline": "bench_pope_full_llava_baseline.meta.json",
        "bench_amber_baseline": "bench_amber_llava_baseline.meta.json",
        "codebook_probe": None,
        "model_class": "continuous",
        "gate_a1": None,
        "gate_a2": None,
    },
    "vila": {
        "sweep": "sweep_vila.jsonl",
        "accuracy": "accuracy_vila.jsonl",
        "bench_pope_baseline": None,
        "bench_amber_baseline": None,
        "codebook_probe": None,
        "model_class": "continuous",
        "gate_a1": None,
        "gate_a2": None,
    },
    "qwen3vl": {
        "sweep": "sweep_qwen3vl.jsonl",
        "accuracy": "accuracy_qwen3vl.jsonl",
        "bench_pope_baseline": "bench_pope_full_qwen3vl_baseline.meta.json",
        "bench_amber_baseline": "bench_amber_qwen3vl_baseline.meta.json",
        "codebook_probe": None,
        "model_class": "continuous",
        "gate_a1": None,
        "gate_a2": None,
    },
    "haplo": {
        "sweep": "sweep_haplo_pilot.jsonl",
        "accuracy": None,
        "bench_pope_baseline": None,
        "bench_amber_baseline": None,
        "codebook_probe": None,
        "model_class": "continuous",
        "gate_a1": None,
        "gate_a2": None,
    },
    "lavit": {
        "sweep": "sweep_lavit_pilot.jsonl",
        "accuracy": None,
        "bench_pope_baseline": None,
        "bench_amber_baseline": None,
        "codebook_probe": None,
        "model_class": "continuous",
        "gate_a1": None,
        "gate_a2": None,
    },
}


# ── loaders ────────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sweep_stats(path: Path) -> dict[str, float]:
    """Compute WARN rates and early visual score from a sweep JSONL."""
    rows = _load_jsonl(path)
    if not rows:
        return {}

    # Group by record_id to get per-record logit_clean / logit_corrupt
    from collections import defaultdict
    rec_map: dict[str, dict] = {}
    scores: dict[tuple, float] = {}  # (record_id, layer_start, token_group) → score

    for r in rows:
        rid = r["record_id"]
        src = r.get("source", "")
        rec_map[rid] = {"source": src, "logit_clean": r["logit_clean"],
                        "logit_corrupt": r["logit_corrupt"]}
        scores[(rid, r.get("layer_start", 0), r.get("token_group", ""))] = r["score"]

    # WARN per source
    stats: dict[str, float] = {}
    for src in ("naturalbench", "pope"):
        src_ids = [rid for rid, v in rec_map.items() if v["source"] == src]
        if not src_ids:
            continue
        warn_ids = [rid for rid in src_ids
                    if rec_map[rid]["logit_clean"] <= rec_map[rid]["logit_corrupt"]]
        stats[f"warn_rate_{src}"] = len(warn_ids) / len(src_ids)
        stats[f"n_{src}"] = len(src_ids)

    # Early visual score ([0,4) window, POPE, non-WARN)
    pope_non_warn = {rid for rid, v in rec_map.items()
                     if v["source"] == "pope"
                     and v["logit_clean"] > v["logit_corrupt"]}
    ev_vals = [scores[(rid, 0, "visual")]
               for rid in pope_non_warn
               if (rid, 0, "visual") in scores]
    if ev_vals:
        stats["early_visual_score_pope"] = float(np.mean(ev_vals))

    # Visual >1.0 rate on POPE [0,4) (signature of off-manifold substitution)
    stats["visual_gt1_rate_pope"] = (float(np.mean([v > 1.0 for v in ev_vals]))
                                     if ev_vals else float("nan"))

    return stats


def _accuracy_stats(path: Path) -> dict[str, float]:
    """Compute yes_rate and accuracy from a probe-cache accuracy JSONL."""
    rows = _load_jsonl(path)
    if not rows:
        return {}

    stats: dict[str, float] = {}
    for src in ("naturalbench", "pope"):
        sub = [r for r in rows if r.get("source") == src]
        if not sub:
            continue
        stats[f"acc_{src}"] = float(np.mean([r["correct"] for r in sub]))
        yes_preds = [r for r in sub if str(r.get("pred", "")).lower() in ("yes", "1")]
        stats[f"yes_rate_{src}"] = len(yes_preds) / len(sub)

    # Combined yes_rate
    yes_all = [r for r in rows if str(r.get("pred", "")).lower() in ("yes", "1")]
    stats["yes_rate_all"] = len(yes_all) / len(rows) if rows else float("nan")
    return stats


def _bench_meta_stats(path: Path) -> dict[str, float]:
    """Extract yes_rate and main metric from a bench .meta.json sidecar."""
    with open(path) as f:
        meta = json.load(f)
    metrics = meta.get("metrics", {})
    bench = meta.get("bench", "")
    stats: dict[str, float] = {}
    if "yes_rate" in metrics:
        stats[f"yes_rate_{bench}"] = metrics["yes_rate"]
    if "f1" in metrics:
        stats[f"f1_{bench}"] = metrics["f1"]
    if "accuracy" in metrics:
        stats[f"acc_{bench}"] = metrics["accuracy"]
    return stats


def _codebook_probe_stats(path: Path) -> dict[str, float]:
    """Compute mean drift severity from B_off_manifold test rows."""
    rows = [r for r in _load_jsonl(path) if r.get("test") == "B_off_manifold"]
    if not rows:
        return {}
    cos_self = [r["cos_sim_noisy_to_clean_self"] for r in rows]
    cos_other = [r["cos_sim_noisy_to_nearest_other"] for r in rows]
    closer = [r["noisy_closer_to_other"] for r in rows]
    return {
        "drift_cos_self_mean": float(np.mean(cos_self)),
        "drift_cos_other_mean": float(np.mean(cos_other)),
        "drift_closer_to_other_rate": float(np.mean(closer)),
        "drift_severity": float(np.mean(cos_other)) - float(np.mean(cos_self)),
        "drift_n": len(rows),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def build_cohort_table(results_dir: Path) -> list[dict]:
    rows = []
    for model, cfg in _MODELS.items():
        row: dict[str, Any] = {"model": model,
                                "model_class": cfg["model_class"],
                                "gate_a1": cfg["gate_a1"],
                                "gate_a2": cfg["gate_a2"]}

        # sweep
        if cfg["sweep"] and (results_dir / cfg["sweep"]).exists():
            row.update(_sweep_stats(results_dir / cfg["sweep"]))

        # accuracy
        if cfg["accuracy"] and (results_dir / cfg["accuracy"]).exists():
            row.update(_accuracy_stats(results_dir / cfg["accuracy"]))

        # bench meta
        for key in ("bench_pope_baseline", "bench_amber_baseline"):
            if cfg[key] and (results_dir / cfg[key]).exists():
                row.update(_bench_meta_stats(results_dir / cfg[key]))

        # codebook probe
        if cfg["codebook_probe"] and (results_dir / cfg["codebook_probe"]).exists():
            row.update(_codebook_probe_stats(results_dir / cfg["codebook_probe"]))

        rows.append(row)
    return rows


def _corr(xs: list[float], ys: list[float]) -> float:
    """Pearson r, NaN if not enough data."""
    mask = [i for i, (x, y) in enumerate(zip(xs, ys))
            if not (np.isnan(x) or np.isnan(y))]
    if len(mask) < 3:
        return float("nan")
    a = np.array([xs[i] for i in mask])
    b = np.array([ys[i] for i in mask])
    return float(np.corrcoef(a, b)[0, 1])


def compute_correlations(table: list[dict]) -> dict[str, float]:
    def col(key: str) -> list[float]:
        return [float(r.get(key, float("nan"))) for r in table]

    pairs = [
        ("drift_severity", "yes_rate_pope",    "drift vs POPE yes-rate"),
        ("drift_severity", "yes_rate_amber",   "drift vs AMBER yes-rate"),
        ("drift_severity", "yes_rate_all",     "drift vs probe-cache yes-rate"),
        ("warn_rate_pope", "yes_rate_all",     "POPE WARN-rate vs probe yes-rate"),
        ("early_visual_score_pope", "yes_rate_all", "early-visual vs probe yes-rate"),
        ("visual_gt1_rate_pope", "yes_rate_all", "visual>1 rate vs probe yes-rate"),
    ]
    result = {}
    for x_key, y_key, label in pairs:
        r = _corr(col(x_key), col(y_key))
        result[label] = r
    return result


# ── figures ────────────────────────────────────────────────────────────────────

_CLASS_COLOR = {"vq": "#d62728", "continuous": "#1f77b4"}


def _scatter(ax: plt.Axes, xs: list[float], ys: list[float],
             classes: list[str], labels: list[str],
             xlabel: str, ylabel: str, title: str) -> None:
    for x, y, cls, lbl in zip(xs, ys, classes, labels):
        if np.isnan(x) or np.isnan(y):
            continue
        ax.scatter(x, y, color=_CLASS_COLOR[cls], s=60, zorder=3)
        ax.annotate(lbl, (x, y), fontsize=7, textcoords="offset points",
                    xytext=(4, 2))
    # trend line
    valid = [(x, y) for x, y, c in zip(xs, ys, classes)
             if not (np.isnan(x) or np.isnan(y))]
    if len(valid) >= 3:
        vx, vy = zip(*valid)
        m, b = np.polyfit(vx, vy, 1)
        xr = np.array([min(vx), max(vx)])
        ax.plot(xr, m * xr + b, "k--", lw=0.8, alpha=0.5)
        r = float(np.corrcoef(vx, vy)[0, 1])
        ax.set_title(f"{title}  r={r:.2f}", fontsize=9)
    else:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def make_figures(table: list[dict], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    def col(key: str) -> list[float]:
        return [float(r.get(key, float("nan"))) for r in table]

    models = [r["model"] for r in table]
    classes = [r["model_class"] for r in table]

    # Figure 1: drift severity vs yes-rate variants
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pairs = [
        ("drift_severity", "yes_rate_all",  "Probe-cache yes-rate"),
        ("drift_severity", "yes_rate_pope", "POPE full yes-rate"),
        ("drift_severity", "yes_rate_amber","AMBER yes-rate"),
    ]
    for ax, (xk, yk, ylabel) in zip(axes, pairs):
        _scatter(ax, col(xk), col(yk), classes, models,
                 "Drift severity (cos_other − cos_self)", ylabel,
                 f"Drift vs {ylabel}")

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=_CLASS_COLOR[c], markersize=8, label=c)
               for c in ("vq", "continuous")]
    axes[0].legend(handles=handles, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "drift_vs_yes_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: class-split bar chart of POPE WARN rate and early visual score
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (key, ylabel) in zip(axes, [("warn_rate_pope", "POPE WARN rate"),
                                         ("early_visual_score_pope", "Early visual score [0,4)")]):
        vq_vals = [(r["model"], float(r.get(key, float("nan"))))
                   for r in table if r["model_class"] == "vq"]
        ct_vals = [(r["model"], float(r.get(key, float("nan"))))
                   for r in table if r["model_class"] == "continuous"]
        for color, vals in [(_CLASS_COLOR["vq"], vq_vals),
                             (_CLASS_COLOR["continuous"], ct_vals)]:
            names = [v[0] for v in vals if not np.isnan(v[1])]
            nums  = [v[1] for v in vals if not np.isnan(v[1])]
            bars = ax.bar(names, nums, color=color, alpha=0.7)
            for bar, num in zip(bars, nums):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{num:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "by_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: gate status summary
    gate_order = {"pass": 2, "partial": 1, "fail": 0, None: -1}
    gate_a1 = [gate_order[r.get("gate_a1")] for r in table]
    gate_a2 = [gate_order[r.get("gate_a2")] for r in table]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(models))
    w = 0.35
    bars1 = ax.bar(x - w / 2, gate_a1, w, label="Gate A.1 (rel_div)", alpha=0.8)
    bars2 = ax.bar(x + w / 2, gate_a2, w, label="Gate A.2 (head-weight)", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, fontsize=8)
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_yticklabels(["N/A", "fail", "partial", "pass"], fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("N-055 diagnostic gate status per model", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "gate_status.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="results",
                    help="Directory containing all result files")
    ap.add_argument("--out", default="results/yes_bias_correlation.json",
                    help="Output JSON summary")
    ap.add_argument("--fig_dir", default="figures/yes_bias",
                    help="Output figure directory")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)

    print("Building cohort table...")
    table = build_cohort_table(results_dir)

    print("\n=== Per-model metrics ===")
    keys_to_print = [
        "model", "model_class",
        "warn_rate_pope", "warn_rate_naturalbench",
        "yes_rate_all", "yes_rate_pope", "yes_rate_amber",
        "early_visual_score_pope", "visual_gt1_rate_pope",
        "drift_severity", "drift_closer_to_other_rate",
        "gate_a1", "gate_a2",
    ]
    header = "  ".join(f"{k:<30}" for k in keys_to_print)
    print(header)
    for row in table:
        vals = []
        for k in keys_to_print:
            v = row.get(k)
            if isinstance(v, float):
                vals.append(f"{v:<30.3f}")
            else:
                vals.append(f"{str(v):<30}")
        print("  ".join(vals))

    print("\n=== Cross-model correlations ===")
    corrs = compute_correlations(table)
    for label, r in corrs.items():
        print(f"  {label:<45} r = {r:+.3f}" if not np.isnan(r)
              else f"  {label:<45} r = N/A")

    print(f"\nGenerating figures in {fig_dir} ...")
    make_figures(table, fig_dir)

    out = {
        "models": table,
        "correlations": {k: (v if not np.isnan(v) else None)
                         for k, v in corrs.items()},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
