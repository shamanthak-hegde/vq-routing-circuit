"""Paired-record bootstrap confidence intervals for intervention Δ-metrics.

For deterministic prefill benchmarks (L0 ablation, L1 ablation, VTI), the only
source of variance is which records happened to be in the benchmark set. The
correct CI uses a *paired* bootstrap: for each resample, draw N records with
replacement, compute Δ_i = metric_int_i − metric_base_i per record, then take
the mean Δ. This is tighter and correct; independent bootstrap of baseline and
intervention overstates uncertainty because it ignores per-record correlation.

Usage (standalone):
    source activate sae
    python -m probe.analysis.paired_bootstrap \\
        --baseline results/bench_pope_full_vilau_baseline.jsonl \\
        --intervention results/bench_pope_full_vilau_L0.jsonl \\
        --metric acc --n_boot 10000 --out results/bootstrap_vilau_pope_L0.json

    # Batch over all headline pairs
    python -m probe.analysis.paired_bootstrap --batch

Output JSON (single pair):
    {
      "n_records": 3000,
      "mean_delta": 0.042,
      "ci_lo": 0.037,
      "ci_hi": 0.047,
      "metric": "acc",
      "n_boot": 10000,
      "alpha": 0.05
    }
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np


# ── Core ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _record_metric(rec: dict, metric: str) -> float:
    if metric == "acc":
        return float(rec["correct"])
    if metric == "yes_rate":
        return float(rec.get("pred", "") == "yes")
    if metric == "logit_gap":
        return float(rec["logit_yes"]) - float(rec["logit_no"])
    raise ValueError(f"Unknown metric {metric!r}. Choose: acc, yes_rate, logit_gap")


def paired_bootstrap(
    baseline_path: str | Path,
    intervention_path: str | Path,
    metric: str = "acc",
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Compute paired-record bootstrap CI for mean Δ(metric).

    Parameters
    ----------
    baseline_path, intervention_path : paths to benchmark JSONL files
    metric : 'acc' | 'yes_rate' | 'logit_gap'
    n_boot : number of bootstrap resamples
    alpha  : CI width (0.05 → 95%)
    seed   : numpy random seed

    Returns
    -------
    dict with keys: n_records, mean_delta, ci_lo, ci_hi, metric, n_boot, alpha,
                    n_matched, n_unmatched_base, n_unmatched_int
    """
    base_recs = load_jsonl(baseline_path)
    int_recs = load_jsonl(intervention_path)

    base_by_id = {r["record_id"]: r for r in base_recs}
    int_by_id = {r["record_id"]: r for r in int_recs}

    matched_ids = sorted(set(base_by_id) & set(int_by_id))
    n_unmatched_base = len(base_by_id) - len(matched_ids)
    n_unmatched_int = len(int_by_id) - len(matched_ids)

    if not matched_ids:
        raise ValueError("No matching record_ids between baseline and intervention files")

    if n_unmatched_base or n_unmatched_int:
        print(f"[warn] {n_unmatched_base} baseline-only, {n_unmatched_int} intervention-only records dropped")

    deltas = np.array([
        _record_metric(int_by_id[rid], metric) - _record_metric(base_by_id[rid], metric)
        for rid in matched_ids
    ], dtype=np.float64)

    rng = np.random.default_rng(seed)
    n = len(deltas)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = deltas[idx].mean()

    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "n_records": n,
        "n_matched": len(matched_ids),
        "n_unmatched_base": n_unmatched_base,
        "n_unmatched_int": n_unmatched_int,
        "mean_delta": float(deltas.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "metric": metric,
        "n_boot": n_boot,
        "alpha": alpha,
    }


# ── Batch mode ────────────────────────────────────────────────────────────────

_HEADLINE_PAIRS = [
    # (label, baseline_jsonl, intervention_jsonl, benchmark)
    # VILA-U
    ("vilau_pope_L0",   "bench_pope_full_vilau_baseline.jsonl",   "bench_pope_full_vilau_L0.jsonl",    "pope"),
    ("vilau_pope_L1",   "bench_pope_full_vilau_baseline.jsonl",   "bench_pope_full_vilau_L1.jsonl",    "pope"),
    ("vilau_amber_L1",  "bench_amber_vilau_baseline.jsonl",       "bench_amber_vilau_L1.jsonl",        "amber"),
    ("vilau_amber_vti", "bench_amber_vilau_baseline.jsonl",       "bench_amber_vilau_vti_a0.005.jsonl","amber"),
    # Liquid
    ("liquid_pope_L0",  "bench_pope_full_liquid_baseline.jsonl",  "bench_pope_full_liquid_L0.jsonl",   "pope"),
    ("liquid_amber_L0", "bench_amber_liquid_baseline.jsonl",      "bench_amber_liquid_L0.jsonl",       "amber"),
    ("liquid_pope_L1",  "bench_pope_full_liquid_baseline.jsonl",  "bench_pope_full_liquid_L1.jsonl",   "pope"),
    ("liquid_amber_L1", "bench_amber_liquid_baseline.jsonl",      "bench_amber_liquid_L1.jsonl",       "amber"),
    # Chameleon
    ("chameleon_pope_L0", "bench_pope_full_chameleon_baseline.jsonl", "bench_pope_full_chameleon_L0.jsonl", "pope"),
    ("chameleon_amber_L0","bench_amber_chameleon_baseline.jsonl",    "bench_amber_chameleon_L0.jsonl",     "amber"),
    ("chameleon_pope_L1", "bench_pope_full_chameleon_baseline.jsonl","bench_pope_full_chameleon_L1.jsonl",  "pope"),
    ("chameleon_amber_L1","bench_amber_chameleon_baseline.jsonl",    "bench_amber_chameleon_L1.jsonl",     "amber"),
]


def _run_batch(results_dir: str, n_boot: int, out_tsv: str) -> None:
    rd = Path(results_dir)
    rows = []
    header = ["label", "benchmark", "metric", "n_records", "mean_delta", "ci_lo", "ci_hi",
              "ci_width", "n_boot", "base_file", "int_file"]

    for label, base_fn, int_fn, bench in _HEADLINE_PAIRS:
        base_path = rd / base_fn
        int_path  = rd / int_fn
        if not base_path.exists():
            print(f"[skip] {label}: baseline not found at {base_path}")
            continue
        if not int_path.exists():
            print(f"[skip] {label}: intervention not found at {int_path}")
            continue

        for metric in ("acc", "yes_rate"):
            try:
                r = paired_bootstrap(base_path, int_path, metric=metric, n_boot=n_boot)
                rows.append({
                    "label": label,
                    "benchmark": bench,
                    "metric": metric,
                    "n_records": r["n_records"],
                    "mean_delta": f"{r['mean_delta']:+.4f}",
                    "ci_lo": f"{r['ci_lo']:+.4f}",
                    "ci_hi": f"{r['ci_hi']:+.4f}",
                    "ci_width": f"{r['ci_hi'] - r['ci_lo']:.4f}",
                    "n_boot": r["n_boot"],
                    "base_file": base_fn,
                    "int_file": int_fn,
                })
                print(f"  {label} [{metric}]: Δ={r['mean_delta']:+.4f}  "
                      f"95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
            except Exception as e:
                print(f"  [error] {label} [{metric}]: {e}")

    # Write TSV
    with open(out_tsv, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(row[k]) for k in header) + "\n")
    print(f"\nResults written → {out_tsv}  ({len(rows)} rows)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired-record bootstrap CIs for intervention Δ-metrics"
    )
    parser.add_argument("--baseline", default=None,
                        help="Path to baseline JSONL (single-pair mode)")
    parser.add_argument("--intervention", default=None,
                        help="Path to intervention JSONL (single-pair mode)")
    parser.add_argument("--metric", default="acc",
                        choices=["acc", "yes_rate", "logit_gap"])
    parser.add_argument("--n_boot", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None,
                        help="Output JSON path (single-pair mode)")
    parser.add_argument("--batch", action="store_true",
                        help="Run over all headline pairs in _HEADLINE_PAIRS")
    parser.add_argument("--results_dir", default="results",
                        help="[batch] directory containing JSONL files")
    parser.add_argument("--out_tsv", default="results/paired_bootstrap_summary.tsv",
                        help="[batch] output TSV path")
    args = parser.parse_args()

    if args.batch:
        _run_batch(args.results_dir, args.n_boot, args.out_tsv)
        return

    if not args.baseline or not args.intervention:
        parser.error("--baseline and --intervention are required in single-pair mode (or use --batch)")

    result = paired_bootstrap(
        args.baseline, args.intervention,
        metric=args.metric, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
    )
    print(f"n_records : {result['n_records']}")
    print(f"mean Δ    : {result['mean_delta']:+.4f}")
    print(f"95% CI    : [{result['ci_lo']:+.4f}, {result['ci_hi']:+.4f}]")
    print(f"CI width  : {result['ci_hi'] - result['ci_lo']:.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"Written → {args.out}")


if __name__ == "__main__":
    _main()
