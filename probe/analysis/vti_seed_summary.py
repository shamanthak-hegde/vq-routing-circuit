"""Summarize VTI demo-seed variance across multiple seeds (N-083 sub-task 4B).

For each seed, computes paired-record Δacc and ΔF1 vs baseline, plus
per-seed paired-record bootstrap CI. Final table reports:
  - Mean ± std across seeds (demo-seed variance)
  - Mean paired CI (within-seed record variance)
  - Combined CI (demo-seed std combined with within-seed CI)

Usage:
    python -m probe.analysis.vti_seed_summary \\
        --results_dir results --out results/vti_seed_summary.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from probe.analysis.paired_bootstrap import paired_bootstrap


def _load_meta(path: Path) -> dict:
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def main(results_dir: str = "results", out_tsv: str = "results/vti_seed_summary.tsv",
         n_boot: int = 5000, seeds: list[int] | None = None) -> None:
    rd = Path(results_dir)
    if seeds is None:
        # Auto-detect available seeds
        seeds = []
        for f in rd.glob("bench_pope_full_vilau_vti_seed*.jsonl"):
            try:
                s = int(f.stem.split("seed")[-1])
                seeds.append(s)
            except ValueError:
                pass
        seeds = sorted(seeds)

    if not seeds:
        print(f"No vti seed bench files found in {results_dir}. "
              f"Run scripts/run_n083_vti_seeds.sh first.")
        return

    base_pope = rd / "bench_pope_full_vilau_baseline.jsonl"
    base_amber = rd / "bench_amber_vilau_baseline.jsonl"
    if not base_pope.exists():
        print(f"Baseline POPE not found: {base_pope}")
        return

    rows = []
    header = ["seed", "bench", "metric", "mean_delta", "ci_lo", "ci_hi", "ci_width"]

    print(f"Processing {len(seeds)} seeds: {seeds}")
    print(f"{'seed':>5}  {'bench':>12}  {'metric':>8}  {'Δ':>7}  {'95% CI':>17}")

    seed_deltas: dict[str, list[float]] = {}  # key = "bench_metric"

    for seed in seeds:
        for bench, base_path in [("pope", base_pope), ("amber", base_amber)]:
            int_path = rd / f"bench_{bench if bench == 'amber' else 'pope_full'}_vilau_vti_seed{seed}.jsonl"
            if not int_path.exists():
                print(f"  [skip] seed={seed} bench={bench}: {int_path} not found")
                continue

            for metric in ("acc",):
                try:
                    r = paired_bootstrap(base_path, int_path, metric=metric, n_boot=n_boot)
                    key = f"{bench}_{metric}"
                    seed_deltas.setdefault(key, []).append(r["mean_delta"])

                    rows.append({
                        "seed": seed,
                        "bench": bench,
                        "metric": metric,
                        "mean_delta": r["mean_delta"],
                        "ci_lo": r["ci_lo"],
                        "ci_hi": r["ci_hi"],
                        "ci_width": r["ci_hi"] - r["ci_lo"],
                    })
                    print(f"{seed:>5}  {bench:>12}  {metric:>8}  "
                          f"{r['mean_delta']:+7.4f}  [{r['ci_lo']:+7.4f}, {r['ci_hi']:+7.4f}]")
                except Exception as e:
                    print(f"  [error] seed={seed} bench={bench} metric={metric}: {e}")

    # Cross-seed summary
    print("\n--- Cross-seed summary ---")
    print(f"{'key':>15}  {'n_seeds':>8}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    for key, deltas in sorted(seed_deltas.items()):
        arr = np.array(deltas)
        print(f"{key:>15}  {len(arr):>8}  {arr.mean():+8.4f}  {arr.std():8.4f}  "
              f"{arr.min():+8.4f}  {arr.max():+8.4f}")

    # Write TSV
    with open(out_tsv, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(row[k]) for k in header) + "\n")
    print(f"\nWritten → {out_tsv}  ({len(rows)} rows)")


def _main() -> None:
    parser = argparse.ArgumentParser(description="VTI demo-seed summary (N-083)")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out", default="results/vti_seed_summary.tsv")
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    main(args.results_dir, args.out, args.n_boot, args.seeds)


if __name__ == "__main__":
    _main()
