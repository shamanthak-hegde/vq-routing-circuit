"""Per-model best visual restoration window from sweep JSONL (N-054).

Given a sweep_<backend>.jsonl produced by run_sweep.py, find the stride-4 window
[layer_start, layer_end) with the highest mean visual restoration score (NB + POPE
pooled), matching the aggregation used by run_sweep._print_heatmap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def best_visual_window(sweep_jsonl: Path) -> tuple[int, int, float]:
    """Return (layer_start, layer_end, mean_score) for the argmax visual window.

    Reads non-WARN rows (logit_clean > logit_corrupt) with token_group == 'visual',
    averages score by (layer_start, layer_end) across all records (NB + POPE pooled),
    and returns the window with the highest mean. Matching the filter used by
    run_sweep._print_heatmap so the chosen window is consistent with the heatmap output.
    """
    sums: dict[tuple[int, int], float] = {}
    counts: dict[tuple[int, int], int] = {}
    with open(sweep_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("token_group") != "visual":
                continue
            # exclude WARN records where corruption didn't hurt (sign is inverted)
            if float(row.get("logit_clean", 1)) <= float(row.get("logit_corrupt", 0)):
                continue
            key = (int(row["layer_start"]), int(row["layer_end"]))
            s = float(row["score"])
            sums[key] = sums.get(key, 0.0) + s
            counts[key] = counts.get(key, 0) + 1
    if not sums:
        raise ValueError(f"No non-WARN visual rows found in {sweep_jsonl}")
    best_key = max(sums, key=lambda k: sums[k] / counts[k])
    mean_score = sums[best_key] / counts[best_key]
    return best_key[0], best_key[1], mean_score


_BACKENDS = ["vilau", "llava", "unitok", "qwen3vl", "showo", "seed"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print best visual window from sweep JSONL (N-054)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--backend", choices=_BACKENDS)
    group.add_argument("--all", action="store_true", dest="all_backends")
    parser.add_argument(
        "--sweep_dir", default="results",
        help="Directory containing sweep_<backend>.jsonl files (default: results/)",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)

    if args.all_backends:
        print(f"{'backend':<12}  {'window':>9}  {'mean_visual_score':>18}")
        print("-" * 43)
        for backend in _BACKENDS:
            p = sweep_dir / f"sweep_{backend}.jsonl"
            if not p.exists():
                print(f"{backend:<12}  {'(no sweep file)':>28}")
                continue
            try:
                l_start, l_end, score = best_visual_window(p)
                print(f"{backend:<12}  [{l_start:>2},{l_end:>3})  {score:>18.4f}")
            except ValueError as e:
                print(f"{backend:<12}  ERROR: {e}")
    else:
        p = sweep_dir / f"sweep_{args.backend}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Sweep file not found: {p}")
        l_start, l_end, score = best_visual_window(p)
        print(
            f"backend={args.backend}  best_visual_window=[{l_start},{l_end})"
            f"  mean_score={score:.4f}"
        )


if __name__ == "__main__":
    main()
