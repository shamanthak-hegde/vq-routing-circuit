"""Deduplication utility for benchmark JSONL files.

Benchmark resume logic can produce duplicate rows when a run is restarted
without clean state. This module provides a canonical dedup helper used by
run_bench.py and as a standalone CLI.

Usage (CLI):
    python -m probe.benchmarks._dedupe results/bench_pope_full_lavit_baseline.jsonl
    python -m probe.benchmarks._dedupe results/bench_pope_full_lavit_baseline.jsonl --score pope_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dedupe_jsonl_by_record_id(path: Path) -> tuple[int, int]:
    """Deduplicate a JSONL file by `record_id`, keeping the first occurrence.

    Rewrites the file in place. Returns (total_before, total_after).
    """
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    seen: set[str] = set()
    deduped: list[dict] = []
    for row in rows:
        rid = row.get("record_id", "")
        if rid not in seen:
            seen.add(rid)
            deduped.append(row)

    total_before = len(rows)
    total_after = len(deduped)

    if total_before != total_after:
        with path.open("w") as f:
            for row in deduped:
                f.write(json.dumps(row) + "\n")

    return total_before, total_after


def rescore_and_write_meta(jsonl_path: Path, bench: str, meta_dict_extra: dict | None = None) -> dict:
    """Score a deduped JSONL file and write the .meta.json sidecar.

    Returns the metrics dict.
    """
    rows: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if bench == "pope_full":
        from probe.benchmarks.pope_full import score_pope
        metrics = score_pope(rows)
    elif bench == "nb_full":
        from probe.benchmarks.naturalbench_full import score_naturalbench
        metrics = score_naturalbench(rows)
    elif bench == "hb":
        from probe.benchmarks.hallusionbench import score_hallusionbench
        metrics = score_hallusionbench(rows)
    elif bench == "amber":
        from probe.benchmarks.amber import score_amber
        metrics = score_amber(rows)
    else:
        raise ValueError(f"Unknown bench {bench!r}")

    meta: dict = {
        "bench": bench,
        "n_records": len(rows),
        "metrics": metrics,
    }
    if meta_dict_extra:
        meta.update(meta_dict_extra)

    meta_path = jsonl_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate a benchmark JSONL file by record_id")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--score", choices=["pope_full", "nb_full", "hb", "amber"],
                        help="If provided, re-score and write .meta.json after dedup")
    args = parser.parse_args()

    before, after = dedupe_jsonl_by_record_id(args.jsonl)
    print(f"{args.jsonl.name}: {before} → {after} rows (removed {before - after} duplicates)")

    if args.score:
        metrics = rescore_and_write_meta(args.jsonl, args.score)
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
        print(f"Meta written to {args.jsonl.with_suffix('.meta.json')}")
