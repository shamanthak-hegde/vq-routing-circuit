"""AMBER discriminative benchmark loader and F1 scorer (N-041D).

AMBER (A Multi-dimensional Benchmark for Hallucination Evaluation) has two
tracks: discriminative (yes/no binary QA) and generative. This module covers
the discriminative track only, which is what this paper needs.

Reads from the local copy at AMBER_LOCAL (/scratch/shegde23/data/AMBER).
Layout:
  data/query/query_discriminative.json  — [{id, image, query}, ...]
  data/annotations.json                 — [{id, type, truth}, ...]
  data/image/AMBER_<n>.jpg              — 1,004 images shared across 14,216 queries

Usage (smoke test)
------------------
    python -m probe.benchmarks.amber --smoke_test
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


AMBER_LOCAL = Path("/scratch/shegde23/data/AMBER")


@dataclass
class BenchRecord:
    id: str
    source: str
    image_path: str
    question: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
    answer_token_id: Optional[int] = None
    pair_id: str = ""


def load_amber(
    amber_local: Optional[Path] = None,
    img_dir: Optional[Path] = None,
    max_records: Optional[int] = None,
) -> list[BenchRecord]:
    """Load AMBER discriminative track from local dataset copy.

    Reads query_discriminative.json and annotations.json from AMBER_LOCAL.
    Images are read directly from AMBER_LOCAL/data/image/ — no copying needed.
    """
    if amber_local is None:
        amber_local = AMBER_LOCAL

    query_path = amber_local / "data" / "query" / "query_discriminative.json"
    annot_path = amber_local / "data" / "annotations.json"
    img_dir_local = amber_local / "data" / "image"

    if not query_path.exists():
        raise FileNotFoundError(f"AMBER queries not found at {query_path}")
    if not annot_path.exists():
        raise FileNotFoundError(f"AMBER annotations not found at {annot_path}")

    print("Loading AMBER (discriminative track) …")

    with open(query_path) as f:
        queries: list[dict] = json.load(f)
    with open(annot_path) as f:
        annotations: list[dict] = json.load(f)

    truth_map: dict[int, str] = {}
    type_map: dict[int, str] = {}
    for ann in annotations:
        if isinstance(ann.get("truth"), str):
            truth_map[ann["id"]] = ann["truth"].lower().strip()
            type_map[ann["id"]] = ann.get("type", "")

    records: list[BenchRecord] = []
    for q in queries:
        rec_id = q["id"]
        truth = truth_map.get(rec_id, "")
        if truth not in ("yes", "no"):
            continue

        img_path = img_dir_local / q["image"]
        if not img_path.exists():
            print(f"  Skipping amber_{rec_id}: image not found at {img_path}")
            continue

        records.append(BenchRecord(
            id=f"amber_{rec_id}",
            source="amber",
            pair_id=f"amber_{rec_id}",
            image_path=str(img_path.resolve()),
            question=q["query"],
            answer=truth,
            metadata={"task": type_map.get(rec_id, ""), "amber_id": str(rec_id)},
        ))

        if max_records is not None and len(records) >= max_records:
            break

    print(f"  Loaded {len(records)} AMBER discriminative records")
    return records


def score_amber(predictions: list[dict]) -> dict[str, float]:
    """Compute AMBER discriminative F1, precision, recall, accuracy.

    predictions: list of dicts with gt_answer and pred.
    """
    tp = fp = tn = fn = 0
    n_total = 0

    for row in predictions:
        gt = row["gt_answer"].lower().strip()
        pred = row["pred"].lower().strip()
        n_total += 1
        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "no" and pred == "no":
            tn += 1
        elif gt == "yes" and pred == "no":
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n_total if n_total else 0.0
    yes_rate = (tp + fp) / n_total if n_total else 0.0

    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "yes_rate": round(yes_rate, 4),
        "n": n_total,
    }


def _smoke_test() -> None:
    records = load_amber(max_records=8)
    assert len(records) >= 1, "Expected at least 1 record"
    for r in records:
        assert r.answer in ("yes", "no"), f"Bad answer: {r.answer}"
        assert Path(r.image_path).exists(), f"Missing image: {r.image_path}"

    fake_preds = [{"gt_answer": r.answer, "pred": r.answer} for r in records]
    metrics = score_amber(fake_preds)
    assert metrics["accuracy"] == 1.0, f"Perfect preds got acc={metrics['accuracy']}"
    print("smoke test passed:", metrics)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--max_records", type=int, default=None)
    args = parser.parse_args()
    if args.smoke_test:
        _smoke_test()
    else:
        recs = load_amber(max_records=args.max_records)
        print(f"Loaded {len(recs)} records")
