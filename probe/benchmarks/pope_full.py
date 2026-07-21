"""Full POPE adversarial split loader and F1 scorer.

Loads the complete lmms-lab/POPE adversarial split (~3,000 records) — NOT the
paired 200-record probe-cache subset. Returns BenchRecord objects compatible
with run_bench.py.

Scorer: F1 + accuracy + yes-rate (same metrics as LLaVA/llava/eval/eval_pope.py).

Usage (smoke test)
------------------
    python -m probe.benchmarks.pope_full --smoke_test
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


POPE_IMG_DIR = Path(__file__).parent.parent.parent / "micro_benchmark" / "images" / "pope"


@dataclass
class BenchRecord:
    id: str
    source: str
    image_path: str
    question: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # filled by run_bench
    answer_token_id: Optional[int] = None
    pair_id: str = ""


def load_pope_full(
    category: str = "adversarial",
    img_dir: Optional[Path] = None,
    max_records: Optional[int] = None,
) -> list[BenchRecord]:
    """Load the full POPE split for a given category.

    Downloads images to img_dir if not already present.
    """
    from datasets import load_dataset

    if img_dir is None:
        img_dir = POPE_IMG_DIR
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading POPE (category={category}) …")
    ds = load_dataset("lmms-lab/POPE", split="test")

    records: list[BenchRecord] = []
    n_saved = 0

    for ex in ds:
        if ex.get("category") != category:
            continue
        img_src = ex["image_source"]
        img_path = img_dir / f"{img_src}.jpg"
        if not img_path.exists():
            img = ex["image"]
            if img is not None:
                img.save(str(img_path))
                n_saved += 1

        records.append(BenchRecord(
            id=f"pope_{ex['question_id']}",
            source="pope",
            pair_id=f"pope_{img_src}",
            image_path=str(img_path.resolve()),
            question=ex["question"],
            answer=ex["answer"].lower().strip(),
            metadata={"category": category, "image_source": img_src},
        ))

        if max_records is not None and len(records) >= max_records:
            break

    if n_saved:
        print(f"  Saved {n_saved} new images to {img_dir}")
    print(f"  Loaded {len(records)} {category} POPE records")
    return records


def score_pope(predictions: list[dict]) -> dict[str, float]:
    """Compute POPE F1, precision, recall, accuracy, yes-rate.

    Each element of predictions should be a dict with keys:
      gt_answer  (str "yes"/"no")
      pred       (str "yes"/"no")
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
    records = load_pope_full(max_records=8)
    assert len(records) == 8, f"Expected 8, got {len(records)}"
    assert all(r.answer in ("yes", "no") for r in records), "Bad answers"
    assert all(Path(r.image_path).exists() for r in records), "Missing image"

    fake_preds = [{"gt_answer": r.answer, "pred": r.answer} for r in records]
    metrics = score_pope(fake_preds)
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
        recs = load_pope_full(max_records=args.max_records)
        print(f"Loaded {len(recs)} records")
