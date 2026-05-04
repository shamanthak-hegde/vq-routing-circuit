"""Full NaturalBench loader and Group-Acc scorer (N-041B).

Loads all NaturalBench yes/no questions from BaiqiL/NaturalBench (~1,900 rows).
Implements Group-Acc (G-Acc): per 4-way group (image1×Q1, image1×Q2, image2×Q1,
image2×Q2), a group counts as correct only if all 4 questions are answered
correctly. G-Acc = fraction of groups with all 4 correct.

Also reports pair accuracy (image-pair correct = both image answers correct)
and standard per-record accuracy.

Usage (smoke test)
------------------
    python -m probe.benchmarks.naturalbench_full --smoke_test
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


NB_IMG_DIR = Path(__file__).parent.parent.parent / "micro_benchmark" / "images" / "naturalbench"


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
    # NB-specific: group_id identifies the 4-way group
    group_id: str = ""


def load_naturalbench_full(
    img_dir: Optional[Path] = None,
    max_records: Optional[int] = None,
) -> list[BenchRecord]:
    """Load the full NaturalBench dataset.

    Each NaturalBench example has two images (image_1, image_2) and two
    questions (question_1, question_2), giving 4 yes/no QA pairs per row.
    """
    from datasets import load_dataset

    if img_dir is None:
        img_dir = NB_IMG_DIR
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Loading NaturalBench …")
    ds = load_dataset("BaiqiL/NaturalBench", split="train")

    records: list[BenchRecord] = []
    n_saved = 0

    for row_idx, ex in enumerate(ds):
        group_id = f"nb_group_{row_idx}"
        # BaiqiL/NaturalBench column names: Image_0/Image_1, Question_0/Question_1,
        # Image_0_Question_0 / Image_1_Question_0 / Image_0_Question_1 / Image_1_Question_1
        q1 = ex["Question_0"]
        q2 = ex["Question_1"]

        for img_key, img_label, q_ans_prefix in [
            ("Image_0", "I0", "Image_0"),
            ("Image_1", "I1", "Image_1"),
        ]:
            img = ex[img_key]
            img_path = img_dir / f"nb_{row_idx}_{img_label}.jpg"
            if not img_path.exists() and img is not None:
                img.save(str(img_path))
                n_saved += 1

            abs_path = str(img_path.resolve())
            pair_id = f"{group_id}_{img_label}"

            for q_label, q, ans_col in [
                ("Q0", q1, f"{q_ans_prefix}_Question_0"),
                ("Q1", q2, f"{q_ans_prefix}_Question_1"),
            ]:
                ans = ex.get(ans_col, "").lower().strip()
                if ans not in ("yes", "no"):
                    continue

                records.append(BenchRecord(
                    id=f"nb_{row_idx}_{img_label}_{q_label}",
                    source="naturalbench",
                    pair_id=pair_id,
                    group_id=group_id,
                    image_path=abs_path,
                    question=q,
                    answer=ans,
                    metadata={
                        "row_idx": row_idx,
                        "img_label": img_label,
                        "q_label": q_label,
                    },
                ))
                if max_records is not None and len(records) >= max_records:
                    break

            if max_records is not None and len(records) >= max_records:
                break

        if max_records is not None and len(records) >= max_records:
            break

    if n_saved:
        print(f"  Saved {n_saved} new images to {img_dir}")
    print(f"  Loaded {len(records)} NaturalBench records")
    return records


def score_naturalbench(predictions: list[dict]) -> dict[str, float]:
    """Compute per-record accuracy, pair accuracy, and Group-Acc (G-Acc).

    predictions: list of dicts with keys gt_answer, pred, and optionally
    group_id (for G-Acc) and pair_id (for pair accuracy).
    """
    n_correct = n_total = 0
    pair_results: dict[str, list[bool]] = defaultdict(list)
    group_results: dict[str, list[bool]] = defaultdict(list)

    for row in predictions:
        gt = row["gt_answer"].lower().strip()
        pred = row["pred"].lower().strip()
        correct = (pred == gt)
        n_correct += int(correct)
        n_total += 1

        pid = row.get("pair_id", "")
        if pid:
            pair_results[pid].append(correct)

        gid = row.get("group_id", "")
        if gid:
            group_results[gid].append(correct)

    accuracy = n_correct / n_total if n_total else 0.0

    # Pair accuracy: both questions for the same image correct
    n_pair_correct = sum(1 for v in pair_results.values() if all(v))
    pair_accuracy = n_pair_correct / len(pair_results) if pair_results else float("nan")

    # G-Acc: all 4 questions in the 4-way group correct
    n_group_correct = sum(1 for v in group_results.values() if all(v))
    g_acc = n_group_correct / len(group_results) if group_results else float("nan")

    return {
        "accuracy": round(accuracy, 4),
        "pair_accuracy": round(pair_accuracy, 4) if pair_results else float("nan"),
        "g_acc": round(g_acc, 4) if group_results else float("nan"),
        "n": n_total,
        "n_pairs": len(pair_results),
        "n_groups": len(group_results),
    }


def _smoke_test() -> None:
    records = load_naturalbench_full(max_records=8)
    assert 1 <= len(records) <= 8, f"Expected ≤8, got {len(records)}"
    for r in records:
        assert r.answer in ("yes", "no"), f"Bad answer: {r.answer}"
        assert Path(r.image_path).exists(), f"Missing image: {r.image_path}"

    fake_preds = [
        {"gt_answer": r.answer, "pred": r.answer,
         "pair_id": r.pair_id, "group_id": r.group_id}
        for r in records
    ]
    metrics = score_naturalbench(fake_preds)
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
        recs = load_naturalbench_full(max_records=args.max_records)
        print(f"Loaded {len(recs)} records")
