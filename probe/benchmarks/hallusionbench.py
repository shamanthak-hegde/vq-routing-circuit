"""HallusionBench loader and aAcc/qAcc/fAcc scorer.

HallusionBench tests visual-language hallucination with figure-question pairs.
Each figure has multiple yes/no questions. Metrics:
  aAcc (all-question accuracy) — fraction of individual QA pairs correct
  qAcc (per-question accuracy) — per unique question, fraction correct across figures
  fAcc (per-figure accuracy)   — per figure, fraction of its questions correct

Some images are already on disk at micro_benchmark/images/hallusionbench/.
Full dataset pulled from lmms-lab/HallusionBench.

Usage (smoke test)
------------------
    python -m probe.benchmarks.hallusionbench --smoke_test
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


HB_IMG_DIR = Path(__file__).parent.parent.parent / "micro_benchmark" / "images" / "hallusionbench"


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
    figure_id: str = ""
    question_id: str = ""


_ANSWER_MAP = {"0": "no", "1": "yes"}


def load_hallusionbench(
    img_dir: Optional[Path] = None,
    max_records: Optional[int] = None,
) -> list[BenchRecord]:
    """Load the full HallusionBench dataset (image split, 951 records).

    Downloads images to img_dir if not already present.
    gt_answer is binary 0/1; we map to no/yes.
    figure_id in the dataset is not globally unique — use the sanitized
    filename (e.g. VS_chart_0_1) as the per-figure grouping key.
    """
    from datasets import load_dataset

    if img_dir is None:
        img_dir = HB_IMG_DIR
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HallusionBench …")
    ds = load_dataset("lmms-lab/HallusionBench", split="image")

    records: list[BenchRecord] = []
    n_saved = 0

    for ex in ds:
        ans_binary = str(ex.get("gt_answer", "")).strip()
        ans_raw = _ANSWER_MAP.get(ans_binary)
        if ans_raw is None:
            continue

        # Sanitize filename to a filesystem-safe unique key (e.g. VS_chart_0_1)
        fname = ex.get("filename", f"fig_{len(records)}")
        fig_key = fname.lstrip("./").replace("/", "_").rsplit(".", 1)[0]

        q_id = str(ex.get("question_id", f"q_{len(records)}"))
        rec_id = f"hb_{fig_key}_{q_id}"

        img_path = img_dir / f"{fig_key}.png"
        if not img_path.exists():
            img = ex.get("image")
            if img is None:
                print(f"  Skipping {rec_id}: no inline image in dataset row")
                continue
            img.save(str(img_path))
            n_saved += 1

        if not img_path.exists():
            continue

        question = str(ex.get("question", ""))

        records.append(BenchRecord(
            id=rec_id,
            source="hallusionbench",
            pair_id=fig_key,
            figure_id=fig_key,
            question_id=q_id,
            image_path=str(img_path.resolve()),
            question=question,
            answer=ans_raw,
            metadata={"figure_id": fig_key, "question_id": q_id},
        ))

        if max_records is not None and len(records) >= max_records:
            break

    if n_saved:
        print(f"  Saved {n_saved} new images to {img_dir}")
    print(f"  Loaded {len(records)} HallusionBench records")
    return records


def score_hallusionbench(predictions: list[dict]) -> dict[str, float]:
    """Compute aAcc, qAcc, fAcc.

    predictions: list of dicts with gt_answer, pred, and optionally
    figure_id and question_id (from metadata).
    """
    n_correct = n_total = 0
    # per-figure: list of (correct, n_questions_for_figure)
    fig_results: dict[str, list[bool]] = defaultdict(list)
    # per-question text: list of bools
    q_results: dict[str, list[bool]] = defaultdict(list)

    for row in predictions:
        gt = row["gt_answer"].lower().strip()
        pred = row["pred"].lower().strip()
        correct = pred == gt
        n_correct += int(correct)
        n_total += 1

        fid = row.get("figure_id") or row.get("metadata", {}).get("figure_id", "")
        qid = row.get("question_id") or row.get("metadata", {}).get("question_id", "")
        q_text = row.get("question", qid)

        if fid:
            fig_results[fid].append(correct)
        if q_text:
            q_results[q_text].append(correct)

    a_acc = n_correct / n_total if n_total else 0.0
    # qAcc: mean per-question accuracy (fraction correct across all figures that have that Q)
    q_acc_vals = [sum(v) / len(v) for v in q_results.values() if v]
    q_acc = sum(q_acc_vals) / len(q_acc_vals) if q_acc_vals else float("nan")
    # fAcc: mean per-figure accuracy
    f_acc_vals = [sum(v) / len(v) for v in fig_results.values() if v]
    f_acc = sum(f_acc_vals) / len(f_acc_vals) if f_acc_vals else float("nan")

    return {
        "a_acc": round(a_acc, 4),
        "q_acc": round(q_acc, 4) if q_acc_vals else float("nan"),
        "f_acc": round(f_acc, 4) if f_acc_vals else float("nan"),
        "n": n_total,
        "n_figures": len(fig_results),
        "n_questions": len(q_results),
    }


def _smoke_test() -> None:
    records = load_hallusionbench(max_records=8)
    assert len(records) >= 1, "Expected at least 1 record"
    for r in records:
        assert r.answer in ("yes", "no"), f"Bad answer: {r.answer}"
        assert Path(r.image_path).exists(), f"Missing image: {r.image_path}"

    fake_preds = [
        {
            "gt_answer": r.answer,
            "pred": r.answer,
            "figure_id": r.figure_id,
            "question_id": r.question_id,
            "question": r.question,
        }
        for r in records
    ]
    metrics = score_hallusionbench(fake_preds)
    assert metrics["a_acc"] == 1.0, f"Perfect preds got a_acc={metrics['a_acc']}"
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
        recs = load_hallusionbench(max_records=args.max_records)
        print(f"Loaded {len(recs)} records")
