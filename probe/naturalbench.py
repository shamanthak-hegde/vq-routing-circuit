"""
NaturalBench loader for the probe module.

Paired structure
----------------
Each NaturalBench row contains two images (Image_0, Image_1) and two yes/no
questions (Question_0, Question_1).  For a given question Q, exactly one image
answers "yes" and the other answers "no".

We emit **two ProbeRecords per row** (one per question), always treating
Image_0 as the anchor and Image_1 as the foil:

    id = nb_{index}_q{0|1}
    image_path       → Image_0
    foil_image_path  → Image_1
    question         → Question_{0|1}
    answer           → Image_0_Question_{0|1}  (which may be "yes" or "no")
    corruption_mode  → "image_swap"

This gives 150 rows × 2 questions = 300 records with strictly balanced yes/no
(the NaturalBench design guarantees one yes and one no per question across
the image pair, so the per-question balance depends on which image is anchor).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from PIL import Image

from .schema import ProbeRecord

SEED = 42
TARGET_ROWS = 150          # → 300 ProbeRecords (2 per row)
IMG_DIR = Path(__file__).parent.parent / "micro_benchmark" / "images" / "naturalbench"


def _save_jpeg(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, format="JPEG", quality=85)


def build_naturalbench_records(
    target_rows: int = TARGET_ROWS,
    seed: int = SEED,
    img_dir: Optional[Path] = None,
) -> list[ProbeRecord]:
    """Download NaturalBench, sample *target_rows* yes_no rows, return ProbeRecords.

    Images are saved to *img_dir* (defaults to micro_benchmark/images/naturalbench/).
    Subsequent calls reuse cached images on disk.
    """
    from datasets import load_dataset

    if img_dir is None:
        img_dir = IMG_DIR

    rng = random.Random(seed)

    print("Loading NaturalBench …")
    ds = load_dataset("BaiqiL/NaturalBench", split="train")

    # ── collect index metadata first (no image tensors in memory) ────────────
    yes_no_rows = [
        (i, ex["Index"], ex["Source"])
        for i, ex in enumerate(ds)
        if ex["Question_Type"] == "yes_no"
    ]
    docci  = [(i, idx, src) for i, idx, src in yes_no_rows if src == "DOCCI"]
    flickr = [(i, idx, src) for i, idx, src in yes_no_rows if src == "Flickr"]

    n_each  = target_rows // 2
    chosen  = (
        rng.sample(docci,  min(n_each, len(docci)))  +
        rng.sample(flickr, min(target_rows - n_each, len(flickr)))
    )
    chosen.sort(key=lambda x: x[0])          # sequential read is faster
    chosen_set = {pos for pos, _, _ in chosen}

    # ── single pass through dataset ───────────────────────────────────────────
    records: list[ProbeRecord] = []
    n_saved = 0

    for i, ex in enumerate(ds):
        if i not in chosen_set:
            continue

        idx = ex["Index"]
        img0_path = img_dir / f"{idx}_img0.jpg"
        img1_path = img_dir / f"{idx}_img1.jpg"

        if not img0_path.exists():
            _save_jpeg(ex["Image_0"], img0_path)
            n_saved += 1
        if not img1_path.exists():
            _save_jpeg(ex["Image_1"], img1_path)
            n_saved += 1

        for q_idx, (q_col, a0_col, a1_col) in enumerate([
            ("Question_0", "Image_0_Question_0", "Image_1_Question_0"),
            ("Question_1", "Image_0_Question_1", "Image_1_Question_1"),
        ]):
            records.append(ProbeRecord(
                id=f"nb_{idx}_q{q_idx}",
                source="naturalbench",
                pair_id=f"nb_{idx}",
                image_path=str(img0_path.resolve()),
                foil_image_path=str(img1_path.resolve()),
                question=ex[q_col],
                answer=ex[a0_col].strip().lower(),
                answer_token_id=None,
                corruption_mode="image_swap",
                metadata={
                    "nb_index": idx,
                    "nb_question_idx": q_idx,
                    "source_dataset": ex["Source"],
                    # foil answer (Image_1's answer to the same question)
                    "foil_answer": ex[a1_col].strip().lower(),
                },
            ))

    print(
        f"  NaturalBench: {len(records)} records from {len(chosen)} rows "
        f"({n_saved} new images saved)"
    )
    return records
