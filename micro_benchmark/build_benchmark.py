"""
Build a 500–1000 example paired micro-benchmark with yes/no answers.

Sources:
  - NaturalBench (paired structure, yes_no questions only)
  - POPE adversarial (object hallucination, yes/no)
  - HallusionBench image split (VD: illusion/math/figure + VS: ocr/map)

Each output record:
  {
    "id":         unique string,
    "source":     "naturalbench" | "pope" | "hallusionbench",
    "pair_id":    groups paired QA sharing the same image context,
    "image_path": path relative to micro_benchmark/ dir,
    "question":   str,
    "answer":     "yes" | "no",
    "metadata":   source-specific dict
  }

Memory strategy: load one dataset at a time, flush each record to JSONL
immediately after saving its image, then del the dataset object.
"""

import gc
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from PIL import Image

SEED = 42
OUT_DIR = Path(__file__).parent
IMG_DIR = OUT_DIR / "images"
OUT_JSONL = OUT_DIR / "micro_benchmark.jsonl"

NB_ROWS = 150          # rows × 4 QA pairs = 600 NaturalBench examples
POPE_N = 150           # 75 yes / 75 no from adversarial category
HB_N = 150             # from targeted subcategories

# HallusionBench subcategories to include (single-image, visually grounded)
HB_KEEP_SUBCATS = {"illusion", "math", "figure", "ocr", "map"}

random.seed(SEED)


def save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, format="JPEG", quality=85)


def build_naturalbench(out: list) -> None:
    print("Loading NaturalBench …")
    ds = load_dataset("BaiqiL/NaturalBench", split="train")

    # Filter to yes_no, collect only the index/metadata (no images yet)
    yes_no_indices = [
        (i, ex["Index"], ex["Source"])
        for i, ex in enumerate(ds)
        if ex["Question_Type"] == "yes_no"
    ]
    docci = [(i, idx, src) for i, idx, src in yes_no_indices if src == "DOCCI"]
    flickr = [(i, idx, src) for i, idx, src in yes_no_indices if src == "Flickr"]
    n_each = NB_ROWS // 2
    chosen_positions = (
        random.sample(docci, min(n_each, len(docci))) +
        random.sample(flickr, min(NB_ROWS - n_each, len(flickr)))
    )
    chosen_positions.sort(key=lambda x: x[0])  # sequential access is faster

    chosen_set = {pos for pos, _, _ in chosen_positions}

    for i, ex in enumerate(ds):
        if i not in chosen_set:
            continue
        idx = ex["Index"]
        pair_id = f"nb_{idx}"

        qa_pairs = [
            ("img0", "q0", ex["Image_0"], ex["Question_0"], ex["Image_0_Question_0"]),
            ("img1", "q0", ex["Image_1"], ex["Question_0"], ex["Image_1_Question_0"]),
            ("img0", "q1", ex["Image_0"], ex["Question_1"], ex["Image_0_Question_1"]),
            ("img1", "q1", ex["Image_1"], ex["Question_1"], ex["Image_1_Question_1"]),
        ]

        for img_key, q_key, img, question, answer in qa_pairs:
            img_path = IMG_DIR / "naturalbench" / f"{idx}_{img_key}.jpg"
            if not img_path.exists():
                save_image(img, img_path)
            out.append({
                "id": f"nb_{idx}_{img_key}_{q_key}",
                "source": "naturalbench",
                "pair_id": pair_id,
                "image_path": str(img_path.relative_to(OUT_DIR)),
                "question": question,
                "answer": answer.strip().lower(),
                "metadata": {
                    "nb_index": idx,
                    "nb_image": img_key,
                    "nb_question": q_key,
                    "source_dataset": ex["Source"],
                },
            })

    del ds
    gc.collect()
    n = len([r for r in out if r["source"] == "naturalbench"])
    print(f"  NaturalBench: {n} QA pairs from {len(chosen_positions)} rows")


def build_pope(out: list) -> None:
    print("Loading POPE …")
    ds = load_dataset("lmms-lab/POPE", split="test")

    # Collect metadata only first to decide which rows to keep
    adv_meta = [
        (i, ex["question_id"], ex["answer"], ex["image_source"])
        for i, ex in enumerate(ds)
        if ex["category"] == "adversarial"
    ]
    yes_pool = [(i, qid, ans, src) for i, qid, ans, src in adv_meta if ans == "yes"]
    no_pool  = [(i, qid, ans, src) for i, qid, ans, src in adv_meta if ans == "no"]
    n_each = POPE_N // 2
    chosen = (
        random.sample(yes_pool, min(n_each, len(yes_pool))) +
        random.sample(no_pool,  min(POPE_N - n_each, len(no_pool)))
    )
    chosen.sort(key=lambda x: x[0])
    chosen_set = {pos for pos, _, _, _ in chosen}

    for i, ex in enumerate(ds):
        if i not in chosen_set:
            continue
        qid = ex["question_id"]
        img_path = IMG_DIR / "pope" / f"{ex['image_source']}.jpg"
        if not img_path.exists():
            save_image(ex["image"], img_path)
        out.append({
            "id": f"pope_{qid}",
            "source": "pope",
            "pair_id": f"pope_{ex['image_source']}",
            "image_path": str(img_path.relative_to(OUT_DIR)),
            "question": ex["question"],
            "answer": ex["answer"].strip().lower(),
            "metadata": {
                "question_id": qid,
                "category": ex["category"],
                "image_source": ex["image_source"],
            },
        })

    del ds
    gc.collect()
    print(f"  POPE: {len([r for r in out if r['source'] == 'pope'])} examples")


def build_hallusionbench(out: list) -> None:
    print("Loading HallusionBench …")
    ds = load_dataset("lmms-lab/HallusionBench", split="image")

    targeted_meta = [
        (i, ex["set_id"], ex["figure_id"], ex["question_id"],
         ex["gt_answer"], ex["subcategory"], ex["filename"])
        for i, ex in enumerate(ds)
        if ex["subcategory"] in HB_KEEP_SUBCATS
    ]
    yes_pool = [t for t in targeted_meta if t[4] == "1"]
    no_pool  = [t for t in targeted_meta if t[4] == "0"]
    n_each = HB_N // 2
    chosen = (
        random.sample(yes_pool, min(n_each, len(yes_pool))) +
        random.sample(no_pool,  min(HB_N - n_each, len(no_pool)))
    )
    chosen.sort(key=lambda x: x[0])
    chosen_set = {pos for pos, *_ in chosen}

    for i, ex in enumerate(ds):
        if i not in chosen_set:
            continue
        set_id = ex["set_id"]
        fig_id = ex["figure_id"]
        q_id   = ex["question_id"]
        img_stem = ex["filename"].replace("/", "_").replace(".", "_")
        img_path = IMG_DIR / "hallusionbench" / f"{img_stem}.jpg"
        if not img_path.exists():
            save_image(ex["image"], img_path)
        answer = "yes" if ex["gt_answer"] == "1" else "no"
        out.append({
            "id": f"hb_{set_id}_{fig_id}_{q_id}",
            "source": "hallusionbench",
            "pair_id": f"hb_{set_id}_{fig_id}",
            "image_path": str(img_path.relative_to(OUT_DIR)),
            "question": ex["question"],
            "answer": answer,
            "metadata": {
                "set_id": set_id,
                "figure_id": fig_id,
                "question_id": q_id,
                "category": ex["category"],
                "subcategory": ex["subcategory"],
                "sample_note": ex["sample_note"],
                "original_answer": ex["gt_answer"],
                "answer_details": ex["gt_answer_details"],
            },
        })

    del ds
    gc.collect()
    print(f"  HallusionBench: {len([r for r in out if r['source'] == 'hallusionbench'])} examples")


def print_stats(records: list) -> None:
    total = len(records)
    by_source = Counter(r["source"] for r in records)
    by_answer = Counter(r["answer"] for r in records)
    n_pairs = len(set(r["pair_id"] for r in records))

    print(f"\n{'='*50}")
    print(f"Total examples : {total}")
    print(f"Unique pair_ids: {n_pairs}")
    print(f"By source      : {dict(by_source)}")
    print(f"Answer balance : {dict(by_answer)}")
    yes_pct = 100 * by_answer.get("yes", 0) / total
    print(f"Yes %          : {yes_pct:.1f}%")
    print(f"{'='*50}\n")


def main():
    records: list = []
    build_naturalbench(records)
    build_pope(records)
    build_hallusionbench(records)

    print_stats(records)

    with open(OUT_JSONL, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} records → {OUT_JSONL}")


if __name__ == "__main__":
    main()
