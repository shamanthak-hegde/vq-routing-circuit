"""
POPE loader for the probe module.

Pairing strategy
----------------
POPE is not natively paired: each example is (image, question, yes|no)
independently.  We construct pairs by finding images that appear with *both*
a yes-answerable and a no-answerable question in the adversarial split, then
sampling one yes-question and one no-question per image:

    pair A:  (img_X, q_yes, "yes")   corruption_mode="gaussian_noise"
    pair B:  (img_X, q_no,  "no")    corruption_mode="gaussian_noise"
    pair_id: pope_{image_source}

Corruption mode — gaussian_noise
---------------------------------
The foil for POPE is *not* a real image.  Instead, zero-mean Gaussian noise
is injected into the visual-token representations (patch embeddings) after the
visual encoder, before any transformer attention layer.  This lets activation-
patching experiments ask: "which heads rely on the visual token content vs.
language priors?"

Because the foil is generated at inference time (at the token level), there is
no foil image file on disk.  foil_image_path is None for all POPE records.

Callers must apply noise at the visual-token level, *not* at pixel level.
A suggested protocol: draw ε ~ N(0, σ²I) with σ tuned so the cosine similarity
between clean and noisy patch embeddings is ≈ 0.5.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from PIL import Image

from .schema import ProbeRecord

SEED = 42
TARGET_PAIRS = 100          # 100 images × 2 questions = 200 ProbeRecords
IMG_DIR = Path(__file__).parent.parent / "micro_benchmark" / "images" / "pope"


def _save_jpeg(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, format="JPEG", quality=85)


def build_pope_records(
    target_pairs: int = TARGET_PAIRS,
    seed: int = SEED,
    img_dir: Optional[Path] = None,
) -> list[ProbeRecord]:
    """Download POPE adversarial, build paired ProbeRecords, return list.

    Selects *target_pairs* images that each have ≥1 yes and ≥1 no question,
    yielding 2 × target_pairs records total.
    """
    from datasets import load_dataset

    if img_dir is None:
        img_dir = IMG_DIR

    rng = random.Random(seed)

    print("Loading POPE …")
    ds = load_dataset("lmms-lab/POPE", split="test")

    # ── group adversarial examples by image ──────────────────────────────────
    by_img: dict[str, dict[str, list]] = defaultdict(lambda: {"yes": [], "no": []})
    for ex in ds:
        if ex["category"] != "adversarial":
            continue
        by_img[ex["image_source"]][ex["answer"]].append(ex)

    # keep only images with at least one yes and one no question
    pairable = {
        src: buckets
        for src, buckets in by_img.items()
        if buckets["yes"] and buckets["no"]
    }
    print(f"  Images with both yes/no questions (adversarial): {len(pairable)}")

    chosen_imgs = rng.sample(sorted(pairable.keys()), min(target_pairs, len(pairable)))

    records: list[ProbeRecord] = []
    n_saved = 0

    for img_src in chosen_imgs:
        buckets = pairable[img_src]
        ex_yes = rng.choice(buckets["yes"])
        ex_no  = rng.choice(buckets["no"])

        img_path = img_dir / f"{img_src}.jpg"
        if not img_path.exists():
            _save_jpeg(ex_yes["image"], img_path)
            n_saved += 1

        abs_path = str(img_path.resolve())
        pair_id  = f"pope_{img_src}"

        for ex, ans in [(ex_yes, "yes"), (ex_no, "no")]:
            records.append(ProbeRecord(
                id=f"pope_{ex['question_id']}",
                source="pope",
                pair_id=pair_id,
                image_path=abs_path,
                foil_image_path=None,   # foil is gaussian noise at token level
                question=ex["question"],
                answer=ans,
                answer_token_id=None,
                corruption_mode="gaussian_noise",
                metadata={
                    "question_id": ex["question_id"],
                    "category": ex["category"],
                    "image_source": img_src,
                    # foil_note documents the expected inference-time protocol
                    "foil_note": (
                        "Apply zero-mean Gaussian noise to visual patch embeddings "
                        "after the visual encoder (before transformer layer 0). "
                        "Scale σ so cosine-sim(clean, noisy) ≈ 0.5."
                    ),
                },
            ))

    print(
        f"  POPE: {len(records)} records from {len(chosen_imgs)} images "
        f"({n_saved} new images saved)"
    )
    return records
