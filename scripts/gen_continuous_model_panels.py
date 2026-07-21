"""Qualitative samples from the continuous-encoder models (LLaVA, Qwen2.5-VL).

These contrast with the VQ "obvious wrong location" panels: on this slice the
continuous models' false positives are POPE label errors or category confusions,
so their answers track real evidence, reinterpret, or self-correct rather than
fabricating an object from nothing. Transcripts are mined from already-inferred
runs (no GPU). Each case visually verified.

    module load mamba; source activate sae
    python scripts/gen_continuous_model_panels.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

CACHE = "results/pope_records_cache_50.json"
OUT = Path("figures/continuous_samples")

# (file, record, model, object, where_answer, note, title)
CASES = [
    ("panel_qwen_car_labelerror.png", "pope_8", "Qwen2.5-VL", "car",
     "The car appears to be parked in the background, behind the person holding "
     "the umbrella and the child standing in front of them.",
     "A dark vehicle IS parked in the background, so POPE's 'car: no' is a label "
     "error. Qwen points straight at it — a correct answer on a mislabeled item, "
     "not a hallucination. Continuous-encoder models track the real pixels.",
     "Qwen2.5-VL locates a vehicle POPE mislabeled as absent"),
    ("panel_llava_selfcorrect.png", "pope_42", "LLaVA-1.6", "dining table",
     "The dining table is not visible in the image. The man is standing in an "
     "office environment, and there is a desk with a computer monitor and papers "
     "on it. There is also a window with blinds in the background.",
     "Asked where the (POPE-negative) dining table is, LLaVA declines to place "
     "it and reinterprets the scene as an office desk — it self-corrects instead "
     "of confabulating a location (single-turn follow-up).",
     "LLaVA-1.6 self-corrects instead of inventing a location"),
    ("panel_llava_truck_catconfusion.png", "pope_30", "LLaVA-1.6", "truck",
     "The truck is in the middle of the street, surrounded by other vehicles and "
     "pedestrians.",
     "Through the windshield there is a white car/van ahead but no truck. LLaVA "
     "gives a plausible location for the misread vehicle — a fine-grained "
     "category confusion, not an object invented from nothing.",
     "LLaVA-1.6: category confusion (a van read as a truck)"),
    ("panel_qwen_desk_reinterpret.png", "pope_42", "Qwen2.5-VL", "dining table",
     "The dining table in the image appears to be a desk or a table that is part "
     "of an office setting, as indicated by the computer monitor, papers, and a "
     "coffee cup on it.",
     "Qwen hedges rather than fabricating: it grounds the 'dining table' in the "
     "real desk the man eats at (COCO's dining-table class is broad). Grounded "
     "reinterpretation, not a phantom placement.",
     "Qwen2.5-VL reinterprets the real desk as the 'dining table'"),
]


def _wrap(s, w=64):
    return "\n".join(textwrap.fill(line, w) for line in s.splitlines() or [s])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {r["id"]: r for r in json.load(open(CACHE))}
    for fname, rid, model, obj, ans, note, title in CASES:
        rec = cache[rid]
        img = Image.open(rec["image_path"]).convert("RGB")
        lines = [
            f"Q1: Is there a {obj} in the image?",
            f"    → {model} answers “Yes”   (POPE ground truth: no)",
            "",
            f"Q2: Where is the {obj} in the image?",
            f"A2: “{ans}”",
            "",
            "What's going on:",
            note,
        ]
        text = "\n".join(_wrap(l) for l in lines)
        fig, (axi, axt) = plt.subplots(
            1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.15]})
        axi.imshow(img); axi.axis("off")
        axi.set_title(f"{Path(rec['image_path']).name}", fontsize=9)
        axt.axis("off")
        axt.text(0.0, 1.0, text, va="top", ha="left", family="monospace",
                 fontsize=10, transform=axt.transAxes)
        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(OUT / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {OUT/fname}")
    print(f"\n{len(CASES)} panels written to {OUT}/")


if __name__ == "__main__":
    main()
