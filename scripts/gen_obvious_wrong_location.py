"""Qualitative panels: models that confidently give an OBVIOUSLY WRONG location
for an object that is not in the image.

All transcripts are mined from already-inferred runs (no GPU): the conversational
follow-up / re-prompt files results/{halluc_followup_conv_,where_report_}*.jsonl.
Each case was visually verified against the image. Image left, transcript +
"why it's obviously wrong" right.

    module load mamba; source activate sae
    python scripts/gen_obvious_wrong_location.py
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
OUT = Path("figures/obvious_wrong_location")

# (file, record, model, object, where_answer, why_obviously_wrong, title)
CASES = [
    ("panel_book_on_dogs_head.png", "pope_36", "Chameleon-7b", "book",
     "The book is on the dog's head.",
     "The image clearly shows a dog and a newspaper lying on the table next to "
     "it. There is no book on the dog's head — the model places a (non-existent) "
     "book at a location that is plainly false even though the dog is real.",
     "Chameleon: “the book is on the dog’s head” (it is on the table)"),
    ("panel_bottle_filecabinet.png", "pope_48", "Chameleon-7b", "bottle",
     "The bottle is on top of a file cabinet.",
     "The image is an office kitchen with microwaves and a wall notice. There is "
     "no bottle and no file cabinet anywhere in the picture.",
     "Chameleon: “bottle on top of a file cabinet” (neither exists)"),
    ("panel_truck_next_to_bus.png", "pope_24", "Chameleon-7b", "truck",
     "The truck is next to a bus. It is in the center of the picture.",
     "The only large vehicle is a red double-decker bus. There is no truck "
     "beside it.",
     "Chameleon: “truck next to a bus, center of the picture” (no truck)"),
    ("panel_motorcycle_foreground.png", "pope_16", "Chameleon-7b", "motorcycle",
     "The motorcycle is in the foreground, on the left side of the image.",
     "The scene is a transit station with buses and one bicycle. There is no "
     "motorcycle in the foreground or anywhere else.",
     "Chameleon: “motorcycle in the foreground, left side” (none present)"),
    ("panel_vase_on_the_bus.png", "pope_22", "Anole-7b", "vase",
     "The vase is on the bus.",
     "The image is a street scene with a red double-decker bus. There is no vase "
     "anywhere — Anole invents one and places it on the bus.",
     "Anole: “the vase is on the bus” (no vase exists)"),
]


def _wrap(s, w=64):
    return "\n".join(textwrap.fill(line, w) for line in s.splitlines() or [s])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {r["id"]: r for r in json.load(open(CACHE))}

    for fname, rid, model, obj, ans, why, title in CASES:
        rec = cache[rid]
        img = Image.open(rec["image_path"]).convert("RGB")
        lines = [
            f"Q1: Is there a {obj} in the image?",
            f"    → {model} answers “Yes”   (WRONG — no {obj} present)",
            "",
            f"Q2: Where is the {obj} in the image?",
            f"A2: “{ans}”",
            "",
            "Why this is obviously wrong:",
            why,
        ]
        text = "\n".join(_wrap(l) for l in lines)

        fig, (axi, axt) = plt.subplots(
            1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.15]})
        axi.imshow(img); axi.axis("off")
        axi.set_title(f"{Path(rec['image_path']).name}  (ground truth: no {obj})",
                      fontsize=9)
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
