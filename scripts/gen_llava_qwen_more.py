"""More LLaVA / Qwen samples, mined from a larger POPE slice (records 50-349).

Run files: results/where_report_{llava,qwen3vl}_b.jsonl (cache
results/pope_records_cache_50_350.json). Each case visually verified.

Finding: on a larger slice LLaVA *does* produce genuine confident wrong-location
hallucinations, driven by scene priors (baseball -> ball, road -> traffic light).
Qwen still essentially does not fabricate — its new FPs are a label error
(bottles in a shower) and a self-correction (wakeboard, not skis).

    module load mamba; source activate sae
    python scripts/gen_llava_qwen_more.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

CACHE = "results/pope_records_cache_50_350.json"
OUT = Path("figures/llava_qwen_more")

# (file, record, model, object, answer, note, title)
CASES = [
    ("panel_llava_sportsball_baseball.png", "pope_104", "LLaVA-1.6", "sports ball",
     "The sports ball is located in the top right corner of the image.",
     "It is a baseball at-bat, but no ball is visible (the pitch is not in "
     "frame). The top-right corner is just the crowd in the stands. LLaVA invents "
     "a ball from the baseball scene prior and confidently places it where there "
     "is none.",
     "LLaVA: sports ball “in the top right corner” (no ball; that's the crowd)"),
    ("panel_llava_trafficlight_car.png", "pope_172", "LLaVA-1.6", "traffic light",
     "In the background, on the right side of the image.",
     "The image is a dog looking out a car window onto a tree-lined road. There "
     "is no traffic light anywhere — the right side is trees. A road-scene prior "
     "drives a confident but baseless placement.",
     "LLaVA: traffic light “background, right side” (only trees there)"),
    ("panel_qwen_skis_wakeboard.png", "pope_200", "Qwen2.5-VL", "skis",
     "The image shows a person wakeboarding, not skiing. The board or ski-like "
     "object is attached to the person's feet and is visible under the water.",
     "Qwen answered \"yes\" at the yes/no stage, but when asked where, it "
     "self-corrects: it identifies the real wakeboard and explicitly says it is "
     "not skiing. Grounded reinterpretation, not a phantom placement.",
     "Qwen self-corrects: “wakeboarding, not skiing” (no skis fabricated)"),
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
