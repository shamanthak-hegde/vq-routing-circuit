"""Consolidated qualitative report: VLMs asked where a (hallucinated) object is.

Writes all panels plus REPORT.md into a single folder (figures/location_report/).
Panels are minimal: model, image name, ground truth, the yes/no question and
answer, and the "where" question and answer. No reasoning, no arrows, no em
dashes.

    module load mamba; source activate sae
    python scripts/gen_location_report.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

CACHES = ["results/pope_records_cache_50.json",
          "results/pope_records_cache_50_350.json"]
OUT = Path("figures/location_report")

# (file, model, record, object, where_answer)
CASES = [
    # continuous-encoder models
    ("01_llava_dining_table.png", "LLaVA-1.6-7b", "pope_42", "dining table",
     "The dining table is not visible in the image. The man is standing in an "
     "office environment, and there is a desk with a computer monitor and some "
     "papers on it. There is also a window with blinds in the background."),
    ("02_llava_truck.png", "LLaVA-1.6-7b", "pope_30", "truck",
     "The truck is in the middle of the street, surrounded by other vehicles "
     "and pedestrians."),
    ("03_llava_sports_ball.png", "LLaVA-1.6-7b", "pope_104", "sports ball",
     "The sports ball is located in the top right corner of the image."),
    ("04_llava_traffic_light.png", "LLaVA-1.6-7b", "pope_172", "traffic light",
     "In the background, on the right side of the image."),
    ("05_qwen_car.png", "Qwen2.5-VL-7b", "pope_8", "car",
     "The car appears to be parked in the background, behind the person holding "
     "the umbrella and the child standing in front of them."),
    ("06_qwen_dining_table.png", "Qwen2.5-VL-7b", "pope_42", "dining table",
     "The dining table in the image appears to be a desk or a table that is part "
     "of an office setting, as indicated by the presence of a computer monitor, "
     "papers, and a coffee cup on it."),
    ("07_qwen_skis.png", "Qwen2.5-VL-7b", "pope_200", "skis",
     "The image shows a person wakeboarding, not skiing. The board or ski-like "
     "object is attached to the person's feet and is visible under the water."),
    # VQ + weak generator
    ("08_vilau_dog.png", "VILA-U-7b", "pope_6", "dog", "The dog"),
    ("09_vilau_dining_table.png", "VILA-U-7b", "pope_14", "dining table",
     "In the image"),
    # VQ + strong generator
    ("10_chameleon_motorcycle.png", "Chameleon-7b", "pope_16", "motorcycle",
     "The motorcycle is in the foreground, on the left side of the image."),
    ("11_chameleon_truck.png", "Chameleon-7b", "pope_24", "truck",
     "The truck is next to a bus. It is in the center of the picture."),
    ("12_chameleon_book.png", "Chameleon-7b", "pope_36", "book",
     "The book is on the dog's head."),
    ("13_chameleon_bottle.png", "Chameleon-7b", "pope_48", "bottle",
     "The bottle is on top of a file cabinet."),
    ("14_anole_vase.png", "Anole-7b", "pope_22", "vase", "The vase is on the bus."),
]


def _clean(s: str) -> str:
    return s.replace("—", ",").replace("–", ",").replace("→", " ")


def _wrap(s, w=58):
    return "\n".join(textwrap.fill(line, w) for line in s.splitlines() or [s])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    for c in CACHES:
        for r in json.load(open(c)):
            cache.setdefault(r["id"], r)

    rows = []
    for fname, model, rid, obj, ans in CASES:
        rec = cache[rid]
        img_name = Path(rec["image_path"]).name
        ans = _clean(ans)
        img = Image.open(rec["image_path"]).convert("RGB")
        lines = [
            f"Model: {model}",
            f"Image name: {img_name}",
            f"Ground truth: no {obj} in the image",
            "",
            f"Q1: Is there a {obj} in the image?",
            "A1: Yes",
            "",
            f"Q2: Where is the {obj} in the image?",
            f"A2: {ans}",
        ]
        text = "\n".join(_wrap(l) for l in lines)

        fig, (axi, axt) = plt.subplots(
            1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.1]})
        axi.imshow(img); axi.axis("off")
        axt.axis("off")
        axt.text(0.0, 1.0, text, va="top", ha="left", family="monospace",
                 fontsize=11, transform=axt.transAxes)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        rows.append((fname, model, img_name, obj, ans))
        print(f"wrote {OUT/fname}")

    # report
    md = ["# Asking VLMs where a hallucinated object is",
          "",
          "Each case is a POPE-adversarial record where the ground truth is that "
          "the object is absent, the model still answers \"Yes\", and we then ask "
          "where the object is. Panels (one per case) are the PNG files in this "
          "folder.",
          "",
          "| # | Model | Image name | Ground truth | Question | Answer |",
          "|---|-------|-----------|--------------|----------|--------|"]
    for i, (fname, model, img_name, obj, ans) in enumerate(rows, 1):
        md.append(f"| {i} | {model} | {img_name} | no {obj} | "
                  f"Where is the {obj} in the image? | {ans} |")
    md += ["",
           "## Panel files",
           ""]
    for fname, model, img_name, obj, ans in rows:
        md.append(f"- `{fname}` ({model}, {obj}, {img_name})")
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {OUT/'REPORT.md'}")
    print(f"{len(rows)} panels in {OUT}/")


if __name__ == "__main__":
    main()
