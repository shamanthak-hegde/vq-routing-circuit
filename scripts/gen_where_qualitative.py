"""Qualitative figures for the 'where is the hallucinated object?' report.

Each panel: the real image (which does NOT contain the queried object) beside the
multi-turn transcript(s) of one or more models confabulating its location.

    module load mamba; source activate sae
    python scripts/gen_where_qualitative.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPORTS = {
    "LLaVA-1.6": "results/where_report_llava.jsonl",
    "Qwen2.5-VL": "results/where_report_qwen3vl.jsonl",
    "Chameleon": "results/where_report_chameleon.jsonl",
    "VILA-U": "results/where_report_vilau.jsonl",
}
CACHE = "results/pope_records_cache_50.json"
OUT = Path("figures/where_qualitative")

# Curated panels: (filename, record_id, mode, [models], title)
#   mode="final"  -> show each model's final where-answer (cross-model)
#   mode="ladder" -> show the full re-prompt ladder for one model
# All panels use VISUALLY-VERIFIED clean negatives (the queried object is
# unambiguously absent from the image). pope_8 (car) and pope_42 (dining table)
# were dropped as POPE label noise — see the report's label-noise audit.
PANELS = [
    ("panel_dog_emptyslope.png", "pope_6", "ladder", ["VILA-U"],
     "VILA-U claims a dog on an empty ski slope, then degenerates"),
    ("panel_diningtable_station.png", "pope_14", "ladder", ["VILA-U"],
     "VILA-U places a dining table at a bus station; re-prompts collapse to junk"),
    ("panel_motorcycle_chameleon.png", "pope_16", "ladder", ["Chameleon"],
     "Chameleon refuses first; a direct re-prompt locates a non-existent motorcycle"),
    ("panel_truck_chameleon.png", "pope_24", "ladder", ["Chameleon"],
     "Double-decker bus, no truck: Chameleon confabulates a truck location"),
]


def _load():
    by_model = {}
    for m, fn in REPORTS.items():
        d = {}
        for line in open(fn):
            r = json.loads(line)
            d[r["record_id"]] = r
        by_model[m] = d
    cache = {r["id"]: r for r in json.load(open(CACHE))}
    return by_model, cache


def _wrap(s, w=72):
    return "\n".join(textwrap.fill(line, w) for line in s.splitlines() or [s])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_model, cache = _load()

    for fname, rid, mode, models, title in PANELS:
        rec = cache[rid]
        img = Image.open(rec["image_path"]).convert("RGB")
        ref = next(by_model[m][rid] for m in models if rid in by_model[m])
        obj = ref["object"]

        lines = [f"Q:  Is there a {obj} in the image?",
                 f"    → model answers \"Yes\"  (WRONG — no {obj} is present)", ""]
        if mode == "final":
            for m in models:
                r = by_model[m].get(rid)
                if not r:
                    continue
                tag = "" if r["got_meaningful_location"] else "  [vague/degenerate]"
                lines.append(f"▸ {m}")
                lines.append(f"   Q: Where is the {obj}?")
                lines.append(f"   A: “{r['final_answer']}”{tag}")
                if r["n_attempts"] > 1:
                    lines.append(f"      (after {r['n_attempts']} re-prompts)")
                lines.append("")
        else:  # ladder
            r = by_model[models[0]][rid]
            lines.append(f"▸ {models[0]}  (asked again with more explicit prompts)")
            lines.append("")
            for i, att in enumerate(r["attempts"], 1):
                tag = "MEANINGFUL" if att["meaningful"] else "vague"
                q = att["prompt"] if len(att["prompt"]) < 90 else att["prompt"][:87] + "..."
                lines.append(f"   Q{i}: {q}")
                lines.append(f"   A{i}: “{att['answer']}”   [{tag}]")
                lines.append("")

        text = "\n".join(_wrap(l) for l in lines)

        fig, (axi, axt) = plt.subplots(
            1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.35]})
        axi.imshow(img); axi.axis("off")
        axi.set_title(f"{rec['image_path'].split('/')[-1]}  (ground truth: no {obj})",
                      fontsize=9)
        axt.axis("off")
        axt.text(0.0, 1.0, text, va="top", ha="left", family="monospace",
                 fontsize=9, transform=axt.transAxes)
        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {OUT/fname}")

    print(f"\n{len(PANELS)} panels written to {OUT}/")


if __name__ == "__main__":
    main()
