"""Caption-quality / informativeness metrics for the CHAIR caption sets. Complements the object-mention metrics in
`chair_robustness.py` with standard reference-based scores (CIDEr, METEOR, SPICE)
against the COCO val2014 5-reference captions.

Purpose: show whether L0 ablation reduces hallucination *while preserving caption
informativeness*, or whether it just makes captions shorter/vaguer. SPICE
(scene-graph F1) is the most relevant for detailed captions; CIDEr/METEOR are
verbosity-sensitive against short COCO refs, so read the baseline→L0 *direction*,
not the absolute values.

Usage:
    source activate sae
    python -m probe.analysis.chair_caption_quality

Outputs:
  results/chair_caption_quality.json
  results/chair_caption_quality.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

# SPICE is OFF by default: its Java scene-graph parser is intractable on the
# ~158-word detailed captions in these sets (single runs exceed 20 min and hog
# the node). The object-mention / distinct-object / recall metrics in
# chair_robustness.py are the informativeness signal we rely on. Set
# CHAIR_SPICE=1 to enable it on a dedicated node.
_USE_SPICE = os.environ.get("CHAIR_SPICE", "0") == "1"

RESULTS = Path("results")
COCO_CAPS = "probe/benchmarks/coco_annotations/captions_val2014.json"

PAIRS = {
    "vilau": ("results/chair_captions_vilau_baseline.jsonl",
              "results/chair_captions_vilau_L0.jsonl"),
    "llava_vq_K65536": ("results/chair_captions_llava_vq_K65536_baseline.jsonl",
                        "results/chair_captions_llava_vq_K65536_L0.jsonl"),
}


def _load_refs() -> dict[int, list[dict]]:
    data = json.load(open(COCO_CAPS))
    refs: dict[int, list[dict]] = {}
    for a in data["annotations"]:
        refs.setdefault(a["image_id"], []).append({"caption": a["caption"]})
    return refs


def _candidates(jsonl: str) -> dict[int, list[dict]]:
    out = {}
    for line in open(jsonl):
        if line.strip():
            d = json.loads(line)
            out[d["image_id"]] = [{"caption": d["caption"]}]
    return out


def _score(refs: dict, jsonl: str, tok: PTBTokenizer,
           scorers: list) -> dict[str, float]:
    cand = _candidates(jsonl)
    ids = sorted(set(cand) & set(refs))
    gts = tok.tokenize({i: refs[i] for i in ids})
    res = tok.tokenize({i: cand[i] for i in ids})
    out = {}
    for scorer, names in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
        except Exception as e:  # keep partial results if one scorer fails
            print(f"  [warn] {names} failed: {e}")
            continue
        if isinstance(names, list):
            for n, s in zip(names, score):
                out[n] = float(s)
        else:
            out[names] = float(score)
    return out


def main() -> None:
    refs = _load_refs()
    tok = PTBTokenizer()
    scorers = [
        (Cider(), "CIDEr"),
        (Meteor(), "METEOR"),
    ]
    if _USE_SPICE:
        from pycocoevalcap.spice.spice import Spice
        scorers.append((Spice(), "SPICE"))
    report = {"models": {}}
    for model, (base_f, l0_f) in PAIRS.items():
        b = _score(refs, base_f, tok, scorers)
        l = _score(refs, l0_f, tok, scorers)
        report["models"][model] = {
            "baseline": b, "l0": l,
            "delta": {k: l[k] - b[k] for k in b},
        }

    (RESULTS / "chair_caption_quality.json").write_text(json.dumps(report, indent=2))

    L = ["# Caption-quality metrics: CIDEr / METEOR / SPICE\n",
         "Reference-based scores vs COCO val2014 5-reference captions. SPICE "
         "(scene-graph F1) is the informativeness-relevant metric for detailed "
         "captions; CIDEr/METEOR penalize verbosity against short refs — read the "
         "baseline→L0 *direction*.\n"]
    for model, m in report["models"].items():
        L.append(f"\n## {model}\n")
        L.append("| metric | baseline | L0 ablation | Δ |")
        L.append("|---|---|---|---|")
        for k in m["baseline"]:
            L.append(f"| {k} | {m['baseline'][k]:.4f} | {m['l0'][k]:.4f} "
                     f"| {m['delta'][k]:+.4f} |")
    (RESULTS / "chair_caption_quality.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nWrote results/chair_caption_quality.{json,md}")


if __name__ == "__main__":
    main()
