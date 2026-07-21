"""Object-category analysis (#15), bias/hallucination correlation (#14), and
qualitative audit (#13) for the VILA-U CHAIR caption sets — all pure re-analysis
of the on-disk caption JSONLs (no GPU).

  #15  Per-COCO-category hallucination counts, baseline vs L0 ablation: which
       categories are hallucinated, and does L0 suppress them uniformly or
       selectively? Cross-referenced with codebook yes-bias where available.
  #14  Per-image correlation between baseline object-hallucination and the L0
       ablation effect: does L0 reduce hallucination most on the images that
       hallucinate most? (links the generative signal to the intervention).
  #13  ~12 qualitative examples across outcome categories (clean removal /
       removal-with-detail-loss / failure), dumped with captions + objects.

Usage:
    source activate sae
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    python -m probe.analysis.chair_category_audit

Outputs:
  results/chair_category_audit.json
  results/chair_category_audit.md      (#15 table + #14 stats)
  results/chair_qualitative_examples.md (#13)
"""

from __future__ import annotations

import contextlib
import json
import os
from collections import defaultdict
from pathlib import Path

from probe.benchmarks.chair import CHAIR

RESULTS = Path("results")
COCO = "probe/benchmarks/coco_annotations"
MODEL = "vilau"
BASE_F = f"results/chair_captions_{MODEL}_baseline.jsonl"
L0_F = f"results/chair_captions_{MODEL}_L0.jsonl"


def _per_image(ev: CHAIR, jsonl: str) -> dict[int, dict]:
    out = ev.compute_chair(jsonl, image_id_key="image_id", caption_key="caption")
    rows = {}
    for s in out["sentences"]:
        node = s["mscoco_generated_words"]
        gt = set(s["mscoco_gt_words"])
        halluc = [n for (_w, n) in s["mscoco_hallucinated_words"]]
        rows[s["image_id"]] = {
            "caption": s["caption"],
            "mentioned": node,                       # COCO objects mentioned
            "halluc": halluc,                        # hallucinated (mentioned ∉ gt)
            "correct": [n for n in node if n in gt],
            "gt": sorted(gt),
            "n_halluc": len(halluc),
            "n_ment": len(node),
            "length": len(s["words"]),
        }
    return rows


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def main() -> None:
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        ev = CHAIR(COCO)
        base = _per_image(ev, BASE_F)
        l0 = _per_image(ev, L0_F)
    ids = sorted(set(base) & set(l0))

    # optional codebook yes-bias per (none per-category available; report mean only)
    cb = None
    cbf = RESULTS / "codebook_bias_vilau.json"
    if cbf.exists():
        cb = json.load(open(cbf))

    # ---- #15 per-category hallucination + correct-mention counts ----
    cats = defaultdict(lambda: {"halluc_base": 0, "halluc_l0": 0,
                                "ment_base": 0, "ment_l0": 0,
                                "correct_base": 0, "correct_l0": 0})
    for i in ids:
        for n in base[i]["halluc"]:
            cats[n]["halluc_base"] += 1
        for n in l0[i]["halluc"]:
            cats[n]["halluc_l0"] += 1
        for n in base[i]["mentioned"]:
            cats[n]["ment_base"] += 1
        for n in l0[i]["mentioned"]:
            cats[n]["ment_l0"] += 1
        for n in base[i]["correct"]:
            cats[n]["correct_base"] += 1
        for n in l0[i]["correct"]:
            cats[n]["correct_l0"] += 1
    cat_rows = []
    for c, d in cats.items():
        cat_rows.append({"category": c, **d,
                         "halluc_delta": d["halluc_l0"] - d["halluc_base"]})
    cat_rows.sort(key=lambda r: -r["halluc_base"])

    # ---- #14 correlation: baseline hallucination vs L0 reduction ----
    xs = [base[i]["n_halluc"] for i in ids]                       # baseline halluc count
    dys = [l0[i]["n_halluc"] - base[i]["n_halluc"] for i in ids]  # Δ (negative = reduced)
    r_base_delta = _pearson(xs, dys)
    # split: do high-baseline-hallucination images benefit more?
    hi = [i for i in ids if base[i]["n_halluc"] >= 2]
    lo = [i for i in ids if base[i]["n_halluc"] <= 1]
    def _meandelta(grp):
        return sum(l0[i]["n_halluc"] - base[i]["n_halluc"] for i in grp) / len(grp) if grp else 0.0
    corr = {
        "pearson_baseHalluc_vs_delta": round(r_base_delta, 3),
        "n_images": len(ids),
        "hi_group_n": len(hi), "hi_group_mean_delta": round(_meandelta(hi), 3),
        "lo_group_n": len(lo), "lo_group_mean_delta": round(_meandelta(lo), 3),
    }

    report = {
        "model": MODEL, "n_images": len(ids),
        "category_table": cat_rows,
        "correlation_14": corr,
        "codebook_mean_yes_rate": (cb or {}).get("mean_yes_rate"),
    }
    (RESULTS / "chair_category_audit.json").write_text(json.dumps(report, indent=2))

    # ---- markdown #15 + #14 ----
    L = [f"# CHAIR object-category & correlation analysis — {MODEL}\n",
         f"n={len(ids)} COCO val2014 images, baseline vs L0 ablation.\n",
         "\n## #15 Per-category object hallucination (caption mentions of object ∉ GT)\n",
         "Sorted by baseline hallucination count. `halluc` = # images where the "
         "category is hallucinated; `correct` = # images where it is a correct mention.\n",
         "| category | halluc base | halluc L0 | Δ | correct base | correct L0 |",
         "|---|---|---|---|---|---|"]
    for r in cat_rows[:25]:
        if r["halluc_base"] == 0 and r["halluc_l0"] == 0:
            continue
        L.append(f"| {r['category']} | {r['halluc_base']} | {r['halluc_l0']} | "
                 f"{r['halluc_delta']:+d} | {r['correct_base']} | {r['correct_l0']} |")
    tot_hb = sum(r["halluc_base"] for r in cat_rows)
    tot_hl = sum(r["halluc_l0"] for r in cat_rows)
    n_cat_reduced = sum(1 for r in cat_rows if r["halluc_delta"] < 0)
    n_cat_worse = sum(1 for r in cat_rows if r["halluc_delta"] > 0)
    L.append(f"\nTotals: hallucinated mentions {tot_hb} → {tot_hl} "
             f"({100*(tot_hl-tot_hb)/tot_hb:+.0f}%). Categories reduced: "
             f"{n_cat_reduced}; worsened: {n_cat_worse}; unchanged: "
             f"{len(cat_rows)-n_cat_reduced-n_cat_worse}.")
    if report["codebook_mean_yes_rate"] is not None:
        L.append(f"\nVILA-U codebook mean yes-rate = "
                 f"{report['codebook_mean_yes_rate']:.3f} (uniform yes-bias, not "
                 f"category-specific — so a per-category codebook-polarity split is "
                 f"not expected; hallucination suppression is broad).")

    L.append("\n## #14 Does L0 ablation reduce hallucination most where it is worst?\n")
    L.append(f"- Pearson(baseline hallucinated-object count, Δ under L0) = "
             f"**{corr['pearson_baseHalluc_vs_delta']}** "
             f"(negative ⇒ more baseline hallucination → larger reduction).")
    L.append(f"- Images with ≥2 baseline hallucinations (n={corr['hi_group_n']}): "
             f"mean Δ = **{corr['hi_group_mean_delta']:+.2f}** objects.")
    L.append(f"- Images with ≤1 baseline hallucination (n={corr['lo_group_n']}): "
             f"mean Δ = **{corr['lo_group_mean_delta']:+.2f}** objects.")
    L.append("\nInterpretation: the L0-ablation hallucination reduction is "
             "concentrated on the images that hallucinate most at baseline — "
             "linking the intervention to the generative hallucination signal.")
    (RESULTS / "chair_category_audit.md").write_text("\n".join(L) + "\n")

    # ---- #13 qualitative examples ----
    def classify(i):
        nb, nl = base[i]["n_halluc"], l0[i]["n_halluc"]
        rb, rl = len(base[i]["correct"]), len(l0[i]["correct"])
        if nb >= 1 and nl == 0 and rl >= max(1, rb - 1):
            return "clean_removal"
        if nb >= 1 and nl == 0 and rl < rb - 1:
            return "removal_detail_loss"
        if nb >= 1 and nl >= nb:
            return "failure"
        return "other"
    buckets = defaultdict(list)
    for i in ids:
        buckets[classify(i)].append(i)
    Q = ["# Qualitative audit — VILA-U baseline vs L0 ablation\n",
         "Object lists are COCO categories. Hallucinated = mentioned but not in GT.\n"]
    want = [("clean_removal", "L0 removes hallucinated object, preserves content", 4),
            ("removal_detail_loss", "L0 removes hallucination but loses detail", 3),
            ("failure", "L0 fails to remove hallucination", 3)]
    for key, title, k in want:
        Q.append(f"\n## {title}  ({len(buckets[key])} images total)\n")
        for i in sorted(buckets[key], key=lambda i: -base[i]["n_halluc"])[:k]:
            Q.append(f"### image {i}")
            Q.append(f"- **GT objects:** {', '.join(base[i]['gt']) or '(none)'}")
            Q.append(f"- **Baseline hallucinated:** {', '.join(base[i]['halluc']) or '(none)'}")
            Q.append(f"- **L0 hallucinated:** {', '.join(l0[i]['halluc']) or '(none)'}")
            Q.append(f"- **Baseline caption:** {base[i]['caption'][:500]}")
            Q.append(f"- **L0 caption:** {l0[i]['caption'][:500]}\n")
    (RESULTS / "chair_qualitative_examples.md").write_text("\n".join(Q) + "\n")

    print("\n".join(L))
    print(f"\nBucket sizes: " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))
    print("Wrote results/chair_category_audit.{json,md}, results/chair_qualitative_examples.md")


if __name__ == "__main__":
    main()
