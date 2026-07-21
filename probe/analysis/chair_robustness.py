"""CHAIR robustness re-analysis for the L0-ablation intervention claim.

Covers #5 (confidence intervals), #3 (length-matched CHAIR),
and #4 (caption-quality / informativeness metrics) — all as pure re-analysis of
the already-generated caption JSONLs (no GPU, no regeneration). The two model
pairs on disk are VILA-U and LLaVA-VQ-K65536, each with baseline + L0-ablation
captions over the same 500 COCO val2014 images (seed=0).

  #5  Paired image-level bootstrap 95% CIs for CHAIRi, CHAIRs, Recall, mean object
      mentions, and mean caption length, plus the baseline→L0 Δ for each.
  #3  Length controls for the CHAIRi reduction:
        (B) per-image length match — truncate each baseline caption to the word
            count of its paired L0 caption (same image_id), re-tokenize, recompute.
        (C) object-mention bins — compare CHAIRi within fixed mention-count bins.
  #4  Informativeness: mean object mentions / caption, distinct objects / caption,
      total distinct objects, fraction of captions with >=1 object mention.

CHAIRi is corpus-level: sum(hallucinated object mentions) / sum(object mentions).
The per-image arrays needed for all three analyses come from one CHAIR scoring
pass (output['sentences']).

Usage:
    source activate sae
    python -m probe.analysis.chair_robustness

Outputs:
  results/chair_robustness.json
  results/chair_robustness.md
"""

from __future__ import annotations

import contextlib
import json
import os
import random
from pathlib import Path

from probe.benchmarks.chair import CHAIR

RESULTS = Path("results")
COCO = "probe/benchmarks/coco_annotations"
N_BOOT = 10000
SEED = 0

PAIRS = {
    "vilau": ("results/chair_captions_vilau_baseline.jsonl",
              "results/chair_captions_vilau_L0.jsonl"),
    "llava_vq_K65536": ("results/chair_captions_llava_vq_K65536_baseline.jsonl",
                        "results/chair_captions_llava_vq_K65536_L0.jsonl"),
}


def _per_image(ev: CHAIR, jsonl: str) -> dict[int, dict]:
    """image_id -> {halluc, mentions, recall_num, gt_num, length, distinct, has_obj}."""
    out = ev.compute_chair(jsonl, image_id_key="image_id", caption_key="caption")
    rows = {}
    for s in out["sentences"]:
        node = s["mscoco_generated_words"]
        gt = set(s["mscoco_gt_words"])
        recall_objs = {n for n in node if n in gt}
        rows[s["image_id"]] = {
            "halluc": len(s["mscoco_hallucinated_words"]),
            "mentions": len(node),
            "recall_num": len(recall_objs),
            "gt_num": len(gt),
            "length": len(s["words"]),
            "distinct": len(set(node)),
            "has_obj": int(len(node) > 0),
        }
    return rows, out["overall_metrics"]


def _corpus_metrics(ids: list[int], R: dict[int, dict]) -> dict:
    H = sum(R[i]["halluc"] for i in ids)
    M = sum(R[i]["mentions"] for i in ids)
    rec = sum(R[i]["recall_num"] for i in ids)
    gt = sum(R[i]["gt_num"] for i in ids)
    n = len(ids)
    return {
        "CHAIRi": 100 * H / M if M else 0.0,
        "CHAIRs": 100 * sum(1 for i in ids if R[i]["halluc"] > 0) / n,
        "Recall": 100 * rec / gt if gt else 0.0,
        "mentions": sum(R[i]["mentions"] for i in ids) / n,
        "distinct": sum(R[i]["distinct"] for i in ids) / n,
        "length": sum(R[i]["length"] for i in ids) / n,
        "has_obj_frac": 100 * sum(R[i]["has_obj"] for i in ids) / n,
    }


def _ci(vals: list[float]) -> tuple[float, float]:
    s = sorted(vals)
    lo = s[int(0.025 * len(s))]
    hi = s[int(0.975 * len(s)) - 1]
    return lo, hi


def _bootstrap(ids: list[int], base: dict, l0: dict) -> dict:
    rng = random.Random(SEED)
    keys = ["CHAIRi", "CHAIRs", "Recall", "mentions", "distinct", "length", "has_obj_frac"]
    samp = {f"{c}_{k}": [] for c in ("base", "l0", "delta") for k in keys}
    n = len(ids)
    for _ in range(N_BOOT):
        res = [ids[rng.randrange(n)] for _ in range(n)]
        mb = _corpus_metrics(res, base)
        ml = _corpus_metrics(res, l0)
        for k in keys:
            samp[f"base_{k}"].append(mb[k])
            samp[f"l0_{k}"].append(ml[k])
            samp[f"delta_{k}"].append(ml[k] - mb[k])
    point_b = _corpus_metrics(ids, base)
    point_l = _corpus_metrics(ids, l0)
    res = {}
    for k in keys:
        lo_d, hi_d = _ci(samp[f"delta_{k}"])
        lo_b, hi_b = _ci(samp[f"base_{k}"])
        lo_l, hi_l = _ci(samp[f"l0_{k}"])
        res[k] = {
            "base": point_b[k], "base_ci": [lo_b, hi_b],
            "l0": point_l[k], "l0_ci": [lo_l, hi_l],
            "delta": point_l[k] - point_b[k], "delta_ci": [lo_d, hi_d],
            "delta_sig": (lo_d > 0) or (hi_d < 0),
        }
    return res


def _truncate_words(text: str, n_words: int) -> str:
    return " ".join(text.split()[:max(0, n_words)])


def _length_match(ev: CHAIR, base_jsonl: str, l0_rows: dict[int, dict]) -> dict:
    """#3(B): truncate each baseline caption to its paired L0 word count, recompute
    corpus CHAIR on the truncated baseline. Returns corpus metrics + per-image rows."""
    rows = {}
    for line in open(base_jsonl):
        if not line.strip():
            continue
        d = json.loads(line)
        imid = d["image_id"]
        if imid not in l0_rows:
            continue
        target = l0_rows[imid]["length"]  # raw-word count of the paired L0 caption
        trunc = _truncate_words(d["caption"], target)
        words, node_words, _, _ = ev.caption_to_words(trunc)
        gt = set(ev.imid_to_objects[imid])
        halluc = sum(1 for n in node_words if n not in gt)
        recall_objs = {n for n in node_words if n in gt}
        rows[imid] = {
            "halluc": halluc, "mentions": len(node_words),
            "recall_num": len(recall_objs), "gt_num": len(gt),
            "length": len(trunc.split()), "distinct": len(set(node_words)),
            "has_obj": int(len(node_words) > 0),
        }
    return rows


def _object_count_bins(ids: list[int], base: dict, l0: dict) -> list[dict]:
    """#3(C): bin images by baseline object-mention count; CHAIRi per bin per cond."""
    bins = [(0, 0), (1, 2), (3, 4), (5, 8), (9, 999)]
    out = []
    for lo, hi in bins:
        sel = [i for i in ids if lo <= base[i]["mentions"] <= hi]
        if not sel:
            continue
        out.append({
            "bin": f"{lo}-{hi if hi < 999 else '+'}",
            "n": len(sel),
            "base_CHAIRi": _corpus_metrics(sel, base)["CHAIRi"],
            "l0_CHAIRi": _corpus_metrics(sel, l0)["CHAIRi"],
            "base_mentions": _corpus_metrics(sel, base)["mentions"],
            "l0_mentions": _corpus_metrics(sel, l0)["mentions"],
        })
    return out


def main() -> None:
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        ev = CHAIR(COCO)  # one shared evaluator (silence 292k-mask build spam)
    report = {"n_boot": N_BOOT, "seed": SEED, "models": {}}

    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        for model, (base_f, l0_f) in PAIRS.items():
            base_rows, _ = _per_image(ev, base_f)
            l0_rows, _ = _per_image(ev, l0_f)
            ids = sorted(set(base_rows) & set(l0_rows))

            boot = _bootstrap(ids, base_rows, l0_rows)
            trunc_rows = _length_match(ev, base_f, l0_rows)
            ids_t = sorted(set(trunc_rows) & set(l0_rows))
            lm_base = _corpus_metrics(ids_t, base_rows)
            lm_trunc = _corpus_metrics(ids_t, trunc_rows)
            lm_l0 = _corpus_metrics(ids_t, l0_rows)
            obj_bins = _object_count_bins(ids, base_rows, l0_rows)

            report["models"][model] = {
                "n_images": len(ids),
                "bootstrap": boot,
                "length_match": {
                    "base_full": lm_base, "base_truncated_to_L0_len": lm_trunc,
                    "l0": lm_l0,
                },
                "object_count_bins": obj_bins,
            }

    (RESULTS / "chair_robustness.json").write_text(json.dumps(report, indent=2))

    # ---- markdown ----
    L = ["# CHAIR robustness re-analysis\n",
         f"Image-level paired bootstrap, {N_BOOT} resamples, seed {SEED}. "
         "CHAIRi/CHAIRs/Recall/has-obj in %, lengths in words.\n"]
    for model, m in report["models"].items():
        L.append(f"\n## {model}  (n={m['n_images']} images)\n")
        L.append("### #5 Bootstrap 95% CIs (baseline → L0 ablation)\n")
        L.append("| metric | baseline [95% CI] | L0 ablation [95% CI] | Δ [95% CI] | sig |")
        L.append("|---|---|---|---|---|")
        names = {"CHAIRi": "CHAIRi ↓", "CHAIRs": "CHAIRs ↓", "Recall": "Recall ↑",
                 "mentions": "obj mentions/cap", "distinct": "distinct obj/cap",
                 "length": "avg length (words)", "has_obj_frac": "% caps w/ ≥1 obj"}
        for k, lab in names.items():
            b = m["bootstrap"][k]
            L.append(f"| {lab} | {b['base']:.2f} [{b['base_ci'][0]:.2f}, {b['base_ci'][1]:.2f}] "
                     f"| {b['l0']:.2f} [{b['l0_ci'][0]:.2f}, {b['l0_ci'][1]:.2f}] "
                     f"| {b['delta']:+.2f} [{b['delta_ci'][0]:+.2f}, {b['delta_ci'][1]:+.2f}] "
                     f"| {'**yes**' if b['delta_sig'] else 'no'} |")

        lm = m["length_match"]
        L.append("\n### #3(B) Per-image length-matched CHAIR (baseline truncated to paired-L0 length)\n")
        L.append("| condition | avg length | obj mentions/cap | CHAIRi ↓ | CHAIRs ↓ | Recall |")
        L.append("|---|---|---|---|---|---|")
        for cond, lab in [("base_full", "baseline (full)"),
                          ("base_truncated_to_L0_len", "baseline (truncated→L0 len)"),
                          ("l0", "L0 ablation")]:
            c = lm[cond]
            L.append(f"| {lab} | {c['length']:.1f} | {c['mentions']:.2f} | "
                     f"{c['CHAIRi']:.2f} | {c['CHAIRs']:.2f} | {c['Recall']:.2f} |")
        d = lm["l0"]["CHAIRi"] - lm["base_truncated_to_L0_len"]["CHAIRi"]
        L.append(f"\n*Length-controlled ΔCHAIRi (L0 − truncated-baseline) = {d:+.2f} pp "
                 f"(if still < 0, the CHAIRi gain is not explained by brevity).*\n")

        L.append("\n### #3(C) CHAIRi within baseline object-mention bins\n")
        L.append("| bin (baseline mentions) | n | baseline CHAIRi | L0 CHAIRi | Δ |")
        L.append("|---|---|---|---|---|")
        for bn in m["object_count_bins"]:
            L.append(f"| {bn['bin']} | {bn['n']} | {bn['base_CHAIRi']:.2f} | "
                     f"{bn['l0_CHAIRi']:.2f} | {bn['l0_CHAIRi'] - bn['base_CHAIRi']:+.2f} |")

    (RESULTS / "chair_robustness.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nWrote results/chair_robustness.{json,md}")


if __name__ == "__main__":
    main()
