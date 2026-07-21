"""L0-attenuation dose-response on CHAIR.

Scores the VILA-U open-ended captioning runs at five L0-ablation strengths and
reports, at each strength, the hallucination rate (CHAIRi) and a fine-detail-
retention metric. Pure re-analysis of caption JSONLs (no GPU); the caption files
are produced by probe.benchmarks.run_chair.

Attenuation strength alpha is the fraction of the L0 self-attention output that is
removed:  alpha = 1 - coeff, where coeff is the ScalarLayerScale retained fraction.
  alpha = 0.00  -> baseline           (coeff 1.0; chair_captions_vilau_baseline.jsonl)
  alpha = 0.25  -> coeff 0.75
  alpha = 0.50  -> coeff 0.50
  alpha = 0.75  -> coeff 0.25
  alpha = 1.00  -> full L0 knockout   (coeff 0.0; chair_captions_vilau_L0.jsonl)

Fine-detail-retention metric = corpus Recall (fraction of ground-truth COCO objects
mentioned). It is the direct informativeness complement to CHAIRi: an intervention
that suppressed hallucination merely by saying less would show Recall collapsing in
lock-step. Secondary detail signals reported: distinct objects/caption and caption
length. Metrics are computed on the image set common to all five conditions so the
curve is over identical images.

Delta-vs-baseline CIs (CHAIRi, Recall) come from a paired image-level bootstrap.

Usage:
    source activate sae
    python -m probe.analysis.chair_dose_response

Outputs:
  results/chair_dose_response_vilau.json
  results/chair_dose_response_vilau.md
"""

from __future__ import annotations

import contextlib
import json
import os
import random
from pathlib import Path

from probe.benchmarks.chair import CHAIR
from probe.analysis.chair_robustness import _per_image, _corpus_metrics, _ci

RESULTS = Path("results")
COCO = "probe/benchmarks/coco_annotations"
N_BOOT = 10000
SEED = 0

# alpha (attenuation strength) -> caption JSONL. Endpoints reuse the files.
CONDITIONS = [
    (0.00, "results/chair_captions_vilau_baseline.jsonl"),
    (0.25, "results/chair_captions_vilau_scalarL0_c0.75.jsonl"),
    (0.50, "results/chair_captions_vilau_scalarL0_c0.50.jsonl"),
    (0.75, "results/chair_captions_vilau_scalarL0_c0.25.jsonl"),
    (1.00, "results/chair_captions_vilau_L0.jsonl"),
]


def _delta_ci(ids: list[int], base: dict, other: dict, keys: list[str]) -> dict:
    rng = random.Random(SEED)
    n = len(ids)
    samp = {k: [] for k in keys}
    for _ in range(N_BOOT):
        res = [ids[rng.randrange(n)] for _ in range(n)]
        mb = _corpus_metrics(res, base)
        mo = _corpus_metrics(res, other)
        for k in keys:
            samp[k].append(mo[k] - mb[k])
    pb = _corpus_metrics(ids, base)
    po = _corpus_metrics(ids, other)
    out = {}
    for k in keys:
        lo, hi = _ci(samp[k])
        out[k] = {"delta": po[k] - pb[k], "ci": [lo, hi], "sig": (lo > 0) or (hi < 0)}
    return out


def main() -> None:
    missing = [f for _, f in CONDITIONS if not Path(f).exists()]
    if missing:
        raise SystemExit("Missing caption files (run scripts/run_chair_dose_response.sh):\n  "
                         + "\n  ".join(missing))

    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        ev = CHAIR(COCO)
        rows = {}
        for alpha, f in CONDITIONS:
            per_img, _ = _per_image(ev, f)
            rows[alpha] = per_img

    # image set common to every condition -> curve over identical images
    common = set.intersection(*[set(rows[a]) for a, _ in CONDITIONS])
    ids = sorted(common)

    metric_keys = ["CHAIRi", "CHAIRs", "Recall", "mentions", "distinct", "length", "has_obj_frac"]
    curve = []
    base_rows = rows[0.00]
    for alpha, f in CONDITIONS:
        m = _corpus_metrics(ids, rows[alpha])
        entry = {"alpha": alpha, "coeff": round(1 - alpha, 2),
                 "captions": os.path.basename(f), **{k: m[k] for k in metric_keys}}
        if alpha > 0:
            entry["delta_vs_baseline"] = _delta_ci(
                ids, base_rows, rows[alpha], ["CHAIRi", "Recall"])
        curve.append(entry)

    report = {"n_images": len(ids), "n_boot": N_BOOT, "seed": SEED,
              "detail_retention_metric": "Recall (fraction of GT COCO objects mentioned)",
              "curve": curve}
    (RESULTS / "chair_dose_response_vilau.json").write_text(json.dumps(report, indent=2))

    L: list[str] = []
    L.append("# L0-attenuation dose-response on CHAIR — VILA-U\n")
    L.append(f"n={len(ids)} COCO val2014 images common to all five conditions "
             f"(seed {SEED}). alpha = attenuation strength (1 = full L0 knockout, the "
             "reported intervention). Fine-detail-retention metric = **Recall** "
             "(fraction of ground-truth COCO objects mentioned); higher = more detail "
             "preserved. Δ vs baseline (α=0) with paired image-level bootstrap 95% CI "
             f"({N_BOOT} resamples).\n")

    L.append("| α (ablation) | coeff | CHAIRi ↓ | Recall ↑ (detail) | distinct obj/cap | "
             "avg len | ΔCHAIRi [95% CI] | ΔRecall [95% CI] |")
    L.append("|---|---|---|---|---|---|---|---|")
    for e in curve:
        tag = "  ← reported" if e["alpha"] == 1.00 else (
            "  (baseline)" if e["alpha"] == 0.00 else "")
        if "delta_vs_baseline" in e:
            dc = e["delta_vs_baseline"]["CHAIRi"]
            dr = e["delta_vs_baseline"]["Recall"]
            dci = (f"{dc['delta']:+.2f} [{dc['ci'][0]:+.2f}, {dc['ci'][1]:+.2f}]"
                   f"{'*' if dc['sig'] else ''}")
            dri = (f"{dr['delta']:+.2f} [{dr['ci'][0]:+.2f}, {dr['ci'][1]:+.2f}]"
                   f"{'*' if dr['sig'] else ''}")
        else:
            dci = dri = "—"
        L.append(f"| {e['alpha']:.2f}{tag} | {e['coeff']:.2f} | {e['CHAIRi']:.2f} | "
                 f"{e['Recall']:.2f} | {e['distinct']:.2f} | {e['length']:.1f} | "
                 f"{dci} | {dri} |")

    L.append("\n\\* Δ 95% CI excludes 0. CHAIRi/CHAIRs/Recall/has-obj in %, lengths in words.\n")
    (RESULTS / "chair_dose_response_vilau.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nWrote results/chair_dose_response_vilau.{json,md}")


if __name__ == "__main__":
    main()
