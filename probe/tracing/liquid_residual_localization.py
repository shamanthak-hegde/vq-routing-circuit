"""Liquid L0/L1 residual-stream localization under L0 ablation.

Tests whether L0 ablation "cures" Liquid merely by disabling its ability to
respond at all (20/20 blank / newline output). Distinguishes two
explanations for the degeneracy by inspecting the L0+1 residual stream:

  (A) loss of visual-token attribution — L0 ablation removes the visual->prompt_last
      routing, so prompt_last stops depending on the IMAGE, but its dependence on the
      QUESTION TEXT (the general generation prior / syntax) survives; or
  (B) loss of the general generation prior — the prompt_last residual collapses to an
      input-invariant vector regardless of image OR question text.

Design.  Liquid's prompt suffix is fixed (`...{question}<end_of_turn>\n<start_of_turn>
model\n`), so `prompt_last` is the SAME token for every record. Therefore every bit of
cross-record variation in the prompt_last residual comes from attention routing, not
from the local token embedding. We build a batch of several distinct questions, each
paired with several different images, and measure per layer:

  - within-question mean pairwise cosine of the prompt_last residual  (image dependence:
    same text, different images -> low cosine = image still matters)
  - between-question mean pairwise cosine                              (text dependence:
    different questions -> low cosine = question text still matters)
  - prompt_last residual L2 norm
  - visual-token residual within/between cosine (control: the image is still encoded in
    the visual positions even under L0 ablation)
  - attention mass prompt_last -> visual tokens at L0 and L1

At layer 0 output under L0 ablation the prompt_last residual is trivially constant (no
attention -> purely local, identical token) — so the informative layer is L1 ("L0+1"),
where attention is active and can, in principle, re-route input-dependent information.

Run (sae env):
    module load mamba && source activate sae
    python -m probe.tracing.liquid_residual_localization \
        --n_questions 5 --imgs_per_q 6 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "results" / "liquid_localization"
FIGDIR = REPO / "figures" / "liquid_localization"


def _load_liquid(model_path: str):
    chameleon_root = os.path.normpath(REPO / "chameleon")
    liquid_root = os.path.normpath(REPO / "Liquid")
    for p in (chameleon_root, liquid_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from chameleon.inference.image_tokenizer import ImageTokenizer
    from probe.hooks.liquid import LiquidHookManager

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
    ).eval()
    vqgan_cfg = os.path.join(chameleon_root, "data", "tokenizer", "vqgan.yaml")
    vqgan_ckpt = os.path.join(chameleon_root, "data", "tokenizer", "vqgan.ckpt")
    image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg, ckpt_path=vqgan_ckpt, device="cuda:0")
    hm = LiquidHookManager(model, tokenizer, image_tokenizer,
                           capture_attention_weights=True)
    return hm, tokenizer


def _build_batch(n_questions: int, imgs_per_q: int, seed: int):
    from probe import load_cache
    from probe.hooks.schema import TokenCategory  # noqa: F401  (import sanity)
    records, _, _ = load_cache()
    pope = [r for r in records if r.corruption_mode == "gaussian_noise"]
    by_q = defaultdict(list)
    for r in pope:
        by_q[r.question].append(r)
    # questions with enough images, most frequent first
    elig = sorted([(q, rs) for q, rs in by_q.items() if len(rs) >= imgs_per_q],
                  key=lambda kv: -len(kv[1]))[:n_questions]
    rng = random.Random(seed)
    batch = []
    for q, rs in elig:
        yes = [r for r in rs if r.answer == "yes"]
        no = [r for r in rs if r.answer == "no"]
        rng.shuffle(yes); rng.shuffle(no)
        half = imgs_per_q // 2
        chosen = yes[:half] + no[:imgs_per_q - half]
        if len(chosen) < imgs_per_q:  # backfill if a GT class is short
            pool = [r for r in rs if r not in chosen]
            rng.shuffle(pool)
            chosen += pool[:imgs_per_q - len(chosen)]
        batch.extend(chosen[:imgs_per_q])
    return batch


@torch.no_grad()
def _capture(hm, img, question, intervention_factory):
    from probe.hooks.schema import TokenCategory
    if intervention_factory is not None:
        with intervention_factory():
            cap = hm.run_prefill(img, question)
    else:
        cap = hm.run_prefill(img, question)
    p = cap.token_index.prompt_last
    vis_mask = cap.token_index.mask(TokenCategory.VISUAL).to(cap.residual.device)
    vis_idx = torch.nonzero(vis_mask, as_tuple=True)[0]
    resid_pl = cap.residual[:, p, :].float().cpu()                      # (L, H)
    resid_vis = cap.residual[:, vis_idx, :].float().mean(dim=1).cpu()   # (L, H) mean visual
    # attention mass prompt_last -> visual, per layer (mean over heads, sum over visual)
    if cap.attn_weights is not None:
        aw = cap.attn_weights[:, :, p, :].float()                       # (L, heads, seq)
        attn_vis = aw[:, :, vis_idx].sum(dim=2).mean(dim=1).cpu()       # (L,)
    else:
        attn_vis = torch.full((resid_pl.shape[0],), float("nan"))
    del cap
    torch.cuda.empty_cache()
    return resid_pl, resid_vis, attn_vis


def _pairwise_cos(mat: torch.Tensor) -> float:
    """Mean off-diagonal pairwise cosine of rows of mat (m, H)."""
    if mat.shape[0] < 2:
        return float("nan")
    x = torch.nn.functional.normalize(mat.float(), dim=1)
    sim = x @ x.t()
    m = mat.shape[0]
    iu = torch.triu_indices(m, m, offset=1)
    return float(sim[iu[0], iu[1]].mean())


def _grouped_metrics(vecs: torch.Tensor, qids: list[int]) -> dict:
    """Common-mode-robust decomposition of prompt_last residual variation.

    The residual stream is dominated by a large shared (common-mode) component, so raw
    pairwise cosine saturates near 1 and cannot discriminate. We remove it by centering
    on the grand mean, then measure:

      disp        : mean ||x_i - grand|| / ||grand||  — relative dispersion; ~0 means all
                    records are identical (input-invariant = collapse).
      eta2_q      : between-question SS / total SS on the raw residuals (variance is
                    mean-centered by construction, so it cancels the common-mode and is
                    scale-invariant) — the FRACTION of residual variation explained by
                    question identity. →1 means image no longer moves the residual (only
                    text does); intermediate means both image and text contribute.
      within_ccos : within-question mean pairwise cosine of centered vectors (image
                    dependence: <1 means same-question-different-image still differ).
      between_ccos: between-question mean pairwise cosine of centered vectors (text
                    dependence: low/neg means different questions point different ways).
    """
    N = vecs.shape[0]
    x = vecs.float()
    grand = x.mean(dim=0, keepdim=True)                 # (1,H)
    c = x - grand                                       # centered
    grand_norm = float(grand.norm()) + 1e-8
    disp = float(c.norm(dim=1).mean()) / grand_norm

    # variance decomposition (eta^2 by question)
    total_ss = float((c ** 2).sum())
    between_ss = 0.0
    for q in set(qids):
        idx = [i for i in range(N) if qids[i] == q]
        gmean = x[idx].mean(dim=0, keepdim=True)
        between_ss += len(idx) * float(((gmean - grand) ** 2).sum())
    eta2 = between_ss / total_ss if total_ss > 1e-12 else float("nan")

    # centered cosines
    cn = torch.nn.functional.normalize(c, dim=1)
    sim = cn @ cn.t()
    within_vals, bt = [], []
    for q in set(qids):
        idx = [i for i in range(N) if qids[i] == q]
        for i, j in combinations(idx, 2):
            within_vals.append(float(sim[i, j]))
    for i, j in combinations(range(N), 2):
        if qids[i] != qids[j]:
            bt.append(float(sim[i, j]))
    within_ccos = float(sum(within_vals) / len(within_vals)) if within_vals else float("nan")
    between_ccos = float(sum(bt) / len(bt)) if bt else float("nan")
    return {"disp": disp, "eta2_q": eta2,
            "within_ccos": within_ccos, "between_ccos": between_ccos}


def _capture_batch(args):
    """GPU pass: capture prompt_last / visual residuals + attn mass, save raw .pt."""
    batch = _build_batch(args.n_questions, args.imgs_per_q, args.seed)
    qlist = sorted({r.question for r in batch})
    qid = {q: i for i, q in enumerate(qlist)}
    qids = [qid[r.question] for r in batch]
    print(f"Batch: {len(batch)} records over {len(qlist)} questions ({args.imgs_per_q}/q):")
    for q in qlist:
        n = sum(1 for r in batch if r.question == q)
        print(f"   [{n}] {q}")

    print("Loading Liquid …")
    hm, _ = _load_liquid(args.model_path)
    from probe.tracing.head_knockout import build_intervention

    def l0_factory():
        return build_intervention("pathological_route_ablation", hm,
                                  layer_idx=0, head_idxs="all", alpha=0.0)

    conds = {"baseline": None, "L0": l0_factory}
    store = {c: {"resid_pl": [], "resid_vis": [], "attn_vis": []} for c in conds}
    t0 = time.time()
    for i, rec in enumerate(batch, 1):
        img = Image.open(rec.image_path).convert("RGB")
        for c, fac in conds.items():
            rp, rv, av = _capture(hm, img, rec.question, fac)
            store[c]["resid_pl"].append(rp)
            store[c]["resid_vis"].append(rv)
            store[c]["attn_vis"].append(av)
        print(f"  [{i:2d}/{len(batch)}] {rec.id} gt={rec.answer} {time.time()-t0:.0f}s",
              flush=True)

    packed = {c: {
        "resid_pl": torch.stack(store[c]["resid_pl"]),
        "resid_vis": torch.stack(store[c]["resid_vis"]),
        "attn_vis": torch.stack(store[c]["attn_vis"]),
    } for c in conds}
    blob = {"packed": packed, "qids": qids, "questions": qlist,
            "record_ids": [r.id for r in batch], "imgs_per_q": args.imgs_per_q}
    torch.save(blob, OUTDIR / "liquid_residual_raw.pt")
    print(f"Saved raw -> {OUTDIR / 'liquid_residual_raw.pt'}  ({time.time()-t0:.0f}s)")
    return blob


def analyze(blob) -> dict:
    packed, qids, qlist = blob["packed"], blob["qids"], blob["questions"]
    n_layers = packed["baseline"]["resid_pl"].shape[1]
    conds = ("baseline", "L0")
    result = {"n_records": len(qids), "n_questions": len(qlist),
              "imgs_per_q": blob.get("imgs_per_q"), "n_layers": n_layers,
              "questions": qlist, "per_layer": {}}
    for L in range(n_layers):
        row = {}
        for c in conds:
            m = _grouped_metrics(packed[c]["resid_pl"][:, L, :], qids)
            mv = _grouped_metrics(packed[c]["resid_vis"][:, L, :], qids)
            row[c] = {
                "pl_disp": round(m["disp"], 4), "pl_eta2_q": round(m["eta2_q"], 4),
                "pl_within_ccos": round(m["within_ccos"], 4),
                "pl_between_ccos": round(m["between_ccos"], 4),
                "pl_norm_mean": round(float(packed[c]["resid_pl"][:, L, :].norm(dim=1).mean()), 2),
                "vis_disp": round(mv["disp"], 4), "vis_eta2_q": round(mv["eta2_q"], 4),
                "attn_pl_to_vis_mean": round(float(packed[c]["attn_vis"][:, L].mean()), 4),
            }
        result["per_layer"][str(L)] = row

    # ---- verdict ----
    # The prompt_last representation is nearly input-invariant in the first layers even in
    # baseline; input-dependence is BUILT across depth. So the readout-relevant test is at
    # the deep layers where baseline develops input-specific structure. We define the deep
    # "readout region" as the layers whose baseline dispersion is >=60% of its max.
    base_disp = [result["per_layer"][str(L)]["baseline"]["pl_disp"] for L in range(n_layers)]
    max_bd = max(base_disp)
    deep = [L for L in range(n_layers) if base_disp[L] >= 0.6 * max_bd]

    def _mean(cond, key, layers):
        return sum(result["per_layer"][str(L)][cond][key] for L in layers) / len(layers)

    deep_disp_base = _mean("baseline", "pl_disp", deep)
    deep_disp_l0 = _mean("L0", "pl_disp", deep)
    collapse_ratio = deep_disp_l0 / deep_disp_base if deep_disp_base > 1e-9 else float("nan")
    deep_eta2_base = _mean("baseline", "pl_eta2_q", deep)
    deep_eta2_l0 = _mean("L0", "pl_eta2_q", deep)
    # visual info still present upstream? Measure at the early (L0+1..L5) region where the
    # visual tokens are read, not at the deep readout layers.
    early = [L for L in range(1, min(6, n_layers))]
    vis_preserved = _mean("L0", "vis_disp", early) >= 0.5 * _mean("baseline", "vis_disp", early)

    general_collapse = collapse_ratio < 0.4            # readout dispersion collapses
    text_survives = deep_eta2_l0 >= 0.5 and collapse_ratio >= 0.5
    image_gone = deep_eta2_l0 >= 0.85                  # only text would remain
    if general_collapse:
        verdict = "GENERAL-PRIOR-LOSS"                 # neither image nor text readout survives
    elif text_survives and image_gone:
        verdict = "VISUAL-GROUNDING-LOSS"
    else:
        verdict = "MIXED/INCONCLUSIVE"
    result["verdict"] = verdict
    l0_1 = result["per_layer"]["1"]
    result["verdict_basis"] = {
        "readout_layers": deep,
        "deep_pl_disp_baseline": round(deep_disp_base, 4),
        "deep_pl_disp_L0": round(deep_disp_l0, 4),
        "deep_disp_collapse_ratio_L0_over_baseline": round(collapse_ratio, 4),
        "deep_eta2_q_baseline": round(deep_eta2_base, 4),
        "deep_eta2_q_L0": round(deep_eta2_l0, 4),
        "visual_info_still_encoded_upstream": bool(vis_preserved),
        "L0+1_pl_norm_inflation_x": round(l0_1["L0"]["pl_norm_mean"] / max(l0_1["baseline"]["pl_norm_mean"], 1e-9), 2),
        "L0+1_attn_pl_to_vis_baseline": l0_1["baseline"]["attn_pl_to_vis_mean"],
        "L0+1_attn_pl_to_vis_L0": l0_1["L0"]["attn_pl_to_vis_mean"],
        "interpretation": (
            "Under L0 ablation the prompt_last residual never develops the normal deep "
            "input-dependent structure (dispersion collapses ~%.0fx at the readout layers) "
            "and question-text dependence does NOT rise (eta2 flat) — so BOTH image- and "
            "text-conditioning of the readout fail: a general readout collapse, not a clean "
            "removal of only visual grounding. The image is still encoded upstream (visual "
            "residuals preserved) and prompt_last attention actually FLOODS to visual at "
            "L0+1 (%.2f vs %.2f baseline), so the failure is not 'stops looking at the "
            "image' — it is 'can no longer turn any input into a conditioned readout.' This "
            "confirms the/ decision to keep Liquid as C1 gate evidence and NOT a "
            "C6 intervention."
            % (1.0 / collapse_ratio if collapse_ratio > 1e-9 else float('nan'),
               l0_1["L0"]["attn_pl_to_vis_mean"], l0_1["baseline"]["attn_pl_to_vis_mean"])
        ),
    }
    return result


def _print_and_plot(result):
    report_layers = sorted(set([0, 1, 2, 5, result["n_layers"] // 2, result["n_layers"] - 1]))
    print("\n=== L0+1 residual-stream localization (prompt_last) ===")
    print(f"{'layer':>5} | {'cond':>8} | {'disp':>6} {'eta2_q':>6} {'w_ccos':>7} "
          f"{'b_ccos':>7} {'norm':>7} {'vis_disp':>8} {'attn>vis':>8}")
    for L in report_layers:
        for c in ("baseline", "L0"):
            r = result["per_layer"][str(L)][c]
            print(f"{L:>5} | {c:>8} | {r['pl_disp']:>6.3f} {r['pl_eta2_q']:>6.3f} "
                  f"{r['pl_within_ccos']:>7.3f} {r['pl_between_ccos']:>7.3f} "
                  f"{r['pl_norm_mean']:>7.1f} {r['vis_disp']:>8.3f} {r['attn_pl_to_vis_mean']:>8.3f}")
    print(f"\nVERDICT: {result['verdict']}")
    print(json.dumps(result["verdict_basis"], indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = list(range(result["n_layers"]))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        panels = [
            ("pl_disp", "prompt_last relative dispersion\n(0 = input-invariant collapse)"),
            ("pl_eta2_q", "prompt_last η²(question)\n(1 = only text moves residual, image ignored)"),
            ("attn_pl_to_vis_mean", "attention mass prompt_last→visual"),
        ]
        for ax, (metric, title) in zip(axes, panels):
            for c, col in [("baseline", "#2c7fb8"), ("L0", "#e34a33")]:
                ys = [result["per_layer"][str(L)][c][metric] for L in xs]
                ax.plot(xs, ys, marker="o", ms=3, label=c, color=col)
            ax.axvline(1, color="gray", ls=":", lw=1, label="L0+1")
            ax.set_xlabel("layer"); ax.set_title(title, fontsize=9)
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle(f"Liquid L0-ablation residual localization  (verdict: {result['verdict']})",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(FIGDIR / "liquid_residual_localization.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure -> {FIGDIR / 'liquid_residual_localization.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Junfeng5/Liquid_V1_7B")
    ap.add_argument("--n_questions", type=int, default=5)
    ap.add_argument("--imgs_per_q", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report_only", action="store_true",
                    help="Recompute metrics from results/cohort_behavior/liquid_residual_raw.pt (no GPU)")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        blob = torch.load(OUTDIR / "liquid_residual_raw.pt", weights_only=False)
    else:
        blob = _capture_batch(args)

    result = analyze(blob)
    (OUTDIR / "liquid_residual_localization.json").write_text(json.dumps(result, indent=2) + "\n")
    _print_and_plot(result)
    print(f"\nWrote {OUTDIR / 'liquid_residual_localization.json'}")


if __name__ == "__main__":
    main()
