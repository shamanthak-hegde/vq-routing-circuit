"""
Codebook probe for VILA-U >1.0 restoration records.

Tests the mechanistic hypothesis behind restoration scores > 1.0 on VILA-U POPE:

  Hypothesis: VQ tokenizer creates an off-manifold effect.
  The mm_projector output lives on a tight manifold (finite codebook → finite
  projected vectors).  Adding Gaussian noise moves embeddings OFF this manifold.
  Models trained only on on-manifold tokens respond to off-manifold tokens with
  confidently-wrong outputs — not confused ones.

Two sub-tests
─────────────
A. VQ code sensitivity (image-space noise):
   Add Gaussian noise to the processed image tensor (before the vision encoder).
   Compare clean vs noisy VQ codes.  Key questions:
   - What fraction of spatial tokens change their top code?
   - Are the noisy codes consistent across noise seeds (same wrong code) or random?

B. Off-manifold effect (post-projector noise, matching sweep corruption):
   For each visual token, after adding σ=1.0 noise to the mm_projector output,
   how does the noisy embedding relate to the clean embeddings of OTHER records?
   If noisy[i] is closer to wrong_record[j]'s clean embedding than to clean[i],
   that is the substitution story in LM space.

Usage
─────
    conda activate vila-u
    python -m probe.tracing.codebook_probe \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --sweep    results/sweep_vilau.jsonl \\
        --accuracy results/accuracy_vilau.jsonl \\
        --n_records 20 \\
        --sigma 1.0 \\
        --out results/codebook_probe_vilau.jsonl

    # Report only (no model needed):
    python -m probe.tracing.codebook_probe --report_only \\
        --out results/codebook_probe_vilau.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_high_restoration_ids(sweep_path: str, accuracy_path: str, n: int, seed: int = 0) -> list[str]:
    """Return up to n record_ids where restoration score > 1.0, non-WARN, and correct."""
    # Load WARN flags and max restoration score per record
    sweep: dict[str, dict] = {}
    with open(sweep_path) as f:
        for line in f:
            row = json.loads(line.strip())
            rid = row["record_id"]
            if row.get("source") != "pope":
                continue
            is_warn = row["logit_clean"] <= row["logit_corrupt"]
            if rid not in sweep:
                sweep[rid] = {"is_warn": is_warn, "max_score": row["score"]}
            else:
                sweep[rid]["max_score"] = max(sweep[rid]["max_score"], row["score"])
                if is_warn:
                    sweep[rid]["is_warn"] = True

    # Load correct flags
    correct_ids: set[str] = set()
    with open(accuracy_path) as f:
        for line in f:
            row = json.loads(line.strip())
            if row.get("correct"):
                correct_ids.add(row["record_id"])

    candidates = [
        rid for rid, v in sweep.items()
        if v["max_score"] > 1.0 and not v["is_warn"] and rid in correct_ids
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]
    print(f"Found {len(candidates)} POPE records with max_score > 1.0, non-WARN, correct; "
          f"using {len(selected)}")
    return selected


def _load_all_pope_ids(sweep_path: str, accuracy_path: str) -> list[str]:
    """Return all non-WARN + correct POPE record_ids (for off-manifold baseline)."""
    warn_flags: dict[str, bool] = {}
    with open(sweep_path) as f:
        for line in f:
            row = json.loads(line.strip())
            if row.get("source") != "pope":
                continue
            rid = row["record_id"]
            if rid not in warn_flags:
                warn_flags[rid] = row["logit_clean"] <= row["logit_corrupt"]
    correct_ids: set[str] = set()
    with open(accuracy_path) as f:
        for line in f:
            row = json.loads(line.strip())
            if row.get("correct"):
                correct_ids.add(row["record_id"])
    return [rid for rid, w in warn_flags.items() if not w and rid in correct_ids]


def _load_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


# ── report ────────────────────────────────────────────────────────────────────

def print_report(rows: list[dict]) -> None:
    if not rows:
        print("No rows to report.")
        return

    sep = "=" * 68

    # Sub-test A: VQ code sensitivity
    a_rows = [r for r in rows if r["test"] == "A_vq_sensitivity"]
    if a_rows:
        n_seeds = max(r["noise_seed"] for r in a_rows) + 1
        print(f"\n{sep}")
        print("SUB-TEST A: VQ code sensitivity (image-space noise)")
        print(sep)
        for seed in range(n_seeds):
            seed_rows = [r for r in a_rows if r["noise_seed"] == seed]
            if not seed_rows:
                continue
            frac_changed = sum(r["frac_codes_changed"] for r in seed_rows) / len(seed_rows)
            print(f"  seed={seed}: mean frac_codes_changed = {frac_changed:.3f}  "
                  f"({len(seed_rows)} records)")

        # Consistency across seeds: for each record, do noisy codes agree across seeds?
        by_record: dict[str, list[dict]] = {}
        for r in a_rows:
            by_record.setdefault(r["record_id"], []).append(r)
        consistent_counts = []
        for rid, rrows in by_record.items():
            if len(rrows) < 2:
                continue
            # Compare dominant_noisy_code across seeds
            codes = [r.get("dominant_noisy_code") for r in rrows if r.get("dominant_noisy_code") is not None]
            if len(codes) >= 2:
                consistent_counts.append(len(set(codes)) == 1)
        if consistent_counts:
            pct = 100 * sum(consistent_counts) / len(consistent_counts)
            verdict = "CONSISTENTLY WRONG" if pct > 70 else "RANDOM"
            print(f"\n  Cross-seed code consistency: {pct:.0f}% of records → {verdict}")
            print(f"  (>70% same wrong code across seeds = information substitution)")
        print(sep)

    # Sub-test B: off-manifold effect
    b_rows = [r for r in rows if r["test"] == "B_off_manifold"]
    if b_rows:
        print(f"\n{sep}")
        print("SUB-TEST B: off-manifold effect (post-projector noise, sweep σ)")
        print(sep)
        n = len(b_rows)
        cos_self   = sum(r["cos_sim_noisy_to_clean_self"]  for r in b_rows) / n
        cos_other  = sum(r["cos_sim_noisy_to_nearest_other"] for r in b_rows) / n
        pct_closer = 100 * sum(1 for r in b_rows if r["noisy_closer_to_other"]) / n
        print(f"  Records:                              {n}")
        print(f"  Mean cos-sim(noisy, clean_self):      {cos_self:.4f}  (calibrated target: ~0.39)")
        print(f"  Mean cos-sim(noisy, nearest_other):   {cos_other:.4f}")
        print(f"  Pct where noisy closer to other rec:  {pct_closer:.1f}%")
        verdict = "SUBSTITUTION" if pct_closer > 20 else "DEGRADATION"
        print(f"\n  Verdict: {verdict}")
        print(f"  (>20% closer to wrong record = off-manifold substitution effect)")
        print(sep)


# ── main probe ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_probe(
    hm,
    target_records,
    all_records,
    sigma: float,
    n_noise_seeds: int,
    out_path: Path,
    sigma_img: float = 0.05,
) -> None:
    """Run both sub-tests and write JSONL rows.

    Parameters
    ----------
    hm            : VilaUHookManager
    target_records: list of ProbeRecord with max_score > 1.0
    all_records   : list of all non-WARN+correct POPE ProbeRecord (for off-manifold baseline)
    sigma         : post-projector noise level (matches sweep σ, default 1.0)
    n_noise_seeds : number of noise seeds for consistency check in sub-test A
    sigma_img     : image-space noise σ for sub-test A (relative to pixel value range)
    """
    # Access the RQ vision tower internals
    vision_tower = hm.model.vision_tower.vision_tower.rqvaesiglip
    quantizer    = vision_tower.quantizer

    fout = out_path.open("a")
    try:
        # Pre-compute clean mm_projector outputs for all records (needed for sub-test B)
        print("Pre-computing clean mm_projector embeddings for off-manifold baseline ...")
        clean_embeds: dict[str, torch.Tensor] = {}  # record_id → (n_vis, H) mean visual embed
        for rec in all_records:
            img = Image.open(rec.image_path).convert("RGB")
            input_ids, images_tensor, image_sizes = hm._build_prompt(img, rec.question)
            embeds, n_vis = hm._prepare_embeds(input_ids, images_tensor, image_sizes)
            sentinel_pos = int((input_ids[0] == hm._IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)[0, 0])
            vis_embeds = embeds[0, sentinel_pos:sentinel_pos + n_vis, :].float()
            clean_embeds[rec.id] = vis_embeds  # (n_vis, H)
        print(f"  Cached {len(clean_embeds)} records\n")

        for rec in target_records:
            img = Image.open(rec.image_path).convert("RGB")
            input_ids, images_tensor, image_sizes = hm._build_prompt(img, rec.question)

            # ── Sub-test A: VQ code sensitivity (image-space noise) ────────────
            print(f"[A] {rec.id}")
            # Clean codes
            clean_code, _ = vision_tower.encode_image(images_tensor)
            # clean_code: (1, h, w, d) — RQ codes

            for seed in range(n_noise_seeds):
                g = torch.Generator(device=images_tensor.device)
                g.manual_seed(seed)
                noise_img = torch.randn(images_tensor.shape, generator=g,
                                        device=images_tensor.device, dtype=torch.float32) * sigma_img
                noisy_images_tensor = images_tensor + noise_img.to(images_tensor.dtype)
                noisy_code, _ = vision_tower.encode_image(noisy_images_tensor)

                # Compare codes: shape (1, h, w, d)
                changed = (clean_code != noisy_code).float()
                frac_changed = changed.mean().item()

                # Dominant noisy code = most frequent changed-to index (flat over spatial+depth)
                noisy_flat = noisy_code[0].reshape(-1).tolist()
                clean_flat = clean_code[0].reshape(-1).tolist()
                changed_noisy = [n for c, n in zip(clean_flat, noisy_flat) if c != n]
                dominant_noisy = max(set(changed_noisy), key=changed_noisy.count) if changed_noisy else None

                row = {
                    "test": "A_vq_sensitivity",
                    "record_id": rec.id,
                    "noise_seed": seed,
                    "sigma_img": sigma_img,
                    "frac_codes_changed": round(frac_changed, 4),
                    "dominant_noisy_code": dominant_noisy,
                    "n_spatial_tokens": clean_code.shape[1] * clean_code.shape[2],
                    "rq_depth": clean_code.shape[3],
                }
                fout.write(json.dumps(row) + "\n")
                print(f"    seed={seed}: frac_changed={frac_changed:.3f}  dom_noisy_code={dominant_noisy}")

            fout.flush()

            # ── Sub-test B: off-manifold (post-projector noise) ────────────────
            print(f"[B] {rec.id}")
            embeds, n_vis = hm._prepare_embeds(input_ids, images_tensor, image_sizes)
            sentinel_pos = int((input_ids[0] == hm._IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)[0, 0])
            vis_clean = embeds[0, sentinel_pos:sentinel_pos + n_vis, :].float()  # (n_vis, H)

            # Add post-projector noise (matches sweep corruption)
            g = torch.Generator(device=vis_clean.device)
            g.manual_seed(0)
            noise_proj = torch.randn(vis_clean.shape, generator=g,
                                     device=vis_clean.device, dtype=torch.float32) * sigma
            vis_noisy = vis_clean + noise_proj  # (n_vis, H)

            # Per-token: cos-sim(noisy, clean_self)
            cos_self_per_tok = F.cosine_similarity(vis_noisy, vis_clean, dim=-1)  # (n_vis,)
            cos_self = cos_self_per_tok.mean().item()

            # Per-token: cos-sim(noisy, nearest_other_record's clean embed)
            # Use mean embedding per record to keep computation tractable
            other_ids  = [rid for rid in clean_embeds if rid != rec.id]
            if other_ids:
                # (n_other, H) — mean visual embedding per record
                other_means = torch.stack([clean_embeds[oid].mean(0) for oid in other_ids])
                noisy_mean  = vis_noisy.mean(0, keepdim=True)   # (1, H)
                clean_mean  = vis_clean.mean(0, keepdim=True)   # (1, H)
                cos_to_others = F.cosine_similarity(noisy_mean, other_means, dim=-1)  # (n_other,)
                cos_other_max = cos_to_others.max().item()
                nearest_other_id = other_ids[int(cos_to_others.argmax())]
                noisy_closer_to_other = bool(cos_other_max > cos_self)
            else:
                cos_other_max = float("nan")
                nearest_other_id = None
                noisy_closer_to_other = False

            row = {
                "test": "B_off_manifold",
                "record_id": rec.id,
                "sigma": sigma,
                "cos_sim_noisy_to_clean_self":    round(cos_self, 4),
                "cos_sim_noisy_to_nearest_other": round(cos_other_max, 4),
                "nearest_other_record_id":        nearest_other_id,
                "noisy_closer_to_other":          noisy_closer_to_other,
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            print(f"    cos_self={cos_self:.4f}  cos_other={cos_other_max:.4f}  "
                  f"closer_to_other={noisy_closer_to_other}\n")

    finally:
        fout.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Codebook probe for VILA-U >1.0 restoration")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--sweep",      default="results/sweep_vilau.jsonl")
    parser.add_argument("--accuracy",   default="results/accuracy_vilau.jsonl")
    parser.add_argument("--out",        default="results/codebook_probe_vilau.jsonl")
    parser.add_argument("--n_records",  type=int, default=20)
    parser.add_argument("--sigma",      type=float, default=1.0,
                        help="Post-projector noise level (match sweep σ)")
    parser.add_argument("--sigma_img",  type=float, default=0.05,
                        help="Image-space noise σ for sub-test A")
    parser.add_argument("--n_seeds",    type=int, default=3,
                        help="Noise seeds for code-consistency check (sub-test A)")
    parser.add_argument("--report_only", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        rows = _load_rows(out_path)
        print_report(rows)
        return

    if not args.model_path:
        parser.error("--model_path is required unless --report_only")

    # Load model
    _vilau = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "vila-u"))
    if _vilau not in sys.path:
        sys.path.insert(0, _vilau)
    from vila_u.model.builder import load_pretrained_model
    from probe.hooks.vilau import VilaUHookManager
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, attn_implementation="eager"
    )
    model.eval()
    hm = VilaUHookManager(model, tokenizer, image_processor)

    # Load probe records
    from probe import load_cache, resolve_answer_token_ids
    records, _, _ = load_cache()
    resolve_answer_token_ids(records, tokenizer)

    target_ids = set(_load_high_restoration_ids(
        args.sweep, args.accuracy, args.n_records
    ))
    all_usable_ids = set(_load_all_pope_ids(args.sweep, args.accuracy))

    target_records = [r for r in records if r.id in target_ids and r.source == "pope"]
    all_records    = [r for r in records if r.id in all_usable_ids and r.source == "pope"]

    print(f"Target (>1.0) records: {len(target_records)}")
    print(f"All usable POPE records (baseline pool): {len(all_records)}\n")

    run_probe(hm, target_records, all_records,
              sigma=args.sigma, n_noise_seeds=args.n_seeds,
              sigma_img=args.sigma_img, out_path=out_path)

    rows = _load_rows(out_path)
    print_report(rows)


if __name__ == "__main__":
    main()
