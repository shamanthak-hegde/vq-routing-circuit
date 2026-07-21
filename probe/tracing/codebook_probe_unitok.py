"""
Sub-test B (off-manifold substitution) for UniTok —.

Tests whether UniTok's post-projector embeddings share VILA-U's off-manifold substitution
property: after adding calibrated Gaussian noise (σ=0.2, UniTok's σ_cal), does the noisy
embedding land closer to a different record's clean embedding than to its own?

VILA-U result: 100% of 20 POPE records showed substitution.
UniTok hypothesis: richer projector (TokenEmbedder) → denser manifold → lower rate.

Sub-test A (VQ code sensitivity) is VILA-U-specific (RQ-VAE) and is intentionally omitted.
The correct filter is omitted because accuracy_unitok.jsonl does not exist; correct_filter_applied=false
is written on every output row so the difference is auditable.

Usage
─────
    conda activate unitok
    python -m probe.tracing.codebook_probe_unitok \\
        --model_path FoundationVision/unitok_mllm \\
        --tokenizer_path /path/to/UniTok/checkpoint/unitok_tokenizer.pth \\
        --sweep    results/sweep_unitok.jsonl \\
        --n_records 20 \\
        --sigma 0.2 \\
        --out results/codebook_probe_unitok.jsonl

    # Report only (no model needed):
    python -m probe.tracing.codebook_probe_unitok --report_only \\
        --out results/codebook_probe_unitok.jsonl
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

def _load_high_restoration_ids(sweep_path: str, n: int, seed: int = 0) -> list[str]:
    """Return up to n POPE record_ids with max_score > 1.0 and non-WARN."""
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

    candidates = [
        rid for rid, v in sweep.items()
        if v["max_score"] > 1.0 and not v["is_warn"]
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]
    print(f"Found {len(candidates)} POPE records with max_score > 1.0 and non-WARN; "
          f"using {len(selected)}")
    return selected


def _load_all_non_warn_pope_ids(sweep_path: str) -> list[str]:
    """Return all non-WARN POPE record_ids (baseline pool for off-manifold comparison)."""
    warn_flags: dict[str, bool] = {}
    with open(sweep_path) as f:
        for line in f:
            row = json.loads(line.strip())
            if row.get("source") != "pope":
                continue
            rid = row["record_id"]
            if rid not in warn_flags:
                warn_flags[rid] = row["logit_clean"] <= row["logit_corrupt"]
    result = [rid for rid, w in warn_flags.items() if not w]
    print(f"Non-WARN POPE pool: {len(result)} records")
    return result


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
    b_rows = [r for r in rows if r["test"] == "B_off_manifold"]
    if not b_rows:
        print("No Sub-test B rows found.")
        return

    sep = "=" * 68
    print(f"\n{sep}")
    print("SUB-TEST B: off-manifold effect (post-projector noise, σ=σ_cal)")
    print("Backend: UniTok  |  correct_filter_applied: False")
    print("Comparison: VILA-U (cos_self=0.38, cos_other=0.87, pct=100%)")
    print(sep)

    n = len(b_rows)
    cos_self  = sum(r["cos_sim_noisy_to_clean_self"]    for r in b_rows) / n
    cos_other = sum(r["cos_sim_noisy_to_nearest_other"] for r in b_rows) / n
    pct_closer = 100 * sum(1 for r in b_rows if r["noisy_closer_to_other"]) / n

    # Distribution of cos_other (deciles)
    cos_other_vals = sorted(r["cos_sim_noisy_to_nearest_other"] for r in b_rows)
    p10 = cos_other_vals[max(0, int(0.10 * n) - 1)]
    p50 = cos_other_vals[int(0.50 * n) - 1]
    p90 = cos_other_vals[min(n - 1, int(0.90 * n))]

    sigma = b_rows[0].get("sigma", "?")
    print(f"  Records:                              {n}")
    print(f"  Noise level σ:                        {sigma}")
    print(f"  Mean cos-sim(noisy, clean_self):      {cos_self:.4f}  (σ_cal target: ~0.66)")
    print(f"  Mean cos-sim(noisy, nearest_other):   {cos_other:.4f}")
    print(f"  cos_other distribution p10/p50/p90:   {p10:.4f} / {p50:.4f} / {p90:.4f}")
    print(f"  Pct where noisy closer to other rec:  {pct_closer:.1f}%")

    if pct_closer < 20:
        verdict = "DEGRADATION  — manifold-richness confirmed; noise stays near self"
    elif pct_closer < 50:
        verdict = "MIXED        — manifold-richness plausible; some substitution present"
    elif pct_closer < 80:
        verdict = "PARTIAL      — geometry argument needs more nuance"
    else:
        verdict = "SUBSTITUTION — contradicts hypothesis; revisit thesis"

    print(f"\n  Verdict: {verdict}")
    print(f"  (>80% closer to wrong record = off-manifold substitution like VILA-U)")
    print(sep)


# ── main probe ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_probe(
    hm,
    target_records,
    all_records,
    sigma: float,
    out_path: Path,
) -> None:
    """Run Sub-test B and write JSONL rows.

    Parameters
    ----------
    hm             : UniTokHookManager
    target_records : ProbeRecords with max_score > 1.0 (n=20)
    all_records    : all non-WARN POPE ProbeRecords (baseline pool for off-manifold cos-sim)
    sigma          : post-projector noise level (UniTok σ_cal = 0.2)
    out_path       : output JSONL path
    """
    fout = out_path.open("a")
    try:
        print("Pre-computing clean multi_embedder outputs for off-manifold baseline ...")
        clean_embeds: dict[str, torch.Tensor] = {}  # record_id → (n_vis, H) on CPU
        for i, rec in enumerate(all_records):
            img = Image.open(rec.image_path).convert("RGB")
            input_ids, vq_code, image_sizes = hm._build_prompt(img, rec.question)
            embeds, n_vis = hm._prepare_embeds(input_ids, vq_code, image_sizes)
            assert n_vis == 256, f"Expected 256 visual tokens, got {n_vis}"
            assert (input_ids[0] == hm._IMAGE_TOKEN_INDEX).sum() == 1, \
                f"Expected exactly 1 IMAGE_TOKEN_INDEX sentinel in input_ids"
            sentinel_pos = int(
                (input_ids[0] == hm._IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)[0, 0]
            )
            vis_embeds = embeds[0, sentinel_pos:sentinel_pos + n_vis, :].float().cpu()
            clean_embeds[rec.id] = vis_embeds  # (256, H) on CPU
            if (i + 1) % 20 == 0:
                print(f"  Cached {i+1}/{len(all_records)} ...")
        print(f"  Cached {len(clean_embeds)} records\n")

        for rec in target_records:
            img = Image.open(rec.image_path).convert("RGB")
            input_ids, vq_code, image_sizes = hm._build_prompt(img, rec.question)

            # ── Sub-test B: off-manifold (post-projector noise) ──────────────
            print(f"[B] {rec.id}")
            embeds, n_vis = hm._prepare_embeds(input_ids, vq_code, image_sizes)
            sentinel_pos = int(
                (input_ids[0] == hm._IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)[0, 0]
            )
            vis_clean = embeds[0, sentinel_pos:sentinel_pos + n_vis, :].float()  # (256, H) on GPU

            g = torch.Generator(device=vis_clean.device)
            g.manual_seed(0)
            noise_proj = torch.randn(vis_clean.shape, generator=g,
                                     device=vis_clean.device, dtype=torch.float32) * sigma
            vis_noisy = vis_clean + noise_proj  # (256, H)

            cos_self_per_tok = F.cosine_similarity(vis_noisy, vis_clean, dim=-1)  # (256,)
            cos_self = cos_self_per_tok.mean().item()

            other_ids = [rid for rid in clean_embeds if rid != rec.id]
            if other_ids:
                device = vis_noisy.device
                other_means = torch.stack(
                    [clean_embeds[oid].mean(0) for oid in other_ids]
                ).to(device)  # (n_other, H) moved from CPU
                noisy_mean = vis_noisy.mean(0, keepdim=True)  # (1, H)
                cos_to_others = F.cosine_similarity(noisy_mean, other_means, dim=-1)
                cos_other_max = cos_to_others.max().item()
                nearest_other_id = other_ids[int(cos_to_others.argmax())]
                noisy_closer_to_other = bool(cos_other_max > cos_self)
            else:
                cos_other_max = float("nan")
                nearest_other_id = None
                noisy_closer_to_other = False

            row = {
                "test": "B_off_manifold",
                "backend": "unitok",
                "record_id": rec.id,
                "sigma": sigma,
                "cos_sim_noisy_to_clean_self":    round(cos_self, 4),
                "cos_sim_noisy_to_nearest_other": round(cos_other_max, 4),
                "nearest_other_record_id":        nearest_other_id,
                "noisy_closer_to_other":          noisy_closer_to_other,
                "correct_filter_applied":         False,
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            print(f"    cos_self={cos_self:.4f}  cos_other={cos_other_max:.4f}  "
                  f"closer_to_other={noisy_closer_to_other}\n")

    finally:
        fout.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sub-test B (off-manifold substitution) for UniTok"
    )
    parser.add_argument("--model_path",     default="FoundationVision/unitok_mllm")
    parser.add_argument("--tokenizer_path", default=None,
                        help="Path to unitok_tokenizer.pth checkpoint")
    parser.add_argument("--sweep",      default="results/sweep_unitok.jsonl")
    parser.add_argument("--out",        default="results/codebook_probe_unitok.jsonl")
    parser.add_argument("--n_records",  type=int, default=20)
    parser.add_argument("--sigma",      type=float, default=0.2,
                        help="Post-projector noise level (UniTok σ_cal=0.2)")
    parser.add_argument("--report_only", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        rows = _load_rows(out_path)
        print_report(rows)
        return

    # Load UniTok model (mirrors calibrate.py --backend unitok)
    _unitok = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "UniTok"))
    _liquid = os.path.join(_unitok, "eval", "liquid")
    for _p in (_unitok, _liquid):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import torch
    from model.builder import load_pretrained_model
    from mm_utils import get_model_name_from_path
    from models.unitok import UniTok
    from utils.config import Args as UniTokArgs
    from probe.hooks.unitok import UniTokHookManager

    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, _, _ = load_pretrained_model(
        os.path.expanduser(args.model_path), None, model_name,
        attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device

    if args.tokenizer_path is None:
        parser.error("--tokenizer_path is required")
    ckpt = torch.load(os.path.expanduser(args.tokenizer_path), map_location="cpu")
    vae_cfg = UniTokArgs()
    vae_cfg.load_state_dict(ckpt["args"])
    vq_model = UniTok(vae_cfg)
    vq_model.load_state_dict(ckpt["trainer"]["unitok"])
    vq_model.to(device).eval()
    del ckpt

    hm = UniTokHookManager(model, tokenizer, vq_model)

    from probe import load_cache, resolve_answer_token_ids
    records, _, _ = load_cache()
    resolve_answer_token_ids(records, tokenizer)

    target_ids = set(_load_high_restoration_ids(args.sweep, args.n_records))
    all_non_warn_ids = set(_load_all_non_warn_pope_ids(args.sweep))

    target_records = [r for r in records if r.id in target_ids and r.source == "pope"]
    all_records    = [r for r in records if r.id in all_non_warn_ids and r.source == "pope"]

    print(f"\nTarget (>1.0 non-WARN) records: {len(target_records)}")
    print(f"Baseline pool (non-WARN POPE):  {len(all_records)}\n")

    run_probe(hm, target_records, all_records, sigma=args.sigma, out_path=out_path)

    rows = _load_rows(out_path)
    print_report(rows)


if __name__ == "__main__":
    main()
