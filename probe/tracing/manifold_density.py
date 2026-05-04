"""
Manifold-density measurement for VLM backends (N-035).

For each backend, collects post-projector visual token embeddings across the
probe set and computes three manifold-richness descriptors:

  (a) Intrinsic dimensionality via TwoNN (Facco et al. 2017)
      — and PCA-95 as a robustness check.
  (b) Mean per-token cross-record nearest-neighbour cosine distance
      — tokens close to other records' tokens → sparse/substitutable manifold.
  (c) VQ codebook utilization rate and Shannon entropy (VQ backends only).

Outputs
-------
  results/manifold_density_<backend>.jsonl  — per-record rows
  results/manifold_density_<backend>.json   — aggregate summary

Usage
─────
    source activate sae
    python -m probe.tracing.manifold_density \\
        --backend llava \\
        --model_path liuhaotian/llava-v1.6-vicuna-7b

    source activate vila
    python -m probe.tracing.manifold_density \\
        --backend vila \\
        --model_path Efficient-Large-Model/VILA-7b

    source activate vila-u
    python -m probe.tracing.manifold_density \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256

    source activate unitok
    python -m probe.tracing.manifold_density \\
        --backend unitok \\
        --model_path FoundationVision/unitok_mllm \\
        --tokenizer_path /path/to/unitok_tokenizer.pth
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image


# ── intrinsic dimensionality ──────────────────────────────────────────────────

def _twonn_id_chunked(tokens: torch.Tensor, chunk_size: int = 512) -> float:
    """Chunked TwoNN for large token sets (avoids O(N²) all-at-once cdist).

    Memory per chunk: chunk_size × N × 4 bytes.
    At chunk_size=512, N=128k: ~262 MB — safe on CPU.
    """
    n = tokens.shape[0]
    if n < 4:
        return float("nan")

    r1_all = torch.full((n,), float("inf"), device=tokens.device)
    r2_all = torch.full((n,), float("inf"), device=tokens.device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = tokens[start:end]                     # (C, D)
        dists = torch.cdist(chunk, tokens, p=2)       # (C, N)
        # mask self-distances
        for j in range(end - start):
            dists[j, start + j] = float("inf")
        top2, _ = dists.topk(2, dim=1, largest=False)  # (C, 2)
        r1_all[start:end] = top2[:, 0]
        r2_all[start:end] = top2[:, 1]
        if (start // chunk_size) % 50 == 0:
            print(f"    TwoNN chunk {start // chunk_size} / {math.ceil(n / chunk_size)} …",
                  flush=True)

    r1 = r1_all.clamp(min=1e-12)
    r2 = r2_all.clamp(min=1e-12)
    mu = (r2 / r1).clamp(min=1.0 + 1e-9)
    id_est = float(n / torch.log(mu).sum().item())
    return id_est


def _twonn_id(tokens: torch.Tensor) -> float:
    """Estimate intrinsic dimensionality via the TwoNN estimator.

    Parameters
    ----------
    tokens : (N, D) float tensor (CPU or GPU).

    Returns
    -------
    Estimated intrinsic dimensionality (MLE of exponential distribution of
    log(μ_i) where μ_i = d(x_i, 2nd-NN) / d(x_i, 1st-NN)).
    """
    n = tokens.shape[0]
    if n < 4:
        return float("nan")

    # Pairwise L2 distances — keep on CPU for large tensors
    # Chunk to avoid OOM; 5000×5000 float32 ≈ 93 MB — fine
    device = tokens.device
    dists = torch.cdist(tokens, tokens, p=2)  # (N, N)

    # Mask self-distance (set diagonal to inf)
    dists.fill_diagonal_(float("inf"))

    top2, _ = dists.topk(2, dim=1, largest=False)  # (N, 2) — r1, r2
    r1 = top2[:, 0].clamp(min=1e-12)
    r2 = top2[:, 1].clamp(min=1e-12)
    mu = (r2 / r1).clamp(min=1.0 + 1e-9)

    log_mu = torch.log(mu)
    id_est = float(n / log_mu.sum().item())
    return id_est


def _pca95_dim(tokens: torch.Tensor) -> int:
    """Number of principal components capturing ≥95% of variance."""
    n, d = tokens.shape
    if n <= 1 or d <= 1:
        return 1
    centered = tokens - tokens.mean(0, keepdim=True)
    try:
        _, s, _ = torch.linalg.svd(centered, full_matrices=False)
        var = s ** 2
        cumvar = var.cumsum(0) / var.sum()
        dims = int((cumvar < 0.95).sum().item()) + 1
        return min(dims, d)
    except Exception:
        return -1


# ── cross-record NN distance ──────────────────────────────────────────────────

def _cross_record_nn_cos_dist(
    tokens: torch.Tensor,
    record_ids: list[int],
    chunk_size: int = 1024,
) -> float:
    """Mean cosine distance from each token to its nearest cross-record neighbour.

    Chunked to avoid O(N²) full matrix — safe at N=128k (512 MB peak per chunk).
    Lower = tokens sit close to other records → sparse/substitutable manifold.
    """
    n = tokens.shape[0]
    if n < 2:
        return float("nan")

    normed = F.normalize(tokens.float(), p=2, dim=-1)  # (N, D)
    rec_ids_t = torch.tensor(record_ids, dtype=torch.long, device=tokens.device)
    best_cos_sim = torch.full((n,), -2.0, device=tokens.device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = normed[start:end]                       # (C, D)
        sim_row = chunk @ normed.T                      # (C, N)
        # mask same-record tokens in this chunk
        chunk_rids = rec_ids_t[start:end].unsqueeze(1)  # (C, 1)
        same = chunk_rids == rec_ids_t.unsqueeze(0)     # (C, N)
        sim_row[same] = -2.0
        best_chunk, _ = sim_row.max(dim=1)              # (C,)
        best_cos_sim[start:end] = best_chunk
        if (start // chunk_size) % 50 == 0:
            print(f"    cross-NN chunk {start // chunk_size} / {math.ceil(n / chunk_size)} …",
                  flush=True)

    valid = best_cos_sim > -1.5
    if valid.sum() == 0:
        return float("nan")
    return float((1.0 - best_cos_sim[valid]).mean().item())


# ── per-record-pair cross-NN distance ────────────────────────────────────────

def _cross_pair_nn_cos_dist(
    tokens: torch.Tensor,
    record_ids: list[int],
) -> float:
    """Per-record-pair mean min cosine distance (audit version of _cross_record_nn_cos_dist).

    For each ordered pair (A, B) of distinct records:
      For each token in A, compute min cosine distance to all tokens in B.
      Pair distance = mean over A's tokens.
    Return mean over all (A, B) ordered pairs.

    Complexity: O(n_pairs × k_A × k_B) with torch matmul.
    For 500 records × 256 toks/rec × 499 pairs: ~16B multiply-adds (~1-5s on CPU).
    """
    normed = F.normalize(tokens.float(), p=2, dim=-1)
    rids = torch.tensor(record_ids, dtype=torch.long, device=tokens.device)
    unique_rids = rids.unique().tolist()

    pair_dists: list[float] = []
    n_unique = len(unique_rids)

    for i, a_rid in enumerate(unique_rids):
        a_mask = rids == a_rid
        a_toks = normed[a_mask]  # (k_A, D)
        for b_rid in unique_rids:
            if a_rid == b_rid:
                continue
            b_mask = rids == b_rid
            b_toks = normed[b_mask]  # (k_B, D)
            sim = a_toks @ b_toks.T          # (k_A, k_B)
            max_sim = sim.max(dim=1).values  # (k_A,)
            pair_dists.append(float((1.0 - max_sim).mean().item()))

        if (i + 1) % 50 == 0:
            print(f"    Per-pair cross-NN: {i+1}/{n_unique} source records done …",
                  flush=True)

    return float(sum(pair_dists) / len(pair_dists)) if pair_dists else float("nan")


# ── VQ codebook stats ─────────────────────────────────────────────────────────

def _vq_stats(
    all_codes: list[list[torch.Tensor]],
    codebook_sizes: list[int],
) -> dict:
    """Compute per-level codebook utilization and Shannon entropy.

    Parameters
    ----------
    all_codes : list of per-record results. Each element is a list of 1-D
                int64 tensors (one per codebook level).
    codebook_sizes : list of ints, one per level.

    Returns
    -------
    dict with 'per_level' (list of dicts) and scalar means.
    """
    if not all_codes:
        return {}

    n_levels = len(all_codes[0])
    per_level = []
    for lv in range(n_levels):
        pooled = torch.cat([rec[lv] for rec in all_codes])
        counts = torch.bincount(pooled, minlength=codebook_sizes[lv]).float()
        total = counts.sum().item()
        n_unique = int((counts > 0).sum().item())

        utilization = n_unique / codebook_sizes[lv]

        probs = counts[counts > 0] / total
        entropy_nats = float(-(probs * probs.log()).sum().item())

        per_level.append(
            {
                "level": lv,
                "codebook_size": codebook_sizes[lv],
                "n_unique": n_unique,
                "utilization": round(utilization, 4),
                "entropy_nats": round(entropy_nats, 4),
            }
        )

    mean_util = sum(lv["utilization"] for lv in per_level) / n_levels
    mean_ent  = sum(lv["entropy_nats"] for lv in per_level) / n_levels
    return {
        "per_level": per_level,
        "mean_utilization": round(mean_util, 4),
        "mean_entropy_nats": round(mean_ent, 4),
    }


# ── per-record row ────────────────────────────────────────────────────────────

def _record_vq_codes_used(codes_result) -> Optional[list[int]]:
    """Count unique codes used per level for one record."""
    if codes_result is None:
        return None
    return [int(len(lv.unique())) for lv in codes_result["levels"]]


# ── main measurement ──────────────────────────────────────────────────────────

@torch.no_grad()
def run_manifold_density(
    hm,
    records: list,
    n_subsample: int,
    seed: int,
    out_dir: Path,
    backend: str,
    audit: bool = False,
    embeddings_ckpt: Optional[Path] = None,
) -> dict:
    """Collect embeddings and compute manifold-density metrics.

    Memory-efficient: subsamples a fixed number of tokens PER RECORD during
    collection so the in-memory buffer never exceeds n_subsample × hidden,
    and all geometry is computed on CPU (no GPU pressure from distance matrices).

    Parameters
    ----------
    hm          : VLMHookManager with _prepare_embeds and optionally _get_vq_codes
    records     : ProbeRecord list (all 500)
    n_subsample : total tokens to retain across all records for TwoNN and NN distance
    seed        : RNG seed for per-record token selection
    out_dir     : directory for output files
    backend     : backend name (for output filenames)

    Returns
    -------
    Aggregate summary dict (also written to JSON).
    """
    suffix = "_audit" if audit else ""
    out_jsonl = out_dir / f"manifold_density_{backend}{suffix}.jsonl"
    out_json  = out_dir / f"manifold_density_{backend}{suffix}.json"

    rng = random.Random(seed)

    # Budget: how many tokens to keep per record to hit n_subsample total
    toks_per_rec = max(1, n_subsample // len(records))
    print(f"Token budget: {toks_per_rec} tokens/record × {len(records)} records "
          f"≈ {toks_per_rec * len(records)} total (target subsample: {n_subsample})")

    # Subsample collected in fp16 on CPU — never exceeds ~n_subsample × hidden × 2 bytes
    sub_embeds: list[torch.Tensor] = []   # each (k, H) fp16 CPU
    sub_rids:   list[int]          = []   # record index per token
    all_vq_codes: list             = []   # per-record VQ results (or None)
    codebook_sizes: Optional[list] = None
    n_vis_total = 0

    # If a prior checkpoint exists, skip the expensive GPU forward-pass phase
    if embeddings_ckpt is not None and embeddings_ckpt.exists():
        print(f"Loading embeddings checkpoint from {embeddings_ckpt} (skipping forward passes) …")
        ckpt_data = torch.load(str(embeddings_ckpt), map_location="cpu")
        # Move to GPU for fast chunked TwoNN / cross-NN (no model loaded at this point)
        _geom_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sub_tok     = ckpt_data["sub_tok"].float().to(_geom_device)
        sub_rids    = ckpt_data["sub_rids"]
        n_vis_total = ckpt_data.get("n_vis_total", 0)
        all_vq_codes   = ckpt_data.get("all_vq_codes", [])
        codebook_sizes = ckpt_data.get("codebook_sizes", None)
        n_actual = sub_tok.shape[0]
        print(f"Loaded {n_actual} tokens, {sub_tok.shape[1]}-dim from checkpoint "
              f"(device: {_geom_device})")
    else:
        sub_embeds: list[torch.Tensor] = []
        sub_rids:   list[int]          = []
        all_vq_codes: list             = []
        codebook_sizes: Optional[list] = None
        n_vis_total = 0

        fout = out_jsonl.open("w")
        try:
            for i, rec in enumerate(records):
                img = Image.open(rec.image_path).convert("RGB")

                input_ids, images_tensor, image_sizes = hm._build_prompt(img, rec.question)
                embeds, n_vis = hm._prepare_embeds(input_ids, images_tensor, image_sizes)
                n_vis_total += n_vis

                sentinel_pos = int((input_ids[0] == -200).nonzero(as_tuple=False)[0, 0])
                vis_full = embeds[0, sentinel_pos : sentinel_pos + n_vis, :].detach().cpu()
                k = min(n_vis, toks_per_rec)
                idx = torch.tensor(rng.sample(range(n_vis), k), dtype=torch.long)
                vis_sub = vis_full[idx].half()
                sub_embeds.append(vis_sub)
                sub_rids.extend([i] * k)
                del embeds, vis_full, vis_sub

                vq = hm._get_vq_codes(img)
                all_vq_codes.append(vq)
                if vq is not None and codebook_sizes is None:
                    codebook_sizes = vq["codebook_sizes"]

                n_unique_per_level = _record_vq_codes_used(vq)
                row = {
                    "record_id":           rec.id,
                    "source":              rec.source,
                    "n_visual_tokens":     int(n_vis),
                    "vq_unique_per_level": n_unique_per_level,
                }
                fout.write(json.dumps(row) + "\n")

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{len(records)} records ...")
        finally:
            fout.close()

        # Pool tokens before saving checkpoint
        sub_tok  = torch.cat(sub_embeds, dim=0).float()
        sub_rids_list = list(sub_rids)

        # Save embedding checkpoint so geometry can resume without re-running GPU
        if embeddings_ckpt is not None:
            torch.save({
                "sub_tok":       sub_tok.half(),   # fp16 to save ~1GB
                "sub_rids":      sub_rids_list,
                "n_vis_total":   n_vis_total,
                "all_vq_codes":  all_vq_codes,
                "codebook_sizes": codebook_sizes,
            }, str(embeddings_ckpt))
            print(f"Embedding checkpoint saved to {embeddings_ckpt}")

        sub_rids = sub_rids_list

    n_actual = sub_tok.shape[0]
    print(f"\nSubsample: {n_actual} tokens, {sub_tok.shape[1]}-dim "
          f"({sub_tok.numel() * 4 / 1e6:.0f} MB fp32 on CPU)")

    # (a) Intrinsic dimensionality (CPU — avoids competing with model GPU memory)
    if audit:
        print("Computing TwoNN intrinsic dimensionality (chunked, N ≥ 100k) ...")
        id_twonn = _twonn_id_chunked(sub_tok)
    else:
        print("Computing TwoNN intrinsic dimensionality ...")
        id_twonn = _twonn_id(sub_tok)
    print(f"  TwoNN ID = {id_twonn:.2f}")

    print("Computing PCA-95 dimensionality ...")
    id_pca95 = _pca95_dim(sub_tok)
    print(f"  PCA-95 dim = {id_pca95}")

    # Save partial results now so TwoNN/PCA survive even if cross-NN is killed
    partial = {
        "backend": backend, "audit": audit, "n_records": -1,
        "n_subsample": n_actual, "toks_per_rec": toks_per_rec,
        "intrinsic_dim_twonn": round(id_twonn, 3) if not math.isnan(id_twonn) else None,
        "intrinsic_dim_pca95": id_pca95,
        "mean_cross_nn_cos_dist": None, "cross_pair_nn_cos_dist": None,
        "status": "partial_after_twonn",
    }
    out_json.write_text(json.dumps(partial, indent=2))

    # (b) Cross-record NN cosine distance (CPU)
    print("Computing cross-record nearest-neighbour cosine distance ...")
    cross_nn_cos_dist = _cross_record_nn_cos_dist(sub_tok, sub_rids)
    print(f"  Mean cross-record NN cosine distance = {cross_nn_cos_dist:.4f}")

    # (b2) Per-record-pair cross-NN distance (audit mode only)
    cross_pair_nn_cos_dist: Optional[float] = None
    if audit:
        print("Computing per-record-pair cross-NN cosine distance (audit) ...")
        cross_pair_nn_cos_dist = _cross_pair_nn_cos_dist(sub_tok, sub_rids)
        if not math.isnan(cross_pair_nn_cos_dist):
            print(f"  Mean per-pair cross-NN cos dist = {cross_pair_nn_cos_dist:.4f}")

    # (c) VQ codebook utilization & entropy
    vq_summary: dict = {}
    vq_codes_list = [r["levels"] for r in all_vq_codes if r is not None]
    if vq_codes_list and codebook_sizes:
        print("Computing VQ codebook utilization and entropy ...")
        vq_summary = _vq_stats(vq_codes_list, codebook_sizes)
        for lv in vq_summary["per_level"]:
            print(
                f"  Level {lv['level']}: util={lv['utilization']:.3f} "
                f"entropy={lv['entropy_nats']:.3f} nats "
                f"(unique {lv['n_unique']}/{lv['codebook_size']})"
            )

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    summary = {
        "backend":                  backend,
        "audit":                    audit,
        "n_records":                len(records),
        "n_visual_tokens_total":    n_vis_total,
        "n_subsample":              n_actual,
        "toks_per_rec":             toks_per_rec,
        "seed":                     seed,
        "git_sha":                  git_sha,
        "intrinsic_dim_twonn":      round(id_twonn, 3) if not math.isnan(id_twonn) else None,
        "intrinsic_dim_pca95":      id_pca95,
        "mean_cross_nn_cos_dist":   round(cross_nn_cos_dist, 4) if not math.isnan(cross_nn_cos_dist) else None,
        "cross_pair_nn_cos_dist":   round(cross_pair_nn_cos_dist, 4) if (cross_pair_nn_cos_dist is not None and not math.isnan(cross_pair_nn_cos_dist)) else None,
    }
    if vq_summary:
        summary["vq"] = vq_summary

    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {out_json}")
    print(f"Per-record JSONL written to {out_jsonl}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_hook_manager(args):
    if args.backend == "llava":
        _llava = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        )
        if _llava not in sys.path:
            sys.path.insert(0, _llava)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.hooks import LlavaHookManager

        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, model_base=None, model_name=model_name,
            attn_implementation="sdpa",
        )
        model.eval()
        return LlavaHookManager(model, tokenizer, image_processor)

    if args.backend == "vila":
        _vila = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "VILA")
        )
        if _vila not in sys.path:
            sys.path.insert(0, _vila)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.hooks.vila import VilaHookManager

        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, None, model_name,
        )
        model.eval()
        return VilaHookManager(model, tokenizer, image_processor)

    if args.backend == "vilau":
        _vilau = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
        )
        if _vilau not in sys.path:
            sys.path.insert(0, _vilau)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks import VilaUHookManager

        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, attn_implementation="eager",
        )
        model.eval()
        return VilaUHookManager(model, tokenizer, image_processor)

    if args.backend == "unitok":
        _unitok = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "UniTok")
        )
        _liquid = os.path.join(_unitok, "eval", "liquid")
        for _p in (_unitok, _liquid):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from model.builder import load_pretrained_model
        from mm_utils import get_model_name_from_path
        from models.unitok import UniTok
        from utils.config import Args as UniTokArgs
        from probe.hooks.unitok import UniTokHookManager

        if not args.tokenizer_path:
            raise ValueError("--tokenizer_path is required for --backend unitok")
        model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
        tokenizer, model, _, _ = load_pretrained_model(
            os.path.expanduser(args.model_path), None, model_name,
            attn_implementation="eager",
        )
        model.eval()
        device = next(model.parameters()).device
        ckpt = torch.load(os.path.expanduser(args.tokenizer_path), map_location="cpu")
        vae_cfg = UniTokArgs()
        vae_cfg.load_state_dict(ckpt["args"])
        vq_model = UniTok(vae_cfg)
        vq_model.load_state_dict(ckpt["trainer"]["unitok"])
        vq_model.to(device).eval()
        del ckpt
        return UniTokHookManager(model, tokenizer, vq_model)

    raise ValueError(f"Unknown backend: {args.backend!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Manifold-density measurement for VLM backends (N-035)"
    )
    parser.add_argument(
        "--backend", required=True,
        choices=["llava", "vila", "vilau", "unitok"],
        help="Model backend",
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None,
                        help="[unitok only] Path to unitok_tokenizer.pth")
    parser.add_argument(
        "--n_subsample", type=int, default=2000,
        help="Total tokens to retain for TwoNN and cross-NN distance (default 2000). "
             "Tokens are sampled per-record during collection so RAM stays bounded.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results",
                        help="Output directory for JSONL + JSON (default: results/)")
    parser.add_argument(
        "--audit", action="store_true",
        help="N-037 audit mode: use chunked TwoNN + per-record-pair cross-NN. "
             "Required when n_subsample >= 100k.",
    )
    parser.add_argument(
        "--embeddings_ckpt", default=None,
        help="Path to save/load collected embeddings tensor (fp16 .pt). "
             "If the file already exists the GPU forward-pass phase is skipped. "
             "Useful for resuming after a kill during TwoNN/cross-NN.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.backend} model from {args.model_path} ...")
    hm = _build_hook_manager(args)

    from probe import load_cache, resolve_answer_token_ids
    records, _, _ = load_cache()
    resolve_answer_token_ids(records, hm.tokenizer)
    print(f"Probe set: {len(records)} records\n")

    ckpt_path = Path(args.embeddings_ckpt) if args.embeddings_ckpt else None
    summary = run_manifold_density(
        hm=hm,
        records=records,
        n_subsample=args.n_subsample,
        seed=args.seed,
        out_dir=out_dir,
        backend=args.backend,
        audit=args.audit,
        embeddings_ckpt=ckpt_path,
    )

    # Print summary
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"MANIFOLD DENSITY SUMMARY — {args.backend}")
    print(sep)
    print(f"  n_records:             {summary['n_records']}")
    print(f"  n_tokens (total):      {summary['n_visual_tokens_total']}")
    print(f"  intrinsic_dim (TwoNN): {summary['intrinsic_dim_twonn']}")
    print(f"  intrinsic_dim (PCA95): {summary['intrinsic_dim_pca95']}")
    print(f"  cross-NN cos dist:     {summary['mean_cross_nn_cos_dist']}")
    if summary.get("cross_pair_nn_cos_dist") is not None:
        print(f"  per-pair cross-NN:     {summary['cross_pair_nn_cos_dist']}")
    if "vq" in summary:
        vq = summary["vq"]
        print(f"  VQ mean utilization:   {vq['mean_utilization']:.4f}")
        print(f"  VQ mean entropy (nat): {vq['mean_entropy_nats']:.4f}")
    print(sep)


if __name__ == "__main__":
    main()
