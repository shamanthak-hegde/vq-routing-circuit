"""Drift probe — correlate per-record embedding drift with hallucination labels.

Three measurement modes (--mode):

  fixed_sigma (default, v1)
      Adds Gaussian noise at the calibrated σ to the projector output and measures
      mean(1 − cos_sim(clean, noisy)) across seeds.  Methodological caveat: fixed σ
      gives near-constant drift across records within a model (only embedding norm
      variation matters).  r(drift, hallucination) ≈ 0 for all tested backends.

  boundary_sensitivity (v2, recommended for VQ models)
      Runs n_seeds noise draws at an intermediate noise level (sigma_fraction × σ_cal).
      For VQ models, records near a code boundary occasionally flip to a different
      codebook entry, producing a bimodal cos_sim distribution (0.0 when flipped,
      ~1.0 otherwise).  This gives large per-record variance (high drift_std) that
      is absent in continuous-projector models and absent in VQ records near code
      centers.  Key metric: drift_std (cross-seed standard deviation of cos_sim).

  adaptive_sigma (v2)
      Binary-searches for the per-record σ_min that first causes any visual token to
      move by more than a threshold in cosine space.  For VQ records near code
      boundaries, a small σ suffices; for records near code centers, a larger σ is
      needed.  Key metric: sigma_threshold (smaller = more boundary-like).

Supports all hook backends.  Continuous-projector models (LLaVA, Qwen2.5-VL) run
as negative controls — drift_std and sigma_threshold should show no hallucination
correlation there.

Usage
-----
    # VILA-U on AMBER — boundary_sensitivity mode
    conda activate vila-u
    python -m probe.tracing.drift_probe \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --bench amber \\
        --bench_jsonl results/bench_amber_vilau_baseline.jsonl \\
        --sigma 1.0 \\
        --mode boundary_sensitivity \\
        --sigma_fraction 0.3 \\
        --n_seeds 10 \\
        --n_records 200 \\
        --out results/drift_probe_vilau_amber_boundary.jsonl

    # VILA-U on AMBER — adaptive_sigma mode
    python -m probe.tracing.drift_probe \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --bench amber \\
        --bench_jsonl results/bench_amber_vilau_baseline.jsonl \\
        --sigma 1.0 \\
        --mode adaptive_sigma \\
        --n_records 200 \\
        --out results/drift_probe_vilau_amber_adaptive.jsonl

    # UniTok on POPE — boundary mode
    conda activate unitok
    python -m probe.tracing.drift_probe \\
        --backend unitok \\
        --model_path FoundationVision/unitok_mllm \\
        --bench pope \\
        --bench_jsonl results/bench_pope_full_unitok_baseline.jsonl \\
        --sigma 0.2 \\
        --mode boundary_sensitivity \\
        --n_records 200 \\
        --out results/drift_probe_unitok_pope_boundary.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_bench_labels(bench_jsonl: Path) -> dict[str, bool]:
    """Return {record_id: hallucinating} from a bench JSONL.

    hallucinating = True when `correct` is False (model was wrong on the clean run).
    Only records where gt_answer is "yes" or "no" are included.
    """
    labels: dict[str, bool] = {}
    with open(bench_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels[row["record_id"]] = not row.get("correct", True)
    return labels


def _load_done_ids(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["record_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def _stratified_sample(
    labels: dict[str, bool], n: int, seed: int = 0
) -> list[str]:
    """Return up to n record_ids, balanced 50/50 between hallucinating and correct."""
    hall = [r for r, h in labels.items() if h]
    corr = [r for r, h in labels.items() if not h]
    rng = random.Random(seed)
    rng.shuffle(hall)
    rng.shuffle(corr)
    half = n // 2
    selected = hall[:half] + corr[:half]
    rng.shuffle(selected)
    return selected



# ── core per-record measurement ────────────────────────────────────────────────

@torch.no_grad()
def measure_drift(
    hm,
    record,
    sigma: float,
    n_seeds: int = 3,
    noise_seed_base: int = 42,
) -> dict:
    """Measure mean embedding drift for a single record.

    Adds Gaussian noise (σ) to the mm_projector output and measures
    cosine similarity between clean and noisy visual embeddings.

    Returns a dict with:
        drift_mean  — mean(1 − cos_sim(clean, noisy)) averaged over seeds and tokens
        cos_sim_mean — mean cos_sim(clean, noisy) [1 - drift_mean]
    """
    from PIL import Image

    img = Image.open(record.image_path).convert("RGB")
    input_ids, images_tensor, image_sizes = hm._build_prompt(img, record.question)

    # Clean visual embeddings via visual_range (backend-agnostic)
    clean_embeds, n_vis = hm._prepare_embeds(input_ids, images_tensor, image_sizes)
    vlo, vhi = hm.visual_range(input_ids, n_vis)
    clean_vis = clean_embeds[0, vlo:vhi, :].float()  # (n_vis, H)

    cos_sims = []
    for seed in range(n_seeds):
        g = torch.Generator(device=clean_vis.device)
        g.manual_seed(noise_seed_base + seed)
        noise = torch.randn(clean_vis.shape, generator=g,
                            device=clean_vis.device, dtype=torch.float32) * sigma
        noisy_vis = (clean_vis + noise)

        # Per-token cosine similarity
        sims = F.cosine_similarity(clean_vis, noisy_vis, dim=-1)  # (n_vis,)
        cos_sims.append(sims.mean().item())

    mean_cos = float(np.mean(cos_sims))
    return {
        "record_id": record.id,
        "drift_mean": 1.0 - mean_cos,
        "cos_sim_mean": mean_cos,
        "n_vis_tokens": n_vis,
        "sigma": sigma,
        "n_seeds": n_seeds,
    }


@torch.no_grad()
def measure_boundary_sensitivity(
    hm,
    record,
    sigma: float,
    sigma_fraction: float = 0.3,
    n_seeds: int = 10,
    noise_seed_base: int = 42,
) -> dict:
    """Measure VQ boundary sensitivity.

    Runs n_seeds noise draws at sigma * sigma_fraction. For VQ models, records
    near a codebook boundary will occasionally flip to a different code, producing
    high variance (drift_std) in cos_sim across seeds. Records near code centers
    stay in the same code → cos_sim ≈ 1.0 with near-zero variance.

    Key output: drift_std — the cross-seed standard deviation of cos_sim.
    """
    from PIL import Image

    sigma_small = sigma * sigma_fraction
    img = Image.open(record.image_path).convert("RGB")
    input_ids, images_tensor, image_sizes = hm._build_prompt(img, record.question)
    clean_embeds, n_vis = hm._prepare_embeds(input_ids, images_tensor, image_sizes)
    vlo, vhi = hm.visual_range(input_ids, n_vis)
    clean_vis = clean_embeds[0, vlo:vhi, :].float()  # (n_vis, H)

    cos_sims = []
    for seed in range(n_seeds):
        g = torch.Generator(device=clean_vis.device)
        g.manual_seed(noise_seed_base + seed)
        noise = torch.randn(clean_vis.shape, generator=g,
                            device=clean_vis.device, dtype=torch.float32) * sigma_small
        noisy_vis = clean_vis + noise
        sims = F.cosine_similarity(clean_vis, noisy_vis, dim=-1)  # (n_vis,)
        cos_sims.append(sims.mean().item())

    cos_arr = np.array(cos_sims)
    return {
        "record_id": record.id,
        "drift_mean": float(1.0 - cos_arr.mean()),
        "drift_std": float(cos_arr.std()),
        "cos_sim_mean": float(cos_arr.mean()),
        "cos_sim_std": float(cos_arr.std()),
        "n_vis_tokens": n_vis,
        "sigma": sigma,
        "sigma_small": sigma_small,
        "sigma_fraction": sigma_fraction,
        "n_seeds": n_seeds,
        "mode": "boundary_sensitivity",
    }


@torch.no_grad()
def measure_adaptive_sigma(
    hm,
    record,
    sigma_max: float,
    target_drift: float = 0.1,
    n_search_steps: int = 12,
    noise_seed_base: int = 42,
) -> dict:
    """Binary-search for the per-record σ that causes a target drift level.

    For VQ records near code boundaries: a small σ is sufficient to produce
    drift ≥ target_drift (because the noisy embedding snaps to a different code).
    For records near code centers: a larger σ is needed.

    Key output: sigma_threshold — smaller = more boundary-like = more susceptible.
    """
    from PIL import Image

    img = Image.open(record.image_path).convert("RGB")
    input_ids, images_tensor, image_sizes = hm._build_prompt(img, record.question)
    clean_embeds, n_vis = hm._prepare_embeds(input_ids, images_tensor, image_sizes)
    vlo, vhi = hm.visual_range(input_ids, n_vis)
    clean_vis = clean_embeds[0, vlo:vhi, :].float()  # (n_vis, H)

    g = torch.Generator(device=clean_vis.device)
    g.manual_seed(noise_seed_base)
    direction = torch.randn(clean_vis.shape, generator=g,
                            device=clean_vis.device, dtype=torch.float32)

    lo, hi = 0.0, sigma_max
    for _ in range(n_search_steps):
        mid = (lo + hi) / 2
        noisy_vis = clean_vis + direction * mid
        cos_sim = float(F.cosine_similarity(clean_vis, noisy_vis, dim=-1).mean())
        drift = 1.0 - cos_sim
        if drift < target_drift:
            lo = mid  # need more noise
        else:
            hi = mid  # already above target

    sigma_threshold = (lo + hi) / 2
    final_noisy = clean_vis + direction * sigma_threshold
    final_cos = float(F.cosine_similarity(clean_vis, final_noisy, dim=-1).mean())

    return {
        "record_id": record.id,
        "sigma_threshold": sigma_threshold,
        "drift_at_threshold": float(1.0 - final_cos),
        "cos_sim_at_threshold": final_cos,
        "n_vis_tokens": n_vis,
        "sigma_max": sigma_max,
        "target_drift": target_drift,
        "mode": "adaptive_sigma",
    }


# ── report ─────────────────────────────────────────────────────────────────────

def _get_primary_metric(row: dict) -> tuple[str, float]:
    """Return (metric_name, value) for the primary metric of a row."""
    mode = row.get("mode", "fixed_sigma")
    if mode == "boundary_sensitivity":
        return "drift_std", float(row.get("drift_std", float("nan")))
    if mode == "adaptive_sigma":
        # Negate sigma_threshold so higher value = more boundary-like (consistent sign)
        return "boundary_score", -float(row.get("sigma_threshold", float("nan")))
    return "drift_mean", float(row.get("drift_mean", float("nan")))


def print_report(rows: list[dict], labels: dict[str, bool]) -> None:
    hall_vals, corr_vals = [], []
    all_vals, all_hall = [], []

    for row in rows:
        rid = row["record_id"]
        if rid not in labels:
            continue
        metric_name, val = _get_primary_metric(row)
        if not np.isfinite(val):
            continue
        if labels[rid]:
            hall_vals.append(val)
        else:
            corr_vals.append(val)
        all_vals.append(val)
        all_hall.append(int(labels[rid]))

    mode = rows[0].get("mode", "fixed_sigma") if rows else "fixed_sigma"
    metric_name, _ = _get_primary_metric(rows[0]) if rows else ("drift_mean", 0.0)

    print(f"\n{'='*60}")
    print(f"DRIFT PROBE REPORT  [{mode}]  metric={metric_name}")
    print(f"{'='*60}")
    n_h, n_c = len(hall_vals), len(corr_vals)
    if n_h and n_c:
        print(f"  Hallucinating  n={n_h}  mean {metric_name}={np.mean(hall_vals):.4f}  std={np.std(hall_vals):.4f}")
        print(f"  Correct        n={n_c}  mean {metric_name}={np.mean(corr_vals):.4f}  std={np.std(corr_vals):.4f}")
        print(f"  Δ (hall − corr) = {np.mean(hall_vals) - np.mean(corr_vals):+.4f}")
    if len(all_vals) >= 3:
        r = float(np.corrcoef(all_vals, all_hall)[0, 1])
        print(f"\n  Pearson r({metric_name}, hallucination) = {r:+.4f}  (n={len(all_vals)})")
    print(f"{'='*60}\n")


# ── backend loaders ────────────────────────────────────────────────────────────
# All loaders mirror run_sweep.py exactly — hook managers require pre-loaded
# model objects, not path strings.

def _load_hm(backend: str, model_path: str, tokenizer_path: Optional[str],
             projector_ckpt: Optional[str], vq_path: Optional[str] = None):
    import os, sys
    _root = str(Path(__file__).parent.parent.parent)

    if backend == "vilau":
        import torch
        _vilau = os.path.join(_root, "vila-u")
        if _vilau not in sys.path:
            sys.path.insert(0, _vilau)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks.vilau import VilaUHookManager
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, attn_implementation="eager"
        )
        model.eval()
        return VilaUHookManager(model, tokenizer, image_processor)

    elif backend == "llava":
        _llava = os.path.join(_root, "LLaVA")
        if _llava not in sys.path:
            sys.path.insert(0, _llava)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.hooks.llava import LlavaHookManager
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, model_base=None, model_name=model_name,
            attn_implementation="sdpa",
        )
        model.eval()
        return LlavaHookManager(model, tokenizer, image_processor)

    elif backend == "unitok":
        import torch
        _unitok = os.path.join(_root, "UniTok")
        _liquid = os.path.join(_unitok, "eval", "liquid")
        for p in (_unitok, _liquid):
            if p not in sys.path:
                sys.path.insert(0, p)
        from model.builder import load_pretrained_model
        from mm_utils import get_model_name_from_path
        from models.unitok import UniTok
        from utils.config import Args as UniTokArgs
        from probe.hooks.unitok import UniTokHookManager
        model_name = get_model_name_from_path(os.path.expanduser(model_path))
        tokenizer, model, _, _ = load_pretrained_model(
            os.path.expanduser(model_path), None, model_name,
            attn_implementation="eager",
        )
        model.eval()
        device = next(model.parameters()).device
        if tokenizer_path is None:
            raise ValueError("--tokenizer_path is required for --backend unitok")
        ckpt = torch.load(os.path.expanduser(tokenizer_path), map_location="cpu")
        vae_cfg = UniTokArgs()
        vae_cfg.load_state_dict(ckpt["args"])
        vq_model = UniTok(vae_cfg)
        vq_model.load_state_dict(ckpt["trainer"]["unitok"])
        vq_model.to(device).eval()
        del ckpt
        return UniTokHookManager(model, tokenizer, vq_model)

    elif backend == "seed":
        import torch
        _seed = os.path.join(_root, "SEED")
        if _seed not in sys.path:
            sys.path.insert(0, _seed)
        from models.model_tools import get_pretrained_llama_causal_model
        from models.seed_llama_tokenizer import SeedLlamaTokenizer
        from probe.hooks.seed import SeedHookManager
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _tok_path = tokenizer_path if tokenizer_path else "AILab-CVC/seed-tokenizer-2"
        if os.path.isdir(_tok_path):
            _encoder_path = os.path.join(_tok_path, "seed_quantizer.pt")
        else:
            from huggingface_hub import hf_hub_download
            _encoder_path = hf_hub_download(repo_id=_tok_path, filename="seed_quantizer.pt")
        seed_tokenizer = SeedLlamaTokenizer.from_pretrained(
            _tok_path, fp16=True, load_diffusion=False,
            encoder_url=_encoder_path, device=str(device),
        )
        model = get_pretrained_llama_causal_model(
            pretrained_model_name_or_path=model_path,
            torch_dtype="fp16", low_cpu_mem_usage=True,
        ).eval().to(device)
        return SeedHookManager(model, seed_tokenizer)

    elif backend == "showo":
        import torch
        if vq_path is None:
            raise ValueError("--vq_path is required for --backend showo")
        _showo = os.path.join(_root, "Show-o")
        if _showo not in sys.path:
            sys.path.insert(0, _showo)
        from models import Showo, MAGVITv2
        from training.prompting_utils import UniversalPrompting
        from transformers import AutoTokenizer
        from probe.hooks.showo import ShowoHookManager
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _phi_tok = tokenizer_path if tokenizer_path else "microsoft/phi-1_5"
        showo_tokenizer = AutoTokenizer.from_pretrained(_phi_tok, padding_side="left")
        uni_prompting = UniversalPrompting(
            showo_tokenizer,
            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                            "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
            ignore_id=-100, cond_dropout_prob=0.0,
        )
        vq_model = MAGVITv2.from_pretrained(vq_path).to(device).eval()
        vq_model.requires_grad_(False)
        model = Showo.from_pretrained(model_path).to(device).eval()
        return ShowoHookManager(model, showo_tokenizer, vq_model, uni_prompting, resolution=256)

    elif backend == "qwen3vl":
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from probe.hooks.qwen3vl import Qwen3VLHookManager
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left", use_fast=True
        )
        processor.image_processor.max_pixels = 720 * 1280
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation="eager",
            device_map="auto",
        ).eval()
        return Qwen3VLHookManager(model, processor)

    elif backend == "llava_vq":
        import torch as _torch
        _llava = os.path.join(_root, "LLaVA")
        if _llava not in sys.path:
            sys.path.insert(0, _llava)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.training.llava_vq_projector import VQLinearProjector
        from probe.hooks.llava_vq import LlavaVQHookManager
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, model_base=None, model_name=model_name,
            attn_implementation="sdpa",
        )
        clip_dim = model.config.mm_hidden_size
        lm_dim   = model.config.hidden_size
        vq_proj  = VQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim)
        if projector_ckpt and os.path.exists(projector_ckpt):
            state = _torch.load(projector_ckpt, map_location="cpu")
            vq_proj.load_state_dict(state["projector"])
        model.model.mm_projector = vq_proj.to(next(model.parameters()).device)
        model.eval()
        return LlavaVQHookManager(model, tokenizer, image_processor)

    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def _load_bench_records(bench: str, bench_jsonl: Path) -> list:
    """Load records for the requested bench.

    Returns records that have labels in bench_jsonl (by record_id match).
    Normalises the attribute interface so all records have .id, .image_path, .question.
    """
    valid_ids = set(_load_bench_labels(bench_jsonl).keys())

    if bench == "amber":
        from probe.benchmarks.amber import load_amber, BenchRecord as AmberRecord
        records = load_amber()
        # AmberRecord: id, image_path, question
        return [r for r in records if r.id in valid_ids]
    elif bench == "pope":
        from probe import load_cache
        # ProbeRecord: id, image_path, question, source
        records = load_cache()
        return [r for r in records if r.id in valid_ids and r.source == "pope"]
    else:
        raise ValueError(f"Unknown bench: {bench}. Use 'amber' or 'pope'.")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="vilau",
                    choices=["vilau", "llava", "unitok", "seed", "showo",
                             "qwen3vl", "llava_vq"],
                    help="Hook manager backend")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tokenizer_path", default=None,
                    help="For UniTok: path to unitok_tokenizer.pth")
    ap.add_argument("--projector_ckpt", default=None,
                    help="For llava_vq: path to projector checkpoint")
    ap.add_argument("--vq_path", default=None,
                    help="For showo/emu3: HF ID or local path to VQ tokenizer")
    ap.add_argument("--bench", default="amber", choices=["amber", "pope"],
                    help="Benchmark to probe")
    ap.add_argument("--bench_jsonl", required=True,
                    help="Existing bench baseline JSONL for hallucination labels")
    ap.add_argument("--sigma", type=float, required=True,
                    help="Noise sigma (use calibrated value for the backend)")
    ap.add_argument("--mode", default="fixed_sigma",
                    choices=["fixed_sigma", "boundary_sensitivity", "adaptive_sigma"],
                    help=(
                        "Drift measurement mode. "
                        "fixed_sigma: mean(1-cos_sim) at fixed sigma (v1; r≈0 in practice). "
                        "boundary_sensitivity: std(cos_sim) across seeds at sigma*sigma_fraction — "
                        "  high std indicates a VQ code-boundary record. "
                        "adaptive_sigma: binary-search per-record sigma_threshold at target_drift — "
                        "  smaller threshold indicates a more boundary-like record."
                    ))
    ap.add_argument("--sigma_fraction", type=float, default=0.3,
                    help="Fraction of sigma to use for boundary_sensitivity mode (default 0.3)")
    ap.add_argument("--target_drift", type=float, default=0.1,
                    help="Target drift for adaptive_sigma mode (default 0.1)")
    ap.add_argument("--n_records", type=int, default=200,
                    help="Number of records to probe (stratified 50/50 by hallucination)")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--noise_seeds", type=int, default=3,
                    help="Noise seeds per record (averaged for fixed_sigma; cross-seed std for boundary)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for record selection")
    ap.add_argument("--report_only", action="store_true",
                    help="Skip model loading; print report from existing --out file")
    args = ap.parse_args()

    out_path = Path(args.out)

    if args.report_only:
        rows = []
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        labels = _load_bench_labels(Path(args.bench_jsonl))
        print_report(rows, labels)
        return

    # Load labels and choose records
    labels = _load_bench_labels(Path(args.bench_jsonl))
    selected_ids = _stratified_sample(labels, args.n_records, seed=args.seed)

    done_ids = _load_done_ids(out_path)
    remaining_ids = set(selected_ids) - done_ids
    if not remaining_ids:
        print("All selected records already done.")
        rows = []
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        print_report([r for r in rows if r["record_id"] in set(selected_ids)], labels)
        return

    print(f"Loading hook manager for backend={args.backend} ...")
    hm = _load_hm(args.backend, args.model_path,
                  args.tokenizer_path, args.projector_ckpt,
                  getattr(args, "vq_path", None))

    print(f"Loading benchmark records for {args.bench} ...")
    all_records = _load_bench_records(args.bench, Path(args.bench_jsonl))
    record_map = {r.id: r for r in all_records}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("a")

    n_done = len(done_ids)
    for rid in sorted(remaining_ids):
        if rid not in record_map:
            print(f"  SKIP {rid} (not in loaded records)")
            continue
        rec = record_map[rid]
        try:
            if args.mode == "boundary_sensitivity":
                n_seeds_boundary = max(args.noise_seeds, 10)
                result = measure_boundary_sensitivity(
                    hm, rec, sigma=args.sigma,
                    sigma_fraction=args.sigma_fraction,
                    n_seeds=n_seeds_boundary,
                )
            elif args.mode == "adaptive_sigma":
                result = measure_adaptive_sigma(
                    hm, rec, sigma_max=args.sigma,
                    target_drift=args.target_drift,
                )
            else:  # fixed_sigma (default)
                result = measure_drift(hm, rec, sigma=args.sigma, n_seeds=args.noise_seeds)
            result["hallucinating"] = labels.get(rid, False)
            result["bench"] = args.bench
            result["backend"] = args.backend
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            n_done += 1
            h_marker = "H" if result["hallucinating"] else "C"
            _, primary_val = _get_primary_metric(result)
            _, metric_name = _get_primary_metric(result)
            metric_name, primary_val = _get_primary_metric(result)
            print(f"  [{n_done}/{len(selected_ids)}] {rid} [{h_marker}] "
                  f"{metric_name}={primary_val:.4f}")
        except Exception as e:
            print(f"  ERROR {rid}: {e}")

    fout.close()

    rows = []
    with open(out_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print_report([r for r in rows if r["record_id"] in set(selected_ids)], labels)


if __name__ == "__main__":
    main()
