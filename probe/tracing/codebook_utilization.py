"""Codebook utilization measurement for VQ-class models (N-074).

Measures active-code fraction, marginal entropy, and effective codebook size
on 500 POPE records from the probe set.  Used to validate C3 (polarity rule):
low utilization predicts YES-promoting L0 circuit; high utilization (e.g. FSQ)
predicts YES-suppressing circuit.

Output: JSON with per-backend statistics, suitable for a polarity-vs-utilization
scatter plot combining natural specimens and induced specimens.

Usage
-----
    source activate sae

    python -m probe.tracing.codebook_utilization \\
        --backend chameleon \\
        --model_path facebook/chameleon-7b \\
        --out results/codebook_utilization_chameleon.json

    python -m probe.tracing.codebook_utilization \\
        --backend liquid \\
        --model_path Junfeng5/Liquid_V1_7B \\
        --out results/codebook_utilization_liquid.json

    # LLaVA-VQ collapsed (v1 projector):
    python -m probe.tracing.codebook_utilization \\
        --backend llava_vq \\
        --model_path liuhaotian/llava-v1.6-vicuna-7b \\
        --projector_ckpt checkpoints/llava_vq/projector_final.pt \\
        --out results/codebook_utilization_llava_vq.json

    # VILA-U:
    python -m probe.tracing.codebook_utilization \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --out results/codebook_utilization_vilau.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image


# ── model loading (mirrors calibrate.py dispatch) ──────────────────────────────

def _load_hm(args: argparse.Namespace):
    """Load hook manager and return (hm, tokenizer)."""
    print(f"\nLoading {args.backend} model from {args.model_path} ...")

    if args.backend == "chameleon":
        from transformers import ChameleonForConditionalGeneration, ChameleonProcessor
        from probe.hooks.chameleon_hf import ChameleonHFHookManager
        processor = ChameleonProcessor.from_pretrained(args.model_path)
        model = ChameleonForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        ).eval()
        hm = ChameleonHFHookManager(model, processor)
        return hm, processor.tokenizer

    if args.backend == "liquid":
        _chameleon_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "chameleon")
        )
        _liquid_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Liquid")
        )
        for _p in (_chameleon_root, _liquid_root):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from chameleon.inference.image_tokenizer import ImageTokenizer
        from probe.hooks.liquid import LiquidHookManager
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        ).eval()
        vqgan_cfg = args.tokenizer_path or os.path.join(
            _chameleon_root, "data", "tokenizer", "vqgan.yaml"
        )
        vqgan_ckpt = args.vq_path or os.path.join(
            _chameleon_root, "data", "tokenizer", "vqgan.ckpt"
        )
        image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg, ckpt_path=vqgan_ckpt, device="cuda:0")
        hm = LiquidHookManager(model, tokenizer, image_tokenizer)
        return hm, tokenizer

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
        hm = VilaUHookManager(model, tokenizer, image_processor)
        return hm, tokenizer

    if args.backend == "llava_vq":
        _llava = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        )
        if _llava not in sys.path:
            sys.path.insert(0, _llava)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.training.llava_vq_projector import VQLinearProjector
        from probe.hooks.llava_vq import LlavaVQHookManager
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, model_base=None, model_name=model_name,
            attn_implementation="sdpa",
        )
        clip_dim = model.config.mm_hidden_size
        lm_dim = model.config.hidden_size
        vq_proj = VQLinearProjector(
            clip_dim=clip_dim, lm_dim=lm_dim,
            codebook_size=getattr(args, "codebook_size", 16384),
        )
        if args.projector_ckpt and os.path.exists(args.projector_ckpt):
            state = torch.load(args.projector_ckpt, map_location="cpu")
            vq_proj.load_state_dict(state["projector"])
        model.model.mm_projector = vq_proj.to(next(model.parameters()).device)
        model.eval()
        hm = LlavaVQHookManager(model, tokenizer, image_processor)
        return hm, tokenizer

    raise ValueError(f"Unsupported backend: {args.backend!r}. "
                     f"Add a loader branch or use one of: chameleon, liquid, vilau, llava_vq")


# ── measurement ───────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_utilization(hm, records, n_records: int, seed: int = 0) -> dict:
    """Compute per-token code frequency across POPE records.

    Returns
    -------
    dict with:
      n_records_processed  : int
      codebook_size        : int   — declared K from _get_vq_codes
      n_active_codes       : int   — codes seen at least once
      active_fraction      : float — n_active / codebook_size
      marginal_entropy_bits: float — H(p) in bits over observed code distribution
      effective_K          : float — 2^H (effective codebook size by entropy)
      top10_code_freq      : list  — top-10 (code_id, count) pairs
    """
    import random
    pope = [r for r in records if r.corruption_mode == "gaussian_noise"]
    rng = random.Random(seed)
    sample = rng.sample(pope, min(n_records, len(pope)))

    code_counter: Counter = Counter()
    codebook_size: int = 0

    for i, rec in enumerate(sample, 1):
        img = Image.open(rec.image_path).convert("RGB")
        info = hm._get_vq_codes(img)
        if info is None:
            print(f"  [{i}] _get_vq_codes returned None — backend may not support VQ codes")
            continue
        codes = info["levels"][0].tolist()   # raw code indices (ints)
        codebook_size = max(codebook_size, info["codebook_sizes"][0])
        code_counter.update(codes)
        if i % 50 == 0:
            print(f"  {i}/{len(sample)} processed ...", flush=True)

    n_active = len(code_counter)
    active_fraction = n_active / codebook_size if codebook_size > 0 else 0.0

    total_tokens = sum(code_counter.values())
    if total_tokens > 0:
        probs = [c / total_tokens for c in code_counter.values()]
        entropy_bits = -sum(p * math.log2(p) for p in probs if p > 0)
    else:
        entropy_bits = 0.0
    effective_K = 2 ** entropy_bits

    top10 = [(int(k), int(v)) for k, v in code_counter.most_common(10)]

    return {
        "n_records_processed": len(sample),
        "codebook_size": codebook_size,
        "n_active_codes": n_active,
        "active_fraction": round(active_fraction, 6),
        "marginal_entropy_bits": round(entropy_bits, 4),
        "effective_K": round(effective_K, 2),
        "top10_code_freq": top10,
    }


def print_report(result: dict, backend: str) -> None:
    print(f"\n{'='*60}")
    print(f"CODEBOOK UTILIZATION — {backend.upper()}")
    print(f"{'='*60}")
    print(f"  Records processed : {result['n_records_processed']}")
    print(f"  Codebook size (K) : {result['codebook_size']}")
    print(f"  Active codes      : {result['n_active_codes']}")
    print(f"  Active fraction   : {result['active_fraction']:.4f}  "
          f"({result['active_fraction']*100:.2f}%)")
    print(f"  Marginal entropy  : {result['marginal_entropy_bits']:.2f} bits")
    print(f"  Effective K (2^H) : {result['effective_K']:.1f}")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True,
                        choices=["chameleon", "liquid", "vilau", "llava_vq"],
                        help="Model backend")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--projector_ckpt", default=None,
                        help="[llava_vq] Path to trained VQLinearProjector checkpoint")
    parser.add_argument("--tokenizer_path", default=None,
                        help="[liquid] Path to vqgan.yaml config (default: chameleon/data/tokenizer/vqgan.yaml)")
    parser.add_argument("--vq_path", default=None,
                        help="[liquid] Path to vqgan.ckpt (default: chameleon/data/tokenizer/vqgan.ckpt)")
    parser.add_argument("--codebook_size", type=int, default=16384,
                        help="[llava_vq] Codebook size of the trained projector")
    parser.add_argument("--n_records", type=int, default=500,
                        help="Number of POPE records to process (default: 500)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True,
                        help="Output JSON path")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    hm, _ = _load_hm(args)

    from probe import load_cache
    records, _, _ = load_cache()

    print(f"\nMeasuring codebook utilization on {args.n_records} POPE records ...")
    result = measure_utilization(hm, records, n_records=args.n_records, seed=args.seed)
    result["backend"] = args.backend
    result["model_path"] = args.model_path

    print_report(result, args.backend)

    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
