"""σ calibration for POPE gaussian_noise corruption.

Sweeps candidate σ values and measures mean cosine-similarity between
clean and noisy projected visual-token embeddings across a POPE sample.
Target: mean cos-sim ≈ 0.5 (following README recommendation).

Usage
-----
    source activate sae
    python -m probe.tracing.calibrate --model_path liuhaotian/llava-v1.6-vicuna-7b

    # optional overrides
    python -m probe.tracing.calibrate --model_path ... --n_samples 30 \
        --sigmas "0.05,0.1,0.2,0.5,1.0,2.0"
"""

from __future__ import annotations

import argparse
import random
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image


@torch.no_grad()
def calibrate_sigma(
    hm,
    records,
    sigmas: Sequence[float] = (0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0),
    n_samples: int = 50,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """Return [(sigma, mean_cos_sim), ...] averaged over a POPE sample.

    For each record the visual encoder runs once; noise is cheap to resample
    per-σ on CPU, so cost is n_samples projector forwards regardless of how
    many σ values are swept.

    Parameters
    ----------
    hm        : VLMHookManager (image encoder used via _build_prompt/_prepare_embeds)
    records   : ProbeRecord list — filtered to POPE records internally
    sigmas    : σ candidates to evaluate
    n_samples : how many POPE records to sample (50 is sufficient for a stable mean)
    seed      : RNG seed for record sampling
    """
    pope = [r for r in records if r.corruption_mode == "gaussian_noise"]
    if not pope:
        raise ValueError("No gaussian_noise records found.")
    rng = random.Random(seed)
    sample = rng.sample(pope, min(n_samples, len(pope)))

    # acc[sigma] = list of per-token mean cos-sim values (one per record)
    acc: dict[float, list[float]] = {s: [] for s in sigmas}

    for rec in sample:
        img = Image.open(rec.image_path).convert("RGB")
        input_ids, images_tensor, image_sizes = hm._build_prompt(img, rec.question)
        embeds, n_image_tokens = hm._prepare_embeds(input_ids, images_tensor, image_sizes)

        # visual slice: need visual_range; compute via token_index
        # _prepare_embeds expanded one sentinel into n_image_tokens
        # we need the start position of the image block (same logic as _build_token_index)
        # Both LLaVA and VILA-U use -200 as the IMAGE_TOKEN_INDEX sentinel.
        sentinel_pos = int((input_ids[0] == -200).nonzero(as_tuple=False)[0, 0])
        vlo, vhi = sentinel_pos, sentinel_pos + n_image_tokens

        clean = embeds[0, vlo:vhi, :].float()   # (n_vis, H) — float32 for cos-sim precision

        for sigma in sigmas:
            noise = torch.randn_like(clean) * sigma
            noisy = clean + noise
            per_token_cs = F.cosine_similarity(clean, noisy, dim=-1)  # (n_vis,)
            acc[sigma].append(per_token_cs.mean().item())

    results = [(s, sum(acc[s]) / len(acc[s])) for s in sigmas]
    return results


def _main():
    import sys, os

    parser = argparse.ArgumentParser(description="Calibrate sigma for POPE gaussian_noise")
    parser.add_argument("--backend", default="llava", choices=["llava", "vilau", "vila"],
                        help="Model backend (default: llava)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument(
        "--sigmas", default=None,
        help="Comma-separated floats, e.g. '0.05,0.1,0.2,0.5,1.0,2.0'. "
             "Default: 0.025,0.05,0.1,0.2,0.5,1.0,2.0,4.0",
    )
    args = parser.parse_args()

    sigmas = (
        [float(s) for s in args.sigmas.split(",")]
        if args.sigmas
        else [0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
    )

    from probe import load_cache

    print(f"Loading {args.backend} model from {args.model_path} ...")
    if args.backend == "llava":
        _llava = os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
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
        hm = LlavaHookManager(model, tokenizer, image_processor)
    elif args.backend == "vilau":
        _vilau = os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
        if _vilau not in sys.path:
            sys.path.insert(0, _vilau)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks import VilaUHookManager
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, attn_implementation="eager",
        )
        model.eval()
        hm = VilaUHookManager(model, tokenizer, image_processor)
    else:  # vila
        _vila = os.path.join(os.path.dirname(__file__), "..", "..", "VILA")
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
        hm = VilaHookManager(model, tokenizer, image_processor)

    records, _, _ = load_cache()
    print(f"Running calibration on {args.n_samples} POPE records ...\n")
    results = calibrate_sigma(hm, records, sigmas=sigmas, n_samples=args.n_samples)

    best_sigma, _ = min(results, key=lambda x: abs(x[1] - 0.5))

    print(f"  {'sigma':>10}   {'mean cos-sim':>12}")
    print(f"  {'-'*10}   {'-'*12}")
    for sigma, cs in results:
        marker = "  <-- recommended" if sigma == best_sigma else ""
        if sigma == 0.1:
            marker += "  (current default)" if not marker.strip() else " / current default"
        print(f"  {sigma:>10.4f}   {cs:>12.4f}{marker}")

    print(f"\nRecommended sigma = {best_sigma} (mean cos-sim = {dict(results)[best_sigma]:.4f})")
    print(
        f"\nTo apply: edit probe/tracing/corrupt.py line 32: "
        f"sigma: float = {best_sigma}"
    )


if __name__ == "__main__":
    _main()
