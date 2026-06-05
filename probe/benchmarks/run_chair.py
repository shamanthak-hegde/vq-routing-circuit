"""Open-ended captioning runner for CHAIR evaluation (N-076).

Usage:
    # Baseline
    python -m probe.benchmarks.run_chair --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --out results/chair_captions_vilau_baseline.jsonl

    # L0 intervention
    python -m probe.benchmarks.run_chair --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --knockout_layer 0 \\
        --out results/chair_captions_vilau_L0.jsonl

    # Smoke test (4 randomly sampled images, prints captions)
    python -m probe.benchmarks.run_chair --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --smoke_test --out /tmp/smoke_chair.jsonl

Images are drawn from /scratch/shegde23/data/val2014 (full COCO val2014, ~40k images).
Default: 500 randomly sampled with --seed 0 (reproducible; all conditions use same seed).
Supported backends: vilau, chameleon, liquid, anole
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
DEFAULT_IMG_DIR = Path("/scratch/shegde23/data/val2014")


def _parse_image_id(fname: str) -> int:
    m = re.search(r'COCO_val2014_0+(\d+)', fname)
    if m is None:
        raise ValueError(f"Cannot parse COCO image ID from filename: {fname}")
    return int(m.group(1))


def _decode_generated(tokenizer, cap) -> str:
    ids = cap.generated_ids
    if ids is None:
        return ""
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _load_backend(args: argparse.Namespace):
    """Load model and hook manager. Returns (hm, tokenizer)."""
    backend = args.backend
    model_path = args.model_path

    if backend == "vilau":
        _vilau = os.path.normpath(REPO / "vila-u")
        if _vilau not in sys.path:
            sys.path.insert(0, _vilau)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks import VilaUHookManager
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, attn_implementation="eager",
        )
        model.eval()
        return VilaUHookManager(model, tokenizer, image_processor), tokenizer

    elif backend == "chameleon":
        import torch
        from transformers.models.chameleon import ChameleonForConditionalGeneration, ChameleonProcessor
        from probe.hooks.chameleon_hf import ChameleonHFHookManager
        processor = ChameleonProcessor.from_pretrained(model_path)
        model = ChameleonForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        ).eval()
        return ChameleonHFHookManager(model, processor), processor.tokenizer

    elif backend == "liquid":
        import torch
        _chameleon_root = os.path.normpath(REPO / "chameleon")
        _liquid_root = os.path.normpath(REPO / "Liquid")
        for p in (_chameleon_root, _liquid_root):
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
        vqgan_cfg = args.tokenizer_path or os.path.join(
            _chameleon_root, "data", "tokenizer", "vqgan.yaml"
        )
        vqgan_ckpt = args.vq_path or os.path.join(
            _chameleon_root, "data", "tokenizer", "vqgan.ckpt"
        )
        image_tokenizer = ImageTokenizer(
            cfg_path=vqgan_cfg, ckpt_path=vqgan_ckpt, device="cuda:0"
        )
        return LiquidHookManager(model, tokenizer, image_tokenizer), tokenizer

    elif backend == "anole":
        from probe.hooks.anole import AnoleHookManager
        hm = AnoleHookManager.from_pretrained(model_path)
        return hm, hm.processor.tokenizer

    elif backend == "llava_vq":
        import torch as _torch
        _llava = os.path.normpath(REPO / "LLaVA")
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
        lm_dim = model.config.hidden_size
        ckpt_path = getattr(args, "projector_ckpt", None)
        if ckpt_path and os.path.exists(ckpt_path):
            state = _torch.load(ckpt_path, map_location="cpu")
            _cb = state["projector"]["codebook"]
            vq_proj = VQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim,
                                        codebook_size=_cb.shape[0], code_dim=_cb.shape[1])
            vq_proj.load_state_dict(state["projector"])
        else:
            vq_proj = VQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim)
            print("[warn] no --projector_ckpt given — using untrained VQ projector")
        model.model.mm_projector = vq_proj.to(next(model.parameters()).device)
        model.eval()
        return LlavaVQHookManager(model, tokenizer, image_processor), tokenizer

    else:
        raise ValueError(f"Unsupported backend: {backend!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open-ended captioning runner for CHAIR evaluation (N-076)"
    )
    parser.add_argument("--backend", required=True,
                        choices=["vilau", "chameleon", "liquid", "anole", "llava_vq"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None,
                        help="[liquid] path to vqgan.yaml (default: chameleon/data/tokenizer/vqgan.yaml)")
    parser.add_argument("--vq_path", default=None,
                        help="[liquid] path to vqgan.ckpt (default: chameleon/data/tokenizer/vqgan.ckpt)")
    parser.add_argument("--projector_ckpt", default=None,
                        help="[llava_vq] path to VQLinearProjector checkpoint (.pt)")
    parser.add_argument("--knockout_layer", type=int, default=None,
                        help="Decoder layer to zero (pathological_route_ablation); omit for baseline")
    parser.add_argument(
        "--decoder", default="standard",
        choices=["standard", "vcd", "dola"],
        help="standard: normal generation.  "
             "vcd: Visual Contrastive Decoding (N-082B) — requires --decoder_sigma.  "
             "dola: Decoding by Contrasting Layers (N-082B) — uses --decoder_early_layer.",
    )
    parser.add_argument("--decoder_sigma", type=float, default=40.0,
                        help="[vcd] gaussian noise std-dev in pixel space 0–255 (default 40.0)")
    parser.add_argument("--decoder_alpha", type=float, default=1.0,
                        help="[vcd/dola] contrastive weight (default 1.0; use 0.5 for dola per paper)")
    parser.add_argument("--decoder_early_layer", type=int, default=None,
                        help="[dola] premature layer index (default: n_layers // 2)")
    parser.add_argument("--prompt", default="Please describe this image in detail.",
                        help="Captioning prompt (default: CHAIR standard)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--img_dir", default=str(DEFAULT_IMG_DIR),
                        help=f"Directory of COCO_val2014_*.jpg images (default: {DEFAULT_IMG_DIR})")
    parser.add_argument("--n_images", type=int, default=500,
                        help="Number of randomly sampled images to caption (default: 500)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for image sampling (default: 0)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run 4 images and print captions; overrides --n_images")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    all_images = sorted(img_dir.glob("COCO_val2014_*.jpg"))
    if not all_images:
        raise RuntimeError(f"No COCO_val2014_*.jpg images found in {img_dir}")

    rng = random.Random(args.seed)
    sampled = list(all_images)
    rng.shuffle(sampled)

    if args.smoke_test:
        images = sampled[:4]
    else:
        images = sampled[:args.n_images]

    print(f"Images: {len(images)} sampled (seed={args.seed}) from {len(all_images)} in {img_dir}")
    print(f"Loading {args.backend} model from {args.model_path} ...")
    hm, tokenizer = _load_backend(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from probe.tracing.head_knockout import LayerAttnKnockout

    knockout_label = args.knockout_layer
    if knockout_label is not None:
        ctx = LayerAttnKnockout(hm, knockout_label)
        print(f"Intervention: L{knockout_label} attention knockout")
    else:
        ctx = nullcontext()
        print("Intervention: none (baseline)")

    _decoder = args.decoder
    _decoder_alpha = args.decoder_alpha
    _decoder_sigma = args.decoder_sigma
    _decoder_early_layer = args.decoder_early_layer

    if _decoder == "dola" and _decoder_early_layer is None:
        _decoder_early_layer = len(hm._get_decoder_layers()) // 2

    print(f"Decoder: {_decoder}" + (
        f"  alpha={_decoder_alpha}  sigma={_decoder_sigma}" if _decoder == "vcd"
        else f"  alpha={_decoder_alpha}  early_layer={_decoder_early_layer}" if _decoder == "dola"
        else ""
    ))
    print(f"Prompt: {args.prompt!r}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"Output: {out_path}\n")

    t0 = time.time()
    n_written = 0
    n_empty = 0

    # DoLA processor is stateless per-image but its hook stays across the loop.
    # Re-create it inside the loop so it resets _early_hidden each image.
    # VCD processor is also re-created each image (new noisy image each time).

    with out_path.open("w") as fout, ctx:
        for i, img_path in enumerate(images, 1):
            t_img = time.time()
            image_id = _parse_image_id(img_path.name)
            image = Image.open(img_path).convert("RGB")

            if _decoder == "vcd":
                from probe.baselines.vcd import make_noisy_image
                from probe.baselines.vcd_generation import make_vcd_processor
                from transformers import LogitsProcessorList
                noisy_img = make_noisy_image(image, sigma=_decoder_sigma)
                proc = make_vcd_processor(hm, noisy_img, args.prompt, alpha=_decoder_alpha)
                cap = hm.run_generate(
                    image, args.prompt, max_new_tokens=args.max_new_tokens,
                    logits_processor=LogitsProcessorList([proc]),
                )
            elif _decoder == "dola":
                from probe.baselines.dola_generation import DoLAGenerationLogitsProcessor
                from transformers import LogitsProcessorList
                proc = DoLAGenerationLogitsProcessor(
                    hm, early_layer=_decoder_early_layer, alpha=_decoder_alpha
                )
                try:
                    cap = hm.run_generate(
                        image, args.prompt, max_new_tokens=args.max_new_tokens,
                        logits_processor=LogitsProcessorList([proc]),
                    )
                finally:
                    proc.remove()
            else:
                cap = hm.run_generate(image, args.prompt, max_new_tokens=args.max_new_tokens)

            text = _decode_generated(tokenizer, cap)

            if not text:
                n_empty += 1

            row = {
                "image_id": image_id,
                "caption": text,
                "backend": args.backend,
                "knockout_layer": knockout_label,
                "decoder": _decoder,
                "decoder_alpha": _decoder_alpha if _decoder != "standard" else None,
                "decoder_sigma": _decoder_sigma if _decoder == "vcd" else None,
                "decoder_early_layer": _decoder_early_layer if _decoder == "dola" else None,
                "prompt": args.prompt,
                "seed": args.seed,
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n_written += 1

            elapsed = time.time() - t_img
            total_elapsed = time.time() - t0
            eta = (total_elapsed / i) * (len(images) - i)

            if args.smoke_test:
                print(f"  [{i}/{len(images)}] id={image_id}  {elapsed:.1f}s  "
                      f"caption='{text[:80]}'")
            else:
                print(f"  [{i:4d}/{len(images)}] id={image_id}  {elapsed:.1f}s  "
                      f"ETA {eta / 60:.1f}m  '{text[:40]}'", flush=True)

    dt = time.time() - t0
    print(f"\nDone: {n_written} captions written in {dt / 60:.1f}m "
          f"({n_empty} empty, {n_written - n_empty} non-empty)")
    print(f"Written -> {out_path}")


if __name__ == "__main__":
    main()
