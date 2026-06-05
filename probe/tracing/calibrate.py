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

        # visual slice: delegate to the hook manager so non-LLaVA backends
        # (e.g. Qwen3VL which has no -200 sentinel) can override the range logic.
        vlo, vhi = hm.visual_range(input_ids, n_image_tokens)

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
    parser.add_argument("--backend", default="llava",
                        choices=["llava", "llava_vq", "llava_vq_fsq", "llava_mlp", "vilau", "vila",
                                 "unitok", "qwen3vl", "haplo", "emu3", "lavit", "showo", "seed",
                                 "gill", "janus", "chameleon", "anole", "liquid", "lumina_mgpt"],
                        help="Model backend (default: llava)")
    parser.add_argument("--tokenizer_path", default=None,
                        help="[unitok only] Path to unitok_tokenizer.pth")
    parser.add_argument("--vq_path", default=None,
                        help="[emu3/showo only] Path or HF hub ID of VQ tokenizer "
                             "(Emu3-VisionTokenizer or showlab/magvitv2)")
    parser.add_argument("--projector_ckpt", default=None,
                        help="[llava_vq only] Path to trained VQLinearProjector checkpoint "
                             "(checkpoints/llava_vq/projector_final.pt)")
    parser.add_argument("--max_image_size", type=int, default=256,
                        help="[emu3 only] Cap image dimensions before VQ-encoding (default 256→1024 tokens)")
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
    elif args.backend == "llava_vq":
        _llava = os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
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
        ckpt_path = getattr(args, "projector_ckpt", None)
        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            _cb = state["projector"]["codebook"]
            vq_proj = VQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim,
                                        codebook_size=_cb.shape[0], code_dim=_cb.shape[1])
            vq_proj.load_state_dict(state["projector"])
        else:
            vq_proj = VQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim)
        model.model.mm_projector = vq_proj.to(next(model.parameters()).device)
        model.eval()
        hm = LlavaVQHookManager(model, tokenizer, image_processor)
    elif args.backend == "llava_mlp":
        _llava = os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        if _llava not in sys.path:
            sys.path.insert(0, _llava)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.training.llava_mlp_projector import MLPProjector
        from probe.hooks.llava_vq import LlavaVQHookManager
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, model_base=None, model_name=model_name,
            attn_implementation="sdpa",
        )
        clip_dim = model.config.mm_hidden_size
        lm_dim = model.config.hidden_size
        mlp_proj = MLPProjector(clip_dim=clip_dim, lm_dim=lm_dim)
        ckpt_path = getattr(args, "projector_ckpt", None)
        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            mlp_proj.load_state_dict(state["projector"])
        model.model.mm_projector = mlp_proj.to(next(model.parameters()).device)
        model.eval()
        hm = LlavaVQHookManager(model, tokenizer, image_processor)
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
    elif args.backend == "unitok":
        _unitok = os.path.join(os.path.dirname(__file__), "..", "..", "UniTok")
        _liquid = os.path.join(_unitok, "eval", "liquid")
        for _p in (_unitok, _liquid):
            if _p not in sys.path:
                sys.path.insert(0, _p)
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
            raise ValueError("--tokenizer_path is required for --backend unitok")
        ckpt = torch.load(os.path.expanduser(args.tokenizer_path), map_location="cpu")
        vae_cfg = UniTokArgs()
        vae_cfg.load_state_dict(ckpt["args"])
        vq_model = UniTok(vae_cfg)
        vq_model.load_state_dict(ckpt["trainer"]["unitok"])
        vq_model.to(device).eval()
        del ckpt
        hm = UniTokHookManager(model, tokenizer, vq_model)
    elif args.backend == "qwen3vl":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from probe.hooks.qwen3vl import Qwen3VLHookManager
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True, padding_side="left", use_fast=True
        )
        processor.image_processor.max_pixels = 720 * 1280
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto",
        ).eval()
        hm = Qwen3VLHookManager(model, processor)
    elif args.backend == "haplo":
        _haplo = os.path.join(os.path.dirname(__file__), "..", "..", "HaploVLM")
        _haplo_model = os.path.join(_haplo, "haploomni", "model")
        for _p in (_haplo, _haplo_model):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from haploomni import HaploOmniForConditionalGeneration, HaploOmniProcessor
        from probe.hooks.haplo import HaploOmniHookManager
        processor = HaploOmniProcessor.from_pretrained(args.model_path)
        model = HaploOmniForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        hm = HaploOmniHookManager(model, processor)
    elif args.backend == "emu3":
        if not hasattr(args, "vq_path") or args.vq_path is None:
            raise ValueError("--vq_path is required for --backend emu3")
        from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
        from probe.hooks.emu3 import Emu3HookManager
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True, padding_side="left"
        )
        image_processor = AutoImageProcessor.from_pretrained(
            args.vq_path, trust_remote_code=True
        )
        image_tokenizer = AutoModel.from_pretrained(
            args.vq_path, device_map="cuda:0", trust_remote_code=True
        ).eval()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
        ).eval()
        hm = Emu3HookManager(model, tokenizer, image_processor, image_tokenizer,
                             max_image_size=args.max_image_size)
    elif args.backend == "lavit":
        _lavit = os.path.join(os.path.dirname(__file__), "..", "..", "LaVIT")
        if _lavit not in sys.path:
            sys.path.insert(0, _lavit)
        from models import build_model
        from probe.hooks.lavit import LavitHookManager
        model = build_model(
            model_path=args.model_path,
            model_dtype="bf16",
            device_id=0,
            use_xformers=False,
            understanding=True,
            local_files_only=True,
        )
        model = model.to("cuda")
        model.eval()
        hm = LavitHookManager(model)
    elif args.backend == "seed":
        _seed = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "SEED")
        )
        if _seed not in sys.path:
            sys.path.insert(0, _seed)
        from models.model_tools import get_pretrained_llama_causal_model
        from models.seed_llama_tokenizer import SeedLlamaTokenizer
        from probe.hooks.seed import SeedHookManager
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _tok_path = args.tokenizer_path if args.tokenizer_path else "AILab-CVC/seed-tokenizer-2"
        if os.path.isdir(_tok_path):
            _encoder_path = os.path.join(_tok_path, "seed_quantizer.pt")
        else:
            from huggingface_hub import hf_hub_download
            _encoder_path = hf_hub_download(repo_id=_tok_path, filename="seed_quantizer.pt")
        tokenizer = SeedLlamaTokenizer.from_pretrained(
            _tok_path,
            fp16=True,
            load_diffusion=False,
            encoder_url=_encoder_path,
            device=str(device),
        )
        model = get_pretrained_llama_causal_model(
            pretrained_model_name_or_path=args.model_path,
            torch_dtype="fp16",
            low_cpu_mem_usage=True,
        )
        model = model.eval().to(device)
        hm = SeedHookManager(model, tokenizer)
    elif args.backend == "showo":
        _showo = os.path.join(os.path.dirname(__file__), "..", "..", "Show-o")
        if _showo not in sys.path:
            sys.path.insert(0, _showo)
        if args.vq_path is None:
            raise ValueError("--vq_path is required for --backend showo (e.g. showlab/magvitv2)")
        from models import Showo, MAGVITv2
        from training.prompting_utils import UniversalPrompting
        from transformers import AutoTokenizer
        from probe.hooks.showo import ShowoHookManager
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Tokenizer is the Phi-1.5 backbone tokenizer, not the Show-o checkpoint itself.
        _phi_tok_path = args.tokenizer_path if args.tokenizer_path else "microsoft/phi-1_5"
        tokenizer = AutoTokenizer.from_pretrained(_phi_tok_path, padding_side="left")
        uni_prompting = UniversalPrompting(
            tokenizer,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
            ),
            ignore_id=-100,
            cond_dropout_prob=0.0,
        )
        vq_model = MAGVITv2.from_pretrained(args.vq_path).to(device).eval()
        vq_model.requires_grad_(False)
        model = Showo.from_pretrained(args.model_path).to(device).eval()
        hm = ShowoHookManager(model, tokenizer, vq_model, uni_prompting, resolution=256)
    elif args.backend == "gill":
        _gill = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "gill"))
        if _gill not in sys.path:
            sys.path.insert(0, _gill)
        from probe.hooks.gill import GillHookManager
        hm = GillHookManager(model_path=args.model_path)
    elif args.backend == "janus":
        from transformers import AutoModelForCausalLM
        from probe.hooks.janus import JanusProHookManager
        try:
            from janus.models import VLChatProcessor
        except ImportError:
            from transformers import AutoProcessor as VLChatProcessor
        processor = VLChatProcessor.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        hm = JanusProHookManager(model, processor)
    elif args.backend == "chameleon":
        from transformers import ChameleonForConditionalGeneration, ChameleonProcessor
        from probe.hooks.chameleon_hf import ChameleonHFHookManager
        processor = ChameleonProcessor.from_pretrained(args.model_path)
        model = ChameleonForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        ).eval()
        hm = ChameleonHFHookManager(model, processor)
    elif args.backend == "anole":
        from probe.hooks.anole import AnoleHookManager
        hm = AnoleHookManager.from_pretrained(args.model_path)
    elif args.backend == "llava_vq_fsq":
        _llava = os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        if _llava not in sys.path:
            sys.path.insert(0, _llava)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.training.llava_vq_fsq_projector import FSQLinearProjector
        from probe.hooks.llava_vq import LlavaVQHookManager
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, model_base=None, model_name=model_name,
            attn_implementation="sdpa",
        )
        clip_dim = model.config.mm_hidden_size
        lm_dim = model.config.hidden_size
        _ckpt = getattr(args, "projector_ckpt", None)
        if _ckpt and os.path.exists(_ckpt):
            ckpt_data = torch.load(_ckpt, map_location="cpu")
            levels = ckpt_data.get("fsq_levels", [8, 8, 8, 5, 5, 5])
            fsq_proj = FSQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim, levels=levels)
            fsq_proj.load_state_dict(ckpt_data["projector"])
        else:
            fsq_proj = FSQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim)
        model.model.mm_projector = fsq_proj.to(next(model.parameters()).device)
        model.eval()
        hm = LlavaVQHookManager(model, tokenizer, image_processor)
    elif args.backend == "liquid":
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
    elif args.backend == "lumina_mgpt":
        _lumina_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Lumina-mGPT")
        )
        _lumina_data = os.path.join(_lumina_root, "lumina_mgpt")
        for _p in (_lumina_root, _lumina_data):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from lumina_mgpt.model.chameleon import ChameleonForConditionalGeneration
        from lumina_mgpt.data.item_processor import FlexARItemProcessor
        from probe.hooks.lumina_mgpt import LuminaMGPTHookManager
        model = ChameleonForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        ).eval()
        item_processor = FlexARItemProcessor(tokenizer=args.model_path, target_size=512)
        hm = LuminaMGPTHookManager(model, item_processor)
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
        if sigma == 0.5:
            marker += "  (calibrated default)" if not marker.strip() else " / calibrated default"
        print(f"  {sigma:>10.4f}   {cs:>12.4f}{marker}")

    print(f"\nRecommended sigma = {best_sigma} (mean cos-sim = {dict(results)[best_sigma]:.4f})")
    print(
        f"\nTo apply: edit probe/tracing/corrupt.py line 32: "
        f"sigma: float = {best_sigma}"
    )


if __name__ == "__main__":
    _main()
