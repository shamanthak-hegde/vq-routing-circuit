"""
Per-layer residual divergence at prompt_last: VILA-U vs UniTok.

For each probe record, runs a clean prefill and a noisy prefill (Gaussian noise
added to post-projector visual token embeddings at σ_cal), then measures how much
the residual stream at the prompt_last position has diverged from clean at every
layer.  Divergence growing across layers = LM amplifies the substituted signal;
divergence flat or shrinking = LM attenuates it.

Hypothesis (Stage-2):
  VILA-U amplifies the substituted visual embedding in early layers → visual-dominant
  POPE circuit and >1.0 restoration.  UniTok attenuates it → prompt_last-dominant
  circuit despite identical Stage-1 (off-manifold) substitution.

Usage
-----
    # VILA-U (conda: vila-u)
    python -m probe.tracing.residual_divergence \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --sigma 1.0 \\
        --out results/residual_divergence_vilau.json

    # UniTok (conda: unitok)
    python -m probe.tracing.residual_divergence \\
        --backend unitok \\
        --model_path FoundationVision/unitok_mllm \\
        --tokenizer_path /path/to/UniTok/checkpoint/unitok_tokenizer.pth \\
        --sigma 0.2 \\
        --out results/residual_divergence_unitok.json

    # Show-o (conda: showo)
    python -m probe.tracing.residual_divergence \\
        --backend showo \\
        --model_path showlab/show-o \\
        --vq_path showlab/magvitv2 \\
        --sigma 0.1 \\
        --source naturalbench \\
        --out results/residual_divergence_showo.json

    # Figure (any env with matplotlib)
    python -m probe.tracing.residual_divergence_figure
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

from probe.tracing.corrupt import (
    noisy_embeds, dropout_embeds, shuffle_embeds, blurred_embeds,
)


# ── residual capture ──────────────────────────────────────────────────────────

@torch.no_grad()
def _residual_from_embeds(hm, inputs_embeds: torch.Tensor) -> torch.Tensor:
    """Run LM forward from pre-built inputs_embeds; return residual on CPU.

    Cannot use register_captures/finalize_store here because finalize_store
    asserts exactly one projector call — which won't fire when passing
    inputs_embeds directly.  Instead, register decoder-layer hooks inline.

    Returns
    -------
    torch.Tensor of shape (n_layers, seq_len, hidden).
    """
    n_layers = len(hm._get_decoder_layers())
    bufs: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    handles = []
    for i, layer in enumerate(hm._get_decoder_layers()):
        def _hook(mod, inp, out, _i=i):
            t = out[0] if isinstance(out, (tuple, list)) else out
            bufs[_i].append(t.squeeze(0).detach().cpu())  # (seq_len, H)
        handles.append(layer.register_forward_hook(_hook))
    try:
        hm._get_lm_forward()(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()
    return torch.stack([calls[0] for calls in bufs], dim=0)  # (n_layers, seq_len, H)


# ── per-record measurement ────────────────────────────────────────────────────

@torch.no_grad()
def measure_record(hm, record, sigma: float, seed: int = 0,
                   corruption: str = "gaussian", drop_frac: float = 0.5,
                   blur_radius: float = 8.0) -> dict:
    """Compute per-layer residual divergence at prompt_last for one record.

    corruption (alternate visual corruptions for Gate A.1):
      "gaussian" — add N(0, sigma^2) to projected visual tokens (default/canonical)
      "dropout"  — zero a random `drop_frac` of visual tokens
      "shuffle"  — randomly permute the visual-token embeddings
      "blur"     — Gaussian-blur the pixel image (radius `blur_radius`) pre-encoder
    The divergence normalization (vis_noise_rms = ‖noisy_vis − clean_vis‖) is
    corruption-agnostic, so rel_div is comparable across corruption types.

    Returns dict with keys: record_id, prompt_last, abs_div, rel_div, vis_noise_rms.
    """
    img = Image.open(record.image_path).convert("RGB")

    # ── clean prefill ─────────────────────────────────────────────────────────
    clean_cap = hm.run_prefill(img, record.question)
    pl = clean_cap.token_index.prompt_last
    vr = clean_cap.token_index.visual_range
    n_layers = clean_cap.residual.shape[0]

    # ── noisy forward ─────────────────────────────────────────────────────────
    if corruption == "gaussian":
        noisy_emb = noisy_embeds(hm, img, record.question,
                                 visual_range=vr, sigma=sigma, seed=seed)
    elif corruption == "dropout":
        noisy_emb = dropout_embeds(hm, img, record.question,
                                   visual_range=vr, drop_frac=drop_frac, seed=seed)
    elif corruption == "shuffle":
        noisy_emb = shuffle_embeds(hm, img, record.question,
                                   visual_range=vr, seed=seed)
    elif corruption == "blur":
        noisy_emb = blurred_embeds(hm, img, record.question, blur_radius=blur_radius)
    else:
        raise ValueError(f"unknown corruption: {corruption}")
    noisy_res = _residual_from_embeds(hm, noisy_emb)  # (n_layers, seq_len, H)

    # ── divergence at prompt_last per layer ───────────────────────────────────
    abs_div, rel_div = [], []
    for L in range(n_layers):
        cv = clean_cap.residual[L, pl, :].float()
        nv = noisy_res[L, pl, :].float()
        ad = (cv - nv).norm().item()
        rd = ad / cv.norm().clamp(min=1e-8).item()
        abs_div.append(round(ad, 6))
        rel_div.append(round(rd, 6))

    # ── magnitude of noise injected at visual positions (for normalization) ──
    clean_vis = clean_cap.visual_embeds.float()                    # (n_vis, H) on CPU
    if vr is not None:
        noisy_vis = noisy_emb[0, vr[0]:vr[1], :].float().cpu()    # (n_vis, H)
        noise_vec = noisy_vis - clean_vis
        vis_noise_rms = float(noise_vec.norm() / (noise_vec.numel() ** 0.5))
    else:
        vis_noise_rms = float("nan")

    return {
        "record_id":    record.id,
        "prompt_last":  int(pl),
        "abs_div":      abs_div,
        "rel_div":      rel_div,
        "vis_noise_rms": round(vis_noise_rms, 6),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-layer residual divergence at prompt_last"
    )
    parser.add_argument("--backend", required=True,
                        choices=["vilau", "unitok", "showo", "seed", "llava_vq", "llava_vq_fsq",
                                 "llava_mlp", "qwen_vq", "emu3", "chameleon", "anole", "liquid",
                                 "lumina_mgpt"],
                        help="Model backend to run")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None,
                        help="[unitok only] Path to unitok_tokenizer.pth")
    parser.add_argument("--vq_path", default=None,
                        help="[showo/emu3] HuggingFace path for VQ tokenizer")
    parser.add_argument("--projector_ckpt", default=None,
                        help="[llava_vq only] Path to trained VQLinearProjector checkpoint")
    parser.add_argument("--max_image_size", type=int, default=256,
                        help="[emu3 only] Cap image dimensions before VQ-encoding (default 256)")
    parser.add_argument("--sigma", type=float, required=True,
                        help="Post-projector noise σ (use σ_cal for each backend)")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: results/residual_divergence_<backend>.json)")
    parser.add_argument("--records_from",
                        default=None,
                        help="JSONL file from which to extract record_ids to measure "
                             "(default: results/codebook_probe_<backend>.jsonl; "
                             "if that file is absent, falls back to non-WARN records "
                             "from results/sweep_<backend>.jsonl)")
    parser.add_argument("--source", default="pope",
                        choices=["pope", "naturalbench", "all"],
                        help="Which probe-set source to sample from (default: pope). "
                             "Use naturalbench for Show-o where POPE WARN rate is 86%%.")
    parser.add_argument("--n_records", type=int, default=None,
                        help="Max records to process (default: all from --records_from)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Noise RNG seed (default: 0)")
    parser.add_argument("--corruption", default="gaussian",
                        choices=["gaussian", "dropout", "shuffle", "blur"],
                        help="visual corruption for Gate A.1 "
                             "(default gaussian = canonical). Alternate modes write "
                             "results/residual_divergence_<backend>_<corruption>.json")
    parser.add_argument("--drop_frac", type=float, default=0.5,
                        help="[--corruption dropout] fraction of visual tokens to zero")
    parser.add_argument("--blur_radius", type=float, default=8.0,
                        help="[--corruption blur] PIL Gaussian blur radius")
    args = parser.parse_args()

    _suffix = "" if args.corruption == "gaussian" else f"_{args.corruption}"
    out_path = Path(args.out or f"results/residual_divergence_{args.backend}{_suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load target record IDs ────────────────────────────────────────────────
    if args.records_from is None:
        args.records_from = f"results/codebook_probe_{args.backend}.jsonl"

    target_ids: list[str] = []
    records_from = Path(args.records_from)
    if records_from.exists():
        seen = set()
        with open(records_from) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rid = json.loads(line)["record_id"]
                        if rid not in seen:
                            seen.add(rid)
                            target_ids.append(rid)
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"Loaded {len(target_ids)} record IDs from {records_from}")
    else:
        print(f"Warning: {records_from} not found — will use first n_records non-WARN POPE records")

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.backend} model from {args.model_path} ...")
    if args.backend == "llava_vq":
        import torch  # noqa: needed; seed branch also imports torch locally
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

    elif args.backend == "llava_vq_fsq":
        import torch
        _llava = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        )
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
        ckpt_path = getattr(args, "projector_ckpt", None)
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt_data = torch.load(ckpt_path, map_location="cpu")
            levels = ckpt_data.get("fsq_levels", [8, 8, 8, 5, 5, 5])
            fsq_proj = FSQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim, levels=levels)
            fsq_proj.load_state_dict(ckpt_data["projector"])
        else:
            fsq_proj = FSQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim)
        model.model.mm_projector = fsq_proj.to(next(model.parameters()).device)
        model.eval()
        hm = LlavaVQHookManager(model, tokenizer, image_processor)

    elif args.backend == "llava_mlp":
        import torch
        _llava = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        )
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

    elif args.backend == "qwen_vq":
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from probe.training.llava_vq_projector import VQLinearProjector
        from probe.training.train_qwen_vq import QwenVQMerger
        from probe.hooks.qwen_vq import QwenVQHookManager
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True, padding_side="left"
        )
        processor.image_processor.max_pixels = 336 * 336
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation="eager",
            device_map="auto",
        ).eval()
        ckpt_path = getattr(args, "projector_ckpt", None)
        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            vq_proj = VQLinearProjector(
                clip_dim=state["clip_dim"], lm_dim=state["lm_dim"],
                code_dim=state["code_dim"], codebook_size=state["codebook_size"],
            )
            vq_proj.load_state_dict(state["projector"])
            model.visual.merger = QwenVQMerger(
                model.visual.merger, vq_proj.to(next(model.parameters()).device)
            )
        model.eval()
        hm = QwenVQHookManager(model, processor)
        tokenizer = processor.tokenizer

    elif args.backend == "vilau":
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

    elif args.backend == "unitok":
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

        model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
        tokenizer, model, _, _ = load_pretrained_model(
            os.path.expanduser(args.model_path), None, model_name,
            attn_implementation="eager",
        )
        model.eval()
        device = next(model.parameters()).device
        if args.tokenizer_path is None:
            parser.error("--tokenizer_path is required for --backend unitok")
        ckpt = torch.load(os.path.expanduser(args.tokenizer_path), map_location="cpu")
        vae_cfg = UniTokArgs()
        vae_cfg.load_state_dict(ckpt["args"])
        vq_model = UniTok(vae_cfg)
        vq_model.load_state_dict(ckpt["trainer"]["unitok"])
        vq_model.to(device).eval()
        del ckpt
        hm = UniTokHookManager(model, tokenizer, vq_model)

    elif args.backend == "seed":
        import torch
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
        _showo = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Show-o")
        )
        if _showo not in sys.path:
            sys.path.insert(0, _showo)
        if args.vq_path is None:
            parser.error("--vq_path is required for --backend showo (e.g. showlab/magvitv2)")
        from models import Showo, MAGVITv2
        from training.prompting_utils import UniversalPrompting
        from transformers import AutoTokenizer
        from probe.hooks.showo import ShowoHookManager

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _phi_tok = args.tokenizer_path if args.tokenizer_path else "microsoft/phi-1_5"
        tokenizer = AutoTokenizer.from_pretrained(_phi_tok, padding_side="left")
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

    elif args.backend == "emu3":
        import torch
        if args.vq_path is None:
            parser.error("--vq_path is required for --backend emu3 (e.g. BAAI/Emu3-VisionTokenizer)")
        from transformers import (AutoTokenizer, AutoModel,
                                  AutoImageProcessor, AutoModelForCausalLM)
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
            args.model_path, device_map="cuda:0", torch_dtype=torch.bfloat16,
            attn_implementation="eager", trust_remote_code=True,
        ).eval()
        hm = Emu3HookManager(model, tokenizer, image_processor, image_tokenizer,
                             max_image_size=args.max_image_size)

    elif args.backend == "chameleon":
        import torch
        from transformers import ChameleonForConditionalGeneration, ChameleonProcessor
        from probe.hooks.chameleon_hf import ChameleonHFHookManager
        processor = ChameleonProcessor.from_pretrained(args.model_path)
        model = ChameleonForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        ).eval()
        hm = ChameleonHFHookManager(model, processor)
        tokenizer = processor.tokenizer

    elif args.backend == "anole":
        from probe.hooks.anole import AnoleHookManager
        hm = AnoleHookManager.from_pretrained(
            hf_checkpoint_path=args.model_path,
        )
        tokenizer = hm.processor.tokenizer

    elif args.backend == "liquid":
        import torch
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
        import torch
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
        tokenizer = item_processor.tokenizer.tokenizer

    else:
        raise ValueError(f"Unknown backend {args.backend!r}")

    # ── load probe records ────────────────────────────────────────────────────
    from probe import load_cache, resolve_answer_token_ids
    records, _, _ = load_cache()
    resolve_answer_token_ids(records, tokenizer)

    src_filter = (lambda r: True) if args.source == "all" else \
                 (lambda r: r.source == args.source)

    if target_ids:
        id_set = set(target_ids)
        todo = [r for r in records if r.id in id_set and src_filter(r)]
        order = {rid: i for i, rid in enumerate(target_ids)}
        todo.sort(key=lambda r: order.get(r.id, 9999))
    else:
        # Fallback: first n non-WARN records from sweep (filtered by --source).
        # If sweep file also absent, use all source records (e.g. new backends
        # like llava_mlp where no sweep has been run yet).
        sweep_path = Path(f"results/sweep_{args.backend}.jsonl")
        non_warn: set[str] = set()
        if sweep_path.exists():
            warn_flags: dict[str, bool] = {}
            sweep_src = None if args.source == "all" else args.source
            with open(sweep_path) as f:
                for line in f:
                    row = json.loads(line.strip())
                    if sweep_src and row.get("source") != sweep_src:
                        continue
                    rid = row["record_id"]
                    from probe.tracing.filters import is_warn_row
                    warn_flags[rid] = warn_flags.get(rid, False) or is_warn_row(row)
            non_warn = {rid for rid, w in warn_flags.items() if not w}
            todo = [r for r in records if src_filter(r) and r.id in non_warn]
        else:
            print(f"Warning: {sweep_path} also not found — using all {args.source} records")
            todo = [r for r in records if src_filter(r)]

    if args.n_records is not None:
        todo = todo[:args.n_records]

    print(f"Records to measure: {len(todo)}")
    print(f"sigma={args.sigma}  seed={args.seed}  out={out_path}\n")

    # ── measure ───────────────────────────────────────────────────────────────
    per_record = []
    for i, rec in enumerate(todo, 1):
        print(f"  [{i:2d}/{len(todo)}] {rec.id}", end="  ", flush=True)
        result = measure_record(hm, rec, sigma=args.sigma, seed=args.seed,
                                corruption=args.corruption, drop_frac=args.drop_frac,
                                blur_radius=args.blur_radius)
        per_record.append(result)
        _n = len(result["abs_div"])
        _late_start = max(8, 3 * _n // 4)
        early = result["abs_div"][:8]
        late  = result["abs_div"][_late_start:]
        print(f"abs_div [0,8) mean={sum(early)/len(early):.2f}  "
              f"[{_late_start},{_n}) mean={sum(late)/len(late) if late else float('nan'):.2f}  "
              f"noise_rms={result['vis_noise_rms']:.3f}")

    if not per_record:
        print("No records measured — check --records_from path.")
        return

    # ── aggregate ─────────────────────────────────────────────────────────────
    n_layers = len(per_record[0]["abs_div"])
    mean_abs = [
        sum(r["abs_div"][L] for r in per_record) / len(per_record)
        for L in range(n_layers)
    ]
    mean_rel = [
        sum(r["rel_div"][L] for r in per_record) / len(per_record)
        for L in range(n_layers)
    ]
    mean_noise_rms = sum(r["vis_noise_rms"] for r in per_record) / len(per_record)

    # Divergence normalized by layer-0 value (shows amplification vs attenuation)
    base = mean_abs[0] if mean_abs[0] > 0 else 1.0
    norm_abs = [round(v / base, 6) for v in mean_abs]

    output = {
        "backend":        args.backend,
        "model_path":     args.model_path,
        "sigma":          args.sigma,
        "corruption":     args.corruption,
        "seed":           args.seed,
        "n_records":      len(per_record),
        "n_layers":       n_layers,
        "mean_noise_rms": round(mean_noise_rms, 6),
        "mean_abs_div":   [round(v, 6) for v in mean_abs],
        "mean_rel_div":   [round(v, 6) for v in mean_rel],
        "norm_abs_div":   norm_abs,
        "per_record":     per_record,
    }

    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWritten → {out_path}")

    # ── print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"RESIDUAL DIVERGENCE AT PROMPT_LAST — {args.backend.upper()}")
    print(f"sigma={args.sigma}  n_records={len(per_record)}  n_layers={n_layers}")
    print("=" * 64)
    print(f"  {'layer':>6}  {'abs_div':>10}  {'rel_div':>10}  {'norm_abs':>10}")
    print(f"  {'------':>6}  {'-------':>10}  {'-------':>10}  {'--------':>10}")
    for L in range(n_layers):
        marker = " ◄" if L < 8 else ""
        print(f"  {L:6d}  {mean_abs[L]:10.4f}  {mean_rel[L]:10.4f}  {norm_abs[L]:10.4f}{marker}")
    print("=" * 64)
    late_start = max(8, 3 * n_layers // 4)
    early_norm = norm_abs[:8]
    late_norm  = norm_abs[late_start:]
    print(f"\nNorm-abs mean [0,8):           {sum(early_norm)/len(early_norm):.4f}")
    if late_norm:
        print(f"Norm-abs mean [{late_start},{n_layers}): {sum(late_norm)/len(late_norm):.4f}")
    verdict = "AMPLIFICATION" if sum(early_norm)/len(early_norm) > 1.05 else \
              "ATTENUATION"   if sum(early_norm)/len(early_norm) < 0.95 else "FLAT"
    print(f"Early-layer verdict:           {verdict}")


if __name__ == "__main__":
    main()
