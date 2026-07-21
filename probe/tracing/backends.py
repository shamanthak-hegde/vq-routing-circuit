"""Shared model-loading for tracing scripts.

`load_backend(args)` returns `(hook_manager, tokenizer)` for any supported
backend.  The body is the same dispatch used by
`probe/tracing/residual_divergence.py`; it is factored out here so multiple
tracing entry points (e.g. `verified_subset.py`) can load models identically
without duplicating ~300 lines or importing `main()`.

`args` must carry: backend, model_path, and (per-backend) tokenizer_path,
vq_path, projector_ckpt, max_image_size.
"""

from __future__ import annotations

import os
import sys

import torch


def _err(parser, msg: str) -> None:
    if parser is not None:
        parser.error(msg)
    raise ValueError(msg)


def load_backend(args, parser=None):
    """Load a model + hook manager for args.backend. Returns (hm, tokenizer)."""
    print(f"\nLoading {args.backend} model from {args.model_path} ...")
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

    elif args.backend == "llava":
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
        hm = LlavaHookManager(model, tokenizer, image_processor)

    elif args.backend == "qwen_vq":
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

    elif args.backend == "qwen3vl":
        from probe.hooks.qwen3vl import Qwen3VLHookManager
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True, padding_side="left", use_fast=True
        )
        processor.image_processor.max_pixels = 720 * 1280
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, trust_remote_code=True, torch_dtype="auto",
            attn_implementation="eager", device_map="auto",
        ).eval()
        hm = Qwen3VLHookManager(model, processor)
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
            _err(parser, "--tokenizer_path is required for --backend unitok")
        ckpt = torch.load(os.path.expanduser(args.tokenizer_path), map_location="cpu")
        vae_cfg = UniTokArgs()
        vae_cfg.load_state_dict(ckpt["args"])
        vq_model = UniTok(vae_cfg)
        vq_model.load_state_dict(ckpt["trainer"]["unitok"])
        vq_model.to(device).eval()
        del ckpt
        hm = UniTokHookManager(model, tokenizer, vq_model)

    elif args.backend == "chameleon":
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
        hm = AnoleHookManager.from_pretrained(hf_checkpoint_path=args.model_path)
        tokenizer = hm.processor.tokenizer

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

    else:
        _err(parser, f"Unknown/unsupported backend {args.backend!r} for load_backend")

    return hm, tokenizer
