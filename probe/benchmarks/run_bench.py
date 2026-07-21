"""Intervention-aware benchmark runner.

Runs a full benchmark (pope_full / nb_full / hb / amber) with optional L0
intervention, writes per-record JSONL, and prints aggregate metrics.

Usage
-----
    # Baseline VILA-U on full POPE
    source activate vila-u
    python -m probe.benchmarks.run_bench \\
        --bench pope_full \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --out results/bench_pope_vilau.jsonl

    # VILA-U with winning selective top-3 intervention
    python -m probe.benchmarks.run_bench \\
        --bench pope_full \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --knockout_mode selective --heads 6,7,14 --knockout_layer 0 \\
        --out results/bench_pope_vilau_intervention.jsonl

    # LLaVA baseline (architecture-specificity: expect null delta)
    source activate sae
    python -m probe.benchmarks.run_bench \\
        --bench pope_full \\
        --backend llava \\
        --model_path liuhaotian/llava-v1.6-vicuna-7b \\
        --knockout_mode selective --heads 6,7,14 --knockout_layer 0 \\
        --out results/bench_pope_llava_intervention.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image


def _load_hook_manager(args: argparse.Namespace):
    print(f"Loading {args.backend} model from {args.model_path} …")
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
        return LlavaHookManager(model, tokenizer, image_processor), tokenizer

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
        return VilaUHookManager(model, tokenizer, image_processor), tokenizer

    if args.backend == "vilau_mlp":
        import torch as _torch
        _vilau = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
        )
        if _vilau not in sys.path:
            sys.path.insert(0, _vilau)
        from vila_u.model.builder import load_pretrained_model
        from probe.training.train_vilau_mlp import SigLIPMLPAdapter
        from probe.hooks.vilau_mlp import VilaUMLPHookManager
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, attn_implementation="eager",
        )
        model.eval()
        _ckpt = getattr(args, "projector_ckpt", None)
        if _ckpt and os.path.exists(_ckpt):
            _state = _torch.load(_ckpt, map_location="cpu")
            _adapter = SigLIPMLPAdapter(_state["siglip_dim"], _state["lm_dim"])
            _adapter.load_state_dict(_state["adapter"])
            _ref = next(model.llm.parameters())
            model.mm_projector = _adapter.to(device=_ref.device, dtype=_ref.dtype).eval()
        return VilaUMLPHookManager(model, tokenizer, image_processor), tokenizer

    if args.backend == "unitok":
        _unitok = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "UniTok")
        )
        _liquid = os.path.join(_unitok, "eval", "liquid")
        for p in (_unitok, _liquid):
            if p not in sys.path:
                sys.path.insert(0, p)
        from model.builder import load_pretrained_model
        from mm_utils import get_model_name_from_path
        from models.unitok import UniTok
        from utils.config import Args as UniTokArgs
        from probe.hooks.unitok import UniTokHookManager
        import torch
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, _, _ = load_pretrained_model(
            args.model_path, None, model_name, attn_implementation="eager",
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
        return UniTokHookManager(model, tokenizer, vq_model), tokenizer

    if args.backend == "qwen3vl":
        from probe.hooks.qwen3vl import Qwen3VLHookManager
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True, padding_side="left", use_fast=True
        )
        processor.image_processor.max_pixels = 720 * 1280
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation="eager",
            device_map="auto",
        )
        model.eval()
        return Qwen3VLHookManager(model, processor), processor.tokenizer

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
        return VilaHookManager(model, tokenizer, image_processor), tokenizer

    if args.backend == "haplo":
        _haplo = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "HaploVLM")
        )
        _haplo_model = os.path.join(_haplo, "haploomni", "model")
        for _p in (_haplo, _haplo_model):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from haploomni import HaploOmniForConditionalGeneration, HaploOmniProcessor
        from probe.hooks.haplo import HaploOmniHookManager
        import torch
        processor = HaploOmniProcessor.from_pretrained(args.model_path)
        model = HaploOmniForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        hm = HaploOmniHookManager(model, processor)
        return hm, hm.tokenizer

    if args.backend == "emu3":
        if args.vq_path is None:
            raise ValueError("--vq_path is required for --backend emu3")
        from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
        from probe.hooks.emu3 import Emu3HookManager
        import torch
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
                             max_image_size=getattr(args, "max_image_size", 256))
        return hm, tokenizer

    if args.backend == "lavit":
        import torch
        _lavit = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "LaVIT")
        )
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
        return LavitHookManager(model), model.llama_tokenizer

    if args.backend == "seed":
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
        _tok_path = args.tokenizer_path or "AILab-CVC/seed-tokenizer-2"
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
        return SeedHookManager(model, tokenizer), tokenizer

    if args.backend == "gill":
        _gill = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "gill")
        )
        if _gill not in sys.path:
            sys.path.insert(0, _gill)
        from probe.hooks.gill import GillHookManager
        hm = GillHookManager(model_path=args.model_path)
        return hm, hm.tokenizer

    if args.backend == "showo":
        import torch
        _showo = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Show-o")
        )
        if _showo not in sys.path:
            sys.path.insert(0, _showo)
        if args.vq_path is None:
            raise ValueError("--vq_path is required for --backend showo (e.g. showlab/magvitv2)")
        from models import Showo, MAGVITv2
        from training.prompting_utils import UniversalPrompting
        from transformers import AutoTokenizer
        from probe.hooks.showo import ShowoHookManager
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _phi_tok = args.tokenizer_path or "microsoft/phi-1_5"
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
        return ShowoHookManager(model, tokenizer, vq_model, uni_prompting, resolution=256), tokenizer

    if args.backend == "llava_vq":
        import torch as _torch
        _llava = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        )
        if _llava not in sys.path:
            sys.path.insert(0, _llava)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.training.llava_vq_projector import VQLinearProjector
        from probe.hooks.llava_vq import LlavaVQHookManager
        _ckpt = getattr(args, "projector_ckpt", None)
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, model_base=None, model_name=model_name,
            attn_implementation="sdpa",
        )
        clip_dim = model.config.mm_hidden_size
        lm_dim = model.config.hidden_size
        if _ckpt and os.path.exists(_ckpt):
            state = _torch.load(_ckpt, map_location="cpu")
            _cb = state["projector"]["codebook"]
            vq_proj = VQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim,
                                        codebook_size=_cb.shape[0], code_dim=_cb.shape[1])
            vq_proj.load_state_dict(state["projector"])
        else:
            vq_proj = VQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim)
        model.model.mm_projector = vq_proj.to(next(model.parameters()).device)
        model.eval()
        return LlavaVQHookManager(model, tokenizer, image_processor), tokenizer

    if args.backend == "llava_mlp":
        import torch as _torch
        _llava = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        )
        if _llava not in sys.path:
            sys.path.insert(0, _llava)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.training.llava_mlp_projector import MLPProjector
        from probe.hooks.llava_vq import LlavaVQHookManager
        _ckpt = getattr(args, "projector_ckpt", None)
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, model_base=None, model_name=model_name,
            attn_implementation="sdpa",
        )
        clip_dim = model.config.mm_hidden_size
        lm_dim = model.config.hidden_size
        mlp_proj = MLPProjector(clip_dim=clip_dim, lm_dim=lm_dim)
        if _ckpt and os.path.exists(_ckpt):
            state = _torch.load(_ckpt, map_location="cpu")
            mlp_proj.load_state_dict(state["projector"])
        model.model.mm_projector = mlp_proj.to(next(model.parameters()).device)
        model.eval()
        return LlavaVQHookManager(model, tokenizer, image_processor), tokenizer

    if args.backend == "janus":
        import torch as _torch
        from transformers import AutoModelForCausalLM
        from probe.hooks.janus import JanusProHookManager
        # VLChatProcessor lives in the janus package (trust_remote_code)
        try:
            from janus.models import VLChatProcessor
        except ImportError:
            # Fallback: load from HuggingFace custom code
            from transformers import AutoProcessor
            VLChatProcessor = AutoProcessor
        processor = VLChatProcessor.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=_torch.bfloat16,
            device_map="auto",
        ).eval()
        return JanusProHookManager(model, processor), processor.tokenizer

    if args.backend == "qwen_vq":
        import torch as _torch
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
            torch_dtype=_torch.bfloat16, attn_implementation="eager",
            device_map="auto",
        ).eval()
        ckpt_path = getattr(args, "projector_ckpt", None)
        if ckpt_path and os.path.exists(ckpt_path):
            state = _torch.load(ckpt_path, map_location="cpu")
            vq_proj = VQLinearProjector(
                clip_dim=state["clip_dim"], lm_dim=state["lm_dim"],
                code_dim=state["code_dim"], codebook_size=state["codebook_size"],
            )
            vq_proj.load_state_dict(state["projector"])
            model.visual.merger = QwenVQMerger(
                model.visual.merger, vq_proj.to(next(model.parameters()).device)
            )
        model.eval()
        return QwenVQHookManager(model, processor), processor.tokenizer

    if args.backend == "chameleon":
        import torch as _torch
        from transformers import ChameleonForConditionalGeneration, ChameleonProcessor
        from probe.hooks.chameleon_hf import ChameleonHFHookManager
        processor = ChameleonProcessor.from_pretrained(args.model_path)
        model = ChameleonForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=_torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        ).eval()
        return ChameleonHFHookManager(model, processor), processor.tokenizer

    if args.backend == "anole":
        from probe.hooks.anole import AnoleHookManager
        hm = AnoleHookManager.from_pretrained(args.model_path)
        return hm, hm.processor.tokenizer

    if args.backend == "llava_vq_fsq":
        import torch as _torch
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
        _ckpt = getattr(args, "projector_ckpt", None)
        if _ckpt and os.path.exists(_ckpt):
            ckpt_data = _torch.load(_ckpt, map_location="cpu")
            levels = ckpt_data.get("fsq_levels", [8, 8, 8, 5, 5, 5])
            fsq_proj = FSQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim, levels=levels)
            fsq_proj.load_state_dict(ckpt_data["projector"])
        else:
            fsq_proj = FSQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim)
            print("[warn] no --projector_ckpt given — using untrained FSQ projector")
        model.model.mm_projector = fsq_proj.to(next(model.parameters()).device)
        model.eval()
        return LlavaVQHookManager(model, tokenizer, image_processor), tokenizer

    if args.backend == "lumina_mgpt":
        import torch as _torch
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
            args.model_path, torch_dtype=_torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        ).eval()
        item_processor = FlexARItemProcessor(tokenizer=args.model_path, target_size=512)
        return LuminaMGPTHookManager(model, item_processor), item_processor.tokenizer.tokenizer

    if args.backend == "liquid":
        import torch as _torch
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
            args.model_path, torch_dtype=_torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        ).eval()
        _vqgan_cfg = getattr(args, "tokenizer_path", None) or os.path.join(
            _chameleon_root, "data", "tokenizer", "vqgan.yaml"
        )
        _vqgan_ckpt = getattr(args, "vq_path", None) or os.path.join(
            _chameleon_root, "data", "tokenizer", "vqgan.ckpt"
        )
        image_tokenizer = ImageTokenizer(
            cfg_path=_vqgan_cfg, ckpt_path=_vqgan_ckpt, device="cuda:0"
        )
        return LiquidHookManager(model, tokenizer, image_tokenizer), tokenizer

    raise ValueError(f"Unknown backend {args.backend!r}")


def _load_records(bench: str, max_records: int | None) -> list[Any]:
    if bench == "pope_full":
        from probe.benchmarks.pope_full import load_pope_full
        return load_pope_full(max_records=max_records)
    if bench == "nb_full":
        from probe.benchmarks.naturalbench_full import load_naturalbench_full
        return load_naturalbench_full(max_records=max_records)
    if bench == "hb":
        from probe.benchmarks.hallusionbench import load_hallusionbench
        return load_hallusionbench(max_records=max_records)
    if bench == "amber":
        from probe.benchmarks.amber import load_amber
        return load_amber(max_records=max_records)
    raise ValueError(f"Unknown bench {bench!r}")


def _score(bench: str, predictions: list[dict]) -> dict:
    if bench == "pope_full":
        from probe.benchmarks.pope_full import score_pope
        return score_pope(predictions)
    if bench == "nb_full":
        from probe.benchmarks.naturalbench_full import score_naturalbench
        return score_naturalbench(predictions)
    if bench == "hb":
        from probe.benchmarks.hallusionbench import score_hallusionbench
        return score_hallusionbench(predictions)
    if bench == "amber":
        from probe.benchmarks.amber import score_amber
        return score_amber(predictions)
    raise ValueError(f"Unknown bench {bench!r}")


def _load_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    from probe.benchmarks._dedupe import dedupe_jsonl_by_record_id
    before, after = dedupe_jsonl_by_record_id(out_path)
    if before != after:
        print(f"[resume] Removed {before - after} duplicate rows from {out_path.name}")
    done: set[str] = set()
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["record_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a full benchmark with optional L0 intervention"
    )
    parser.add_argument(
        "--bench",
        required=True,
        choices=["pope_full", "nb_full", "hb", "amber"],
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["llava", "vila", "vilau", "vilau_mlp", "unitok", "qwen3vl", "haplo", "emu3", "lavit", "seed", "gill", "showo", "llava_vq", "llava_vq_fsq", "llava_mlp", "janus", "qwen_vq", "chameleon", "anole", "lumina_mgpt", "liquid"],
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None,
                        help="[unitok/seed/showo] path to tokenizer checkpoint")
    parser.add_argument("--vq_path", default=None,
                        help="[emu3/showo] path or HF hub ID of VQ tokenizer")
    parser.add_argument("--projector_ckpt", default=None,
                        help="[llava_vq] path to trained VQLinearProjector checkpoint")
    parser.add_argument("--max_image_size", type=int, default=256,
                        help="[emu3 only] cap image dimensions before VQ-encoding (default 256→1024 tokens)")
    parser.add_argument("--out", required=True, help="Output JSONL (resumable)")
    parser.add_argument("--max_records", type=int, default=None)
    # Intervention args
    parser.add_argument(
        "--knockout_mode",
        choices=[
            "pathological_route_ablation", "full_zero",
            "selective", "scalar", "selective_scalar",
            "window_attn_knockout",
            "vti_textual",
            "projector_scale",
        ],
        default=None,
        help="full_zero is deprecated alias; window_attn_knockout requires --knockout_layer_end or "
             "--auto_window; vti_textual requires --vti_direction; "
             "projector_scale multiplies projector output by --alpha",
    )
    parser.add_argument("--knockout_layer", type=int, default=0)
    parser.add_argument(
        "--knockout_layer_end", type=int, default=None,
        help="End layer (exclusive) for window_attn_knockout. Auto-set when --auto_window is used.",
    )
    parser.add_argument(
        "--auto_window", action="store_true",
        help="Read best visual window from results/sweep_<backend>.jsonl and use window_attn_knockout",
    )
    parser.add_argument("--heads", default="6,7,14")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--vti_direction", default=None,
        help="[vti_textual] path to direction tensor (n_layers, hidden) from vti_calibrate.py",
    )
    # Decoder-mode baselines: modify how logits are derived at inference time.
    parser.add_argument(
        "--decoder", default="standard",
        choices=["standard", "vcd", "dola"],
        help="standard: normal prefill logits.  "
             "vcd: Visual Contrastive Decoding (Leng et al. CVPR 2024) — requires --decoder_sigma.  "
             "dola: Decoding by Contrasting Layers (Chuang et al. ICLR 2024) — uses --decoder_early_layer.",
    )
    parser.add_argument("--decoder_sigma", type=float, default=40.0,
                        help="[vcd] gaussian noise std-dev in pixel space (0–255 scale, default 40.0)")
    parser.add_argument("--decoder_early_layer", type=int, default=None,
                        help="[dola] premature layer index (default: n_layers // 2)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = _load_records(args.bench, args.max_records)
    hm, tokenizer = _load_hook_manager(args)

    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]
    for r in records:
        r.answer_token_id = yes_id if r.answer == "yes" else no_id

    if args.auto_window:
        from probe.tracing.best_window import best_visual_window
        from pathlib import Path as _Path
        _sweep_path = _Path(f"results/sweep_{args.backend}.jsonl")
        if not _sweep_path.exists():
            raise FileNotFoundError(
                f"--auto_window requires {_sweep_path}; run the sweep for {args.backend} first"
            )
        _l_start, _l_end, _window_score = best_visual_window(_sweep_path)
        print(
            f"Auto-window ({args.backend}): [{_l_start},{_l_end})"
            f" mean_visual_score={_window_score:.4f}"
        )
        args.knockout_layer = _l_start
        args.knockout_layer_end = _l_end
        if args.knockout_mode is None:
            args.knockout_mode = "window_attn_knockout"

    intervention = None
    if args.knockout_mode is not None:
        from probe.tracing.head_knockout import build_intervention, _parse_heads
        intervention = build_intervention(
            args.knockout_mode,
            hm,
            args.knockout_layer,
            _parse_heads(args.heads),
            args.alpha,
            layer_end=args.knockout_layer_end,
            direction_path=args.vti_direction,
        )
        if args.knockout_mode == "window_attn_knockout":
            print(
                f"Intervention: mode={args.knockout_mode},"
                f" window=[{args.knockout_layer},{args.knockout_layer_end})"
            )
        elif args.knockout_mode == "vti_textual":
            print(
                f"Intervention: mode={args.knockout_mode},"
                f" alpha={args.alpha}, direction={args.vti_direction}"
            )
        elif args.knockout_mode == "projector_scale":
            print(f"Intervention: mode={args.knockout_mode}, alpha={args.alpha}")
        else:
            print(
                f"Intervention: mode={args.knockout_mode}, layer={args.knockout_layer}"
                + (f", heads={args.heads}" if args.knockout_mode in ("selective", "selective_scalar") else "")
                + (f", alpha={args.alpha}" if args.knockout_mode in ("scalar", "selective_scalar") else "")
            )

    done_ids = _load_done_ids(out_path)
    todo = [r for r in records if r.id not in done_ids]
    print(f"Records: {len(records)} total, {len(done_ids)} done, {len(todo)} to run")
    if not todo:
        print("Nothing to run. Loading existing results for scoring …")
        rows: list[dict] = []
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        metrics = _score(args.bench, rows)
        print(json.dumps(metrics, indent=2))
        meta_path = out_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps({
            "bench": args.bench,
            "backend": args.backend,
            "model_path": args.model_path,
            "intervention_mode": args.knockout_mode,
            "decoder": getattr(args, "decoder", "standard"),
            "knockout_layer": args.knockout_layer,
            "knockout_layer_end": args.knockout_layer_end,
            "heads": args.heads if args.knockout_mode in ("selective", "selective_scalar") else None,
            "alpha": args.alpha if args.knockout_mode in ("scalar", "selective_scalar", "vti_textual", "projector_scale") else None,
            "vti_direction": args.vti_direction if args.knockout_mode == "vti_textual" else None,
            "decoder_sigma": getattr(args, "decoder_sigma", None) if getattr(args, "decoder", "standard") == "vcd" else None,
            "decoder_early_layer": getattr(args, "decoder_early_layer", None) if getattr(args, "decoder", "standard") == "dola" else None,
            "n_records": len(rows),
            "metrics": metrics,
        }, indent=2) + "\n")
        print(f"Meta written to {meta_path}")
        return

    # Prepare decoder-mode helpers (imported lazily to avoid loading in standard mode)
    _decoder = getattr(args, "decoder", "standard")
    _dola_early_layer = getattr(args, "decoder_early_layer", None)
    if _decoder == "dola" and _dola_early_layer is None:
        _dola_early_layer = len(hm._get_decoder_layers()) // 2
        print(f"[dola] Using early_layer={_dola_early_layer} (n_layers // 2)")

    t_start = time.time()
    all_rows: list[dict] = []

    with out_path.open("a") as fout:
        for i, rec in enumerate(todo, 1):
            t0 = time.time()
            img = Image.open(rec.image_path).convert("RGB")

            if _decoder == "vcd":
                from probe.baselines.vcd import make_noisy_image, vcd_logits
                if intervention is not None:
                    with intervention:
                        cap = hm.run_prefill(img, rec.question)
                else:
                    cap = hm.run_prefill(img, rec.question)
                noisy_img = make_noisy_image(img, sigma=args.decoder_sigma)
                last_logits = vcd_logits(cap, hm, noisy_img, rec.question,
                                         alpha=args.alpha,
                                         force_ids=[yes_id, no_id])
            elif _decoder == "dola":
                from probe.baselines.dola import dola_logits
                if intervention is not None:
                    with intervention:
                        cap = hm.run_prefill(img, rec.question)
                else:
                    cap = hm.run_prefill(img, rec.question)
                last_logits = dola_logits(cap, hm, early_layer=_dola_early_layer,
                                          alpha=args.alpha,
                                          force_ids=[yes_id, no_id])
            else:
                if intervention is not None:
                    with intervention:
                        cap = hm.run_prefill(img, rec.question)
                else:
                    cap = hm.run_prefill(img, rec.question)
                last_logits = cap.logits[cap.token_index.prompt_last].float()

            logit_yes = float(last_logits[yes_id])
            logit_no = float(last_logits[no_id])
            pred = "yes" if logit_yes > logit_no else "no"
            correct = pred == rec.answer

            extra: dict[str, Any] = {}
            for fld in ("pair_id", "group_id", "figure_id", "question_id"):
                val = getattr(rec, fld, None)
                if val:
                    extra[fld] = val

            row = {
                "record_id": rec.id,
                "source": rec.source,
                "gt_answer": rec.answer,
                "pred": pred,
                "correct": correct,
                "logit_yes": round(logit_yes, 4),
                "logit_no": round(logit_no, 4),
                **extra,
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            all_rows.append(row)

            elapsed = time.time() - t0
            eta = (time.time() - t_start) / i * (len(todo) - i)
            print(
                f"  [{i:4d}/{len(todo)}] {'OK' if correct else 'ERR'}"
                f"  {rec.source:<16} {rec.id:<28}"
                f"  gt={rec.answer}  pred={pred}"
                f"  {elapsed:.1f}s  ETA {eta/60:.1f}m",
                flush=True,
            )

    print(f"\nDone. {len(todo)} records in {(time.time()-t_start)/60:.1f}m")
    # Score from full JSONL (not just all_rows) to guard against partial-resume artifacts.
    # Dedupe first so duplicate rows from a re-started run don't bias metrics.
    from probe.benchmarks._dedupe import dedupe_jsonl_by_record_id
    before, after = dedupe_jsonl_by_record_id(out_path)
    if before != after:
        print(f"[score] Removed {before - after} duplicate rows before scoring")
    full_rows: list[dict] = []
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    full_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    metrics = _score(args.bench, full_rows)
    print(json.dumps(metrics, indent=2))

    _dec = getattr(args, "decoder", "standard")
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "bench": args.bench,
        "backend": args.backend,
        "model_path": args.model_path,
        "projector_ckpt": getattr(args, "projector_ckpt", None),
        "intervention_mode": args.knockout_mode,
        "knockout_layer": args.knockout_layer,
        "knockout_layer_end": args.knockout_layer_end,
        "heads": args.heads if args.knockout_mode in ("selective", "selective_scalar") else None,
        "alpha": args.alpha if args.knockout_mode in ("scalar", "selective_scalar", "vti_textual") else None,
        "vti_direction": args.vti_direction if args.knockout_mode == "vti_textual" else None,
        "decoder": _dec,
        "decoder_alpha": args.alpha if _dec in ("vcd", "dola") else None,
        "decoder_sigma": getattr(args, "decoder_sigma", None) if _dec == "vcd" else None,
        "decoder_early_layer": getattr(args, "decoder_early_layer", None) if _dec == "dola" else None,
        "n_records": len(full_rows),
        "metrics": metrics,
    }, indent=2) + "\n")
    print(f"Meta written to {meta_path}")


if __name__ == "__main__":
    main()
