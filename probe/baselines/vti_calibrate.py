"""Calibrate per-layer VTI textual steering directions for a VLM (N-047).

Runs 2*N_DEMOS prefill passes (clean + corrupted) on POPE records, captures
last-token hidden states at every decoder layer, computes rank-1 PCA of the
(corrupted - clean) difference vectors per layer, and saves:

  results/vti_direction_<backend>.pt          (n_layers, hidden) float32
  results/vti_direction_<backend>.meta.json   calibration provenance

Usage (VILA-U):
  source activate vila-u
  python -m probe.baselines.vti_calibrate \\
      --backend vilau --model_path mit-han-lab/vila-u-7b-256 \\
      --sigma 1.0 --n_demos 50 --out results/vti_direction_vilau.pt

Usage (LLaVA-VQ):
  source activate sae
  python -m probe.baselines.vti_calibrate \\
      --backend llava_vq --model_path liuhaotian/llava-v1.6-vicuna-7b \\
      --projector_ckpt results/llava_vq_projector.pt \\
      --sigma 0.033 --n_demos 200 --out results/vti_direction_llava_vq.pt

Usage (SEED-LLaMA):
  source activate sae
  python -m probe.baselines.vti_calibrate \\
      --backend seed --model_path AILab-CVC/seed-llama-8b-sft \\
      --tokenizer_path AILab-CVC/seed-tokenizer-2 \\
      --sigma 0.033 --n_demos 200 --out results/vti_direction_seed.pt

Usage (Show-o):
  source activate showo
  python -m probe.baselines.vti_calibrate \\
      --backend showo --model_path showlab/show-o --vq_path showlab/magvitv2 \\
      --sigma 0.1 --n_demos 200 --out results/vti_direction_showo.pt
  # Uses naturalbench automatically (POPE has 86% WARN rate for Show-o)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image


# ── Backend loaders ───────────────────────────────────────────────────────────

def _load_hook_manager(args: argparse.Namespace):
    if args.backend == "vilau":
        vila_u_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
        )
        if vila_u_root not in sys.path:
            sys.path.insert(0, vila_u_root)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks import VilaUHookManager
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, attn_implementation="eager"
        )
        model.eval()
        return VilaUHookManager(model, tokenizer, image_processor), tokenizer

    if args.backend == "llava":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA"))
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.hooks import LlavaHookManager
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, model_base=None, model_name=model_name
        )
        model.eval()
        return LlavaHookManager(model, tokenizer, image_processor), tokenizer

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
        from probe.hooks import UniTokHookManager
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, _, _ = load_pretrained_model(
            args.model_path, None, model_name, attn_implementation="eager"
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
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from probe.hooks.qwen3vl import Qwen3VLHookManager
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True, padding_side="left", use_fast=True
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="cuda"
        )
        model.eval()
        return Qwen3VLHookManager(model, processor), processor.tokenizer

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
        vq_proj = VQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim)
        if args.projector_ckpt and os.path.exists(args.projector_ckpt):
            state = torch.load(args.projector_ckpt, map_location="cpu")
            vq_proj.load_state_dict(state["projector"])
        model.model.mm_projector = vq_proj.to(next(model.parameters()).device)
        model.eval()
        return LlavaVQHookManager(model, tokenizer, image_processor), tokenizer

    if args.backend == "seed":
        _seed = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "SEED")
        )
        if _seed not in sys.path:
            sys.path.insert(0, _seed)
        from models.model_tools import get_pretrained_llama_causal_model
        from models.seed_llama_tokenizer import SeedLlamaTokenizer
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
        from probe.hooks.seed import SeedHookManager
        model = get_pretrained_llama_causal_model(
            pretrained_model_name_or_path=args.model_path,
            torch_dtype="fp16",
            low_cpu_mem_usage=True,
        )
        model = model.eval().to(device)
        return SeedHookManager(model, tokenizer), tokenizer

    if args.backend == "showo":
        showo_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Show-o")
        )
        if showo_root not in sys.path:
            sys.path.insert(0, showo_root)
        if args.vq_path is None:
            raise ValueError("--vq_path is required for --backend showo (e.g. showlab/magvitv2)")
        import torch as _torch
        from models import Showo, MAGVITv2
        from training.prompting_utils import UniversalPrompting
        from transformers import AutoTokenizer
        from probe.hooks.showo import ShowoHookManager
        device = _torch.device("cuda:0" if _torch.cuda.is_available() else "cpu")
        _phi_tok = args.tokenizer_path if args.tokenizer_path else "microsoft/phi-1_5"
        tokenizer = AutoTokenizer.from_pretrained(_phi_tok, padding_side="left")
        uni_prompting = UniversalPrompting(
            tokenizer,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
            ),
            ignore_id=-100, cond_dropout_prob=0.0,
        )
        vq_model = MAGVITv2.from_pretrained(args.vq_path).to(device).eval()
        vq_model.requires_grad_(False)
        try:
            model = Showo.from_pretrained(args.model_path, attn_implementation="eager").to(device).eval()
        except TypeError:
            model = Showo.from_pretrained(args.model_path).to(device).eval()
        return ShowoHookManager(model, tokenizer, vq_model, uni_prompting, resolution=256), tokenizer

    if args.backend == "janus":
        _janus = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Janus")
        )
        if _janus not in sys.path:
            sys.path.insert(0, _janus)
        from janus.models import MultiModalityCausalLM, VLChatProcessor
        from probe.hooks.janus import JanusProHookManager
        model_path = args.model_path
        processor = VLChatProcessor.from_pretrained(model_path)
        model = MultiModalityCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="cuda",
        )
        model.eval()
        return JanusProHookManager(model, processor), processor.tokenizer

    if args.backend == "chameleon":
        from transformers import ChameleonForConditionalGeneration, ChameleonProcessor
        from probe.hooks.chameleon_hf import ChameleonHFHookManager
        model_path = args.model_path
        processor = ChameleonProcessor.from_pretrained(model_path)
        model = ChameleonForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        )
        model.eval()
        return ChameleonHFHookManager(model, processor), processor.tokenizer

    if args.backend == "anole":
        from probe.hooks.anole import AnoleHookManager
        hm = AnoleHookManager.from_pretrained(args.model_path)
        return hm, hm.processor.tokenizer

    if args.backend == "liquid":
        _chameleon_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "chameleon")
        )
        _liquid_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Liquid")
        )
        for p in (_chameleon_root, _liquid_root):
            if p not in sys.path:
                sys.path.insert(0, p)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from chameleon.inference.image_tokenizer import ImageTokenizer
        from probe.hooks.liquid import LiquidHookManager
        model_path = args.model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        )
        model.eval()
        # Default paths: chameleon/data/tokenizer/ after running chameleon/download_data.sh
        vqgan_cfg = args.tokenizer_path or os.path.join(_chameleon_root, "data", "tokenizer", "vqgan.yaml")
        vqgan_ckpt = args.vq_path or os.path.join(_chameleon_root, "data", "tokenizer", "vqgan.ckpt")
        image_tokenizer = ImageTokenizer(
            cfg_path=vqgan_cfg, ckpt_path=vqgan_ckpt, device="cuda:0"
        )
        return LiquidHookManager(model, tokenizer, image_tokenizer), tokenizer

    if args.backend == "lumina_mgpt":
        _lumina_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Lumina-mGPT")
        )
        _lumina_data = os.path.join(_lumina_root, "lumina_mgpt")
        for p in (_lumina_root, _lumina_data):
            if p not in sys.path:
                sys.path.insert(0, p)
        from lumina_mgpt.model.chameleon import ChameleonForConditionalGeneration
        from lumina_mgpt.data.item_processor import FlexARItemProcessor
        from probe.hooks.lumina_mgpt import LuminaMGPTHookManager
        model_path = args.model_path
        model = ChameleonForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        )
        model.eval()
        item_processor = FlexARItemProcessor(tokenizer=model_path, target_size=512)
        return LuminaMGPTHookManager(model, item_processor), item_processor.tokenizer.tokenizer

    if args.backend == "llava_vq_fsq":
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
        # Load FSQ projector checkpoint
        if args.projector_ckpt and os.path.exists(args.projector_ckpt):
            ckpt = torch.load(args.projector_ckpt, map_location="cpu")
            levels = ckpt.get("fsq_levels", [8, 8, 8, 5, 5, 5])
            clip_dim = model.config.mm_hidden_size
            lm_dim = model.config.hidden_size
            fsq_proj = FSQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim, levels=levels)
            fsq_proj.load_state_dict(ckpt["projector"])
        else:
            clip_dim = model.config.mm_hidden_size
            lm_dim = model.config.hidden_size
            fsq_proj = FSQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim)
            print("[warn] no projector_ckpt given — using untrained FSQ projector")
        model.model.mm_projector = fsq_proj.to(next(model.parameters()).device)
        model.eval()
        return LlavaVQHookManager(model, tokenizer, image_processor), tokenizer

    raise ValueError(f"Unknown backend {args.backend!r}")


# ── Direction computation ─────────────────────────────────────────────────────

def _rank1_direction(diff_matrix: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Rank-1 PCA of diff_matrix (n_demos, hidden).

    Returns (direction, evr_top1, evr_top2):
      direction : (hidden,) float32 — leading PC, sign-aligned to mean diff
      evr_top1  : explained variance ratio of the leading singular value
      evr_top2  : EVR of the second singular value (0.0 if fewer than 2 rows)
    """
    X = diff_matrix.float()
    finite_mask = torch.isfinite(X).all(dim=1)
    n_bad = (~finite_mask).sum().item()
    if n_bad:
        print(f"  [warn] SVD: dropping {n_bad}/{len(X)} non-finite diff rows")
        X = X[finite_mask]
    if len(X) == 0:
        raise RuntimeError("All diff rows are non-finite for this layer; cannot compute direction")
    mean_diff = X.mean(dim=0)
    X = X - mean_diff.unsqueeze(0)
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    pc = Vh[0]  # (hidden,)
    if torch.dot(pc, mean_diff) < 0:
        pc = -pc

    # Explained variance ratios from singular values
    var = S.pow(2)
    total_var = var.sum().clamp(min=1e-12)
    evr_top1 = float((var[0] / total_var).item())
    evr_top2 = float((var[1] / total_var).item()) if S.numel() > 1 else 0.0
    return pc, evr_top1, evr_top2


def _capture_residuals_at(
    hm: Any, inputs_embeds: torch.Tensor, prompt_last: int
) -> torch.Tensor:
    """Run a forward pass with pre-built embeddings and capture the residual stream.

    Registers lightweight per-layer hooks directly — avoids register_captures/
    finalize_store which require the projector to fire (it doesn't when we skip
    _prepare_embeds and pass inputs_embeds directly).

    Returns (n_layers, hidden) float32 tensor on CPU.
    """
    layers = hm._get_decoder_layers()
    n = len(layers)
    captured: list[torch.Tensor | None] = [None] * n
    handles: list[Any] = []

    def _make_hook(idx: int):
        def _hook(module: Any, inp: Any, output: Any) -> None:
            x = output[0] if isinstance(output, (tuple, list)) else output
            captured[idx] = x[0, prompt_last].detach().float().cpu()
        return _hook

    for l, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(_make_hook(l)))

    try:
        hm._get_lm_forward()(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()

    return torch.stack(captured, dim=0)  # (n_layers, hidden)


@torch.no_grad()
def compute_directions(
    hm: Any,
    records: list[Any],
    sigma: float,
    n_demos: int,
    seed: int = 0,
) -> torch.Tensor:
    """Compute (n_layers, hidden) VTI textual directions.

    For each POPE record: run clean prefill and noisy prefill, extract
    last-token residual at every decoder layer, accumulate the difference.
    Then per-layer SVD to get rank-1 direction.
    """
    from probe.tracing.corrupt import noisy_embeds

    n_layers = len(hm._get_decoder_layers())

    # diffs[l] accumulates (corrupted - clean) at prompt_last per layer
    diffs: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    print(f"Running calibration on {min(len(records), n_demos)} records …")

    done = 0
    for rec in records:
        if done >= n_demos:
            break
        t0 = time.time()
        img = Image.open(rec.image_path).convert("RGB")

        # ── clean pass ────────────────────────────────────────────────────────
        cap_clean = hm.run_prefill(img, rec.question)
        pl = cap_clean.token_index.prompt_last
        clean_hidden = cap_clean.residual[:, pl, :].float().cpu()  # (n_layers, hidden)

        # ── corrupt pass (gaussian noise on visual tokens) ────────────────────
        vis_lo = int(cap_clean.token_index.visual_range[0])
        vis_hi = int(cap_clean.token_index.visual_range[1])
        corrupt_emb = noisy_embeds(
            hm, img, rec.question,
            visual_range=(vis_lo, vis_hi),
            sigma=sigma,
            seed=seed + done,
        )
        # Use minimal per-layer hooks — register_captures requires the projector
        # to fire (it doesn't when inputs_embeds is passed directly).
        corrupt_hidden = _capture_residuals_at(hm, corrupt_emb, pl)  # (n_layers, hidden)

        for l in range(n_layers):
            diffs[l].append((clean_hidden[l] - corrupt_hidden[l]).float())

        done += 1
        print(f"  [{done:3d}/{n_demos}] {rec.id:<28} {time.time()-t0:.1f}s", flush=True)

    if done == 0:
        raise RuntimeError("No records processed — check probe cache and POPE filter")

    # Per-layer rank-1 PCA
    directions: list[torch.Tensor] = []
    evr_top1_per_layer: list[float] = []
    evr_top2_per_layer: list[float] = []
    for l in range(n_layers):
        diff_mat = torch.stack(diffs[l], dim=0)  # (done, hidden)
        pc, evr1, evr2 = _rank1_direction(diff_mat)
        directions.append(pc)
        evr_top1_per_layer.append(evr1)
        evr_top2_per_layer.append(evr2)

    # Attach EVR lists as attributes so the caller can write them to meta.json
    result = torch.stack(directions, dim=0)  # (n_layers, hidden)
    result._evr_top1 = evr_top1_per_layer   # type: ignore[attr-defined]
    result._evr_top2 = evr_top2_per_layer   # type: ignore[attr-defined]
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute VTI textual steering directions for a VLM (N-047)"
    )
    parser.add_argument("--backend", required=True,
                        choices=["vilau", "llava", "unitok", "qwen3vl", "showo", "seed", "llava_vq",
                                 "llava_vq_fsq", "janus", "chameleon", "anole", "liquid", "lumina_mgpt"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None,
                        help="[unitok/seed/showo] path to tokenizer checkpoint or HF repo")
    parser.add_argument("--vq_path", default=None,
                        help="[showo] HF path for MAGVITv2 (e.g. showlab/magvitv2)")
    parser.add_argument("--projector_ckpt", default=None,
                        help="[llava_vq] path to trained VQLinearProjector checkpoint")
    parser.add_argument("--source", default=None,
                        choices=["pope", "naturalbench"],
                        help="Probe-set source for calibration records. "
                             "Defaults to pope for most backends, naturalbench for showo "
                             "(86%% WARN rate on POPE makes pope unusable for showo).")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Gaussian noise σ for visual token corruption. "
                             "If omitted, reads from results/sweep_<backend>.meta.json")
    parser.add_argument("--n_demos", type=int, default=50,
                        help="Number of POPE records to use for calibration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output .pt path for direction tensor")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve σ from sweep meta if not given
    sigma = args.sigma
    sweep_meta_source: str | None = None
    if sigma is None:
        meta_path = Path(f"results/sweep_{args.backend}.meta.json")
        if not meta_path.exists():
            meta_path = Path(f"results/sweep_{args.backend}.jsonl")
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"--sigma not given and no results/sweep_{args.backend}.meta.json found"
                )
        # Try to read sigma from meta
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            sigma = float(meta["sigma"])
            sweep_meta_source = str(meta_path)
            print(f"Resolved σ={sigma} from {meta_path}")
        except Exception:
            raise RuntimeError(
                f"Could not read sigma from {meta_path}. Pass --sigma explicitly."
            )

    print(f"Loading {args.backend} model from {args.model_path} …")
    hm, tokenizer = _load_hook_manager(args)

    from probe import load_cache, resolve_answer_token_ids

    records, _, _ = load_cache()
    resolve_answer_token_ids(records, tokenizer)

    source = args.source or ("naturalbench" if args.backend == "showo" else "pope")
    cal_records = [r for r in records if r.source == source]

    if not cal_records:
        raise RuntimeError(f"No {source!r} records in probe cache")

    print(f"{source!r} records available: {len(cal_records)}, using first {args.n_demos}")

    direction = compute_directions(hm, cal_records, sigma, args.n_demos, args.seed)

    torch.save(direction, out_path)
    print(f"\nDirection tensor saved → {out_path}  shape={list(direction.shape)}")

    evr_top1_per_layer = getattr(direction, "_evr_top1", [])
    evr_top2_per_layer = getattr(direction, "_evr_top2", [])
    best_layer = int(max(range(len(evr_top1_per_layer)), key=lambda i: evr_top1_per_layer[i])) \
        if evr_top1_per_layer else -1

    meta = {
        "backend": args.backend,
        "model_path": args.model_path,
        "sigma": sigma,
        "source": source,
        "n_demos": args.n_demos,
        "seed": args.seed,
        "direction_shape": list(direction.shape),
        "sweep_meta_source": sweep_meta_source,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        # Triage signal: EVR of leading singular value per layer.
        # evr_top1 >= 0.35 AND evr_top1/evr_top2 >= 2.0 → candidate for full port.
        "evr_top1_per_layer": evr_top1_per_layer,
        "evr_top2_per_layer": evr_top2_per_layer,
        "evr_top1": evr_top1_per_layer[best_layer] if best_layer >= 0 else None,
        "evr_top2": evr_top2_per_layer[best_layer] if best_layer >= 0 else None,
        "evr_best_layer": best_layer,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    if evr_top1_per_layer:
        evr1 = evr_top1_per_layer[best_layer]
        evr2 = evr_top2_per_layer[best_layer]
        ratio = evr1 / max(evr2, 1e-8)
        verdict = "CANDIDATE" if evr1 >= 0.35 and ratio >= 2.0 else \
                  "BORDERLINE" if evr1 >= 0.20 else "SKIP"
        print(f"EVR triage: layer={best_layer} evr_top1={evr1:.3f} ratio={ratio:.1f} → {verdict}")
    print(f"Meta written → {meta_path}")


if __name__ == "__main__":
    main()
