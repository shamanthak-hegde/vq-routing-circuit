"""Sub-test B (off-manifold substitution) for any VQ-class backend —.

Generalizes codebook_probe.py and codebook_probe_unitok.py to support all
hook-manager backends. Runs sub-test B only: after adding calibrated Gaussian
noise to the mm_projector output, does the noisy embedding land closer to a
different record's clean embedding than to its own?

Sub-test A (VQ code sensitivity) is VILA-U / backend-specific and lives in the
original codebook_probe.py — this script focuses on the cross-model paper claim.

Usage
-----
    # LLaVA-VQ (sae env)
    source activate sae
    python -m probe.tracing.codebook_probe_offmanifold \\
        --backend llava_vq \\
        --model_path liuhaotian/llava-v1.6-vicuna-7b \\
        --projector_ckpt checkpoints/llava_vq/projector_final.pt \\
        --sweep results/sweep_llava_vq.jsonl \\
        --n_records 200 --sigma 2.0 \\
        --out results/codebook_probe_llava_vq.jsonl

    # SEED-LLaMA (seed_llama env)
    python -m probe.tracing.codebook_probe_offmanifold \\
        --backend seed \\
        --model_path AILab-CVC/seed-llama-8b-sft \\
        --sweep results/sweep_seed.jsonl \\
        --n_records 200 --sigma 0.033 \\
        --out results/codebook_probe_seed.jsonl

    # Show-o NB records (showo env) — use --source naturalbench, POPE is 86% WARN
    python -m probe.tracing.codebook_probe_offmanifold \\
        --backend showo \\
        --model_path showlab/show-o --vq_path showlab/magvitv2 \\
        --sweep results/sweep_showo.jsonl \\
        --n_records 200 --sigma 0.1 --source naturalbench \\
        --out results/codebook_probe_showo.jsonl

    # Emu3 (emu env)
    python -m probe.tracing.codebook_probe_offmanifold \\
        --backend emu3 \\
        --model_path BAAI/Emu3-Chat --vq_path BAAI/Emu3-VisionTokenizer \\
        --sweep results/sweep_emu3.jsonl \\
        --n_records 200 --sigma 0.01 \\
        --out results/codebook_probe_emu3.jsonl

    # Report only (no model loading):
    python -m probe.tracing.codebook_probe_offmanifold \\
        --report_only --out results/codebook_probe_llava_vq.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image


# ── record selection ──────────────────────────────────────────────────────────

def _select_records(
    sweep_path: str,
    source: str,
    n: int,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Return (target_ids, pool_ids).

    target_ids: up to n records with max visual score > 1.0 and non-WARN.
                Falls back to any non-WARN records if no >1.0 found.
    pool_ids:   all non-WARN records in the source (for off-manifold baseline).
    """
    sweep: dict[str, dict] = {}
    with open(sweep_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row["record_id"]
            src = row.get("source", "")
            if source != "all" and src != source:
                continue
            is_warn = row["logit_clean"] <= row["logit_corrupt"]
            score = row.get("score", 0.0)
            if rid not in sweep:
                sweep[rid] = {"is_warn": is_warn, "max_score": score}
            else:
                sweep[rid]["max_score"] = max(sweep[rid]["max_score"], score)
                if is_warn:
                    sweep[rid]["is_warn"] = True

    pool_ids = [rid for rid, v in sweep.items() if not v["is_warn"]]
    gt1_ids  = [rid for rid in pool_ids if sweep[rid]["max_score"] > 1.0]

    rng = random.Random(seed)
    if gt1_ids:
        rng.shuffle(gt1_ids)
        target_ids = gt1_ids[:n]
        print(f"Found {len(gt1_ids)} non-WARN records with max_score>1.0; using {len(target_ids)}")
    else:
        rng.shuffle(pool_ids)
        target_ids = pool_ids[:n]
        print(f"No >1.0 records found; using {len(target_ids)} non-WARN records (max_score<1.0)")

    print(f"Off-manifold pool: {len(pool_ids)} non-WARN records")
    return target_ids, pool_ids


# ── model loading ─────────────────────────────────────────────────────────────

def _load_hm(
    backend: str,
    model_path: str,
    tokenizer_path: Optional[str],
    projector_ckpt: Optional[str],
    vq_path: Optional[str],
    codebook_size: int = 16384,
):
    """Load the appropriate hook manager. Mirrors calibrate.py dispatch."""
    if backend == "llava_vq":
        import os, sys, torch as _torch
        _llava = str(Path(__file__).parent.parent.parent / "LLaVA")
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
        vq_proj  = VQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim,
                                     codebook_size=codebook_size)
        if projector_ckpt and os.path.exists(projector_ckpt):
            state = _torch.load(projector_ckpt, map_location="cpu")
            vq_proj.load_state_dict(state["projector"])
        model.model.mm_projector = vq_proj.to(next(model.parameters()).device)
        model.eval()
        return LlavaVQHookManager(model, tokenizer, image_processor)

    elif backend == "vilau":
        import os, sys
        _vilau = str(Path(__file__).parent.parent.parent / "vila-u")
        if _vilau not in sys.path:
            sys.path.insert(0, _vilau)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks.vilau import VilaUHookManager
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, attn_implementation="eager"
        )
        model.eval()
        return VilaUHookManager(model, tokenizer, image_processor)

    elif backend == "unitok":
        import os, sys, torch
        _unitok = str(Path(__file__).parent.parent.parent / "UniTok")
        _liquid = str(Path(_unitok) / "eval" / "liquid")
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
        import os, sys, torch
        _seed = str(Path(__file__).parent.parent.parent / "SEED")
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
        import torch, sys
        if vq_path is None:
            raise ValueError("--vq_path is required for --backend showo (e.g. showlab/magvitv2)")
        _showo = str(Path(__file__).parent.parent.parent / "Show-o")
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
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
            ),
            ignore_id=-100, cond_dropout_prob=0.0,
        )
        vq_model = MAGVITv2.from_pretrained(vq_path).to(device).eval()
        vq_model.requires_grad_(False)
        model = Showo.from_pretrained(model_path).to(device).eval()
        return ShowoHookManager(model, showo_tokenizer, vq_model, uni_prompting, resolution=256)

    elif backend == "emu3":
        import torch
        if vq_path is None:
            raise ValueError("--vq_path is required for --backend emu3 (e.g. BAAI/Emu3-VisionTokenizer)")
        from transformers import (AutoTokenizer, AutoModel,
                                  AutoImageProcessor, AutoModelForCausalLM)
        from probe.hooks.emu3 import Emu3HookManager
        emu_tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        image_processor = AutoImageProcessor.from_pretrained(vq_path, trust_remote_code=True)
        image_tokenizer = AutoModel.from_pretrained(
            vq_path, device_map="cuda:0", trust_remote_code=True
        ).eval()
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda:0", torch_dtype=torch.bfloat16,
            attn_implementation="eager", trust_remote_code=True,
        ).eval()
        return Emu3HookManager(model, emu_tokenizer, image_processor, image_tokenizer)

    elif backend == "chameleon":
        import torch
        from transformers import ChameleonForConditionalGeneration, ChameleonProcessor
        from probe.hooks.chameleon_hf import ChameleonHFHookManager
        processor = ChameleonProcessor.from_pretrained(model_path)
        model = ChameleonForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        ).eval()
        return ChameleonHFHookManager(model, processor)

    elif backend == "anole":
        from probe.hooks.anole import AnoleHookManager
        return AnoleHookManager.from_pretrained(hf_checkpoint_path=model_path)

    elif backend == "liquid":
        import os as _os, sys as _sys, torch
        _chameleon_root = _os.path.normpath(
            _os.path.join(_os.path.dirname(__file__), "..", "..", "chameleon")
        )
        _liquid_root = _os.path.normpath(
            _os.path.join(_os.path.dirname(__file__), "..", "..", "Liquid")
        )
        for _p in (_chameleon_root, _liquid_root):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from chameleon.inference.image_tokenizer import ImageTokenizer
        from probe.hooks.liquid import LiquidHookManager
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        ).eval()
        vqgan_cfg = tokenizer_path or _os.path.join(
            _chameleon_root, "data", "tokenizer", "vqgan.yaml"
        )
        vqgan_ckpt = vq_path or _os.path.join(
            _chameleon_root, "data", "tokenizer", "vqgan.ckpt"
        )
        image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg, ckpt_path=vqgan_ckpt, device="cuda:0")
        return LiquidHookManager(model, tokenizer, image_tokenizer)

    elif backend == "lumina_mgpt":
        import os as _os, sys as _sys, torch
        _lumina_root = _os.path.normpath(
            _os.path.join(_os.path.dirname(__file__), "..", "..", "Lumina-mGPT")
        )
        _lumina_data = _os.path.join(_lumina_root, "lumina_mgpt")
        for _p in (_lumina_root, _lumina_data):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from lumina_mgpt.model.chameleon import ChameleonForConditionalGeneration
        from lumina_mgpt.data.item_processor import FlexARItemProcessor
        from probe.hooks.lumina_mgpt import LuminaMGPTHookManager
        model = ChameleonForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        ).eval()
        item_processor = FlexARItemProcessor(tokenizer=model_path, target_size=512)
        return LuminaMGPTHookManager(model, item_processor)

    else:
        raise ValueError(f"Unknown backend: {backend!r}")


# ── probe core ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_probe(
    hm,
    target_records,
    pool_records,
    sigma: float,
    backend: str,
    out_path: Path,
) -> None:
    """Run sub-test B (off-manifold substitution) for any backend.

    Uses hm.visual_range() to locate visual tokens — backend-agnostic.
    """
    fout = out_path.open("a")
    done_ids: set[str] = set()
    # Load already-written records to support resuming
    if out_path.stat().st_size > 0 if out_path.exists() else False:
        with out_path.open() as fin:
            for line in fin:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line)["record_id"])
                    except Exception:
                        pass

    try:
        print(f"Pre-computing clean visual embeddings for pool ({len(pool_records)} records) ...")
        clean_embeds: dict[str, torch.Tensor] = {}
        for i, rec in enumerate(pool_records):
            img = Image.open(rec.image_path).convert("RGB")
            input_ids, images_tensor, image_sizes = hm._build_prompt(img, rec.question)
            embeds, n_vis = hm._prepare_embeds(input_ids, images_tensor, image_sizes)
            vlo, vhi = hm.visual_range(input_ids, n_vis)
            clean_embeds[rec.id] = embeds[0, vlo:vhi, :].float().cpu()
            if (i + 1) % 25 == 0:
                print(f"  Cached {i+1}/{len(pool_records)} ...")
        print(f"  Done — {len(clean_embeds)} records cached\n")

        for rec in target_records:
            if rec.id in done_ids:
                print(f"  SKIP {rec.id} (already written)")
                continue
            img = Image.open(rec.image_path).convert("RGB")
            input_ids, images_tensor, image_sizes = hm._build_prompt(img, rec.question)
            embeds, n_vis = hm._prepare_embeds(input_ids, images_tensor, image_sizes)
            vlo, vhi = hm.visual_range(input_ids, n_vis)
            vis_clean = embeds[0, vlo:vhi, :].float()

            g = torch.Generator(device=vis_clean.device)
            g.manual_seed(0)
            noise = torch.randn(vis_clean.shape, generator=g,
                                device=vis_clean.device, dtype=torch.float32) * sigma
            vis_noisy = vis_clean + noise

            cos_self = F.cosine_similarity(vis_noisy, vis_clean, dim=-1).mean().item()

            other_ids = [rid for rid in clean_embeds if rid != rec.id]
            if other_ids:
                device = vis_noisy.device
                other_means = torch.stack(
                    [clean_embeds[oid].mean(0) for oid in other_ids]
                ).to(device)
                noisy_mean = vis_noisy.mean(0, keepdim=True)
                cos_to_others = F.cosine_similarity(noisy_mean, other_means, dim=-1)
                cos_other_max = cos_to_others.max().item()
                nearest_other_id = other_ids[int(cos_to_others.argmax())]
                noisy_closer = bool(cos_other_max > cos_self)
            else:
                cos_other_max = float("nan")
                nearest_other_id = None
                noisy_closer = False

            row = {
                "test": "B_off_manifold",
                "backend": backend,
                "record_id": rec.id,
                "sigma": sigma,
                "n_vis_tokens": n_vis,
                "cos_sim_noisy_to_clean_self":    round(cos_self, 4),
                "cos_sim_noisy_to_nearest_other": round(cos_other_max, 4),
                "nearest_other_record_id":        nearest_other_id,
                "noisy_closer_to_other":          noisy_closer,
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            print(f"  [{rec.id}] cos_self={cos_self:.4f}  cos_other={cos_other_max:.4f}  "
                  f"closer={noisy_closer}")

    finally:
        fout.close()


# ── report ────────────────────────────────────────────────────────────────────

def print_report(rows: list[dict]) -> None:
    b_rows = [r for r in rows if r.get("test") == "B_off_manifold"]
    if not b_rows:
        print("No Sub-test B rows found.")
        return
    n = len(b_rows)
    cos_self  = sum(r["cos_sim_noisy_to_clean_self"]    for r in b_rows) / n
    cos_other = sum(r["cos_sim_noisy_to_nearest_other"] for r in b_rows) / n
    pct = 100 * sum(1 for r in b_rows if r["noisy_closer_to_other"]) / n
    backend = b_rows[0].get("backend", "?")
    sigma   = b_rows[0].get("sigma", "?")
    print(f"\n{'='*60}")
    print(f"SUB-TEST B: off-manifold substitution  backend={backend}  σ={sigma}")
    print(f"{'='*60}")
    print(f"  n                                    {n}")
    print(f"  mean cos_sim(noisy, clean_self)      {cos_self:.4f}")
    print(f"  mean cos_sim(noisy, nearest_other)   {cos_other:.4f}")
    print(f"  % noisy closer to other record       {pct:.1f}%")
    verdict = "SUBSTITUTION" if pct > 80 else ("PARTIAL" if pct > 20 else "DEGRADATION")
    print(f"\n  Verdict: {verdict}")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="llava_vq",
                    choices=["llava_vq", "vilau", "unitok", "seed", "showo", "emu3",
                             "chameleon", "anole", "liquid", "lumina_mgpt"])
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--tokenizer_path", default=None,
                    help="[unitok] Path to unitok_tokenizer.pth")
    ap.add_argument("--projector_ckpt", default=None,
                    help="[llava_vq] Path to VQLinearProjector checkpoint")
    ap.add_argument("--codebook_size", type=int, default=16384,
                    help="[llava_vq] Codebook size K (default 16384)")
    ap.add_argument("--vq_path", default=None,
                    help="[showo/emu3] HF ID or path to VQ tokenizer")
    ap.add_argument("--sweep", required=False, default=None,
                    help="Sweep JSONL to select high-restoration records from")
    ap.add_argument("--source", default="pope", choices=["pope", "naturalbench", "all"],
                    help="Which source to use from sweep (default: pope; use naturalbench for Show-o)")
    ap.add_argument("--n_records", type=int, default=200,
                    help="Number of target records to probe")
    ap.add_argument("--sigma", type=float, required=False, default=None,
                    help="Post-projector noise σ (use calibrated value per model)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        rows = []
        if out_path.exists():
            with out_path.open() as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        print_report(rows)
        return

    if not args.model_path:
        ap.error("--model_path is required unless --report_only")
    if args.sigma is None:
        ap.error("--sigma is required (use calibrated value: vilau=1.0, unitok=0.2, seed=0.033, showo=0.1, emu3=0.01, llava_vq=2.0)")
    if not args.sweep:
        ap.error("--sweep is required (path to sweep JSONL)")

    target_ids, pool_ids = _select_records(args.sweep, args.source, args.n_records)

    from probe import load_cache
    records, _, _ = load_cache()
    rec_map = {r.id: r for r in records}

    target_records = [rec_map[rid] for rid in target_ids if rid in rec_map]
    pool_records   = [rec_map[rid] for rid in pool_ids   if rid in rec_map]
    print(f"target={len(target_records)}  pool={len(pool_records)}")

    print(f"Loading {args.backend} model ...")
    hm = _load_hm(args.backend, args.model_path,
                  args.tokenizer_path, args.projector_ckpt, args.vq_path,
                  codebook_size=args.codebook_size)

    run_probe(hm, target_records, pool_records,
              sigma=args.sigma, backend=args.backend, out_path=out_path)

    rows = []
    with out_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print_report(rows)


if __name__ == "__main__":
    main()
