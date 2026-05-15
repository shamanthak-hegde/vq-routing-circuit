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
        unitok_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "UniTok")
        )
        if unitok_root not in sys.path:
            sys.path.insert(0, unitok_root)
        from model.builder import load_pretrained_model
        from mm_utils import get_model_name_from_path
        from models.unitok import UniTok
        from probe.hooks import UniTokHookManager
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, _, _ = load_pretrained_model(
            args.model_path, None, model_name, attn_implementation="eager"
        )
        model.eval()
        device = next(model.parameters()).device
        import yaml
        vae_cfg_path = os.path.join(unitok_root, "models", "configs", "unitok_tokenizer.yaml")
        with open(vae_cfg_path) as f:
            vae_cfg = yaml.safe_load(f)
        ckpt = torch.load(args.tokenizer_path, map_location="cpu")
        vq_model = UniTok(vae_cfg)
        vq_model.load_state_dict(ckpt["trainer"]["unitok"])
        vq_model.to(device).eval()
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

    raise ValueError(f"Unknown backend {args.backend!r}")


# ── Direction computation ─────────────────────────────────────────────────────

def _rank1_direction(diff_matrix: torch.Tensor) -> torch.Tensor:
    """Rank-1 PCA of diff_matrix (n_demos, hidden) → direction (hidden,) float32.

    Uses SVD on the mean-centered matrix; returns the leading right singular
    vector (first PC in the column space of the diffs), sign-aligned so that
    the mean projection is positive (direction points toward the mean diff).
    """
    X = diff_matrix.float()
    X = X - X.mean(dim=0, keepdim=True)
    # economy SVD: U (n,1), S (1,), Vh (1, hidden)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    pc = Vh[0]  # (hidden,)
    # Align sign: positive projection onto mean diff
    mean_diff = diff_matrix.float().mean(dim=0)
    if torch.dot(pc, mean_diff) < 0:
        pc = -pc
    return pc


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
    for l in range(n_layers):
        diff_mat = torch.stack(diffs[l], dim=0)  # (done, hidden)
        directions.append(_rank1_direction(diff_mat))

    return torch.stack(directions, dim=0)  # (n_layers, hidden)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute VTI textual steering directions for a VLM (N-047)"
    )
    parser.add_argument("--backend", required=True,
                        choices=["vilau", "llava", "unitok", "qwen3vl"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None,
                        help="[unitok] path to tokenizer checkpoint")
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
    pope_records = [r for r in records if r.source == "pope"]

    if not pope_records:
        raise RuntimeError("No POPE records in probe cache")

    print(f"POPE records available: {len(pope_records)}, using first {args.n_demos}")

    direction = compute_directions(hm, pope_records, sigma, args.n_demos, args.seed)

    torch.save(direction, out_path)
    print(f"\nDirection tensor saved → {out_path}  shape={list(direction.shape)}")

    meta = {
        "backend": args.backend,
        "model_path": args.model_path,
        "sigma": sigma,
        "n_demos": args.n_demos,
        "seed": args.seed,
        "direction_shape": list(direction.shape),
        "sweep_meta_source": sweep_meta_source,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Meta written → {meta_path}")


if __name__ == "__main__":
    main()
