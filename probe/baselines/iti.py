"""Inference-Time Intervention (ITI) direction computation — Li et al. NeurIPS 2023.

ITI trains a linear probe on head activations to find the "truthful direction",
then steers generation by adding scaled direction vectors at inference time.

Here we adapt ITI for VLM yes-bias correction:
  - Training: collect prompt_last residuals from POPE records (clean image).
    Label 1 = model says "yes", label 0 = model says "no".
    Fit logistic regression at each layer; direction = probe weight vector (unit norm).
  - Inference: add -alpha * direction to the residual at selected layers.
    Because the existing vti_textual mechanism already handles additive direction
    steering (probe/tracing/head_knockout.py → VTITextualSteer), the ITI direction
    can be used directly with --knockout_mode vti_textual.

Usage
-----
    # Compute ITI directions and save to .pt (n_layers, hidden)
    source activate vila-u
    python -m probe.baselines.iti \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --n_demos 200 \\
        --out results/iti_direction_vilau.pt

    # Use at inference (same as VTI):
    python -m probe.benchmarks.run_bench --bench pope_full --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --knockout_mode vti_textual \\
        --vti_direction results/iti_direction_vilau.pt \\
        --alpha 0.005 \\
        --out results/bench_pope_vilau_iti.jsonl

Difference from VTI
--------------------
VTI direction: mean(residual[corrupt] - residual[clean]) — discriminates by
               image-identity perturbation.
ITI direction: logistic-regression weight from residual[clean] → predicts_yes
               — discriminates by the model's output bias, not by image identity.
Both target the "yes direction" in activation space but from different angles.

Reference: Li et al. "Inference-Time Intervention: Eliciting Truthful Answers
           from a Language Model", NeurIPS 2023.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image


# ── Direction computation ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_iti_directions(
    hm,
    records,
    n_demos: int = 200,
    seed: int = 0,
) -> torch.Tensor:
    """Compute per-layer ITI directions (logistic-probe weight vectors).

    For each decoder layer, fits a logistic regression on prompt_last residuals
    with label = (model_predicts_yes).  Returns unit-norm weight vectors.

    Parameters
    ----------
    hm        : VLMHookManager (eval mode)
    records   : POPE ProbeRecord list (gaussian_noise type — only POPE has yes-bias)
    n_demos   : number of clean-image prefill passes (200 recommended)
    seed      : RNG seed for reproducibility

    Returns
    -------
    directions : (n_layers, hidden) float32 tensor
    """
    import random
    from sklearn.linear_model import LogisticRegression

    pope = [r for r in records if r.corruption_mode == "gaussian_noise"]
    if not pope:
        raise ValueError("No POPE (gaussian_noise) records found.")
    rng = random.Random(seed)
    sample = rng.sample(pope, min(n_demos, len(pope)))

    yes_ids = []
    for tok in ("yes", "Yes", "YES"):
        toks = hm.tokenizer.encode(tok, add_special_tokens=False)
        if toks:
            yes_ids.append(toks[0])
    if not yes_ids:
        raise ValueError("Could not find yes token id")
    yes_id = yes_ids[0]

    all_residuals: list[list[torch.Tensor]] = []  # [record_idx][layer]
    labels: list[int] = []

    print(f"[ITI] Collecting {len(sample)} clean-image prefills …")
    t0 = time.time()
    for i, rec in enumerate(sample, 1):
        img = Image.open(rec.image_path).convert("RGB")
        cap = hm.run_prefill(img, rec.question)
        prompt_last = cap.token_index.prompt_last
        # Collect prompt_last hidden state at every layer
        per_layer = [cap.residual[l, prompt_last, :].float().cpu()
                     for l in range(cap.residual.shape[0])]
        all_residuals.append(per_layer)
        # Label: did the model predict yes?
        last_logits = cap.logits[prompt_last].float()
        pred_yes = int(last_logits[yes_id] > 0)
        labels.append(pred_yes)
        if i % 20 == 0:
            print(f"  {i}/{len(sample)}  ({time.time()-t0:.0f}s)", flush=True)

    n_layers = len(all_residuals[0])
    hidden = all_residuals[0][0].shape[0]
    print(f"[ITI] Fitting {n_layers} logistic probes …")

    directions = torch.zeros(n_layers, hidden)
    for l in range(n_layers):
        X = torch.stack([all_residuals[i][l] for i in range(len(sample))]).numpy()
        y = labels
        if len(set(y)) < 2:
            # degenerate: all same label — direction is zero
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X, y)
        w = torch.from_numpy(clf.coef_[0]).float()
        w = w / (w.norm() + 1e-8)
        directions[l] = w

    return directions


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_hook_manager(args: argparse.Namespace):
    if args.backend == "vilau":
        root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
        )
        if root not in sys.path:
            sys.path.insert(0, root)
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
        root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "UniTok")
        )
        liquid = os.path.join(root, "eval", "liquid")
        for p in (root, liquid):
            if p not in sys.path:
                sys.path.insert(0, p)
        from model.builder import load_pretrained_model
        from mm_utils import get_model_name_from_path
        from models.unitok import UniTok
        from utils.config import Args as UniTokArgs
        from probe.hooks.unitok import UniTokHookManager
        import torch as _torch
        model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
        tokenizer, model, _, _ = load_pretrained_model(
            os.path.expanduser(args.model_path), None, model_name,
            attn_implementation="eager",
        )
        model.eval()
        device = next(model.parameters()).device
        ckpt = _torch.load(os.path.expanduser(args.tokenizer_path), map_location="cpu")
        cfg = UniTokArgs()
        cfg.load_state_dict(ckpt["args"])
        vq_model = UniTok(cfg)
        vq_model.load_state_dict(ckpt["trainer"]["unitok"])
        vq_model.to(device).eval()
        del ckpt
        return UniTokHookManager(model, tokenizer, vq_model), tokenizer

    raise ValueError(f"Unknown backend {args.backend!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ITI direction (logistic-probe weight) for a VLM"
    )
    parser.add_argument("--backend", required=True,
                        choices=["vilau", "llava", "unitok"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None,
                        help="[unitok] path to unitok_tokenizer.pth")
    parser.add_argument("--n_demos", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True,
                        help="Output path for .pt direction file")
    args = parser.parse_args()

    hm, _ = _load_hook_manager(args)

    from probe import load_cache
    records = load_cache()

    directions = compute_iti_directions(hm, records, n_demos=args.n_demos, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(directions, out)
    print(f"ITI directions saved to {out}  shape={list(directions.shape)}")

    meta = {
        "backend": args.backend,
        "model_path": args.model_path,
        "n_demos": args.n_demos,
        "seed": args.seed,
        "shape": list(directions.shape),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Meta written to {meta_path}")


if __name__ == "__main__":
    main()
