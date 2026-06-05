"""Cross-model VTI direction transfer experiment (N-065).

Computes a 4×4 pairwise cosine-similarity matrix between VTI directions from
{VILA-U, UniTok, SEED, LLaVA-VQ}, then measures POPE accuracy when one model's
direction is applied to another.

The analysis tests C4's mechanistic claim: if VQ-class models share a common
"yes-bias direction" in activation space, pairwise cos-sim within VQ class
should be higher than across class (VQ vs continuous-projector).

Usage (requires pre-computed direction files)
---------------------------------------------
    # Step 1: make sure directions exist (use vti_calibrate.py for each model)
    # e.g. results/vti_direction_vilau_v2.pt, results/vti_direction_unitok.pt,
    #      results/vti_direction_seed.pt, results/vti_direction_llava_vq.pt

    # Step 2: compute similarity matrix
    python -m probe.tracing.direction_transfer \\
        --directions vilau:results/vti_direction_vilau_v2.pt \\
                     unitok:results/vti_direction_unitok.pt \\
                     seed:results/vti_direction_seed.pt \\
                     llava_vq:results/vti_direction_llava_vq.pt \\
        --out results/direction_transfer.json

    # Step 3 (optional, GPU): cross-apply a direction to a different model
    python -m probe.tracing.direction_transfer \\
        --cross_apply \\
        --source_direction results/vti_direction_vilau_v2.pt \\
        --source_name vilau \\
        --backend unitok --model_path ... --tokenizer_path ... \\
        --out results/direction_transfer_vilau_to_unitok.jsonl
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
import torch.nn.functional as F


# ── Similarity matrix ─────────────────────────────────────────────────────────

def compute_direction_similarity(
    directions: dict[str, torch.Tensor],
    layer_range: tuple[int, int] | None = None,
) -> dict:
    """Compute pairwise cosine similarity between direction tensors.

    Parameters
    ----------
    directions  : {name: (n_layers, hidden) tensor}
    layer_range : (start, end) inclusive; if None, uses all layers

    Returns
    -------
    dict with 'names', 'matrix' (list of lists), 'per_layer' (list of dicts)
    """
    names = sorted(directions.keys())
    n = len(names)

    # Normalise each direction to unit norm per layer
    normed = {}
    for name, d in directions.items():
        norms = d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed[name] = d / norms

    if layer_range is not None:
        lo, hi = layer_range
        for name in normed:
            normed[name] = normed[name][lo:hi + 1]

    n_layers = normed[names[0]].shape[0]

    # Global cos-sim (mean over layers)
    matrix = [[0.0] * n for _ in range(n)]
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            cs = (normed[ni] * normed[nj]).sum(dim=-1).mean().item()
            matrix[i][j] = round(cs, 4)

    # Per-layer cos-sim for selected pairs
    per_layer = []
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i >= j:
                continue
            cs_per_layer = (normed[ni] * normed[nj]).sum(dim=-1).tolist()
            per_layer.append({
                "pair": f"{ni}×{nj}",
                "values": [round(v, 4) for v in cs_per_layer],
            })

    return {"names": names, "matrix": matrix, "per_layer": per_layer}


# ── Cross-model direction application ────────────────────────────────────────

@torch.no_grad()
def cross_apply_direction(
    hm,
    tokenizer,
    records,
    direction: torch.Tensor,
    source_name: str,
    target_name: str,
    alpha: float = 0.005,
    out_path: Path | None = None,
) -> dict:
    """Apply a direction from model A to model B; return accuracy metrics.

    Parameters
    ----------
    hm          : VLMHookManager for target model
    records     : POPE records
    direction   : (n_layers, hidden) direction from source model
    source_name : label for the source model
    target_name : label for the target model
    alpha       : steering coefficient

    Returns
    -------
    metrics dict with 'source', 'target', 'alpha', 'pope_acc', 'yes_rate'
    """
    from probe.tracing.head_knockout import build_intervention

    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]
    pope = [r for r in records if r.corruption_mode == "gaussian_noise"]

    results = []
    direction_path_tmp = Path(f"/tmp/direction_transfer_{source_name}_to_{target_name}.pt")
    torch.save(direction, direction_path_tmp)

    intervention = build_intervention(
        "vti_textual", hm, 0, [], alpha,
        direction_path=str(direction_path_tmp)
    )

    from PIL import Image as _Image
    for rec in pope:
        img = _Image.open(rec.image_path).convert("RGB")
        with intervention:
            cap = hm.run_prefill(img, rec.question)
        last_logits = cap.logits[cap.token_index.prompt_last].float()
        pred = "yes" if last_logits[yes_id] > last_logits[no_id] else "no"
        results.append({"correct": pred == rec.answer, "pred": pred})

    direction_path_tmp.unlink(missing_ok=True)

    acc = sum(r["correct"] for r in results) / len(results)
    yes_rate = sum(1 for r in results if r["pred"] == "yes") / len(results)

    metrics = {
        "source": source_name,
        "target": target_name,
        "alpha": alpha,
        "n_records": len(results),
        "pope_acc": round(acc, 4),
        "yes_rate": round(yes_rate, 4),
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "\n".join(json.dumps(r) for r in results) + "\n"
        )

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-model VTI direction transfer analysis (N-065)"
    )
    subp = parser.add_subparsers(dest="mode", required=True)

    # --- similarity matrix mode
    sim_p = subp.add_parser("similarity",
                             help="Compute pairwise direction cos-sim matrix")
    sim_p.add_argument("--directions", nargs="+", required=True,
                       metavar="NAME:PATH",
                       help="name:path pairs, e.g. vilau:results/vti_direction_vilau_v2.pt")
    sim_p.add_argument("--layer_range", default=None,
                       help="Layer range 'start-end' (inclusive), e.g. '0-7'")
    sim_p.add_argument("--out", required=True,
                       help="Output JSON path")

    # --- cross-apply mode (GPU)
    cross_p = subp.add_parser("cross_apply",
                               help="Apply source direction to target model (GPU)")
    cross_p.add_argument("--source_direction", required=True,
                         help="Path to .pt direction file (n_layers, hidden)")
    cross_p.add_argument("--source_name", required=True)
    cross_p.add_argument("--backend", required=True,
                         choices=["vilau", "llava", "unitok"])
    cross_p.add_argument("--model_path", required=True)
    cross_p.add_argument("--tokenizer_path", default=None)
    cross_p.add_argument("--alpha", type=float, default=0.005)
    cross_p.add_argument("--out", required=True, help="Output JSONL + .meta.json")

    args = parser.parse_args()

    if args.mode == "similarity":
        directions = {}
        for item in args.directions:
            if ":" not in item:
                parser.error(f"--directions items must be name:path (got {item!r})")
            name, path = item.split(":", 1)
            directions[name] = torch.load(path, map_location="cpu")
            print(f"Loaded {name}: {list(directions[name].shape)}")

        layer_range = None
        if args.layer_range:
            lo, hi = args.layer_range.split("-")
            layer_range = (int(lo), int(hi))

        result = compute_direction_similarity(directions, layer_range=layer_range)

        # Print matrix
        names = result["names"]
        print("\nPairwise cosine similarity (mean over layers):")
        print("       " + "  ".join(f"{n:>8}" for n in names))
        for i, ni in enumerate(names):
            row = result["matrix"][i]
            print(f"{ni:>6}" + "  ".join(f"{v:>8.4f}" for v in row))

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nResults written to {out}")

    elif args.mode == "cross_apply":
        direction = torch.load(args.source_direction, map_location="cpu")
        print(f"Loaded source direction ({args.source_name}): {list(direction.shape)}")

        # Load target model
        hm, tokenizer = _load_backend(args)
        target_name = args.backend

        from probe import load_cache
        records, _, _ = load_cache()

        out_path = Path(args.out)
        metrics = cross_apply_direction(
            hm, tokenizer, records, direction,
            source_name=args.source_name,
            target_name=target_name,
            alpha=args.alpha,
            out_path=out_path,
        )
        print(json.dumps(metrics, indent=2))

        meta_path = out_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps({
            **metrics,
            "source_direction": args.source_direction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n")
        print(f"Meta: {meta_path}")


def _load_backend(args):
    if args.backend == "vilau":
        root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "vila-u"))
        if root not in sys.path:
            sys.path.insert(0, root)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks import VilaUHookManager
        tokenizer, model, ip, _ = load_pretrained_model(args.model_path, attn_implementation="eager")
        model.eval()
        return VilaUHookManager(model, tokenizer, ip), tokenizer

    if args.backend == "llava":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA"))
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.hooks import LlavaHookManager
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, ip, _ = load_pretrained_model(args.model_path, model_base=None, model_name=model_name)
        model.eval()
        return LlavaHookManager(model, tokenizer, ip), tokenizer

    if args.backend == "unitok":
        root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "UniTok"))
        liquid = os.path.join(root, "eval", "liquid")
        for p in (root, liquid):
            if p not in sys.path:
                sys.path.insert(0, p)
        from model.builder import load_pretrained_model
        from mm_utils import get_model_name_from_path
        from models.unitok import UniTok
        from utils.config import Args as UniTokArgs
        from probe.hooks.unitok import UniTokHookManager
        model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
        tokenizer, model, _, _ = load_pretrained_model(
            os.path.expanduser(args.model_path), None, model_name, attn_implementation="eager")
        model.eval()
        device = next(model.parameters()).device
        ckpt = torch.load(os.path.expanduser(args.tokenizer_path), map_location="cpu")
        cfg = UniTokArgs(); cfg.load_state_dict(ckpt["args"])
        vq_model = UniTok(cfg); vq_model.load_state_dict(ckpt["trainer"]["unitok"])
        vq_model.to(device).eval()
        return UniTokHookManager(model, tokenizer, vq_model), tokenizer

    raise ValueError(f"Unknown backend {args.backend!r}")


if __name__ == "__main__":
    main()
