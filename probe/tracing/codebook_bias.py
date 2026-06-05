"""Codebook entry yes-bias analysis for VQ-VLM backends (N-069/N-078).

For each active codebook entry, measures what fraction of POPE prefills where
that entry appears in the visual tokens predict "yes".

Usage
-----
    # VILA-U (vila-u env):
    python -m probe.tracing.codebook_bias --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 --n_records 200 \\
        --out results/codebook_bias_vilau.json

    # Chameleon (sae env):
    python -m probe.tracing.codebook_bias --backend chameleon \\
        --model_path facebook/chameleon-7b --n_records 200 \\
        --out results/codebook_bias_chameleon.json

    # Liquid (sae env):
    python -m probe.tracing.codebook_bias --backend liquid \\
        --model_path Junfeng5/Liquid_V1_7B --n_records 200 \\
        --out results/codebook_bias_liquid.json

    # Report only (no model needed):
    python -m probe.tracing.codebook_bias --report_only \\
        --out results/codebook_bias_chameleon.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
from pathlib import Path

import torch
from PIL import Image


@torch.no_grad()
def compute_codebook_bias(
    hm,
    records,
    n_records: int = 500,
    seed: int = 0,
) -> dict:
    """Measure per-codebook-entry yes-prediction rate on POPE records.

    Parameters
    ----------
    hm        : VilaUHookManager (must have working _get_vq_codes())
    records   : ProbeRecord list — POPE gaussian_noise type used (clean images)
    n_records : how many records to process
    seed      : RNG seed

    Returns
    -------
    dict with:
      'entry_yes_count'   : {str(code_id): yes_count}
      'entry_total_count' : {str(code_id): total_count}
      'entry_yes_rate'    : {str(code_id): float}
      'n_records'         : int
      'n_active_entries'  : int
      'mean_yes_rate'     : float  (should be ~50% for unbiased)
      'max_yes_rate'      : float
      'min_yes_rate'      : float
      'yes_biased_entries': list of code_ids with yes_rate > 0.75
    """
    pope = [r for r in records if r.corruption_mode == "gaussian_noise"]
    rng = random.Random(seed)
    sample = rng.sample(pope, min(n_records, len(pope)))

    # Resolve tokenizer: some backends store it as hm.tokenizer, others via hm.processor
    tok = getattr(hm, "tokenizer", None) or hm.processor.tokenizer
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]

    entry_yes: dict[int, int] = {}
    entry_total: dict[int, int] = {}

    print(f"[codebook_bias] Processing {len(sample)} POPE records ...")
    for i, rec in enumerate(sample, 1):
        img = Image.open(rec.image_path).convert("RGB")

        # Get VQ code indices for this image (clean image, no noise)
        codes_info = hm._get_vq_codes(img)
        if codes_info is None:
            continue
        # levels[0]: (N,) int64 code indices for each spatial token
        codes = codes_info["levels"][0].tolist()  # list of int code indices
        unique_codes = set(codes)

        # Get model's yes/no prediction on this clean image
        cap = hm.run_prefill(img, rec.question)
        last_logits = cap.logits[cap.token_index.prompt_last].float()
        pred_yes = int(last_logits[yes_id] > last_logits[no_id])

        for code_id in unique_codes:
            entry_total[code_id] = entry_total.get(code_id, 0) + 1
            if pred_yes:
                entry_yes[code_id] = entry_yes.get(code_id, 0) + 1

        if i % 50 == 0:
            print(f"  {i}/{len(sample)}", flush=True)

    # Compute per-entry yes-rates (only for entries seen ≥ 5 times)
    entry_yes_rate = {}
    for code_id, total in entry_total.items():
        if total >= 5:
            entry_yes_rate[code_id] = entry_yes.get(code_id, 0) / total

    if not entry_yes_rate:
        return {"error": "no entries seen ≥ 5 times", "n_records": len(sample)}

    rates = list(entry_yes_rate.values())
    yes_biased = [cid for cid, r in entry_yes_rate.items() if r > 0.75]
    no_biased = [cid for cid, r in entry_yes_rate.items() if r < 0.25]

    return {
        "entry_yes_count":    {str(k): v for k, v in entry_yes.items()},
        "entry_total_count":  {str(k): v for k, v in entry_total.items()},
        "entry_yes_rate":     {str(k): round(v, 4) for k, v in entry_yes_rate.items()},
        "n_records":          len(sample),
        "n_active_entries":   len(entry_yes_rate),
        "mean_yes_rate":      round(sum(rates) / len(rates), 4),
        "max_yes_rate":       round(max(rates), 4),
        "min_yes_rate":       round(min(rates), 4),
        "yes_biased_entries": sorted([str(c) for c in yes_biased]),
        "no_biased_entries":  sorted([str(c) for c in no_biased]),
        "n_yes_biased":       len(yes_biased),
        "n_no_biased":        len(no_biased),
    }


def plot_codebook_bias(result: dict, out_path: str | Path) -> None:
    """Histogram of per-entry yes-rates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = [float(v) for v in result["entry_yes_rate"].values()]
    if not rates:
        print("No rates to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rates, bins=20, range=(0, 1), color="#e74c3c", edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.5, label="chance (0.5)")
    ax.axvline(0.75, color="#e67e22", linestyle=":", linewidth=1.5, label="yes-biased threshold (0.75)")
    ax.set_xlabel("Per-entry yes-prediction rate")
    ax.set_ylabel("Number of active codebook entries")
    backend = result.get("backend", "vilau")
    ax.set_title(
        f"{backend} codebook entry yes-bias distribution\n"
        f"n_entries={result['n_active_entries']}, "
        f"yes-biased (>0.75)={result['n_yes_biased']}, "
        f"mean={result['mean_yes_rate']:.2f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def print_report(result: dict) -> None:
    print("\n=== Codebook yes-bias report ===")
    print(f"  Records processed:  {result['n_records']}")
    print(f"  Active entries:     {result['n_active_entries']}")
    print(f"  Mean yes-rate:      {result['mean_yes_rate']:.3f}  (0.5 = unbiased)")
    print(f"  Max yes-rate:       {result['max_yes_rate']:.3f}")
    print(f"  Min yes-rate:       {result['min_yes_rate']:.3f}")
    print(f"  Yes-biased (>0.75): {result['n_yes_biased']} entries")
    print(f"  No-biased  (<0.25): {result['n_no_biased']} entries")
    if result.get("yes_biased_entries"):
        print(f"  Top yes-biased codes: {result['yes_biased_entries'][:10]}")


def _load_hm(args):
    """Load the hook manager for the requested backend."""
    backend = args.backend

    if backend == "vilau":
        _VILAU_ROOT = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
        )
        if _VILAU_ROOT not in sys.path:
            sys.path.insert(0, _VILAU_ROOT)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks import VilaUHookManager
        print(f"Loading vilau {args.model_path} ...")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, attn_implementation="eager"
        )
        model.eval()
        return VilaUHookManager(model, tokenizer, image_processor)

    elif backend == "chameleon":
        import torch
        from transformers import ChameleonForConditionalGeneration, ChameleonProcessor
        from probe.hooks.chameleon_hf import ChameleonHFHookManager
        print(f"Loading chameleon {args.model_path} ...")
        processor = ChameleonProcessor.from_pretrained(args.model_path)
        model = ChameleonForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        ).eval()
        return ChameleonHFHookManager(model, processor)

    elif backend == "liquid":
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
        print(f"Loading liquid {args.model_path} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
        ).eval()
        _vqgan_cfg = args.tokenizer_path or os.path.join(
            _chameleon_root, "data", "tokenizer", "vqgan.yaml"
        )
        _vqgan_ckpt = args.vq_path or os.path.join(
            _chameleon_root, "data", "tokenizer", "vqgan.ckpt"
        )
        image_tokenizer = ImageTokenizer(
            cfg_path=_vqgan_cfg, ckpt_path=_vqgan_ckpt, device="cuda:0"
        )
        return LiquidHookManager(model, tokenizer, image_tokenizer)

    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Codebook entry yes-bias analysis (N-069/N-078)"
    )
    parser.add_argument("--backend", default="vilau",
                        choices=["vilau", "chameleon", "liquid"],
                        help="VLM backend to analyse")
    parser.add_argument("--model_path", default="mit-han-lab/vila-u-7b-256")
    parser.add_argument("--tokenizer_path", default=None,
                        help="(liquid only) path to vqgan.yaml")
    parser.add_argument("--vq_path", default=None,
                        help="(liquid only) path to vqgan.ckpt")
    parser.add_argument("--n_records", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--figure", default=None, help="Optional figure path (.png)")
    parser.add_argument("--report_only", action="store_true",
                        help="Load existing JSON and print report + figure without running model")
    args = parser.parse_args()

    out = Path(args.out)

    if args.report_only:
        result = json.loads(out.read_text())
        print_report(result)
        if args.figure:
            Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
            plot_codebook_bias(result, args.figure)
        return

    hm = _load_hm(args)

    from probe import load_cache
    records, _, _ = load_cache()

    result = compute_codebook_bias(hm, records, n_records=args.n_records, seed=args.seed)
    result["backend"] = args.backend
    print_report(result)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Results written to {out}")

    if args.figure:
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        plot_codebook_bias(result, args.figure)


if __name__ == "__main__":
    main()
