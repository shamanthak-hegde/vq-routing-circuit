"""Where does VILA-U *look* when it falsely confirms an absent object?

For VILA-U false-positive POPE records, capture the prefill attention from the
decision position (prompt_last) to the 256 visual tokens during the yes/no
question, reshape to the 16x16 patch grid, and report/overlay the peak region.
This answers "where is it seeing the object" at the attention level, since
VILA-U's free-text generation is too degenerate to localize reliably.

Usage:
    module load mamba; source activate vila-u
    python -m probe.benchmarks.vilau_hallucination_attention \
        --model_path mit-han-lab/vila-u-7b-256 \
        --records_json results/pope_records_cache_50.json \
        --fp_jsonl results/halluc_followup_vilau.jsonl \
        --out_dir figures/vilau_halluc_attn
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


def _grid_region_name(row: int, col: int, n: int) -> str:
    v = "top" if row < n / 3 else ("bottom" if row >= 2 * n / 3 else "middle")
    h = "left" if col < n / 3 else ("right" if col >= 2 * n / 3 else "center")
    return f"{v}-{h}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="mit-han-lab/vila-u-7b-256")
    p.add_argument("--records_json", required=True)
    p.add_argument("--fp_jsonl", required=True)
    p.add_argument("--out_dir", default="figures/vilau_halluc_attn")
    p.add_argument("--n_cases", type=int, default=6)
    p.add_argument("--layers", default="16,24",
                   help="comma-separated layer indices to average for the map")
    p.add_argument("--condition", choices=["fp", "tp"], default="fp",
                   help="fp: false-positive hallucinations (gt=no, pred=yes); "
                        "tp: true positives (gt=yes, pred=yes) as a grounding control")
    args = p.parse_args()

    import torch
    from probe.benchmarks.run_bench import _load_hook_manager

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_ids = [int(x) for x in args.layers.split(",")]

    rec_by_id = {r["id"]: r for r in json.load(open(args.records_json))}
    rows = [json.loads(l) for l in open(args.fp_jsonl)]
    if args.condition == "fp":
        fp = [r for r in rows if r["false_positive_hallucination"]]
    else:  # tp: true positives — object present and correctly confirmed
        fp = [r for r in rows if r["gt_answer"] == "yes" and r["pred"] == "yes"]
    fp = fp[:args.n_cases]

    hm, tokenizer = _load_hook_manager(SimpleNamespace(
        backend="vilau", model_path=args.model_path,
        tokenizer_path=None, vq_path=None, projector_ckpt=None,
        max_image_size=256,
    ))
    hm.capture_attention_weights = True

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_mpl = True
    except Exception:
        have_mpl = False

    print(f"layers averaged: {layer_ids}\n")
    for r in fp:
        rec = rec_by_id[r["record_id"]]
        obj = r["object"]
        img = Image.open(rec["image_path"]).convert("RGB")
        cap = hm.run_prefill(img, rec["question"])

        aw = cap.attn_weights            # (n_layers, heads, S, S)
        vs, ve = cap.token_index.visual_range
        pl = cap.token_index.prompt_last
        n_vis = ve - vs
        grid = int(math.sqrt(n_vis))

        # prompt_last → visual attention, averaged over heads and chosen layers
        sel = aw[layer_ids][:, :, pl, vs:ve].float()   # (n_layers_sel, heads, n_vis)
        att = sel.mean(dim=(0, 1)).numpy()             # (n_vis,)
        # fraction of the decision position's total attention that lands on image
        img_mass = float(aw[layer_ids][:, :, pl, vs:ve].float().sum(-1).mean())

        att_grid = att.reshape(grid, grid)
        peak = np.unravel_index(int(att_grid.argmax()), att_grid.shape)
        region = _grid_region_name(peak[0], peak[1], grid)
        top3 = np.dstack(np.unravel_index(
            np.argsort(att_grid.ravel())[::-1][:3], att_grid.shape))[0]

        print(f"{r['record_id']:<12} {obj:<14} img-attn-mass={img_mass:.3f}  "
              f"peak=({peak[0]},{peak[1]})/{grid} [{region}]  "
              f"top3={[tuple(int(x) for x in c) for c in top3]}")

        if have_mpl:
            heat = att_grid / (att_grid.max() + 1e-8)
            heat_img = np.array(Image.fromarray((heat * 255).astype(np.uint8))
                                .resize(img.size, Image.BILINEAR)) / 255.0
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img); ax[0].set_title("image"); ax[0].axis("off")
            ax[1].imshow(img)
            ax[1].imshow(heat_img, cmap="jet", alpha=0.5)
            ax[1].set_title(f"VILA-U attn → '{obj}' [{region}]")
            ax[1].axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"{r['record_id']}_{obj.replace(' ', '_')}.png",
                        dpi=90)
            plt.close(fig)

    if have_mpl:
        print(f"\nOverlays written to {out_dir}/")


if __name__ == "__main__":
    main()
