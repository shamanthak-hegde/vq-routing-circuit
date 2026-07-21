"""
Download LLaVA-Pretrain-558K and create a 50k-entry subset.

LLaVA-Pretrain-558K (liuhaotian/LLaVA-Pretrain) contains:
  - blip_laion_cc_sbu_558k.json  — list of {id, image, conversations}
  - images/                      — JPEG/PNG images

The subset is deterministically sampled (seeded) and written as
data/llava_pretrain/subset_50k.json with a matching subset_50k_images/
symlink directory so the train script can use either.

Usage
-----
    python -m probe.training.download_pretrain \\
        --out_dir data/llava_pretrain \\
        --n_subset 50000 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random


def _main() -> None:
    parser = argparse.ArgumentParser(description="Download LLaVA-Pretrain-558K + create 50k subset")
    parser.add_argument("--out_dir", default="data/llava_pretrain",
                        help="Directory to download into (default: data/llava_pretrain)")
    parser.add_argument("--n_subset", type=int, default=50000,
                        help="Subset size (default 50000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo_id", default="liuhaotian/LLaVA-Pretrain",
                        help="HuggingFace repo ID (default: liuhaotian/LLaVA-Pretrain)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub not installed. Run: pip install huggingface_hub")

    print(f"Downloading {args.repo_id} → {args.out_dir}  (this may take ~30 GB of disk) ...")
    local_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.out_dir,
        ignore_patterns=["*.md", "*.gitattributes"],
    )
    print(f"Downloaded to: {local_dir}")

    # Locate the main JSON manifest
    manifest_candidates = [
        os.path.join(local_dir, "blip_laion_cc_sbu_558k.json"),
        os.path.join(local_dir, "llava_pretrain_558k.json"),
    ]
    manifest_path = None
    for c in manifest_candidates:
        if os.path.exists(c):
            manifest_path = c
            break
    if manifest_path is None:
        jsons = [f for f in os.listdir(local_dir) if f.endswith(".json")]
        if jsons:
            manifest_path = os.path.join(local_dir, jsons[0])
        else:
            raise FileNotFoundError(f"No JSON manifest found in {local_dir}; check repo structure.")

    print(f"Found manifest: {manifest_path}")
    with open(manifest_path) as f:
        full = json.load(f)
    print(f"Total entries: {len(full):,}")

    rng = random.Random(args.seed)
    subset = rng.sample(full, min(args.n_subset, len(full)))

    subset_path = os.path.join(args.out_dir, f"subset_{args.n_subset // 1000}k.json")
    with open(subset_path, "w") as f:
        json.dump(subset, f)
    print(f"Subset ({len(subset):,} entries) → {subset_path}")

    # Record where images live for the train script
    # LLaVA-Pretrain ships images as images.zip rather than an extracted directory.
    image_zip = os.path.join(local_dir, "images.zip")
    image_dir = os.path.join(local_dir, "images")
    if os.path.isfile(image_zip) and not os.path.isdir(image_dir):
        print(f"Note: images are in {image_zip} (not extracted).")
        print("Pass --image_dir to that zip path, or extract it with:")
        print(f"  unzip {image_zip} -d {local_dir}")
    meta = {
        "repo_id": args.repo_id,
        "local_dir": local_dir,
        "manifest": manifest_path,
        "image_dir": image_dir,
        "image_zip": image_zip if os.path.isfile(image_zip) else None,
        "n_total": len(full),
        "subset_path": subset_path,
        "n_subset": len(subset),
        "seed": args.seed,
    }
    meta_path = os.path.join(args.out_dir, "download_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata → {meta_path}")
    print(f"\nImage directory: {image_dir}")
    print("Done.")


if __name__ == "__main__":
    _main()
