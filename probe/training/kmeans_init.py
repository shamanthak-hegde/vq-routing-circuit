"""
K-means codebook initialisation for VQLinearProjector (N-056).

Runs a single k-means pass on CLIP features from a sample of LLaVA-Pretrain
images to initialise the VQ codebook before STE training.  Random init leads
to slow convergence and high codebook collapse risk with 1-2k steps.

Usage
-----
    source activate sae
    python -m probe.training.kmeans_init \\
        --data_json  data/llava_pretrain/subset_50k.json \\
        --image_dir  data/llava_pretrain/images \\
        --model_path liuhaotian/llava-v1.6-vicuna-7b \\
        --n_samples  10000 \\
        --codebook_size 16384 \\
        --code_dim   32 \\
        --out        checkpoints/codebook_init.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import io
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


_LLAVA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
)


def _open_image(src, rel_path: str) -> Image.Image | None:
    """Open one image from either a directory root or an open ZipFile."""
    try:
        if isinstance(src, zipfile.ZipFile):
            data = src.read(rel_path)
            return Image.open(io.BytesIO(data)).convert("RGB")
        else:
            return Image.open(os.path.join(src, rel_path)).convert("RGB")
    except Exception:
        return None


@torch.no_grad()
def collect_clip_features(
    model,
    rel_paths: list[str],
    src,          # str (directory root) or zipfile.ZipFile
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Return raw CLIP patch features (N_tokens_total, mm_hidden_size) on CPU.

    `src` is either a directory path (str) or an already-open ZipFile.
    Uses the tower's CLIPImageProcessor for standard 336×336 encoding —
    avoids process_images whose anyres output cannot be batched as a tensor.
    """
    tower = model.get_model().get_vision_tower()
    clip_proc = tower.image_processor

    # Quick existence check on first few entries
    if isinstance(src, str):
        n_exist = sum(1 for p in rel_paths[:20] if os.path.exists(os.path.join(src, p)))
        print(f"  Path check (first 20): {n_exist}/20 exist  [src={src}]")
        if n_exist == 0:
            print(f"  Sample path: {os.path.join(src, rel_paths[0])}")
            raise RuntimeError(
                "No images found. --image_dir must point to the directory "
                "that directly contains the subdirectories listed in the manifest "
                f"(e.g. '00217/002170933.jpg'). Tried: {os.path.join(src, rel_paths[0])}"
            )

    feats: list[torch.Tensor] = []
    n_load_errors = 0
    n_tower_errors = 0
    for start in range(0, len(rel_paths), batch_size):
        batch = rel_paths[start:start + batch_size]
        images = [img for p in batch if (img := _open_image(src, p)) is not None]
        n_load_errors += len(batch) - len(images)
        if not images:
            continue
        try:
            pixel_values = clip_proc(images, return_tensors="pt")["pixel_values"]
            pixel_values = pixel_values.to(device, dtype=torch.float16)
            out = tower(pixel_values)               # (B, n_patches, mm_hidden_size)
            feats.append(out.reshape(-1, out.shape[-1]).cpu().float())
        except Exception as e:
            n_tower_errors += 1
            if n_tower_errors <= 3:
                print(f"  [tower error] batch {start}: {e}", flush=True)
            continue
        if (start // batch_size) % 20 == 0:
            n_done = min(start + batch_size, len(rel_paths))
            print(f"  CLIP features: {n_done}/{len(rel_paths)} images, "
                  f"{sum(f.shape[0] for f in feats):,} tokens", flush=True)

    if not feats:
        raise RuntimeError(
            f"No CLIP features collected — load_errors={n_load_errors}, "
            f"tower_errors={n_tower_errors}."
        )
    if n_load_errors or n_tower_errors:
        print(f"  Skipped: {n_load_errors} load errors, {n_tower_errors} tower errors.")
    return torch.cat(feats, dim=0)


def _project_down(feats: torch.Tensor, down: torch.nn.Module) -> torch.Tensor:
    """Apply the down projection in chunks to avoid OOM."""
    device = next(down.parameters()).device
    chunks = []
    chunk_size = 4096
    for i in range(0, feats.shape[0], chunk_size):
        chunks.append(down(feats[i:i + chunk_size].to(device)).cpu())
    return torch.cat(chunks, dim=0).float()


def kmeans(
    data: torch.Tensor,
    K: int,
    n_iters: int = 50,
    seed: int = 42,
    chunk: int = 8192,
) -> torch.Tensor:
    """Lloyd k-means with chunked distance computation (avoids N×K OOM).

    Computes assignments in `chunk`-row slices so peak memory stays at
    chunk×K×4 bytes (~512 MB for chunk=8192, K=16384) rather than N×K.
    """
    torch.manual_seed(seed)
    N, D = data.shape
    idx = torch.randperm(N)[:K]
    centroids = data[idx].clone()   # (K, D)
    cent_sq = centroids.pow(2).sum(1)   # (K,) — recomputed each iter

    for it in range(n_iters):
        # Chunked assignment: avoid building the full (N, K) distance matrix
        assigns = torch.empty(N, dtype=torch.long)
        for lo in range(0, N, chunk):
            hi = min(lo + chunk, N)
            chunk_data = data[lo:hi]            # (chunk, D)
            dist = (
                chunk_data.pow(2).sum(1, keepdim=True)   # (chunk, 1)
                - 2 * chunk_data @ centroids.t()          # (chunk, K)
                + cent_sq.unsqueeze(0)                    # (1, K)
            )
            assigns[lo:hi] = dist.argmin(dim=1)

        # Update centroids via scatter-add
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(K, dtype=torch.float32)
        new_centroids.scatter_add_(0, assigns.unsqueeze(1).expand(-1, D), data)
        counts.scatter_add_(0, assigns, torch.ones(N))

        empty = counts == 0
        if empty.any():
            # Reinit empty clusters from random data points
            reinit_idx = torch.randint(N, (int(empty.sum().item()),))
            new_centroids[empty] = data[reinit_idx]
            counts[empty] = 1.0

        new_centroids /= counts.unsqueeze(1)
        centroids = new_centroids
        cent_sq = centroids.pow(2).sum(1)

        used = int((counts > 0).sum().item())
        if (it + 1) % 10 == 0:
            print(f"  k-means iter {it+1}/{n_iters}: {used}/{K} active clusters")

    return centroids  # (K, D)


def _main() -> None:
    parser = argparse.ArgumentParser(description="K-means codebook init for VQLinearProjector")
    parser.add_argument("--data_json", required=True,
                        help="JSON manifest (list of dicts with 'image' field)")
    parser.add_argument("--image_dir", required=True,
                        help="Root directory for images in data_json")
    parser.add_argument("--model_path", default="liuhaotian/llava-v1.6-vicuna-7b",
                        help="LLaVA model path (for CLIP tower + image processor)")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of images to sample for CLIP feature collection (default 10000)")
    parser.add_argument("--max_tokens", type=int, default=300000,
                        help="Max token vectors fed to k-means (default 300000; "
                             "10k images × 576 patches = 5.76M, which causes OOM)")
    parser.add_argument("--codebook_size", type=int, default=16384)
    parser.add_argument("--code_dim", type=int, default=32,
                        help="Projection dimension (same as VQLinearProjector code_dim)")
    parser.add_argument("--n_iters", type=int, default=50,
                        help="K-means Lloyd iterations (default 50)")
    parser.add_argument("--out", default="checkpoints/codebook_init.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Load manifest — use relative paths from manifest (the 'image' field value)
    with open(args.data_json) as f:
        manifest = json.load(f)
    all_rel = [r["image"] for r in manifest]
    rng = random.Random(args.seed)
    sample_rel = rng.sample(all_rel, min(args.n_samples, len(all_rel)))
    print(f"Selected {len(sample_rel)} images from {len(all_rel)} in manifest.")

    # Resolve image source: directory OR images.zip (LLaVA-Pretrain ships as a zip)
    image_dir = args.image_dir
    zip_candidate = os.path.join(os.path.dirname(image_dir.rstrip("/")), "images.zip")
    if os.path.isdir(image_dir):
        src = image_dir            # directory root; will join with rel path
        print(f"Image source: directory {image_dir}")
    elif os.path.isfile(image_dir) and image_dir.endswith(".zip"):
        src = zipfile.ZipFile(image_dir, "r")
        print(f"Image source: zip {image_dir}")
    elif os.path.isfile(zip_candidate):
        src = zipfile.ZipFile(zip_candidate, "r")
        print(f"Image source: zip {zip_candidate} (auto-detected)")
    else:
        raise FileNotFoundError(
            f"--image_dir '{image_dir}' is neither a directory nor a .zip file, "
            f"and '{zip_candidate}' was not found either.\n"
            "Pass the path to the images/ directory or to images.zip directly."
        )

    # Load LLaVA model (CLIP tower only; we never run the LM)
    if _LLAVA_ROOT not in sys.path:
        sys.path.insert(0, _LLAVA_ROOT)
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading LLaVA ({args.model_path}) on {device} ...")
    model_name = get_model_name_from_path(args.model_path)
    _, model, image_processor, _ = load_pretrained_model(
        args.model_path, model_base=None, model_name=model_name,
        attn_implementation="sdpa",
    )
    model.eval()

    # Collect CLIP features
    print(f"\nCollecting CLIP features for {len(sample_rel)} images ...")
    try:
        feats = collect_clip_features(model, sample_rel, src, device)
    finally:
        if isinstance(src, zipfile.ZipFile):
            src.close()
    print(f"Raw CLIP features: {tuple(feats.shape)}")

    # Subsample tokens for k-means — 5.76M rows at K=16384 is way too large.
    # 300k tokens covers ~18 per codebook entry (K=16384) — sufficient for init.
    if feats.shape[0] > args.max_tokens:
        torch.manual_seed(args.seed)
        idx = torch.randperm(feats.shape[0])[:args.max_tokens]
        feats = feats[idx]
        print(f"Subsampled to {args.max_tokens:,} tokens for k-means.")

    # Down-project to code_dim using a fresh (random) Linear — same as what will be trained
    # We initialise k-means in the down-projected space so the initial codebook is in the
    # same embedding space as the trained VQLinearProjector.
    print(f"\nApplying random down-projection Linear({feats.shape[-1]}, {args.code_dim}) ...")
    down = torch.nn.Linear(feats.shape[-1], args.code_dim, bias=True)
    nn.init.xavier_uniform_(down.weight)
    nn.init.zeros_(down.bias)
    down.to(device)
    projected = _project_down(feats, down)
    print(f"Projected features: {tuple(projected.shape)}")
    del feats

    # K-means
    print(f"\nRunning k-means (K={args.codebook_size}, D={args.code_dim}, "
          f"n_iters={args.n_iters}) ...")
    centroids = kmeans(projected, K=args.codebook_size, n_iters=args.n_iters, seed=args.seed)

    # Save
    torch.save({"codebook": centroids.float()}, args.out)
    print(f"\nSaved → {args.out}  shape={tuple(centroids.shape)}")

    # Quick sanity: codebook spread
    pairwise = torch.cdist(centroids[:1000], centroids[:1000])
    print(f"Mean inter-centroid distance (first 1k): {pairwise[pairwise > 0].mean():.4f}")


if __name__ == "__main__":
    _main()
