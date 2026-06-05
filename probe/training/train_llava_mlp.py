"""Matched-compute LLaVA-MLP control training (N-081).

Trains a 2-layer GeLU MLP projector for LLaVA-1.6 under *identical*
hyperparameters as train_llava_vq.py (LLaVA-VQ v1):
  - Dataset: 50k CC3M caption pairs
  - Steps: 2000
  - Batch: 8, grad_accum: 4
  - LR: 2e-3, warmup: 100 steps
  - CLIP + Vicuna-7B frozen; only projector trains

No VQ bottleneck → no codebook collapse → prediction: L0 routing circuit
does NOT form; gate A.2 fails; POPE yes-rate remains near-calibrated.

This is the strongest control against the reviewer critique "maybe the
pathology comes from brief projector-only training, not VQ specifically."

Usage:
    source activate sae
    python -m probe.training.train_llava_mlp \\
        --model_path liuhaotian/llava-v1.6-vicuna-7b \\
        --data_json  data/llava_pretrain/subset_50k.json \\
        --image_dir  data/llava_pretrain/images \\
        --out_dir    checkpoints/llava_mlp \\
        --max_steps  2000
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import time
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

_LLAVA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
)


# ── Dataset (identical to train_llava_vq.py) ──────────────────────────────────

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path: str, image_src) -> None:
        with open(manifest_path) as f:
            self.records = json.load(f)
        self.image_src = image_src

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        rel = rec["image"]
        try:
            if isinstance(self.image_src, zipfile.ZipFile):
                data = self.image_src.read(rel)
                img = Image.open(io.BytesIO(data)).convert("RGB")
            else:
                img = Image.open(os.path.join(self.image_src, rel)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (336, 336), color=(128, 128, 128))
        caption = rec["conversations"][1]["value"] if len(rec["conversations"]) > 1 else ""
        return {"image": img, "caption": caption}


def _collate_fn(batch: list[dict]):
    return {"images": [b["image"] for b in batch], "captions": [b["caption"] for b in batch]}


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    if _LLAVA_ROOT not in sys.path:
        sys.path.insert(0, _LLAVA_ROOT)
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    from probe.training.llava_mlp_projector import MLPProjector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading {args.model_path} ...")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, model_base=None, model_name=model_name,
        attn_implementation="sdpa",
    )
    model = model.to(device)
    model.requires_grad_(False)

    clip_dim = model.config.mm_hidden_size   # 1024
    lm_dim = model.config.hidden_size        # 4096
    print(f"Replacing mm_projector with MLPProjector: clip_dim={clip_dim}, lm_dim={lm_dim}")

    mlp_proj = MLPProjector(clip_dim=clip_dim, lm_dim=lm_dim)
    model.model.mm_projector = mlp_proj.to(device)
    model.model.mm_projector.requires_grad_(True)

    trainable_params = [p for p in mlp_proj.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {n_trainable:,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    image_dir = args.image_dir
    zip_candidate = os.path.join(os.path.dirname(image_dir.rstrip("/")), "images.zip")
    if os.path.isdir(image_dir):
        image_src = image_dir
    elif os.path.isfile(image_dir) and image_dir.endswith(".zip"):
        image_src = zipfile.ZipFile(image_dir, "r")
    elif os.path.isfile(zip_candidate):
        image_src = zipfile.ZipFile(zip_candidate, "r")
        print(f"Image source: {zip_candidate} (auto-detected)")
    else:
        raise FileNotFoundError(
            f"--image_dir '{image_dir}' not found as directory or zip. "
            f"Also checked '{zip_candidate}'."
        )

    dataset = PretrainDataset(args.data_json, image_src)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if isinstance(image_src, zipfile.ZipFile) else 4,
        collate_fn=_collate_fn,
        pin_memory=(device.type == "cuda" and not isinstance(image_src, zipfile.ZipFile)),
        drop_last=True,
    )
    print(f"Dataset: {len(dataset):,} examples, batch={args.batch_size}, max_steps={args.max_steps}")

    # ── Optimizer + scheduler (identical to VQ run) ───────────────────────────
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)

    def _lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    tower = model.get_model().get_vision_tower()
    clip_proc = tower.image_processor

    model.eval()
    model.model.mm_projector.train()

    step = 0
    grad_accum_steps = args.grad_accum
    optimizer.zero_grad()
    t_start = time.time()

    for epoch in range(9999):
        for batch in loader:
            if step >= args.max_steps:
                break

            pixel_values = clip_proc(
                batch["images"], return_tensors="pt"
            )["pixel_values"].to(device, dtype=torch.float16)

            with torch.no_grad():
                clip_feats = tower(pixel_values)   # (B, n_patches, 1024)

            lm_losses = []
            for img_feat, caption in zip(clip_feats.unbind(0), batch["captions"]):
                cap_ids = tokenizer.encode(caption, add_special_tokens=False)
                if not cap_ids:
                    continue

                proj_out = model.model.mm_projector(img_feat.unsqueeze(0))  # (1, N, 4096)
                n_vis = proj_out.shape[1]

                cap_tensor = torch.tensor(
                    [tokenizer.bos_token_id] + cap_ids + [tokenizer.eos_token_id],
                    dtype=torch.long, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    cap_embeds = model.model.embed_tokens(cap_tensor)

                inputs_embeds = torch.cat([proj_out, cap_embeds], dim=1)

                labels_cap = torch.cat([
                    torch.full((1,), -100, device=device),
                    torch.tensor(cap_ids, device=device),
                    torch.tensor([tokenizer.eos_token_id], device=device),
                ], dim=0).unsqueeze(0)
                labels = torch.cat([
                    torch.full((1, n_vis), -100, dtype=torch.long, device=device),
                    labels_cap,
                ], dim=1)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    out = model(inputs_embeds=inputs_embeds, labels=labels, return_dict=True)
                lm_losses.append(out.loss)

            if not lm_losses:
                continue

            lm_loss = torch.stack(lm_losses).mean()
            loss = lm_loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                lr_now = scheduler.get_last_lr()[0]
                row = {
                    "step": step,
                    "lm_loss": round(lm_loss.item(), 4),
                    "lr": round(lr_now, 6),
                    "elapsed_s": round(elapsed, 1),
                }
                print(
                    f"step={step:4d}  lm_loss={lm_loss.item():.4f}  lr={lr_now:.2e}",
                    flush=True,
                )
                with open(log_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

            if step > 0 and step % args.save_every == 0:
                ckpt_path = out_dir / f"step_{step:05d}.pt"
                torch.save({
                    "step": step,
                    "projector": model.model.mm_projector.state_dict(),
                    "quantizer": "mlp",
                    "clip_dim": clip_dim,
                    "lm_dim": lm_dim,
                }, ckpt_path)
                print(f"Checkpoint → {ckpt_path}")

            step += 1
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    final_path = out_dir / "projector_final.pt"
    torch.save({
        "step": step,
        "projector": model.model.mm_projector.state_dict(),
        "quantizer": "mlp",
        "clip_dim": clip_dim,
        "lm_dim": lm_dim,
    }, final_path)
    print(f"\nFinal checkpoint → {final_path}")
    print(f"Training complete. {step} steps, {time.time()-t_start:.0f}s.")
    if isinstance(image_src, zipfile.ZipFile):
        image_src.close()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched-compute LLaVA-MLP control training (N-081)"
    )
    parser.add_argument("--model_path", default="liuhaotian/llava-v1.6-vicuna-7b")
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_dir", default="checkpoints/llava_mlp")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    _main()
