"""
Projector-only training for Qwen2.5-VL-VQ.

Replaces Qwen2.5-VL's continuous visual merger (Qwen2_5_VLPatchMerger) with a
VQLinearProjector, freezes the entire model except the new projector, then trains
on CC3M captions.

This is the constructive induction experiment for Qwen2.5-VL:
  Qwen2.5-VL (continuous merger, no yes-bias) →
  Qwen-VQ (VQ projector, expected yes-bias + L0 routing circuit)

Mirrors probe/training/train_llava_vq.py but adapts for Qwen's visual path:
  - Visual encoder: model.visual (includes patch_embed + transformer blocks + merger)
  - Original merger: model.visual.merger (Qwen2_5_VLPatchMerger)
  - After replacement: model.visual.merger = VQLinearProjector
  - Visual features flow: patch_embed → blocks → [VQLinearProjector] → LM

The clip_dim for VQLinearProjector is auto-detected by running one forward pass
with a hook on the merger to capture its input feature dimension.

Usage
-----
    source activate qwen25
    python -m probe.training.train_qwen_vq \\
        --model_path Qwen/Qwen2.5-VL-7B-Instruct \\
        --data_json  data/llava_pretrain/subset_50k.json \\
        --image_dir  data/llava_pretrain/images \\
        --codebook_init checkpoints/codebook_init_qwen.pt \\
        --out_dir    checkpoints/qwen_vq \\
        --max_steps  2000
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image


# ── Dataset (identical to train_llava_vq.py) ─────────────────────────────────

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


def _collate_fn(batch):
    return {"images": [b["image"] for b in batch], "captions": [b["caption"] for b in batch]}


# ── Helper: detect merger input dimension via a probe forward pass ─────────────

# ── QwenVQMerger: mirrors Qwen2_5_VLPatchMerger but with VQ instead of MLP ──

class QwenVQMerger(nn.Module):
    """Drop-in replacement for Qwen2_5_VLPatchMerger.

    Qwen merger forward:
        x: (N, context_dim)  [raw ViT patch features]
        ln_q(x).view(-1, context_dim * spatial_merge_size²)  → (N/4, 5120)
        mlp(...)  →  (N/4, lm_dim)

    This module replicates that shape contract:
        ln_q(x)  →  (N, context_dim)
        .view(-1, hidden_size)  →  (N/4, hidden_size)  [spatial merge, same as original]
        VQLinearProjector  →  (N/4, lm_dim)

    This ensures the merged token count matches the number of <image_pad> tokens
    in input_ids, preventing sequence-length mismatch and NaN loss.
    """

    def __init__(self, orig_merger, vq_proj: nn.Module) -> None:
        super().__init__()
        self.ln_q = orig_merger.ln_q          # borrow trained/init'd RMSNorm
        self.hidden_size = orig_merger.hidden_size  # context_dim * merge_size² = 5120
        self.vq = vq_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate original merger shape contract exactly
        x = self.ln_q(x).view(-1, self.hidden_size)  # (N/4, 5120)
        return self.vq(x)                             # (N/4, lm_dim)

    # Expose VQLinearProjector attrs used by training loop
    @property
    def _commit_loss(self):
        return self.vq._commit_loss

    @property
    def _entropy_loss(self):
        return self.vq._entropy_loss

    def codebook_entropy(self):
        return self.vq.codebook_entropy()

    def codebook_active_fraction(self):
        return self.vq.codebook_active_fraction()

    def state_dict(self, **kwargs):
        return self.vq.state_dict(**kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.vq.load_state_dict(*args, **kwargs)

    @property
    def _last_codes(self):
        return self.vq._last_codes if hasattr(self.vq, "_last_codes") else None

    @property
    def codebook_size(self):
        return self.vq.codebook_size


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from probe.training.llava_vq_projector import VQLinearProjector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading {args.model_path} ...")
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="left"
    )
    processor.image_processor.max_pixels = 336 * 336  # fixed res for training
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    ).eval()

    model.requires_grad_(False)
    # Gradient checkpointing: recomputes activations during backward instead of
    # storing all 32 layers. Trades ~40% speed for ~60% activation memory saving.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    tokenizer = processor.tokenizer

    # Read post-spatial-merge dim directly from merger.hidden_size:
    # Qwen merger: x(N,ctx) → ln_q → .view(-1, ctx*s²) → mlp(ctx*s², lm_dim)
    # VQLinearProjector must receive (N/4, ctx*s²), so clip_dim = merger.hidden_size
    orig_merger = model.visual.merger
    clip_dim = int(orig_merger.hidden_size)   # = context_dim * spatial_merge_size² = 5120
    lm_dim = model.config.hidden_size
    print(f"clip_dim (post-spatial-merge)={clip_dim}, lm_dim={lm_dim}")

    # Build VQLinearProjector operating in post-spatial-merge space
    vq_proj = VQLinearProjector(
        clip_dim=clip_dim,
        lm_dim=lm_dim,
        code_dim=args.code_dim,
        codebook_size=args.codebook_size,
        commitment=args.commitment,
        ema_decay=args.ema_decay,
    )

    if args.codebook_init and os.path.exists(args.codebook_init):
        print(f"Loading k-means codebook from {args.codebook_init} ...")
        vq_proj.load_kmeans_codebook(args.codebook_init)
    else:
        print("Warning: no codebook_init provided — random init (may collapse).")

    # Wrap in QwenVQMerger to replicate spatial-merge shape contract
    model.visual.merger = QwenVQMerger(orig_merger, vq_proj.to(device))
    model.visual.merger.vq.requires_grad_(True)

    trainable_params = [p for p in vq_proj.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Dataset
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
        raise FileNotFoundError(f"--image_dir '{image_dir}' not found.")

    dataset = PretrainDataset(args.data_json, image_src)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if isinstance(image_src, zipfile.ZipFile) else 2,
        collate_fn=_collate_fn,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset):,} examples, batch={args.batch_size}, max_steps={args.max_steps}")

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

    model.eval()
    model.visual.merger.vq.train()  # only VQ projector trains; ln_q stays in eval

    step = 0
    grad_accum_steps = args.grad_accum
    optimizer.zero_grad()
    t_start = time.time()

    for epoch in range(9999):
        for batch in loader:
            if step >= args.max_steps:
                break

            commit_losses = []
            lm_losses = []

            for img, caption in zip(batch["images"], batch["captions"]):
                if not caption.strip():
                    continue

                # Build prompt: image + caption
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe the image briefly."},
                    ],
                }]
                try:
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = processor(
                        text=[text], images=[img], return_tensors="pt"
                    ).to(device)
                except Exception:
                    continue

                # Caption label ids
                cap_ids = tokenizer.encode(caption, add_special_tokens=False)
                if not cap_ids:
                    continue

                # Forward through model (hooks active; merger = VQLinearProjector)
                input_len = inputs["input_ids"].shape[1]
                cap_tensor = torch.tensor(
                    cap_ids + [tokenizer.eos_token_id],
                    dtype=torch.long, device=device,
                ).unsqueeze(0)

                # Append caption ids to input
                full_input_ids = torch.cat([inputs["input_ids"], cap_tensor], dim=1)
                labels = torch.full_like(full_input_ids, -100)
                labels[0, input_len:] = full_input_ids[0, input_len:]

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                    out = model(
                        input_ids=full_input_ids,
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
                        labels=labels,
                        return_dict=True,
                    )
                lm_losses.append(out.loss)
                commit_losses.append(model.visual.merger._commit_loss)

            if not lm_losses:
                continue

            lm_loss = torch.stack(lm_losses).mean()
            commit_loss = torch.stack(commit_losses).mean()
            entropy_reg = torch.zeros(1, device=device)
            if args.entropy_coef > 0:
                entropy_reg = -args.entropy_coef * model.visual.merger._entropy_loss
            loss = (lm_loss + commit_loss + entropy_reg) / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                entropy = model.visual.merger.codebook_entropy()
                active = model.visual.merger.codebook_active_fraction()
                lr_now = scheduler.get_last_lr()[0]
                row = {
                    "step": step, "lm_loss": round(lm_loss.item(), 4),
                    "commit_loss": round(commit_loss.item(), 4),
                    "entropy_reg": round(entropy_reg.item(), 6),
                    "codebook_entropy": round(entropy, 4),
                    "codebook_active": round(active, 4),
                    "lr": round(lr_now, 6), "elapsed_s": round(elapsed, 1),
                }
                print(
                    f"step={step:4d}  lm_loss={lm_loss.item():.4f}  "
                    f"commit={commit_loss.item():.4f}  "
                    f"entropy={entropy:.3f}  active={active:.3f}  "
                    f"lr={lr_now:.2e}", flush=True,
                )
                with open(log_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

            if step > 0 and step % args.save_every == 0:
                ckpt_path = out_dir / f"step_{step:05d}.pt"
                torch.save(
                    {"step": step,
                     "projector": model.visual.merger.state_dict(),
                     "code_dim": args.code_dim, "codebook_size": args.codebook_size,
                     "clip_dim": clip_dim, "lm_dim": lm_dim},
                    ckpt_path,
                )
                print(f"Checkpoint → {ckpt_path}")

            step += 1
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    final_path = out_dir / "projector_final.pt"
    torch.save(
        {"step": step, "projector": model.visual.merger.state_dict(),
         "code_dim": args.code_dim, "codebook_size": args.codebook_size,
         "clip_dim": clip_dim, "lm_dim": lm_dim},
        final_path,
    )
    print(f"\nFinal checkpoint → {final_path}")
    print(f"Training complete. {step} steps, {time.time()-t_start:.0f}s.")
    if isinstance(image_src, zipfile.ZipFile):
        image_src.close()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Projector-only training for Qwen2.5-VL-VQ")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--codebook_init", default=None)
    parser.add_argument("--out_dir", default="checkpoints/qwen_vq")
    parser.add_argument("--code_dim", type=int, default=32)
    parser.add_argument("--codebook_size", type=int, default=16384)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    _main()
