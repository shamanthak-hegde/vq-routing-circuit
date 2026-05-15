"""
Projector-only training for LLaVA-VQ (N-056).

Freezes the CLIP vision tower and the Vicuna-7B LM; trains only the
VQLinearProjector (down Linear + EMA VQ codebook + up Linear).

Loss = causal LM loss on caption tokens + commitment_loss (β=0.25)

Usage
-----
    source activate sae
    python -m probe.training.train_llava_vq \\
        --model_path liuhaotian/llava-v1.6-vicuna-7b \\
        --data_json  data/llava_pretrain/subset_50k.json \\
        --image_dir  data/llava_pretrain/images \\
        --codebook_init checkpoints/codebook_init.pt \\
        --out_dir    checkpoints/llava_vq \\
        --max_steps  2000

Monitor codebook_entropy (should stay > ln(K)×0.5 ≈ 4.8) and lm_loss
(should drop from ~10 to ~3-5 over 2k steps on a caption dataset).
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


# ── Dataset ───────────────────────────────────────────────────────────────────

class PretrainDataset(torch.utils.data.Dataset):
    """LLaVA-Pretrain captions dataset.

    Each entry: {image: rel_path, conversations: [{from: human, value: ...},
                                                   {from: gpt, value: <caption>}]}
    Only the GPT/caption text is used as the LM training target.

    image_src can be a directory path (str) or a ZipFile object.
    """

    def __init__(self, manifest_path: str, image_src) -> None:
        with open(manifest_path) as f:
            self.records = json.load(f)
        self.image_src = image_src   # str (dir) or zipfile.ZipFile

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
    return {
        "images": [b["image"] for b in batch],
        "captions": [b["caption"] for b in batch],
    }


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_input_and_target(tokenizer, image_token_ids: list[int],
                             caption_ids: list[int], device):
    """Build input_ids and labels (−100 for image/prompt tokens, caption for loss)."""
    # Minimal vicuna_v1-style prompt: "USER: <image>\n<caption>\nASSISTANT:"
    # For pretraining we just want the caption as the prediction target.
    # Token layout: [image_tokens] [caption_ids] [eos]
    img_tensor = torch.tensor(image_token_ids, dtype=torch.long)
    cap_tensor = torch.tensor(caption_ids + [tokenizer.eos_token_id], dtype=torch.long)
    input_ids = torch.cat([img_tensor, cap_tensor]).unsqueeze(0).to(device)
    labels = torch.full_like(input_ids, fill_value=-100)
    # Only supervise on caption tokens
    labels[0, len(image_token_ids):] = cap_tensor
    return input_ids, labels


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    if _LLAVA_ROOT not in sys.path:
        sys.path.insert(0, _LLAVA_ROOT)
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    from probe.training.llava_vq_projector import VQLinearProjector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load base LLaVA model ─────────────────────────────────────────────────
    print(f"\nLoading {args.model_path} ...")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, model_base=None, model_name=model_name,
        attn_implementation="sdpa",
    )
    model = model.to(device)

    # ── Freeze LM + CLIP; only projector trains ───────────────────────────────
    model.requires_grad_(False)

    # ── Replace mm_projector with VQLinearProjector ───────────────────────────
    clip_dim = model.config.mm_hidden_size            # 1024
    lm_dim = model.config.hidden_size                 # 4096
    print(f"Replacing mm_projector: clip_dim={clip_dim}, lm_dim={lm_dim}, "
          f"code_dim={args.code_dim}, K={args.codebook_size}")

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
        print("Warning: no codebook_init provided — random codebook init (may collapse).")

    model.model.mm_projector = vq_proj.to(device)
    model.model.mm_projector.requires_grad_(True)

    # EMA buffers are not Parameters — make Linears trainable
    trainable_params = [p for p in vq_proj.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {n_trainable:,}")

    # ── Dataset — resolve image source (directory or images.zip) ─────────────
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
    print(f"Dataset: {len(dataset):,} examples, batch={args.batch_size}, "
          f"max_steps={args.max_steps}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)

    def _lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    # ── CLIP tower + image processor (hoist; do not call process_images) ────────
    # process_images applies anyres tiling → 5D tensor (B, n_tiles, 3, H, W).
    # CLIPVisionModel expects 4D (B, 3, H, W).  Use tower.image_processor
    # directly: it crops/resizes each image to exactly 336×336 with no tiling.
    tower = model.get_model().get_vision_tower()
    clip_proc = tower.image_processor

    # ── Training ──────────────────────────────────────────────────────────────
    model.eval()                           # LM frozen → eval mode
    model.model.mm_projector.train()       # projector in train mode (EMA active)

    step = 0
    grad_accum_steps = args.grad_accum
    optimizer.zero_grad()
    t_start = time.time()

    for epoch in range(9999):
        for batch in loader:
            if step >= args.max_steps:
                break

            # Encode all images in the batch through CLIP in one forward pass.
            # clip_proc returns (B, 3, 336, 336) — no anyres tiling.
            pixel_values = clip_proc(
                batch["images"], return_tensors="pt"
            )["pixel_values"].to(device, dtype=torch.float16)

            with torch.no_grad():
                clip_feats = tower(pixel_values)   # (B, n_patches, 1024)

            # Tokenize captions (no image sentinel in targets)
            commit_losses = []
            lm_losses = []
            for img_feat, caption in zip(clip_feats.unbind(0), batch["captions"]):
                cap_ids = tokenizer.encode(caption, add_special_tokens=False)
                if not cap_ids:
                    continue

                # Project through VQLinearProjector (img_feat already on device)
                proj_out = model.model.mm_projector(img_feat.unsqueeze(0))  # (1, N, 4096)
                n_vis = proj_out.shape[1]
                commit_losses.append(model.model.mm_projector._commit_loss)

                # Build simple input: [image_embeds | caption_token_embeds]
                cap_tensor = torch.tensor(
                    [tokenizer.bos_token_id] + cap_ids + [tokenizer.eos_token_id],
                    dtype=torch.long, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    cap_embeds = model.model.embed_tokens(cap_tensor)  # (1, L, 4096)

                # Concatenate: image embeds + caption embeds
                inputs_embeds = torch.cat(
                    [proj_out, cap_embeds], dim=1
                )  # (1, N+L, 4096)

                # Labels: −100 for image positions; caption token ids for supervision
                labels_cap = torch.cat([
                    torch.full((1,), -100, device=device),  # bos
                    torch.tensor(cap_ids, device=device),
                    torch.tensor([tokenizer.eos_token_id], device=device),
                ], dim=0).unsqueeze(0)
                labels = torch.cat([
                    torch.full((1, n_vis), -100, dtype=torch.long, device=device),
                    labels_cap,
                ], dim=1)  # (1, N+L)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    out = model(inputs_embeds=inputs_embeds, labels=labels, return_dict=True)
                lm_losses.append(out.loss)

            if not lm_losses:
                continue

            lm_loss = torch.stack(lm_losses).mean()
            commit_loss = torch.stack(commit_losses).mean()
            # _entropy_loss = -H (negative entropy); multiplying by -coef maximises H
            entropy_reg = torch.zeros(1, device=device)
            if args.entropy_coef > 0:
                entropy_reg = -args.entropy_coef * model.model.mm_projector._entropy_loss
            loss = (lm_loss + commit_loss + entropy_reg) / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                entropy = model.model.mm_projector.codebook_entropy()
                active = model.model.mm_projector.codebook_active_fraction()
                lr_now = scheduler.get_last_lr()[0]
                row = {
                    "step": step,
                    "lm_loss": round(lm_loss.item(), 4),
                    "commit_loss": round(commit_loss.item(), 4),
                    "entropy_reg": round(entropy_reg.item(), 6),
                    "codebook_entropy": round(entropy, 4),
                    "codebook_active": round(active, 4),
                    "lr": round(lr_now, 6),
                    "elapsed_s": round(elapsed, 1),
                }
                print(
                    f"step={step:4d}  lm_loss={lm_loss.item():.4f}  "
                    f"commit={commit_loss.item():.4f}  "
                    f"entropy_reg={entropy_reg.item():.4f}  "
                    f"entropy={entropy:.3f}  active={active:.3f}  "
                    f"lr={lr_now:.2e}",
                    flush=True,
                )
                with open(log_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

            # Checkpoint
            if step > 0 and step % args.save_every == 0:
                ckpt_path = out_dir / f"step_{step:05d}.pt"
                torch.save(
                    {"step": step,
                     "projector": model.model.mm_projector.state_dict(),
                     "code_dim": args.code_dim,
                     "codebook_size": args.codebook_size,
                     "clip_dim": clip_dim,
                     "lm_dim": lm_dim},
                    ckpt_path,
                )
                print(f"Checkpoint → {ckpt_path}")

            step += 1
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    # ── Final checkpoint ──────────────────────────────────────────────────────
    final_path = out_dir / "projector_final.pt"
    torch.save(
        {"step": step,
         "projector": model.model.mm_projector.state_dict(),
         "code_dim": args.code_dim,
         "codebook_size": args.codebook_size,
         "clip_dim": clip_dim,
         "lm_dim": lm_dim},
        final_path,
    )
    print(f"\nFinal checkpoint → {final_path}")
    print(f"Training complete. {step} steps, {time.time()-t_start:.0f}s.")
    if isinstance(image_src, zipfile.ZipFile):
        image_src.close()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Projector-only training for LLaVA-VQ (N-056)")
    parser.add_argument("--model_path", default="liuhaotian/llava-v1.6-vicuna-7b")
    parser.add_argument("--data_json", required=True,
                        help="Path to subset JSON (e.g. data/llava_pretrain/subset_50k.json)")
    parser.add_argument("--image_dir", required=True,
                        help="Root directory containing the pretrain images")
    parser.add_argument("--codebook_init", default=None,
                        help="Path to k-means codebook checkpoint (checkpoints/codebook_init.pt)")
    parser.add_argument("--out_dir", default="checkpoints/llava_vq")
    parser.add_argument("--code_dim", type=int, default=32)
    parser.add_argument("--codebook_size", type=int, default=16384)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch×grad_accum)")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--entropy_coef", type=float, default=0.0,
                        help="Entropy regularization coefficient (0=disabled). "
                             "Adds coef×H(marginal_assignment) to the loss to prevent "
                             "codebook collapse. Recommended: 0.1 for v2 retrain.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    _main()
