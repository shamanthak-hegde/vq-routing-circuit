"""
Reverse ablation: VILA-U with continuous SigLIP adapter (N-069).

Replaces VILA-U's VQ projector (RQVAE → mm_projector) with a continuous
SigLIP MLP adapter that bypasses the RQVAE quantization step entirely.
Freezes the VILA-U LLM and SigLIP encoder; trains only the new MLP adapter.

This is the REVERSE of LLaVA-VQ (N-056):
  LLaVA + VQ projector → pathology appears        (constructive induction)
  VILA-U + continuous adapter → pathology should disappear  (reverse ablation)

Together these form a complete causal loop proving the VQ quantization step
specifically causes the pathological L0 routing circuit.

VILA-U pipeline (original):
  image → SigLIP encoder → RQVAE quantizer → VQ codes → mm_projector → LLM

VILA-U-MLP pipeline (this script):
  image → SigLIP encoder → [skip RQVAE] → new_mlp_adapter → LLM

The SigLIP features are captured via a forward hook on the penultimate SigLIP
encoder layer (where encode_image() branches to the quantizer). This avoids
modifying VILA-U's source code.

Architecture
------------
- SigLIP feature dim: auto-detected from model (typically 1152 for SigLIP-400M)
- LLM hidden dim: model.llm.config.hidden_size (4096 for VILA-U-7B)
- MLP adapter: Linear(1152 → 4096×2) + GELU + Linear(4096×2 → 4096)
  (matches LLaVA mm_projector structure: mlp2x_gelu)

Usage
-----
    source activate vila-u
    python -m probe.training.train_vilau_mlp \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --data_json  data/llava_pretrain/subset_50k.json \\
        --image_dir  data/llava_pretrain/images \\
        --out_dir    checkpoints/vilau_mlp \\
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
import torch.nn.functional as F
from PIL import Image

_VILAU_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
)


# ── Continuous MLP adapter (mirrors LLaVA mm_projector mlp2x_gelu) ───────────

class SigLIPMLPAdapter(nn.Module):
    """Two-layer MLP: siglip_dim → 2*lm_dim → lm_dim (with GELU).

    Mirrors LLaVA's mlp2x_gelu mm_projector so the two experiments are
    architecturally comparable: same structure, only the input source differs
    (continuous SigLIP vs VQ codebook embeddings).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 2)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(out_dim * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


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


# ── SigLIP feature capture via forward hook ───────────────────────────────────

class SigLIPFeatureCapture:
    """Captures continuous SigLIP features at the penultimate encoder layer.

    VILA-U's encode_image() runs the SigLIP ViT and at the penultimate layer
    (layer n-2) hands off to the RQVAE quantizer.  This hook captures those
    pre-quantization features and stores them for use by the MLP adapter.
    """

    def __init__(self) -> None:
        self._feats: torch.Tensor | None = None
        self._handle = None

    def register(self, encoder_layer_n_minus_2) -> None:
        self._handle = encoder_layer_n_minus_2.register_forward_hook(self._hook)

    def _hook(self, module, inp, out) -> None:
        self._feats = out[0]  # (B, L, C) — continuous SigLIP hidden states

    def get(self) -> torch.Tensor | None:
        feats = self._feats
        self._feats = None
        return feats

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    if _VILAU_ROOT not in sys.path:
        sys.path.insert(0, _VILAU_ROOT)

    from vila_u.model.builder import load_pretrained_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading {args.model_path} ...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, attn_implementation="eager"
    )
    model = model.eval()
    model.requires_grad_(False)

    # Access the RQVAE-SigLIP module
    rqvaesiglip = model.vision_tower.vision_tower.rqvaesiglip
    siglip_vm = rqvaesiglip.siglip_model.vision_model
    siglip_layers = siglip_vm.encoder.layers
    n_siglip_layers = len(siglip_layers)

    # Detect actual SigLIP feature dim by probing a dummy forward pass.
    # encode_image() stops at layer n-2 and quantizes; the penultimate layer's
    # output is the continuous SigLIP feature we want (shape (B, L, C)).
    _probe_hs = []
    def _probe_hook(m, inp, out): _probe_hs.append(out[0].shape[-1])
    _probe_h = siglip_layers[n_siglip_layers - 2].register_forward_hook(_probe_hook)
    with torch.no_grad():
        dummy_px = image_processor(
            [Image.new("RGB", (256, 256))], return_tensors="pt"
        )["pixel_values"].to(device, dtype=model.llm.dtype)
        rqvaesiglip.encode_image(dummy_px)
    _probe_h.remove()
    siglip_dim = int(_probe_hs[0]) if _probe_hs else 1152
    lm_dim = model.llm.config.hidden_size
    print(f"SigLIP dim (detected): {siglip_dim}, LM dim: {lm_dim}")

    # Replace mm_projector with SigLIPMLPAdapter
    mlp_adapter = SigLIPMLPAdapter(siglip_dim, lm_dim).to(device)
    if model.mm_projector is not None:
        mlp_adapter = mlp_adapter.to(next(model.mm_projector.parameters()).dtype)
    model.mm_projector = mlp_adapter
    model.mm_projector.requires_grad_(True)

    trainable_params = list(mlp_adapter.parameters())
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
    model.mm_projector.train()

    step = 0
    grad_accum_steps = args.grad_accum
    optimizer.zero_grad()
    t_start = time.time()

    for epoch in range(9999):
        for batch in loader:
            if step >= args.max_steps:
                break

            lm_losses = []
            for img, caption in zip(batch["images"], batch["captions"]):
                if not caption.strip():
                    continue

                cap_ids = tokenizer.encode(caption, add_special_tokens=False)
                if not cap_ids:
                    continue

                # Step 1: Run SigLIP to penultimate layer (bypass VQ quantization)
                pixel_values = image_processor(
                    [img], return_tensors="pt"
                )["pixel_values"].to(device, dtype=model.llm.dtype)

                with torch.no_grad():
                    hs = siglip_vm.embeddings(pixel_values)
                    for i, layer in enumerate(siglip_layers):
                        hs = layer(hs, attention_mask=None)[0]
                        if i == n_siglip_layers - 2:
                            break
                # hs: (1, L, siglip_dim) — continuous SigLIP features

                # Step 2: Apply MLP adapter
                B, L, C = hs.shape
                vis_e = mlp_adapter(
                    hs.reshape(L, C).to(dtype=next(mlp_adapter.parameters()).dtype)
                )  # (L, lm_dim)
                vis_e = vis_e.unsqueeze(0).to(dtype=model.llm.dtype)  # (1, L, lm_dim)

                # Step 3: Embed caption tokens
                cap_tensor = torch.tensor(
                    cap_ids + [tokenizer.eos_token_id],
                    dtype=torch.long, device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    cap_e = model.llm.model.embed_tokens(cap_tensor)

                # Step 4: Build inputs_embeds + labels
                inputs_embeds = torch.cat([vis_e, cap_e], dim=1)
                labels_step = torch.full(
                    (1, L + cap_tensor.shape[1]), -100,
                    dtype=torch.long, device=device,
                )
                labels_step[0, L:] = cap_tensor[0]

                # Step 5: LLM forward + manual loss (model.llm may return None loss
                # when inputs_embeds is passed without input_ids)
                with torch.cuda.amp.autocast(
                    enabled=(device.type == "cuda"), dtype=model.llm.dtype
                ):
                    out = model.llm(
                        inputs_embeds=inputs_embeds,
                        return_dict=True,
                    )
                logits = out.logits  # (1, L+cap, vocab)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels_step[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                lm_losses.append(loss)

            if not lm_losses:
                continue

            loss = torch.stack(lm_losses).mean() / grad_accum_steps
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
                    "step": step, "lm_loss": round(lm_losses[-1].item(), 4),
                    "lr": round(lr_now, 6), "elapsed_s": round(elapsed, 1),
                }
                print(
                    f"step={step:4d}  lm_loss={lm_losses[-1].item():.4f}  "
                    f"lr={lr_now:.2e}", flush=True,
                )
                with open(log_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

            if step > 0 and step % args.save_every == 0:
                ckpt_path = out_dir / f"step_{step:05d}.pt"
                torch.save(
                    {"step": step,
                     "adapter": model.mm_projector.state_dict(),
                     "siglip_dim": siglip_dim, "lm_dim": lm_dim},
                    ckpt_path,
                )
                print(f"Checkpoint → {ckpt_path}")

            step += 1
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    final_path = out_dir / "adapter_final.pt"
    torch.save(
        {"step": step, "adapter": model.mm_projector.state_dict(),
         "siglip_dim": siglip_dim, "lm_dim": lm_dim},
        final_path,
    )
    print(f"\nFinal checkpoint → {final_path}")
    print(f"Training complete. {step} steps, {time.time()-t_start:.0f}s.")
    if isinstance(image_src, zipfile.ZipFile):
        image_src.close()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="VILA-U reverse ablation: continuous SigLIP adapter (N-069)"
    )
    parser.add_argument("--model_path", default="mit-han-lab/vila-u-7b-256")
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_dir", default="checkpoints/vilau_mlp")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    _main()
