"""
Precomputed pixel-value cache for the probe set.

Layout on disk (all under cache_dir)
--------------------------------------
  records.json             serialised ProbeRecord list (no tensors)
  pixel_values.pt          {id → float16 tensor (3, H, W)}  clean images
  foil_pixel_values.pt     {id → float16 tensor (3, H, W)}  foil images
                           only present for records with foil_image_path set
                           (i.e. NaturalBench); absent for POPE records.

Image preprocessing
-------------------
Default transform: resize shortest edge to *image_size* (default 336),
centre-crop to image_size × image_size, convert to float in [0, 1], then
apply ImageNet mean/std normalisation.  This matches the input contract of
most ViT-based visual encoders (CLIP, SigLIP, InternViT, …).

Pass a custom *transform* callable to use a model-specific processor instead.

Answer token IDs
----------------
answer_token_id is NOT stored in the cache.  Call
    probe.schema.resolve_answer_token_ids(records, tokenizer)
after loading, once you have the model's tokenizer.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .schema import ProbeRecord

DEFAULT_IMAGE_SIZE = 336
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

_CACHE_DIR = Path(__file__).parent / "cached"


# ── default image transform ───────────────────────────────────────────────────

def _default_transform(img: Image.Image, image_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Resize-crop-normalise; returns float32 tensor (3, H, W)."""
    # resize shortest edge
    w, h = img.size
    scale = image_size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    # centre crop
    left  = (new_w - image_size) // 2
    upper = (new_h - image_size) // 2
    img   = img.crop((left, upper, left + image_size, upper + image_size))
    t = TF.to_tensor(img)                    # (3, H, W) in [0, 1]
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
    return t


# ── serialisation helpers ─────────────────────────────────────────────────────

def _record_to_dict(r: ProbeRecord) -> dict[str, Any]:
    d = dataclasses.asdict(r)
    # answer_token_id not persisted — computed at load time from tokenizer
    d.pop("answer_token_id", None)
    return d


def _dict_to_record(d: dict[str, Any]) -> ProbeRecord:
    d["answer_token_id"] = None
    return ProbeRecord(**d)


# ── build cache ───────────────────────────────────────────────────────────────

def build_cache(
    records: list[ProbeRecord],
    cache_dir: Optional[Path] = None,
    transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Preprocess all images and save tensors + metadata to *cache_dir*.

    Skips files that already exist so incremental rebuilds are cheap.
    """
    if cache_dir is None:
        cache_dir = _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    _xfm = transform or (lambda img: _default_transform(img, image_size))

    pv_path      = cache_dir / "pixel_values.pt"
    foil_pv_path = cache_dir / "foil_pixel_values.pt"
    rec_path     = cache_dir / "records.json"

    # load existing tensors so we can update incrementally
    pixel_values: dict[str, torch.Tensor]      = torch.load(pv_path)      if pv_path.exists()      else {}
    foil_pixel_values: dict[str, torch.Tensor] = torch.load(foil_pv_path) if foil_pv_path.exists() else {}

    new_pv = new_foil = 0
    for rec in records:
        if rec.id not in pixel_values:
            img = Image.open(rec.image_path).convert("RGB")
            pixel_values[rec.id] = _xfm(img).to(dtype)
            new_pv += 1

        if rec.foil_image_path is not None and rec.id not in foil_pixel_values:
            foil = Image.open(rec.foil_image_path).convert("RGB")
            foil_pixel_values[rec.id] = _xfm(foil).to(dtype)
            new_foil += 1

    torch.save(pixel_values,      pv_path)
    torch.save(foil_pixel_values, foil_pv_path)

    with open(rec_path, "w") as f:
        json.dump([_record_to_dict(r) for r in records], f, indent=2)

    print(
        f"Cache written to {cache_dir}\n"
        f"  {len(pixel_values)} clean tensors  (+{new_pv} new)\n"
        f"  {len(foil_pixel_values)} foil tensors   (+{new_foil} new)\n"
        f"  {len(records)} records"
    )


# ── load cache ────────────────────────────────────────────────────────────────

def load_cache(
    cache_dir: Optional[Path] = None,
    device: str = "cpu",
) -> tuple[list[ProbeRecord], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load cached records and pixel-value tensors.

    Returns
    -------
    records           list[ProbeRecord]  (answer_token_id is None — fill with
                                          resolve_answer_token_ids() if needed)
    pixel_values      dict[id → tensor]  clean images, shape (3, H, W)
    foil_pixel_values dict[id → tensor]  foil images (NaturalBench only);
                                          POPE ids absent — use gaussian noise
    """
    if cache_dir is None:
        cache_dir = _CACHE_DIR

    rec_path     = cache_dir / "records.json"
    pv_path      = cache_dir / "pixel_values.pt"
    foil_pv_path = cache_dir / "foil_pixel_values.pt"

    for p in (rec_path, pv_path, foil_pv_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run build_cache() first."
            )

    with open(rec_path) as f:
        records = [_dict_to_record(d) for d in json.load(f)]

    pixel_values      = torch.load(pv_path,      map_location=device)
    foil_pixel_values = torch.load(foil_pv_path, map_location=device)

    return records, pixel_values, foil_pixel_values
