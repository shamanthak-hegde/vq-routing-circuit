"""Corruption modes for activation-patching experiments.

Each function returns inputs_embeds ready to feed to the LM forward without
running a full forward pass.  Call once per record; reuse the returned tensor
across all patching windows for that record to keep the noise realization fixed.
"""

from __future__ import annotations

import torch
from PIL import Image, ImageFilter


@torch.no_grad()
def foil_embeds(hm, foil_image: Image.Image, question: str) -> torch.Tensor:
    """inputs_embeds (1, S, H) built from the foil image on the model device.

    Used for NaturalBench image_swap: runs x' through the visual encoder so the
    LM sees corrupted visual tokens while the question text is unchanged.
    """
    input_ids, images_tensor, image_sizes = hm._build_prompt(foil_image, question)
    if hasattr(hm, "_prepare_embeds_with_last_keep"):
        embeds, _ = hm._prepare_embeds_with_last_keep(
            input_ids, images_tensor, image_sizes
        )
    else:
        embeds, _ = hm._prepare_embeds(input_ids, images_tensor, image_sizes)
    return embeds.detach()


@torch.no_grad()
def noisy_embeds(
    hm,
    clean_image: Image.Image,
    question: str,
    visual_range: tuple[int, int],
    sigma: float,
    seed: int | None = None,
) -> torch.Tensor:
    """inputs_embeds (1, S, H) with Gaussian noise on projected visual tokens.

    Used for POPE gaussian_noise: adds N(0, sigma^2) to the visual-token slice
    of inputs_embeds after the mm_projector and before the first LM layer.
    The noise tensor is sampled once here so all patching passes for this
    record see the same corruption.

    seed: if set, seeds the RNG locally so different seeds give independent draws
    while the rest of the program's RNG state is unaffected.
    """
    input_ids, images_tensor, image_sizes = hm._build_prompt(clean_image, question)
    if hasattr(hm, "_prepare_embeds_with_last_keep"):
        embeds, _ = hm._prepare_embeds_with_last_keep(
            input_ids, images_tensor, image_sizes
        )
    else:
        embeds, _ = hm._prepare_embeds(input_ids, images_tensor, image_sizes)
    embeds = embeds.clone()
    lo, hi = visual_range
    if seed is not None:
        gen = torch.Generator(device=embeds.device)
        gen.manual_seed(seed)
        noise = torch.randn(
            hi - lo, embeds.shape[-1],
            dtype=embeds.dtype, device=embeds.device,
            generator=gen,
        ) * sigma
    else:
        noise = torch.randn(
            hi - lo, embeds.shape[-1],
            dtype=embeds.dtype, device=embeds.device,
        ) * sigma
    embeds[0, lo:hi, :] += noise
    return embeds.detach()


def _prep(hm, image: Image.Image, question: str) -> torch.Tensor:
    input_ids, images_tensor, image_sizes = hm._build_prompt(image, question)
    if hasattr(hm, "_prepare_embeds_with_last_keep"):
        embeds, _ = hm._prepare_embeds_with_last_keep(input_ids, images_tensor, image_sizes)
    else:
        embeds, _ = hm._prepare_embeds(input_ids, images_tensor, image_sizes)
    return embeds


def _gen(device, seed: int | None):
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


@torch.no_grad()
def dropout_embeds(
    hm,
    clean_image: Image.Image,
    question: str,
    visual_range: tuple[int, int],
    drop_frac: float,
    seed: int | None = None,
) -> torch.Tensor:
    """Alternate corruption: zero a random `drop_frac` of visual
    tokens (token masking) rather than adding Gaussian noise. Tests whether the
    L0 restoration signature is specific to embedding-space Gaussian noise."""
    embeds = _prep(hm, clean_image, question).clone()
    lo, hi = visual_range
    n = hi - lo
    k = int(round(drop_frac * n))
    if k > 0:
        gen = _gen(embeds.device, seed)
        perm = torch.randperm(n, device=embeds.device, generator=gen)
        idx = (lo + perm[:k])
        embeds[0, idx, :] = 0.0
    return embeds.detach()


@torch.no_grad()
def shuffle_embeds(
    hm,
    clean_image: Image.Image,
    question: str,
    visual_range: tuple[int, int],
    seed: int | None = None,
) -> torch.Tensor:
    """Alternate corruption: randomly permute the visual-token
    embeddings (patch-shuffle in embedding space). Destroys spatial/semantic
    arrangement while preserving the token distribution."""
    embeds = _prep(hm, clean_image, question).clone()
    lo, hi = visual_range
    n = hi - lo
    gen = _gen(embeds.device, seed)
    perm = torch.randperm(n, device=embeds.device, generator=gen)
    embeds[0, lo:hi, :] = embeds[0, lo:hi, :][perm]
    return embeds.detach()


@torch.no_grad()
def blurred_embeds(
    hm,
    clean_image: Image.Image,
    question: str,
    blur_radius: float = 8.0,
) -> torch.Tensor:
    """Alternate corruption: image-space Gaussian blur applied to
    the pixel image before the visual encoder, so the LM sees genuinely degraded
    visual content (vs. embedding-space noise). The whole prompt is rebuilt from
    the blurred image; text tokens are unchanged because the question is the same."""
    blurred = clean_image.convert("RGB").filter(ImageFilter.GaussianBlur(blur_radius))
    return _prep(hm, blurred, question).detach()
