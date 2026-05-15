"""Corruption modes for activation-patching experiments.

Each function returns inputs_embeds ready to feed to the LM forward without
running a full forward pass.  Call once per record; reuse the returned tensor
across all patching windows for that record to keep the noise realization fixed.
"""

from __future__ import annotations

import torch
from PIL import Image


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
