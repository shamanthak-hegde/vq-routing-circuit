"""Visual Contrastive Decoding (VCD) — Leng et al. CVPR 2024.

For yes/no benchmarks we only need the first generated token logits, so VCD
reduces to a prefill-time logit contrast:

    logits_vcd = logits_clean - alpha * logits_noisy

where logits_noisy are from a second prefill with a gaussian-noisy image.

This is functionally equivalent to VCD's full generation version when the
answer is a single token because there is no auto-regressive decoding to
contrast beyond the first step.

Usage in run_bench.py (via --decoder vcd)
-----------------------------------------
    # vcd_logits() takes the clean Capture and a noisy image path
    logits = vcd_logits(cap_clean, hm, noisy_img, rec.question,
                        sigma=args.decoder_sigma, alpha=args.alpha)

Standalone
----------
    from probe.baselines.vcd import vcd_logits, make_noisy_image

Reference: Leng et al. "Mitigating Object Hallucinations in Large Vision-Language
           Models through Visual Contrastive Decoding", CVPR 2024.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image


def make_noisy_image(image: Image.Image, sigma: float = 1.0) -> Image.Image:
    """Return a PIL image with gaussian noise added (in pixel space, σ in [0,255] scale).

    sigma is the std-dev of noise added to pixel values before clamping to [0,255].
    Typical values: 0.3–1.5 for VCD (weaker degradation than POPE calibration σ).
    """
    import numpy as np
    arr = np.array(image).astype(float)
    arr = arr + np.random.randn(*arr.shape) * sigma
    arr = arr.clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


@torch.no_grad()
def vcd_logits(
    cap_clean,
    hm,
    noisy_image: Image.Image,
    question: str,
    alpha: float = 1.0,
    force_ids: list[int] | None = None,
) -> torch.Tensor:
    """Compute VCD-adjusted logits at the final prompt position.

    Parameters
    ----------
    cap_clean   : Capture from a clean-image run_prefill call
    hm          : the same VLMHookManager used for cap_clean
    noisy_image : degraded version of the original image
    question    : same question string used for cap_clean
    alpha       : contrastive weight (default 1.0 per VCD paper)
    force_ids   : token IDs that must always be included in the scored set
                  regardless of the plausibility threshold (e.g., [yes_id, no_id]).
                  Use this for binary classification to prevent both answer tokens
                  from being masked out when the model is confident about a third token.

    Returns
    -------
    logits_vcd  : (vocab,) float32 tensor on CPU
    """
    cap_noisy = hm.run_prefill(noisy_image, question)
    prompt_last = cap_clean.token_index.prompt_last

    logits_clean = cap_clean.logits[prompt_last].float()
    # Sanitize noisy logits: NaN/inf from extreme noisy-image activations would
    # propagate through the subtraction and corrupt the yes/no comparison.
    logits_noisy = torch.nan_to_num(
        cap_noisy.logits[prompt_last].float(),
        nan=0.0, posinf=0.0, neginf=0.0,
    )

    # Adaptive plausibility constraint: only contrast tokens above a threshold
    # in the clean distribution (prevents amplifying impossible tokens).
    # Threshold = 0.1 × max_prob (from VCD paper Appendix A).
    probs_clean = F.softmax(logits_clean, dim=-1)
    threshold = 0.1 * probs_clean.max()
    mask = probs_clean >= threshold

    # Ensure force_ids (e.g., yes/no token IDs) are always in the scored set.
    # Without this, a model confident about a third token (e.g., '\n') would
    # mask out both answer tokens, producing -inf for both and forcing all-NO.
    if force_ids:
        for tok_id in force_ids:
            mask[tok_id] = True

    logits_vcd = torch.full_like(logits_clean, float("-inf"))
    logits_vcd[mask] = logits_clean[mask] - alpha * logits_noisy[mask]
    return logits_vcd


@torch.no_grad()
def vcd_logits_from_images(
    hm,
    clean_image: Image.Image,
    question: str,
    sigma: float = 1.0,
    alpha: float = 1.0,
    pixel_sigma: float = 40.0,
) -> tuple[torch.Tensor, object]:
    """Run both prefills and return (vcd_logits, cap_clean).

    pixel_sigma: gaussian noise std-dev in pixel space (0–255 scale).
    """
    noisy_image = make_noisy_image(clean_image, sigma=pixel_sigma)
    cap_clean = hm.run_prefill(clean_image, question)
    logits = vcd_logits(cap_clean, hm, noisy_image, question, alpha=alpha)
    return logits, cap_clean
