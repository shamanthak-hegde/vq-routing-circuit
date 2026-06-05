"""Decoding by Contrasting Layers (DoLA) — Chuang et al. ICLR 2024.

For yes/no benchmarks we only need the first-token logit, so DoLA reduces to
a single prefill contrast between mature (final) and premature (early) layers:

    logits_dola = log_softmax(logits_final) - log_softmax(logits_premature)

We obtain logits_premature by passing the captured early-layer residual through
the model's own lm_head (no second forward pass needed — residuals from all
layers are already in the Capture object).

Usage
-----
    cap = hm.run_prefill(img, question)
    logits = dola_logits(cap, hm, early_layer=16)

Reference: Chuang et al. "DoLa: Decoding by Contrasting Layers Improves
           Factuality in Large Language Models", ICLR 2024.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def dola_logits(
    cap,
    hm,
    early_layer: int = 16,
    alpha: float = 0.5,
    force_ids: list[int] | None = None,
) -> torch.Tensor:
    """Compute DoLA-adjusted logits at the final prompt position.

    Parameters
    ----------
    cap         : Capture from run_prefill (must have residual tensor)
    hm          : VLMHookManager (used to access lm_head)
    early_layer : premature layer index (default 16 for 32-layer model)
    alpha       : contrast weight; 0.5 means equal blend of mature and contrast
    force_ids   : token IDs always included in the scored set regardless of the
                  plausibility threshold (e.g., [yes_id, no_id] for binary tasks).

    Returns
    -------
    logits_dola : (vocab,) float32 tensor on CPU
    """
    n_layers = cap.residual.shape[0]
    if early_layer >= n_layers:
        raise ValueError(
            f"early_layer={early_layer} >= n_layers={n_layers}; "
            "use a smaller value (e.g. n_layers // 2)"
        )

    prompt_last = cap.token_index.prompt_last
    lm_head = hm._get_lm_head()
    device = hm._model_device
    dtype = next(lm_head.parameters()).dtype

    # Mature logits: already in cap.logits
    logits_mature = cap.logits[prompt_last].float()

    # Premature logits: run lm_head on the early-layer residual.
    # Sanitize to guard against NaN/inf from early-layer residuals at extreme
    # positions (e.g., early_layer=0 where hidden state is raw embeddings).
    early_hidden = cap.residual[early_layer, prompt_last, :].unsqueeze(0)
    early_hidden = early_hidden.to(device=device, dtype=dtype)
    with torch.no_grad():
        logits_premature = lm_head(early_hidden).squeeze(0).float().cpu()
    logits_premature = torch.nan_to_num(logits_premature, nan=0.0, posinf=0.0, neginf=0.0)

    # Adaptive plausibility constraint (same as VCD).
    probs_mature = F.softmax(logits_mature, dim=-1)
    threshold = 0.1 * probs_mature.max()
    mask = probs_mature >= threshold

    # Ensure force_ids are always scored (prevents all-NO collapse on binary tasks).
    if force_ids:
        for tok_id in force_ids:
            mask[tok_id] = True

    logits_dola = torch.full_like(logits_mature, float("-inf"))
    # DoLA: contrast log_softmax distributions
    log_mature = F.log_softmax(logits_mature, dim=-1)
    log_premature = F.log_softmax(logits_premature, dim=-1)
    logits_dola[mask] = log_mature[mask] - alpha * log_premature[mask]
    return logits_dola


def default_early_layer(hm) -> int:
    """Return a sensible premature layer: the middle of the decoder."""
    n_layers = len(hm._get_decoder_layers())
    return n_layers // 2
