"""Block-overwrite windowed activation patching (ROME-style).

For a window [layer_start, layer_end), every layer l in that range gets a
forward hook that overwrites the residual-stream output at the target token
positions with clean_residual[l, token_indices, :].  One forward pass tests
the entire window simultaneously.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


@torch.no_grad()
def run_patched_forward(
    model,
    corrupted_embeds: Tensor,    # (1, S, H) on model device
    clean_residual: Tensor,      # (n_layers, S, H) — clean_cap.residual, on CPU
    layer_start: int,
    layer_end: int,              # exclusive
    token_indices: Tensor,       # (k,) int64, positions to overwrite
    prompt_last: int,
) -> Tensor:
    """Forward pass on corrupted_embeds with block-overwrite patching in [layer_start, layer_end).

    At every layer l ∈ [layer_start, layer_end), the residual-stream output
    is cloned and then overwritten at token_indices with clean_residual[l].
    Layers outside the window are untouched.

    Returns the lm_head logit vector at prompt_last: (vocab_size,) on CPU.
    """
    device = corrupted_embeds.device
    dtype  = corrupted_embeds.dtype
    layers = model.model.layers
    handles: list = []
    logit_bucket: list[Tensor] = []

    # ── lm_head capture ───────────────────────────────────────────────────────
    def _lmhead_hook(module, inp, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        logit_bucket.append(t.squeeze(0)[prompt_last].detach().cpu())

    handles.append(model.lm_head.register_forward_hook(_lmhead_hook))

    # ── per-layer overwrite hooks ─────────────────────────────────────────────
    # token_indices and clean slices are prepared once, then captured via closure.
    idx = token_indices.to(device)

    for l in range(layer_start, layer_end):
        clean_slice = clean_residual[l][token_indices].to(device=device, dtype=dtype)

        def _patch_hook(module, inp, output, _slice=clean_slice):
            # Decoder layers may return either a bare hidden-state tensor or a
            # tuple whose first element is hidden_states.
            if torch.is_tensor(output):
                hidden = output.clone()       # (1, S, H)
                hidden[0].index_copy_(0, idx, _slice)
                return hidden

            hidden = output[0].clone()        # (1, S, H)
            hidden[0].index_copy_(0, idx, _slice)
            return (hidden,) + output[1:]

        handles.append(layers[l].register_forward_hook(_patch_hook))

    try:
        model(
            inputs_embeds=corrupted_embeds,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()

    return logit_bucket[0]  # (vocab_size,) on CPU
