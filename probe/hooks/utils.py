"""
Hook-registration helpers and accumulation buffers.

Design
------
``register_captures`` installs one hook per module attachment point and
returns a (handles, store) pair.  ``store`` is a plain dict that is
populated in-place as forward calls execute.

List-based accumulation handles both prefill (one call per layer) and
generate (one call per token per layer) identically — callers concatenate
with ``finalize_store`` afterwards.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _as_tensor(output) -> Tensor:
    """Unwrap tuple/list returns to the first tensor."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Expected tensor or tuple, got {type(output)}")


def _as_attn_weight(output) -> Optional[Tensor]:
    """Extract attention weight from self_attn output tuple (index 1).

    Returns None when the backend does not materialise weights (SDPA, Flash).
    """
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        w = output[1]
        return w if (w is not None and torch.is_tensor(w)) else None
    return None


def register_captures(
    projector,
    layers,
    lm_head,
    capture_attn_weights: bool = False,
) -> tuple[list, dict]:
    """Register all four hook families on the given modules.

    Parameters
    ----------
    projector   : the mm_projector module (post-projection visual output)
    layers      : iterable of decoder layers (each has .self_attn and .mlp)
    lm_head     : the language-model head module (produces logits)
    capture_attn_weights : if True, also register hooks for raw attn weights

    Returns
    -------
    handles : list of RemovableHandle
        Call ``h.remove()`` on each to detach hooks when done.
    store : dict
        Keys::

            'projector'     : list[Tensor]  (length ≤ 1 after any run)
            'attn_out'      : list[list[Tensor]]  [layer][call]
            'mlp_out'       : list[list[Tensor]]
            'residual'      : list[list[Tensor]]
            'attn_weights'  : list[list[Tensor]] or None
            'lm_head'       : list[Tensor]        (one per fwd call)

        The inner lists grow by one entry per forward call (prefill = 1,
        each generate step = 1).  ``finalize_store`` concatenates them.
    """
    n_layers = len(layers)

    store: dict = {
        "projector":    [],
        "attn_out":     [[] for _ in range(n_layers)],
        "mlp_out":      [[] for _ in range(n_layers)],
        "residual":     [[] for _ in range(n_layers)],
        "attn_weights": ([[] for _ in range(n_layers)] if capture_attn_weights else None),
        "lm_head":      [],
    }
    handles = []

    # ── Hook 1: projector ────────────────────────────────────────────────────
    def _proj_hook(module, inp, output):
        store["projector"].append(output.detach().cpu())

    handles.append(projector.register_forward_hook(_proj_hook))

    # ── Hooks 2-4: per-layer ─────────────────────────────────────────────────
    for i, layer in enumerate(layers):
        # self-attention output (post o_proj, pre residual-add)
        def _attn_hook(module, inp, output, _i=i):
            store["attn_out"][_i].append(_as_tensor(output).detach().cpu())

        # optional: raw attention weights (eager only)
        def _attn_w_hook(module, inp, output, _i=i):
            w = _as_attn_weight(output)
            if w is not None:
                store["attn_weights"][_i].append(w.detach().cpu())

        # MLP output (post down_proj, pre residual-add)
        def _mlp_hook(module, inp, output, _i=i):
            store["mlp_out"][_i].append(_as_tensor(output).detach().cpu())

        # full decoder-layer output = residual stream after this layer
        def _layer_hook(module, inp, output, _i=i):
            store["residual"][_i].append(_as_tensor(output).detach().cpu())

        handles.append(layer.self_attn.register_forward_hook(_attn_hook))
        if capture_attn_weights:
            handles.append(layer.self_attn.register_forward_hook(_attn_w_hook))
        handles.append(layer.mlp.register_forward_hook(_mlp_hook))
        handles.append(layer.register_forward_hook(_layer_hook))

    # ── lm_head output (final logits) ────────────────────────────────────────
    def _lmhead_hook(module, inp, output):
        store["lm_head"].append(_as_tensor(output).detach().cpu())

    handles.append(lm_head.register_forward_hook(_lmhead_hook))

    return handles, store


def finalize_store(store: dict) -> dict[str, Tensor]:
    """Concatenate per-call lists → stacked tensors.

    Output shapes (B=1 squeezed):

    ============== ====================================
    key            shape
    ============== ====================================
    projector      (num_crops, patches_per_crop, H)
    attn_out       (n_layers, seq_len, H)
    mlp_out        (n_layers, seq_len, H)
    residual       (n_layers, seq_len, H)
    attn_weights   (n_layers, heads, seq_len, seq_len) or None
    logits         (seq_len, vocab)
    ============== ====================================

    ``seq_len = prompt_len`` for prefill; ``prompt_len + n_generated`` for
    generate.
    """
    n_layers = len(store["attn_out"])

    def _cat_calls(call_lists):
        # call_lists[layer] = [tensor_call0, tensor_call1, ...]
        # each tensor_call has shape (1, call_seq_len, H) — squeeze batch dim
        stacked = []
        for layer_calls in call_lists:
            seq_parts = [t.squeeze(0) for t in layer_calls]   # (call_seq, H)
            stacked.append(torch.cat(seq_parts, dim=0))        # (total_seq, H)
        return torch.stack(stacked, dim=0)                     # (n_layers, total_seq, H)

    out: dict[str, Tensor] = {}

    # projector: list of one tensor per projector call
    assert len(store["projector"]) == 1, (
        f"Expected exactly 1 projector call, got {len(store['projector'])}"
    )
    out["projected_visual"] = store["projector"][0]  # (crops, patches, H)

    out["attn_out"] = _cat_calls(store["attn_out"])
    out["mlp_out"]  = _cat_calls(store["mlp_out"])
    out["residual"] = _cat_calls(store["residual"])

    # lm_head: concatenate along seq dim
    lmhead_parts = [t.squeeze(0) for t in store["lm_head"]]  # each (seq, vocab) or (1, vocab)
    out["logits"] = torch.cat(lmhead_parts, dim=0)            # (total_seq, vocab)

    if store["attn_weights"] is not None:
        # In generate mode attention weights grow: prefill is (heads, S, S),
        # decode step k is (heads, 1, S+k).  Stacking them requires padding
        # or keeping them separate.  For now, only store the prefill block
        # (which covers all prompt positions) and append decode slices as-is.
        # Callers who need per-step decode attention should use inference_grounding.py.
        # Here we concatenate only the prefill call (first element per layer).
        aw_layers = []
        for layer_calls in store["attn_weights"]:
            if layer_calls:
                # take only prefill (first call) — shape (1, heads, S, S)
                aw_layers.append(layer_calls[0].squeeze(0))   # (heads, S, S)
        if aw_layers:
            out["attn_weights"] = torch.stack(aw_layers, dim=0)  # (n_layers, heads, S, S)
        else:
            out["attn_weights"] = None
    else:
        out["attn_weights"] = None

    return out


def remove_handles(handles: list) -> None:
    for h in handles:
        h.remove()
