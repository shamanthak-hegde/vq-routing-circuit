"""DoLA LogitsProcessor for multi-token generation (N-082B).

For yes/no benchmarks, the prefill-mode dola.py is sufficient (D-030).
This module extends DoLA to open-ended generation (CHAIR) by hooking into
an early decoder layer at every step and contrasting with the mature layer.

Algorithm (per decode step t):
  1. A forward hook on layers[early_layer] captures the last-position hidden
     state before the final global norm.
  2. In __call__: apply backbone.norm → lm_head to get premature logits.
  3. Apply adaptive plausibility constraint (β=0.1 × max_prob of mature dist).
  4. Return log_softmax(mature) - alpha × log_softmax(premature) over plausible set.

The hook must be removed after generation via proc.remove().

Usage (run_chair.py):
    from probe.baselines.dola_generation import DoLAGenerationLogitsProcessor
    proc = DoLAGenerationLogitsProcessor(hm, early_layer=16, alpha=0.5)
    try:
        cap = hm.run_generate(image, question, logits_processor=[proc])
    finally:
        proc.remove()
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DoLAGenerationLogitsProcessor:
    """LogitsProcessor that applies Decoding by Contrasting Layers at every step.

    Compatible with transformers' LogitsProcessorList — pass an instance to
    hm.run_generate(..., logits_processor=[proc]).

    Call proc.remove() after generation to clean up the registered hook.

    Parameters
    ----------
    hm           : VLMHookManager with _get_decoder_layers() and model.llm
    early_layer  : premature layer index (default: n_layers // 2)
    alpha        : contrast weight (default 0.5)
    beta         : adaptive plausibility threshold fraction (default 0.1)
    """

    def __init__(self, hm, early_layer: int | None = None, alpha: float = 0.5, beta: float = 0.1):
        self._hm = hm
        layers = hm._get_decoder_layers()
        if early_layer is None:
            early_layer = len(layers) // 2
        if early_layer >= len(layers):
            raise ValueError(
                f"early_layer={early_layer} >= n_layers={len(layers)}"
            )
        self.early_layer = early_layer
        self.alpha = alpha
        self.beta = beta
        self._early_hidden: torch.Tensor | None = None
        self._handle = self._register_hook(layers[early_layer])

    def _register_hook(self, layer):
        def _hook(module, args, output):
            # LlamaDecoderLayer returns (hidden_states, ...) or just hidden_states.
            hidden = output[0] if isinstance(output, tuple) else output
            # Capture last-position hidden state (before final global norm).
            self._early_hidden = hidden[:, -1:, :].detach()
        return layer.register_forward_hook(_hook)

    def remove(self):
        """Remove the forward hook. Call after generation completes."""
        self._handle.remove()

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self._early_hidden is None:
            return scores

        llm = self._hm.model.llm
        backbone = llm.model  # LlamaModel
        lm_head = llm.lm_head
        device = self._hm._model_device
        dtype = self._hm._model_dtype

        # Apply the final global norm to the premature hidden state, then lm_head.
        # (backbone.norm is LlamaRMSNorm; decoder layer outputs are pre-norm.)
        # Sanitize to guard against NaN/inf from early-layer residuals.
        early_h = self._early_hidden.to(device=device, dtype=dtype)
        normed = backbone.norm(early_h)
        premature_logits = torch.nan_to_num(
            lm_head(normed).squeeze(1).float().to(scores.device),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

        # Adaptive plausibility constraint.
        probs_mature = F.softmax(scores.float(), dim=-1)
        threshold = self.beta * probs_mature.max(dim=-1, keepdim=True).values
        mask = probs_mature >= threshold

        log_mature = F.log_softmax(scores.float(), dim=-1)
        log_premature = F.log_softmax(premature_logits, dim=-1)

        result = torch.full_like(scores, float("-inf"))
        result[mask] = log_mature[mask] - self.alpha * log_premature[mask]
        return result
