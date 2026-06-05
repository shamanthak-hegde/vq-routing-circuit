"""VCD LogitsProcessor for multi-token generation (N-082B).

For yes/no benchmarks, the prefill-mode vcd.py is sufficient (D-030).
This module extends VCD to open-ended generation (CHAIR) by running a
parallel noisy-branch forward at every decode step via HuggingFace's
LogitsProcessor interface.

Algorithm (per decode step t):
  1. Step 0: full noisy-prefix forward → KV cache + noisy logits at last pos.
  2. Steps 1+: single-token forward on noisy branch using cached KV → noisy logits.
  3. Apply adaptive plausibility constraint (β=0.1 × max_prob of clean dist).
  4. Return clean_logits - alpha × noisy_logits over the plausible set.

Usage (run_chair.py):
    from probe.baselines.vcd_generation import VCDGenerationLogitsProcessor, make_vcd_processor
    proc = make_vcd_processor(hm, noisy_image, question, alpha=1.0)
    cap = hm.run_generate(image, question, logits_processor=[proc])
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image


def make_vcd_processor(
    hm,
    noisy_image: Image.Image,
    question: str,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> "VCDGenerationLogitsProcessor":
    """Construct a VCDGenerationLogitsProcessor from a HookManager and noisy image.

    Pre-computes the noisy branch's prompt embeddings so they are ready for
    the first decode step without any extra work inside __call__.
    """
    # Build prompt for the noisy image — same question, different image.
    input_ids, images_tensor, image_sizes = hm._build_prompt(noisy_image, question)

    # prepare_inputs_labels_for_multimodal returns:
    #   (input_ids, position_ids, attn_mask, past_kv, inputs_embeds, labels)
    result = hm.model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None, images_tensor
    )
    noisy_embeds = result[4].to(hm._model_dtype)
    attn_mask = result[2]
    return VCDGenerationLogitsProcessor(hm, noisy_embeds, attn_mask, alpha=alpha, beta=beta)


class VCDGenerationLogitsProcessor:
    """LogitsProcessor that applies Visual Contrastive Decoding at every step.

    Compatible with transformers' LogitsProcessorList — pass an instance to
    hm.run_generate(..., logits_processor=[proc]).

    Parameters
    ----------
    hm           : VilaUHookManager (or any HookManager with model.llm.model)
    noisy_embeds : pre-computed noisy-image prompt embeddings (1, seq, hidden)
    attn_mask    : attention mask for the noisy prefix
    alpha        : contrastive weight
    beta         : adaptive plausibility threshold fraction (default 0.1)
    """

    def __init__(self, hm, noisy_embeds, attn_mask, alpha: float = 1.0, beta: float = 0.1):
        self._hm = hm
        self._noisy_embeds = noisy_embeds
        self._attn_mask = attn_mask
        self.alpha = alpha
        self.beta = beta
        self._noisy_past = None
        self._step = 0

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        llm = self._hm.model.llm
        backbone = llm.model  # LlamaModel: embed_tokens, layers, norm
        lm_head = llm.lm_head
        device = self._hm._model_device
        dtype = self._hm._model_dtype

        if self._step == 0:
            # Run full noisy prefix to populate KV cache.
            out = backbone(
                inputs_embeds=self._noisy_embeds,
                attention_mask=self._attn_mask,
                use_cache=True,
                return_dict=True,
            )
            self._noisy_past = out.past_key_values
            # last_hidden_state is already after backbone.norm.
            noisy_h = out.last_hidden_state[:, -1:, :]
        else:
            # Single-token forward on noisy branch using cached KV.
            new_tok = input_ids[:, -1:]
            embed = backbone.embed_tokens(new_tok).to(device=device, dtype=dtype)
            out = backbone(
                inputs_embeds=embed,
                past_key_values=self._noisy_past,
                use_cache=True,
                return_dict=True,
            )
            self._noisy_past = out.past_key_values
            noisy_h = out.last_hidden_state[:, -1:, :]

        self._step += 1
        # Sanitize noisy logits: NaN/inf from extreme noisy-image activations
        # would corrupt the subtraction and produce nan/inf in the result.
        noisy_logits = torch.nan_to_num(
            lm_head(noisy_h).squeeze(1).float().to(scores.device),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

        # Adaptive plausibility constraint: only contrast tokens that are
        # plausible in the clean distribution (prevents amplifying near-zero probs).
        probs_clean = F.softmax(scores.float(), dim=-1)
        threshold = self.beta * probs_clean.max(dim=-1, keepdim=True).values
        mask = probs_clean >= threshold

        result = torch.full_like(scores, float("-inf"))
        result[mask] = scores[mask] - self.alpha * noisy_logits[mask]
        return result
