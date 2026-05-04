"""
LavitHookManager — concrete hook manager for LaVIT-7B-v2.

Architecture (understanding inference path)
--------------------------------------------
- Visual tokenizer : DynamicVisualTokenizer
    EVA-CLIP ViT (1408-d) → Gumbel token-drop → causal_encoder
    → vit_proj (Linear 1408→4096, projector hook point)
    → variable T_real tokens (1-256) per image
    NOTE: the VQ codebook fires ONLY for image generation, not understanding.
    This path uses continuous post-projection features.
- LLM              : LlamaForCausalLM at model.llama_model
- Decoder layers   : model.llama_model.model.layers
- lm_head          : model.llama_model.lm_head

Token layout (no -200 sentinel; image block prepended to text)
--------------------------------------------------------------
  [img_start] [VISUAL×T_real] [img_end] [Question: {q} Answer:]
    OTHER          VISUAL         OTHER    QUESTION   ...  OTHER

  img_start / img_end are special token embeddings (IDs 32000/32001) that
  appear as the first and last positions of the image block.  The trailing
  " Answer:" tokens are the role indicator (OTHER), equivalent to " ASSISTANT:"
  in Vicuna-style models.

Determinism
-----------
encode_features uses Gumbel-softmax (hard=True) to select which patches to
keep.  The random draw is seeded to torch.manual_seed(42) per _prepare_embeds
call so clean/corrupt runs on the same record produce the same T_real and the
same mask, avoiding shape mismatches during patching.

Generate contract
-----------------
model.llama_model.generate(inputs_embeds=...) is called directly (no
model.generate wrapper).  _call_generate re-runs compute_dynamic_visual_embeds
with hooks active so the vit_proj hook fires during generate.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
from PIL import Image

from .base import VLMHookManager
from .schema import Capture, TokenCategory, TokenIndex

_LAVIT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "LaVIT")
)


class LavitHookManager(VLMHookManager):
    """Hook manager for LaVIT-7B-v2 (understanding path).

    Parameters
    ----------
    model           : LaVITforUnderstanding in eval mode
    capture_attention_weights : opt-in; requires attn_implementation='eager'
    """

    def __init__(
        self,
        model,
        capture_attention_weights: bool = False,
    ):
        # Insert LaVIT into sys.path so LaVIT's internal imports work.
        if _LAVIT_ROOT not in sys.path:
            sys.path.insert(0, _LAVIT_ROOT)

        # Bypass base __init__ guard — set attrs directly.
        # LaVIT loads with standard HF from_pretrained; no FA2 hardcoded.
        # capture_attention_weights works as long as the LLM was loaded with
        # attn_implementation='eager'.
        self.model = model
        self.capture_attention_weights = capture_attention_weights

        self._model_device = next(model.llama_model.parameters()).device
        self._model_dtype = model.llama_model.dtype

        self._asst_suffix_len: int = self._compute_asst_suffix_len(
            model.llama_tokenizer
        )
        print(
            f"LavitHookManager: asst_suffix_len={self._asst_suffix_len}, "
            f"device={self._model_device}, dtype={self._model_dtype}",
            flush=True,
        )

    # ── Module accessors ──────────────────────────────────────────────────────

    def _get_projector(self):
        return self.model.visual_tokenizer.vit_proj

    def _get_decoder_layers(self):
        return self.model.llama_model.model.layers

    def _get_lm_head(self):
        return self.model.llama_model.lm_head

    def _get_lm_forward(self):
        return self.model.llama_model

    # ── _build_prompt ─────────────────────────────────────────────────────────

    def _build_prompt(self, image: Image.Image, question: str) -> tuple:
        """Return (prompt_input_ids, image_tensor, None).

        prompt_input_ids : tokenized "Question: {q} Answer:" — no image sentinel.
        image_tensor     : (1, 3, 224, 224) float32 on model device.
        """
        image_rgb = image.convert("RGB")
        image_tensor = self.model.processer(image_rgb)  # (3, 224, 224)
        image_tensor = image_tensor.unsqueeze(0).to(self._model_device)

        text = f"Question: {question} Answer:"
        self.model.llama_tokenizer.padding_side = "left"
        prompt_tokens = self.model.llama_tokenizer(
            [text],
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self._model_device)

        return prompt_tokens.input_ids, image_tensor, None

    # ── _prepare_embeds ────────────────────────────────────────────────────────

    def _prepare_embeds(
        self,
        input_ids,          # (1, prompt_text_len) — text-only token ids
        images_tensor,      # (1, 3, 224, 224) raw image tensor
        image_sizes,        # ignored
    ) -> tuple[torch.Tensor, int]:
        """Assemble the full inputs_embeds = [img_start, vis×N, img_end, text].

        Returns (inputs_embeds, n_image_tokens) where n_image_tokens is the
        number of true visual tokens (excluding img_start / img_end tokens).
        """
        # Seed Gumbel-softmax for reproducibility across clean/corrupt runs.
        torch.manual_seed(42)

        with self.model.maybe_autocast():
            image_embeds, image_attns = self.model.compute_dynamic_visual_embeds(
                images_tensor
            )
            prompt_embeds = self.model.llama_model.get_input_embeddings()(input_ids)

        # attention mask from tokenizer (ones for batch_size=1 with no padding)
        prompt_attn = torch.ones(
            input_ids.shape, dtype=torch.long, device=self._model_device
        )

        inputs_embeds, attention_mask = self.model.pad_input_embeds(
            image_embeds, image_attns, prompt_embeds, prompt_attn
        )
        inputs_embeds = inputs_embeds.to(self._model_dtype)

        # image_attns includes img_start + T_real visual + img_end (all ones).
        # Subtract 2 to get only the true visual token count.
        n_img_total = int(image_attns.sum().item())  # T_real + 2
        n_image_tokens = max(n_img_total - 2, 0)

        # Stash attention_mask for _call_generate.
        self._last_attn_mask = attention_mask

        return inputs_embeds, n_image_tokens

    # ── _build_token_index ─────────────────────────────────────────────────────

    def _build_token_index(
        self,
        input_ids,
        n_image_tokens: int,
        prompt_len: int,
        n_generated: int = 0,
    ) -> TokenIndex:
        """Classify positions for LaVIT's layout (no -200 sentinel).

        Layout in inputs_embeds:
          pos 0             : img_start embedding (OTHER)
          pos 1 .. N        : VISUAL (n_image_tokens = N)
          pos N+1           : img_end embedding (OTHER)
          pos N+2 .. P-S-1  : QUESTION (question text)
          pos P-S .. P-1    : OTHER (trailing " Answer:" role suffix)
          pos P .. P+G-1    : ANSWER (n_generated)
        where P = prompt_len, S = _asst_suffix_len, G = n_generated.
        """
        vis_start = 1
        vis_end = 1 + n_image_tokens
        text_start = vis_end + 1  # position after img_end

        categories = torch.zeros(prompt_len + n_generated, dtype=torch.int8)
        categories[vis_start:vis_end] = int(TokenCategory.VISUAL)

        # Question region: text_start to (prompt_len - asst_suffix_len)
        asst_len = getattr(self, "_asst_suffix_len", 0)
        q_end = prompt_len - asst_len
        if text_start < q_end:
            categories[text_start:q_end] = int(TokenCategory.QUESTION)

        if n_generated > 0:
            categories[prompt_len : prompt_len + n_generated] = int(TokenCategory.ANSWER)

        visual_range = (vis_start, vis_end)

        q_mask = categories[:prompt_len] == int(TokenCategory.QUESTION)
        q_idx = q_mask.nonzero(as_tuple=False)
        question_range = (
            (int(q_idx[0, 0].item()), int(q_idx[-1, 0].item()) + 1)
            if q_idx.numel() > 0
            else None
        )
        answer_start = prompt_len if n_generated > 0 else None

        return TokenIndex(
            categories=categories,
            visual_range=visual_range,
            question_range=question_range,
            answer_start=answer_start,
        )

    # ── visual_range ──────────────────────────────────────────────────────────

    def visual_range(self, input_ids: torch.Tensor, n_image_tokens: int) -> tuple:
        """Visual tokens occupy positions [1, 1 + n_image_tokens)."""
        return (1, 1 + n_image_tokens)

    # ── _slice_visual ─────────────────────────────────────────────────────────

    def _slice_visual(
        self,
        inputs_embeds,
        token_index: Optional[TokenIndex],
        n_image_tokens: Optional[int] = None,
        input_ids=None,
    ) -> torch.Tensor:
        if token_index is not None:
            vr = token_index.visual_range
            return inputs_embeds[0, vr[0] : vr[1], :].detach().cpu()
        # Called from run_generate before token_index is built.
        assert n_image_tokens is not None
        return inputs_embeds[0, 1 : 1 + n_image_tokens, :].detach().cpu()

    # ── _call_generate ─────────────────────────────────────────────────────────

    def _call_generate(
        self,
        input_ids,
        images_tensor,
        image_sizes,        # ignored
        max_new_tokens: int,
        **kwargs,
    ):
        """Re-run embedding assembly with hooks active, then call llama_model.generate."""
        torch.manual_seed(42)

        with self.model.maybe_autocast():
            image_embeds, image_attns = self.model.compute_dynamic_visual_embeds(
                images_tensor
            )
            prompt_embeds = self.model.llama_model.get_input_embeddings()(input_ids)

        prompt_attn = torch.ones(
            input_ids.shape, dtype=torch.long, device=self._model_device
        )
        inputs_embeds, attention_mask = self.model.pad_input_embeds(
            image_embeds, image_attns, prompt_embeds, prompt_attn
        )
        inputs_embeds = inputs_embeds.to(self._model_dtype)

        supress_range = 32000 + self.model.visual_vocab_size + 2
        suppress_tokens = list(range(32000, supress_range))

        with self.model.maybe_autocast():
            return self.model.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                suppress_tokens=suppress_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )

    # ── ASSISTANT tag detection ───────────────────────────────────────────────

    def _compute_asst_suffix_len(self, tokenizer) -> int:
        """Count tokens occupied by the trailing ' Answer:' role suffix.

        Differential tokenization: two test prompts with different questions
        → common suffix length.  Immune to BPE context effects.
        """
        def _tok(q: str) -> list:
            text = f"Question: {q} Answer:"
            ids = tokenizer(text, add_special_tokens=False).input_ids
            return ids if isinstance(ids, list) else list(ids)

        ids1 = _tok("X")
        ids2 = _tok("W V W")
        n = 0
        while n < min(len(ids1), len(ids2)):
            if ids1[-(n + 1)] == ids2[-(n + 1)]:
                n += 1
            else:
                break
        return n
