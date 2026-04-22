"""
LlavaHookManager — concrete hook manager for LLaVA-1.6.

Model assumptions
-----------------
- Architecture : LlavaLlamaForCausalLM  (llava-v1.6-vicuna-7b)
- Conv mode    : llava_v1  (SeparatorStyle.TWO, sep=" ", sep2="</s>")
- Projector    : model.model.mm_projector  (mlp2x_gelu)
- Decoder      : model.model.layers[0..31]
- mm config    : mm_patch_merge_type="spatial_unpad", image_aspect_ratio="anyres"

Token layout after prepare_inputs_labels_for_multimodal
-------------------------------------------------------
  [BOS] [system prompt + "USER: "] [N×visual_tokens] [\n question] [" ASSISTANT:"]
   OTHER        OTHER                    VISUAL            QUESTION       OTHER

The " ASSISTANT:" suffix is located by scanning for ``asst_tag_ids`` —
tokens of ``" ASSISTANT:"`` — starting just after the image block.
Isolating this in ``_locate_assistant_tag`` means the Week-3 VILA-U
subclass only needs to override that one method to keep axes comparable.

n_image_tokens calculation (LLaVA-1.6 anyres with spatial_unpad)
-----------------------------------------------------------------
  n_image_tokens = inputs_embeds_expanded.shape[1] - input_ids.shape[1] + 1
The ``+1`` accounts for the single IMAGE_TOKEN_INDEX sentinel that was
replaced by N patch embeddings.
"""

from __future__ import annotations

import sys
import os
from typing import Optional

import torch
from PIL import Image

# Make llava importable (same pattern as inference_grounding.py)
_LLAVA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
if _LLAVA_ROOT not in sys.path:
    sys.path.insert(0, _LLAVA_ROOT)

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token

from .base import VLMHookManager
from .schema import Capture, TokenCategory, TokenIndex


class LlavaHookManager(VLMHookManager):
    """Hook manager for LLaVA-1.6 (llava-v1.6-vicuna-7b and compatible).

    Parameters
    ----------
    model               : LlavaLlamaForCausalLM in eval mode
    tokenizer           : matching tokenizer
    image_processor     : matching CLIPImageProcessor (from load_pretrained_model)
    conv_mode           : conversation template key (default "llava_v1")
    capture_attention_weights : opt-in; requires attn_implementation="eager"
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        conv_mode: str = "llava_v1",
        capture_attention_weights: bool = False,
    ):
        super().__init__(model, capture_attention_weights)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = conv_mode

        mm_use = getattr(model.config, "mm_use_im_start_end", False)
        if mm_use:
            self._image_placeholder = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
        else:
            self._image_placeholder = DEFAULT_IMAGE_TOKEN

        self._model_device = next(model.parameters()).device

        # Compute how many tokens the trailing "ASSISTANT:" tag occupies.
        # We use differential tokenization (two test prompts with different
        # questions) and take the common suffix length.  This is immune to BPE
        # context effects that make standalone encode() unreliable.
        self._asst_suffix_len: int = self._compute_asst_suffix_len(
            tokenizer, conv_mode
        )

    # ── _build_prompt ─────────────────────────────────────────────────────────

    def _build_prompt(
        self, image: Image.Image, question: str
    ) -> tuple:
        """Tokenize image+question → (input_ids, images_tensor, image_sizes)."""
        qs = self._image_placeholder + "\n" + question
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self._model_device)
        )

        images_tensor = process_images(
            [image], self.image_processor, self.model.config
        ).to(self._model_device, dtype=torch.float16)

        image_sizes = [image.size]  # (width, height)

        return input_ids, images_tensor, image_sizes

    # ── _prepare_embeds ────────────────────────────────────────────────────────

    def _prepare_embeds(
        self,
        input_ids,
        images_tensor,
        image_sizes,
    ) -> tuple[torch.Tensor, int]:
        """Expand the IMAGE_TOKEN_INDEX sentinel into real patch embeddings.

        Returns (inputs_embeds, n_image_tokens).
        """
        _, _, _, _, inputs_embeds, _ = (
            self.model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, images_tensor, image_sizes
            )
        )
        n_image_tokens = inputs_embeds.shape[1] - input_ids.shape[1] + 1
        return inputs_embeds, n_image_tokens

    # ── _build_token_index ─────────────────────────────────────────────────────

    def _build_token_index(
        self,
        input_ids,
        n_image_tokens: int,
        prompt_len: int,
        n_generated: int = 0,
    ) -> TokenIndex:
        """Tag every embedding position with a TokenCategory.

        Works on the *pre-splice* input_ids (which still contains the single
        IMAGE_TOKEN_INDEX sentinel) to locate the image position and the
        ASSISTANT tag, then projects onto the post-splice sequence.
        """
        ids_1d = input_ids[0]  # (prompt_tokens_raw,)

        # 1. Image sentinel position in the raw (pre-splice) sequence
        sentinel_pos = int(
            (ids_1d == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)[0, 0].item()
        )

        # 2. Post-splice positions
        pos_img_start = sentinel_pos
        pos_img_end   = sentinel_pos + n_image_tokens
        # Everything after the image in the raw sequence is shifted right by
        # (n_image_tokens - 1) because the single sentinel became N tokens.
        shift = n_image_tokens - 1

        # 3. Locate " ASSISTANT:" tag in the raw sequence (after sentinel)
        pos_asst_tag_raw = self._locate_assistant_tag(ids_1d, sentinel_pos + 1)

        # 4. Build category tensor over the full post-splice prompt
        categories = torch.zeros(prompt_len + n_generated, dtype=torch.int8)

        # VISUAL block
        categories[pos_img_start:pos_img_end] = int(TokenCategory.VISUAL)

        # QUESTION: between end of image block and start of assistant tag
        if pos_asst_tag_raw is not None:
            pos_asst_tag_splice = pos_asst_tag_raw + shift
            q_start = pos_img_end
            q_end   = pos_asst_tag_splice
            if q_start < q_end:
                categories[q_start:q_end] = int(TokenCategory.QUESTION)
            # " ASSISTANT:" and beyond → OTHER (already 0)
        else:
            # Fallback: everything after image block is QUESTION
            categories[pos_img_end:prompt_len] = int(TokenCategory.QUESTION)

        # ANSWER block (generate mode)
        if n_generated > 0:
            categories[prompt_len:prompt_len + n_generated] = int(TokenCategory.ANSWER)

        # Derive ranges for quick access
        visual_range   = (pos_img_start, pos_img_end)
        q_mask         = (categories[:prompt_len] == int(TokenCategory.QUESTION))
        q_indices      = q_mask.nonzero(as_tuple=False)
        question_range = (
            (int(q_indices[0, 0].item()), int(q_indices[-1, 0].item()) + 1)
            if q_indices.numel() > 0 else None
        )
        answer_start = prompt_len if n_generated > 0 else None

        return TokenIndex(
            categories=categories,
            visual_range=visual_range,
            question_range=question_range,
            answer_start=answer_start,
        )

    def _compute_asst_suffix_len(self, tokenizer, conv_mode: str) -> int:
        """Return the number of raw tokens occupied by the ASSISTANT role tag.

        Compares two test prompts with different single-word questions and
        counts their common suffix.  This avoids standalone encode() which is
        sensitive to BPE context at the prompt/suffix boundary.
        """
        def _tok(q: str) -> list:
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0],
                                self._image_placeholder + "\n" + q)
            conv.append_message(conv.roles[1], None)
            ids = tokenizer_image_token(
                conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX
            )
            return ids if isinstance(ids, list) else ids.tolist()

        ids1 = _tok("X")
        ids2 = _tok("W V")
        n = 0
        while n < min(len(ids1), len(ids2)):
            if ids1[-(n + 1)] == ids2[-(n + 1)]:
                n += 1
            else:
                break
        return n

    def _locate_assistant_tag(
        self, ids_1d: torch.Tensor, search_from: int
    ) -> Optional[int]:
        """Return the pre-splice position where the ASSISTANT role tag begins.

        Length-based: avoids BPE context sensitivity of standalone tokenize().
        Override this in VILA-U / other models to match their template.
        """
        if self._asst_suffix_len <= 0:
            return None
        return len(ids_1d) - self._asst_suffix_len

    # ── _call_generate ─────────────────────────────────────────────────────────

    def _call_generate(
        self,
        input_ids,
        images_tensor,
        image_sizes,
        max_new_tokens: int,
        **kwargs,
    ):
        return self.model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            **kwargs,
        )
