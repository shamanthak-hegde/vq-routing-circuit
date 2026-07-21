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
  [BOS] [system prompt + "USER: "] [N×visual_tokens] [\\n question] [" ASSISTANT:"]
   OTHER        OTHER                    VISUAL            QUESTION       OTHER

The " ASSISTANT:" suffix is located by scanning for ``asst_tag_ids`` —
tokens of ``" ASSISTANT:"`` — starting just after the image block.
Isolating this in ``_locate_assistant_tag`` means the VILA subclass only
needs to override that one method to keep axes comparable.

n_image_tokens calculation (LLaVA-1.6 anyres with spatial_unpad)
-----------------------------------------------------------------
  n_image_tokens = inputs_embeds_expanded.shape[1] - input_ids.shape[1] + 1
The ``+1`` accounts for the single IMAGE_TOKEN_INDEX sentinel that was
replaced by N patch embeddings.

All LLaVA-specific imports are deferred to __init__ (lazy) so that importing
probe.hooks in a non-LLaVA environment does not pollute sys.modules with
llava.* module cache entries — which would interfere with the VILA hook manager
(same llava.* namespace, different vendored subtree).
"""

from __future__ import annotations

import os
import sys

import torch
from PIL import Image

from .base import VLMHookManager

_LLAVA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
)


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

        self._DEFAULT_IMAGE_TOKEN   = DEFAULT_IMAGE_TOKEN
        self._DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self._DEFAULT_IM_END_TOKEN   = DEFAULT_IM_END_TOKEN
        self._IMAGE_TOKEN_INDEX      = IMAGE_TOKEN_INDEX
        self._conv_templates         = conv_templates
        self._process_images         = process_images
        self._tokenizer_image_token  = tokenizer_image_token

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

        self._asst_suffix_len: int = self._compute_asst_suffix_len(
            tokenizer, conv_mode
        )

    # ── _build_prompt ─────────────────────────────────────────────────────────

    def _build_prompt(
        self, image: Image.Image, question: str
    ) -> tuple:
        """Tokenize image+question → (input_ids, images_tensor, image_sizes)."""
        conv = self._conv_templates[self.conv_mode].copy()
        prior = getattr(self, "_followup_prior", None)
        if prior:
            pq, pa = prior
            conv.append_message(conv.roles[0], self._image_placeholder + "\n" + pq)
            conv.append_message(conv.roles[1], pa)
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
        else:
            conv.append_message(conv.roles[0], self._image_placeholder + "\n" + question)
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            self._tokenizer_image_token(
                prompt, self.tokenizer, self._IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self._model_device)
        )

        images_tensor = self._process_images(
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

    def _compute_asst_suffix_len(self, tokenizer, conv_mode: str) -> int:
        """Return the number of raw tokens occupied by the ASSISTANT role tag.

        Compares two test prompts with different single-word questions and
        counts their common suffix.  This avoids standalone encode() which is
        sensitive to BPE context at the prompt/suffix boundary.
        """
        def _tok(q: str) -> list:
            conv = self._conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0],
                                self._image_placeholder + "\n" + q)
            conv.append_message(conv.roles[1], None)
            ids = self._tokenizer_image_token(
                conv.get_prompt(), self.tokenizer, self._IMAGE_TOKEN_INDEX
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
