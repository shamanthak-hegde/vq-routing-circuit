"""
VilaHookManager — concrete hook manager for VILA v1.0 (Efficient-Large-Model/VILA-7b).

All vila imports are deferred to __init__ so that importing probe.hooks
without VILA installed (e.g., when running LLaVA tests) works fine.

Model assumptions
-----------------
- Architecture : LlavaLlamaForCausalLM (VILA fork of LLaVA-1.5 codebase)
- Conv mode    : vicuna_v1 (SeparatorStyle.TWO, sep=" ", sep2="</s>")
- Projector    : model.model.mm_projector
- Decoder      : model.model.layers[0..31]
- LM head      : model.lm_head
- n_image_tokens: inputs_embeds.shape[1] - input_ids.shape[1] + 1
  (576 for VILA-7b with CLIP ViT-L/14-336: (336//14)**2)

Token layout (identical to VILA-U — both use Vicuna-v1):
  [BOS] [sys + "USER: "] [N×VISUAL] [\\n question] [" ASSISTANT:"]
   OTHER      OTHER          VISUAL      QUESTION        OTHER

FlashAttention2
---------------
Vendored transformers (v4.36.2 with llava/train/transformers_replace patches)
hardcodes LlamaFlashAttention2 in LlamaDecoderLayer.__init__.
_patch_flash_attn() rebinds each layer's self_attn class to LlamaAttention
(the parent). No state-dict copy needed — LlamaFlashAttention2 only overrides
forward(), so all weight attrs are already present on the instance.

Generate contract
-----------------
LlavaLlamaForCausalLM does not override generate() and is not decorated with
@torch.inference_mode. Use model.generate(input_ids, images=...) directly —
the LLaVA-1.5 pattern. Hooks fire normally during generate.
"""

from __future__ import annotations

import os
import sys

import torch
from PIL import Image

from .base import VLMHookManager

_VILA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "VILA")
)


class VilaHookManager(VLMHookManager):
    """Hook manager for VILA v1.0 (Efficient-Large-Model/VILA-7b).

    Parameters
    ----------
    model               : LlavaLlamaForCausalLM (VILA fork) in eval mode
    tokenizer           : matching tokenizer
    image_processor     : matching CLIPImageProcessor (from load_pretrained_model)
    conv_mode           : conversation template key (default "vicuna_v1")
    capture_attention_weights : opt-in; FA2 patch always applies
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        conv_mode: str = "vicuna_v1",
        capture_attention_weights: bool = False,
    ):
        if _VILA_ROOT not in sys.path:
            sys.path.insert(0, _VILA_ROOT)

        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        self._DEFAULT_IMAGE_TOKEN   = DEFAULT_IMAGE_TOKEN
        self._IMAGE_TOKEN_INDEX     = IMAGE_TOKEN_INDEX
        self._conv_templates        = conv_templates
        self._process_images        = process_images
        self._tokenizer_image_token = tokenizer_image_token

        n_patched = self._patch_flash_attn(model)
        print(
            f"VilaHookManager: patched {n_patched} FlashAttention2 "
            "layers to eager forward",
            flush=True,
        )

        # Set base-class attrs directly to bypass the attn_implementation guard
        # (the FA2 patch already makes the runtime eager).
        self.model = model
        self.capture_attention_weights = capture_attention_weights

        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.conv_mode       = conv_mode
        self._model_device   = next(model.parameters()).device
        self._model_dtype    = next(model.parameters()).dtype

        self._asst_suffix_len: int = self._compute_asst_suffix_len(
            tokenizer, conv_mode
        )

    # ── FA2 monkey-patch ──────────────────────────────────────────────────────

    @staticmethod
    def _patch_flash_attn(model) -> int:
        """Rebind LlamaFlashAttention2 → LlamaAttention by class substitution.

        LlamaFlashAttention2 only overrides forward(); all weight tensors are
        inherited from LlamaAttention. Rebinding __class__ restores eager forward
        without any state-dict copy. Returns the count of layers patched.
        """
        from transformers.models.llama.modeling_llama import LlamaAttention
        n = 0
        for layer in model.model.layers:
            attn = layer.self_attn
            if attn.__class__.__name__ == "LlamaFlashAttention2":
                attn.__class__ = LlamaAttention
                n += 1
        return n

    # ── Module accessors ──────────────────────────────────────────────────────

    def _get_lm_forward(self):
        """Return a forward callable that injects a required attention_mask.

        VILA's LlavaLlamaForCausalLM.forward() calls attention_mask.sum(-1)
        unconditionally for packed-sequence seqlen dispatch, so passing
        attention_mask=None (the base-class default) raises AttributeError.
        This wrapper synthesises an all-ones mask from inputs_embeds.shape.
        """
        model = self.model
        device = self._model_device

        def _forward(inputs_embeds, **kwargs):
            attn_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.long, device=device
            )
            return model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                **kwargs,
            )

        return _forward

    # ── _build_prompt ─────────────────────────────────────────────────────────

    def _build_prompt(self, image: Image.Image, question: str) -> tuple:
        """Tokenize image+question → (input_ids, images_tensor, None)."""
        conv = self._conv_templates[self.conv_mode].copy()
        prior = getattr(self, "_followup_prior", None)
        if prior:
            pq, pa = prior
            conv.append_message(conv.roles[0], self._DEFAULT_IMAGE_TOKEN + "\n" + pq)
            conv.append_message(conv.roles[1], pa)
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
        else:
            conv.append_message(conv.roles[0], self._DEFAULT_IMAGE_TOKEN + "\n" + question)
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
        ).to(self._model_device, dtype=self._model_dtype)

        return input_ids, images_tensor, None  # image_sizes not used by VILA v1.0

    # ── _prepare_embeds ────────────────────────────────────────────────────────

    def _prepare_embeds(
        self,
        input_ids,
        images_tensor,
        image_sizes,  # ignored — VILA v1.0 has no image_sizes arg
    ) -> tuple[torch.Tensor, int]:
        """Expand IMAGE_TOKEN_INDEX sentinel into visual patch embeddings."""
        _, _, _, _, inputs_embeds, _ = (
            self.model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, images_tensor
            )
        )
        n_image_tokens = inputs_embeds.shape[1] - input_ids.shape[1] + 1
        return inputs_embeds, n_image_tokens

    # ── _call_generate ─────────────────────────────────────────────────────────

    def _call_generate(
        self,
        input_ids,
        images_tensor,
        image_sizes,  # ignored
        max_new_tokens: int,
        **kwargs,
    ):
        # output_scores=True is required: VILA uses patched transformers 4.36.2
        # which lacks logits_to_keep, so lm_head fires on the full prompt sequence
        # at the prefill step (shape: prompt_len × vocab).  The base class scores
        # path slices tensors["logits"][prompt_len-1:][:n_scores] to extract one
        # decision logit per generated token, which only works when gen_out.scores
        # provides the authoritative count of decode steps.
        return self.model.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )

    # ── ASSISTANT tag detection ───────────────────────────────────────────────

    def _compute_asst_suffix_len(self, tokenizer, conv_mode: str) -> int:
        """Count tokens occupied by the trailing " ASSISTANT:" role tag.

        Differential tokenization: two test prompts with different questions
        → common suffix length.  Immune to BPE context effects.
        """
        def _tok(q: str) -> list:
            conv = self._conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], self._DEFAULT_IMAGE_TOKEN + "\n" + q)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            ids = self._tokenizer_image_token(
                prompt, tokenizer, self._IMAGE_TOKEN_INDEX
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
