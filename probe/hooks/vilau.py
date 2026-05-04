"""
VilaUHookManager — concrete hook manager for VILA-U (vila-u-7b-256 / vila-u-7b-384).

All vila_u imports are deferred to __init__ so that importing probe.hooks
without VILA-U installed (e.g., when running LLaVA tests) works fine.

Model assumptions
-----------------
- Architecture : VILAULlamaModel wrapping model.llm (LlamaForCausalLM)
- Conv mode    : vicuna_v1 (SeparatorStyle.TWO, sep=" ", sep2="</s>")
- Projector    : model.mm_projector  (MultimodalProjector, mlp2x_gelu)
- Decoder      : model.llm.model.layers[0..31]
- n_image_tokens : model.vision_tower.image_tokens (256 or 729)

Token layout (identical to LLaVA-1.6 — both use Vicuna-v1):
  [BOS] [sys + "USER: "] [N×VISUAL] [\n question] [" ASSISTANT:"]
   OTHER      OTHER          VISUAL      QUESTION        OTHER

FlashAttention2
---------------
Published checkpoints hardcode FA2 in config.json.  FA2 returns None for
attention weights and disrupts o_proj output hooks.  _patch_flash_attn()
rebinds each layer's forward to the parent LlamaAttention forward (eager).
Applied unconditionally in __init__.

Generate contract
-----------------
model.generate() is @torch.inference_mode() — use model.llm.generate(inputs_embeds=…)
directly after an explicit prepare_inputs_labels_for_multimodal call.
gen_out.sequences contains only new tokens; _extract_generated_ids handles this.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Optional

import torch
from PIL import Image

from .base import VLMHookManager

_VILAU_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
)


class VilaUHookManager(VLMHookManager):
    """Hook manager for VILA-U (vila-u-7b-256 and vila-u-7b-384).

    Parameters
    ----------
    model               : VILAULlamaModel in eval mode
    tokenizer           : matching tokenizer (from load_pretrained_model)
    image_processor     : model.vision_tower.image_processor
    conv_mode           : conversation template key (default "vicuna_v1")
    capture_attention_weights : opt-in; FA2 patch always applies so this
                          works even with a checkpoint that has FA2 in config
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        conv_mode: str = "vicuna_v1",
        capture_attention_weights: bool = False,
    ):
        # Make vila_u importable — must happen before any vila_u import below.
        if _VILAU_ROOT not in sys.path:
            sys.path.insert(0, _VILAU_ROOT)

        # Lazy imports: avoided at module level so LLaVA tests don't require
        # vila_u (or its loguru dependency) to be installed.
        from vila_u.constants import IMAGE_TOKEN_INDEX
        import vila_u.conversation as vila_conv
        from vila_u.mm_utils import tokenizer_image_token, process_images
        from vila_u.utils.media import extract_media
        from vila_u.utils.tokenizer import tokenize_conversation

        # Store vila_u callables as instance attrs — used by _build_prompt etc.
        self._IMAGE_TOKEN_INDEX   = IMAGE_TOKEN_INDEX
        self._vila_conv           = vila_conv
        self._tokenizer_image_token = tokenizer_image_token
        self._process_images      = process_images
        self._extract_media       = extract_media
        self._tokenize_conversation = tokenize_conversation

        # Apply FA2 patch before base init (base checks _attn_implementation,
        # which may be misleading for VILA-U; we skip the guard and set attrs
        # directly).
        n_patched = self._patch_flash_attn(model)
        print(
            f"VilaUHookManager: patched {n_patched} FlashAttention2 "
            "layers to eager forward",
            flush=True,
        )

        # Set base-class attrs directly to bypass the attn_implementation
        # guard (FA2 patch already makes the runtime eager).
        self.model = model
        self.capture_attention_weights = capture_attention_weights

        self.tokenizer  = tokenizer
        self.image_processor = image_processor
        self.conv_mode  = conv_mode
        self._model_device = next(model.llm.parameters()).device
        self._model_dtype  = model.llm.dtype

        self._asst_suffix_len: int = self._compute_asst_suffix_len(
            tokenizer, conv_mode
        )

    # ── Module accessors ──────────────────────────────────────────────────────

    def _get_projector(self):
        return self.model.mm_projector

    def _get_decoder_layers(self):
        return self.model.llm.model.layers

    def _get_lm_head(self):
        return self.model.llm.lm_head

    def _get_lm_forward(self):
        return self.model.llm

    # ── FA2 monkey-patch ──────────────────────────────────────────────────────

    @staticmethod
    def _patch_flash_attn(model) -> int:
        """Rebind LlamaFlashAttention2.forward → parent LlamaAttention.forward.

        Returns the number of layers patched (0 if checkpoint already uses eager).
        """
        n = 0
        for layer in model.llm.model.layers:
            attn = layer.self_attn
            if attn.__class__.__name__ == "LlamaFlashAttention2":
                parent_fwd = attn.__class__.__bases__[0].forward
                attn.forward = types.MethodType(parent_fwd, attn)
                n += 1
        return n

    # ── _build_prompt ─────────────────────────────────────────────────────────

    def _build_prompt(self, image: Image.Image, question: str) -> tuple:
        """Build (input_ids, images_tensor, None) from a PIL image and question.

        A fresh conversation list is built each call because extract_media
        mutates message["value"] in-place.
        """
        conversation = [{"from": "human", "value": [image, question]}]
        media = self._extract_media(conversation, self.model.config)
        # conversation[0]["value"] is now "<image>\nquestion" (string)

        images_tensor = self._process_images(
            media["image"], self.image_processor, self.model.config
        ).to(self._model_device, dtype=self._model_dtype)

        input_ids = (
            self._tokenize_conversation(
                conversation, self.tokenizer, add_generation_prompt=True
            )
            .unsqueeze(0)
            .to(self._model_device)
        )

        return input_ids, images_tensor, None  # image_sizes not used by VILA-U

    # ── _prepare_embeds ────────────────────────────────────────────────────────

    def _prepare_embeds(
        self,
        input_ids,
        images_tensor,
        image_sizes,  # ignored — VILA-U has no image_sizes arg
    ) -> tuple[torch.Tensor, int]:
        """Expand the IMAGE_TOKEN_INDEX sentinel into visual patch embeddings."""
        _, _, _, _, inputs_embeds, _ = (
            self.model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, images_tensor
            )
        )
        inputs_embeds = inputs_embeds.to(self._model_dtype)
        n_image_tokens = self.model.vision_tower.image_tokens
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
        # Re-run prepare so the projector hook fires with hooks active.
        # Bypass model.generate() which is @torch.inference_mode() decorated.
        _, _, attn_mask, _, embeds, _ = (
            self.model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, images_tensor
            )
        )
        embeds = embeds.to(self._model_dtype)
        return self.model.llm.generate(
            inputs_embeds=embeds,
            attention_mask=attn_mask,
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

    # ── VQ codes ──────────────────────────────────────────────────────────────

    def _get_vq_codes(self, image: Image.Image):
        """Return RQ-VAE code indices from VILA-U's RQVAE-SigLIP vision tower.

        Returns one level per RQ depth slot (shape: h*w,) as int64 CPU tensors.
        """
        img = image.convert("RGB")
        images_tensor = self._process_images(
            [img], self.image_processor, self.model.config
        ).to(self._model_device, dtype=self._model_dtype)

        vision_tower = self.model.vision_tower.vision_tower.rqvaesiglip
        codes, _ = vision_tower.encode_image(images_tensor)  # (1, h, w, d)
        d = codes.shape[3]

        quantizer = vision_tower.quantizer
        cb_size = getattr(quantizer, "codebook_size", None)
        if cb_size is None:
            cb_size = int(codes.max().item()) + 1
        codebook_sizes = [int(cb_size)] * d

        levels = [codes[0, :, :, i].reshape(-1).cpu() for i in range(d)]
        return {"levels": levels, "codebook_sizes": codebook_sizes}

    # ── ASSISTANT tag detection ───────────────────────────────────────────────

    def _compute_asst_suffix_len(self, tokenizer, conv_mode: str) -> int:
        """Count tokens occupied by the trailing " ASSISTANT:" role tag.

        Differential tokenization: two test prompts with different questions
        → common suffix length.  Immune to BPE context effects.
        """
        def _tok(q: str) -> list:
            conv = self._vila_conv.conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            ids = self._tokenizer_image_token(prompt, tokenizer)
            return ids if isinstance(ids, list) else list(ids)

        ids1 = _tok("X")
        ids2 = _tok("W V")
        n = 0
        while n < min(len(ids1), len(ids2)):
            if ids1[-(n + 1)] == ids2[-(n + 1)]:
                n += 1
            else:
                break
        return n
