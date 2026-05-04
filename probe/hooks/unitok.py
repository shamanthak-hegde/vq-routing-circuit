"""
UniTokHookManager — concrete hook manager for the UniTok MLLM
(FoundationVision/unitok_mllm).

Architecture
------------
- MLLM class     : MiniGeminiLlamaForCausalLM (eval/liquid/model/language_model/)
- VQ tokenizer   : UniTok (models/unitok.py), loaded from unitok_tokenizer.pth
- Projector      : model.model.multi_embedder (TokenEmbedder)
                   TokenEmbedder: (bz, 8, 256) → (bz, 256, hidden)
- Decoder layers : model.model.layers (LlamaModel.layers)
- LM head        : model.lm_head
- n_image_tokens : 256 (fixed — 8 codebooks × 256 VQ positions → 256 embeddings)
- Conv template  : vicuna_v1 (identical layout to LLaVA-1.6)
- IMAGE_TOKEN_INDEX : -200 (same as LLaVA)

Image encoding path
-------------------
1. PIL image → pad-to-square → Resize(256×256) → Normalize(0.5) → img_tensor
2. vq_model.img_to_idx(img_tensor)  → (1, 8, 256)  VQ code indices
3. vq_code.unsqueeze(0)             → (1, 1, 8, 256)  batch × n_images × codebooks × tokens
4. model.prepare_inputs_for_multimodal(..., images=vq_code, ...)
   calls multi_embedder(vq_code[0]) = multi_embedder((1, 8, 256)) → (1, 256, hidden)
   then splices the 256 visual embeddings into the prompt sequence.

Generate contract
-----------------
model.generate(input_ids, images=vq_code, ...) calls prepare_inputs_for_multimodal
internally (firing multi_embedder once, captured by the projector hook), then
delegates to LlamaForCausalLM.generate via inputs_embeds=...

gen_out.sequences contains only the newly generated token IDs (no prompt prefix),
since the prompt was given as inputs_embeds (no input_ids for the prefill).
_extract_generated_ids returns gen_out.sequences[0] directly.

Flash-attention note
--------------------
Load the MLLM with attn_implementation="eager" to avoid FA2 o_proj hook issues.
The grounding inference script also uses eager for this reason.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from .base import VLMHookManager

_UNITOK_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "UniTok")
)
_LIQUID_DIR = os.path.join(_UNITOK_ROOT, "eval", "liquid")

_N_IMAGE_TOKENS = 256  # fixed: 8 VQ codebooks → 256 output embeddings


def _expand2square(pil_img: Image.Image, background_color=(122, 116, 104)) -> Image.Image:
    w, h = pil_img.size
    if w == h:
        return pil_img
    side = max(w, h)
    result = Image.new(pil_img.mode, (side, side), background_color)
    result.paste(pil_img, ((side - w) // 2, (side - h) // 2))
    return result


class UniTokHookManager(VLMHookManager):
    """Hook manager for the UniTok MLLM (FoundationVision/unitok_mllm).

    Parameters
    ----------
    model       : MiniGeminiLlamaForCausalLM in eval mode;
                  load with attn_implementation="eager"
    tokenizer   : AutoTokenizer from load_pretrained_model
    vq_model    : UniTok VQ tokenizer loaded from unitok_tokenizer.pth;
                  must be on the same device as model, in eval mode
    conv_mode   : conversation template key (default "vicuna_v1")
    capture_attention_weights : opt-in; off by default
    """

    def __init__(
        self,
        model,
        tokenizer,
        vq_model,
        conv_mode: str = "vicuna_v1",
        capture_attention_weights: bool = False,
    ):
        # Insert UniTok eval/liquid into sys.path for lazy imports.
        for _p in (_UNITOK_ROOT, _LIQUID_DIR):
            if _p not in sys.path:
                sys.path.insert(0, _p)

        from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from conversation import conv_templates
        from mm_utils import tokenizer_image_token

        self._IMAGE_TOKEN_INDEX   = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._conv_templates      = conv_templates
        self._tokenizer_image_token = tokenizer_image_token

        # Bypass base __init__ (no attn_implementation guard needed here).
        self.model                    = model
        self.capture_attention_weights = capture_attention_weights
        self.tokenizer                = tokenizer
        self.vq_model                 = vq_model
        self.conv_mode                = conv_mode

        self._model_device = next(model.parameters()).device
        self._model_dtype  = next(model.parameters()).dtype

        self._img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self._asst_suffix_len: int = self._compute_asst_suffix_len(tokenizer, conv_mode)
        print(
            f"UniTokHookManager: asst_suffix_len={self._asst_suffix_len}, "
            f"n_image_tokens={_N_IMAGE_TOKENS}, device={self._model_device}",
            flush=True,
        )

    # ── Module accessors ──────────────────────────────────────────────────────

    def _get_projector(self):
        return self.model.model.multi_embedder

    def _get_decoder_layers(self):
        return self.model.model.layers

    def _get_lm_head(self):
        return self.model.lm_head

    def _get_lm_forward(self):
        return self.model

    # ── _build_prompt ─────────────────────────────────────────────────────────

    def _build_prompt(self, image: Image.Image, question: str) -> tuple:
        """Build (input_ids, vq_code, None) from a PIL image and question.

        vq_code shape: (1, 1, 8, 256) = (batch=1, n_images=1, codebooks=8, tokens=256).
        """
        pad_img = _expand2square(image.convert("RGB"))
        img_tensor = self._img_transform(pad_img).unsqueeze(0).to(self._model_device)
        with torch.no_grad():
            vq_code = self.vq_model.img_to_idx(img_tensor)  # (1, 8, 256)
        vq_code = vq_code.unsqueeze(0)  # (1, 1, 8, 256)

        qs = self._DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = self._conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            self._tokenizer_image_token(
                prompt, self.tokenizer, self._IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self._model_device)
        )

        return input_ids, vq_code, None  # image_sizes not used by UniTok

    # ── _prepare_embeds ────────────────────────────────────────────────────────

    def _prepare_embeds(
        self,
        input_ids,
        images_tensor,   # vq_code: (1, 1, 8, 256)
        image_sizes,     # ignored
    ) -> tuple[torch.Tensor, int]:
        """Expand IMAGE_TOKEN_INDEX sentinel into 256 VQ visual embeddings.

        Calls model.prepare_inputs_for_multimodal, which fires multi_embedder
        (the projector hook target) and splices the 256 visual token embeddings
        into the prompt sequence.

        Returns (inputs_embeds, 256) where inputs_embeds shape is
        (1, prompt_len + 255, hidden) — the IMAGE_TOKEN_INDEX is replaced by
        256 visual embeddings (net expansion: 255 extra tokens).
        """
        (_, _, _, _, inputs_embeds, _) = self.model.prepare_inputs_for_multimodal(
            input_ids, None, None, None, None, images_tensor, None
        )
        return inputs_embeds, _N_IMAGE_TOKENS

    # ── _call_generate ─────────────────────────────────────────────────────────

    def _call_generate(
        self,
        input_ids,
        images_tensor,   # vq_code: (1, 1, 8, 256)
        image_sizes,     # ignored
        max_new_tokens: int,
        **kwargs,
    ):
        """Call model.generate() with VQ image tokens.

        model.generate() calls prepare_inputs_for_multimodal internally,
        firing the multi_embedder (projector hook) once, then delegates to
        LlamaForCausalLM.generate via inputs_embeds.
        """
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

    # ── VQ codes ──────────────────────────────────────────────────────────────

    def _get_vq_codes(self, image: Image.Image):
        """Return residual VQ code indices from UniTok's VQ tokenizer.

        Returns one level per residual codebook (shape: n_tokens,) as int64 CPU tensors.
        """
        pad_img = _expand2square(image.convert("RGB"))
        img_tensor = self._img_transform(pad_img).unsqueeze(0).to(self._model_device)
        with torch.no_grad():
            vq_code = self.vq_model.img_to_idx(img_tensor)  # (1, n_codebooks, n_tokens)

        n_codebooks = vq_code.shape[1]

        # Try known attribute names for codebook size, fall back to observed max+1
        cb_size = None
        for attr in ("n_codes", "codebook_size", "num_codes", "n_embed"):
            val = getattr(self.vq_model, attr, None)
            if val is not None:
                cb_size = int(val)
                break
        if cb_size is None:
            cb_size = int(vq_code.max().item()) + 1

        levels = [vq_code[0, k, :].cpu() for k in range(n_codebooks)]
        codebook_sizes = [cb_size] * n_codebooks
        return {"levels": levels, "codebook_sizes": codebook_sizes}

    # ── ASSISTANT tag detection ───────────────────────────────────────────────

    def _compute_asst_suffix_len(self, tokenizer, conv_mode: str) -> int:
        """Count tokens occupied by the trailing " ASSISTANT:" role tag.

        Differential tokenization: two test prompts with different questions
        → common suffix length. Immune to BPE context effects.
        """
        def _tok(q: str) -> list:
            conv = self._conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            ids = self._tokenizer_image_token(
                prompt, tokenizer, self._IMAGE_TOKEN_INDEX
            )
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
