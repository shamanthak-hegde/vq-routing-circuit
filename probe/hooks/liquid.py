"""
LiquidHookManager — hook manager for Liquid_V1_7B (unified VQ-VLM).

Architecture
------------
Liquid is a Chameleon-based unified-tokenizer VLM:
  - LM          : AutoModelForCausalLM (Chameleon architecture, HF-compatible)
  - VQ tokenizer: VQGAN from Meta's chameleon repo (ImageTokenizer)
  - Token layout: image VQ codes are offset by the text vocab size and placed
                  directly in the shared embedding, delimited by <boi>/<eoi> markers.

VQ code encoding
----------------
  raw_code  ∈ [0, vqgan_codebook_size)          (from image_tokenizer)
  token_id  = raw_code + text_vocab_size         (injected into input_ids)

This is identical in spirit to Emu3's unified-vocab approach.  We use the same
_Emu3VisualEmbedder thin-wrapper trick to satisfy the projector-hook contract.

Token layout (Gemma template)
------------------------------
  <start_of_turn>user\\n<boi>[vq_codes]<eoi>\\n{question}<end_of_turn>\\n<start_of_turn>model\\n
   OTHER                   OTHER  VISUAL          OTHER  QUESTION  OTHER

n_image_tokens = number of VQ codes from VQGAN (typically 1024 for 512×512 input).

Requirements
------------
  - chameleon repo must be on sys.path so that
      chameleon.inference.image_tokenizer.ImageTokenizer is importable.
  - Model loaded with attn_implementation="eager" (flash_attn blocks hooks).
  - vqgan_cfg_path and vqgan_ckpt_path must point to the chameleon VQGAN weights.

Conda env: sae  (or whichever env has the chameleon package + Liquid's HF model)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
from PIL import Image
from torch import nn

from .base import VLMHookManager
from .schema import TokenCategory, TokenIndex

_CHAMELEON_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "chameleon")
)
_LIQUID_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Liquid")
)

_GEMMA_ASST_OPEN = "<start_of_turn>model\n"


class _LiquidVisualEmbedder(nn.Module):
    """Thin hook-target that calls embed_tokens on the visual block IDs.

    Identical purpose to _Emu3VisualEmbedder: gives register_captures a
    hookable module whose forward fires exactly once per run_prefill /
    _call_generate, satisfying finalize_store's projector-hook assertion.
    """

    def __init__(self, embed_tokens: nn.Embedding) -> None:
        super().__init__()
        self.__dict__["_embed_ref"] = embed_tokens

    def forward(self, vis_ids: torch.Tensor) -> torch.Tensor:
        # vis_ids: (1, n_vis) int64 — the VQ code IDs (offset by vocab_size)
        return self.__dict__["_embed_ref"](vis_ids)


class LiquidHookManager(VLMHookManager):
    """Hook manager for Junfeng5/Liquid_V1_7B (Chameleon-based unified VQ-VLM).

    Parameters
    ----------
    model             : AutoModelForCausalLM (Chameleon arch), loaded with
                        attn_implementation="eager"
    tokenizer         : AutoTokenizer.from_pretrained(model_path)
    image_tokenizer   : ImageTokenizer (chameleon VQGAN) — call as:
                          from chameleon.inference.image_tokenizer import ImageTokenizer
                          image_tokenizer = ImageTokenizer(cfg_path=..., ckpt_path=...,
                                                          device="cuda:0")
    image_size        : resize both dims of input images to this value before
                        VQ-encoding. Default 512 (matches Liquid's eval script).
    capture_attention_weights : opt-in; requires attn_implementation="eager"
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_tokenizer,
        image_size: int = 512,
        capture_attention_weights: bool = False,
    ) -> None:
        for p in (_CHAMELEON_ROOT, _LIQUID_ROOT):
            if p not in sys.path:
                sys.path.insert(0, p)

        self.model = model
        self.tokenizer = tokenizer
        self.image_tokenizer = image_tokenizer
        self.image_size = image_size
        self.capture_attention_weights = capture_attention_weights

        self._model_device = next(model.parameters()).device
        self._model_dtype = next(model.parameters()).dtype
        self._text_vocab_size: int = len(tokenizer)

        # Special delimiter token IDs — look them up from the tokenizer.
        # Liquid prompt uses <boi> and <eoi> as plain text markers; if they are
        # not registered as special tokens they will each tokenize as subwords.
        # We look them up with add_special_tokens=False and take the first/last
        # token ID respectively, then scan input_ids at runtime.
        boi_ids = tokenizer.encode("<boi>", add_special_tokens=False)
        eoi_ids = tokenizer.encode("<eoi>", add_special_tokens=False)
        self._boi_id: int = boi_ids[0] if len(boi_ids) == 1 else -1
        self._eoi_id: int = eoi_ids[0] if len(eoi_ids) == 1 else -1
        self._n_boi_tokens: int = len(boi_ids)
        self._n_eoi_tokens: int = len(eoi_ids)
        self._boi_ids_seq = boi_ids
        self._eoi_ids_seq = eoi_ids

        # Thin projector wrapper
        self._visual_embedder = _LiquidVisualEmbedder(model.get_input_embeddings())

        # ASSISTANT tag detection (differential tokenization)
        self._asst_suffix_len: int = self._compute_asst_suffix_len()

        print(
            f"LiquidHookManager: n_layers={len(model.model.layers)}, "
            f"hidden={model.config.hidden_size}, text_vocab={self._text_vocab_size}, "
            f"device={self._model_device}, image_size={image_size}",
            flush=True,
        )

    # ── Module accessors ──────────────────────────────────────────────────────

    def _get_projector(self):
        return self._visual_embedder

    def _get_decoder_layers(self):
        return self.model.model.layers

    def _get_lm_head(self):
        return self.model.lm_head

    def _get_lm_forward(self):
        return self.model

    # ── _build_prompt ─────────────────────────────────────────────────────────

    def _build_prompt(self, image: Image.Image, question: str) -> tuple:
        """Tokenize image + question into input_ids with VQ codes in-place.

        Returns (input_ids, None, None).  Visual tokens are the VQ codes
        offset by the text vocab size, embedded in the shared vocabulary.
        """
        # Resize image (Liquid eval uses 512×512)
        img = image.convert("RGB")
        pad = _expand2square(img, (122, 116, 104))
        resized = pad.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Get VQ codes (1-D tensor in [0, codebook_size))
        with torch.no_grad():
            vq_code = self.image_tokenizer.img_tokens_from_pil(resized).cpu()
        vq_ids = (vq_code + self._text_vocab_size).long()  # shift into unified vocab

        # Build Gemma-style prompt text around <boi><image><eoi>
        boi_text = "<boi>"
        eoi_text = "<eoi>"
        prefix_text = f"<start_of_turn>user\n{boi_text}"
        suffix_text = f"{eoi_text}\n{question}<end_of_turn>\n{_GEMMA_ASST_OPEN}"

        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=True,
                                           return_tensors="pt")[0]
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False,
                                           return_tensors="pt")[0]

        input_ids = torch.cat([prefix_ids, vq_ids, suffix_ids]).unsqueeze(0)
        input_ids = input_ids.to(self._model_device)
        return input_ids, None, None

    # ── _prepare_embeds ───────────────────────────────────────────────────────

    def _prepare_embeds(
        self,
        input_ids,
        _images_tensor,
        _image_sizes,
    ) -> tuple[torch.Tensor, int]:
        """Embed all tokens (text + visual) via embed_tokens.

        Also fires the projector hook by calling _visual_embedder on the
        visual block.  Detects visual tokens by their ID range (>= text_vocab_size).
        """
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        vis_lo, vis_hi = self._find_visual_range(input_ids[0])
        n_image_tokens = vis_hi - vis_lo

        with torch.no_grad():
            self._visual_embedder(input_ids[:, vis_lo:vis_hi])

        return inputs_embeds, n_image_tokens

    # ── visual_range ──────────────────────────────────────────────────────────

    def visual_range(self, input_ids, n_image_tokens: int) -> tuple[int, int]:
        ids_1d = input_ids[0] if input_ids.ndim == 2 else input_ids
        return self._find_visual_range(ids_1d)

    def _find_visual_range(self, ids_1d: torch.Tensor) -> tuple[int, int]:
        """Find the contiguous block of VQ-code IDs (>= text_vocab_size)."""
        is_vis = ids_1d >= self._text_vocab_size
        vis_positions = is_vis.nonzero(as_tuple=False)[:, 0]
        if vis_positions.numel() == 0:
            raise RuntimeError("No visual tokens found in Liquid input_ids")
        return int(vis_positions[0].item()), int(vis_positions[-1].item()) + 1

    # ── _build_token_index ────────────────────────────────────────────────────

    def _build_token_index(
        self,
        input_ids,
        n_image_tokens: int,
        prompt_len: int,
        n_generated: int = 0,
    ) -> TokenIndex:
        ids_1d = input_ids[0]
        vis_lo, vis_hi = self._find_visual_range(ids_1d)
        asst_pos = len(ids_1d) - self._asst_suffix_len

        categories = torch.zeros(prompt_len + n_generated, dtype=torch.int8)
        categories[vis_lo:vis_hi] = int(TokenCategory.VISUAL)

        q_start = vis_hi
        if q_start < asst_pos:
            categories[q_start:asst_pos] = int(TokenCategory.QUESTION)

        if n_generated > 0:
            categories[prompt_len:prompt_len + n_generated] = int(TokenCategory.ANSWER)

        q_range = (q_start, asst_pos) if q_start < asst_pos else None
        return TokenIndex(
            categories=categories,
            visual_range=(vis_lo, vis_hi),
            question_range=q_range,
            answer_start=prompt_len if n_generated > 0 else None,
        )

    # ── _call_generate ────────────────────────────────────────────────────────

    def _call_generate(
        self,
        input_ids,
        _images_tensor,
        _image_sizes,
        max_new_tokens: int,
        **kwargs,
    ):
        # Fire projector hook to satisfy finalize_store's assertion
        vis_lo, vis_hi = self._find_visual_range(input_ids[0])
        with torch.no_grad():
            self._visual_embedder(input_ids[:, vis_lo:vis_hi])

        return self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )

    # ── _extract_input_ids_for_capture ────────────────────────────────────────

    def _extract_input_ids_for_capture(self, input_ids) -> torch.Tensor:
        return input_ids.squeeze(0).cpu()

    # ── _slice_visual ─────────────────────────────────────────────────────────

    def _slice_visual(self, inputs_embeds, token_index, n_image_tokens=None,
                      input_ids=None) -> torch.Tensor:
        if token_index is not None:
            vr = token_index.visual_range
            return inputs_embeds[0, vr[0]:vr[1], :].detach().cpu()
        vis_lo, vis_hi = self._find_visual_range(input_ids[0])
        return inputs_embeds[0, vis_lo:vis_hi, :].detach().cpu()

    # ── ASSISTANT tag suffix ───────────────────────────────────────────────────

    def _compute_asst_suffix_len(self) -> int:
        """Differential tokenization to find the trailing assistant-tag length."""
        def _tok(q: str) -> list:
            text = (f"<start_of_turn>user\n<boi>PLACEHOLDER<eoi>\n"
                    f"{q}<end_of_turn>\n{_GEMMA_ASST_OPEN}")
            return self.tokenizer.encode(text, add_special_tokens=True)

        ids1 = _tok("X")
        ids2 = _tok("W V")
        n = 0
        while n < min(len(ids1), len(ids2)):
            if ids1[-(n + 1)] == ids2[-(n + 1)]:
                n += 1
            else:
                break
        return n

    def _locate_assistant_tag(self, ids_1d: torch.Tensor, search_from: int) -> Optional[int]:
        suffix_len = getattr(self, "_asst_suffix_len", 0)
        if suffix_len <= 0:
            return None
        return len(ids_1d) - suffix_len

    # ── VQ codes ──────────────────────────────────────────────────────────────

    def _get_vq_codes(self, image: Image.Image):
        """Return raw VQ code indices from the Chameleon VQGAN.

        Reuses the same resize + pad + img_tokens_from_pil path as _build_prompt.
        Codebook size is read from the quantizer; falls back to max(codes)+1.
        """
        img = image.convert("RGB")
        pad = _expand2square(img, (122, 116, 104))
        resized = pad.resize((self.image_size, self.image_size), Image.LANCZOS)
        with torch.no_grad():
            codes = self.image_tokenizer.img_tokens_from_pil(resized).cpu()
        # codes: 1-D LongTensor in [0, codebook_size)
        # ImageTokenizer stores the VQModel as _vq_model (private attribute)
        try:
            n_vq = int(self.image_tokenizer._vq_model.quantize.n_e)
        except AttributeError:
            n_vq = int(codes.max().item()) + 1
        return {"levels": [codes], "codebook_sizes": [n_vq]}


# ── Image utilities ────────────────────────────────────────────────────────────

def _expand2square(pil_img: Image.Image, background_color: tuple) -> Image.Image:
    """Pad to square by adding background on the short side."""
    w, h = pil_img.size
    if w == h:
        return pil_img
    size = max(w, h)
    result = Image.new(pil_img.mode, (size, size), background_color)
    paste_x = (size - w) // 2
    paste_y = (size - h) // 2
    result.paste(pil_img, (paste_x, paste_y))
    return result
