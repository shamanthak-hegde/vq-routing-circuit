"""
AnyGptHookManager — hook manager for AnyGPT-chat (fnlp/AnyGPT-chat).

Architecture
------------
- LLM backbone     : LlamaForCausalLM (standard HuggingFace), 32 layers, hidden=4096
- Visual tokenizer : SEED-2 ImageTokenizer (AILab-CVC/seed-tokenizer-2)
                     Q-Former VQ, codebook 8192, 32 codes per image
- No projector     : image codes enter via embed_tokens directly (vocab-injection)

Visual pipeline
---------------
No mm_projector in AnyGPT. The image path is purely discrete:
  1. PIL image → ImageTokenizer.encode() → 32 code indices ∈ [0, 8191]
  2. Each index i is mapped to the special token string "<👀{i}>" in the extended vocab
  3. Token strings are wrapped in <soim>...</soim> and concatenated into input_ids
  4. model.model.embed_tokens(input_ids) embeds ALL tokens uniformly — no projector

The projector-hook contract is satisfied by _AnyGptVisualEmbedder: a thin nn.Module
that calls embed_tokens on the visual block, identical to the SEED / Liquid pattern.

Token layout
------------
  pos:  0     1..k   k+1    k+2..k+33    k+34    k+35..asst-1   asst..end
        <s>  [Human]:  <soim>  vis×32    <eoim>   question+<eoh>   [MMGPT]:
        OTHER  OTHER   OTHER   VISUAL     OTHER     QUESTION          OTHER

where k+1 = <soim> position, k+2 = first visual code, k+33 = last visual code,
k+34 = <eoim> position, and asst = prompt_len - _asst_suffix_len.

Predicted gate result (pre-registered in handoffs/2026-05-23_N-084_pre_registration.md):
  Gate A.1: FAIL — no discrete early peak (monotonic ramp expected, like SEED-LLaMA)
  Gate A.2: FAIL — routing distributed to late layers, L0 mass < 5
  σ_cal: very low (0.01–0.05), same tier as SEED-LLaMA (vocab-injection, no amplification)

Requirements
------------
  - AnyGPT repo must be on sys.path (or seed2/ must be importable)
  - Model loaded with attn_implementation="eager" for attention weight capture
  - image_tokenizer: ImageTokenizer from AnyGPT/seed2/seed_llama_tokenizer.py
  - LlamaForCausalLM.generate() (standard HuggingFace; supports logits_to_keep)
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

_ANYGPT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "AnyGPT")
)
_ANYGPT_SRC = os.path.join(_ANYGPT_ROOT, "anygpt", "src")

_IMAGE_PREFIX = "👀"
_IMAGE_VOCAB_SIZE = 8192
_NUM_VIS_TOKENS = 32
_SOI_TOKEN = "<soim>"
_EOI_TOKEN = "<eoim>"


def _img_code_to_str(i: int) -> str:
    return f"<{_IMAGE_PREFIX}{i}>"


class _AnyGptVisualEmbedder(nn.Module):
    """Thin hook-target wrapping embed_tokens for the visual block.

    No real projector exists in AnyGPT; this fires once per run_prefill /
    run_generate to satisfy finalize_store's ≥1 projector-hook assertion.
    Same design as _SeedVisualEmbedder in seed.py and _LiquidVisualEmbedder in liquid.py.
    """

    def __init__(self, embed_tokens: nn.Embedding) -> None:
        super().__init__()
        self.__dict__["_embed_ref"] = embed_tokens

    def forward(self, vis_ids: torch.Tensor) -> torch.Tensor:
        # vis_ids: (1, 32) int64 — image special-token IDs from extended vocab
        return self.__dict__["_embed_ref"](vis_ids)


class AnyGptHookManager(VLMHookManager):
    """Hook manager for AnyGPT-chat (LlamaForCausalLM + SEED-2 VQ).

    Parameters
    ----------
    model           : LlamaForCausalLM, eval mode on GPU, loaded with
                      attn_implementation="eager" for attention weight capture
    tokenizer       : LlamaTokenizer extended with AnyGPT multimodal tokens
    image_tokenizer : ImageTokenizer from AnyGPT/seed2/seed_llama_tokenizer.py
    capture_attention_weights : opt-in; requires attn_implementation="eager"
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_tokenizer,
        capture_attention_weights: bool = False,
    ) -> None:
        for p in (_ANYGPT_ROOT, _ANYGPT_SRC):
            if p not in sys.path:
                sys.path.insert(0, p)

        self.model = model
        self.tokenizer = tokenizer
        self.image_tokenizer = image_tokenizer
        self.capture_attention_weights = capture_attention_weights

        self._model_device = next(model.parameters()).device
        self._model_dtype = next(model.parameters()).dtype

        # Detect image token ID range from the extended tokenizer vocab.
        # AnyGPT adds <👀0>..<👀8191> as special tokens; we use the ID range
        # for visual-token detection in input_ids at runtime.
        self._vis_id_lo: int = tokenizer.convert_tokens_to_ids(_img_code_to_str(0))
        self._vis_id_hi: int = tokenizer.convert_tokens_to_ids(
            _img_code_to_str(_IMAGE_VOCAB_SIZE - 1)
        ) + 1  # exclusive

        # Sentinel token lengths
        self._soi_ids = tokenizer.encode(_SOI_TOKEN, add_special_tokens=False)
        self._eoi_ids = tokenizer.encode(_EOI_TOKEN, add_special_tokens=False)
        self._n_soi = len(self._soi_ids)
        self._n_eoi = len(self._eoi_ids)

        # Phantom projector hook wrapper
        self._visual_embedder = _AnyGptVisualEmbedder(model.model.embed_tokens)

        # ASSISTANT tag suffix length
        self._asst_suffix_len: int = self._compute_asst_suffix_len()

        n_layers = len(model.model.layers)
        hidden = model.config.hidden_size
        vocab = model.config.vocab_size
        print(
            f"AnyGptHookManager: n_layers={n_layers}, hidden={hidden}, vocab={vocab}, "
            f"vis_id_lo={self._vis_id_lo}, vis_id_hi={self._vis_id_hi}, "
            f"n_soi={self._n_soi}, n_eoi={self._n_eoi}, "
            f"asst_suffix_len={self._asst_suffix_len}, "
            f"device={self._model_device}, dtype={self._model_dtype}",
            flush=True,
        )

    # ── Module accessors ──────────────────────────────────────────────────────

    def _get_projector(self):
        return self._visual_embedder

    # _get_decoder_layers : base default → self.model.model.layers  ✓ for LLaMA
    # _get_lm_head        : base default → self.model.lm_head       ✓
    # _get_lm_forward     : base default → self.model               ✓

    # ── _build_prompt ─────────────────────────────────────────────────────────

    def _build_prompt(self, image: Image.Image, question: str) -> tuple:
        """Build input_ids for AnyGPT-chat VQA.

        Layout: [<s>][Human]: <soim><👀...×32><eoim> {question}<eoh> [MMGPT]:

        Returns (input_ids, None, None) — no images_tensor or image_sizes.
        """
        # PIL → 32 SEED-2 code indices
        img = image.convert("RGB")
        img_tensor = self.image_tokenizer.processor(img).to(self._model_device)
        with torch.no_grad():
            code_ids = self.image_tokenizer.encode(img_tensor)  # (1, 32) int64
        code_ids_1d = code_ids.view(-1).cpu().numpy()

        # Format as <soim><👀i0>...<👀i31><eoim>
        img_str = _SOI_TOKEN + "".join(
            _img_code_to_str(int(c)) for c in code_ids_1d
        ) + _EOI_TOKEN

        # AnyGPT chat template
        prompt_str = (
            self.tokenizer.bos_token
            + "[Human]: "
            + img_str
            + " "
            + question
            + "<eoh> [MMGPT]:"
        )

        input_ids = self.tokenizer(
            prompt_str, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self._model_device)

        return input_ids, None, None

    # ── _prepare_embeds ────────────────────────────────────────────────────────

    def _prepare_embeds(
        self,
        input_ids,
        _images_tensor,
        _image_sizes,
    ) -> tuple[torch.Tensor, int]:
        """Embed all tokens via embed_tokens; phantom-fire projector hook."""
        inputs_embeds = self.model.model.embed_tokens(input_ids)

        vis_lo, vis_hi = self._find_visual_range(input_ids[0])
        with torch.no_grad():
            self._visual_embedder(input_ids[:, vis_lo:vis_hi])

        return inputs_embeds, _NUM_VIS_TOKENS

    # ── visual_range ───────────────────────────────────────────────────────────

    def visual_range(
        self, input_ids: torch.Tensor, n_image_tokens: int
    ) -> tuple[int, int]:
        vlo, vhi = self._find_visual_range(input_ids[0])
        return (vlo, vhi)

    def _find_visual_range(self, ids_1d: torch.Tensor) -> tuple[int, int]:
        """Return (vlo, vhi) for the visual block via image token ID range."""
        vis_mask = (ids_1d >= self._vis_id_lo) & (ids_1d < self._vis_id_hi)
        positions = vis_mask.nonzero(as_tuple=False).squeeze(-1)
        vlo = int(positions[0])
        vhi = int(positions[-1]) + 1
        return vlo, vhi

    # ── _slice_visual ──────────────────────────────────────────────────────────

    def _slice_visual(
        self,
        inputs_embeds: torch.Tensor,
        token_index: Optional[TokenIndex],
        n_image_tokens: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if token_index is not None:
            vr = token_index.visual_range
            return inputs_embeds[0, vr[0]:vr[1], :].detach().cpu()
        vlo, vhi = self._find_visual_range(input_ids[0])
        return inputs_embeds[0, vlo:vhi, :].detach().cpu()

    # ── _build_token_index ─────────────────────────────────────────────────────

    def _build_token_index(
        self,
        input_ids,
        n_image_tokens: int,
        prompt_len: int,
        n_generated: int = 0,
    ) -> TokenIndex:
        """Classify positions: OTHER / VISUAL / QUESTION / ANSWER.

        Layout (by construction in _build_prompt):
          [<s>][Human]: <soim> [vis×32] <eoim> {question}<eoh> [MMGPT]: [ANSWER]
           OTHER OTHER  OTHER   VISUAL   OTHER   QUESTION            OTHER  ANSWER
        """
        ids_1d = input_ids[0]

        # Visual range: 32 contiguous tokens with IDs in [vis_id_lo, vis_id_hi)
        vis_mask = (ids_1d >= self._vis_id_lo) & (ids_1d < self._vis_id_hi)
        vis_positions = vis_mask.nonzero(as_tuple=False).squeeze(-1)
        vlo = int(vis_positions[0])
        vhi = vlo + _NUM_VIS_TOKENS  # exclusive

        # Question starts after </eoim> (which follows the visual block)
        q_start = vhi + self._n_eoi

        # ASSISTANT tag: last _asst_suffix_len tokens of the prompt
        asst_pos = prompt_len - self._asst_suffix_len

        categories = torch.zeros(prompt_len + n_generated, dtype=torch.int8)
        categories[vlo:vhi] = int(TokenCategory.VISUAL)

        if q_start < asst_pos:
            categories[q_start:asst_pos] = int(TokenCategory.QUESTION)

        if n_generated > 0:
            categories[prompt_len : prompt_len + n_generated] = int(TokenCategory.ANSWER)

        vis_range = (vlo, vhi)
        q_range = (q_start, asst_pos) if q_start < asst_pos else (q_start, q_start)
        answer_start = prompt_len if n_generated > 0 else None

        return TokenIndex(
            categories=categories,
            visual_range=vis_range,
            question_range=q_range,
            answer_start=answer_start,
        )

    # ── _call_generate ─────────────────────────────────────────────────────────

    def _call_generate(
        self,
        input_ids,
        _images_tensor,
        _image_sizes,
        max_new_tokens: int,
        **kwargs,
    ):
        """Generate via standard HuggingFace LlamaForCausalLM.generate.

        Phantom-fires the projector hook to satisfy finalize_store's assertion.
        Uses output_scores=True for per-step logit capture (lm_head hook accumulation).
        """
        vis_lo, vis_hi = self._find_visual_range(input_ids[0])
        with torch.no_grad():
            self._visual_embedder(input_ids[:, vis_lo:vis_hi])

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )

    # ── _extract_input_ids_for_capture ─────────────────────────────────────────

    def _extract_input_ids_for_capture(self, input_ids) -> torch.Tensor:
        return input_ids.squeeze(0).cpu()

    # ── ASSISTANT tag suffix length ────────────────────────────────────────────

    def _compute_asst_suffix_len(self) -> int:
        """Count trailing tokens of ' [MMGPT]:' via common-suffix detection."""

        def _tok(q: str) -> list:
            s = (
                self.tokenizer.bos_token
                + "[Human]: "
                + _SOI_TOKEN
                + _EOI_TOKEN
                + " "
                + q
                + "<eoh> [MMGPT]:"
            )
            return self.tokenizer(s, add_special_tokens=False)["input_ids"]

        ids1 = _tok("Is this a cat?")
        ids2 = _tok("Does this picture show a dog and a cat running together?")
        n = 0
        while n < min(len(ids1), len(ids2)):
            if ids1[-(n + 1)] == ids2[-(n + 1)]:
                n += 1
            else:
                break
        return n
