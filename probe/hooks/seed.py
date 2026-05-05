"""
SeedHookManager — concrete hook manager for SEED-LLaMA-8B.

Architecture
------------
- LLM backbone      : LlamaForCausalLM (xformers; SEED/models/llama_xformer.py)
                      32 layers, hidden=4096, vocab=40192 (32000 text + 8192 image)
- Visual tokenizer  : SeedLlamaTokenizer wrapping Blip2QformerQuantizer (Q-Former VQ)
                      Codebook size 8192; 32 codes per image
- Decoder layers    : model.model.layers
- LM head           : model.lm_head

Visual pipeline
---------------
SEED has NO projector module. The image path is purely discrete:
  1. PIL image → SeedLlamaTokenizer.encode_image() → 32 codebook indices (int64)
  2. Indices shifted into LLM vocab: idx + IMAGE_ID_SHIFT (32000)
  3. Shifted IDs concatenated into input_ids between <img> and </img> text sentinels
  4. model.model.embed_tokens(input_ids) embeds ALL tokens uniformly — no projector.

The "projector hook" contract is satisfied by _SeedVisualEmbedder: a thin nn.Module
that re-calls embed_tokens on the [BOI+1 : EOI) visual block when invoked, firing
once per run_prefill and once per run_generate.

Token layout
------------
  pos:   0     1..k    k+1     k+2..k+33   k+34      k+35..asst-1   asst..end
         <s>   USER:   <img>   vis×32      </img>     QUESTION        [ASSISTANT:]
         OTHER  OTHER   OTHER   VISUAL      OTHER       QUESTION        OTHER

where k+1 = BOI position, k+2 = first visual code, k+33 = last visual code,
k+34 = EOI position, and asst = prompt_len - _asst_suffix_len.

Special token IDs
-----------------
  Visual codes : input_id in [IMAGE_ID_SHIFT, IMAGE_ID_SHIFT + NUM_IMG_CODES)
                 i.e. [32000, 40192) — detected by ID range, not sentinel.

Attention backend
-----------------
The xformers LlamaForCausalLM uses xops.memory_efficient_attention with
LowerTriangularMask. For our hook captures, attention_mask=None is passed to
all forward calls; the model builds its own causal mask internally.
transformers 4.30.2 lacks logits_to_keep and cache_position — we use
output_scores=True + the lm_head hook accumulation for per-step generate logits,
identical to the vilau.py / vila.py pattern.
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

_SEED_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "SEED")
)

BOI_TOKEN = "<img>"
EOI_TOKEN = "</img>"
IMG_TOKEN = "<img_{:05d}>"
NUM_IMG_TOKENS = 32
IMAGE_ID_SHIFT = 32000
NUM_IMG_CODES = 8192


class _SeedVisualEmbedder(nn.Module):
    """Thin hook-target wrapping embed_tokens for the visual block.

    No real projector exists in SEED; this fires once per run_prefill /
    run_generate to satisfy finalize_store's ≥1 projector-hook assertion.
    Same design as _ShowoVisualEmbedder in showo.py.
    """

    def __init__(self, embed_tokens: nn.Embedding) -> None:
        super().__init__()
        self.__dict__["_embed_ref"] = embed_tokens

    def forward(self, vis_block_ids: torch.Tensor) -> torch.Tensor:
        # vis_block_ids: (1, 32) int64 — shifted codebook IDs
        # returns: (1, 32, hidden)
        return self.__dict__["_embed_ref"](vis_block_ids)


class SeedHookManager(VLMHookManager):
    """Hook manager for SEED-LLaMA-8B (xformers LLaMA + VQ-into-vocab tokenizer).

    Parameters
    ----------
    model     : LlamaForCausalLM from SEED/models/llama_xformer.py, eval mode on GPU
    tokenizer : SeedLlamaTokenizer (handles both text tokenization and image encoding)
    capture_attention_weights : not supported for xformers backend (must be False)
    """

    def __init__(
        self,
        model,
        tokenizer,
        capture_attention_weights: bool = False,
    ) -> None:
        if _SEED_ROOT not in sys.path:
            sys.path.insert(0, _SEED_ROOT)

        # Bypass base __init__ — xformers LLaMA has no _attn_implementation attribute.
        self.model = model
        self.capture_attention_weights = capture_attention_weights
        self.tokenizer = tokenizer

        if capture_attention_weights:
            raise ValueError(
                "capture_attention_weights=True is not supported for the SEED xformers backend."
            )

        self._model_device = next(model.parameters()).device
        self._model_dtype = next(model.parameters()).dtype

        # Visual-token ID range (VQ codes shifted into LLM vocab)
        self._vis_id_lo = IMAGE_ID_SHIFT
        self._vis_id_hi = IMAGE_ID_SHIFT + NUM_IMG_CODES

        # BOI/EOI token lengths (to correctly place question_start after EOI)
        self._boi_len = len(tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids)
        self._eoi_len = len(tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids)

        # Phantom projector-hook wrapper (not a model submodule, avoids double-hook)
        self._visual_embedder = _SeedVisualEmbedder(model.model.embed_tokens)

        # ASSISTANT tag suffix length for QUESTION boundary detection
        self._asst_suffix_len: int = self._compute_asst_suffix_len()

        n_layers = len(model.model.layers)
        hidden = model.config.hidden_size
        vocab = model.config.vocab_size
        print(
            f"SeedHookManager: n_layers={n_layers}, hidden={hidden}, vocab={vocab}, "
            f"device={self._model_device}, dtype={self._model_dtype}, "
            f"boi_len={self._boi_len}, eoi_len={self._eoi_len}, "
            f"asst_suffix_len={self._asst_suffix_len}",
            flush=True,
        )

    # ── Module accessors ──────────────────────────────────────────────────────

    def _get_projector(self):
        return self._visual_embedder

    # _get_decoder_layers : base default → self.model.model.layers  ✓ for SEED xformers LLaMA
    # _get_lm_head        : base default → self.model.lm_head       ✓
    # _get_lm_forward     : base default → self.model               ✓

    # ── _build_prompt ─────────────────────────────────────────────────────────

    def _build_prompt(self, image: Image.Image, question: str) -> tuple:
        """Build input_ids for SEED-LLaMA VQA.

        Layout: [<s>][USER: ][<img>][vis_codes×32][</img>][question][\\n][ASSISTANT:]
        Returns (input_ids, None, None) — no images_tensor or image_sizes.
        """
        # PIL → 32 codebook indices via SeedLlamaTokenizer.encode_image
        img_ids = self.tokenizer.encode_image(image_pil=image)  # (1, 32) int64
        img_ids = img_ids.view(-1).cpu().numpy()

        img_tokens = BOI_TOKEN + "".join(IMG_TOKEN.format(int(i)) for i in img_ids) + EOI_TOKEN

        s_token = "USER:"
        e_token = "ASSISTANT:"
        sep = "\n"
        input_str = self.tokenizer.bos_token + s_token + " " + img_tokens + question + sep + e_token

        input_ids = self.tokenizer(
            input_str, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self._model_device)

        return input_ids, None, None

    # ── _prepare_embeds ────────────────────────────────────────────────────────

    def _prepare_embeds(
        self,
        input_ids,
        images_tensor,  # None for SEED
        image_sizes,    # None for SEED
    ) -> tuple[torch.Tensor, int]:
        """Embed all tokens via embed_tokens; phantom-fire projector hook on visual block."""
        inputs_embeds = self.model.model.embed_tokens(input_ids)

        # Find the 32 contiguous visual code positions by ID range
        ids_1d = input_ids[0]
        vis_mask = (ids_1d >= self._vis_id_lo) & (ids_1d < self._vis_id_hi)
        vis_positions = vis_mask.nonzero(as_tuple=False).squeeze(-1)

        with torch.no_grad():
            # Phantom-fire to satisfy finalize_store's ≥1 projector assertion
            self._visual_embedder(
                input_ids[:, vis_positions[0] : vis_positions[-1] + 1]
            )

        return inputs_embeds, NUM_IMG_TOKENS

    # ── visual_range ───────────────────────────────────────────────────────────

    def visual_range(
        self, input_ids: torch.Tensor, n_image_tokens: int
    ) -> tuple[int, int]:
        """Return (vlo, vhi) for the visual block via VQ-code ID range."""
        ids_1d = input_ids[0]
        vis_mask = (ids_1d >= self._vis_id_lo) & (ids_1d < self._vis_id_hi)
        positions = vis_mask.nonzero(as_tuple=False).squeeze(-1)
        vlo = int(positions[0])
        return (vlo, vlo + NUM_IMG_TOKENS)

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
            return inputs_embeds[0, vr[0] : vr[1], :].detach().cpu()
        ids_1d = input_ids[0]
        vis_mask = (ids_1d >= self._vis_id_lo) & (ids_1d < self._vis_id_hi)
        positions = vis_mask.nonzero(as_tuple=False).squeeze(-1)
        vlo = int(positions[0])
        return inputs_embeds[0, vlo : vlo + NUM_IMG_TOKENS, :].detach().cpu()

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
          [<s>][USER: ][<img>][vis×32][</img>][question][\\n][ASSISTANT:][ANSWER]
           OTHER OTHER   OTHER  VISUAL  OTHER    QUESTION  OTHER  OTHER    ANSWER
        """
        ids_1d = input_ids[0]

        # Visual range: 32 contiguous tokens with id in [IMAGE_ID_SHIFT, IMAGE_ID_SHIFT+8192)
        vis_mask = (ids_1d >= self._vis_id_lo) & (ids_1d < self._vis_id_hi)
        vis_positions = vis_mask.nonzero(as_tuple=False).squeeze(-1)
        vlo = int(vis_positions[0])
        vhi = vlo + NUM_IMG_TOKENS  # exclusive; position vhi = </img> or first EOI token

        # EOI tokens immediately follow the visual block (</img> is _eoi_len tokens)
        q_start = vhi + self._eoi_len

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
        images_tensor,  # None for SEED
        image_sizes,    # None for SEED
        max_new_tokens: int,
        **kwargs,
    ):
        """Generate via xformers LlamaForCausalLM.generate.

        Fires the projector hook once (run_generate calls _prepare_embeds before
        hooks are registered, so the call there doesn't fire).

        output_scores=True is required because transformers 4.30.2 lacks
        logits_to_keep — per-step logits come from the lm_head hook accumulation
        sliced by len(gen_out.scores) in base.py:240-242.
        """
        # Phantom-fire to satisfy finalize_store's ≥1 projector assertion
        ids_1d = input_ids[0]
        vis_mask = (ids_1d >= self._vis_id_lo) & (ids_1d < self._vis_id_hi)
        vis_positions = vis_mask.nonzero(as_tuple=False).squeeze(-1)
        with torch.no_grad():
            self._visual_embedder(
                input_ids[:, vis_positions[0] : vis_positions[-1] + 1]
            )

        return self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )

    # ── ASSISTANT tag suffix length ────────────────────────────────────────────

    def _compute_asst_suffix_len(self) -> int:
        """Count trailing tokens of '\\nASSISTANT:' via common-suffix detection."""

        def _tok(q: str) -> list:
            s = self.tokenizer.bos_token + "USER: " + q + "\n" + "ASSISTANT:"
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
