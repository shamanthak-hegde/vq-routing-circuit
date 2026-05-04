"""
Emu3HookManager — concrete hook manager for BAAI/Emu3-Chat.

Architecture
------------
- MLLM class     : Emu3ForCausalLM (BAAI/Emu3-Chat)
- VQ tokenizer   : Emu3VisionVQModel (BAAI/Emu3-VisionTokenizer), separate model
- Processor      : Emu3Processor(image_processor, image_tokenizer, tokenizer)
- Decoder layers : model.model.layers (32 flat LLaMA-style layers, no staging)
- LM head        : model.lm_head
- Hidden size    : 4096 (default Emu3 config)
- Vocab size     : 184622 (unified text + visual token IDs)
- n_image_tokens : eoi_pos - img_marker_pos - 1 (visual block span including
                   interspersed EOL row-separators and trailing EOF token)

Visual pipeline
---------------
Emu3 has NO separate projector module. The full pipeline is:
  1. PIL image → image_processor → pixel tensor
  2. VQ encoder (spatial_factor=8) → discrete 2D code grid (H×W, codebook_size=32768)
  3. Formatted as <|visual token NNNNNN|> strings → tokenized into input_ids
  4. Visual IDs are delimited by <|image token|> (IMG) and <|image end|> (EOI)
  5. LLM's embed_tokens(input_ids) embeds ALL tokens (text + visual) uniformly.

The "projector hook" requirement is satisfied by _Emu3VisualEmbedder: a thin
nn.Module that re-calls embed_tokens on the [IMG+1 : EOI) visual block when
invoked. This fires register_forward_hook exactly once per run_prefill and
once per run_generate (_call_generate fires it explicitly because run_generate
calls _prepare_embeds before hooks are registered).

Token layout after Emu3Processor (mode='U')
-------------------------------------------
  [BOS] [You are a helpful assistant. USER: ] [BOI] [{H}*{W}] [IMG]
   OTHER              OTHER                   OTHER   OTHER   OTHER
  [vis_codes/EOLs/EOF] [EOI] [question] [". ASSISTANT:"]
         VISUAL         OTHER  QUESTION       OTHER

where:
  BOI = <|image start|>  (id 151852)
  IMG = <|image token|>  (looked up via tokenizer.img_token)
  EOI = <|image end|>    (id 151853)
  vis_block = [VQ codes interspersed with H-1 EOL (row separator) tokens,
               followed by one EOF token]
  n_image_tokens = eoi_pos - img_marker_pos - 1

Chat template (Emu3Processor default)
--------------------------------------
  "You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:"
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

_EMU3_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Emu3")
)


class _Emu3VisualEmbedder(nn.Module):
    """Thin hook-target module that wraps embed_tokens for the visual block.

    Sole purpose: give register_captures a hookable nn.Module whose forward()
    fires with (1, vis_block_size, hidden) so finalize_store's projector
    assertion is satisfied. The computation is a redundant re-lookup of
    visual-block IDs through embed_tokens (identical to what the main forward
    already does for those positions).

    The embed_tokens reference is stored via __dict__ to avoid nn.Module
    submodule registration (which would affect model.parameters() counts and
    could cause unexpected double-hooks if embed_tokens itself is hooked).
    """

    def __init__(self, embed_tokens: nn.Embedding) -> None:
        super().__init__()
        self.__dict__["_embed_ref"] = embed_tokens

    def forward(self, vis_block_ids: torch.Tensor) -> torch.Tensor:
        # vis_block_ids: (1, vis_block_size) int64
        # returns:       (1, vis_block_size, hidden)
        return self.__dict__["_embed_ref"](vis_block_ids)


class Emu3HookManager(VLMHookManager):
    """Hook manager for BAAI/Emu3-Chat.

    Parameters
    ----------
    model               : Emu3ForCausalLM in eval mode;
                          load with attn_implementation="eager"
    tokenizer           : Emu3Tokenizer from BAAI/Emu3-Chat
    image_processor     : AutoImageProcessor from BAAI/Emu3-VisionTokenizer
    image_tokenizer     : Emu3VisionVQModel from BAAI/Emu3-VisionTokenizer
    max_image_size      : cap both dimensions of the input PIL image before
                          VQ-encoding. Default 256 → 32×32 = 1024 visual tokens.
                          Emu3's default processor area (~720×720) gives 8100 tokens,
                          which OOMs under eager attention on all but the largest GPUs.
                          Set to None to use the processor's native resolution.
    capture_attention_weights : opt-in; requires attn_implementation="eager"
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        image_tokenizer,
        max_image_size: Optional[int] = 256,
        capture_attention_weights: bool = False,
    ) -> None:
        if _EMU3_ROOT not in sys.path:
            sys.path.insert(0, _EMU3_ROOT)

        from emu3.mllm.processing_emu3 import Emu3Processor

        self._processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

        # Bypass base __init__ — we load with eager, no attn_implementation guard needed.
        self.model = model
        self.capture_attention_weights = capture_attention_weights
        self.tokenizer = tokenizer

        self._model_device = next(model.parameters()).device
        self._model_dtype = next(model.parameters()).dtype

        # Special token IDs used for visual-range detection
        self._boi_id = tokenizer.convert_tokens_to_ids(tokenizer.boi_token)
        self._eoi_id = tokenizer.convert_tokens_to_ids(tokenizer.eoi_token)
        self._img_marker_id = tokenizer.convert_tokens_to_ids(tokenizer.img_token)

        # Thin projector-hook wrapper (not a model submodule)
        self._visual_embedder = _Emu3VisualEmbedder(model.get_input_embeddings())
        self._max_image_size = max_image_size

        # ASSISTANT tag suffix length (differential tokenization trick)
        self._asst_suffix_len: int = self._compute_asst_suffix_len()

        print(
            f"Emu3HookManager: n_layers={len(model.model.layers)}, "
            f"hidden={model.config.hidden_size}, vocab={model.config.vocab_size}, "
            f"device={self._model_device}, max_image_size={max_image_size}",
            flush=True,
        )
        print(
            f"  boi_id={self._boi_id}, eoi_id={self._eoi_id}, "
            f"img_marker_id={self._img_marker_id}, "
            f"asst_suffix_len={self._asst_suffix_len}",
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
        """Tokenize image + question via Emu3Processor in mode='U'.

        Returns (input_ids, None, None).  Visual tokens are encoded as discrete
        IDs within input_ids via the VQ tokenizer; no separate images_tensor
        is passed to the LM forward.

        If max_image_size is set, the image is thumbnail-resized before encoding
        to keep the VQ token count manageable (default 256 → 1024 visual tokens).
        Without this cap, Emu3's default processor area (~720×720) produces ~8100
        tokens, which OOMs under eager attention on most GPUs.
        """
        if self._max_image_size is not None:
            image = image.copy()
            image.thumbnail(
                (self._max_image_size, self._max_image_size), Image.LANCZOS
            )
        inputs = self._processor(
            text=[question],
            image=[image],
            mode="U",
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self._model_device)
        return input_ids, None, None

    # ── _prepare_embeds ────────────────────────────────────────────────────────

    def _prepare_embeds(
        self,
        input_ids,
        images_tensor,   # None for Emu3
        image_sizes,     # None for Emu3
    ) -> tuple[torch.Tensor, int]:
        """Embed all tokens (text + visual) via embed_tokens.

        Also fires the projector hook by calling _visual_embedder on the
        [IMG+1 : EOI) visual block.  This hook fires from run_prefill (which
        registers hooks first, then calls _prepare_embeds) but NOT from
        run_generate (which calls _prepare_embeds before registering hooks).
        _call_generate fires the hook explicitly to cover the generate path.

        Returns (inputs_embeds, n_image_tokens) where n_image_tokens is the
        total number of positions in the visual block span [IMG+1 : EOI).
        """
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        ids_1d = input_ids[0]
        img_pos = int((ids_1d == self._img_marker_id).nonzero(as_tuple=False)[0, 0])
        eoi_pos = int((ids_1d == self._eoi_id).nonzero(as_tuple=False)[0, 0])

        # Phantom forward to satisfy the projector hook (no_grad, no side effects)
        with torch.no_grad():
            self._visual_embedder(input_ids[:, img_pos + 1 : eoi_pos])

        n_image_tokens = eoi_pos - img_pos - 1
        return inputs_embeds, n_image_tokens

    # ── visual_range ───────────────────────────────────────────────────────────

    def visual_range(
        self, input_ids: torch.Tensor, n_image_tokens: int
    ) -> tuple[int, int]:
        """Return (vlo, vhi) for the visual block in inputs_embeds.

        Override: Emu3 has no -200 sentinel; find range via IMG marker position.
        """
        ids_1d = input_ids[0]
        img_pos = int((ids_1d == self._img_marker_id).nonzero(as_tuple=False)[0, 0])
        return (img_pos + 1, img_pos + 1 + n_image_tokens)

    # ── _slice_visual ──────────────────────────────────────────────────────────

    def _slice_visual(
        self,
        inputs_embeds: torch.Tensor,
        token_index: Optional[TokenIndex],
        n_image_tokens: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract the visual-token slice from inputs_embeds.

        Override: avoids the base-class -200 scan for the run_generate path.
        """
        if token_index is not None:
            vr = token_index.visual_range
            return inputs_embeds[0, vr[0] : vr[1], :].detach().cpu()
        ids_1d = input_ids[0]
        img_pos = int((ids_1d == self._img_marker_id).nonzero(as_tuple=False)[0, 0])
        return inputs_embeds[
            0, img_pos + 1 : img_pos + 1 + n_image_tokens, :
        ].detach().cpu()

    # ── _build_token_index ─────────────────────────────────────────────────────

    def _build_token_index(
        self,
        input_ids,
        n_image_tokens: int,
        prompt_len: int,
        n_generated: int = 0,
    ) -> TokenIndex:
        """Classify every position into a TokenCategory.

        Override: Emu3 token layout uses IMG/EOI markers instead of -200 sentinel.
        For Emu3, prompt_len == input_ids.shape[1] (no sentinel expansion).

        Layout:
          [BOS][sys+role][BOI][H*W prefix][IMG][visual_block][EOI][question][. ASSISTANT:][ANSWER]
           OTHER  OTHER  OTHER    OTHER   OTHER    VISUAL      OTHER QUESTION      OTHER    ANSWER
        """
        ids_1d = input_ids[0]

        img_pos = int((ids_1d == self._img_marker_id).nonzero(as_tuple=False)[0, 0])
        eoi_pos = int((ids_1d == self._eoi_id).nonzero(as_tuple=False)[0, 0])
        asst_pos = len(ids_1d) - self._asst_suffix_len

        categories = torch.zeros(prompt_len + n_generated, dtype=torch.int8)
        categories[img_pos + 1 : eoi_pos] = int(TokenCategory.VISUAL)

        q_start = eoi_pos + 1  # first token of question (EOI itself is OTHER)
        if q_start < asst_pos:
            categories[q_start : asst_pos] = int(TokenCategory.QUESTION)

        if n_generated > 0:
            categories[prompt_len : prompt_len + n_generated] = int(TokenCategory.ANSWER)

        vis_range = (img_pos + 1, eoi_pos)
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
        images_tensor,   # None for Emu3
        image_sizes,     # None for Emu3
        max_new_tokens: int,
        **kwargs,
    ):
        """Generate with input_ids (visual tokens already encoded as discrete IDs).

        Fires the projector hook once before model.generate so that finalize_store
        finds len(store["projector"]) == 1.  (run_generate calls _prepare_embeds
        before registering hooks, so the hook inside _prepare_embeds doesn't fire.)
        """
        from transformers import GenerationConfig

        # Fire projector hook to satisfy finalize_store's assertion
        ids_1d = input_ids[0]
        img_pos = int((ids_1d == self._img_marker_id).nonzero(as_tuple=False)[0, 0])
        eoi_pos = int((ids_1d == self._eoi_id).nonzero(as_tuple=False)[0, 0])
        with torch.no_grad():
            self._visual_embedder(input_ids[:, img_pos + 1 : eoi_pos])

        gen_config = GenerationConfig(
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=gen_config,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )

    # ── ASSISTANT tag suffix ───────────────────────────────────────────────────

    def _compute_asst_suffix_len(self) -> int:
        """Count tokens occupied by the trailing '. ASSISTANT:' tag.

        Uses the same differential-tokenization trick as VilaUHookManager and
        UniTokHookManager: two prompts with different questions, common suffix
        length = number of tokens in the fixed '. ASSISTANT:' tail.
        """
        chat_template = self._processor.chat_template
        bos = self.tokenizer.bos_token or ""

        def _tok(q: str) -> list:
            prompt = bos + chat_template.format(image_prompt="", text_prompt=q)
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
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
