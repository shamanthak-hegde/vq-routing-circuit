"""
LuminaMGPTHookManager — hook manager for Alpha-VLLM/Lumina-mGPT-7B (unified VQ-VLM).

Architecture
------------
Lumina-mGPT is built on Chameleon (HF-compatible ChameleonForConditionalGeneration).
Image tokens have IDs in the range [4, 8195] inclusive (8192 VQ codes).
Special delimiters:
  image_start_token = "<racm3:break>"
  image_end_token   = "<eoss>"

The model has no separate projector — visual tokens are embedded directly
through the shared vocabulary table.  We use the _VisualEmbedder thin-wrapper
trick (same as Emu3 and Liquid) to satisfy the projector-hook contract.

Prompt building
---------------
Uses FlexARItemProcessor from lumina_mgpt/data/item_processor.py:
  item = {"image": pil_image, "conversations": [
      {"from": "human", "value": question},
      {"from": "gpt",   "value": ""},
  ]}
  _prompt = item_processor.process_item(item)
  # Flatten the list of (int | {"input_ids": [...]}) → torch.LongTensor

Requirements
------------
  - Lumina-mGPT repo must be on sys.path (lumina_mgpt/, xllmx/, data/)
  - Model loaded via:
      from lumina_mgpt.model.chameleon import ChameleonForConditionalGeneration
      model = ChameleonForConditionalGeneration.from_pretrained(model_path, ...)
  - attn_implementation="eager" to allow hook capture (flash_attn blocks hooks)
  - item_processor: FlexARItemProcessor(target_size=512)

Conda env: sae (after installing lumina_mgpt dependencies)
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

_LUMINA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Lumina-mGPT")
)
_LUMINA_DATA_ROOT = os.path.join(_LUMINA_ROOT, "lumina_mgpt")

# Image token ID range in the unified vocab (hard-coded from inference_solver.py)
_IMAGE_TOKEN_ID_MIN = 4
_IMAGE_TOKEN_ID_MAX = 8195  # inclusive — 8192 codes total


class _LuminaVisualEmbedder(nn.Module):
    """Thin hook-target wrapping embed_tokens for the visual block.

    Identical purpose to _Emu3VisualEmbedder.
    """

    def __init__(self, embed_tokens: nn.Embedding) -> None:
        super().__init__()
        self.__dict__["_embed_ref"] = embed_tokens

    def forward(self, vis_ids: torch.Tensor) -> torch.Tensor:
        return self.__dict__["_embed_ref"](vis_ids)


class LuminaMGPTHookManager(VLMHookManager):
    """Hook manager for Alpha-VLLM/Lumina-mGPT-7B.

    Parameters
    ----------
    model          : ChameleonForConditionalGeneration (from lumina_mgpt.model.chameleon)
                     loaded with attn_implementation="eager"
    item_processor : FlexARItemProcessor instance for prompt building
    capture_attention_weights : opt-in; requires attn_implementation="eager"
    """

    def __init__(
        self,
        model,
        item_processor,
        capture_attention_weights: bool = False,
    ) -> None:
        for p in (_LUMINA_ROOT, _LUMINA_DATA_ROOT):
            if p not in sys.path:
                sys.path.insert(0, p)

        self.model = model
        self.item_processor = item_processor
        self.capture_attention_weights = capture_attention_weights

        self._model_device = next(model.parameters()).device
        self._model_dtype = next(model.parameters()).dtype

        # Thin projector wrapper
        self._visual_embedder = _LuminaVisualEmbedder(model.get_input_embeddings())

        # Find image delimiter token IDs from the item_processor's tokenizer
        tok = item_processor.tokenizer
        self._img_start_id: int = tok.encode(
            item_processor.image_start_token, bos=False, eos=False
        )[0]
        self._img_end_id: int = tok.encode(
            item_processor.image_end_token, bos=False, eos=False
        )[0]

        # ASSISTANT tag: find the trailing tokens that differ between prompts
        self._asst_suffix_len: int = self._compute_asst_suffix_len()

        print(
            f"LuminaMGPTHookManager: n_layers={len(model.model.layers)}, "
            f"hidden={model.config.hidden_size}, "
            f"img_start_id={self._img_start_id}, img_end_id={self._img_end_id}, "
            f"device={self._model_device}",
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
        """Build input_ids via FlexARItemProcessor. Returns (input_ids, None, None)."""
        item = {
            "image": image,
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt",   "value": ""},
            ],
        }
        raw_prompt = self.item_processor.process_item(item)
        flat: list[int] = []
        for v in raw_prompt:
            if isinstance(v, int):
                flat.append(v)
            else:
                flat.extend(v["input_ids"])

        input_ids = torch.tensor(flat, dtype=torch.long).unsqueeze(0).to(self._model_device)
        return input_ids, None, None

    # ── _prepare_embeds ───────────────────────────────────────────────────────

    def _prepare_embeds(self, input_ids, _images_tensor, _image_sizes) -> tuple:
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
        """Find contiguous visual-token block by ID range [4, 8195]."""
        is_vis = (ids_1d >= _IMAGE_TOKEN_ID_MIN) & (ids_1d <= _IMAGE_TOKEN_ID_MAX)
        vis_positions = is_vis.nonzero(as_tuple=False)[:, 0]
        if vis_positions.numel() == 0:
            raise RuntimeError("No visual tokens found in Lumina-mGPT input_ids")
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
        # Fire projector hook (run_generate calls _prepare_embeds before hooks)
        vis_lo, vis_hi = self._find_visual_range(input_ids[0])
        with torch.no_grad():
            self._visual_embedder(input_ids[:, vis_lo:vis_hi])

        return self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_input_ids_for_capture(self, input_ids) -> torch.Tensor:
        return input_ids.squeeze(0).cpu()

    def _slice_visual(self, inputs_embeds, token_index, n_image_tokens=None,
                      input_ids=None) -> torch.Tensor:
        if token_index is not None:
            vr = token_index.visual_range
            return inputs_embeds[0, vr[0]:vr[1], :].detach().cpu()
        vis_lo, vis_hi = self._find_visual_range(input_ids[0])
        return inputs_embeds[0, vis_lo:vis_hi, :].detach().cpu()

    def _locate_assistant_tag(self, ids_1d: torch.Tensor, search_from: int) -> Optional[int]:
        suffix_len = getattr(self, "_asst_suffix_len", 0)
        if suffix_len <= 0:
            return None
        return len(ids_1d) - suffix_len

    def _compute_asst_suffix_len(self) -> int:
        tok = self.item_processor.tokenizer

        def _build(q: str) -> list:
            item = {
                "image": None,
                "conversations": [
                    {"from": "human", "value": q},
                    {"from": "gpt",   "value": ""},
                ],
            }
            try:
                raw = self.item_processor.process_item(item)
            except Exception:
                return []
            flat: list[int] = []
            for v in raw:
                if isinstance(v, int):
                    flat.append(v)
                else:
                    flat.extend(v.get("input_ids", []))
            return flat

        ids1 = _build("X")
        ids2 = _build("W V")
        if not ids1 or not ids2:
            return 0
        n = 0
        while n < min(len(ids1), len(ids2)):
            if ids1[-(n + 1)] == ids2[-(n + 1)]:
                n += 1
            else:
                break
        return n
