"""
ShowoHookManager — concrete hook manager for Show-o (MAGVIT-v2 path).

Architecture
------------
- Top-level wrapper  : Showo (ModelMixin/ConfigMixin from Show-o/models/modeling_showo.py)
- LLM backbone       : PhiForCausalLM (microsoft/phi-1_5, vocab resized to 58498)
- VQ tokenizer       : MAGVITv2 (showlab/magvitv2, LFQ codebook size 8192)
- Decoder layers     : model.showo.model.layers (Phi-1.5: 24 layers, hidden=2048)
- LM head            : model.showo.lm_head

Visual pipeline
---------------
Show-o (MAGVIT path, w_clip_vit=False) has NO projector module:
  1. PIL image → image_transform (bicubic resize+crop, normalize to [-1, 1])
  2. MAGVITv2.get_code → (1, N_vis) discrete codebook indices (LFQ, codebook_size=8192)
  3. Codes shifted into LLM vocab: code_ids + len(tokenizer) (text vocab offset)
  4. Visual IDs concatenated into input_ids between <|soi|> and <|eoi|> markers
  5. embed_tokens(input_ids) embeds ALL tokens (text + visual) uniformly — no projector.

The "projector hook" requirement is satisfied by _ShowoVisualEmbedder: a thin
nn.Module that re-calls embed_tokens on the [soi+1 : eoi) visual block when
invoked, firing once per run_prefill and once per run_generate.

Token layout (MAGVIT path)
--------------------------
  pos:   0      1       2..N+1    N+2     N+3      N+4..asst-1  asst..end
         <|mmu|> <|soi|>  vis×N   <|eoi|> <|sot|>    QUESTION    [ASSISTANT:]
         OTHER   OTHER    VISUAL   OTHER   OTHER       QUESTION    OTHER

where N = num_vq_tokens (256 for 256-res, 1024 for 512-res).

Special token IDs (via UniversalPrompting)
------------------------------------------
  <|mmu|>, <|soi|>, <|eoi|>  — add_tokens() IDs added after base Phi vocab + [PAD]
  <|sot|>                    — tokenizer.bos_token_id (explicitly mapped)
  <|eot|>                    — tokenizer.eos_token_id (explicitly mapped)

Attention mask
--------------
Both run_prefill and run_generate use attention_mask=None, which triggers
PhiSdpaAttention's built-in is_causal=True path (lower-triangular causal masking).
This differs from Show-o's original mmu_generate which uses a custom 4D MMU mask
(bidirectional within image block), but gives consistent prefill/generate behaviour
for hook capture purposes and avoids shape-compatibility issues with the vendored phi.py.
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

_SHOWO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Show-o")
)


class _ShowoVisualEmbedder(nn.Module):
    """Thin hook-target wrapping embed_tokens for the visual block.

    Same design as _Emu3VisualEmbedder: gives register_captures a hookable
    nn.Module whose forward() fires with (1, N_vis, hidden), satisfying
    finalize_store's projector-hook assertion without modifying the model.
    """

    def __init__(self, embed_tokens: nn.Embedding) -> None:
        super().__init__()
        self.__dict__["_embed_ref"] = embed_tokens

    def forward(self, vis_block_ids: torch.Tensor) -> torch.Tensor:
        # vis_block_ids: (1, N_vis) int64
        # returns:       (1, N_vis, hidden)
        return self.__dict__["_embed_ref"](vis_block_ids)


class ShowoHookManager(VLMHookManager):
    """Hook manager for Show-o (MAGVIT-v2 path, w_clip_vit=False).

    Parameters
    ----------
    model         : Showo instance from models.Showo.from_pretrained()
    tokenizer     : AutoTokenizer for microsoft/phi-1_5 with Show-o special
                    tokens already registered via UniversalPrompting
    vq_model      : MAGVITv2 from showlab/magvitv2, in eval mode on GPU
    uni_prompting : UniversalPrompting instance (provides sptids_dict)
    resolution    : image resolution used for VQ encoding; 256 → 256 tokens
    capture_attention_weights : opt-in eager-attention weight capture
    """

    def __init__(
        self,
        model,
        tokenizer,
        vq_model,
        uni_prompting,
        resolution: int = 256,
        capture_attention_weights: bool = False,
    ) -> None:
        if _SHOWO_ROOT not in sys.path:
            sys.path.insert(0, _SHOWO_ROOT)

        # Bypass base __init__ — no attn_implementation guard needed for Phi SDPA.
        self.model = model
        self.capture_attention_weights = capture_attention_weights
        self.tokenizer = tokenizer
        self._vq_model = vq_model
        self._resolution = resolution

        self._model_device = next(model.parameters()).device
        self._model_dtype = next(model.parameters()).dtype

        # Special token IDs from UniversalPrompting.sptids_dict
        self._mmu_id = int(uni_prompting.sptids_dict["<|mmu|>"])
        self._soi_id = int(uni_prompting.sptids_dict["<|soi|>"])
        self._eoi_id = int(uni_prompting.sptids_dict["<|eoi|>"])
        self._sot_id = int(uni_prompting.sptids_dict["<|sot|>"])
        self._eot_id = int(uni_prompting.sptids_dict["<|eot|>"])

        # Shift VQ codebook indices into LLM vocab range
        self._text_vocab_offset = len(tokenizer)

        # Thin projector-hook wrapper (not a model submodule, avoids double-hook)
        self._visual_embedder = _ShowoVisualEmbedder(
            model.showo.model.embed_tokens
        )

        # ASSISTANT tag suffix length for prompt_last / QUESTION boundary detection
        self._asst_suffix_len: int = self._compute_asst_suffix_len()

        print(
            f"ShowoHookManager: n_layers={len(model.showo.model.layers)}, "
            f"hidden={model.showo.config.hidden_size}, "
            f"vocab={model.showo.config.vocab_size}, "
            f"device={self._model_device}, resolution={resolution}",
            flush=True,
        )
        print(
            f"  mmu_id={self._mmu_id}, soi_id={self._soi_id}, "
            f"eoi_id={self._eoi_id}, sot_id={self._sot_id}, "
            f"text_vocab_offset={self._text_vocab_offset}, "
            f"asst_suffix_len={self._asst_suffix_len}",
            flush=True,
        )

    # ── Module accessors ──────────────────────────────────────────────────────

    def _get_projector(self):
        return self._visual_embedder

    def _get_decoder_layers(self):
        return self.model.showo.model.layers

    def _get_lm_head(self):
        return self.model.showo.lm_head

    def _get_lm_forward(self):
        """Return PhiForCausalLM callable.

        attention_mask=None lets PhiSdpaAttention use is_causal=True automatically
        (standard causal masking). Consistent with _call_generate's behaviour.
        """
        return self.model.showo

    # ── _build_prompt ─────────────────────────────────────────────────────────

    def _build_prompt(self, image: Image.Image, question: str) -> tuple:
        """Build input_ids for Show-o MMU (MAGVIT path).

        Layout: [<|mmu|>, <|soi|>, vis_codes×N, <|eoi|>, <|sot|>, q_ids]

        Returns (input_ids, None, None) — no images_tensor or image_sizes.
        """
        from training.utils import image_transform

        # PIL → normalised tensor → VQ codes (shifted into LLM vocab)
        img_t = (
            image_transform(image, resolution=self._resolution)
            .unsqueeze(0)
            .to(self._model_device, dtype=self._model_dtype)
        )
        with torch.no_grad():
            image_codes = (
                self._vq_model.get_code(img_t) + self._text_vocab_offset
            )  # (1, N_vis)

        # Question tokenization — mirrors inference_mmu.py MAGVIT path
        q_ids = self.tokenizer(
            ["USER: \n" + question + " ASSISTANT:"],
            return_tensors="pt",
        ).input_ids.to(self._model_device)

        mmu = torch.full((1, 1), self._mmu_id, dtype=torch.long, device=self._model_device)
        soi = torch.full((1, 1), self._soi_id, dtype=torch.long, device=self._model_device)
        eoi = torch.full((1, 1), self._eoi_id, dtype=torch.long, device=self._model_device)
        sot = torch.full((1, 1), self._sot_id, dtype=torch.long, device=self._model_device)

        input_ids = torch.cat([mmu, soi, image_codes, eoi, sot, q_ids], dim=1)
        return input_ids, None, None

    # ── _prepare_embeds ────────────────────────────────────────────────────────

    def _prepare_embeds(
        self,
        input_ids,
        images_tensor,   # None for Show-o MAGVIT path
        image_sizes,     # None for Show-o MAGVIT path
    ) -> tuple[torch.Tensor, int]:
        """Embed all tokens via embed_tokens; fire projector hook on visual block."""
        inputs_embeds = self.model.showo.model.embed_tokens(input_ids)

        ids_1d = input_ids[0]
        soi_pos = int((ids_1d == self._soi_id).nonzero(as_tuple=False)[0, 0])
        eoi_pos = int((ids_1d == self._eoi_id).nonzero(as_tuple=False)[0, 0])

        # Phantom-fire to satisfy finalize_store's ≥1 projector assertion
        with torch.no_grad():
            self._visual_embedder(input_ids[:, soi_pos + 1 : eoi_pos])

        n_image_tokens = eoi_pos - soi_pos - 1
        return inputs_embeds, n_image_tokens

    # ── visual_range ───────────────────────────────────────────────────────────

    def visual_range(
        self, input_ids: torch.Tensor, n_image_tokens: int
    ) -> tuple[int, int]:
        """Return (vlo, vhi) for the visual block via <|soi|> position."""
        ids_1d = input_ids[0]
        soi_pos = int((ids_1d == self._soi_id).nonzero(as_tuple=False)[0, 0])
        return (soi_pos + 1, soi_pos + 1 + n_image_tokens)

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
        soi_pos = int((ids_1d == self._soi_id).nonzero(as_tuple=False)[0, 0])
        return inputs_embeds[
            0, soi_pos + 1 : soi_pos + 1 + n_image_tokens, :
        ].detach().cpu()

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
          [<|mmu|>][<|soi|>][vis×N][<|eoi|>][<|sot|>][USER:...Q... ASSISTANT:][ANSWER]
            OTHER    OTHER   VISUAL   OTHER    OTHER        QUESTION        OTHER  ANSWER
        """
        ids_1d = input_ids[0]

        soi_pos = int((ids_1d == self._soi_id).nonzero(as_tuple=False)[0, 0])
        eoi_pos = int((ids_1d == self._eoi_id).nonzero(as_tuple=False)[0, 0])
        # sot is always immediately after eoi by construction
        sot_pos = eoi_pos + 1
        asst_pos = len(ids_1d) - self._asst_suffix_len

        categories = torch.zeros(prompt_len + n_generated, dtype=torch.int8)
        categories[soi_pos + 1 : eoi_pos] = int(TokenCategory.VISUAL)

        q_start = sot_pos + 1
        if q_start < asst_pos:
            categories[q_start : asst_pos] = int(TokenCategory.QUESTION)

        if n_generated > 0:
            categories[prompt_len : prompt_len + n_generated] = int(TokenCategory.ANSWER)

        vis_range = (soi_pos + 1, eoi_pos)
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
        images_tensor,   # None for Show-o MAGVIT path
        image_sizes,     # None for Show-o MAGVIT path
        max_new_tokens: int,
        **kwargs,
    ):
        """Manual KV-cache generate loop calling PhiForCausalLM directly.

        Avoids transformers.generate() which auto-creates a long attention_mask
        that SDPA rejects. With attention_mask=None:
          - prefill (q_len > 1): PhiSdpaAttention uses is_causal=True
          - decode (q_len = 1):  is_causal=False (always, q_len==1)

        Fires the projector hook once before the loop (run_generate calls
        _prepare_embeds before hooks are registered, so the hook inside
        _prepare_embeds doesn't fire).

        Returns a synthetic object with .sequences and .scores compatible with
        the base class's gen_out processing.
        """
        # Fire projector hook to satisfy finalize_store's ≥1 assertion
        ids_1d = input_ids[0]
        soi_pos = int((ids_1d == self._soi_id).nonzero(as_tuple=False)[0, 0])
        eoi_pos = int((ids_1d == self._eoi_id).nonzero(as_tuple=False)[0, 0])
        with torch.no_grad():
            self._visual_embedder(input_ids[:, soi_pos + 1 : eoi_pos])

        # Prefill: run the full prompt, build KV cache, capture last-position logit
        prefill_out = self.model.showo(
            input_ids=input_ids,
            attention_mask=None,
            use_cache=True,
            return_dict=True,
        )
        past_kv = prefill_out.past_key_values
        # prefill_out.logits: (1, L, vocab); take last position
        last_logit = prefill_out.logits[:, -1, :]  # (1, vocab)

        scores = [last_logit]
        generated = [last_logit.argmax(dim=-1, keepdim=True)]  # (1, 1) greedy

        for _ in range(max_new_tokens - 1):
            if generated[-1].item() == self._eot_id:
                break
            decode_out = self.model.showo(
                input_ids=generated[-1],   # (1, 1)
                attention_mask=None,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = decode_out.past_key_values
            step_logit = decode_out.logits[:, -1, :]  # (1, vocab)
            scores.append(step_logit)
            generated.append(step_logit.argmax(dim=-1, keepdim=True))

        sequences = torch.cat([input_ids] + generated, dim=1)  # (1, L + n_gen)

        class _GenOut:
            pass

        out = _GenOut()
        out.sequences = sequences
        out.scores = scores   # list of (1, vocab) — base class checks len for slicing
        return out

    # ── ASSISTANT tag suffix ───────────────────────────────────────────────────

    def _compute_asst_suffix_len(self) -> int:
        """Count tokens in the trailing ' ASSISTANT:' suffix via common-suffix trick."""

        def _tok(q: str) -> list:
            return self.tokenizer(
                ["USER: \n" + q + " ASSISTANT:"],
                return_tensors=None,
            )["input_ids"][0]

        ids1 = _tok("Is this a cat?")
        ids2 = _tok("Does this picture show a dog and a cat running together?")
        n = 0
        while n < min(len(ids1), len(ids2)):
            if ids1[-(n + 1)] == ids2[-(n + 1)]:
                n += 1
            else:
                break
        return n
