"""
GillHookManager — hook manager for GILL (Generating Images with LLMs, NeurIPS 2023).

Architecture
------------
- LM backbone      : OPTForCausalLM (facebook/opt-6.7b)
                      32 layers, hidden=4096, vocab~50281 (50265 base + cls + 8 IMG)
- Vision encoder   : CLIP-ViT-L/14 (openai/clip-vit-large-patch14)
                      Pooler output (1024-D) projected via nn.Linear to 4×4096
- Projector        : model.visual_embeddings: nn.Linear(1024, 4×4096)
                      → reshape (B, 4, 4096) — 4 continuous visual tokens
- Decoder layers   : model.lm.model.decoder.layers (32 OPTDecoderLayer)
- LM head          : model.lm.lm_head

Visual pipeline
---------------
  PIL image → CLIPVisionModel.forward().pooler_output (1, 1024)
            → _GillProjectorModule (wraps visual_embeddings + reshape)
            → (1, 4, 4096) — 4 projected visual tokens
            → concatenated with text embeddings → OPT forward

No VQ: GILL uses a continuous linear projection (not a codebook).
Calibrated σ expected near 0.5 (same cluster as LLaVA/VILA/HaploOmni/LaVIT).

Token layout (in inputs_embeds)
---------------------------------
  pos: 0..3     4       5..asst-1     asst..end
       VISUAL×4  OTHER   QUESTION      OTHER ("A:")
       (visual)  (BOS)   (question)    (prompt tag)

Prompt format
-------------
  [PIL image, f"Q: {question}\\nA:"]
  Text tokenized with add_special_tokens=True; OPT BOS (</s>) appears at position 4.
  No chat template.

Generation
----------
Custom token-by-token greedy loop mirroring GILLModel.generate (L443).
use_cache=False: full sequence re-forwarded each step → hooks fire fresh each step.
This matches SEED-LLaMA's xformers pattern; base.py handles accumulated captures.
transformers==4.30.2 lacks logits_to_keep — returns per-step scores as a tuple.
IMG output tokens masked with -inf during yes/no QA.

OPT layer quirk
---------------
OPTDecoderLayer has .fc1 and .fc2 (not .mlp).
We monkey-patch layer.mlp = layer.fc2 on all layers so register_captures can
attach the standard mlp-output hook (fc2 fires right before the residual add).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Optional

import torch
from PIL import Image
from torch import nn

from .base import VLMHookManager
from .schema import TokenCategory, TokenIndex

_GILL_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "gill")
)

N_VISUAL_TOKENS = 4  # from checkpoints/gill_opt/model_args.json: n_visual_tokens=4


class _GillProjectorModule(nn.Module):
    """Thin hook-target that wraps visual_embeddings and reshapes the output.

    visual_embeddings is nn.Linear(1024, 4*4096), returning (B, 4*4096).
    This wrapper reshapes to (B, 4, 4096) so projected_visual satisfies the
    ndim=3 and shape[-1]==hidden contract checked by the smoke test.
    """

    def __init__(self, linear: nn.Linear, n_visual: int) -> None:
        super().__init__()
        # Store reference without registering as a submodule (avoids double-counting params)
        self.__dict__["_linear_ref"] = linear
        self._n_visual = n_visual

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        # encoder_output: (B, 1024) — CLIP pooler output
        out = self.__dict__["_linear_ref"](encoder_output)  # (B, 4096*n)
        return out.view(encoder_output.shape[0], self._n_visual, -1)  # (B, n, 4096)


class GillHookManager(VLMHookManager):
    """Hook manager for GILL (OPT-6.7B + CLIP-ViT-L/14 continuous projector).

    Parameters
    ----------
    model_path             : path to the gill_opt checkpoint directory
                             (contains pretrained_ckpt.pth.tar + model_args.json)
    capture_attention_weights : must be False (OPT in transformers 4.30.2 has no
                             eager mode toggle via _attn_implementation)
    """

    def __init__(
        self,
        model_path: str = "gill/checkpoints/gill_opt",
        capture_attention_weights: bool = False,
    ) -> None:
        if _GILL_ROOT not in sys.path:
            sys.path.insert(0, _GILL_ROOT)

        from gill.models import load_gill  # noqa: PLC0415

        if capture_attention_weights:
            raise ValueError(
                "capture_attention_weights=True is not supported for the GILL backend."
            )

        print(f"Loading GILL from {model_path} ...", flush=True)
        self._gill_outer = load_gill(model_path)   # outer GILL wrapper
        inner = self._gill_outer.model              # GILLModel (has .lm, .visual_model, etc.)

        # Bypass base __init__: OPT in transformers 4.30.2 has no _attn_implementation.
        self.model = inner
        self.capture_attention_weights = False
        self.tokenizer = inner.tokenizer

        self._model_device = next(inner.lm.parameters()).device
        self._model_dtype = next(inner.lm.parameters()).dtype
        self._n_visual_tokens = N_VISUAL_TOKENS

        # OPTDecoderLayer exposes .fc1 and .fc2, not .mlp.
        # Monkey-patch layer.mlp = layer.fc2 so register_captures can attach
        # the mlp-output hook to the feedforward output (post-fc2, pre-residual).
        for layer in inner.lm.model.decoder.layers:
            layer.mlp = layer.fc2

        # Projector wrapper: wraps visual_embeddings + reshape for the hook target.
        self._projector_module = _GillProjectorModule(
            inner.visual_embeddings, N_VISUAL_TOKENS
        )

        # IMG token IDs to mask during yes/no generation (prevents image-output path).
        self._img_token_ids = inner.retrieval_token_idx  # list[int], 8 IDs

        # Trailing "A:" suffix token count for QUESTION/OTHER boundary detection.
        self._asst_suffix_len: int = self._compute_asst_suffix_len()

        n_layers = len(inner.lm.model.decoder.layers)
        hidden = inner.lm.config.hidden_size
        vocab = inner.lm.config.vocab_size
        print(
            f"GillHookManager: n_layers={n_layers}, hidden={hidden}, vocab={vocab}, "
            f"n_visual_tokens={N_VISUAL_TOKENS}, device={self._model_device}, "
            f"dtype={self._model_dtype}, asst_suffix_len={self._asst_suffix_len}",
            flush=True,
        )

    # ── Module accessors ──────────────────────────────────────────────────────

    def _get_projector(self):
        return self._projector_module  # _GillProjectorModule (hook target)

    def _get_decoder_layers(self):
        return self.model.lm.model.decoder.layers

    def _get_lm_head(self):
        return self.model.lm.lm_head

    def _get_lm_forward(self):
        return self.model.lm  # OPTForCausalLM (accepts inputs_embeds)

    # ── _build_prompt ─────────────────────────────────────────────────────────

    def _build_prompt(self, image: Image.Image, question: str) -> tuple:
        """Build inputs for GILL yes/no VQA.

        Prompt: [PIL image, f"Q: {question}\\nA:"] — no chat template.
        Returns (input_ids, pixel_values, None).
        """
        from gill import utils as gill_utils  # noqa: PLC0415

        text = f"Q: {question}\nA:"
        input_ids = self.tokenizer(
            text, add_special_tokens=True, return_tensors="pt"
        ).input_ids.to(self._model_device)

        pixel_values = gill_utils.get_pixel_values_for_model(
            self.model.feature_extractor, image
        )  # (3, H, W) — no batch dim
        pixel_values = pixel_values.unsqueeze(0).to(
            device=self._model_device, dtype=self._model_dtype
        )  # (1, 3, H, W)
        return input_ids, pixel_values, None

    # ── _prepare_embeds ────────────────────────────────────────────────────────

    def _prepare_embeds(
        self,
        input_ids,
        pixel_values,
        image_sizes,  # None for GILL
    ) -> tuple[torch.Tensor, int]:
        """Build inputs_embeds: [VISUAL×4, text tokens].

        Runs CLIP manually, then calls _projector_module (fires the projector hook)
        to get the reshaped visual embeddings.
        """
        clip_out = self.model.visual_model(pixel_values)
        encoder_output = clip_out.pooler_output  # (1, 1024)

        # _projector_module fires the projector hook; returns (1, 4, 4096)
        visual = self._projector_module(encoder_output)

        text_embeds = self.model.input_embeddings(input_ids)  # (1, T, 4096)
        inputs_embeds = torch.cat([visual, text_embeds], dim=1)  # (1, 4+T, 4096)
        return inputs_embeds, N_VISUAL_TOKENS

    # ── visual_range ───────────────────────────────────────────────────────────

    def visual_range(
        self, input_ids: torch.Tensor, n_image_tokens: int
    ) -> tuple[int, int]:
        """Visual tokens are always at positions [0, n_image_tokens)."""
        return (0, n_image_tokens)

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
        # called from run_generate before token_index is built — visual always at front
        return inputs_embeds[0, 0:N_VISUAL_TOKENS, :].detach().cpu()

    # ── _build_token_index ─────────────────────────────────────────────────────

    def _build_token_index(
        self,
        input_ids,
        n_image_tokens: int,
        prompt_len: int,
        n_generated: int = 0,
    ) -> TokenIndex:
        """Classify positions for GILL's prompt layout.

        inputs_embeds layout (prompt_len = 4 + T where T = len(text_ids)):
          [0:4)           VISUAL   — projected CLIP features
          [4]             OTHER    — OPT BOS token (</s>)
          [5 : asst_pos)  QUESTION — "Q: question\\n" tokens
          [asst_pos : prompt_len)  OTHER — trailing "A:" tokens
          [prompt_len : ...)       ANSWER — generated tokens
        """
        N = n_image_tokens  # 4
        asst_pos = prompt_len - self._asst_suffix_len

        categories = torch.zeros(prompt_len + n_generated, dtype=torch.int8)
        categories[0:N] = int(TokenCategory.VISUAL)

        # Position N (BOS) stays OTHER (0); question starts at N+1
        q_start = N + 1
        if q_start < asst_pos:
            categories[q_start:asst_pos] = int(TokenCategory.QUESTION)

        if n_generated > 0:
            categories[prompt_len : prompt_len + n_generated] = int(TokenCategory.ANSWER)

        vis_range = (0, N)
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
        pixel_values,
        image_sizes,  # None for GILL
        max_new_tokens: int,
        **kwargs,
    ):
        """Greedy decode loop (use_cache=False) mirroring GILLModel.generate.

        Full sequence re-forwarded each step so hooks fire fresh per step.
        IMG output tokens masked to prevent image-generation paths.
        Returns SimpleNamespace(sequences, scores) compatible with base.py logit
        slicing (output_scores=True style).
        """
        # _prepare_embeds fires the projector hook (hooks are live at this point)
        inputs_embeds, _ = self._prepare_embeds(input_ids, pixel_values, None)
        accum = inputs_embeds

        generated_ids: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []
        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            out = self.model.lm(
                inputs_embeds=accum,
                use_cache=False,
                return_dict=True,
            )
            next_logits = out.logits[:, -1, :].float()  # (1, vocab)
            next_logits[:, self._img_token_ids] = float("-inf")  # mask IMG tokens
            scores.append(next_logits)

            next_id = next_logits.argmax(dim=-1)  # (1,)
            generated_ids.append(next_id)

            if next_id.item() == eos_id:
                break

            next_emb = self.model.input_embeddings(next_id.unsqueeze(0))  # (1,1,H)
            accum = torch.cat([accum, next_emb], dim=1)

        # Build sequences: [vis_placeholder×4, text_ids, gen_ids]
        # _extract_generated_ids slices from prompt_len = 4 + len(text_ids)
        vis_placeholder = torch.full(
            (1, N_VISUAL_TOKENS), -1, dtype=torch.long, device=self._model_device
        )
        gen_tensor = torch.stack(generated_ids, dim=1)  # (1, n_gen)
        sequences = torch.cat([vis_placeholder, input_ids, gen_tensor], dim=1)

        return SimpleNamespace(sequences=sequences, scores=tuple(scores))

    # ── ASSISTANT suffix detection ─────────────────────────────────────────────

    def _compute_asst_suffix_len(self) -> int:
        """Count trailing tokens shared by '\\nA:' via common-suffix detection."""

        def _tok(q: str) -> list:
            s = f"Q: {q}\nA:"
            return self.tokenizer(s, add_special_tokens=True)["input_ids"]

        ids1 = _tok("Is this a cat?")
        ids2 = _tok("Does this picture show a dog and a cat running together?")
        n = 0
        while n < min(len(ids1), len(ids2)):
            if ids1[-(n + 1)] == ids2[-(n + 1)]:
                n += 1
            else:
                break
        return n
