"""
ChameleonHFHookManager — hook manager for Meta Chameleon-7B (HF transformers).

This targets the HF-native ChameleonForConditionalGeneration model, not the
Meta research repo in chameleon/.  It requires transformers >= 4.41.

Architecture
------------
HF Chameleon is a unified-tokenizer VQ-VLM:
  - LM:       ChameleonForConditionalGeneration  (model.model.layers)
  - VQVAE:    model.model.vqmodel               (encodes pixel_values → VQ codes)
  - Vocab:    model.model.vocabulary_mapping     (maps VQ codes → BPE token IDs)
  - No separate linear projector — image features go through the shared embedding.

Visual pipeline (HF model internal)
-------------------------------------
  1. pixel_values (float32, 3×512×512) → model.model.vqmodel.encode → VQ codes (int)
  2. codes → model.model.vocabulary_mapping.convert_img2bpe → BPE image token IDs
  3. BPE image IDs → model.model.embed_tokens → visual embeddings
  4. Visual embeddings replace the single `<image>` placeholder in inputs_embeds.

Hook strategy (projector-hook contract)
-----------------------------------------
We wrap model.model.get_image_features() in a thin _ChameleonVisualEmbedder module
and call it explicitly during _prepare_embeds. This fires the register_captures hook
exactly once per run_prefill and _call_generate, satisfying finalize_store.

In _prepare_embeds we manually expand the single <image> position to n_vq_codes
positions, producing inputs_embeds of shape (1, expanded_len, hidden).

Token layout (Chameleon template)
-----------------------------------
  [BOS] [system] [HUMAN:] [<image>] [question] [ASSISTANT:] [ANSWER]
   OTHER  OTHER   OTHER    VISUAL    QUESTION      OTHER      ANSWER

n_image_tokens: determined at runtime (call model.model.get_image_tokens(pv)).
Typical value: 1024 for 512×512 images (32×32 VQ grid).

Requirements
------------
  - transformers >= 4.41 (for ChameleonForConditionalGeneration / ChameleonProcessor)
  - Model checkpoint: meta-llama/chameleon-7b or equivalent
  - Load with attn_implementation="eager" to enable hook capture
    (flash_attn blocks attention weight hooks).
  - Pass model_path to ChameleonProcessor.from_pretrained(model_path).

Conda env: sae  (transformers 4.57 is present; Chameleon is available)
"""

from __future__ import annotations

from typing import Optional

import torch
from PIL import Image
from torch import nn

from .base import VLMHookManager
from .schema import TokenCategory, TokenIndex


class _ChameleonVisualEmbedder(nn.Module):
    """Thin hook-target that wraps model.model.get_image_features(pixel_values).

    Calling forward fires exactly once per _prepare_embeds, satisfying the
    projector-hook contract.  The computation is identical to what the model
    does internally — no side effects, no extra computation.
    """

    def __init__(self, chameleon_model) -> None:
        super().__init__()
        # Store by __dict__ to avoid registering as a submodule
        self.__dict__["_model_ref"] = chameleon_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Returns (1, n_vq_tokens, hidden)
        return self.__dict__["_model_ref"].get_image_features(pixel_values)


class ChameleonHFHookManager(VLMHookManager):
    """Hook manager for ChameleonForConditionalGeneration (HF transformers >= 4.41).

    Parameters
    ----------
    model     : ChameleonForConditionalGeneration loaded with attn_implementation="eager"
    processor : ChameleonProcessor.from_pretrained(model_path)
    capture_attention_weights : opt-in; requires attn_implementation="eager"

    Typical loading
    ---------------
    from transformers.models.chameleon import ChameleonForConditionalGeneration, ChameleonProcessor
    import torch

    model_path = "facebook/chameleon-7b"  # or local path
    processor = ChameleonProcessor.from_pretrained(model_path)
    model = ChameleonForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
    )
    model.eval()
    hm = ChameleonHFHookManager(model, processor)
    """

    def __init__(
        self,
        model,
        processor,
        capture_attention_weights: bool = False,
    ) -> None:
        super().__init__(model, capture_attention_weights)

        self.processor = processor
        self._model_device = next(model.parameters()).device
        self._model_dtype = next(model.parameters()).dtype

        # Single <image> placeholder token ID (from config.vocabulary_map)
        self._image_token_id: int = int(model.model.vocabulary_mapping.image_token_id)

        # Thin projector wrapper
        self._visual_embedder = _ChameleonVisualEmbedder(model.model)

        # ASSISTANT tag suffix length (differential tokenization)
        self._asst_suffix_len: int = self._compute_asst_suffix_len()

        print(
            f"ChameleonHFHookManager: n_layers={len(model.model.layers)}, "
            f"hidden={model.config.hidden_size}, "
            f"image_token_id={self._image_token_id}, "
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
        """Build (input_ids, pixel_values, None) for Chameleon.

        Uses ChameleonProcessor with a simple VQA template:
          "USER: <image>\\n{question}\\nASSISTANT:"
        """
        # Chameleon base model: no chat template. Use a direct completion format
        # so the model naturally predicts "yes" or "no" as the first token.
        prior = getattr(self, "_followup_prior", None)
        if prior:
            pq, pa = prior
            prompt = f"<image>{pq}\nAnswer: {pa}\n{question}\nAnswer:"
        else:
            prompt = f"<image>{question}\nAnswer:"
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self._model_device)
        pixel_values = inputs.pixel_values.to(self._model_device, dtype=self._model_dtype)
        return input_ids, pixel_values, None

    # ── _prepare_embeds ───────────────────────────────────────────────────────

    def _prepare_embeds(
        self,
        input_ids,
        pixel_values,
        _image_sizes,
    ) -> tuple[torch.Tensor, int]:
        """Replace image_token_id positions with VQVAE visual embeddings.

        HF ChameleonProcessor already expands <image> to 1024 copies of
        image_token_id in input_ids — no sequence-length expansion is needed
        here. We just embed all tokens, then scatter the VQVAE embeddings into
        the 1024 image_token_id positions (identical to what the model does
        internally in ChameleonModel.forward).

        Returns (inputs_embeds, n_image_tokens).
        """
        inputs_embeds = self.model.model.embed_tokens(input_ids)  # (1, full_len, hidden)

        # Call visual embedder (fires projector hook) → (1, n_vq, hidden)
        visual_embeds = self._visual_embedder(pixel_values)
        n_vq = visual_embeds.shape[1]

        # Scatter VQVAE embeddings into the image_token_id positions in order.
        # masked_scatter flattens the source, so pass visual_embeds directly.
        img_mask = (input_ids == self._image_token_id)  # (1, full_len)
        n_img_in_ids = int(img_mask.sum().item())
        if n_img_in_ids != n_vq:
            raise ValueError(
                f"Chameleon image token count mismatch: "
                f"{n_img_in_ids} image_token_id positions in input_ids "
                f"vs {n_vq} VQVAE embeddings from pixel_values"
            )
        img_mask_3d = img_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(img_mask_3d, visual_embeds)

        return inputs_embeds, n_vq

    # ── visual_range ──────────────────────────────────────────────────────────

    def visual_range(self, input_ids, n_image_tokens: int) -> tuple[int, int]:
        ids_1d = input_ids[0] if input_ids.ndim == 2 else input_ids
        img_positions = (ids_1d == self._image_token_id).nonzero(as_tuple=False)[:, 0]
        img_pos = int(img_positions[0].item())
        return (img_pos, img_pos + n_image_tokens)

    # ── _build_token_index ────────────────────────────────────────────────────

    def _build_token_index(
        self,
        input_ids,
        n_image_tokens: int,
        prompt_len: int,
        n_generated: int = 0,
    ) -> TokenIndex:
        """Map embedding positions to TokenCategory.

        ChameleonProcessor puts all n_image_tokens (1024) image_token_id values
        directly into input_ids, so input_ids and inputs_embeds share the same
        coordinate system — no shift adjustment is needed.
        """
        ids_1d = input_ids[0]
        img_positions = (ids_1d == self._image_token_id).nonzero(as_tuple=False)[:, 0]
        img_pos = int(img_positions[0].item())
        vis_lo = img_pos
        vis_hi = img_pos + n_image_tokens
        # No shift: ChameleonProcessor puts all 1024 image_token_ids directly
        # into input_ids, so input_ids coordinates == inputs_embeds coordinates.

        asst_raw = self._locate_assistant_tag(ids_1d, img_pos + 1)

        categories = torch.zeros(prompt_len + n_generated, dtype=torch.int8)
        categories[vis_lo:vis_hi] = int(TokenCategory.VISUAL)

        if asst_raw is not None:
            q_start = vis_hi
            q_end = asst_raw  # no shift
            if q_start < q_end:
                categories[q_start:q_end] = int(TokenCategory.QUESTION)
        else:
            categories[vis_hi:prompt_len] = int(TokenCategory.QUESTION)

        if n_generated > 0:
            categories[prompt_len:prompt_len + n_generated] = int(TokenCategory.ANSWER)

        q_mask = categories[:prompt_len] == int(TokenCategory.QUESTION)
        q_indices = q_mask.nonzero(as_tuple=False)
        q_range = (
            (int(q_indices[0, 0].item()), int(q_indices[-1, 0].item()) + 1)
            if q_indices.numel() > 0 else None
        )
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
        pixel_values,
        _image_sizes,
        max_new_tokens: int,
        **kwargs,
    ):
        # Fire projector hook (run_generate calls _prepare_embeds before hooks active)
        with torch.no_grad():
            self._visual_embedder(pixel_values)

        attention_mask = torch.ones_like(input_ids)
        return self.model.generate(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=1.0,    # neutral; do_sample=False makes this a no-op
            top_p=1.0,          # neutral; do_sample=False makes this a no-op
            pad_token_id=self.model.config.eos_token_id,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )

    # ── _slice_visual ─────────────────────────────────────────────────────────

    def _slice_visual(
        self,
        inputs_embeds,
        token_index,
        n_image_tokens=None,
        input_ids=None,
    ) -> torch.Tensor:
        if token_index is not None:
            vr = token_index.visual_range
            return inputs_embeds[0, vr[0]:vr[1], :].detach().cpu()
        img_positions = (input_ids[0] == self._image_token_id).nonzero(as_tuple=False)[:, 0]
        img_pos = int(img_positions[0].item())
        return inputs_embeds[0, img_pos:img_pos + n_image_tokens, :].detach().cpu()

    # ── _get_lm_forward override ──────────────────────────────────────────────

    def _get_lm_forward(self):
        # Chameleon's forward expects pixel_values= but we pass inputs_embeds=.
        # The base run_prefill already passes inputs_embeds=; pixel_values defaults
        # to None in ChameleonForConditionalGeneration.forward() which is correct
        # (visual embedding is already done in _prepare_embeds).
        return self.model

    # ── ASSISTANT tag suffix ───────────────────────────────────────────────────

    def _compute_asst_suffix_len(self) -> int:
        """Differential tokenization to find 'Answer:' suffix length."""
        def _tok(q: str) -> list:
            prompt = f"<image>{q}\nAnswer:"
            ids = self.processor.tokenizer.encode(prompt, add_special_tokens=True)
            return ids

        ids1 = _tok("X")
        ids2 = _tok("W V")
        n = 0
        while n < min(len(ids1), len(ids2)):
            if ids1[-(n + 1)] == ids2[-(n + 1)]:
                n += 1
            else:
                break
        return n

    def _locate_assistant_tag(
        self, ids_1d: torch.Tensor, search_from: int
    ) -> Optional[int]:
        suffix_len = getattr(self, "_asst_suffix_len", 0)
        if suffix_len <= 0:
            return None
        return len(ids_1d) - suffix_len

    # ── VQ codes ──────────────────────────────────────────────────────────────

    def _get_vq_codes(self, image: Image.Image):
        """Return raw VQ code indices from the Chameleon VQVAE.

        model.model.vqmodel.encode returns (quant, emb_loss, min_encoding_indices)
        where min_encoding_indices is a flat LongTensor of argmin code indices.
        Codebook size from quantizer.num_embeddings.
        """
        inputs = self.processor(
            text="<image>X\nAnswer:", images=image, return_tensors="pt"
        )
        pixel_values = inputs.pixel_values.to(self._model_device, dtype=self._model_dtype)
        with torch.no_grad():
            vq_out = self.model.model.vqmodel.encode(pixel_values)
        # vq_out = (quant, emb_loss, min_encoding_indices)
        codes = vq_out[2].flatten().cpu()
        n_vq = int(self.model.model.vqmodel.quantize.num_embeddings)
        return {"levels": [codes], "codebook_sizes": [n_vq]}
