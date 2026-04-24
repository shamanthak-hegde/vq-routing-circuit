"""
Abstract VLMHookManager interface.

Subclasses implement the model-specific parts:
  - _build_prompt      → (input_ids, images_tensor, image_sizes)
  - _prepare_embeds    → (inputs_embeds, n_image_tokens)
  - _call_generate     → raw generate output

Everything else (hook registration, token classification, Capture assembly)
is handled by the base class so model ports only override three methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from PIL import Image

from .schema import Capture, TokenCategory, TokenIndex
from .utils import finalize_store, register_captures, remove_handles


class VLMHookManager(ABC):
    """Base class for VLM activation-capture managers.

    Parameters
    ----------
    model               : the VLM (must be eval mode)
    capture_attention_weights : if True, also capture raw attention weight
                          tensors (requires ``attn_implementation="eager"``)
    """

    def __init__(self, model, capture_attention_weights: bool = False):
        self.model = model
        self.capture_attention_weights = capture_attention_weights
        if capture_attention_weights:
            impl = getattr(model.config, "_attn_implementation", None)
            if impl != "eager":
                raise ValueError(
                    "capture_attention_weights=True requires the model to be loaded "
                    "with attn_implementation='eager' "
                    f"(found: {impl!r}).  Reload the model with that argument."
                )

    # ── Module accessors (override in subclasses for different architectures) ──

    def _get_projector(self):
        """Return the multimodal projector module."""
        return self.model.model.mm_projector

    def _get_decoder_layers(self):
        """Return the list of decoder layers."""
        return self.model.model.layers

    def _get_lm_head(self):
        """Return the language-model head module."""
        return self.model.lm_head

    def _get_lm_forward(self):
        """Return the callable module for the prefill forward pass."""
        return self.model

    # ── Abstract interface (model-specific) ───────────────────────────────────

    @abstractmethod
    def _build_prompt(
        self, image: Image.Image, question: str
    ) -> tuple:
        """Return (input_ids, images_tensor, image_sizes) ready for the model."""
        ...

    @abstractmethod
    def _prepare_embeds(
        self,
        input_ids,
        images_tensor,
        image_sizes,
    ) -> tuple:
        """Call the model's multimodal prepare method.

        Returns (inputs_embeds, n_image_tokens).
        ``inputs_embeds`` has shape (1, prompt_len_expanded, hidden).
        """
        ...

    def _build_token_index(
        self,
        input_ids,
        n_image_tokens: int,
        prompt_len: int,
        n_generated: int = 0,
    ) -> TokenIndex:
        """Map every embedding position to a TokenCategory.

        Default implementation handles Vicuna-v1 style templates where a
        single IMAGE_TOKEN_INDEX (-200) sentinel is expanded to n_image_tokens
        contiguous positions and the ASSISTANT: tag closes the prompt.
        Override for models with a different token layout.
        """
        ids_1d = input_ids[0]
        cats, vis_range, q_range, ans_start = self._classify_positions(
            ids_1d, n_image_tokens, prompt_len, n_generated
        )
        return TokenIndex(
            categories=cats,
            visual_range=vis_range,
            question_range=q_range,
            answer_start=ans_start,
        )

    # ── Concrete run methods ──────────────────────────────────────────────────

    @torch.no_grad()
    def run_prefill(
        self,
        image: Image.Image,
        question: str,
        device: Optional[str] = None,
    ) -> Capture:
        """Single forward pass; no generation.

        Captures activations at all four hook points.  Logits at the final
        position predict the first answer token (yes/no for the probe set).
        """
        input_ids, images_tensor, image_sizes = self._build_prompt(image, question)

        # Register hooks BEFORE _prepare_embeds so the projector hook fires
        # during the image-embedding step (before the LM forward pass).
        handles, store = register_captures(
            self._get_projector(),
            self._get_decoder_layers(),
            self._get_lm_head(),
            capture_attn_weights=self.capture_attention_weights,
        )
        try:
            inputs_embeds, n_image_tokens = self._prepare_embeds(
                input_ids, images_tensor, image_sizes
            )
            prompt_len = inputs_embeds.shape[1]
            token_index = self._build_token_index(input_ids, n_image_tokens, prompt_len)

            self._get_lm_forward()(
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                return_dict=True,
            )
        finally:
            remove_handles(handles)

        tensors = finalize_store(store)
        visual_embeds = self._slice_visual(inputs_embeds, token_index)

        cap = Capture(
            input_ids=input_ids.squeeze(0).cpu(),
            token_index=token_index,
            projected_visual=tensors["projected_visual"],
            visual_embeds=visual_embeds,
            attn_out=tensors["attn_out"],
            mlp_out=tensors["mlp_out"],
            residual=tensors["residual"],
            logits=tensors["logits"],
            n_image_tokens=n_image_tokens,
            generated_ids=None,
            attn_weights=tensors.get("attn_weights"),
        )
        return cap if device is None else cap.to(device)

    @torch.no_grad()
    def run_generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 64,
        device: Optional[str] = None,
        **generate_kwargs,
    ) -> Capture:
        """Run model.generate() with hooks capturing every prefill + decode step.

        ``Capture.residual`` / ``attn_out`` / ``mlp_out`` have shape
        ``(n_layers, prompt_len + n_generated_captured, hidden)`` — prompt
        tokens first, then generated-token positions that were actually fed
        back through the model during decoding.

        In decoder-only autoregressive generation, the final sampled token is
        chosen from the previous step's logits and is not itself forwarded
        through the model, so typically
        ``n_generated_captured = max(total_generated - 1, 0)``.

        ``Capture.token_index.mask(TokenCategory.ANSWER)`` selects the
        generated positions present in the captured activations.  Separately,
        ``Capture.logits`` stores one row per generation decision step because
        modern ``transformers.generate()`` calls decoder-only models with
        ``logits_to_keep=1``.
        """
        input_ids, images_tensor, image_sizes = self._build_prompt(image, question)

        # Call prepare_inputs_labels_for_multimodal BEFORE registering hooks
        # so we know n_image_tokens and can build the TokenIndex.
        # The model.generate() call below will invoke it again internally
        # (unavoidable given LlavaLlamaForCausalLM.generate's contract), but
        # hooks are not yet registered at this point.
        inputs_embeds_probe, n_image_tokens = self._prepare_embeds(
            input_ids, images_tensor, image_sizes
        )
        prompt_len = inputs_embeds_probe.shape[1]
        visual_embeds = self._slice_visual(inputs_embeds_probe, None,
                                           n_image_tokens, input_ids)

        handles, store = register_captures(
            self._get_projector(),
            self._get_decoder_layers(),
            self._get_lm_head(),
            capture_attn_weights=self.capture_attention_weights,
        )
        try:
            gen_out = self._call_generate(
                input_ids, images_tensor, image_sizes, max_new_tokens,
                **generate_kwargs
            )
        finally:
            remove_handles(handles)

        tensors = finalize_store(store)
        # When the backend returns per-step scores (output_scores=True), use
        # len(scores) as the ground-truth decode-step count — sequences may
        # contain EOS-padding beyond the real generation steps.  Slice the
        # raw lm_head hook accumulation rather than copying gen_out.scores
        # (scores are post-logits_processor; the hook fires before any
        # processor, so its values are raw lm_head output).
        # Hook layout during VILA-U generate:
        #   rows 0 … prompt_len-1  → prefill positions
        #   rows prompt_len …      → forwarded-back decode tokens
        # So rows [prompt_last : prompt_last + n_scores] give one raw row per
        # decode decision (step 1 from prefill_last, steps 2+ from decode).
        # Falls back to the hook accumulation as-is for backends with
        # logits_to_keep=1 (e.g. LLaVA), which already produce one row/step.
        if getattr(gen_out, "scores", None):
            n_scores = len(gen_out.scores)
            tensors["logits"] = tensors["logits"][prompt_len - 1:][:n_scores]
            generated_ids = self._extract_generated_ids(gen_out, prompt_len)[:n_scores]
        else:
            generated_ids = self._extract_generated_ids(gen_out, prompt_len)
        captured_seq_len = tensors["residual"].shape[1]
        n_generated_captured = max(captured_seq_len - prompt_len, 0)

        token_index = self._build_token_index(
            input_ids, n_image_tokens, prompt_len, n_generated_captured
        )

        cap = Capture(
            input_ids=input_ids.squeeze(0).cpu(),
            token_index=token_index,
            projected_visual=tensors["projected_visual"],
            visual_embeds=visual_embeds,
            attn_out=tensors["attn_out"],
            mlp_out=tensors["mlp_out"],
            residual=tensors["residual"],
            logits=tensors["logits"],
            n_image_tokens=n_image_tokens,
            generated_ids=generated_ids.cpu(),
            attn_weights=tensors.get("attn_weights"),
        )
        return cap if device is None else cap.to(device)

    # ── Token classification helpers ─────────────────────────────────────────

    def _locate_assistant_tag(
        self, ids_1d: torch.Tensor, search_from: int
    ) -> Optional[int]:
        """Return the pre-splice index where the ASSISTANT role tag starts.

        Default: length-based detection using self._asst_suffix_len (computed
        by the subclass).  Override for models with non-Vicuna templates.
        """
        suffix_len = getattr(self, "_asst_suffix_len", 0)
        if suffix_len <= 0:
            return None
        return len(ids_1d) - suffix_len

    def _classify_positions(
        self,
        ids_1d: torch.Tensor,
        n_image_tokens: int,
        prompt_len: int,
        n_generated: int,
        image_sentinel: int = -200,
    ) -> tuple:
        """Classify embedding positions into TokenCategory values.

        Handles the standard layout:
          [BOS] [sys+role] [VISUAL×N] [QUESTION] [ASSISTANT-tag (OTHER)] [ANSWER]

        Returns (categories, visual_range, question_range, answer_start).
        """
        sentinel_pos = int(
            (ids_1d == image_sentinel).nonzero(as_tuple=False)[0, 0].item()
        )
        pos_img_start = sentinel_pos
        pos_img_end   = sentinel_pos + n_image_tokens
        shift = n_image_tokens - 1  # sentinel→N expansion offset

        pos_asst_raw = self._locate_assistant_tag(ids_1d, sentinel_pos + 1)

        categories = torch.zeros(prompt_len + n_generated, dtype=torch.int8)
        categories[pos_img_start:pos_img_end] = int(TokenCategory.VISUAL)

        if pos_asst_raw is not None:
            q_start = pos_img_end
            q_end   = pos_asst_raw + shift
            if q_start < q_end:
                categories[q_start:q_end] = int(TokenCategory.QUESTION)
        else:
            categories[pos_img_end:prompt_len] = int(TokenCategory.QUESTION)

        if n_generated > 0:
            categories[prompt_len:prompt_len + n_generated] = int(TokenCategory.ANSWER)

        visual_range = (pos_img_start, pos_img_end)
        q_mask    = (categories[:prompt_len] == int(TokenCategory.QUESTION))
        q_indices = q_mask.nonzero(as_tuple=False)
        question_range = (
            (int(q_indices[0, 0].item()), int(q_indices[-1, 0].item()) + 1)
            if q_indices.numel() > 0 else None
        )
        answer_start = prompt_len if n_generated > 0 else None

        return categories, visual_range, question_range, answer_start

    # ── Helpers (may be overridden) ───────────────────────────────────────────

    def _slice_visual(
        self,
        inputs_embeds,
        token_index: Optional[TokenIndex],
        n_image_tokens: Optional[int] = None,
        input_ids=None,
    ) -> torch.Tensor:
        """Extract the visual-token slice from inputs_embeds."""
        if token_index is not None:
            vr = token_index.visual_range
            return inputs_embeds[0, vr[0]:vr[1], :].detach().cpu()
        # called from run_generate before token_index is built
        assert n_image_tokens is not None and input_ids is not None
        pos_img_start = (
            (input_ids[0] == -200).nonzero(as_tuple=False)[0, 0].item()
        )
        return inputs_embeds[
            0, pos_img_start:pos_img_start + n_image_tokens, :
        ].detach().cpu()

    @abstractmethod
    def _call_generate(
        self,
        input_ids,
        images_tensor,
        image_sizes,
        max_new_tokens: int,
        **kwargs,
    ):
        """Call model.generate() in a model-specific way; return raw output."""
        ...

    @staticmethod
    def _extract_generated_ids(gen_out, prompt_len: int) -> torch.Tensor:
        """Pull newly-generated token ids out of the generate output."""
        if hasattr(gen_out, "sequences"):
            seqs = gen_out.sequences
        else:
            seqs = gen_out
        # seqs shape: (1, prompt_len + n_gen) or (1, n_gen) depending on model
        seq = seqs[0]
        if seq.shape[0] > prompt_len:
            return seq[prompt_len:]
        return seq  # model returned only the new tokens
