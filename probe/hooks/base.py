"""
Abstract VLMHookManager interface.

Subclasses implement the model-specific parts:
  - _build_prompt      → (input_ids, images_tensor, image_sizes)
  - _prepare_embeds    → (inputs_embeds, n_image_tokens)
  - _build_token_index → TokenIndex

Everything else (hook registration, finalization, Capture assembly) is
handled by the base class so Week-3 ports only override three methods.
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

    @abstractmethod
    def _build_token_index(
        self,
        input_ids,
        n_image_tokens: int,
        prompt_len: int,
        n_generated: int = 0,
    ) -> TokenIndex:
        """Map every embedding position to a TokenCategory."""
        ...

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
            self.model, capture_attn_weights=self.capture_attention_weights
        )
        try:
            inputs_embeds, n_image_tokens = self._prepare_embeds(
                input_ids, images_tensor, image_sizes
            )
            prompt_len = inputs_embeds.shape[1]
            token_index = self._build_token_index(input_ids, n_image_tokens, prompt_len)

            self.model(
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
            self.model, capture_attn_weights=self.capture_attention_weights
        )
        try:
            gen_out = self._call_generate(
                input_ids, images_tensor, image_sizes, max_new_tokens,
                **generate_kwargs
            )
        finally:
            remove_handles(handles)

        tensors = finalize_store(store)
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
