"""
Qwen3VLHookManager — concrete hook manager for Qwen2.5-VL-7B-Instruct.

NOTE: The local clone is in Qwen3-VL/ (repo renamed by authors), but the
      model class is Qwen2_5_VLForConditionalGeneration from transformers.

Architecture
------------
- Model class    : Qwen2_5_VLForConditionalGeneration (transformers)
- Visual tower   : model.visual (Qwen2_5_VLVisionTransformer)
  - Projector    : model.visual.merger (Qwen2_5_VLPatchMerger → LM hidden dim)
- Decoder layers : model.language_model.layers (Qwen2_5_VLTextModel.layers)
- LM head        : model.lm_head
- Chat template  : ChatML (<|im_start|> user / assistant)
- n_image_tokens : DYNAMIC — varies per image (function of image_grid_thw)

Token layout (pre-splice, 1:1 with inputs_embeds — no sentinel expansion)
--------------------------------------------------------------------------
<|im_start|> user \n <|vision_start|> <|image_pad|>×N <|vision_end|>
  [question text ...] <|im_end|> \n <|im_start|> assistant \n

run_prefill / run_generate design
----------------------------------
Both methods are overridden entirely (rather than reusing the base class path)
because Qwen2.5-VL requires image_grid_thw to compute 3D multi-modal RoPE
(mrope). Calling model.model.forward(inputs_embeds=...) without image_grid_thw
would silently fall back to 1D rope, causing prefill/generate logit mismatch
that breaks the smoke-test parity assertion.

In run_prefill: model(input_ids, pixel_values, image_grid_thw) is called with
hooks active — the visual tower fires once, firing the projector hook.
In run_generate: model.generate(input_ids, pixel_values, image_grid_thw, ...)
is called with hooks active — the projector fires once during the prefill step.

calibrate.py compatibility
--------------------------
_prepare_embeds is only called from calibrate.py (not from run_prefill/run_generate).
It manually embeds input_ids + scatters visual tokens into the sequence so that
visual_range(input_ids, n_image_tokens) can slice the visual block for cos-sim.
"""

from __future__ import annotations

import sys
import os
from typing import Optional

import torch
from PIL import Image

from .base import VLMHookManager
from .schema import Capture, TokenCategory, TokenIndex
from .utils import finalize_store, register_captures, remove_handles

_QWEN_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Qwen3-VL")
)
_QWEN_VL_UTILS = os.path.join(_QWEN_ROOT, "qwen-vl-utils", "src")


def _ensure_qwen_utils():
    for _p in (_QWEN_ROOT, _QWEN_VL_UTILS):
        if _p not in sys.path:
            sys.path.insert(0, _p)


class Qwen3VLHookManager(VLMHookManager):
    """Hook manager for Qwen2.5-VL-7B-Instruct.

    Parameters
    ----------
    model     : Qwen2_5_VLForConditionalGeneration in eval mode;
                must be loaded with attn_implementation="eager"
    processor : AutoProcessor from AutoProcessor.from_pretrained(model_id)
                with processor.image_processor.max_pixels capped as needed
    capture_attention_weights : opt-in; off by default
    """

    def __init__(
        self,
        model,
        processor,
        capture_attention_weights: bool = False,
    ):
        _ensure_qwen_utils()

        # Bypass base __init__ (we verify eager impl ourselves below).
        self.model = model
        self.capture_attention_weights = capture_attention_weights
        self.processor = processor
        self.tokenizer = processor.tokenizer

        impl = getattr(model.config, "_attn_implementation", None)
        if impl != "eager":
            raise ValueError(
                f"Qwen3VLHookManager requires attn_implementation='eager' "
                f"(found: {impl!r}). Reload the model with that argument."
            )

        self._model_device = next(model.parameters()).device
        self._model_dtype = next(model.parameters()).dtype

        tok = self.tokenizer
        self._vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
        self._vision_end_id   = tok.convert_tokens_to_ids("<|vision_end|>")
        self._im_start_id     = tok.convert_tokens_to_ids("<|im_start|>")

        # image_token_id from config, with fallback
        cfg_img_id = getattr(model.config, "image_token_id", None)
        if cfg_img_id is None:
            cfg_img_id = tok.convert_tokens_to_ids("<|image_pad|>")
        self._image_token_id = int(cfg_img_id)
        self._last_input_ids = None
        self._last_image_grid_thw = None
        self._last_attention_mask = None

        print(
            f"Qwen3VLHookManager: device={self._model_device}, "
            f"vision_start_id={self._vision_start_id}, "
            f"vision_end_id={self._vision_end_id}, "
            f"image_token_id={self._image_token_id}",
            flush=True,
        )

    # ── Module accessors ──────────────────────────────────────────────────────

    def _get_projector(self):
        return self.model.visual.merger

    def _get_decoder_layers(self):
        return self.model.language_model.layers

    def _get_lm_head(self):
        return self.model.lm_head

    def _get_lm_forward(self):
        def _forward_with_cached_prompt(**kwargs):
            if (
                "inputs_embeds" in kwargs
                and "input_ids" not in kwargs
                and "position_ids" not in kwargs
            ):
                if self._last_input_ids is None or self._last_image_grid_thw is None:
                    raise RuntimeError(
                        "Qwen3VLHookManager tracing forward requires cached prompt "
                        "metadata. Call _prepare_embeds() first."
                    )
                kwargs["input_ids"] = self._last_input_ids
                kwargs["image_grid_thw"] = self._last_image_grid_thw
                kwargs.setdefault("attention_mask", self._last_attention_mask)
            return self.model(**kwargs)

        return _forward_with_cached_prompt

    # ── Visual token count ────────────────────────────────────────────────────

    def _n_image_tokens_from_ids(self, input_ids: torch.Tensor) -> int:
        """Count image-pad tokens in input_ids — the ground-truth token count."""
        return int((input_ids[0] == self._image_token_id).sum().item())

    # ── visual_range (calibrate.py compatibility) ─────────────────────────────

    def visual_range(
        self, input_ids: torch.Tensor, n_image_tokens: int
    ) -> tuple[int, int]:
        """Return (vlo, vhi) positions of visual tokens in inputs_embeds.

        For Qwen2.5-VL there is no -200 sentinel; visual tokens occupy positions
        immediately after <|vision_start|> in the already-expanded input_ids.
        """
        ids_1d = input_ids[0]
        vstart = int(
            (ids_1d == self._vision_start_id)
            .nonzero(as_tuple=False)[0, 0]
            .item()
        )
        return (vstart + 1, vstart + 1 + n_image_tokens)

    # ── _build_prompt ─────────────────────────────────────────────────────────

    def _build_prompt(self, image: Image.Image, question: str) -> tuple:
        """Build processor inputs from a PIL image and question text.

        Returns (input_ids, pixel_values, image_grid_thw) repacked into the
        abstract tuple slots (images_tensor=pixel_values, image_sizes=image_grid_thw).
        """
        from qwen_vl_utils import process_vision_info

        prior = getattr(self, "_followup_prior", None)
        if prior:
            pq, pa = prior
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": pq},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": pa}]},
                {"role": "user", "content": [{"type": "text", "text": question}]},
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(self._model_device)

        return (
            inputs["input_ids"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
        )

    # ── _prepare_embeds (calibrate.py path only) ──────────────────────────────

    def _prepare_embeds(
        self,
        input_ids,
        pixel_values,    # images_tensor slot
        image_grid_thw,  # image_sizes slot
    ) -> tuple[torch.Tensor, int]:
        """Compute inputs_embeds for calibrate.py cosine-similarity measurement.

        Manually merges visual embeddings into text embeddings so that
        visual_range(input_ids, n_image_tokens) can slice the visual block.
        NOT called from run_prefill / run_generate.
        """
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        mask = (input_ids[0] == self._image_token_id)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[0][mask] = image_embeds.to(inputs_embeds.dtype)
        self._last_input_ids = input_ids
        self._last_image_grid_thw = image_grid_thw
        self._last_attention_mask = torch.ones_like(input_ids)
        n_image_tokens = int(mask.sum().item())
        return inputs_embeds, n_image_tokens

    # ── _build_token_index ────────────────────────────────────────────────────

    def _build_token_index(
        self,
        input_ids,
        n_image_tokens: int,
        prompt_len: int,
        n_generated: int = 0,
    ) -> TokenIndex:
        """Classify positions using <|vision_start|>/<|vision_end|> sentinels.

        For Qwen2.5-VL there is no -200 expansion — input_ids positions map 1:1
        to inputs_embeds positions (the processor already expands image-pad tokens).
        """
        ids_1d = input_ids[0]

        vstart = int(
            (ids_1d == self._vision_start_id)
            .nonzero(as_tuple=False)[0, 0]
            .item()
        )
        vend = int(
            (ids_1d == self._vision_end_id)
            .nonzero(as_tuple=False)[0, 0]
            .item()
        )
        # Last occurrence of <|im_start|> is the assistant tag
        asst_tag_start = int(
            (ids_1d == self._im_start_id)
            .nonzero(as_tuple=False)[-1, 0]
            .item()
        )

        categories = torch.zeros(prompt_len + n_generated, dtype=torch.int8)

        # VISUAL: positions [vstart+1, vend)
        categories[vstart + 1 : vend] = int(TokenCategory.VISUAL)

        # QUESTION: positions [vend+1, asst_tag_start)
        q_start = vend + 1
        q_end   = asst_tag_start
        if q_start < q_end:
            categories[q_start:q_end] = int(TokenCategory.QUESTION)

        # ANSWER: generated positions
        if n_generated > 0:
            categories[prompt_len : prompt_len + n_generated] = int(TokenCategory.ANSWER)

        visual_range   = (vstart + 1, vend)
        question_range = (q_start, q_end) if q_start < q_end else None
        answer_start   = prompt_len if n_generated > 0 else None

        return TokenIndex(
            categories=categories,
            visual_range=visual_range,
            question_range=question_range,
            answer_start=answer_start,
        )

    # ── _call_generate (abstract compliance) ─────────────────────────────────

    def _call_generate(
        self,
        input_ids,
        pixel_values,
        image_grid_thw,
        max_new_tokens: int,
        **kwargs,
    ):
        """Implement abstract; run_generate overrides the calling context."""
        return self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
            **kwargs,
        )

    # ── run_prefill (full override) ───────────────────────────────────────────

    @torch.no_grad()
    def run_prefill(
        self,
        image: Image.Image,
        question: str,
        device: Optional[str] = None,
    ) -> Capture:
        """Single forward pass with proper Qwen 3D mrope and activation capture.

        Calls model(input_ids, pixel_values, image_grid_thw) — the visual tower
        fires once inside hooks (projector hook captured), and the decoder sees
        correct 3D rotary embeddings. No inputs_embeds path is used so that
        mrope is computed from input_ids + image_grid_thw as intended.
        """
        input_ids, pixel_values, image_grid_thw = self._build_prompt(image, question)
        n_image_tokens = self._n_image_tokens_from_ids(input_ids)
        prompt_len = input_ids.shape[1]
        token_index = self._build_token_index(input_ids, n_image_tokens, prompt_len)

        handles, store = register_captures(
            self._get_projector(),
            self._get_decoder_layers(),
            self._get_lm_head(),
            capture_attn_weights=self.capture_attention_weights,
        )
        try:
            self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=False,
                output_attentions=self.capture_attention_weights,
                return_dict=True,
            )
        finally:
            remove_handles(handles)

        tensors = finalize_store(store)
        pv = tensors["projected_visual"]
        visual_embeds = pv.reshape(-1, pv.shape[-1])[:n_image_tokens].cpu()

        cap = Capture(
            input_ids=input_ids.squeeze(0).cpu(),
            token_index=token_index,
            projected_visual=pv,
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

    # ── run_generate (full override) ──────────────────────────────────────────

    @torch.no_grad()
    def run_generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 64,
        device: Optional[str] = None,
        **generate_kwargs,
    ) -> Capture:
        """Generate with hooks capturing prefill + all decode steps.

        With modern transformers (logits_to_keep=1), lm_head is called once per
        generation step: once during prefill (1 row) and once per decode step.
        Total logit rows = n_generated. Residual/attn/mlp accumulate all
        positions: prompt_len + (n_generated - 1) = prompt_len + n_gen_captured.
        """
        input_ids, pixel_values, image_grid_thw = self._build_prompt(image, question)
        n_image_tokens = self._n_image_tokens_from_ids(input_ids)
        prompt_len = input_ids.shape[1]

        handles, store = register_captures(
            self._get_projector(),
            self._get_decoder_layers(),
            self._get_lm_head(),
            capture_attn_weights=self.capture_attention_weights,
        )
        try:
            gen_out = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
                **generate_kwargs,
            )
        finally:
            remove_handles(handles)

        tensors = finalize_store(store)

        # With logits_to_keep=1, tensors["logits"] already has n_scores rows.
        n_scores = len(gen_out.scores) if gen_out.scores else 0
        if n_scores > 0:
            tensors["logits"] = tensors["logits"][:n_scores]

        generated_ids = gen_out.sequences[0][prompt_len:]
        if n_scores > 0:
            generated_ids = generated_ids[:n_scores]

        captured_seq_len = tensors["residual"].shape[1]
        n_generated_captured = max(captured_seq_len - prompt_len, 0)
        token_index = self._build_token_index(
            input_ids, n_image_tokens, prompt_len, n_generated_captured
        )

        pv = tensors["projected_visual"]
        visual_embeds = pv.reshape(-1, pv.shape[-1])[:n_image_tokens].cpu()

        cap = Capture(
            input_ids=input_ids.squeeze(0).cpu(),
            token_index=token_index,
            projected_visual=pv,
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
