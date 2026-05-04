"""
HaploOmniHookManager — concrete hook manager for HaploOmni.

Architecture
------------
- Model class    : HaploOmniForConditionalGeneration (haploomni package)
- Projector      : model.model.pre_connector (HaploOmniConnector)
- Decoder layers : model.model.layers (pre-stage + main-stage + post-stage combined)
- LM head        : model.lm_head
- n_image_tokens : count(input_ids == model.config.image_token_index)
- Logits         : output_scores=True (logits_to_keep not guaranteed in HaploOmni env)

Token layout (pre-expanded by processor — no -200 sentinel expansion)
----------------------------------------------------------------------
  [sys tokens] [user tokens] [image_token×N] [question tokens] [asst marker]
      OTHER        OTHER           VISUAL          QUESTION         OTHER

run_prefill / run_generate design
----------------------------------
Both are fully overridden because HaploOmni requires pixel_values + image_sizes
passed to model() — calling model.model.forward(inputs_embeds=...) would bypass
the visual encoder injection step in HaploOmniModel.forward().

_prepare_embeds (calibrate.py only)
------------------------------------
Uses a registered hook on pre_connector to capture the projected visual block
during a real model forward pass, then builds a synthetic inputs_embeds with the
visual slice filled in. Slower than the Qwen approach but guaranteed correct.

Conda env: haplovlm
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Optional

import torch
from PIL import Image

from .base import VLMHookManager
from .schema import Capture, TokenCategory, TokenIndex
from .utils import finalize_store, register_captures, remove_handles

_HAPLO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "HaploVLM")
)


def _ensure_haplo_path():
    haplo_model = os.path.join(_HAPLO_ROOT, "haploomni", "model")
    for _p in (_HAPLO_ROOT, haplo_model):
        if _p not in sys.path:
            sys.path.insert(0, _p)


class HaploOmniHookManager(VLMHookManager):
    """Hook manager for HaploOmni.

    Parameters
    ----------
    model     : HaploOmniForConditionalGeneration in eval mode;
                must be loaded with attn_implementation="eager"
    processor : HaploOmniProcessor from HaploOmniProcessor.from_pretrained(...)
    capture_attention_weights : opt-in; off by default
    """

    def __init__(
        self,
        model,
        processor,
        capture_attention_weights: bool = False,
    ):
        _ensure_haplo_path()

        self.model = model
        self.capture_attention_weights = capture_attention_weights
        self.processor = processor
        self.tokenizer = processor.tokenizer

        impl = getattr(model.config, "_attn_implementation", None)
        # HaploOmni ignores attn_implementation="eager" in from_pretrained.
        # We only need eager for raw attention-weight capture; residual/attn_out
        # hooks fire regardless.  Warn instead of raising for non-eager.
        if capture_attention_weights and impl != "eager":
            raise ValueError(
                f"HaploOmniHookManager with capture_attention_weights=True "
                f"requires attn_implementation='eager' (found: {impl!r})."
            )
        if impl != "eager":
            print(
                f"[HaploOmniHookManager] attn_implementation={impl!r} "
                f"(not eager; capture_attention_weights must stay False)",
                flush=True,
            )

        self._model_device = next(model.parameters()).device
        self._model_dtype = next(model.parameters()).dtype

        self._image_token_id = int(model.config.image_token_index)

        # HaploOmni model.model.layers is a FLAT list combining three stages:
        #   [0 .. pre_n-1]         : vision encoder  (hidden = pre_config.hidden_size)
        #   [pre_n .. pre_n+n-1]   : LLM decoder     (hidden = config.hidden_size)
        #   [pre_n+n .. end]       : post-decoder     (different hidden size)
        # The stages have DIFFERENT hidden sizes; we hook only the LLM stage so
        # finalize_store can safely stack all captured tensors to (n_layers, S, H).
        pre_n = model.config.pre_config.num_hidden_layers
        llm_n = model.config.num_hidden_layers
        self._llm_layer_start = pre_n           # inclusive
        self._llm_layer_end   = pre_n + llm_n   # exclusive

        # Locate assistant-start marker in the chat template.
        # HaploOmni uses ChatML: <|im_start|>assistant\n
        tok = self.tokenizer
        self._im_start_id = tok.convert_tokens_to_ids("<|im_start|>")

        print(
            f"HaploOmniHookManager: device={self._model_device}, "
            f"llm_layers=[{self._llm_layer_start},{self._llm_layer_end}) "
            f"({llm_n} layers, hidden={model.config.hidden_size}), "
            f"image_token_id={self._image_token_id}, "
            f"im_start_id={self._im_start_id}",
            flush=True,
        )

    # ── Module accessors ──────────────────────────────────────────────────────

    def _get_projector(self):
        return self.model.model.pre_connector

    def _get_decoder_layers(self):
        # Return only the LLM stage; vision-encoder and post-decoder stages have
        # different hidden sizes and must not be mixed in the hook store.
        return self.model.model.layers[self._llm_layer_start : self._llm_layer_end]

    def _get_lm_head(self):
        return self.model.lm_head

    # ── _build_prompt ─────────────────────────────────────────────────────────

    def _build_prompt(self, image: Image.Image, question: str) -> tuple:
        """Build processor inputs from a PIL image and question text.

        Saves the PIL image to a temp file (HaploOmni processor expects a path).
        Returns (input_ids, pixel_values, image_sizes) repacked into the
        abstract tuple slots (images_tensor=pixel_values, image_sizes=image_sizes).
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            image.save(tmp_path)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": tmp_path},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self._model_device, dtype=self._model_dtype)
        finally:
            os.unlink(tmp_path)

        self._last_inputs = inputs
        return (
            inputs["input_ids"],
            inputs.get("pixel_values"),
            inputs.get("image_sizes"),
        )

    # ── _prepare_embeds (calibrate.py path only) ──────────────────────────────

    def _prepare_embeds(
        self,
        input_ids,
        pixel_values,
        image_sizes,
    ) -> tuple:
        """Capture full hidden states at the LLM-stage boundary (after pre_connector).

        Hooks the forward pre-hook of layers[llm_layer_start] to intercept the
        hidden states entering the first LLM layer. This gives the full sequence
        at LLM hidden dim (3584), with text tokens properly processed through the
        pre-stage and visual tokens projected by pre_connector.

        calibrate.py slices embeds[0, vlo:vhi, :] for cosine-sim computation.
        corrupt.py (noisy_embeds / foil_embeds) adds noise to that slice and
        returns the full tensor for _forward_logit / run_patched_forward.
        """
        boundary_store: list[torch.Tensor] = []

        def _boundary_hook(_module, args):
            # Keep on GPU in model dtype (bfloat16) so corrupted_embeds.device/dtype
            # match the model for patch.py / sweep.py index_copy_ compatibility.
            # calibrate.py does its own .float() cast before cosine-sim.
            boundary_store.append(args[0].detach())

        handle = self.model.model.layers[self._llm_layer_start].register_forward_pre_hook(
            _boundary_hook
        )
        try:
            with torch.no_grad():
                inputs = {k: v for k, v in self._last_inputs.items()}
                inputs["use_cache"] = False
                inputs["return_dict"] = True
                self.model(**inputs)
        finally:
            handle.remove()

        n_image_tokens = int((input_ids[0] == self._image_token_id).sum().item())
        if not boundary_store:
            raise RuntimeError("_prepare_embeds: boundary hook did not fire.")
        inputs_embeds = boundary_store[0]  # (1, seq_len, 3584) on model device, model dtype
        return inputs_embeds, n_image_tokens

    # ── visual_range (calibrate.py compatibility) ─────────────────────────────

    def visual_range(
        self, input_ids: torch.Tensor, n_image_tokens: int
    ) -> tuple[int, int]:
        """Return (vlo, vhi) — visual-token positions in inputs_embeds.

        For HaploOmni there is no -200 sentinel; the processor pre-expands
        image placeholder tokens to image_token_index (151646) × N.
        """
        ids_1d = input_ids[0]
        positions = (ids_1d == self._image_token_id).nonzero(as_tuple=False)
        if positions.numel() == 0:
            raise ValueError("No image tokens found in input_ids.")
        vlo = int(positions[0, 0].item())
        return (vlo, vlo + n_image_tokens)

    # ── _build_token_index ────────────────────────────────────────────────────

    def _build_token_index(
        self,
        input_ids,
        n_image_tokens: int,
        prompt_len: int,
        n_generated: int = 0,
    ) -> TokenIndex:
        """Classify positions using image_token_index and <|im_start|> sentinel.

        For HaploOmni there is no -200 expansion — input_ids map 1:1 to
        inputs_embeds positions (processor already expands the image block).
        """
        ids_1d = input_ids[0]

        positions = (ids_1d == self._image_token_id).nonzero(as_tuple=False)
        if positions.numel() == 0:
            raise ValueError("No image tokens found in input_ids.")
        vlo = int(positions[0, 0].item())
        vhi = vlo + n_image_tokens

        # Last <|im_start|> marks the assistant turn start.
        asst_positions = (ids_1d == self._im_start_id).nonzero(as_tuple=False)
        if asst_positions.numel() > 0:
            asst_start = int(asst_positions[-1, 0].item())
        else:
            asst_start = prompt_len

        categories = torch.zeros(prompt_len + n_generated, dtype=torch.int8)
        categories[vlo:vhi] = int(TokenCategory.VISUAL)

        q_start = vhi
        q_end = asst_start
        if q_start < q_end:
            categories[q_start:q_end] = int(TokenCategory.QUESTION)

        if n_generated > 0:
            categories[prompt_len : prompt_len + n_generated] = int(
                TokenCategory.ANSWER
            )

        visual_range = (vlo, vhi)
        question_range = (q_start, q_end) if q_start < q_end else None
        answer_start = prompt_len if n_generated > 0 else None

        return TokenIndex(
            categories=categories,
            visual_range=visual_range,
            question_range=question_range,
            answer_start=answer_start,
        )

    # ── _get_lm_forward (sweep injection at LLM-stage boundary) ──────────────

    def _get_lm_forward(self):
        """Return a callable for the sweep's _forward_logit / run_patched_forward.

        sweep.py and patch.py call _get_lm_forward()(inputs_embeds=corrupt_emb, ...)
        where corrupt_emb is at LLM hidden dim (3584). HaploOmni's model.forward()
        would feed this through pre_norm (1024-dim) first — wrong.

        Instead, we run the full model with the original (clean) inputs but inject
        corrupt_emb into the hidden states just before the first LLM layer (layer 24)
        via a forward_pre_hook. The pre-stage always sees the clean image; the LLM
        stage sees the injected corrupt/patched embeddings. This is correct for the
        causal sweep, which only patches LLM-stage layers (relative 0-27).
        """
        device = self._model_device
        dtype = self._model_dtype
        first_llm_layer = self.model.model.layers[self._llm_layer_start]

        def _inject(inputs_embeds, use_cache=False, **kwargs):
            inject = inputs_embeds.to(device=device, dtype=dtype)
            fired = [False]

            def _pre_hook(module, args):
                if fired[0]:
                    return args
                fired[0] = True
                return (inject,) + args[1:]

            handle = first_llm_layer.register_forward_pre_hook(_pre_hook)
            try:
                fwd = {k: v for k, v in self._last_inputs.items()}
                fwd["use_cache"] = False
                fwd["return_dict"] = True
                return self.model(**fwd)
            finally:
                handle.remove()

        return _inject

    # ── _call_generate (abstract compliance) ─────────────────────────────────

    def _call_generate(
        self,
        input_ids,
        pixel_values,
        image_sizes,
        max_new_tokens: int,
        **kwargs,
    ):
        """Implement abstract; run_generate overrides the calling context."""
        gen_inputs = {k: v for k, v in self._last_inputs.items()}
        return self.model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
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
        """Single forward pass with activation capture.

        Calls model(**inputs) — the visual tower fires once inside hooks
        (projector hook captured), and the decoder sees the correct sequence.
        """
        input_ids, pixel_values, image_sizes = self._build_prompt(image, question)
        n_image_tokens = int((input_ids[0] == self._image_token_id).sum().item())
        prompt_len = input_ids.shape[1]
        token_index = self._build_token_index(input_ids, n_image_tokens, prompt_len)

        handles, store = register_captures(
            self._get_projector(),
            self._get_decoder_layers(),
            self._get_lm_head(),
            capture_attn_weights=self.capture_attention_weights,
        )
        try:
            fwd_inputs = {k: v for k, v in self._last_inputs.items()}
            fwd_inputs["use_cache"] = False
            fwd_inputs["output_attentions"] = self.capture_attention_weights
            fwd_inputs["return_dict"] = True
            self.model(**fwd_inputs)
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

        With output_scores=True, gen_out.scores gives one entry per decode step.
        Residual/attn/mlp accumulate: prompt_len + (n_generated - 1) positions.
        """
        input_ids, pixel_values, image_sizes = self._build_prompt(image, question)
        n_image_tokens = int((input_ids[0] == self._image_token_id).sum().item())
        prompt_len = input_ids.shape[1]

        handles, store = register_captures(
            self._get_projector(),
            self._get_decoder_layers(),
            self._get_lm_head(),
            capture_attn_weights=self.capture_attention_weights,
        )
        try:
            gen_inputs = {k: v for k, v in self._last_inputs.items()}
            # Use output_scores=True as the compatibility path (same as VILA-U/VILA).
            # logits_to_keep is not guaranteed in HaploOmni's transformers version.
            gen_out = self.model.generate(
                **gen_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True,
                **generate_kwargs,
            )
        finally:
            remove_handles(handles)

        tensors = finalize_store(store)

        # output_scores=True path: len(gen_out.scores) gives true decode step count.
        n_scores = len(gen_out.scores) if getattr(gen_out, "scores", None) else 0
        generated_ids = gen_out.sequences[0][prompt_len:]
        if n_scores > 0:
            # Hook accumulates lm_head rows from prompt_last onward; slice to n_scores.
            tensors["logits"] = tensors["logits"][prompt_len - 1:][:n_scores]
            generated_ids = generated_ids[:n_scores]
        n_generated = len(generated_ids)

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
