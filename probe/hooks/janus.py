"""
JanusProHookManager — concrete hook manager for Janus-Pro-7B (DeepSeek).

Architecture
------------
Janus-Pro uses DECOUPLED visual encoders:
  - Understanding path : SigLIP → model.aligner (MLP) → language_model
  - Generation path    : model.gen_vision_model (VQ) → model.gen_aligner

Only the understanding path is relevant for yes/no visual grounding.
The understanding path is a CONTINUOUS projector (SigLIP+MLP), NOT VQ.
This means Janus-Pro is predicted to be continuous-projector class — same
circuit as LLaVA/Qwen/HaploOmni, NOT the VQ pathology of VILA-U.

Module accessors
-----------------
  Projector      : model.aligner           (understanding MLP)
  Decoder layers : model.language_model.model.layers
  LM head        : model.language_model.lm_head
  LM forward     : model.language_model

Token layout (Janus conversation template)
-------------------------------------------
  [BOS] [system tokens] [<|User|> + image_placeholder×N + question + <|Assistant|>]
            OTHER              OTHER          VISUAL              QUESTION      OTHER

n_image_tokens = 576 (SigLIP 24×24 patches, fixed regardless of image size).

Conda env: janus
  Install: pip install git+https://github.com/deepseek-ai/Janus
  Load:    AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)

Generate contract
-----------------
model.language_model.generate(inputs_embeds=...) after
model.prepare_inputs_embeds(**prepare_inputs).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
from PIL import Image

from .base import VLMHookManager
from .schema import TokenCategory, TokenIndex

_JANUS_N_IMAGE_TOKENS = 576  # SigLIP 24×24 — fixed for Janus-Pro


class JanusProHookManager(VLMHookManager):
    """Hook manager for Janus-Pro-7B.

    Parameters
    ----------
    model               : MultiModalityCausalLM in eval mode (trust_remote_code=True)
    processor           : VLChatProcessor from VLChatProcessor.from_pretrained(...)
    capture_attention_weights : opt-in; off by default
    """

    def __init__(
        self,
        model,
        processor,
        capture_attention_weights: bool = False,
    ):
        # All janus imports are deferred; they only need to be present in the
        # janus conda env.  Importing at the top level would break other envs.
        from janus.models import VLChatProcessor  # noqa: F401 — confirms env

        self.model = model
        self.processor = processor
        self.capture_attention_weights = capture_attention_weights

        # Locate underlying LLM for device / dtype
        self._llm = model.language_model
        self._model_device = next(self._llm.parameters()).device
        self._model_dtype = next(self._llm.parameters()).dtype

        # image_token_index: the special token that marks image placeholder
        # positions in input_ids.  Janus stores it in processor.image_start_id
        # or tokenizer.convert_tokens_to_ids("<image_placeholder>").
        self._image_token_id = processor.tokenizer.convert_tokens_to_ids(
            processor.image_token
        )

        # ASSISTANT tag length — differential tokenisation over two prompts
        self._asst_suffix_len: int = self._compute_asst_suffix_len(processor)

    # ── Module accessors ──────────────────────────────────────────────────────

    def _get_projector(self):
        return self.model.aligner  # understanding MLP projector (SigLIP output → LLM dim)

    def _get_decoder_layers(self):
        return self.model.language_model.model.layers

    def _get_lm_head(self):
        return self.model.language_model.lm_head

    def _get_lm_forward(self):
        return self.model.language_model

    # ── Prompt building ───────────────────────────────────────────────────────

    def _build_prompt(self, image: Image.Image, question: str) -> tuple:
        """Build (prepare_inputs, None, None) for Janus-Pro.

        Returns the full VLChatProcessor output dict in place of the usual
        (input_ids, images_tensor, image_sizes) triple.  _prepare_embeds and
        _call_generate both accept this form.
        """
        prior = getattr(self, "_followup_prior", None)
        if prior:
            pq, pa = prior
            conversation = [
                {"role": "<|User|>", "content": f"<image_placeholder>\n{pq}",
                 "images": [image]},
                {"role": "<|Assistant|>", "content": pa},
                {"role": "<|User|>", "content": question},
                {"role": "<|Assistant|>", "content": ""},
            ]
        else:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{question}",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        prepare_inputs = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
        ).to(self._model_device)
        # input_ids from the processor contain the expanded image token positions
        return prepare_inputs, None, None

    # ── Embedding preparation ────────────────────────────────────────────────

    def _prepare_embeds(
        self,
        prepare_inputs,    # the full VLChatProcessor output dict
        _images_tensor,    # unused
        _image_sizes,      # unused
    ) -> tuple[torch.Tensor, int]:
        """Expand image tokens into SigLIP projected embeddings."""
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        inputs_embeds = inputs_embeds.to(self._model_dtype)
        return inputs_embeds, _JANUS_N_IMAGE_TOKENS

    # ── Token classification ─────────────────────────────────────────────────

    def _build_token_index(
        self,
        prepare_inputs,
        n_image_tokens: int,
        prompt_len: int,
        n_generated: int = 0,
    ) -> TokenIndex:
        """Map embedding positions to TokenCategory for Janus token layout.

        Janus uses the image_token_id sentinel (not -200) to mark image
        positions in input_ids before expansion.
        """
        input_ids = prepare_inputs.input_ids[0]   # (seq_pre_expand,)

        # Find contiguous image_token block
        is_img = (input_ids == self._image_token_id)
        img_positions = is_img.nonzero(as_tuple=False)[:, 0]
        if img_positions.numel() == 0:
            raise ValueError("No image tokens found in Janus input_ids")

        pos_img_start = int(img_positions[0].item())
        n_img_in_ids = int(img_positions.numel())
        # Each sentinel expands to n_image_tokens / n_img_in_ids slots.
        # Janus places one sentinel per image, so expansion = n_image_tokens.
        expansion = _JANUS_N_IMAGE_TOKENS

        pos_img_end = pos_img_start + expansion  # in embedding space
        shift = expansion - n_img_in_ids

        # ASSISTANT tag end (in pre-expansion id space)
        pos_asst_raw = self._locate_assistant_tag(input_ids, pos_img_start + 1)

        cats = torch.zeros(prompt_len + n_generated, dtype=torch.int8)
        cats[pos_img_start:pos_img_end] = int(TokenCategory.VISUAL)

        if pos_asst_raw is not None:
            q_start = pos_img_end
            q_end = pos_asst_raw + shift
            if q_start < q_end:
                cats[q_start:q_end] = int(TokenCategory.QUESTION)
        else:
            cats[pos_img_end:prompt_len] = int(TokenCategory.QUESTION)

        if n_generated > 0:
            cats[prompt_len:prompt_len + n_generated] = int(TokenCategory.ANSWER)

        q_mask = (cats[:prompt_len] == int(TokenCategory.QUESTION))
        q_indices = q_mask.nonzero(as_tuple=False)
        q_range = (
            (int(q_indices[0, 0].item()), int(q_indices[-1, 0].item()) + 1)
            if q_indices.numel() > 0 else None
        )

        return TokenIndex(
            categories=cats,
            visual_range=(pos_img_start, pos_img_end),
            question_range=q_range,
            answer_start=prompt_len if n_generated > 0 else None,
        )

    def visual_range(self, prepare_inputs, n_image_tokens: int) -> tuple[int, int]:
        """Return (vlo, vhi) — the visual token slice in inputs_embeds."""
        input_ids = prepare_inputs.input_ids[0]
        is_img = (input_ids == self._image_token_id)
        img_positions = is_img.nonzero(as_tuple=False)[:, 0]
        pos_img_start = int(img_positions[0].item())
        return (pos_img_start, pos_img_start + _JANUS_N_IMAGE_TOKENS)

    # ── Generate ──────────────────────────────────────────────────────────────

    def _call_generate(
        self,
        prepare_inputs,
        _images_tensor,
        _image_sizes,
        max_new_tokens: int,
        **kwargs,
    ):
        # Re-run prepare_inputs_embeds so the projector hook fires with hooks active.
        embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        embeds = embeds.to(self._model_dtype)
        attn_mask = getattr(prepare_inputs, "attention_mask", None)
        return self.model.language_model.generate(
            inputs_embeds=embeds,
            attention_mask=attn_mask,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )

    def _extract_input_ids_for_capture(self, prepare_inputs) -> "torch.Tensor":
        return prepare_inputs.input_ids.squeeze(0).cpu()

    # ── Visual slice helper ───────────────────────────────────────────────────

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
        # called from run_generate before token_index is built
        assert n_image_tokens is not None
        # input_ids here is actually prepare_inputs
        vlo, vhi = self.visual_range(input_ids, n_image_tokens)
        return inputs_embeds[0, vlo:vhi, :].detach().cpu()

    # ── ASSISTANT tag detection ───────────────────────────────────────────────

    def _compute_asst_suffix_len(self, processor) -> int:
        """Count tokens occupied by the <|Assistant|>: closing tag.

        Differential tokenisation: two prompts with different questions,
        common suffix = assistant tag.
        """
        def _tok(q: str) -> list:
            conversation = [
                {"role": "<|User|>", "content": f"<image_placeholder>\n{q}",
                 "images": []},
                {"role": "<|Assistant|>", "content": ""},
            ]
            text = processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=processor.sft_format,
                system_prompt=processor.system_prompt,
            )
            return processor.tokenizer.encode(text)

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
