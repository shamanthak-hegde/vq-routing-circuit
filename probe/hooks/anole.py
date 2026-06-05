"""
AnoleHookManager — hook manager for GAIR/Anole-7b-v0.1.

Loading strategy
----------------
Anole weights are stored in Meta's consolidated.pth format.  A one-time
conversion script (run_anole_conversion.py) remaps keys and saves an HF-format
checkpoint to anole/Anole-7b-v0.1-hf/.  The LM (ChameleonForCausalLM) is loaded
from that HF checkpoint.

Image encoding
--------------
The HF VQGAN (model.model.vqmodel) has cuDNN compatibility issues in this
environment.  We bypass it entirely and use the Meta ImageTokenizer directly
(anole/Anole-7b-v0.1/tokenizer/vqgan.{yaml,ckpt}), the same path used by
anole/inference.py.  This gives:

    PIL image → Meta ImageTokenizer → VQ codes [0,8191]
                → VocabTranslation → BPE IDs [4,8195]
                → model.model.embed_tokens → (1, n_vq, hidden)

ChameleonProcessor expands <image> into [BOI, 1024×65536, EOI] in input_ids.
_prepare_embeds replaces the 65536 slots with real BPE IDs before embed_tokens.
_call_generate inlines the same BPE IDs and calls model.generate without pixel_values.

Everything else (layer hooks, token index, head_knockout) is inherited from
ChameleonHFHookManager unchanged.

Converted HF checkpoint: anole/Anole-7b-v0.1-hf/
Meta tokenizer source:    anole/Anole-7b-v0.1/tokenizer/
"""

from __future__ import annotations

import json
import os
import sys

import torch
from PIL import Image
from torch import nn

from .chameleon_hf import ChameleonHFHookManager

_ANOLE_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "anole")
)
_ANOLE_TRANSFORMERS = os.path.join(_ANOLE_ROOT, "transformers", "src")
_ANOLE_CHAMELEON   = _ANOLE_ROOT   # chameleon.inference.* is under anole/

_META_TOK_DIR  = os.path.join(_ANOLE_ROOT, "Anole-7b-v0.1", "tokenizer")
_DEFAULT_HF_PATH = os.path.join(_ANOLE_ROOT, "Anole-7b-v0.1-hf")


# ── visual embedder (bypasses HF VQGAN) ──────────────────────────────────────

class _AnoleVisualEmbedder(nn.Module):
    """Call embed_tokens on BPE image-token IDs.

    Input: (1, n_vq) int32 or int64 tensor of BPE IDs produced by
    VocabTranslation.convert_img2bp2 — NOT pixel_values.

    Satisfies the projector-hook contract: fires exactly once per
    _prepare_embeds / _call_generate, letting register_captures capture the
    visual embeddings.
    """

    def __init__(self, embed_tokens: nn.Embedding) -> None:
        super().__init__()
        self.__dict__["_embed_ref"] = embed_tokens

    def forward(self, bpe_ids: torch.Tensor) -> torch.Tensor:
        return self.__dict__["_embed_ref"](bpe_ids.long())  # (1, n_vq, hidden)


# ── main class ────────────────────────────────────────────────────────────────

class AnoleHookManager(ChameleonHFHookManager):
    """Hook manager for GAIR/Anole-7b-v0.1.

    Use AnoleHookManager.from_pretrained() to load.

    Parameters
    ----------
    model          : ChameleonForCausalLM, attn_implementation="eager"
    processor      : ChameleonProcessor
    image_tokenizer: Meta ImageTokenizer (from anole/Anole-7b-v0.1/tokenizer/)
    vocab_trans    : VocabTranslation (codes → BPE IDs)
    image_size     : resize images to this before VQ-encoding (default 512)
    """

    def __init__(
        self,
        model,
        processor,
        image_tokenizer,
        vocab_trans,
        image_size: int = 512,
        capture_attention_weights: bool = False,
    ) -> None:
        # Ensure paths for vendored transformers and Meta chameleon
        for p in (_ANOLE_TRANSFORMERS, _ANOLE_CHAMELEON):
            if p not in sys.path:
                sys.path.insert(0, p)

        # Bootstrap directly from VLMHookManager — skip ChameleonHFHookManager.__init__
        # because it calls model.model.get_image_features which doesn't exist here.
        from .base import VLMHookManager
        VLMHookManager.__init__(self, model, capture_attention_weights)

        self.processor = processor
        self._image_tokenizer = image_tokenizer
        self._vocab_trans = vocab_trans
        self._image_size = image_size

        self._model_device = next(model.parameters()).device
        self._model_dtype  = next(model.parameters()).dtype

        # Stored for compatibility with inherited methods (not used in our overrides)
        self._image_token_id: int = int(model.model.vocabulary_mapping.image_token_id)

        # Visual embedder: wraps embed_tokens (no VQGAN call)
        self._visual_embedder = _AnoleVisualEmbedder(model.model.embed_tokens)

        self._asst_suffix_len: int = self._compute_asst_suffix_len()

        print(
            f"AnoleHookManager: n_layers={len(model.model.layers)}, "
            f"hidden={model.config.hidden_size}, "
            f"image_token_id={self._image_token_id}, "
            f"device={self._model_device}",
            flush=True,
        )

    # ── class factory ─────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        hf_checkpoint_path: str = _DEFAULT_HF_PATH,
        capture_attention_weights: bool = False,
        image_size: int = 512,
    ) -> "AnoleHookManager":
        """Load from the converted HF checkpoint + Meta tokenizer.

        hf_checkpoint_path : path produced by run_anole_conversion.py
            (default: anole/Anole-7b-v0.1-hf/)
        """
        for p in (_ANOLE_TRANSFORMERS, _ANOLE_CHAMELEON):
            if p not in sys.path:
                sys.path.insert(0, p)

        import torch as _torch
        from transformers import ChameleonForCausalLM, ChameleonProcessor
        from chameleon.inference.image_tokenizer import ImageTokenizer
        from chameleon.inference.vocab import VocabInfo, VocabTranslation

        print(f"Loading AnoleHookManager from {hf_checkpoint_path} ...", flush=True)
        processor = ChameleonProcessor.from_pretrained(hf_checkpoint_path)
        model = ChameleonForCausalLM.from_pretrained(
            hf_checkpoint_path,
            torch_dtype=_torch.bfloat16,
            attn_implementation="eager",
            device_map="cuda",
        ).eval()

        # Meta ImageTokenizer (bypasses HF VQGAN)
        vqgan_cfg  = os.path.join(_META_TOK_DIR, "vqgan.yaml")
        vqgan_ckpt = os.path.join(_META_TOK_DIR, "vqgan.ckpt")
        tok_json   = os.path.join(_META_TOK_DIR, "text_tokenizer.json")

        print(f"Loading Meta ImageTokenizer from {vqgan_cfg} ...", flush=True)
        image_tokenizer = ImageTokenizer(
            cfg_path=vqgan_cfg, ckpt_path=vqgan_ckpt, device="cuda"
        )

        with open(tok_json) as f:
            tok_cfg = json.load(f)
        vocab_map = dict(tok_cfg["model"]["vocab"])
        for entry in tok_cfg.get("added_tokens", []):
            vocab_map[entry["content"]] = entry["id"]

        vocab       = VocabInfo(vocab_map)
        vocab_trans = VocabTranslation(vocab_info=vocab, device="cuda")

        return cls(
            model, processor, image_tokenizer, vocab_trans,
            image_size=image_size,
            capture_attention_weights=capture_attention_weights,
        )

    # ── image helpers ─────────────────────────────────────────────────────────

    def _pad_and_resize(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB")
        w, h = img.size
        size = max(w, h)
        bg = Image.new("RGB", (size, size), (122, 116, 104))
        bg.paste(img, ((size - w) // 2, (size - h) // 2))
        return bg.resize((self._image_size, self._image_size), Image.LANCZOS)

    def _image_to_bpe_ids(self, image: Image.Image) -> torch.Tensor:
        """PIL image → (1, n_vq) BPE ID tensor on model device."""
        img_sq = self._pad_and_resize(image)
        with torch.no_grad():
            vq_codes = self._image_tokenizer.img_tokens_from_pil(img_sq).cpu()
        bpe_ids = self._vocab_trans.convert_img2bp2(vq_codes).long()
        return bpe_ids.unsqueeze(0).to(self._model_device)  # (1, n_vq)

    # ── overridden interface ──────────────────────────────────────────────────

    # Image-slot placeholder value used by ChameleonProcessor: one past vocab_size.
    _PLACEHOLDER_ID: int = 65536

    def _build_prompt(self, image: Image.Image, question: str) -> tuple:
        """Return (input_ids_with_placeholders, raw_pil_image, None).

        ChameleonProcessor expands <image> into [BOI, 1024×65536, EOI] in
        input_ids and returns pixel_values separately.  We keep the raw PIL
        image instead so downstream can use the Meta ImageTokenizer directly.
        """
        prompt = f"<image>{question}\nAnswer:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        input_ids = inputs.input_ids.to(self._model_device)
        return input_ids, image, None   # PIL image as second element

    def _prepare_embeds(
        self,
        input_ids,
        pil_image: Image.Image,
        _image_sizes,
    ) -> tuple[torch.Tensor, int]:
        """Fill the 65536 image slots with real BPE IDs, then embed everything.

        1. Convert PIL → VQ codes → BPE IDs via Meta ImageTokenizer
        2. Replace the 1024 placeholder slots (value 65536) in input_ids
        3. Call embed_tokens on the fixed sequence (no splicing — same length)
        4. Fire the projector hook via _visual_embedder
        """
        # Identify the 1024 placeholder positions
        ids_1d = input_ids[0]
        placeholder_mask = ids_1d == self._PLACEHOLDER_ID          # (seq,) bool

        # Get BPE IDs from Meta ImageTokenizer
        bpe_ids = self._image_to_bpe_ids(pil_image)                # (1, n_vq)
        n_vq = bpe_ids.shape[1]

        # Replace placeholders with real BPE IDs
        fixed_ids = input_ids.clone()
        fixed_ids[0, placeholder_mask] = bpe_ids.squeeze(0)

        # Fire projector hook
        self._visual_embedder(bpe_ids)                              # (1, n_vq, H)

        # Embed full fixed sequence (no length change)
        inputs_embeds = self.model.model.embed_tokens(fixed_ids)   # (1, seq, H)
        return inputs_embeds, n_vq

    def visual_range(self, input_ids, n_image_tokens: int) -> tuple[int, int]:
        """Visual token range = where input_ids has 65536 placeholders."""
        ids_1d = input_ids[0] if input_ids.ndim == 2 else input_ids
        positions = (ids_1d == self._PLACEHOLDER_ID).nonzero(as_tuple=False)[:, 0]
        return int(positions[0].item()), int(positions[-1].item()) + 1

    def _build_token_index(
        self,
        input_ids,
        n_image_tokens: int,
        prompt_len: int,
        n_generated: int = 0,
    ):
        """TokenIndex using 65536-slot positions for the visual range."""
        from .schema import TokenCategory, TokenIndex

        ids_1d = input_ids[0]
        vis_lo, vis_hi = self.visual_range(ids_1d, n_image_tokens)

        asst_pos = self._locate_assistant_tag(ids_1d, vis_hi)
        q_end = asst_pos if asst_pos is not None else prompt_len

        categories = torch.zeros(prompt_len + n_generated, dtype=torch.int8)
        categories[vis_lo:vis_hi] = int(TokenCategory.VISUAL)

        q_start = vis_hi + 1  # skip EOI token after image block
        if q_start < q_end:
            categories[q_start:q_end] = int(TokenCategory.QUESTION)

        if n_generated > 0:
            categories[prompt_len : prompt_len + n_generated] = int(TokenCategory.ANSWER)

        q_mask = categories[:prompt_len] == int(TokenCategory.QUESTION)
        q_idxs = q_mask.nonzero(as_tuple=False)
        q_range = (
            (int(q_idxs[0, 0]), int(q_idxs[-1, 0]) + 1)
            if q_idxs.numel() > 0 else None
        )
        return TokenIndex(
            categories=categories,
            visual_range=(vis_lo, vis_hi),
            question_range=q_range,
            answer_start=prompt_len if n_generated > 0 else None,
        )

    def _slice_visual(
        self,
        inputs_embeds,
        token_index,
        n_image_tokens=None,
        input_ids=None,
    ) -> torch.Tensor:
        """Extract visual-token rows from inputs_embeds.

        When token_index is available (run_prefill path), use its visual_range.
        When not available (run_generate pre-build path), use the 65536-slot
        positions from input_ids — not _image_token_id=8711 which is absent.
        """
        if token_index is not None:
            vr = token_index.visual_range
            return inputs_embeds[0, vr[0]:vr[1], :].detach().cpu()
        vis_lo, vis_hi = self.visual_range(input_ids, n_image_tokens)
        return inputs_embeds[0, vis_lo:vis_hi, :].detach().cpu()

    def _call_generate(
        self,
        input_ids,
        pil_image: Image.Image,
        _image_sizes,
        max_new_tokens: int,
        **kwargs,
    ):
        """Generate with BPE IDs inlined — no pixel_values, no HF VQGAN call.

        Also fires the projector hook once to satisfy finalize_store.
        """
        bpe_ids = self._image_to_bpe_ids(pil_image)               # (1, n_vq)
        with torch.no_grad():
            self._visual_embedder(bpe_ids)                         # fires hook

        # Replace 65536 placeholders with real BPE IDs
        fixed_ids = input_ids.clone()
        fixed_ids[0, input_ids[0] == self._PLACEHOLDER_ID] = bpe_ids.squeeze(0)

        return self.model.generate(
            fixed_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )
