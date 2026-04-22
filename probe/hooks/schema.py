"""
TokenCategory, TokenIndex, and Capture dataclasses.

The token-category mapping lives here — not in the downstream tracing code —
so that heatmap axes stay comparable when this hook layer is ported to VILA-U
and other unified models in Week 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import torch
from torch import Tensor


class TokenCategory(IntEnum):
    OTHER    = 0  # BOS, system prompt, chat-template scaffolding, role tags
    VISUAL   = 1  # image patch tokens (includes anyres newline tokens for LLaVA-1.6)
    QUESTION = 2  # the user's question text
    ANSWER   = 3  # model-generated tokens (absent in prefill-only mode)


@dataclass
class TokenIndex:
    """Per-position token-category labels for a single forward pass.

    ``categories`` is defined over the *post-splice* sequence length (after
    visual tokens have been expanded from the -200 sentinel into N real
    patch embeddings).
    """
    categories:     Tensor                   # (seq_len,) int8
    visual_range:   Optional[tuple[int,int]] # [start, end) — contiguous image block
    question_range: Optional[tuple[int,int]] # [start, end) — contiguous question text
    answer_start:   Optional[int]            # first generated-token position, or None

    def mask(self, category: TokenCategory) -> Tensor:
        """Boolean mask of shape (seq_len,)."""
        return self.categories == int(category)

    @property
    def prompt_last(self) -> int:
        """Index of the last prompt token — where yes/no next-token logits live."""
        if self.answer_start is not None:
            return self.answer_start - 1
        return int(self.categories.shape[0]) - 1

    def last_token_is(self, category: TokenCategory) -> bool:
        return int(self.categories[self.prompt_last]) == int(category)


@dataclass
class Capture:
    """All activations captured during one (prefill or generate) run.

    Tensor conventions
    ------------------
    - Batch dimension is squeezed (B=1 assumed throughout).
    - ``seq_len = prompt_len`` in prefill mode;
      ``seq_len = prompt_len + n_generated`` in generate mode.
    - In prefill mode, ``attn_out`` / ``mlp_out`` / ``residual`` / ``logits``
      all share ``seq_len``.
    - In generate mode, ``attn_out`` / ``mlp_out`` / ``residual`` are aligned
      to captured sequence positions, while ``logits`` contains one row per
      generation decision step (typically ``n_generated`` rows).
    """

    # ── identity ──────────────────────────────────────────────────────────────
    input_ids:   Tensor       # (prompt_len,) with IMAGE_TOKEN_INDEX (-200) sentinel
    token_index: TokenIndex

    # ── Hook 1: visual tokens as emitted by the projector ────────────────────
    # Raw projector output before anyres unpadding / newline insertion / splicing.
    # Shape: (num_crops, num_patches_per_crop, hidden_size)  e.g. (5, 576, 4096)
    projected_visual: Tensor
    # The contiguous visual-token block that the LM actually sees (post-splice).
    # Shape: (n_image_tokens, hidden_size)
    visual_embeds:    Tensor

    # ── Hooks 2-4: per-layer LM activations ──────────────────────────────────
    # All three have shape (n_layers, seq_len, hidden_size).
    attn_out:  Tensor   # post-o_proj self-attention output (pre-residual-add)
    mlp_out:   Tensor   # MLP down-proj output (pre-residual-add)
    residual:  Tensor   # full residual stream after each decoder layer

    # ── Final outputs ─────────────────────────────────────────────────────────
    logits:          Tensor          # prefill: (seq_len, vocab_size); generate: (n_steps, vocab_size)
    n_image_tokens:  int
    # Populated only in generate mode; None in prefill.
    generated_ids:   Optional[Tensor] = None

    # ── Opt-in: raw attention weights (eager backend only) ───────────────────
    # (n_layers, n_heads, seq_len, seq_len) in prefill;
    # None when HookManager was created with capture_attention_weights=False.
    attn_weights: Optional[Tensor] = None

    # ── Convenience ───────────────────────────────────────────────────────────

    def last_token_residual(self) -> Tensor:
        """Residual stream at the prompt-last position: (n_layers, hidden_size)."""
        return self.residual[:, self.token_index.prompt_last, :]

    def by_category(
        self,
        category: TokenCategory,
        which: str = "residual",
    ) -> Tensor:
        """Slice a (n_layers, seq_len, hidden) tensor to the given category.

        Returns (n_layers, n_cat_tokens, hidden).
        """
        mask = self.token_index.mask(category)   # (seq_len,)
        t = getattr(self, which)                  # (n_layers, seq_len, hidden)
        return t[:, mask, :]

    def logit_lens(self, lm_head: torch.nn.Module, ln_f: torch.nn.Module) -> Tensor:
        """Project residual stream through the unembedding matrix at each layer.

        Returns (n_layers, seq_len, vocab_size).
        Requires the model's ``lm_head`` and final layer-norm ``model.model.norm``.
        """
        n_layers, seq_len, hidden = self.residual.shape
        flat = self.residual.reshape(n_layers * seq_len, hidden)
        with torch.no_grad():
            normed = ln_f(flat)
            logits = lm_head(normed)
        return logits.reshape(n_layers, seq_len, -1)

    def to(self, device: str) -> "Capture":
        """Move all tensors to *device* in-place; returns self."""
        for attr_name in (
            "input_ids", "projected_visual", "visual_embeds",
            "attn_out", "mlp_out", "residual", "logits",
        ):
            val = getattr(self, attr_name)
            if val is not None:
                setattr(self, attr_name, val.to(device))
        for attr_name in ("generated_ids", "attn_weights"):
            val = getattr(self, attr_name)
            if val is not None:
                setattr(self, attr_name, val.to(device))
        self.token_index.categories = self.token_index.categories.to(device)
        return self

    def save(self, path: str) -> None:
        """Persist all tensors to a single .pt file."""
        payload = {
            "input_ids":        self.input_ids,
            "categories":       self.token_index.categories,
            "visual_range":     self.token_index.visual_range,
            "question_range":   self.token_index.question_range,
            "answer_start":     self.token_index.answer_start,
            "projected_visual": self.projected_visual,
            "visual_embeds":    self.visual_embeds,
            "attn_out":         self.attn_out,
            "mlp_out":          self.mlp_out,
            "residual":         self.residual,
            "logits":           self.logits,
            "n_image_tokens":   self.n_image_tokens,
            "generated_ids":    self.generated_ids,
            "attn_weights":     self.attn_weights,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "Capture":
        p = torch.load(path, map_location=device)
        token_index = TokenIndex(
            categories=p["categories"],
            visual_range=p["visual_range"],
            question_range=p["question_range"],
            answer_start=p["answer_start"],
        )
        return cls(
            input_ids=p["input_ids"],
            token_index=token_index,
            projected_visual=p["projected_visual"],
            visual_embeds=p["visual_embeds"],
            attn_out=p["attn_out"],
            mlp_out=p["mlp_out"],
            residual=p["residual"],
            logits=p["logits"],
            n_image_tokens=p["n_image_tokens"],
            generated_ids=p.get("generated_ids"),
            attn_weights=p.get("attn_weights"),
        )
