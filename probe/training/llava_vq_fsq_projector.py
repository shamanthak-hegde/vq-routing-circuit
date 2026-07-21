"""
FSQLinearProjector — collapse-proof drop-in for LLaVA's mm_projector.

Replaces the single-level VQ codebook (K=16384, collapses to ~90 active codes)
with Finite Scalar Quantization (Mentzer et al. 2024).  FSQ quantizes each
embedding dimension independently to a fixed set of scalar levels, so codebook
collapse is provably impossible — every combination of per-dim levels is a
valid code, and the quantizer visits all of them by construction.

Design choices
--------------
  levels  = [8, 8, 8, 5, 5, 5]  →  8³ × 5³ = 64 000 implicit codes
  code_dim = 6                   (one scalar per level entry)

API contract is identical to VQLinearProjector so LlavaVQHookManager and
train_llava_vq.py can swap projectors transparently:
  - _last_codes : LongTensor(B, N) — unique code index per token
  - forward(x)  → (B, N, lm_dim) in the original dtype
  - codebook_active_fraction() → fraction of implicit codes seen this forward
  - codebook_entropy() → stub (always returns log(n_codes) — FSQ is full-coverage)

Architecture
------------
  CLIP features (B, N, clip_dim)
  → down: Linear(clip_dim, code_dim)      # compress to 6-D FSQ space
  → FSQ:  per-dim scalar quantisation     # no codebook table, no EMA
  → up:   Linear(code_dim, lm_dim)        # match Vicuna hidden dim
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FSQLinearProjector(nn.Module):
    """CLIP → FSQ → LM-hidden projector for LLaVA-VQ-FSQ.

    Parameters
    ----------
    clip_dim    : int  — CLIP hidden size (1024 for ViT-L/14-336)
    lm_dim      : int  — LM hidden size (4096 for Vicuna-7b)
    levels      : list[int] — per-dimension quantization levels;
                  default [8,8,8,5,5,5] → 64 000 implicit codes
    """

    def __init__(
        self,
        clip_dim: int = 1024,
        lm_dim: int = 4096,
        levels: list[int] | None = None,
    ) -> None:
        super().__init__()
        if levels is None:
            levels = [8, 8, 8, 5, 5, 5]

        self.levels = levels
        self.code_dim = len(levels)

        # Precompute quantization boundaries per dimension.
        # For L levels, the L quantization points are:
        #   { -(L-1)/2, -(L-3)/2, …, (L-1)/2 }  (uniformly spaced in tanh-space)
        # We store them as a buffer so they move to GPU with the module.
        grids = []
        for L in levels:
            pts = torch.linspace(-(L - 1) / 2.0, (L - 1) / 2.0, L)
            grids.append(pts)
        # pad to rectangular tensor (max_L,) — shorter dims stay in [:L] slice
        max_L = max(levels)
        grid_tensor = torch.zeros(self.code_dim, max_L)
        for d, pts in enumerate(grids):
            grid_tensor[d, : len(pts)] = pts
        self.register_buffer("_grids", grid_tensor)  # (code_dim, max_L)

        # Stride vector for multi-index → linear code index
        strides = [1]
        for L in reversed(levels[1:]):
            strides.insert(0, strides[0] * L)
        self.register_buffer("_strides", torch.tensor(strides, dtype=torch.long))

        self.n_codes = math.prod(levels)
        self.codebook_size = self.n_codes  # alias used by train loop logging

        self.down = nn.Linear(clip_dim, self.code_dim)
        self.up = nn.Linear(self.code_dim, lm_dim)

        # Track codes seen this forward (same API as VQLinearProjector._last_codes)
        self.register_buffer("_last_codes", torch.zeros(1, 1, dtype=torch.long),
                             persistent=False)
        # Fraction of codes seen (updated in forward; used by logging)
        self._last_active_fraction: float = 0.0

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, clip_dim) or (N, clip_dim) — CLIP patch features

        Returns
        -------
        (B, N, lm_dim) or (N, lm_dim) — projected embeddings in original dtype
        """
        orig_dtype = x.dtype
        squeeze_batch = x.ndim == 2
        if squeeze_batch:
            x = x.unsqueeze(0)
        B, N, _ = x.shape

        x = x.float()
        z = self.down(x)                   # (B, N, code_dim)
        flat = z.reshape(-1, self.code_dim)  # (B*N, code_dim)

        # Per-dimension scalar quantization (straight-through estimator)
        # Scale z into tanh range for each dimension, then round to nearest grid point.
        quant_flat = torch.zeros_like(flat)
        indices = torch.zeros(flat.shape[0], self.code_dim, dtype=torch.long, device=x.device)
        for d, L in enumerate(self.levels):
            pts = self._grids[d, :L]                    # (L,)
            z_d = flat[:, d:d+1]                       # (B*N, 1)
            # Nearest-neighbour in 1-D: argmin |z_d - pts|
            dist_d = (z_d - pts.unsqueeze(0)).abs()    # (B*N, L)
            idx_d = dist_d.argmin(dim=1)               # (B*N,)
            indices[:, d] = idx_d
            # Quantized value (STE: gradient flows through z_d)
            q_d = pts[idx_d]                           # (B*N,)
            quant_flat[:, d] = (q_d - flat[:, d]).detach() + flat[:, d]

        # Compute linear code index and store
        code_indices = (indices * self._strides.unsqueeze(0)).sum(dim=1)  # (B*N,)
        codes_reshaped = code_indices.reshape(B, N)
        if self._last_codes.shape != codes_reshaped.shape:
            object.__setattr__(self, "_last_codes", codes_reshaped.detach())
        else:
            self._last_codes.data = codes_reshaped.detach()

        # Track utilization (fraction of implicit codes seen)
        unique_codes = code_indices.unique().numel()
        self._last_active_fraction = unique_codes / self.n_codes

        quant = quant_flat.reshape(B, N, self.code_dim)
        out = self.up(quant)               # (B, N, lm_dim) float32
        out = out.to(orig_dtype)
        if squeeze_batch:
            out = out.squeeze(0)
        return out

    # ── Codebook helpers (same API as VQLinearProjector) ─────────────────────

    def codebook_active_fraction(self, threshold: float = 1.0) -> float:
        """Fraction of implicit FSQ codes seen in the last forward pass."""
        return self._last_active_fraction

    def codebook_entropy(self) -> float:
        """FSQ has uniform coverage by construction; return log(n_codes)."""
        return math.log(self.n_codes)

    # No EMA buffers — FSQ has no learnable codebook. _commit_loss stub
    # lets the training loop work without special-casing.
    @property
    def _commit_loss(self) -> "torch.Tensor":
        return torch.zeros(1, device=next(self.parameters()).device)

    # Training-time: entropy reg not applicable to FSQ (always max entropy)
    @property
    def _entropy_loss(self) -> "torch.Tensor":
        return torch.zeros(1, device=next(self.parameters()).device)
