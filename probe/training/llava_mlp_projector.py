"""MLPProjector — matched-compute MLP control for LLaVA-VQ.

2-layer GeLU MLP matching LLaVA-1.6's native mm_projector architecture:
    CLIP features (B, N, 1024)
    → Linear(1024, lm_dim)   [down + up collapsed into two layers with GeLU]
    → GeLU
    → Linear(lm_dim, lm_dim)

This is a drop-in replacement for VQLinearProjector; used by train_llava_mlp.py
to train a matched-compute control under identical hyperparameters as LLaVA-VQ.

Matched-compute means: same dataset (50k CC3M), same steps (2000), same
batch size (8), same grad-accum (4), same LR (2e-3), same CLIP & Vicuna frozen.
No VQ bottleneck → no codebook collapse → no L0 routing pathology (prediction).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """2-layer GeLU MLP projector matching LLaVA-1.6's mm_projector.

    Parameters
    ----------
    clip_dim : int — CLIP hidden size (1024 for ViT-L/14-336)
    lm_dim   : int — LM hidden size (4096 for Vicuna-7b)
    """

    def __init__(self, clip_dim: int = 1024, lm_dim: int = 4096) -> None:
        super().__init__()
        self.fc1 = nn.Linear(clip_dim, lm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(lm_dim, lm_dim)

        # Placeholders so training loop can call these without branching
        self._commit_loss: torch.Tensor = torch.zeros(1)
        self._entropy_loss: torch.Tensor = torch.zeros(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, clip_dim)  — CLIP patch features

        Returns
        -------
        (B, N, lm_dim)  — same dtype as input
        """
        orig_dtype = x.dtype
        squeeze_batch = x.ndim == 2
        if squeeze_batch:
            x = x.unsqueeze(0)

        x = x.float()
        out = self.fc2(self.act(self.fc1(x)))

        # Keep commit/entropy loss attributes at zero — no VQ bottleneck
        device = out.device
        self._commit_loss = torch.zeros(1, device=device)
        self._entropy_loss = torch.zeros(1, device=device)

        out = out.to(orig_dtype)
        if squeeze_batch:
            out = out.squeeze(0)
        return out

    # Stub codebook helpers so training-loop codebook_entropy() calls don't crash
    def codebook_entropy(self) -> float:
        return 0.0

    def codebook_active_fraction(self, threshold: float = 1.0) -> float:
        return 1.0
