"""
VQLinearProjector — drop-in replacement for LLaVA's mm_projector.

Replaces the mlp2x_gelu MLP with:
    CLIP features (B, N, 1024)
    → down: Linear(1024, code_dim)        # compress to VQ embedding space
    → VQ:   single-level VQ               # codebook nearest-neighbour lookup
    → up:   Linear(code_dim, lm_hidden)   # match Vicuna hidden dim (4096)

The VQ bottleneck uses EMA codebook updates (no codebook loss term) and a
straight-through estimator (STE) for gradient flow through the argmin.

K-means initialisation is strongly recommended before STE training — see
probe/training/kmeans_init.py.  Pass a (K, code_dim) tensor as `init_codebook`
to __init__, or call `load_kmeans_codebook(path)` after construction.

VQ codes recorded in `_last_codes: LongTensor(B, N)` after each forward
so that LlavaVQHookManager._get_vq_codes() can read them without a second pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQLinearProjector(nn.Module):
    """CLIP → VQ → LM-hidden projector for LLaVA-VQ.

    Parameters
    ----------
    clip_dim    : int  — CLIP hidden size (1024 for ViT-L/14-336)
    lm_dim      : int  — LM hidden size (4096 for Vicuna-7b)
    code_dim    : int  — VQ codebook entry dimension (32 recommended)
    codebook_size: int — number of codebook entries (16384 recommended)
    commitment  : float — commitment loss coefficient β (0.25)
    ema_decay   : float — EMA decay for codebook update (0.99)
    init_codebook: optional (K, code_dim) tensor — skip EMA warm-up
    """

    def __init__(
        self,
        clip_dim: int = 1024,
        lm_dim: int = 4096,
        code_dim: int = 32,
        codebook_size: int = 16384,
        commitment: float = 0.25,
        ema_decay: float = 0.99,
        init_codebook: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.codebook_size = codebook_size
        self.commitment = commitment
        self.ema_decay = ema_decay

        self.down = nn.Linear(clip_dim, code_dim)
        self.up = nn.Linear(code_dim, lm_dim)

        # EMA codebook: not a Parameter — updated by exponential moving average
        embed = torch.randn(codebook_size, code_dim) if init_codebook is None \
                else init_codebook.float().clone()
        self.register_buffer("codebook", embed)
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_embed_sum", embed.clone())

        # Last seen VQ code indices — read by _get_vq_codes without re-forward.
        # persistent=False: excluded from state_dict (shape is dynamic, per-batch).
        self.register_buffer("_last_codes", torch.zeros(1, 1, dtype=torch.long),
                             persistent=False)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, clip_dim)  — CLIP patch features, any float dtype

        Returns
        -------
        (B, N, lm_dim)  — projected embeddings for the LM, same dtype as input
        """
        orig_dtype = x.dtype
        # Qwen's visual.merger passes (N_total, C); LLaVA passes (B, N, C).
        # Normalise to 3-D for the rest of the forward, then restore on exit.
        squeeze_batch = x.ndim == 2
        if squeeze_batch:
            x = x.unsqueeze(0)          # (1, N_total, C)
        B, N, _ = x.shape

        # Run all projector arithmetic in float32 so Linear weights (fp32 by
        # default) and input (possibly fp16 from CLIP) have matching dtypes.
        x = x.float()

        # Compress to code space (fp32 for argmin precision)
        z = self.down(x)                   # (B, N, code_dim)

        # VQ argmin lookup
        flat = z.reshape(-1, self.code_dim)                      # (B*N, code_dim)
        with torch.no_grad():
            dist = (
                flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.codebook.t()
                + self.codebook.pow(2).sum(1, keepdim=True).t()
            )                                                     # (B*N, K)
            codes = dist.argmin(dim=1)                           # (B*N,)

        # Store for inspection — use .data= to avoid re-registering the buffer
        # as persistent=True (plain assignment would re-register with default persistent=True)
        codes_reshaped = codes.reshape(B, N)
        if self._last_codes.shape != codes_reshaped.shape:
            object.__setattr__(self, "_last_codes", codes_reshaped.detach())
        else:
            self._last_codes.data = codes_reshaped.detach()

        # Quantized embedding via STE
        quant = self.codebook[codes]                             # (B*N, code_dim)
        quant_z = (quant - flat).detach() + flat                 # STE passthrough
        quant_z = quant_z.reshape(B, N, self.code_dim)

        # EMA codebook update (only during training)
        if self.training:
            self._ema_update(flat.detach(), codes)

        # Commitment loss stored as attribute; accessed by train loop
        self._commit_loss = self.commitment * F.mse_loss(z.reshape(-1, self.code_dim), quant.detach())

        # Differentiable entropy regularization: marginal entropy of soft assignments.
        # Gradient flows through flat (the encoder output) to encourage diverse code use.
        # Codebook is detached so only the encoder is incentivised to spread across codes.
        # _entropy_loss is NEGATIVE entropy; train loop adds -coef * _entropy_loss to
        # the total loss (i.e. coef * entropy, maximising entropy = minimising loss).
        if self.training:
            # Temperature = mean squared norm of flat divided by code_dim, so that
            # distances are normalised to O(1) regardless of encoder output scale.
            # This prevents gradient blow-up through the K=16384 distance matrix.
            with torch.no_grad():
                temperature = (flat.detach().pow(2).sum(-1).mean() / self.code_dim).clamp(min=0.1)
            soft_dist = (
                flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.codebook.detach().t()
                + self.codebook.detach().pow(2).sum(1, keepdim=True).t()
            )                                                     # (B*N, K)
            soft_probs = F.softmax(-soft_dist / temperature, dim=-1)  # (B*N, K)
            marginal = soft_probs.mean(0)                         # (K,)
            self._entropy_loss = (marginal * (marginal + 1e-8).log()).sum()  # = -H
        else:
            self._entropy_loss = torch.zeros(1, device=x.device)

        # Project to LM hidden dim; cast to original dtype (fp16) for the LM
        out = self.up(quant_z)                                  # (B, N, lm_dim) float32
        out = out.to(orig_dtype)
        if squeeze_batch:
            out = out.squeeze(0)        # (N_total, lm_dim)
        return out

    # ── EMA update ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, codes: torch.Tensor) -> None:
        """Update codebook via exponential moving average (no gradient needed)."""
        K = self.codebook_size
        one_hot = F.one_hot(codes, num_classes=K).float()        # (B*N, K)
        count = one_hot.sum(0)                                   # (K,)
        embed_sum = one_hot.t() @ flat                           # (K, code_dim)

        self.ema_cluster_size.mul_(self.ema_decay).add_(count, alpha=1 - self.ema_decay)
        self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

        # Laplace-smoothed counts
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + 1e-5) / (n + K * 1e-5) * n
        self.codebook.copy_(self.ema_embed_sum / smoothed.unsqueeze(1))

    # ── codebook helpers ──────────────────────────────────────────────────────

    def load_kmeans_codebook(self, path: str) -> None:
        """Load a (K, code_dim) tensor from disk and initialise the codebook."""
        data = torch.load(path, map_location="cpu")
        cb = data["codebook"] if isinstance(data, dict) else data
        assert cb.shape == (self.codebook_size, self.code_dim), \
            f"Codebook shape mismatch: expected ({self.codebook_size}, {self.code_dim}), got {tuple(cb.shape)}"
        self.codebook.copy_(cb.float())
        self.ema_embed_sum.copy_(cb.float())

    def codebook_entropy(self) -> float:
        """Approximate codebook usage entropy (nats) from EMA cluster sizes."""
        p = self.ema_cluster_size / self.ema_cluster_size.sum().clamp(min=1e-8)
        return float(-(p * (p + 1e-8).log()).sum().item())

    def codebook_active_fraction(self, threshold: float = 1.0) -> float:
        """Fraction of codebook entries with EMA cluster size > threshold."""
        return float((self.ema_cluster_size > threshold).float().mean().item())
