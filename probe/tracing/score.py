"""Normalized restoration score for activation-patching experiments."""

from __future__ import annotations


def normalized_restoration(
    logit_patched: float,
    logit_corrupt: float,
    logit_clean: float,
) -> float:
    """(logit_patched - logit_corrupt) / (logit_clean - logit_corrupt).

    Returns 0.0 when the denominator is near-zero (clean ≈ corrupt),
    meaning there is nothing to restore. No clamping — raw ratio.
    """
    denom = logit_clean - logit_corrupt
    if abs(denom) < 1e-6:
        return 0.0
    return float((logit_patched - logit_corrupt) / denom)
