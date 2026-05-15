"""Canonical record-level filter helpers for sweep and tracing code.

A WARN record is one where the model prefers the corrupt image over the clean
image: logit_clean <= logit_corrupt.  WARN records are excluded from causal
sweep score interpretation because the restoration score denominator
(logit_clean - logit_corrupt) would be ≤ 0, making the normalized score
meaningless or negative-direction.
"""

from __future__ import annotations


def is_warn_row(row: dict) -> bool:
    """Return True if `row` is a WARN record (clean logit ≤ corrupt logit)."""
    return float(row["logit_clean"]) <= float(row["logit_corrupt"])


def filter_non_warn(rows: list[dict]) -> list[dict]:
    """Return only non-WARN rows from a sweep JSONL record list."""
    return [r for r in rows if not is_warn_row(r)]


def partition_warn(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (non_warn_rows, warn_rows) partitions."""
    non_warn = [r for r in rows if not is_warn_row(r)]
    warn = [r for r in rows if is_warn_row(r)]
    return non_warn, warn
