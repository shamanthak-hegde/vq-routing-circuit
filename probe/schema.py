"""
Unified ProbeRecord schema for the VLM paired micro-benchmark.

Corruption modes
----------------
image_swap (NaturalBench)
    x and x' are semantically distinct real images. The model sees the same
    question q against both; the ground-truth answer flips.  Foil is stored
    as a real image path.

gaussian_noise (POPE)
    x' is x with Gaussian noise injected at the *visual-token* level (i.e.
    after the patch-embedding projection, before any attention layer). No
    altered image file is saved on disk. foil_image_path is None.
    Noise is applied at inference time; callers must respect this contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


CorruptionMode = Literal["image_swap", "gaussian_noise"]


@dataclass
class ProbeRecord:
    # ── identity ──────────────────────────────────────────────────────────────
    id: str
    source: Literal["naturalbench", "pope"]
    # groups records that share the same image context / paired structure
    pair_id: str

    # ── visual inputs ─────────────────────────────────────────────────────────
    # absolute path to the clean image
    image_path: str
    # absolute path to the foil image, or None when the foil is synthetic
    # (gaussian_noise mode: noise is applied at the visual-token level at
    # inference time; the foil is not a real image file)
    foil_image_path: Optional[str]

    # ── language ──────────────────────────────────────────────────────────────
    question: str
    answer: Literal["yes", "no"]

    # filled by resolve_answer_token_ids() once a tokenizer is available;
    # left as None when the record is first constructed
    answer_token_id: Optional[int]

    # ── patching metadata ─────────────────────────────────────────────────────
    corruption_mode: CorruptionMode

    # source-specific extra fields (nb_index, category, …)
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_answer_token_ids(
    records: list[ProbeRecord],
    tokenizer,
    yes_str: str = "yes",
    no_str: str = "no",
) -> None:
    """Fill answer_token_id in-place from *tokenizer*.

    Expects the tokenizer to map "yes"/"no" to a single token.  Raises if
    either maps to more than one token (add_special_tokens=False is used).
    """
    for s in (yes_str, no_str):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"'{s}' encodes to {len(ids)} tokens with this tokenizer. "
                "Pick a tokenizer where yes/no are single tokens."
            )
    yes_id = tokenizer.encode(yes_str, add_special_tokens=False)[0]
    no_id  = tokenizer.encode(no_str,  add_special_tokens=False)[0]

    for r in records:
        r.answer_token_id = yes_id if r.answer == "yes" else no_id
