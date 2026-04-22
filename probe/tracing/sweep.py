"""Windowed activation-patching sweep over probe records.

Produces a flat list of PatchResult rows (one per record × window × token_group)
that can be pivoted into a heatmap downstream:

    import pandas as pd
    df = pd.DataFrame([vars(r) for r in results])
    heatmap = df.groupby(["layer_start", "token_group"])["score"].mean().unstack()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from PIL import Image

from probe.schema import ProbeRecord
from probe.hooks.base import VLMHookManager
from .corrupt import foil_embeds, noisy_embeds
from .patch import run_patched_forward
from .score import normalized_restoration


@dataclass
class PatchResult:
    record_id:     str
    source:        str       # "naturalbench" | "pope"
    layer_start:   int
    layer_end:     int
    token_group:   str       # "visual" | "question" | "prompt_last"
    score:         float
    logit_clean:   float
    logit_corrupt: float
    logit_patched: float


def _forward_logit(model, embeds: Tensor, prompt_last: int, answer_id: int) -> float:
    """Minimal forward; returns scalar logit at (prompt_last, answer_id) as float."""
    bucket: list[Tensor] = []

    def _hook(module, inp, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        bucket.append(t.squeeze(0)[prompt_last, answer_id].detach().cpu())

    h = model.lm_head.register_forward_hook(_hook)
    try:
        model(inputs_embeds=embeds, use_cache=False,
              output_attentions=False, return_dict=True)
    finally:
        h.remove()
    return float(bucket[0])


def _token_groups(token_index) -> dict[str, Tensor]:
    """Resolve the three sweep token groups to int64 index tensors."""
    groups: dict[str, Tensor] = {}
    vr = token_index.visual_range
    if vr is not None:
        groups["visual"] = torch.arange(vr[0], vr[1], dtype=torch.long)
    qr = token_index.question_range
    if qr is not None:
        groups["question"] = torch.arange(qr[0], qr[1], dtype=torch.long)
    groups["prompt_last"] = torch.tensor(
        [token_index.prompt_last], dtype=torch.long
    )
    return groups


@torch.no_grad()
def sweep_record(
    hm: VLMHookManager,
    record: ProbeRecord,
    window_size: int = 4,
    sigma: float = 0.1,
) -> list[PatchResult]:
    """Run the full (window × token_group) sweep for a single probe record.

    Requires record.answer_token_id to be set (call resolve_answer_token_ids
    before sweeping).

    Images are loaded from record.image_path / foil_image_path on disk.
    """
    assert record.answer_token_id is not None, (
        f"answer_token_id is None for {record.id}. "
        "Call resolve_answer_token_ids(records, tokenizer) first."
    )
    answer_id = record.answer_token_id

    # ── 1. Clean prefill ──────────────────────────────────────────────────────
    clean_img = Image.open(record.image_path).convert("RGB")
    clean_cap = hm.run_prefill(clean_img, record.question)

    token_index = clean_cap.token_index
    prompt_last = token_index.prompt_last
    logit_clean = float(clean_cap.logits[prompt_last, answer_id])

    # ── 2. Corrupted inputs_embeds ────────────────────────────────────────────
    if record.corruption_mode == "image_swap":
        assert record.foil_image_path is not None
        foil_img = Image.open(record.foil_image_path).convert("RGB")
        corrupt_emb = foil_embeds(hm, foil_img, record.question)
    else:  # gaussian_noise
        corrupt_emb = noisy_embeds(
            hm, clean_img, record.question,
            visual_range=token_index.visual_range,
            sigma=sigma,
        )

    # ── 3. Corrupted logit (one bare forward, no activation capture) ──────────
    logit_corrupt = _forward_logit(hm.model, corrupt_emb, prompt_last, answer_id)

    # ── 4. Sweep ──────────────────────────────────────────────────────────────
    n_layers = len(hm.model.model.layers)
    groups   = _token_groups(token_index)
    results: list[PatchResult] = []

    for l_start in range(0, n_layers, window_size):
        l_end = min(l_start + window_size, n_layers)

        for group_name, idx in groups.items():
            logit_vec = run_patched_forward(
                hm.model,
                corrupt_emb,
                clean_cap.residual,  # (n_layers, S, H) on CPU
                l_start,
                l_end,
                idx,
                prompt_last,
            )
            logit_patched = float(logit_vec[answer_id])
            score = normalized_restoration(logit_patched, logit_corrupt, logit_clean)

            results.append(PatchResult(
                record_id=record.id,
                source=record.source,
                layer_start=l_start,
                layer_end=l_end,
                token_group=group_name,
                score=score,
                logit_clean=logit_clean,
                logit_corrupt=logit_corrupt,
                logit_patched=logit_patched,
            ))

    return results


def sweep_windows(
    hm: VLMHookManager,
    records: Sequence[ProbeRecord],
    window_size: int = 4,
    sigma: float = 0.1,
) -> list[PatchResult]:
    """Run sweep_record over every record; return concatenated results.

    records must have answer_token_id filled (call resolve_answer_token_ids first).
    """
    return [row for rec in records for row in sweep_record(hm, rec, window_size, sigma)]
