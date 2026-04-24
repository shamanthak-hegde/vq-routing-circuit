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
    logit_corrupt: float     # mean across seeds for POPE multi-seed
    logit_patched: float     # mean across seeds for POPE multi-seed
    n_seeds:       int = 1   # number of noise seeds averaged (POPE only)


def _forward_logit(hm, embeds: Tensor, prompt_last: int, answer_id: int) -> float:
    """Minimal forward; returns scalar logit at (prompt_last, answer_id) as float."""
    bucket: list[Tensor] = []

    def _hook(module, inp, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        bucket.append(t.squeeze(0)[prompt_last, answer_id].detach().cpu())

    h = hm._get_lm_head().register_forward_hook(_hook)
    try:
        hm._get_lm_forward()(inputs_embeds=embeds, use_cache=False,
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


def _sweep_one_corrupt(
    hm,
    corrupt_emb: Tensor,
    clean_residual: Tensor,
    groups: dict[str, Tensor],
    prompt_last: int,
    answer_id: int,
    n_layers: int,
    window_size: int,
    logit_clean: float,
) -> tuple[float, dict[tuple[int, int, str], tuple[float, float]]]:
    """Run all window × group patching passes for a single corrupt_emb.

    Returns (logit_corrupt, {(l_start, l_end, group): (score, logit_patched)}).
    """
    logit_corrupt = _forward_logit(hm, corrupt_emb, prompt_last, answer_id)
    cell: dict[tuple[int, int, str], tuple[float, float]] = {}

    for l_start in range(0, n_layers, window_size):
        l_end = min(l_start + window_size, n_layers)
        for group_name, idx in groups.items():
            logit_vec = run_patched_forward(
                hm, corrupt_emb, clean_residual,
                l_start, l_end, idx, prompt_last,
            )
            lp = float(logit_vec[answer_id])
            sc = normalized_restoration(lp, logit_corrupt, logit_clean)
            cell[(l_start, l_end, group_name)] = (sc, lp)

    return logit_corrupt, cell


@torch.no_grad()
def sweep_record(
    hm: VLMHookManager,
    record: ProbeRecord,
    sigma: float,
    window_size: int = 4,
    n_seeds: int = 1,
) -> list[PatchResult]:
    """Run the full (window × token_group) sweep for a single probe record.

    Requires record.answer_token_id to be set (call resolve_answer_token_ids
    before sweeping).

    Images are loaded from record.image_path / foil_image_path on disk.

    For POPE records, n_seeds > 1 runs that many independent noise draws and
    averages the resulting scores (per-seed normalization before averaging).
    NaturalBench records ignore n_seeds (image_swap is deterministic).
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

    n_layers = len(hm._get_decoder_layers())
    groups   = _token_groups(token_index)

    # ── 2. Build corrupt embeds and sweep ─────────────────────────────────────
    if record.corruption_mode == "image_swap":
        # Deterministic — n_seeds doesn't apply
        assert record.foil_image_path is not None
        foil_img = Image.open(record.foil_image_path).convert("RGB")
        if foil_img.size != clean_img.size:
            foil_img = foil_img.resize(clean_img.size, Image.LANCZOS)
        corrupt_emb = foil_embeds(hm, foil_img, record.question)
        logit_corrupt, cells = _sweep_one_corrupt(
            hm, corrupt_emb, clean_cap.residual,
            groups, prompt_last, answer_id, n_layers, window_size, logit_clean,
        )
        effective_seeds = 1
        # Wrap in list to share averaging code below
        all_corrupt = [logit_corrupt]
        all_cells   = [cells]

    else:  # gaussian_noise — average over n_seeds independent draws
        effective_seeds = max(1, n_seeds)
        all_corrupt: list[float] = []
        all_cells: list[dict] = []
        for seed_idx in range(effective_seeds):
            corrupt_emb = noisy_embeds(
                hm, clean_img, record.question,
                visual_range=token_index.visual_range,
                sigma=sigma,
                seed=seed_idx,
            )
            lc, cells = _sweep_one_corrupt(
                hm, corrupt_emb, clean_cap.residual,
                groups, prompt_last, answer_id, n_layers, window_size, logit_clean,
            )
            all_corrupt.append(lc)
            all_cells.append(cells)

    # ── 3. Average across seeds and assemble results ───────────────────────────
    mean_logit_corrupt = sum(all_corrupt) / len(all_corrupt)
    keys = list(all_cells[0].keys())
    results: list[PatchResult] = []

    for key in keys:
        l_start, l_end, group_name = key
        scores    = [c[key][0] for c in all_cells]
        lp_values = [c[key][1] for c in all_cells]
        results.append(PatchResult(
            record_id=record.id,
            source=record.source,
            layer_start=l_start,
            layer_end=l_end,
            token_group=group_name,
            score=sum(scores) / len(scores),
            logit_clean=logit_clean,
            logit_corrupt=mean_logit_corrupt,
            logit_patched=sum(lp_values) / len(lp_values),
            n_seeds=effective_seeds,
        ))

    return results


def sweep_windows(
    hm: VLMHookManager,
    records: Sequence[ProbeRecord],
    sigma: float,
    window_size: int = 4,
    n_seeds: int = 1,
) -> list[PatchResult]:
    """Run sweep_record over every record; return concatenated results.

    records must have answer_token_id filled (call resolve_answer_token_ids first).
    """
    return [row for rec in records for row in sweep_record(hm, rec, sigma, window_size, n_seeds)]
