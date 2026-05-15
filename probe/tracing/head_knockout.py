"""Pathological-route ablation and refined intervention variants (N-039, N-040A).

Canonical intervention (D-009): pathological_route_ablation (alias: full_zero) —
zeroes the full self-attention output of decoder layer 0, disabling the
pathological early visual→prompt_last routing circuit identified in VILA-U.
Named for the mechanism it targets, not the operation it performs.
"full_zero" is retained as a deprecated alias for backwards compatibility with
existing JSONL meta files.

N-040A additions (three refined shapes):
  selective        — pre-hook on o_proj; zeros head_dim slices for listed heads.
  scalar           — post-hook on self_attn; multiplies attn output by alpha.
  selective_scalar — pre-hook on o_proj; scales listed head slices by alpha.

All four are context managers with the same __enter__/__exit__ interface and work
uniformly across all four hook managers (vilau, unitok, qwen3vl, llava) because all
use LLaMA-style self_attn with a separate o_proj Linear.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


BaselineIndex = dict[str, dict[tuple[int, int, str], dict[str, Any]]]


def _parse_heads(heads_arg: str | list[int] | None) -> list[int]:
    """Accept comma-separated string "6,7,14" or a list; return sorted int list."""
    if heads_arg is None:
        return []
    if isinstance(heads_arg, (list, tuple)):
        return sorted(int(h) for h in heads_arg)
    return sorted(int(h.strip()) for h in str(heads_arg).split(",") if h.strip())


def _get_head_dim(layer: Any) -> int:
    """Infer per-head dimension for the attention output projection.

    Works for both MHA (num_heads == num_kv_heads) and GQA (num_kv_heads <
    num_heads).  For all LLaMA-family models the o_proj input size equals
    num_attention_heads * head_dim regardless of GQA grouping.
    """
    sa = layer.self_attn
    # 1. Prefer an explicit head_dim attribute on the attention module
    for attr in ("head_dim", "attn_head_size", "attention_head_size"):
        if hasattr(sa, attr):
            return int(getattr(sa, attr))
    # 2. Infer from o_proj input size and num_attention_heads (works for GQA)
    n_heads: int | None = None
    try:
        n_heads = int(sa.config.num_attention_heads)
    except AttributeError:
        pass
    if n_heads is None:
        for attr in ("num_heads", "num_attention_heads"):
            if hasattr(sa, attr):
                n_heads = int(getattr(sa, attr))
                break
    if n_heads is None:
        n_heads = 32  # safe fallback for 7B LLaMA-family models
    return sa.o_proj.weight.shape[1] // n_heads


def _zero_attn_output(module: Any, inp: tuple[Any, ...], output: Any) -> Any:
    """Forward hook: zero the self-attention output tensor (or first element of tuple/list)."""
    if torch.is_tensor(output):
        return torch.zeros_like(output)
    if isinstance(output, tuple):
        return (torch.zeros_like(output[0]),) + output[1:]
    if isinstance(output, list):
        return [torch.zeros_like(output[0]), *output[1:]]
    raise TypeError(f"Unsupported self_attn output type: {type(output)}")


class LayerAttnKnockout:
    """Zero one decoder layer's self-attention output inside a context block (N-039)."""

    def __init__(self, hm: Any, layer_idx: int):
        self.hm = hm
        self.layer_idx = layer_idx
        self.handle: Any | None = None

    def __enter__(self) -> "LayerAttnKnockout":
        layers = self.hm._get_decoder_layers()
        if self.layer_idx < 0 or self.layer_idx >= len(layers):
            raise IndexError(f"knockout_layer={self.layer_idx} outside 0..{len(layers) - 1}")
        self.handle = layers[self.layer_idx].self_attn.register_forward_hook(
            _zero_attn_output,
            prepend=True,
        )
        return self

    def __exit__(self, *exc: object) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class WindowAttnKnockout:
    """Zero self-attention output on every layer in [layer_start, layer_end) (N-054).

    Per-model causal-window ablation: the window is chosen as the stride-4 block
    with maximum mean visual restoration score from the model's sweep JSONL, rather
    than always targeting layer 0.
    """

    def __init__(self, hm: Any, layer_start: int, layer_end: int):
        self.hm = hm
        self.layer_start = layer_start
        self.layer_end = layer_end
        self._handles: list[Any] = []

    def __enter__(self) -> "WindowAttnKnockout":
        layers = self.hm._get_decoder_layers()
        n = len(layers)
        if self.layer_start < 0 or self.layer_end > n or self.layer_start >= self.layer_end:
            raise IndexError(
                f"window [{self.layer_start},{self.layer_end}) invalid for model with {n} layers"
            )
        for l in range(self.layer_start, self.layer_end):
            h = layers[l].self_attn.register_forward_hook(_zero_attn_output, prepend=True)
            self._handles.append(h)
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


class SelectiveHeadKnockout:
    """Zero the o_proj input slices for specific attention heads (N-040A).

    Hooks o_proj with a forward_pre_hook. The o_proj input has shape
    (B, seq, n_heads * head_dim); we zero the slice for each listed head.
    This is equivalent to zeroing those heads' value-weighted attention
    contribution before the output projection — more surgical than full-layer
    zeroing and works uniformly across LLaMA-style attention (all four backends).
    """

    def __init__(self, hm: Any, layer_idx: int, head_idxs: list[int] | str):
        self.hm = hm
        self.layer_idx = layer_idx
        self.head_idxs = _parse_heads(head_idxs)
        self.handle: Any | None = None
        self._head_dim: int | None = None

    def _pre_hook(self, module: Any, inp: tuple[Any, ...]) -> tuple[Any, ...]:
        x = inp[0].clone()  # clone — never mutate a cached/aliased tensor in-place
        hd = self._head_dim
        for h in self.head_idxs:
            x[..., h * hd : (h + 1) * hd] = 0.0
        return (x,) + inp[1:]

    def __enter__(self) -> "SelectiveHeadKnockout":
        layers = self.hm._get_decoder_layers()
        if self.layer_idx < 0 or self.layer_idx >= len(layers):
            raise IndexError(f"layer_idx={self.layer_idx} outside 0..{len(layers) - 1}")
        layer = layers[self.layer_idx]
        self._head_dim = _get_head_dim(layer)
        self.handle = layer.self_attn.o_proj.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class ScalarLayerScale:
    """Multiply one decoder layer's self-attention output by a scalar alpha (N-040A).

    alpha=0.0 reproduces LayerAttnKnockout (full zeroing).
    alpha=1.0 is the identity.
    Values in (0, 1) provide soft down-weighting.
    """

    def __init__(self, hm: Any, layer_idx: int, alpha: float):
        self.hm = hm
        self.layer_idx = layer_idx
        self.alpha = float(alpha)
        self.handle: Any | None = None

    def _hook(self, module: Any, inp: tuple[Any, ...], output: Any) -> Any:
        if torch.is_tensor(output):
            return output * self.alpha
        if isinstance(output, tuple):
            return (output[0] * self.alpha,) + output[1:]
        if isinstance(output, list):
            return [output[0] * self.alpha, *output[1:]]
        raise TypeError(f"Unsupported self_attn output type: {type(output)}")

    def __enter__(self) -> "ScalarLayerScale":
        layers = self.hm._get_decoder_layers()
        if self.layer_idx < 0 or self.layer_idx >= len(layers):
            raise IndexError(f"layer_idx={self.layer_idx} outside 0..{len(layers) - 1}")
        self.handle = layers[self.layer_idx].self_attn.register_forward_hook(
            self._hook, prepend=True
        )
        return self

    def __exit__(self, *exc: object) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class SelectiveHeadScale:
    """Scale the o_proj input slices for specific heads by alpha (N-040A).

    Combines SelectiveHeadKnockout (head selection) and ScalarLayerScale (soft
    suppression). alpha=0.0 is equivalent to SelectiveHeadKnockout; alpha=1.0
    is the identity.
    """

    def __init__(self, hm: Any, layer_idx: int, head_idxs: list[int] | str, alpha: float):
        self.hm = hm
        self.layer_idx = layer_idx
        self.head_idxs = _parse_heads(head_idxs)
        self.alpha = float(alpha)
        self.handle: Any | None = None
        self._head_dim: int | None = None

    def _pre_hook(self, module: Any, inp: tuple[Any, ...]) -> tuple[Any, ...]:
        x = inp[0].clone()  # clone — never mutate a cached/aliased tensor in-place
        hd = self._head_dim
        for h in self.head_idxs:
            x[..., h * hd : (h + 1) * hd] *= self.alpha
        return (x,) + inp[1:]

    def __enter__(self) -> "SelectiveHeadScale":
        layers = self.hm._get_decoder_layers()
        if self.layer_idx < 0 or self.layer_idx >= len(layers):
            raise IndexError(f"layer_idx={self.layer_idx} outside 0..{len(layers) - 1}")
        layer = layers[self.layer_idx]
        self._head_dim = _get_head_dim(layer)
        self.handle = layer.self_attn.o_proj.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class VTITextualSteer:
    """VTI textual intervention: add a precomputed per-layer steering direction
    to the decoder block output (residual stream) during inference (N-047).

    Direction tensor shape: (n_layers, hidden), computed by
    probe/baselines/vti_calibrate.py from (clean, corrupted) prefill pairs.
    At each layer the hook normalizes the current residual, adds alpha * d_norm,
    re-normalizes, then restores the original L2 norm — preserving representation
    scale while shifting orientation toward the clean-stable direction.

    The steering formula per token is:
        x_steered = normalize(normalize(x) + alpha * d) * ||x||

    where d is the pre-normalized direction vector and alpha is the step size.
    A good operating point for VILA-U is alpha ≈ 0.05 (calibrated on 200 POPE
    records via probe/baselines/vti_calibrate.py).

    Hook fires on the decoder block output (after attn + MLP), matching the
    residual-stream semantics used by the rest of this module. Model modules
    are never mutated, so __exit__ is always a clean slate.
    """

    def __init__(self, hm: Any, direction_path: str | Path, alpha: float):
        self.hm = hm
        self.direction = torch.load(str(direction_path), map_location="cpu")
        self.alpha = float(alpha)
        self._handles: list[Any] = []

    def _make_hook(self, layer_idx: int):
        d_norm = F.normalize(self.direction[layer_idx].float(), dim=-1)
        alpha = self.alpha

        def _hook(module: Any, inp: tuple, output: Any) -> Any:
            if isinstance(output, (tuple, list)):
                x = output[0]
            else:
                x = output
            orig_dtype = x.dtype
            x32 = x.float()
            # Per-token L2 norm for re-scaling after direction injection
            norm = torch.linalg.vector_norm(x32, dim=-1, keepdim=True)
            d = d_norm.to(x.device)
            steered = (
                F.normalize(F.normalize(x32, dim=-1) + alpha * d, dim=-1) * norm
            )
            steered = steered.to(orig_dtype)
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            if isinstance(output, list):
                return [steered] + output[1:]
            return steered

        return _hook

    def __enter__(self) -> "VTITextualSteer":
        layers = self.hm._get_decoder_layers()
        if self.direction.shape[0] != len(layers):
            raise ValueError(
                f"VTI direction n_layers={self.direction.shape[0]} "
                f"!= model n_layers={len(layers)}"
            )
        for l, layer in enumerate(layers):
            h = layer.register_forward_hook(self._make_hook(l), prepend=False)
            self._handles.append(h)
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


class ProjectorOutputScale:
    """Multiply the projector output by a scalar alpha (N-057).

    Registers a forward hook on hm._get_projector() that returns output * alpha.
    alpha=1.0 is the identity; alpha<1 down-scales; alpha>1 amplifies.
    Used for the projector-norm scaling ablation: sweep alpha across {0.25, 0.5, 1.0, 1.5, 2.0}
    and measure L0 sink mass and POPE accuracy at each level.
    """

    def __init__(self, hm: Any, alpha: float):
        self.hm = hm
        self.alpha = float(alpha)
        self.handle: Any | None = None

    def _hook(self, module: Any, inp: tuple[Any, ...], output: Any) -> Any:
        if torch.is_tensor(output):
            return output * self.alpha
        if isinstance(output, tuple):
            return (output[0] * self.alpha,) + output[1:]
        if isinstance(output, list):
            return [output[0] * self.alpha, *output[1:]]
        raise TypeError(f"Unsupported projector output type: {type(output)}")

    def __enter__(self) -> "ProjectorOutputScale":
        projector = self.hm._get_projector()
        self.handle = projector.register_forward_hook(self._hook, prepend=True)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def build_intervention(
    mode: str,
    hm: Any,
    layer_idx: int,
    head_idxs: list[int] | str,
    alpha: float,
    layer_end: int | None = None,
    direction_path: str | Path | None = None,
) -> Any:
    """Factory: return the appropriate intervention context manager by mode name."""
    if mode in ("pathological_route_ablation", "full_zero"):
        return LayerAttnKnockout(hm, layer_idx)
    if mode == "window_attn_knockout":
        if layer_end is None:
            raise ValueError("window_attn_knockout requires layer_end (use --knockout_layer_end or --auto_window)")
        return WindowAttnKnockout(hm, layer_idx, layer_end)
    if mode == "selective":
        return SelectiveHeadKnockout(hm, layer_idx, head_idxs)
    if mode == "scalar":
        return ScalarLayerScale(hm, layer_idx, alpha)
    if mode == "selective_scalar":
        return SelectiveHeadScale(hm, layer_idx, head_idxs, alpha)
    if mode == "vti_textual":
        if direction_path is None:
            raise ValueError("vti_textual requires --vti_direction <path>")
        return VTITextualSteer(hm, direction_path, alpha)
    if mode == "projector_scale":
        return ProjectorOutputScale(hm, alpha)
    raise ValueError(f"Unknown intervention mode {mode!r}")


def parse_window_filter(text: str) -> set[tuple[int, int]]:
    windows: set[tuple[int, int]] = set()
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Bad window spec {item!r}; expected START-END")
        start_s, end_s = item.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        if end <= start:
            raise ValueError(f"Bad window spec {item!r}; END must be > START")
        windows.add((start, end))
    if not windows:
        raise ValueError("--window_filter selected no windows")
    return windows


def load_baseline(path: Path) -> tuple[BaselineIndex, dict[str, list[dict[str, Any]]], list[str]]:
    index: BaselineIndex = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    seen: set[str] = set()

    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            rid = row["record_id"]
            if rid not in seen:
                seen.add(rid)
                order.append(rid)
            grouped[rid].append(row)
            key = (int(row["layer_start"]), int(row["layer_end"]), str(row["token_group"]))
            index.setdefault(rid, {})[key] = row

    return index, grouped, order


def select_target_ids(
    grouped: dict[str, list[dict[str, Any]]],
    order: list[str],
    windows: set[tuple[int, int]],
    score_threshold: float,
) -> list[str]:
    selected: list[str] = []
    for rid in order:
        rows = grouped[rid]
        if not rows or rows[0].get("source") != "pope":
            continue
        from probe.tracing.filters import is_warn_row
        if any(is_warn_row(r) for r in rows):
            continue
        has_early_visual = any(
            r["token_group"] == "visual"
            and (int(r["layer_start"]), int(r["layer_end"])) in windows
            and float(r["score"]) > score_threshold
            for r in rows
        )
        if has_early_visual:
            selected.append(rid)
    return selected


def infer_baseline_n_seeds(
    grouped: dict[str, list[dict[str, Any]]],
    target_ids: list[str],
) -> int:
    vals = {
        int(row.get("n_seeds", 1))
        for rid in target_ids
        for row in grouped[rid]
    }
    if len(vals) != 1:
        raise ValueError(f"Cannot infer one baseline n_seeds value from: {sorted(vals)}")
    return vals.pop()


def load_hook_manager(args: Any) -> tuple[Any, Any]:
    """Load the appropriate hook manager based on args.backend."""
    backend = getattr(args, "backend", "vilau")
    model_path = args.model_path

    if backend == "llava_vq":
        import torch as _torch
        _llava = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        )
        if _llava not in sys.path:
            sys.path.insert(0, _llava)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.training.llava_vq_projector import VQLinearProjector
        from probe.hooks.llava_vq import LlavaVQHookManager

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, model_base=None, model_name=model_name,
            attn_implementation="eager",
        )
        vq_proj = VQLinearProjector(
            clip_dim=model.config.mm_hidden_size,
            lm_dim=model.config.hidden_size,
        )
        ckpt_path = getattr(args, "projector_ckpt", None)
        if ckpt_path and os.path.exists(ckpt_path):
            state = _torch.load(ckpt_path, map_location="cpu")
            vq_proj.load_state_dict(state["projector"])
        model.model.mm_projector = vq_proj.to(next(model.parameters()).device)
        model.eval()
        hm = LlavaVQHookManager(model, tokenizer, image_processor)
        return hm, tokenizer

    if backend == "vilau":
        vila_u_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
        )
        if vila_u_root not in sys.path:
            sys.path.insert(0, vila_u_root)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks import VilaUHookManager

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            attn_implementation="eager",
        )
        model.eval()
        hm = VilaUHookManager(model, tokenizer, image_processor)
        return hm, tokenizer

    # showo
    _showo = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "Show-o")
    )
    if _showo not in sys.path:
        sys.path.insert(0, _showo)
    vq_path = getattr(args, "vq_path", None)
    if vq_path is None:
        raise ValueError("--vq_path is required for --backend showo (e.g. showlab/magvitv2)")
    from models import Showo, MAGVITv2
    from training.prompting_utils import UniversalPrompting
    from transformers import AutoTokenizer
    from probe.hooks.showo import ShowoHookManager

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer_path = getattr(args, "tokenizer_path", None)
    _phi_tok = tokenizer_path if tokenizer_path else "microsoft/phi-1_5"
    tokenizer = AutoTokenizer.from_pretrained(_phi_tok, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
            "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=0.0,
    )
    vq_model = MAGVITv2.from_pretrained(vq_path).to(device).eval()
    vq_model.requires_grad_(False)
    model = Showo.from_pretrained(model_path).to(device).eval()
    hm = ShowoHookManager(model, tokenizer, vq_model, uni_prompting, resolution=256)
    return hm, tokenizer


def load_target_records(tokenizer: Any, target_ids: list[str], max_records: int | None) -> list[Any]:
    from probe import load_cache, resolve_answer_token_ids

    records, _, _ = load_cache()
    resolve_answer_token_ids(records, tokenizer)
    by_id = {record.id: record for record in records}
    missing = [rid for rid in target_ids if rid not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} target ids were not found in probe cache")

    selected = [by_id[rid] for rid in target_ids]
    if max_records is not None:
        selected = selected[:max_records]
    return selected


def max_early_visual(rows: list[dict[str, Any]], windows: set[tuple[int, int]]) -> float:
    vals = [
        float(r["score"])
        for r in rows
        if r["token_group"] == "visual"
        and (int(r["layer_start"]), int(r["layer_end"])) in windows
    ]
    return max(vals) if vals else float("nan")


def run_sanity_checks(
    hm: Any,
    record: Any,
    baseline_index: BaselineIndex,
    args: argparse.Namespace,
    windows: set[tuple[int, int]],
) -> dict[str, float]:
    from probe.tracing.sweep import sweep_record

    print("\nSanity checks on first target record:", record.id)
    rows = sweep_record(
        hm,
        record,
        sigma=args.sigma,
        window_size=args.window_size,
        n_seeds=args.n_seeds,
    )
    max_score_abs_diff = 0.0
    for row in rows:
        key = (row.layer_start, row.layer_end, row.token_group)
        baseline_row = baseline_index[row.record_id].get(key)
        if baseline_row is None:
            continue
        max_score_abs_diff = max(
            max_score_abs_diff,
            abs(float(row.score) - float(baseline_row["score"])),
        )

    if max_score_abs_diff > args.identity_tolerance:
        raise RuntimeError(
            "Identity check failed: max baseline score diff "
            f"{max_score_abs_diff:.6f} > tolerance {args.identity_tolerance:.6f}"
        )

    image = Image.open(record.image_path).convert("RGB")
    ref_cap = hm.run_prefill(image, record.question)
    sanity_intervention = build_intervention(
        args.mode,
        hm,
        args.knockout_layer,
        _parse_heads(args.heads),
        args.alpha,
        direction_path=args.vti_direction,
    )
    with sanity_intervention:
        ko_cap = hm.run_prefill(image, record.question)

    attn_max_abs = float(ko_cap.attn_out[args.knockout_layer].abs().max().item())
    if args.mode in ("pathological_route_ablation", "full_zero") and attn_max_abs != 0.0:
        raise RuntimeError(
            f"Knockout capture check failed: L{args.knockout_layer} attn_out "
            f"max abs is {attn_max_abs}"
        )

    prompt_last = ref_cap.token_index.prompt_last
    residual_delta = float(
        torch.linalg.vector_norm(
            ko_cap.residual[args.knockout_layer, prompt_last].float()
            - ref_cap.residual[args.knockout_layer, prompt_last].float()
        ).item()
    )
    baseline_max = max_early_visual(list(baseline_index[record.id].values()), windows)
    print(f"  identity max score diff: {max_score_abs_diff:.6f}")
    print(f"  knockout attn_out max abs: {attn_max_abs:.6f}")
    print(f"  prompt_last residual delta at L{args.knockout_layer}: {residual_delta:.6f}")
    print(f"  baseline early visual max: {baseline_max:.6f}\n")

    return {
        "identity_max_score_abs_diff": max_score_abs_diff,
        "knockout_attn_out_max_abs": attn_max_abs,
        "prompt_last_residual_delta": residual_delta,
        "baseline_early_visual_max": baseline_max,
    }


def pair_rows(knockout_rows: list[Any], baseline_index: BaselineIndex) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    for row in knockout_rows:
        key = (row.layer_start, row.layer_end, row.token_group)
        baseline_row = baseline_index.get(row.record_id, {}).get(key)
        if baseline_row is None:
            raise RuntimeError(f"Missing baseline row for {row.record_id} {key}")
        paired.append({
            "record_id": row.record_id,
            "source": row.source,
            "layer_start": row.layer_start,
            "layer_end": row.layer_end,
            "token_group": row.token_group,
            "baseline_score": float(baseline_row["score"]),
            "knockout_score": float(row.score),
            "delta": float(row.score) - float(baseline_row["score"]),
            "baseline_logit_clean": float(baseline_row["logit_clean"]),
            "knockout_logit_clean": float(row.logit_clean),
            "baseline_logit_corrupt": float(baseline_row["logit_corrupt"]),
            "knockout_logit_corrupt": float(row.logit_corrupt),
            "baseline_logit_patched": float(baseline_row["logit_patched"]),
            "knockout_logit_patched": float(row.logit_patched),
            "baseline_n_seeds": int(baseline_row.get("n_seeds", 1)),
            "knockout_n_seeds": int(row.n_seeds),
        })
    return paired


def summarize(
    paired: list[dict[str, Any]],
    windows: set[tuple[int, int]],
    score_threshold: float,
) -> tuple[dict[str, Any], str]:
    summary: dict[str, Any] = {}
    ordered_windows = sorted(windows)
    groups = sorted({row["token_group"] for row in paired})

    for layer_start, layer_end in ordered_windows:
        for group in groups:
            cell = [
                row for row in paired
                if row["layer_start"] == layer_start
                and row["layer_end"] == layer_end
                and row["token_group"] == group
            ]
            if not cell:
                continue
            stem = f"({layer_start},{layer_end})_{group}"
            baseline_vals = [row["baseline_score"] for row in cell]
            knockout_vals = [row["knockout_score"] for row in cell]
            eligible = [row for row in cell if row["baseline_score"] > score_threshold]
            summary[f"{stem}_baseline_mean"] = sum(baseline_vals) / len(baseline_vals)
            summary[f"{stem}_knockout_mean"] = sum(knockout_vals) / len(knockout_vals)
            summary[f"{stem}_delta_mean"] = (
                summary[f"{stem}_knockout_mean"] - summary[f"{stem}_baseline_mean"]
            )
            summary[f"{stem}_n_records"] = len(cell)
            summary[f"{stem}_n_baseline_gt_threshold"] = len(eligible)
            summary[f"{stem}_n_collapsed_below_1"] = sum(
                1 for row in eligible if row["knockout_score"] < score_threshold
            )
            summary[f"{stem}_n_at_or_below_1"] = sum(
                1 for row in eligible if row["knockout_score"] <= score_threshold
            )

    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        if (
            row["token_group"] == "visual"
            and (row["layer_start"], row["layer_end"]) in windows
        ):
            by_record[row["record_id"]].append(row)

    record_baseline_max: list[float] = []
    record_knockout_max: list[float] = []
    collapsed = 0
    no_longer_gt = 0
    for rows in by_record.values():
        baseline_max = max(row["baseline_score"] for row in rows)
        knockout_max = max(row["knockout_score"] for row in rows)
        if baseline_max <= score_threshold:
            continue
        record_baseline_max.append(baseline_max)
        record_knockout_max.append(knockout_max)
        if knockout_max < score_threshold:
            collapsed += 1
        if knockout_max <= score_threshold:
            no_longer_gt += 1

    n_basis = len(record_baseline_max)
    if n_basis == 0:
        raise RuntimeError("No records qualified for verdict basis")
    baseline_mean = sum(record_baseline_max) / n_basis
    knockout_mean = sum(record_knockout_max) / n_basis
    mean_drop = baseline_mean - knockout_mean
    collapse_fraction = collapsed / n_basis
    no_longer_gt_fraction = no_longer_gt / n_basis
    persists_fraction = 1.0 - no_longer_gt_fraction

    if no_longer_gt_fraction > 0.70:
        verdict = "COLLAPSE"
    elif mean_drop > 0.50 and persists_fraction > 0.30:
        verdict = "PARTIAL"
    elif mean_drop > 0.50:
        verdict = "PARTIAL"
    else:
        verdict = "PERSISTS"

    summary["verdict_basis"] = {
        "score_threshold": score_threshold,
        "n_records": n_basis,
        "baseline_early_visual_max_mean": baseline_mean,
        "knockout_early_visual_max_mean": knockout_mean,
        "mean_drop": mean_drop,
        "n_collapsed_below_threshold": collapsed,
        "n_no_longer_gt_threshold": no_longer_gt,
        "collapse_fraction": collapse_fraction,
        "no_longer_gt_fraction": no_longer_gt_fraction,
        "persists_fraction": persists_fraction,
    }
    return summary, verdict


def print_summary(output: dict[str, Any], windows: set[tuple[int, int]]) -> None:
    print("\n" + "=" * 78)
    print("VILA-U L0 ATTENTION KNOCKOUT")
    print("=" * 78)
    print(
        f"records={output['n_records']}  layer={output['knockout_layer']}  "
        f"sigma={output['sigma']}  verdict={output['verdict']}"
    )
    print(f"out={output['out']}\n")
    print(f"  {'window':>10}  {'group':>11}  {'baseline':>10}  {'knockout':>10}  {'delta':>10}")
    print("  " + "-" * 59)
    for layer_start, layer_end in sorted(windows):
        for group in ("visual", "question", "prompt_last"):
            stem = f"({layer_start},{layer_end})_{group}"
            if f"{stem}_baseline_mean" not in output["summary"]:
                continue
            baseline = output["summary"][f"{stem}_baseline_mean"]
            knockout = output["summary"][f"{stem}_knockout_mean"]
            delta = output["summary"][f"{stem}_delta_mean"]
            print(
                f"  [{layer_start:2d},{layer_end:2d})  {group:>11}  "
                f"{baseline:10.3f}  {knockout:10.3f}  {delta:10.3f}"
            )

    basis = output["summary"]["verdict_basis"]
    print("\nVerdict basis:")
    print(
        "  early visual max mean: "
        f"{basis['baseline_early_visual_max_mean']:.3f} -> "
        f"{basis['knockout_early_visual_max_mean']:.3f} "
        f"(drop {basis['mean_drop']:.3f})"
    )
    print(
        "  no longer > threshold: "
        f"{basis['n_no_longer_gt_threshold']}/{basis['n_records']} "
        f"({100 * basis['no_longer_gt_fraction']:.1f}%)"
    )
    print(
        "  strictly below threshold: "
        f"{basis['n_collapsed_below_threshold']}/{basis['n_records']} "
        f"({100 * basis['collapse_fraction']:.1f}%)"
    )
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-attention knockout / intervention over the >1.0 subset (multi-backend)"
    )
    parser.add_argument("--backend", default="vilau", choices=["vilau", "showo", "llava_vq"],
                        help="Model backend (default: vilau)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None,
                        help="Tokenizer path (showo: defaults to microsoft/phi-1_5)")
    parser.add_argument("--vq_path", default=None,
                        help="[showo only] HuggingFace path for MAGVITv2 (e.g. showlab/magvitv2)")
    parser.add_argument("--projector_ckpt", default=None,
                        help="[llava_vq only] Path to trained VQLinearProjector checkpoint")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--knockout_layer", type=int, default=0)
    parser.add_argument("--knockout_layer_end", type=int, default=None,
                        help="Exclusive end layer for window_attn_knockout mode")
    parser.add_argument(
        "--mode",
        choices=["pathological_route_ablation", "full_zero", "window_attn_knockout",
                 "selective", "scalar", "selective_scalar", "vti_textual"],
        default="pathological_route_ablation",
        help=(
            "Intervention shape: pathological_route_ablation (canonical D-009; "
            "full_zero is a deprecated alias), window_attn_knockout (N-054: "
            "zero all attn in [knockout_layer, knockout_layer_end)), selective (per-head zero), "
            "scalar (layer-wide alpha scaling), selective_scalar (per-head alpha scaling), "
            "vti_textual (N-047: per-layer residual-stream steering; requires --vti_direction)"
        ),
    )
    parser.add_argument(
        "--vti_direction",
        default=None,
        help="Path to direction tensor (n_layers, hidden) for vti_textual mode",
    )
    parser.add_argument(
        "--heads",
        default="6,7,14",
        help="Comma-separated head indices for selective/selective_scalar modes",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Scale factor for scalar/selective_scalar modes (0.0=zero, 1.0=identity)",
    )
    parser.add_argument("--baseline", default=None,
                        help="Baseline sweep JSONL (default: results/sweep_<backend>.jsonl)")
    parser.add_argument("--score_threshold", type=float, default=1.0)
    parser.add_argument("--window_filter", default="0-4,4-8,8-12")
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None,
        help="Noise seeds for POPE records (default: infer from baseline)",
    )
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: results/head_knockout_<backend>.json)")
    parser.add_argument("--skip_sanity_checks", action="store_true")
    parser.add_argument("--identity_tolerance", type=float, default=0.05)
    args = parser.parse_args()

    # Apply backend-aware defaults
    if args.baseline is None:
        args.baseline = f"results/sweep_{args.backend}.jsonl"
    if args.out is None:
        args.out = f"results/head_knockout_{args.backend}.json"

    started_at = datetime.now(timezone.utc).isoformat()
    baseline_path = Path(args.baseline)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    windows = parse_window_filter(args.window_filter)

    baseline_index, grouped, order = load_baseline(baseline_path)
    target_ids = select_target_ids(
        grouped,
        order,
        windows=windows,
        score_threshold=args.score_threshold,
    )
    if not target_ids:
        raise RuntimeError("No target POPE non-WARN >1.0 records found")
    if args.n_seeds is None:
        args.n_seeds = infer_baseline_n_seeds(grouped, target_ids)

    print(f"Baseline: {baseline_path}")
    print(
        f"Target filter: source=pope, non-WARN, visual score > "
        f"{args.score_threshold} in {args.window_filter}"
    )
    print(f"Target records: {len(target_ids)} before --max_records")
    print(f"n_seeds={args.n_seeds}")

    print(f"\nLoading {args.backend} model from {args.model_path} ...")
    hm, tokenizer = load_hook_manager(args)
    records = load_target_records(tokenizer, target_ids, args.max_records)
    print(f"Records to run: {len(records)}")
    if not records:
        raise RuntimeError("No records selected for knockout sweep")

    sanity: dict[str, float] = {}
    if not args.skip_sanity_checks:
        sanity = run_sanity_checks(hm, records[0], baseline_index, args, windows)

    from probe.tracing.sweep import sweep_record

    head_idxs = _parse_heads(args.heads)
    intervention = build_intervention(
        args.mode, hm, args.knockout_layer, head_idxs, args.alpha,
        layer_end=args.knockout_layer_end,
        direction_path=args.vti_direction,
    )

    _layer_desc = (f"layer={args.knockout_layer}"
                   if args.mode != "window_attn_knockout"
                   else f"layers=[{args.knockout_layer},{args.knockout_layer_end})")
    print(
        f"\nIntervention: mode={args.mode}, {_layer_desc}"
        + (f", heads={head_idxs}" if args.mode in ("selective", "selective_scalar") else "")
        + (f", alpha={args.alpha}" if args.mode in ("scalar", "selective_scalar") else "")
        + (f", direction={args.vti_direction}" if args.mode == "vti_textual" else "")
    )

    knockout_rows: list[Any] = []
    t_start = time.time()
    with intervention:
        for idx, record in enumerate(records, 1):
            t0 = time.time()
            rows = sweep_record(
                hm,
                record,
                sigma=args.sigma,
                window_size=args.window_size,
                n_seeds=args.n_seeds,
            )
            knockout_rows.extend(rows)
            elapsed = time.time() - t0
            baseline_max = max_early_visual(list(baseline_index[record.id].values()), windows)
            knockout_max = max_early_visual([asdict(row) for row in rows], windows)
            eta = ((time.time() - t_start) / idx) * (len(records) - idx)
            print(
                f"  [{idx:3d}/{len(records)}] {record.id:<20} "
                f"{elapsed:5.1f}s  early_visual_max "
                f"{baseline_max:6.3f} -> {knockout_max:6.3f}  "
                f"ETA {eta / 60:5.1f}m",
                flush=True,
            )

    paired = pair_rows(knockout_rows, baseline_index)
    summary, verdict = summarize(paired, windows, args.score_threshold)

    output = {
        "backend": args.backend,
        "model_path": args.model_path,
        "sigma": args.sigma,
        "knockout_layer": args.knockout_layer,
        "knockout_layer_end": args.knockout_layer_end,
        "mode": args.mode,
        "heads": head_idxs,
        "alpha": args.alpha,
        "score_threshold": args.score_threshold,
        "window_filter": args.window_filter,
        "window_size": args.window_size,
        "n_seeds": args.n_seeds,
        "baseline": str(baseline_path),
        "out": str(out_path),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "target_filter": {
            "source": "pope",
            "non_warn": True,
            "token_group": "visual",
            "score_gt": args.score_threshold,
            "windows": [list(window) for window in sorted(windows)],
        },
        "n_records_available": len(target_ids),
        "n_records": len(records),
        "selected_record_ids": [record.id for record in records],
        "sanity_checks": sanity,
        "rows": paired,
        "summary": summary,
        "verdict": verdict,
    }
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWritten -> {out_path}")
    print_summary(output, windows)


if __name__ == "__main__":
    main()
