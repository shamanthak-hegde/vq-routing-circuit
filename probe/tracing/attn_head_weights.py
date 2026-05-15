"""
Per-head visual-to-prompt_last attention weights (N-038, N-043).

N-038: clean-prefill measurement for VILA-U and UniTok.
N-043: extends with --include_noisy to also capture noisy-prefill attention
       (sigma-calibrated Gaussian noise on post-projector visual tokens), enabling
       a clean-vs-noisy side-by-side comparison to verify L0 routing is input-invariant.

Usage
-----
    python -m probe.tracing.attn_head_weights \
        --backend vilau \
        --model_path mit-han-lab/vila-u-7b-256 \
        --out results/attn_head_weights_vilau.json

    # N-043: include noisy prefill
    python -m probe.tracing.attn_head_weights \
        --backend vilau \
        --model_path mit-han-lab/vila-u-7b-256 \
        --include_noisy --sigma 1.0 \
        --out results/attn_head_weights_vilau_clean_noisy.json

    python -m probe.tracing.attn_head_weights \
        --backend unitok \
        --model_path FoundationVision/unitok_mllm \
        --tokenizer_path UniTok/checkpoint/unitok_tokenizer.pth \
        --out results/attn_head_weights_unitok.json

    # Show-o (conda: showo) — uses NB source because POPE WARN rate is 86%
    python -m probe.tracing.attn_head_weights \
        --backend showo \
        --model_path showlab/show-o \
        --vq_path showlab/magvitv2 \
        --source naturalbench \
        --out results/attn_head_weights_showo.json

    # Emu3 (conda: emu) — sweep is all naturalbench
    python -m probe.tracing.attn_head_weights \
        --backend emu3 \
        --model_path BAAI/Emu3-Chat \
        --vq_path BAAI/Emu3-VisionTokenizer \
        --sigma 0.01 \
        --source naturalbench \
        --n_records 20 \
        --out results/attn_head_weights_emu3.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def _default_target_window(backend: str, n_layers: int) -> tuple[int, int]:
    if backend in ("vilau", "llava_vq", "emu3"):
        return (0, min(8, n_layers))
    if backend == "showo":
        return (min(4, n_layers), min(8, n_layers))
    return (min(12, n_layers), min(14, n_layers))


def _load_target_ids(records_from: Path) -> list[str]:
    target_ids: list[str] = []
    if not records_from.exists():
        print(f"Warning: {records_from} not found; will use sweep fallback.")
        return target_ids

    seen: set[str] = set()
    with records_from.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rid = json.loads(line)["record_id"]
            except (json.JSONDecodeError, KeyError):
                continue
            if rid not in seen:
                seen.add(rid)
                target_ids.append(rid)
    print(f"Loaded {len(target_ids)} record IDs from {records_from}")
    return target_ids


def _fallback_non_warn_ids(backend: str, source: str = "pope") -> list[str]:
    sweep_path = Path(f"results/sweep_{backend}.jsonl")
    if not sweep_path.exists():
        print(f"Warning: {sweep_path} not found; will use first {source} records.")
        return []

    warn_flags: dict[str, bool] = {}
    with sweep_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if source != "all" and row.get("source") != source:
                continue
            rid = row["record_id"]
            is_warn = row["logit_clean"] <= row["logit_corrupt"]
            warn_flags[rid] = warn_flags.get(rid, False) or is_warn

    ids = [rid for rid, is_warn in warn_flags.items() if not is_warn]
    print(f"Fallback pool: {len(ids)} non-WARN {source} records from {sweep_path}")
    return ids


def _load_records(tokenizer, backend: str, records_from: Path, n_records: int,
                  source: str = "pope") -> list:
    from probe import load_cache, resolve_answer_token_ids

    records, _, _ = load_cache()
    resolve_answer_token_ids(records, tokenizer)

    src_filter = (lambda r: True) if source == "all" else (lambda r: r.source == source)

    def _filter_by_ids(ids: list) -> list:
        id_set = set(ids)
        order = {rid: i for i, rid in enumerate(ids)}
        result = [r for r in records if src_filter(r) and r.id in id_set]
        result.sort(key=lambda r: order.get(r.id, 999999))
        return result

    target_ids = _load_target_ids(records_from)
    if target_ids:
        todo = _filter_by_ids(target_ids)
        if not todo:
            print(f"Warning: records_from IDs produce no {source!r} records; "
                  f"falling back to sweep.")
            target_ids = []

    if not target_ids:
        target_ids = _fallback_non_warn_ids(backend, source=source)
        todo = _filter_by_ids(target_ids) if target_ids else [r for r in records if src_filter(r)]

    return todo[:n_records]


def _load_hook_manager(args: argparse.Namespace):
    print(f"\nLoading {args.backend} model from {args.model_path} ...")
    if args.backend == "llava_vq":
        import torch  # noqa: unitok/showo branches also import torch locally
        _llava = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        )
        if _llava not in sys.path:
            sys.path.insert(0, _llava)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.training.llava_vq_projector import VQLinearProjector
        from probe.hooks.llava_vq import LlavaVQHookManager

        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, model_base=None, model_name=model_name,
            attn_implementation="eager",
        )
        clip_dim = model.config.mm_hidden_size
        lm_dim = model.config.hidden_size
        vq_proj = VQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim)
        ckpt_path = getattr(args, "projector_ckpt", None)
        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            vq_proj.load_state_dict(state["projector"])
        model.model.mm_projector = vq_proj.to(next(model.parameters()).device)
        model.eval()
        hm = LlavaVQHookManager(model, tokenizer, image_processor,
                                 capture_attention_weights=True)
        return hm, tokenizer

    if args.backend == "vilau":
        _vilau = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
        )
        if _vilau not in sys.path:
            sys.path.insert(0, _vilau)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks import VilaUHookManager

        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path,
            attn_implementation="eager",
        )
        model.eval()
        hm = VilaUHookManager(
            model, tokenizer, image_processor, capture_attention_weights=True
        )
        return hm, tokenizer

    if args.backend == "unitok":
        _unitok = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "UniTok")
        )
        _liquid = os.path.join(_unitok, "eval", "liquid")
        for path in (_unitok, _liquid):
            if path not in sys.path:
                sys.path.insert(0, path)

        from model.builder import load_pretrained_model
        from mm_utils import get_model_name_from_path
        from models.unitok import UniTok
        from utils.config import Args as UniTokArgs
        from probe.hooks.unitok import UniTokHookManager

        if args.tokenizer_path is None:
            raise ValueError("--tokenizer_path is required for --backend unitok")

        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, _, _ = load_pretrained_model(
            model_path,
            None,
            model_name,
            attn_implementation="eager",
        )
        model.eval()

        device = next(model.parameters()).device
        ckpt = torch.load(os.path.expanduser(args.tokenizer_path), map_location="cpu")
        vae_cfg = UniTokArgs()
        vae_cfg.load_state_dict(ckpt["args"])
        vq_model = UniTok(vae_cfg)
        vq_model.load_state_dict(ckpt["trainer"]["unitok"])
        vq_model.to(device).eval()
        del ckpt

        hm = UniTokHookManager(
            model, tokenizer, vq_model, capture_attention_weights=True
        )
        return hm, tokenizer

    if args.backend == "emu3":
        import torch
        if getattr(args, "vq_path", None) is None:
            raise ValueError("--vq_path is required for --backend emu3 (e.g. BAAI/Emu3-VisionTokenizer)")
        from transformers import (AutoTokenizer, AutoModel,
                                  AutoImageProcessor, AutoModelForCausalLM)
        from probe.hooks.emu3 import Emu3HookManager
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True, padding_side="left"
        )
        image_processor = AutoImageProcessor.from_pretrained(
            args.vq_path, trust_remote_code=True
        )
        image_tokenizer = AutoModel.from_pretrained(
            args.vq_path, device_map="cuda:0", trust_remote_code=True
        ).eval()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map="cuda:0", torch_dtype=torch.bfloat16,
            attn_implementation="eager", trust_remote_code=True,
        ).eval()
        hm = Emu3HookManager(model, tokenizer, image_processor, image_tokenizer,
                             max_image_size=getattr(args, "max_image_size", 128),
                             capture_attention_weights=True)
        return hm, tokenizer

    # showo
    _showo = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "Show-o")
    )
    if _showo not in sys.path:
        sys.path.insert(0, _showo)
    if getattr(args, "vq_path", None) is None:
        raise ValueError("--vq_path is required for --backend showo (e.g. showlab/magvitv2)")

    from models import Showo, MAGVITv2
    from training.prompting_utils import UniversalPrompting
    from transformers import AutoTokenizer
    from probe.hooks.showo import ShowoHookManager

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _phi_tok = args.tokenizer_path if args.tokenizer_path else "microsoft/phi-1_5"
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
    vq_model = MAGVITv2.from_pretrained(args.vq_path).to(device).eval()
    vq_model.requires_grad_(False)
    # Attempt eager attention so output_attentions=True is supported.
    # If Showo.from_pretrained does not accept attn_implementation, the kwarg
    # is silently ignored and PhiSdpaAttention will be used instead — in that
    # case capture_attention_weights will raise at runtime with a clear message.
    try:
        model = Showo.from_pretrained(args.model_path, attn_implementation="eager").to(device).eval()
    except TypeError:
        model = Showo.from_pretrained(args.model_path).to(device).eval()
        print("Warning: Showo.from_pretrained does not accept attn_implementation='eager'. "
              "Attention-weight capture may fail if PhiSdpaAttention is active.")

    hm = ShowoHookManager(
        model, tokenizer, vq_model, uni_prompting, resolution=256,
        capture_attention_weights=True,
    )
    return hm, tokenizer


def _extract_attn_stats(cap, record_id: str, source: str, answer: str) -> dict:
    """Extract per-head visual->prompt_last attention stats from a Capture."""
    if cap.attn_weights is None:
        raise RuntimeError(
            "Attention weights were not captured. Load the model with "
            "attn_implementation='eager' and set capture_attention_weights=True."
        )

    aw = cap.attn_weights
    if aw.ndim != 4:
        raise AssertionError(f"Expected attn_weights ndim=4, got shape={tuple(aw.shape)}")
    n_layers, n_heads, seq_q, seq_k = aw.shape
    if not (8 <= n_heads <= 128):
        raise AssertionError(f"Unexpected n_heads={n_heads}; expected in [8, 128]")
    if seq_q != seq_k:
        raise AssertionError(f"Expected square attention matrix, got {seq_q}x{seq_k}")

    pl = cap.token_index.prompt_last
    vlo, vhi = cap.token_index.visual_range
    if not (0 <= pl < seq_q and 0 <= vlo < vhi <= seq_k):
        raise AssertionError(
            f"Bad prompt/visual indices: prompt_last={pl}, visual_range={(vlo, vhi)}, "
            f"attn_shape={tuple(aw.shape)}"
        )

    prompt_rows = aw[:, :, pl, :].float()
    row_sums = prompt_rows.sum(dim=-1)
    max_row_sum_err = (row_sums - 1.0).abs().max().item()
    if max_row_sum_err > 0.02:
        raise AssertionError(
            f"Attention row-sum sanity failed: max abs error={max_row_sum_err:.4f}"
        )

    head_vis = aw[:, :, pl, vlo:vhi].float()
    head_visual_sum = head_vis.sum(dim=-1)
    head_visual_max = head_vis.max(dim=-1).values
    head_visual_mean = head_vis.mean(dim=-1)
    if head_visual_sum.min().item() < -1e-5 or head_visual_sum.max().item() > 1.02:
        raise AssertionError("Visual attention mass outside expected [0, 1] range")

    return {
        "record_id": record_id,
        "source": source,
        "answer": answer,
        "prompt_last": int(pl),
        "visual_range": [int(vlo), int(vhi)],
        "n_visual": int(vhi - vlo),
        "row_sum_max_abs_err": round(max_row_sum_err, 6),
        "head_visual_sum": head_visual_sum.tolist(),
        "head_visual_max": head_visual_max.tolist(),
        "head_visual_mean": head_visual_mean.tolist(),
    }


@torch.no_grad()
def measure_record(hm, record) -> dict:
    img = Image.open(record.image_path).convert("RGB")
    cap = hm.run_prefill(img, record.question)
    return _extract_attn_stats(cap, record.id, record.source, record.answer)


@torch.no_grad()
def measure_record_noisy(hm, record, sigma: float = 1.0, seed: int = 0) -> dict:
    """Like measure_record but runs a noisy-prefill pass (Gaussian noise on visual tokens).

    Used by N-043 to confirm that L0 routing is input-invariant: the visual->prompt_last
    routing pattern should look nearly identical on clean and noisy inputs because the
    model routes whatever is present (clean or substituted VQ tokens) through the same heads.
    """
    from probe.tracing.corrupt import noisy_embeds
    from probe.hooks.utils import remove_handles

    img = Image.open(record.image_path).convert("RGB")

    # Clean prefill to get token_index (visual_range, prompt_last)
    clean_cap = hm.run_prefill(img, record.question)
    visual_range = clean_cap.token_index.visual_range

    # Build noisy inputs_embeds (skips projector — noise applied post-projector)
    noisy_inp = noisy_embeds(hm, img, record.question, visual_range, sigma=sigma, seed=seed)

    # Register ONLY self-attention weight hooks (no projector — it doesn't fire here)
    layers = hm._get_decoder_layers()
    n_layers = len(layers)
    attn_weight_store: list[list] = [[] for _ in range(n_layers)]
    handles = []

    def _make_hook(i):
        def _hook(module, inp, output):
            # output is (attn_out, attn_weights, ...) when output_attentions=True
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attn_weight_store[i].append(output[1].detach().cpu())
        return _hook

    for i, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(_make_hook(i)))

    try:
        hm._get_lm_forward()(
            inputs_embeds=noisy_inp,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    finally:
        remove_handles(handles)

    # Stack captured attention weights: (n_layers, n_heads, seq, seq)
    stacked_weights = []
    for layer_calls in attn_weight_store:
        if layer_calls:
            # Each call: (1, n_heads, seq, seq) → squeeze batch
            stacked_weights.append(torch.cat([t.squeeze(0) for t in layer_calls], dim=-2))
        else:
            stacked_weights.append(None)

    if any(w is None for w in stacked_weights):
        raise RuntimeError(
            "Some layers did not produce attention weights. "
            "Ensure the model is loaded with attn_implementation='eager'."
        )

    aw = torch.stack(stacked_weights, dim=0)  # (n_layers, n_heads, seq, seq)

    # Build a minimal duck-type object for _extract_attn_stats
    import types as _types
    noisy_cap = _types.SimpleNamespace(
        token_index=clean_cap.token_index,
        attn_weights=aw,
    )

    result = _extract_attn_stats(noisy_cap, record.id, record.source, record.answer)
    result["noisy"] = True
    result["sigma"] = sigma
    result["seed"] = seed
    return result


def _mean_matrix(per_record: list[dict], key: str) -> torch.Tensor:
    stacked = torch.tensor([r[key] for r in per_record], dtype=torch.float32)
    return stacked.mean(dim=0)


def _select_candidates(
    mean_sum: torch.Tensor,
    mean_max: torch.Tensor,
    target_window: tuple[int, int],
    top_k: int,
) -> list[dict]:
    layer_start, layer_end = target_window
    scored: list[dict] = []
    for layer in range(layer_start, layer_end):
        for head in range(mean_sum.shape[1]):
            scored.append({
                "layer": int(layer),
                "head": int(head),
                "mean_sum": round(float(mean_sum[layer, head]), 6),
                "mean_max": round(float(mean_max[layer, head]), 6),
            })
    scored.sort(key=lambda row: row["mean_sum"], reverse=True)
    for rank, row in enumerate(scored[:top_k], 1):
        row["rank"] = rank
    return scored[:top_k]


def _print_summary(output: dict) -> None:
    print("\n" + "=" * 72)
    print(f"VISUAL -> PROMPT_LAST ATTENTION HEADS - {output['backend'].upper()}")
    print("=" * 72)
    print(f"records={output['n_records']}  layers={output['n_layers']}  "
          f"heads={output['n_heads']}  target_window={output['target_window']}")
    print(f"max row-sum error={output['max_row_sum_abs_err']:.6f}\n")
    print(f"  {'rank':>4}  {'layer':>5}  {'head':>4}  {'mean_sum':>10}  {'mean_max':>10}")
    print("  " + "-" * 43)
    for row in output["candidates"]:
        print(f"  {row['rank']:>4}  {row['layer']:>5}  {row['head']:>4}  "
              f"{row['mean_sum']:>10.6f}  {row['mean_max']:>10.6f}")

    layer_mass = output["mean_layer_visual_sum"]
    top_layers = sorted(
        enumerate(layer_mass), key=lambda pair: pair[1], reverse=True
    )[:5]
    print("\nTop layers by summed visual attention mass across heads:")
    for layer, mass in top_layers:
        print(f"  L{layer:02d}: {mass:.4f}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-head visual-to-prompt_last attention weights (N-038, N-043)"
    )
    parser.add_argument("--backend", required=True, choices=["vilau", "unitok", "showo", "llava_vq", "emu3"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None,
                        help="[unitok only] Path to unitok_tokenizer.pth")
    parser.add_argument("--vq_path", default=None,
                        help="[showo/emu3] HuggingFace path for VQ tokenizer")
    parser.add_argument("--projector_ckpt", default=None,
                        help="[llava_vq only] Path to trained VQLinearProjector checkpoint")
    parser.add_argument("--source", default="pope",
                        choices=["pope", "naturalbench", "all"],
                        help="Which probe-set source to sample from (default: pope). "
                             "Use naturalbench for Show-o where POPE WARN rate is 86%%.")
    parser.add_argument("--records_from",
                        default=None,
                        help="JSONL file providing record_id values "
                             "(default: results/codebook_probe_<backend>.jsonl)")
    parser.add_argument("--n_records", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: results/attn_head_weights_<backend>.json)")
    # N-043: clean-vs-noisy comparison
    parser.add_argument(
        "--include_noisy",
        action="store_true",
        help="[N-043] Also run a noisy-prefill pass and include noisy head stats",
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0,
        help="Gaussian noise sigma for noisy pass (default 1.0 = VILA-U calibrated σ)",
    )
    parser.add_argument(
        "--projector_scale", type=float, default=None,
        help="[N-057] Scale projector output by this alpha before measuring L0 mass",
    )
    parser.add_argument(
        "--max_image_size", type=int, default=128,
        help="[emu3] Cap image dimensions before VQ encoding (default 128 → 256 vis tokens). "
             "256 → 1024 vis tokens but OOMs under eager attention on <40 GB GPU.",
    )
    args = parser.parse_args()

    if args.records_from is None:
        args.records_from = f"results/codebook_probe_{args.backend}.jsonl"

    out_path = Path(args.out or f"results/attn_head_weights_{args.backend}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hm, tokenizer = _load_hook_manager(args)
    records = _load_records(
        tokenizer,
        backend=args.backend,
        records_from=Path(args.records_from),
        n_records=args.n_records,
        source=args.source,
    )
    print(f"Records to measure: {len(records)}")
    if not records:
        raise RuntimeError("No records selected for measurement")

    proj_ctx: Any = None
    if args.projector_scale is not None:
        from probe.tracing.head_knockout import ProjectorOutputScale
        proj_ctx = ProjectorOutputScale(hm, args.projector_scale)
        proj_ctx.__enter__()
        print(f"ProjectorOutputScale active: alpha={args.projector_scale}")

    per_record_clean = []
    per_record_noisy = []
    try:
        for i, rec in enumerate(records, 1):
            print(f"  [{i:2d}/{len(records)}] {rec.id}", end="", flush=True)
            per_record_clean.append(measure_record(hm, rec))
            if args.include_noisy:
                per_record_noisy.append(measure_record_noisy(hm, rec, sigma=args.sigma))
                print(" (clean + noisy)", flush=True)
            else:
                print(flush=True)
    finally:
        if proj_ctx is not None:
            proj_ctx.__exit__(None, None, None)

    per_record = per_record_clean

    mean_sum = _mean_matrix(per_record, "head_visual_sum")
    mean_max = _mean_matrix(per_record, "head_visual_max")
    mean_mean = _mean_matrix(per_record, "head_visual_mean")
    n_layers, n_heads = mean_sum.shape
    target_window = _default_target_window(args.backend, n_layers)
    candidates = _select_candidates(mean_sum, mean_max, target_window, args.top_k)
    max_row_sum_abs_err = max(r["row_sum_max_abs_err"] for r in per_record)

    output: dict = {
        "backend": args.backend,
        "model_path": args.model_path,
        "records_from": args.records_from,
        "projector_scale": args.projector_scale,
        "n_records": len(per_record),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "target_window": list(target_window),
        "max_row_sum_abs_err": round(max_row_sum_abs_err, 6),
        "per_record": per_record,
        "mean_head_visual_sum": mean_sum.tolist(),
        "mean_head_visual_max": mean_max.tolist(),
        "mean_head_visual_mean": mean_mean.tolist(),
        "mean_layer_visual_sum": mean_sum.sum(dim=1).tolist(),
        "candidates": candidates,
    }

    if args.include_noisy and per_record_noisy:
        noisy_sum = _mean_matrix(per_record_noisy, "head_visual_sum")
        noisy_max = _mean_matrix(per_record_noisy, "head_visual_max")
        noisy_mean = _mean_matrix(per_record_noisy, "head_visual_mean")
        output["include_noisy"] = True
        output["noisy_sigma"] = args.sigma
        output["per_record_noisy"] = per_record_noisy
        output["noisy_mean_head_visual_sum"] = noisy_sum.tolist()
        output["noisy_mean_head_visual_max"] = noisy_max.tolist()
        output["noisy_mean_head_visual_mean"] = noisy_mean.tolist()
        output["noisy_mean_layer_visual_sum"] = noisy_sum.sum(dim=1).tolist()
        # Correlation between clean and noisy L0 mass as an input-invariance diagnostic
        clean_l0 = mean_sum[0]
        noisy_l0 = noisy_sum[0]
        if clean_l0.std() > 1e-6 and noisy_l0.std() > 1e-6:
            corr = torch.corrcoef(torch.stack([clean_l0, noisy_l0]))[0, 1].item()
        else:
            corr = float("nan")
        output["l0_clean_noisy_pearson_r"] = round(corr, 4)
        print(f"\nL0 clean-vs-noisy Pearson r = {corr:.4f}")

    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWritten -> {out_path}")
    _print_summary(output)


if __name__ == "__main__":
    main()
