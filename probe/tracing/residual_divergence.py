"""
Per-layer residual divergence at prompt_last: VILA-U vs UniTok (N-036).

For each probe record, runs a clean prefill and a noisy prefill (Gaussian noise
added to post-projector visual token embeddings at σ_cal), then measures how much
the residual stream at the prompt_last position has diverged from clean at every
layer.  Divergence growing across layers = LM amplifies the substituted signal;
divergence flat or shrinking = LM attenuates it.

Hypothesis (Stage-2):
  VILA-U amplifies the substituted visual embedding in early layers → visual-dominant
  POPE circuit and >1.0 restoration.  UniTok attenuates it → prompt_last-dominant
  circuit despite identical Stage-1 (off-manifold) substitution.

Usage
-----
    # VILA-U (conda: vila-u)
    python -m probe.tracing.residual_divergence \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --sigma 1.0 \\
        --out results/residual_divergence_vilau.json

    # UniTok (conda: unitok)
    python -m probe.tracing.residual_divergence \\
        --backend unitok \\
        --model_path FoundationVision/unitok_mllm \\
        --tokenizer_path /home/shegde23/VLM_Attention_Psych/UniTok/checkpoint/unitok_tokenizer.pth \\
        --sigma 0.2 \\
        --out results/residual_divergence_unitok.json

    # Figure (any env with matplotlib)
    python -m probe.tracing.residual_divergence_figure
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

from probe.tracing.corrupt import noisy_embeds


# ── residual capture ──────────────────────────────────────────────────────────

@torch.no_grad()
def _residual_from_embeds(hm, inputs_embeds: torch.Tensor) -> torch.Tensor:
    """Run LM forward from pre-built inputs_embeds; return residual on CPU.

    Cannot use register_captures/finalize_store here because finalize_store
    asserts exactly one projector call — which won't fire when passing
    inputs_embeds directly.  Instead, register decoder-layer hooks inline.

    Returns
    -------
    torch.Tensor of shape (n_layers, seq_len, hidden).
    """
    n_layers = len(hm._get_decoder_layers())
    bufs: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    handles = []
    for i, layer in enumerate(hm._get_decoder_layers()):
        def _hook(mod, inp, out, _i=i):
            t = out[0] if isinstance(out, (tuple, list)) else out
            bufs[_i].append(t.squeeze(0).detach().cpu())  # (seq_len, H)
        handles.append(layer.register_forward_hook(_hook))
    try:
        hm._get_lm_forward()(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()
    return torch.stack([calls[0] for calls in bufs], dim=0)  # (n_layers, seq_len, H)


# ── per-record measurement ────────────────────────────────────────────────────

@torch.no_grad()
def measure_record(hm, record, sigma: float, seed: int = 0) -> dict:
    """Compute per-layer residual divergence at prompt_last for one record.

    Returns
    -------
    dict with keys:
      record_id  : str
      prompt_last: int
      abs_div    : list[float]  — ‖clean[L,pl] - noisy[L,pl]‖₂ per layer
      rel_div    : list[float]  — abs_div[L] / ‖clean[L,pl]‖₂ per layer
      vis_noise_rms : float     — RMS of noise added to visual positions
    """
    img = Image.open(record.image_path).convert("RGB")

    # ── clean prefill ─────────────────────────────────────────────────────────
    clean_cap = hm.run_prefill(img, record.question)
    pl = clean_cap.token_index.prompt_last
    vr = clean_cap.token_index.visual_range
    n_layers = clean_cap.residual.shape[0]

    # ── noisy forward ─────────────────────────────────────────────────────────
    noisy_emb = noisy_embeds(hm, img, record.question,
                              visual_range=vr, sigma=sigma, seed=seed)
    noisy_res = _residual_from_embeds(hm, noisy_emb)  # (n_layers, seq_len, H)

    # ── divergence at prompt_last per layer ───────────────────────────────────
    abs_div, rel_div = [], []
    for L in range(n_layers):
        cv = clean_cap.residual[L, pl, :].float()
        nv = noisy_res[L, pl, :].float()
        ad = (cv - nv).norm().item()
        rd = ad / cv.norm().clamp(min=1e-8).item()
        abs_div.append(round(ad, 6))
        rel_div.append(round(rd, 6))

    # ── magnitude of noise injected at visual positions (for normalization) ──
    clean_vis = clean_cap.visual_embeds.float()                    # (n_vis, H) on CPU
    if vr is not None:
        noisy_vis = noisy_emb[0, vr[0]:vr[1], :].float().cpu()    # (n_vis, H)
        noise_vec = noisy_vis - clean_vis
        vis_noise_rms = float(noise_vec.norm() / (noise_vec.numel() ** 0.5))
    else:
        vis_noise_rms = float("nan")

    return {
        "record_id":    record.id,
        "prompt_last":  int(pl),
        "abs_div":      abs_div,
        "rel_div":      rel_div,
        "vis_noise_rms": round(vis_noise_rms, 6),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-layer residual divergence at prompt_last (N-036)"
    )
    parser.add_argument("--backend", required=True,
                        choices=["vilau", "unitok"],
                        help="Model backend to run")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None,
                        help="[unitok only] Path to unitok_tokenizer.pth")
    parser.add_argument("--sigma", type=float, required=True,
                        help="Post-projector noise σ (use σ_cal for each backend)")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: results/residual_divergence_<backend>.json)")
    parser.add_argument("--records_from",
                        default="results/codebook_probe_unitok.jsonl",
                        help="JSONL file from which to extract record_ids to measure "
                             "(default: codebook_probe_unitok.jsonl — the 20 Sub-test B records)")
    parser.add_argument("--n_records", type=int, default=None,
                        help="Max records to process (default: all from --records_from)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Noise RNG seed (default: 0)")
    args = parser.parse_args()

    out_path = Path(args.out or f"results/residual_divergence_{args.backend}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load target record IDs ────────────────────────────────────────────────
    target_ids: list[str] = []
    records_from = Path(args.records_from)
    if records_from.exists():
        seen = set()
        with open(records_from) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rid = json.loads(line)["record_id"]
                        if rid not in seen:
                            seen.add(rid)
                            target_ids.append(rid)
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"Loaded {len(target_ids)} record IDs from {records_from}")
    else:
        print(f"Warning: {records_from} not found — will use first n_records non-WARN POPE records")

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.backend} model from {args.model_path} ...")
    if args.backend == "vilau":
        _vilau = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
        )
        if _vilau not in sys.path:
            sys.path.insert(0, _vilau)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks import VilaUHookManager
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, attn_implementation="eager",
        )
        model.eval()
        hm = VilaUHookManager(model, tokenizer, image_processor)

    else:  # unitok
        _unitok = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "UniTok")
        )
        _liquid = os.path.join(_unitok, "eval", "liquid")
        for _p in (_unitok, _liquid):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from model.builder import load_pretrained_model
        from mm_utils import get_model_name_from_path
        from models.unitok import UniTok
        from utils.config import Args as UniTokArgs
        from probe.hooks.unitok import UniTokHookManager

        model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
        tokenizer, model, _, _ = load_pretrained_model(
            os.path.expanduser(args.model_path), None, model_name,
            attn_implementation="eager",
        )
        model.eval()
        device = next(model.parameters()).device
        if args.tokenizer_path is None:
            parser.error("--tokenizer_path is required for --backend unitok")
        ckpt = torch.load(os.path.expanduser(args.tokenizer_path), map_location="cpu")
        vae_cfg = UniTokArgs()
        vae_cfg.load_state_dict(ckpt["args"])
        vq_model = UniTok(vae_cfg)
        vq_model.load_state_dict(ckpt["trainer"]["unitok"])
        vq_model.to(device).eval()
        del ckpt
        hm = UniTokHookManager(model, tokenizer, vq_model)

    # ── load probe records ────────────────────────────────────────────────────
    from probe import load_cache, resolve_answer_token_ids
    records, _, _ = load_cache()
    resolve_answer_token_ids(records, tokenizer)

    if target_ids:
        id_set = set(target_ids)
        todo = [r for r in records if r.id in id_set and r.source == "pope"]
        # Preserve order from target_ids
        order = {rid: i for i, rid in enumerate(target_ids)}
        todo.sort(key=lambda r: order.get(r.id, 9999))
    else:
        # Fallback: first n non-WARN POPE records from sweep
        sweep_path = Path(f"results/sweep_{args.backend}.jsonl")
        non_warn: set[str] = set()
        if sweep_path.exists():
            warn_flags: dict[str, bool] = {}
            with open(sweep_path) as f:
                for line in f:
                    row = json.loads(line.strip())
                    if row.get("source") != "pope":
                        continue
                    rid = row["record_id"]
                    is_warn = row["logit_clean"] <= row["logit_corrupt"]
                    warn_flags[rid] = warn_flags.get(rid, False) or is_warn
            non_warn = {rid for rid, w in warn_flags.items() if not w}
        todo = [r for r in records if r.source == "pope" and r.id in non_warn]

    if args.n_records is not None:
        todo = todo[:args.n_records]

    print(f"Records to measure: {len(todo)}")
    print(f"sigma={args.sigma}  seed={args.seed}  out={out_path}\n")

    # ── measure ───────────────────────────────────────────────────────────────
    per_record = []
    for i, rec in enumerate(todo, 1):
        print(f"  [{i:2d}/{len(todo)}] {rec.id}", end="  ", flush=True)
        result = measure_record(hm, rec, sigma=args.sigma, seed=args.seed)
        per_record.append(result)
        # Print early-layer abs_div summary
        early = result["abs_div"][:8]
        late  = result["abs_div"][24:]
        print(f"abs_div [0,8) mean={sum(early)/len(early):.2f}  "
              f"[24,32) mean={sum(late)/len(late):.2f}  "
              f"noise_rms={result['vis_noise_rms']:.3f}")

    if not per_record:
        print("No records measured — check --records_from path.")
        return

    # ── aggregate ─────────────────────────────────────────────────────────────
    n_layers = len(per_record[0]["abs_div"])
    mean_abs = [
        sum(r["abs_div"][L] for r in per_record) / len(per_record)
        for L in range(n_layers)
    ]
    mean_rel = [
        sum(r["rel_div"][L] for r in per_record) / len(per_record)
        for L in range(n_layers)
    ]
    mean_noise_rms = sum(r["vis_noise_rms"] for r in per_record) / len(per_record)

    # Divergence normalized by layer-0 value (shows amplification vs attenuation)
    base = mean_abs[0] if mean_abs[0] > 0 else 1.0
    norm_abs = [round(v / base, 6) for v in mean_abs]

    output = {
        "backend":        args.backend,
        "model_path":     args.model_path,
        "sigma":          args.sigma,
        "seed":           args.seed,
        "n_records":      len(per_record),
        "n_layers":       n_layers,
        "mean_noise_rms": round(mean_noise_rms, 6),
        "mean_abs_div":   [round(v, 6) for v in mean_abs],
        "mean_rel_div":   [round(v, 6) for v in mean_rel],
        "norm_abs_div":   norm_abs,
        "per_record":     per_record,
    }

    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWritten → {out_path}")

    # ── print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"RESIDUAL DIVERGENCE AT PROMPT_LAST — {args.backend.upper()}")
    print(f"sigma={args.sigma}  n_records={len(per_record)}  n_layers={n_layers}")
    print("=" * 64)
    print(f"  {'layer':>6}  {'abs_div':>10}  {'rel_div':>10}  {'norm_abs':>10}")
    print(f"  {'------':>6}  {'-------':>10}  {'-------':>10}  {'--------':>10}")
    for L in range(n_layers):
        marker = " ◄" if L < 8 else ""
        print(f"  {L:6d}  {mean_abs[L]:10.4f}  {mean_rel[L]:10.4f}  {norm_abs[L]:10.4f}{marker}")
    print("=" * 64)
    early_norm = norm_abs[:8]
    late_norm  = norm_abs[24:]
    print(f"\nNorm-abs mean [0,8):  {sum(early_norm)/len(early_norm):.4f}")
    print(f"Norm-abs mean [24,32): {sum(late_norm)/len(late_norm):.4f}")
    verdict = "AMPLIFICATION" if sum(early_norm)/len(early_norm) > 1.05 else \
              "ATTENUATION"   if sum(early_norm)/len(early_norm) < 0.95 else "FLAT"
    print(f"Early-layer verdict:  {verdict}")


if __name__ == "__main__":
    main()
