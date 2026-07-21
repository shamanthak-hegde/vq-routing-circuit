"""
Clean-run accuracy sweep: prefill only, no patching.

Records the top-1 prediction (yes vs no) at the last prompt position, and
cross-tabulates with WARN flags from an existing sweep JSONL so you can see
whether the usable (non-WARN) slice is also the correct-answer slice.

Usage
-----
    source activate sae
    python -m probe.tracing.run_accuracy \\
        --model_path liuhaotian/llava-v1.6-vicuna-7b \\
        --sweep_results results/sweep_llava.jsonl \\
        --out results/accuracy_llava.jsonl

    conda activate vila
    python -m probe.tracing.run_accuracy \\
        --backend vila \\
        --model_path Efficient-Large-Model/VILA-7b \\
        --sweep_results results/sweep_vila.jsonl \\
        --out results/accuracy_vila.jsonl

    conda activate vila-u
    python -m probe.tracing.run_accuracy \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --sweep_results results/sweep_vilau.jsonl \\
        --out results/accuracy_vilau.jsonl

Flags
-----
    --report_only     Skip the forward passes; just print the report from
                      an existing --out file (requires --sweep_results too).
    --source          all | naturalbench | pope
    --limit           cap record count for quick smoke tests
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_warn_flags(sweep_path: str | None) -> dict[str, bool]:
    """Return {record_id: is_warn} using the first row seen per record."""
    if not sweep_path or not Path(sweep_path).exists():
        return {}
    flags: dict[str, bool] = {}
    with open(sweep_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rid = row["record_id"]
                if rid not in flags:
                    flags[rid] = row["logit_clean"] <= row["logit_corrupt"]
            except (json.JSONDecodeError, KeyError):
                pass
    return flags


def _load_done_ids(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["record_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def _load_rows(out_path: Path) -> list[dict]:
    rows = []
    if not out_path.exists():
        return rows
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


# ── report ───────────────────────────────────────────────────────────────────

def _acc(subset: list[dict]) -> tuple[int, int, float]:
    if not subset:
        return 0, 0, float("nan")
    n_c = sum(1 for r in subset if r["correct"])
    n = len(subset)
    return n_c, n, 100.0 * n_c / n


def print_report(rows: list[dict], warn_flags: dict[str, bool]) -> None:
    sep = "=" * 68

    print(f"\n{sep}")
    print("CLEAN-RUN ACCURACY REPORT")
    print(sep)

    for src in ("naturalbench", "pope", "all"):
        subset = rows if src == "all" else [r for r in rows if r["source"] == src]
        if not subset:
            continue
        label = "ALL" if src == "all" else src.upper()
        n_c, n, pct = _acc(subset)
        print(f"\n{label}  (n={n})")
        print(f"  Overall accuracy : {n_c}/{n} = {pct:.1f}%")

        for ans in ("yes", "no"):
            s = [r for r in subset if r["gt_answer"] == ans]
            nc, nn, p = _acc(s)
            print(f"  answer={ans}       : {nc}/{nn} = {p:.1f}%")

        # WARN cross-tabulation
        if warn_flags:
            in_sweep    = [r for r in subset if r["record_id"] in warn_flags]
            not_in_sweep = [r for r in subset if r["record_id"] not in warn_flags]
            nowarn = [r for r in in_sweep if not warn_flags[r["record_id"]]]
            warn   = [r for r in in_sweep if     warn_flags[r["record_id"]]]

            if in_sweep:
                pct_warn = 100.0 * len(warn) / len(in_sweep)
                print(f"  WARN rate (sweep): {len(warn)}/{len(in_sweep)} = {pct_warn:.1f}%")
                nc, nn, p = _acc(nowarn)
                print(f"  non-WARN accuracy: {nc}/{nn} = {p:.1f}%  ← usable slice")
                nc, nn, p = _acc(warn)
                print(f"  WARN accuracy    : {nc}/{nn} = {p:.1f}%  ← excluded slice")
            if not_in_sweep:
                nc, nn, p = _acc(not_in_sweep)
                print(f"  (no sweep data)  : {nc}/{nn} = {p:.1f}%")

    # NB pair-level consistency
    nb = [r for r in rows if r["source"] == "naturalbench"]
    if nb:
        by_pair: dict[str, list] = {}
        for r in nb:
            by_pair.setdefault(r.get("pair_id", r["record_id"]), []).append(r)
        n_pairs = len(by_pair)
        both_correct = sum(
            1 for v in by_pair.values()
            if len(v) == 2 and all(x["correct"] for x in v)
        )
        both_wrong = sum(
            1 for v in by_pair.values()
            if len(v) == 2 and not any(x["correct"] for x in v)
        )
        print(f"\nNB PAIR CONSISTENCY  (n_pairs={n_pairs})")
        print(f"  Both correct : {both_correct}/{n_pairs} = {100*both_correct/n_pairs:.1f}%")
        print(f"  Both wrong   : {both_wrong}/{n_pairs} = {100*both_wrong/n_pairs:.1f}%")

    # Accuracy × WARN 2×2 table for NB (the core diagnostic)
    nb_in_sweep = [r for r in rows
                   if r["source"] == "naturalbench" and r["record_id"] in warn_flags]
    if nb_in_sweep:
        correct_nowarn = sum(1 for r in nb_in_sweep if r["correct"] and not warn_flags[r["record_id"]])
        wrong_nowarn   = sum(1 for r in nb_in_sweep if not r["correct"] and not warn_flags[r["record_id"]])
        correct_warn   = sum(1 for r in nb_in_sweep if r["correct"] and warn_flags[r["record_id"]])
        wrong_warn     = sum(1 for r in nb_in_sweep if not r["correct"] and warn_flags[r["record_id"]])
        print(f"\nNB ACCURACY × WARN  (2×2 table)")
        print(f"  {'':12s}  {'correct':>8}  {'wrong':>8}  {'total':>8}")
        nw = correct_nowarn + wrong_nowarn
        w  = correct_warn   + wrong_warn
        print(f"  {'non-WARN':12s}  {correct_nowarn:>8}  {wrong_nowarn:>8}  {nw:>8}")
        print(f"  {'WARN':12s}  {correct_warn:>8}  {wrong_warn:>8}  {w:>8}")

    print(f"\n{sep}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean-run accuracy sweep")
    parser.add_argument("--backend", default="llava",
                        choices=["llava", "llava_vq", "vilau", "vila", "unitok", "qwen3vl", "lavit"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--sweep_results", default=None,
                        help="Existing sweep JSONL for WARN cross-tabulation")
    parser.add_argument("--out", default="results/accuracy.jsonl",
                        help="Output JSONL (appended; safe to resume)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", default="all",
                        choices=["all", "naturalbench", "pope"])
    parser.add_argument("--report_only", action="store_true",
                        help="Skip forward passes; just print report from --out")
    #/C: optional intervention applied around every run_prefill call
    parser.add_argument(
        "--knockout_mode",
        choices=[
            "pathological_route_ablation", "full_zero",
            "selective", "scalar", "selective_scalar",
            "window_attn_knockout",
            "vti_textual",
            "projector_scale",
        ],
        default=None,
        help="If set, apply intervention from head_knockout.py (full_zero is deprecated alias); "
             "vti_textual requires --vti_direction; "
             "projector_scale multiplies projector output by --alpha",
    )
    parser.add_argument("--knockout_layer", type=int, default=0)
    parser.add_argument(
        "--knockout_layer_end", type=int, default=None,
        help="End layer (exclusive) for window_attn_knockout. Auto-set when --auto_window is used.",
    )
    parser.add_argument(
        "--auto_window", action="store_true",
        help="Read best visual window from results/sweep_<backend>.jsonl and use window_attn_knockout",
    )
    parser.add_argument(
        "--heads", default="6,7,14",
        help="Comma-separated head indices (selective / selective_scalar modes)",
    )
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Scale factor (scalar / selective_scalar / vti_textual modes)")
    parser.add_argument(
        "--vti_direction", default=None,
        help="[vti_textual] path to direction tensor (n_layers, hidden) from vti_calibrate.py",
    )
    parser.add_argument("--tokenizer_path", default=None,
                        help="[unitok only] path to unitok_tokenizer.pth")
    parser.add_argument("--projector_ckpt", default=None,
                        help="[llava_vq only] path to VQLinearProjector checkpoint")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    warn_flags = _load_warn_flags(args.sweep_results)
    n_sw = len(warn_flags)
    if n_sw:
        n_warn = sum(1 for v in warn_flags.values() if v)
        print(f"Loaded {n_sw} WARN flags ({n_warn} WARN = {100*n_warn/n_sw:.1f}%)"
              f" from {args.sweep_results}")
    else:
        print("No sweep results loaded — WARN cross-tabulation will be skipped.")

    if args.report_only:
        rows = _load_rows(out_path)
        if not rows:
            print(f"No rows found in {out_path}.")
            return
        print_report(rows, warn_flags)
        return

    if not args.model_path:
        parser.error("--model_path is required unless --report_only is set")

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading {args.backend} model from {args.model_path} ...")
    if args.backend == "llava":
        _llava = os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        if _llava not in sys.path:
            sys.path.insert(0, _llava)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.hooks import LlavaHookManager
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, model_base=None, model_name=model_name,
            attn_implementation="sdpa",
        )
        model.eval()
        hm = LlavaHookManager(model, tokenizer, image_processor)
    elif args.backend == "llava_vq":
        import torch as _torch
        _llava = os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        if _llava not in sys.path:
            sys.path.insert(0, _llava)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.training.llava_vq_projector import VQLinearProjector
        from probe.hooks.llava_vq import LlavaVQHookManager
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, model_base=None, model_name=model_name,
            attn_implementation="sdpa",
        )
        clip_dim = model.config.mm_hidden_size
        lm_dim = model.config.hidden_size
        vq_proj = VQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim)
        ckpt_path = getattr(args, "projector_ckpt", None)
        if ckpt_path and os.path.exists(ckpt_path):
            state = _torch.load(ckpt_path, map_location="cpu")
            vq_proj.load_state_dict(state["projector"])
        model.model.mm_projector = vq_proj.to(next(model.parameters()).device)
        model.eval()
        hm = LlavaVQHookManager(model, tokenizer, image_processor)
    elif args.backend == "vilau":
        _vilau = os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
        if _vilau not in sys.path:
            sys.path.insert(0, _vilau)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks import VilaUHookManager
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, attn_implementation="eager",
        )
        model.eval()
        hm = VilaUHookManager(model, tokenizer, image_processor)
    elif args.backend == "unitok":
        import torch
        _unitok = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "UniTok"))
        _liquid = os.path.join(_unitok, "eval", "liquid")
        for p in (_unitok, _liquid):
            if p not in sys.path:
                sys.path.insert(0, p)
        from model.builder import load_pretrained_model as _unitok_load
        from mm_utils import get_model_name_from_path as _unitok_name
        from models.unitok import UniTok
        from utils.config import Args as UniTokArgs
        from probe.hooks.unitok import UniTokHookManager
        if not args.tokenizer_path:
            parser.error("--tokenizer_path is required for --backend unitok")
        model_name = _unitok_name(args.model_path)
        tokenizer, model, _, _ = _unitok_load(
            args.model_path, None, model_name, attn_implementation="eager",
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
        hm = UniTokHookManager(model, tokenizer, vq_model)
    elif args.backend == "qwen3vl":
        from probe.hooks.qwen3vl import Qwen3VLHookManager
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True, padding_side="left", use_fast=True
        )
        processor.image_processor.max_pixels = 720 * 1280
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation="eager",
            device_map="auto",
        )
        model.eval()
        hm = Qwen3VLHookManager(model, processor)
        tokenizer = processor.tokenizer
    elif args.backend == "lavit":
        _lavit = os.path.join(os.path.dirname(__file__), "..", "..", "LaVIT")
        if _lavit not in sys.path:
            sys.path.insert(0, _lavit)
        from models import build_model
        from probe.hooks.lavit import LavitHookManager
        model = build_model(
            model_path=args.model_path,
            model_dtype="bf16",
            device_id=0,
            use_xformers=False,
            understanding=True,
            local_files_only=True,
        )
        model = model.to("cuda")
        model.eval()
        hm = LavitHookManager(model)
        tokenizer = model.llama_tokenizer
    else:  # vila
        _vila = os.path.join(os.path.dirname(__file__), "..", "..", "VILA")
        if _vila not in sys.path:
            sys.path.insert(0, _vila)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.hooks.vila import VilaHookManager
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, None, model_name,
        )
        model.eval()
        hm = VilaHookManager(model, tokenizer, image_processor)

    # ── build optional intervention ───────────────────────────────────────────
    if args.auto_window:
        from probe.tracing.best_window import best_visual_window
        from pathlib import Path as _Path
        _sweep_path = _Path(f"results/sweep_{args.backend}.jsonl")
        if not _sweep_path.exists():
            raise FileNotFoundError(
                f"--auto_window requires {_sweep_path}; run the sweep for {args.backend} first"
            )
        _l_start, _l_end, _score = best_visual_window(_sweep_path)
        print(f"Auto-window ({args.backend}): [{_l_start},{_l_end}) mean_visual_score={_score:.4f}")
        args.knockout_layer = _l_start
        args.knockout_layer_end = _l_end
        if args.knockout_mode is None:
            args.knockout_mode = "window_attn_knockout"

    intervention = None
    if args.knockout_mode is not None:
        from probe.tracing.head_knockout import build_intervention, _parse_heads
        intervention = build_intervention(
            args.knockout_mode,
            hm,
            args.knockout_layer,
            _parse_heads(args.heads),
            args.alpha,
            layer_end=args.knockout_layer_end,
            direction_path=args.vti_direction,
        )
        if args.knockout_mode == "window_attn_knockout":
            print(
                f"Intervention: mode={args.knockout_mode},"
                f" window=[{args.knockout_layer},{args.knockout_layer_end})"
            )
        elif args.knockout_mode == "vti_textual":
            print(
                f"Intervention: mode={args.knockout_mode},"
                f" alpha={args.alpha}, direction={args.vti_direction}"
            )
        elif args.knockout_mode == "projector_scale":
            print(f"Intervention: mode={args.knockout_mode}, alpha={args.alpha}")
        else:
            print(
                f"Intervention: mode={args.knockout_mode}, layer={args.knockout_layer}"
                + (f", heads={args.heads}" if args.knockout_mode in ("selective", "selective_scalar") else "")
                + (f", alpha={args.alpha}" if args.knockout_mode in ("scalar", "selective_scalar") else "")
            )

    # ── load probe records ────────────────────────────────────────────────────
    import torch
    from PIL import Image
    from probe import load_cache, resolve_answer_token_ids

    records, _, _ = load_cache()
    resolve_answer_token_ids(records, tokenizer)

    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("no",  add_special_tokens=False)[0]

    if args.source != "all":
        records = [r for r in records if r.source == args.source]
    if args.limit:
        records = records[:args.limit]

    done_ids = _load_done_ids(out_path)
    todo = [r for r in records if r.id not in done_ids]
    print(f"Records: {len(records)} total, {len(done_ids)} done, {len(todo)} to run")
    print(f"yes_id={yes_id}  no_id={no_id}\n")

    if not todo:
        print("Nothing to run.")
        rows = _load_rows(out_path)
        print_report(rows, warn_flags)
        return

    # ── sweep ─────────────────────────────────────────────────────────────────
    t_start = time.time()

    with out_path.open("a") as fout:
        for i, rec in enumerate(todo, 1):
            t0 = time.time()
            img = Image.open(rec.image_path).convert("RGB")
            if intervention is not None:
                with intervention:
                    cap = hm.run_prefill(img, rec.question)
            else:
                cap = hm.run_prefill(img, rec.question)

            # logits at the last prompt position: (vocab,)
            last_logits = cap.logits[cap.token_index.prompt_last].float()
            logit_yes = float(last_logits[yes_id])
            logit_no  = float(last_logits[no_id])
            pred = "yes" if logit_yes > logit_no else "no"
            correct = (pred == rec.answer)

            row = {
                "record_id":  rec.id,
                "source":     rec.source,
                "pair_id":    rec.pair_id,
                "gt_answer":  rec.answer,
                "pred":       pred,
                "correct":    correct,
                "logit_yes":  round(logit_yes, 4),
                "logit_no":   round(logit_no, 4),
                "logit_gt":   round(logit_yes if rec.answer == "yes" else logit_no, 4),
                "logit_foil": round(logit_no  if rec.answer == "yes" else logit_yes, 4),
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()

            elapsed = time.time() - t0
            eta = (time.time() - t_start) / i * (len(todo) - i)
            warn_tag = ""
            if rec.id in warn_flags:
                warn_tag = " [WARN]" if warn_flags[rec.id] else ""
            corr_tag = "OK " if correct else "ERR"
            print(
                f"  [{i:3d}/{len(todo)}] {corr_tag}  {rec.source:<14} {rec.id:<22}"
                f"  gt={rec.answer}  pred={pred}"
                f"  yes={logit_yes:6.2f}  no={logit_no:6.2f}"
                f"  {elapsed:.1f}s  ETA {eta/60:.1f}m"
                f"{warn_tag}"
            )

    total = time.time() - t_start
    print(f"\nDone. {len(todo)} records in {total/60:.1f}m")

    rows = _load_rows(out_path)
    print_report(rows, warn_flags)


if __name__ == "__main__":
    main()
