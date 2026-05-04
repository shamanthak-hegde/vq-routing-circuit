"""Intervention-aware benchmark runner (N-042).

Runs a full benchmark (pope_full / nb_full / hb / amber) with optional L0
intervention, writes per-record JSONL, and prints aggregate metrics.

Usage
-----
    # Baseline VILA-U on full POPE
    source activate vila-u
    python -m probe.benchmarks.run_bench \\
        --bench pope_full \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --out results/bench_pope_vilau.jsonl

    # VILA-U with winning selective top-3 intervention
    python -m probe.benchmarks.run_bench \\
        --bench pope_full \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --knockout_mode selective --heads 6,7,14 --knockout_layer 0 \\
        --out results/bench_pope_vilau_intervention.jsonl

    # LLaVA baseline (architecture-specificity: expect null delta)
    source activate sae
    python -m probe.benchmarks.run_bench \\
        --bench pope_full \\
        --backend llava \\
        --model_path liuhaotian/llava-v1.6-vicuna-7b \\
        --knockout_mode selective --heads 6,7,14 --knockout_layer 0 \\
        --out results/bench_pope_llava_intervention.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image


def _load_hook_manager(args: argparse.Namespace):
    print(f"Loading {args.backend} model from {args.model_path} …")
    if args.backend == "llava":
        _llava = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        )
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
        return LlavaHookManager(model, tokenizer, image_processor), tokenizer

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
        return VilaUHookManager(model, tokenizer, image_processor), tokenizer

    if args.backend == "unitok":
        _unitok = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "UniTok")
        )
        _liquid = os.path.join(_unitok, "eval", "liquid")
        for p in (_unitok, _liquid):
            if p not in sys.path:
                sys.path.insert(0, p)
        from model.builder import load_pretrained_model
        from mm_utils import get_model_name_from_path
        from models.unitok import UniTok
        from utils.config import Args as UniTokArgs
        from probe.hooks.unitok import UniTokHookManager
        import torch
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, _, _ = load_pretrained_model(
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
        return UniTokHookManager(model, tokenizer, vq_model), tokenizer

    if args.backend == "qwen3vl":
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
        return Qwen3VLHookManager(model, processor), processor.tokenizer

    if args.backend == "vila":
        _vila = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "VILA")
        )
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
        return VilaHookManager(model, tokenizer, image_processor), tokenizer

    if args.backend == "haplo":
        _haplo = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "HaploVLM")
        )
        _haplo_model = os.path.join(_haplo, "haploomni", "model")
        for _p in (_haplo, _haplo_model):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from haploomni import HaploOmniForConditionalGeneration, HaploOmniProcessor
        from probe.hooks.haplo import HaploOmniHookManager
        processor = HaploOmniProcessor.from_pretrained(args.model_path)
        model = HaploOmniForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        hm = HaploOmniHookManager(model, processor)
        return hm, hm.tokenizer

    if args.backend == "emu3":
        if args.vq_path is None:
            raise ValueError("--vq_path is required for --backend emu3")
        from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
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
            args.model_path,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
        ).eval()
        hm = Emu3HookManager(model, tokenizer, image_processor, image_tokenizer,
                             max_image_size=getattr(args, "max_image_size", 256))
        return hm, tokenizer

    raise ValueError(f"Unknown backend {args.backend!r}")


def _load_records(bench: str, max_records: int | None) -> list[Any]:
    if bench == "pope_full":
        from probe.benchmarks.pope_full import load_pope_full
        return load_pope_full(max_records=max_records)
    if bench == "nb_full":
        from probe.benchmarks.naturalbench_full import load_naturalbench_full
        return load_naturalbench_full(max_records=max_records)
    if bench == "hb":
        from probe.benchmarks.hallusionbench import load_hallusionbench
        return load_hallusionbench(max_records=max_records)
    if bench == "amber":
        from probe.benchmarks.amber import load_amber
        return load_amber(max_records=max_records)
    raise ValueError(f"Unknown bench {bench!r}")


def _score(bench: str, predictions: list[dict]) -> dict:
    if bench == "pope_full":
        from probe.benchmarks.pope_full import score_pope
        return score_pope(predictions)
    if bench == "nb_full":
        from probe.benchmarks.naturalbench_full import score_naturalbench
        return score_naturalbench(predictions)
    if bench == "hb":
        from probe.benchmarks.hallusionbench import score_hallusionbench
        return score_hallusionbench(predictions)
    if bench == "amber":
        from probe.benchmarks.amber import score_amber
        return score_amber(predictions)
    raise ValueError(f"Unknown bench {bench!r}")


def _load_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["record_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a full benchmark with optional L0 intervention"
    )
    parser.add_argument(
        "--bench",
        required=True,
        choices=["pope_full", "nb_full", "hb", "amber"],
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["llava", "vila", "vilau", "unitok", "qwen3vl", "haplo", "emu3"],
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None,
                        help="[unitok only] path to unitok_tokenizer.pth")
    parser.add_argument("--vq_path", default=None,
                        help="[emu3 only] path or HF hub ID of Emu3-VisionTokenizer")
    parser.add_argument("--max_image_size", type=int, default=256,
                        help="[emu3 only] cap image dimensions before VQ-encoding (default 256→1024 tokens)")
    parser.add_argument("--out", required=True, help="Output JSONL (resumable)")
    parser.add_argument("--max_records", type=int, default=None)
    # Intervention args
    parser.add_argument(
        "--knockout_mode",
        choices=["pathological_route_ablation", "full_zero", "selective", "scalar", "selective_scalar"],
        default=None,
        help="full_zero is a deprecated alias for pathological_route_ablation",
    )
    parser.add_argument("--knockout_layer", type=int, default=0)
    parser.add_argument("--heads", default="6,7,14")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = _load_records(args.bench, args.max_records)
    hm, tokenizer = _load_hook_manager(args)

    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]
    for r in records:
        r.answer_token_id = yes_id if r.answer == "yes" else no_id

    intervention = None
    if args.knockout_mode is not None:
        from probe.tracing.head_knockout import build_intervention, _parse_heads
        intervention = build_intervention(
            args.knockout_mode,
            hm,
            args.knockout_layer,
            _parse_heads(args.heads),
            args.alpha,
        )
        print(
            f"Intervention: mode={args.knockout_mode}, layer={args.knockout_layer}"
            + (f", heads={args.heads}" if args.knockout_mode in ("selective", "selective_scalar") else "")
            + (f", alpha={args.alpha}" if args.knockout_mode in ("scalar", "selective_scalar") else "")
        )

    done_ids = _load_done_ids(out_path)
    todo = [r for r in records if r.id not in done_ids]
    print(f"Records: {len(records)} total, {len(done_ids)} done, {len(todo)} to run")
    if not todo:
        print("Nothing to run. Loading existing results for scoring …")
        rows: list[dict] = []
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        metrics = _score(args.bench, rows)
        print(json.dumps(metrics, indent=2))
        meta_path = out_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps({
            "bench": args.bench,
            "backend": args.backend,
            "model_path": args.model_path,
            "intervention_mode": args.knockout_mode,
            "knockout_layer": args.knockout_layer,
            "heads": args.heads if args.knockout_mode in ("selective", "selective_scalar") else None,
            "alpha": args.alpha if args.knockout_mode in ("scalar", "selective_scalar") else None,
            "n_records": len(rows),
            "metrics": metrics,
        }, indent=2) + "\n")
        print(f"Meta written to {meta_path}")
        return

    t_start = time.time()
    all_rows: list[dict] = []

    with out_path.open("a") as fout:
        for i, rec in enumerate(todo, 1):
            t0 = time.time()
            img = Image.open(rec.image_path).convert("RGB")
            if intervention is not None:
                with intervention:
                    cap = hm.run_prefill(img, rec.question)
            else:
                cap = hm.run_prefill(img, rec.question)

            last_logits = cap.logits[cap.token_index.prompt_last].float()
            logit_yes = float(last_logits[yes_id])
            logit_no = float(last_logits[no_id])
            pred = "yes" if logit_yes > logit_no else "no"
            correct = pred == rec.answer

            extra: dict[str, Any] = {}
            for fld in ("pair_id", "group_id", "figure_id", "question_id"):
                val = getattr(rec, fld, None)
                if val:
                    extra[fld] = val

            row = {
                "record_id": rec.id,
                "source": rec.source,
                "gt_answer": rec.answer,
                "pred": pred,
                "correct": correct,
                "logit_yes": round(logit_yes, 4),
                "logit_no": round(logit_no, 4),
                **extra,
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            all_rows.append(row)

            elapsed = time.time() - t0
            eta = (time.time() - t_start) / i * (len(todo) - i)
            print(
                f"  [{i:4d}/{len(todo)}] {'OK' if correct else 'ERR'}"
                f"  {rec.source:<16} {rec.id:<28}"
                f"  gt={rec.answer}  pred={pred}"
                f"  {elapsed:.1f}s  ETA {eta/60:.1f}m",
                flush=True,
            )

    print(f"\nDone. {len(todo)} records in {(time.time()-t_start)/60:.1f}m")
    # Score from full JSONL (not just all_rows) to guard against partial-resume artifacts.
    full_rows: list[dict] = []
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    full_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    metrics = _score(args.bench, full_rows)
    print(json.dumps(metrics, indent=2))

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "bench": args.bench,
        "backend": args.backend,
        "model_path": args.model_path,
        "intervention_mode": args.knockout_mode,
        "knockout_layer": args.knockout_layer,
        "heads": args.heads if args.knockout_mode in ("selective", "selective_scalar") else None,
        "alpha": args.alpha if args.knockout_mode in ("scalar", "selective_scalar") else None,
        "n_records": len(full_rows),
        "metrics": metrics,
    }, indent=2) + "\n")
    print(f"Meta written to {meta_path}")


if __name__ == "__main__":
    main()
