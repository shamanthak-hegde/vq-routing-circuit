"""Full probe-set sweep runner with incremental checkpointing.

Writes one JSON line per PatchResult to --out (JSONL format).  If the output
file already exists, records whose record_id already appears in it are skipped,
so interrupted runs can be resumed safely.

Usage
-----
    source activate sae
    python -m probe.tracing.run_sweep \\
        --model_path liuhaotian/llava-v1.6-vicuna-7b \\
        --out results/sweep_llava.jsonl

    conda activate vila-u
    python -m probe.tracing.run_sweep \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --out results/sweep_vilau.jsonl

Optional flags
--------------
    --backend     llava       "llava" | "vilau" | "vila" | "unitok" | "qwen3vl" (default "llava")
    --window_size 4          layer-window width (default 4)
    --sigma       <float>    noise σ for POPE records (required; use calibrated value)
    --limit       500        cap number of records (default: all)
    --source      all        "all" | "naturalbench" | "pope"
    --print_heatmap          print aggregate heatmap to stdout when done
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def _probe_set_hash() -> str:
    records_json = Path(__file__).parent.parent / "cached" / "records.json"
    if not records_json.exists():
        return "unknown"
    return hashlib.sha256(records_json.read_bytes()).hexdigest()[:12]



def _load_done_ids(out_path: Path) -> set[str]:
    """Return record_ids already written to the checkpoint file."""
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


def _print_heatmap(out_path: Path) -> None:
    rows = []
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not rows:
        print("No results to aggregate.")
        return

    # filter to records where corruption actually hurt (logit_clean > logit_corrupt)
    rows = [r for r in rows if r["logit_clean"] > r["logit_corrupt"]]
    if not rows:
        print("All records were WARN (logit_clean ≤ logit_corrupt) — nothing to show.")
        return

    from itertools import groupby as _groupby

    sources = sorted(set(r["source"] for r in rows))
    for src in sources:
        src_rows = [r for r in rows if r["source"] == src]
        windows = sorted(set((r["layer_start"], r["layer_end"]) for r in src_rows))
        groups  = sorted(set(r["token_group"] for r in src_rows))

        print(f"\n── {src} (n={len(set(r['record_id'] for r in src_rows))} records) ──")
        header = f"  {'layers':>10}" + "".join(f"  {g:>12}" for g in groups)
        print(header)

        for l_s, l_e in windows:
            cells = []
            for g in groups:
                vals = [r["score"] for r in src_rows
                        if r["layer_start"] == l_s and r["token_group"] == g]
                cells.append(f"{sum(vals)/len(vals):>12.3f}" if vals else f"{'—':>12}")
            print(f"  [{l_s:2d},{l_e:2d})    " + "".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description="Full probe-set patching sweep")
    parser.add_argument("--backend", default="llava",
                        choices=["llava", "vilau", "vila", "unitok", "qwen3vl", "haplo", "emu3", "lavit"],
                        help="Model backend (default: llava)")
    parser.add_argument("--tokenizer_path", default=None,
                        help="[unitok only] Path to unitok_tokenizer.pth")
    parser.add_argument("--vq_path", default=None,
                        help="[emu3 only] Path or HF hub ID of Emu3-VisionTokenizer")
    parser.add_argument("--max_image_size", type=int, default=256,
                        help="[emu3 only] Cap image dimensions before VQ-encoding (default 256→1024 tokens)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out", default="sweep_results.jsonl",
                        help="Output JSONL file (appended; safe to resume)")
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--sigma", type=float, required=True,
                        help="Noise σ for POPE gaussian_noise records (use calibrated value)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max records to process (default: all)")
    parser.add_argument("--source", default="all",
                        choices=["all", "naturalbench", "pope"])
    parser.add_argument("--n_seeds", type=int, default=1,
                        help="Noise seeds to average for POPE records (default 1)")
    parser.add_argument("--print_heatmap", action="store_true")
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from probe import load_cache, resolve_answer_token_ids
    from probe.tracing.sweep import sweep_record

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
        _unitok = os.path.join(os.path.dirname(__file__), "..", "..", "UniTok")
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
            raise ValueError("--tokenizer_path is required for --backend unitok")
        ckpt = torch.load(os.path.expanduser(args.tokenizer_path), map_location="cpu")
        vae_cfg = UniTokArgs()
        vae_cfg.load_state_dict(ckpt["args"])
        vq_model = UniTok(vae_cfg)
        vq_model.load_state_dict(ckpt["trainer"]["unitok"])
        vq_model.to(device).eval()
        del ckpt
        hm = UniTokHookManager(model, tokenizer, vq_model)
    elif args.backend == "qwen3vl":
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from probe.hooks.qwen3vl import Qwen3VLHookManager
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True, padding_side="left", use_fast=True
        )
        processor.image_processor.max_pixels = 720 * 1280
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto",
        ).eval()
        hm = Qwen3VLHookManager(model, processor)
        tokenizer = hm.tokenizer
    elif args.backend == "haplo":
        import torch
        _haplo = os.path.join(os.path.dirname(__file__), "..", "..", "HaploVLM")
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
        ).eval()
        hm = HaploOmniHookManager(model, processor)
        tokenizer = hm.tokenizer
    elif args.backend == "emu3":
        import torch
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
                             max_image_size=args.max_image_size)
    elif args.backend == "lavit":
        import torch
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

    # ── load records ──────────────────────────────────────────────────────────
    records, _, _ = load_cache()
    resolve_answer_token_ids(records, tokenizer)

    if args.source != "all":
        records = [r for r in records if r.source == args.source]
    if args.limit:
        records = records[:args.limit]

    done_ids = _load_done_ids(out_path)
    todo = [r for r in records if r.id not in done_ids]

    n_total = len(records)
    n_done  = len(done_ids) if done_ids else 0
    print(f"Records: {n_total} total, {n_done} already done, {len(todo)} to run")
    print(f"window_size={args.window_size}  sigma={args.sigma}  n_seeds={args.n_seeds}  out={out_path}\n")

    if not todo:
        print("Nothing to do.")
        if args.print_heatmap:
            _print_heatmap(out_path)
        return

    # ── sweep ─────────────────────────────────────────────────────────────────
    t_sweep_start = time.time()
    warn_count = 0

    with out_path.open("a") as fout:
        for i, rec in enumerate(todo, start=1):
            t0 = time.time()
            rows = sweep_record(hm, rec, window_size=args.window_size, sigma=args.sigma, n_seeds=args.n_seeds)

            # detect WARN (corruption didn't hurt)
            is_warn = rows[0].logit_clean <= rows[0].logit_corrupt if rows else False
            if is_warn:
                warn_count += 1

            for row in rows:
                fout.write(json.dumps(asdict(row)) + "\n")
            fout.flush()

            elapsed_rec = time.time() - t0
            elapsed_total = time.time() - t_sweep_start
            remaining = len(todo) - i
            eta = (elapsed_total / i) * remaining if i > 0 else 0

            warn_flag = " [WARN]" if is_warn else ""
            print(
                f"  [{i:3d}/{len(todo)}] {rec.source:<14} {rec.id:<20}"
                f"  {elapsed_rec:4.1f}s"
                f"  logit_clean={rows[0].logit_clean:6.2f}"
                f"  logit_corrupt={rows[0].logit_corrupt:6.2f}"
                f"  ETA {eta/60:4.1f}m"
                f"{warn_flag}"
            )

    total_elapsed = time.time() - t_sweep_start
    print(
        f"\nDone. {len(todo)} records in {total_elapsed/60:.1f}m  "
        f"({warn_count} WARNs = {100*warn_count/len(todo):.1f}%)"
    )

    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "model_path": args.model_path,
        "backend": args.backend,
        "sigma": args.sigma,
        "window_size": args.window_size,
        "n_seeds": args.n_seeds,
        "source": args.source,
        "limit": args.limit,
        "probe_set_hash": _probe_set_hash(),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "n_records_this_run": len(todo),
        "n_warn_this_run": warn_count,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Metadata → {meta_path}")

    if args.print_heatmap:
        _print_heatmap(out_path)


if __name__ == "__main__":
    main()
