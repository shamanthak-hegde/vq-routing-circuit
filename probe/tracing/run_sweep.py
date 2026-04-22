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

Optional flags
--------------
    --window_size 4          layer-window width (default 4)
    --sigma       0.5        noise σ for POPE records (default = corrupt.py default)
    --limit       500        cap number of records (default: all)
    --source      all        "all" | "naturalbench" | "pope"
    --print_heatmap          print aggregate heatmap to stdout when done
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path


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
    _llava = os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
    if _llava not in sys.path:
        sys.path.insert(0, _llava)

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from probe.hooks import LlavaHookManager
    from probe import load_cache, resolve_answer_token_ids
    from probe.tracing.sweep import sweep_record
    from probe.tracing.corrupt import noisy_embeds  # import to read default sigma

    import inspect
    _default_sigma = inspect.signature(noisy_embeds).parameters["sigma"].default

    parser = argparse.ArgumentParser(description="Full probe-set patching sweep")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out", default="sweep_results.jsonl",
                        help="Output JSONL file (appended; safe to resume)")
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=_default_sigma)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max records to process (default: all)")
    parser.add_argument("--source", default="all",
                        choices=["all", "naturalbench", "pope"])
    parser.add_argument("--print_heatmap", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {args.model_path} ...")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, model_base=None, model_name=model_name,
        attn_implementation="sdpa",
    )
    model.eval()
    hm = LlavaHookManager(model, tokenizer, image_processor)

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
    print(f"window_size={args.window_size}  sigma={args.sigma}  out={out_path}\n")

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
            rows = sweep_record(hm, rec, window_size=args.window_size, sigma=args.sigma)

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

    if args.print_heatmap:
        _print_heatmap(out_path)


if __name__ == "__main__":
    main()
