"""R-A: Liquid L0-ablation degenerate-emitter sanity check (GPU generation).

For 20 records (default: 5 POPE-yes + 5 POPE-no + 5 AMBER-yes + 5 AMBER-no),
runs Liquid's `run_generate` (max_new_tokens=20) in baseline and L0-knockout
modes and records the decoded text along with logit_yes / logit_no.

Outputs:
  - results/sanity_check_liquid_generated.jsonl
  - results/sanity_check_liquid_generated.meta.json

Run (sae env per CLAUDE.md):
    module load mamba && source activate sae
    python -m probe.tracing.liquid_sanity_generate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT_JSONL = RESULTS / "sanity_check_liquid_generated.jsonl"
OUT_META = RESULTS / "sanity_check_liquid_generated.meta.json"


def _load_liquid(model_path: str):
    chameleon_root = os.path.normpath(REPO / "chameleon")
    liquid_root = os.path.normpath(REPO / "Liquid")
    for p in (chameleon_root, liquid_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from chameleon.inference.image_tokenizer import ImageTokenizer
    from probe.hooks.liquid import LiquidHookManager

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda",
    ).eval()
    vqgan_cfg = os.path.join(chameleon_root, "data", "tokenizer", "vqgan.yaml")
    vqgan_ckpt = os.path.join(chameleon_root, "data", "tokenizer", "vqgan.ckpt")
    image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg, ckpt_path=vqgan_ckpt, device="cuda:0")
    return LiquidHookManager(model, tokenizer, image_tokenizer), tokenizer


def _balanced_sample(records, n_yes: int, n_no: int, seed: int) -> list:
    rng = random.Random(seed)
    yes_pool = [r for r in records if r.answer == "yes"]
    no_pool = [r for r in records if r.answer == "no"]
    rng.shuffle(yes_pool)
    rng.shuffle(no_pool)
    return yes_pool[:n_yes] + no_pool[:n_no]


def _load_records(seed: int) -> list:
    from probe.benchmarks.pope_full import load_pope_full
    from probe.benchmarks.amber import load_amber

    print("Loading POPE adversarial …")
    pope = load_pope_full()
    print("Loading AMBER discriminative …")
    amber = load_amber()

    chosen: list = []
    for src, rs, ny, nn in [("pope", pope, 5, 5), ("amber", amber, 5, 5)]:
        sub = _balanced_sample(rs, ny, nn, seed)
        for r in sub:
            r.source = src
        chosen.extend(sub)
    return chosen


def _decode_generated(tokenizer, cap) -> str:
    ids = cap.generated_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return text.strip()


def _run_one(hm, tokenizer, rec, intervention_factory, yes_id: int, no_id: int,
             max_new_tokens: int) -> dict:
    """One image+question. Two passes:
      - run_prefill (under intervention) → logit_yes / logit_no at prompt_last
      - run_generate (under intervention) → decoded text

    intervention_factory: callable taking no args, returning a fresh context
    manager.  Pass None to skip intervention.
    """
    img = Image.open(rec.image_path).convert("RGB")

    if intervention_factory is not None:
        with intervention_factory():
            cap_p = hm.run_prefill(img, rec.question)
    else:
        cap_p = hm.run_prefill(img, rec.question)
    prompt_last_logits = cap_p.logits[cap_p.token_index.prompt_last].float()
    logit_yes = float(prompt_last_logits[yes_id])
    logit_no = float(prompt_last_logits[no_id])

    if intervention_factory is not None:
        with intervention_factory():
            cap_g = hm.run_generate(img, rec.question, max_new_tokens=max_new_tokens)
    else:
        cap_g = hm.run_generate(img, rec.question, max_new_tokens=max_new_tokens)
    text = _decode_generated(tokenizer, cap_g)

    return {
        "text": text,
        "logit_yes": round(logit_yes, 4),
        "logit_no": round(logit_no, 4),
        "gap": round(logit_yes - logit_no, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Junfeng5/Liquid_V1_7B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--n_yes", type=int, default=5,
                    help="yes-GT samples per source")
    ap.add_argument("--n_no", type=int, default=5,
                    help="no-GT samples per source")
    args = ap.parse_args()

    if OUT_JSONL.exists():
        OUT_JSONL.unlink()

    records = _load_records(args.seed)
    print(f"Sampled {len(records)} records "
          f"({sum(r.answer=='yes' for r in records)} yes / "
          f"{sum(r.answer=='no' for r in records)} no)")

    print("Loading Liquid …")
    hm, tokenizer = _load_liquid(args.model_path)
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]
    print(f"yes_id={yes_id}  no_id={no_id}")

    from probe.tracing.head_knockout import build_intervention

    t0 = time.time()
    rows: list[dict] = []
    with OUT_JSONL.open("w") as fout:
        for i, rec in enumerate(records, 1):
            t_rec = time.time()
            baseline = _run_one(hm, tokenizer, rec, None, yes_id, no_id,
                                args.max_new_tokens)
            l0_factory = lambda: build_intervention(
                "pathological_route_ablation", hm,
                layer_idx=0, head_idxs="all", alpha=0.0,
            )
            l0 = _run_one(hm, tokenizer, rec, l0_factory, yes_id, no_id,
                          args.max_new_tokens)
            row = {
                "record_id": rec.id,
                "source": rec.source,
                "gt_answer": rec.answer,
                "question": rec.question,
                "image_path": rec.image_path,
                "baseline_text": baseline["text"],
                "baseline_logit_yes": baseline["logit_yes"],
                "baseline_logit_no": baseline["logit_no"],
                "baseline_gap": baseline["gap"],
                "L0_text": l0["text"],
                "L0_logit_yes": l0["logit_yes"],
                "L0_logit_no": l0["logit_no"],
                "L0_gap": l0["gap"],
            }
            rows.append(row)
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            elapsed = time.time() - t_rec
            print(f"  [{i:2d}/{len(records)}] {rec.source:5s} gt={rec.answer:3s} "
                  f"id={rec.id}  "
                  f"base_gap={baseline['gap']:+.2f}  base='{baseline['text'][:30]}'  "
                  f"L0_gap={l0['gap']:+.2f}  L0='{l0['text'][:30]}'  "
                  f"{elapsed:.1f}s", flush=True)

    dt = time.time() - t0
    print(f"\nDone in {dt/60:.1f}m")

    OUT_META.write_text(json.dumps({
        "node": "R-A",
        "model_path": args.model_path,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "n_records": len(rows),
        "intervention": "pathological_route_ablation @ layer 0",
        "out_jsonl": str(OUT_JSONL.relative_to(REPO)),
    }, indent=2) + "\n")
    print(f"meta -> {OUT_META}")


if __name__ == "__main__":
    main()
