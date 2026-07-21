"""Diagnose VILA-U terse hallucination follow-ups (manual probe).

For a handful of VILA-U false-positive POPE records, ask the "where is it?"
follow-up with several prompt variants and report, for each:
  - the generated text
  - n_generated_tokens and whether EOS was reached before the token budget
    (settles whether terse answers are truncation vs early-EOS)

Usage:
    module load mamba; source activate vila-u
    python -m probe.benchmarks.vilau_halluc_probe \
        --model_path mit-han-lab/vila-u-7b-256 \
        --records_json results/pope_records_cache_50.json \
        --fp_jsonl results/halluc_followup_vilau.jsonl
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from PIL import Image


PROMPT_VARIANTS = [
    ("original", "Where is the {obj} in the image?"),
    ("detailed", "Describe in detail where the {obj} is located in the image. "
                 "Answer in one or two complete sentences."),
    ("position", "The image contains a {obj}. State its position in the image "
                 "(for example: top-left, center, bottom-right) and describe "
                 "what it looks like."),
]


def _decode(tokenizer, cap) -> str:
    ids = cap.generated_ids
    if ids is None:
        return ""
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _n_gen(cap) -> int:
    ids = cap.generated_ids
    if ids is None:
        return 0
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="mit-han-lab/vila-u-7b-256")
    p.add_argument("--records_json", required=True)
    p.add_argument("--fp_jsonl", required=True)
    p.add_argument("--n_cases", type=int, default=6)
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()

    from probe.benchmarks.run_bench import _load_hook_manager

    rec_by_id = {r["id"]: r for r in json.load(open(args.records_json))}
    fp = [json.loads(l) for l in open(args.fp_jsonl)]
    fp = [r for r in fp if r["false_positive_hallucination"]][:args.n_cases]

    hm, tokenizer = _load_hook_manager(SimpleNamespace(
        backend="vilau", model_path=args.model_path,
        tokenizer_path=None, vq_path=None, projector_ckpt=None,
        max_image_size=256,
    ))

    for r in fp:
        rec = rec_by_id[r["record_id"]]
        obj = r["object"]
        img = Image.open(rec["image_path"]).convert("RGB")
        print(f"\n{'='*72}\n{r['record_id']}  object={obj!r}  (gt=no, model said YES)")
        print(f"original Q: {rec['question']}")
        for name, tmpl in PROMPT_VARIANTS:
            q = tmpl.format(obj=obj)
            cap = hm.run_generate(img, q, max_new_tokens=args.max_new_tokens)
            txt = _decode(tokenizer, cap)
            n = _n_gen(cap)
            eos = "EOS" if n < args.max_new_tokens else "HIT-MAX"
            print(f"  [{name:8s}] n_gen={n:3d} ({eos})  ->  {txt!r}")


if __name__ == "__main__":
    main()
