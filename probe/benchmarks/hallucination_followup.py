"""Hallucination follow-up probe (qualitative, exploratory).

Runs N POPE samples through a model. For every record where the ground truth is
"no" (the object is NOT in the image) but the model answers "yes" (a
false-positive object hallucination), it issues a fresh single-turn follow-up
question "Where is the <object> in the image?" and records the model's
generated answer. A model that has genuinely hallucinated the object will
typically invent a plausible location instead of refusing.

The yes/no decision uses the same logit-comparison convention as
probe.benchmarks.run_bench (logit_yes vs logit_no at prompt_last). The
follow-up is a new single-turn query on the same image (not a chat
continuation) to keep prompt formatting model-agnostic.

Usage
-----
    module load mamba; source activate sae
    python -m probe.benchmarks.hallucination_followup \
        --backend llava --model_path liuhaotian/llava-v1.6-vicuna-7b \
        --max_records 50 --out results/halluc_followup_llava.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


_OBJ_RE = re.compile(r"is there an?\s+(.+?)\s+in the (?:image|picture)", re.IGNORECASE)


def _extract_object(question: str) -> str:
    m = _OBJ_RE.search(question)
    if m:
        return m.group(1).strip()
    # Fallback: strip leading "Is there a/an" and trailing "in the image?"
    q = re.sub(r"^is there an?\s+", "", question.strip(), flags=re.IGNORECASE)
    q = re.sub(r"\s+in the (image|picture)\??$", "", q, flags=re.IGNORECASE)
    return q.strip() or "object"


def _decode_generated(tokenizer, cap) -> str:
    ids = cap.generated_ids
    if ids is None:
        return ""
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--vq_path", default=None)
    parser.add_argument("--projector_ckpt", default=None)
    parser.add_argument("--max_image_size", type=int, default=256)
    parser.add_argument("--max_records", type=int, default=50,
                        help="Number of POPE records to scan for hallucinations")
    parser.add_argument("--category", default="adversarial")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--records_json", default=None,
                        help="Load POPE records from a cached JSON list "
                             "(for envs without the `datasets` package)")
    parser.add_argument("--conversational", action="store_true",
                        help="Feed the model's own 'Yes.' answer back as chat "
                             "history before asking the follow-up (multi-turn). "
                             "Default is an independent single-turn follow-up.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from probe.benchmarks.run_bench import _load_hook_manager

    if args.records_json:
        from probe.benchmarks.pope_full import BenchRecord
        with open(args.records_json) as f:
            raw = json.load(f)
        records = [BenchRecord(**r) for r in raw][:args.max_records]
        print(f"  Loaded {len(records)} cached POPE records from {args.records_json}")
    else:
        from probe.benchmarks.pope_full import load_pope_full
        records = load_pope_full(category=args.category, max_records=args.max_records)

    loader_args = SimpleNamespace(
        backend=args.backend, model_path=args.model_path,
        tokenizer_path=args.tokenizer_path, vq_path=args.vq_path,
        projector_ckpt=args.projector_ckpt, max_image_size=args.max_image_size,
    )
    hm, tokenizer = _load_hook_manager(loader_args)

    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_fp = n_correct = 0
    rows: list[dict] = []
    t_start = time.time()

    with out_path.open("w") as fout:
        for i, rec in enumerate(records, 1):
            img = Image.open(rec.image_path).convert("RGB")
            cap = hm.run_prefill(img, rec.question)
            last = cap.logits[cap.token_index.prompt_last].float()
            logit_yes, logit_no = float(last[yes_id]), float(last[no_id])
            pred = "yes" if logit_yes > logit_no else "no"
            correct = pred == rec.answer
            n_correct += int(correct)

            obj = _extract_object(rec.question)
            is_fp = (rec.answer == "no" and pred == "yes")

            followup_q = followup_a = None
            if is_fp:
                n_fp += 1
                followup_q = f"Where is the {obj} in the image?"
                if args.conversational:
                    hm._followup_prior = (rec.question, "Yes.")
                try:
                    gcap = hm.run_generate(img, followup_q,
                                           max_new_tokens=args.max_new_tokens)
                finally:
                    hm._followup_prior = None
                followup_a = _decode_generated(tokenizer, gcap)

            row = {
                "record_id": rec.id,
                "image": Path(rec.image_path).name,
                "object": obj,
                "question": rec.question,
                "gt_answer": rec.answer,
                "pred": pred,
                "logit_yes": round(logit_yes, 3),
                "logit_no": round(logit_no, 3),
                "false_positive_hallucination": is_fp,
                "followup_mode": "conversational" if args.conversational else "independent",
                "followup_question": followup_q,
                "followup_answer": followup_a,
            }
            rows.append(row)
            fout.write(json.dumps(row) + "\n")
            fout.flush()

            tag = "HALLUC" if is_fp else ("OK" if correct else "err")
            msg = (f"  [{i:3d}/{len(records)}] {tag:6s} gt={rec.answer} "
                   f"pred={pred}  {obj}")
            if is_fp:
                msg += f"\n        Q: {followup_q}\n        A: {followup_a}"
            print(msg, flush=True)

    acc = n_correct / len(records) if records else 0.0
    print(f"\nDone in {(time.time()-t_start)/60:.1f}m. "
          f"acc={acc:.3f}  false-positive hallucinations={n_fp}/{len(records)}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
