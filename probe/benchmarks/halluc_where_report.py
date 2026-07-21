"""Thesis-defense probe: when a VLM wrongly says an object is present, ask it
where the object is.

For each false-positive POPE record (gt="no", model answers "yes"), this asks a
*conversational* follow-up — the model's own "Yes." is in the dialogue history,
then we ask where the object is. If the answer is vague/meaningless (refusal,
degenerate tokens, or a contentless deflection like "In the image"), we retry
with a ladder of more explicit prompts until we get a meaningful localization or
exhaust the ladder.

Writes per-case JSONL (every attempt) and prints a readable transcript.

Usage:
    module load mamba; source activate <env>
    python -m probe.benchmarks.halluc_where_report \
        --backend llava --model_path liuhaotian/llava-v1.6-vicuna-7b \
        --records_json results/pope_records_cache_50.json \
        --max_records 50 --max_cases 8 \
        --out results/where_report_llava.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


_OBJ_RE = re.compile(r"is there an?\s+(.+?)\s+in the (?:image|picture)", re.IGNORECASE)

# Conversational prompt ladder. {obj} is filled per record. Tried in order;
# we stop at the first response judged meaningful.
PROMPT_LADDER = [
    "Where is the {obj} in the image?",
    "Can you describe where exactly the {obj} is located? For example, is it on "
    "the left, right, top, or bottom, and is it in the foreground or background?",
    "Please point to the {obj}. What object is it next to, and which part of the "
    "picture is it in?",
]

_REFUSAL = re.compile(
    r"unable|apolog|cannot comply|can't take|i'm sorry|cannot assist|can't assist|"
    r"i cannot|i can't help", re.IGNORECASE)
_SPATIAL = re.compile(
    r"\b(left|right|top|bottom|center|centre|middle|background|foreground|front|"
    r"behind|next to|beside|near|on the|under(neath)?|above|below|corner|side|edge|"
    r"in the (kitchen|restaurant|street|room|driveway|sky|water|grass|field|"
    r"background|foreground))\b", re.IGNORECASE)


def _extract_object(question: str) -> str:
    m = _OBJ_RE.search(question)
    if m:
        return m.group(1).strip()
    q = re.sub(r"^is there an?\s+", "", question.strip(), flags=re.IGNORECASE)
    q = re.sub(r"\s+in the (image|picture)\??$", "", q, flags=re.IGNORECASE)
    return q.strip() or "object"


def is_meaningful(ans: str, obj: str) -> bool:
    if not ans:
        return False
    s = ans.strip().lower()
    if _REFUSAL.search(s):
        return False
    alnum = re.sub(r"[^a-z]", "", s)
    if len(alnum) < 3:                       # mostly digits/punct
        return False
    if re.fullmatch(r"[\d\s\.\,\-\)\(]+", s):
        return False
    toks = s.split()
    if len(toks) >= 4 and len(set(toks)) <= 2:   # token-repetition loop
        return False
    deflections = {
        "in the image", "in the image.", "in the picture", "in the picture.",
        "the " + obj.lower(), obj.lower(), "the image is taken.", "yes", "yes.",
    }
    if s in deflections:
        return False
    if _SPATIAL.search(s):
        return True
    return len(toks) >= 6                     # a descriptive sentence


def _decode(tokenizer, cap) -> str:
    ids = cap.generated_ids
    if ids is None:
        return ""
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--records_json", required=True)
    p.add_argument("--max_records", type=int, default=50)
    p.add_argument("--max_cases", type=int, default=8,
                   help="cap on number of false-positive cases to probe")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from probe.benchmarks.run_bench import _load_hook_manager
    from probe.benchmarks.pope_full import BenchRecord

    raw = json.load(open(args.records_json))[:args.max_records]
    records = [BenchRecord(**r) for r in raw]

    hm, tokenizer = _load_hook_manager(SimpleNamespace(
        backend=args.backend, model_path=args.model_path,
        tokenizer_path=None, vq_path=None, projector_ckpt=None, max_image_size=256,
    ))
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_cases = 0
    n_meaningful = 0
    rows = []
    with out_path.open("w") as fout:
        for rec in records:
            img = Image.open(rec.image_path).convert("RGB")
            cap = hm.run_prefill(img, rec.question)
            last = cap.logits[cap.token_index.prompt_last].float()
            pred = "yes" if float(last[yes_id]) > float(last[no_id]) else "no"
            if not (rec.answer == "no" and pred == "yes"):
                continue
            n_cases += 1
            obj = _extract_object(rec.question)

            print(f"\n{'='*74}")
            print(f"{rec.id}  image={Path(rec.image_path).name}")
            print(f"  Q1: {rec.question}")
            print(f"  A1: Yes.   <-- WRONG (object is NOT in the image)")

            attempts = []
            final = None
            for tmpl in PROMPT_LADDER:
                q = tmpl.format(obj=obj)
                hm._followup_prior = (rec.question, "Yes.")
                try:
                    gcap = hm.run_generate(img, q, max_new_tokens=args.max_new_tokens)
                finally:
                    hm._followup_prior = None
                ans = _decode(tokenizer, gcap)
                ok = is_meaningful(ans, obj)
                attempts.append({"prompt": q, "answer": ans, "meaningful": ok})
                tag = "MEANINGFUL" if ok else "vague"
                print(f"  Q: {q}")
                print(f"  A: {ans!r}   [{tag}]")
                if ok:
                    final = attempts[-1]
                    break
            if final is not None:
                n_meaningful += 1
            else:
                final = attempts[-1]  # last attempt, even if vague

            row = {
                "record_id": rec.id,
                "image": Path(rec.image_path).name,
                "object": obj,
                "presence_question": rec.question,
                "model_presence_answer": "yes",
                "ground_truth": "no",
                "got_meaningful_location": final["meaningful"],
                "final_prompt": final["prompt"],
                "final_answer": final["answer"],
                "n_attempts": len(attempts),
                "attempts": attempts,
            }
            rows.append(row)
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            if n_cases >= args.max_cases:
                break

    print(f"\n{'='*74}")
    print(f"{args.backend}: {n_cases} false-positive cases probed; "
          f"{n_meaningful} produced a meaningful location "
          f"({n_meaningful}/{n_cases}).")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
