"""Consistent cohort behavioral summary: POPE/AMBER base accuracy and L0-ablation
delta accuracy + delta yes-rate, computed identically from the bench JSONLs.

Replaces the old tab:cohort-behavior which mixed acc-deltas and yes-rate-deltas.
Drops CIs and the sparse A/B/C sanity tier; carries degeneracy as a flag instead.

Writes results/cohort_behavior_summary.tsv and prints a LaTeX tabular body.
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"

# (display_name, pope_stem, pope_L0_suffix, amber_stem, amber_L0_suffix, flag)
# flag: "d" generation-confirmed degenerate emitter; "c" catastrophic collapse; "" none
COHORT = [
    ("VILA-U",          "vilau",            "L0",           "vilau",            "intervention", ""),
    ("Liquid",          "liquid",           "L0",           "liquid",           "L0",           "d"),
    ("Chameleon-7B",    "chameleon",        "L0",           "chameleon",        "L0",           ""),
    ("Anole",           "anole",            "L0",           "anole",            "L0",           "d"),
    ("Lumina-mGPT",     "lumina_mgpt",      "L0",           "lumina_mgpt",      "L0",           ""),
    ("LLaVA-VQ v1",     "llava_vq",         "intervention", "llava_vq",         "intervention", ""),
    ("LLaVA-VQ-FSQ v1", "llava_vq_fsq",     "L0",           "llava_vq_fsq",     "L0",           ""),
    ("LLaVA-VQ-FSQ v2", "llava_vq_fsq_v2",  "L0",           "llava_vq_fsq_v2",  "L0",           ""),
    ("LLaVA-MLP ctrl",  "llava_mlp",        "L0",           "llava_mlp",        "L0",           ""),
    ("LLaVA-1.6",       "llava",            "intervention", "llava",            "intervention", ""),
    ("Qwen2.5-VL",      "qwen3vl",          "intervention", "qwen3vl",          "intervention", "c"),
    ("Emu3-Chat",       "emu3",             "L0",           "emu3",             "L0",           "c"),
    ("HaploOmni",       "haplo",            "L0",           "haplo",            "L0",           "c"),
    ("LLaVA-VQ-K4096",  "llava_vq_K4096",   "L0",           "llava_vq_K4096",   "L0",           ""),
    ("LLaVA-VQ-K65536", "llava_vq_K65536",  "L0",           "llava_vq_K65536",  "L0",           ""),
]


def load(stem: str) -> list[dict] | None:
    p = RESULTS / f"{stem}.jsonl"
    if not p.exists():
        return None
    return [json.loads(line) for line in p.open() if line.strip()]


def acc(rows: list[dict]) -> float:
    return sum(1 for r in rows if r.get("correct")) / len(rows)


def yes_rate(rows: list[dict]) -> float:
    return sum(1 for r in rows if str(r.get("pred", "")).lower().strip() == "yes") / len(rows)


def pair(bench: str, stem: str, suf: str):
    base = load(f"bench_{bench}_{stem}_baseline")
    l0 = load(f"bench_{bench}_{stem}_{suf}")
    if base is None or l0 is None:
        return None
    return {
        "base_acc": acc(base),
        "d_acc": (acc(l0) - acc(base)) * 100,
        "d_yr": (yes_rate(l0) - yes_rate(base)) * 100,
        "n": len(l0),
    }


def main():
    rows_out = []
    hdr = ["model", "pope_base", "pope_dacc", "pope_dyr", "amber_base",
           "amber_dacc", "amber_dyr", "flag"]
    for name, ps, psuf, as_, asuf, flag in COHORT:
        p = pair("pope_full", ps, psuf)
        a = pair("amber", as_, asuf)
        rows_out.append({
            "model": name,
            "pope_base": f"{p['base_acc']:.3f}" if p else "--",
            "pope_dacc": f"{p['d_acc']:+.1f}" if p else "--",
            "pope_dyr": f"{p['d_yr']:+.1f}" if p else "--",
            "amber_base": f"{a['base_acc']:.3f}" if a else "--",
            "amber_dacc": f"{a['d_acc']:+.1f}" if a else "--",
            "amber_dyr": f"{a['d_yr']:+.1f}" if a else "--",
            "flag": flag,
        })

    out = RESULTS / "cohort_behavior_summary.tsv"
    with out.open("w") as f:
        f.write("\t".join(hdr) + "\n")
        for r in rows_out:
            f.write("\t".join(r[k] for k in hdr) + "\n")
    print(f"wrote {out}")

    mark = {"d": r"$^{d}$", "c": r"$^{c}$", "": ""}
    print("\n% --- LaTeX tabular body ---")
    for r in rows_out:
        print(f"    {r['model']}{mark[r['flag']]} & {r['pope_base']} & {r['pope_dacc']} & "
              f"{r['pope_dyr']} & {r['amber_base']} & {r['amber_dacc']} & {r['amber_dyr']} \\\\")


if __name__ == "__main__":
    main()
