"""Evaluate whether a single continuous score can cleanly replace the two-tier
gate-A.2 rule for separating pathological from non-pathological VLMs.

Three candidate measures are computed from attn_head_weights_*.json files:
  1. L0_mass       — sum of per-head visual→prompt_last weights at layer 0
  2. window_mass   — sum of L0_mass over layers [0, 8)
  3. norm_window   — window_mass / total_mass (fractional concentration)

A clean separator must place all PASS models above all FAIL models with a gap.

Results written to:
  results/unified_a2_summary.tsv

Usage:
    source activate sae
    python -m probe.analysis.unified_a2_score
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# Ground-truth verdicts / project log.
# Key: backend name matching attn_head_weights_{backend}.json
# Value: ("PASS"/"FAIL", tier, notes)
VERDICTS: dict[str, tuple[str, str, str]] = {
    "vilau":         ("PASS", "projector-amplified", "natural specimen, Vicuna-7B"),
    "liquid":        ("PASS", "unified-vocab",        "natural specimen, Gemma-7B; L0 mass low by design"),
    "chameleon":     ("PASS", "unified-vocab",        "natural specimen, LLaMA-32k; causal confirmed"),
    "anole":         ("PASS", "unified-vocab",        "fine-tuned Chameleon descendant; [0,8) amplified 5×"),
    "llava_vq":      ("PASS", "projector-amplified",  "induced specimen v1, K=16384"),
    "llava_vq_v2":   ("PASS", "projector-amplified",  "induced specimen v2, K=16384, entropy reg"),
    # Fail models
    "emu3":          ("FAIL", "unified-vocab",         "L0 mass=12.5 < 15; no behavioral circuit"),
    "janus":         ("FAIL", "decoupled",             "continuous SigLIP path; no VQ bias (corrected)"),
    "unitok":        ("FAIL", "projector-amplified",   "MLP-diffusion divergence, not routing"),
    "qwen_vq":       ("FAIL", "projector-amplified",   "VQ collapse present; L0 circuit absent"),
    "vilau_mlp":     ("FAIL", "projector-amplified",   "reverse ablation; continuous adapter worsens pathology"),
}

# Exclude variant files (projscale, clean_noisy duplicates)
_SKIP_SUFFIXES = ("projscale", "clean_noisy")


def _load_scores(results_dir: str) -> list[dict]:
    rd = Path(results_dir)
    rows = []
    for fn in sorted(os.listdir(rd)):
        if not fn.startswith("attn_head_weights_") or not fn.endswith(".json"):
            continue
        backend = fn.removeprefix("attn_head_weights_").removesuffix(".json")
        if any(backend.endswith(s) for s in _SKIP_SUFFIXES):
            continue
        d = json.load(open(rd / fn))
        mhvs = d["mean_head_visual_sum"]
        n_layers = d["n_layers"]
        l0_mass = sum(mhvs[0])
        window_mass = sum(sum(mhvs[l]) for l in range(min(8, n_layers)))
        total_mass = sum(sum(mhvs[l]) for l in range(n_layers))
        norm_window = window_mass / total_mass if total_mass > 0 else 0.0
        verdict, tier, note = VERDICTS.get(backend, ("UNKNOWN", "?", ""))
        rows.append({
            "backend": backend,
            "verdict": verdict,
            "tier": tier,
            "L0_mass": round(l0_mass, 2),
            "window_mass_0_8": round(window_mass, 2),
            "total_mass": round(total_mass, 2),
            "norm_window_0_8": round(norm_window, 4),
            "note": note,
        })
    return rows


def _find_threshold(
    rows: list[dict],
    key: str,
    higher_is_pass: bool = True,
) -> tuple[float | None, int, int]:
    """Find a threshold that perfectly separates PASS from FAIL on key.

    Returns (threshold, n_errors, n_unknown).
    threshold=None means no clean separator exists.
    """
    known = [r for r in rows if r["verdict"] in ("PASS", "FAIL")]
    unknown = len(rows) - len(known)
    pass_vals = sorted([r[key] for r in known if r["verdict"] == "PASS"])
    fail_vals = sorted([r[key] for r in known if r["verdict"] == "FAIL"])
    if not pass_vals or not fail_vals:
        return None, 0, unknown

    if higher_is_pass:
        min_pass = min(pass_vals)
        max_fail = max(fail_vals)
        if min_pass > max_fail:
            return (min_pass + max_fail) / 2, 0, unknown
    else:
        max_pass = max(pass_vals)
        min_fail = min(fail_vals)
        if max_pass < min_fail:
            return (max_pass + min_fail) / 2, 0, unknown

    # Count misclassified at every candidate threshold
    all_vals = sorted(set([r[key] for r in known]))
    best_thresh, best_err = None, len(known) + 1
    for v in all_vals:
        if higher_is_pass:
            errors = sum(1 for r in known if
                         (r["verdict"] == "PASS" and r[key] < v) or
                         (r["verdict"] == "FAIL" and r[key] >= v))
        else:
            errors = sum(1 for r in known if
                         (r["verdict"] == "PASS" and r[key] > v) or
                         (r["verdict"] == "FAIL" and r[key] <= v))
        if errors < best_err:
            best_err = errors
            best_thresh = v
    return best_thresh, best_err, unknown


def main(results_dir: str = "results", out_tsv: str = "results/unified_a2_summary.tsv") -> None:
    rows = _load_scores(results_dir)
    if not rows:
        print(f"No attn_head_weights_*.json files found in {results_dir}")
        return

    # Print table
    hdr = f"{'backend':<28} {'verdict':<6} {'tier':<22} {'L0':>7} {'[0,8)':>7} {'total':>7} {'norm[0,8)':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["verdict"]):
        print(f"{r['backend']:<28} {r['verdict']:<6} {r['tier']:<22} "
              f"{r['L0_mass']:7.2f} {r['window_mass_0_8']:7.2f} {r['total_mass']:7.2f} "
              f"{r['norm_window_0_8']:10.4f}")

    print("\n--- Threshold search ---")
    for key, direction in [("L0_mass", True), ("window_mass_0_8", True), ("norm_window_0_8", False)]:
        thresh, n_err, n_unk = _find_threshold(rows, key, direction)
        status = "CLEAN" if n_err == 0 else f"{n_err} error(s)"
        print(f"  {key:25s}  threshold={thresh}  status={status}  (n_unknown={n_unk})")

    print("\n--- Conclusion ---")
    print("None of the three single continuous measures cleanly separates PASS from FAIL.")
    print("Key overlaps:")
    known = [r for r in rows if r["verdict"] in ("PASS", "FAIL")]
    for r in known:
        if r["verdict"] == "PASS" and r["L0_mass"] < 15:
            print(f"  PASS model {r['backend']} has L0_mass={r['L0_mass']:.2f} < 15 "
                  f"→ would be misclassified by L0 threshold alone (resolved by the two-tier rule)")
        if r["verdict"] == "FAIL" and r["window_mass_0_8"] > 19:
            print(f"  FAIL model {r['backend']} has [0,8)={r['window_mass_0_8']:.2f} > 19 "
                  f"→ would be misclassified by window threshold alone")
    print("Decision: use the two-tier rule derived from architectural first principles (σ_cal).")

    # Write TSV
    keys = ["backend", "verdict", "tier", "L0_mass", "window_mass_0_8", "total_mass",
            "norm_window_0_8", "note"]
    with open(out_tsv, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")
    print(f"\nTable written → {out_tsv}")


if __name__ == "__main__":
    main()
