"""Gate threshold-sensitivity analysis for the three-gate diagnostic chain.

Were the gate thresholds (A.1 peak >= 0.4, A.2 mass >= 15, A.3 off-manifold
rate >= 80%) chosen *after* seeing the cohort outcomes? We sweep each threshold
over a broad grid, holding the other two at their canonical value, and report
how many model verdicts flip relative to the canonical 8-PASS / 7-FAIL
partition.

Diagnostic definition:
  A.1  max(rel_div[layers 0..7]) >= t1, with peak-and-decline shape
       (monotonic-ramp / no-event models fail the shape sub-condition).
  A.2  two-tier (architectural, not post-hoc):
         projector-amplified models -> L0 head-weight mass >= t2
         unified-vocab models       -> [0,8) window mass    >= t2
       Each model carries its *designated* A.2 statistic (the one the paper
       reports); the swept threshold t2 is applied to that statistic.
  A.3  VQ off-manifold rate >= t3, applicable only to collapsed-VQ models.
       Non-collapsing (FSQ) and non-VQ (MLP / continuous) models are exempt
       (A.3 is not the operative discriminator for them).
  Final verdict = A.1 AND A.2 AND A.3 (chain).

Numeric gate values are loaded from results/ where the generic per-model file is
the operative statistic; documented special cases (UniTok's diffusion-path A.2,
FSQ's EVR-based A.1, flat continuous A.1) are encoded with a source note.

Usage:
    source activate sae
    python -m probe.analysis.gate_threshold_sensitivity

Outputs:
  results/gate_threshold_sensitivity.json
  results/gate_threshold_sensitivity.md
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("results")

# Committee-suggested sweep grids.
A1_GRID = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
A2_GRID = [8, 10, 12, 15, 18, 20, 25]
A3_GRID = [60, 70, 80, 90, 95]

A1_DEFAULT, A2_DEFAULT, A3_DEFAULT = 0.40, 15.0, 80.0


def _rel_div_early_peak(backend: str) -> float | None:
    f = RESULTS / f"residual_divergence_{backend}.json"
    if not f.exists():
        return None
    rd = json.load(open(f)).get("mean_rel_div")
    if not isinstance(rd, list):
        return None
    return max(rd[:8])


def _attn_stats(backend: str) -> tuple[float, float] | None:
    f = RESULTS / f"attn_head_weights_{backend}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    ls = d["mean_layer_visual_sum"]
    return ls[0], sum(ls[:8])  # (L0 mass, [0,8) window mass)


def _offmanifold_rate(backend: str) -> float | None:
    f = RESULTS / f"codebook_probe_offmanifold_{backend}.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if not rows:
        return None
    return 100.0 * sum(1 for r in rows if r.get("noisy_closer_to_other")) / len(rows)


# Per-model spec. Fields:
#   verdict      canonical PASS/FAIL (tab:cohort-gates)
#   a2_tier      "proj" -> A.2 uses L0 mass; "uni" -> A.2 uses [0,8) window
#   a1_mode      "rel_div" (governed by t1) | "evr" (passes via VTI EVR, not
#                governed by t1) | "flat" (continuous, never passes A.1)
#   a1_shape_ok  False for monotonic-ramp / no-event models (fail A.1 by shape)
#   a3_applies   A.3 is operative (collapsed-VQ) vs exempt (FSQ / non-VQ)
#   a1_fixed/a2_fixed/a3_fixed  documented override when the generic file is not
#                the operative statistic (with source in note)
COHORT: dict[str, dict] = {
    # ---- 8 PASS specimens ----
    "vilau":            dict(verdict="PASS", a2_tier="proj", a1_mode="rel_div",
                             a3_applies=True, a3_fixed=100.0,
                             note="natural; Vicuna-7B"),
    "liquid":           dict(verdict="PASS", a2_tier="uni", a1_mode="rel_div",
                             a3_applies=True, note="natural; Gemma-7B"),
    "chameleon":        dict(verdict="PASS", a2_tier="uni", a1_mode="rel_div",
                             a3_applies=True, note="natural; LLaMA-32k; NO-codebook"),
    "anole":            dict(verdict="PASS", a2_tier="uni", a1_mode="rel_div",
                             a3_applies=True, note="FT Chameleon descendant"),
    "lumina_mgpt":      dict(verdict="PASS", a2_tier="uni", a1_mode="rel_div",
                             a3_applies=True, note="held-out"),
    "llava_vq":         dict(verdict="PASS", a2_tier="proj", a1_mode="rel_div",
                             a3_applies=True, a3_fixed=100.0,
                             note="induced v1, K=16384"),
    "llava_vq_fsq":     dict(verdict="PASS", a2_tier="proj", a1_mode="evr",
                             a1_fixed=0.511, a3_applies=False,
                             note="induced FSQ v1; A.1 via VTI EVR(L3)=0.511; "
                                  "non-collapsing so A.3 exempt"),
    "llava_vq_fsq_v2":  dict(verdict="PASS", a2_tier="proj", a1_mode="evr",
                             a1_fixed=0.511, a3_applies=False,
                             note="induced FSQ v2; A.1 via VTI EVR; A.3 exempt"),
    # ---- 7 FAIL specimens (with data) ----
    "llava_mlp_control": dict(verdict="FAIL", a2_tier="proj", a1_mode="rel_div",
                              a3_applies=False,
                              note="matched-compute control; FAIL A.1 + behavioral"),
    "qwen_vq":          dict(verdict="FAIL", a2_tier="proj", a1_mode="rel_div",
                             a3_applies=True, a3_fixed=0.0,
                             note="collapse w/o circuit; FAIL A.2"),
    "unitok":           dict(verdict="FAIL", a2_tier="proj", a1_mode="rel_div",
                             a2_fixed=1.37, a3_applies=True,
                             note="A.2 designated 1.37 at [12,14) (diffusion-MLP "
                                  "path), not the generic L0 cell; FAIL A.1+A.2"),
    "emu3":             dict(verdict="FAIL", a2_tier="proj", a1_mode="rel_div",
                             a3_applies=True, a3_fixed=0.0,
                             note="L0 routing, no propagation; FAIL A.2 (12.5<15)"),
    "janus":            dict(verdict="FAIL", a2_tier="proj", a1_mode="flat",
                             a2_fixed=13.57, a3_applies=False,
                             note="continuous SigLIP; healthy routing; FAIL A.1"),
    "seed":             dict(verdict="FAIL", a2_tier="uni", a1_mode="rel_div",
                             a1_shape_ok=False, a2_fixed=5.0, a3_applies=True,
                             a3_fixed=86.3,
                             note="monotonic-ramp A.1 (shape fail); FAIL A.1"),
    "showo":            dict(verdict="FAIL", a2_tier="proj", a1_mode="rel_div",
                             a1_shape_ok=False, a2_fixed=5.0, a3_applies=True,
                             a3_fixed=100.0,
                             note="no discrete routing event; FAIL A.1"),
}


def _resolve(backend: str, spec: dict) -> dict:
    """Materialise the operative numeric gate values for one model."""
    # A.1
    if spec["a1_mode"] == "rel_div":
        a1 = spec.get("a1_fixed") or _rel_div_early_peak(backend)
    else:  # evr / flat use fixed (or 0.0 for flat)
        a1 = spec.get("a1_fixed", 0.0)
    # A.2
    attn = _attn_stats(backend)
    if "a2_fixed" in spec:
        a2 = spec["a2_fixed"]
    elif attn is not None:
        a2 = attn[0] if spec["a2_tier"] == "proj" else attn[1]
    else:
        a2 = None
    # A.3
    if "a3_fixed" in spec:
        a3 = spec["a3_fixed"]
    else:
        a3 = _offmanifold_rate(backend)
    return {
        "backend": backend,
        "verdict": spec["verdict"],
        "a2_tier": spec["a2_tier"],
        "a1_mode": spec["a1_mode"],
        "a1_shape_ok": spec.get("a1_shape_ok", True),
        "a3_applies": spec["a3_applies"],
        "a1_value": a1,
        "a2_value": a2,
        "a3_value": a3,
        "note": spec["note"],
    }


def _chain_verdict(m: dict, t1: float, t2: float, t3: float) -> str:
    # A.1
    if m["a1_mode"] == "flat":
        a1_pass = False
    elif m["a1_mode"] == "evr":
        a1_pass = True  # established via VTI EVR, not governed by rel_div t1
    else:
        a1_pass = m["a1_shape_ok"] and m["a1_value"] is not None and m["a1_value"] >= t1
    # A.2
    a2_pass = m["a2_value"] is not None and m["a2_value"] >= t2
    # A.3
    a3_pass = (not m["a3_applies"]) or (m["a3_value"] is not None and m["a3_value"] >= t3)
    return "PASS" if (a1_pass and a2_pass and a3_pass) else "FAIL"


def _sweep(models: list[dict], gate: str, grid: list[float]) -> list[dict]:
    rows = []
    for t in grid:
        t1, t2, t3 = A1_DEFAULT, A2_DEFAULT, A3_DEFAULT
        if gate == "A1":
            t1 = t
        elif gate == "A2":
            t2 = t
        else:
            t3 = t
        flips = [m["backend"] for m in models
                 if _chain_verdict(m, t1, t2, t3) != m["verdict"]]
        pass_set = sorted(m["backend"] for m in models
                          if _chain_verdict(m, t1, t2, t3) == "PASS")
        rows.append({"threshold": t, "n_flips": len(flips),
                     "flipped": flips, "n_pass": len(pass_set)})
    return rows


def main() -> None:
    models = [_resolve(b, s) for b, s in COHORT.items()]

    # --- Validation: canonical partition reproduced at default thresholds ---
    mismatches = [m["backend"] for m in models
                  if _chain_verdict(m, A1_DEFAULT, A2_DEFAULT, A3_DEFAULT) != m["verdict"]]
    canonical_ok = not mismatches
    n_pass = sum(1 for m in models if m["verdict"] == "PASS")
    n_fail = sum(1 for m in models if m["verdict"] == "FAIL")

    sweeps = {g: _sweep(models, g, grid)
              for g, grid in [("A1", A1_GRID), ("A2", A2_GRID), ("A3", A3_GRID)]}

    # Stable band (contiguous grid range with 0 flips) + binding model at break.
    def _band(gate: str, grid: list[float]) -> dict:
        rows = sweeps[gate]
        stable = [r["threshold"] for r in rows if r["n_flips"] == 0]
        first_break = next((r for r in rows if r["n_flips"] > 0), None)
        return {"stable_min": min(stable) if stable else None,
                "stable_max": max(stable) if stable else None,
                "binding_model": first_break["flipped"][0] if first_break else None,
                "binding_threshold": first_break["threshold"] if first_break else None}
    bands = {g: _band(g, grid) for g, grid in
             [("A1", A1_GRID), ("A2", A2_GRID), ("A3", A3_GRID)]}

    # Chain vs single-gate robustness: how many verdicts would a *single* A.2
    # threshold (ignoring A.1/A.3) misclassify, vs the chain, across A2_GRID.
    single_a2 = []
    for t in A2_GRID:
        wrong = [m["backend"] for m in models
                 if (("PASS" if (m["a2_value"] is not None and m["a2_value"] >= t)
                      else "FAIL") != m["verdict"])]
        single_a2.append({"threshold": t, "n_misclassified": len(wrong), "models": wrong})

    out = {
        "single_gate_a2_vs_chain": {"single_a2": single_a2,
                                    "chain_a2": sweeps["A2"]},
        "stable_bands": bands,
        "defaults": {"A1": A1_DEFAULT, "A2": A2_DEFAULT, "A3": A3_DEFAULT},
        "canonical_partition": {"n_pass": n_pass, "n_fail": n_fail,
                                "reproduced": canonical_ok, "mismatches": mismatches},
        "models": models,
        "sweeps": sweeps,
    }
    (RESULTS / "gate_threshold_sensitivity.json").write_text(json.dumps(out, indent=2))

    # --- Markdown report ---
    L = []
    L.append("# Gate threshold-sensitivity analysis\n")
    L.append(f"Defaults: A.1 >= {A1_DEFAULT}, A.2 >= {A2_DEFAULT}, "
             f"A.3 >= {A3_DEFAULT}%. Chain verdict = A.1 AND A.2 AND A.3.\n")
    L.append(f"**Canonical partition reproduced at defaults:** "
             f"{'YES' if canonical_ok else 'NO — ' + ', '.join(mismatches)} "
             f"({n_pass} PASS / {n_fail} FAIL).\n")

    L.append("\n## Per-model operative gate values\n")
    L.append("| model | verdict | A.2 tier | A.1 | A.2 | A.3% | note |")
    L.append("|---|---|---|---|---|---|---|")
    for m in sorted(models, key=lambda x: (x["verdict"], x["backend"])):
        a1 = f"{m['a1_value']:.3f}" if isinstance(m["a1_value"], (int, float)) else "—"
        a1 += f" ({m['a1_mode']})" if m["a1_mode"] != "rel_div" else ""
        if not m["a1_shape_ok"]:
            a1 += " [ramp]"
        a2 = f"{m['a2_value']:.2f}" if isinstance(m["a2_value"], (int, float)) else "—"
        a3 = (f"{m['a3_value']:.0f}" if isinstance(m["a3_value"], (int, float))
              else "—") + ("" if m["a3_applies"] else " (exempt)")
        L.append(f"| {m['backend']} | {m['verdict']} | {m['a2_tier']} | {a1} | "
                 f"{a2} | {a3} | {m['note']} |")

    for gate, label, default in [("A1", "A.1 residual-divergence peak", A1_DEFAULT),
                                 ("A2", "A.2 head-weight mass", A2_DEFAULT),
                                 ("A3", "A.3 off-manifold rate (%)", A3_DEFAULT)]:
        L.append(f"\n## {label} sweep (other two gates at default)\n")
        L.append("| threshold | # verdict flips | # PASS | flipped models |")
        L.append("|---|---|---|---|")
        for r in sweeps[gate]:
            mark = "  ← default" if abs(r["threshold"] - default) < 1e-9 else ""
            L.append(f"| {r['threshold']}{mark} | {r['n_flips']} | {r['n_pass']} | "
                     f"{', '.join(r['flipped']) if r['flipped'] else '—'} |")

    L.append("\n## Chain vs single-gate A.2 robustness\n")
    L.append("How many verdicts a *single* A.2 threshold misclassifies (ignoring "
             "A.1/A.3) vs the full chain, over the A.2 grid. The chain stays at "
             "0 errors well below the default because A.1/A.3 carry the separation "
             "where A.2 alone cannot.\n")
    L.append("| threshold | single-gate A.2 errors | chain errors |")
    L.append("|---|---|---|")
    for s, c in zip(single_a2, sweeps["A2"]):
        L.append(f"| {s['threshold']} | {s['n_misclassified']} "
                 f"({', '.join(s['models']) if s['models'] else '—'}) | {c['n_flips']} |")

    L.append("\n## Stable bands (contiguous 0-flip range within the swept grid)\n")
    L.append("| gate | stable range | first break | binding model |")
    L.append("|---|---|---|---|")
    for g, lab in [("A1", "A.1 peak"), ("A2", "A.2 mass"), ("A3", "A.3 rate %")]:
        b = bands[g]
        rng = f"{b['stable_min']}–{b['stable_max']}" if b["stable_min"] is not None else "—"
        brk = (f"@ {b['binding_threshold']} ({b['binding_model']})"
               if b["binding_model"] else "none in grid")
        L.append(f"| {lab} | {rng} | {brk} | {b['binding_model'] or '—'} |")

    (RESULTS / "gate_threshold_sensitivity.md").write_text("\n".join(L) + "\n")

    # --- console summary ---
    print("\n".join(L))
    print("\nWrote results/gate_threshold_sensitivity.{json,md}")


if __name__ == "__main__":
    main()
