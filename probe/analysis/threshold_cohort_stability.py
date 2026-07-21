"""Threshold-range cohort verdict stability.

Asks whether the cohort PASS/FAIL partition is stable when the two numeric gate
thresholds are perturbed across a wide band, rather than whether any individual
model sits near its gate. This is a pure re-analysis over the existing
residual-divergence and head-weight-mass curves (no GPU, no regeneration); it
reuses the per-model gate machinery of probe.analysis.gate_threshold_sensitivity
so the chain-verdict definition is identical.

Scope (deliberately narrow):
  * A.1 residual-divergence peak threshold swept over [0.20, 0.60].
  * A.2 head-weight mass threshold swept over [8, 25].
  * We report ONLY cohort-level verdict stability — the size of the PASS set and,
    critically, the *direction* of any verdict change:
      - false_positive : a canonically-FAIL model that the chain would ADMIT
                         (FAIL -> PASS). This is the dangerous direction: it would
                         mean the thresholds manufacture specimens.
      - lost           : a canonically-PASS model that the chain would DROP
                         (PASS -> FAIL) because the bar rose above its statistic.
  * No per-model gate values and no binding-model identities are emitted here;
    see gate_threshold_sensitivity.py for the per-model table.

The A.3 gate is held at its canonical value throughout: this analysis concerns
the two continuously-valued gates (A.1 divergence, A.2 mass); A.3 is an off-manifold rate
that is either ~100% (collapsed VQ) or ~0% (everything else) and carries no
threshold ambiguity in the swept range (A.3 sweep: 0 flips over [60,95]).

Usage:
    source activate sae
    python -m probe.analysis.threshold_cohort_stability

Outputs:
  results/threshold_cohort_stability.json
  results/threshold_cohort_stability.md
"""

from __future__ import annotations

import json
from pathlib import Path

from probe.analysis.gate_threshold_sensitivity import (
    COHORT,
    A1_DEFAULT,
    A2_DEFAULT,
    A3_DEFAULT,
    _resolve,
    _chain_verdict,
)

RESULTS = Path("results")

# W4-requested ranges.
A1_LO, A1_HI, A1_STEP = 0.20, 0.60, 0.05
A2_LO, A2_HI, A2_STEP = 8, 25, 1


def _frange(lo: float, hi: float, step: float) -> list[float]:
    n = round((hi - lo) / step)
    return [round(lo + i * step, 4) for i in range(n + 1)]


def _cohort_row(models: list[dict], t1: float, t2: float, t3: float) -> dict:
    """Cohort-level summary at one (t1,t2,t3); no per-model values leaked."""
    n_pass = 0
    false_pos = 0   # FAIL -> PASS   (thresholds admit a non-specimen)
    lost = 0        # PASS -> FAIL   (thresholds drop a genuine specimen)
    for m in models:
        v = _chain_verdict(m, t1, t2, t3)
        if v == "PASS":
            n_pass += 1
        if v != m["verdict"]:
            if m["verdict"] == "FAIL":
                false_pos += 1
            else:
                lost += 1
    return {
        "n_pass": n_pass,
        "n_flips": false_pos + lost,
        "false_positive": false_pos,
        "lost": lost,
    }


def _sweep_axis(models: list[dict], axis: str, grid: list[float]) -> list[dict]:
    rows = []
    for t in grid:
        t1, t2, t3 = A1_DEFAULT, A2_DEFAULT, A3_DEFAULT
        if axis == "A1":
            t1 = t
        else:
            t2 = t
        row = _cohort_row(models, t1, t2, t3)
        row["threshold"] = t
        rows.append(row)
    return rows


def _band(rows: list[dict], key: str) -> tuple[float | None, float | None]:
    """Contiguous threshold range with 0 of `key` (e.g. false_positive / n_flips)."""
    ok = [r["threshold"] for r in rows if r[key] == 0]
    return (min(ok), max(ok)) if ok else (None, None)


def main() -> None:
    models = [_resolve(b, s) for b, s in COHORT.items()]
    n_pass0 = sum(1 for m in models if m["verdict"] == "PASS")
    n_fail0 = sum(1 for m in models if m["verdict"] == "FAIL")

    a1_grid = _frange(A1_LO, A1_HI, A1_STEP)
    a2_grid = _frange(A2_LO, A2_HI, A2_STEP)
    a1_rows = _sweep_axis(models, "A1", a1_grid)
    a2_rows = _sweep_axis(models, "A2", [int(x) for x in a2_grid])

    # Full 2-D grid: does ANY (t1,t2) combination in the requested box ever admit a
    # canonically-FAIL model? This is the strongest cohort-stability statement.
    grid2d_false_pos = 0
    grid2d_cells = 0
    worst_false_pos = 0
    for t1 in a1_grid:
        for t2 in a2_grid:
            r = _cohort_row(models, t1, int(t2), A3_DEFAULT)
            grid2d_cells += 1
            if r["false_positive"] > 0:
                grid2d_false_pos += 1
            worst_false_pos = max(worst_false_pos, r["false_positive"])

    a1_fp_band = _band(a1_rows, "false_positive")
    a1_flip_band = _band(a1_rows, "n_flips")
    a2_fp_band = _band(a2_rows, "false_positive")
    a2_flip_band = _band(a2_rows, "n_flips")

    out = {
        "canonical": {"n_pass": n_pass0, "n_fail": n_fail0,
                      "defaults": {"A1": A1_DEFAULT, "A2": A2_DEFAULT, "A3": A3_DEFAULT}},
        "ranges": {"A1": [A1_LO, A1_HI], "A2": [A2_LO, A2_HI]},
        "A1_sweep": a1_rows,
        "A2_sweep": a2_rows,
        "grid_2d": {"cells": grid2d_cells,
                    "cells_with_false_positive": grid2d_false_pos,
                    "max_false_positive_in_any_cell": worst_false_pos},
        "bands": {
            "A1_no_false_positive": a1_fp_band,
            "A1_no_flip": a1_flip_band,
            "A2_no_false_positive": a2_fp_band,
            "A2_no_flip": a2_flip_band,
        },
    }
    (RESULTS / "threshold_cohort_stability.json").write_text(json.dumps(out, indent=2))

    L: list[str] = []
    L.append("# Threshold-range cohort verdict stability\n")
    L.append(f"Canonical partition: {n_pass0} PASS / {n_fail0} FAIL at the reported "
             f"thresholds (A.1={A1_DEFAULT}, A.2={A2_DEFAULT}, A.3={A3_DEFAULT}%). "
             "Chain verdict = A.1 AND A.2 AND A.3. A.3 held at its canonical value "
             "(it carries no threshold ambiguity in this range).\n")
    L.append("We report cohort-level verdict stability only. `false_pos` counts "
             "canonically-FAIL models the chain would *admit* (FAIL→PASS, the "
             "dangerous direction); `lost` counts canonically-PASS models the chain "
             "would *drop* (PASS→FAIL) as the bar rises above their statistic.\n")

    L.append(f"\n## A.1 residual-divergence peak threshold ∈ [{A1_LO}, {A1_HI}] "
             "(A.2, A.3 at default)\n")
    L.append("| A.1 threshold | # PASS | false_pos | lost |")
    L.append("|---|---|---|---|")
    for r in a1_rows:
        mark = "  ← reported" if abs(r["threshold"] - A1_DEFAULT) < 1e-9 else ""
        L.append(f"| {r['threshold']:.2f}{mark} | {r['n_pass']} | "
                 f"{r['false_positive']} | {r['lost']} |")

    L.append(f"\n## A.2 head-weight mass threshold ∈ [{A2_LO}, {A2_HI}] "
             "(A.1, A.3 at default)\n")
    L.append("| A.2 threshold | # PASS | false_pos | lost |")
    L.append("|---|---|---|---|")
    for r in a2_rows:
        mark = "  ← reported" if abs(r["threshold"] - A2_DEFAULT) < 1e-9 else ""
        L.append(f"| {int(r['threshold'])}{mark} | {r['n_pass']} | "
                 f"{r['false_positive']} | {r['lost']} |")

    fp2d = out["grid_2d"]["cells_with_false_positive"]
    L.append("\n## Joint 2-D threshold box\n")
    L.append(f"Over all {out['grid_2d']['cells']} (A.1, A.2) combinations in the box "
             f"[{A1_LO},{A1_HI}] × [{A2_LO},{A2_HI}], the number of cells that admit "
             f"*any* canonically-FAIL model (FAIL→PASS) is **{fp2d}**; the maximum "
             f"such admissions in a single cell is "
             f"{out['grid_2d']['max_false_positive_in_any_cell']}.\n")

    def _fmt(b: tuple[float | None, float | None]) -> str:
        return f"[{b[0]}, {b[1]}]" if b[0] is not None else "none"

    L.append("\n## Stability bands (contiguous 0-count sub-range within the swept range)\n")
    L.append("| axis | no FAIL→PASS over | no verdict change over |")
    L.append("|---|---|---|")
    L.append(f"| A.1 | {_fmt(a1_fp_band)} | {_fmt(a1_flip_band)} |")
    L.append(f"| A.2 | {_fmt(a2_fp_band)} | {_fmt(a2_flip_band)} |")

    (RESULTS / "threshold_cohort_stability.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nWrote results/threshold_cohort_stability.{json,md}")


if __name__ == "__main__":
    main()
