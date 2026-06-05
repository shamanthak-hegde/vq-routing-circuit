"""R-A-soft: Liquid scalar L0 scaling sweep — recover C6 number, or confirm binary.

Runs `probe.benchmarks.run_bench --knockout_mode scalar --knockout_layer 0`
on Liquid POPE-full for α ∈ {0.25, 0.5, 0.75} sequentially, then reports the
question-conditional sanity-check statistics on each output.

Decision rule per D-040: a usable C6 alpha must satisfy
  - overall logit-gap std > 1.0   (question-conditional signal preserved)
  - yes_rate ∈ [0.4, 0.6]         (no longer all-no, no longer all-yes)

If no α satisfies both, the conclusion is "L0 in Liquid is binary."

Skips AMBER intentionally: POPE-full (3000 records) is sufficient to decide
the binary-vs-graded question. The paper-strengthening session can run AMBER
on the winning α as a follow-up.

Outputs:
  - results/bench_pope_full_liquid_scalarL0_a{0.25,0.50,0.75}.jsonl (+ meta)
  - results/sanity_check_liquid_scalar_sweep.md
  - figures/sanity_check_liquid_scalar/logit_gap_pope_a{...}.png

Run (sae env):
    module load mamba && source activate sae
    python -m probe.tracing.liquid_scalar_l0_sweep
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
FIG_DIR = REPO / "figures" / "sanity_check_liquid_scalar"
REPORT = RESULTS / "sanity_check_liquid_scalar_sweep.md"

ALPHAS = (0.25, 0.50, 0.75)
BASELINE = RESULTS / "bench_pope_full_liquid_baseline.jsonl"
L0_ZERO = RESULTS / "bench_pope_full_liquid_L0.jsonl"

MODEL_PATH = "Junfeng5/Liquid_V1_7B"


def out_path_for(alpha: float) -> Path:
    return RESULTS / f"bench_pope_full_liquid_scalarL0_a{alpha:.2f}.jsonl"


def run_bench_one(alpha: float) -> None:
    out = out_path_for(alpha)
    if out.exists():
        # Use resume behaviour built into run_bench.py
        print(f"[α={alpha}] resume on existing {out.name}")
    cmd = [
        sys.executable, "-m", "probe.benchmarks.run_bench",
        "--bench", "pope_full",
        "--backend", "liquid",
        "--model_path", MODEL_PATH,
        "--knockout_mode", "scalar",
        "--knockout_layer", "0",
        "--alpha", f"{alpha:.2f}",
        "--out", str(out),
    ]
    print(f"[α={alpha}] launch:  {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(REPO))
    print(f"[α={alpha}] done in {(time.time()-t0)/60:.1f} m", flush=True)


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def stats(rows: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    yes_gaps: list[float] = []
    no_gaps: list[float] = []
    all_gaps: list[float] = []
    for r in rows:
        gt = r["gt_answer"].lower().strip()
        pred = r["pred"].lower().strip()
        gap = float(r["logit_yes"]) - float(r["logit_no"])
        all_gaps.append(gap)
        if gt == "yes":
            yes_gaps.append(gap)
        else:
            no_gaps.append(gap)
        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "no" and pred == "no":
            tn += 1
        elif gt == "yes" and pred == "no":
            fn += 1
    n = tp + fp + tn + fn
    yes_rate = (tp + fp) / n if n else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    p_y = tp / (tp + fp) if (tp + fp) else 0.0
    r_y = tp / (tp + fn) if (tp + fn) else 0.0
    f1_y = 2 * p_y * r_y / (p_y + r_y) if (p_y + r_y) else 0.0
    p_n = tn / (tn + fn) if (tn + fn) else 0.0
    r_n = tn / (tn + fp) if (tn + fp) else 0.0
    f1_n = 2 * p_n * r_n / (p_n + r_n) if (p_n + r_n) else 0.0

    def _agg(xs: list[float]) -> tuple[int, float, float]:
        if not xs:
            return 0, 0.0, 0.0
        return len(xs), statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)

    yn, ymean, ystd = _agg(yes_gaps)
    nn, nmean, nstd = _agg(no_gaps)
    an, amean, astd = _agg(all_gaps)
    return {
        "n": n,
        "yes_rate": yes_rate,
        "accuracy": accuracy,
        "f1_yes": f1_y,
        "f1_no": f1_n,
        "yes_gt": {"n": yn, "mean": ymean, "std": ystd},
        "no_gt": {"n": nn, "mean": nmean, "std": nstd},
        "all": {"n": an, "mean": amean, "std": astd},
    }


def verdict(s: dict) -> str:
    """Apply D-040 rule: usable C6 = std > 1.0 AND yes_rate ∈ [0.4, 0.6]."""
    std_ok = s["all"]["std"] > 1.0
    yr_ok = 0.4 <= s["yes_rate"] <= 0.6
    if std_ok and yr_ok:
        return "USABLE C6 (preserves question-conditional + bias near 50/50)"
    if std_ok:
        return f"PARTIAL — gap-std OK ({s['all']['std']:.2f}>1.0) but yes_rate={s['yes_rate']:.2f} outside [0.4,0.6]"
    if yr_ok:
        return f"PARTIAL — yes_rate OK ({s['yes_rate']:.2f}) but gap-std collapsed ({s['all']['std']:.2f}<1.0)"
    return f"FAIL — gap-std={s['all']['std']:.2f}, yes_rate={s['yes_rate']:.2f}"


def plot_hist(label: str, gaps_base: list[float], gaps_now: list[float], out: Path) -> None:
    lo = min(min(gaps_base), min(gaps_now))
    hi = max(max(gaps_base), max(gaps_now))
    bins = [lo + (hi - lo) * i / 80 for i in range(81)]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.hist(gaps_base, bins=bins, alpha=0.55, label=f"baseline (n={len(gaps_base)})",
            color="#1f77b4")
    ax.hist(gaps_now, bins=bins, alpha=0.75, label=label, color="#d62728")
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("logit(yes) − logit(no)")
    ax.set_ylabel("count")
    ax.set_title(f"Liquid POPE — {label}")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def build_report(per_alpha: dict[float, dict], base_stats: dict, l0_stats: dict) -> str:
    lines: list[str] = []
    lines.append("# Liquid scalar L0 scaling sweep (R-A-soft)\n")
    lines.append(
        "Tests whether a partial L0 ablation preserves Liquid's question-conditional "
        "signal while reducing yes-bias. Decision rule (D-040): a usable C6 α "
        "needs `gap-std > 1.0` AND `yes_rate ∈ [0.4, 0.6]`.\n"
    )

    lines.append("## Per-α metrics  (POPE adversarial, n=3000)\n")
    lines.append("| α | yes_rate | accuracy | F1_yes | F1_no | gap mean (yes-GT) | gap mean (no-GT) | gap mean Δ | gap std | verdict |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    def _row(label: str, s: dict) -> str:
        delta = s["yes_gt"]["mean"] - s["no_gt"]["mean"]
        return (f"| {label} | {s['yes_rate']:.3f} | {s['accuracy']:.3f} | "
                f"{s['f1_yes']:.3f} | {s['f1_no']:.3f} | "
                f"{s['yes_gt']['mean']:+.2f} | {s['no_gt']['mean']:+.2f} | "
                f"{delta:+.2f} | {s['all']['std']:.3f} | {verdict(s)} |")

    lines.append(_row("baseline (α=1.00)", base_stats))
    for alpha in sorted(per_alpha):
        lines.append(_row(f"{alpha:.2f}", per_alpha[alpha]))
    lines.append(_row("L0 zero (α=0.00)", l0_stats))
    lines.append("")

    best_alpha = None
    for alpha in sorted(per_alpha):
        s = per_alpha[alpha]
        if s["all"]["std"] > 1.0 and 0.4 <= s["yes_rate"] <= 0.6:
            best_alpha = alpha
            break
    lines.append("## Conclusion\n")
    if best_alpha is not None:
        lines.append(f"**Usable C6 α found: α={best_alpha:.2f}** — gap-std and yes-rate both pass.\n")
        lines.append(
            "Follow-up (not in R-A-soft scope): re-run on full AMBER under this α and "
            "report the per-class F1 alongside VILA-U's L1/VTI numbers.\n"
        )
    else:
        lines.append(
            "**No usable α.** L0 in Liquid behaves as a binary signal: any partial "
            "scaling that meaningfully shifts yes-rate also collapses gap-std below "
            "1.0. This is itself a publishable C3 prediction — collapsed-VQ models "
            "concentrate yes-signal entirely in L0, with no graded fallback. The "
            "paper's C6 evidence for Liquid stays C1-only (D-040).\n"
        )

    lines.append("\n---\n")
    lines.append("## Figures\n")
    for alpha in sorted(per_alpha):
        fig_path = FIG_DIR / f"logit_gap_pope_a{alpha:.2f}.png"
        lines.append(f"![logit-gap α={alpha}]({fig_path.relative_to(REPO)})\n")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_bench", action="store_true",
                    help="Skip GPU runs; just rebuild report from existing JSONLs")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_bench:
        for alpha in ALPHAS:
            run_bench_one(alpha)

    base_rows = load_rows(BASELINE)
    l0_rows = load_rows(L0_ZERO)
    base_stats = stats(base_rows)
    l0_stats = stats(l0_rows)
    base_gaps = [float(r["logit_yes"]) - float(r["logit_no"]) for r in base_rows]

    per_alpha: dict[float, dict] = {}
    for alpha in ALPHAS:
        path = out_path_for(alpha)
        if not path.exists():
            print(f"WARN: {path.name} missing; skipping in report")
            continue
        rows = load_rows(path)
        s = stats(rows)
        per_alpha[alpha] = s
        gaps_now = [float(r["logit_yes"]) - float(r["logit_no"]) for r in rows]
        plot_hist(f"α={alpha:.2f} (n={len(rows)})", base_gaps, gaps_now,
                  FIG_DIR / f"logit_gap_pope_a{alpha:.2f}.png")

    text = build_report(per_alpha, base_stats, l0_stats)
    REPORT.write_text(text)
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
