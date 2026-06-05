"""R-A/A2/A3: L0-ablation degenerate-emitter sanity check (no-GPU report).

Reads existing per-model POPE/AMBER baseline + L0-knockout JSONLs and emits:
  - results/sanity_check_<model>_report.md         (canonical record per node)
  - figures/sanity_check_<model>/logit_gap_<split>.png

Run:
    python -m probe.tracing.liquid_sanity_report                    # default: liquid (R-A)
    python -m probe.tracing.liquid_sanity_report --model chameleon  # R-A2
    python -m probe.tracing.liquid_sanity_report --model vilau      # R-A3
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"


# Per-model configuration.  L0 file names differ slightly across backends
# because the bench harness was used inconsistently — VILA-U's AMBER L0 file
# is `bench_amber_vilau_intervention.jsonl` (mode=full_zero, layer=0); the
# others follow the `bench_<bench>_<model>_L0.jsonl` convention.
MODEL_CONFIGS: dict[str, dict] = {
    "liquid": {
        "node": "R-A",
        "pope": {
            "baseline": RESULTS / "bench_pope_full_liquid_baseline.jsonl",
            "L0": RESULTS / "bench_pope_full_liquid_L0.jsonl",
        },
        "amber": {
            "baseline": RESULTS / "bench_amber_liquid_baseline.jsonl",
            "L0": RESULTS / "bench_amber_liquid_L0.jsonl",
        },
        "gen_jsonl": RESULTS / "sanity_check_liquid_generated.jsonl",
    },
    "chameleon": {
        "node": "R-A2",
        "pope": {
            "baseline": RESULTS / "bench_pope_full_chameleon_baseline.jsonl",
            "L0": RESULTS / "bench_pope_full_chameleon_L0.jsonl",
        },
    },
    "vilau": {
        "node": "R-A3",
        "pope": {
            "baseline": RESULTS / "bench_pope_full_vilau_baseline.jsonl",
            "L0": RESULTS / "bench_pope_full_vilau_L0.jsonl",
        },
        "amber": {
            "baseline": RESULTS / "bench_amber_vilau_baseline.jsonl",
            "L0": RESULTS / "bench_amber_vilau_intervention.jsonl",
        },
    },
}


@dataclass
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def yes_rate(self) -> float:
        return (self.tp + self.fp) / self.n if self.n else 0.0

    def _f1(self, p_num: int, p_den: int, r_num: int, r_den: int) -> tuple[float, float, float]:
        p = p_num / p_den if p_den else 0.0
        r = r_num / r_den if r_den else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    def yes_f1(self) -> tuple[float, float, float]:
        return self._f1(self.tp, self.tp + self.fp, self.tp, self.tp + self.fn)

    def no_f1(self) -> tuple[float, float, float]:
        return self._f1(self.tn, self.tn + self.fn, self.tn, self.tn + self.fp)


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def confusion(rows: list[dict]) -> Confusion:
    tp = fp = tn = fn = 0
    for r in rows:
        gt = r["gt_answer"].lower().strip()
        pred = r["pred"].lower().strip()
        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "no" and pred == "no":
            tn += 1
        elif gt == "yes" and pred == "no":
            fn += 1
    return Confusion(tp, fp, tn, fn)


@dataclass
class GapStats:
    n: int
    mean: float
    std: float
    lo: float
    hi: float
    p10: float
    p50: float
    p90: float

    @classmethod
    def from_list(cls, xs: list[float]) -> "GapStats":
        xs_sorted = sorted(xs)
        n = len(xs_sorted)

        def pct(p: float) -> float:
            if not xs_sorted:
                return 0.0
            i = max(0, min(n - 1, int(round(p * (n - 1)))))
            return xs_sorted[i]

        return cls(
            n=n,
            mean=statistics.mean(xs) if xs else 0.0,
            std=statistics.stdev(xs) if n > 1 else 0.0,
            lo=min(xs) if xs else 0.0,
            hi=max(xs) if xs else 0.0,
            p10=pct(0.10),
            p50=pct(0.50),
            p90=pct(0.90),
        )


def split_gaps(rows: list[dict]) -> tuple[GapStats, GapStats, GapStats]:
    yes_gaps, no_gaps, all_gaps = [], [], []
    for r in rows:
        gap = float(r["logit_yes"]) - float(r["logit_no"])
        all_gaps.append(gap)
        if r["gt_answer"].lower().strip() == "yes":
            yes_gaps.append(gap)
        else:
            no_gaps.append(gap)
    return (
        GapStats.from_list(yes_gaps),
        GapStats.from_list(no_gaps),
        GapStats.from_list(all_gaps),
    )


def plot_gap_hist(model: str, split: str, baseline_rows: list[dict],
                  l0_rows: list[dict], out: Path) -> None:
    base_gaps = [float(r["logit_yes"]) - float(r["logit_no"]) for r in baseline_rows]
    l0_gaps = [float(r["logit_yes"]) - float(r["logit_no"]) for r in l0_rows]

    lo = min(min(base_gaps), min(l0_gaps))
    hi = max(max(base_gaps), max(l0_gaps))
    bins = [lo + (hi - lo) * i / 80 for i in range(81)]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.hist(base_gaps, bins=bins, alpha=0.55, label=f"baseline (n={len(base_gaps)})",
            color="#1f77b4")
    ax.hist(l0_gaps, bins=bins, alpha=0.75, label=f"L0 knockout (n={len(l0_gaps)})",
            color="#d62728")
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("logit(yes) − logit(no)")
    ax.set_ylabel("count")
    ax.set_title(f"{model} logit-gap distribution — {split.upper()}")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fmt_pct(x: float) -> str:
    return f"{100 * x:5.2f}%"


def build_report(model: str, fig_dir: Path) -> str:
    cfg = MODEL_CONFIGS[model]
    splits = {k: v for k, v in cfg.items() if k in ("pope", "amber")}

    lines: list[str] = []
    lines.append(
        f"# {model} L0-ablation sanity-check report ({cfg['node']})\n"
    )
    lines.append(
        f"Computed by `probe/tracing/liquid_sanity_report.py --model {model}`. "
        f"Inputs are existing `bench_*_{model}_*` JSONLs; no GPU, no resweep.\n"
    )

    for split, files in splits.items():
        base_rows = load_rows(files["baseline"])
        l0_rows = load_rows(files["L0"])
        base_cm = confusion(base_rows)
        l0_cm = confusion(l0_rows)

        bp_y, br_y, bf_y = base_cm.yes_f1()
        bp_n, br_n, bf_n = base_cm.no_f1()
        lp_y, lr_y, lf_y = l0_cm.yes_f1()
        lp_n, lr_n, lf_n = l0_cm.no_f1()

        lines.append(f"## {split.upper()}\n")
        lines.append(f"- baseline rows: `{files['baseline'].relative_to(REPO)}` (n={base_cm.n})")
        lines.append(f"- L0 knockout rows: `{files['L0'].relative_to(REPO)}` (n={l0_cm.n})\n")

        lines.append("### Confusion matrix\n")
        lines.append("| condition | TP | FP | TN | FN | yes-pred | no-pred |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        lines.append(f"| baseline | {base_cm.tp} | {base_cm.fp} | {base_cm.tn} | {base_cm.fn} | "
                     f"{base_cm.tp + base_cm.fp} | {base_cm.tn + base_cm.fn} |")
        lines.append(f"| L0 knockout | {l0_cm.tp} | {l0_cm.fp} | {l0_cm.tn} | {l0_cm.fn} | "
                     f"{l0_cm.tp + l0_cm.fp} | {l0_cm.tn + l0_cm.fn} |")
        lines.append("")

        lines.append("### Per-class metrics\n")
        lines.append("| condition | accuracy | yes_rate | YES P / R / F1 | NO P / R / F1 |")
        lines.append("|---|---:|---:|---|---|")
        lines.append(f"| baseline | {fmt_pct(base_cm.accuracy)} | {fmt_pct(base_cm.yes_rate)} | "
                     f"{bp_y:.3f} / {br_y:.3f} / **{bf_y:.3f}** | "
                     f"{bp_n:.3f} / {br_n:.3f} / **{bf_n:.3f}** |")
        lines.append(f"| L0 knockout | {fmt_pct(l0_cm.accuracy)} | {fmt_pct(l0_cm.yes_rate)} | "
                     f"{lp_y:.3f} / {lr_y:.3f} / **{lf_y:.3f}** | "
                     f"{lp_n:.3f} / {lr_n:.3f} / **{lf_n:.3f}** |")
        lines.append("")

        ys_b, ns_b, all_b = split_gaps(base_rows)
        ys_l, ns_l, all_l = split_gaps(l0_rows)

        lines.append("### Logit-gap distribution  (logit_yes − logit_no)\n")
        lines.append("| condition | GT | n | mean | std | min | p10 | p50 | p90 | max |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for label, cond, gs in [
            ("baseline", "yes", ys_b), ("baseline", "no", ns_b),
            ("baseline", "all", all_b),
            ("L0 knockout", "yes", ys_l), ("L0 knockout", "no", ns_l),
            ("L0 knockout", "all", all_l),
        ]:
            lines.append(
                f"| {label} | {cond} | {gs.n} | {gs.mean:+.3f} | {gs.std:.3f} | "
                f"{gs.lo:+.2f} | {gs.p10:+.2f} | {gs.p50:+.2f} | {gs.p90:+.2f} | {gs.hi:+.2f} |"
            )
        lines.append("")

        gap_means_match = abs(ys_l.mean - ns_l.mean) < 0.05
        std_collapsed = all_l.std < 0.5
        verdict = (
            "**DEGENERATE-EMITTER signature confirmed**" if gap_means_match and std_collapsed
            else "question-conditional signal partially preserved"
        )
        lines.append(f"**Verdict ({split.upper()}):** {verdict} — "
                     f"yes-GT gap mean {ys_l.mean:+.2f} vs no-GT gap mean {ns_l.mean:+.2f} "
                     f"(difference {ys_l.mean - ns_l.mean:+.3f} logits); "
                     f"overall std {all_l.std:.3f} under L0 vs {all_b.std:.3f} baseline "
                     f"({all_b.std / max(all_l.std, 1e-9):.1f}× collapse).\n")

        fig_path = fig_dir / f"logit_gap_{split}.png"
        plot_gap_hist(model, split, base_rows, l0_rows, fig_path)
        lines.append(f"![logit-gap histogram]({fig_path.relative_to(REPO)})\n")

    gen_jsonl = cfg.get("gen_jsonl")
    if gen_jsonl is not None and gen_jsonl.exists():
        lines.append("## Generated text on 20 random records (GPU)\n")
        lines.append(
            f"Source: `{gen_jsonl.relative_to(REPO)}` — produced by "
            "`probe/tracing/liquid_sanity_generate.py` (run_prefill for logits "
            "+ run_generate for text, max_new_tokens=20, both passes under the "
            "same intervention).\n"
        )
        gen_rows: list[dict] = []
        with gen_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    gen_rows.append(json.loads(line))

        l0_empty = sum(1 for r in gen_rows if not r["L0_text"].strip())
        base_empty = sum(1 for r in gen_rows if not r["baseline_text"].strip())
        l0_starts_yes = sum(
            1 for r in gen_rows if r["L0_text"].strip().lower().startswith("yes")
        )
        l0_starts_no = sum(
            1 for r in gen_rows if r["L0_text"].strip().lower().startswith("no")
        )
        base_starts_yes = sum(
            1 for r in gen_rows
            if r["baseline_text"].strip().lower().startswith("yes")
        )
        base_starts_no = sum(
            1 for r in gen_rows
            if r["baseline_text"].strip().lower().startswith("no")
        )

        lines.append("### Aggregate behaviour\n")
        lines.append("| condition | n | empty | starts 'yes' | starts 'no' | other |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        lines.append(
            f"| baseline | {len(gen_rows)} | {base_empty} | {base_starts_yes} | "
            f"{base_starts_no} | "
            f"{len(gen_rows) - base_empty - base_starts_yes - base_starts_no} |"
        )
        lines.append(
            f"| L0 knockout | {len(gen_rows)} | **{l0_empty}** | {l0_starts_yes} | "
            f"{l0_starts_no} | "
            f"{len(gen_rows) - l0_empty - l0_starts_yes - l0_starts_no} |"
        )
        lines.append("")
        lines.append(
            "All 20 L0 outputs are empty strings after `skip_special_tokens=True`. "
            "Raw token-ID inspection (see handoff §3) shows the model emits token "
            "108 (`\\n`) repeatedly under L0 ablation — pure whitespace, not even "
            "the literal token 'no'. The argmax over (yes_id, no_id) reads 'no' "
            "only because the *gap* (logit_yes − logit_no) is slightly negative; "
            "neither is the top-1 prediction.\n"
        )

        lines.append("### Per-record table (truncated to 30 chars per text)\n")
        lines.append(
            "| record | gt | base_gap | baseline_text | L0_gap | L0_text |"
        )
        lines.append("|---|---|---:|---|---:|---|")
        for r in gen_rows:
            bt = r["baseline_text"].replace("|", "/")[:30]
            lt = r["L0_text"].replace("|", "/")[:30] or "*(empty)*"
            lines.append(
                f"| {r['record_id']} ({r['source']}) | {r['gt_answer']} | "
                f"{r['baseline_gap']:+.2f} | {bt} | {r['L0_gap']:+.2f} | {lt} |"
            )
        lines.append("")

    lines.append("---\n")
    lines.append("## Mechanical verdict (computed)\n")

    verdicts: list[str] = []
    for split, files in splits.items():
        base_rows = load_rows(files["baseline"])
        l0_rows = load_rows(files["L0"])
        _, _, all_b = split_gaps(base_rows)
        ys_l, ns_l, all_l = split_gaps(l0_rows)
        gap_means_match = abs(ys_l.mean - ns_l.mean) < 0.05
        std_collapsed = all_l.std < 0.5
        collapse_ratio = all_b.std / max(all_l.std, 1e-9)
        if gap_means_match and std_collapsed:
            tag = "DEGENERATE-EMITTER"
        elif collapse_ratio > 3.0 or std_collapsed:
            tag = "PARTIAL DEGENERATION"
        else:
            tag = "QUESTION-CONDITIONAL SIGNAL PRESERVED"
        verdicts.append(
            f"- **{split.upper()}** → **{tag}**  "
            f"(gap-mean Δ={ys_l.mean - ns_l.mean:+.3f}, "
            f"gap-std {all_b.std:.2f} → {all_l.std:.2f}, "
            f"{collapse_ratio:.1f}× collapse)"
        )
    lines.extend(verdicts)
    lines.append("")
    lines.append(
        "*Thresholds:* DEGENERATE-EMITTER requires yes/no gap-mean Δ < 0.05 AND "
        "L0 gap-std < 0.5; PARTIAL DEGENERATION fires on either >3× std collapse "
        "OR L0 std < 0.5 alone; otherwise the question-conditional signal is "
        "considered preserved.\n"
    )
    return "\n".join(lines)


def build_report_layer(model: str, layer: int, fig_dir: Path) -> str:
    """Like build_report but compares baseline vs L{layer} instead of L0."""
    layer_tag = f"L{layer}"
    cfg = MODEL_CONFIGS[model]
    splits_base = {k: v for k, v in cfg.items() if k in ("pope", "amber")}

    # Override the intervention file to use the requested layer tag
    splits: dict[str, dict] = {}
    for split, files in splits_base.items():
        baseline_path = files["baseline"]
        # Derive L-layer path from L0 path by substituting the tag
        l0_path = files["L0"]
        new_path = Path(str(l0_path).replace("_L0.jsonl", f"_{layer_tag}.jsonl"))
        if not new_path.exists():
            print(f"[SKIP] {new_path} does not exist yet")
            continue
        splits[split] = {"baseline": baseline_path, layer_tag: new_path}

    if not splits:
        return f"# {model} L{layer}-ablation sanity check\n\nNo result files found yet.\n"

    # Patch cfg for build_report by temporarily substituting L0 → Lx in MODEL_CONFIGS
    original_splits = {}
    for split, files in splits.items():
        original_splits[split] = cfg.get(split, {}).copy()
        cfg[split] = {"baseline": files["baseline"], "L0": files[layer_tag]}

    try:
        report_text = build_report(model, fig_dir)
    finally:
        # Restore original config
        for split, files in original_splits.items():
            cfg[split] = files

    # Replace "L0" label text with the actual layer tag
    report_text = report_text.replace(
        f"L0-ablation sanity-check report", f"L{layer}-ablation sanity-check report"
    ).replace("L0 knockout", f"{layer_tag} knockout")
    return report_text


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="liquid",
        help="Which backend's bench JSONLs to analyse",
    )
    ap.add_argument(
        "--layer", type=int, default=0,
        help="Which layer index to treat as the intervention (default 0 = L0)",
    )
    args = ap.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    layer_tag = f"L{args.layer}"
    fig_dir = REPO / "figures" / f"sanity_check_{args.model}_{layer_tag}"
    report = RESULTS / f"sanity_check_{args.model}_{layer_tag}_report.md"

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.layer == 0:
        text = build_report(args.model, fig_dir)
    else:
        text = build_report_layer(args.model, args.layer, fig_dir)

    report.write_text(text)
    print(f"wrote {report}")
    for split in cfg:
        if split in ("pope", "amber"):
            print(f"wrote {fig_dir / f'logit_gap_{split}.png'}")


if __name__ == "__main__":
    main()
