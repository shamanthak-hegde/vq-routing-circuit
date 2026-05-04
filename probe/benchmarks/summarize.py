"""Aggregate N-042 benchmark results into a paper-ready summary table.

Reads all results/bench_*.meta.json sidecars (written by run_bench.py),
builds a tidy (model, bench, mode) → metrics table, writes:
  results/bench_summary.json    — machine-readable
  results/bench_summary.md      — markdown table for paper
  figures/intervention/benchmark_summary.png — 4-panel grouped bar chart

Usage
-----
    source activate sae
    python -m probe.benchmarks.summarize
    python -m probe.benchmarks.summarize --results_dir results --out_dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BENCH_LABEL = {
    "pope_full": "POPE-Adv",
    "nb_full":   "NB G-Acc",
    "hb":        "HB aAcc",
    "amber":     "AMBER F1",
}

BENCH_METRIC = {
    "pope_full": "accuracy",
    "nb_full":   "g_acc",
    "hb":        "a_acc",
    "amber":     "f1",
}

MODEL_LABEL = {
    "vilau":   "VILA-U",
    "llava":   "LLaVA",
    "unitok":  "UniTok",
    "qwen3vl": "Qwen2.5-VL",
}

MODEL_ORDER  = ["vilau", "llava", "unitok", "qwen3vl"]
BENCH_ORDER  = ["pope_full", "nb_full", "hb", "amber"]
MODE_ORDER   = ["baseline", "intervention"]


def load_meta(results_dir: Path) -> list[dict]:
    metas = []
    for path in sorted(results_dir.glob("bench_*.meta.json")):
        try:
            m = json.loads(path.read_text())
            metas.append(m)
        except Exception:
            print(f"  Warning: could not parse {path.name}")
    return metas


def extract_primary_metric(meta: dict) -> float | None:
    bench = meta.get("bench")
    key = BENCH_METRIC.get(bench)
    if key is None:
        return None
    return meta.get("metrics", {}).get(key)


def build_table(metas: list[dict]) -> dict:
    """Return {(model, bench, mode): {metric: val, ...}}."""
    table: dict = {}
    for m in metas:
        model   = m.get("backend")
        bench   = m.get("bench")
        # intervention_mode is the new field; knockout_mode is the legacy alias.
        imode   = m.get("intervention_mode") or m.get("knockout_mode")
        mode    = "baseline" if imode is None else "intervention"
        if not (model and bench):
            continue
        key = (model, bench, mode)
        table[key] = {
            "primary": extract_primary_metric(m),
            "metrics": m.get("metrics", {}),
            "n_records": m.get("n_records"),
            "intervention_mode": imode,
            "knockout_layer": m.get("knockout_layer"),
        }
    return table


def render_markdown(table: dict) -> str:
    lines = []
    lines.append("# Benchmark Summary — Pathological-route ablation (L0)\n")

    for bench in BENCH_ORDER:
        label = BENCH_LABEL.get(bench, bench)
        metric_key = BENCH_METRIC.get(bench, "accuracy")
        lines.append(f"## {label} (`{metric_key}`)\n")

        header = "| Model | Baseline | Intervention | Δ |"
        sep    = "|---|---|---|---|"
        lines.append(header)
        lines.append(sep)

        for model in MODEL_ORDER:
            model_label = MODEL_LABEL.get(model, model)
            base = table.get((model, bench, "baseline"), {}).get("primary")
            intr = table.get((model, bench, "intervention"), {}).get("primary")

            def fmt(v: float | None) -> str:
                if v is None:
                    return "—"
                return f"{v * 100:.1f}%"

            delta_str = "—"
            if base is not None and intr is not None:
                delta = (intr - base) * 100
                sign = "+" if delta >= 0 else ""
                delta_str = f"{sign}{delta:.1f}pt"

            lines.append(f"| {model_label} | {fmt(base)} | {fmt(intr)} | {delta_str} |")

        lines.append("")

    return "\n".join(lines)


def render_figure(table: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping figure")
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Pathological-route ablation (L0) — effect by model and benchmark", fontsize=13)

    colors = {"baseline": "#4878d0", "intervention": "#ee854a"}
    model_order = [m for m in MODEL_ORDER if any(
        (m, bench, mode) in table for bench in BENCH_ORDER for mode in MODE_ORDER
    )]
    model_labels = [MODEL_LABEL.get(m, m) for m in model_order]

    for ax, bench in zip(axes, BENCH_ORDER):
        label = BENCH_LABEL[bench]
        x = np.arange(len(model_order))
        width = 0.35
        for i, (mode, color) in enumerate(colors.items()):
            vals = []
            for model in model_order:
                v = table.get((model, bench, mode), {}).get("primary")
                vals.append((v or 0) * 100)
            bars = ax.bar(x + (i - 0.5) * width, vals, width, label=mode.capitalize(),
                          color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{v:.0f}", ha="center", va="bottom", fontsize=7)

        # Draw Δ arrows for VILA-U
        if "vilau" in model_order:
            idx = model_order.index("vilau")
            b = table.get(("vilau", bench, "baseline"), {}).get("primary")
            n = table.get(("vilau", bench, "intervention"), {}).get("primary")
            if b is not None and n is not None:
                delta = (n - b) * 100
                sign = "+" if delta >= 0 else ""
                ax.annotate(f"{sign}{delta:.1f}pt",
                            xy=(x[idx], max(b, n) * 100 + 2),
                            ha="center", va="bottom", fontsize=8,
                            color="green" if delta >= 0 else "red",
                            fontweight="bold")

        ax.set_title(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Score (%)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_summary.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate N-042 benchmark results into a summary table + figure"
    )
    parser.add_argument("--results_dir", default="results",
                        help="Directory containing bench_*.meta.json files")
    parser.add_argument("--out_dir", default="results",
                        help="Output directory for bench_summary.{json,md}")
    parser.add_argument("--fig_dir", default="figures/intervention",
                        help="Output directory for benchmark_summary.png")
    parser.add_argument("--no_figure", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = load_meta(results_dir)
    print(f"Loaded {len(metas)} meta files from {results_dir}")

    if not metas:
        print("No meta files found. Run N-042 sweep first.")
        return

    table = build_table(metas)
    print(f"Built table with {len(table)} (model, bench, mode) entries")

    # Dump machine-readable JSON
    json_out = {
        f"{model}/{bench}/{mode}": entry
        for (model, bench, mode), entry in sorted(table.items())
    }
    (out_dir / "bench_summary.json").write_text(json.dumps(json_out, indent=2) + "\n")
    print(f"  Wrote {out_dir / 'bench_summary.json'}")

    # Markdown table
    md = render_markdown(table)
    (out_dir / "bench_summary.md").write_text(md)
    print(f"  Wrote {out_dir / 'bench_summary.md'}")
    print()
    print(md)

    if not args.no_figure:
        render_figure(table, fig_dir)


if __name__ == "__main__":
    main()
