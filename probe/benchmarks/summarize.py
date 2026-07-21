"""Aggregate benchmark results into a paper-ready summary table.

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
    "seed":    "SEED-LLaMA",
    "showo":   "Show-o",
    "haplo":   "HaploOmni",
    "lavit":   "LaVIT",
}

MODEL_ORDER  = ["vilau", "llava", "unitok", "qwen3vl", "seed", "showo", "haplo", "lavit"]
BENCH_ORDER  = ["pope_full", "nb_full", "hb", "amber"]
# Canonical modes: "baseline", "intervention" (pathological_route_ablation),
# "vti" (VTITextualSteer), "vista" (future VISTASteer)
MODE_ORDER   = ["baseline", "vti", "intervention"]

# Map intervention_mode field value → canonical table mode key
_INTERVENTION_MODE_MAP: dict[str, str] = {
    None:                           "baseline",
    "pathological_route_ablation":  "intervention",
    "full_zero":                    "intervention",   # legacy alias
    "window_attn_knockout":         "intervention",
    "vti_textual":                  "vti",
    "vista":                        "vista",
}


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
    """Return {(model, bench, mode): {metric: val, ...}}.

    Modes: "baseline", "intervention", "vti", "vista"
    """
    table: dict = {}
    for m in metas:
        model = m.get("backend")
        bench = m.get("bench")
        imode = m.get("intervention_mode") or m.get("knockout_mode")
        mode  = _INTERVENTION_MODE_MAP.get(imode, "intervention")
        if not (model and bench):
            continue
        key = (model, bench, mode)
        n = m.get("n_records") or 0
        entry = {
            "primary": extract_primary_metric(m),
            "metrics": m.get("metrics", {}),
            "n_records": n,
            "intervention_mode": imode,
            "knockout_layer": m.get("knockout_layer"),
        }
        # Keep the entry with the larger n_records (smoke runs don't overwrite real runs)
        if key not in table or n > (table[key].get("n_records") or 0):
            table[key] = entry
    return table


def render_markdown(table: dict) -> str:
    lines = []
    lines.append("# Benchmark Summary — Pathological-route ablation (L0)\n")

    def fmt(v: float | None) -> str:
        return "—" if v is None else f"{v * 100:.1f}%"

    def delta_str(base: float | None, intr: float | None) -> str:
        if base is None or intr is None:
            return "—"
        d = (intr - base) * 100
        return f"{'+'if d >= 0 else ''}{d:.1f}pt"

    for bench in BENCH_ORDER:
        label = BENCH_LABEL.get(bench, bench)
        metric_key = BENCH_METRIC.get(bench, "accuracy")
        lines.append(f"## {label} (`{metric_key}`)\n")

        # Determine which non-baseline modes have any data for this bench
        extra_modes = [m for m in MODE_ORDER[1:]
                       if any((mdl, bench, m) in table for mdl in MODEL_ORDER)]

        col_headers = ["Baseline"] + [m.upper() for m in extra_modes] + \
                      ["Δ " + m.upper() for m in extra_modes]
        header = "| Model | " + " | ".join(col_headers) + " |"
        sep    = "|" + "|".join(["---"] * (len(col_headers) + 1)) + "|"
        lines.append(header)
        lines.append(sep)

        for model in MODEL_ORDER:
            mlabel = MODEL_LABEL.get(model, model)
            base   = table.get((model, bench, "baseline"), {}).get("primary")
            row_vals = [fmt(base)]
            for m in extra_modes:
                v = table.get((model, bench, m), {}).get("primary")
                row_vals.append(fmt(v))
            for m in extra_modes:
                v = table.get((model, bench, m), {}).get("primary")
                row_vals.append(delta_str(base, v))
            lines.append("| " + mlabel + " | " + " | ".join(row_vals) + " |")

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

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Benchmark results: baseline / VTI / pathological-route ablation (L0)",
                 fontsize=12)

    mode_colors = {
        "baseline":     "#4878d0",
        "vti":          "#6acc65",
        "intervention": "#ee854a",
        "vista":        "#d65f5f",
    }
    model_order = [m for m in MODEL_ORDER if any(
        (m, bench, mode) in table for bench in BENCH_ORDER for mode in MODE_ORDER
    )]
    model_labels = [MODEL_LABEL.get(m, m) for m in model_order]

    for ax, bench in zip(axes, BENCH_ORDER):
        label = BENCH_LABEL[bench]
        active_modes = [m for m in MODE_ORDER
                        if any((mdl, bench, m) in table for mdl in model_order)]
        x = np.arange(len(model_order))
        n_modes = len(active_modes)
        width = 0.7 / max(n_modes, 1)

        for i, mode in enumerate(active_modes):
            offset = (i - (n_modes - 1) / 2.0) * width
            vals = [(table.get((mdl, bench, mode), {}).get("primary") or 0) * 100
                    for mdl in model_order]
            color = mode_colors.get(mode, "#888888")
            bars = ax.bar(x + offset, vals, width * 0.9, label=mode.upper(),
                          color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                            f"{v:.0f}", ha="center", va="bottom", fontsize=6)

        ax.set_title(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 108)
        ax.set_ylabel("Score (%)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_summary.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark results into a summary table + figure"
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
        print("No meta files found. Run the sweep first.")
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
