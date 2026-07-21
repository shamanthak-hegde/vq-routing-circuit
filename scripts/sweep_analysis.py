"""Analysis and visualization for full probe-set sweep results.

Usage
-----
    # Single file
    python scripts/sweep_analysis.py --results results/sweep_llava.jsonl

    # Merge multiple files (e.g. naturalbench from one run, pope multi-seed from another)
    python scripts/sweep_analysis.py \\
        --results results/sweep_llava.jsonl results/sweep_pope_multiseed.jsonl

    When the same record_id appears in multiple files, the last file wins.

    # 2-model comparison (LLaVA vs VILA-U)
    python scripts/sweep_analysis.py --compare \\
        --llava results/sweep_llava.jsonl results/sweep_pope_multiseed.jsonl \\
        --vilau results/sweep_vilau.jsonl \\
        --out_dir figures/comparison

    # 3-model comparison (LLaVA | VILA | VILA-U + two Δ columns)
    python scripts/sweep_analysis.py --compare \\
        --llava results/sweep_llava.jsonl results/sweep_pope_multiseed.jsonl \\
        --vila  results/sweep_vila.jsonl \\
        --vilau results/sweep_vilau.jsonl \\
        --out_dir figures/comparison

Outputs
-------
    figures/heatmap_naturalbench.png
    figures/heatmap_pope.png
    figures/heatmap_yes_no.png     -- yes vs no split for each source
    figures/pair_consistency.png   -- per-pair dominant-window agreement rate
    figures/comparison/compare_<source>.png   -- compare mode figures
    + printed summary to stdout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ── load ──────────────────────────────────────────────────────────────────────

def load_results(paths: list[str]) -> pd.DataFrame:
    """Load and merge one or more JSONL result files.

    When the same (record_id, layer_start, token_group) key appears in multiple
    files the last file wins, so a multi-seed pope file can supersede an earlier
    single-seed run without touching naturalbench rows.
    """
    seen: dict[tuple, dict] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (row["record_id"], row.get("layer_start"), row.get("token_group"))
                seen[key] = row
    df = pd.DataFrame(list(seen.values()))
    df["warn"] = df["logit_clean"] <= df["logit_corrupt"]
    return df


def apply_correct_filter(df: pd.DataFrame, accuracy_path: str) -> pd.DataFrame:
    """Mark records as warn=True where clean-run top-1 prediction was wrong.

    Loads results/accuracy_*.jsonl (produced by run_accuracy.py) and excludes
    any record_id where correct=False from the usable slice.  All downstream
    code uses ~df.warn so no other changes are needed.
    """
    correct_ids: set[str] = set()
    with open(accuracy_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("correct"):
                correct_ids.add(row["record_id"])
    df = df.copy()
    df.loc[~df["record_id"].isin(correct_ids), "warn"] = True
    return df


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, correct_filter: bool = False) -> None:
    n_records = df.record_id.nunique()
    n_warn    = df[df.warn].record_id.nunique()
    label = "Results summary (WARN + wrong-baseline excluded)" if correct_filter else "Results summary"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total records   : {n_records}")
    print(f"  WARN (excluded) : {n_warn}  ({100*n_warn/n_records:.1f}%)")
    print(f"  Usable records  : {n_records - n_warn}")
    print()
    for src, g in df.groupby("source"):
        recs   = g.record_id.nunique()
        warned = g[g.warn].record_id.nunique()
        print(f"  {src:<16}  {recs:3d} records   {warned:3d} WARNs ({100*warned/recs:.1f}%)")
    print(f"{'='*60}\n")


# ── heatmap helpers ───────────────────────────────────────────────────────────

_GROUP_ORDER = ["visual", "question", "prompt_last"]
_CMAP        = "RdYlGn"


def _pivot(df: pd.DataFrame) -> pd.DataFrame:
    piv = (
        df.groupby(["layer_start", "token_group"])["score"]
        .mean()
        .unstack()
        .reindex(columns=_GROUP_ORDER)
    )
    return piv


def _draw_heatmap(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    title: str,
    n_records: int,
    vmin: float = -0.3,
    vmax: float = 1.0,
) -> None:
    layer_ends = [
        r + (pivot.index[1] - pivot.index[0]) if len(pivot.index) > 1 else r + 4
        for r in pivot.index
    ]
    ylabels = [f"[{ls},{le})" for ls, le in zip(pivot.index, layer_ends)]

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".2f",
        vmin=vmin,
        vmax=vmax,
        cmap=_CMAP,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_yticklabels(ylabels, rotation=0, fontsize=9)
    ax.set_xticklabels(pivot.columns, rotation=15, ha="right", fontsize=9)
    ax.set_xlabel("Token group", fontsize=10)
    ax.set_ylabel("Layer window", fontsize=10)
    ax.set_title(f"{title}\n(n={n_records} records)", fontsize=11)


def _draw_diff_heatmap(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    title: str,
    vmax: float = 0.5,
) -> None:
    layer_ends = [
        r + (pivot.index[1] - pivot.index[0]) if len(pivot.index) > 1 else r + 4
        for r in pivot.index
    ]
    ylabels = [f"[{ls},{le})" for ls, le in zip(pivot.index, layer_ends)]
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt="+.2f",
        vmin=-vmax,
        vmax=vmax,
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_yticklabels(ylabels, rotation=0, fontsize=9)
    ax.set_xticklabels(pivot.columns, rotation=15, ha="right", fontsize=9)
    ax.set_xlabel("Token group", fontsize=10)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=11)


def plot_cross_model(
    llava_df: pd.DataFrame,
    vilau_df: pd.DataFrame,
    out_dir: Path,
    vila_df: pd.DataFrame | None = None,
    qwen_df: pd.DataFrame | None = None,
    haplo_df: pd.DataFrame | None = None,
    emu3_df: pd.DataFrame | None = None,
) -> None:
    """Per source: comparison heatmaps across models.

    2-model:                         3-panel [LLaVA | VILA-U | Δ(LLaVA−VILA-U)]
    3-model (vila):                  5-panel [LLaVA | VILA | VILA-U | Δ(VILA−LLaVA) | Δ(VILA-U−LLaVA)]
    4-model (vila+qwen):             6-panel [LLaVA | VILA | Qwen | VILA-U | Δ(VILA-U−LLaVA) | Δ(Qwen−LLaVA)]
    5-model (vila+qwen+haplo|emu3):  7-panel [LLaVA | VILA | Qwen | {Haplo|Emu3} | VILA-U | Δ(VILA-U−LLaVA) | Δ({Haplo|Emu3}−LLaVA)]
    """
    # emu3_df fills the haplo slot when haplo_df is absent
    if haplo_df is None and emu3_df is not None:
        haplo_df = emu3_df
        _haplo_label = "Emu3"
        _delta_label = "Δ  (Emu3 − LLaVA)"
    else:
        _haplo_label = "HaploOmni"
        _delta_label = "Δ  (HaploOmni − LLaVA)"
    l_clean = llava_df[~llava_df.warn]
    vu_clean = vilau_df[~vilau_df.warn]
    sources = sorted(set(l_clean["source"].unique()) & set(vu_clean["source"].unique()))
    if not sources:
        print("  [skip] cross-model: no overlapping sources")
        return

    vi_clean = vila_df[~vila_df.warn] if vila_df is not None else None
    q_clean  = qwen_df[~qwen_df.warn] if qwen_df is not None else None
    h_clean  = haplo_df[~haplo_df.warn] if haplo_df is not None else None

    for src in sources:
        l_g  = l_clean[l_clean["source"] == src]
        vu_g = vu_clean[vu_clean["source"] == src]
        l_piv  = _pivot(l_g)
        vu_piv = _pivot(vu_g)

        common_rows = l_piv.index.intersection(vu_piv.index)

        if vi_clean is not None:
            vi_g   = vi_clean[vi_clean["source"] == src]
            vi_piv = _pivot(vi_g)
            common_rows = common_rows.intersection(vi_piv.index)
            vi_piv = vi_piv.reindex(index=common_rows, columns=_GROUP_ORDER)

        if q_clean is not None:
            q_g   = q_clean[q_clean["source"] == src]
            q_piv = _pivot(q_g)
            common_rows = common_rows.intersection(q_piv.index)
            q_piv = q_piv.reindex(index=common_rows, columns=_GROUP_ORDER)

        if h_clean is not None:
            h_g   = h_clean[h_clean["source"] == src]
            h_piv = _pivot(h_g)
            common_rows = common_rows.intersection(h_piv.index)
            h_piv = h_piv.reindex(index=common_rows, columns=_GROUP_ORDER)

        l_piv  = l_piv.reindex(index=common_rows, columns=_GROUP_ORDER)
        vu_piv = vu_piv.reindex(index=common_rows, columns=_GROUP_ORDER)

        if vi_clean is not None and q_clean is not None and h_clean is not None:
            # 5-model: LLaVA | VILA | Qwen | {Haplo|Emu3} | VILA-U | Δ(VILA-U−LLaVA) | Δ({Haplo|Emu3}−LLaVA)
            d_vu_piv = vu_piv - l_piv
            d_h_piv  = h_piv  - l_piv
            fig, axes = plt.subplots(1, 7, figsize=(38, 5))
            _draw_heatmap(axes[0], l_piv,  f"LLaVA-1.6 — {src}",         l_g.record_id.nunique())
            _draw_heatmap(axes[1], vi_piv, f"VILA v1.0 — {src}",         vi_g.record_id.nunique())
            _draw_heatmap(axes[2], q_piv,  f"Qwen2.5-VL — {src}",        q_g.record_id.nunique())
            _draw_heatmap(axes[3], h_piv,  f"{_haplo_label} — {src}",    h_g.record_id.nunique())
            _draw_heatmap(axes[4], vu_piv, f"VILA-U — {src}",            vu_g.record_id.nunique())
            _draw_diff_heatmap(axes[5], d_vu_piv, "Δ  (VILA-U − LLaVA)")
            _draw_diff_heatmap(axes[6], d_h_piv,  _delta_label)
            fig.suptitle(f"5-model causal restoration — {src}", fontsize=13)
        elif vi_clean is not None and q_clean is not None:
            # 4-model: LLaVA | VILA | Qwen | VILA-U | Δ(VILA-U−LLaVA) | Δ(Qwen−LLaVA)
            d_vu_piv = vu_piv - l_piv
            d_q_piv  = q_piv  - l_piv
            fig, axes = plt.subplots(1, 6, figsize=(33, 5))
            _draw_heatmap(axes[0], l_piv,  f"LLaVA-1.6 — {src}",    l_g.record_id.nunique())
            _draw_heatmap(axes[1], vi_piv, f"VILA v1.0 — {src}",    vi_g.record_id.nunique())
            _draw_heatmap(axes[2], q_piv,  f"Qwen2.5-VL — {src}",   q_g.record_id.nunique())
            _draw_heatmap(axes[3], vu_piv, f"VILA-U — {src}",       vu_g.record_id.nunique())
            _draw_diff_heatmap(axes[4], d_vu_piv, "Δ  (VILA-U − LLaVA)")
            _draw_diff_heatmap(axes[5], d_q_piv,  "Δ  (Qwen − LLaVA)")
            fig.suptitle(f"4-model causal restoration — {src}", fontsize=13)
        elif vi_clean is not None:
            d_vi_piv  = vi_piv  - l_piv
            d_vu_piv  = vu_piv  - l_piv
            fig, axes = plt.subplots(1, 5, figsize=(27, 5))
            _draw_heatmap(axes[0], l_piv,  f"LLaVA-1.6 — {src}",  l_g.record_id.nunique())
            _draw_heatmap(axes[1], vi_piv, f"VILA v1.0 — {src}",  vi_g.record_id.nunique())
            _draw_heatmap(axes[2], vu_piv, f"VILA-U — {src}",     vu_g.record_id.nunique())
            _draw_diff_heatmap(axes[3], d_vi_piv, "Δ  (VILA − LLaVA)")
            _draw_diff_heatmap(axes[4], d_vu_piv, "Δ  (VILA-U − LLaVA)")
            fig.suptitle(f"3-model causal restoration — {src}", fontsize=13)
        else:
            d_piv = l_piv - vu_piv
            fig, axes = plt.subplots(1, 3, figsize=(17, 5))
            _draw_heatmap(axes[0], l_piv,  f"LLaVA-1.6 — {src}", l_g.record_id.nunique())
            _draw_heatmap(axes[1], vu_piv, f"VILA-U — {src}",    vu_g.record_id.nunique())
            _draw_diff_heatmap(axes[2], d_piv, "Δ  (LLaVA − VILA-U)")
            fig.suptitle(f"Cross-model causal restoration — {src}", fontsize=13)

        fig.tight_layout()
        p = out_dir / f"compare_{src}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  saved {p}")


def plot_per_source(df: pd.DataFrame, out_dir: Path) -> None:
    """One heatmap per source (NaturalBench, POPE)."""
    clean = df[~df.warn]
    for src, g in clean.groupby("source"):
        fig, ax = plt.subplots(figsize=(6, 5))
        piv = _pivot(g)
        _draw_heatmap(ax, piv, src, g.record_id.nunique())
        fig.tight_layout()
        p = out_dir / f"heatmap_{src}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  saved {p}")


def plot_yes_no_split(df: pd.DataFrame, out_dir: Path) -> None:
    """Side-by-side yes/no heatmaps for each source."""
    clean = df[~df.warn]

    # need answer per record_id — infer from logit_clean direction
    # answer=="yes" → logit_clean for yes_id was high; we don't store
    # answer directly but record_id encodes it (probe schema uses _q0/_q1).
    # Instead: re-derive from record metadata if available, else skip.
    # We do store nothing about answer here; skip if not available.
    # Fallback: split by record_id suffix heuristic (fragile) — better to
    # load records from cache and merge.
    try:
        from probe import load_cache
        records, _, _ = load_cache()
        answer_map = {r.id: r.answer for r in records}
        clean = clean.copy()
        clean["answer"] = clean["record_id"].map(answer_map)
        if clean["answer"].isna().all():
            print("  [skip] yes/no split: record_ids not found in cache")
            return
    except Exception as e:
        print(f"  [skip] yes/no split: {e}")
        return

    sources = clean["source"].unique()
    for src in sources:
        g = clean[clean["source"] == src]
        if g["answer"].isna().all():
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
        for ax, ans in zip(axes, ["yes", "no"]):
            sub = g[g["answer"] == ans]
            if sub.empty:
                ax.set_visible(False)
                continue
            piv = _pivot(sub)
            _draw_heatmap(ax, piv, f"{src} — answer={ans}", sub.record_id.nunique())
        fig.suptitle(f"{src}: yes vs no answer split", fontsize=12)
        fig.tight_layout()
        p = out_dir / f"heatmap_yes_no_{src}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  saved {p}")


def plot_pair_consistency(df: pd.DataFrame, out_dir: Path) -> None:
    """For each pair_id, do yes+no records agree on the dominant window?

    'Dominant window' = layer window with highest mean score across token groups.
    Consistency = fraction of pairs where both records share the same dominant window.
    Plotted as a bar chart per source.
    """
    clean = df[~df.warn]

    try:
        from probe import load_cache
        records, _, _ = load_cache()
        pair_map = {r.id: r.pair_id for r in records}
        clean = clean.copy()
        clean["pair_id"] = clean["record_id"].map(pair_map)
    except Exception as e:
        print(f"  [skip] pair consistency: {e}")
        return

    if clean["pair_id"].isna().all():
        print("  [skip] pair consistency: pair_ids not found")
        return

    results = []
    for src, g in clean.groupby("source"):
        for pair_id, pg in g.groupby("pair_id"):
            rec_ids = pg["record_id"].unique()
            if len(rec_ids) < 2:
                continue
            dominant = {}
            for rid in rec_ids:
                rg = pg[pg["record_id"] == rid]
                mean_by_window = rg.groupby("layer_start")["score"].mean()
                dominant[rid] = int(mean_by_window.idxmax())
            doms = list(dominant.values())
            results.append({
                "source": src,
                "pair_id": pair_id,
                "consistent": len(set(doms)) == 1,
                "dominant_windows": doms,
            })

    if not results:
        return

    rdf = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(6, 4))
    consistency = rdf.groupby("source")["consistent"].mean() * 100
    consistency.plot(kind="bar", ax=ax, color=["steelblue", "coral"][:len(consistency)],
                     edgecolor="white", width=0.5)
    ax.set_ylabel("Pair consistency (%)")
    ax.set_title("Dominant window agreement within pairs\n(both records agree on which layer window restores most)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticklabels(consistency.index, rotation=0)
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.1f}%",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height() + 1),
                    ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    p = out_dir / "pair_consistency.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")


# ── score distribution ────────────────────────────────────────────────────────

def print_score_stats(df: pd.DataFrame) -> None:
    clean = df[~df.warn]
    print("  Mean restoration score by (source, token_group):")
    tbl = (
        clean.groupby(["source", "token_group"])["score"]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    print(tbl.to_string())
    print()

    print("  Peak window (highest mean score) by source:")
    for src, g in clean.groupby("source"):
        peak = g.groupby("layer_start")["score"].mean().idxmax()
        w_end = peak + (g["layer_end"].iloc[0] - g["layer_start"].iloc[0])
        peak_score = g.groupby("layer_start")["score"].mean().max()
        print(f"    {src:<16}  window [{peak},{w_end})  mean_score={peak_score:.3f}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", default=["results/sweep_llava.jsonl"],
                        help="One or more JSONL result files to merge (later files override earlier)")
    parser.add_argument("--compare", action="store_true",
                        help="Cross-model comparison mode: requires --llava and --vilau")
    parser.add_argument("--llava", nargs="+",
                        help="LLaVA result files (compare mode)")
    parser.add_argument("--vila", nargs="+",
                        help="VILA v1.0 result files (optional 3rd model in compare mode)")
    parser.add_argument("--vilau", nargs="+",
                        help="VILA-U result files (compare mode)")
    parser.add_argument("--qwen", nargs="+",
                        help="Qwen2.5-VL result files (optional 4th model in compare mode)")
    parser.add_argument("--haplo", nargs="+",
                        help="HaploOmni result files (optional 5th model in compare mode)")
    parser.add_argument("--emu3", nargs="+",
                        help="Emu3 result files (optional; fills haplo slot when haplo absent)")
    parser.add_argument("--out_dir", default="figures")
    parser.add_argument(
        "--correct_filter", action="store_true",
        help="Exclude records where the model's clean-run top-1 prediction was wrong. "
             "Requires --accuracy (single mode) or --llava_acc/--vila_acc/--vilau_acc (compare mode).",
    )
    parser.add_argument("--accuracy",    default=None, help="Accuracy JSONL for single mode")
    parser.add_argument("--llava_acc",   default=None, help="LLaVA accuracy JSONL (compare mode)")
    parser.add_argument("--vila_acc",    default=None, help="VILA accuracy JSONL (compare mode)")
    parser.add_argument("--vilau_acc",   default=None, help="VILA-U accuracy JSONL (compare mode)")
    parser.add_argument("--emu3_acc",    default=None, help="Emu3 accuracy JSONL (compare mode)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        if not args.llava or not args.vilau:
            parser.error("--compare requires --llava and --vilau")
        print(f"Loading LLaVA: {args.llava}")
        llava_df = load_results(args.llava)
        print(f"Loading VILA-U: {args.vilau}")
        vilau_df = load_results(args.vilau)
        vila_df = None
        if args.vila:
            print(f"Loading VILA: {args.vila}")
            vila_df = load_results(args.vila)
        qwen_df = None
        if args.qwen:
            print(f"Loading Qwen2.5-VL: {args.qwen}")
            qwen_df = load_results(args.qwen)
        haplo_df = None
        if args.haplo:
            print(f"Loading HaploOmni: {args.haplo}")
            haplo_df = load_results(args.haplo)
        emu3_df = None
        if args.emu3:
            print(f"Loading Emu3: {args.emu3}")
            emu3_df = load_results(args.emu3)

        if args.correct_filter:
            if args.llava_acc:
                llava_df = apply_correct_filter(llava_df, args.llava_acc)
            if args.vilau_acc:
                vilau_df = apply_correct_filter(vilau_df, args.vilau_acc)
            if vila_df is not None and args.vila_acc:
                vila_df = apply_correct_filter(vila_df, args.vila_acc)

        print("\n--- LLaVA ---")
        print_summary(llava_df, args.correct_filter)
        if vila_df is not None:
            print("--- VILA ---")
            print_summary(vila_df, args.correct_filter)
        if qwen_df is not None:
            print("--- Qwen2.5-VL ---")
            print_summary(qwen_df, args.correct_filter)
        if haplo_df is not None:
            print("--- HaploOmni ---")
            print_summary(haplo_df, args.correct_filter)
        if emu3_df is not None:
            print("--- Emu3 ---")
            print_summary(emu3_df, args.correct_filter)
        print("--- VILA-U ---")
        print_summary(vilau_df, args.correct_filter)
        print("Generating cross-model figures ...")
        plot_cross_model(llava_df, vilau_df, out_dir, vila_df=vila_df, qwen_df=qwen_df, haplo_df=haplo_df, emu3_df=emu3_df)
        print("\nDone.")
        return

    print(f"Loading {args.results} ...")
    df = load_results(args.results)

    if args.correct_filter:
        if not args.accuracy:
            parser.error("--correct_filter requires --accuracy in single mode")
        df = apply_correct_filter(df, args.accuracy)

    print_summary(df, args.correct_filter)
    print_score_stats(df)

    print("Generating figures ...")
    plot_per_source(df, out_dir)
    plot_yes_no_split(df, out_dir)
    plot_pair_consistency(df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
