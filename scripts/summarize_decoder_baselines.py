"""sweep result summarizer.

Reads all results/decoder_baseline_sweep/bench_*.meta.json files, picks the best-accuracy
config per (method, bench), and writes a summary TSV.

Usage:
    python scripts/summarize_decoder_baselines.py
    python scripts/summarize_decoder_baselines.py --out results/decoder_baseline_sweep_summary.tsv
    python scripts/summarize_decoder_baselines.py --collapse_threshold 0.51

Collapse criterion: configs with accuracy < collapse_threshold are flagged
as COLLAPSED and excluded from winner selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_meta_files(sweep_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(sweep_dir.glob("bench_*.meta.json")):
        try:
            meta = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] Could not read {p.name}: {e}", file=sys.stderr)
            continue
        if meta.get("decoder", "standard") == "standard":
            continue  # skip baseline runs stored here
        rows.append({"_file": p.name, **meta})
    return rows


def parse_config(row: dict) -> tuple[str, str, float, str | int | float]:
    """Extract (method, bench, alpha, param_value) from meta dict."""
    decoder = row.get("decoder", "standard")
    bench = row.get("bench", "?")
    alpha = row.get("alpha") or row.get("decoder_alpha", float("nan"))
    if decoder == "vcd":
        param = row.get("decoder_sigma", float("nan"))
    elif decoder == "dola":
        param = row.get("decoder_early_layer", -1)
    else:
        param = None
    return decoder, bench, float(alpha), param


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize VCD/DOLA sweep results")
    parser.add_argument("--sweep_dir", default="results/decoder_baseline_sweep",
                        help="Directory containing bench_*.meta.json files")
    parser.add_argument("--out", default="results/decoder_baseline_sweep_summary.tsv",
                        help="Output TSV path")
    parser.add_argument("--collapse_threshold", type=float, default=0.51,
                        help="Accuracy below this value is flagged COLLAPSED (default 0.51)")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: {sweep_dir} does not exist. Run the sweep first.", file=sys.stderr)
        sys.exit(1)

    rows = load_meta_files(sweep_dir)
    if not rows:
        print(f"No bench_*.meta.json files found in {sweep_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(rows)} meta files from {sweep_dir}")

    # Build a table with parsed config and metrics.
    table = []
    for row in rows:
        decoder, bench, alpha, param = parse_config(row)
        metrics = row.get("metrics", {})
        acc = metrics.get("accuracy", float("nan"))
        f1 = metrics.get("f1", float("nan"))
        yes_rate = metrics.get("yes_rate", float("nan"))
        n_records = row.get("n_records", 0)

        collapsed = acc < args.collapse_threshold if acc == acc else True  # nan → collapsed
        param_key = "sigma" if decoder == "vcd" else "early_layer"

        table.append({
            "method": decoder,
            "bench": bench,
            "alpha": alpha,
            param_key: param,
            "acc": acc,
            "f1": f1,
            "yes_rate": yes_rate,
            "n_records": n_records,
            "collapsed": collapsed,
            "_file": row["_file"],
        })

    # Print raw table sorted by (method, bench, acc desc).
    table.sort(key=lambda r: (r["method"], r["bench"], -(r["acc"] if r["acc"] == r["acc"] else -1)))

    # ── Per-(method, bench) winners ────────────────────────────────────────
    winners: dict[tuple[str, str], dict] = {}
    for row in table:
        key = (row["method"], row["bench"])
        if row["collapsed"]:
            continue
        if key not in winners or row["acc"] > winners[key]["acc"]:
            winners[key] = row

    # ── Print all results ──────────────────────────────────────────────────
    print("\nAll configs (sorted by method, bench, acc desc):")
    print(f"{'method':<8} {'bench':<12} {'alpha':>6} {'param':>10} {'acc':>7} {'f1':>7} "
          f"{'yes_rate':>9} {'n':>6} {'status':<12} file")
    print("-" * 100)
    for row in table:
        m = row["method"]
        b = row["bench"]
        param_key = "sigma" if m == "vcd" else "early_layer"
        param = row.get(param_key, row.get("sigma", row.get("early_layer", "?")))
        is_winner = winners.get((m, b), {}).get("_file") == row["_file"]
        status = "WINNER" if is_winner else ("COLLAPSED" if row["collapsed"] else "")
        print(
            f"{m:<8} {b:<12} {row['alpha']:>6.2f} {str(param):>10} "
            f"{row['acc']:>7.4f} {row['f1']:>7.4f} {row['yes_rate']:>9.4f} "
            f"{row['n_records']:>6} {status:<12} {row['_file']}"
        )

    # ── Write summary TSV ──────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tsv_rows = []
    for row in table:
        m = row["method"]
        b = row["bench"]
        param_key = "sigma" if m == "vcd" else "early_layer"
        param = row.get(param_key, row.get("sigma", row.get("early_layer", "")))
        is_winner = winners.get((m, b), {}).get("_file") == row["_file"]
        tsv_rows.append({
            "method": m,
            "bench": b,
            "alpha": row["alpha"],
            "sigma_or_layer": param,
            "acc": row["acc"],
            "f1": row["f1"],
            "yes_rate": row["yes_rate"],
            "n_records": row["n_records"],
            "collapsed": "1" if row["collapsed"] else "0",
            "win_flag": "1" if is_winner else "0",
            "file": row["_file"],
        })

    header = ["method", "bench", "alpha", "sigma_or_layer", "acc", "f1", "yes_rate",
              "n_records", "collapsed", "win_flag", "file"]
    with out_path.open("w") as f:
        f.write("\t".join(header) + "\n")
        for r in tsv_rows:
            f.write("\t".join(str(r[h]) for h in header) + "\n")

    print(f"\nSummary TSV written to {out_path}")

    # ── Print winners table ────────────────────────────────────────────────
    print("\nWinners (best accuracy per method×bench, excluding collapsed):")
    print(f"{'method':<8} {'bench':<12} {'alpha':>6} {'param':>10} {'acc':>7} {'f1':>7} {'yes_rate':>9}")
    print("-" * 65)
    for (m, b), row in sorted(winners.items()):
        param_key = "sigma" if m == "vcd" else "early_layer"
        param = row.get(param_key, row.get("sigma", row.get("early_layer", "?")))
        print(f"{m:<8} {b:<12} {row['alpha']:>6.2f} {str(param):>10} "
              f"{row['acc']:>7.4f} {row['f1']:>7.4f} {row['yes_rate']:>9.4f}")

    if not winners:
        print("  (no non-collapsed configs found yet — run the sweep first)")

    # ── Missing configs ────────────────────────────────────────────────────
    expected_vcd = {
        f"bench_{b}_vilau_vcd_a{a}_s{s}.meta.json"
        for b in ("pope_full", "amber")
        for a in (0.5, 1.0, 1.5, 2.0)
        for s in (10, 20, 40, 80)
    }
    expected_dola = {
        f"bench_{b}_vilau_dola_a{a}_l{l}.meta.json"
        for b in ("pope_full", "amber")
        for a in (0.1, 0.3, 0.5, 1.0)
        for l in (0, 4, 8, 16, 24)
    }
    found = {r["_file"] for r in rows}
    missing_vcd = expected_vcd - found
    missing_dola = expected_dola - found
    total_missing = len(missing_vcd) + len(missing_dola)
    if total_missing:
        print(f"\n{total_missing} configs not yet run (of {len(expected_vcd)+len(expected_dola)} total):")
        for f in sorted(missing_vcd | missing_dola):
            print(f"  MISSING: {f}")
    else:
        print(f"\nAll {len(expected_vcd)+len(expected_dola)} expected configs found.")


if __name__ == "__main__":
    main()
