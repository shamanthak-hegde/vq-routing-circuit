"""σ-sensitivity audit for activation-patching restoration scores.

Calibration criterion (from probe/tracing/calibrate.py):
  Sample N=50 POPE records.  For each record, project the clean image through
  mm_projector to get clean_tok ∈ ℝ^(n_vis × H) (float32).  For each
  candidate σ, add independent N(0,σ²) noise to every token:
      noisy_tok = clean_tok + ε,  ε ~ N(0, σ²·I_H)
  Compute the per-token cosine similarity and average across all tokens and
  records.  Choose σ so that mean cos-sim ≈ 0.5.

  Why 0.5?  cos-sim=1 means no corruption; cos-sim=0 means orthogonal pure
  noise.  The midpoint 0.5 places the noisy tokens at the edge of the clean
  manifold neighbourhood: the model is reliably degraded (WARN rate low) while
  the tokens remain structurally related to the originals.

Calibrated σ per model (audit-confirmed):
  LLaVA-1.6-7b : σ = 0.50  (tok-norm ≈ 17; cos-sim ≈ 0.50 at σ_cal)
  VILA-7b       : σ = 0.50  (tok-norm ≈ 17; cos-sim ≈ 0.48)
  VILA-U-7b-256 : σ = 1.00  (tok-norm ≈ 26; cos-sim ≈ 0.39; WARN inherent)

  Note: LLaVA and VILA share similar token norms (~17) so they share the same
  calibrated σ = 0.50.  VILA-U needs 2× larger σ because its VQ projector
  produces tokens of norm ~26, and WARN persists regardless because the VQ
  discretisation makes the model insensitive to continuous additive noise.

This script:
  1. Re-runs calibrate_sigma at each candidate σ on the same 50-record sample.
  2. Runs the full windowed sweep (window_size=4, n_seeds=1) at each σ.
  3. Reports per-σ: cos-sim, WARN rate, mean restoration score per token group.
  4. Combined multi-model table via --combine.

Usage
-----
    # Per-model (run in the model's own conda env):
    source activate sae
    python -m probe.tracing.calibrate_audit \\
        --backend llava \\
        --model_path liuhaotian/llava-v1.6-vicuna-7b \\
        --sigmas 0.25,0.5,1.0 \\
        --out results/audit_llava.jsonl

    conda activate vila
    python -m probe.tracing.calibrate_audit \\
        --backend vila \\
        --model_path Efficient-Large-Model/VILA-7b \\
        --sigmas 0.25,0.5,1.0 \\
        --out results/audit_vila.jsonl

    conda activate vila-u
    python -m probe.tracing.calibrate_audit \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --sigmas 0.5,1.0,2.0 \\
        --out results/audit_vilau.jsonl

    # Single-model table (no model needed):
    python -m probe.tracing.calibrate_audit --report_only \\
        --out results/audit_llava.jsonl

    # Combined 3-model table (no model needed):
    python -m probe.tracing.calibrate_audit --combine \\
        results/audit_llava.jsonl results/audit_vila.jsonl results/audit_vilau.jsonl \\
        --labels LLaVA VILA VILA-U
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import torch


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_done_pairs(out_path: Path) -> set[tuple[str, float]]:
    done: set[tuple[str, float]] = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add((row["record_id"], float(row["sigma"])))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _load_rows(out_path: Path) -> list[dict]:
    rows = []
    if not out_path.exists():
        return rows
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


# ── table formatting ──────────────────────────────────────────────────────────

def _mean(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else float("nan")


def _fmt(v: float, width: int = 7) -> str:
    if v != v:  # NaN
        return f"{'—':>{width}}"
    return f"{v:>{width}.3f}"


def print_table(rows: list[dict], model_label: str = "") -> None:
    """Print per-σ sensitivity table from audit JSONL rows."""
    if not rows:
        print("  (no rows)")
        return

    sigmas = sorted(set(float(r["sigma"]) for r in rows))
    header = (
        f"  {'σ':>7}  {'cos-sim':>7}  {'WARN%':>6}  "
        f"{'n_use':>5}  {'visual':>7}  {'question':>8}  {'prompt_last':>11}"
    )
    sep = "  " + "─" * (len(header) - 2)

    if model_label:
        print(f"\n{'═'*68}")
        print(f"  {model_label}")
        print(f"{'═'*68}")
    print(header)
    print(sep)

    for sigma in sigmas:
        sigma_rows = [r for r in rows if abs(float(r["sigma"]) - sigma) < 1e-9]
        cos_sim = _mean([r["cos_sim"] for r in sigma_rows])

        # WARN flag is per-record (first row for each record_id)
        by_record: dict[str, dict] = {}
        for r in sigma_rows:
            if r["record_id"] not in by_record:
                by_record[r["record_id"]] = r
        n_total = len(by_record)
        n_warn  = sum(1 for r in by_record.values() if r["is_warn"])
        warn_pct = 100.0 * n_warn / n_total if n_total else 0.0

        usable = [r for r in sigma_rows if not r["is_warn"]]
        n_usable = len(set(r["record_id"] for r in usable))

        by_group: dict[str, list[float]] = {}
        for r in usable:
            by_group.setdefault(r["token_group"], []).append(r["score"])

        visual      = _mean(by_group.get("visual",      []))
        question    = _mean(by_group.get("question",    []))
        prompt_last = _mean(by_group.get("prompt_last", []))

        cal_mark = " ←cal" if abs(cos_sim - 0.5) <= 0.08 else "     "
        print(
            f"  {sigma:>7.4f}  {cos_sim:>7.4f}  {warn_pct:>5.1f}%"
            f"  {n_usable:>5d}"
            f"  {_fmt(visual)}  {_fmt(question)}  {_fmt(prompt_last)}{cal_mark}"
        )

    print(sep)


def print_combined_table(files: list[Path], labels: list[str]) -> None:
    """Print per-model sensitivity tables side by side."""
    for path, label in zip(files, labels):
        rows = _load_rows(path)
        if not rows:
            print(f"\n  {label}: no data in {path}")
            continue
        print_table(rows, model_label=label)


def print_criterion_summary() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  σ-CALIBRATION CRITERION                                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Noise added post mm_projector (projected visual tokens in LM space) ║
║  Noise distribution : N(0, σ²·I_H), independent per token/dim      ║
║  Similarity metric  : mean cos-sim(clean_tok, noisy_tok)            ║
║  over all visual tokens and 50 randomly sampled POPE records        ║
║  Target             : mean cos-sim ≈ 0.5                            ║
║                                                                      ║
║  Calibrated σ per model:                                            ║
║    LLaVA-1.6-7b   : σ = 0.50  (cos-sim ≈ 0.50, tok-norm ≈ 17)     ║
║    VILA-7b        : σ = 0.50  (cos-sim ≈ 0.48, tok-norm ≈ 17)      ║
║    VILA-U-7b-256  : σ = 1.00  (cos-sim ≈ 0.39, tok-norm ≈ 26)      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# ── core sweep ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_audit(
    hm,
    records,
    sigmas: Sequence[float],
    n_samples: int = 50,
    seed: int = 0,
    out_path: Path | None = None,
    done_pairs: set[tuple[str, float]] | None = None,
) -> list[dict]:
    """Full σ-sensitivity audit: calibrate_sigma + windowed sweep.

    Returns list of row dicts (same schema as written to out_path JSONL).
    """
    from probe.tracing.calibrate import calibrate_sigma
    from probe.tracing.sweep import sweep_record

    pope = [r for r in records if r.corruption_mode == "gaussian_noise"]
    if not pope:
        raise ValueError("No gaussian_noise records found.")
    rng = random.Random(seed)
    sample = rng.sample(pope, min(n_samples, len(pope)))
    print(f"Sampled {len(sample)} POPE records (seed={seed})\n")

    # Cos-sim pass — projector only, fast
    print("Computing cos-sim for all σ values (projector pass) ...")
    cs_results = calibrate_sigma(hm, sample, sigmas=list(sigmas), n_samples=len(sample), seed=seed)
    cos_sim_map: dict[float, float] = dict(cs_results)
    for s, cs in cs_results:
        cal = " ← calibrated (target ≈ 0.5)" if abs(cs - 0.5) <= 0.08 else ""
        print(f"  σ={s:.4f}  cos-sim={cs:.4f}{cal}")
    print()

    done_pairs = done_pairs or set()
    all_rows: list[dict] = []

    fout = out_path.open("a") if out_path else None
    try:
        total_records = len(sample) * len(sigmas)
        done_count = 0

        for sigma in sigmas:
            cos_sim_val = cos_sim_map[sigma]
            print(f"─── σ={sigma:.4f}  cos-sim={cos_sim_val:.4f} ───────────────────────────")

            t_sigma_start = time.time()
            sigma_done = 0

            for i, rec in enumerate(sample, 1):
                pair_key = (rec.id, float(sigma))
                if pair_key in done_pairs:
                    sigma_done += 1
                    done_count += 1
                    continue

                t0 = time.time()
                try:
                    patch_results = sweep_record(
                        hm, rec, sigma=sigma, window_size=4, n_seeds=1
                    )
                except Exception as e:
                    print(f"  [{i:3d}/{len(sample)}] ERROR {rec.id}: {e}")
                    continue

                is_warn = patch_results[0].logit_clean <= patch_results[0].logit_corrupt

                for pr in patch_results:
                    row = {
                        "sigma":         sigma,
                        "cos_sim":       round(cos_sim_val, 6),
                        "record_id":     pr.record_id,
                        "source":        pr.source,
                        "is_warn":       bool(is_warn),
                        "layer_start":   pr.layer_start,
                        "layer_end":     pr.layer_end,
                        "token_group":   pr.token_group,
                        "score":         round(pr.score, 6),
                        "logit_clean":   round(pr.logit_clean, 4),
                        "logit_corrupt": round(pr.logit_corrupt, 4),
                        "logit_patched": round(pr.logit_patched, 4),
                    }
                    all_rows.append(row)
                    if fout:
                        fout.write(json.dumps(row) + "\n")

                if fout:
                    fout.flush()

                elapsed = time.time() - t0
                sigma_done += 1
                done_count += 1
                remaining = total_records - done_count
                eta = (time.time() - t_sigma_start) / sigma_done * max(remaining, 0)
                warn_tag = " WARN" if is_warn else ""
                print(
                    f"  [{i:3d}/{len(sample)}]  {rec.id:<22}"
                    f"  {elapsed:.1f}s  ETA {eta/60:.0f}m{warn_tag}"
                )

            print()
    finally:
        if fout:
            fout.close()

    return all_rows


# ── model loading ─────────────────────────────────────────────────────────────

def _load_model(backend: str, model_path: str):
    if backend == "llava":
        _llava = os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        if _llava not in sys.path:
            sys.path.insert(0, _llava)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.hooks import LlavaHookManager
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, model_base=None, model_name=model_name,
            attn_implementation="sdpa",
        )
        model.eval()
        return LlavaHookManager(model, tokenizer, image_processor), tokenizer

    elif backend == "vilau":
        _vilau = os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
        if _vilau not in sys.path:
            sys.path.insert(0, _vilau)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks import VilaUHookManager
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, attn_implementation="eager",
        )
        model.eval()
        return VilaUHookManager(model, tokenizer, image_processor), tokenizer

    else:  # vila
        _vila = os.path.join(os.path.dirname(__file__), "..", "..", "VILA")
        if _vila not in sys.path:
            sys.path.insert(0, _vila)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.hooks.vila import VilaHookManager
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name,
        )
        model.eval()
        return VilaHookManager(model, tokenizer, image_processor), tokenizer


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="σ-sensitivity audit for activation-patching restoration scores"
    )
    parser.add_argument("--backend", default="llava", choices=["llava", "vilau", "vila"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument(
        "--sigmas", default=None,
        help="Comma-separated σ values (3 recommended). "
             "Defaults: llava=0.05,0.1,0.2 | vila=0.25,0.5,1.0 | vilau=0.5,1.0,2.0",
    )
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument(
        "--out", default=None,
        help="Output JSONL (resumable). Default: results/audit_{backend}.jsonl",
    )
    parser.add_argument(
        "--report_only", action="store_true",
        help="Skip model loading; print table from --out file",
    )
    parser.add_argument(
        "--combine", nargs="+", metavar="JSONL",
        help="Print combined multi-model table from these JSONL files (no model needed)",
    )
    parser.add_argument(
        "--labels", nargs="+",
        help="Model labels for --combine (same order as files)",
    )
    args = parser.parse_args()

    print_criterion_summary()

    # ── combined table mode ───────────────────────────────────────────────────
    if args.combine:
        files  = [Path(p) for p in args.combine]
        labels = args.labels if args.labels else [p.stem for p in files]
        if len(labels) != len(files):
            parser.error("--labels must have the same count as --combine files")
        print_combined_table(files, labels)
        return

    out_path = Path(args.out) if args.out else Path(f"results/audit_{args.backend}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── report-only mode ──────────────────────────────────────────────────────
    if args.report_only:
        rows = _load_rows(out_path)
        if not rows:
            print(f"No rows found in {out_path}.")
            return
        print_table(rows, model_label=args.backend)
        return

    if not args.model_path:
        parser.error("--model_path is required unless --report_only or --combine is set")

    default_sigmas: dict[str, list[float]] = {
        "llava":  [0.25, 0.5,  1.0],
        "vila":   [0.25, 0.5,  1.0],
        "vilau":  [0.5,  1.0,  2.0],
    }
    sigmas: list[float] = (
        [float(s) for s in args.sigmas.split(",")]
        if args.sigmas
        else default_sigmas[args.backend]
    )
    print(f"Backend : {args.backend}")
    print(f"σ values: {sigmas}")
    print(f"Samples : {args.n_samples}")
    print(f"Output  : {out_path}\n")

    # ── load model ────────────────────────────────────────────────────────────
    hm, tokenizer = _load_model(args.backend, args.model_path)

    from probe import load_cache, resolve_answer_token_ids
    records, _, _ = load_cache()
    resolve_answer_token_ids(records, tokenizer)

    done_pairs = _load_done_pairs(out_path)
    if done_pairs:
        n_sigmas = len({s for _, s in done_pairs})
        print(f"Resuming: {len(done_pairs)} (record_id, σ) pairs already in {out_path}\n")

    rows = run_audit(
        hm, records,
        sigmas=sigmas,
        n_samples=args.n_samples,
        out_path=out_path,
        done_pairs=done_pairs,
    )

    # Reload from file so resumed rows are included
    all_rows = _load_rows(out_path)
    print_table(all_rows, model_label=args.backend)


if __name__ == "__main__":
    _main()
