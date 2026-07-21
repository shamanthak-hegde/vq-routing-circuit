#!/usr/bin/env bash
# VTI demo-seed variance for VILA-U (, sub-task 4B).
#
# Runs VTI calibration with 5 different demo seeds (0-4), each selecting
# a different random subset of 200 POPE records for direction computation.
# For each seed, runs POPE + AMBER bench with the resulting direction.
# Paired-record bootstrap CIs are computed afterward with paired_bootstrap.py.
#
# Usage (from repo root, with sae environment active):
#   source activate sae
#   bash scripts/run_vti_seed_variance.sh
#
# Expected runtime: ~3h on A100 (5 seeds × ~30min calibration + ~6min bench each)
# Outputs:
#   results/vti_direction_vilau_seed{0..4}.pt
#   results/bench_pope_full_vilau_vti_seed{0..4}.jsonl
#   results/bench_amber_vilau_vti_seed{0..4}.jsonl
#   results/vti_seed_summary.tsv   (written by probe.analysis.vti_seed_summary)

set -euo pipefail

MODEL_PATH="mit-han-lab/vila-u-7b-256"
SIGMA=1.0
N_DEMOS=200

for SEED in 0 1 2; do
    echo ""
    echo "=== VTI seed=$SEED ==="

    DIR_OUT="results/vti_direction_vilau_seed${SEED}.pt"

    if [ ! -f "$DIR_OUT" ]; then
        python -m probe.baselines.vti_calibrate \
            --backend vilau \
            --model_path "$MODEL_PATH" \
            --sigma "$SIGMA" \
            --n_demos "$N_DEMOS" \
            --seed "$SEED" \
            --out "$DIR_OUT"
    else
        echo "  [skip] direction already exists: $DIR_OUT"
    fi

    for BENCH in pope_full amber; do
        OUT="results/bench_${BENCH}_vilau_vti_seed${SEED}.jsonl"
        if [ ! -f "$OUT" ]; then
            python -m probe.benchmarks.run_bench \
                --backend vilau \
                --model_path "$MODEL_PATH" \
                --bench "$BENCH" \
                --knockout_mode vti_textual \
                --vti_direction "$DIR_OUT" \
                --alpha 0.005 \
                --out "$OUT"
        else
            echo "  [skip] bench already exists: $OUT"
        fi
    done
done

echo ""
echo "=== Computing seed summary ==="
python -m probe.analysis.vti_seed_summary \
    --results_dir results \
    --out results/vti_seed_summary.tsv

echo ""
echo "Done. Results in results/vti_seed_summary.tsv"
