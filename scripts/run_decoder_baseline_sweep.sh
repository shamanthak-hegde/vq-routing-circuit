#!/usr/bin/env bash
# Tuned VCD/DOLA baseline hyperparameter sweep on VILA-U.
#
# Sweeps VCD (alpha × decoder_sigma) and DoLA (alpha × early_layer) on
# POPE-full and AMBER for VILA-U, outputting to results/decoder_baseline_sweep/.
#
# Usage:
#   source activate vila-u
#   bash scripts/run_decoder_baseline_sweep.sh
#
# To resume a partial run: the benchmark runner is resumable — re-running
# the same command appends only missing records.
#
# Logs: results/decoder_baseline_sweep/logs/<run_id>.log

set -euo pipefail

MODEL_PATH="mit-han-lab/vila-u-7b-256"
BENCH_CMD="python -m probe.benchmarks.run_bench"
OUT_DIR="results/decoder_baseline_sweep"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES=0

# ── VCD hyperparameter grid ────────────────────────────────────────────────
# alpha ∈ {0.5, 1.0, 1.5, 2.0}  ×  decoder_sigma ∈ {10, 20, 40, 80}
# 40 is the default; re-running it gives a sanity-check baseline.

VCD_ALPHAS=(1.5)
VCD_SIGMAS=(80)

for BENCH in amber; do
    for ALPHA in "${VCD_ALPHAS[@]}"; do
        for SIGMA in "${VCD_SIGMAS[@]}"; do
            SLUG="bench_${BENCH}_vilau_vcd_a${ALPHA}_s${SIGMA}"
            OUT="${OUT_DIR}/${SLUG}.jsonl"
            LOG="${LOG_DIR}/${SLUG}.log"
            echo "[VCD] bench=${BENCH} alpha=${ALPHA} sigma=${SIGMA} -> ${OUT}"
            ${BENCH_CMD} \
                --bench "${BENCH}" \
                --backend vilau \
                --model_path "${MODEL_PATH}" \
                --decoder vcd \
                --alpha "${ALPHA}" \
                --decoder_sigma "${SIGMA}" \
                --out "${OUT}" \
                2>&1 | tee "${LOG}"
        done
    done
done


echo ""
echo "Sweep complete. Run scripts/summarize_decoder_baselines.py to pick winners."
