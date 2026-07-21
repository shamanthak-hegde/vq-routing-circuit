#!/usr/bin/env bash
# head-level localization of the L0 circuit on VILA-U.
#
# L0 has 32 heads; the visual->prompt_last routing mass is distributed (top 22/32
# carry 80%; results/attn_head_weights_vilau.json). This tests causally whether a
# small head subset reproduces the full-L0 ablation effect (+4.23pp POPE acc) or
# whether the circuit is distributed:
#   top8 = 8 highest-mass L0 heads (33% of routing mass)
#   bot8 = 8 lowest-mass  L0 heads (13% of routing mass)
#   rnd8 = 8 random L0 heads (control)
# Compare to baseline (61.93%) and full-L0-all-32 (66.17%, +4.23pp).
#
# Usage (GPU node, sae env):
#   PATH=$CONDA_PREFIX/bin:$PATH
#   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#   export CUDA_VISIBLE_DEVICES=<free gpu>
#   bash scripts/run_head_localization.sh

set -euo pipefail
MODEL_PATH="mit-han-lab/vila-u-7b-256"
BENCH="pope_full"

declare -A SETS=(
  [top8]="2,3,6,7,13,14,17,23"
  [bot8]="0,8,15,18,25,26,28,29"
  [rnd8]="1,8,13,15,16,24,28,31"
)

for TAG in top8 bot8 rnd8; do
    OUT="results/bench_pope_full_vilau_L0heads_${TAG}.jsonl"
    if [ ! -f "$OUT" ]; then
        echo "=== L0 selective heads ${TAG} = ${SETS[$TAG]} ==="
        python -m probe.benchmarks.run_bench \
            --backend vilau --model_path "$MODEL_PATH" --bench "$BENCH" \
            --knockout_mode selective --knockout_layer 0 --heads "${SETS[$TAG]}" \
            --max_records "${MAXREC:-300}" \
            --out "$OUT"
    else
        echo "  [skip] $OUT"
    fi
done

echo ""
echo "Summarize: python -m probe.benchmarks.summarize results/bench_pope_full_vilau_L0heads_*.jsonl"
