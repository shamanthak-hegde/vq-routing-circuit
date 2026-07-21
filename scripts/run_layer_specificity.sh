#!/usr/bin/env bash
# intervention specificity controls on VILA-U.
#
# Tests whether the L0 ablation benefit is *localized* to L0 or a generic
# degradation that any-layer ablation would reproduce. Uses the existing
# run_bench.py intervention modes (no new code):
#   (A) random-LAYER ablation: pathological_route_ablation at L0/L2/L4/L8/L16.
#       If only L0 (and the known L1) move POPE meaningfully, the effect is
#       layer-localized, not generic.
#   (B) attention-output SCALING at L0: scalar mode, alpha in {0.25,0.5,0.75}.
#       A graded dose-response (vs. all-or-nothing) argues for a real routing
#       channel rather than a brittle zeroing artifact.
#
# Baseline (bench_pope_full_vilau_baseline.jsonl) and L0 ablation
# (bench_pope_full_vilau_L0.jsonl) already exist; this only fills the controls.
#
# Usage (GPU node, sae env):
#   source activate sae
#   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#   bash scripts/run_layer_specificity.sh
#
# Expected runtime: ~6 min/run on A100 x (5 layers + 3 alphas) ≈ 50 min.
# Outputs: results/bench_pope_full_vilau_L{2,4,8,16}.jsonl,
#          results/bench_pope_full_vilau_scalarL0_a{0.25,0.50,0.75}.jsonl
# Then summarize with: python -m probe.benchmarks.summarize results/bench_pope_full_vilau_*.jsonl

set -euo pipefail
MODEL_PATH="mit-han-lab/vila-u-7b-256"
BENCH="pope_full"

# (A) random-layer ablation specificity
for L in 2 4 8 16; do
    OUT="results/bench_${BENCH}_vilau_L${L}.jsonl"
    if [ ! -f "$OUT" ]; then
        echo "=== pathological_route_ablation @ L${L} ==="
        python -m probe.benchmarks.run_bench \
            --backend vilau --model_path "$MODEL_PATH" --bench "$BENCH" \
            --knockout_mode pathological_route_ablation --knockout_layer "$L" \
            --out "$OUT"
    else
        echo "  [skip] $OUT"
    fi
done

# (B) attention-output scaling dose-response at L0
for A in 0.25 0.50 0.75; do
    OUT="results/bench_${BENCH}_vilau_scalarL0_a${A}.jsonl"
    if [ ! -f "$OUT" ]; then
        echo "=== scalar L0 scaling alpha=${A} ==="
        python -m probe.benchmarks.run_bench \
            --backend vilau --model_path "$MODEL_PATH" --bench "$BENCH" \
            --knockout_mode scalar --knockout_layer 0 --alpha "$A" \
            --out "$OUT"
    else
        echo "  [skip] $OUT"
    fi
done

echo ""
echo "Done. Summarize:"
echo "  python -m probe.benchmarks.summarize results/bench_${BENCH}_vilau_*.jsonl"
