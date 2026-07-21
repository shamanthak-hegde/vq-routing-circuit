#!/usr/bin/env bash
# VILA-U CHAIR L0-attenuation dose-response.
# Interior alpha only; endpoints (baseline=coeff1.0, full L0=coeff0.0) already on disk.
#   scale_alpha = retained fraction (ScalarLayerScale coeff); attenuation strength = 1 - coeff.
#   coeff 0.75 -> attenuation 0.25 ; 0.50 -> 0.50 ; 0.25 -> 0.75
# Env: vila-u.  Run from repo root.
#   PATH=$CONDA_PREFIX/bin:$PATH
#   LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
set -euo pipefail
MODEL=mit-han-lab/vila-u-7b-256
for C in 0.75 0.50 0.25; do
  echo "=== coeff=$C (attenuation $(python3 -c "print(1-$C)")) ==="
  python -m probe.benchmarks.run_chair --backend vilau --model_path "$MODEL" \
    --knockout_layer 0 --scale_alpha "$C" \
    --n_images 500 --seed 0 \
    --out "results/chair_captions_vilau_scalarL0_c${C}.jsonl"
done
echo "ALL DOSE-RESPONSE RUNS COMPLETE"
