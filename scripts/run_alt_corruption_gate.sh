#!/usr/bin/env bash
# alternate visual corruptions for Gate A.1.
#
# Re-runs the residual-divergence (Gate A.1) measurement under three non-Gaussian
# corruptions to test whether the early-layer (L0/L1) restoration peak is specific
# to embedding-space Gaussian noise or persists across corruption types:
#   dropout  — zero 50% of visual tokens
#   shuffle  — permute the visual-token embeddings
#   blur     — Gaussian-blur the pixel image before the encoder (image-space)
#
# Run on the two cleanest specimens (one projector-amplified, one unified-vocab);
# extend to others as desired. Compare the early-window peak + peak-and-decline
# shape against the canonical Gaussian results
# (results/residual_divergence_{vilau,chameleon}.json).
#
# Usage (GPU node, sae env):
#   source activate sae
#   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#   bash scripts/run_alt_corruption_gate.sh
#
# Outputs: results/residual_divergence_<backend>_{dropout,shuffle,blur}.json
# (canonical gaussian files are NOT overwritten — alternate modes get a suffix).

set -euo pipefail
RD="python -m probe.tracing.residual_divergence"

# backend  model_path  sigma(for-reference-only; unused by dropout/shuffle/blur)
run_one () {
    local BACKEND="$1" MODEL="$2" SIGMA="$3"
    for CORR in dropout shuffle blur; do
        OUT="results/residual_divergence_${BACKEND}_${CORR}.json"
        if [ ! -f "$OUT" ]; then
            echo "=== ${BACKEND} / ${CORR} ==="
            $RD --backend "$BACKEND" --model_path "$MODEL" --sigma "$SIGMA" \
                --corruption "$CORR"
        else
            echo "  [skip] $OUT"
        fi
    done
}

run_one vilau     "mit-han-lab/vila-u-7b-256"  1.0
run_one chameleon "facebook/chameleon-7b"      0.00625

echo ""
echo "Compare early[0,8) peaks vs canonical gaussian:"
echo "  python -c \"import json,glob; [print(f, round(max(json.load(open(f))['mean_rel_div'][:8]),3)) for f in sorted(glob.glob('results/residual_divergence_vilau*.json')+glob.glob('results/residual_divergence_chameleon*.json'))]\""
