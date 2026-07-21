#!/bin/bash
set -e
# module load mamba;
# source activate sae
cd "$(dirname "$0")/.."

# ── Step 3: Full sweep (produces non-WARN record pool) ───────────────────────
echo "=== Step 3: Causal sweep ==="
python -m probe.tracing.run_sweep --backend llava_vq --model_path liuhaotian/llava-v1.6-vicuna-7b --projector_ckpt checkpoints/llava_vq/projector_final.pt --sigma 2.0 --out results/sweep_llava_vq.jsonl

# ── Step 4: Gate A.1 — residual divergence on POPE ───────────────────────────
echo "=== Step 4: Gate A.1 — residual divergence ==="
python -m probe.tracing.residual_divergence --backend llava_vq --model_path liuhaotian/llava-v1.6-vicuna-7b --projector_ckpt checkpoints/llava_vq/projector_final.pt --sigma 2.0 --source pope --out results/residual_divergence_llava_vq.json

# ── Step 5: Gate A.2 — head-weight analysis (run only if Gate A.1 passes) ────
echo "=== Step 5: Gate A.2 — head-weight analysis ==="
python -m probe.tracing.attn_head_weights --backend llava_vq --model_path liuhaotian/llava-v1.6-vicuna-7b --projector_ckpt checkpoints/llava_vq/projector_final.pt --source pope --n_records 20 --records_from results/sweep_llava_vq.jsonl --out results/attn_head_weights_llava_vq.json
