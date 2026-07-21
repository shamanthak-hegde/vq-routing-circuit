#!/bin/bash
set -e
# module load mamba;
# source activate sae
cd "$(dirname "$0")/.."

# Intervention capstone — apply pathological_route_ablation to LLaVA-VQ
# Gate: yes_rate drops + accuracy improves + circuit collapses under L0 ablation

# Step 1: Baseline accuracy (no intervention)
# echo "=== Step 1: Baseline accuracy ==="
# python -m probe.tracing.run_accuracy --backend llava_vq --model_path liuhaotian/llava-v1.6-vicuna-7b --projector_ckpt checkpoints/llava_vq/projector_final.pt --out results/accuracy_llava_vq_baseline.jsonl

# # Step 2: L0 ablation (primary intervention — pathological_route_ablation)
# echo "=== Step 2: L0 ablation ==="
# python -m probe.tracing.run_accuracy --backend llava_vq --model_path liuhaotian/llava-v1.6-vicuna-7b --projector_ckpt checkpoints/llava_vq/projector_final.pt --knockout_mode pathological_route_ablation --knockout_layer 0 --out results/accuracy_llava_vq_pathological_route_ablation.jsonl

# Step 3: Circuit collapse verification (head_knockout.py)
echo "=== Step 3: Circuit collapse verification ==="
python -m probe.tracing.head_knockout --backend llava_vq --model_path liuhaotian/llava-v1.6-vicuna-7b --projector_ckpt checkpoints/llava_vq/projector_final.pt --knockout_layer 0 --sigma 2.0 --out results/head_knockout_llava_vq.json

# Step 4 (optional): L0+L1 window ablation — uncomment if Step 2 delta is small
# echo "=== Step 4: L0+L1 window ablation ==="
# python -m probe.tracing.run_accuracy --backend llava_vq --model_path liuhaotian/llava-v1.6-vicuna-7b --projector_ckpt checkpoints/llava_vq/projector_final.pt --knockout_mode window_attn_knockout --knockout_layer 0 --knockout_layer_end 2 --out results/accuracy_llava_vq_window_attn_knockout_0_2.jsonl
