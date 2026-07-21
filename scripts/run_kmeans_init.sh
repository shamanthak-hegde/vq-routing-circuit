#!/bin/bash
set -e
# module load mamba;
# source activate sae
cd "$(dirname "$0")/.."

python -m probe.training.kmeans_init \
    --data_json data/llava_pretrain/subset_50k.json \
    --image_dir data/llava_pretrain/images \
    --model_path liuhaotian/llava-v1.6-vicuna-7b \
    --n_samples 10000 \
    --codebook_size 16384 \
    --code_dim 32 \
    --n_iters 50 \
    --out checkpoints/codebook_init.pt
