#!/bin/bash
set -e
# module load mamba;
# source activate sae
cd /scratch/shegde23/unified_mech

python -m probe.training.train_llava_vq \
    --model_path liuhaotian/llava-v1.6-vicuna-7b \
    --data_json data/llava_pretrain/subset_50k.json \
    --image_dir data/llava_pretrain/images \
    --codebook_init checkpoints/codebook_init.pt \
    --out_dir checkpoints/llava_vq \
    --max_steps 2000
