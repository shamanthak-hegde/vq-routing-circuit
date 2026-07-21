
# Back-fill the 4 missing AMBER cells in tab:cohort-behavior:
#   LLaVA-VQ-K4096, LLaVA-VQ-K65536, LLaVA-MLP ctrl, Lumina-mGPT
# Each needs a baseline + an L0 (pathological_route_ablation) run, then a paired bootstrap.
# Requires a GPU node and the `sae` conda env. run_bench is resumable (safe to re-run).
set -euo pipefail
cd "$(dirname "$0")/.."


MP_LLAVA="liuhaotian/llava-v1.6-vicuna-7b"

# tag  backend     projector_ckpt                                  model_path
RUNS=(
  "llava_vq_K4096   llava_vq    checkpoints/llava_vq_K4096/projector_final.pt    $MP_LLAVA"
  "llava_vq_K65536  llava_vq    checkpoints/llava_vq_K65536/projector_final.pt   $MP_LLAVA"
  "llava_mlp        llava_mlp   checkpoints/llava_mlp/projector_final.pt         $MP_LLAVA"
  "lumina_mgpt      lumina_mgpt -                                                Alpha-VLLM/Lumina-mGPT-7B-768"
)

for row in "${RUNS[@]}"; do
  read -r tag backend ckpt mp <<<"$row"
  ckpt_arg=(); [[ "$ckpt" != "-" ]] && ckpt_arg=(--projector_ckpt "$ckpt")

  # echo "=== $tag : AMBER baseline ==="
  # python -m probe.benchmarks.run_bench --bench amber --backend "$backend" \
  #   --model_path "$mp" "${ckpt_arg[@]}" \
  #   --out "results/bench_amber_${tag}_baseline.jsonl"

  # echo "=== $tag : AMBER L0 ablation ==="
  # python -m probe.benchmarks.run_bench --bench amber --backend "$backend" \
  #   --model_path "$mp" "${ckpt_arg[@]}" \
  #   --knockout_mode pathological_route_ablation --knockout_layer 0 \
  #   --out "results/bench_amber_${tag}_L0.jsonl"

  echo "=== $tag : paired bootstrap (acc + yes_rate) ==="
  for metric in acc yes_rate; do
    python -m probe.analysis.paired_bootstrap \
      --baseline "results/bench_amber_${tag}_baseline.jsonl" \
      --intervention "results/bench_amber_${tag}_L0.jsonl" \
      --metric "$metric" --n_boot 10000 \
      --out "results/bootstrap_${tag}_amber_L0$([[ $metric == yes_rate ]] && echo _yr).json"
  done
done
echo "DONE. base/L0 accuracy + yes_rate are in results/bench_amber_*_{baseline,L0}.meta.json;"
echo "CIs in results/bootstrap_*_amber_L0*.json"
