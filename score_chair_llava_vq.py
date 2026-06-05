from probe.benchmarks.chair import score_chair_from_jsonl
coco_path = "probe/benchmarks/coco_annotations"
for cond in ["baseline", "L0"]:
    result = score_chair_from_jsonl(f"results/chair_captions_llava_vq_K65536_{cond}.jsonl", coco_path)
    m = result["overall_metrics"]
    print(f"llava_vq_K65536 {cond:8s}  CHAIRs={m['CHAIRs']*100:.1f}  CHAIRi={m['CHAIRi']*100:.1f}  Recall={m['Recall']*100:.1f}  AvgLen={m['Len']*100:.1f}")
