# vq-routing-circuit

Tools for measuring an early-layer attention routing circuit in vision-language
models that encode images with a vector-quantized codebook, and for turning that
circuit off to see what changes.

Models with this circuit tend to answer "yes" when asked about objects that are
not in the image. The code here lets you test a model for the circuit, ablate it,
and measure the effect on hallucination benchmarks.

## Install

Python 3.11 and a CUDA build of PyTorch.

```bash
pip install -r requirements.txt
```

Each model family is loaded through its own upstream repository, so clone the
models you want to run and add them to `PYTHONPATH`. For LLaVA, edit
`llava/model/language_model/llava_llama.py` so `forward()` accepts
`cache_position`, `logits_to_keep`, and `**kwargs`; without that it fails on
current transformers.

Datasets are not included. Put COCO `val2014` and AMBER under `data/`, and COCO
annotations under `probe/benchmarks/coco_annotations/`.

Run every command from the repository root.

## Build the probe set

Downloads the source datasets and caches the preprocessed images, about 518 MB
into `probe/cached/`. Do this once before anything else.

```bash
python -c "from probe import build_probe_set; build_probe_set()"
```

Afterwards, load it in Python:

```python
from probe import load_cache, resolve_answer_token_ids
records, pixel_values, foil_pixel_values = load_cache()
resolve_answer_token_ids(records, tokenizer)
```

You get 500 records in 250 pairs. Each pair is one "yes" record and one "no"
record, so you always have a minimal contrast to patch between. All questions are
yes/no. The tracing code assumes a single-token answer, so it will not work on
captioning data.

## Test a model for the circuit

Three measurements. A model needs all three to count as carrying the circuit.

Does corrupting the image move the residual stream early, at the last prompt
position? Threshold 0.4.

```bash
python -m probe.tracing.residual_divergence \
    --backend vilau --model_path mit-han-lab/vila-u-7b-256 \
    --sigma 1.0 --out results/residual_divergence_vilau.json
```

How much layer-0 attention runs from the last prompt token to the image tokens?
Threshold 15.

```bash
python -m probe.tracing.attn_head_weights \
    --backend vilau --model_path mit-han-lab/vila-u-7b-256 \
    --out results/attn_head_weights_vilau.json
```

Does noise push the image embeddings off the codebook? Threshold 80 percent.
Only applies to models whose codebook has collapsed.

```bash
python -m probe.tracing.codebook_probe_offmanifold \
    --backend llava_vq --model_path liuhaotian/llava-v1.6-vicuna-7b \
    --projector_ckpt checkpoints/llava_vq/projector_final.pt \
    --sweep results/sweep_llava_vq.jsonl \
    --n_records 200 --sigma 2.0 \
    --out results/codebook_probe_llava_vq.jsonl
```

To check the thresholds are not tuned to give the answer you want, sweep them and
count how many models change verdict:

```bash
python -m probe.analysis.gate_threshold_sensitivity
```

Supported backends are in `probe/hooks/`: LLaVA-1.6, VILA, VILA-U, UniTok,
Chameleon, Anole, Liquid, Janus, Emu3, SEED-LLaMA, Show-o, LaVIT, Lumina-mGPT,
HaploVLM, GILL, AnyGPT, Qwen2.5-VL, and trained LLaVA variants. Pass the name as
`--backend`. To add a model, subclass `VLMHookManager` in `probe/hooks/base.py`.

## Turn the circuit off

Run a benchmark with and without the ablation and compare. Works on POPE, AMBER,
NaturalBench, and HallusionBench.

```bash
# baseline
python -m probe.benchmarks.run_bench \
    --bench pope_full --backend vilau \
    --model_path mit-han-lab/vila-u-7b-256 \
    --out results/bench_pope_vilau_baseline.jsonl

# layer 0 ablated
python -m probe.benchmarks.run_bench \
    --bench pope_full --backend vilau \
    --model_path mit-han-lab/vila-u-7b-256 \
    --knockout_mode pathological_route_ablation --knockout_layer 0 \
    --out results/bench_pope_vilau_L0.jsonl
```

`pathological_route_ablation` zeroes the whole self-attention output of the
layer. `--knockout_mode` is required; without it the run is a baseline even if
you pass `--knockout_layer`. To zero single heads instead, use
`--knockout_mode selective --heads 6,7,14`. To scale the output rather than zero
it, use `--knockout_mode scalar --alpha 0.5`.

Valid `--bench` values are `pope_full`, `amber`, `nb_full`, and `hb`.

For free-form captioning, scored with CHAIR:

```bash
python -m probe.benchmarks.run_chair --backend vilau \
    --model_path mit-han-lab/vila-u-7b-256 \
    --knockout_layer 0 \
    --out results/chair_captions_vilau_L0.jsonl
```

Add `--smoke_test` to run four images and print the captions.

Before believing an accuracy gain, check that the model is still producing real
output. Some models go quiet under the ablation and score well only because they
stopped answering. `probe/tracing/liquid_residual_localization.py` and the
sanity-check reporters in `probe/tracing/` exist to catch that.

## Compare against other methods

VCD, DoLA, ITI, and VTI are in `probe/baselines/`. Run them through the same
benchmark runner with `--decoder vcd` or `--decoder dola`, then:

```bash
bash scripts/run_decoder_baseline_sweep.sh
python scripts/summarize_decoder_baselines.py
```

## Build the circuit into a model

`probe/training/` trains replacement projectors for LLaVA-1.6: a VQ projector, an
FSQ projector that does not collapse, and a plain MLP trained on the same data
for the same number of steps as a control.

```bash
bash scripts/run_train_llava_vq.sh
```

Then run the three measurements on the result. A VQ projector installs the
circuit; the matched MLP does not.

## Other tools

`probe/analysis/` has bootstrap confidence intervals, threshold stability checks,
and CHAIR robustness re-analysis. `scripts/gen_*.py` regenerate figures from
results already on disk. `scripts/sweep_analysis.py` builds per-layer heatmaps
from a sweep file.

## Tests

```bash
python -m probe.hooks.test_hooks                    # no GPU needed
python -m probe.hooks.test_hooks --model_path ...   # needs GPU
```

Per-model versions sit next to it, for example `test_hooks_vilau.py`.

## License

MIT, see LICENSE. Use it for research or anything else.

The images in `micro_benchmark/images/` are not covered by that. They come from
NaturalBench, POPE, and HallusionBench, and most trace back to COCO. Check those
datasets' terms before redistributing the images.
