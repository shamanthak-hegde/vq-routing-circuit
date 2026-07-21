# vq-routing-circuit

Code for studying an early-layer attention routing circuit in vision-language
models that tokenize images through a vector-quantized codebook.

The short version: VLMs that pass images through a VQ codebook tend to say "yes"
to questions about objects that aren't there. We traced this to a specific
routing pattern in decoder layer 0, where the last prompt token attends broadly
to the visual tokens in a way that is largely independent of what the image
actually contains. Continuous-projector VLMs (CLIP + MLP, as in LLaVA) don't show
it. Ablating layer 0 attenuates the behavior, and swapping a continuous projector
for a VQ one installs it.

This repository has the measurement code, the intervention code, the paired probe
set, and the small benchmark used throughout.

## What's here

`probe/` is the main Python package.

* `probe/hooks/` is the activation-capture layer. One hook manager per model
  family, all subclassing `VLMHookManager`. Currently: LLaVA-1.6, VILA, VILA-U,
  UniTok, Chameleon, Anole, Liquid, Janus, Emu3, SEED-LLaMA, Show-o, LaVIT,
  Lumina-mGPT, HaploVLM, GILL, AnyGPT, Qwen2.5-VL, and the VQ/MLP LLaVA variants
  we trained.
* `probe/tracing/` holds the three diagnostic gates and the intervention library.
* `probe/benchmarks/` runs POPE, AMBER, NaturalBench, HallusionBench, and CHAIR,
  with interventions applied inline.
* `probe/baselines/` implements VCD, DoLA, ITI, and VTI for comparison.
* `probe/analysis/` is post-hoc analysis: bootstrap CIs, threshold sensitivity,
  cohort summaries, CHAIR robustness checks.
* `probe/training/` trains the projector variants used for the induction
  experiment (VQ, FSQ, and a matched-compute MLP control).

`micro_benchmark/` is a 900-example yes/no benchmark drawn from NaturalBench,
POPE, and HallusionBench, with images included. `scripts/` has sweep drivers and
figure generators. `results/` holds a few small summary outputs; the bulk
per-record generation logs are not checked in.

## Setup

You need Python 3.11 and a recent PyTorch with CUDA. Install the Python
dependencies:

```bash
pip install -r requirements.txt
```

Most of the model-specific hooks import from the upstream model repository rather
than from `transformers`, so you need to clone whichever models you plan to run
and put them on your `PYTHONPATH`. LLaVA needs one patch: `forward()` in
`llava/model/language_model/llava_llama.py` must accept `cache_position`,
`logits_to_keep`, and `**kwargs` to work with newer transformers.

The benchmark loaders expect COCO `val2014` images and the AMBER data under
`data/`, and CHAIR scoring needs COCO annotations in
`probe/benchmarks/coco_annotations/`. None of that is checked in.

Run everything from the repository root. Several modules resolve paths relative
to the working directory.

## The probe set

500 paired records over NaturalBench and POPE. Every `pair_id` groups exactly two
records, one answering "yes" and one answering "no", so you always have a minimal
contrast to patch between. NaturalBench pairs are two different images against
the same question; POPE pairs are two different questions against the same image,
with corruption applied as noise on the visual embeddings.

```python
from probe import build_probe_set, load_cache, resolve_answer_token_ids

records, pixel_values, foil_pixel_values = build_probe_set()  # first run
records, pixel_values, foil_pixel_values = load_cache()       # afterwards
resolve_answer_token_ids(records, tokenizer)
```

`build_probe_set()` downloads the source datasets and precomputes about 518 MB of
pixel tensors into `probe/cached/`. That cache is not in the repository, so the
first call takes a while.

Only yes/no visual-grounding questions are included. Don't run the tracing code
on captioning data; the token-position bookkeeping assumes a single-token answer.

## The three gates

The diagnostic is a chain. A model has to pass all three to count as carrying the
circuit.

Gate A.1 asks whether corrupting the visual tokens produces an early, peaked
divergence in the residual stream at the last prompt position:

```bash
python -m probe.tracing.residual_divergence \
    --backend vilau --model_path mit-han-lab/vila-u-7b-256 \
    --sigma 1.0 --out results/residual_divergence_vilau.json
```

Gate A.2 measures how much layer-0 attention mass runs from the last prompt token
to the visual tokens:

```bash
python -m probe.tracing.attn_head_weights \
    --backend vilau --model_path mit-han-lab/vila-u-7b-256 \
    --out results/attn_head_weights_vilau.json
```

Gate A.3 checks whether noise pushes visual embeddings off the codebook manifold,
which only applies to models with a collapsed VQ codebook:

```bash
python -m probe.tracing.codebook_probe_offmanifold \
    --backend llava_vq --model_path liuhaotian/llava-v1.6-vicuna-7b \
    --projector_ckpt checkpoints/llava_vq/projector_final.pt \
    --sweep results/sweep_llava_vq.jsonl \
    --n_records 200 --sigma 2.0 \
    --out results/codebook_probe_llava_vq.jsonl
```

The thresholds are 0.4, 15, and 80% respectively. Gate A.2 uses a two-tier rule
because projector-amplified and unified-vocab architectures concentrate their
routing mass differently. `probe/analysis/gate_threshold_sensitivity.py` sweeps
all three thresholds and reports how many verdicts flip, which is the honest way
to check the thresholds weren't fitted to the outcome.

## Interventions

`run_bench.py` applies interventions inline on the binary benchmarks:

```bash
python -m probe.benchmarks.run_bench \
    --bench pope_full --backend vilau \
    --model_path mit-han-lab/vila-u-7b-256 \
    --knockout_layer 0 \
    --out results/bench_pope_vilau_L0.jsonl
```

The canonical intervention zeroes the whole self-attention output of layer 0. It
is registered as `pathological_route_ablation`; `full_zero` still works as a
deprecated alias. There are also selective (per-head) and scalar (scaled rather
than zeroed) variants, which matter for the dose-response analysis.

For open-ended captioning, `run_chair.py` does the same thing during generation.
This is where the intervention separates from the decoding-time baselines: tuned
DoLA wins on binary yes/no calibration, but only the layer-0 ablation moves CHAIR
on free-form captions.

## Caveats

The layer-0 ablation is not uniformly safe. On some architectures it is close to
harmless, on others it is catastrophic, and on at least one model it produces a
degenerate emitter that scores well only because it stops producing real output.
`probe/tracing/liquid_residual_localization.py` exists specifically to
distinguish that case from a genuine effect, and the sanity-check reporters flag
it. Check the logit-gap output before reading an accuracy gain as real.

Gate A.2 is not measurable on every model. Some vendored backbones never
materialize attention weights, so the hook manager raises rather than silently
returning zeros.

## Tests

```bash
python -m probe.hooks.test_hooks                    # structural, no GPU
python -m probe.hooks.test_hooks --model_path ...   # full, needs GPU
```

There are per-model test modules alongside it for the other backends.
