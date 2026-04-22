# unified_mech — VLM Mechanistic Interpretability

Paired micro-benchmark and probe infrastructure for mechanistic interpretability
experiments on vision-language models.

---

## Repository layout

```
unified_mech/
├── micro_benchmark/          # 900-example raw benchmark (JSONL + images)
│   ├── build_benchmark.py    # build script (run once to reproduce)
│   ├── micro_benchmark.jsonl # 900 records, one per line
│   └── images/
│       ├── naturalbench/     # 103 MB — 300 JPEG images
│       ├── pope/             #  15 MB — ~150 JPEG images
│       └── hallusionbench/   # 7.6 MB — ~150 JPEG images
│
├── probe/                    # Python module — paired probe set + hook layer
│   ├── __init__.py           # public API + build_probe_set()
│   ├── schema.py             # ProbeRecord dataclass
│   ├── naturalbench.py       # NaturalBench loader (300 records)
│   ├── pope.py               # POPE loader (200 records)
│   ├── cache.py              # precompute / load pixel-value tensors
│   ├── cached/               # 518 MB precomputed cache
│   │   ├── records.json
│   │   ├── pixel_values.pt
│   │   └── foil_pixel_values.pt
│   └── hooks/                # activation-capture hook layer
│       ├── __init__.py       # public API: LlavaHookManager, Capture, TokenCategory
│       ├── schema.py         # TokenCategory, TokenIndex, Capture dataclasses
│       ├── base.py           # abstract VLMHookManager (subclass for VILA-U in Week 3)
│       ├── llava.py          # LlavaHookManager (LLaVA-1.6 concrete impl)
│       ├── utils.py          # hook registration + finalization helpers
│       └── test_hooks.py     # smoke test
│
├── DreamLLM/
├── HaploVLM/
├── LLaVA/
├── Qwen3-VL/
├── UniTok/
├── VILA/
└── vila-u/
```

---

## micro_benchmark — 900-example raw benchmark

Built from three hallucination / visual-grounding datasets, all with yes/no
answers (single-token targets for logit restoration).  Captioning data is
explicitly excluded.

| Source | Count | yes | no | Notes |
|--------|------:|----:|---:|-------|
| NaturalBench | 600 | 300 | 300 | yes/no questions only; paired image structure |
| POPE (adversarial) | 150 | 75 | 75 | object existence questions |
| HallusionBench (image) | 150 | 75 | 75 | illusion / math / figure / ocr / map subcategories |
| **Total** | **900** | **450** | **450** | 50 % yes, 50 % no |

### Record schema (`micro_benchmark.jsonl`)

```jsonc
{
  "id":         "nb_1279_img0_q1",
  "source":     "naturalbench",          // "naturalbench" | "pope" | "hallusionbench"
  "pair_id":    "nb_1279",               // groups paired QA sharing the same image context
  "image_path": "images/naturalbench/1279_img0.jpg",   // relative to micro_benchmark/
  "question":   "Is there someone's hair red?",
  "answer":     "no",                    // always lowercase "yes" or "no"
  "metadata":   { ... }                  // source-specific extras
}
```

### Rebuild

```bash
source activate sae
cd micro_benchmark/
python build_benchmark.py
```

NaturalBench is downloaded from `BaiqiL/NaturalBench` on HuggingFace.
POPE from `lmms-lab/POPE`, HallusionBench from `lmms-lab/HallusionBench`.

---

## probe — paired probe set for activation patching

A Python module that exposes a **unified `(x, x′, q, y)` interface** over
NaturalBench and POPE.  500 records total, precomputed pixel-value tensors
saved to `probe/cached/`.

**Do not run tracing experiments on captioning data.**  The probe set
contains only yes/no visual-grounding questions.

### Composition

| Source | Records | Pairs | Corruption mode |
|--------|--------:|------:|-----------------|
| NaturalBench | 300 | 150 pair_ids × 2 questions | `image_swap` |
| POPE (adversarial) | 200 | 100 pair_ids × 2 questions | `gaussian_noise` |
| **Total** | **500** | **250** | |

Each `pair_id` groups exactly **2 records** — one "yes" and one "no" — sharing
the same image context.  This makes it straightforward to set up a
paired patching run.

### Quick start

```python
# build once (downloads datasets, saves images, precomputes tensors)
from probe import build_probe_set
records, pixel_values, foil_pixel_values = build_probe_set()

# load on subsequent runs
from probe import load_cache
records, pixel_values, foil_pixel_values = load_cache()

# attach answer token IDs once you have the model tokenizer
from probe import resolve_answer_token_ids
resolve_answer_token_ids(records, tokenizer)
```

### ProbeRecord schema

```python
@dataclass
class ProbeRecord:
    id:                str
    source:            "naturalbench" | "pope"
    pair_id:           str          # groups the two paired records
    image_path:        str          # absolute path to clean JPEG
    foil_image_path:   str | None   # see corruption modes below
    question:          str
    answer:            "yes" | "no"
    answer_token_id:   int | None   # None until resolve_answer_token_ids()
    corruption_mode:   "image_swap" | "gaussian_noise"
    metadata:          dict
```

### Corruption modes

**`image_swap` — NaturalBench**

`x` and `x′` are two distinct real images.  For a given question `q`, one
image answers "yes" and the other "no".  Both have real JPEG files on disk;
`foil_image_path` points to `x′`.

Typical patching protocol: run model on `x′`, collect residual-stream or
attention activations at layer `l`, patch them into the forward pass on `x`,
observe whether the logit for the correct answer is restored.

**`gaussian_noise` — POPE**

`x′` is `x` with zero-mean Gaussian noise injected into the **visual patch
embeddings** after the visual encoder, before transformer layer 0.
`foil_image_path` is `None` — there is no foil JPEG.  Noise must be applied
at the visual-token level at inference time.

Suggested noise scale: tune σ on a held-out set so that
cosine-sim(clean embedding, noisy embedding) ≈ 0.5.

The pairing for POPE: two questions on the **same image**, one with answer
"yes" and one with answer "no", share a `pair_id`.  This lets experiments
ask which components encode the presence/absence signal vs. question semantics.

### Precomputed cache

Images are preprocessed with a standard ViT-compatible transform
(resize shortest edge to 336, centre-crop 336×336, ImageNet normalisation)
and stored as **float16** tensors of shape `(3, 336, 336)`.

| File | Contents | Size |
|------|----------|------|
| `cached/records.json` | all 500 ProbeRecord dicts (no tensors) | — |
| `cached/pixel_values.pt` | `{id → tensor}` clean images, all 500 | |
| `cached/foil_pixel_values.pt` | `{id → tensor}` foil images, NB only (300) | |
| **Total** | | **518 MB** |

Pass `transform=your_processor` to `build_cache()` to use a model-specific
image processor instead of the default.  Pass `image_size=224` for
CLIP-ViT-L/14 models.

### Iterating paired records

```python
from itertools import groupby
from probe import load_cache

records, pv, foil_pv = load_cache()

# sort by pair_id so groupby works
records.sort(key=lambda r: r.pair_id)

for pair_id, group in groupby(records, key=lambda r: r.pair_id):
    pair = list(group)          # always length 2
    yes_rec = next(r for r in pair if r.answer == "yes")
    no_rec  = next(r for r in pair if r.answer == "no")

    x_clean  = pv[yes_rec.id]                     # (3, 336, 336) float16
    x_foil   = foil_pv.get(yes_rec.id)            # tensor or None (POPE)
    # ... run patching experiment
```

---

---

## probe/hooks — activation-capture hook layer (LLaVA-1.6)

Four hook points exposed during a single model forward pass:

| # | Hook | Module | Shape |
|---|------|--------|-------|
| 1 | Projected visual tokens | `model.model.mm_projector` | `(num_crops, 576, H)` |
| 2 | Self-attention output | `model.model.layers[i].self_attn` | `(n_layers, seq, H)` |
| 3 | MLP output | `model.model.layers[i].mlp` | `(n_layers, seq, H)` |
| 4 | Residual stream | `model.model.layers[i]` | `(n_layers, seq, H)` |

### Token-category abstraction

Every embedding position is tagged with one of four categories:

| Category | Value | Positions |
|----------|-------|-----------|
| `OTHER`    | 0 | BOS, system prompt, chat-template scaffolding, role tags |
| `VISUAL`   | 1 | Image patch tokens (including anyres newline tokens) |
| `QUESTION` | 2 | User's question text |
| `ANSWER`   | 3 | Model-generated tokens (generate mode only) |

This mapping is built **inside the hook layer**, not in the tracing code, so
heatmap axes stay comparable when porting to VILA-U in Week 3.  Only
`LlavaHookManager._locate_assistant_tag()` needs to be overridden per model.

### Quick start

```python
from probe.hooks import LlavaHookManager, TokenCategory

# load model (done once outside the loop)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path, model_base=None,
    model_name=get_model_name_from_path(model_path),
    attn_implementation="sdpa",
)
model.eval()

hm = LlavaHookManager(model, tokenizer, image_processor)

# prefill — primary path for yes/no logit restoration
cap = hm.run_prefill(image=PIL_img, question="Is there a cat?")

# generate — captures prefill + all decode steps
cap = hm.run_generate(image=PIL_img, question="Is there a cat?",
                      max_new_tokens=8)

# analysis
last_res  = cap.last_token_residual()                      # (32, 4096)
vis_res   = cap.by_category(TokenCategory.VISUAL)          # (32, n_img, 4096)
ans_res   = cap.by_category(TokenCategory.ANSWER)          # (32, n_gen, 4096)
logit_map = cap.logit_lens(model.lm_head, model.model.norm) # (32, seq, vocab)
yes_no    = cap.logits[cap.token_index.prompt_last, [yes_id, no_id]]
```

Opt-in attention weights (requires `attn_implementation="eager"`):

```python
hm = LlavaHookManager(model, tokenizer, image_processor,
                      capture_attention_weights=True)
cap = hm.run_prefill(img, q)
# cap.attn_weights: (n_layers, n_heads, seq_len, seq_len)
```

### Smoke test

```bash
source activate sae
cd /scratch/shegde23/unified_mech

# structural tests only (no GPU needed)
python -m probe.hooks.test_hooks

# full real-model tests
python -m probe.hooks.test_hooks --model_path /path/to/llava-v1.6-vicuna-7b
```

---

## Environment

```bash
source activate sae   # Python 3.11, torch 2.11+cu130, transformers 4.57
```

Required packages: `datasets`, `torch`, `torchvision`, `Pillow`, `tqdm`.
