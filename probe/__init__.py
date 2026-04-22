"""
probe — paired micro-benchmark for VLM mechanistic interpretability.

Quick start
-----------
Build the cache once (downloads datasets, saves images + tensors):

    from probe import build_probe_set
    records, pixel_values, foil_pixel_values = build_probe_set()

Load on subsequent runs:

    from probe import load_cache
    records, pixel_values, foil_pixel_values = load_cache()

Attach answer token IDs when you have a tokenizer:

    from probe import resolve_answer_token_ids
    resolve_answer_token_ids(records, tokenizer)

Record schema (ProbeRecord)
---------------------------
  id                  str
  source              "naturalbench" | "pope"
  pair_id             str  — groups conceptually paired records
  image_path          str  — absolute path to clean image JPEG
  foil_image_path     str | None
                          NaturalBench → path to the other image in the pair
                          POPE         → None (foil is gaussian noise at
                                         the visual-token level; see pope.py)
  question            str
  answer              "yes" | "no"
  answer_token_id     int | None  (None until resolve_answer_token_ids is called)
  corruption_mode     "image_swap" | "gaussian_noise"
  metadata            dict  (source-specific extras)

Corruption modes
----------------
image_swap      (NaturalBench)
    x' is a real image with the semantically opposite answer.
    Use activation patching: run model on x', collect layer activations,
    patch them into the run on x, observe answer change.

gaussian_noise  (POPE)
    x' is x with zero-mean Gaussian noise injected into visual patch
    embeddings after the visual encoder (before transformer layer 0).
    foil_image_path is None.  Callers must implement the noise injection.
    Suggested σ: tune so cosine-sim(clean, noisy) ≈ 0.5 on a held-out set.
"""

from .cache import build_cache, load_cache
from .naturalbench import build_naturalbench_records
from .pope import build_pope_records
from .schema import ProbeRecord, resolve_answer_token_ids


def build_probe_set(
    nb_rows: int = 150,
    pope_pairs: int = 100,
    seed: int = 42,
    **cache_kwargs,
) -> tuple[list[ProbeRecord], dict, dict]:
    """Download, build, cache, and return the full probe set.

    Parameters
    ----------
    nb_rows     : NaturalBench rows to sample (→ nb_rows × 2 records)
    pope_pairs  : POPE image pairs to sample (→ pope_pairs × 2 records)
    seed        : random seed for sampling
    **cache_kwargs : forwarded to build_cache() (e.g. image_size, dtype)

    Returns
    -------
    records, pixel_values, foil_pixel_values
    """
    nb_records   = build_naturalbench_records(target_rows=nb_rows,   seed=seed)
    pope_records = build_pope_records(         target_pairs=pope_pairs, seed=seed)
    all_records  = nb_records + pope_records
    build_cache(all_records, **cache_kwargs)
    return load_cache()


__all__ = [
    "ProbeRecord",
    "resolve_answer_token_ids",
    "build_naturalbench_records",
    "build_pope_records",
    "build_cache",
    "load_cache",
    "build_probe_set",
]
