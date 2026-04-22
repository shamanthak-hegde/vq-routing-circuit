"""
probe.tracing — windowed activation-patching engine.

Quick start
-----------
    from probe import load_cache, resolve_answer_token_ids
    from probe.hooks import LlavaHookManager
    from probe.tracing import sweep_windows

    records, _, _ = load_cache()
    resolve_answer_token_ids(records, tokenizer)

    hm = LlavaHookManager(model, tokenizer, image_processor)
    results = sweep_windows(hm, records, window_size=4)

    # pivot to heatmap
    import pandas as pd
    df = pd.DataFrame([vars(r) for r in results])
    heatmap = df.groupby(["layer_start", "token_group"])["score"].mean().unstack()
"""

from .sweep  import sweep_windows, sweep_record, PatchResult
from .score  import normalized_restoration
from .corrupt import foil_embeds, noisy_embeds

__all__ = [
    "sweep_windows",
    "sweep_record",
    "PatchResult",
    "normalized_restoration",
    "foil_embeds",
    "noisy_embeds",
]
