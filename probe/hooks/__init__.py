"""
probe.hooks — activation-capture hook layer for VLMs.

Supported models
----------------
LLaVA-1.6 (llava-v1.6-vicuna-7b)  →  LlavaHookManager

Usage
-----
    from probe.hooks import LlavaHookManager, TokenCategory

    # load model once (outside the hot loop)
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_base=None, model_name=model_name,
        attn_implementation="sdpa",   # or "eager" if capture_attention_weights=True
    )
    model.eval()

    hm = LlavaHookManager(model, tokenizer, image_processor)

    # prefill (primary path — single-token yes/no logit restoration)
    cap = hm.run_prefill(image=PIL_img, question="Is there a cat?")

    # generate (multi-token tracing)
    cap = hm.run_generate(image=PIL_img, question="Describe this image.",
                          max_new_tokens=32)

    # downstream analysis
    last_res  = cap.last_token_residual()          # (n_layers, hidden)
    vis_res   = cap.by_category(TokenCategory.VISUAL)  # (n_layers, n_img, hidden)
    logit_map = cap.logit_lens(model.lm_head, model.model.norm)  # (n_layers, seq, vocab)

Opt-in attention weights (eager backend required)
-------------------------------------------------
    tokenizer, model, image_processor, _ = load_pretrained_model(
        ..., attn_implementation="eager"
    )
    hm = LlavaHookManager(model, tokenizer, image_processor,
                          capture_attention_weights=True)
    cap = hm.run_prefill(...)
    # cap.attn_weights: (n_layers, heads, seq_len, seq_len)
"""

from .schema import Capture, TokenCategory, TokenIndex
from .llava import LlavaHookManager

__all__ = [
    "Capture",
    "TokenCategory",
    "TokenIndex",
    "LlavaHookManager",
]
