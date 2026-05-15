"""
Smoke test for LlavaVQHookManager (N-056).

Runs prefill + generate on 3 probe records (1 NB, 1 POPE yes, 1 POPE no)
using a VQLinearProjector-swapped LLaVA model and asserts the same 9
structural invariants as the base LLaVA smoke test.

Usage (requires GPU + sae env, LLaVA base model):

    source activate sae
    cd /scratch/shegde23/unified_mech
    python -m probe.hooks.test_hooks_llava_vq \\
        --model_path liuhaotian/llava-v1.6-vicuna-7b

Optional: pass a trained checkpoint to test with the actual trained projector:

    python -m probe.hooks.test_hooks_llava_vq \\
        --model_path liuhaotian/llava-v1.6-vicuna-7b \\
        --projector_ckpt checkpoints/llava_vq/projector_final.pt

Without --projector_ckpt, uses a randomly-initialised VQLinearProjector
(sufficient to verify hook invariants; outputs will be semantically wrong).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from PIL import Image


_LLAVA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
)


def _load_model_with_vq(model_path: str, projector_ckpt: str | None):
    """Load LLaVA-1.6 and replace mm_projector with VQLinearProjector."""
    if _LLAVA_ROOT not in sys.path:
        sys.path.insert(0, _LLAVA_ROOT)
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from probe.training.llava_vq_projector import VQLinearProjector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_base=None, model_name=model_name,
        attn_implementation="eager",
    )

    clip_dim = model.config.mm_hidden_size   # 1024
    lm_dim = model.config.hidden_size         # 4096

    if projector_ckpt and os.path.exists(projector_ckpt):
        state = torch.load(projector_ckpt, map_location="cpu")
        code_dim = state.get("code_dim", 32)
        codebook_size = state.get("codebook_size", 16384)
        print(f"Loading projector from {projector_ckpt} "
              f"(code_dim={code_dim}, K={codebook_size})")
        vq_proj = VQLinearProjector(
            clip_dim=clip_dim, lm_dim=lm_dim,
            code_dim=code_dim, codebook_size=codebook_size,
        )
        vq_proj.load_state_dict(state["projector"])
    else:
        print("No projector checkpoint — using random VQLinearProjector (invariants only)")
        vq_proj = VQLinearProjector(clip_dim=clip_dim, lm_dim=lm_dim)

    model.model.mm_projector = vq_proj.to(device).eval()
    model.eval()
    return model, tokenizer, image_processor


def _run_smoke(model_path: str, projector_ckpt: str | None) -> None:
    from probe.hooks.llava_vq import LlavaVQHookManager
    from probe.hooks.schema import TokenCategory
    from probe import load_cache

    model, tokenizer, image_processor = _load_model_with_vq(model_path, projector_ckpt)
    device = next(model.parameters()).device

    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    vocab = model.config.vocab_size

    hm = LlavaVQHookManager(model, tokenizer, image_processor,
                             capture_attention_weights=True)

    records, _, _ = load_cache()
    nb = next(r for r in records if r.source == "naturalbench")
    pope_yes = next(r for r in records if r.source == "pope" and r.answer == "yes")
    pope_no = next(r for r in records if r.source == "pope" and r.answer == "no")
    test_records = [nb, pope_yes, pope_no]

    n_passed = 0
    for rec in test_records:
        img = Image.open(rec.image_path).convert("RGB")

        # ── prefill ───────────────────────────────────────────────────────────
        t0 = time.time()
        cap = hm.run_prefill(img, rec.question)
        elapsed = time.time() - t0
        S = cap.residual.shape[1]
        print(f"\n[{rec.source}/{rec.answer}] seq_len={S} prefill={elapsed:.2f}s")

        # 1. Shape assertions
        assert cap.residual.shape == (n_layers, S, hidden), \
            f"residual shape: {cap.residual.shape}"
        assert cap.attn_out.shape == (n_layers, S, hidden)
        assert cap.mlp_out.shape == (n_layers, S, hidden)
        assert cap.logits.shape == (S, vocab)
        assert cap.visual_embeds.shape == (cap.n_image_tokens, hidden)
        print("  [PASS] shapes")
        n_passed += 1

        # 2. No ANSWER tokens in prefill
        cats = cap.token_index.categories
        assert (cats == int(TokenCategory.ANSWER)).sum() == 0
        print("  [PASS] no ANSWER tokens in prefill")
        n_passed += 1

        # 3. VISUAL count matches n_image_tokens
        n_vis = (cats == int(TokenCategory.VISUAL)).sum().item()
        assert n_vis == cap.n_image_tokens, \
            f"VISUAL count {n_vis} != n_image_tokens {cap.n_image_tokens}"
        print(f"  [PASS] n_image_tokens={cap.n_image_tokens}")
        n_passed += 1

        # 4. Attention weights captured (capture_attention_weights=True)
        assert cap.attn_weights is not None, "attn_weights is None"
        assert cap.attn_weights.shape == (n_layers, model.config.num_attention_heads, S, S), \
            f"attn_weights shape: {cap.attn_weights.shape}"
        print("  [PASS] attention weights shape")
        n_passed += 1

        # 5. VQ codes populated after prefill (projector fires during _prepare_embeds)
        proj = model.model.mm_projector
        codes = proj._last_codes
        assert codes.numel() == cap.n_image_tokens, \
            f"_last_codes numel {codes.numel()} != n_image_tokens {cap.n_image_tokens}"
        assert codes.max().item() < proj.codebook_size
        print(f"  [PASS] VQ codes: {codes.numel()} tokens, "
              f"max_code={codes.max().item()} < K={proj.codebook_size}")
        n_passed += 1

        # 6. Residual arithmetic (layer 1 delta)
        delta = cap.residual[1] - cap.residual[0]
        expected = cap.attn_out[1] + cap.mlp_out[1]
        diff = (delta - expected.to(delta.device)).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        assert max_err < 0.5 and mean_err < 0.02, \
            f"Residual arithmetic: max_err={max_err:.4f}, mean_err={mean_err:.4f}"
        print(f"  [PASS] residual arithmetic (max_err={max_err:.4f})")
        n_passed += 1

        # ── generate ──────────────────────────────────────────────────────────
        t0 = time.time()
        cap_gen = hm.run_generate(img, rec.question, max_new_tokens=4)
        elapsed_gen = time.time() - t0
        S_gen = cap_gen.residual.shape[1]
        n_gen_cap = S_gen - S

        # 7. Generate shape
        assert cap_gen.residual.shape[0] == n_layers
        assert cap_gen.residual.shape[2] == hidden
        print(f"  [PASS] generate shape: residual {tuple(cap_gen.residual.shape)} "
              f"(+{n_gen_cap} decoded) {elapsed_gen:.2f}s")
        n_passed += 1

        # 8. Generated ids non-empty
        assert cap_gen.generated_ids is not None and cap_gen.generated_ids.numel() > 0
        decoded = tokenizer.decode(cap_gen.generated_ids.tolist(), skip_special_tokens=True)
        print(f"  [PASS] generated: {decoded!r}")
        n_passed += 1

        # 9. Logits shape in generate
        n_scores = cap_gen.logits.shape[0]
        assert n_scores >= cap_gen.generated_ids.shape[0] - 1
        print(f"  [PASS] generate logits: {cap_gen.logits.shape}")
        n_passed += 1

    print(f"\n{'='*60}")
    print(f"Smoke test PASSED: {n_passed}/9 assertions across 3 records.")
    print(f"{'='*60}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="LlavaVQHookManager smoke test (N-056)")
    parser.add_argument("--model_path", required=True,
                        help="Base LLaVA model (e.g. liuhaotian/llava-v1.6-vicuna-7b)")
    parser.add_argument("--projector_ckpt", default=None,
                        help="Trained VQLinearProjector checkpoint "
                             "(default: random init, checks structure only)")
    args = parser.parse_args()
    _run_smoke(args.model_path, args.projector_ckpt)


if __name__ == "__main__":
    _main()
