"""
Smoke test for probe.hooks.

Runs prefill + generate on 3 probe records (1 NB, 1 POPE yes, 1 POPE no),
asserts all structural invariants, and prints a timing / memory report.

Usage:
    source activate sae
    cd <repo-root>
    python -m probe.hooks.test_hooks [--model_path <path>]

If --model_path is omitted, the script only validates the TokenIndex /
Capture dataclass logic with a dummy model to keep CI fast.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# ── dummy model test (no GPU needed) ─────────────────────────────────────────

def _test_token_index_invariants():
    """Pure-python invariant checks that don't require a real model."""
    from probe.hooks.schema import TokenCategory, TokenIndex

    cats = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0], dtype=torch.int8
    )
    ti = TokenIndex(
        categories=cats,
        visual_range=(3, 7),
        question_range=(7, 10),
        answer_start=None,
    )
    assert ti.mask(TokenCategory.VISUAL).sum() == 4
    assert ti.mask(TokenCategory.QUESTION).sum() == 3
    assert ti.mask(TokenCategory.OTHER).sum() == 5
    assert ti.mask(TokenCategory.ANSWER).sum() == 0
    assert ti.prompt_last == 11
    # visual is contiguous
    vis_idx = ti.mask(TokenCategory.VISUAL).nonzero(as_tuple=False).squeeze(-1)
    assert torch.all(vis_idx[1:] - vis_idx[:-1] == 1), "VISUAL not contiguous"
    # question is contiguous
    q_idx = ti.mask(TokenCategory.QUESTION).nonzero(as_tuple=False).squeeze(-1)
    assert torch.all(q_idx[1:] - q_idx[:-1] == 1), "QUESTION not contiguous"
    print("[PASS] TokenIndex invariants")


def _test_capture_by_category():
    from probe.hooks.schema import Capture, TokenCategory, TokenIndex

    n_layers, seq_len, hidden = 4, 10, 8
    cats = torch.tensor([0]*2 + [1]*3 + [2]*4 + [0]*1, dtype=torch.int8)
    ti = TokenIndex(
        categories=cats, visual_range=(2,5), question_range=(5,9), answer_start=None
    )
    cap = Capture(
        input_ids=torch.zeros(seq_len, dtype=torch.long),
        token_index=ti,
        projected_visual=torch.randn(2, 5, hidden),
        visual_embeds=torch.randn(3, hidden),
        attn_out=torch.randn(n_layers, seq_len, hidden),
        mlp_out=torch.randn(n_layers, seq_len, hidden),
        residual=torch.randn(n_layers, seq_len, hidden),
        logits=torch.randn(seq_len, 32000),
        n_image_tokens=3,
    )
    vis = cap.by_category(TokenCategory.VISUAL)
    assert vis.shape == (n_layers, 3, hidden), f"unexpected shape {vis.shape}"
    q = cap.by_category(TokenCategory.QUESTION)
    assert q.shape == (n_layers, 4, hidden)
    lr = cap.last_token_residual()
    assert lr.shape == (n_layers, hidden)
    print("[PASS] Capture.by_category / last_token_residual")


# ── real-model tests ──────────────────────────────────────────────────────────

def _test_real_model(model_path: str):
    print(f"\nLoading model from {model_path} ...")
    import sys, os
    _llava = os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
    if _llava not in sys.path:
        sys.path.insert(0, _llava)

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from probe.hooks import LlavaHookManager, TokenCategory
    from probe import load_cache

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_base=None, model_name=model_name,
        attn_implementation="sdpa",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden   = model.config.hidden_size
    vocab    = model.config.vocab_size

    hm = LlavaHookManager(model, tokenizer, image_processor)
    records, pv_cache, _ = load_cache()

    # pick 3 test records: 1 NB, 1 POPE yes, 1 POPE no
    nb   = next(r for r in records if r.source == "naturalbench")
    pope_yes = next(r for r in records if r.source == "pope" and r.answer == "yes")
    pope_no  = next(r for r in records if r.source == "pope" and r.answer == "no")
    test_records = [nb, pope_yes, pope_no]

    from PIL import Image
    for rec in test_records:
        img = Image.open(rec.image_path).convert("RGB")
        q   = rec.question

        t0 = time.time()
        cap = hm.run_prefill(img, q)
        elapsed = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        S = cap.residual.shape[1]
        print(f"\n  [{rec.source}] q={q!r:.50s}... | prefill seq_len={S} | {elapsed:.2f}s | peak_gpu={mem:.1f}GB")

        # ── shape assertions ──────────────────────────────────────────────────
        assert cap.residual.shape == (n_layers, S, hidden), \
            f"residual shape mismatch: {cap.residual.shape}"
        assert cap.attn_out.shape  == (n_layers, S, hidden)
        assert cap.mlp_out.shape   == (n_layers, S, hidden)
        assert cap.logits.shape    == (S, vocab)
        assert cap.projected_visual.ndim == 3
        assert cap.projected_visual.shape[-1] == hidden
        assert cap.visual_embeds.shape == (cap.n_image_tokens, hidden)
        print("    [PASS] shapes")

        # ── token-category invariants ─────────────────────────────────────────
        cats = cap.token_index.categories
        # no ANSWER in prefill
        assert (cats == int(TokenCategory.ANSWER)).sum() == 0, \
            "ANSWER tokens found in prefill"
        # VISUAL count matches n_image_tokens
        n_vis = (cats == int(TokenCategory.VISUAL)).sum().item()
        assert n_vis == cap.n_image_tokens, \
            f"VISUAL count {n_vis} != n_image_tokens {cap.n_image_tokens}"
        # first token is OTHER (BOS)
        assert int(cats[0]) == int(TokenCategory.OTHER), "First token not OTHER"
        # last prompt token is OTHER (" ASSISTANT:")
        assert int(cats[cap.token_index.prompt_last]) == int(TokenCategory.OTHER), \
            f"Last prompt token is {TokenCategory(int(cats[cap.token_index.prompt_last])).name}, expected OTHER"
        # VISUAL is contiguous
        vis_idx = (cats == int(TokenCategory.VISUAL)).nonzero(as_tuple=False).squeeze(-1)
        assert torch.all(vis_idx[1:] - vis_idx[:-1] == 1), "VISUAL not contiguous"
        # QUESTION is contiguous if present
        q_idx = (cats == int(TokenCategory.QUESTION)).nonzero(as_tuple=False).squeeze(-1)
        if q_idx.numel() > 1:
            assert torch.all(q_idx[1:] - q_idx[:-1] == 1), "QUESTION not contiguous"
        # VISUAL appears before QUESTION
        if vis_idx.numel() > 0 and q_idx.numel() > 0:
            assert int(vis_idx[-1]) < int(q_idx[0]), \
                "VISUAL tokens appear after QUESTION tokens"
        print("    [PASS] token-category invariants (prefill)")

        # ── residual stream arithmetic check (layer 0) ────────────────────────
        # residual[0] ≈ input_embeds + attn_out[0] + mlp_out[0]
        # (Llama: h = h + attn; h = h + mlp; so residual[0] = embed + attn[0] + mlp[0])
        # We test this loosely since input_embeds aren't directly in Capture.
        # Instead: residual[1] - residual[0] ≈ attn_out[1] + mlp_out[1]
        # Captures are stored in model dtype (typically fp16), so a strict
        # max-only bound is noisy.  Guard on both the worst-case and mean
        # absolute error to distinguish rounding from a real hook mismatch.
        delta = cap.residual[1] - cap.residual[0]
        expected = cap.attn_out[1] + cap.mlp_out[1]
        diff = (delta - expected.to(delta.device)).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        assert max_err < 0.2 and mean_err < 0.01, (
            "Residual arithmetic check failed: "
            f"max err = {max_err:.4f}, mean err = {mean_err:.4f}"
        )
        print(
            "    [PASS] residual stream arithmetic "
            f"(layer1 delta max_err={max_err:.4f}, mean_err={mean_err:.4f})"
        )

        # ── logit sanity ──────────────────────────────────────────────────────
        last_logits = cap.logits[cap.token_index.prompt_last]  # (vocab,)
        top_token   = tokenizer.decode([int(last_logits.argmax())])
        print(f"    top predicted token: {top_token!r}  (gt answer: {rec.answer!r})")

        # ── generate mode ─────────────────────────────────────────────────────
        t0 = time.time()
        cap_gen = hm.run_generate(img, q, max_new_tokens=4)
        elapsed_gen = time.time() - t0
        S_gen = cap_gen.residual.shape[1]
        n_gen = cap_gen.generated_ids.shape[0] if cap_gen.generated_ids is not None else 0
        n_gen_captured = S_gen - S
        print(
            f"    generate seq_len={S_gen} "
            f"(prompt={S}, gen={n_gen}, captured_gen={n_gen_captured}) | "
            f"{elapsed_gen:.2f}s"
        )

        assert cap_gen.residual.shape == (n_layers, S_gen, hidden)
        # generate() runs with logits_to_keep=1 under current transformers, so
        # lm_head only materializes one logit row per generation step: the
        # prompt-last prediction, then one per subsequent decode step.
        assert cap_gen.logits.shape[0] == n_gen
        assert cap_gen.token_index.answer_start == S
        assert (cap_gen.token_index.categories == int(TokenCategory.ANSWER)).sum() == n_gen_captured
        prefill_last_logits = cap.logits[cap.token_index.prompt_last].to(cap_gen.logits.device)
        gen_first_logits = cap_gen.logits[0]
        prefill_topv, prefill_topi = torch.topk(prefill_last_logits.float(), k=5)
        gen_topv, gen_topi = torch.topk(gen_first_logits.float(), k=5)
        top1_match = int(prefill_topi[0]) == int(gen_topi[0])
        top5_overlap = len(set(prefill_topi.tolist()) & set(gen_topi.tolist()))
        top5_max_err = (prefill_topv - gen_topv).abs().max().item()
        assert top1_match and top5_overlap >= 4 and top5_max_err < 0.1, (
            "Prefill/generate first-logit parity failed: "
            f"top1_match={top1_match}, top5_overlap={top5_overlap}, "
            f"top5_max_err={top5_max_err:.4e}"
        )
        print("    [PASS] generate shapes + ANSWER category")

        # ── prefill vs generate parity (prompt positions) ─────────────────────
        err_parity = (
            cap.residual[:, :S, :].to(cap_gen.residual.device)
            - cap_gen.residual[:, :S, :]
        ).abs()
        parity_max_err = err_parity.max().item()
        parity_mean_err = err_parity.mean().item()
        assert parity_max_err < 2.5 and parity_mean_err < 0.02, (
            "Prefill/generate residual parity failed: "
            f"max_err={parity_max_err:.4f}, mean_err={parity_mean_err:.4f}"
        )
        print(
            "    [PASS] prefill/generate parity "
            f"(max_err={parity_max_err:.4f}, mean_err={parity_mean_err:.4f})"
        )

    print("\n[ALL TESTS PASSED]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None,
                        help="Path to llava-v1.6 checkpoint. If omitted, only dummy tests run.")
    args = parser.parse_args()

    _test_token_index_invariants()
    _test_capture_by_category()

    if args.model_path:
        _test_real_model(args.model_path)
    else:
        print("\n(Skipping real-model tests — pass --model_path to run them)")


if __name__ == "__main__":
    main()
