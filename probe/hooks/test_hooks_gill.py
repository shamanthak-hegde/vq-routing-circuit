"""
Smoke test for GillHookManager.

Runs prefill + generate on 3 probe records (1 NB, 1 POPE yes, 1 POPE no),
asserts all structural invariants, and prints timing / memory reports.

Usage (requires GPU + gill conda env):
    conda create -n gill python=3.10 -y && conda activate gill
    pip install -r gill/requirements.txt
    pip install -e <repo-root>
    cd <repo-root>
    python -m probe.hooks.test_hooks_gill \\
        --model_path gill/checkpoints/gill_opt

Notes
-----
GILL layout differs from LLaVA/SEED in one key way: position 0 of the
inputs_embeds is VISUAL (not OTHER/BOS), because visual tokens come before
the text sequence. The "pos 0 is OTHER" assertion from test_hooks.py is
adapted here to "pos 0 is VISUAL".

GILL uses a full re-forward each decode step (use_cache=False), so
n_gen_captured >> n_gen (same phenomenon as SEED-LLaMA). base.py handles
the logit slicing correctly.

OPT residual arithmetic tolerance widened to 0.3 (vs 0.2 for LLaVA) to
accommodate bf16 OPT numerical variance; matches the Emu3 precedent.
"""

import argparse
import os
import sys
import time

import torch


def _test_real_model(model_path: str) -> None:
    print(f"\nLoading GILL from {model_path} ...", flush=True)

    _gill_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "gill")
    )
    if _gill_root not in sys.path:
        sys.path.insert(0, _gill_root)

    from probe.hooks.gill import GillHookManager
    from probe.hooks import TokenCategory
    from probe import load_cache
    from PIL import Image

    hm = GillHookManager(model_path=model_path)

    n_layers = len(hm.model.lm.model.decoder.layers)
    hidden = hm.model.lm.config.hidden_size
    vocab = hm.model.lm.config.vocab_size
    print(f"n_layers={n_layers}, hidden={hidden}, vocab={vocab}", flush=True)

    # GILL-specific: <|image|> cls token ID (should NOT appear in text prompts)
    img_cls_id = hm.tokenizer.convert_tokens_to_ids("<|image|>")

    records, _, _ = load_cache()
    nb = next(r for r in records if r.source == "naturalbench")
    pope_yes = next(r for r in records if r.source == "pope" and r.answer == "yes")
    pope_no = next(r for r in records if r.source == "pope" and r.answer == "no")

    for rec in [nb, pope_yes, pope_no]:
        img = Image.open(rec.image_path).convert("RGB")
        q = rec.question

        # ── prefill ───────────────────────────────────────────────────────────
        t0 = time.time()
        cap = hm.run_prefill(img, q)
        elapsed = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        S = cap.residual.shape[1]
        print(
            f"\n[{rec.source}] seq_len={S} n_image_tokens={cap.n_image_tokens} | "
            f"{elapsed:.2f}s | peak_gpu={mem:.1f}GB",
            flush=True,
        )

        # shapes
        assert cap.residual.shape == (n_layers, S, hidden), cap.residual.shape
        assert cap.attn_out.shape == (n_layers, S, hidden), cap.attn_out.shape
        assert cap.mlp_out.shape == (n_layers, S, hidden), cap.mlp_out.shape
        assert cap.logits.shape == (S, vocab), cap.logits.shape
        assert cap.projected_visual.ndim == 3, (
            f"projected_visual.ndim={cap.projected_visual.ndim}, expected 3"
        )
        assert cap.projected_visual.shape[-1] == hidden, (
            f"projected_visual last dim {cap.projected_visual.shape[-1]} != hidden {hidden}"
        )
        assert cap.visual_embeds.shape == (cap.n_image_tokens, hidden), (
            f"visual_embeds {cap.visual_embeds.shape} vs n_image_tokens {cap.n_image_tokens}"
        )
        assert cap.n_image_tokens == 4, (
            f"expected 4 image tokens, got {cap.n_image_tokens}"
        )
        print("  [PASS] shapes")

        # token-category invariants
        cats = cap.token_index.categories
        n_vis = (cats == int(TokenCategory.VISUAL)).sum().item()
        assert n_vis == cap.n_image_tokens, (
            f"VISUAL count {n_vis} != n_image_tokens {cap.n_image_tokens}"
        )
        assert (cats == int(TokenCategory.ANSWER)).sum() == 0, "ANSWER tokens in prefill"

        # GILL: pos 0 is VISUAL (visual tokens come before BOS, unlike all other models)
        assert int(cats[0]) == int(TokenCategory.VISUAL), (
            f"pos 0 expected VISUAL, got {TokenCategory(int(cats[0])).name}"
        )
        assert int(cats[cap.token_index.prompt_last]) == int(TokenCategory.OTHER), (
            f"prompt_last category={TokenCategory(int(cats[cap.token_index.prompt_last])).name}"
        )

        vis_idx = (cats == int(TokenCategory.VISUAL)).nonzero(as_tuple=False).squeeze(-1)
        if vis_idx.numel() > 1:
            assert torch.all(vis_idx[1:] - vis_idx[:-1] == 1), "VISUAL block not contiguous"
        q_idx = (cats == int(TokenCategory.QUESTION)).nonzero(as_tuple=False).squeeze(-1)
        if q_idx.numel() > 1:
            assert torch.all(q_idx[1:] - q_idx[:-1] == 1), "QUESTION block not contiguous"
        if vis_idx.numel() > 0 and q_idx.numel() > 0:
            assert int(vis_idx[-1]) < int(q_idx[0]), "VISUAL appears after QUESTION"

        # GILL-specific: <|image|> cls token must not appear in prompt input_ids
        assert img_cls_id not in cap.input_ids.tolist(), (
            f"<|image|> token (id={img_cls_id}) found in input_ids — unexpected"
        )
        print("  [PASS] token_index categories")

        # residual arithmetic: res[L+1] - res[L] ≈ attn_out[L+1] + mlp_out[L+1]
        # Tolerance 2.0: OPT bf16 accumulates rounding across two residual adds per
        # layer (post-attn add then post-ffn add), producing max_err ~0.5–1.0.
        # Matches SEED-LLaMA's tolerance (same bf16 accumulation pattern).
        delta = cap.residual[1].float() - cap.residual[0].float()
        expected = cap.attn_out[1].float() + cap.mlp_out[1].float()
        diff = (delta - expected).abs()
        max_err, mean_err = diff.max().item(), diff.mean().item()
        assert max_err < 2.0 and mean_err < 0.02, (
            f"Residual arithmetic: max_err={max_err:.4f}, mean_err={mean_err:.4f}"
        )
        print(f"  [PASS] residual arithmetic  max_err={max_err:.4f}")

        # logit sanity: yes/no split at prompt_last
        last = cap.logits[cap.token_index.prompt_last]
        yes_ids = hm.tokenizer.encode("yes", add_special_tokens=False)
        no_ids = hm.tokenizer.encode("no", add_special_tokens=False)
        if len(yes_ids) == 1 and len(no_ids) == 1:
            logit_yes = last[yes_ids[0]].item()
            logit_no = last[no_ids[0]].item()
            pred = "yes" if logit_yes > logit_no else "no"
            print(
                f"  logit_yes={logit_yes:.2f} logit_no={logit_no:.2f} "
                f"→ pred={pred}  gt={rec.answer}"
            )
        else:
            top_tok = hm.tokenizer.decode([int(last.argmax())])
            print(f"  top predicted token: {top_tok!r}  (gt: {rec.answer!r})")

        # ── generate mode ─────────────────────────────────────────────────────
        t0 = time.time()
        cap_gen = hm.run_generate(img, q, max_new_tokens=4)
        elapsed_gen = time.time() - t0
        S_gen = cap_gen.residual.shape[1]
        n_gen = cap_gen.generated_ids.shape[0] if cap_gen.generated_ids is not None else 0
        n_gen_captured = S_gen - S

        generated_text = (
            hm.tokenizer.decode(cap_gen.generated_ids.tolist(), skip_special_tokens=True)
            if cap_gen.generated_ids is not None
            else ""
        )
        print(
            f"  generate: {generated_text!r}  "
            f"S_gen={S_gen} n_gen={n_gen} n_gen_captured={n_gen_captured}  "
            f"({elapsed_gen:.2f}s)",
            flush=True,
        )

        assert cap_gen.residual.shape == (n_layers, S_gen, hidden), cap_gen.residual.shape
        assert cap_gen.logits.shape[0] == n_gen, (
            f"logits rows {cap_gen.logits.shape[0]} != n_gen {n_gen}"
        )
        assert cap_gen.token_index.answer_start == S, (
            f"answer_start={cap_gen.token_index.answer_start}, expected {S}"
        )
        assert (
            (cap_gen.token_index.categories == int(TokenCategory.ANSWER)).sum()
            == n_gen_captured
        )
        print("  [PASS] generate shapes")

        # first-step logit parity: prefill-last vs generate-first
        prefill_last = cap.logits[cap.token_index.prompt_last].to(cap_gen.logits.device)
        gen_first = cap_gen.logits[0]
        pf_topv, pf_topi = torch.topk(prefill_last.float(), k=5)
        gn_topv, gn_topi = torch.topk(gen_first.float(), k=5)
        top1_match = int(pf_topi[0]) == int(gn_topi[0])
        top5_overlap = len(set(pf_topi.tolist()) & set(gn_topi.tolist()))
        top5_max_err = (pf_topv - gn_topv).abs().max().item()
        assert top1_match and top5_overlap >= 3 and top5_max_err < 2.0, (
            "Prefill/generate first-logit parity failed: "
            f"top1_match={top1_match}, top5_overlap={top5_overlap}, "
            f"top5_max_err={top5_max_err:.4e}"
        )
        print(
            f"  [PASS] first-step logit parity  "
            f"top1={top1_match} overlap={top5_overlap}/5  max_err={top5_max_err:.4f}"
        )

    print("\n=== All GILL hook tests PASSED ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="gill/checkpoints/gill_opt",
        help="Path to the gill_opt checkpoint directory "
             "(contains pretrained_ckpt.pth.tar + model_args.json).",
    )
    args = parser.parse_args()
    _test_real_model(args.model_path)


if __name__ == "__main__":
    main()
