"""
Smoke test for VilaUHookManager.

Runs prefill + generate on 3 probe records (1 NB, 1 POPE yes, 1 POPE no),
asserts all structural invariants, and prints a timing / memory report.

Usage:
    source activate sae
    cd <repo-root>
    python -m probe.hooks.test_hooks_vilau --model_path mit-han-lab/vila-u-7b-256

Mirrors test_hooks.py but targets VILA-U:
  - Projector: model.mm_projector (256 or 729 image tokens)
  - Decoder:   model.llm.model.layers
  - Generate:  model.llm.generate(inputs_embeds=...)
"""

import argparse
import time

import torch


def _test_real_model(model_path: str) -> None:
    print(f"\nLoading VILA-U model from {model_path} ...", flush=True)

    import os, sys
    _vilau = os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
    if _vilau not in sys.path:
        sys.path.insert(0, _vilau)

    from vila_u.model.builder import load_pretrained_model
    from probe.hooks import VilaUHookManager, TokenCategory
    from probe import load_cache
    from PIL import Image

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        attn_implementation="eager",  # FA2 patch handles any hardcoded FA2
    )
    model.eval()

    n_layers = model.llm.config.num_hidden_layers
    hidden   = model.llm.config.hidden_size
    vocab    = model.llm.config.vocab_size

    hm = VilaUHookManager(model, tokenizer, image_processor)
    expected_n_img = model.vision_tower.image_tokens
    print(
        f"  n_layers={n_layers}, hidden={hidden}, n_image_tokens={expected_n_img}",
        flush=True,
    )

    records, pv_cache, _ = load_cache()
    nb       = next(r for r in records if r.source == "naturalbench")
    pope_yes = next(r for r in records if r.source == "pope" and r.answer == "yes")
    pope_no  = next(r for r in records if r.source == "pope" and r.answer == "no")

    for rec in [nb, pope_yes, pope_no]:
        img = Image.open(rec.image_path).convert("RGB")
        q   = rec.question

        # ── prefill ───────────────────────────────────────────────────────────
        t0 = time.time()
        cap = hm.run_prefill(img, q)
        elapsed = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        S = cap.residual.shape[1]
        print(
            f"\n  [{rec.source}] q={q!r:.50s}... | prefill seq_len={S} | "
            f"{elapsed:.2f}s | peak_gpu={mem:.1f}GB",
            flush=True,
        )

        # shapes
        assert cap.residual.shape       == (n_layers, S, hidden), cap.residual.shape
        assert cap.attn_out.shape       == (n_layers, S, hidden)
        assert cap.mlp_out.shape        == (n_layers, S, hidden)
        assert cap.logits.shape         == (S, vocab)
        assert cap.projected_visual.ndim == 3
        assert cap.projected_visual.shape[-1] == hidden
        assert cap.n_image_tokens == expected_n_img, (
            f"n_image_tokens={cap.n_image_tokens}, expected={expected_n_img}"
        )
        assert cap.visual_embeds.shape == (cap.n_image_tokens, hidden)
        print("    [PASS] shapes")

        # token-category invariants
        cats = cap.token_index.categories
        assert (cats == int(TokenCategory.ANSWER)).sum() == 0, "ANSWER in prefill"
        n_vis = (cats == int(TokenCategory.VISUAL)).sum().item()
        assert n_vis == cap.n_image_tokens, \
            f"VISUAL count {n_vis} != n_image_tokens {cap.n_image_tokens}"
        assert int(cats[0]) == int(TokenCategory.OTHER), "First token not OTHER"
        assert int(cats[cap.token_index.prompt_last]) == int(TokenCategory.OTHER), (
            f"Last prompt token is "
            f"{TokenCategory(int(cats[cap.token_index.prompt_last])).name}, expected OTHER"
        )
        vis_idx = (cats == int(TokenCategory.VISUAL)).nonzero(as_tuple=False).squeeze(-1)
        assert torch.all(vis_idx[1:] - vis_idx[:-1] == 1), "VISUAL not contiguous"
        q_idx = (cats == int(TokenCategory.QUESTION)).nonzero(as_tuple=False).squeeze(-1)
        if q_idx.numel() > 1:
            assert torch.all(q_idx[1:] - q_idx[:-1] == 1), "QUESTION not contiguous"
        if vis_idx.numel() > 0 and q_idx.numel() > 0:
            assert int(vis_idx[-1]) < int(q_idx[0]), "VISUAL after QUESTION"
        print("    [PASS] token-category invariants (prefill)")

        # residual stream arithmetic (layer 1): delta ≈ attn + mlp
        delta    = cap.residual[1] - cap.residual[0]
        expected = cap.attn_out[1] + cap.mlp_out[1]
        diff     = (delta - expected.to(delta.device)).abs()
        max_err, mean_err = diff.max().item(), diff.mean().item()
        # bf16 has 7-bit mantissa; single-position rounding outliers up to ~1.0
        # are expected. mean_err << 0.01 confirms hooks are correct.
        assert max_err < 1.5 and mean_err < 0.01, (
            f"Residual arithmetic failed: max_err={max_err:.4f}, mean_err={mean_err:.4f}"
        )
        print(
            f"    [PASS] residual arithmetic (max_err={max_err:.4f}, mean_err={mean_err:.4f})"
        )

        # logit sanity
        last_logits = cap.logits[cap.token_index.prompt_last]
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
            f"{elapsed_gen:.2f}s",
            flush=True,
        )

        assert cap_gen.residual.shape == (n_layers, S_gen, hidden)
        assert cap_gen.logits.shape[0] == n_gen, \
            f"logits.shape[0]={cap_gen.logits.shape[0]}, expected n_gen={n_gen}"
        assert cap_gen.token_index.answer_start == S
        assert (
            (cap_gen.token_index.categories == int(TokenCategory.ANSWER)).sum()
            == n_gen_captured
        )
        print("    [PASS] generate shapes + ANSWER category")

        # first-step logit parity (prefill last-token vs generate first step)
        prefill_last = cap.logits[cap.token_index.prompt_last].to(cap_gen.logits.device)
        gen_first    = cap_gen.logits[0]
        pf_topv, pf_topi = torch.topk(prefill_last.float(), k=5)
        gn_topv, gn_topi = torch.topk(gen_first.float(), k=5)
        top1_match   = int(pf_topi[0]) == int(gn_topi[0])
        top5_overlap = len(set(pf_topi.tolist()) & set(gn_topi.tolist()))
        top5_max_err = (pf_topv - gn_topv).abs().max().item()
        assert top1_match and top5_overlap >= 4 and top5_max_err < 1.0, (
            "Prefill/generate first-logit parity failed: "
            f"top1_match={top1_match}, top5_overlap={top5_overlap}, "
            f"top5_max_err={top5_max_err:.4e}"
        )
        print("    [PASS] first-step logit parity")

        # prompt-position activation parity (prefill vs generate)
        err = (
            cap.residual[:, :S, :].to(cap_gen.residual.device)
            - cap_gen.residual[:, :S, :]
        ).abs()
        parity_max, parity_mean = err.max().item(), err.mean().item()
        assert parity_max < 4.0 and parity_mean < 0.02, (
            f"Prefill/generate parity failed: max={parity_max:.4f}, mean={parity_mean:.4f}"
        )
        print(
            f"    [PASS] prefill/generate parity "
            f"(max_err={parity_max:.4f}, mean_err={parity_mean:.4f})"
        )

    print("\n[ALL TESTS PASSED]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="mit-han-lab/vila-u-7b-256",
        help="Path or HF hub ID of the VILA-U checkpoint.",
    )
    args = parser.parse_args()
    _test_real_model(args.model_path)


if __name__ == "__main__":
    main()
