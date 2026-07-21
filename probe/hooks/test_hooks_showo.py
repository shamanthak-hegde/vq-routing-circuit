"""
Smoke test for ShowoHookManager.

Runs prefill + generate on 3 probe records (1 NB, 1 POPE yes, 1 POPE no),
asserts all structural invariants, and prints timing / memory reports.

Usage (requires GPU):
    source activate showo
    cd <repo-root>
    python -m probe.hooks.test_hooks_showo \\
        --model_path showlab/show-o \\
        --vq_model_path showlab/magvitv2 \\
        --resolution 256
"""

import argparse
import os
import sys
import time

import torch


def _test_real_model(
    model_path: str, vq_model_path: str, resolution: int, tokenizer_path: str
) -> None:
    print(f"\nLoading Show-o model from {model_path} ...", flush=True)

    _showo = os.path.join(os.path.dirname(__file__), "..", "..", "Show-o")
    if _showo not in sys.path:
        sys.path.insert(0, _showo)

    from models import Showo, MAGVITv2
    from training.prompting_utils import UniversalPrompting
    from transformers import AutoTokenizer

    from probe.hooks.showo import ShowoHookManager
    from probe.hooks import TokenCategory
    from probe import load_cache
    from PIL import Image

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Tokenizer is Phi-1.5's — loaded separately from the LLM backbone path.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
            "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=0.0,
    )

    vq_model = MAGVITv2.from_pretrained(vq_model_path).to(device).eval()
    vq_model.requires_grad_(False)

    model = Showo.from_pretrained(model_path).to(device).eval()

    n_layers = len(model.showo.model.layers)
    hidden   = model.showo.config.hidden_size
    vocab    = model.showo.config.vocab_size
    print(f"n_layers={n_layers}, hidden={hidden}, vocab={vocab}", flush=True)

    hm = ShowoHookManager(
        model, tokenizer, vq_model, uni_prompting, resolution=resolution
    )

    records, _, _ = load_cache()
    nb       = next(r for r in records if r.source == "naturalbench")
    pope_yes = next(r for r in records if r.source == "pope" and r.answer == "yes")
    pope_no  = next(r for r in records if r.source == "pope"  and r.answer == "no")

    for rec in [nb, pope_yes, pope_no]:
        img = Image.open(rec.image_path).convert("RGB")
        q   = rec.question

        # ── prefill ───────────────────────────────────────────────────────────
        t0  = time.time()
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
        assert cap.residual.shape    == (n_layers, S, hidden), cap.residual.shape
        assert cap.attn_out.shape    == (n_layers, S, hidden), cap.attn_out.shape
        assert cap.mlp_out.shape     == (n_layers, S, hidden), cap.mlp_out.shape
        assert cap.logits.shape      == (S, vocab),            cap.logits.shape
        assert cap.projected_visual.ndim == 3
        assert cap.projected_visual.shape[-1] == hidden
        assert cap.visual_embeds.shape == (cap.n_image_tokens, hidden), (
            f"visual_embeds {cap.visual_embeds.shape} vs n_image_tokens {cap.n_image_tokens}"
        )
        assert cap.n_image_tokens == (resolution // 16) ** 2, (
            f"expected {(resolution // 16) ** 2} tokens, got {cap.n_image_tokens}"
        )
        print("  [PASS] shapes")

        # token-category invariants
        cats  = cap.token_index.categories
        n_vis = (cats == int(TokenCategory.VISUAL)).sum().item()
        assert n_vis == cap.n_image_tokens, \
            f"VISUAL count {n_vis} != n_image_tokens {cap.n_image_tokens}"
        assert (cats == int(TokenCategory.ANSWER)).sum() == 0, "ANSWER tokens in prefill"
        assert int(cats[0]) == int(TokenCategory.OTHER), "pos 0 not OTHER"
        assert int(cats[cap.token_index.prompt_last]) == int(TokenCategory.OTHER), (
            f"prompt_last category={TokenCategory(int(cats[cap.token_index.prompt_last])).name}"
        )
        vis_idx = (cats == int(TokenCategory.VISUAL)).nonzero(as_tuple=False).squeeze(-1)
        assert torch.all(vis_idx[1:] - vis_idx[:-1] == 1), "VISUAL block not contiguous"
        q_idx = (cats == int(TokenCategory.QUESTION)).nonzero(as_tuple=False).squeeze(-1)
        if q_idx.numel() > 1:
            assert torch.all(q_idx[1:] - q_idx[:-1] == 1), "QUESTION block not contiguous"
        if vis_idx.numel() > 0 and q_idx.numel() > 0:
            assert int(vis_idx[-1]) < int(q_idx[0]), "VISUAL appears after QUESTION"
        print("  [PASS] token_index categories")

        # residual arithmetic: res[L+1] - res[L] ≈ attn_out[L+1] + mlp_out[L+1]
        delta    = cap.residual[1].float() - cap.residual[0].float()
        expected = cap.attn_out[1].float() + cap.mlp_out[1].float()
        diff     = (delta - expected).abs()
        max_err, mean_err = diff.max().item(), diff.mean().item()
        assert max_err < 2.0 and mean_err < 0.02, (
            f"Residual arithmetic: max_err={max_err:.4f}, mean_err={mean_err:.4f}"
        )
        print(f"  [PASS] residual arithmetic  max_err={max_err:.4f}")

        # logit sanity: yes/no split
        last = cap.logits[cap.token_index.prompt_last]
        yes_ids = tokenizer.encode("yes", add_special_tokens=False)
        no_ids  = tokenizer.encode("no",  add_special_tokens=False)
        if len(yes_ids) == 1 and len(no_ids) == 1:
            logit_yes = last[yes_ids[0]].item()
            logit_no  = last[no_ids[0]].item()
            pred = "yes" if logit_yes > logit_no else "no"
            print(
                f"  logit_yes={logit_yes:.2f} logit_no={logit_no:.2f} "
                f"→ pred={pred}  gt={rec.answer}"
            )
        else:
            top_tok = tokenizer.decode([int(last.argmax())])
            print(f"  top predicted token: {top_tok!r}  (gt: {rec.answer!r})")

        # ── generate mode ─────────────────────────────────────────────────────
        t0 = time.time()
        cap_gen = hm.run_generate(img, q, max_new_tokens=4)
        elapsed_gen = time.time() - t0
        S_gen = cap_gen.residual.shape[1]
        n_gen = cap_gen.generated_ids.shape[0] if cap_gen.generated_ids is not None else 0
        n_gen_captured = S_gen - S

        generated_text = tokenizer.decode(
            cap_gen.generated_ids.tolist(), skip_special_tokens=True
        ) if cap_gen.generated_ids is not None else ""
        print(
            f"  [PASS] generate shapes  "
            f"(generated: {generated_text!r}  "
            f"S_gen={S_gen} n_gen={n_gen} n_gen_captured={n_gen_captured})"
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

        # first-step logit parity: prefill-last vs generate-first
        prefill_last = cap.logits[cap.token_index.prompt_last].to(cap_gen.logits.device)
        gen_first    = cap_gen.logits[0]
        pf_topv, pf_topi = torch.topk(prefill_last.float(), k=5)
        gn_topv, gn_topi = torch.topk(gen_first.float(), k=5)
        top1_match   = int(pf_topi[0]) == int(gn_topi[0])
        top5_overlap = len(set(pf_topi.tolist()) & set(gn_topi.tolist()))
        top5_max_err = (pf_topv - gn_topv).abs().max().item()
        assert top1_match and top5_overlap >= 3 and top5_max_err < 2.0, (
            "Prefill/generate first-logit parity failed: "
            f"top1_match={top1_match}, top5_overlap={top5_overlap}, "
            f"top5_max_err={top5_max_err:.4e}"
        )
        print(
            f"  [PASS] first-step logit parity  "
            f"top1={top1_match} overlap={top5_overlap}/5  ({elapsed_gen:.2f}s)"
        )

    print("\n=== All Show-o hook tests PASSED ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="showlab/show-o",
        help="Path or HF hub ID of the Show-o checkpoint.",
    )
    parser.add_argument(
        "--vq_model_path",
        default="showlab/magvitv2",
        help="Path or HF hub ID of the MAGVITv2 VQ tokenizer.",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="microsoft/phi-1_5",
        help="Path or HF hub ID for the Phi-1.5 tokenizer (LLM backbone).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Image resolution (256→256 VQ tokens, 512→1024 VQ tokens).",
    )
    args = parser.parse_args()
    _test_real_model(args.model_path, args.vq_model_path, args.resolution, args.tokenizer_path)


if __name__ == "__main__":
    main()
