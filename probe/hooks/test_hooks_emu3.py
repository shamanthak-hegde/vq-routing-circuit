"""
Smoke test for Emu3HookManager.

Runs prefill + generate on 3 probe records (1 NB, 1 POPE yes, 1 POPE no),
asserts all structural invariants, and prints timing / memory reports.

Usage (requires GPU):
    source activate emu
    module load cuda-12.4.1-gcc-12.1.0
    cd <repo-root>
    python -m probe.hooks.test_hooks_emu3 \\
        --model_path BAAI/Emu3-Chat \\
        --vq_path BAAI/Emu3-VisionTokenizer
"""

import argparse
import time

import torch


def _test_real_model(model_path: str, vq_path: str) -> None:
    print(f"\nLoading Emu3 model from {model_path} ...", flush=True)

    import os, sys
    _emu3 = os.path.join(os.path.dirname(__file__), "..", "..", "Emu3")
    if _emu3 not in sys.path:
        sys.path.insert(0, _emu3)

    from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
    from probe.hooks.emu3 import Emu3HookManager
    from probe.hooks import TokenCategory
    from probe import load_cache
    from PIL import Image

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    image_processor = AutoImageProcessor.from_pretrained(vq_path, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(
        vq_path, device_map="cuda:0", trust_remote_code=True
    ).eval()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).eval()

    n_layers = model.config.num_hidden_layers
    hidden   = model.config.hidden_size
    vocab    = model.config.vocab_size
    print(
        f"n_layers={n_layers}, hidden={hidden}, vocab={vocab}, "
        f"device=cuda:0",
        flush=True,
    )

    hm = Emu3HookManager(model, tokenizer, image_processor, image_tokenizer,
                         max_image_size=256)

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
        assert cap.attn_out.shape    == (n_layers, S, hidden)
        assert cap.mlp_out.shape     == (n_layers, S, hidden)
        assert cap.logits.shape      == (S, vocab)
        assert cap.projected_visual.ndim == 3
        assert cap.projected_visual.shape[-1] == hidden
        assert cap.visual_embeds.shape == (cap.n_image_tokens, hidden), (
            f"visual_embeds {cap.visual_embeds.shape} vs n_image_tokens {cap.n_image_tokens}"
        )
        print("  [PASS] shapes")

        # token-category invariants
        cats  = cap.token_index.categories
        n_vis = (cats == int(TokenCategory.VISUAL)).sum().item()
        assert n_vis == cap.n_image_tokens, \
            f"VISUAL count {n_vis} != n_image_tokens {cap.n_image_tokens}"
        assert (cats == int(TokenCategory.ANSWER)).sum() == 0, "ANSWER in prefill"
        assert int(cats[0]) == int(TokenCategory.OTHER), "First token not OTHER"
        assert int(cats[cap.token_index.prompt_last]) == int(TokenCategory.OTHER), (
            f"Last prompt token category={TokenCategory(int(cats[cap.token_index.prompt_last])).name}"
        )
        vis_idx = (cats == int(TokenCategory.VISUAL)).nonzero(as_tuple=False).squeeze(-1)
        assert torch.all(vis_idx[1:] - vis_idx[:-1] == 1), "VISUAL block not contiguous"
        q_idx = (cats == int(TokenCategory.QUESTION)).nonzero(as_tuple=False).squeeze(-1)
        if q_idx.numel() > 1:
            assert torch.all(q_idx[1:] - q_idx[:-1] == 1), "QUESTION block not contiguous"
        if vis_idx.numel() > 0 and q_idx.numel() > 0:
            assert int(vis_idx[-1]) < int(q_idx[0]), "VISUAL appears after QUESTION"
        print("  [PASS] token_index categories")

        # Compare in fp32 before the arithmetic; even then, Emu3 can produce
        # isolated ~1-2 unit outliers while the mean error stays near zero.
        delta    = cap.residual[1].float() - cap.residual[0].float()
        expected = cap.attn_out[1].float() + cap.mlp_out[1].float()
        diff     = (delta - expected).abs()
        max_err, mean_err = diff.max().item(), diff.mean().item()
        assert max_err < 2.0 and mean_err < 0.01, (
            f"Residual arithmetic: max_err={max_err:.4f}, mean_err={mean_err:.4f}"
        )
        print(f"  [PASS] residual[0].norm()={cap.residual[0].norm():.4f} > 0")

        # logit sanity: print yes/no logits
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
        print(
            f"  [PASS] generate shapes  "
            f"(generated: {tokenizer.decode(cap_gen.generated_ids.tolist(), skip_special_tokens=True)!r})"
        )

        assert cap_gen.residual.shape == (n_layers, S_gen, hidden)
        assert cap_gen.logits.shape[0] == n_gen, \
            f"logits.shape[0]={cap_gen.logits.shape[0]}, expected n_gen={n_gen}"
        assert cap_gen.token_index.answer_start == S
        assert (
            (cap_gen.token_index.categories == int(TokenCategory.ANSWER)).sum()
            == n_gen_captured
        )

        # first-step logit parity
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
        print(f"  [PASS] first-step logit parity ({elapsed_gen:.2f}s)")

    print("\n=== All Emu3 hook tests PASSED ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="BAAI/Emu3-Chat",
        help="Path or HF hub ID of the Emu3-Chat checkpoint.",
    )
    parser.add_argument(
        "--vq_path",
        default="BAAI/Emu3-VisionTokenizer",
        help="Path or HF hub ID of the Emu3-VisionTokenizer.",
    )
    args = parser.parse_args()
    _test_real_model(args.model_path, args.vq_path)


if __name__ == "__main__":
    main()
