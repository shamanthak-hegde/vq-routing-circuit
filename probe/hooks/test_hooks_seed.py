"""
Smoke test for SeedHookManager.

Runs prefill + generate on 3 probe records (1 NB, 1 POPE yes, 1 POPE no),
asserts all structural invariants, and prints timing / memory reports.

Usage (requires GPU + seed_llama conda env):
    conda create -n seed_llama python=3.10 -y && conda activate seed_llama
    pip install -r SEED/requirements.txt
    cd <repo-root>
    python -m probe.hooks.test_hooks_seed \\
        --model_path AILab-CVC/seed-llama-8b-sft \\
        --tokenizer_path AILab-CVC/seed-tokenizer-2
"""

import argparse
import os
import sys
import time

import torch


def _test_real_model(model_path: str, tokenizer_path: str) -> None:
    print(f"\nLoading SEED-LLaMA model from {model_path} ...", flush=True)

    _seed = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "SEED")
    )
    if _seed not in sys.path:
        sys.path.insert(0, _seed)

    from models.model_tools import get_pretrained_llama_causal_model
    from models.seed_llama_tokenizer import SeedLlamaTokenizer

    from probe.hooks.seed import SeedHookManager
    from probe.hooks import TokenCategory
    from probe import load_cache
    from PIL import Image

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Resolve quantizer weights — local dir or HF hub ID
    print(f"Loading SeedLlamaTokenizer from {tokenizer_path} ...", flush=True)
    if os.path.isdir(tokenizer_path):
        encoder_path = os.path.join(tokenizer_path, "seed_quantizer.pt")
    else:
        from huggingface_hub import hf_hub_download
        encoder_path = hf_hub_download(repo_id=tokenizer_path, filename="seed_quantizer.pt")
    tokenizer = SeedLlamaTokenizer.from_pretrained(
        tokenizer_path,
        fp16=True,
        load_diffusion=False,
        encoder_url=encoder_path,
        device=str(device),
    )

    print(f"Loading LlamaForCausalLM (xformers) from {model_path} ...", flush=True)
    model = get_pretrained_llama_causal_model(
        pretrained_model_name_or_path=model_path,
        torch_dtype="fp16",
        low_cpu_mem_usage=True,
    )
    model = model.eval().to(device)

    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size
    vocab = model.config.vocab_size
    print(f"n_layers={n_layers}, hidden={hidden}, vocab={vocab}", flush=True)

    hm = SeedHookManager(model, tokenizer)

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
        assert cap.projected_visual.ndim == 3
        assert cap.projected_visual.shape[-1] == hidden
        assert cap.visual_embeds.shape == (cap.n_image_tokens, hidden), (
            f"visual_embeds {cap.visual_embeds.shape} vs n_image_tokens {cap.n_image_tokens}"
        )
        assert cap.n_image_tokens == 32, (
            f"expected 32 image tokens, got {cap.n_image_tokens}"
        )
        print("  [PASS] shapes")

        # token-category invariants
        cats = cap.token_index.categories
        n_vis = (cats == int(TokenCategory.VISUAL)).sum().item()
        assert n_vis == cap.n_image_tokens, (
            f"VISUAL count {n_vis} != n_image_tokens {cap.n_image_tokens}"
        )
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
        delta = cap.residual[1].float() - cap.residual[0].float()
        expected = cap.attn_out[1].float() + cap.mlp_out[1].float()
        diff = (delta - expected).abs()
        max_err, mean_err = diff.max().item(), diff.mean().item()
        assert max_err < 2.0 and mean_err < 0.02, (
            f"Residual arithmetic: max_err={max_err:.4f}, mean_err={mean_err:.4f}"
        )
        print(f"  [PASS] residual arithmetic  max_err={max_err:.4f}")

        # logit sanity: yes/no split
        last = cap.logits[cap.token_index.prompt_last]
        yes_ids = tokenizer.encode("yes", add_special_tokens=False)
        no_ids = tokenizer.encode("no", add_special_tokens=False)
        if len(yes_ids) == 1 and len(no_ids) == 1:
            logit_yes = last[yes_ids[0]].item()
            logit_no = last[no_ids[0]].item()
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

        generated_text = (
            tokenizer.decode(cap_gen.generated_ids.tolist(), skip_special_tokens=True)
            if cap_gen.generated_ids is not None
            else ""
        )
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
            f"top1={top1_match} overlap={top5_overlap}/5  ({elapsed_gen:.2f}s)"
        )

    print("\n=== All SEED-LLaMA hook tests PASSED ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="AILab-CVC/seed-llama-8b-sft",
        help="Path or HF hub ID of the SEED-LLaMA-8B checkpoint.",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="AILab-CVC/seed-tokenizer-2",
        help="Path or HF hub ID of the SeedLlamaTokenizer.",
    )
    args = parser.parse_args()
    _test_real_model(args.model_path, args.tokenizer_path)


if __name__ == "__main__":
    main()
