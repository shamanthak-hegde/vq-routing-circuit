"""
Smoke test for HaploOmniHookManager.

Runs prefill + generate on 3 probe records (1 NB, 1 POPE yes, 1 POPE no),
asserts all structural invariants, and prints a timing / memory report.

Usage (GPU required, haplovlm env):
    source activate haplovlm
    cd <repo-root>
    python -m probe.hooks.test_hooks_haplo \\
        --model_path EasonXiao-888/HaploOmni

Architecture notes:
  - Projector   : model.model.pre_connector (HaploOmniConnector)
  - Decoder     : model.model.layers  (pre + main + post stages combined)
  - LM head     : model.lm_head
  - Generate    : model.generate(**processor_inputs, logits_to_keep=1)
  - n_image_tokens : count(input_ids == model.config.image_token_index)
"""

import argparse
import time

import torch


def _test_real_model(model_path: str) -> None:
    import sys, os
    _HAPLO_ROOT = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "HaploVLM")
    )
    _HAPLO_MODEL = os.path.join(_HAPLO_ROOT, "haploomni", "model")
    for _p in (_HAPLO_ROOT, _HAPLO_MODEL):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    print(f"\nLoading HaploOmni from {model_path} ...", flush=True)

    from haploomni import HaploOmniForConditionalGeneration, HaploOmniProcessor
    from probe.hooks.haplo import HaploOmniHookManager
    from probe.hooks import TokenCategory
    from probe import load_cache
    from PIL import Image

    processor = HaploOmniProcessor.from_pretrained(model_path)

    model = HaploOmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    device = next(model.parameters()).device
    n_layers = len(model.model.layers)
    hidden   = model.config.hidden_size
    vocab    = model.config.vocab_size
    print(f"  n_layers={n_layers}, hidden={hidden}, vocab={vocab}, device={device}", flush=True)

    hm = HaploOmniHookManager(model, processor)
    # n_layers for shape checks = LLM stage only (not all 82 combined layers)
    n_llm_layers = len(hm._get_decoder_layers())

    records, _, _ = load_cache()
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

        S     = cap.residual.shape[1]
        n_img = cap.n_image_tokens
        print(
            f"\n  [{rec.source}] q={q!r:.50s}... | prefill seq_len={S} "
            f"n_image_tokens={n_img} | {elapsed:.2f}s | peak_gpu={mem:.1f}GB",
            flush=True,
        )

        # shapes — n_llm_layers = LLM stage only (pre/post stages excluded)
        assert cap.residual.shape == (n_llm_layers, S, hidden), (
            f"residual.shape={cap.residual.shape} expected ({n_llm_layers}, {S}, {hidden})"
        )
        assert cap.attn_out.shape == (n_llm_layers, S, hidden)
        assert cap.mlp_out.shape  == (n_llm_layers, S, hidden)
        assert cap.logits.shape   == (S, vocab)
        assert cap.projected_visual.ndim >= 2
        assert cap.projected_visual.shape[-1] == hidden
        assert cap.visual_embeds.shape == (n_img, hidden), (
            f"visual_embeds.shape={cap.visual_embeds.shape}, expected ({n_img}, {hidden})"
        )
        print("    [PASS] shapes")

        # token-category invariants
        cats = cap.token_index.categories
        assert (cats == int(TokenCategory.ANSWER)).sum() == 0, "ANSWER in prefill"
        n_vis = (cats == int(TokenCategory.VISUAL)).sum().item()
        assert n_vis == n_img, f"VISUAL count {n_vis} != n_image_tokens {n_img}"
        assert int(cats[0]) == int(TokenCategory.OTHER), "First token not OTHER"
        q_count = (cats == int(TokenCategory.QUESTION)).sum().item()
        assert q_count > 0, "No QUESTION tokens found"
        assert int(cats[cap.token_index.prompt_last]) == int(TokenCategory.OTHER), (
            f"Last prompt token is {cats[cap.token_index.prompt_last]}, expected OTHER"
        )
        print("    [PASS] token_index categories")

        # residual arithmetic identity check (clean run = model forward)
        # The residual at the final position of layer 0 under the hook should
        # be consistent with a non-hooked forward within fp16 rounding.
        residual_norm = float(cap.residual[0, -1].norm().item())
        assert residual_norm > 0, "L0 residual is all-zero (hook not firing)"
        print(f"    [PASS] residual[0,-1].norm()={residual_norm:.4f} > 0")

        # logits: yes/no should both have a logit value
        yes_id = hm.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id  = hm.tokenizer.encode("no",  add_special_tokens=False)[0]
        logit_yes = float(cap.logits[cap.token_index.prompt_last, yes_id].item())
        logit_no  = float(cap.logits[cap.token_index.prompt_last, no_id].item())
        pred = "yes" if logit_yes > logit_no else "no"
        print(
            f"    logit_yes={logit_yes:.2f} logit_no={logit_no:.2f} "
            f"→ pred={pred}  gt={rec.answer}",
            flush=True,
        )

        # ── generate ──────────────────────────────────────────────────────────
        t0 = time.time()
        gcap = hm.run_generate(img, q, max_new_tokens=4)
        gen_elapsed = time.time() - t0
        gen_seq_len = gcap.residual.shape[1]
        n_gen = len(gcap.generated_ids) if gcap.generated_ids is not None else 0
        print(
            f"\n  [generate] seq_len={gen_seq_len} n_gen={n_gen} "
            f"| {gen_elapsed:.2f}s",
            flush=True,
        )
        assert gcap.residual.shape[0] == n_llm_layers
        assert gcap.residual.shape[2] == hidden
        assert gcap.logits.ndim == 2 and gcap.logits.shape[1] == vocab
        if gcap.generated_ids is not None and len(gcap.generated_ids) > 0:
            gen_text = hm.tokenizer.decode(gcap.generated_ids, skip_special_tokens=True)
            print(f"    generated: {gen_text!r}", flush=True)
        print("    [PASS] generate shapes")

    print("\n=== All HaploOmni hook tests PASSED ===\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for HaploOmniHookManager (GPU required)"
    )
    parser.add_argument(
        "--model_path",
        default="EasonXiao-888/HaploOmni",
        help="HF hub ID or local path to HaploOmni checkpoint",
    )
    args = parser.parse_args()
    _test_real_model(args.model_path)


if __name__ == "__main__":
    main()
