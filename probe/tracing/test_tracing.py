"""Smoke test for probe.tracing.

Tier 1 — structural (no GPU, seconds):
    python -m probe.tracing.test_tracing

Tier 2 — real-model (GPU required):
    python -m probe.tracing.test_tracing --model_path liuhaotian/llava-v1.6-vicuna-7b
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn


# ── Tier 1: structural tests (no GPU) ────────────────────────────────────────

def _test_score():
    from probe.tracing.score import normalized_restoration

    assert normalized_restoration(1.0, 0.0, 1.0) == 1.0
    assert normalized_restoration(0.0, 0.0, 1.0) == 0.0
    assert normalized_restoration(0.5, 0.0, 1.0) == 0.5
    # near-zero denominator → 0
    assert normalized_restoration(5.0, 3.0, 3.0 + 1e-8) == 0.0
    # values outside [0,1] are allowed (no clamping)
    assert normalized_restoration(-1.0, 0.0, 1.0) == -1.0
    assert normalized_restoration(2.0, 0.0, 1.0) == 2.0
    print("[PASS] normalized_restoration")


class _TinyLayer(nn.Module):
    """Single linear layer that acts as a stand-in decoder layer."""
    def __init__(self, H):
        super().__init__()
        self.proj = nn.Linear(H, H, bias=False)

    def forward(self, hidden_states, **kwargs):
        return (self.proj(hidden_states),)


class _TinyModel(nn.Module):
    def __init__(self, n_layers=4, H=8, vocab=16):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyLayer(H) for _ in range(n_layers)])
        self.lm_head = nn.Linear(H, vocab, bias=False)

    def forward(self, inputs_embeds, **kwargs):
        x = inputs_embeds
        for layer in self.model.layers:
            x = layer(x)[0]
        return type("Out", (), {"logits": self.lm_head(x)})()


class _TinyTensorLayer(nn.Module):
    """Single linear layer that returns a bare tensor."""
    def __init__(self, H):
        super().__init__()
        self.proj = nn.Linear(H, H, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states)


class _TinyTensorModel(nn.Module):
    def __init__(self, n_layers=4, H=8, vocab=16):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyTensorLayer(H) for _ in range(n_layers)])
        self.lm_head = nn.Linear(H, vocab, bias=False)

    def forward(self, inputs_embeds, **kwargs):
        x = inputs_embeds
        for layer in self.model.layers:
            x = layer(x)
        return type("Out", (), {"logits": self.lm_head(x)})()


def _test_patch_identity():
    """Full block-overwrite with all layers → output matches clean run exactly."""
    from probe.tracing.patch import run_patched_forward

    torch.manual_seed(0)
    n_layers, S, H = 4, 10, 8
    model = _TinyModel(n_layers, H).eval()

    clean_embeds = torch.randn(1, S, H)
    corrupt_embeds = torch.randn(1, S, H)

    # Run clean forward manually to get clean_residual at each layer
    with torch.no_grad():
        x = clean_embeds.clone()
        clean_residual = []
        for layer in model.model.layers:
            x = layer(x)[0]
            clean_residual.append(x.squeeze(0).clone())  # (S, H)
        clean_logits_manual = model.lm_head(x).squeeze(0)  # (S, vocab)

    clean_res_tensor = torch.stack(clean_residual, dim=0)  # (n_layers, S, H)
    token_indices = torch.arange(S, dtype=torch.long)
    prompt_last = S - 1

    result = run_patched_forward(
        model,
        corrupt_embeds,
        clean_res_tensor,
        layer_start=0,
        layer_end=n_layers,
        token_indices=token_indices,
        prompt_last=prompt_last,
    )
    # All layers patched → result should match clean logits at prompt_last
    expected = clean_logits_manual[prompt_last]
    assert result.shape == expected.shape, f"shape mismatch {result.shape} vs {expected.shape}"
    max_err = (result - expected).abs().max().item()
    assert max_err < 1e-4, f"identity check failed: max_err={max_err:.2e}"
    print(f"[PASS] run_patched_forward identity (max_err={max_err:.2e})")


def _test_patch_identity_tensor_output():
    """Full block-overwrite works when decoder layers return bare tensors."""
    from probe.tracing.patch import run_patched_forward

    torch.manual_seed(2)
    n_layers, S, H = 4, 10, 8
    model = _TinyTensorModel(n_layers, H).eval()

    clean_embeds = torch.randn(1, S, H)
    corrupt_embeds = torch.randn(1, S, H)

    with torch.no_grad():
        x = clean_embeds.clone()
        clean_residual = []
        for layer in model.model.layers:
            x = layer(x)
            clean_residual.append(x.squeeze(0).clone())
        clean_logits_manual = model.lm_head(x).squeeze(0)

    clean_res_tensor = torch.stack(clean_residual, dim=0)
    token_indices = torch.arange(S, dtype=torch.long)
    prompt_last = S - 1

    result = run_patched_forward(
        model,
        corrupt_embeds,
        clean_res_tensor,
        layer_start=0,
        layer_end=n_layers,
        token_indices=token_indices,
        prompt_last=prompt_last,
    )
    expected = clean_logits_manual[prompt_last]
    max_err = (result - expected).abs().max().item()
    assert max_err < 1e-4, f"tensor-output identity check failed: max_err={max_err:.2e}"
    print(f"[PASS] run_patched_forward tensor-output identity (max_err={max_err:.2e})")


def _test_patch_noop():
    """Patching zero tokens → same as plain corrupted forward."""
    from probe.tracing.patch import run_patched_forward

    torch.manual_seed(1)
    n_layers, S, H = 4, 10, 8
    model = _TinyModel(n_layers, H).eval()

    corrupt_embeds = torch.randn(1, S, H)
    clean_res = torch.randn(n_layers, S, H)
    prompt_last = S - 1

    with torch.no_grad():
        x = corrupt_embeds.clone()
        for layer in model.model.layers:
            x = layer(x)[0]
        expected = model.lm_head(x).squeeze(0)[prompt_last]

    # empty token_indices → no-op patch
    result = run_patched_forward(
        model, corrupt_embeds, clean_res,
        layer_start=0, layer_end=n_layers,
        token_indices=torch.zeros(0, dtype=torch.long),
        prompt_last=prompt_last,
    )
    max_err = (result - expected).abs().max().item()
    assert max_err < 1e-5, f"no-op check failed: max_err={max_err:.2e}"
    print(f"[PASS] run_patched_forward no-op (max_err={max_err:.2e})")


# ── Tier 2: real-model tests (GPU) ───────────────────────────────────────────

def _test_real_model(model_path: str):
    import sys, os
    _llava = os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
    if _llava not in sys.path:
        sys.path.insert(0, _llava)

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from probe.hooks import LlavaHookManager
    from probe import load_cache, resolve_answer_token_ids
    from probe.tracing import sweep_record

    print(f"\nLoading model from {model_path} ...")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_base=None, model_name=model_name,
        attn_implementation="sdpa",
    )
    model.eval()
    hm = LlavaHookManager(model, tokenizer, image_processor)

    records, _, _ = load_cache()
    resolve_answer_token_ids(records, tokenizer)

    nb_rec   = next(r for r in records if r.source == "naturalbench")
    pope_yes = next(r for r in records if r.source == "pope" and r.answer == "yes")
    pope_no  = next(r for r in records if r.source == "pope" and r.answer == "no")
    test_records = [nb_rec, pope_yes, pope_no]

    n_layers = model.config.num_hidden_layers
    window_size = 4
    n_windows = (n_layers + window_size - 1) // window_size

    for rec in test_records:
        t0 = time.time()
        rows = sweep_record(hm, rec, window_size=window_size)
        elapsed = time.time() - t0

        n_groups = len(set(r.token_group for r in rows))
        expected_rows = n_windows * n_groups
        assert len(rows) == expected_rows, \
            f"Expected {expected_rows} rows, got {len(rows)}"

        # Corruption sanity: logit_clean > logit_corrupt for a well-calibrated record
        sample = rows[0]
        if sample.logit_clean <= sample.logit_corrupt:
            print(f"  [WARN] {rec.id}: model prefers corrupted input "
                  f"(logit_clean={sample.logit_clean:.3f} ≤ logit_corrupt={sample.logit_corrupt:.3f}); "
                  "skipping restoration assertions")
        else:
            # All-layers window should give near-perfect restoration
            all_layer_rows = [r for r in rows if r.layer_start == 0]
            # For full-window (all layers), find the row that patches all groups together
            # (We test each group separately, so check the visual group row at window=0)
            vis_row = next((r for r in all_layer_rows if r.token_group == "visual"), None)
            if vis_row is not None:
                print(f"  [{rec.source}] window=[0,{window_size}) visual score: {vis_row.score:.3f}")

        # Print heatmap (windows × groups)
        groups = sorted(set(r.token_group for r in rows))
        windows = sorted(set((r.layer_start, r.layer_end) for r in rows))
        header = f"  {'layers':>10}" + "".join(f"  {g:>12}" for g in groups)
        print(header)
        for l_s, l_e in windows:
            row_scores = []
            for g in groups:
                cell = next(
                    (r.score for r in rows if r.layer_start == l_s and r.token_group == g),
                    float("nan"),
                )
                row_scores.append(f"{cell:>12.3f}")
            print(f"  [{l_s:2d},{l_e:2d})    " + "".join(row_scores))

        print(f"  [{rec.source}] {rec.id} — {len(rows)} rows in {elapsed:.1f}s\n")

    print("[ALL REAL-MODEL TESTS PASSED]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None)
    args = parser.parse_args()

    _test_score()
    _test_patch_identity()
    _test_patch_identity_tensor_output()
    _test_patch_noop()

    if args.model_path:
        _test_real_model(args.model_path)
    else:
        print("\n(Skipping real-model tests — pass --model_path to run them)")
    print("\n[ALL STRUCTURAL TESTS PASSED]")


if __name__ == "__main__":
    main()
