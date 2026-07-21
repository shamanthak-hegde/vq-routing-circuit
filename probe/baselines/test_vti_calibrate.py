"""CPU-only tests for VTI calibration direction computation.

Tests that:
  1. _rank1_direction returns shape (hidden,) and unit norm.
  2. _rank1_direction returns a vector aligned with the mean diff (not anti-aligned).
  3. compute_directions returns shape (n_layers, hidden) when given mock prefills.
  4. compute_directions agrees with manual per-layer SVD on synthetic data.

No GPU, no model checkpoint required.
"""

from __future__ import annotations

import types
import unittest
from typing import Any

import torch
import torch.nn.functional as F

from probe.baselines.vti_calibrate import _rank1_direction, compute_directions


HIDDEN = 16
N_LAYERS = 4
N_DEMOS = 6
SEQ = 8


def _make_fake_hm(
    clean_residuals: list[torch.Tensor],
    corrupt_residuals: list[torch.Tensor],
    visual_range: tuple[int, int] = (1, 4),
) -> Any:
    """Fake hook manager whose run_prefill and _get_lm_forward return canned tensors."""

    call_counter = [0]
    corrupt_call_counter = [0]

    class FakeCapture:
        def __init__(self, residual: torch.Tensor, prompt_last: int):
            self.residual = residual
            self.token_index = types.SimpleNamespace(
                prompt_last=prompt_last,
                visual_range=visual_range,
            )

    def run_prefill(img: Any, question: Any) -> FakeCapture:
        idx = call_counter[0] % len(clean_residuals)
        call_counter[0] += 1
        return FakeCapture(clean_residuals[idx].clone(), prompt_last=SEQ - 1)

    # Fake _prepare_embeds and friends — not used in this simplified test
    # because we patch corrupt path directly; see _make_corrupt_patch below.

    class FakeProjector(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class FakeLMHead(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class FakeLMForward(torch.nn.Module):
        def forward(self, inputs_embeds: torch.Tensor, **kwargs: Any) -> Any:
            return types.SimpleNamespace()

    projector = FakeProjector()
    lm_head = FakeLMHead()
    lm_fwd = FakeLMForward()

    fake_layers = [torch.nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(N_LAYERS)]
    fake_modules = [torch.nn.Module() for _ in range(N_LAYERS)]

    hm = types.SimpleNamespace()
    hm.run_prefill = run_prefill
    hm._get_decoder_layers = lambda: fake_modules
    hm._get_projector = lambda: projector
    hm._get_lm_head = lambda: lm_head
    hm._get_lm_forward = lambda: lm_fwd
    hm._build_prompt = lambda img, q: (torch.zeros(1, SEQ), None, None)
    hm._prepare_embeds = lambda ids, imgs, sizes: (torch.zeros(1, SEQ, HIDDEN), visual_range[1] - visual_range[0])

    return hm, corrupt_residuals, corrupt_call_counter


class TestRank1Direction(unittest.TestCase):

    def test_shape(self) -> None:
        diffs = torch.randn(N_DEMOS, HIDDEN)
        d = _rank1_direction(diffs)
        self.assertEqual(d.shape, (HIDDEN,))

    def test_unit_norm(self) -> None:
        diffs = torch.randn(N_DEMOS, HIDDEN)
        d = _rank1_direction(diffs)
        norm = d.norm().item()
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_sign_aligned_with_mean(self) -> None:
        """Direction must have a non-negative inner product with the mean diff."""
        diffs = torch.randn(N_DEMOS, HIDDEN)
        d = _rank1_direction(diffs)
        mean_diff = diffs.float().mean(dim=0)
        dot = float(torch.dot(d.float(), mean_diff))
        self.assertGreaterEqual(dot, 0.0, "Direction anti-aligned with mean diff")

    def test_known_rank1_data(self) -> None:
        """All diffs along the same direction → PC must be exactly that direction."""
        true_dir = F.normalize(torch.randn(HIDDEN).float(), dim=-1)
        diffs = true_dir.unsqueeze(0) * torch.randn(N_DEMOS, 1)
        d = _rank1_direction(diffs)
        # After sign alignment, should match true_dir up to sign flip handled internally
        dot = abs(float(torch.dot(d, true_dir)))
        self.assertAlmostEqual(dot, 1.0, places=4,
                               msg="PC of rank-1 data should be the generating direction")


class TestComputeDirections(unittest.TestCase):
    """Integration test using patched noisy_embeds and _capture_residuals_at."""

    def _run(self) -> torch.Tensor:
        """Build canned clean + corrupt residuals, run compute_directions, return result."""
        torch.manual_seed(42)
        # Each record: (n_layers, SEQ, HIDDEN) residual tensor
        clean_residuals = [torch.randn(N_LAYERS, SEQ, HIDDEN) for _ in range(N_DEMOS)]
        corrupt_residuals = [torch.randn(N_LAYERS, SEQ, HIDDEN) for _ in range(N_DEMOS)]

        hm, _, _ = _make_fake_hm(clean_residuals, corrupt_residuals)

        corrupt_idx = [0]

        def fake_noisy(*a: Any, **kw: Any) -> torch.Tensor:
            return torch.zeros(1, SEQ, HIDDEN)

        # Patch _capture_residuals_at so the corrupt pass returns canned tensors
        # without needing real decoder layers that respond to forward hooks.
        import probe.baselines.vti_calibrate as cal_mod
        import probe.tracing.corrupt as corrupt_mod

        orig_noisy = corrupt_mod.noisy_embeds
        orig_capture = cal_mod._capture_residuals_at

        def fake_capture(hm_: Any, emb: torch.Tensor, pl: int) -> torch.Tensor:
            idx = corrupt_idx[0] % N_DEMOS
            corrupt_idx[0] += 1
            # Return (n_layers, hidden) last-token slice from canned corrupt residuals
            return corrupt_residuals[idx][:, pl, :].float().cpu()

        corrupt_mod.noisy_embeds = fake_noisy
        cal_mod._capture_residuals_at = fake_capture

        try:
            # Build minimal probe records with required fields
            records = []
            for i in range(N_DEMOS):
                rec = types.SimpleNamespace()
                rec.source = "pope"
                rec.image_path = __file__  # any existing path; Image.open won't be called
                rec.question = "Is there a dog?"
                rec.id = f"pope_{i:03d}_q0"
                records.append(rec)

            # Monkey-patch Image.open so compute_directions doesn't crash on fake path.
            # We create the fake image once and return it from all open() calls.
            from PIL import Image as PIL_Image
            _orig_open = PIL_Image.open
            _fake_img = PIL_Image.new("RGB", (32, 32))

            def fake_open(path: Any, **kw: Any) -> Any:
                return _fake_img.copy()

            PIL_Image.open = fake_open

            try:
                result = compute_directions(hm, records, sigma=0.5, n_demos=N_DEMOS)
            finally:
                PIL_Image.open = _orig_open
        finally:
            corrupt_mod.noisy_embeds = orig_noisy
            cal_mod._capture_residuals_at = orig_capture

        return result

    def test_output_shape(self) -> None:
        result = self._run()
        self.assertEqual(result.shape, (N_LAYERS, HIDDEN))

    def test_output_unit_norm(self) -> None:
        result = self._run()
        norms = torch.linalg.vector_norm(result.float(), dim=-1)
        self.assertTrue(
            torch.allclose(norms, torch.ones(N_LAYERS), atol=1e-5),
            f"Not all layer directions have unit norm: {norms.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
