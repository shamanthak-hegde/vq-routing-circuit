"""CPU-only sanity tests for the N-040A intervention library.

Tests that:
  1. SelectiveHeadKnockout zeros exactly the listed head slices and leaves others unchanged.
  2. ScalarLayerScale with alpha=0.0 is equivalent to full-zero output.
  3. ScalarLayerScale with alpha=1.0 is the identity.
  4. SelectiveHeadScale with alpha=0.0 zeroes listed heads and leaves others unchanged.
  5. SelectiveHeadScale with alpha=1.0 is the identity.
  6. build_intervention dispatches to the right class.
  7. _parse_heads handles string and list inputs.
  8. _get_head_dim infers correct head_dim from a mock layer.

All tests use lightweight fake modules — no GPU, no model checkpoint required.
"""

from __future__ import annotations

import types
import unittest
from typing import Any

import torch
import torch.nn as nn

from probe.tracing.head_knockout import (
    SelectiveHeadKnockout,
    ScalarLayerScale,
    SelectiveHeadScale,
    LayerAttnKnockout,
    build_intervention,
    _parse_heads,
    _get_head_dim,
)


N_HEADS = 8
HEAD_DIM = 4
HIDDEN = N_HEADS * HEAD_DIM  # 32
SEQ = 6
BATCH = 1


def _make_layer() -> nn.Module:
    """Build a minimal fake decoder layer with self_attn.o_proj."""
    o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
    nn.init.eye_(o_proj.weight)  # identity weights for easy checking

    self_attn = nn.Module()
    self_attn.o_proj = o_proj
    # Expose config with n_heads for _get_head_dim
    config = types.SimpleNamespace(num_attention_heads=N_HEADS)
    self_attn.config = config

    layer = nn.Module()
    layer.self_attn = self_attn
    return layer


def _make_hm(layer: nn.Module) -> Any:
    """Fake hook manager that returns a single-layer list."""
    hm = types.SimpleNamespace()
    hm._get_decoder_layers = lambda: [layer]
    return hm


class TestParseHeads(unittest.TestCase):
    def test_string(self) -> None:
        self.assertEqual(_parse_heads("6,7,14"), [6, 7, 14])

    def test_list(self) -> None:
        self.assertEqual(_parse_heads([14, 6, 7]), [6, 7, 14])

    def test_none(self) -> None:
        self.assertEqual(_parse_heads(None), [])

    def test_single(self) -> None:
        self.assertEqual(_parse_heads("3"), [3])


class TestGetHeadDim(unittest.TestCase):
    def test_infer(self) -> None:
        layer = _make_layer()
        self.assertEqual(_get_head_dim(layer), HEAD_DIM)


class TestSelectiveHeadKnockout(unittest.TestCase):
    def _run(self, head_idxs: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run o_proj on random input with and without selective knockout."""
        layer = _make_layer()
        hm = _make_hm(layer)
        x = torch.randn(BATCH, SEQ, HIDDEN)
        x_ref = x.clone()

        ref_out = layer.self_attn.o_proj(x_ref)

        x_ko = x.clone()
        with SelectiveHeadKnockout(hm, 0, head_idxs):
            ko_out = layer.self_attn.o_proj(x_ko)

        return ref_out, ko_out

    def test_listed_heads_zeroed(self) -> None:
        heads = [2, 5]
        ref_out, ko_out = self._run(heads)
        for h in heads:
            ko_slice = ko_out[..., h * HEAD_DIM : (h + 1) * HEAD_DIM]
            self.assertTrue(
                ko_slice.abs().max().item() == 0.0,
                f"Head {h} slice not zeroed in o_proj output",
            )

    def test_unlisted_heads_unchanged(self) -> None:
        heads = [2, 5]
        ref_out, ko_out = self._run(heads)
        for h in range(N_HEADS):
            if h in heads:
                continue
            ref_slice = ref_out[..., h * HEAD_DIM : (h + 1) * HEAD_DIM]
            ko_slice = ko_out[..., h * HEAD_DIM : (h + 1) * HEAD_DIM]
            self.assertTrue(
                torch.allclose(ref_slice, ko_slice),
                f"Head {h} (unlisted) changed under selective knockout",
            )

    def test_no_side_effects_outside_context(self) -> None:
        layer = _make_layer()
        hm = _make_hm(layer)
        x = torch.randn(BATCH, SEQ, HIDDEN)

        with SelectiveHeadKnockout(hm, 0, [1, 3]):
            pass  # enter and exit

        x2 = x.clone()
        out_after = layer.self_attn.o_proj(x2)
        out_ref = layer.self_attn.o_proj(x)
        self.assertTrue(torch.allclose(out_after, out_ref), "Hook not removed after __exit__")


class TestScalarLayerScale(unittest.TestCase):
    def _make_full_hm(self) -> tuple[Any, Any]:
        """Hook manager wrapping a fake self_attn that returns a known tensor."""
        layer = _make_layer()
        hm = _make_hm(layer)
        return hm, layer

    def test_alpha_zero_zeroes_output(self) -> None:
        hm, layer = self._make_full_hm()
        captured: list[Any] = []

        def fake_forward(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, None]:
            out = torch.ones(BATCH, SEQ, HIDDEN)
            captured.append(out)
            return (out,)

        layer.self_attn.forward = fake_forward

        with ScalarLayerScale(hm, 0, alpha=0.0):
            result = layer.self_attn(None)

        self.assertTrue(result[0].abs().max().item() == 0.0)

    def test_alpha_one_is_identity(self) -> None:
        layer = _make_layer()
        hm = _make_hm(layer)
        captured: list[Any] = []

        expected = torch.randn(BATCH, SEQ, HIDDEN)

        def fake_forward(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, None]:
            return (expected.clone(),)

        layer.self_attn.forward = fake_forward

        with ScalarLayerScale(hm, 0, alpha=1.0):
            result = layer.self_attn(None)

        self.assertTrue(torch.allclose(result[0], expected))

    def test_alpha_half(self) -> None:
        layer = _make_layer()
        hm = _make_hm(layer)
        val = torch.ones(BATCH, SEQ, HIDDEN) * 4.0

        def fake_forward(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, None]:
            return (val.clone(),)

        layer.self_attn.forward = fake_forward

        with ScalarLayerScale(hm, 0, alpha=0.5):
            result = layer.self_attn(None)

        self.assertTrue(torch.allclose(result[0], val * 0.5))


class TestSelectiveHeadScale(unittest.TestCase):
    def _run(self, head_idxs: list[int], alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
        layer = _make_layer()
        hm = _make_hm(layer)
        x = torch.randn(BATCH, SEQ, HIDDEN)

        ref_out = layer.self_attn.o_proj(x.clone())

        with SelectiveHeadScale(hm, 0, head_idxs, alpha):
            ko_out = layer.self_attn.o_proj(x.clone())

        return ref_out, ko_out

    def test_alpha_zero_zeroes_listed(self) -> None:
        heads = [1, 4]
        ref_out, ko_out = self._run(heads, 0.0)
        for h in heads:
            self.assertTrue(
                ko_out[..., h * HEAD_DIM : (h + 1) * HEAD_DIM].abs().max().item() == 0.0
            )

    def test_alpha_zero_unlisted_unchanged(self) -> None:
        heads = [1, 4]
        ref_out, ko_out = self._run(heads, 0.0)
        for h in range(N_HEADS):
            if h in heads:
                continue
            self.assertTrue(
                torch.allclose(
                    ref_out[..., h * HEAD_DIM : (h + 1) * HEAD_DIM],
                    ko_out[..., h * HEAD_DIM : (h + 1) * HEAD_DIM],
                )
            )

    def test_alpha_one_is_identity(self) -> None:
        heads = [0, 3, 7]
        ref_out, ko_out = self._run(heads, 1.0)
        self.assertTrue(torch.allclose(ref_out, ko_out))

    def test_alpha_half_scales_listed(self) -> None:
        heads = [2]
        ref_out, ko_out = self._run(heads, 0.5)
        ref_slice = ref_out[..., 2 * HEAD_DIM : 3 * HEAD_DIM]
        ko_slice = ko_out[..., 2 * HEAD_DIM : 3 * HEAD_DIM]
        self.assertTrue(torch.allclose(ko_slice, ref_slice * 0.5, atol=1e-5))


class TestBuildIntervention(unittest.TestCase):
    def _layer_hm(self) -> Any:
        return _make_hm(_make_layer())

    def test_pathological_route_ablation(self) -> None:
        hm = self._layer_hm()
        iv = build_intervention("pathological_route_ablation", hm, 0, [], 0.5)
        self.assertIsInstance(iv, LayerAttnKnockout)

    def test_full_zero_alias(self) -> None:
        hm = self._layer_hm()
        iv = build_intervention("full_zero", hm, 0, [], 0.5)
        self.assertIsInstance(iv, LayerAttnKnockout)

    def test_selective(self) -> None:
        hm = self._layer_hm()
        iv = build_intervention("selective", hm, 0, [6, 7], 0.5)
        self.assertIsInstance(iv, SelectiveHeadKnockout)

    def test_scalar(self) -> None:
        hm = self._layer_hm()
        iv = build_intervention("scalar", hm, 0, [], 0.5)
        self.assertIsInstance(iv, ScalarLayerScale)

    def test_selective_scalar(self) -> None:
        hm = self._layer_hm()
        iv = build_intervention("selective_scalar", hm, 0, [6, 7], 0.5)
        self.assertIsInstance(iv, SelectiveHeadScale)

    def test_unknown_mode(self) -> None:
        hm = self._layer_hm()
        with self.assertRaises(ValueError):
            build_intervention("nonexistent", hm, 0, [], 0.5)


if __name__ == "__main__":
    unittest.main()
