"""Structural tests for WindowAttnKnockout. CPU-only, no GPU required.

Run:
    python -m probe.tracing.test_window_knockout
"""

from __future__ import annotations

import types
import unittest

import torch
import torch.nn as nn


# ── stub hook manager ─────────────────────────────────────────────────────────

class _StubAttn(nn.Module):
    """Fake self_attn that returns a constant tensor so we can detect zeroing."""

    def __init__(self, val: float):
        super().__init__()
        self._val = val

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        out = torch.full_like(x, self._val)
        return (out, None)


class _StubLayer(nn.Module):
    def __init__(self, val: float):
        super().__init__()
        self.self_attn = _StubAttn(val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x)
        return x + attn_out


class _StubHookManager:
    def __init__(self, n_layers: int):
        # start at 1.0 so that no layer naturally produces all-zero output
        self._layers = nn.ModuleList([_StubLayer(float(i + 1)) for i in range(n_layers)])

    def _get_decoder_layers(self):
        return self._layers

    def run_layer(self, l: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (attn_out, layer_out) for layer l."""
        attn_bucket: list[torch.Tensor] = []

        def _capture(module, inp, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            attn_bucket.append(t.clone())

        h = self._layers[l].self_attn.register_forward_hook(_capture)
        layer_out = self._layers[l](x)
        h.remove()
        return attn_bucket[0], layer_out


# ── tests ─────────────────────────────────────────────────────────────────────

class TestWindowAttnKnockout(unittest.TestCase):

    def setUp(self):
        from probe.tracing.head_knockout import WindowAttnKnockout
        self.WindowAttnKnockout = WindowAttnKnockout
        self.n_layers = 8
        self.hm = _StubHookManager(self.n_layers)
        self.x = torch.ones(1, 4, 8)  # (batch, seq, hidden)

    def _attn_out(self, l: int) -> torch.Tensor:
        out, _ = self.hm.run_layer(l, self.x)
        return out

    def test_window_zeroes_target_layers(self):
        """Layers 2,3,4,5 in [2,6) must produce zero attn output."""
        with self.WindowAttnKnockout(self.hm, 2, 6):
            for l in range(2, 6):
                out = self._attn_out(l)
                self.assertTrue(
                    out.eq(0).all().item(),
                    f"Layer {l} inside [2,6) should be zeroed but got {out.unique()}"
                )

    def test_window_leaves_other_layers_intact(self):
        """Layers 0,1,6,7 outside [2,6) must NOT be zeroed."""
        with self.WindowAttnKnockout(self.hm, 2, 6):
            for l in [0, 1, 6, 7]:
                out = self._attn_out(l)
                self.assertFalse(
                    out.eq(0).all().item(),
                    f"Layer {l} outside [2,6) should be non-zero but got zeros"
                )

    def test_hooks_removed_after_exit(self):
        """After the context exits, layers inside the window should be non-zero again."""
        with self.WindowAttnKnockout(self.hm, 2, 6):
            pass  # enter and immediately exit
        for l in range(2, 6):
            out = self._attn_out(l)
            self.assertFalse(
                out.eq(0).all().item(),
                f"Layer {l} should be non-zero after context exit but got zeros"
            )

    def test_handles_cleared_after_exit(self):
        wk = self.WindowAttnKnockout(self.hm, 0, 4)
        with wk:
            self.assertEqual(len(wk._handles), 4)
        self.assertEqual(len(wk._handles), 0)

    def test_invalid_window_raises(self):
        with self.assertRaises(IndexError):
            with self.WindowAttnKnockout(self.hm, 5, 3):  # start >= end
                pass

    def test_out_of_bounds_raises(self):
        with self.assertRaises(IndexError):
            with self.WindowAttnKnockout(self.hm, 0, self.n_layers + 2):
                pass

    def test_single_layer_window(self):
        """A window of size 1 behaves like LayerAttnKnockout."""
        with self.WindowAttnKnockout(self.hm, 3, 4):
            out = self._attn_out(3)
            self.assertTrue(out.eq(0).all().item())
            out_other = self._attn_out(2)
            self.assertFalse(out_other.eq(0).all().item())

    def test_tuple_output_zeroed(self):
        """Verify _zero_attn_output handles tuple-output self_attn correctly."""
        from probe.tracing.head_knockout import _zero_attn_output
        t = torch.ones(2, 3)
        result = _zero_attn_output(None, (), (t, None))
        self.assertTrue(result[0].eq(0).all().item())
        self.assertIsNone(result[1])

    def test_tensor_output_zeroed(self):
        from probe.tracing.head_knockout import _zero_attn_output
        t = torch.ones(2, 3)
        result = _zero_attn_output(None, (), t)
        self.assertTrue(result.eq(0).all().item())


class TestBestVisualWindow(unittest.TestCase):

    def test_argmax_correct(self):
        """best_visual_window returns the window with the highest mean visual score,
        WARN records (logit_clean <= logit_corrupt) excluded."""
        import json
        import tempfile
        from pathlib import Path
        from probe.tracing.best_window import best_visual_window

        rows = [
            # non-WARN: logit_clean > logit_corrupt
            {"token_group": "visual", "layer_start": 0, "layer_end": 4, "score": 0.5,
             "logit_clean": 2.0, "logit_corrupt": 1.0},
            {"token_group": "visual", "layer_start": 0, "layer_end": 4, "score": 0.7,
             "logit_clean": 2.0, "logit_corrupt": 1.0},
            {"token_group": "visual", "layer_start": 4, "layer_end": 8, "score": 1.2,
             "logit_clean": 2.0, "logit_corrupt": 1.0},
            {"token_group": "visual", "layer_start": 4, "layer_end": 8, "score": 0.8,
             "logit_clean": 2.0, "logit_corrupt": 1.0},
            # WARN record (logit_clean <= logit_corrupt): should be excluded
            {"token_group": "visual", "layer_start": 0, "layer_end": 4, "score": 99.0,
             "logit_clean": 1.0, "logit_corrupt": 2.0},
            # different token_group: should be excluded
            {"token_group": "question", "layer_start": 0, "layer_end": 4, "score": 9.9,
             "logit_clean": 2.0, "logit_corrupt": 1.0},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            tmp = Path(f.name)

        l_start, l_end, score = best_visual_window(tmp)
        tmp.unlink()
        self.assertEqual((l_start, l_end), (4, 8))
        self.assertAlmostEqual(score, 1.0, places=5)  # mean of 1.2 and 0.8


if __name__ == "__main__":
    unittest.main(verbosity=2)
