"""
QwenVQHookManager — hook manager for Qwen2.5-VL with VQLinearProjector (N-068).

Extends Qwen3VLHookManager.  The only change from stock Qwen2.5-VL is that
model.visual.merger has been replaced by a VQLinearProjector.  All forward-pass
mechanics are identical to Qwen3VLHookManager.

Loading a Qwen-VQ checkpoint:

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from probe.training.llava_vq_projector import VQLinearProjector

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", ...)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", ..., attn_implementation="eager",
    ).eval()

    ckpt = torch.load("checkpoints/qwen_vq/projector_final.pt", map_location="cpu")
    vq_proj = VQLinearProjector(
        clip_dim=ckpt["clip_dim"], lm_dim=ckpt["lm_dim"],
        code_dim=ckpt["code_dim"], codebook_size=ckpt["codebook_size"],
    )
    vq_proj.load_state_dict(ckpt["projector"])
    model.visual.merger = vq_proj.to(next(model.parameters()).device).eval()

    hm = QwenVQHookManager(model, processor)
"""

from __future__ import annotations

from PIL import Image

from .qwen3vl import Qwen3VLHookManager


class QwenVQHookManager(Qwen3VLHookManager):
    """Hook manager for Qwen2.5-VL-VQ (VQLinearProjector replacing visual merger).

    Parameters
    ----------
    model     : Qwen2_5_VLForConditionalGeneration with model.visual.merger
                replaced by a trained VQLinearProjector
    processor : AutoProcessor matching the base Qwen2.5-VL checkpoint
    """

    # ── VQ-specific interface ─────────────────────────────────────────────────

    def _get_vq_codes(self, image: Image.Image) -> dict | None:
        """Return VQ code information from the last forward pass.

        The VQLinearProjector stores code indices in `_last_codes` during its
        forward pass. We trigger a prefill and read the buffer.

        Returns
        -------
        dict with:
          'levels'         — [LongTensor(N,)] — one-level VQ, list of length 1
          'codebook_sizes' — [codebook_size]
        """
        import torch
        proj = self.model.visual.merger
        if not hasattr(proj, "_last_codes"):
            return None
        cap = self.run_prefill(image, "")
        codes_2d = proj._last_codes        # (B, N)
        codes_1d = codes_2d.reshape(-1).cpu()
        return {
            "levels": [codes_1d],
            "codebook_sizes": [int(proj.codebook_size)],
        }
