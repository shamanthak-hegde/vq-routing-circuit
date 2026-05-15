"""
LlavaVQHookManager — hook manager for LLaVA-VQ (N-056).

Identical to LlavaHookManager except:
  - mm_projector is a VQLinearProjector instead of mlp2x_gelu
  - _get_vq_codes() returns the last discrete code indices stored by the
    projector's forward pass (mirrors VilaUHookManager._get_vq_codes)

Loading a LLaVA-VQ checkpoint:

    from llava.model.builder import load_pretrained_model
    tokenizer, model, image_processor, _ = load_pretrained_model(
        "liuhaotian/llava-v1.6-vicuna-7b", ...,
        attn_implementation="eager",   # required for capture_attention_weights
    )
    from probe.training.llava_vq_projector import VQLinearProjector
    vq_proj = VQLinearProjector(...)
    vq_proj.load_kmeans_codebook("checkpoints/codebook_init.pt")
    state = torch.load("checkpoints/llava_vq/projector_final.pt", map_location="cpu")
    vq_proj.load_state_dict(state["projector"])
    model.model.mm_projector = vq_proj.to(device).eval()
    hm = LlavaVQHookManager(model, tokenizer, image_processor)
"""

from __future__ import annotations

from PIL import Image

from .llava import LlavaHookManager


class LlavaVQHookManager(LlavaHookManager):
    """Hook manager for LLaVA-VQ (LLaVA with VQLinearProjector).

    Parameters
    ----------
    model           : LlavaLlamaForCausalLM with mm_projector replaced by
                      VQLinearProjector
    tokenizer       : matching tokenizer
    image_processor : matching CLIPImageProcessor
    conv_mode       : conversation template (default "llava_v1")
    capture_attention_weights : requires attn_implementation="eager"
    """

    # ── VQ-specific interface ─────────────────────────────────────────────────

    def _get_vq_codes(self, image: Image.Image) -> dict | None:
        """Return VQ code information using the last forward pass's codes.

        The VQLinearProjector stores code indices in `_last_codes` during its
        forward pass, so we run a prefill and then read the buffer.

        Returns
        -------
        dict with:
          'levels'         — [LongTensor(N,)] — one-level VQ, so list of length 1
          'codebook_sizes' — [16384]
        """
        import torch
        proj = self.model.model.mm_projector
        if not hasattr(proj, "_last_codes"):
            return None
        cap = self.run_prefill(image, "")
        codes_2d = proj._last_codes        # (B, N) — B=1
        codes_1d = codes_2d.reshape(-1).cpu()
        return {
            "levels": [codes_1d],
            "codebook_sizes": [int(proj.codebook_size)],
        }
