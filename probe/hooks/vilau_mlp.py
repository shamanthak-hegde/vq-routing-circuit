"""
VilaUMLPHookManager — hook manager for VILA-U with continuous SigLIP adapter.

Extends VilaUHookManager.  The mm_projector has been replaced by a
SigLIPMLPAdapter (continuous, no VQ quantization).  Additionally, the
SigLIP feature capture hook is active so that the MLP adapter receives
continuous pre-quantization features rather than VQ code embeddings.

This is the reverse-ablation model:
  - VILA-U (VQ projector): L0 mass ≈ 17.50, yes_rate ≈ 81%, pathological
  - VILA-U-MLP (continuous adapter): L0 mass should drop, yes_rate ≈ 50%, healthy

Loading a VILA-U-MLP checkpoint:

    from probe.training.train_vilau_mlp import SigLIPMLPAdapter
    from probe.hooks.vilau_mlp import VilaUMLPHookManager

    # Load base VILA-U
    tokenizer, model, image_processor, _ = load_pretrained_model(
        "mit-han-lab/vila-u-7b-256", attn_implementation="eager"
    )
    model.eval()

    # Load and install trained adapter
    ckpt = torch.load("checkpoints/vilau_mlp/adapter_final.pt", map_location="cpu")
    adapter = SigLIPMLPAdapter(ckpt["siglip_dim"], ckpt["lm_dim"])
    adapter.load_state_dict(ckpt["adapter"])
    _ref = next(model.llm.parameters())
    model.mm_projector = adapter.to(device=_ref.device, dtype=_ref.dtype).eval()

    hm = VilaUMLPHookManager(model, tokenizer, image_processor)
"""

from __future__ import annotations

import torch

from .vilau import VilaUHookManager
from .utils import finalize_store, register_captures, remove_handles


class VilaUMLPHookManager(VilaUHookManager):
    """Hook manager for VILA-U with a continuous SigLIP MLP adapter.

    Identical to VilaUHookManager except:
    - mm_projector is a SigLIPMLPAdapter (no VQ codes)
    - A forward hook on the SigLIP penultimate encoder layer captures
      continuous features and routes them to the adapter, bypassing RQVAE
    - _get_vq_codes() returns None (no VQ codebook)

    Parameters
    ----------
    model              : VILAULlamaModel in eval mode, with mm_projector
                         replaced by SigLIPMLPAdapter
    tokenizer          : matching tokenizer
    image_processor    : model.vision_tower.image_processor
    conv_mode          : conversation template (default "vicuna_v1")
    """

    def __init__(self, model, tokenizer, image_processor, conv_mode="vicuna_v1",
                 capture_attention_weights=False):
        super().__init__(
            model, tokenizer, image_processor, conv_mode, capture_attention_weights
        )
        self._install_siglip_bypass_hook()

    def _install_siglip_bypass_hook(self) -> None:
        """Register hooks to bypass RQVAE and feed SigLIP features to the adapter."""
        rqvaesiglip = self.model.vision_tower.vision_tower.rqvaesiglip
        vision_model = rqvaesiglip.siglip_model.vision_model
        n_layers = len(vision_model.encoder.layers)

        self._siglip_feats: torch.Tensor | None = None

        def _capture(module, inp, out) -> None:
            self._siglip_feats = out[0]  # (B, L, C)

        def _inject(module, inp):
            if self._siglip_feats is not None:
                feats = self._siglip_feats
                self._siglip_feats = None
                return (feats.to(inp[0].device, dtype=inp[0].dtype),)
            return inp

        self._handle_capture = vision_model.encoder.layers[n_layers - 2].register_forward_hook(_capture)
        self._handle_inject = self.model.mm_projector.register_forward_pre_hook(_inject)

    def _get_vq_codes(self, image) -> None:
        return None  # no VQ codebook in continuous adapter

    def __del__(self) -> None:
        try:
            if hasattr(self, "_handle_capture"):
                self._handle_capture.remove()
            if hasattr(self, "_handle_inject"):
                self._handle_inject.remove()
        except Exception:
            pass
