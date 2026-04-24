"""
probe.hooks — activation-capture hook layer for VLMs.

Supported models
----------------
LLaVA-1.6 (llava-v1.6-vicuna-7b)   →  LlavaHookManager
VILA-U (vila-u-7b-256/384)          →  VilaUHookManager
VILA v1.0 (VILA-7b)                 →  probe.hooks.vila.VilaHookManager
  (import explicitly — VILA and LLaVA share the llava.* namespace; importing
   both in the same process is unsupported; each lives in its own conda env)

All model-specific imports are lazy (inside __init__) so that importing this
package does not pollute sys.modules with llava.* or vila_u.* entries.
"""

from .schema import Capture, TokenCategory, TokenIndex
from .llava import LlavaHookManager
from .vilau import VilaUHookManager

__all__ = [
    "Capture",
    "TokenCategory",
    "TokenIndex",
    "LlavaHookManager",
    "VilaUHookManager",
]
