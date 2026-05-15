"""
probe.hooks — activation-capture hook layer for VLMs.

Supported models
----------------
LLaVA-1.6 (llava-v1.6-vicuna-7b)   →  LlavaHookManager
LLaVA-VQ  (N-056 constructive induction variant)  →  LlavaVQHookManager
VILA-U (vila-u-7b-256/384)          →  VilaUHookManager
VILA v1.0 (VILA-7b)                 →  probe.hooks.vila.VilaHookManager
UniTok MLLM (unitok_mllm)           →  UniTokHookManager
Qwen2.5-VL-7B-Instruct              →  Qwen3VLHookManager
LaVIT-7B-v2                         →  LavitHookManager
  (import explicitly — VILA and LLaVA share the llava.* namespace; importing
   both in the same process is unsupported; each lives in its own conda env)

All model-specific imports are lazy (inside __init__) so that importing this
package does not pollute sys.modules with llava.* or vila_u.* entries.
"""

from .schema import Capture, TokenCategory, TokenIndex
from .llava import LlavaHookManager
from .llava_vq import LlavaVQHookManager
from .vilau import VilaUHookManager
from .unitok import UniTokHookManager
from .qwen3vl import Qwen3VLHookManager
from .emu3 import Emu3HookManager
from .lavit import LavitHookManager

__all__ = [
    "Capture",
    "TokenCategory",
    "TokenIndex",
    "LlavaHookManager",
    "LlavaVQHookManager",
    "VilaUHookManager",
    "UniTokHookManager",
    "Qwen3VLHookManager",
    "Emu3HookManager",
    "LavitHookManager",
]
