"""Runtime shims that the pinned remote model code needs on transformers 5.x.

ByteDance/Ouro-2.6B computes default RoPE with ``ROPE_INIT_FUNCTIONS["default"]``
and does not define ``compute_default_rope_parameters``. transformers 5 removed
that registry entry and, during ``post_init``, calls the missing method
whenever ``rope_type == "default"``. Registering the original formula and
attaching it before the library init path keeps the checkpoint's RoPE intact.
"""

from __future__ import annotations

import torch

_INSTALLED = False


def default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
    """Inverse frequencies for the original RoPE used by Ouro (``rope_type="default"``)."""

    del seq_len, layer_type
    base = config.rope_theta
    partial = getattr(config, "partial_rotary_factor", 1.0) or 1.0
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial)
    exponent = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim
    inv_freq = 1.0 / (base ** exponent)
    return inv_freq, 1.0


def install_runtime_compat() -> None:
    """Install process-wide shims. Safe to call more than once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.modeling_utils import PreTrainedModel

    ROPE_INIT_FUNCTIONS.setdefault("default", default_rope_parameters)

    original = PreTrainedModel._init_weights
    if not getattr(original, "_loop_jlens_wrapped", False):

        def _init_weights(self, module):
            if (
                "RotaryEmbedding" in type(module).__name__
                and getattr(module, "rope_type", None) == "default"
                and hasattr(module, "original_inv_freq")
                and not hasattr(module, "compute_default_rope_parameters")
            ):
                module.compute_default_rope_parameters = default_rope_parameters
            return original(self, module)

        _init_weights._loop_jlens_wrapped = True
        PreTrainedModel._init_weights = _init_weights

    _INSTALLED = True
