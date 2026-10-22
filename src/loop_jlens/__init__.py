"""Jacobian-lens reproduction utilities for looped transformers."""

from loop_jlens.adapters import (
    HuginnLensModel,
    NanbeigeLensModel,
    OuroLensModel,
    VirtualBlock,
    from_looped_hf,
)
from loop_jlens.compat import install_runtime_compat
from loop_jlens.validation import ValidationReport, validate_adapter

install_runtime_compat()

__all__ = [
    "HuginnLensModel",
    "NanbeigeLensModel",
    "OuroLensModel",
    "ValidationReport",
    "VirtualBlock",
    "from_looped_hf",
    "validate_adapter",
]
