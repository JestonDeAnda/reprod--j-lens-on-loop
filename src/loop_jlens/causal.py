"""Shared residual-stream interventions for looped-model causal experiments."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
from torch import Tensor, nn


def unit_direction(vector: Tensor) -> Tensor:
    vector = vector.detach().float()
    norm = torch.linalg.vector_norm(vector)
    if norm == 0:
        raise ValueError("intervention direction must be nonzero")
    return vector / norm


def jacobian_token_direction(jacobian: Tensor, unembedding_vector: Tensor) -> Tensor:
    """Return ``normalize(J.T @ w_token)`` in the source residual basis."""

    return unit_direction(jacobian.detach().float().T @ unembedding_vector.detach().float())


def random_control_direction(reference: Tensor, *, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sample = torch.randn(reference.numel(), generator=generator)
    return unit_direction(sample).to(reference.device)


def orthogonal_control_direction(control: Tensor, target: Tensor) -> Tensor:
    """Remove the target-direction component from a matched control direction."""

    target = unit_direction(target)
    control = control.detach().float().to(target.device)
    return unit_direction(control - torch.dot(control, target) * target)


def intervene_residual(
    hidden: Tensor,
    direction: Tensor,
    *,
    positions: Sequence[int],
    mode: str,
    strength: float,
) -> Tensor:
    """Apply an additive or projection-ablation intervention without mutation."""

    if hidden.ndim != 3:
        raise ValueError("hidden state must have shape [batch, sequence, d_model]")
    if direction.ndim != 1 or direction.numel() != hidden.shape[-1]:
        raise ValueError("direction width must match hidden-state width")
    result = hidden.clone()
    selected = result[:, list(positions), :]
    direction = unit_direction(direction).to(selected.device, selected.dtype)
    if mode == "add":
        scale = torch.linalg.vector_norm(selected.float(), dim=-1, keepdim=True)
        selected = selected + strength * scale.to(selected.dtype) * direction
    elif mode == "ablate":
        projection = torch.sum(selected * direction, dim=-1, keepdim=True)
        selected = selected - strength * projection * direction
    else:
        raise ValueError(f"unsupported intervention mode: {mode}")
    result[:, list(positions), :] = selected
    return result


def _first_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"unsupported block output type: {type(output)!r}")


def _replace_first_tensor(output, tensor: Tensor):
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise TypeError(f"unsupported block output type: {type(output)!r}")


@contextmanager
def residual_intervention(
    block: nn.Module,
    direction: Tensor,
    *,
    positions: Sequence[int] = (-1,),
    mode: str = "add",
    strength: float = 0.5,
) -> Iterator[None]:
    """Temporarily modify one virtual residual-stream block."""

    def hook(_module, _inputs, output):
        changed = intervene_residual(
            _first_tensor(output),
            direction,
            positions=positions,
            mode=mode,
            strength=strength,
        )
        return _replace_first_tensor(output, changed)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def token_rank(logits: Tensor, token_id: int) -> int:
    score = logits[token_id]
    return int(1 + (logits > score).sum().item())
