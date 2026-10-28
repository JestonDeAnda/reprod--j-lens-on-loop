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
    scale_mode: str = "per_position",
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
        norms = torch.linalg.vector_norm(selected.float(), dim=-1, keepdim=True)
        if scale_mode == "per_position":
            scale = norms
        elif scale_mode == "mean":
            scale = norms.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"unsupported scale mode: {scale_mode}")
        selected = selected + strength * scale.to(selected.dtype) * direction
    elif mode == "ablate":
        projection = torch.sum(selected * direction, dim=-1, keepdim=True)
        selected = selected - strength * projection * direction
    else:
        raise ValueError(f"unsupported intervention mode: {mode}")
    result[:, list(positions), :] = selected
    return result


def swap_direction_coordinate(
    hidden: Tensor,
    source_direction: Tensor,
    target_direction: Tensor,
    *,
    positions: Sequence[int],
    strength: float = 1.0,
) -> Tensor:
    """Transfer the source coordinate into the target coordinate."""

    result = hidden.clone()
    selected = result[:, list(positions), :]
    source = unit_direction(source_direction).to(selected.device, selected.dtype)
    target = unit_direction(target_direction).to(selected.device, selected.dtype)
    basis = torch.stack([source, target], dim=1)
    gram_inverse = torch.linalg.pinv((basis.T @ basis).float()).to(basis.dtype)
    coefficients = selected @ basis @ gram_inverse
    source_coefficient = coefficients[..., :1]
    delta = source_coefficient * (target - source)
    result[:, list(positions), :] = selected + strength * delta
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
    scale_mode: str = "per_position",
) -> Iterator[None]:
    """Temporarily modify one virtual residual-stream block."""

    def hook(_module, _inputs, output):
        changed = intervene_residual(
            _first_tensor(output),
            direction,
            positions=positions,
            mode=mode,
            strength=strength,
            scale_mode=scale_mode,
        )
        return _replace_first_tensor(output, changed)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def residual_coordinate_swap(
    block: nn.Module,
    source_direction: Tensor,
    target_direction: Tensor,
    *,
    positions: Sequence[int],
    strength: float = 1.0,
) -> Iterator[None]:
    """Temporarily replace one J-Lens coordinate with another."""

    def hook(_module, _inputs, output):
        changed = swap_direction_coordinate(
            _first_tensor(output),
            source_direction,
            target_direction,
            positions=positions,
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
