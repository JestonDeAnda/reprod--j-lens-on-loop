"""Smoke checks for recurrence-specific silent failure modes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel


@dataclass(frozen=True)
class ValidationReport:
    virtual_layers: int
    deterministic: bool
    distinct_firings: bool
    nonzero_gradients: bool
    max_repeat_error: float


def validate_adapter(
    model: LensModel,
    prompt: str,
    *,
    atol: float = 0.0,
    max_seq_len: int = 64,
) -> ValidationReport:
    """Validate routing, determinism, distinct firings, and gradient flow.

    This intentionally uses a scalar loss instead of fitting a full
    ``d_model²`` Jacobian, making it suitable as a preflight check on GPU.
    """

    input_ids = model.encode(prompt, max_length=max_seq_len)
    at = list(range(model.n_layers))

    def record(*, require_grad: bool) -> dict[int, torch.Tensor]:
        context = torch.enable_grad() if require_grad else torch.no_grad()
        with context, ActivationRecorder(
            model.layers,
            at=at,
            start_graph_at=0 if require_grad else None,
        ) as recorder:
            model.forward(input_ids)
            return {index: value for index, value in recorder.activations.items()}

    first = record(require_grad=False)
    second = record(require_grad=False)
    if sorted(first) != at or sorted(second) != at:
        raise AssertionError("not every virtual layer fired exactly once")

    max_repeat_error = max(
        (first[index] - second[index]).abs().max().item() for index in at
    )
    deterministic = max_repeat_error <= atol

    # Identical adjacent states across the entire stack almost always signal
    # that every virtual proxy captured the final firing by mistake.
    distinct_firings = any(
        not torch.equal(first[index], first[index + 1])
        for index in range(model.n_layers - 1)
    )

    with torch.enable_grad(), ActivationRecorder(
        model.layers, at=at, start_graph_at=0
    ) as recorder:
        model.forward(input_ids)
        target = recorder.activations[model.n_layers - 1].float().sum()
        sources = [recorder.activations[index] for index in at[:-1]]
        gradients = torch.autograd.grad(target, sources, allow_unused=True)
    nonzero_gradients = all(
        gradient is not None and torch.count_nonzero(gradient).item() > 0
        for gradient in gradients
    )

    if not deterministic:
        raise AssertionError(
            f"forward is not deterministic (max absolute error {max_repeat_error:g})"
        )
    if not distinct_firings:
        raise AssertionError("virtual firings did not produce distinct activations")
    if not nonzero_gradients:
        raise AssertionError("one or more virtual layers has a zero/absent gradient")

    return ValidationReport(
        virtual_layers=model.n_layers,
        deterministic=deterministic,
        distinct_firings=distinct_firings,
        nonzero_gradients=nonzero_gradients,
        max_repeat_error=max_repeat_error,
    )
