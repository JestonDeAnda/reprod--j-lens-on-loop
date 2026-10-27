import pytest
import torch
from torch import nn

from loop_jlens.causal import (
    jacobian_token_direction,
    random_control_direction,
    residual_intervention,
    token_rank,
)


def test_jacobian_token_direction_uses_transpose_and_normalizes() -> None:
    jacobian = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    direction = jacobian_token_direction(jacobian, torch.tensor([1.0, 0.0]))
    expected = torch.tensor([1.0, 2.0])
    expected /= torch.linalg.vector_norm(expected)
    assert torch.allclose(direction, expected)


def test_random_control_is_deterministic_and_unit_norm() -> None:
    reference = torch.zeros(8)
    first = random_control_direction(reference, seed=183)
    second = random_control_direction(reference, seed=183)
    assert torch.equal(first, second)
    assert torch.linalg.vector_norm(first).item() == pytest.approx(1.0)


def test_residual_intervention_changes_only_selected_position() -> None:
    block = nn.Identity()
    hidden = torch.tensor([[[3.0, 4.0], [1.0, 2.0]]])
    with residual_intervention(
        block, torch.tensor([1.0, 0.0]), positions=(-1,), mode="add", strength=0.5
    ):
        changed = block(hidden)
    assert torch.equal(changed[:, 0], hidden[:, 0])
    assert changed[0, 1, 0] > hidden[0, 1, 0]
    assert changed[0, 1, 1] == hidden[0, 1, 1]


def test_projection_ablation_removes_direction_component() -> None:
    block = nn.Identity()
    hidden = torch.tensor([[[3.0, 4.0]]])
    with residual_intervention(
        block, torch.tensor([1.0, 0.0]), positions=(-1,), mode="ablate", strength=1.0
    ):
        changed = block(hidden)
    assert torch.allclose(changed, torch.tensor([[[0.0, 4.0]]]))


def test_token_rank() -> None:
    assert token_rank(torch.tensor([1.0, 4.0, 3.0]), 2) == 2
