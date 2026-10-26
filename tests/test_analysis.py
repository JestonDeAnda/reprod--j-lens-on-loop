import pytest
import torch

from loop_jlens.analysis import (
    estimate_linear_operator_norm,
    estimate_operator_norm,
    frobenius_over_sqrt_d,
    matrix_cosine,
    readout_metrics,
    singular_subspace_overlap,
)


def test_operator_norm_exact_and_power() -> None:
    matrix = torch.diag(torch.tensor([5.0, 2.0, 0.5]))
    exact = estimate_operator_norm(matrix, method="exact")
    power = estimate_operator_norm(matrix, tolerance=1e-7, seed=4)

    assert exact.value == pytest.approx(5.0)
    assert power.value == pytest.approx(5.0, rel=1e-5)
    assert power.converged


def test_implicit_composed_error_operator_norm() -> None:
    direct = torch.diag(torch.tensor([6.0, 3.0]))
    first = torch.diag(torch.tensor([2.0, 1.0]))
    second = torch.diag(torch.tensor([3.0, 2.0]))
    estimate = estimate_linear_operator_norm(
        lambda vector: direct @ vector - second @ (first @ vector),
        lambda vector: direct.T @ vector - first.T @ (second.T @ vector),
        2,
        tolerance=1e-7,
        seed=3,
    )

    assert estimate.value == pytest.approx(1.0, rel=1e-5)


def test_frobenius_and_cosine() -> None:
    identity = torch.eye(4)
    assert frobenius_over_sqrt_d(identity) == pytest.approx(1.0)
    assert matrix_cosine(identity, 2 * identity) == pytest.approx(1.0)
    assert matrix_cosine(identity, -identity) == pytest.approx(-1.0)


def test_singular_subspace_overlap_for_same_matrix() -> None:
    torch.manual_seed(0)
    matrix = torch.randn(12, 12)
    overlap = singular_subspace_overlap(matrix, matrix, rank=3)

    assert overlap["left_overlap"] == pytest.approx(1.0, abs=2e-4)
    assert overlap["right_overlap"] == pytest.approx(1.0, abs=2e-4)


def test_readout_metrics_identical_logits() -> None:
    logits = torch.tensor([[1.0, 4.0, 2.0], [5.0, 1.0, -1.0]])
    metrics = readout_metrics(logits, logits)

    assert metrics["positions"] == 2
    assert metrics["top1_agreement"] == 1.0
    assert metrics["target_top1_mean_rank"] == 1.0
    assert metrics["target_to_candidate_kl"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["logit_cosine"] == pytest.approx(1.0)


def test_readout_metrics_reports_rank_and_kl() -> None:
    target = torch.tensor([[0.0, 3.0, 1.0]])
    candidate = torch.tensor([[4.0, 0.0, 2.0]])
    metrics = readout_metrics(candidate, target)

    assert metrics["top1_agreement"] == 0.0
    assert metrics["target_top1_mean_rank"] == 3.0
    assert metrics["target_to_candidate_kl"] > 0
