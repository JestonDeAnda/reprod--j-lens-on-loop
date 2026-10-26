"""Numerically controlled matrix metrics for structural J-Lens analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

Matvec = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class OperatorNormEstimate:
    value: float
    method: str
    iterations: int
    converged: bool
    relative_change: float

    def to_dict(self) -> dict[str, float | str | int | bool]:
        return asdict(self)


def _normal_vector(size: int, *, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    vector = torch.randn(size, generator=generator, device=device, dtype=torch.float32)
    return vector / torch.linalg.vector_norm(vector)


def estimate_operator_norm(
    matrix: Tensor,
    *,
    method: str = "power",
    max_iterations: int = 100,
    tolerance: float = 1e-5,
    seed: int = 0,
) -> OperatorNormEstimate:
    """Return the spectral/operator 2-norm with convergence diagnostics."""

    matrix = matrix.detach().float()
    if method == "exact":
        value = torch.linalg.matrix_norm(matrix, ord=2).item()
        return OperatorNormEstimate(value, method, 1, True, 0.0)
    if method != "power":
        raise ValueError(f"unsupported operator-norm method: {method}")
    return estimate_linear_operator_norm(
        lambda vector: matrix @ vector,
        lambda vector: matrix.T @ vector,
        matrix.shape[1],
        method=method,
        max_iterations=max_iterations,
        tolerance=tolerance,
        seed=seed,
        device=matrix.device,
    )


def estimate_linear_operator_norm(
    matvec: Matvec,
    rmatvec: Matvec,
    input_size: int,
    *,
    method: str = "power",
    max_iterations: int = 100,
    tolerance: float = 1e-5,
    seed: int = 0,
    device: torch.device | None = None,
) -> OperatorNormEstimate:
    """Estimate an implicit linear operator's spectral norm by power iteration."""

    if method != "power":
        raise ValueError("implicit operators only support power iteration")
    device = device or torch.device("cpu")
    vector = _normal_vector(input_size, seed=seed, device=device)
    previous = 0.0
    relative_change = float("inf")
    value = 0.0
    converged = False
    completed = 0
    for iteration in range(1, max_iterations + 1):
        left = matvec(vector)
        left_norm = torch.linalg.vector_norm(left)
        if left_norm == 0:
            return OperatorNormEstimate(0.0, method, iteration, True, 0.0)
        left = left / left_norm
        right = rmatvec(left)
        right_norm = torch.linalg.vector_norm(right)
        if right_norm == 0:
            return OperatorNormEstimate(0.0, method, iteration, True, 0.0)
        vector = right / right_norm
        value = torch.linalg.vector_norm(matvec(vector)).item()
        relative_change = abs(value - previous) / max(value, torch.finfo(torch.float32).eps)
        completed = iteration
        if iteration > 1 and relative_change <= tolerance:
            converged = True
            break
        previous = value
    return OperatorNormEstimate(value, method, completed, converged, relative_change)


def frobenius_over_sqrt_d(matrix: Tensor) -> float:
    matrix = matrix.detach().float()
    return (torch.linalg.matrix_norm(matrix, ord="fro") / matrix.shape[1] ** 0.5).item()


def matrix_cosine(left: Tensor, right: Tensor) -> float:
    left = left.detach().float()
    right = right.detach().float()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if denominator == 0:
        return 0.0
    return (torch.sum(left * right) / denominator).item()


def singular_subspace_overlap(
    left: Tensor,
    right: Tensor,
    *,
    rank: int,
    iterations: int = 4,
    seed: int = 0,
) -> dict[str, float | int]:
    """Compare top left/right singular spaces using randomized low-rank SVD."""

    if rank < 1 or rank > min(left.shape):
        raise ValueError(f"rank must be in [1, {min(left.shape)}]")
    q = min(min(left.shape), rank + 8)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        u_left, _, v_left = torch.svd_lowrank(left.detach().float(), q=q, niter=iterations)
        torch.manual_seed(seed)
        u_right, _, v_right = torch.svd_lowrank(right.detach().float(), q=q, niter=iterations)
    u_overlap = (torch.linalg.matrix_norm(u_left[:, :rank].T @ u_right[:, :rank]) ** 2 / rank)
    v_overlap = (torch.linalg.matrix_norm(v_left[:, :rank].T @ v_right[:, :rank]) ** 2 / rank)
    return {
        "rank": rank,
        "left_overlap": u_overlap.item(),
        "right_overlap": v_overlap.item(),
        "mean_overlap": ((u_overlap + v_overlap) / 2).item(),
    }


def hutchinson_comparison(
    left_matvec: Matvec,
    right_matvec: Matvec,
    input_size: int,
    *,
    samples: int = 32,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict[str, float | int]:
    """Estimate Frobenius norms and cosine for two implicit matrices."""

    if samples < 1:
        raise ValueError("samples must be positive")
    device = device or torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    left_squared = 0.0
    right_squared = 0.0
    error_squared = 0.0
    inner_product = 0.0
    for _ in range(samples):
        vector = torch.randint(
            0,
            2,
            (input_size,),
            generator=generator,
            device=device,
            dtype=torch.int8,
        ).float()
        vector.mul_(2).sub_(1)
        left = left_matvec(vector)
        right = right_matvec(vector)
        left_squared += torch.dot(left, left).item()
        right_squared += torch.dot(right, right).item()
        error_squared += torch.dot(left - right, left - right).item()
        inner_product += torch.dot(left, right).item()
    left_squared /= samples
    right_squared /= samples
    error_squared /= samples
    inner_product /= samples
    denominator = (left_squared * right_squared) ** 0.5
    return {
        "samples": samples,
        "left_frobenius": left_squared**0.5,
        "right_frobenius": right_squared**0.5,
        "error_frobenius": error_squared**0.5,
        "matrix_cosine": inner_product / denominator if denominator else 0.0,
    }


def readout_metrics(candidate_logits: Tensor, target_logits: Tensor) -> dict[str, float]:
    """Compare a transported or logit-lens readout with target-layer logits."""

    if candidate_logits.shape != target_logits.shape:
        raise ValueError(
            "candidate and target logits must have the same shape, got "
            f"{tuple(candidate_logits.shape)} and {tuple(target_logits.shape)}"
        )
    if candidate_logits.ndim != 2:
        raise ValueError("readout logits must have shape [n_positions, vocab_size]")
    candidate = candidate_logits.detach().float()
    target = target_logits.detach().float()
    candidate_log_probs = F.log_softmax(candidate, dim=-1)
    target_log_probs = F.log_softmax(target, dim=-1)
    target_probs = target_log_probs.exp()
    target_top1 = target.argmax(dim=-1)
    candidate_top1 = candidate.argmax(dim=-1)
    target_scores = candidate.gather(-1, target_top1[:, None])
    target_ranks = 1 + (candidate > target_scores).sum(dim=-1)
    kl = torch.sum(target_probs * (target_log_probs - candidate_log_probs), dim=-1)
    cosine = F.cosine_similarity(candidate, target, dim=-1)
    return {
        "positions": float(candidate.shape[0]),
        "top1_agreement": (candidate_top1 == target_top1).float().mean().item(),
        "target_top1_mean_rank": target_ranks.float().mean().item(),
        "target_to_candidate_kl": kl.mean().item(),
        "logit_cosine": cosine.mean().item(),
    }
