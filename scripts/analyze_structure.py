#!/usr/bin/env python
"""Shared-weight symmetry and mean-Jacobian factorisation analyses."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from jlens import JacobianLens

from loop_jlens.analysis import (
    estimate_linear_operator_norm,
    estimate_operator_norm,
    hutchinson_comparison,
    matrix_cosine,
    singular_subspace_overlap,
)


def matrix_vector(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return matrix @ vector


def composed_vector(
    first: torch.Tensor,
    second: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    return second @ (first @ vector)


def composition_error_vector(
    direct: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    return direct @ vector - second @ (first @ vector)


def composition_error_transpose_vector(
    direct: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    return direct.T @ vector - first.T @ (second.T @ vector)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("lens_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("symmetry", "factorisation"), required=True)
    parser.add_argument("--period", type=int, required=True)
    parser.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 48])
    parser.add_argument("--subspace-rank", type=int, default=32)
    parser.add_argument("--operator-iterations", type=int, default=100)
    parser.add_argument("--operator-tolerance", type=float, default=1e-5)
    parser.add_argument("--hutchinson-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_lenses(directory: Path) -> dict[int, tuple[Path, JacobianLens]]:
    lenses: dict[int, tuple[Path, JacobianLens]] = {}
    for path in sorted(directory.glob("*.lens.pt")):
        lens = JacobianLens.load(str(path))
        target = max(lens.source_layers) + 1
        if target in lenses:
            raise ValueError(f"duplicate inferred target {target}: {path} and {lenses[target][0]}")
        lenses[target] = (path, lens)
    if not lenses:
        raise ValueError(f"no *.lens.pt under {directory}")
    return lenses


def symmetry_rows(
    lenses: dict[int, tuple[Path, JacobianLens]],
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    targets = sorted(lenses)
    for left_index, left_target in enumerate(targets):
        for right_target in targets[left_index + 1 :]:
            if left_target % args.period != right_target % args.period:
                continue
            left_path, left_lens = lenses[left_target]
            right_path, right_lens = lenses[right_target]
            for offset in args.offsets:
                left_source = left_target - offset
                right_source = right_target - offset
                if (
                    left_source not in left_lens.jacobians
                    or right_source not in right_lens.jacobians
                ):
                    continue
                left = left_lens.jacobians[left_source].detach().float()
                right = right_lens.jacobians[right_source].detach().float()
                difference = left - right
                row_seed = args.seed + left_target * 1_000_003 + right_target * 101 + offset
                left_norm = estimate_operator_norm(
                    left,
                    max_iterations=args.operator_iterations,
                    tolerance=args.operator_tolerance,
                    seed=row_seed,
                )
                right_norm = estimate_operator_norm(
                    right,
                    max_iterations=args.operator_iterations,
                    tolerance=args.operator_tolerance,
                    seed=row_seed,
                )
                difference_norm = estimate_operator_norm(
                    difference,
                    max_iterations=args.operator_iterations,
                    tolerance=args.operator_tolerance,
                    seed=row_seed,
                )
                rank = min(args.subspace_rank, min(left.shape))
                rows.append(
                    {
                        "left_file": left_path.name,
                        "right_file": right_path.name,
                        "left_source": left_source,
                        "left_target": left_target,
                        "right_source": right_source,
                        "right_target": right_target,
                        "offset": offset,
                        "operator_norm_left": left_norm.to_dict(),
                        "operator_norm_right": right_norm.to_dict(),
                        "operator_norm_difference": difference_norm.to_dict(),
                        "relative_operator_difference": difference_norm.value
                        / max(left_norm.value, right_norm.value),
                        "frobenius_matrix_cosine": matrix_cosine(left, right),
                        "singular_subspace": singular_subspace_overlap(
                            left,
                            right,
                            rank=rank,
                            seed=row_seed,
                        ),
                    }
                )
    return rows


def factorisation_rows(
    lenses: dict[int, tuple[Path, JacobianLens]],
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    for middle, (middle_path, middle_lens) in sorted(lenses.items()):
        for target, (target_path, target_lens) in sorted(lenses.items()):
            if target <= middle or middle not in target_lens.jacobians:
                continue
            if middle % args.period != target % args.period:
                continue
            second = target_lens.jacobians[middle].detach().float()
            for source in middle_lens.source_layers:
                if source not in target_lens.jacobians:
                    continue
                first = middle_lens.jacobians[source].detach().float()
                direct = target_lens.jacobians[source].detach().float()
                row_seed = args.seed + source * 1_000_003 + middle * 101 + target
                direct_norm = estimate_operator_norm(
                    direct,
                    max_iterations=args.operator_iterations,
                    tolerance=args.operator_tolerance,
                    seed=row_seed,
                )

                direct_matvec = partial(matrix_vector, direct)
                composed_matvec = partial(composed_vector, first, second)
                error_matvec = partial(composition_error_vector, direct, first, second)
                error_rmatvec = partial(
                    composition_error_transpose_vector,
                    direct,
                    first,
                    second,
                )

                error_norm = estimate_linear_operator_norm(
                    error_matvec,
                    error_rmatvec,
                    direct.shape[1],
                    max_iterations=args.operator_iterations,
                    tolerance=args.operator_tolerance,
                    seed=row_seed,
                    device=direct.device,
                )
                rows.append(
                    {
                        "source": source,
                        "middle": middle,
                        "target": target,
                        "source_to_middle_file": middle_path.name,
                        "direct_to_target_file": target_path.name,
                        "operator_norm_direct": direct_norm.to_dict(),
                        "operator_norm_error": error_norm.to_dict(),
                        "relative_operator_error": (
                            error_norm.value / direct_norm.value
                            if direct_norm.value > 0
                            else 0.0
                        ),
                        "hutchinson": hutchinson_comparison(
                            direct_matvec,
                            composed_matvec,
                            direct.shape[1],
                            samples=args.hutchinson_samples,
                            seed=row_seed,
                            device=direct.device,
                        ),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    if args.period < 1:
        raise SystemExit("--period must be positive")
    lenses = load_lenses(args.lens_dir)
    rows = (
        symmetry_rows(lenses, args)
        if args.mode == "symmetry"
        else factorisation_rows(lenses, args)
    )
    payload = {
        "mode": args.mode,
        "lens_dir": str(args.lens_dir),
        "period": args.period,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}))


if __name__ == "__main__":
    main()
