#!/usr/bin/env python
"""Distance-controlled transport checks for fitted looped Jacobian lenses.

Reads artefact directories produced by ``fit_family.py`` / ``dlc_paper_fit.sh`` and
reports matrix-level transport norms at matched source-to-target offsets. This
is the structural half of the paper's first milestone; causal families are
separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jlens import JacobianLens

from loop_jlens.analysis import estimate_operator_norm, frobenius_over_sqrt_d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("lens_dir", type=Path, help="directory with *.lens.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--offsets",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 48],
        help="source-to-target distances to report",
    )
    parser.add_argument(
        "--operator-method",
        choices=("power", "exact"),
        default="power",
        help="spectral/operator 2-norm implementation",
    )
    parser.add_argument("--operator-iterations", type=int, default=100)
    parser.add_argument("--operator-tolerance", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def lens_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.lens.pt"))


def report_lens(
    path: Path,
    offsets: list[int],
    *,
    operator_method: str = "power",
    operator_iterations: int = 100,
    operator_tolerance: float = 1e-5,
    seed: int = 0,
) -> dict:
    lens = JacobianLens.load(str(path))
    # Each fitted artefact targets one virtual layer; the largest source index
    # plus one is that target when the family was built as range(start, target).
    target = max(lens.source_layers) + 1
    rows = []
    for offset in offsets:
        source = target - offset
        if source not in lens.jacobians:
            continue
        J = lens.jacobians[source]
        operator = estimate_operator_norm(
            J,
            method=operator_method,
            max_iterations=operator_iterations,
            tolerance=operator_tolerance,
            seed=seed + source,
        )
        frobenius = frobenius_over_sqrt_d(J)
        rows.append(
            {
                "source": source,
                "target": target,
                "offset": offset,
                "operator_norm": operator.to_dict(),
                "frobenius_over_sqrt_d": frobenius,
                "stable_rank": (
                    frobenius**2 * lens.d_model / operator.value**2
                    if operator.value > 0
                    else 0.0
                ),
            }
        )
    return {
        "file": path.name,
        "n_prompts": lens.n_prompts,
        "d_model": lens.d_model,
        "n_sources": len(lens.source_layers),
        "inferred_target": target,
        "source_min": min(lens.source_layers),
        "source_max": max(lens.source_layers),
        "offsets": rows,
    }


def main() -> None:
    args = parse_args()
    files = lens_files(args.lens_dir)
    if not files:
        raise SystemExit(f"no *.lens.pt under {args.lens_dir}")
    reports = [
        report_lens(
            path,
            args.offsets,
            operator_method=args.operator_method,
            operator_iterations=args.operator_iterations,
            operator_tolerance=args.operator_tolerance,
            seed=args.seed,
        )
        for path in files
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"lens_dir": str(args.lens_dir), "lenses": reports}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
