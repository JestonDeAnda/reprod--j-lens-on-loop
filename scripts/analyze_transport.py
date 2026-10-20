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
import math
from pathlib import Path

from jlens import JacobianLens


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
    return parser.parse_args()


def lens_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.lens.pt"))


def report_lens(path: Path, offsets: list[int]) -> dict:
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
        norm = J.norm().item() / math.sqrt(lens.d_model)
        rows.append(
            {
                "source": source,
                "target": target,
                "offset": offset,
                "frobenius_over_sqrt_d": norm,
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
    reports = [report_lens(path, args.offsets) for path in files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"lens_dir": str(args.lens_dir), "lenses": reports}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
