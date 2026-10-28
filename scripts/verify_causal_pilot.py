#!/usr/bin/env python
"""Aggregate paired causal-pilot records without discarding raw trials."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return [center - margin, center + margin]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.trials.read_text().splitlines() if line]
    baseline = {
        (row["family"], row["prompt_id"]): row
        for row in records
        if row["condition"] == "baseline"
    }
    groups = defaultdict(list)
    for row in records:
        groups[(row["family"], row["condition"])].append(row)
    rows = []
    for (family, condition), trials in sorted(groups.items()):
        paired = [baseline[(family, row["prompt_id"])] for row in trials]
        successes = sum(row["target_in_generation"] for row in trials)
        rows.append(
            {
                "family": family,
                "condition": condition,
                "trials": len(trials),
                "target_generation_rate": successes / len(trials),
                "target_generation_wilson_95": wilson(successes, len(trials)),
                "mean_target_logit_delta_vs_baseline": sum(
                    row["target_logit"] - base["target_logit"]
                    for row, base in zip(trials, paired, strict=True)
                )
                / len(trials),
                "mean_target_control_margin_delta_vs_baseline": sum(
                    row["target_control_margin"] - base["target_control_margin"]
                    for row, base in zip(trials, paired, strict=True)
                )
                / len(trials),
                "mean_target_rank_improvement_vs_baseline": sum(
                    base["target_rank"] - row["target_rank"]
                    for row, base in zip(trials, paired, strict=True)
                )
                / len(trials),
            }
        )
    payload = {
        "protocol_status": "synthetic_protocol_gate",
        "trials": str(args.trials),
        "records": len(records),
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
