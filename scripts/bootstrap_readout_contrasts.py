#!/usr/bin/env python
"""Bootstrap confidence intervals for completed released-prompt readouts."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=183)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def interval(samples: list[float]) -> dict:
    return {
        "mean": sum(samples) / len(samples),
        "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)],
    }


def paired_delta(rows: list[dict], left: str, right: str, metric: str, samples: int, seed: int):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["unit"]][row["condition"]].append(row[metric])
    units = sorted(unit for unit, values in grouped.items() if left in values and right in values)
    observed = []
    for unit in units:
        observed.append(
            sum(grouped[unit][left]) / len(grouped[unit][left])
            - sum(grouped[unit][right]) / len(grouped[unit][right])
        )
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        chosen = [observed[rng.randrange(len(observed))] for _ in units]
        draws.append(sum(chosen) / len(chosen))
    return {
        "units": len(units),
        "observed_mean": sum(observed) / len(observed) if observed else None,
        "bootstrap": interval(draws) if draws else None,
    }


def main() -> None:
    args = parse_args()
    jobs = {
        "summoning": (
            "readout-summoning-20261102T0138Z",
            lambda row: row["key"],
            "q2",
            "q1",
        ),
        "language": (
            "readout-language-20261102T0138Z",
            lambda row: row["key"],
            "explicit",
            "automatic",
        ),
        "linecount": (
            "readout-linecount-20261102T0138Z",
            lambda row: row["key"],
            "direct",
            "none",
        ),
        "directed-topic": (
            "directed-topic-all-20261102T1455Z",
            lambda row: row["target"],
            "focus",
            "control",
        ),
    }
    results = {}
    for name, (run, unit_of, left, right) in jobs.items():
        path = args.runs / run / "trials.jsonl"
        rows = []
        for row in load_rows(path):
            condition = row.get("group_kind", row.get("condition"))
            if name == "directed-topic" and condition == "suppress":
                continue
            rows.append(
                {
                    "unit": unit_of(row),
                    "condition": condition,
                    "hit_fraction": row["hit_fraction"],
                    "any_hit": float(row["any_hit"]),
                }
            )
        results[name] = {
            "contrast": f"{left}-{right}",
            "position_hit_rate": paired_delta(
                rows, left, right, "hit_fraction", args.samples, args.seed
            ),
            "any_hit_rate": paired_delta(
                rows, left, right, "any_hit", args.samples, args.seed
            ),
            "source": str(path),
        }
        if name == "directed-topic":
            all_rows = [
                {
                    "unit": row["target"],
                    "condition": row["group_kind"],
                    "hit_fraction": row["hit_fraction"],
                    "any_hit": float(row["any_hit"]),
                }
                for row in load_rows(path)
            ]
            results[name]["suppress_minus_control"] = {
                "position_hit_rate": paired_delta(
                    all_rows, "suppress", "control", "hit_fraction", args.samples, args.seed
                ),
                "any_hit_rate": paired_delta(
                    all_rows, "suppress", "control", "any_hit", args.samples, args.seed
                ),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
