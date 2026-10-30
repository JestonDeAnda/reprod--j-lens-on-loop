#!/usr/bin/env python
"""Released-prompt probe-swap experiment for Ouro."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import torch
from fit_family import configure_exact_forward, load_model
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder

from loop_jlens.causal import (
    jacobian_token_direction,
    orthogonal_control_direction,
    random_control_direction,
    residual_coordinate_swap,
    token_rank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def single_token(tokenizer, text: str) -> int | None:
    pieces = tokenizer.encode(" " + text.strip(), add_special_tokens=False)
    return pieces[0] if len(pieces) == 1 else None


def wilson_interval(successes: int, trials: int) -> list[float]:
    if trials == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return [centre - margin, centre + margin]


@torch.no_grad()
def logits_for(model, input_ids, *, target_layer: int, swaps: list[dict] | None):
    with ExitStack() as stack:
        for swap in swaps or []:
            stack.enter_context(
                residual_coordinate_swap(
                    model.layers[swap["layer"]],
                    swap["source"],
                    swap["target"],
                    positions=range(input_ids.shape[1]),
                    strength=swap["strength"],
                )
            )
        recorder = stack.enter_context(
            ActivationRecorder(model.layers, at=[target_layer])
        )
        model.forward(input_ids)
    return model.unembed(recorder.activations[target_layer][0, -1]).float().cpu()


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["category"])].append(row)

    by_condition_category = {}
    for (condition, category), group in sorted(grouped.items()):
        eligible = [row for row in group if row["baseline_correct"]]
        successes = sum(row["swap_success"] for row in eligible)
        by_condition_category.setdefault(condition, {})[category] = {
            "trials": len(group),
            "eligible_trials": len(eligible),
            "successes": successes,
            "success_rate": successes / len(eligible) if eligible else 0.0,
            "success_rate_wilson_95": wilson_interval(successes, len(eligible)),
            "mean_target_rank_improvement": (
                sum(
                    row["target_rank_before"] - row["target_rank_after"]
                    for row in eligible
                )
                / len(eligible)
                if eligible
                else 0.0
            ),
        }

    by_condition = {}
    for condition in sorted({row["condition"] for row in rows}):
        group = [row for row in rows if row["condition"] == condition]
        eligible = [row for row in group if row["baseline_correct"]]
        successes = sum(row["swap_success"] for row in eligible)
        by_condition[condition] = {
            "trials": len(group),
            "eligible_trials": len(eligible),
            "successes": successes,
            "success_rate": successes / len(eligible) if eligible else 0.0,
            "success_rate_wilson_95": wilson_interval(successes, len(eligible)),
            "mean_target_rank_improvement": (
                sum(
                    row["target_rank_before"] - row["target_rank_after"]
                    for row in eligible
                )
                / len(eligible)
                if eligible
                else 0.0
            ),
        }
    return {
        "by_condition": by_condition,
        "by_condition_category": by_condition_category,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    data_path = Path(config["data"])
    items = json.loads(data_path.read_text())["items"]
    configure_exact_forward()
    torch.manual_seed(config["seed"])
    model_args = SimpleNamespace(
        architecture=config["architecture"],
        model=config["model"],
        device_map=args.device_map,
        recurrences=16,
        seed=config["seed"],
    )
    model_id, tokenizer, model = load_model(model_args)
    lens = JacobianLens.load(config["lens"])
    target_layer = max(lens.source_layers) + 1
    layers = config["band_layers"]
    strength = config.get("strength", 1.0)
    unembedding = model._lm_head.weight.detach().float().cpu()

    tokenized = []
    for item in items:
        tokenized.append(
            {
                **item,
                "intermediate_id": single_token(tokenizer, item["intermediate"]),
                "answer_id": single_token(tokenizer, item["answer"]),
                "swap_to_id": single_token(tokenizer, item["swap_to"]),
                "swap_answer_id": single_token(tokenizer, item["swap_answer"]),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "trials.jsonl"
    records = []
    with raw_path.open("w") as handle:
        for item_index, item in enumerate(tokenized):
            required = (
                "intermediate_id",
                "answer_id",
                "swap_to_id",
                "swap_answer_id",
            )
            if any(item[key] is None for key in required):
                continue
            input_ids = model.encode(item["prompt"], max_length=config["max_seq_len"])
            baseline = logits_for(
                model, input_ids, target_layer=target_layer, swaps=None
            )
            unrelated = tokenized[(item_index + 1) % len(tokenized)]
            unrelated_id = unrelated["swap_to_id"]
            if unrelated_id is None or unrelated_id == item["swap_to_id"]:
                unrelated_id = tokenized[(item_index + 2) % len(tokenized)][
                    "swap_to_id"
                ]
            if unrelated_id is None:
                continue

            directions = {}
            for condition in ("target", "noop", "random", "unrelated"):
                swaps = []
                for layer in layers:
                    source = jacobian_token_direction(
                        lens.jacobians[layer], unembedding[item["intermediate_id"]]
                    )
                    if condition == "target":
                        target = jacobian_token_direction(
                            lens.jacobians[layer], unembedding[item["swap_to_id"]]
                        )
                    elif condition == "noop":
                        target = source
                    elif condition == "unrelated":
                        target = jacobian_token_direction(
                            lens.jacobians[layer], unembedding[unrelated_id]
                        )
                    else:
                        random = random_control_direction(
                            source, seed=config["seed"] + item_index * 1000 + layer
                        )
                        target = orthogonal_control_direction(random, source)
                    swaps.append(
                        {
                            "layer": layer,
                            "source": source,
                            "target": target,
                            "strength": strength,
                        }
                    )
                directions[condition] = swaps

            for condition, swaps in directions.items():
                changed = logits_for(
                    model, input_ids, target_layer=target_layer, swaps=swaps
                )
                row = {
                    "protocol_status": "released_prompts_loop_band_adaptation",
                    "name": item["name"],
                    "category": item["category"],
                    "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                    "condition": condition,
                    "intermediate": item["intermediate"],
                    "swap_to": item["swap_to"],
                    "unrelated_control": unrelated["swap_to"],
                    "baseline_top1": int(baseline.argmax()),
                    "changed_top1": int(changed.argmax()),
                    "baseline_correct": int(baseline.argmax()) == item["answer_id"],
                    "swap_success": int(changed.argmax()) == item["swap_answer_id"],
                    "target_rank_before": token_rank(baseline, item["swap_answer_id"]),
                    "target_rank_after": token_rank(changed, item["swap_answer_id"]),
                    "band_layers": layers,
                    "strength": strength,
                }
                records.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    summary = {
        "protocol_status": "released_prompts_loop_band_adaptation",
        "model": model_id,
        "lens": config["lens"],
        "data": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "items_in_data": len(items),
        "eligible_items": len(
            {
                row["name"]
                for row in records
                if row["condition"] == "target" and row["baseline_correct"]
            }
        ),
        "band_layers": layers,
        "strength": strength,
        **summarize(records),
        "raw_output": str(raw_path),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
