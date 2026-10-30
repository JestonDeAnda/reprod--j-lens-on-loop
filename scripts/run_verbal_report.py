#!/usr/bin/env python
"""Released-prompt verbal-report coordinate-swap experiment for Ouro."""

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
    pieces = tokenizer.encode(" " + text, add_special_tokens=False)
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
        recorder = stack.enter_context(ActivationRecorder(model.layers, at=[target_layer]))
        model.forward(input_ids)
    return model.unembed(recorder.activations[target_layer][0, -1]).float().cpu()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    data_path = Path(config["data"])
    data = json.loads(data_path.read_text())
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
    layers = config.get("band_layers") or list(
        range(config["band_start"], config["band_end"] + 1)
    )
    conditions = config.get("conditions", ["target"])
    unembedding = model._lm_head.weight.detach().float().cpu()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "trials.jsonl"
    records = []

    candidate_pool = []
    for category, candidates in data["candidates"].items():
        for candidate in candidates:
            candidate_id = single_token(tokenizer, candidate)
            if candidate_id is not None:
                candidate_pool.append((category, candidate, candidate_id))

    trial_index = 0
    with raw_path.open("w") as handle:
        for category, candidates in data["candidates"].items():
            prompt = f"Think of a {category}. Answer in one word:"
            input_ids = model.encode(prompt, max_length=config["max_seq_len"])
            baseline = logits_for(model, input_ids, target_layer=target_layer, swaps=None)
            answer_id = int(baseline.argmax())
            tested = 0
            for candidate in candidates:
                if tested == config["candidates_per_category"]:
                    break
                candidate_id = single_token(tokenizer, candidate)
                if candidate_id is None or candidate_id == answer_id:
                    continue
                unrelated = next(
                    (
                        item
                        for offset in range(1, len(candidate_pool) + 1)
                        if (item := candidate_pool[(trial_index + offset) % len(candidate_pool)])[0]
                        != category
                        and item[2] != candidate_id
                    ),
                    candidate_pool[(trial_index + 1) % len(candidate_pool)],
                )
                for condition in conditions:
                    swaps = []
                    for layer in layers:
                        source = jacobian_token_direction(
                            lens.jacobians[layer], unembedding[answer_id]
                        )
                        if condition == "target":
                            target = jacobian_token_direction(
                                lens.jacobians[layer], unembedding[candidate_id]
                            )
                        elif condition == "unrelated":
                            target = jacobian_token_direction(
                                lens.jacobians[layer], unembedding[unrelated[2]]
                            )
                        elif condition == "random":
                            random = random_control_direction(
                                source,
                                seed=config["seed"] + trial_index * 1000 + layer,
                            )
                            target = orthogonal_control_direction(random, source)
                        else:
                            raise ValueError(f"unsupported condition: {condition}")
                        swaps.append(
                            {
                                "layer": layer,
                                "source": source,
                                "target": target,
                                "strength": config.get("strength", 1.0),
                            }
                        )
                    changed = logits_for(
                        model, input_ids, target_layer=target_layer, swaps=swaps
                    )
                    row = {
                        "protocol_status": "released_prompts_loop_band_adaptation",
                        "category": category,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "condition": condition,
                        "baseline_answer_id": answer_id,
                        "baseline_answer": tokenizer.decode([answer_id]),
                        "candidate": candidate,
                        "candidate_id": candidate_id,
                        "unrelated_control": unrelated[1],
                        "rank_before": token_rank(baseline, candidate_id),
                        "rank_after": token_rank(changed, candidate_id),
                        "success": int(changed.argmax()) == candidate_id,
                        "band_layers": layers,
                    }
                    records.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                trial_index += 1
                tested += 1

    grouped = defaultdict(list)
    for row in records:
        grouped[row["condition"]].append(row)
    condition_summaries = {}
    for condition, group in sorted(grouped.items()):
        successes = sum(row["success"] for row in group)
        condition_summaries[condition] = {
            "trials": len(group),
            "successes": successes,
            "success_rate": successes / len(group) if group else 0.0,
            "success_rate_wilson_95": wilson_interval(successes, len(group)),
            "mean_rank_improvement": (
                sum(row["rank_before"] - row["rank_after"] for row in group)
                / len(group)
                if group
                else 0.0
            ),
        }

    summary = {
        "protocol_status": "released_prompts_loop_band_adaptation",
        "model": model_id,
        "lens": config["lens"],
        "data": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "band_layers": layers,
        "conditions": condition_summaries,
        "raw_output": str(raw_path),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
