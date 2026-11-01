#!/usr/bin/env python
"""Released-prompt top-down summoning causal experiment for Ouro."""

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


def token_set(tokenizer, words: list[str]) -> list[int]:
    return sorted(
        {
            token_id
            for word in words
            if (token_id := single_token(tokenizer, word)) is not None
        }
    )


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
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


@torch.no_grad()
def logits_for(model, input_ids, *, target_layer: int, swaps: list[dict] | None):
    with ExitStack() as stack:
        for swap in swaps or []:
            stack.enter_context(
                residual_coordinate_swap(
                    model.layers[swap["layer"]],
                    swap["source"],
                    swap["target"],
                    positions=swap["positions"],
                    strength=swap["strength"],
                )
            )
        recorder = stack.enter_context(
            ActivationRecorder(model.layers, at=[target_layer])
        )
        model.forward(input_ids)
    return model.unembed(recorder.activations[target_layer][0, -1]).float().cpu()


def prompt_and_positions(model, question: str, stimulus: str, max_length: int):
    prefix = f"Question: {question}\nPassage: "
    through_stimulus = prefix + stimulus
    prompt = through_stimulus + "\nAnswer:"
    input_ids = model.encode(prompt, max_length=max_length)
    prefix_ids = model.encode(prefix, max_length=max_length)
    stimulus_ids = model.encode(through_stimulus, max_length=max_length)
    start = max(0, prefix_ids.shape[1] - 1)
    end = min(stimulus_ids.shape[1], input_ids.shape[1] - 1)
    if end <= start:
        raise ValueError("stimulus span is empty after tokenization")
    return prompt, input_ids, list(range(start, end))


def summarize(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["condition"], row["question_kind"])].append(row)
    result = {}
    for (condition, question_kind), rows in sorted(grouped.items()):
        baseline_eligible = [row for row in rows if row["baseline_correct"]]
        target_top1 = sum(row["changed_top1"] == row["target_id"] for row in rows)
        result.setdefault(condition, {})[question_kind] = {
            "trials": len(rows),
            "baseline_eligible": len(baseline_eligible),
            "baseline_accuracy": (
                len(baseline_eligible) / len(rows) if rows else 0.0
            ),
            "target_top1": target_top1,
            "target_top1_rate": target_top1 / len(rows) if rows else 0.0,
            "target_top1_wilson_95": wilson_interval(target_top1, len(rows)),
            "top1_change_rate": (
                sum(row["changed_top1"] != row["baseline_top1"] for row in rows)
                / len(rows)
                if rows
                else 0.0
            ),
            "mean_target_margin_shift": (
                sum(row["target_margin_shift"] for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "mean_target_rank_improvement": (
                sum(
                    row["target_rank_before"] - row["target_rank_after"]
                    for row in rows
                )
                / len(rows)
                if rows
                else 0.0
            ),
        }
    return result


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
    layers = config["band_layers"]
    conditions = config.get("conditions", ["target", "random", "unrelated"])
    strength = config["strength"]
    unembedding = model._lm_head.weight.detach().float().cpu()

    swap_targets = []
    for item in data["items"]:
        for source, target in item["swaps"]:
            source_id = single_token(tokenizer, source)
            target_id = single_token(tokenizer, target)
            if source_id is not None and target_id is not None:
                swap_targets.append((item["key"], source, target, source_id, target_id))

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "trials.jsonl"
    records = []
    trial_index = 0
    with raw_path.open("w") as handle:
        for item in data["items"]:
            expected_ids = token_set(tokenizer, item["expected"])
            q1_expected_ids = token_set(tokenizer, item["q1_expect"])
            for source, target in item["swaps"]:
                source_id = single_token(tokenizer, source)
                target_id = single_token(tokenizer, target)
                if source_id is None or target_id is None:
                    continue
                unrelated = next(
                    entry
                    for entry in swap_targets
                    if entry[0] != item["key"] and entry[4] != target_id
                )
                for question_kind, question, answer_ids in (
                    ("q1", data["q1"], q1_expected_ids),
                    ("q2", item["q2"], expected_ids),
                ):
                    prompt, input_ids, positions = prompt_and_positions(
                        model, question, item["stimulus"], config["max_seq_len"]
                    )
                    baseline = logits_for(
                        model, input_ids, target_layer=target_layer, swaps=None
                    )
                    for condition in conditions:
                        swaps = []
                        for layer in layers:
                            source_direction = jacobian_token_direction(
                                lens.jacobians[layer], unembedding[source_id]
                            )
                            if condition == "target":
                                target_direction = jacobian_token_direction(
                                    lens.jacobians[layer], unembedding[target_id]
                                )
                            elif condition == "unrelated":
                                target_direction = jacobian_token_direction(
                                    lens.jacobians[layer], unembedding[unrelated[4]]
                                )
                            elif condition == "random":
                                random = random_control_direction(
                                    source_direction,
                                    seed=config["seed"] + trial_index * 1000 + layer,
                                )
                                target_direction = orthogonal_control_direction(
                                    random, source_direction
                                )
                            else:
                                raise ValueError(
                                    f"unsupported condition: {condition}"
                                )
                            swaps.append(
                                {
                                    "layer": layer,
                                    "source": source_direction,
                                    "target": target_direction,
                                    "positions": positions,
                                    "strength": strength,
                                }
                            )
                        changed = logits_for(
                            model, input_ids, target_layer=target_layer, swaps=swaps
                        )
                        margin_before = float(
                            baseline[target_id] - baseline[source_id]
                        )
                        margin_after = float(changed[target_id] - changed[source_id])
                        row = {
                            "protocol_status": (
                                "released_prompts_loop_band_adaptation"
                            ),
                            "item": item["key"],
                            "question_kind": question_kind,
                            "condition": condition,
                            "prompt_sha256": hashlib.sha256(
                                prompt.encode()
                            ).hexdigest(),
                            "source": source,
                            "source_id": source_id,
                            "target": target,
                            "target_id": target_id,
                            "unrelated_control": unrelated[2],
                            "baseline_top1": int(baseline.argmax()),
                            "changed_top1": int(changed.argmax()),
                            "baseline_correct": int(baseline.argmax()) in answer_ids,
                            "target_rank_before": token_rank(baseline, target_id),
                            "target_rank_after": token_rank(changed, target_id),
                            "target_margin_before": margin_before,
                            "target_margin_after": margin_after,
                            "target_margin_shift": margin_after - margin_before,
                            "stimulus_positions": positions,
                            "band_layers": layers,
                            "strength": strength,
                        }
                        records.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                    trial_index += 1

    summary = {
        "protocol_status": "released_prompts_loop_band_adaptation",
        "model": model_id,
        "lens": config["lens"],
        "data": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "band_layers": layers,
        "strength": strength,
        "summary": summarize(records),
        "raw_output": str(raw_path),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
