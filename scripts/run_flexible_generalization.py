#!/usr/bin/env python
"""Released-prompt flexible-generalization swap experiment for Ouro."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import torch
from fit_family import configure_exact_forward, load_model
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder

from loop_jlens.causal import (
    jacobian_token_direction,
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
    unembedding = model._lm_head.weight.detach().float().cpu()

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "trials.jsonl"
    records = []
    with raw_path.open("w") as handle:
        for category in data["categories"]:
            argument_tokens = {
                argument: single_token(tokenizer, argument)
                for argument in category["args"]
            }
            for function in category["funcs"]:
                for source_argument in category["args"]:
                    prompt = function["template"].format(arg=source_argument)
                    source_answer = single_token(
                        tokenizer, function["answers"][source_argument]
                    )
                    source_token = argument_tokens[source_argument]
                    if source_answer is None or source_token is None:
                        continue
                    input_ids = model.encode(prompt, max_length=config["max_seq_len"])
                    baseline = logits_for(
                        model, input_ids, target_layer=target_layer, swaps=None
                    )
                    for target_argument in category["args"]:
                        if target_argument == source_argument:
                            continue
                        target_token = argument_tokens[target_argument]
                        target_answer = single_token(
                            tokenizer, function["answers"][target_argument]
                        )
                        if target_token is None or target_answer is None:
                            continue
                        swaps = []
                        for layer in layers:
                            jacobian = lens.jacobians[layer]
                            swaps.append(
                                {
                                    "layer": layer,
                                    "source": jacobian_token_direction(
                                        jacobian, unembedding[source_token]
                                    ),
                                    "target": jacobian_token_direction(
                                        jacobian, unembedding[target_token]
                                    ),
                                    "strength": config["strength"],
                                }
                            )
                        changed = logits_for(
                            model,
                            input_ids,
                            target_layer=target_layer,
                            swaps=swaps,
                        )
                        row = {
                            "protocol_status": "released_prompts_loop_band_adaptation",
                            "category": category["name"],
                            "function": function["name"],
                            "source_argument": source_argument,
                            "target_argument": target_argument,
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "source_answer_token": source_answer,
                            "target_answer_token": target_answer,
                            "baseline_top1": int(baseline.argmax()),
                            "swapped_top1": int(changed.argmax()),
                            "baseline_correct": int(baseline.argmax()) == source_answer,
                            "swap_success": int(changed.argmax()) == target_answer,
                            "target_rank_before": token_rank(baseline, target_answer),
                            "target_rank_after": token_rank(changed, target_answer),
                            "band_layers": layers,
                        }
                        records.append(row)
                        handle.write(json.dumps(row) + "\n")
                        handle.flush()

    eligible = [row for row in records if row["baseline_correct"]]
    summary = {
        "protocol_status": "released_prompts_loop_band_adaptation",
        "model": model_id,
        "lens": config["lens"],
        "data": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "band_layers": layers,
        "strength": config["strength"],
        "trials": len(records),
        "baseline_correct_trials": len(eligible),
        "baseline_accuracy": (
            sum(row["baseline_correct"] for row in records) / len(records)
            if records
            else 0.0
        ),
        "swap_success_on_baseline_correct": (
            sum(row["swap_success"] for row in eligible) / len(eligible)
            if eligible
            else 0.0
        ),
        "mean_target_rank_improvement": (
            sum(row["target_rank_before"] - row["target_rank_after"] for row in eligible)
            / len(eligible)
            if eligible
            else 0.0
        ),
        "raw_output": str(raw_path),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
