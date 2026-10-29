#!/usr/bin/env python
"""Released-prompt verbal-introspection experiment for an Ouro loop band."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import torch
from fit_family import configure_exact_forward, load_model
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder

from loop_jlens.causal import (
    jacobian_token_direction,
    random_control_direction,
    residual_intervention,
    token_rank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def render_trial(tokenizer, data: dict, prefill: str) -> tuple[list[int], list[int]]:
    context = data["intro_prompt"][:2]
    question = data["intro_prompt"][2]
    prefix = tokenizer.apply_chat_template(
        context, tokenize=True, add_generation_prompt=False
    )
    question = tokenizer.apply_chat_template(
        [*context, question], tokenize=True, add_generation_prompt=True
    )
    prefix_ids = prefix["input_ids"] if hasattr(prefix, "keys") else prefix
    question_ids = question["input_ids"] if hasattr(question, "keys") else question
    prefill_ids = tokenizer.encode(prefill, add_special_tokens=False)
    positions = list(range(len(prefix_ids), len(question_ids)))
    return [*question_ids, *prefill_ids], positions


@torch.no_grad()
def trial_logits(
    model,
    input_ids,
    *,
    lens: JacobianLens,
    layers: list[int],
    positions: list[int],
    token_id: int,
    control_token_id: int,
    direction_kind: str,
    seed: int,
    strength: float,
    target_layer: int,
):
    with ExitStack() as stack:
        if strength:
            unembedding = model._lm_head.weight[token_id].detach().float().cpu()
            control_unembedding = (
                model._lm_head.weight[control_token_id].detach().float().cpu()
            )
            for layer in layers:
                target_direction = jacobian_token_direction(
                    lens.jacobians[layer], unembedding
                )
                if direction_kind == "target":
                    direction = target_direction
                elif direction_kind == "unrelated":
                    direction = jacobian_token_direction(
                        lens.jacobians[layer], control_unembedding
                    )
                elif direction_kind == "random":
                    direction = random_control_direction(
                        target_direction, seed=seed + layer
                    )
                else:
                    raise ValueError(f"unsupported direction: {direction_kind}")
                stack.enter_context(
                    residual_intervention(
                        model.layers[layer],
                        direction,
                        positions=positions,
                        mode="add",
                        strength=strength,
                        scale_mode="mean",
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
    missing = [layer for layer in layers if layer not in lens.jacobians]
    if missing:
        raise ValueError(f"lens is missing band layers: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "trials.jsonl"
    aggregates = {}
    with raw_path.open("w") as handle:
        for prefill_name, prefill in data["prefills"].items():
            token_ids, positions = render_trial(tokenizer, data, prefill)
            input_ids = torch.tensor([token_ids], device=model.input_device)
            concepts = data["concepts"]
            conditions = config.get("conditions") or [
                {"name": f"target_{strength}", "direction": "target", "strength": strength}
                for strength in config["strengths"]
            ]
            for concept_index, concept in enumerate(concepts):
                pieces = tokenizer.encode(
                    " " + concept["surface"], add_special_tokens=False
                )
                if len(pieces) != 1:
                    continue
                token_id = pieces[0]
                unrelated = concepts[(concept_index + 1) % len(concepts)]
                control_pieces = tokenizer.encode(
                    " " + unrelated["surface"], add_special_tokens=False
                )
                if len(control_pieces) != 1:
                    continue
                control_token_id = control_pieces[0]
                for condition in conditions:
                    strength = condition["strength"]
                    logits = trial_logits(
                        model,
                        input_ids,
                        lens=lens,
                        layers=layers,
                        positions=positions,
                        token_id=token_id,
                        control_token_id=control_token_id,
                        direction_kind=condition["direction"],
                        seed=config["seed"] + concept_index,
                        strength=strength,
                        target_layer=target_layer,
                    )
                    rank = token_rank(logits, token_id)
                    row = {
                        "protocol_status": "released_prompts_loop_band_adaptation",
                        "prefill": prefill_name,
                        "concept": concept["name"],
                        "surface": concept["surface"],
                        "token_id": token_id,
                        "strength": strength,
                        "condition": condition["name"],
                        "direction": condition["direction"],
                        "control_surface": unrelated["surface"],
                        "rank": rank,
                        "reciprocal_rank": 1.0 / rank,
                        "band_layers": layers,
                        "question_positions": positions,
                    }
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    aggregates.setdefault(
                        (prefill_name, condition["name"], strength), []
                    ).append(
                        1.0 / rank
                    )

    summary = {
        "protocol_status": "released_prompts_loop_band_adaptation",
        "model": model_id,
        "lens": config["lens"],
        "data": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "band_layers": layers,
        "rows": [
            {
                "prefill": prefill,
                "condition": condition,
                "strength": strength,
                "trials": len(values),
                "median_reciprocal_rank": statistics.median(values),
                "mean_reciprocal_rank": statistics.fmean(values),
            }
            for (prefill, condition, strength), values in sorted(aggregates.items())
        ],
        "raw_output": str(raw_path),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
