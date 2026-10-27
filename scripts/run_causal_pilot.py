#!/usr/bin/env python
"""Run paired, synthetic protocol-gate interventions with a fitted J-Lens."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
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


def expand_cases(config: dict) -> list[dict]:
    rows = []
    for family in config["families"]:
        for template_index, template in enumerate(family["templates"]):
            for case_index, case in enumerate(family["cases"]):
                rows.append(
                    {
                        **case,
                        "family": family["name"],
                        "source_offset": family["source_offset"],
                        "prompt_id": f"{family['name']}-{template_index:02d}-{case_index:02d}",
                        "prompt": template.format(**case),
                    }
                )
    return rows


def one_token_id(tokenizer, text: str) -> tuple[int, list[int]]:
    ids = tokenizer.encode(" " + text.strip(), add_special_tokens=False)
    if not ids:
        raise ValueError(f"tokenizer produced no ids for {text!r}")
    return ids[-1], ids


@torch.no_grad()
def forward_logits(model, input_ids, *, target: int, intervention):
    context = (
        nullcontext()
        if intervention is None
        else residual_intervention(
            model.layers[intervention["source"]],
            intervention["direction"],
            positions=(-1,),
            mode=intervention["mode"],
            strength=intervention["strength"],
        )
    )
    with context, ActivationRecorder(model.layers, at=[target]) as recorder:
        model.forward(input_ids)
    return model.unembed(recorder.activations[target][0, -1]).float().cpu()


@torch.no_grad()
def run_condition(
    model,
    tokenizer,
    input_ids,
    *,
    target_layer: int,
    intervention,
    generate_tokens: int,
) -> tuple[torch.Tensor, list[int], str]:
    current = input_ids
    first_logits = None
    generated = []
    for _ in range(generate_tokens):
        logits = forward_logits(
            model,
            current,
            target=target_layer,
            intervention=intervention,
        )
        if first_logits is None:
            first_logits = logits
        token = int(logits.argmax().item())
        generated.append(token)
        current = torch.cat(
            [current, torch.tensor([[token]], device=current.device)],
            dim=1,
        )
    assert first_logits is not None
    return first_logits, generated, tokenizer.decode(generated)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    torch.manual_seed(config["seed"])
    configure_exact_forward()
    model_args = SimpleNamespace(
        architecture=config["architecture"],
        model=config["model"],
        device_map=args.device_map,
        recurrences=config.get("recurrences", 16),
        seed=config["seed"],
    )
    model_id, tokenizer, model = load_model(model_args)
    lens = JacobianLens.load(config["lens"])
    target_layer = max(lens.source_layers) + 1
    cases = expand_cases(config)
    conditions = config["conditions"]

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "trials.jsonl"
    with raw_path.open("w") as handle:
        for case_index, case in enumerate(cases):
            source = target_layer - case["source_offset"]
            if source not in lens.jacobians:
                raise ValueError(f"lens has no source layer {source}")
            target_id, target_pieces = one_token_id(tokenizer, case["target"])
            control_id, control_pieces = one_token_id(tokenizer, case["control"])
            unembedding = model._lm_head.weight.detach().float().cpu()
            jacobian = lens.jacobians[source]
            target_direction = jacobian_token_direction(
                jacobian, unembedding[target_id]
            )
            unrelated_direction = jacobian_token_direction(
                jacobian, unembedding[control_id]
            )
            random_direction = random_control_direction(
                target_direction, seed=config["seed"] + case_index
            )
            directions = {
                "jacobian": target_direction,
                "unrelated": unrelated_direction,
                "random": random_direction,
            }
            input_ids = model.encode(case["prompt"], max_length=config["max_seq_len"])
            for condition in conditions:
                direction_name = condition.get("direction")
                intervention = None
                if direction_name is not None:
                    intervention = {
                        "source": source,
                        "direction": directions[direction_name],
                        "mode": condition["mode"],
                        "strength": condition["strength"],
                    }
                logits, generated_ids, generated_text = run_condition(
                    model,
                    tokenizer,
                    input_ids,
                    target_layer=target_layer,
                    intervention=intervention,
                    generate_tokens=config["generate_tokens"],
                )
                record = {
                    "protocol_status": "synthetic_protocol_gate",
                    "prompt_id": case["prompt_id"],
                    "family": case["family"],
                    "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
                    "condition": condition["name"],
                    "model": model_id,
                    "lens": config["lens"],
                    "lens_prompts": lens.n_prompts,
                    "source_layer": source,
                    "target_layer": target_layer,
                    "position": -1,
                    "direction_token": case["target"],
                    "direction_token_ids": target_pieces,
                    "control_token": case["control"],
                    "control_token_ids": control_pieces,
                    "strength": condition.get("strength", 0.0),
                    "seed": config["seed"] + case_index,
                    "target_logit": float(logits[target_id]),
                    "control_logit": float(logits[control_id]),
                    "target_rank": token_rank(logits, target_id),
                    "control_rank": token_rank(logits, control_id),
                    "generated_ids": generated_ids,
                    "generated_text": generated_text,
                    "target_in_generation": target_id in generated_ids,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

    manifest = {
        "protocol_status": "synthetic_protocol_gate",
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "model": model_id,
        "lens": config["lens"],
        "families": sorted({case["family"] for case in cases}),
        "prompts_per_family": {
            name: sum(case["family"] == name for case in cases)
            for name in sorted({case["family"] for case in cases})
        },
        "conditions": [condition["name"] for condition in conditions],
        "raw_output": str(raw_path),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
