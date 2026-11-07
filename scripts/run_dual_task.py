#!/usr/bin/env python
"""Released-prompt dual-task readout for Ouro.

The model copies one carrier sentence while holding one or two covert tasks.
A task is reachable when any of its target tokens has J-Lens rank at or below
the cutoff somewhere in the band, over the assistant span. Interference is
single-task reachability minus dual-task reachability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from fit_family import configure_exact_forward, load_model
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder

NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def token_ids(tokenizer, words: list[str], *, allow_bare: bool) -> list[int]:
    found = set()
    for word in words:
        candidates = [" " + word.strip()]
        if allow_bare:
            candidates.append(word.strip())
        for candidate in candidates:
            pieces = tokenizer.encode(candidate, add_special_tokens=False)
            if len(pieces) == 1:
                found.add(pieces[0])
    return sorted(found)


def as_ids(encoded) -> list[int]:
    if hasattr(encoded, "input_ids"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        return encoded[0]
    return list(encoded)


def render_copy(tokenizer, user: str, assistant: str) -> tuple[list[int], list[int]] | None:
    header = as_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full = as_ids(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    carrier = tokenizer.encode(assistant, add_special_tokens=False)
    if full[: len(header)] != header:
        return None
    if full[len(header) : len(header) + len(carrier)] != carrier:
        return None
    start = max(0, len(header) - 1)
    end = len(header) + len(carrier)
    return full, list(range(start, end))


def best_rank(logits: torch.Tensor, token_ids_for_task: list[int]) -> int:
    scores = logits[:, token_ids_for_task]
    ranks = 1 + (logits.unsqueeze(1) > scores.unsqueeze(2)).sum(dim=2)
    return int(ranks.min())


def instruction(config: dict, condition: str, **fields: str) -> str:
    return config["instructions"][condition].format(**fields)


def rate(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return sum(flags) / len(flags)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    data = json.loads(Path(config["data"]).read_text())
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
    layers = [layer for layer in config["band_layers"] if layer in lens.jacobians]
    if not layers:
        raise RuntimeError("none of the requested layers have a fitted Jacobian")
    by_key = {item["key"]: item for item in data["pairs"]}
    carrier = data["carrier_sentence"]
    cutoff = config["rank_cutoff"]

    concept_ids = {}
    math_ids = {}
    for item in data["pairs"]:
        concept_ids[item["key"]] = token_ids(tokenizer, item["concept_words"], allow_bare=False)
        answer = item["base"] ** item["exp"] - item["sub"]
        math_ids[item["key"]] = token_ids(
            tokenizer,
            [str(answer), NUMBER_WORDS[answer]],
            allow_bare=True,
        )
    missing = {
        "concept": [key for key, ids in concept_ids.items() if not ids],
        "math": [key for key, ids in math_ids.items() if not ids],
    }
    if missing["concept"] or missing["math"]:
        raise RuntimeError(f"tasks without a single-token target: {missing}")

    trials = []
    for item in data["pairs"]:
        expr = f"{item['base']}^{item['exp']} - {item['sub']}"
        fields = {"carrier": carrier, "concept": item["concept"], "expr": expr}
        for condition in data["concept_math_conditions"]:
            tasks = []
            if condition != "math":
                tasks.append(("concept", item["key"], concept_ids[item["key"]]))
            if condition != "concept":
                tasks.append(("math", item["key"], math_ids[item["key"]]))
            trials.append(
                {
                    "arm": "concept_math",
                    "condition": condition,
                    "key": item["key"],
                    "user": instruction(config, condition, **fields),
                    "tasks": tasks,
                    "single": condition in {"concept", "math"},
                }
            )
    for left_key, right_key in data["concept_pairs"]:
        left = by_key[left_key]
        right = by_key[right_key]
        fields = {
            "carrier": carrier,
            "concept_a": left["concept"],
            "concept_b": right["concept"],
        }
        for condition in data["concept_concept_conditions"]:
            tasks = []
            if condition != "b_only":
                tasks.append(("concept_a", left_key, concept_ids[left_key]))
            if condition != "a_only":
                tasks.append(("concept_b", right_key, concept_ids[right_key]))
            trials.append(
                {
                    "arm": "concept_concept",
                    "condition": condition,
                    "key": f"{left_key}+{right_key}",
                    "user": instruction(config, condition, **fields),
                    "tasks": tasks,
                    "single": condition in {"a_only", "b_only"},
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "trials.jsonl"
    buckets: dict[tuple[str, str], list[bool]] = {}
    completed = 0
    with raw_path.open("w") as handle:
        for trial in trials:
            rendered = render_copy(tokenizer, trial["user"], carrier)
            if rendered is None:
                raise RuntimeError(
                    f"carrier alignment failed for {trial['arm']} {trial['condition']} {trial['key']}"
                )
            input_list, positions = rendered
            input_ids = torch.tensor([input_list], device=model.input_device)
            if input_ids.shape[1] > config["max_seq_len"]:
                continue
            positions = [position for position in positions if position < input_ids.shape[1]]
            with ActivationRecorder(model.layers, at=layers) as recorder:
                model.forward(input_ids)
            task_rows = []
            for role, key, ids in trial["tasks"]:
                lowest = 10**9
                for layer in layers:
                    residual = recorder.activations[layer][0, positions].float()
                    logits = model.unembed(lens.transport(residual, layer)).float().cpu()
                    lowest = min(lowest, best_rank(logits, ids))
                reachable = lowest <= cutoff
                side = "single" if trial["single"] else "dual"
                buckets.setdefault((trial["arm"], role, side), []).append(reachable)
                task_rows.append(
                    {"role": role, "key": key, "min_rank": lowest, "reachable": reachable}
                )
            handle.write(
                json.dumps(
                    {
                        "protocol_status": "released_prompts_loop_band_adaptation",
                        "arm": trial["arm"],
                        "condition": trial["condition"],
                        "key": trial["key"],
                        "single": trial["single"],
                        "positions": len(positions),
                        "tasks": task_rows,
                        "prompt_sha256": hashlib.sha256(trial["user"].encode()).hexdigest(),
                    }
                )
                + "\n"
            )
            completed += 1

    if completed == 0:
        raise RuntimeError("no dual-task trials were aligned")

    def interference(arm: str, role: str) -> dict:
        single = buckets.get((arm, role, "single"), [])
        dual = buckets.get((arm, role, "dual"), [])
        single_rate = rate(single)
        dual_rate = rate(dual)
        return {
            "single_reachability": single_rate,
            "dual_reachability": dual_rate,
            "interference": None
            if single_rate is None or dual_rate is None
            else single_rate - dual_rate,
            "n_single": len(single),
            "n_dual": len(dual),
        }

    summary = {
        "protocol_status": "released_prompts_loop_band_adaptation",
        "prompt_note": (
            "Single-task wording follows jlens/examples.py. "
            "Dual-task order follows the released condition names; "
            "the conjunction is reconstructed."
        ),
        "model": model_id,
        "lens": config["lens"],
        "data": config["data"],
        "rank_cutoff": cutoff,
        "trials_requested": len(trials),
        "trials_completed": completed,
        "band_layers": layers,
        "concept_math": {
            "concept": interference("concept_math", "concept"),
            "math": interference("concept_math", "math"),
        },
        "concept_concept": {
            "concept_a": interference("concept_concept", "concept_a"),
            "concept_b": interference("concept_concept", "concept_b"),
        },
        "raw_output": str(raw_path),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"trials": completed, "requested": len(trials)}))
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
