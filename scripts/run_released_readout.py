#!/usr/bin/env python
"""Released-prompt J-Lens readout experiments for Ouro."""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from fit_family import configure_exact_forward, load_model
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("summoning", "language", "linecount"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def token_ids(tokenizer, words: list[str]) -> list[int]:
    ids = set()
    for word in words:
        pieces = tokenizer.encode(" " + word.strip(), add_special_tokens=False)
        if len(pieces) == 1:
            ids.add(pieces[0])
    return sorted(ids)


def span_after(model, full_prompt: str, prefix: str, max_length: int) -> list[int]:
    full = model.encode(full_prompt, max_length=max_length)
    before = model.encode(prefix, max_length=max_length)
    start = max(0, before.shape[1] - 1)
    return list(range(start, full.shape[1]))


def span_between(
    model, full_prompt: str, prefix: str, through_span: str, max_length: int
) -> list[int]:
    full = model.encode(full_prompt, max_length=max_length)
    before = model.encode(prefix, max_length=max_length)
    through = model.encode(through_span, max_length=max_length)
    start = max(0, before.shape[1] - 1)
    end = min(through.shape[1], full.shape[1])
    return list(range(start, end))


def build_trials(task: str, data: dict, model, tokenizer, max_length: int):
    if task == "summoning":
        for item in data["items"]:
            for condition, question in (("q1", data["q1"]), ("q2", item["q2"])):
                prefix = f"Question: {question}\nPassage: "
                through = prefix + item["stimulus"]
                prompt = through + "\nAnswer:"
                yield {
                    "key": item["key"],
                    "condition": condition,
                    "prompt": prompt,
                    "positions": span_between(
                        model, prompt, prefix, through, max_length
                    ),
                    "targets": token_ids(tokenizer, item["expected"]),
                }
    elif task == "language":
        for passage in data["passages"]:
            for condition, template in (
                ("explicit", data["task"]["explicit_q"]),
                ("automatic", data["task"]["automatic_q"]),
            ):
                prompt = template.format(text=passage["text"])
                passage_end = prompt.index(passage["text"]) + len(passage["text"])
                prefix = prompt[:passage_end]
                yield {
                    "key": passage["key"],
                    "category": passage["category"],
                    "condition": condition,
                    "prompt": prompt,
                    "positions": span_after(model, prompt, prefix, max_length),
                    "targets": token_ids(
                        tokenizer, data["intermediates"][passage["category"]]
                    ),
                }
    else:
        number_words = [
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        ]
        targets = token_ids(
            tokenizer, [str(number) for number in range(10, 100)] + number_words
        )
        for passage in data["passages"]:
            wrapped = textwrap.fill(passage["text"], width=passage["width"])
            conditions = {
                **data["conditions"],
                "continue": {"question": data["explicit_q"], "prefill": ""},
            }
            for condition, spec in conditions.items():
                question = spec["question"]
                prompt = (
                    (question + "\n\n" if question else "")
                    + wrapped
                    + spec["prefill"]
                )
                yield {
                    "key": passage["tag"],
                    "condition": condition,
                    "prompt": prompt,
                    "positions": list(
                        range(model.encode(prompt, max_length=max_length).shape[1])
                    ),
                    "targets": targets,
                    "ground_truth": len(wrapped.splitlines()[0]),
                }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    data_path = Path(config["data"][args.task])
    data = json.loads(data_path.read_text())
    configure_exact_forward()
    model_args = SimpleNamespace(
        architecture=config["architecture"],
        model=config["model"],
        device_map=args.device_map,
        recurrences=16,
        seed=config["seed"],
    )
    model_id, tokenizer, model = load_model(model_args)
    lens = JacobianLens.load(config["lens"])
    layers = config["band_layers"]
    top_k = config["top_k"]
    trials = list(
        build_trials(
            args.task, data, model, tokenizer, config["max_seq_len"]
        )
    )

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "trials.jsonl"
    aggregates = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    with raw_path.open("w") as handle:
        for trial in trials:
            input_ids = model.encode(
                trial["prompt"], max_length=config["max_seq_len"]
            )
            positions = [
                position
                for position in trial["positions"]
                if position < input_ids.shape[1]
            ]
            if not positions or not trial["targets"]:
                continue
            with ActivationRecorder(model.layers, at=layers) as recorder:
                model.forward(input_ids)
            target_tensor = torch.tensor(trial["targets"])
            for layer in layers:
                residual = recorder.activations[layer][0, positions].float()
                logits = model.unembed(lens.transport(residual, layer)).float().cpu()
                top_ids = logits.topk(top_k, dim=-1).indices
                hits = (top_ids[..., None] == target_tensor).any(dim=(-1, -2))
                hit_fraction = float(hits.float().mean())
                any_hit = float(hits.any())
                key = (trial["condition"], layer)
                aggregates[key]["hit_fraction"] += hit_fraction
                aggregates[key]["any_hit"] += any_hit
                counts[key] += 1
                row = {
                    "protocol_status": "released_prompts_loop_band_adaptation",
                    "task": args.task,
                    "key": trial["key"],
                    "condition": trial["condition"],
                    "layer": layer,
                    "positions": len(positions),
                    "target_ids": trial["targets"],
                    "hit_fraction": hit_fraction,
                    "any_hit": bool(any_hit),
                    "prompt_sha256": hashlib.sha256(
                        trial["prompt"].encode()
                    ).hexdigest(),
                }
                if "category" in trial:
                    row["category"] = trial["category"]
                if "ground_truth" in trial:
                    row["ground_truth"] = trial["ground_truth"]
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = []
    for (condition, layer), values in sorted(aggregates.items()):
        count = counts[(condition, layer)]
        rows.append(
            {
                "condition": condition,
                "layer": layer,
                "trials": count,
                "mean_position_hit_rate": values["hit_fraction"] / count,
                "prompt_any_hit_rate": values["any_hit"] / count,
            }
        )
    contrasts = []
    if args.task in {"summoning", "language"}:
        positive, control = (
            ("q2", "q1") if args.task == "summoning" else ("explicit", "automatic")
        )
        indexed = {(row["condition"], row["layer"]): row for row in rows}
        for layer in layers:
            a, b = indexed[(positive, layer)], indexed[(control, layer)]
            contrasts.append(
                {
                    "layer": layer,
                    "contrast": f"{positive}-{control}",
                    "position_hit_rate_delta": (
                        a["mean_position_hit_rate"] - b["mean_position_hit_rate"]
                    ),
                    "any_hit_rate_delta": (
                        a["prompt_any_hit_rate"] - b["prompt_any_hit_rate"]
                    ),
                }
            )
    summary = {
        "protocol_status": "released_prompts_loop_band_adaptation",
        "task": args.task,
        "model": model_id,
        "lens": config["lens"],
        "data": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "top_k": top_k,
        "band_layers": layers,
        "rows": rows,
        "contrasts": contrasts,
        "raw_output": str(raw_path),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"task": args.task, "rows": len(rows)}))
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
