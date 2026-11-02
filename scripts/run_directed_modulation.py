#!/usr/bin/env python
"""Released-prompt directed-modulation readout experiment for Ouro."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from fit_family import configure_exact_forward, load_model
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("topic", "math", "all"), default="all")
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def token_ids(tokenizer, words: list[str]) -> list[int]:
    result = set()
    for word in words:
        pieces = tokenizer.encode(" " + word.strip(), add_special_tokens=False)
        if len(pieces) == 1:
            result.add(pieces[0])
    return sorted(result)


def response_positions(model, prefix: str, prompt: str, max_length: int) -> list[int]:
    before = model.encode(prefix, max_length=max_length)
    full = model.encode(prompt, max_length=max_length)
    start = max(0, before.shape[1] - 1)
    return list(range(start, full.shape[1]))


def targets_for(data: dict, tokenizer, family: str) -> list[dict]:
    targets = []
    if family in {"math", "all"}:
        for item in data["math_problems"]:
            ids = token_ids(tokenizer, [item["answer"]])
            if ids:
                targets.append(
                    {
                        "family": "math",
                        "key": item["expr"],
                        "x": item["expr"],
                        "target_ids": ids,
                        "tier": item["tier"],
                    }
                )
    if family in {"topic", "all"}:
        for item in data["topic_categories"]:
            ids = token_ids(tokenizer, item["members"])
            if ids:
                targets.append(
                    {
                        "family": "topic",
                        "key": item["name"],
                        "x": item["name"],
                        "target_ids": ids,
                    }
                )
    return targets


@torch.no_grad()
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
    layers = config["band_layers"]
    top_k = config["top_k"]
    targets = targets_for(data, tokenizer, args.family)

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "trials.jsonl"
    aggregates = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    with raw_path.open("w") as handle:
        for target_index, target in enumerate(targets):
            target_tensor = torch.tensor(target["target_ids"])
            for phrasing_index, phrasing in enumerate(data["phrasings"]):
                carrier = data["carrier_sentences"][
                    (target_index * len(data["phrasings"]) + phrasing_index)
                    % len(data["carrier_sentences"])
                ]
                instruction = phrasing["text"].format(x=target["x"])
                prefix = instruction + "\n\n"
                prompt = prefix + carrier
                input_ids = model.encode(prompt, max_length=config["max_seq_len"])
                positions = response_positions(
                    model, prefix, prompt, config["max_seq_len"]
                )
                positions = [p for p in positions if p < input_ids.shape[1]]
                if not positions:
                    continue
                with ActivationRecorder(model.layers, at=layers) as recorder:
                    model.forward(input_ids)
                group_kind = data["group_kind"][phrasing["group"]]
                for layer in layers:
                    residual = recorder.activations[layer][0, positions].float()
                    logits = model.unembed(lens.transport(residual, layer)).float().cpu()
                    top_ids = logits.topk(top_k, dim=-1).indices
                    hits = (top_ids[..., None] == target_tensor).any(dim=(-1, -2))
                    hit_fraction = float(hits.float().mean())
                    any_hit = float(hits.any())
                    key = (target["family"], group_kind, layer)
                    aggregates[key]["hit_fraction"] += hit_fraction
                    aggregates[key]["any_hit"] += any_hit
                    counts[key] += 1
                    row = {
                        "protocol_status": "released_prompts_loop_band_adaptation",
                        "family": target["family"],
                        "target": target["key"],
                        "phrasing": phrasing["name"],
                        "group": phrasing["group"],
                        "group_kind": group_kind,
                        "layer": layer,
                        "positions": len(positions),
                        "hit_fraction": hit_fraction,
                        "any_hit": bool(any_hit),
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    }
                    if "tier" in target:
                        row["tier"] = target["tier"]
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = []
    for (family, group_kind, layer), values in sorted(aggregates.items()):
        count = counts[(family, group_kind, layer)]
        rows.append(
            {
                "family": family,
                "group_kind": group_kind,
                "layer": layer,
                "trials": count,
                "mean_position_hit_rate": values["hit_fraction"] / count,
                "prompt_any_hit_rate": values["any_hit"] / count,
            }
        )
    indexed = {
        (row["family"], row["group_kind"], row["layer"]): row for row in rows
    }
    contrasts = []
    for family in sorted({row["family"] for row in rows}):
        for layer in layers:
            control = indexed[(family, "control", layer)]
            for group_kind in ("focus", "suppress"):
                row = indexed[(family, group_kind, layer)]
                contrasts.append(
                    {
                        "family": family,
                        "layer": layer,
                        "contrast": f"{group_kind}-control",
                        "position_hit_rate_delta": (
                            row["mean_position_hit_rate"]
                            - control["mean_position_hit_rate"]
                        ),
                        "any_hit_rate_delta": (
                            row["prompt_any_hit_rate"]
                            - control["prompt_any_hit_rate"]
                        ),
                    }
                )
    summary = {
        "protocol_status": "released_prompts_loop_band_adaptation",
        "model": model_id,
        "lens": config["lens"],
        "data": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "family": args.family,
        "top_k": top_k,
        "band_layers": layers,
        "target_count": len(targets),
        "rows": rows,
        "contrasts": contrasts,
        "raw_output": str(raw_path),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"family": args.family, "targets": len(targets), "rows": len(rows)}
        )
    )
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
