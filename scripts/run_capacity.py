#!/usr/bin/env python
"""Released-prompt list-capacity readout for Ouro.

Each trial is four 20-word blocks drawn from the released single-token canons.
At every comma, count how many words read so far have band-minimum J-Lens rank
at or below the configured cutoffs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def single_token(tokenizer, text: str) -> int | None:
    pieces = tokenizer.encode(text, add_special_tokens=False)
    return pieces[0] if len(pieces) == 1 else None


def canon_for(pool: dict, tokenizer, limit: int) -> list[tuple[str, int]]:
    canon = []
    for word in pool["pool"]:
        token_id = single_token(tokenizer, " " + word.strip())
        if token_id is None:
            continue
        canon.append((word.strip(), token_id))
        if len(canon) == limit:
            break
    return canon


def align_list(model, tokenizer, words: list[str], token_ids: list[int], max_length: int):
    """Prefix keeps every list word on its leading-space token."""
    prompt = "List: " + ", ".join(words)
    bare = tokenizer.encode(prompt, add_special_tokens=False)
    full = model.encode(prompt, max_length=max_length)[0].tolist()
    if len(full) < len(bare):
        return None
    offset = next(
        (
            index
            for index in range(len(full) - len(bare) + 1)
            if full[index : index + len(bare)] == bare
        ),
        None,
    )
    if offset is None:
        return None
    prefix = tokenizer.encode("List:", add_special_tokens=False)
    if bare[: len(prefix)] != prefix:
        return None
    cursor = len(prefix)
    commas = []
    for index, token_id in enumerate(token_ids):
        if cursor >= len(bare) or bare[cursor] != token_id:
            return None
        cursor += 1
        if index == len(token_ids) - 1:
            break
        comma = tokenizer.encode(",", add_special_tokens=False)
        if bare[cursor : cursor + len(comma)] != comma:
            return None
        commas.append(offset + cursor)
        cursor += len(comma)
    return prompt, commas


def ranks_for(logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    scores = logits[token_ids]
    return 1 + (logits.unsqueeze(0) > scores.unsqueeze(1)).sum(dim=1)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    data = json.loads(Path(config["data"]).read_text())
    configure_exact_forward()
    rng = random.Random(config["seed"])
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
    pools = {pool["name"]: pool for pool in data["candidate_pools"]}
    canons = {
        family: canon_for(
            pools[family], tokenizer, data["targets_per_family"][family]
        )
        for family in data["block_families"]
    }
    short = {family: len(words) for family, words in canons.items() if len(words) < 20}
    if short:
        raise RuntimeError(f"single-token canon is shorter than one block: {short}")

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "trials.jsonl"
    curves = defaultdict(lambda: defaultdict(list))
    completed = 0
    with raw_path.open("w") as handle:
        for trial in range(config["trials"]):
            families = list(data["block_families"])
            rng.shuffle(families)
            selected = []
            for family in families:
                selected.extend(rng.sample(canons[family], 20))
            words = [word for word, _token_id in selected]
            token_ids = [token_id for _word, token_id in selected]
            located = align_list(model, tokenizer, words, token_ids, config["max_seq_len"])
            if located is None:
                continue
            prompt, comma_positions = located
            input_ids = model.encode(prompt, max_length=config["max_seq_len"])
            with ActivationRecorder(model.layers, at=layers) as recorder:
                model.forward(input_ids)
            best_rank = torch.full((len(comma_positions), len(token_ids)), 10**9)
            for layer in layers:
                residual = recorder.activations[layer][0, comma_positions].float()
                logits = model.unembed(lens.transport(residual, layer)).float().cpu()
                for position, row in enumerate(logits):
                    word_ranks = ranks_for(row, token_ids)
                    best_rank[position, : position + 1] = torch.minimum(
                        best_rank[position, : position + 1],
                        word_ranks[: position + 1],
                    )
            row = {
                "protocol_status": "released_prompts_loop_band_adaptation",
                "trial": trial,
                "family_order": families,
                "list_prefix": "List:",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "counts": {},
            }
            for cutoff in config["rank_cutoffs"]:
                retained = [
                    int((best_rank[position, : position + 1] <= cutoff).sum())
                    for position in range(len(comma_positions))
                ]
                row["counts"][str(cutoff)] = retained
                for seen, count in enumerate(retained, start=1):
                    curves[cutoff][seen].append(count)
            handle.write(json.dumps(row) + "\n")
            completed += 1

    if completed == 0:
        raise RuntimeError("no capacity trials were aligned")
    summary = {
        "protocol_status": "released_prompts_loop_band_adaptation",
        "model": model_id,
        "lens": config["lens"],
        "data": config["data"],
        "trials_requested": config["trials"],
        "trials_completed": completed,
        "canon_sizes": {family: len(words) for family, words in canons.items()},
        "band_layers": layers,
        "curves": {
            str(cutoff): [
                {
                    "words_read": seen,
                    "mean_retained": sum(values) / len(values),
                    "trials": len(values),
                }
                for seen, values in sorted(points.items())
            ]
            for cutoff, points in sorted(curves.items())
        },
        "raw_output": str(raw_path),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"trials": completed, "cutoffs": config["rank_cutoffs"]}))
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
