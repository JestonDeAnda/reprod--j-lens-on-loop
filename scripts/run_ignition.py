#!/usr/bin/env python
"""Bounded released-prompt ignition readout for Ouro.

The full paper grid is thousands of carrier/pair/alpha forwards. This run
keeps a deterministic prefix of that grid so one GPU can finish it, and
labels the result as a bounded adaptation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import torch
from fit_family import configure_exact_forward, load_model
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder

from loop_jlens.causal import token_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def single_token(tokenizer, text: str) -> int | None:
    pieces = tokenizer.encode(" " + text.strip(), add_special_tokens=False)
    return pieces[0] if len(pieces) == 1 else None


def ranked_pairs(words: list[str], tokenizer, limit: int) -> list[tuple[str, str, int, int]]:
    pairs = []
    for left, right in combinations(words, 2):
        left_id = single_token(tokenizer, left)
        right_id = single_token(tokenizer, right)
        if left_id is None or right_id is None or left_id == right_id:
            continue
        pairs.append((left, right, left_id, right_id))
        if len(pairs) == limit:
            break
    return pairs


def explicit_pairs(pairs: list[list[str]], tokenizer, limit: int):
    result = []
    for left, right in pairs:
        left_id = single_token(tokenizer, left)
        right_id = single_token(tokenizer, right)
        if left_id is None or right_id is None or left_id == right_id:
            continue
        result.append((left, right, left_id, right_id))
        if len(result) == limit:
            break
    return result


def placeholder_position(model, template: str, word: str, token_id: int, max_length: int):
    """Locate the substituted word after the template prefix.

    A trailing space in the prefix is absorbed into the next token
    (``visit {W}`` becomes ``ĠFrance``), so that token sits at
    ``len(prefix_ids) - 1`` rather than at ``len(prefix_ids)``.
    """

    prefix = template.split("{W}")[0]
    prompt = template.replace("{W}", word)
    input_ids = model.encode(prompt, max_length=max_length)
    prefix_ids = model.encode(prefix, max_length=max_length)
    candidates = [prefix_ids.shape[1]]
    if prefix[-1:].isspace() and prefix_ids.shape[1] > 0:
        candidates.insert(0, prefix_ids.shape[1] - 1)
    for position in candidates:
        if position < input_ids.shape[1] and int(input_ids[0, position]) == token_id:
            return prompt, input_ids, position
    return None


@contextmanager
def replace_embedding(embed, position: int, vector: torch.Tensor):
    def hook(_module, _inputs, output):
        changed = output.clone()
        changed[:, position, :] = vector.to(device=changed.device, dtype=changed.dtype)
        return changed

    handle = embed.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def crossing(points: list[tuple[float, float]], level: float) -> float | None:
    for (left_alpha, left_share), (right_alpha, right_share) in zip(points, points[1:]):
        if (left_share - level) * (right_share - level) <= 0 and right_share != left_share:
            weight = (level - left_share) / (right_share - left_share)
            return left_alpha + weight * (right_alpha - left_alpha)
        if left_share == level:
            return left_alpha
    return None


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
    embed_weight = model._embed_tokens.weight.detach()
    families = {
        "countries": (
            ranked_pairs(data["countries_12"], tokenizer, config["country_pairs"]),
            data["ctx_templates"][: config["country_templates"]],
        ),
        "idiom": (
            explicit_pairs(data["idiom_pairs"], tokenizer, config["control_pairs"]),
            data["noun_ctx_templates"][: config["control_templates"]],
        ),
        "scrambled": (
            explicit_pairs(data["scrambled_pairs"], tokenizer, config["control_pairs"]),
            data["noun_ctx_templates"][: config["control_templates"]],
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "trials.jsonl"
    curves = defaultdict(lambda: defaultdict(list))
    forwards = 0
    with raw_path.open("w") as handle:
        for family, (pairs, templates) in families.items():
            for left, right, left_id, right_id in pairs:
                left_vector = embed_weight[left_id].float()
                right_vector = embed_weight[right_id].float()
                prepared = []
                for template in templates:
                    located = placeholder_position(
                        model, template, left, left_id, config["max_seq_len"]
                    )
                    if located is not None:
                        prepared.append(located)
                for alpha in config["alphas"]:
                    mixed = alpha * left_vector + (1 - alpha) * right_vector
                    for prompt, input_ids, position in prepared:
                        with (
                            replace_embedding(model._embed_tokens, position, mixed),
                            ActivationRecorder(model.layers, at=layers) as recorder,
                        ):
                            model.forward(input_ids)
                        forwards += 1
                        for layer in layers:
                            residual = recorder.activations[layer][0, position].float()
                            logits = model.unembed(
                                lens.transport(residual, layer)
                            ).float().cpu()
                            rank_left = token_rank(logits, left_id)
                            rank_right = token_rank(logits, right_id)
                            share = (1 / rank_left) / (1 / rank_left + 1 / rank_right)
                            curves[(family, f"{left}->{right}", layer)][alpha].append(share)
                            handle.write(
                                json.dumps(
                                    {
                                        "protocol_status": "bounded_released_prompt_adaptation",
                                        "family": family,
                                        "pair": f"{left}->{right}",
                                        "alpha": alpha,
                                        "layer": layer,
                                        "position": position,
                                        "rank_left": rank_left,
                                        "rank_right": rank_right,
                                        "reciprocal_rank_share": share,
                                        "prompt_sha256": hashlib.sha256(
                                            prompt.encode()
                                        ).hexdigest(),
                                    }
                                )
                                + "\n"
                            )

    pair_rows = []
    for (family, pair, layer), alpha_values in sorted(curves.items()):
        points = [
            (alpha, sum(values) / len(values))
            for alpha, values in sorted(alpha_values.items())
        ]
        low = crossing(points, 0.1)
        threshold = crossing(points, 0.5)
        high = crossing(points, 0.9)
        pair_rows.append(
            {
                "family": family,
                "pair": pair,
                "layer": layer,
                "threshold_alpha": threshold,
                "transition_width_10_90": (
                    None if low is None or high is None else high - low
                ),
                "share_by_alpha": [
                    {"alpha": alpha, "mean_share": share} for alpha, share in points
                ],
            }
        )

    layer_rows = []
    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[(row["family"], row["layer"])].append(row)
    for (family, layer), rows in sorted(grouped.items()):
        thresholds = [row["threshold_alpha"] for row in rows if row["threshold_alpha"] is not None]
        widths = [
            row["transition_width_10_90"]
            for row in rows
            if row["transition_width_10_90"] is not None
        ]
        layer_rows.append(
            {
                "family": family,
                "layer": layer,
                "pairs": len(rows),
                "threshold_defined": len(thresholds),
                "mean_threshold_alpha": (
                    sum(thresholds) / len(thresholds) if thresholds else None
                ),
                "mean_transition_width_10_90": (
                    sum(widths) / len(widths) if widths else None
                ),
            }
        )

    summary = {
        "protocol_status": "bounded_released_prompt_adaptation",
        "model": model_id,
        "lens": config["lens"],
        "data": config["data"],
        "forwards": forwards,
        "family_sizes": {
            family: {"pairs": len(pairs), "templates": len(templates)}
            for family, (pairs, templates) in families.items()
        },
        "alphas": config["alphas"],
        "band_layers": layers,
        "layer_summary": layer_rows,
        "raw_output": str(raw_path),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "pair-curves.json").write_text(json.dumps(pair_rows, indent=2) + "\n")
    if forwards == 0:
        raise RuntimeError("no ignition forwards were scored; placeholder alignment failed")
    print(json.dumps({"forwards": forwards, "layers": len(layer_rows)}))
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
