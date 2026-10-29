#!/usr/bin/env python
"""Evaluate fitted J-Lens and vanilla logit-lens readouts at explicit targets."""

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

from loop_jlens.analysis import estimate_operator_norm, readout_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture", choices=("ouro", "huginn", "nanbeige", "qwen"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--lens-dir", type=Path, required=True)
    parser.add_argument("--prompt-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--positions", type=int, nargs="+", default=[-1])
    parser.add_argument("--offsets", type=int, nargs="+", required=True)
    parser.add_argument("--period", type=int, required=True)
    parser.add_argument("--all-sources", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def prompt_digest(prompts: list[str]) -> str:
    payload = json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def infer_target(lens: JacobianLens) -> int:
    return max(lens.source_layers) + 1


def select_sources(
    lens: JacobianLens,
    target: int,
    offsets: list[int],
    *,
    all_sources: bool,
) -> list[int]:
    if all_sources:
        return lens.source_layers
    requested = {target - offset for offset in offsets}
    return [source for source in lens.source_layers if source in requested]


def token_summary(logits: torch.Tensor, tokenizer, top_k: int) -> list[dict]:
    values, token_ids = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
    rows = []
    for position_values, position_ids in zip(values, token_ids, strict=True):
        ids = position_ids.tolist()
        rows.append(
            {
                "token_ids": ids,
                "tokens": tokenizer.convert_ids_to_tokens(ids),
                "logits": position_values.tolist(),
            }
        )
    return rows


def update_aggregate(
    aggregates: dict[tuple[str, int, int], dict[str, float]],
    *,
    lens_file: str,
    source: int,
    target: int,
    kind: str,
    metrics: dict[str, float],
) -> None:
    key = (f"{lens_file}:{kind}", source, target)
    row = aggregates[key]
    weight = metrics["positions"]
    row["positions"] += weight
    row["prompt_records"] += 1
    for metric, value in metrics.items():
        if metric != "positions":
            row[metric] += value * weight


def finish_aggregates(
    aggregates: dict[tuple[str, int, int], dict[str, float]],
    period: int,
) -> list[dict]:
    rows = []
    for (label, source, target), values in sorted(aggregates.items()):
        lens_file, kind = label.rsplit(":", 1)
        positions = values["positions"]
        rows.append(
            {
                "lens_file": lens_file,
                "kind": kind,
                "source": source,
                "target": target,
                "offset": target - source,
                "iteration_boundaries_crossed": target // period - source // period,
                "prompt_records": int(values["prompt_records"]),
                "positions": int(positions),
                **{
                    metric: value / positions
                    for metric, value in values.items()
                    if metric not in {"positions", "prompt_records"}
                },
            }
        )
    return rows


@torch.no_grad()
def evaluate_lens(
    lens_path: Path,
    *,
    model,
    tokenizer,
    prompts: list[str],
    positions: list[int],
    offsets: list[int],
    period: int,
    all_sources: bool,
    max_seq_len: int,
    top_k: int,
    raw_handle,
    aggregates: dict[tuple[str, int, int], dict[str, float]],
) -> None:
    lens = JacobianLens.load(str(lens_path))
    target = infer_target(lens)
    sources = select_sources(lens, target, offsets, all_sources=all_sources)
    if not sources:
        raise ValueError(f"{lens_path.name}: no requested source layers are fitted")
    if target >= model.n_layers:
        raise ValueError(f"{lens_path.name}: inferred target {target} is out of range")
    record_at = sorted(set(sources) | {target})

    for prompt_index, prompt in enumerate(prompts):
        input_ids = model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=record_at) as recorder:
            model.forward(input_ids)

        def selected(layer: int) -> torch.Tensor:
            activation = recorder.activations[layer].detach()[0]
            return activation[positions].float()

        target_logits = model.unembed(selected(target)).float().cpu()
        target_tokens = token_summary(target_logits, tokenizer, top_k)
        for source in sources:
            source_residual = selected(source)
            candidates = {
                "jacobian": model.unembed(lens.transport(source_residual, source))
                .float()
                .cpu(),
                "logit_lens": model.unembed(source_residual).float().cpu(),
            }
            for kind, candidate_logits in candidates.items():
                metrics = readout_metrics(candidate_logits, target_logits)
                update_aggregate(
                    aggregates,
                    lens_file=lens_path.name,
                    source=source,
                    target=target,
                    kind=kind,
                    metrics=metrics,
                )
                raw_handle.write(
                    json.dumps(
                        {
                            "lens_file": lens_path.name,
                            "prompt_index": prompt_index,
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "positions": positions,
                            "kind": kind,
                            "source": source,
                            "target": target,
                            "offset": target - source,
                            "iteration_boundaries_crossed": (
                                target // period - source // period
                            ),
                            "metrics": metrics,
                            "target_top_k": target_tokens,
                            "candidate_top_k": token_summary(
                                candidate_logits, tokenizer, top_k
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def load_qwen(model_path: str, device_map: str):
    from jlens import from_hf
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    return model_path, tokenizer, from_hf(hf_model, tokenizer)


def operator_norm_rows(lens_path: Path, offsets: list[int]) -> list[dict]:
    lens = JacobianLens.load(str(lens_path))
    target = infer_target(lens)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for source in select_sources(lens, target, offsets, all_sources=False):
        matrix = lens.jacobians[source].to(device)
        estimate = estimate_operator_norm(matrix, max_iterations=40)
        rows.append(
            {
                "lens_file": lens_path.name,
                "source": source,
                "target": target,
                "offset": target - source,
                **estimate.to_dict(),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.period < 1 or args.prompts < 1 or args.top_k < 1:
        raise SystemExit("--period, --prompts and --top-k must be positive")
    prompts = json.loads(args.prompt_cache.read_text())[: args.prompts]
    if len(prompts) < args.prompts:
        raise SystemExit(f"prompt cache contains only {len(prompts)} prompts")
    lens_paths = sorted(args.lens_dir.glob("*.lens.pt"))
    if not lens_paths:
        raise SystemExit(f"no *.lens.pt under {args.lens_dir}")

    configure_exact_forward()
    torch.manual_seed(args.seed)
    model_args = SimpleNamespace(
        architecture=args.architecture,
        model=args.model,
        device_map=args.device_map,
        recurrences=16,
        seed=args.seed,
    )
    if args.architecture == "qwen":
        model_id, tokenizer, model = load_qwen(args.model, args.device_map)
    else:
        model_id, tokenizer, model = load_model(model_args)
    norms = [
        row
        for lens_path in lens_paths
        for row in operator_norm_rows(lens_path, args.offsets)
    ]
    raw_output = args.raw_output or args.output.with_suffix(".jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    aggregates: dict[tuple[str, int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    with raw_output.open("w") as raw_handle:
        for lens_path in lens_paths:
            evaluate_lens(
                lens_path,
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                positions=args.positions,
                offsets=args.offsets,
                period=args.period,
                all_sources=args.all_sources,
                max_seq_len=args.max_seq_len,
                top_k=args.top_k,
                raw_handle=raw_handle,
                aggregates=aggregates,
            )

    payload = {
        "architecture": args.architecture,
        "model": model_id,
        "lens_dir": str(args.lens_dir),
        "prompt_cache": str(args.prompt_cache),
        "prompt_count": len(prompts),
        "prompt_sha256": prompt_digest(prompts),
        "max_seq_len": args.max_seq_len,
        "positions": args.positions,
        "offsets": args.offsets,
        "period": args.period,
        "raw_output": str(raw_output),
        "rows": finish_aggregates(aggregates, args.period),
        "operator_norms": norms,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(payload["rows"])}))
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
