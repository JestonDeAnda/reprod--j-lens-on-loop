#!/usr/bin/env python
"""Fit the Qwen3.6-27B baseline lenses used by arXiv:2609.01924."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from fit_family import configure_exact_forward, corpus_prompts
from jlens import fit, from_hf
from transformers import AutoModelForMultimodalLM, AutoTokenizer

from loop_jlens import validate_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=1000)
    parser.add_argument("--prompt-cache", type=Path)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument(
        "--targets",
        choices=("final", "band-entry"),
        nargs="+",
        default=("final", "band-entry"),
    )
    parser.add_argument("--source-window", type=int)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--text", action="append")
    return parser.parse_args()


def target_family(model, targets: list[str], source_window: int | None):
    family = []
    for name in targets:
        target = model.n_layers - 1 if name == "final" else 24
        sources = list(range(target))
        if source_window is not None:
            if source_window < 1:
                raise ValueError("source_window must be positive")
            sources = sources[-source_window:]
        family.append((name, target, sources))
    return family


def main() -> None:
    args = parse_args()
    configure_exact_forward()
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    model = from_hf(hf_model, tokenizer)
    report = validate_adapter(
        model,
        "The quick brown fox jumps over the lazy dog. " * 5,
        max_seq_len=64,
    )
    print(json.dumps(report.__dict__, indent=2))
    if args.validate_only:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.prompt_cache
    if cache is None and args.text is None:
        cache = args.output / f"wikitext-{args.prompts}x{args.max_seq_len}.json"
    prompts = args.text or corpus_prompts(
        tokenizer,
        args.prompts,
        args.max_seq_len,
        cache_path=cache,
    )
    metadata = {
        "model": args.model,
        "architecture": "qwen3.6",
        "virtual_depth": model.n_layers,
        "d_model": model.d_model,
        "prompts": len(prompts),
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
        "targets": args.targets,
        "source_window": args.source_window,
        "prompt_cache": str(cache) if cache is not None else None,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for name, target, sources in target_family(model, args.targets, args.source_window):
        checkpoint = args.output / f"{name}.checkpoint.pt"
        lens = fit(
            model,
            prompts,
            source_layers=sources,
            target_layer=target,
            dim_batch=args.dim_batch,
            max_seq_len=args.max_seq_len,
            skip_first=args.skip_first,
            checkpoint_path=str(checkpoint),
            checkpoint_every=args.checkpoint_every,
        )
        lens.save(str(args.output / f"{name}.lens.pt"))


if __name__ == "__main__":
    main()
