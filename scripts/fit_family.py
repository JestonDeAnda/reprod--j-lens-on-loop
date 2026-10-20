#!/usr/bin/env python
"""Fit the per-iteration Jacobian-lens family described in arXiv:2609.01924."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from jlens import configure_logging, fit
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from loop_jlens import HuginnLensModel, OuroLensModel, from_looped_hf, validate_adapter
from loop_jlens.compat import ensure_pad_token_id

MODELS = {
    "ouro": "ByteDance/Ouro-2.6B",
    "huginn": "tomg-group-umd/huginn-0125",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture", choices=sorted(MODELS))
    parser.add_argument("--model", help="override the paper model ID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=1000)
    parser.add_argument("--recurrences", type=int, default=16)
    parser.add_argument("--targets", type=int, nargs="*")
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--full-sources", action="store_true")
    parser.add_argument(
        "--source-window",
        type=int,
        default=None,
        help="keep only the last N sources before each target (plumbing check)",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="fit these prompts instead of streaming WikiText",
    )
    parser.add_argument(
        "--prompt-cache",
        type=Path,
        default=None,
        help="JSON list of prompts; created on first WikiText build, reused later",
    )
    return parser.parse_args()


def corpus_prompts(
    tokenizer,
    count: int,
    length: int,
    *,
    cache_path: Path | None = None,
) -> list[str]:
    if cache_path is not None and cache_path.exists():
        prompts = json.loads(cache_path.read_text())
        if len(prompts) < count:
            raise RuntimeError(
                f"prompt cache {cache_path} has {len(prompts)} prompts, need {count}"
            )
        return prompts[:count]

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    token_buffer: list[int] = []
    prompts: list[str] = []
    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
        token_buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        while len(token_buffer) >= length and len(prompts) < count:
            chunk, token_buffer = token_buffer[:length], token_buffer[length:]
            prompts.append(tokenizer.decode(chunk))
        if len(prompts) == count:
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(prompts) + "\n")
            return prompts
    raise RuntimeError(f"WikiText yielded only {len(prompts)} prompts")


def _limit_sources(sources: list[int], source_window: int | None) -> list[int]:
    if source_window is None:
        return sources
    if source_window < 1:
        raise ValueError("source_window must be positive")
    return sources[-source_window:]


def target_family(
    model,
    requested: list[int] | None,
    *,
    full_sources: bool,
    source_window: int | None = None,
) -> list[tuple[str, int, list[int]]]:
    if isinstance(model, OuroLensModel):
        loops = requested or list(range(1, model.n_loops + 1))
        family = []
        for loop in loops:
            if not 1 <= loop <= model.n_loops:
                raise ValueError(f"Ouro loop target {loop} is out of range")
            start = 0 if full_sources else (loop - 1) * model.n_physical_layers
            target = loop * model.n_physical_layers - 1
            family.append(
                (f"loop-{loop}", target, _limit_sources(list(range(start, target)), source_window))
            )
        return family

    assert isinstance(model, HuginnLensModel)
    recurrences = requested or [4, 8, 12, model.recurrences]
    family = []
    for recurrence in recurrences:
        if not 1 <= recurrence <= model.recurrences:
            raise ValueError(f"Huginn recurrence target {recurrence} is out of range")
        if recurrence == model.recurrences:
            target = model.n_layers - 1  # coda end, as in the primary lens
        else:
            target = model.n_prelude + recurrence * model.n_core - 1
        start = 0 if full_sources else max(0, target - 2 * model.n_core)
        family.append(
            (
                f"recurrence-{recurrence}",
                target,
                _limit_sources(list(range(start, target)), source_window),
            )
        )
    return family


def configure_exact_forward() -> None:
    """Prefer the deterministic attention kernels the Jacobian protocol needs."""

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return
    for name, enabled in (
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_cudnn_sdp", False),
        ("enable_math_sdp", True),
    ):
        toggle = getattr(cuda_backends, name, None)
        if callable(toggle):
            toggle(enabled)


def load_model(args: argparse.Namespace):
    model_id = args.model or MODELS[args.architecture]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    ensure_pad_token_id(config)
    if args.architecture == "ouro":
        # The published Ouro class dispatches to flash/sdpa unless this is eager.
        config._attn_implementation = "eager"
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    model = from_looped_hf(
        hf_model,
        tokenizer,
        architecture=args.architecture,
        recurrences=args.recurrences,
        seed=args.seed,
    )
    return model_id, tokenizer, model


if __name__ == "__main__":
    ARGS = parse_args()
    configure_logging()
    configure_exact_forward()
    torch.manual_seed(ARGS.seed)
    model_id, tokenizer, model = load_model(ARGS)
    report = validate_adapter(
        model,
        "The quick brown fox jumps over the lazy dog. " * 5,
        max_seq_len=64,
    )
    print(json.dumps(report.__dict__, indent=2))
    if ARGS.validate_only:
        raise SystemExit(0)

    ARGS.output.mkdir(parents=True, exist_ok=True)
    cache = ARGS.prompt_cache
    if cache is None and ARGS.text is None:
        cache = ARGS.output / f"wikitext-{ARGS.prompts}x{ARGS.max_seq_len}.json"
    prompts = ARGS.text or corpus_prompts(
        tokenizer, ARGS.prompts, ARGS.max_seq_len, cache_path=cache
    )
    metadata = {
        "model": model_id,
        "architecture": ARGS.architecture,
        "virtual_depth": model.n_layers,
        "prompts": len(prompts),
        "max_seq_len": ARGS.max_seq_len,
        "skip_first": ARGS.skip_first,
        "seed": ARGS.seed,
        "source_window": ARGS.source_window,
        "full_sources": ARGS.full_sources,
        "prompt_cache": str(cache) if cache is not None else None,
    }
    (ARGS.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for name, target, sources in target_family(
        model,
        ARGS.targets,
        full_sources=ARGS.full_sources,
        source_window=ARGS.source_window,
    ):
        checkpoint = ARGS.output / f"{name}.checkpoint.pt"
        lens = fit(
            model,
            prompts,
            source_layers=sources,
            target_layer=target,
            dim_batch=ARGS.dim_batch,
            max_seq_len=ARGS.max_seq_len,
            skip_first=ARGS.skip_first,
            checkpoint_path=str(checkpoint),
            checkpoint_every=ARGS.checkpoint_every,
        )
        lens.save(str(ARGS.output / f"{name}.lens.pt"))
