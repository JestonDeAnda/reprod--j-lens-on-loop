#!/usr/bin/env python
"""Resume a ModelScope sharded checkpoint one weight file at a time."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def expected_shards(destination: Path) -> list[str]:
    indexes = sorted(destination.glob("*.safetensors.index.json"))
    if not indexes:
        raise FileNotFoundError(
            f"no safetensors index found under {destination}; stage metadata first"
        )
    shards: set[str] = set()
    for index in indexes:
        payload = json.loads(index.read_text())
        shards.update(payload.get("weight_map", {}).values())
    if not shards:
        raise ValueError(f"no weight shards listed in {indexes}")
    return sorted(shards)


def complete_shards(destination: Path, shards: list[str]) -> list[str]:
    return [
        shard
        for shard in shards
        if (destination / shard).is_file()
        and (destination / shard).stat().st_size > 0
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=int, default=60)
    args = parser.parse_args()

    if args.retries < 1:
        parser.error("--retries must be at least 1")

    from modelscope import model_file_download

    shards = expected_shards(args.destination)
    complete = set(complete_shards(args.destination, shards))
    print(
        f"model={args.model_id} complete={len(complete)}/{len(shards)}",
        flush=True,
    )

    for shard in shards:
        if shard in complete:
            continue
        for attempt in range(1, args.retries + 1):
            print(
                f"downloading shard={shard} attempt={attempt}/{args.retries}",
                flush=True,
            )
            try:
                model_file_download(
                    args.model_id,
                    shard,
                    local_dir=str(args.destination),
                )
                target = args.destination / shard
                if not target.is_file() or target.stat().st_size == 0:
                    raise RuntimeError(f"download did not create {target}")
                complete.add(shard)
                print(
                    f"completed shard={shard} "
                    f"progress={len(complete)}/{len(shards)} "
                    f"bytes={target.stat().st_size}",
                    flush=True,
                )
                break
            except Exception as error:
                print(
                    f"failed shard={shard} attempt={attempt}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                if attempt == args.retries:
                    raise
                time.sleep(args.retry_delay)

    print(f"RESULT: PASS complete={len(complete)}/{len(shards)}", flush=True)


if __name__ == "__main__":
    main()
