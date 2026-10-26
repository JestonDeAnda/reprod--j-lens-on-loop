#!/usr/bin/env python
"""Stage all reproduction checkpoints into the DSW oss-save archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

MODELS = (
    (
        "ByteDance/Ouro-2.6B",
        Path("/mnt/nas-save/wangjunqi/ouro/models/Ouro-2.6B"),
        False,
    ),
    (
        "tomg-group-umd/huginn-0125",
        Path("/mnt/nas-save/wangjunqi/j-lens-on-loop/models/huginn-0125"),
        False,
    ),
    (
        "Nanbeige/Nanbeige4.2-3B",
        Path(
            "/mnt/nas-save/wangjunqi/j-lens-on-loop-nanbeige/"
            "models/Nanbeige4.2-3B"
        ),
        False,
    ),
    (
        "Qwen/Qwen3.6-27B",
        Path(
            "/mnt/nas-save/wangjunqi/j-lens-on-loop-qwen/"
            "models/Qwen3.6-27B"
        ),
        True,
    ),
    ("ByteDance/Ouro-2.6B-Thinking", None, True),
)


def copy_cached_files(source: Path, destination: Path) -> int:
    copied = 0
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        if any(part in {".cache", ".git"} for part in relative.parts):
            continue
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not item.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == item.stat().st_size:
            continue
        shutil.copyfile(item, target)
        copied += 1
        print(f"copied {item} -> {target}", flush=True)
    return copied


def expected_weight_files(directory: Path) -> set[str]:
    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if not indexes:
        return set()
    expected: set[str] = set()
    for index in indexes:
        payload = json.loads(index.read_text())
        expected.update(payload.get("weight_map", {}).values())
    return expected


def is_complete(directory: Path) -> bool:
    if not (directory / "config.json").is_file():
        return False
    expected = expected_weight_files(directory)
    if expected:
        return all((directory / name).is_file() for name in expected)
    return any(directory.glob("*.safetensors"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path) -> None:
    records = []
    for model_id, _, _ in MODELS:
        directory = root / model_id
        files = [path for path in directory.rglob("*") if path.is_file()]
        records.append(
            {
                "model_id": model_id,
                "path": str(directory),
                "complete": is_complete(directory),
                "files": len(files),
                "bytes": sum(path.stat().st_size for path in files),
                "metadata_sha256": {
                    str(path.relative_to(directory)): sha256(path)
                    for path in sorted(directory.glob("*.json"))
                },
            }
        )
    (root / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/mnt/oss-save/modelscope"),
    )
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    for model_id, source, download_if_incomplete in MODELS:
        destination = args.destination / model_id
        destination.mkdir(parents=True, exist_ok=True)
        if source is not None and source.exists():
            copy_cached_files(source, destination)
        if download_if_incomplete and not is_complete(destination):
            from modelscope import snapshot_download

            print(f"downloading {model_id} into {destination}", flush=True)
            snapshot_download(model_id, local_dir=str(destination))
        print(
            f"model={model_id} complete={is_complete(destination)}",
            flush=True,
        )

    write_manifest(args.destination)
    if not all(is_complete(args.destination / model_id) for model_id, _, _ in MODELS):
        raise SystemExit("one or more model archives are incomplete")
    print("RESULT: PASS", flush=True)


if __name__ == "__main__":
    main()
