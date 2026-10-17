#!/usr/bin/env bash
# One-GPU preflight for the looped Jacobian-lens harness.
# Runs on a DLC worker that can see /mnt/nas-save. Does not start a paper-scale fit.
set -euo pipefail

# DSW mounts the NFS export /nas-save at /mnt/nas-save. A DLC DataSource for the
# same NAS URI mounts the filesystem root, where that export is /mnt/nas-save/nas-save.
NAS=/mnt/nas-save/nas-save
ROOT=$NAS/wangjunqi/j-lens-on-loop
OURO_MODEL=$NAS/wangjunqi/ouro/models/Ouro-2.6B
HUGINN_MODEL=$ROOT/models/huginn-0125
JLENS_SRC=$NAS/wangjunqi/deps/jacobian-lens
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$ROOT/runs/dlc-smoke-$STAMP
if [[ ! -f $ROOT/pyproject.toml || ! -f $OURO_MODEL/config.json || ! -f $JLENS_SRC/pyproject.toml ]]; then
  echo "persistent paths are not visible"
  ls -la /mnt/nas-save | head -40
  ls -la "$NAS/wangjunqi" || true
  exit 1
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/smoke.log") 2>&1

echo "=== host ==="
hostname
date -u
nvidia-smi
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY

export HF_HOME=$ROOT/hf-home
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

echo "=== install ==="
python3 -m pip install --upgrade-strategy only-if-needed -e "$JLENS_SRC"
python3 -m pip install --upgrade-strategy only-if-needed datasets pyyaml tqdm
python3 -m pip install --upgrade-strategy only-if-needed -e "$ROOT" --no-deps

if [[ ! -f $HUGINN_MODEL/config.json ]]; then
  echo "=== download huginn into NAS ==="
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("tomg-group-umd/huginn-0125", local_dir="${HUGINN_MODEL}")
PY
fi

cd "$ROOT"
echo "=== validate ouro ==="
python3 scripts/fit_family.py ouro \
  --model "$OURO_MODEL" \
  --output "$OUT/ouro-validate" \
  --validate-only

echo "=== validate huginn ==="
python3 scripts/fit_family.py huginn \
  --model "$HUGINN_MODEL" \
  --output "$OUT/huginn-validate" \
  --validate-only

SMOKE_TEXT="The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."

echo "=== smoke fit ouro: 1 prompt, loop 1, 2 sources ==="
python3 scripts/fit_family.py ouro \
  --model "$OURO_MODEL" \
  --output "$OUT/ouro-smoke" \
  --prompts 1 \
  --text "$SMOKE_TEXT" \
  --targets 1 \
  --source-window 2 \
  --max-seq-len 32 \
  --dim-batch 16 \
  --checkpoint-every 1

echo "=== smoke fit huginn: 1 prompt, recurrence 4, 2 sources ==="
python3 scripts/fit_family.py huginn \
  --model "$HUGINN_MODEL" \
  --output "$OUT/huginn-smoke" \
  --prompts 1 \
  --text "$SMOKE_TEXT" \
  --targets 4 \
  --recurrences 16 \
  --source-window 2 \
  --max-seq-len 32 \
  --dim-batch 16 \
  --checkpoint-every 1

echo "=== artefacts ==="
find "$OUT" -type f -printf '%s %p\n' | sort
echo "RESULT: PASS"
echo "OUT=$OUT"
