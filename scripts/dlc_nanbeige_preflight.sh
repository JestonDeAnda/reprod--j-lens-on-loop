#!/usr/bin/env bash
# One-GPU Nanbeige4.2-3B adapter validation and one-prompt Jacobian-lens smoke fit.
set -euo pipefail

NAS=/mnt/nas-save/nas-save
ROOT=$NAS/wangjunqi/j-lens-on-loop
MODEL=$ROOT/models/Nanbeige4.2-3B
JLENS_SRC=$NAS/wangjunqi/deps/jacobian-lens
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$ROOT/runs/nanbeige-preflight-$STAMP
OSS_OUT=/mnt/oss-save/wangjunqi/j-lens-on-loop/runs/nanbeige-preflight-$STAMP

if [[ ! -f $ROOT/pyproject.toml || ! -f $JLENS_SRC/pyproject.toml ]]; then
  echo "persistent paths are not visible"
  exit 1
fi

mkdir -p "$OUT" "$OSS_OUT"
exec > >(tee -a "$OUT/preflight.log") 2>&1

copy_artifacts() {
  cp -f "$OUT/preflight.log" "$OSS_OUT/" 2>/dev/null || true
  find "$OUT" -maxdepth 2 -type f \( -name 'metadata.json' -o -name '*.lens.pt' \) \
    -exec cp -f {} "$OSS_OUT/" \; 2>/dev/null || true
}
trap copy_artifacts EXIT

hostname
date -u
nvidia-smi

export HF_HOME=$ROOT/hf-home
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python3 -m pip install --upgrade-strategy only-if-needed -e "$JLENS_SRC"
python3 -m pip install "transformers==4.45.1" datasets pyyaml tqdm
python3 -m pip install --upgrade-strategy only-if-needed -e "$ROOT" --no-deps

if [[ ! -f $MODEL/config.json ]]; then
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    "Nanbeige/Nanbeige4.2-3B",
    local_dir="$MODEL",
)
PY
fi

cd "$ROOT"
python3 scripts/fit_family.py nanbeige \
  --model "$MODEL" \
  --output "$OUT/validate" \
  --validate-only

SMOKE_TEXT="The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."
python3 scripts/fit_family.py nanbeige \
  --model "$MODEL" \
  --output "$OUT/smoke" \
  --prompts 1 \
  --text "$SMOKE_TEXT" \
  --targets 1 \
  --source-window 2 \
  --max-seq-len 32 \
  --dim-batch 16 \
  --checkpoint-every 1

find "$OUT" -type f -printf '%s %p\n' | sort
echo "RESULT: PASS"
echo "OUT=$OUT"
echo "OSS_OUT=$OSS_OUT"
