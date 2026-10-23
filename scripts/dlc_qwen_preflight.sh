#!/usr/bin/env bash
# One-GPU Qwen3.6-27B baseline validation and one-prompt J-Lens smoke fit.
set -euo pipefail

NAS=/mnt/nas-save/nas-save
ROOT=$NAS/wangjunqi/j-lens-on-loop-qwen
MODEL=$ROOT/models/Qwen3.6-27B
REVISION=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
JLENS_SRC=$NAS/wangjunqi/deps/jacobian-lens
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$ROOT/runs/qwen-preflight-$STAMP
OSS_OUT=/mnt/oss-save/wangjunqi/j-lens-on-loop/runs/qwen-preflight-$STAMP

test -f "$ROOT/pyproject.toml"
test -f "$JLENS_SRC/pyproject.toml"
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
python3 -m pip install "transformers==5.20.1" "accelerate>=1.0" datasets pyyaml tqdm
python3 -m pip install --upgrade-strategy only-if-needed -e "$ROOT" --no-deps

if [[ ! -f $MODEL/model.safetensors.index.json ]]; then
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3.6-27B",
    revision="$REVISION",
    local_dir="$MODEL",
)
PY
fi

cd "$ROOT"
python3 scripts/fit_qwen.py \
  --model "$MODEL" \
  --output "$OUT/validate" \
  --validate-only

SMOKE_TEXT="The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."
python3 scripts/fit_qwen.py \
  --model "$MODEL" \
  --output "$OUT/smoke" \
  --prompts 1 \
  --text "$SMOKE_TEXT" \
  --targets final \
  --source-window 2 \
  --max-seq-len 32 \
  --dim-batch 8 \
  --checkpoint-every 1

find "$OUT" -type f -printf '%s %p\n' | sort
echo "RESULT: PASS"
echo "OUT=$OUT"
echo "OSS_OUT=$OSS_OUT"
