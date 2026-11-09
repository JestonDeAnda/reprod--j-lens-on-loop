#!/usr/bin/env bash
set -euo pipefail

NAS=/mnt/nas-save/nas-save
REPO=${REPO:-$NAS/wangjunqi/j-lens-on-loop-qwen}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
ARCHIVE=/mnt/oss-save/modelscope/Qwen/Qwen3.6-27B
MODEL=/tmp/Qwen3.6-27B
OUT=$NAS/wangjunqi/j-lens-on-loop/runs/$RUN_NAME
CONFIG=$REPO/configs/experiments/qwen-probe-swap.json

test -f "$ARCHIVE/model.safetensors.index.json"
test -f "$NAS/wangjunqi/j-lens-on-loop-qwen/runs/qwen-full-20261027T093500Z/lenses/final.lens.pt"
mkdir -p "$OUT" "$MODEL"
exec > >(tee -a "$OUT/run.log") 2>&1
export PYTHONPATH="$REPO/src:$NAS/wangjunqi/deps/jacobian-lens:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "staging Qwen weights"
cp -a "$ARCHIVE"/* "$MODEL/"
python3 -m pip install -q --no-deps \
  -e "$NAS/wangjunqi/deps/jacobian-lens" -e "$REPO"
cd "$REPO"
python3 scripts/run_probe_swap.py --config "$CONFIG" --output "$OUT"

echo "RESULT: PASS"
echo "OUT=$OUT"
