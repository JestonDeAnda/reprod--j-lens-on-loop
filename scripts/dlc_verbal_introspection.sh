#!/usr/bin/env bash
set -euo pipefail

NAS=/mnt/nas-save/nas-save
REPO=$NAS/wangjunqi/j-lens-on-loop-qwen
RUN_NAME=${RUN_NAME:?set RUN_NAME}
OUT=$NAS/wangjunqi/j-lens-on-loop/runs/$RUN_NAME
CONFIG=${CONFIG:-$REPO/configs/experiments/ouro-verbal-introspection.json}

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
export PYTHONPATH="$REPO/src:$NAS/wangjunqi/deps/jacobian-lens:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python3 -m pip install -q --no-deps \
  -e "$NAS/wangjunqi/deps/jacobian-lens" -e "$REPO"
cd "$REPO"
python3 scripts/run_verbal_introspection.py --config "$CONFIG" --output "$OUT"

echo "RESULT: PASS"
echo "OUT=$OUT"
