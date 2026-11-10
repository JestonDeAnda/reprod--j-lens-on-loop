#!/usr/bin/env bash
set -euo pipefail

NAS=/mnt/nas-save/nas-save
REPO=${REPO:-$NAS/wangjunqi/j-lens-on-loop-qwen}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
ROOT=$NAS/wangjunqi/j-lens-on-loop-nanbeige
MODEL=$ROOT/models/Nanbeige4.2-3B
LENS=$ROOT/runs/nanbeige-full-20261022T125900Z/lenses/loop-2.lens.pt
OUT=$NAS/wangjunqi/j-lens-on-loop/runs/$RUN_NAME

test -f "$MODEL/config.json"
test -f "$LENS"
mkdir -p "$OUT/probe-swap" "$OUT/verbal-report"
exec > >(tee -a "$OUT/run.log") 2>&1
export PYTHONPATH="$REPO/src:$NAS/wangjunqi/deps/jacobian-lens:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python3 -m pip install -q --no-deps \
  -e "$NAS/wangjunqi/deps/jacobian-lens" -e "$REPO"
cd "$REPO"

python3 scripts/run_probe_swap.py \
  --config configs/experiments/nanbeige-probe-swap.json \
  --output "$OUT/probe-swap"

python3 scripts/run_verbal_report.py \
  --config configs/experiments/nanbeige-verbal-report.json \
  --output "$OUT/verbal-report"

echo "RESULT: PASS"
echo "OUT=$OUT"
