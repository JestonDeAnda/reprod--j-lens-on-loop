#!/usr/bin/env bash
set -euo pipefail

NAS=/mnt/nas-save/nas-save
REPO=$NAS/wangjunqi/j-lens-on-loop-qwen
MODEL=$NAS/wangjunqi/j-lens-on-loop/models/huginn-0125
LENS_DIR=$NAS/wangjunqi/j-lens-on-loop/runs/paper-fit-20261020T034458Z/huginn-tiling
PROMPT_CACHE=$NAS/wangjunqi/j-lens-on-loop/cache/wikitext-1000x128.json
RUN_NAME=${RUN_NAME:?set RUN_NAME}
OUT=$NAS/wangjunqi/j-lens-on-loop/runs/$RUN_NAME

test -f "$MODEL/config.json"
test -f "$LENS_DIR/recurrence-16.lens.pt"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
export PYTHONPATH="$REPO/src:$NAS/wangjunqi/deps/jacobian-lens:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python3 -m pip install -q --no-deps \
  -e "$NAS/wangjunqi/deps/jacobian-lens" -e "$REPO"
cd "$REPO"
python3 scripts/analyze_readout.py huginn \
  --model "$MODEL" \
  --lens-dir "$LENS_DIR" \
  --prompt-cache "$PROMPT_CACHE" \
  --output "$OUT/summary.json" \
  --raw-output "$OUT/raw.jsonl" \
  --prompts 1000 \
  --max-seq-len 128 \
  --positions -1 -2 -4 -8 \
  --offsets 1 2 4 8 16 32 \
  --period 4

echo "RESULT: PASS"
echo "OUT=$OUT"
