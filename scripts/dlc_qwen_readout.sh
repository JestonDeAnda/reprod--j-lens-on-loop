#!/usr/bin/env bash
# Qwen final and band-entry readout against the shared WikiText cache.
set -euo pipefail

NAS=/mnt/nas-save/nas-save
REPO=$NAS/wangjunqi/j-lens-on-loop-qwen
ARCHIVE=/mnt/oss-save/modelscope/Qwen/Qwen3.6-27B
MODEL=/tmp/Qwen3.6-27B
PROMPT_CACHE=$NAS/wangjunqi/j-lens-on-loop/cache/wikitext-1000x128.json
LENS_DIR=$REPO/runs/qwen-full-20261027T093500Z/lenses
RUN_NAME=${RUN_NAME:?set RUN_NAME}
OUT=$REPO/runs/$RUN_NAME

test -f "$ARCHIVE/model.safetensors.index.json"
test -f "$LENS_DIR/final.lens.pt"
test -f "$LENS_DIR/band-entry.lens.pt"
mkdir -p "$OUT" "$MODEL"
exec > >(tee -a "$OUT/run.log") 2>&1

export PYTHONPATH="$REPO/src:$NAS/wangjunqi/deps/jacobian-lens:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "staging Qwen weights"
cp -a "$ARCHIVE"/* "$MODEL/"
python3 -m pip install -q --no-deps -e "$NAS/wangjunqi/deps/jacobian-lens" -e "$REPO"

cd "$REPO"
python3 scripts/analyze_readout.py qwen \
  --model "$MODEL" \
  --lens-dir "$LENS_DIR" \
  --prompt-cache "$PROMPT_CACHE" \
  --output "$OUT/summary.json" \
  --raw-output "$OUT/raw.jsonl" \
  --prompts 1000 \
  --max-seq-len 128 \
  --positions -1 -2 -4 -8 \
  --offsets 1 2 4 8 16 32 \
  --period 64

echo "RESULT: PASS"
echo "OUT=$OUT"
