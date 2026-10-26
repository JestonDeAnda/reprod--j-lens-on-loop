#!/usr/bin/env bash
# GPU pilot for explicit-target J-Lens versus vanilla logit-lens readout metrics.
set -euo pipefail

ARCHITECTURE=${ARCHITECTURE:?set ARCHITECTURE to ouro or nanbeige}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
PROMPTS=${PROMPTS:-32}
NAS=/mnt/nas-save/nas-save
REPO=$NAS/wangjunqi/j-lens-on-loop-qwen
PROMPT_CACHE=$NAS/wangjunqi/j-lens-on-loop/cache/wikitext-1000x128.json

export PYTHONPATH="$REPO/src:$NAS/wangjunqi/deps/jacobian-lens:${PYTHONPATH:-}"
export HF_HOME="$REPO/hf-home"
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
python3 -m pip install -q --no-deps -e "$NAS/wangjunqi/deps/jacobian-lens" -e "$REPO"

case "$ARCHITECTURE" in
  ouro)
    MODEL=$NAS/wangjunqi/ouro/models/Ouro-2.6B
    LENS_DIR=$NAS/wangjunqi/j-lens-on-loop/runs/paper-fit-20261020T034458Z/ouro-paper
    OUTPUT_ROOT=$NAS/wangjunqi/j-lens-on-loop/runs/$RUN_NAME
    PERIOD=48
    OFFSETS=(1 2 4 8 16 32 48 96 144)
    ;;
  nanbeige)
    MODEL=$NAS/wangjunqi/j-lens-on-loop-nanbeige/models/Nanbeige4.2-3B
    LENS_DIR=$NAS/wangjunqi/j-lens-on-loop-nanbeige/runs/nanbeige-full-20261022T125900Z/lenses
    OUTPUT_ROOT=$NAS/wangjunqi/j-lens-on-loop-nanbeige/runs/$RUN_NAME
    PERIOD=22
    OFFSETS=(1 2 4 8 16 22)
    ;;
  *)
    echo "unsupported ARCHITECTURE=$ARCHITECTURE" >&2
    exit 2
    ;;
esac

for path in "$MODEL/config.json" "$LENS_DIR" "$PROMPT_CACHE"; do
  test -e "$path" || { echo "missing required path: $path" >&2; exit 3; }
done

mkdir -p "$OUTPUT_ROOT"
python3 scripts/analyze_readout.py "$ARCHITECTURE" \
  --model "$MODEL" \
  --lens-dir "$LENS_DIR" \
  --prompt-cache "$PROMPT_CACHE" \
  --output "$OUTPUT_ROOT/summary.json" \
  --raw-output "$OUTPUT_ROOT/raw.jsonl" \
  --prompts "$PROMPTS" \
  --max-seq-len 128 \
  --positions -1 -2 -4 -8 \
  --offsets "${OFFSETS[@]}" \
  --period "$PERIOD" \
  2>&1 | tee "$OUTPUT_ROOT/run.log"
