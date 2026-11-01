#!/usr/bin/env bash
set -euo pipefail

NAS=/mnt/nas-save/nas-save
REPO=${REPO:-$NAS/wangjunqi/j-lens-on-loop-qwen}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
BAND_KIND=${BAND_KIND:-all-loops}
STRENGTH=${STRENGTH:-1.0}
OUT=$NAS/wangjunqi/j-lens-on-loop/runs/$RUN_NAME
BASE=$REPO/configs/experiments/ouro-top-down-summoning.json
CONFIG=$OUT/config.json

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
export PYTHONPATH="$REPO/src:$NAS/wangjunqi/deps/jacobian-lens:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python3 -m pip install -q --no-deps \
  -e "$NAS/wangjunqi/deps/jacobian-lens" -e "$REPO"
python3 - "$BASE" "$CONFIG" "$BAND_KIND" "$STRENGTH" <<'PY'
import json
import sys
from pathlib import Path

base, output, band_kind, strength = sys.argv[1:]
config = json.loads(Path(base).read_text())
if band_kind == "all-loops":
    layers = [
        layer
        for loop_start in (0, 48, 96, 144)
        for layer in range(loop_start + 24, loop_start + 47)
    ]
elif band_kind == "final-loop":
    layers = list(range(168, 191))
else:
    raise ValueError(f"unsupported band: {band_kind}")
config["band_layers"] = layers
config["strength"] = float(strength)
Path(output).write_text(json.dumps(config, indent=2) + "\n")
PY

cd "$REPO"
python3 scripts/run_top_down_summoning.py \
  --config "$CONFIG" \
  --output "$OUT"

echo "RESULT: PASS"
echo "OUT=$OUT"
