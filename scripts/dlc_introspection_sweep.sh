#!/usr/bin/env bash
set -euo pipefail

NAS=/mnt/nas-save/nas-save
REPO=$NAS/wangjunqi/j-lens-on-loop-qwen
RUN_NAME=${RUN_NAME:?set RUN_NAME}
OUT=$NAS/wangjunqi/j-lens-on-loop/runs/$RUN_NAME
BASE=$REPO/configs/experiments/ouro-verbal-introspection.json

mkdir -p "$OUT/configs"
exec > >(tee -a "$OUT/run.log") 2>&1
export PYTHONPATH="$REPO/src:$NAS/wangjunqi/deps/jacobian-lens:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python3 -m pip install -q --no-deps \
  -e "$NAS/wangjunqi/deps/jacobian-lens" -e "$REPO"

python3 - <<PY
import json
from pathlib import Path

base = json.loads(Path("$BASE").read_text())
out = Path("$OUT/configs")
bands = {
    "single": [179],
    "final-loop": list(range(168, 191)),
    "all-loops": [
        layer
        for loop_start in (0, 48, 96, 144)
        for layer in range(loop_start + 24, loop_start + 47)
    ],
}
conditions = [
    {"name": f"target_{strength:.3f}", "direction": "target", "strength": strength}
    for strength in (0.0, 0.025, 0.05, 0.1, 0.15, 0.2)
] + [
    {"name": "random_0.100", "direction": "random", "strength": 0.1},
    {"name": "unrelated_0.100", "direction": "unrelated", "strength": 0.1},
]
for band, layers in bands.items():
    payload = {**base, "band_layers": layers, "conditions": conditions}
    (out / f"{band}.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

cd "$REPO"
for config in "$OUT"/configs/*.json; do
  name=$(basename "$config" .json)
  python3 scripts/run_verbal_introspection.py \
    --config "$config" \
    --output "$OUT/$name"
done

python3 - <<PY
import json
from pathlib import Path

root = Path("$OUT")
rows = []
for path in sorted(root.glob("*/summary.json")):
    rows.append({"band": path.parent.name, **json.loads(path.read_text())})
(root / "sweep-summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps({"bands": len(rows)}, indent=2))
PY

echo "RESULT: PASS"
echo "OUT=$OUT"
