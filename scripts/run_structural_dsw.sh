#!/usr/bin/env bash
# Run the completed Ouro and Huginn-primary structural analyses on DSW.
set -euo pipefail

PYTHON=${PYTHON:-/tmp/jlens-analysis/bin/python}
REPO=${REPO:-/mnt/nas-save/wangjunqi/j-lens-on-loop-qwen}
FIT_ROOT=${FIT_ROOT:-/mnt/nas-save/wangjunqi/j-lens-on-loop/runs/paper-fit-20261020T034458Z}
OUTPUT=${OUTPUT:?set OUTPUT to a durable result directory}
OPERATOR_ITERATIONS=${OPERATOR_ITERATIONS:-60}

OURO=$FIT_ROOT/ouro-paper
HUGINN=$FIT_ROOT/huginn-primary
mkdir -p "$OUTPUT"
cd "$REPO"

"$PYTHON" scripts/analyze_transport.py "$OURO" \
  --output "$OUTPUT/ouro-distance.json" \
  --operator-iterations "$OPERATOR_ITERATIONS"
"$PYTHON" scripts/analyze_structure.py "$OURO" \
  --output "$OUTPUT/ouro-symmetry.json" \
  --mode symmetry --period 48 \
  --operator-iterations "$OPERATOR_ITERATIONS"
"$PYTHON" scripts/analyze_structure.py "$OURO" \
  --output "$OUTPUT/ouro-factorisation.json" \
  --mode factorisation --period 48 \
  --operator-iterations "$OPERATOR_ITERATIONS"

"$PYTHON" scripts/analyze_transport.py "$HUGINN" \
  --output "$OUTPUT/huginn-distance.json" \
  --operator-iterations "$OPERATOR_ITERATIONS"
"$PYTHON" scripts/analyze_structure.py "$HUGINN" \
  --output "$OUTPUT/huginn-symmetry.json" \
  --mode symmetry --period 4 \
  --operator-iterations "$OPERATOR_ITERATIONS"
"$PYTHON" scripts/analyze_structure.py "$HUGINN" \
  --output "$OUTPUT/huginn-factorisation.json" \
  --mode factorisation --period 4 \
  --operator-iterations "$OPERATOR_ITERATIONS"

echo "RESULT: PASS"
