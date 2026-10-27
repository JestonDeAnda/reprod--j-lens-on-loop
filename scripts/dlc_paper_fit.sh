#!/usr/bin/env bash
# Paper-scale Jacobian-lens family fit (arXiv:2609.01924).
# Runs on a DLC worker that can see /mnt/nas-save. Resumable via fit() checkpoints.
#
# STAGE selects the work unit (one DLC job per stage is the intended use):
#   ouro              — Ouro loops 1–4, full sources, 1000 WikiText prompts
#   huginn-primary    — Huginn r=4/8/12/16, full sources
#   huginn-tiling     — Huginn r=1..16 gap-fill tiling, full sources
#   all               — run the three stages sequentially (long)
set -euo pipefail

NAS=/mnt/nas-save/nas-save
ROOT=$NAS/wangjunqi/j-lens-on-loop
OURO_MODEL=$NAS/wangjunqi/ouro/models/Ouro-2.6B
HUGINN_MODEL=$ROOT/models/huginn-0125
JLENS_SRC=$NAS/wangjunqi/deps/jacobian-lens
OSS=${OSS_ROOT:-/mnt/oss-save/wangjunqi/j-lens-on-loop}
STAGE=${STAGE:-all}
PROMPTS=${PROMPTS:-1000}
DIM_BATCH=${DIM_BATCH:-8}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-128}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5}
RUN_NAME=${RUN_NAME:-paper-fit-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$ROOT/runs/$RUN_NAME
OSS_OUT=$OSS/runs/$RUN_NAME

if [[ ! -f $ROOT/pyproject.toml || ! -f $OURO_MODEL/config.json || ! -f $JLENS_SRC/pyproject.toml ]]; then
  echo "persistent paths are not visible"
  ls -la /mnt/nas-save | head -40
  ls -la "$NAS/wangjunqi" || true
  exit 1
fi

mkdir -p "$OUT" "$OSS_OUT" || true
exec > >(tee -a "$OUT/paper-fit.log") 2>&1

copy_artifacts() {
  mkdir -p "$OSS_OUT" 2>/dev/null || true
  if [[ -d $OSS_OUT ]]; then
    cp -f "$OUT/paper-fit.log" "$OSS_OUT/" 2>/dev/null || true
    # Small metadata only; large *.lens.pt / *.checkpoint.pt stay on NAS.
    find "$OUT" -name 'metadata.json' -print0 2>/dev/null \
      | while IFS= read -r -d '' f; do
          rel=${f#"$OUT/"}
          mkdir -p "$OSS_OUT/$(dirname "$rel")"
          cp -f "$f" "$OSS_OUT/$rel" || true
        done
  fi
}
trap copy_artifacts EXIT

echo "=== host ==="
hostname
date -u
echo "STAGE=$STAGE RUN_NAME=$RUN_NAME OUT=$OUT"
nvidia-smi
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY

export HF_HOME=${HF_HOME:-$ROOT/hf-home}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

echo "=== install ==="
python3 -m pip install --upgrade-strategy only-if-needed -e "$JLENS_SRC"
python3 -m pip install --upgrade-strategy only-if-needed datasets pyyaml tqdm
python3 -m pip install --upgrade-strategy only-if-needed -e "$ROOT" --no-deps

if [[ ! -f $HUGINN_MODEL/config.json ]]; then
  echo "=== download huginn into NAS ==="
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("tomg-group-umd/huginn-0125", local_dir="${HUGINN_MODEL}")
PY
fi

cd "$ROOT"

PROMPT_CACHE=$ROOT/cache/wikitext-${PROMPTS}x${MAX_SEQ_LEN}.json
mkdir -p "$(dirname "$PROMPT_CACHE")"

run_ouro() {
  echo "=== paper fit ouro: loops 1-4, full sources, ${PROMPTS} prompts ==="
  python3 scripts/fit_family.py ouro \
    --model "$OURO_MODEL" \
    --output "$OUT/ouro-paper" \
    --prompts "$PROMPTS" \
    --prompt-cache "$PROMPT_CACHE" \
    --full-sources \
    --max-seq-len "$MAX_SEQ_LEN" \
    --dim-batch "$DIM_BATCH" \
    --checkpoint-every "$CHECKPOINT_EVERY"
}

run_huginn_primary() {
  echo "=== paper fit huginn-primary: r=4/8/12/16, full sources, ${PROMPTS} prompts ==="
  python3 scripts/fit_family.py huginn \
    --model "$HUGINN_MODEL" \
    --output "$OUT/huginn-primary" \
    --prompts "$PROMPTS" \
    --prompt-cache "$PROMPT_CACHE" \
    --recurrences 16 \
    --targets 4 8 12 16 \
    --full-sources \
    --max-seq-len "$MAX_SEQ_LEN" \
    --dim-batch "$DIM_BATCH" \
    --checkpoint-every "$CHECKPOINT_EVERY"
}

run_huginn_tiling() {
  local targets=${TILING_TARGETS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16}
  echo "=== paper fit huginn-tiling: targets=${targets}, full sources, ${PROMPTS} prompts ==="
  python3 scripts/fit_family.py huginn \
    --model "$HUGINN_MODEL" \
    --output "$OUT/huginn-tiling" \
    --prompts "$PROMPTS" \
    --prompt-cache "$PROMPT_CACHE" \
    --recurrences 16 \
    --targets $targets \
    --full-sources \
    --max-seq-len "$MAX_SEQ_LEN" \
    --dim-batch "$DIM_BATCH" \
    --checkpoint-every "$CHECKPOINT_EVERY"
}

case "$STAGE" in
  ouro) run_ouro ;;
  huginn-primary) run_huginn_primary ;;
  huginn-tiling) run_huginn_tiling ;;
  all)
    run_ouro
    run_huginn_primary
    run_huginn_tiling
    ;;
  *)
    echo "unknown STAGE=$STAGE (expected ouro|huginn-primary|huginn-tiling|all)"
    exit 2
    ;;
esac

echo "=== artefacts ==="
find "$OUT" -type f -printf '%s %p\n' | sort
python3 - <<PY
import json
from pathlib import Path
out = Path("$OUT")
summary = {"stage": "$STAGE", "run": "$RUN_NAME", "lenses": []}
for lens in sorted(out.rglob("*.lens.pt")):
    meta = lens.parent / "metadata.json"
    summary["lenses"].append({
        "path": str(lens),
        "bytes": lens.stat().st_size,
        "metadata": json.loads(meta.read_text()) if meta.exists() else None,
    })
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "RESULT: PASS"
echo "OUT=$OUT"
echo "OSS_OUT=$OSS_OUT"
