#!/usr/bin/env bash
# Paper-protocol Qwen3.6-27B final-target and L24-target J-Lens fits.
set -euo pipefail

NAS=/mnt/nas-save/nas-save
ROOT=$NAS/wangjunqi/j-lens-on-loop-qwen
MODEL=$ROOT/models/Qwen3.6-27B
JLENS_SRC=$NAS/wangjunqi/deps/jacobian-lens
PROMPT_CACHE=$NAS/wangjunqi/j-lens-on-loop/cache/wikitext-1000x128.json
RUN_NAME=${RUN_NAME:-qwen-full-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$ROOT/runs/$RUN_NAME
OSS_OUT=/mnt/oss-save/wangjunqi/j-lens-on-loop/runs/$RUN_NAME

for path in "$ROOT/pyproject.toml" "$MODEL/model.safetensors.index.json" \
  "$JLENS_SRC/pyproject.toml" "$PROMPT_CACHE"; do
  if [[ ! -f $path ]]; then
    echo "required persistent file is missing: $path"
    exit 1
  fi
done

mkdir -p "$OUT" "$OSS_OUT"
exec > >(tee -a "$OUT/fit.log") 2>&1

copy_artifacts() {
  cp -f "$OUT/fit.log" "$OSS_OUT/" 2>/dev/null || true
  find "$OUT" -maxdepth 2 -type f \( -name 'metadata.json' -o -name 'summary.json' \) \
    -exec cp -f {} "$OSS_OUT/" \; 2>/dev/null || true
}
trap copy_artifacts EXIT

hostname
date -u
nvidia-smi

export HF_HOME=$ROOT/hf-home
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python3 -m pip install --upgrade-strategy only-if-needed -e "$JLENS_SRC"
python3 -m pip install "transformers==5.20.1" "accelerate>=1.0" datasets pyyaml tqdm
python3 -m pip install --upgrade-strategy only-if-needed -e "$ROOT" --no-deps

cd "$ROOT"
python3 scripts/fit_qwen.py \
  --model "$MODEL" \
  --output "$OUT/lenses" \
  --prompts 1000 \
  --prompt-cache "$PROMPT_CACHE" \
  --targets final band-entry \
  --max-seq-len 128 \
  --dim-batch 8 \
  --checkpoint-every 5

python3 - <<PY
import json
from pathlib import Path

out = Path("$OUT")
lenses = [
    {"path": str(path), "bytes": path.stat().st_size}
    for path in sorted(out.rglob("*.lens.pt"))
]
summary = {
    "model": "Qwen/Qwen3.6-27B",
    "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    "run": "$RUN_NAME",
    "prompts": 1000,
    "max_seq_len": 128,
    "targets": ["final", "band-entry"],
    "lenses": lenses,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "RESULT: PASS"
echo "OUT=$OUT"
echo "OSS_OUT=$OSS_OUT"
