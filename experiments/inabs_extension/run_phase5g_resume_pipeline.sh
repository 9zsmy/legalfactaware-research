#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/inabs_lfa_phase2
REPO=/root/autodl-tmp/JUSTNLP_LSUMM_BudgetRecipe_2025
WORK=/root/inabs_phase5g_budgetrecipe
MODEL=/dev/shm/inabs_budgetrecipe_model
EXTRA=/dev/shm/inabs_budgetrecipe_model_extra
PARTIAL="$MODEL/.cache/huggingface/download/IO4xwqmZYzFmxznkwkiNSBwO1H0=.9bcf0b74b81c775a92d104c3a4c0ba9a8465733c8c9d8f6b0b1282cc95acf1c3.incomplete"
EXPECTED=4990095032

trap 'rc=$?; printf "%s\n" "$rc" > "$WORK/resume_pipeline.exitcode"' EXIT

for attempt in $(seq 1 120); do
  size=$(stat -c '%s' "$PARTIAL" 2>/dev/null || printf '0')
  if [[ "$size" -eq "$EXPECTED" ]]; then
    break
  fi
  if [[ -s "$MODEL/model-00001-of-00004.safetensors" ]]; then
    break
  fi
  if [[ "$size" -gt "$EXPECTED" ]]; then
    echo "Shard 1 exceeds expected size: $size" >&2
    exit 1
  fi
  echo "Waiting for shard 1: $size/$EXPECTED bytes"
  sleep 30
done

size=$(stat -c '%s' "$PARTIAL" 2>/dev/null || printf '0')
if [[ "$size" -ne "$EXPECTED" && ! -s "$MODEL/model-00001-of-00004.safetensors" ]]; then
  echo "Shard 1 did not finish in time" >&2
  exit 1
fi

cd "$REPO"
for attempt in $(seq 1 40); do
  echo "Checkpoint download/verification attempt $attempt"
  if HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=600 \
    python3 reproduction/download_model_split.py \
      --primary-dir "$MODEL" --secondary-dir "$EXTRA"; then
    echo "All checkpoint shards verified by huggingface_hub"
    break
  fi
  if [[ "$attempt" -eq 40 ]]; then
    echo "Checkpoint download exhausted retries" >&2
    exit 1
  fi
  sleep 15
done

cd "$ROOT"
PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /root/autodl-tmp/venvs/budgetrecipe/bin/python run_inabs_budgetrecipe.py \
    --model "$MODEL" --dev /root/inabs_phase5d_sende/data/dev.jsonl \
    --repo "$REPO" --output-dir "$WORK/smoke" --max-new-tokens 1024 --limit 3

python3 -c "import json; p='$WORK/smoke/records.jsonl'; x=[json.loads(s) for s in open(p)]; assert len(x)==3 and any(not r['empty'] for r in x), x"

bash run_phase5g_budgetrecipe.sh
