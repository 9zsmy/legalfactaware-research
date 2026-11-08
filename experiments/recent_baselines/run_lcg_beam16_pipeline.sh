#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/legalfactaware_recent_baselines_run
EXP="$ROOT/experiments/recent_baselines_20261031"
MODEL=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
LOG=/root/autodl-tmp/logs_lcg_beam16
REVISION=a09a35458c702b33eeacc393d103063234e8bc28
BASE="https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct/resolve/$REVISION"

mkdir -p "$MODEL" "$LOG"
rm -f "$LOG/DOWNLOAD_FAILED" "$LOG/MODEL_VERIFIED" "$LOG/COMPLETE"

download_shard() {
  local shard="$1"
  aria2c -c -x 8 -s 8 -k 1M --file-allocation=none --max-tries=0 --retry-wait=3 \
    -d "$MODEL" -o "model-0000${shard}-of-00004.safetensors" \
    "$BASE/model-0000${shard}-of-00004.safetensors?download=true" \
    > "$LOG/qwen_shard${shard}.log" 2>&1
}

download_shard 1 & p1=$!
download_shard 2 & p2=$!
download_shard 3 & p3=$!
download_shard 4 & p4=$!

failed=0
for pid in "$p1" "$p2" "$p3" "$p4"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  touch "$LOG/DOWNLOAD_FAILED"
  exit 1
fi

exec "$EXP/start_lcg_beam16_after_download.sh" "$p1" "$p2" "$p3" "$p4"

