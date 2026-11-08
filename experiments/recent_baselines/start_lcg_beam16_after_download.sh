#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 SHARD1_PID SHARD2_PID SHARD3_PID SHARD4_PID" >&2
  exit 2
fi

PIDS=("$@")
ROOT=/root/autodl-tmp/legalfactaware_recent_baselines_run
EXP="$ROOT/experiments/recent_baselines_20261031"
PY=/root/autodl-tmp/venvs/recent_baselines_20261031/bin/python
MODEL=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
STATUS=/root/autodl-tmp/logs_lcg_beam16
REVISION=a09a35458c702b33eeacc393d103063234e8bc28

mkdir -p "$STATUS"
rm -f "$STATUS/DOWNLOAD_FAILED" "$STATUS/MODEL_VERIFIED" "$STATUS/COMPLETE"

while true; do
  alive=0
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
      break
    fi
  done
  [[ "$alive" -eq 0 ]] && break
  sleep 30
done

check_file() {
  local filename="$1"
  local expected_size="$2"
  local expected_sha="$3"
  local path="$MODEL/$filename"
  if [[ ! -f "$path" ]]; then
    echo "missing $path" >&2
    touch "$STATUS/DOWNLOAD_FAILED"
    exit 1
  fi
  local actual_size
  actual_size=$(stat -c %s "$path")
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "$filename size mismatch: $actual_size != $expected_size" >&2
    touch "$STATUS/DOWNLOAD_FAILED"
    exit 1
  fi
  local actual_sha
  actual_sha=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "$filename sha256 mismatch: $actual_sha != $expected_sha" >&2
    touch "$STATUS/DOWNLOAD_FAILED"
    exit 1
  fi
  echo "$filename verified: $actual_size bytes $actual_sha"
}

check_file model-00001-of-00004.safetensors 3945441440 a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7
check_file model-00002-of-00004.safetensors 3864726352 f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185
check_file model-00003-of-00004.safetensors 3864726424 8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5
check_file model-00004-of-00004.safetensors 3556377672 1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd

HF_ENDPOINT=https://hf-mirror.com "$PY" -c \
  "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-7B-Instruct', revision='$REVISION', local_dir='$MODEL', allow_patterns=['*.json', '*.txt'], max_workers=4)"

for required in config.json generation_config.json model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  if [[ ! -s "$MODEL/$required" ]]; then
    echo "missing required metadata file: $required" >&2
    touch "$STATUS/DOWNLOAD_FAILED"
    exit 1
  fi
done

touch "$STATUS/MODEL_VERIFIED"
chmod +x "$EXP/run_lcg_beam16_smoke.sh"
"$EXP/run_lcg_beam16_smoke.sh"
touch "$STATUS/COMPLETE"

