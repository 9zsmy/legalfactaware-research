#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/legalfactaware_runtime_benchmark_20261021
CODE="$ROOT/code"
OUT="$ROOT/outputs"
mkdir -p "$OUT"

# The recovered TermDiffuSum bytecode reads this historical fixed directory.
# Later IN-Abs work repointed it, so switch it to the archived CAIL adapter for
# this benchmark and restore the previous targets on every exit path.
LEGAL_DIR=/media/dxy/DiffuSum-main/datasets/legal_last40
OLD_TRAIN=$(readlink "$LEGAL_DIR/train.jsonl")
OLD_VAL=$(readlink "$LEGAL_DIR/val.jsonl")
OLD_TEST=$(readlink "$LEGAL_DIR/test.jsonl")
restore_links() {
  ln -sfn "$OLD_TRAIN" "$LEGAL_DIR/train.jsonl"
  ln -sfn "$OLD_VAL" "$LEGAL_DIR/val.jsonl"
  ln -sfn "$OLD_TEST" "$LEGAL_DIR/test.jsonl"
}
trap restore_links EXIT INT TERM
ln -sfn /root/autodl-tmp/termdiffusum_reproduction_20260817/data_cail_full/train.jsonl "$LEGAL_DIR/train.jsonl"
ln -sfn /root/autodl-tmp/termdiffusum_reproduction_20260817/data_cail_full/val.jsonl "$LEGAL_DIR/val.jsonl"
ln -sfn /root/autodl-tmp/termdiffusum_reproduction_20260817/data_cail_full/test.jsonl "$LEGAL_DIR/test.jsonl"

run_one() {
  local repeat="$1"
  local run_dir="$OUT/termdiffusum_r${repeat}"
  local command="/root/autodl-tmp/conda_envs/termdiff37/bin/python $CODE/benchmark_termdiff_dev.py --runtime-dir /root/autodl-tmp/termdiffusum_reproduction_20260817/runtime --adapted-data-dir /root/autodl-tmp/termdiffusum_reproduction_20260817/data_cail_full --checkpoint /root/autodl-tmp/termdiffusum_reproduction_20260817/outputs/cail_full_10epoch/checkpoint_latest.pt --bert-config-dir /root/autodl-tmp/fedsum_reproduction_20260826/models/bert-base-uncased --output-dir $run_dir --batch-size 6"
  mkdir -p "$run_dir"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 > "$run_dir/gpu.csv" 2>/dev/null &
  local monitor_pid=$!
  TIMEFORMAT='real_seconds=%R user_seconds=%U sys_seconds=%S'
  { time bash -lc "$command" > "$run_dir/stdout.log" 2> "$run_dir/stderr.log"; } 2> "$run_dir/time.txt"
  local status=$?
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  printf '%s\n' "$status" > "$run_dir/exitcode"
  printf 'termdiffusum repeat=%s exit=%s\n' "$repeat" "$status"
  return "$status"
}

for repeat in 1 2 3; do
  run_one "$repeat" || exit 15
done
touch "$ROOT/COMPLETE"
