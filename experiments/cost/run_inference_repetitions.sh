#!/usr/bin/env bash
set -euo pipefail

BENCH=${BENCH:-/root/autodl-tmp/legalfactaware_cost_analysis_20261023}
CODE=${CODE:-/root/autodl-tmp/legalfactaware_full_runtime_20261021/code}
RUNNER="$CODE/benchmark_unified_inference.py"
OUT="$BENCH/unified_inference"
FROZEN="$BENCH/frozen_models"
DATA=/root/legalfactaware_runs_20260813/data/cail2020-clean-20260811
PY=/root/miniconda3/bin/python
TERMPY=/root/autodl-tmp/conda_envs/termdiff37/bin/python

mkdir -p "$OUT"
printf '%s\tUNIFIED_INFERENCE_START\n' "$(date -Is)" > "$BENCH/INFERENCE_STATUS"

run_method() {
  local method="$1" python="$2"
  if [[ -s "$OUT/$method/manifest.json" ]] && grep -q '"status": "complete"' "$OUT/$method/manifest.json"; then
    printf '%s\t%s\tSKIP_COMPLETE\n' "$(date -Is)" "$method" > "$BENCH/INFERENCE_STATUS"
    return
  fi
  if [[ -d "$OUT/$method" ]]; then
    mv "$OUT/$method" "$OUT/${method}.incomplete.$(date +%Y%m%dT%H%M%S)"
  fi
  printf '%s\t%s\tSTART\n' "$(date -Is)" "$method" > "$BENCH/INFERENCE_STATUS"
  "$python" "$RUNNER" "$method" --repeats 5 --output "$OUT" --frozen "$FROZEN" \
    --train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl" --code-dir "$CODE" \
    > "$OUT/${method}.stdout.log" 2> "$OUT/${method}.stderr.log"
  # Count only the five formal run directories.  Some implementations retain
  # method-internal working predictions that must not affect validation.
  for repeat in 1 2 3 4 5; do
    test "$(wc -l < "$OUT/$method/run_$repeat/dev_predictions.jsonl")" -eq 942
    test -s "$OUT/$method/run_$repeat/run_record.json"
  done
  printf '%s\t%s\tCOMPLETE\n' "$(date -Is)" "$method" > "$BENCH/INFERENCE_STATUS"
}

# Interleave lightweight and heavy methods while keeping exclusive execution.
run_method sse "$PY"
run_method legalfactaware "$PY"
run_method sende "$PY"
run_method memsum "$PY"
run_method termdiffusum "$TERMPY"

printf '%s\tALL_UNIFIED_INFERENCE_COMPLETE\n' "$(date -Is)" > "$BENCH/INFERENCE_STATUS"
touch "$BENCH/INFERENCE_COMPLETE"
