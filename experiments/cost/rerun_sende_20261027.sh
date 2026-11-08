#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/sende_rerun_20261027}
CODE=${CODE:-/root/autodl-tmp/legalfactaware_full_runtime_20261021/code}
RUNNER="$CODE/run_full_runtime.sh"
INFER="$CODE/benchmark_unified_inference.py"
DATA=/root/legalfactaware_runs_20260813/data/cail2020-clean-20260811
PY=/root/miniconda3/bin/python

mkdir -p "$BASE/pipeline" "$BASE/frozen_models/sende" "$BASE/unified_inference"
printf 'repeat\tstart\tend\texit_code\tpredictions\twall_seconds\n' > "$BASE/pipeline_manifest.tsv"

for repeat in 1 2 3; do
  root="$BASE/pipeline/run_$repeat"
  if [[ -e "$root" ]]; then
    mv "$root" "$root.incomplete.$(date +%Y%m%dT%H%M%S)"
  fi
  mkdir -p "$root"
  start=$(date -Is)
  printf '%s\tpipeline_%s\tSTART\n' "$start" "$repeat" > "$BASE/STATUS"
  keep=0
  [[ "$repeat" == 1 ]] && keep=1
  set +e
  "$PY" "$CODE/time_command.py" --output "$root/wall_time.json" -- \
    env ROOT="$root" CODE="$CODE" OUT="$root/outputs" KEEP_SENDE_MODEL="$keep" \
    "$RUNNER" sende > "$root/pipeline.log" 2>&1
  rc=$?
  set -e
  end=$(date -Is)
  pred="$root/outputs/sende/dev_predictions.jsonl"
  count=0
  [[ -f "$pred" ]] && count=$(wc -l < "$pred")
  wall=$(
    "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["real_seconds"])' \
      "$root/wall_time.json" 2>/dev/null || printf 'NA'
  )
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$repeat" "$start" "$end" "$rc" "$count" "$wall" >> "$BASE/pipeline_manifest.tsv"
  if [[ "$rc" -ne 0 || "$count" -ne 942 ]]; then
    printf '%s\tpipeline_%s\tFAILED\trc=%s\tcount=%s\n' "$end" "$repeat" "$rc" "$count" > "$BASE/STATUS"
    exit 1
  fi
  sha256sum "$pred" > "$root/prediction_sha256.txt"
  if [[ "$repeat" == 1 ]]; then
    rm -rf "$BASE/frozen_models/sende/tsdae"
    mv "$root/outputs/sende/tsdae" "$BASE/frozen_models/sende/tsdae"
    cp "$root/outputs/sende/extractor/best.pt" "$BASE/frozen_models/sende/extractor_best.pt"
    find "$BASE/frozen_models/sende" -type f -print0 | sort -z | xargs -0 sha256sum > "$BASE/frozen_models/sende/SHA256SUMS.txt"
  fi
  rm -rf "$root/outputs/sende/oracles" "$root/outputs/sende/features" "$root/outputs/sende/tsdae" "$root/outputs/sende/extractor"
  df -h / /root/autodl-tmp > "$root/disk_after_cleanup.txt"
  printf '%s\tpipeline_%s\tCOMPLETE\n' "$end" "$repeat" > "$BASE/STATUS"
done

printf '%s\tunified_inference\tSTART\n' "$(date -Is)" > "$BASE/STATUS"
"$PY" "$INFER" sende --repeats 5 --output "$BASE/unified_inference" \
  --frozen "$BASE/frozen_models" --train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl" \
  --code-dir "$CODE" > "$BASE/unified_inference/sende.stdout.log" \
  2> "$BASE/unified_inference/sende.stderr.log"

for repeat in 1 2 3 4 5; do
  test "$(wc -l < "$BASE/unified_inference/sende/run_$repeat/dev_predictions.jsonl")" -eq 942
  test -s "$BASE/unified_inference/sende/run_$repeat/run_record.json"
done

printf '%s\tALL_COMPLETE\n' "$(date -Is)" > "$BASE/STATUS"
touch "$BASE/COMPLETE"
