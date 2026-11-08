#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/sende_rerun_20261027/dev_selected
DATA=/root/legalfactaware_runs_20260813/data/cail2020-clean-20260811
SENDE=/root/autodl-tmp/sende_independent_20260826
FROZEN=/root/autodl-tmp/sende_rerun_20261027/frozen_models/sende
PY=/root/miniconda3/bin/python

mkdir -p "$BASE/records" "$BASE/features" "$BASE/evaluation"
printf '%s\tPREPARE_RECORDS\n' "$(date -Is)" > "$BASE/STATUS"
"$PY" "$SENDE/prepare_cail_oracles.py" \
  --data-dir "$DATA" --output-dir "$BASE/records" \
  --splits dev test --workers 12 \
  > "$BASE/prepare.stdout.log" 2> "$BASE/prepare.stderr.log"

printf '%s\tENCODE_DEV\n' "$(date -Is)" > "$BASE/STATUS"
"$PY" "$SENDE/encode_records.py" \
  --records "$BASE/records/dev.jsonl" --model "$FROZEN/tsdae" \
  --output-prefix "$BASE/features/dev" --batch-size 256 \
  > "$BASE/encode_dev.stdout.log" 2> "$BASE/encode_dev.stderr.log"

printf '%s\tENCODE_TEST\n' "$(date -Is)" > "$BASE/STATUS"
"$PY" "$SENDE/encode_records.py" \
  --records "$BASE/records/test.jsonl" --model "$FROZEN/tsdae" \
  --output-prefix "$BASE/features/test" --batch-size 256 \
  > "$BASE/encode_test.stdout.log" 2> "$BASE/encode_test.stderr.log"

printf '%s\tDEV_SELECT_AND_TEST\n' "$(date -Is)" > "$BASE/STATUS"
"$PY" "$SENDE/evaluate_cail.py" \
  --dev-records "$BASE/records/dev.jsonl" \
  --dev-prefix "$BASE/features/dev" \
  --test-records "$BASE/records/test.jsonl" \
  --test-prefix "$BASE/features/test" \
  --checkpoint "$FROZEN/extractor_best.pt" \
  --output-dir "$BASE/evaluation" \
  > "$BASE/evaluate.stdout.log" 2> "$BASE/evaluate.stderr.log"

test "$(wc -l < "$BASE/evaluation/test_predictions.jsonl")" -eq 948
sha256sum "$BASE/evaluation/test_predictions.jsonl" > "$BASE/evaluation/test_predictions.sha256"
printf '%s\tCOMPLETE\n' "$(date -Is)" > "$BASE/STATUS"
touch "$BASE/COMPLETE"
