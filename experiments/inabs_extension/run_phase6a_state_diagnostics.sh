#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/inabs_lfa_phase2
WORK=/root/inabs_phase6a_state_diagnostics

mkdir -p "$WORK"
trap 'status=$?; printf "%s\n" "$status" > "$WORK/exitcode"' EXIT
test ! -e "$WORK/run"

cd "$ROOT"
PYTHONPATH=vendor_pyrouge11 python3 run_inabs_phase6a_state_diagnostics.py \
  --dataset-dir dataset/IN-Abs \
  --phase3-model outputs/phase3_static_full/model_bundle.pkl \
  --output-dir "$WORK/run" 2>&1 | tee "$WORK/run.log"

tar -czf "$WORK/phase6a_state_diagnostics_compact_results.tgz" \
  -C "$ROOT" PREREGISTRATION_PHASE6A_STATE_DIAGNOSTICS.md \
  run_inabs_phase6a_state_diagnostics.py run_phase6a_state_diagnostics.sh \
  -C "$WORK" run/dev_records.jsonl run/metrics.json run.log
