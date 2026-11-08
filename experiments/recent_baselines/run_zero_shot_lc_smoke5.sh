#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/legalfactaware_recent_baselines_run}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
RESULTS="$ROOT/results/zero_shot_lc_smoke5"
mkdir -p "$RESULTS"

COMMON=(
  --input "$ROOT/data/cail_dev_942.jsonl"
  --model-id "$MODEL"
  --target-chars 450
  --num-candidates 3
  --max-revisions 5
  --num-samples 5
  --max-source-chars 8192
  --source-truncation head_tail
  --max-new-tokens 700
  --quantization 4bit
  --top-k 20
  --top-p 0.8
  --temperature 0.7
  --repetition-penalty 1.05
  --seed 0
)

python3 "$ROOT/run_zero_shot_lc.py" "${COMMON[@]}" \
  --length-objective exact \
  --output "$RESULTS/zero_shot_lc_sf3_ar5_exact_smoke5.jsonl" \
  --sample-output "$RESULTS/zero_shot_lc_sf3_exact_smoke5.jsonl" \
  --trace-output "$RESULTS/zero_shot_lc_sf3_ar5_exact_traces_smoke5.jsonl" \
  2>&1 | tee "$RESULTS/exact.log"

python3 "$ROOT/run_zero_shot_lc.py" "${COMMON[@]}" \
  --length-objective upper_bound \
  --output "$RESULTS/zero_shot_lc_sf3_ar5_upper_smoke5.jsonl" \
  --sample-output "$RESULTS/zero_shot_lc_sf3_upper_smoke5.jsonl" \
  --trace-output "$RESULTS/zero_shot_lc_sf3_ar5_upper_traces_smoke5.jsonl" \
  2>&1 | tee "$RESULTS/upper.log"

python3 "$ROOT/evaluate_cail_baselines.py" \
  --dataset "$ROOT/data/cail_dev_942.jsonl" \
  --limit 5 \
  --prediction "LegalFactAware=$ROOT/data/legalfactaware_predictions.jsonl" \
  --prediction "ZeroShotLC-SF3-exact=$RESULTS/zero_shot_lc_sf3_exact_smoke5.jsonl" \
  --prediction "ZeroShotLC-SF3-AR5-exact=$RESULTS/zero_shot_lc_sf3_ar5_exact_smoke5.jsonl" \
  --prediction "ZeroShotLC-SF3-upper=$RESULTS/zero_shot_lc_sf3_upper_smoke5.jsonl" \
  --prediction "ZeroShotLC-SF3-AR5-upper=$RESULTS/zero_shot_lc_sf3_ar5_upper_smoke5.jsonl" \
  --output-dir "$RESULTS/eval" \
  2>&1 | tee "$RESULTS/eval.log"

