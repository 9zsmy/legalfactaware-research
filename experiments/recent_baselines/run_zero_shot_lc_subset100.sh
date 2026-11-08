#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/legalfactaware_recent_baselines_run}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
RESULTS="$ROOT/results/subset100_zero_shot_lc"
mkdir -p "$RESULTS"

python3 "$ROOT/run_zero_shot_lc.py" \
  --input "$ROOT/data/subset100/cail_dev_stratified_100.jsonl" \
  --model-id "$MODEL" \
  --target-chars 450 \
  --length-objective upper_bound \
  --num-candidates 3 \
  --max-revisions 5 \
  --max-source-chars 8192 \
  --source-truncation head_tail \
  --max-new-tokens 700 \
  --quantization 4bit \
  --top-k 20 \
  --top-p 0.8 \
  --temperature 0.7 \
  --repetition-penalty 1.05 \
  --seed 0 \
  --output "$RESULTS/zero_shot_lc_sf3_ar5_upper_100.jsonl" \
  --sample-output "$RESULTS/zero_shot_lc_sf3_upper_100.jsonl" \
  --trace-output "$RESULTS/zero_shot_lc_sf3_ar5_upper_traces_100.jsonl" \
  2>&1 | tee "$RESULTS/run.log"

python3 "$ROOT/evaluate_cail_baselines.py" \
  --dataset "$ROOT/data/subset100/cail_dev_stratified_100.jsonl" \
  --prediction "LegalFactAware=$ROOT/data/subset100/legalfactaware_stratified_100.jsonl" \
  --prediction "ZeroShotLC-SF3-AR5-upper=$RESULTS/zero_shot_lc_sf3_ar5_upper_100.jsonl" \
  --output-dir "$RESULTS/eval" \
  2>&1 | tee "$RESULTS/eval.log"

