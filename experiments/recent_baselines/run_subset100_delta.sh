#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/legalfactaware_recent_baselines_run
EXP="$ROOT/experiments/recent_baselines_20261031"
PY=/root/autodl-tmp/venvs/recent_baselines_20261031/bin/python
MODEL=/root/autodl-tmp/models/Dream-v0-Instruct-7B
OUT="$EXP/results/subset100"

mkdir -p "$OUT"
cd "$ROOT"

"$PY" "$EXP/run_delta_cail.py" \
  --input "$EXP/data/subset100/cail_dev_stratified_100.jsonl" \
  --output "$OUT/delta_dream_100.jsonl" \
  --official-code-dir "$EXP/vendor/delta-diffusion" \
  --model-id "$MODEL" \
  --model-name dream \
  --target-chars 450 \
  --avg-chars-per-token 1.5263636665452414 \
  --quantization 4bit \
  2>&1 | tee "$OUT/delta_dream_100.log"

"$PY" "$EXP/evaluate_cail_baselines.py" \
  --dataset "$EXP/data/subset100/cail_dev_stratified_100.jsonl" \
  --prediction "LegalFactAware-v2.5-B=$EXP/data/subset100/legalfactaware_stratified_100.jsonl" \
  --prediction "DELTA-Dream=$OUT/delta_dream_100.jsonl" \
  --output-dir "$OUT/eval_delta_100"

