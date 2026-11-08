#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/legalfactaware_recent_baselines_run
EXP="$ROOT/experiments/recent_baselines_20261031"
PY=/root/autodl-tmp/venvs/recent_baselines_20261031/bin/python
MODEL=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
OUT="$EXP/results/subset100"

mkdir -p "$OUT"
cd "$ROOT"

"$PY" -m unittest discover -s "$EXP/tests" -v

"$PY" "$EXP/run_lcg_blackbox.py" \
  --input "$EXP/data/subset100/cail_dev_stratified_100.jsonl" \
  --demo "$EXP/data/lcg_demo_train_row0.jsonl" \
  --output "$OUT/lcg_mhis_qwen25_upper_100.jsonl" \
  --initial-output "$OUT/qwen25_direct_upper_100.jsonl" \
  --model-id "$MODEL" \
  --target-chars 450 \
  --length-objective upper_bound \
  --max-iterations 5 \
  --quantization 4bit \
  --top-k 20 --top-p 0.8 --temperature 0.7 \
  --repetition-penalty 1.05 --seed 0 \
  2>&1 | tee "$OUT/lcg_upper_100.log"

"$PY" "$EXP/evaluate_cail_baselines.py" \
  --dataset "$EXP/data/subset100/cail_dev_stratified_100.jsonl" \
  --prediction "LegalFactAware-v2.5-B=$EXP/data/subset100/legalfactaware_stratified_100.jsonl" \
  --prediction "Qwen2.5-Direct-Upper=$OUT/qwen25_direct_upper_100.jsonl" \
  --prediction "LCG-MHIS-Upper=$OUT/lcg_mhis_qwen25_upper_100.jsonl" \
  --output-dir "$OUT/eval_lcg_upper_100"

"$PY" "$EXP/run_lcg_blackbox.py" \
  --input "$EXP/data/subset100/cail_dev_stratified_100.jsonl" \
  --demo "$EXP/data/lcg_demo_train_row0.jsonl" \
  --output "$OUT/lcg_mhis_qwen25_exact_100.jsonl" \
  --initial-output "$OUT/qwen25_direct_exact_100.jsonl" \
  --model-id "$MODEL" \
  --target-chars 450 \
  --length-objective exact \
  --max-iterations 5 \
  --quantization 4bit \
  --top-k 20 --top-p 0.8 --temperature 0.7 \
  --repetition-penalty 1.05 --seed 0 \
  2>&1 | tee "$OUT/lcg_exact_100.log"

"$PY" "$EXP/evaluate_cail_baselines.py" \
  --dataset "$EXP/data/subset100/cail_dev_stratified_100.jsonl" \
  --prediction "LegalFactAware-v2.5-B=$EXP/data/subset100/legalfactaware_stratified_100.jsonl" \
  --prediction "Qwen2.5-Direct-Exact=$OUT/qwen25_direct_exact_100.jsonl" \
  --prediction "LCG-MHIS-Exact=$OUT/lcg_mhis_qwen25_exact_100.jsonl" \
  --output-dir "$OUT/eval_lcg_exact_100"

