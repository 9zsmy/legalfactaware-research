#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/legalfactaware_recent_baselines_run
EXP="$ROOT/experiments/recent_baselines_20261031"
PY=/root/autodl-tmp/venvs/recent_baselines_20261031/bin/python
MODEL=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
OUT="$EXP/results/lcg_beam16_smoke1"

mkdir -p "$OUT"
cd "$ROOT"

"$PY" -m unittest discover -s "$EXP/tests" -v

"$PY" "$EXP/run_lcg_blackbox.py" \
  --input "$EXP/data/cail_dev_942.jsonl" \
  --demo "$EXP/data/lcg_demo_train_row0.jsonl" \
  --output "$OUT/lcg_mhis_beam16_upper_smoke1.jsonl" \
  --initial-output "$OUT/qwen25_beam16_initial_upper_smoke1.jsonl" \
  --model-id "$MODEL" \
  --target-chars 450 --length-objective upper_bound \
  --beam-size 16 --max-iterations 5 --quantization 4bit \
  --top-k 20 --top-p 0.8 --temperature 0.7 \
  --repetition-penalty 1.05 --seed 0 --num-samples 1 \
  2>&1 | tee "$OUT/upper_smoke1.log"

"$PY" "$EXP/audit_lcg_beam_run.py" \
  --result "$OUT/lcg_mhis_beam16_upper_smoke1.jsonl" \
  --initial "$OUT/qwen25_beam16_initial_upper_smoke1.jsonl" \
  > "$OUT/upper_audit.json"

"$PY" "$EXP/run_lcg_blackbox.py" \
  --input "$EXP/data/cail_dev_942.jsonl" \
  --demo "$EXP/data/lcg_demo_train_row0.jsonl" \
  --output "$OUT/lcg_mhis_beam16_exact_smoke1.jsonl" \
  --initial-output "$OUT/qwen25_beam16_initial_exact_smoke1.jsonl" \
  --model-id "$MODEL" \
  --target-chars 450 --length-objective exact \
  --beam-size 16 --max-iterations 5 --quantization 4bit \
  --top-k 20 --top-p 0.8 --temperature 0.7 \
  --repetition-penalty 1.05 --seed 0 --num-samples 1 \
  2>&1 | tee "$OUT/exact_smoke1.log"

"$PY" "$EXP/audit_lcg_beam_run.py" \
  --result "$OUT/lcg_mhis_beam16_exact_smoke1.jsonl" \
  --initial "$OUT/qwen25_beam16_initial_exact_smoke1.jsonl" \
  > "$OUT/exact_audit.json"

