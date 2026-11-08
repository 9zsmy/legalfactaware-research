#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/legalfactaware_recent_baselines_run/experiments/recent_baselines_20261031"
PHASE_ROOT="/root/autodl-tmp/inabs_lfa_phase2"
PY="/root/autodl-tmp/venvs/recent_baselines_20261031/bin/python"
MODEL="/root/autodl-tmp/models/Dream-v0-Instruct-7B"
OUT="$ROOT/results/inabs_delta_pilot50"

mkdir -p "$OUT"

"$PY" "$ROOT/prepare_delta_inabs_pilot.py" \
  --phase-root "$PHASE_ROOT" \
  --output-dir "$OUT/data" \
  --num-samples 50

"$PY" "$ROOT/run_delta_inabs.py" \
  --input "$OUT/data/pilot50_input.jsonl" \
  --output "$OUT/delta_dream_inabs_643w_pilot50.jsonl" \
  --official-code-dir "$ROOT/vendor/delta-diffusion" \
  --model-id "$MODEL" \
  --avg-chars-per-token 4.469938624158604 \
  --max-source-chars 8192 \
  --max-iterations 1200 \
  --quantization 4bit

"$PY" "$ROOT/evaluate_delta_inabs_pilot.py" \
  --predictions "$OUT/delta_dream_inabs_643w_pilot50.jsonl" \
  --output "$OUT/feasibility_metrics.json"

