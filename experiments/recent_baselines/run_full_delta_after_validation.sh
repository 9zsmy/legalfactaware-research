#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/legalfactaware_recent_baselines_run
EXP="$ROOT/experiments/recent_baselines_20261031"
PY=/root/autodl-tmp/venvs/recent_baselines_20261031/bin/python
MODEL=/root/autodl-tmp/models/Dream-v0-Instruct-7B
VALIDATION="$EXP/results/original_task_validation/delta_dream_xsum_50.json"
OUT="$EXP/results/full942"

mkdir -p "$OUT"
cd "$ROOT"

"$PY" -c '
import json, sys
d = json.load(open(sys.argv[1]))
s = d["statistics"]
print("XSum validation:", s)
if s["mean_absolute_error"] > 5.0 or s["within_10_rate"] < 0.90:
    raise SystemExit("DELTA original-task regression gate failed")
' "$VALIDATION"

"$PY" "$EXP/run_delta_cail.py" \
  --input "$EXP/data/cail_dev_942.jsonl" \
  --output "$OUT/delta_dream_942.jsonl" \
  --official-code-dir "$EXP/vendor/delta-diffusion" \
  --model-id "$MODEL" \
  --model-name dream \
  --target-chars 450 \
  --avg-chars-per-token 1.5263636665452414 \
  --quantization 4bit \
  2>&1 | tee "$OUT/delta_dream_942.log"

"$PY" "$EXP/evaluate_cail_baselines.py" \
  --dataset "$EXP/data/cail_dev_942.jsonl" \
  --prediction "LegalFactAware-v2.5-B=$EXP/data/legalfactaware_predictions.jsonl" \
  --prediction "DELTA-Dream-CAIL-adapted=$OUT/delta_dream_942.jsonl" \
  --output-dir "$OUT/eval_delta_942"
