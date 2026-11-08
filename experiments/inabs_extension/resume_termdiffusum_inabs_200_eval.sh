#!/usr/bin/env bash
set -euo pipefail

TD_ROOT=/root/autodl-tmp/termdiffusum_reproduction_20260817
INABS_ROOT=/root/autodl-tmp/inabs_lfa_phase2
DATA_DIR="$TD_ROOT/data_inabs_200_full"
RUN_DIR="$TD_ROOT/outputs/inabs_200_full_10epoch"
EVAL_DIR="$INABS_ROOT/outputs/phase5h_termdiffusum_200_eval"

cd "$TD_ROOT/runtime"
env \
  PYTHONPATH="$TD_ROOT/runtime" \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE=/tmp/termdiff_inabs_200_full_cache \
  /root/autodl-tmp/conda_envs/termdiff37/bin/python \
  "$TD_ROOT/runtime/run_termdiffusum_inabs_200.py" \
  --adapted-data-dir "$DATA_DIR" \
  --output-dir "$RUN_DIR" \
  --epochs 10 \
  --train-batch-size 32 \
  --decode-batch-size 19 \
  --learning-rate 1e-5 \
  --seed 101

cd "$INABS_ROOT"
env PYTHONPATH=/tmp/kmm_pyrouge/lib/python3.10/site-packages \
  python evaluate_phase4b_predictions.py \
  --method TermDiffuSum-INAbs-200-candidate-sensitivity \
  --dataset-dir "$INABS_ROOT/dataset/IN-Abs" \
  --predictions-dir "$RUN_DIR/dev_predictions" \
  --e1-records "$INABS_ROOT/outputs/phase3b_official_eval/official_pyrouge_per_case.jsonl" \
  --output-dir "$EVAL_DIR"

printf '%s\n' 0 > "$TD_ROOT/outputs/inabs_200_eval_resume.exitcode"
