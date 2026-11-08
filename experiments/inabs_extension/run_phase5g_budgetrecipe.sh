#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/inabs_lfa_phase2
REPO=/root/autodl-tmp/JUSTNLP_LSUMM_BudgetRecipe_2025
WORK=/root/inabs_phase5g_budgetrecipe
MODEL=/dev/shm/inabs_budgetrecipe_model
PYTHON=/root/autodl-tmp/venvs/budgetrecipe/bin/python
cd "$ROOT"

test -f "$MODEL/model.safetensors.index.json"
test -f "$MODEL/model-00003-of-00004.safetensors"

PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 "$PYTHON" run_inabs_budgetrecipe.py \
  --model "$MODEL" --dev /root/inabs_phase5d_sende/data/dev.jsonl \
  --repo "$REPO" --output-dir "$WORK/run" --max-new-tokens 1024

PYTHONPATH=vendor_pyrouge11 python3 evaluate_phase4b_predictions.py \
  --method BudgetRecipe-released-checkpoint-zero-shot-IN-Abs \
  --dataset-dir dataset/IN-Abs --predictions-dir "$WORK/run/predictions" \
  --e1-records outputs/phase3c_spacy_full/dev_records.jsonl \
  --output-dir "$WORK/eval"

tar -czf "$WORK/phase5g_budgetrecipe_compact_results.tgz" \
  -C "$ROOT" PREREGISTRATION_PHASE5G_BUDGET_RECIPE.md \
  run_inabs_budgetrecipe.py run_phase5g_budgetrecipe.sh \
  -C "$WORK" run/manifest.json run/records.jsonl \
  eval/metrics.json eval/dev_records.jsonl
