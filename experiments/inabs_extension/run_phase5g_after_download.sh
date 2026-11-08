#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/inabs_lfa_phase2
REPO=/root/autodl-tmp/JUSTNLP_LSUMM_BudgetRecipe_2025
WORK=/root/inabs_phase5g_budgetrecipe
MODEL=/dev/shm/inabs_budgetrecipe_model
PYTHON=/root/autodl-tmp/venvs/budgetrecipe/bin/python

while [[ ! -f "$WORK/download_retry.exitcode" ]]; do
  sleep 20
done
test "$(cat "$WORK/download_retry.exitcode")" = 0
for shard in 00001 00002 00003 00004; do
  test -s "$MODEL/model-${shard}-of-00004.safetensors"
done

cd "$ROOT"
PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 "$PYTHON" run_inabs_budgetrecipe.py \
  --model "$MODEL" --dev /root/inabs_phase5d_sende/data/dev.jsonl \
  --repo "$REPO" --output-dir "$WORK/smoke" --max-new-tokens 1024 --limit 3

python3 -c "import json; p='$WORK/smoke/records.jsonl'; x=[json.loads(s) for s in open(p)]; assert len(x)==3 and any(not r['empty'] for r in x), x"

bash run_phase5g_budgetrecipe.sh
