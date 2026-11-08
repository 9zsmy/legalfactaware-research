#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/inabs_lfa_phase2
SSE=/root/legalfactaware_runs_20260813/sse_reimplementation
WORK=/root/inabs_phase5e_sse
cd "$ROOT"

python3 prepare_inabs_sse.py \
  --source-dir /root/inabs_phase5d_sende/data \
  --output-dir "$WORK/prepared" \
  --sse-scripts "$SSE/scripts"

python3 "$SSE/scripts/train_sse_memsum.py" \
  --prepared-dir "$WORK/prepared" --output-dir "$WORK/train" \
  --embed-dim 128 --hidden-size 512 --num-heads 8 \
  --max-doc-len 400 --max-seq-len 128 --batch-size 4 \
  --epochs 10 --learning-rate 0.0001 --weight-decay 0.000001 \
  --dropout 0.1 --weight-mode paper_ratio \
  --token-softmax-dim sequence --seed 20260814

python3 decode_inabs_sse.py \
  --checkpoint "$WORK/train/best.pt" --corpus "$WORK/prepared/dev.jsonl" \
  --sse-scripts "$SSE/scripts" --output-dir "$WORK/decode"

PYTHONPATH=vendor_pyrouge11 python3 evaluate_phase4b_predictions.py \
  --method SSE-structural-proxy-IN-Abs \
  --dataset-dir dataset/IN-Abs --predictions-dir "$WORK/decode/predictions" \
  --e1-records outputs/phase3c_spacy_full/dev_records.jsonl \
  --output-dir "$WORK/eval"

tar -czf "$WORK/phase5e_sse_compact_results.tgz" \
  -C "$ROOT" PREREGISTRATION_PHASE5E_SSE.md prepare_inabs_sse.py \
  decode_inabs_sse.py run_phase5e_sse.sh \
  -C "$WORK" prepared/manifest.json train/run_config.json \
  train/history.json train/best.pt decode/manifest.json \
  decode/decode_records.jsonl eval/metrics.json eval/dev_records.jsonl
