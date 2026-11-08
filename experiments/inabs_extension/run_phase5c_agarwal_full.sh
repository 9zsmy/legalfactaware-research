#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/inabs_lfa_phase2
MODEL=/root/autodl-tmp/sende_independent_20260826/models/legal_bert_base
DATA="$ROOT/outputs/phase5c_agarwal_st_data"
TRAIN_CACHE="$ROOT/outputs/phase5c_agarwal_train_cache"
DEV_CACHE="$ROOT/outputs/phase5c_agarwal_dev_cache"
RUN="$ROOT/outputs/phase5c_agarwal_st_run"
EVAL="$ROOT/outputs/phase5c_agarwal_st_eval"

cd "$ROOT"
python3 resume_agarwal_sentence_embeddings.py \
  --input "$DATA/train.jsonl" --output-dir "$TRAIN_CACHE" \
  --model "$MODEL" --tokenizer "$MODEL" --batch-size 64 --max-length 512 --device cuda

python3 /root/autodl-tmp/agarwal_st_cail_20260816/scripts/cache_agarwal_sentence_embeddings.py \
  --input "$DATA/dev.jsonl" --output-dir "$DEV_CACHE" \
  --model "$MODEL" --tokenizer "$MODEL" --batch-size 64 --max-length 512 --device cuda --dtype float16

python3 train_agarwal_st_inabs.py \
  --train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl" \
  --train-cache "$TRAIN_CACHE" --dev-cache "$DEV_CACHE" --output-dir "$RUN" \
  --epochs 5 --batch-size 8 --hidden-size 128 --dropout 0.5 \
  --learning-rate 0.00261 --lambda-mmr 0.8 --seed 42

PYTHONPATH=vendor_pyrouge11 python3 evaluate_phase4b_predictions.py \
  --method Agarwal-ST-P50-budget-oracle-restricted-transfer \
  --dataset-dir dataset/IN-Abs --predictions-dir "$RUN/predictions" \
  --e1-records outputs/phase3c_spacy_full/dev_records.jsonl --output-dir "$EVAL"

tar -czf phase5c_agarwal_st_compact_results.tgz \
  PREREGISTRATION_PHASE5C_AGARWAL_ST.md resume_agarwal_sentence_embeddings.py \
  train_agarwal_st_inabs.py run_phase5c_agarwal_full.sh \
  "$TRAIN_CACHE/manifest.json" "$DEV_CACHE/manifest.json" \
  "$RUN/manifest.json" "$RUN/model_epoch5.pt" "$RUN/decode_records.jsonl" \
  "$EVAL/metrics.json" "$EVAL/dev_records.jsonl"

