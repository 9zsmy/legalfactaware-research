#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/inabs_lfa_phase2
WORK=/root/inabs_phase5f_civilsum
MODEL=/root/autodl-tmp/hf_models/hub/models--allenai--led-base-16384/snapshots/38335783885b338d93791936c54bb4be46bebed9
cd "$ROOT"

TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python3 run_inabs_civilsum_led.py \
  --train /root/inabs_phase5d_sende/data/train.jsonl \
  --dev /root/inabs_phase5d_sende/data/dev.jsonl \
  --model "$MODEL" --output "$WORK/run"

PYTHONPATH=vendor_pyrouge11 python3 evaluate_phase4b_predictions.py \
  --method CivilSum-LED-tail-generative-IN-Abs \
  --dataset-dir dataset/IN-Abs --predictions-dir "$WORK/run/predictions" \
  --e1-records outputs/phase3c_spacy_full/dev_records.jsonl \
  --output-dir "$WORK/eval"

tar -czf "$WORK/phase5f_civilsum_compact_results.tgz" \
  -C "$ROOT" PREREGISTRATION_PHASE5F_CIVILSUM_LED.md \
  run_inabs_civilsum_led.py run_phase5f_civilsum.sh \
  -C "$WORK" run/manifest.json run/decode_manifest.json \
  run/dev_records.jsonl eval/metrics.json eval/dev_records.jsonl
