#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/inabs_lfa_phase2
SENDE=/root/autodl-tmp/sende_independent_20260826
WORK=/root/inabs_phase5d_sende
DATA="$WORK/data"
cd "$ROOT"

test "$(cat "$WORK/tsdae.exitcode")" = 0
test -f "$WORK/tsdae/reconstruction_manifest.json"

if [[ ! -f "$DATA/manifest.json" ]]; then
  python3 prepare_inabs_sende.py \
    --source-dir outputs/phase5c_agarwal_st_data \
    --oracles outputs/phase2_lr_full/train_oracles.jsonl \
    --output-dir "$DATA"
fi

python3 "$SENDE/encode_records.py" \
  --records "$DATA/train.jsonl" --model "$WORK/tsdae" \
  --output-prefix "$WORK/train_features" --batch-size 256
python3 "$SENDE/encode_records.py" \
  --records "$DATA/dev.jsonl" --model "$WORK/tsdae" \
  --output-prefix "$WORK/dev_features" --batch-size 256

python3 "$SENDE/train_extractor.py" \
  --train-prefix "$WORK/train_features" --output-dir "$WORK/extractor" \
  --epochs 20 --patience 3 --batch-size 64 --lr 0.0002 \
  --max-sentences 400 --seed 20260826

python3 decode_inabs_sende.py \
  --records "$DATA/dev.jsonl" --features-prefix "$WORK/dev_features" \
  --checkpoint "$WORK/extractor/best.pt" --sende-src "$SENDE" \
  --output-dir "$WORK/decode" --threshold 0.2

PYTHONPATH=vendor_pyrouge11 python3 evaluate_phase4b_predictions.py \
  --method SENDE-common-label-independent-transfer \
  --dataset-dir dataset/IN-Abs --predictions-dir "$WORK/decode/predictions" \
  --e1-records outputs/phase3c_spacy_full/dev_records.jsonl \
  --output-dir "$WORK/eval"

tar -czf "$WORK/phase5d_sende_compact_results.tgz" \
  -C "$ROOT" PREREGISTRATION_PHASE5D_SENDE.md prepare_inabs_sende.py \
  decode_inabs_sende.py run_phase5d_sende_after_tsdae.sh \
  -C "$WORK" data/manifest.json tsdae/reconstruction_manifest.json \
  extractor/manifest.json extractor/best.pt decode/manifest.json \
  decode/decode_records.jsonl eval/metrics.json eval/dev_records.jsonl
