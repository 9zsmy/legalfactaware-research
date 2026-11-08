#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/inabs_lfa_phase2
PAPER=/root/autodl-tmp/paper_reproductions/legal_case_summarization_aacl2022
cd "$ROOT"

python3 train_summarunner_inabs.py \
  --train-jsonl "$ROOT/outputs/phase4b_data_full/summarunner_train.jsonl" \
  --dev-jsonl "$ROOT/outputs/phase4b_data_full/summarunner_dev.jsonl" \
  --word2vec "$ROOT/outputs/phase4b_gist_full/word2vec_model.bin" \
  --upstream-dir "$PAPER/third_party/SummaRuNNer_upstream" \
  --output-dir "$ROOT/outputs/phase4b_summarunner_full_restart" \
  --epochs 30

PYTHONPATH="$ROOT/vendor_pyrouge11" python3 evaluate_phase4b_predictions.py \
  --method summarunner \
  --dataset-dir "$ROOT/dataset/IN-Abs" \
  --predictions-dir "$ROOT/outputs/phase4b_summarunner_full_restart/predictions" \
  --e1-records "$ROOT/outputs/phase3c_spacy_full/dev_records.jsonl" \
  --output-dir "$ROOT/outputs/phase4b_summarunner_eval_restart"
