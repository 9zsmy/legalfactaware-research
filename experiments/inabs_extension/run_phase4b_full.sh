#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/inabs_lfa_phase2
PAPER=/root/autodl-tmp/paper_reproductions/legal_case_summarization_aacl2022
cd "$ROOT"

python3 prepare_inabs_phase4b.py \
  --dataset-dir "$ROOT/dataset/IN-Abs" \
  --archive "$ROOT/dataset.zip" \
  --train-oracles "$ROOT/outputs/phase2_lr_full/train_oracles.jsonl" \
  --output-dir "$ROOT/outputs/phase4b_data_full"

python3 train_gist_inabs_from_frozen_labels.py \
  --labelled-train "$ROOT/outputs/phase4b_data_full/gist_labelled_train" \
  --test-dir "$ROOT/outputs/phase4b_data_full/dev_candidates" \
  --length-file "$ROOT/outputs/phase4b_data_full/dev_lengths.tsv" \
  --cue-phrases "$PAPER/official_repo/extractive/Gist/cue_phrases.txt" \
  --pos-tags "$PAPER/official_repo/extractive/Gist/postags.txt" \
  --output-dir "$ROOT/outputs/phase4b_gist_full" \
  --seed 42

PYTHONPATH="$ROOT/vendor_pyrouge11" python3 evaluate_phase4b_predictions.py \
  --method gist \
  --dataset-dir "$ROOT/dataset/IN-Abs" \
  --predictions-dir "$ROOT/outputs/phase4b_gist_full/predictions" \
  --e1-records "$ROOT/outputs/phase3c_spacy_full/dev_records.jsonl" \
  --output-dir "$ROOT/outputs/phase4b_gist_eval"

python3 train_summarunner_inabs.py \
  --train-jsonl "$ROOT/outputs/phase4b_data_full/summarunner_train.jsonl" \
  --dev-jsonl "$ROOT/outputs/phase4b_data_full/summarunner_dev.jsonl" \
  --word2vec "$ROOT/outputs/phase4b_gist_full/word2vec_model.bin" \
  --upstream-dir "$PAPER/third_party/SummaRuNNer_upstream" \
  --output-dir "$ROOT/outputs/phase4b_summarunner_full" \
  --epochs 30

PYTHONPATH="$ROOT/vendor_pyrouge11" python3 evaluate_phase4b_predictions.py \
  --method summarunner \
  --dataset-dir "$ROOT/dataset/IN-Abs" \
  --predictions-dir "$ROOT/outputs/phase4b_summarunner_full/predictions" \
  --e1-records "$ROOT/outputs/phase3c_spacy_full/dev_records.jsonl" \
  --output-dir "$ROOT/outputs/phase4b_summarunner_eval"
