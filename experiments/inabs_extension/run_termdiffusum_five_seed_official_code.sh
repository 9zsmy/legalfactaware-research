#!/usr/bin/env bash
set -euo pipefail

experiment_root=/root/autodl-tmp/termdiffusum_reproduction_20260817
runtime_root="$experiment_root/runtime"
adapter_root="$experiment_root/data_inabs_200_full"
keyword_file="$runtime_root/termdiffusum_inabs_english_legal_terms.json"
phase_root=/root/autodl-tmp/inabs_lfa_phase2
python37=/root/autodl-tmp/conda_envs/termdiff37/bin/python
pyrouge_path=/tmp/kmm_pyrouge/lib/python3.10/site-packages
dataset_root="$phase_root/dataset/IN-Abs"
e1_records="$phase_root/outputs/phase3b_official_eval/official_pyrouge_per_case.jsonl"

export PYTHONPATH="$runtime_root"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE="$experiment_root/hf_cache_five_seed"

for seed in 100 201 901 1001; do
  output_root="$experiment_root/outputs/inabs_200_official_code_seed${seed}"
  source_eval="$phase_root/outputs/phase5h_termdiffusum_official_code_seed${seed}_source_eval"
  budget_eval="$phase_root/outputs/phase5h_termdiffusum_official_code_seed${seed}_budget_eval"
  mkdir -p "$output_root" "$source_eval" "$budget_eval"

  if [[ ! -f "$output_root/manifest.json" ]]; then
    cd "$runtime_root"
    "$python37" run_termdiffusum_inabs_200_source_faithful.py \
      --adapted-data-dir "$adapter_root" \
      --keywords-file "$keyword_file" \
      --output-dir "$output_root" \
      --epochs 10 \
      --train-batch-size 32 \
      --decode-batch-size 19 \
      --learning-rate 1e-5 \
      --lr-anneal-steps 400000 \
      --seed "$seed" \
      --resume-recovery \
      > "$output_root/run.log" 2>&1
  fi

  if [[ ! -f "$source_eval/metrics.json" ]]; then
    cd "$phase_root"
    PYTHONPATH="$pyrouge_path" python evaluate_inabs_predictions_unbounded.py \
      --method "TermDiffuSum-OfficialCode-seed${seed}-SourceDecoder" \
      --dataset-dir "$dataset_root" \
      --records "$output_root/source_decoder/dev_records.jsonl" \
      --output-dir "$source_eval" \
      > "$source_eval/run.log" 2>&1
  fi

  if [[ ! -f "$budget_eval/metrics.json" ]]; then
    cd "$phase_root"
    PYTHONPATH="$pyrouge_path" python evaluate_phase4b_predictions.py \
      --method "TermDiffuSum-OfficialCode-seed${seed}-BudgetAdapted" \
      --dataset-dir "$dataset_root" \
      --predictions-dir "$output_root/budget_adapted_decoder/predictions" \
      --e1-records "$e1_records" \
      --output-dir "$budget_eval" \
      > "$budget_eval/run.log" 2>&1
  fi

  # Metrics and predictions are sufficient for the five-seed aggregate. The
  # two checkpoint files are removed after successful evaluation because the
  # server has less than 2 GB free and cannot retain five 800 MB model copies.
  if [[ -f "$source_eval/metrics.json" && -f "$budget_eval/metrics.json" ]]; then
    rm -f "$output_root/model_final.pt" "$output_root/checkpoint_recovery_fp16.pt"
  fi
done

cd "$phase_root"
python aggregate_termdiffusum_five_seeds.py \
  --phase-root "$phase_root" \
  --output-dir "$phase_root/outputs/phase5h_termdiffusum_official_code_five_seed_aggregate"
