#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/legalfactaware_runtime_benchmark_20261021
CODE="$ROOT/code"
OUT="$ROOT/outputs"
LOG="$ROOT/logs"
mkdir -p "$OUT" "$LOG"

run_one() {
  local method="$1"
  local repeat="$2"
  local command="$3"
  local run_dir="$OUT/${method}_r${repeat}"
  mkdir -p "$run_dir"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 > "$run_dir/gpu.csv" 2>/dev/null &
  local monitor_pid=$!
  TIMEFORMAT='real_seconds=%R user_seconds=%U sys_seconds=%S'
  { time bash -lc "$command" > "$run_dir/stdout.log" 2> "$run_dir/stderr.log"; } 2> "$run_dir/time.txt"
  local status=$?
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  printf '%s\n' "$status" > "$run_dir/exitcode"
  printf '%s repeat=%s exit=%s\n' "$method" "$repeat" "$status"
  return "$status"
}

BASEPY=/root/miniconda3/bin/python
TERMPY=/root/autodl-tmp/conda_envs/termdiff37/bin/python
LFA_SCRIPTS=/root/legalfactaware_runs_20260813/sse_reimplementation/scripts
DEV=/root/legalfactaware_runs_20260813/data/cail2020-clean-20260811/dev.jsonl
PREP=/root/legalfactaware_runs_20260813/data/sse_cail_clean_prepared_v2_20260814

for repeat in 1 2 3; do
  run_one legalfactaware "$repeat" "$BASEPY $CODE/benchmark_lfa_inference.py --scripts-dir $LFA_SCRIPTS --formal-script $CODE/run_legalfactaware_formal_ablation_dev.py --dev $DEV --vectorizer-model /root/legalfactaware_runs_20260813/artifacts/legalfactaware_v22_oracle_supervision_dev_20260815/model.pkl --classifier-model /root/autodl-tmp/legalfactaware_formal_ablation_20260830/outputs/unified_dev/models/oracle__full.pkl --output $OUT/legalfactaware_r${repeat}/predictions.jsonl" || exit 11
done

for repeat in 1 2 3; do
  run_one sse "$repeat" "$BASEPY /root/legalfactaware_runs_20260813/sse_reimplementation/scripts/run_sse_memsum.py --checkpoint /root/legalfactaware_runs_20260813/artifacts/sse_v2_inverse_sequence_seed20260814/best.pt --corpus $PREP/dev.jsonl --output $OUT/sse_r${repeat}/predictions.jsonl --device cuda --batch-size 4 --max-sentences 8 --max-chars 450 --output-order source" || exit 12
done

for repeat in 1 2 3; do
  run_one memsum "$repeat" "$BASEPY /root/legalfactaware_runs_20260813/sse_reimplementation/scripts/run_memsum_chinese.py --corpus /root/legalfactaware_runs_20260813/data/memsum_cail_strict_v2_20260814/dev.jsonl --model /root/legalfactaware_runs_20260813/artifacts/memsum_cail_strict_v2_seed20260814/model/model_best.pt --vocabulary $PREP/vocabulary.pkl --output $OUT/memsum_r${repeat}/predictions.jsonl --gpu 0 --budget 450 --max-sentences 7 --p-stop 0.6 --embed-dim 128 --hidden-dim 512 --max-doc-len 512 --max-seq-len 128" || exit 13
done

for repeat in 1 2 3; do
  run_one sende "$repeat" "$BASEPY $CODE/benchmark_sende_inference.py --source-dir /root/autodl-tmp/sende_independent_20260826 --records /root/autodl-tmp/sende_independent_20260826/outputs/cail_oracles_full/dev.jsonl --prefix /root/autodl-tmp/sende_independent_20260826/outputs/cail_features_legal_roberta_tsdae/dev --checkpoint /root/autodl-tmp/sende_independent_20260826/outputs/cail_extractor_legal_roberta_tsdae_full/best.pt --output $OUT/sende_r${repeat}/predictions.jsonl --threshold 0.8" || exit 14
done

for repeat in 1 2 3; do
  run_one termdiffusum "$repeat" "$TERMPY $CODE/benchmark_termdiff_dev.py --runtime-dir /root/autodl-tmp/termdiffusum_reproduction_20260817/runtime --adapted-data-dir /root/autodl-tmp/termdiffusum_reproduction_20260817/data_cail_full --checkpoint /root/autodl-tmp/termdiffusum_reproduction_20260817/outputs/cail_full_10epoch/checkpoint_latest.pt --output-dir $OUT/termdiffusum_r${repeat} --batch-size 6" || exit 15
done

touch "$ROOT/COMPLETE"
