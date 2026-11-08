#!/usr/bin/env bash
set -uo pipefail

ROOT=${ROOT:-/root/autodl-tmp/legalfactaware_full_runtime_20261021}
CODE=${CODE:-/root/autodl-tmp/legalfactaware_full_runtime_20261021/code}
OUT=${OUT:-$ROOT/outputs}
CAIL=/root/legalfactaware_runs_20260813/data/cail2020-clean-20260811
SSE_ROOT=/root/legalfactaware_runs_20260813/sse_reimplementation
SSE_SCRIPTS="$SSE_ROOT/scripts"
MEMSUM_ROOT="$SSE_ROOT/third_party/MemSum_acl2022/memsum_demo"
SENDE=/root/autodl-tmp/sende_independent_20260826
TERM=/root/autodl-tmp/termdiffusum_reproduction_20260817
PY=/root/miniconda3/bin/python
TERMPY=/root/autodl-tmp/conda_envs/termdiff37/bin/python

mkdir -p "$OUT"

run_stage() {
  local method="$1"
  local stage="$2"
  shift 2
  local dir="$OUT/$method"
  mkdir -p "$dir"
  printf '%s\t%s\t%s\n' "$(date -Is)" "$method" "$stage" > "$ROOT/STATUS"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 \
    > "$dir/${stage}.gpu.csv" 2>/dev/null &
  local monitor_pid=$!
  set +e
  "$PY" "$CODE/time_command.py" --output "$dir/${stage}.time.json" -- "$@" \
    > "$dir/${stage}.stdout.log" 2> "$dir/${stage}.stderr.log"
  local status=$?
  set -e
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  printf '%s\n' "$status" > "$dir/${stage}.exitcode"
  if [[ "$status" -ne 0 ]]; then
    printf '%s\t%s\tFAILED\t%s\n' "$(date -Is)" "$method" "$stage" > "$ROOT/STATUS"
    touch "$ROOT/FAILED"
    exit "$status"
  fi
}

run_legalfactaware() {
  local dir="$OUT/legalfactaware"
  mkdir -p "$dir"
  run_stage legalfactaware full_pipeline \
    "$PY" "$CODE/benchmark_lfa_end_to_end.py" \
    --scripts-dir "$SSE_SCRIPTS" \
    --formal-script "$CODE/run_legalfactaware_formal_ablation_dev.py" \
    --train "$CAIL/train.jsonl" --dev "$CAIL/dev.jsonl" \
    --vectorizer-template /root/legalfactaware_runs_20260813/artifacts/legalfactaware_v22_oracle_supervision_dev_20260815/model.pkl \
    --output-dir "$dir/result"
  test "$(wc -l < "$dir/result/dev_predictions.jsonl")" -eq 942
}

run_sse() {
  local dir="$OUT/sse"
  local prep="$dir/prepared"
  local model="$dir/model"
  mkdir -p "$dir"
  run_stage sse preprocessing \
    "$PY" "$SSE_SCRIPTS/prepare_sse_cail.py" \
    --source-dir "$CAIL" --output-dir "$prep" --max-vocab 12000 \
    --embed-dim 128 --seed 20260814 --workers 12 --split train --split dev
  run_stage sse training \
    "$PY" "$SSE_SCRIPTS/train_sse_memsum.py" \
    --prepared-dir "$prep" --output-dir "$model" --embed-dim 128 \
    --hidden-size 512 --num-heads 8 --max-doc-len 512 --max-seq-len 128 \
    --batch-size 4 --epochs 10 --learning-rate 0.0001 --weight-decay 0.000001 \
    --dropout 0.1 --weight-mode inverse_ratio --token-softmax-dim sequence \
    --seed 20260814 --device cuda
  run_stage sse inference \
    "$PY" "$SSE_SCRIPTS/run_sse_memsum.py" \
    --checkpoint "$model/best.pt" --corpus "$prep/dev.jsonl" \
    --output "$dir/dev_predictions.jsonl" --device cuda --batch-size 4 \
    --max-sentences 8 --max-chars 450 --output-order source
  test "$(wc -l < "$dir/dev_predictions.jsonl")" -eq 942
}

run_memsum() {
  local dir="$OUT/memsum"
  local sseprep="$dir/sse_prepared"
  local prep="$dir/prepared"
  local artifacts="$dir/artifacts"
  local seedroot="$artifacts/memsum_cail_strict_v2_seed20260814"
  mkdir -p "$dir"
  run_stage memsum preprocessing_sse \
    "$PY" "$SSE_SCRIPTS/prepare_sse_cail.py" \
    --source-dir "$CAIL" --output-dir "$sseprep" --max-vocab 12000 \
    --embed-dim 128 --seed 20260814 --workers 12
  run_stage memsum preprocessing_memsum \
    "$PY" "$SSE_SCRIPTS/prepare_memsum_strict_cail.py" \
    --train-dev-dir "$sseprep" --test-dir "$sseprep" \
    --output-dir "$prep" --artifact-root "$artifacts"
  run_stage memsum training \
    "$PY" "$MEMSUM_ROOT/src/MemSum_Final/train.py" \
    -config_file_path "$seedroot/training.config"
  run_stage memsum inference \
    "$PY" "$SSE_SCRIPTS/run_memsum_chinese.py" \
    --corpus "$prep/dev.jsonl" --model "$seedroot/model/model_best.pt" \
    --vocabulary "$sseprep/vocabulary.pkl" --output "$dir/dev_predictions.jsonl" \
    --gpu 0 --budget 450 --max-sentences 7 --p-stop 0.6 \
    --embed-dim 128 --hidden-dim 512 --max-doc-len 512 --max-seq-len 128
  test "$(wc -l < "$dir/dev_predictions.jsonl")" -eq 942
}

run_sende() {
  local dir="$OUT/sende"
  local records="$dir/oracles"
  local tsdae="$dir/tsdae"
  local features="$dir/features"
  local extractor="$dir/extractor"
  mkdir -p "$features"
  if [[ "$(cat "$dir/oracle.exitcode" 2>/dev/null || true)" != "0" || ! -s "$records/train.jsonl" || ! -s "$records/dev.jsonl" ]]; then
    run_stage sende oracle \
      "$PY" "$SENDE/prepare_cail_oracles.py" \
      --data-dir "$CAIL" --output-dir "$records" --splits train dev --workers 12
  fi
  run_stage sende tsdae_training \
    "$PY" "$CODE/train_tsdae.py" \
    --train-jsonl "$records/train.jsonl" \
    --base-model "$SENDE/models/chinese_legal_roberta_base" \
    --output-dir "$tsdae" --epochs 1 --batch-size 64 --lr 0.00002 \
    --warmup-steps 10000 --max-seq-length 128 --seed 20260826
  # Checkpoints are temporary recovery files; the final encoder is saved at
  # the output root and the checkpoint copies would otherwise consume ~1.2 GB.
  rm -rf "$tsdae/checkpoints"
  run_stage sende encode_train \
    "$PY" "$SENDE/encode_records.py" \
    --records "$records/train.jsonl" --model "$tsdae" \
    --output-prefix "$features/train" --batch-size 256
  run_stage sende encode_dev \
    "$PY" "$SENDE/encode_records.py" \
    --records "$records/dev.jsonl" --model "$tsdae" \
    --output-prefix "$features/dev" --batch-size 256
  run_stage sende extractor_training \
    "$PY" "$SENDE/train_extractor.py" \
    --train-prefix "$features/train" --output-dir "$extractor" \
    --epochs 20 --patience 3 --batch-size 64 --lr 0.0002 \
    --max-sentences 1024 --seed 20260826
  run_stage sende inference \
    "$PY" "$CODE/benchmark_sende_inference.py" \
    --source-dir "$SENDE" --records "$records/dev.jsonl" \
    --prefix "$features/dev" --checkpoint "$extractor/best.pt" \
    --output "$dir/dev_predictions.jsonl" --threshold 0.8
  test "$(wc -l < "$dir/dev_predictions.jsonl")" -eq 942
  sha256sum "$tsdae"/model.safetensors "$extractor/best.pt" > "$dir/model_hashes.txt"
  # Encoded matrices and the TSDAE model are reproducible intermediates. Keep
  # their timings/hashes but reclaim space before TermDiffuSum training.
  if [[ "${KEEP_SENDE_MODEL:-0}" != "1" ]]; then
    rm -rf "$features" "$tsdae"
  fi
}

TERM_LINK_DIR=/media/dxy/DiffuSum-main/datasets/legal_last40
TERM_ORIG_TRAIN=""
TERM_ORIG_VAL=""
TERM_ORIG_TEST=""
restore_term_links() {
  if [[ -n "$TERM_ORIG_TRAIN" ]]; then
    ln -sfn "$TERM_ORIG_TRAIN" "$TERM_LINK_DIR/train.jsonl"
    ln -sfn "$TERM_ORIG_VAL" "$TERM_LINK_DIR/val.jsonl"
    ln -sfn "$TERM_ORIG_TEST" "$TERM_LINK_DIR/test.jsonl"
  fi
}
trap restore_term_links EXIT

run_termdiffusum() {
  local dir="$OUT/termdiffusum"
  local prep="$dir/adapted"
  local model="$dir/model"
  mkdir -p "$dir"
  run_stage termdiffusum preprocessing \
    "$TERMPY" "$TERM/prepare_termdiffusum_cail.py" \
    --input-dir "$CAIL" --output-dir "$prep" --max-sentences 200
  TERM_ORIG_TRAIN=$(readlink "$TERM_LINK_DIR/train.jsonl")
  TERM_ORIG_VAL=$(readlink "$TERM_LINK_DIR/val.jsonl")
  TERM_ORIG_TEST=$(readlink "$TERM_LINK_DIR/test.jsonl")
  ln -sfn "$prep/train.jsonl" "$TERM_LINK_DIR/train.jsonl"
  ln -sfn "$prep/val.jsonl" "$TERM_LINK_DIR/val.jsonl"
  ln -sfn "$prep/test.jsonl" "$TERM_LINK_DIR/test.jsonl"
  run_stage termdiffusum training_and_inference \
    bash -lc "cd '$TERM/runtime' && PYTHONPATH='$TERM/runtime' '$TERMPY' '$CODE/run_termdiffusum_cail_full.py' \
      --adapted-data-dir '$prep' --output-dir '$model' --epochs 10 \
      --train-batch-size 32 --decode-batch-size 6 --learning-rate 0.00001 \
      --max-sentences 8 --max-chars 450 --seed 101 --decode-splits dev \
      --bert-config-dir /root/autodl-tmp/fedsum_reproduction_20260826/models/bert-base-uncased"
  restore_term_links
  TERM_ORIG_TRAIN=""
  test "$(wc -l < "$model/dev_predictions.jsonl")" -eq 942
}

rm -f "$ROOT/FAILED" "$ROOT/COMPLETE"
case "${1:-all}" in
  all)
    run_legalfactaware
    run_sse
    run_memsum
    run_sende
    run_termdiffusum
    ;;
  sende_term)
    run_sende
    run_termdiffusum
    ;;
  term_only)
    run_termdiffusum
    ;;
  legalfactaware)
    run_legalfactaware
    ;;
  sse)
    run_sse
    ;;
  memsum)
    run_memsum
    ;;
  sende)
    run_sende
    ;;
  termdiffusum)
    run_termdiffusum
    ;;
  *)
    echo "usage: $0 [all|sende_term|term_only|legalfactaware|sse|memsum|sende|termdiffusum]" >&2
    exit 2
    ;;
esac
printf '%s\tCOMPLETE\n' "$(date -Is)" > "$ROOT/STATUS"
touch "$ROOT/COMPLETE"
