#!/usr/bin/env bash
set -euo pipefail

BENCH=${BENCH:-/root/autodl-tmp/legalfactaware_cost_analysis_20261023}
CODE=${CODE:-/root/autodl-tmp/legalfactaware_full_runtime_20261021/code}
RUNNER="$CODE/run_full_runtime.sh"
MANIFEST="$BENCH/pipeline_runs.tsv"
STATUS="$BENCH/STATUS"

mkdir -p "$BENCH/pipeline" "$BENCH/environment"

if [[ ! -s "$BENCH/environment/system.txt" ]]; then
  {
    date -Is
    hostname
    uname -a
    lscpu
    free -h
    nvidia-smi
    /root/miniconda3/bin/python "$CODE/time_command.py" --help 2>&1 || true
    /root/miniconda3/bin/python --version 2>&1 || true
    env | sort | grep -E '^(CUDA|CUDNN|OMP|MKL|OPENBLAS|NUMEXPR|PYTHON|TOKENIZERS)' || true
  } > "$BENCH/environment/system.txt"
  sha256sum \
    /root/legalfactaware_runs_20260813/data/cail2020-clean-20260811/train.jsonl \
    /root/legalfactaware_runs_20260813/data/cail2020-clean-20260811/dev.jsonl \
    > "$BENCH/environment/input_sha256.txt"
fi

if [[ ! -e "$MANIFEST" ]]; then
  printf 'round\tmethod\tstart\tend\texit_code\tpredictions\toutput_dir\n' > "$MANIFEST"
fi

prediction_path() {
  local root="$1" method="$2"
  case "$method" in
    legalfactaware) printf '%s\n' "$root/outputs/legalfactaware/result/dev_predictions.jsonl" ;;
    termdiffusum) printf '%s\n' "$root/outputs/termdiffusum/model/dev_predictions.jsonl" ;;
    *) printf '%s\n' "$root/outputs/$method/dev_predictions.jsonl" ;;
  esac
}

archive_and_reclaim() {
  local root="$1" method="$2" pred="$3"
  local out="$root/outputs/$method"
  find "$out" -type f \( -name '*.pt' -o -name '*.pkl' -o -name '*.safetensors' -o -name '*.npy' \) \
    -print0 | sort -z | xargs -0 -r sha256sum > "$out/artifact_hashes_before_cleanup.txt"
  sha256sum "$pred" > "$out/prediction_sha256.txt"
  case "$method" in
    legalfactaware)
      rm -f "$out/result/model.pkl" "$out/result/train_oracle_extracts.jsonl"
      ;;
    sse)
      rm -rf "$out/prepared" "$out/model"
      ;;
    memsum)
      rm -rf "$out/sse_prepared" "$out/prepared" "$out/artifacts"
      ;;
    sende)
      rm -rf "$out/oracles" "$out/features" "$out/tsdae" "$out/extractor"
      ;;
    termdiffusum)
      rm -rf "$out/adapted"
      rm -f "$out/model/checkpoint_latest.pt"
      ;;
  esac
}

freeze_for_inference() {
  local root="$1" method="$2" round="$3"
  [[ "$round" == "1" ]] || return 0
  local out="$root/outputs/$method"
  local frozen="$BENCH/frozen_models/$method"
  mkdir -p "$frozen"
  case "$method" in
    legalfactaware)
      cp "$out/result/model.pkl" "$frozen/model.pkl"
      ;;
    sse)
      cp "$out/model/best.pt" "$frozen/best.pt"
      cp "$out/prepared/vocabulary.pkl" "$frozen/vocabulary.pkl"
      cp "$out/prepared/random_embeddings_128d.pkl" "$frozen/random_embeddings_128d.pkl"
      ;;
    memsum)
      cp "$out/artifacts/memsum_cail_strict_v2_seed20260814/model/model_best.pt" "$frozen/model_best.pt"
      cp "$out/sse_prepared/vocabulary.pkl" "$frozen/vocabulary.pkl"
      ;;
    sende)
      mv "$out/tsdae" "$frozen/tsdae"
      cp "$out/extractor/best.pt" "$frozen/extractor_best.pt"
      ;;
    termdiffusum)
      # The 2026-10-22 checkpoint uses the identical frozen configuration and
      # remains available without duplicating a 1.5-GB file.
      ln -sfn /root/autodl-tmp/legalfactaware_full_runtime_20261021/outputs/termdiffusum/model/checkpoint_latest.pt \
        "$frozen/checkpoint_latest.pt"
      cp "$out/model/config.json" "$frozen/config.json"
      ;;
  esac
  find -L "$frozen" -type f -print0 | sort -z | xargs -0 -r sha256sum > "$frozen/SHA256SUMS.txt"
}

run_one() {
  local round="$1" method="$2"
  local root="$BENCH/pipeline/$method/run_$round"
  local pred
  if [[ -f "$root/COMPLETE" ]]; then
    printf '%s\t%s\tSKIP_COMPLETE\n' "$(date -Is)" "$method" > "$STATUS"
    return
  fi
  if [[ -e "$root" ]]; then
    mv "$root" "${root}.incomplete.$(date +%Y%m%dT%H%M%S)"
  fi
  mkdir -p "$root"
  local start end rc count
  start=$(date -Is)
  printf '%s\tround_%s\t%s\tSTART\n' "$start" "$round" "$method" > "$STATUS"
  set +e
  local keep_sende=0
  [[ "$round" == "1" && "$method" == "sende" ]] && keep_sende=1
  ROOT="$root" CODE="$CODE" OUT="$root/outputs" KEEP_SENDE_MODEL="$keep_sende" \
    "$RUNNER" "$method" > "$root/pipeline.log" 2>&1
  rc=$?
  set -e
  end=$(date -Is)
  pred=$(prediction_path "$root" "$method")
  count=0
  [[ -f "$pred" ]] && count=$(wc -l < "$pred")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$round" "$method" "$start" "$end" "$rc" "$count" "$root" >> "$MANIFEST"
  if [[ "$rc" -ne 0 || "$count" -ne 942 ]]; then
    printf '%s\tround_%s\t%s\tFAILED\trc=%s\tcount=%s\n' "$end" "$round" "$method" "$rc" "$count" > "$STATUS"
    touch "$BENCH/FAILED"
    exit 1
  fi
  freeze_for_inference "$root" "$method" "$round"
  archive_and_reclaim "$root" "$method" "$pred"
  printf '%s\tround_%s\t%s\tCOMPLETE\n' "$end" "$round" "$method" > "$STATUS"
  df -h /root/autodl-tmp >> "$root/disk_after_cleanup.txt"
}

round_1=(sse legalfactaware sende memsum termdiffusum)
round_2=(termdiffusum memsum sende legalfactaware sse)
round_3=(legalfactaware termdiffusum sse memsum sende)

for round in 1 2 3; do
  declare -n methods="round_$round"
  for method in "${methods[@]}"; do
    run_one "$round" "$method"
  done
done

printf '%s\tALL_PIPELINE_RUNS_COMPLETE\n' "$(date -Is)" > "$STATUS"
touch "$BENCH/PIPELINE_COMPLETE"
