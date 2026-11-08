#!/usr/bin/env bash
set -uo pipefail

INABS_ROOT=/root/autodl-tmp/inabs_lfa_phase2
SWEEP_ROOT=/root/autodl-tmp/termdiffusum_reproduction_20260817/outputs/inabs_200_decoder_sweep
DATASET_ROOT="$INABS_ROOT/dataset/IN-Abs"
E1_RECORDS="$INABS_ROOT/outputs/phase3b_official_eval/official_pyrouge_per_case.jsonl"

rm -f "$SWEEP_ROOT/parallel_eval_complete.json"
pids=()
names=()

for records in "$SWEEP_ROOT"/*/dev_records.jsonl; do
  variant=$(basename "$(dirname "$records")")
  output_dir="$SWEEP_ROOT/$variant/eval"
  mkdir -p "$output_dir"
  env PYTHONPATH=/tmp/kmm_pyrouge/lib/python3.10/site-packages \
    python "$INABS_ROOT/evaluate_phase4b_predictions.py" \
      --method "TermDiffuSum-INAbs-200-$variant" \
      --dataset-dir "$DATASET_ROOT" \
      --predictions-dir "$SWEEP_ROOT/$variant/predictions" \
      --e1-records "$E1_RECORDS" \
      --output-dir "$output_dir" \
      > "$output_dir/eval.log" 2>&1 &
  pids+=("$!")
  names+=("$variant")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "FAILED ${names[$index]}" >&2
    failed=$((failed + 1))
  fi
done

python - "$SWEEP_ROOT" "$failed" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
failed = int(sys.argv[2])
rows = []
for metrics_path in sorted(root.glob("*/eval/metrics.json")):
    report = json.loads(metrics_path.read_text())
    rows.append({
        "variant": metrics_path.parents[1].name,
        "f1_percent": report["f1_percent"],
        "paired_difference_e1_minus_method": report["paired_difference_e1_minus_method"],
        "budget_violations": report["budget_violations"],
    })
rows.sort(key=lambda row: (
    row["f1_percent"]["rouge-2"],
    row["f1_percent"]["rouge-l"],
    row["f1_percent"]["rouge-1"],
), reverse=True)
result = {
    "status": "complete" if failed == 0 and len(rows) == 12 else "incomplete",
    "failed": failed,
    "completed": len(rows),
    "official_test_contents_read": False,
    "ranking_by_rouge2": rows,
}
(root / "parallel_eval_complete.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

exit "$failed"
