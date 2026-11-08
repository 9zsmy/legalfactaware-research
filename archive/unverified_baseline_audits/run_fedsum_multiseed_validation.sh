#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/fedsum_reproduction_20260826
WORK=/root/fedsum_multiseed_validation_20260922
SCRATCH=/dev/shm/fedsum_multiseed_validation_20260922
RUNNER="$ROOT/reproduction/run_official_fixed.py"
DATA="$ROOT/data/CNNDM_presumm_corenlp"
BERT="$ROOT/models/bert-base-uncased"

mkdir -p "$WORK/logs" "$SCRATCH"
trap 'status=$?; printf "%s\n" "$status" > "$WORK/exitcode"' EXIT

test -d "$DATA"
test -d "$BERT"
test -f "$ROOT/outputs/fedavg_cnndm_presumm_seed666/manifest.json"
test -f "$ROOT/outputs/fedsum_cnndm_presumm_seed666/manifest.json"

for seed in 123 2025; do
  for algorithm in FedAvg FedSum; do
    lower=$(printf '%s' "$algorithm" | tr '[:upper:]' '[:lower:]')
    output="$SCRATCH/${lower}_seed${seed}"
    log="$WORK/logs/${lower}_seed${seed}.log"
    test ! -e "$output"
    echo "START algorithm=$algorithm seed=$seed time=$(date -Is)" | tee -a "$WORK/progress.log"
    cd "$ROOT"
    python -u "$RUNNER" \
      --algorithm "$algorithm" \
      --data-dir "$DATA" \
      --output-dir "$output" \
      --bert "$BERT" \
      --clients 20 --fraction 0.2 --local-epochs 2 --rounds 5 \
      --batch-size 32 --learning-rate 0.005 --split Dirichlet01 --seed "$seed" \
      > "$log" 2>&1
    cp "$output/manifest.json" "$WORK/${lower}_seed${seed}_manifest.json"
    grep 'ROUGE-[FR](1/2/l)' "$log" > "$WORK/${lower}_seed${seed}_rouge.txt"
    echo "DONE algorithm=$algorithm seed=$seed time=$(date -Is)" | tee -a "$WORK/progress.log"
  done
done

python - "$ROOT" "$WORK" <<'PY'
import json, re, statistics, sys
from pathlib import Path

root, work = map(Path, sys.argv[1:])
pattern = re.compile(r"ROUGE-([FR])\(1/2/l\):\s*([0-9.]+)/([0-9.]+)/([0-9.]+)")
rows = []
for seed in (666, 123, 2025):
    for algorithm in ("fedavg", "fedsum"):
        if seed == 666:
            path = root / "logs" / f"{algorithm}_cnndm_presumm_seed666.log"
        else:
            path = work / "logs" / f"{algorithm}_seed{seed}.log"
        scores = {"F": [], "R": []}
        for line in path.read_text(errors="ignore").splitlines():
            match = pattern.search(line)
            if match:
                scores[match.group(1)].append(tuple(map(float, match.groups()[1:])))
        if not scores["F"] or not scores["R"]:
            raise RuntimeError(f"No ROUGE found in {path}")
        best_index = max(range(len(scores["F"])), key=lambda i: sum(scores["F"][i]))
        rows.append({"algorithm": algorithm, "seed": seed,
                     "best_round_index_1based": best_index + 1,
                     "rouge_f": scores["F"][best_index],
                     "rouge_r": scores["R"][best_index]})

summary = {}
for algorithm in ("fedavg", "fedsum"):
    part = [row for row in rows if row["algorithm"] == algorithm]
    summary[algorithm] = {
        key: {"mean": [statistics.mean(row[key][i] for row in part) for i in range(3)],
              "stdev": [statistics.stdev(row[key][i] for row in part) for i in range(3)]}
        for key in ("rouge_f", "rouge_r")
    }
summary["fedsum_minus_fedavg_mean"] = {
    key: [summary["fedsum"][key]["mean"][i] - summary["fedavg"][key]["mean"][i]
          for i in range(3)] for key in ("rouge_f", "rouge_r")
}
report = {"protocol": "FedSum paper 2K CNNDM, PreSumm+CoreNLP, Dirichlet01, 3 seeds",
          "paper_targets": {"fedavg_f": [34.11, 12.87, 27.17],
                            "fedsum_f": [35.71, 14.18, 28.69]},
          "rows": rows, "summary": summary}
(work / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

tar -czf "$WORK/fedsum_multiseed_validation_compact.tgz" \
  -C "$WORK" logs progress.log metrics.json \
  fedavg_seed123_manifest.json fedsum_seed123_manifest.json \
  fedavg_seed2025_manifest.json fedsum_seed2025_manifest.json \
  fedavg_seed123_rouge.txt fedsum_seed123_rouge.txt \
  fedavg_seed2025_rouge.txt fedsum_seed2025_rouge.txt
