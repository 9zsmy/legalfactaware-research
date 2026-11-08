#!/usr/bin/env python3
"""Fast ROUGE-1/2 screen using the paper-specified py-rouge 1.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rouge import Rouge


parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=Path, required=True)
parser.add_argument("--references", type=Path, required=True)
args = parser.parse_args()

evaluator = Rouge(metrics=["rouge-n"], max_n=2, limit_length=False)
output = {}
for prediction_path in sorted(args.results_dir.glob("*/predictions")):
    variant = prediction_path.parent.name
    names = sorted(path.name for path in prediction_path.glob("*.txt"))
    hypotheses = [" ".join(prediction_path.joinpath(name).read_text(errors="ignore").split()) for name in names]
    references = [" ".join(args.references.joinpath(name).read_text(errors="ignore").split()) for name in names]
    scores = evaluator.get_scores(hypotheses, references)
    output[variant] = {metric: 100 * values["f"] for metric, values in scores.items()}
    print(json.dumps({variant: output[variant]}), flush=True)
(args.results_dir / "exact_pyrouge11_rougen.json").write_text(
    json.dumps(output, indent=2) + "\n", encoding="utf-8"
)
