#!/usr/bin/env python3
"""Compare ROUGE APIs for published UK-Abs SummaRuNNer outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--pltrdy-vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.pltrdy_vendor))
    from rouge import Rouge as PltrdyRouge

    names = sorted(path.name for path in args.references.glob("*.txt"))
    references = [
        (args.references / name).read_text(encoding="utf-8") for name in names
    ]
    google = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )
    pltrdy = PltrdyRouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    report = {}
    for directory in sorted(path for path in args.base_dir.iterdir() if path.is_dir()):
        prediction_dir = directory / "predictions"
        if not prediction_dir.exists():
            continue
        hypotheses = [
            (prediction_dir / name).read_text(encoding="utf-8") for name in names
        ]
        google_rows = [google.score(reference, hypothesis)
                       for reference, hypothesis in zip(references, hypotheses)]
        pltrdy_result = pltrdy.get_scores(hypotheses, references, avg=True)
        report[directory.name] = {
            "google_rouge_score_f1_percent": {
                metric: 100.0 * float(np.mean([
                    row[metric].fmeasure for row in google_rows
                ]))
                for metric in ("rouge1", "rouge2", "rougeL", "rougeLsum")
            },
            "pltrdy_rouge_1_0_1_f1_percent": {
                metric: 100.0 * float(pltrdy_result[metric]["f"])
                for metric in ("rouge-1", "rouge-2", "rouge-l")
            },
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
