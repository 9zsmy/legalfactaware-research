#!/usr/bin/env python3
"""Evaluate SummaRuNNer predictions with the AACL paper's py-rouge protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--vendor-pyrouge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vendor_pyrouge))
    from rouge import Rouge

    names = sorted(path.name for path in args.predictions.glob("*.txt"))
    if len(names) != 100:
        raise RuntimeError(f"expected 100 predictions, got {len(names)}")
    hypotheses = [(args.predictions / name).read_text(encoding="utf-8") for name in names]
    references = [(args.references / name).read_text(encoding="utf-8") for name in names]
    evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False,
        apply_avg=False, apply_best=False,
    )
    raw = evaluator.get_scores(hypotheses, [[reference] for reference in references])
    metrics = {
        metric: 100.0 * float(np.mean([float(item["f"][0]) for item in raw[metric]]))
        for metric in ("rouge-1", "rouge-2", "rouge-l")
    }
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    paper = {"rouge-1": 50.2, "rouge-2": 20.5, "rouge-l": 23.7}
    def used_words(row):
        return row.get("used_words", row.get("used_whitespace_words"))

    report = {
        "method": "SummaRuNNer UK-Abs evaluation",
        "documents": len(names), "py_rouge_distribution": "py-rouge==1.1",
        "paper_protocol": {"max_n": 2, "limit_length": False},
        "f1_percent": metrics, "paper_f1_percent": paper,
        "difference_reproduction_minus_paper": {key: metrics[key] - paper[key] for key in paper},
        "budget_violations": sum(used_words(row) > row["budget"] for row in records),
        "official_inabs_test_read": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
