#!/usr/bin/env python3
"""Evaluate IN-Abs predictions without enforcing a reference-length cap.

This is only for an author's-source decoder track whose original decoding
protocol has no per-document word budget.  It records budget violations but
does not compare the resulting scores directly with budget-constrained models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rouge import Rouge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_inabs_phase1 as phase1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    part = args.dataset_dir / "train-data"
    eligible, excluded = phase1.eligible_train_ids(part, phase1.paired_ids(part))
    _, dev_ids = phase1.fixed_split(eligible)
    rows = {
        row["id"]: row
        for row in map(json.loads, args.records.read_text(encoding="utf-8").splitlines())
    }
    if set(rows) != set(dev_ids):
        raise RuntimeError(f"prediction id mismatch: {len(rows)} vs {len(dev_ids)}")
    hypotheses = [rows[case_id]["prediction"] for case_id in dev_ids]
    references = [phase1.load_case(part, case_id)["reference"] for case_id in dev_ids]
    evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=False,
        apply_avg=False,
        apply_best=False,
    )
    raw = evaluator.get_scores(hypotheses, [[reference] for reference in references])
    arrays = {
        metric: np.asarray([float(row["f"][0]) for row in raw[metric]])
        for metric in ("rouge-1", "rouge-2", "rouge-l")
    }
    report = {
        "method": args.method,
        "track": "authors-source-decoder-without-IN-Abs-length-cap",
        "directly_comparable_to_budget_constrained_table": False,
        "official_test_contents_read": False,
        "dev_documents": len(dev_ids),
        "budget_violations": sum(bool(row["budget_violation"]) for row in rows.values()),
        "mean_selected_sentences": float(
            np.mean([len(row["selected_indices"]) for row in rows.values()])
        ),
        "f1_percent": {
            metric: 100.0 * float(values.mean()) for metric, values in arrays.items()
        },
        "excluded": excluded,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
