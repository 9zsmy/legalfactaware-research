#!/usr/bin/env python3
"""Audit plausible py-rouge length/input variants against two UK-Abs anchors."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


PAPER = {
    "summarunner": {"rouge-1": 50.2, "rouge-2": 20.5, "rouge-l": 23.7},
    "gist": {"rouge-1": 42.7, "rouge-2": 13.2, "rouge-l": 21.5},
}


def score(Rouge, hypotheses: list[str], references: list[str], variant: dict) -> dict[str, float]:
    evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"], max_n=2,
        limit_length=variant["limit_length"],
        length_limit=variant.get("length_limit", 665),
        length_limit_type=variant.get("length_limit_type", "bytes"),
        stemming=variant.get("stemming", True),
        apply_avg=False, apply_best=False,
    )
    raw = evaluator.get_scores(hypotheses, [[reference] for reference in references])
    return {
        metric: 100.0 * float(np.mean([float(item["f"][0]) for item in raw[metric]]))
        for metric in ("rouge-1", "rouge-2", "rouge-l")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarunner", type=Path, required=True)
    parser.add_argument("--gist", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--vendor-pyrouge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vendor_pyrouge))
    from rouge import Rouge

    names = sorted(path.name for path in args.summarunner.glob("*.txt"))
    if len(names) != 100 or any(not (args.gist / name).exists() for name in names):
        raise RuntimeError("prediction sets do not share the official 100 files")
    refs = [(args.references / name).read_text(encoding="utf-8") for name in names]
    predictions = {
        "summarunner": [(args.summarunner / name).read_text(encoding="utf-8") for name in names],
        "gist": [(args.gist / name).read_text(encoding="utf-8") for name in names],
    }
    variants = [{"name": "paper_stated_no_limit", "limit_length": False, "stemming": True}]
    for limit in (500, 665, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000):
        variants.append({"name": f"bytes_{limit}", "limit_length": True,
                         "length_limit": limit, "length_limit_type": "bytes", "stemming": True})
    for limit in (100, 200, 300, 400, 500, 750, 1000, 1500):
        variants.append({"name": f"words_{limit}", "limit_length": True,
                         "length_limit": limit, "length_limit_type": "words", "stemming": True})
    variants.append({"name": "no_limit_no_stemming", "limit_length": False, "stemming": False})

    rows = []
    for index, variant in enumerate(variants, 1):
        result = {method: score(Rouge, values, refs, variant) for method, values in predictions.items()}
        squared = []
        for method in PAPER:
            for metric in PAPER[method]:
                squared.append((result[method][metric] - PAPER[method][metric]) ** 2)
        row = {"variant": variant, "scores": result,
               "joint_rmse": math.sqrt(float(np.mean(squared)))}
        rows.append(row)
        print(json.dumps({"done": index, "total": len(variants), **row}), flush=True)
    rows.sort(key=lambda row: row["joint_rmse"])
    report = {"paper": PAPER, "best_by_joint_rmse": rows[:10], "all": rows}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
