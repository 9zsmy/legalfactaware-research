#!/usr/bin/env python3
"""Evaluate DELTA and frozen E1 on the same IN-Abs pilot documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rouge import Rouge


METRICS = ("rouge-1", "rouge-2", "rouge-l")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bootstrap(values: np.ndarray, seed: int = 42, samples: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        draw = rng.integers(0, len(values), size=len(values))
        means[index] = values[draw].mean()
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--e1-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    predictions = read_jsonl(args.predictions)
    references = {str(row["id"]): row for row in read_jsonl(args.references)}
    e1 = {str(row["id"]): row for row in read_jsonl(args.e1_records)}
    ids = [str(row["id"]) for row in predictions]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise RuntimeError(f"expected 50 unique predictions, got {len(ids)}")
    if any(case_id not in references or case_id not in e1 for case_id in ids):
        raise RuntimeError("prediction/reference/E1 ID mismatch")

    evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=False,
        apply_avg=False,
        apply_best=False,
    )
    hypotheses = [row["prediction"] for row in predictions]
    refs = [references[case_id]["reference"] for case_id in ids]
    raw = evaluator.get_scores(hypotheses, [[reference] for reference in refs])
    delta = {
        metric: np.asarray([float(item["f"][0]) for item in raw[metric]], dtype=float)
        for metric in METRICS
    }
    e1_arrays = {
        metric: np.asarray(
            [
                float(
                    e1[case_id]["tracks"]["643"]["configs"]["ordinary_complete"][
                        "pyrouge"
                    ][metric]
                )
                for case_id in ids
            ],
            dtype=float,
        )
        for metric in METRICS
    }

    per_document = []
    for index, row in enumerate(predictions):
        case_id = ids[index]
        per_document.append(
            {
                "id": case_id,
                "delta_words": int(row["words"]),
                "delta_word_budget_violation": bool(row["word_budget_violation"]),
                "source_truncated": bool(row["source_truncated"]),
                "delta": {metric: float(delta[metric][index]) for metric in METRICS},
                "e1_643_ordinary_complete": {
                    metric: float(e1_arrays[metric][index]) for metric in METRICS
                },
            }
        )

    report = {
        "cases": len(ids),
        "official_test_contents_read": False,
        "evaluation": "py-rouge==1.1 compatible Rouge; max_n=2; no length limit",
        "comparison_status": (
            "diagnostic only: DELTA uses a train-calibrated character target and "
            "may violate the 643-word upper bound"
        ),
        "delta_f1_percent": {
            metric: 100.0 * float(values.mean()) for metric, values in delta.items()
        },
        "e1_643_subset_f1_percent": {
            metric: 100.0 * float(values.mean()) for metric, values in e1_arrays.items()
        },
        "paired_e1_minus_delta": {
            metric: bootstrap(e1_arrays[metric] - delta[metric]) for metric in METRICS
        },
        "delta_word_budget_violations": sum(
            bool(row["word_budget_violation"]) for row in predictions
        ),
        "source_truncated": sum(bool(row["source_truncated"]) for row in predictions),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "quality_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "quality_per_document.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in per_document:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

