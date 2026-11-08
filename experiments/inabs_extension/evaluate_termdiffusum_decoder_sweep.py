#!/usr/bin/env python3
"""Evaluate all frozen-checkpoint decoder variants in one py-rouge pass."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from rouge import Rouge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_inabs_phase1 as phase1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--e1-records", type=Path, required=True)
    args = parser.parse_args()

    part = args.dataset_dir / "train-data"
    eligible, _ = phase1.eligible_train_ids(part, phase1.paired_ids(part))
    _, dev_ids = phase1.fixed_split(eligible)
    references_by_id = {
        case_id: phase1.load_case(part, case_id)["reference"] for case_id in dev_ids
    }
    e1_rows = {
        row["id"]: row for row in map(json.loads, args.e1_records.read_text().splitlines())
    }

    variant_dirs = sorted(
        path for path in args.sweep_dir.iterdir()
        if path.is_dir() and (path / "dev_records.jsonl").exists()
    )
    all_hypotheses = []
    all_references = []
    slices = {}
    selection_stats = {}
    for variant_dir in variant_dirs:
        rows = [
            json.loads(line)
            for line in (variant_dir / "dev_records.jsonl").read_text().splitlines()
            if line.strip()
        ]
        by_id = {row["id"]: row for row in rows}
        if set(by_id) != set(dev_ids):
            raise RuntimeError(
                f"{variant_dir.name}: id mismatch, got {len(by_id)}, expected {len(dev_ids)}"
            )
        if any(row["budget_violation"] for row in rows):
            raise RuntimeError(f"{variant_dir.name}: decoder-reported budget violation")
        start = len(all_hypotheses)
        ordered = [by_id[case_id] for case_id in dev_ids]
        all_hypotheses.extend(row["prediction"] for row in ordered)
        all_references.extend(references_by_id[case_id] for case_id in dev_ids)
        slices[variant_dir.name] = (start, len(all_hypotheses))
        selection_stats[variant_dir.name] = {
            "mean_selected_sentences": float(
                np.mean([len(row["selected_indices"]) for row in ordered])
            ),
            "mean_budget_utilization": float(
                np.mean([
                    row["prediction_spacy_words"] / row["reference_spacy_words"]
                    if row["reference_spacy_words"] else 0.0
                    for row in ordered
                ])
            ),
        }

    started = time.time()
    evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=False,
        apply_avg=False,
        apply_best=False,
    )
    raw = evaluator.get_scores(
        all_hypotheses, [[reference] for reference in all_references]
    )
    arrays = {
        metric: np.asarray([float(row["f"][0]) for row in raw[metric]])
        for metric in ("rouge-1", "rouge-2", "rouge-l")
    }

    e1_arrays = {}
    for metric in arrays:
        values = []
        for case_id in dev_ids:
            row = e1_rows[case_id]
            if "pyrouge" in row:
                values.append(row["pyrouge"][metric])
            else:
                values.append(row["models"]["e1_word1_2"]["pyrouge"][metric])
        e1_arrays[metric] = np.asarray(values)

    results = {}
    for name, (start, end) in slices.items():
        scores = {metric: values[start:end] for metric, values in arrays.items()}
        results[name] = {
            **selection_stats[name],
            "budget_violations": 0,
            "f1_percent": {
                metric: 100.0 * float(values.mean()) for metric, values in scores.items()
            },
            "e1_minus_variant_percent": {
                metric: 100.0 * float((e1_arrays[metric] - values).mean())
                for metric, values in scores.items()
            },
        }
    ranking = sorted(
        results,
        key=lambda name: (
            results[name]["f1_percent"]["rouge-2"],
            results[name]["f1_percent"]["rouge-l"],
            results[name]["f1_percent"]["rouge-1"],
        ),
        reverse=True,
    )
    report = {
        "status": "complete",
        "official_test_contents_read": False,
        "dev_documents": len(dev_ids),
        "variants": len(results),
        "evaluation_seconds": round(time.time() - started, 3),
        "ranking_by_rouge2": ranking,
        "results": results,
    }
    (args.sweep_dir / "sweep_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
