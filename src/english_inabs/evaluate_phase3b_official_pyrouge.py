#!/usr/bin/env python3
"""Re-evaluate frozen IN-Abs phase-3 dev extracts with paper py-rouge.

No model is loaded or trained. Official test contents are never read. The
script reconstructs the already-selected official-length dev summaries and
applies py-rouge 1.1 with the AACL paper's stated no-limit configuration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_inabs_phase1 as phase1

import numpy as np
from rouge import Rouge


SEED = 42
MODELS = ("e0_char2_4", "e1_word1_2", "e2_word1_2_char3_5")
METRICS = ("rouge-1", "rouge-2", "rouge-l")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bootstrap(delta: np.ndarray, samples: int = 10000) -> dict:
    rng = np.random.default_rng(SEED)
    means = np.empty(samples, dtype=float)
    for start in range(0, samples, 250):
        size = min(250, samples - start)
        means[start:start+size] = delta[rng.integers(0, len(delta), size=(size, len(delta)))].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "p_two_sided": float(min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0)))),
    }


def score_arrays(hypotheses: list[str], references: list[str]) -> dict[str, np.ndarray]:
    evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=False,
        apply_avg=False,
        apply_best=False,
    )
    raw = evaluator.get_scores(hypotheses, [[reference] for reference in references])
    return {
        metric: np.asarray([float(item["f"][0]) for item in raw[metric]], dtype=float)
        for metric in METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--phase3-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_part = args.dataset_dir / "train-data"
    eligible, excluded = phase1.eligible_train_ids(train_part, phase1.paired_ids(train_part))
    _, dev_ids = phase1.fixed_split(eligible)
    records = read_jsonl(args.phase3_records)
    if len(records) != 703 or [row["id"] for row in records] != dev_ids:
        raise RuntimeError("phase-3 records do not match frozen 703-document dev split")

    references: list[str] = []
    hypotheses = {model: [] for model in MODELS}
    per_case: list[dict] = []
    for row in records:
        case = phase1.load_case(train_part, row["id"])
        reference = case["reference"]
        references.append(reference)
        detail = {
            "id": row["id"],
            "reference_regex_words": len(phase1.word_tokens(reference)),
            "reference_whitespace_words": len(reference.split()),
            "models": {},
        }
        for model in MODELS:
            indices = row["selected"]["official_reference_length"][model]
            prediction = phase1.summary(case["sentences"], indices)
            hypotheses[model].append(prediction)
            detail["models"][model] = {
                "selected_indices": indices,
                "prediction_regex_words": len(phase1.word_tokens(prediction)),
                "prediction_whitespace_words": len(prediction.split()),
            }
        per_case.append(detail)

    arrays = {model: score_arrays(hypotheses[model], references) for model in MODELS}
    aggregate = {
        model: {metric: 100 * float(arrays[model][metric].mean()) for metric in METRICS}
        for model in MODELS
    }
    comparisons = {}
    for left, right in (
        ("e0_char2_4", "e1_word1_2"),
        ("e0_char2_4", "e2_word1_2_char3_5"),
        ("e1_word1_2", "e2_word1_2_char3_5"),
    ):
        comparisons[f"{right}_minus_{left}"] = {
            metric: bootstrap(arrays[right][metric] - arrays[left][metric]) for metric in METRICS
        }
    for index, detail in enumerate(per_case):
        for model in MODELS:
            detail["models"][model]["pyrouge"] = {
                metric: float(arrays[model][metric][index]) for metric in METRICS
            }

    length_audit = {}
    for model in MODELS:
        regex_violation = sum(
            row["models"][model]["prediction_regex_words"] > row["reference_regex_words"]
            for row in per_case
        )
        whitespace_violation = sum(
            row["models"][model]["prediction_whitespace_words"] > row["reference_whitespace_words"]
            for row in per_case
        )
        length_audit[model] = {
            "regex_budget_violations": regex_violation,
            "whitespace_budget_violations": whitespace_violation,
            "mean_prediction_regex_words": float(np.mean([row["models"][model]["prediction_regex_words"] for row in per_case])),
            "mean_prediction_whitespace_words": float(np.mean([row["models"][model]["prediction_whitespace_words"] for row in per_case])),
        }

    report = {
        "experiment": "Phase 3B existing-output official py-rouge alignment",
        "dataset": "frozen 703-document development split from official IN-Abs train",
        "models_trained_or_changed": False,
        "official_test_contents_read": False,
        "py_rouge_distribution": "py-rouge==1.1 (Diego999/py-rouge)",
        "paper_configuration": {
            "metrics": ["rouge-n", "rouge-l"], "max_n": 2, "limit_length": False,
            "stemming_default": True, "alpha_default": 0.5,
            "weight_factor_default": 1.0, "ensure_compatibility_default": True,
        },
        "aggregate_f1_percent": aggregate,
        "paired_bootstrap": comparisons,
        "length_audit": length_audit,
        "excluded_empty_official_train_pairs": excluded,
    }
    write_json(args.output_dir / "official_pyrouge_metrics.json", report)
    with (args.output_dir / "official_pyrouge_per_case.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_case:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
