#!/usr/bin/env python3
"""Evaluate a Phase 4B prediction directory under the frozen Phase 3C protocol."""
from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_inabs_phase1 as phase1

import numpy as np
import spacy
from rouge import Rouge

SEED = 42


def count(doc) -> int:
    table = str.maketrans("", "", string.punctuation + string.whitespace)
    return sum(len(token.text.translate(table)) > 0 for token in doc)


def bootstrap(values, samples=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    means = np.empty(samples)
    for start in range(0, samples, 250):
        size = min(250, samples - start)
        means[start:start + size] = values[rng.integers(0, len(values), (size, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--predictions-dir", type=Path, required=True)
    p.add_argument("--e1-records", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    part = args.dataset_dir / "train-data"
    eligible, excluded = phase1.eligible_train_ids(part, phase1.paired_ids(part))
    _, dev_ids = phase1.fixed_split(eligible)
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    hypotheses, references, records = [], [], []
    for case_id in dev_ids:
        case = phase1.load_case(part, case_id)
        prediction_path = args.predictions_dir / f"{case_id}.txt"
        if not prediction_path.exists():
            raise RuntimeError(f"missing prediction: {prediction_path}")
        prediction = prediction_path.read_text(encoding="utf-8", errors="ignore").strip()
        pred_words, ref_words = count(nlp(prediction)), count(nlp(case["reference"]))
        if pred_words > ref_words:
            raise RuntimeError(f"budget violation: {case_id} {pred_words}>{ref_words}")
        hypotheses.append(prediction)
        references.append(case["reference"])
        records.append({"id": case_id, "prediction_spacy_words": pred_words,
                        "reference_spacy_words": ref_words, "budget_violation": False})
    evaluator = Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False,
                      apply_avg=False, apply_best=False)
    raw = evaluator.get_scores(hypotheses, [[reference] for reference in references])
    arrays = {metric: np.asarray([float(row["f"][0]) for row in raw[metric]])
              for metric in ("rouge-1", "rouge-2", "rouge-l")}
    for i, record in enumerate(records):
        record["pyrouge"] = {metric: float(values[i]) for metric, values in arrays.items()}
    with (args.output_dir / "dev_records.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    e1 = {row["id"]: row for row in map(json.loads, args.e1_records.read_text().splitlines())}
    def e1_score(case_id, metric):
        row = e1[case_id]
        if "pyrouge" in row:
            return row["pyrouge"][metric]
        return row["models"]["e1_word1_2"]["pyrouge"][metric]

    report = {
        "method": args.method, "dev_documents": 703, "official_test_contents_read": False,
        "budget_violations": 0, "py_rouge_distribution": "py-rouge==1.1",
        "f1_percent": {metric: 100 * float(values.mean()) for metric, values in arrays.items()},
        "bootstrap_mean_ci": {metric: bootstrap(values) for metric, values in arrays.items()},
        "paired_difference_e1_minus_method": {
            metric: bootstrap(np.asarray([e1_score(case_id, metric) for case_id in dev_ids]) - values)
            for metric, values in arrays.items()
        },
        "excluded": excluded,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": "complete", **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
