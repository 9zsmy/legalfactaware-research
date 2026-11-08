#!/usr/bin/env python3
"""Run the CAIL-table LexRank baseline on frozen IN-Abs dev."""
from __future__ import annotations

import argparse
import json
import string
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_inabs_phase1 as phase1
import run_inabs_phase2_lr as phase2

import numpy as np
import spacy
from rouge import Rouge

SEED = 42


def count_doc(doc):
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
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--e1-records", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dev-limit", type=int, default=0)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if phase1.md5(args.archive) != phase1.EXPECTED_ARCHIVE_MD5:
        raise RuntimeError("official archive MD5 mismatch")
    part = args.dataset_dir / "train-data"
    eligible, excluded = phase1.eligible_train_ids(part, phase1.paired_ids(part))
    train_ids, dev_ids = phase1.fixed_split(eligible)
    if len(train_ids) != 6325 or len(dev_ids) != 703:
        raise RuntimeError("frozen split mismatch")
    if args.dev_limit:
        dev_ids = dev_ids[:args.dev_limit]
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    hypotheses, references, records = [], [], []
    started = time.time()
    for position, case_id in enumerate(dev_ids, 1):
        case = phase1.load_case(part, case_id)
        candidates = phase2.candidate_indices(len(case["sentences"]))
        texts = [case["sentences"][index] for index in candidates]
        scores = phase1.lexrank_scores(texts)
        lengths = [count_doc(doc) for doc in nlp.pipe(texts, batch_size=64)]
        budget = count_doc(nlp(case["reference"]))
        ranking = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]))
        selected_local, used = [], 0
        for local in ranking:
            if lengths[local] and used + lengths[local] <= budget:
                selected_local.append(local)
                used += lengths[local]
        selected = sorted(candidates[local] for local in selected_local)
        prediction = phase1.summary(case["sentences"], selected)
        verified = count_doc(nlp(prediction))
        if verified != used or verified > budget:
            raise RuntimeError(f"budget violation {case_id}")
        hypotheses.append(prediction)
        references.append(case["reference"])
        records.append({"id": case_id, "selected_indices": selected,
                        "prediction_spacy_words": verified, "reference_spacy_words": budget,
                        "candidate_count": len(candidates), "budget_violation": False})
        if position % 25 == 0 or position == len(dev_ids):
            print(json.dumps({"stage": "lexrank", "done": position, "total": len(dev_ids)}), flush=True)
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
    report = {
        "method": "EUR-Lex-Sum LexRank IN-Abs adaptation",
        "dev_documents": len(dev_ids), "official_test_contents_read": False,
        "candidate_cap": 400, "threshold": 0.10, "damping": 0.85,
        "budget_violations": 0, "py_rouge_distribution": "py-rouge==1.1",
        "f1_percent": {metric: 100 * float(values.mean()) for metric, values in arrays.items()},
        "bootstrap_mean_ci": {metric: bootstrap(values) for metric, values in arrays.items()},
        "paired_difference_e1_minus_method": {
            metric: bootstrap(np.asarray([e1[case_id]["pyrouge"][metric] for case_id in dev_ids]) - values)
            for metric, values in arrays.items()
        },
        "excluded": excluded, "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": "complete", **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
