#!/usr/bin/env python3
"""Decode frozen E1 with the official Gist spaCy word counter and py-rouge.

No fitting or parameter selection occurs. Official test contents are not read.
"""
from __future__ import annotations

import argparse
import json
import pickle
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
from scipy.sparse import csr_matrix, hstack


SEED = 42
MODEL = "e1_word1_2"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def official_count_doc(doc) -> int:
    """Exact count_word logic in official Gist generate_summary.py."""
    table = str.maketrans("", "", string.punctuation + string.whitespace)
    return sum(len(token.text.translate(table)) > 0 for token in doc)


def official_counts(nlp, texts: list[str]) -> list[int]:
    return [official_count_doc(doc) for doc in nlp.pipe(texts, batch_size=64)]


def normalize(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    return (values - low) / (high - low) if high > low else np.full_like(values, 0.5)


def decode_with_lengths(sentences, candidates, probabilities, lengths, budget):
    relevance = normalize(probabilities)
    grams = [phase2.word_bigram_set(sentences[index]) for index in candidates]
    remaining, selected, used = set(range(len(candidates))), [], 0
    max_redundancy = np.zeros(len(candidates), dtype=np.float32)
    while remaining:
        fitting = [item for item in remaining if lengths[item] and used + lengths[item] <= budget]
        if not fitting:
            break
        best = max(
            fitting,
            key=lambda item: (
                float(relevance[item]) - phase2.REDUNDANCY_WEIGHT * float(max_redundancy[item]),
                float(relevance[item]),
                -candidates[item],
            ),
        )
        selected.append(candidates[best])
        used += lengths[best]
        remaining.remove(best)
        for item in remaining:
            max_redundancy[item] = max(
                max_redundancy[item], phase2.jaccard(grams[item], grams[best])
            )
    return sorted(selected), used


def feature_matrix(vectorizer, sentences, candidates):
    total = len(sentences)
    decorated = [phase2.decorated(sentences[index], index, total) for index in candidates]
    text = vectorizer.transform(decorated)
    extras = csr_matrix(
        [phase2.explicit20(sentences[index], index, total) for index in candidates],
        dtype=np.float32,
    )
    return hstack([text, extras], format="csr")


def bootstrap(values: np.ndarray, samples: int = 10000) -> dict:
    rng = np.random.default_rng(SEED)
    means = np.empty(samples, dtype=float)
    for start in range(0, samples, 250):
        size = min(250, samples - start)
        means[start:start+size] = values[
            rng.integers(0, len(values), size=(size, len(values)))
        ].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--phase3-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-limit", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if phase1.md5(args.archive) != phase1.EXPECTED_ARCHIVE_MD5:
        raise RuntimeError("official archive MD5 mismatch")

    train_part = args.dataset_dir / "train-data"
    eligible, excluded = phase1.eligible_train_ids(train_part, phase1.paired_ids(train_part))
    train_ids, dev_ids = phase1.fixed_split(eligible)
    if len(train_ids) != 6325 or len(dev_ids) != 703:
        raise RuntimeError("frozen split mismatch")
    if args.dev_limit:
        dev_ids = dev_ids[:args.dev_limit]
    with args.phase3_model.open("rb") as handle:
        bundle = pickle.load(handle)
    vectorizer = bundle["vectorizers"]["word"]
    classifier = bundle["models"][MODEL]
    try:
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    except Exception as error:
        raise RuntimeError("official en_core_web_sm tokenizer is required") from error

    started = time.time()
    records, hypotheses, references = [], [], []
    for position, case_id in enumerate(dev_ids, 1):
        case = phase1.load_case(train_part, case_id)
        candidates = phase2.candidate_indices(len(case["sentences"]))
        matrix = feature_matrix(vectorizer, case["sentences"], candidates)
        probabilities = classifier.predict_proba(matrix)[:, 1]
        reference_budget = official_counts(nlp, [case["reference"]])[0]
        candidate_lengths = official_counts(nlp, [case["sentences"][index] for index in candidates])
        selected, decoder_words = decode_with_lengths(
            case["sentences"], candidates, probabilities, candidate_lengths, reference_budget
        )
        prediction = phase1.summary(case["sentences"], selected)
        verified_words = official_counts(nlp, [prediction])[0]
        if verified_words != decoder_words or verified_words > reference_budget:
            raise RuntimeError(f"official word-budget violation for {case_id}")
        records.append({
            "id": case_id,
            "selected_indices": selected,
            "reference_spacy_words": reference_budget,
            "prediction_spacy_words": verified_words,
            "budget_violation": False,
            "candidate_count": len(candidates),
        })
        hypotheses.append(prediction)
        references.append(case["reference"])
        if position % 25 == 0 or position == len(dev_ids):
            print(json.dumps({"phase": "decode", "docs": position, "total": len(dev_ids)}), flush=True)

    evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False,
        apply_avg=False, apply_best=False,
    )
    print(json.dumps({"phase": "pyrouge", "docs": len(dev_ids)}), flush=True)
    raw = evaluator.get_scores(hypotheses, [[reference] for reference in references])
    arrays = {
        metric: np.asarray([float(item["f"][0]) for item in raw[metric]], dtype=float)
        for metric in ("rouge-1", "rouge-2", "rouge-l")
    }
    for index, record in enumerate(records):
        record["pyrouge"] = {metric: float(values[index]) for metric, values in arrays.items()}
    with (args.output_dir / "dev_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    report = {
        "experiment": "IN-Abs Phase 3C frozen E1 official spaCy decode",
        "model": MODEL,
        "models_trained_or_changed": False,
        "train_documents": 6325,
        "dev_documents": len(dev_ids),
        "official_test_contents_read": False,
        "candidate_rule": "reference-free evenly-spaced maximum 400 sentences",
        "word_counter": "official Gist spaCy en_core_web_sm punctuation-filtered tokens",
        "spacy_version": spacy.__version__,
        "spacy_model": nlp.meta.get("name"),
        "budget_violations": sum(row["budget_violation"] for row in records),
        "mean_reference_words": float(np.mean([row["reference_spacy_words"] for row in records])),
        "mean_prediction_words": float(np.mean([row["prediction_spacy_words"] for row in records])),
        "py_rouge_distribution": "py-rouge==1.1",
        "py_rouge_configuration": {"max_n": 2, "limit_length": False, "other_parameters": "defaults"},
        "f1_percent": {metric: 100 * float(values.mean()) for metric, values in arrays.items()},
        "bootstrap_mean_ci": {metric: bootstrap(values) for metric, values in arrays.items()},
        "excluded_empty_train_pairs": excluded,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "metrics.json", report)
    print(json.dumps({"phase": "complete", **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
