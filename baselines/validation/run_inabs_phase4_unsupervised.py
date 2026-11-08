#!/usr/bin/env python3
"""Unified-protocol DSDR and CaseSummarizer adaptations on frozen IN-Abs dev."""
from __future__ import annotations

import argparse
import json
import math
import string
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_inabs_phase1 as phase1
import run_inabs_phase2_lr as phase2

import numpy as np
import spacy
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer


SEED = 42
DSDR_LAMBDA = 0.1
METHODS = ("dsdr", "casesummarizer")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def official_count_doc(doc) -> int:
    table = str.maketrans("", "", string.punctuation + string.whitespace)
    return sum(len(token.text.translate(table)) > 0 for token in doc)


def official_counts(nlp, texts: list[str]) -> list[int]:
    return [official_count_doc(doc) for doc in nlp.pipe(texts, batch_size=64)]


def bootstrap(values: np.ndarray, samples: int = 10000) -> dict:
    rng = np.random.default_rng(SEED)
    means = np.empty(samples, dtype=float)
    for start in range(0, samples, 250):
        size = min(250, samples - start)
        means[start:start + size] = values[
            rng.integers(0, len(values), size=(size, len(values)))
        ].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def dsdr_select(sentences: list[str], candidates: list[int], lengths: list[int], budget: int) -> list[int]:
    texts = [sentences[index] for index in candidates]
    try:
        vectors = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 1), norm="l2"
        ).fit_transform(texts)
        gram = (vectors @ vectors.T).toarray().astype(np.float64, copy=False)
    except ValueError:
        gram = np.eye(len(candidates), dtype=np.float64)
    matrix = gram / DSDR_LAMBDA
    remaining = set(range(len(candidates)))
    selected: list[int] = []
    used = 0
    while remaining:
        fitting = [item for item in remaining if lengths[item] and used + lengths[item] <= budget]
        if not fitting:
            break
        scores = np.sum(matrix * matrix, axis=0) / (1.0 + np.diag(matrix))
        best = max(fitting, key=lambda item: (float(scores[item]), -candidates[item]))
        selected.append(candidates[best])
        used += lengths[best]
        remaining.remove(best)
        column = matrix[:, best].copy()
        denominator = 1.0 + matrix[best, best]
        matrix -= np.outer(column, column) / denominator
    return sorted(selected)


def normalized_tokens(doc) -> list[str]:
    return [
        (token.lemma_ or token.text).lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]


def case_features(nlp, cases: list[dict], candidate_lists: list[list[int]], legal_words: set[str]):
    prepared: list[list[dict]] = []
    document_frequency: Counter = Counter()
    for position, (case, candidates) in enumerate(zip(cases, candidate_lists), 1):
        texts = [case["sentences"][index] for index in candidates]
        rows = []
        document_vocabulary = set()
        for doc in nlp.pipe(texts, batch_size=64):
            tokens = normalized_tokens(doc)
            document_vocabulary.update(tokens)
            rows.append({
                "tokens": tokens,
                "entity_tokens": sum(len(ent) for ent in doc.ents),
                "has_digit": any(token.like_num or any(ch.isdigit() for ch in token.text) for token in doc),
            })
        document_frequency.update(document_vocabulary)
        prepared.append(rows)
        if position % 25 == 0 or position == len(cases):
            print(json.dumps({"phase": "casesummarizer_features", "docs": position, "total": len(cases)}), flush=True)

    total_documents = len(cases)
    rankings: list[list[int]] = []
    for case, candidates, rows in zip(cases, candidate_lists, prepared):
        document_counts = Counter(token for row in rows for token in row["tokens"])
        document_total = max(1, sum(document_counts.values()))
        weights = {
            token: (count / document_total) * math.log10(total_documents / document_frequency[token])
            for token, count in document_counts.items()
        }
        base = np.asarray([
            (sum(weights.get(token, 0.0) for token in row["tokens"]) / len(row["tokens"]))
            if row["tokens"] else 0.0
            for row in rows
        ], dtype=float)
        deviation = float(np.std(base))
        scored = []
        for local, row in enumerate(rows):
            denominator = max(1, len(row["tokens"]))
            entity_density = row["entity_tokens"] / denominator
            legal_density = len(set(row["tokens"]) & legal_words) / denominator
            score = base[local] + deviation * (
                0.2 * float(row["has_digit"]) + 0.3 * entity_density + 1.5 * legal_density
            )
            scored.append((score, candidates[local]))
        rankings.append([index for _, index in sorted(scored, key=lambda item: (-item[0], item[1]))])
    return rankings


def pack_ranking(ranking: list[int], candidates: list[int], lengths: list[int], budget: int) -> list[int]:
    length_by_index = dict(zip(candidates, lengths))
    selected, used = [], 0
    for index in ranking:
        length = length_by_index[index]
        if length and used + length <= budget:
            selected.append(index)
            used += length
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--e1-records", type=Path, required=True)
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

    # Full pipeline for CaseSummarizer; the same tokenizer supplies official counts.
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    cases = [phase1.load_case(train_part, case_id) for case_id in dev_ids]
    candidate_lists = [phase2.candidate_indices(len(case["sentences"])) for case in cases]
    budgets = official_counts(nlp, [case["reference"] for case in cases])
    candidate_lengths = [
        official_counts(nlp, [case["sentences"][index] for index in candidates])
        for case, candidates in zip(cases, candidate_lists)
    ]
    legal_words = {
        word.lower()
        for line in args.dictionary.read_text(encoding="utf-8", errors="ignore").splitlines()
        for word in line.split()
        if word.isalpha()
    }

    started = time.time()
    case_rankings = case_features(nlp, cases, candidate_lists, legal_words)
    selections = {method: [] for method in METHODS}
    for position, (case, candidates, lengths, budget, case_ranking) in enumerate(
        zip(cases, candidate_lists, candidate_lengths, budgets, case_rankings), 1
    ):
        selections["dsdr"].append(dsdr_select(case["sentences"], candidates, lengths, budget))
        selections["casesummarizer"].append(pack_ranking(case_ranking, candidates, lengths, budget))
        if position % 25 == 0 or position == len(cases):
            print(json.dumps({"phase": "select", "docs": position, "total": len(cases)}), flush=True)

    evaluator = Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False,
                      apply_avg=False, apply_best=False)
    references = [case["reference"] for case in cases]
    all_records = {}
    metric_arrays = {}
    for method in METHODS:
        hypotheses = [
            phase1.summary(case["sentences"], selected)
            for case, selected in zip(cases, selections[method])
        ]
        verified = official_counts(nlp, hypotheses)
        if any(words > budget for words, budget in zip(verified, budgets)):
            raise RuntimeError(f"budget violation in {method}")
        print(json.dumps({"phase": "pyrouge", "method": method, "docs": len(cases)}), flush=True)
        raw = evaluator.get_scores(hypotheses, [[reference] for reference in references])
        arrays = {
            metric: np.asarray([float(item["f"][0]) for item in raw[metric]], dtype=float)
            for metric in ("rouge-1", "rouge-2", "rouge-l")
        }
        metric_arrays[method] = arrays
        records = []
        for i, case in enumerate(cases):
            records.append({
                "id": case["id"],
                "selected_indices": selections[method][i],
                "reference_spacy_words": budgets[i],
                "prediction_spacy_words": verified[i],
                "budget_violation": False,
                "candidate_count": len(candidate_lists[i]),
                "pyrouge": {metric: float(values[i]) for metric, values in arrays.items()},
            })
        all_records[method] = records
        with (args.output_dir / f"{method}_dev_records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    e1_rows = [json.loads(line) for line in args.e1_records.read_text().splitlines() if line.strip()]
    e1_by_id = {row["id"]: row for row in e1_rows}
    if set(e1_by_id) != set(dev_ids) and not args.dev_limit:
        raise RuntimeError("E1 record IDs do not match frozen dev")
    report = {
        "experiment": "IN-Abs Phase 4A unified unsupervised baseline adaptations",
        "adaptation_status": {
            "dsdr": "published-algorithm budget adaptation",
            "casesummarizer": "restricted faithful Python 3/spaCy adaptation",
        },
        "dev_documents": len(dev_ids),
        "official_test_contents_read": False,
        "candidate_rule": "reference-free evenly-spaced maximum 400 sentences",
        "word_counter": "official Gist spaCy en_core_web_sm punctuation-filtered tokens",
        "budget_violations": {method: 0 for method in METHODS},
        "py_rouge_distribution": "py-rouge==1.1",
        "f1_percent": {},
        "bootstrap_mean_ci": {},
        "paired_difference_vs_e1": {},
        "excluded_empty_train_pairs": excluded,
        "elapsed_seconds": time.time() - started,
    }
    for method, arrays in metric_arrays.items():
        report["f1_percent"][method] = {
            metric: 100 * float(values.mean()) for metric, values in arrays.items()
        }
        report["bootstrap_mean_ci"][method] = {
            metric: bootstrap(values) for metric, values in arrays.items()
        }
        report["paired_difference_vs_e1"][method] = {}
        for metric, values in arrays.items():
            e1 = np.asarray([float(e1_by_id[case_id]["pyrouge"][metric]) for case_id in dev_ids])
            # Positive means E1 is better.
            report["paired_difference_vs_e1"][method][metric] = bootstrap(e1 - values)
    write_json(args.output_dir / "metrics.json", report)
    print(json.dumps({"phase": "complete", **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
