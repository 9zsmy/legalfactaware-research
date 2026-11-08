#!/usr/bin/env python3
"""Rebuild the preregistered LegalFactAware-v2 selector on CAIL clean.

This deterministic implementation follows the frozen handoff protocol:
character 2--4 gram TF-IDF, 14 explicit legal/structure features, balanced
logistic regression, and greedy marginal selection under an 8 sentence / 450
character budget.  Test is evaluated only with the two preregistered work
points; no test-driven tuning is performed here.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_legal_factaware_lcsts import (
    explicit_features,
    extracted_fact_sets,
    inference_rows,
    rouge2_precision,
    tokens,
    bigrams,
    decorated_text,
)


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:limit] if limit else rows


def sentences(row: dict) -> list[str]:
    raw = row.get("text") or []
    return [
        str(item.get("sentence") or item.get("text") or "")
        if isinstance(item, dict)
        else str(item)
        for item in raw
    ]


def reference(row: dict) -> str:
    return str(row.get("summary") or row.get("reference_summary") or "")


def build_training_rows(rows: list[dict], threshold: float) -> tuple[list[dict], list[int]]:
    features: list[dict] = []
    labels: list[int] = []
    for document_index, row in enumerate(rows, 1):
        document_sentences = sentences(row)
        reference_counts = bigrams(tokens(reference(row)))
        for index, sentence in enumerate(document_sentences):
            features.append(
                {
                    "doc_id": document_index - 1,
                    "index": index,
                    "total": len(document_sentences),
                    "sentence": sentence,
                }
            )
            labels.append(int(rouge2_precision(sentence, reference_counts) > threshold))
        if document_index % 1000 == 0 or document_index == len(rows):
            print(f"labeled {document_index}/{len(rows)} documents", flush=True)
    return features, labels


def fit_model(rows: list[dict], threshold: float, max_features: int) -> dict:
    training_rows, labels = build_training_rows(rows, threshold)
    if len(set(labels)) != 2:
        raise RuntimeError("training labels must contain both classes")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )
    text_matrix = vectorizer.fit_transform(decorated_text(row) for row in training_rows)
    explicit_matrix = csr_matrix([explicit_features(row) for row in training_rows])
    matrix = hstack([text_matrix, explicit_matrix], format="csr")
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    ).fit(matrix, labels)
    return {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "training_cases": len(rows),
        "training_sentences": len(training_rows),
        "positive_labels": sum(labels),
        "label_threshold": threshold,
        "max_features": max_features,
    }


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def char_bigrams(text: str) -> set[tuple[str, str]]:
    characters = [character for character in text if not character.isspace()]
    return set(zip(characters, characters[1:]))


def jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def source_fact_items(text: str) -> set[tuple[str, str]]:
    facts = extracted_fact_sets(text)
    return {(kind, value) for kind, values in facts.items() for value in values}


def select_indices(
    document_sentences: list[str],
    probabilities: list[float],
    *,
    fact_weight: float,
    redundancy_weight: float,
    max_sentences: int,
) -> list[int]:
    normalized = minmax(probabilities)
    bigram_sets = [char_bigrams(sentence) for sentence in document_sentences]
    fact_sets = [source_fact_items(sentence) for sentence in document_sentences]
    selected: list[int] = []
    covered: set[tuple[str, str]] = set()
    remaining = set(range(len(document_sentences)))
    while remaining and len(selected) < max_sentences:
        best_index, best_score = None, -float("inf")
        for index in sorted(remaining):
            redundancy = max(
                (jaccard(bigram_sets[index], bigram_sets[other]) for other in selected),
                default=0.0,
            )
            marginal_facts = len(fact_sets[index] - covered)
            fact_gain = marginal_facts / max(len(fact_sets[index]), 1)
            score = (
                normalized[index]
                + fact_weight * fact_gain
                - redundancy_weight * redundancy
            )
            if score > best_score:
                best_index, best_score = index, score
        assert best_index is not None
        selected.append(best_index)
        covered.update(fact_sets[best_index])
        remaining.remove(best_index)
    return sorted(selected)


def predict_probabilities(model: dict, document_sentences: list[str]) -> list[float]:
    rows = inference_rows(document_sentences)
    if not rows:
        return []
    text_matrix = model["vectorizer"].transform(decorated_text(row) for row in rows)
    explicit_matrix = csr_matrix([explicit_features(row) for row in rows])
    matrix = hstack([text_matrix, explicit_matrix], format="csr")
    return model["classifier"].predict_proba(matrix)[:, 1].tolist()


def generate_split(
    rows: list[dict],
    model: dict,
    output: Path,
    *,
    fact_weight: float,
    redundancy_weight: float,
    max_sentences: int,
    max_chars: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for document_index, row in enumerate(rows, 1):
            document_sentences = sentences(row)
            probabilities = predict_probabilities(model, document_sentences)
            selected = select_indices(
                document_sentences,
                probabilities,
                fact_weight=fact_weight,
                redundancy_weight=redundancy_weight,
                max_sentences=max_sentences,
            )
            summary = "".join(document_sentences[index] for index in selected)[:max_chars]
            handle.write(
                json.dumps(
                    {
                        "id": str(row.get("id") or row.get("case_id") or ""),
                        "generated_summary": summary,
                        "part_indices": [selected],
                        "selected_indices": selected,
                        "reference_summary": reference(row),
                        "document": "".join(document_sentences),
                        "fact_weight": fact_weight,
                        "redundancy_weight": redundancy_weight,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if document_index % 100 == 0 or document_index == len(rows):
                print(f"generated {document_index}/{len(rows)} -> {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=120000)
    parser.add_argument("--label-threshold", type=float, default=0.1)
    parser.add_argument("--redundancy-weight", type=float, default=0.2)
    parser.add_argument("--quality-fact-weight", type=float, default=0.0)
    parser.add_argument("--fact-point-weight", type=float, default=0.05)
    parser.add_argument("--max-sentences", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=450)
    args = parser.parse_args()

    train = read_jsonl(args.train, args.train_limit)
    dev = read_jsonl(args.dev, args.dev_limit)
    test = read_jsonl(args.test, args.test_limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = fit_model(train, args.label_threshold, args.max_features)
    with (args.output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    for split_name, split_rows in (("dev", dev), ("test", test)):
        for point_name, fact_weight in (
            ("quality", args.quality_fact_weight),
            ("fact", args.fact_point_weight),
        ):
            generate_split(
                split_rows,
                model,
                args.output_dir / f"{split_name}_{point_name}_predictions.jsonl",
                fact_weight=fact_weight,
                redundancy_weight=args.redundancy_weight,
                max_sentences=args.max_sentences,
                max_chars=args.max_chars,
            )
    manifest = {
        "identity": "protocol-faithful Mac/server rebuild of LegalFactAware-v2",
        "not_historical_prediction_reuse": True,
        "test_used_for_tuning": False,
        "train_cases": len(train),
        "dev_cases": len(dev),
        "test_cases": len(test),
        "training_sentences": model["training_sentences"],
        "positive_labels": model["positive_labels"],
        "arguments": vars(args) | {
            key: str(value)
            for key, value in vars(args).items()
            if isinstance(value, Path)
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
