#!/usr/bin/env python3
"""Dev-only LegalFactAware-v2.2 budget-aware oracle-supervision experiment.

The LegalFactAware-v2.1-Budget decoder and all representation/classifier
settings are frozen.  The only changed experimental variable is the binary
training target: membership in a character ROUGE-1+ROUGE-2 F1 greedy oracle
under the same 450-character / 8-sentence complete-sentence constraints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression

import run_legalfactaware_v21_budget_decoder_dev as frozen
import run_legalfactaware_v2_cail_clean as base
from evaluate_sse_cail import character_tokens, rouge_n


EXPERIMENT = "LegalFactAware-v2.2-BudgetAwareOracleSupervision"
EXPECTED_V21_DECODER_SHA256 = "e8084e0b0ca6e1803a9bedd1c51bb36143c09136a22d70e0da2e3c19b74c93dd"
EXPECTED_FROZEN_MODEL_SHA256 = "22689296b98f398bba1f4b591db3e02e74703284da1ffdbeed1ea94cba334fac"
EXPECTED_TRAIN_SHA256 = "8b19e6a1aa086e55c3a1d694e86cb4cf5abfe6929ee1258a3621b291bde564a2"
EXPECTED_DEV_SHA256 = "1a056c68dcfa5f11e9574d5f1b818fa475bb5607ea7f6565e2868072a55de2c0"
EXPECTED_V21_DEV_PREDICTIONS_SHA256 = "ba539d5b6e7a36177b4d38d15238a78ae523d42b65a799f81f6cfa5ed0a513eb"
TRAIN_CASES = 7495
DEV_CASES = 942
MAX_CHARS = 450
MAX_SENTENCES = 8
FACT_WEIGHT = 0.0
REDUNDANCY_WEIGHT = 0.2
WEAK_LABEL_THRESHOLD = 0.1
ORACLE_OBJECTIVE = "character_rouge1_f1_plus_character_rouge2_f1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"unexpected {label} SHA256: {actual}; expected {expected}")
    return actual


def oracle_score(reference: str, summary: str) -> float:
    return rouge_n(reference, summary, 1, character_tokens) + rouge_n(
        reference, summary, 2, character_tokens
    )


def select_budget_oracle(document_sentences: list[str], reference: str) -> tuple[list[int], float]:
    selected: list[int] = []
    remaining = set(range(len(document_sentences)))
    current_score = 0.0
    while remaining and len(selected) < MAX_SENTENCES:
        best_index = None
        best_score = current_score
        for index in sorted(remaining):
            ordered = sorted(selected + [index])
            candidate = "".join(document_sentences[item] for item in ordered)
            if len(candidate) > MAX_CHARS:
                continue
            score = oracle_score(reference, candidate)
            if score > best_score + 1e-12:
                best_index = index
                best_score = score
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
        current_score = best_score
    selected.sort()
    summary = "".join(document_sentences[index] for index in selected)
    assert len(selected) <= MAX_SENTENCES
    assert len(summary) <= MAX_CHARS
    return selected, current_score


def binary_agreement(left: list[int], right: list[int]) -> dict:
    if len(left) != len(right):
        raise ValueError("label length mismatch")
    tp = sum(a == 1 and b == 1 for a, b in zip(left, right))
    tn = sum(a == 0 and b == 0 for a, b in zip(left, right))
    fp = sum(a == 1 and b == 0 for a, b in zip(left, right))
    fn = sum(a == 0 and b == 1 for a, b in zip(left, right))
    total = len(left)
    return {
        "cases": total,
        "agreement": (tp + tn) / total if total else 1.0,
        "positive_jaccard": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "left_positive_right_positive": tp,
        "left_positive_right_negative": fp,
        "left_negative_right_positive": fn,
        "left_negative_right_negative": tn,
    }


def build_oracle_training(rows: list[dict]) -> tuple[list[dict], list[int], dict, list[dict]]:
    training_rows: list[dict] = []
    oracle_labels: list[int] = []
    weak_labels: list[int] = []
    official_labels: list[int] = []
    oracle_documents: list[dict] = []
    for document_index, row in enumerate(rows, 1):
        document_sentences = base.sentences(row)
        ref = base.reference(row)
        selected, score = select_budget_oracle(document_sentences, ref)
        selected_set = set(selected)
        reference_counts = base.bigrams(base.tokens(ref))
        row_official = frozen.source_labels(row)
        for index, sentence in enumerate(document_sentences):
            training_rows.append(
                {
                    "doc_id": document_index - 1,
                    "index": index,
                    "total": len(document_sentences),
                    "sentence": sentence,
                }
            )
            oracle_labels.append(int(index in selected_set))
            weak_labels.append(
                int(base.rouge2_precision(sentence, reference_counts) > WEAK_LABEL_THRESHOLD)
            )
            official_labels.append(row_official[index] if index < len(row_official) else 0)
        oracle_documents.append(
            {
                "id": frozen.row_id(row),
                "oracle_selected_indices": selected,
                "oracle_sentence_count": len(selected),
                "oracle_characters": len("".join(document_sentences[i] for i in selected)),
                "oracle_objective": score,
            }
        )
        if document_index % 500 == 0 or document_index == len(rows):
            print(f"oracle-labeled {document_index}/{len(rows)} train documents", flush=True)
    if len(set(oracle_labels)) != 2:
        raise RuntimeError("oracle labels must contain both classes")
    audit = {
        "training_documents": len(rows),
        "training_sentences": len(oracle_labels),
        "oracle_positive_labels": sum(oracle_labels),
        "oracle_positive_rate": mean(oracle_labels),
        "weak_positive_labels": sum(weak_labels),
        "weak_positive_rate": mean(weak_labels),
        "official_positive_labels": sum(official_labels),
        "official_positive_rate": mean(official_labels),
        "oracle_vs_weak": binary_agreement(oracle_labels, weak_labels),
        "oracle_vs_official": binary_agreement(oracle_labels, official_labels),
        "oracle_sentence_count": frozen.distribution(
            [row["oracle_sentence_count"] for row in oracle_documents]
        ),
        "oracle_characters": frozen.distribution(
            [row["oracle_characters"] for row in oracle_documents]
        ),
        "oracle_objective": frozen.distribution(
            [row["oracle_objective"] for row in oracle_documents]
        ),
    }
    return training_rows, oracle_labels, audit, oracle_documents


def fit_oracle_classifier(
    training_rows: list[dict], labels: list[int], frozen_model: dict
) -> dict:
    vectorizer = frozen_model["vectorizer"]
    text_matrix = vectorizer.transform(base.decorated_text(row) for row in training_rows)
    explicit_matrix = csr_matrix([base.explicit_features(row) for row in training_rows])
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
        "training_cases": TRAIN_CASES,
        "training_sentences": len(training_rows),
        "positive_labels": sum(labels),
        "label_method": "budget_aware_greedy_oracle",
        "oracle_objective": ORACLE_OBJECTIVE,
        "max_features": frozen_model.get("max_features", 120000),
    }


def generate_dev(rows: list[dict], model: dict) -> list[dict]:
    predictions: list[dict] = []
    for document_number, row in enumerate(rows, 1):
        document_sentences = base.sentences(row)
        probabilities = base.predict_probabilities(model, document_sentences)
        selected, skipped = frozen.select_budget_greedy(
            document_sentences,
            probabilities,
            fact_weight=FACT_WEIGHT,
            redundancy_weight=REDUNDANCY_WEIGHT,
            max_sentences=MAX_SENTENCES,
            max_chars=MAX_CHARS,
        )
        predictions.append(
            frozen.prediction_row(
                row,
                selected,
                decoder="budget_greedy",
                fact_weight=FACT_WEIGHT,
                redundancy_weight=REDUNDANCY_WEIGHT,
                max_chars=MAX_CHARS,
                skipped_due_to_budget=skipped,
            )
        )
        if document_number % 100 == 0 or document_number == len(rows):
            print(f"generated {document_number}/{len(rows)} v2.2 dev cases", flush=True)
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--v21-dev-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    decoder_path = Path(frozen.__file__).resolve()
    decoder_hash = require_hash(decoder_path, EXPECTED_V21_DECODER_SHA256, "v2.1 decoder")
    model_hash = require_hash(
        args.frozen_model, EXPECTED_FROZEN_MODEL_SHA256, "frozen v2 model"
    )
    train_hash = require_hash(args.train, EXPECTED_TRAIN_SHA256, "train split")
    dev_hash = require_hash(args.dev, EXPECTED_DEV_SHA256, "dev split")
    baseline_hash = require_hash(
        args.v21_dev_predictions,
        EXPECTED_V21_DEV_PREDICTIONS_SHA256,
        "v2.1 dev predictions",
    )
    train_rows = base.read_jsonl(args.train)
    dev_rows = base.read_jsonl(args.dev)
    if len(train_rows) != TRAIN_CASES or len(dev_rows) != DEV_CASES:
        raise RuntimeError(
            f"unexpected split sizes: train={len(train_rows)}, dev={len(dev_rows)}"
        )
    with args.frozen_model.open("rb") as handle:
        frozen_model = pickle.load(handle)

    training_rows, labels, label_audit, oracle_documents = build_oracle_training(train_rows)
    model = fit_oracle_classifier(training_rows, labels, frozen_model)
    predictions = generate_dev(dev_rows, model)
    baseline_predictions = base.read_jsonl(args.v21_dev_predictions)
    if [frozen.row_id(row) for row in baseline_predictions] != [
        frozen.row_id(row) for row in predictions
    ]:
        raise RuntimeError("v2.1/v2.2 dev ID order mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    frozen.write_jsonl(args.output_dir / "train_oracle_extracts.jsonl", oracle_documents)
    frozen.write_jsonl(args.output_dir / "dev_v22_predictions.jsonl", predictions)

    metrics: dict[str, dict] = {"v21_budget": {}, "v22_oracle": {}}
    for unit in ("lawa", "character"):
        baseline_metrics, baseline_details = frozen.evaluate_predictions(
            baseline_predictions,
            dev_rows,
            rouge_unit=unit,
            max_chars=MAX_CHARS,
            max_sentences=MAX_SENTENCES,
        )
        oracle_metrics, oracle_details = frozen.evaluate_predictions(
            predictions,
            dev_rows,
            rouge_unit=unit,
            max_chars=MAX_CHARS,
            max_sentences=MAX_SENTENCES,
        )
        metrics["v21_budget"][unit] = baseline_metrics
        metrics["v22_oracle"][unit] = oracle_metrics
        frozen.write_jsonl(
            args.output_dir / f"dev_v21_{unit}_per_sample.jsonl", baseline_details
        )
        frozen.write_jsonl(
            args.output_dir / f"dev_v22_{unit}_per_sample.jsonl", oracle_details
        )
    diagnostics = {
        "v21_budget": frozen.decoder_diagnostics(
            baseline_predictions, max_chars=MAX_CHARS
        ),
        "v22_oracle": frozen.decoder_diagnostics(predictions, max_chars=MAX_CHARS),
        "selection_comparison": frozen.compare_selections(
            baseline_predictions, predictions
        )[0],
    }
    metrics["v21_budget"]["decoder_diagnostics"] = diagnostics["v21_budget"]
    metrics["v22_oracle"]["decoder_diagnostics"] = diagnostics["v22_oracle"]
    frozen.save_json(args.output_dir / "dev_v21_metrics.json", metrics["v21_budget"])
    frozen.save_json(args.output_dir / "dev_v22_metrics.json", metrics["v22_oracle"])
    frozen.save_json(args.output_dir / "decoder_diagnostics.json", diagnostics)
    frozen.save_json(args.output_dir / "training_label_audit.json", label_audit)

    config = {
        "experiment": EXPERIMENT,
        "status": "dev_complete_test_untouched",
        "selection_basis": "train_and_dev_only",
        "test_read_or_generated": False,
        "single_changed_variable": "binary_training_labels",
        "baseline_label": "sentence_rouge2_precision_gt_0.1",
        "new_label": "membership_in_budget_aware_greedy_oracle",
        "oracle_objective": ORACLE_OBJECTIVE,
        "oracle_requires_positive_marginal_gain": True,
        "oracle_tie_break": "lowest_source_index",
        "decoder": "frozen_v2.1_budget_greedy",
        "decoder_sha256": decoder_hash,
        "frozen_model_sha256": model_hash,
        "train_sha256": train_hash,
        "dev_sha256": dev_hash,
        "v21_dev_predictions_sha256": baseline_hash,
        "representation_reused_from_frozen_model": True,
        "classifier": "LogisticRegression",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": 42,
        "max_iter": 1000,
        "max_chars": MAX_CHARS,
        "max_sentences": MAX_SENTENCES,
        "fact_weight": FACT_WEIGHT,
        "redundancy_weight": REDUNDANCY_WEIGHT,
    }
    frozen.save_json(args.output_dir / "config.json", config)
    manifest = {
        **config,
        "script_sha256": sha256(script_path),
        "model_sha256": sha256(model_path),
        "train_cases": len(train_rows),
        "dev_cases": len(dev_rows),
        "training_label_audit": label_audit,
        "primary_delta_v22_minus_v21": {
            metric: metrics["v22_oracle"]["lawa"][metric]
            - metrics["v21_budget"]["lawa"][metric]
            for metric in ("rouge_1_f1", "rouge_2_f1", "rouge_l_f1")
        },
        "outputs": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(args.output_dir.iterdir())
            if path.is_file()
        },
    }
    frozen.save_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
