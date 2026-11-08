#!/usr/bin/env python3
"""Recompute the v2.4 TF-IDF-only control in the audited local environment."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression


PROJECT = Path("/Users/admin/Documents/SummarizaionGeneration")
OUT = Path(
    "/Users/admin/Documents/ChatGPT/1/论文写作/英文论文/"
    "statistics_recomputed_20261010/tfidf_only_reconciliation_20261011.json"
)
EXPECTED = {
    "train": "8b19e6a1aa086e55c3a1d694e86cb4cf5abfe6929ee1258a3621b291bde564a2",
    "dev": "1a056c68dcfa5f11e9574d5f1b818fa475bb5607ea7f6565e2868072a55de2c0",
    "oracle": "de20bf0a98cb04de5f11f2b0869701fa5c940a392f5804f0b85a09fad5f19960",
    "v22_model": "2cc0c9fe4b1fdaca6c1b2e37e545fcece3e5da83533296c1c12ba1d8bb8f70e9",
    "decoder": "e8084e0b0ca6e1803a9bedd1c51bb36143c09136a22d70e0da2e3c19b74c93dd",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")


def main() -> None:
    sys.path.insert(0, str(PROJECT / "scripts"))
    import run_legalfactaware_v21_budget_decoder_dev as frozen
    import run_legalfactaware_v2_cail_clean as base

    data = PROJECT / "third_party/JUSTNLP_LSUMM_BudgetRecipe_2025/cail2020_clean_source"
    train_path = data / "train.jsonl"
    dev_path = data / "dev.jsonl"
    results = PROJECT / "experiments/legalfactaware_decoder_optimization/v2_2_oracle_supervision/results"
    oracle_path = results / "train_oracle_extracts.jsonl"
    model_path = results / "model.pkl"
    decoder_path = Path(frozen.__file__).resolve()
    for path, key in [
        (train_path, "train"),
        (dev_path, "dev"),
        (oracle_path, "oracle"),
        (model_path, "v22_model"),
        (decoder_path, "decoder"),
    ]:
        require(path, EXPECTED[key])

    train = base.read_jsonl(train_path)
    dev = base.read_jsonl(dev_path)
    oracle_rows = base.read_jsonl(oracle_path)
    with model_path.open("rb") as handle:
        vectorizer = pickle.load(handle)["vectorizer"]

    training_rows = []
    labels = []
    for doc_index, (row, oracle) in enumerate(zip(train, oracle_rows)):
        if frozen.row_id(row) != frozen.row_id(oracle):
            raise RuntimeError("train/oracle ID mismatch")
        sentences = base.sentences(row)
        selected = set(map(int, oracle["oracle_selected_indices"]))
        for index, sentence in enumerate(sentences):
            training_rows.append(
                {
                    "doc_id": doc_index,
                    "index": index,
                    "total": len(sentences),
                    "sentence": sentence,
                }
            )
            labels.append(int(index in selected))
        if (doc_index + 1) % 1000 == 0 or doc_index + 1 == len(train):
            print(f"prepared {doc_index + 1}/{len(train)} documents", flush=True)

    text = vectorizer.transform(base.decorated_text(row) for row in training_rows)
    classifier = LogisticRegression(
        C=1.0,
        penalty="l2",
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        tol=1e-4,
        fit_intercept=True,
        random_state=42,
    )
    classifier.fit(text, labels)

    predictions = []
    for number, row in enumerate(dev, 1):
        sentences = base.sentences(row)
        feature_rows = base.inference_rows(sentences)
        matrix = vectorizer.transform(base.decorated_text(item) for item in feature_rows)
        scores = classifier.predict_proba(matrix)[:, 1].tolist()
        selected, skipped = frozen.select_budget_greedy(
            sentences,
            scores,
            fact_weight=0.0,
            redundancy_weight=0.2,
            max_sentences=8,
            max_chars=450,
        )
        predictions.append(
            frozen.prediction_row(
                row,
                selected,
                decoder="budget_greedy",
                fact_weight=0.0,
                redundancy_weight=0.2,
                max_chars=450,
                skipped_due_to_budget=skipped,
            )
        )
        if number % 100 == 0 or number == len(dev):
            print(f"generated {number}/{len(dev)} documents", flush=True)

    aggregate, _ = frozen.evaluate_predictions(
        predictions, dev, rouge_unit="lawa", max_chars=450, max_sentences=8
    )
    import sklearn
    import scipy
    import numpy
    import lawa

    payload = {
        "audit_date": "2026-10-11",
        "purpose": "reconcile historical v2.4 TF-only means with the audited formal batch",
        "document_count": len(dev),
        "training_sentences": len(labels),
        "positive_labels": int(sum(labels)),
        "hashes": EXPECTED,
        "runtime": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "numpy": numpy.__version__,
            "lawa": getattr(lawa, "__version__", "unknown"),
        },
        "configuration": {
            "features": "frozen position-augmented character 2-4 gram TF-IDF only",
            "supervision": "frozen budget-consistent Oracle membership",
            "classifier": "LogisticRegression(C=1.0, penalty=l2, class_weight=balanced, solver=liblinear, tol=1e-4, max_iter=1000, fit_intercept=True, random_state=42)",
            "decoder": "frozen complete-sentence budget greedy",
            "max_chars": 450,
            "max_sentences": 8,
            "redundancy_weight": 0.2,
            "fact_weight": 0.0,
        },
        "historical_reported_percent": {"R1": 60.31, "R2": 41.26, "RL": 50.02},
        "formal_batch_percent": {"R1": 59.90, "R2": 40.66, "RL": 49.27},
        "recomputed_metrics": aggregate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
