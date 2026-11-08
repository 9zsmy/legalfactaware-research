#!/usr/bin/env python3
"""Time the selected LegalFactAware pipeline from frozen CAIL JSONL to dev summaries."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripts-dir", type=Path, required=True)
    parser.add_argument("--formal-script", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--vectorizer-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.scripts_dir))
    sys.path.insert(0, str(args.formal_script.parent))
    import run_legalfactaware_v22_oracle_supervision_dev as oracle_impl
    import run_legalfactaware_v25_structure_feature_dev as section_impl
    import run_legalfactaware_v2_cail_clean as base
    import run_legalfactaware_formal_ablation_dev as formal

    total_started = time.perf_counter()
    train_rows = base.read_jsonl(args.train)
    dev_rows = base.read_jsonl(args.dev)
    if (len(train_rows), len(dev_rows)) != (7495, 942):
        raise RuntimeError("unexpected CAIL split sizes")

    started = time.perf_counter()
    feature_rows, oracle_labels, oracle_audit, oracle_documents = (
        oracle_impl.build_oracle_training(train_rows)
    )
    oracle_seconds = time.perf_counter() - started
    write_jsonl(args.output_dir / "train_oracle_extracts.jsonl", oracle_documents)

    offset = 0
    for row in train_rows:
        sentences = base.sentences(row)
        sections = section_impl.section_labels(sentences)
        for index, section in enumerate(sections):
            feature_rows[offset + index]["section"] = section
        offset += len(sentences)

    started = time.perf_counter()
    with args.vectorizer_template.open("rb") as handle:
        template = pickle.load(handle)["vectorizer"]
    params = template.get_params(deep=False)
    params["vocabulary"] = None
    vectorizer = TfidfVectorizer(**params)
    text = vectorizer.fit_transform(base.decorated_text(row) for row in feature_rows)
    legal = csr_matrix([base.explicit_features(row) for row in feature_rows])
    section = csr_matrix(
        [section_impl.section_one_hot(row["section"]) for row in feature_rows]
    )
    matrix = hstack([text, legal, section], format="csr")
    representation_seconds = time.perf_counter() - started

    started = time.perf_counter()
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    classifier.fit(matrix, np.asarray(oracle_labels, dtype=np.int8))
    training_seconds = time.perf_counter() - started
    model = {"classifier": classifier, "feature_mode": "full"}
    with (args.output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({**model, "vectorizer": vectorizer}, handle)

    started = time.perf_counter()
    score_rows: list[list[float]] = []
    for row in dev_rows:
        sentences = base.sentences(row)
        dev_matrix = formal.matrix_for_sentences("full", vectorizer, sentences)
        score_rows.append(classifier.predict_proba(dev_matrix)[:, 1].tolist())
    predictions, _ = formal.generate_variant(
        dev_rows,
        score_rows,
        {"name": "LegalFactAware", "model": "oracle__full"},
    )
    inference_seconds = time.perf_counter() - started
    write_jsonl(args.output_dir / "dev_predictions.jsonl", predictions)

    timings = {
        "method": "LegalFactAware",
        "train_documents": len(train_rows),
        "dev_documents": len(dev_rows),
        "training_sentences": len(feature_rows),
        "oracle_seconds": oracle_seconds,
        "representation_seconds": representation_seconds,
        "model_training_seconds": training_seconds,
        "dev_inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "oracle_audit": oracle_audit,
    }
    (args.output_dir / "stage_timings.json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(timings, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
