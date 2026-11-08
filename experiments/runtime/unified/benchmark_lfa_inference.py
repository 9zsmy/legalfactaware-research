#!/usr/bin/env python3
"""Time frozen LegalFactAware inference from JSONL text to summaries."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripts-dir", type=Path, required=True)
    parser.add_argument("--formal-script", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--vectorizer-model", type=Path, required=True)
    parser.add_argument("--classifier-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.scripts_dir))
    sys.path.insert(0, str(args.formal_script.parent))
    import run_legalfactaware_formal_ablation_dev as formal

    dev = formal.base.read_jsonl(args.dev)
    with args.vectorizer_model.open("rb") as handle:
        vectorizer = pickle.load(handle)["vectorizer"]
    with args.classifier_model.open("rb") as handle:
        model = pickle.load(handle)

    classifier = model["classifier"]
    score_rows: list[list[float]] = []
    for row in dev:
        sentences = formal.base.sentences(row)
        matrix = formal.matrix_for_sentences(model["feature_mode"], vectorizer, sentences)
        score_rows.append(classifier.predict_proba(matrix)[:, 1].tolist())

    predictions, _ = formal.generate_variant(
        dev,
        score_rows,
        {"name": "LegalFactAware", "model": "oracle__full"},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"method": "LegalFactAware", "cases": len(predictions)}))


if __name__ == "__main__":
    main()
