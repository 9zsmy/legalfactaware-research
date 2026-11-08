#!/usr/bin/env python3
"""Strict dev-only ``-TF-IDF`` ablation for frozen LegalFactAware-v2.5-B.

The only representation change is removing character 2--4 gram TF-IDF and
retaining Legal14 + Section6. Oracle targets, class-balanced LR settings,
complete-sentence decoder, redundancy weight, budgets, data split, and
evaluation are identical to the unified formal ablation. This script has no
test argument by design.
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression

import run_legalfactaware_formal_ablation_dev as formal
import run_legalfactaware_v2_cail_clean as base
import run_legalfactaware_v25_structure_feature_dev as section_impl


EXPERIMENT = "LegalFactAware-v2.5-B-minus-tfidf-dev-20260830"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--oracle-extracts", type=Path, required=True)
    parser.add_argument("--v22-model", type=Path, required=True)
    parser.add_argument("--formal-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def legal_section_matrix(sentences: list[str]):
    rows = section_impl.document_rows(sentences)
    legal = csr_matrix([base.explicit_features(row) for row in rows])
    section = csr_matrix([
        section_impl.section_one_hot(row["section"]) for row in rows
    ])
    return hstack([legal, section], format="csr")


def score_dev(dev: list[dict], model: dict):
    scores: list[list[float]] = []
    elapsed = 0.0
    sentence_total = 0
    classifier = model["classifier"]
    for number, row in enumerate(dev, 1):
        sentences = base.sentences(row)
        sentence_total += len(sentences)
        matrix = legal_section_matrix(sentences)
        started = time.perf_counter()
        values = classifier.predict_proba(matrix)[:, 1]
        elapsed += time.perf_counter() - started
        scores.append(np.asarray(values, dtype=float).tolist())
        if number % 100 == 0 or number == len(dev):
            print(f"scored -TF-IDF dev documents {number}/{len(dev)}", flush=True)
    return scores, {
        "classifier_score_seconds": elapsed,
        "documents_per_second": len(dev) / elapsed if elapsed else 0.0,
        "sentences_per_second": sentence_total / elapsed if elapsed else 0.0,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("models", "predictions", "per_sample", "tables"):
        (args.output_dir / name).mkdir(exist_ok=True)

    hashes = {
        "train_sha256": formal.require(
            args.train, formal.EXPECTED_TRAIN_SHA256, "train"
        ),
        "dev_sha256": formal.require(args.dev, formal.EXPECTED_DEV_SHA256, "dev"),
        "oracle_extracts_sha256": formal.require(
            args.oracle_extracts, formal.EXPECTED_ORACLE_SHA256, "oracle extracts"
        ),
        "v22_model_sha256": formal.require(
            args.v22_model, formal.EXPECTED_V22_MODEL_SHA256, "v2.2 model"
        ),
    }
    full_per_sample = args.formal_output / "per_sample" / "full_lawa.jsonl"
    full_metrics_path = args.formal_output / "all_metrics.json"
    if not full_per_sample.is_file() or not full_metrics_path.is_file():
        raise FileNotFoundError("formal Full evidence is incomplete")
    hashes["formal_full_per_sample_sha256"] = formal.sha256(full_per_sample)
    hashes["formal_all_metrics_sha256"] = formal.sha256(full_metrics_path)

    train = base.read_jsonl(args.train)
    dev = base.read_jsonl(args.dev)
    oracle_rows = base.read_jsonl(args.oracle_extracts)
    if (len(train), len(dev), len(oracle_rows)) != (
        formal.TRAIN_CASES,
        formal.DEV_CASES,
        formal.TRAIN_CASES,
    ):
        raise RuntimeError("unexpected split size")

    features, labels, _ = formal.build_training(train, oracle_rows)
    print("building Legal14+Section6 matrix without TF-IDF", flush=True)
    legal = csr_matrix([base.explicit_features(row) for row in features])
    section = csr_matrix([
        section_impl.section_one_hot(row["section"]) for row in features
    ])
    matrix = hstack([legal, section], format="csr")
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    model, fit_timing = formal.fit_classifier(
        "oracle__legal14_section6",
        classifier,
        matrix,
        labels["oracle"],
        "legal14_section6",
    )
    model["removed_feature"] = "character 2-4 gram TF-IDF"
    model_path = args.output_dir / "models" / "oracle__legal14_section6.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    del matrix, legal, section

    score_rows, inference_timing = score_dev(dev, model)
    predictions, decode_timing = formal.generate_variant(
        dev,
        score_rows,
        {"name": "-TF-IDF", "model": "oracle__legal14_section6"},
    )
    prediction_path = args.output_dir / "predictions" / "minus_tfidf.jsonl"
    formal.write_jsonl(prediction_path, predictions)
    evaluated = formal.evaluate_variant(
        "minus_tfidf", predictions, dev, formal.MAX_CHARS, args.output_dir
    )
    compact = formal.compact_row("-TF-IDF", evaluated)
    formal.write_csv(args.output_dir / "tables" / "Table_minus_tfidf.csv", [compact])

    full_details = base.read_jsonl(full_per_sample)
    ablated_details = evaluated.pop("lawa_details")
    significance = {
        metric: formal.paired_test(full_details, ablated_details, metric)
        for metric in (
            "rouge_1_f1",
            "rouge_2_f1",
            "rouge_l_f1",
            "reference_fact_f1",
            "sentence_f1",
        )
    }
    formal.save_json(args.output_dir / "metrics.json", evaluated)
    formal.save_json(args.output_dir / "significance_vs_full.json", significance)
    timings = {
        "fit": fit_timing,
        "inference": inference_timing,
        "decoder": decode_timing,
    }
    formal.save_json(args.output_dir / "timings.json", timings)

    config = {
        "experiment": EXPERIMENT,
        "status": "dev_complete_test_untouched",
        "test_argument_exists": False,
        "single_changed_variable": {
            "full": "frozen TF-IDF + Legal14 + Section6",
            "ablation": "Legal14 + Section6 only (TF-IDF removed)",
        },
        "unchanged": {
            "training_cases": len(train),
            "dev_cases": len(dev),
            "training_supervision": "budget-consistent Oracle",
            "classifier": 'LogisticRegression(class_weight="balanced")',
            "solver": "liblinear",
            "max_iter": 1000,
            "random_state": 42,
            "decoder": "budget-aware complete-sentence with redundancy=0.2",
            "max_chars": formal.MAX_CHARS,
            "max_sentences": formal.MAX_SENTENCES,
        },
        "feature_dimensions": {
            "Legal14": int(legal_section_matrix(["示例句子。"]).shape[1] - 6),
            "Section6": 6,
            "total": int(legal_section_matrix(["示例句子。"]).shape[1]),
        },
        "oracle_positive_labels": int(labels["oracle"].sum()),
        "oracle_positive_rate": float(labels["oracle"].mean()),
        "python": platform.python_version(),
        "sklearn": sklearn_version,
        **hashes,
    }
    formal.save_json(args.output_dir / "config.json", config)
    manifest = {
        **config,
        "script_sha256": formal.sha256(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
        "results": compact,
        "timings": timings,
        "outputs": {
            str(path.relative_to(args.output_dir)): {
                "sha256": formal.sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(args.output_dir.rglob("*"))
            if path.is_file()
        },
    }
    formal.save_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
