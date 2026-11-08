#!/usr/bin/env python3
"""Strict dev-only TF-IDF -> TF-only representation comparison.

The frozen 120,000-feature character 2--4 gram vocabulary, analyzer,
sublinear TF scaling, L2 normalization, Oracle targets, Legal14, Section6,
class-balanced LR, decoder and budgets are unchanged. The sole representation
change is ``use_idf=True`` -> ``use_idf=False``. No test argument exists.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import run_legalfactaware_formal_ablation_dev as formal
import run_legalfactaware_v2_cail_clean as base
import run_legalfactaware_v25_structure_feature_dev as section_impl


EXPERIMENT = "LegalFactAware-v2.5-B-tfidf-to-tf-only-dev-20260830"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--oracle-extracts", type=Path, required=True)
    parser.add_argument("--v22-model", type=Path, required=True)
    parser.add_argument("--formal-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def make_tf_only_vectorizer(original: TfidfVectorizer) -> TfidfVectorizer:
    params = original.get_params(deep=False)
    params["vocabulary"] = dict(original.vocabulary_)
    params["use_idf"] = False
    vectorizer = TfidfVectorizer(**params)
    if vectorizer.analyzer != "char" or vectorizer.ngram_range != (2, 4):
        raise RuntimeError("unexpected frozen analyzer configuration")
    if vectorizer.sublinear_tf is not True or vectorizer.norm != "l2":
        raise RuntimeError("TF scaling or normalization drifted")
    if len(vectorizer.vocabulary) != 120000:
        raise RuntimeError("frozen vocabulary size drifted")
    return vectorizer


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
    with args.v22_model.open("rb") as handle:
        original_vectorizer = pickle.load(handle)["vectorizer"]
    original_params = {
        name: getattr(original_vectorizer, name)
        for name in (
            "analyzer", "ngram_range", "min_df", "max_df", "max_features",
            "binary", "use_idf", "smooth_idf", "sublinear_tf", "norm",
            "lowercase", "dtype", "strip_accents", "token_pattern",
        )
    }
    tf_vectorizer = make_tf_only_vectorizer(original_vectorizer)

    features, labels, _ = formal.build_training(train, oracle_rows)
    print("building fixed-vocabulary TF-only + Legal14 + Section6 matrix", flush=True)
    text = tf_vectorizer.fit_transform(
        base.decorated_text(row) for row in features
    )
    if text.shape[1] != len(original_vectorizer.vocabulary_):
        raise RuntimeError("TF-only matrix vocabulary dimension drifted")
    legal = csr_matrix([base.explicit_features(row) for row in features])
    section = csr_matrix([
        section_impl.section_one_hot(row["section"]) for row in features
    ])
    matrix = hstack([text, legal, section], format="csr")
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    model, fit_timing = formal.fit_classifier(
        "oracle__tf_only_full",
        classifier,
        matrix,
        labels["oracle"],
        "full",
    )
    model["text_representation"] = "fixed-vocabulary sublinear TF, no IDF"
    with (args.output_dir / "models" / "oracle__tf_only_full.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    with (args.output_dir / "models" / "tf_only_vectorizer.pkl").open("wb") as handle:
        pickle.dump(tf_vectorizer, handle)
    del matrix, text, legal, section

    score_rows, inference_timing = formal.precompute_dev_scores(
        dev, {"oracle__tf_only_full": model}, tf_vectorizer
    )
    predictions, decode_timing = formal.generate_variant(
        dev,
        score_rows["oracle__tf_only_full"],
        {"name": "TF-IDF -> TF-only", "model": "oracle__tf_only_full"},
    )
    prediction_path = args.output_dir / "predictions" / "tf_only_full.jsonl"
    formal.write_jsonl(prediction_path, predictions)
    evaluated = formal.evaluate_variant(
        "tf_only_full", predictions, dev, formal.MAX_CHARS, args.output_dir
    )
    compact = formal.compact_row("TF-IDF -> TF-only", evaluated)
    formal.write_csv(args.output_dir / "tables" / "Table_tf_only.csv", [compact])

    full_details = base.read_jsonl(full_per_sample)
    comparison_details = evaluated.pop("lawa_details")
    significance = {
        metric: formal.paired_test(full_details, comparison_details, metric)
        for metric in (
            "rouge_1_f1", "rouge_2_f1", "rouge_l_f1",
            "reference_fact_f1", "sentence_f1",
        )
    }
    formal.save_json(args.output_dir / "metrics.json", evaluated)
    formal.save_json(args.output_dir / "significance_vs_full.json", significance)
    timings = {
        "fit": fit_timing,
        "inference": inference_timing["oracle__tf_only_full"],
        "decoder": decode_timing,
    }
    formal.save_json(args.output_dir / "timings.json", timings)

    serializable_original = {
        key: (str(value) if key == "dtype" else value)
        for key, value in original_params.items()
    }
    config = {
        "experiment": EXPERIMENT,
        "status": "dev_complete_test_untouched",
        "test_argument_exists": False,
        "single_changed_variable": {
            "full": "TfidfVectorizer(use_idf=True)",
            "comparison": "TfidfVectorizer(use_idf=False)",
        },
        "original_vectorizer": serializable_original,
        "tf_only_vectorizer": {
            **serializable_original,
            "use_idf": False,
            "vocabulary": "exact frozen 120000-feature training vocabulary",
        },
        "unchanged": {
            "training_cases": len(train),
            "dev_cases": len(dev),
            "training_supervision": "budget-consistent Oracle",
            "features": "text representation + Legal14 + Section6",
            "classifier": 'LogisticRegression(class_weight="balanced")',
            "solver": "liblinear",
            "max_iter": 1000,
            "random_state": 42,
            "decoder": "budget-aware complete-sentence with redundancy=0.2",
            "max_chars": formal.MAX_CHARS,
            "max_sentences": formal.MAX_SENTENCES,
        },
        "matrix_columns": int(fit_timing["matrix_columns"]),
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
