#!/usr/bin/env python3
"""Execute the released Gist feature and summary functions without rewriting them.

Only runtime-compatibility shims are applied for APIs removed from current
gensim/scikit-learn and for the released Windows-only path separators.  The
official feature and decoding functions themselves are imported unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import sklearn.preprocessing
from gensim.models.keyedvectors import KeyedVectors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-gist-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--word2vec", type=Path, required=True)
    parser.add_argument("--comparison-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()

    if (args.output_dir.exists() and any(args.output_dir.iterdir()) and
            not args.reuse_features):
        raise RuntimeError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = args.output_dir / "features"
    predictions = args.output_dir / "predictions"
    features.mkdir(exist_ok=True)

    # gensim 3.x exposed wv.vocab; the released code only performs membership
    # checks against it, so mapping it to gensim 4.x key_to_index is exact.
    KeyedVectors.vocab = property(lambda self: self.key_to_index)

    # Old sklearn pickles import this removed module path.
    legacy_label = types.ModuleType("sklearn.preprocessing.label")
    for name in dir(sklearn.preprocessing):
        setattr(legacy_label, name, getattr(sklearn.preprocessing, name))
    sys.modules["sklearn.preprocessing.label"] = legacy_label

    sys.path.insert(0, str(args.official_gist_dir))
    from codes import generate_features, generate_summary

    feature_args = Namespace(
        data_path=str(args.data_dir) + os.sep,
        features_path=str(features) + os.sep,
        important_words=str(args.official_gist_dir / "cue_phrases.txt"),
        pos_tags=str(args.official_gist_dir / "postags.txt"),
        w2v=False,
        word2vec_path=str(args.word2vec),
    )
    if not args.reuse_features:
        generate_features.features(feature_args)

    # generate_summary.py contains literal Windows backslashes in its directory
    # names.  Alias those names to the folders just produced, without changing
    # the released source.
    for name in ("features_quant", "features_impwords", "features_wordemb",
                 "features_pos", "features_openword"):
        alias = Path(str(features / name) + "\\")
        if alias.is_symlink():
            alias.unlink()
        alias.symlink_to((features / name).resolve(), target_is_directory=True)

    budgets = {}
    for line in args.stats_file.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            budgets[fields[0]] = int(fields[2])
    length_file = args.output_dir / "length.txt"
    length_file.write_text("".join(
        f"{name}\t{budgets[name]}\n" for name in sorted(budgets)
    ), encoding="utf-8")

    original_load = joblib.load

    class PredictorProxy:
        def __init__(self, classifier):
            self.classifier = classifier

        def predict_proba(self, matrix):
            positive = np.asarray(self.classifier._Booster.predict(matrix))
            return np.column_stack([1.0 - positive, positive])

    def compatible_load(path):
        return PredictorProxy(original_load(path))

    joblib.load = compatible_load
    generate_summary.joblib.load = compatible_load
    try:
        summary_args = Namespace(
            data_path=str(args.data_dir) + os.sep,
            features_path=str(features) + os.sep,
            summary_path=str(predictions),
            model_path=str(args.model),
            length_file=str(length_file),
        )
        generate_summary.summary(summary_args)
    finally:
        joblib.load = original_load

    names = sorted(path.name for path in predictions.glob("*.txt"))
    comparisons = []
    for name in names:
        generated = predictions / name
        previous = args.comparison_predictions / name
        comparisons.append({
            "file": name,
            "exact_text_match": previous.exists() and
                                generated.read_text(encoding="utf-8") ==
                                previous.read_text(encoding="utf-8"),
            "official_code_sha256": sha256(generated),
            "compatibility_runner_sha256": sha256(previous) if previous.exists() else None,
        })
    report = {
        "experiment": "Gist released-code equivalence audit",
        "documents": len(names),
        "exact_text_matches": sum(row["exact_text_match"] for row in comparisons),
        "all_exact": bool(comparisons) and all(row["exact_text_match"] for row in comparisons),
        "runtime_shims": [
            "gensim3 wv.vocab membership alias",
            "sklearn legacy pickle module alias",
            "LightGBM Booster predict_proba proxy",
            "Windows feature-directory separator aliases",
        ],
    }
    (args.output_dir / "comparison.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in comparisons), encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
