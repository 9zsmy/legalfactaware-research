#!/usr/bin/env python3
"""Rebuild missing dev variants and recompute manuscript statistics.

This audit reuses the frozen vectorizer, split hashes, feature code, model
settings, and decoder.  It retrains only the two variants whose per-document
files were removed from the compact archive (weak supervision and no TF-IDF),
and regenerates the post-hoc truncation control from the frozen full model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression


EXPECTED = {
    "train": "8b19e6a1aa086e55c3a1d694e86cb4cf5abfe6929ee1258a3621b291bde564a2",
    "dev": "1a056c68dcfa5f11e9574d5f1b818fa475bb5607ea7f6565e2868072a55de2c0",
    "v22_model": "2cc0c9fe4b1fdaca6c1b2e37e545fcece3e5da83533296c1c12ba1d8bb8f70e9",
    "full_dev": "fba8edc13449413e446fd185f34ffab4764a79a65c4d4e71a0a8c379d60abf08",
    "full_model": "0858b3e80f641336ead5e70c0a4343e254677c72b2bfc9427170564744648dab",
}
METRICS = ("rouge_1_f1", "rouge_2_f1", "rouge_l_f1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: {actual} != {expected}")
    return actual


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def paired_test(full: list[dict], variant: list[dict], metric: str) -> dict:
    differences = np.asarray(
        [float(a[metric]) - float(b[metric]) for a, b in zip(full, variant)],
        dtype=float,
    )
    rng = np.random.default_rng(42)
    samples = np.empty(10_000, dtype=float)
    for start in range(0, 10_000, 500):
        stop = min(start + 500, 10_000)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        samples[start:stop] = differences[indices].mean(axis=1)
    p_value = 1.0 if np.allclose(differences, 0) else float(
        wilcoxon(differences, zero_method="pratt").pvalue
    )
    return {
        "metric": metric,
        "cases": len(differences),
        "mean_delta_full_minus_variant": float(differences.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "wilcoxon_p_raw": p_value,
        "full_win_rate": float(np.mean(differences > 0)),
        "tie_rate": float(np.mean(np.isclose(differences, 0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    sys.path.insert(0, str(project / "scripts"))
    import run_legalfactaware_v2_cail_clean as base
    import run_legalfactaware_v21_budget_decoder_dev as decoder
    import run_legalfactaware_v25_structure_feature_dev as section

    train_path = project / "third_party/JUSTNLP_LSUMM_BudgetRecipe_2025/cail2020_clean_source/train.jsonl"
    dev_path = project / "third_party/JUSTNLP_LSUMM_BudgetRecipe_2025/cail2020_clean_source/dev.jsonl"
    v22_model_path = project / "experiments/legalfactaware_decoder_optimization/v2_2_oracle_supervision/results/model.pkl"
    full_dir = project / "experiments/legalfactaware_state_aware_dynamic_20260827/v25_rebuild"
    full_model_path = full_dir / "v25_b_tfidf_legal14_section_model.pkl"
    full_details_path = full_dir / "dev_v25_b_tfidf_legal14_section_lawa_per_sample.jsonl"
    no_legal14_path = full_dir / "dev_v25_a_tfidf_section_lawa_per_sample.jsonl"
    no_section6_path = full_dir / "dev_baseline_frozen_v22_lawa_per_sample.jsonl"

    hashes = {
        "train": require(train_path, EXPECTED["train"], "train"),
        "dev": require(dev_path, EXPECTED["dev"], "dev"),
        "v22_model": require(v22_model_path, EXPECTED["v22_model"], "v22 model"),
        "full_model": require(full_model_path, EXPECTED["full_model"], "full model"),
        "full_dev": require(full_details_path, EXPECTED["full_dev"], "full dev metrics"),
    }
    train = base.read_jsonl(train_path)
    dev = base.read_jsonl(dev_path)
    if (len(train), len(dev)) != (7495, 942):
        raise RuntimeError("unexpected train/dev sizes")
    with v22_model_path.open("rb") as handle:
        vectorizer = pickle.load(handle)["vectorizer"]
    with full_model_path.open("rb") as handle:
        full_model = pickle.load(handle)

    features: list[dict] = []
    weak_labels: list[int] = []
    oracle_labels: list[int] = []
    oracle_rows = read_jsonl(
        project / "experiments/legalfactaware_decoder_optimization/v2_2_oracle_supervision/results/train_oracle_extracts.jsonl"
    )
    for doc_id, (row, oracle_row) in enumerate(zip(train, oracle_rows)):
        sentences = base.sentences(row)
        sections = section.section_labels(sentences)
        selected = set(map(int, oracle_row["oracle_selected_indices"]))
        reference_counts = base.bigrams(base.tokens(base.reference(row)))
        for index, sentence_text in enumerate(sentences):
            features.append({
                "doc_id": doc_id,
                "index": index,
                "total": len(sentences),
                "sentence": sentence_text,
                "section": sections[index],
            })
            weak_labels.append(int(base.rouge2_precision(sentence_text, reference_counts) > 0.1))
            oracle_labels.append(int(index in selected))
        if (doc_id + 1) % 1000 == 0:
            print(f"prepared {doc_id + 1}/7495 training documents", flush=True)

    text_matrix = vectorizer.transform(base.decorated_text(row) for row in features)
    legal_matrix = csr_matrix([base.explicit_features(row) for row in features])
    section_matrix = csr_matrix([section.section_one_hot(row["section"]) for row in features])
    weak_matrix = hstack([text_matrix, legal_matrix, section_matrix], format="csr")
    domain_matrix = hstack([legal_matrix, section_matrix], format="csr")
    lr_args = dict(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)
    print("fitting weak-supervision and no-TF-IDF controls", flush=True)
    weak_classifier = LogisticRegression(**lr_args).fit(weak_matrix, np.asarray(weak_labels))
    no_tfidf_classifier = LogisticRegression(**lr_args).fit(domain_matrix, np.asarray(oracle_labels))

    variants: dict[str, list[dict]] = {
        "full": read_jsonl(full_details_path),
        "minus_legal14": read_jsonl(no_legal14_path),
        "minus_section6": read_jsonl(no_section6_path),
    }
    weak_predictions: list[dict] = []
    no_tfidf_predictions: list[dict] = []
    post_truncation_predictions: list[dict] = []
    for number, row in enumerate(dev, 1):
        sentences = base.sentences(row)
        doc_rows = section.document_rows(sentences)
        text = vectorizer.transform(base.decorated_text(item) for item in doc_rows)
        legal = csr_matrix([base.explicit_features(item) for item in doc_rows])
        sec = csr_matrix([section.section_one_hot(item["section"]) for item in doc_rows])
        weak_scores = weak_classifier.predict_proba(hstack([text, legal, sec], format="csr"))[:, 1].tolist()
        no_tfidf_scores = no_tfidf_classifier.predict_proba(hstack([legal, sec], format="csr"))[:, 1].tolist()
        full_scores = section.predict_scores(full_model, vectorizer, sentences)

        for target, scores in ((weak_predictions, weak_scores), (no_tfidf_predictions, no_tfidf_scores)):
            selected, skipped = decoder.select_budget_greedy(
                sentences, scores, fact_weight=0.0, redundancy_weight=0.2,
                max_sentences=8, max_chars=450,
            )
            target.append(decoder.prediction_row(
                row, selected, decoder="budget_greedy", fact_weight=0.0,
                redundancy_weight=0.2, max_chars=450, skipped_due_to_budget=skipped,
            ))

        legacy_selected = base.select_indices(
            sentences, full_scores, fact_weight=0.0, redundancy_weight=0.2, max_sentences=8
        )
        post_truncation_predictions.append(decoder.prediction_row(
            row, legacy_selected, decoder="legacy", fact_weight=0.0,
            redundancy_weight=0.2, max_chars=450,
        ))
        if number % 100 == 0 or number == len(dev):
            print(f"evaluated {number}/{len(dev)} dev documents", flush=True)

    for name, predictions in (
        ("weak_supervision", weak_predictions),
        ("minus_tfidf", no_tfidf_predictions),
        ("post_truncation", post_truncation_predictions),
    ):
        aggregate, details = decoder.evaluate_predictions(
            predictions, dev, rouge_unit="lawa", max_chars=450, max_sentences=8
        )
        variants[name] = details
        write_jsonl(args.output_dir / f"{name}_lawa_per_sample.jsonl", details)
        (args.output_dir / f"{name}_metrics.json").write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    ids = [str(row["id"]) for row in variants["full"]]
    if len(ids) != 942 or len(set(ids)) != 942:
        raise RuntimeError("full dev ID audit failed")
    for name, rows in variants.items():
        if [str(row["id"]) for row in rows] != ids:
            raise RuntimeError(f"ID/order mismatch for {name}")

    results: list[dict] = []
    for name in ("weak_supervision", "minus_tfidf", "minus_legal14", "minus_section6", "post_truncation"):
        for metric in METRICS:
            row = paired_test(variants["full"], variants[name], metric)
            row["variant"] = name
            results.append(row)
    adjusted = holm_adjust([row["wilcoxon_p_raw"] for row in results])
    for row, value in zip(results, adjusted):
        row["wilcoxon_p_holm_15_rouge_tests"] = value
        low, high = row["bootstrap_95_ci"]
        row["supported_at_0_05"] = bool(low > 0 and high > 0 and value < 0.05)

    payload = {
        "audit_date": "2026-10-10",
        "protocol": {
            "cases": 942,
            "paired_bootstrap_samples": 10_000,
            "bootstrap_seed": 42,
            "wilcoxon": "two-sided, zero_method=pratt",
            "holm_family": "5 variants x 3 primary ROUGE metrics",
            "evaluation": "document-macro Lawa-tokenized ROUGE-1/2/L F1",
        },
        "verified_hashes": hashes,
        "weak_label": "positive iff character-bigram ROUGE-2 precision(sentence, reference) > 0.1",
        "variant_means": {
            name: {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}
            for name, rows in variants.items()
        },
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "recomputed_statistics.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
