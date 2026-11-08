#!/usr/bin/env python3
"""Unified dev-only formal ablations for LegalFactAware-v2.5-B.

This script deliberately has no test argument.  It rebuilds every trainable
ablation from the same frozen TF-IDF vectorizer, train split, label arrays,
classifier settings and dev evaluator.  It then separates supervision,
features, decoding, redundancy, budget, structure coverage and scorer
capacity into paper-ready tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from scipy.stats import wilcoxon
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import run_legalfactaware_v21_budget_decoder_dev as frozen
import run_legalfactaware_v22_oracle_supervision_dev as oracle_impl
import run_legalfactaware_v25_structure_feature_dev as section_impl
import run_legalfactaware_v2_cail_clean as base
from evaluate_sse_cail import FACT_PATTERNS


EXPERIMENT = "LegalFactAware-v2.5-B-formal-dev-ablation-20260821"
EXPECTED_TRAIN_SHA256 = "8b19e6a1aa086e55c3a1d694e86cb4cf5abfe6929ee1258a3621b291bde564a2"
EXPECTED_DEV_SHA256 = "1a056c68dcfa5f11e9574d5f1b818fa475bb5607ea7f6565e2868072a55de2c0"
EXPECTED_ORACLE_SHA256 = "de20bf0a98cb04de5f11f2b0869701fa5c940a392f5804f0b85a09fad5f19960"
EXPECTED_V22_MODEL_SHA256 = "2cc0c9fe4b1fdaca6c1b2e37e545fcece3e5da83533296c1c12ba1d8bb8f70e9"
EXPECTED_V25_MODEL_SHA256 = "65fc714698f22da944cab62cbc4d28022181c31fea4f9c3f783ee11f8d776539"
TRAIN_CASES = 7495
DEV_CASES = 942
MAX_CHARS = 450
MAX_SENTENCES = 8
FACT_WEIGHT = 0.0
FULL_REDUNDANCY = 0.2
WEAK_THRESHOLD = 0.1
STRUCTURE_LAMBDAS = (0.01, 0.025, 0.05, 0.10)
REDUNDANCY_LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4)
BUDGETS = (200, 250, 300, 350, 400, 450)
MEANINGFUL_SECTIONS = ("claim", "defense", "fact", "reason", "decision")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"unexpected {label} SHA256: {actual}; expected {expected}")
    return actual


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def build_training(rows: list[dict], oracle_rows: list[dict]):
    features: list[dict] = []
    oracle_labels: list[int] = []
    weak_labels: list[int] = []
    official_labels: list[int] = []
    document_slices: list[tuple[int, int]] = []
    start = 0
    for doc, (row, oracle_row) in enumerate(zip(rows, oracle_rows)):
        if frozen.row_id(row) != frozen.row_id(oracle_row):
            raise RuntimeError("train/oracle ID mismatch")
        sentences = base.sentences(row)
        sections = section_impl.section_labels(sentences)
        selected = set(map(int, oracle_row["oracle_selected_indices"]))
        reference_counts = base.bigrams(base.tokens(base.reference(row)))
        official = frozen.source_labels(row)
        for index, sentence in enumerate(sentences):
            features.append({
                "doc_id": doc,
                "index": index,
                "total": len(sentences),
                "sentence": sentence,
                "section": sections[index],
            })
            oracle_labels.append(int(index in selected))
            weak_labels.append(int(base.rouge2_precision(sentence, reference_counts) > WEAK_THRESHOLD))
            official_labels.append(official[index] if index < len(official) else 0)
        document_slices.append((start, start + len(sentences)))
        start += len(sentences)
        if (doc + 1) % 500 == 0 or doc + 1 == len(rows):
            print(f"prepared unified labels {doc + 1}/{len(rows)}", flush=True)
    return features, {
        "oracle": np.asarray(oracle_labels, dtype=np.int8),
        "weak": np.asarray(weak_labels, dtype=np.int8),
        "official": np.asarray(official_labels, dtype=np.int8),
    }, document_slices


def fit_classifier(name: str, classifier, matrix, labels: np.ndarray, feature_mode: str):
    print(f"fitting {name} on {matrix.shape} with {int(labels.sum())} positives", flush=True)
    started = time.perf_counter()
    classifier.fit(matrix, labels)
    elapsed = time.perf_counter() - started
    print(f"fitted {name} in {elapsed:.2f}s", flush=True)
    return {
        "classifier": classifier,
        "feature_mode": feature_mode,
        "training_label": name.split("__", 1)[0],
    }, {
        "fit_seconds": elapsed,
        "matrix_rows": int(matrix.shape[0]),
        "matrix_columns": int(matrix.shape[1]),
        "positive_labels": int(labels.sum()),
    }


def train_models(text, legal, section, labels: dict[str, np.ndarray], n_jobs: int):
    full = hstack([text, legal, section], format="csr")
    legal_matrix = hstack([text, legal], format="csr")
    section_matrix = hstack([text, section], format="csr")
    lr = lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42
    )
    specs = {
        "oracle__tfidf": (lr(), text, labels["oracle"], "tfidf"),
        "oracle__tfidf_legal14": (lr(), legal_matrix, labels["oracle"], "tfidf_legal14"),
        "oracle__tfidf_section6": (lr(), section_matrix, labels["oracle"], "tfidf_section6"),
        "oracle__full": (lr(), full, labels["oracle"], "full"),
        "weak__full": (lr(), full, labels["weak"], "full"),
        "official__full": (lr(), full, labels["official"], "full"),
        "oracle__svm_full": (
            LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=5000, random_state=42),
            full, labels["oracle"], "full",
        ),
        "oracle__lightgbm_full": (
            LGBMClassifier(
                objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
                max_depth=-1, min_child_samples=20, subsample=1.0, colsample_bytree=1.0,
                reg_alpha=0.0, reg_lambda=0.0, class_weight="balanced", random_state=42,
                n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbosity=-1,
            ),
            full, labels["oracle"], "full",
        ),
    }
    models, timings = {}, {}
    for name, (classifier, matrix, target, mode) in specs.items():
        models[name], timings[name] = fit_classifier(name, classifier, matrix, target, mode)
    return models, timings


def matrix_for_sentences(mode: str, vectorizer, sentences: list[str]):
    rows = section_impl.document_rows(sentences)
    text = vectorizer.transform(base.decorated_text(row) for row in rows)
    if mode == "tfidf":
        return text
    legal = csr_matrix([base.explicit_features(row) for row in rows])
    if mode == "tfidf_legal14":
        return hstack([text, legal], format="csr")
    section = csr_matrix([section_impl.section_one_hot(row["section"]) for row in rows])
    if mode == "tfidf_section6":
        return hstack([text, section], format="csr")
    if mode == "full":
        return hstack([text, legal, section], format="csr")
    raise ValueError(mode)


def precompute_dev_scores(dev: list[dict], models: dict, vectorizer):
    scores = {name: [] for name in models}
    timings = {name: 0.0 for name in models}
    sentence_total = sum(len(base.sentences(row)) for row in dev)
    for number, row in enumerate(dev, 1):
        sentences = base.sentences(row)
        matrices: dict[str, object] = {}
        for name, model in models.items():
            mode = model["feature_mode"]
            if mode not in matrices:
                matrices[mode] = matrix_for_sentences(mode, vectorizer, sentences)
            started = time.perf_counter()
            classifier = model["classifier"]
            matrix = matrices[mode]
            if hasattr(classifier, "predict_proba"):
                values = classifier.predict_proba(matrix)[:, 1]
            else:
                values = classifier.decision_function(matrix)
            timings[name] += time.perf_counter() - started
            scores[name].append(np.asarray(values, dtype=float).tolist())
        if number % 100 == 0 or number == len(dev):
            print(f"scored unified dev documents {number}/{len(dev)}", flush=True)
    timing_rows = {
        name: {
            "classifier_score_seconds": elapsed,
            "documents_per_second": len(dev) / elapsed if elapsed else None,
            "sentences_per_second": sentence_total / elapsed if elapsed else None,
        }
        for name, elapsed in timings.items()
    }
    return scores, timing_rows


def select_structure_budget(
    sentences: list[str], scores: list[float], sections: list[str], section_lambda: float,
    max_chars: int = MAX_CHARS,
):
    normalized = base.minmax(scores)
    bigrams = [base.char_bigrams(sentence) for sentence in sentences]
    selected: list[int] = []
    covered: set[str] = set()
    remaining = set(range(len(sentences)))
    skipped: list[dict] = []
    while remaining and len(selected) < MAX_SENTENCES:
        best_index, best_score = None, -float("inf")
        for index in sorted(remaining):
            ordered = sorted(selected + [index])
            candidate = "".join(sentences[item] for item in ordered)
            redundancy = max((base.jaccard(bigrams[index], bigrams[item]) for item in selected), default=0.0)
            gain = float(sections[index] in MEANINGFUL_SECTIONS and sections[index] not in covered)
            score = normalized[index] - FULL_REDUNDANCY * redundancy + section_lambda * gain
            if len(candidate) > max_chars:
                skipped.append({"index": index, "length": len(sentences[index]), "score": score})
                continue
            if score > best_score:
                best_index, best_score = index, score
        if best_index is None:
            break
        selected.append(best_index)
        if sections[best_index] in MEANINGFUL_SECTIONS:
            covered.add(sections[best_index])
        remaining.remove(best_index)
    return sorted(selected), skipped


def generate_variant(dev: list[dict], score_rows: list[list[float]], spec: dict):
    outputs = []
    started = time.perf_counter()
    for number, (row, scores) in enumerate(zip(dev, score_rows), 1):
        sentences = base.sentences(row)
        budget = int(spec.get("max_chars", MAX_CHARS))
        redundancy = float(spec.get("redundancy", FULL_REDUNDANCY))
        decoder = spec.get("decoder", "budget")
        if decoder == "legacy":
            selected = base.select_indices(
                sentences, scores, fact_weight=FACT_WEIGHT,
                redundancy_weight=redundancy, max_sentences=MAX_SENTENCES,
            )
            skipped = []
            output = frozen.prediction_row(
                row, selected, decoder="legacy", fact_weight=FACT_WEIGHT,
                redundancy_weight=redundancy, max_chars=budget,
            )
        elif decoder == "structure":
            sections = section_impl.section_labels(sentences)
            selected, skipped = select_structure_budget(
                sentences, scores, sections, float(spec["section_lambda"]), budget
            )
            output = frozen.prediction_row(
                row, selected, decoder="budget_greedy", fact_weight=FACT_WEIGHT,
                redundancy_weight=redundancy, max_chars=budget, skipped_due_to_budget=skipped,
            )
            output["selected_sections"] = [sections[index] for index in selected]
            output["section_lambda"] = float(spec["section_lambda"])
        else:
            selected, skipped = frozen.select_budget_greedy(
                sentences, scores, fact_weight=FACT_WEIGHT,
                redundancy_weight=redundancy, max_sentences=MAX_SENTENCES,
                max_chars=budget,
            )
            output = frozen.prediction_row(
                row, selected, decoder="budget_greedy", fact_weight=FACT_WEIGHT,
                redundancy_weight=redundancy, max_chars=budget, skipped_due_to_budget=skipped,
            )
        output["method"] = spec["name"]
        output["training_model"] = spec["model"]
        output["evaluation_budget"] = budget
        outputs.append(output)
        if number % 200 == 0 or number == len(dev):
            print(f"generated {number}/{len(dev)} {spec['name']}", flush=True)
    elapsed = time.perf_counter() - started
    return outputs, {
        "decode_seconds": elapsed,
        "documents_per_second": len(dev) / elapsed if elapsed else None,
        "sentences_per_second": sum(len(base.sentences(row)) for row in dev) / elapsed if elapsed else None,
    }


def fact_reference_metrics(reference: str, prediction: str):
    precisions, recalls, f1s = [], [], []
    for pattern in FACT_PATTERNS.values():
        ref = set(pattern.findall(reference))
        pred = set(pattern.findall(prediction))
        if not ref and not pred:
            continue
        overlap = len(ref & pred)
        precision = overlap / len(pred) if pred else 0.0
        recall = overlap / len(ref) if ref else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    if not f1s:
        return 1.0, 1.0, 1.0
    return mean(precisions), mean(recalls), mean(f1s)


def mean_pairwise_redundancy(sentences: list[str], selected: list[int]) -> float:
    if len(selected) < 2:
        return 0.0
    sets = {index: base.char_bigrams(sentences[index]) for index in selected}
    values = [
        base.jaccard(sets[left], sets[right])
        for pos, left in enumerate(selected)
        for right in selected[pos + 1:]
    ]
    return mean(values) if values else 0.0


def covered_sections(dev_row: dict, prediction: dict) -> int:
    sections = section_impl.section_labels(base.sentences(dev_row))
    selected = set(map(int, prediction["selected_indices"]))
    return len({sections[index] for index in selected} & set(MEANINGFUL_SECTIONS))


def evaluate_variant(name: str, predictions: list[dict], dev: list[dict], budget: int, output_dir: Path):
    result: dict[str, object] = {}
    for unit in ("lawa", "character"):
        aggregate, details = frozen.evaluate_predictions(
            predictions, dev, rouge_unit=unit, max_chars=budget, max_sentences=MAX_SENTENCES
        )
        for detail, prediction, gold in zip(details, predictions, dev):
            precision, reference_recall, fact_f1 = fact_reference_metrics(
                base.reference(gold), prediction["generated_summary"]
            )
            detail["reference_fact_precision"] = precision
            detail["reference_fact_recall_macro"] = reference_recall
            detail["reference_fact_f1"] = fact_f1
            detail["pairwise_redundancy"] = mean_pairwise_redundancy(
                base.sentences(gold), list(map(int, prediction["selected_indices"]))
            )
            detail["unused_budget"] = budget - len(prediction["generated_summary"])
            detail["incomplete_sentence"] = int(bool(prediction.get("hard_truncated")))
            detail["covered_sections"] = covered_sections(gold, prediction)
        for key in (
            "reference_fact_precision", "reference_fact_recall_macro", "reference_fact_f1",
            "pairwise_redundancy", "unused_budget", "incomplete_sentence", "covered_sections",
            "characters", "selected_sentences",
        ):
            aggregate[key] = mean(float(row[key]) for row in details)
        aggregate["hard_truncation_count"] = sum(int(row["hard_truncated"]) for row in details)
        aggregate["incomplete_sentence_count"] = sum(int(row["incomplete_sentence"]) for row in details)
        result[unit] = aggregate
        if unit == "lawa":
            write_jsonl(output_dir / "per_sample" / f"{name}_lawa.jsonl", details)
            result["lawa_details"] = details
    result["decoder_diagnostics"] = frozen.decoder_diagnostics(predictions, max_chars=budget)
    return result


def binary_diagnostics(candidate: np.ndarray, oracle: np.ndarray):
    tp = int(np.sum((candidate == 1) & (oracle == 1)))
    fp = int(np.sum((candidate == 1) & (oracle == 0)))
    fn = int(np.sum((candidate == 0) & (oracle == 1)))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision_vs_oracle": precision, "recall_vs_oracle": recall, "f1_vs_oracle": f1}


def label_diagnostics(
    train: list[dict], labels: dict[str, np.ndarray], slices: list[tuple[int, int]], sample_docs: int = 1000,
):
    rng = np.random.default_rng(42)
    sample = set(map(int, rng.choice(len(train), size=min(sample_docs, len(train)), replace=False)))
    output = {}
    oracle = labels["oracle"]
    for label_name, values in labels.items():
        counts, redundancies = [], []
        standalone, marginal = [], []
        positive_marginal = 0
        marginal_total = 0
        for doc, (row, (start, stop)) in enumerate(zip(train, slices)):
            indices = np.flatnonzero(values[start:stop]).astype(int).tolist()
            counts.append(len(indices))
            sentences = base.sentences(row)
            redundancies.append(mean_pairwise_redundancy(sentences, indices))
            if doc not in sample:
                continue
            reference = base.reference(row)
            current_indices: list[int] = []
            current_score = 0.0
            for index in indices:
                standalone.append(oracle_impl.oracle_score(reference, sentences[index]))
                candidate_indices = sorted(current_indices + [index])
                candidate = "".join(sentences[item] for item in candidate_indices)
                if len(candidate) > MAX_CHARS or len(candidate_indices) > MAX_SENTENCES:
                    continue
                score = oracle_impl.oracle_score(reference, candidate)
                gain = score - current_score
                marginal.append(gain)
                marginal_total += 1
                positive_marginal += int(gain > 0)
                current_indices = candidate_indices
                current_score = score
        entry = {
            "positive_labels": int(values.sum()),
            "positive_rate": float(values.mean()),
            "average_positive_sentences_per_document": float(np.mean(counts)),
            "mean_positive_pairwise_redundancy": float(np.mean(redundancies)),
            "marginal_protocol": "fixed source-order positives, same 450-char/8-sentence feasibility; 1000 deterministic train documents",
            "marginal_sample_documents": len(sample),
            "standalone_oracle_objective": distribution(standalone),
            "budgeted_source_order_marginal_gain": distribution(marginal),
            "positive_marginal_gain_rate": positive_marginal / marginal_total if marginal_total else 0.0,
            **binary_diagnostics(values, oracle),
        }
        output[label_name] = entry
    return output


def paired_test(full_details: list[dict], ablated_details: list[dict], metric: str, seed: int = 42):
    if [row["id"] for row in full_details] != [row["id"] for row in ablated_details]:
        raise RuntimeError("paired detail ID mismatch")
    differences = np.asarray([
        float(full[metric]) - float(ablated[metric])
        for full, ablated in zip(full_details, ablated_details)
    ])
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(10000, dtype=float)
    for start in range(0, 10000, 500):
        stop = min(start + 500, 10000)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        bootstrap[start:stop] = differences[indices].mean(axis=1)
    try:
        p_value = float(wilcoxon(differences, zero_method="pratt").pvalue) if not np.allclose(differences, 0) else 1.0
    except ValueError:
        p_value = 1.0
    return {
        "metric": metric,
        "mean_delta_full_minus_ablation": float(differences.mean()),
        "bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "wilcoxon_p": p_value if math.isfinite(p_value) else 1.0,
        "full_win_rate": float(np.mean(differences > 0)),
    }


def compact_row(label: str, metrics: dict, extra: dict | None = None):
    lawa = metrics["lawa"]
    row = {
        "method": label,
        "R1": lawa["rouge_1_f1"],
        "R2": lawa["rouge_2_f1"],
        "RL": lawa["rouge_l_f1"],
        "Fact_P": lawa["reference_fact_precision"],
        "Fact_R": lawa["fact_recall"],
        "Fact_F1": lawa["reference_fact_f1"],
        "Sent_F1": lawa["sentence_f1"],
        "Avg_chars": lawa["characters"],
        "Avg_sentences": lawa["selected_sentences"],
        "Unused_budget": lawa["unused_budget"],
        "Redundancy": lawa["pairwise_redundancy"],
        "Covered_sections": lawa["covered_sections"],
        "Hard_truncations": lawa["hard_truncation_count"],
        "Budget_violations": lawa["budget_violations"],
    }
    if extra:
        row.update(extra)
    return row


def make_tables(metrics: dict, timings: dict, output_dir: Path):
    component = [
        compact_row("Full", metrics["full"]),
        compact_row("-Oracle", metrics["minus_oracle"]),
        compact_row("-Legal14", metrics["minus_legal14"]),
        compact_row("-Section6", metrics["minus_section6"]),
        compact_row("-Legal14-Section6", metrics["tfidf_only"]),
        compact_row("-Budget Decoder", metrics["post_truncation"]),
    ]
    supervision_decoder = [
        compact_row("Independent weak label", metrics["minus_oracle"], {"group": "supervision"}),
        compact_row("Official sentence label", metrics["official_full"], {"group": "supervision"}),
        compact_row("Budget-constrained Oracle", metrics["full"], {"group": "supervision"}),
        compact_row("Post-truncation", metrics["post_truncation"], {"group": "decoder"}),
        compact_row("Complete-sentence budget", metrics["complete_no_redundancy"], {"group": "decoder"}),
        compact_row("Budget-aware + redundancy", metrics["full"], {"group": "decoder"}),
    ]
    features = [
        compact_row("TF-IDF", metrics["tfidf_only"], {"Legal14": 0, "Section6": 0}),
        compact_row("TF-IDF+Legal14", metrics["minus_section6"], {"Legal14": 1, "Section6": 0}),
        compact_row("TF-IDF+Section6", metrics["minus_legal14"], {"Legal14": 0, "Section6": 1}),
        compact_row("TF-IDF+Legal14+Section6", metrics["full"], {"Legal14": 1, "Section6": 1}),
    ]
    structure = [
        compact_row("No structure", metrics["minus_section6"], {"scorer_section": 0, "coverage_reward": 0.0}),
        compact_row("Soft structure", metrics["full"], {"scorer_section": 1, "coverage_reward": 0.0}),
    ] + [
        compact_row(
            f"Soft + coverage lambda={value}", metrics[f"structure_{str(value).replace('.', 'p')}"],
            {"scorer_section": 1, "coverage_reward": value},
        )
        for value in STRUCTURE_LAMBDAS
    ]
    redundancy = [
        compact_row(
            f"lambda_r={value}",
            metrics["full" if value == FULL_REDUNDANCY else f"redundancy_{str(value).replace('.', 'p')}"] ,
            {"lambda_r": value},
        )
        for value in REDUNDANCY_LAMBDAS
    ]
    budget = [
        compact_row(
            f"B={value}", metrics["full" if value == MAX_CHARS else f"budget_{value}"], {"budget": value}
        )
        for value in BUDGETS
    ]
    scorer = []
    for label, key, model_key in (
        ("Logistic Regression", "full", "oracle__full"),
        ("Linear SVM", "svm_full", "oracle__svm_full"),
        ("LightGBM", "lightgbm_full", "oracle__lightgbm_full"),
    ):
        row = compact_row(label, metrics[key])
        row.update(timings[model_key])
        scorer.append(row)
    write_csv(output_dir / "tables" / "Table_A_component_ablation.csv", component)
    write_csv(output_dir / "tables" / "Table_B_supervision_decoder.csv", supervision_decoder)
    write_csv(output_dir / "tables" / "Table_C_feature_2x2.csv", features)
    write_csv(output_dir / "tables" / "Table_D_structure_negative.csv", structure)
    write_csv(output_dir / "tables" / "Appendix_redundancy_sweep.csv", redundancy)
    write_csv(output_dir / "tables" / "Figure_budget_curve.csv", budget)
    write_csv(output_dir / "tables" / "Appendix_scorer_efficiency.csv", scorer)
    return {
        "component": component,
        "supervision_decoder": supervision_decoder,
        "features": features,
        "structure": structure,
        "redundancy": redundancy,
        "budget": budget,
        "scorer": scorer,
    }


def render_plots(tables: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    budget = tables["budget"]
    x = [row["budget"] for row in budget]
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, label in (("R1", "ROUGE-1"), ("R2", "ROUGE-2"), ("RL", "ROUGE-L")):
        ax.plot(x, [100 * row[metric] for row in budget], marker="o", label=label)
    ax.set_xlabel("Summary budget (characters)")
    ax.set_ylabel("Lawa ROUGE F1 (%)")
    ax.set_title("LegalFactAware quality vs summary budget (dev)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "quality_vs_budget.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [100 * row["Fact_F1"] for row in budget], marker="o", label="Fact-F1")
    ax.plot(x, [100 * row["Fact_R"] for row in budget], marker="s", label="Fact recall")
    ax.set_xlabel("Summary budget (characters)")
    ax.set_ylabel("Fact metric (%)")
    ax.set_title("LegalFactAware fact coverage vs summary budget (dev)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "fact_vs_budget.png", dpi=180)
    plt.close(fig)


def build_report(tables: dict, label_audit: dict, significance: dict) -> str:
    full = tables["component"][0]
    lines = [
        "# LegalFactAware-v2.5-B 正式消融（统一 dev-only 协议）",
        "",
        "## 协议",
        "",
        "- 所有可训练变体在同一 CAIL2020-clean train=7,495 上重新训练。",
        "- 所有选择与分析只使用 dev=942；脚本没有 test 参数。",
        "- 相同 TF-IDF vectorizer、LR 参数、450 字/8 句上限和评测器。",
        "- 主结果采用 Lawa ROUGE，同时保留字符级逐样本结果。",
        "",
        "## Full dev 结果",
        "",
        f"Full: R1/R2/RL={full['R1']:.6f}/{full['R2']:.6f}/{full['RL']:.6f}, "
        f"Fact-F1={full['Fact_F1']:.6f}, Fact-R={full['Fact_R']:.6f}, Sent-F1={full['Sent_F1']:.6f}, "
        f"平均{full['Avg_chars']:.2f}字，硬截断{full['Hard_truncations']}。",
        "",
        "## 标签诊断",
        "",
        "| 监督 | 正例率 | 对Oracle P/R/F1 | 每篇正例 | 正例冗余 | 平均预算边际增益 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("weak", "Weak"), ("official", "Official"), ("oracle", "Oracle")):
        row = label_audit[key]
        lines.append(
            f"| {label} | {row['positive_rate']:.4f} | "
            f"{row['precision_vs_oracle']:.4f}/{row['recall_vs_oracle']:.4f}/{row['f1_vs_oracle']:.4f} | "
            f"{row['average_positive_sentences_per_document']:.2f} | "
            f"{row['mean_positive_pairwise_redundancy']:.4f} | "
            f"{row['budgeted_source_order_marginal_gain']['mean']:.6f} |"
        )
    lines.extend([
        "",
        "## 统计文件",
        "",
        "Full 对五个核心拆除项的 10,000 次 paired bootstrap、Wilcoxon 和胜率见 `significance_core.json`。",
        "",
        "## 使用边界",
        "",
        "- 本报告是开发集机制分析，不替代已经冻结的一次性 test 主结果。",
        "- 冗余、结构覆盖和长度预算曲线仅用于解释与参数稳健性，不再产生新版本。",
        "- 若某项 dev 消融优于 Full，也不能据此回头修改已冻结 test 结果。",
        "",
    ])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--oracle-extracts", type=Path, required=True)
    parser.add_argument("--v22-model", type=Path, required=True)
    parser.add_argument("--frozen-v25-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)
    (args.output_dir / "predictions").mkdir(exist_ok=True)
    (args.output_dir / "per_sample").mkdir(exist_ok=True)
    (args.output_dir / "tables").mkdir(exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    hashes = {
        "train_sha256": require(args.train, EXPECTED_TRAIN_SHA256, "train"),
        "dev_sha256": require(args.dev, EXPECTED_DEV_SHA256, "dev"),
        "oracle_extracts_sha256": require(args.oracle_extracts, EXPECTED_ORACLE_SHA256, "oracle extracts"),
        "v22_model_sha256": require(args.v22_model, EXPECTED_V22_MODEL_SHA256, "v2.2 model"),
        "frozen_v25_model_sha256": require(args.frozen_v25_model, EXPECTED_V25_MODEL_SHA256, "frozen v2.5-B model"),
    }
    train = base.read_jsonl(args.train)
    dev = base.read_jsonl(args.dev)
    oracle_rows = base.read_jsonl(args.oracle_extracts)
    if (len(train), len(dev), len(oracle_rows)) != (TRAIN_CASES, DEV_CASES, TRAIN_CASES):
        raise RuntimeError("unexpected split size")
    with args.v22_model.open("rb") as handle:
        vectorizer = pickle.load(handle)["vectorizer"]

    features, labels, document_slices = build_training(train, oracle_rows)
    print("building shared frozen-representation matrices", flush=True)
    text = vectorizer.transform(base.decorated_text(row) for row in features)
    legal = csr_matrix([base.explicit_features(row) for row in features])
    section = csr_matrix([section_impl.section_one_hot(row["section"]) for row in features])
    models, fit_timings = train_models(text, legal, section, labels, args.n_jobs)
    for name, model in models.items():
        with (args.output_dir / "models" / f"{name}.pkl").open("wb") as handle:
            pickle.dump(model, handle)
    del text, legal, section

    label_audit = label_diagnostics(train, labels, document_slices)
    save_json(args.output_dir / "label_diagnostics.json", label_audit)
    label_rows = []
    for name, values in label_audit.items():
        label_rows.append({
            "supervision": name,
            "positive_rate": values["positive_rate"],
            "precision_vs_oracle": values["precision_vs_oracle"],
            "recall_vs_oracle": values["recall_vs_oracle"],
            "f1_vs_oracle": values["f1_vs_oracle"],
            "avg_positive_sentences": values["average_positive_sentences_per_document"],
            "positive_pairwise_redundancy": values["mean_positive_pairwise_redundancy"],
            "standalone_objective": values["standalone_oracle_objective"]["mean"],
            "budgeted_marginal_gain": values["budgeted_source_order_marginal_gain"]["mean"],
            "positive_marginal_rate": values["positive_marginal_gain_rate"],
        })
    write_csv(args.output_dir / "tables" / "Label_diagnostics.csv", label_rows)

    score_rows, inference_timings = precompute_dev_scores(dev, models, vectorizer)
    specs = {
        "full": {"name": "Full", "model": "oracle__full"},
        "minus_oracle": {"name": "-Oracle", "model": "weak__full"},
        "official_full": {"name": "Official supervision", "model": "official__full"},
        "minus_legal14": {"name": "-Legal14", "model": "oracle__tfidf_section6"},
        "minus_section6": {"name": "-Section6", "model": "oracle__tfidf_legal14"},
        "tfidf_only": {"name": "-Legal14-Section6", "model": "oracle__tfidf"},
        "post_truncation": {"name": "-Budget Decoder", "model": "oracle__full", "decoder": "legacy"},
        "complete_no_redundancy": {"name": "Complete budget without redundancy", "model": "oracle__full", "redundancy": 0.0},
        "svm_full": {"name": "Linear SVM", "model": "oracle__svm_full"},
        "lightgbm_full": {"name": "LightGBM", "model": "oracle__lightgbm_full"},
    }
    for value in REDUNDANCY_LAMBDAS:
        if value in (0.0, FULL_REDUNDANCY):
            continue
        key = f"redundancy_{str(value).replace('.', 'p')}"
        specs[key] = {"name": f"Redundancy lambda={value}", "model": "oracle__full", "redundancy": value}
    specs["redundancy_0p0"] = specs["complete_no_redundancy"]
    for value in STRUCTURE_LAMBDAS:
        key = f"structure_{str(value).replace('.', 'p')}"
        specs[key] = {
            "name": f"Structure coverage lambda={value}", "model": "oracle__full",
            "decoder": "structure", "section_lambda": value,
        }
    for budget in BUDGETS:
        if budget == MAX_CHARS:
            continue
        specs[f"budget_{budget}"] = {"name": f"Budget={budget}", "model": "oracle__full", "max_chars": budget}

    predictions, decode_timings, metrics = {}, {}, {}
    for key, spec in specs.items():
        # redundancy_0p0 is an alias, so do not regenerate it.
        if key == "redundancy_0p0":
            continue
        rows, timing = generate_variant(dev, score_rows[spec["model"]], spec)
        predictions[key] = rows
        decode_timings[key] = timing
        write_jsonl(args.output_dir / "predictions" / f"{key}.jsonl", rows)
        budget = int(spec.get("max_chars", MAX_CHARS))
        metrics[key] = evaluate_variant(key, rows, dev, budget, args.output_dir)
        print(
            f"evaluated {key}: "
            f"{metrics[key]['lawa']['rouge_1_f1']:.6f}/"
            f"{metrics[key]['lawa']['rouge_2_f1']:.6f}/"
            f"{metrics[key]['lawa']['rouge_l_f1']:.6f}",
            flush=True,
        )
    predictions["redundancy_0p0"] = predictions["complete_no_redundancy"]
    metrics["redundancy_0p0"] = metrics["complete_no_redundancy"]
    decode_timings["redundancy_0p0"] = decode_timings["complete_no_redundancy"]

    combined_timings = {}
    for model_key in models:
        relevant = {
            "oracle__full": "full", "oracle__svm_full": "svm_full",
            "oracle__lightgbm_full": "lightgbm_full",
        }.get(model_key)
        combined_timings[model_key] = {
            **fit_timings[model_key],
            **inference_timings[model_key],
            **(decode_timings[relevant] if relevant else {}),
        }
    tables = make_tables(metrics, combined_timings, args.output_dir)
    render_plots(tables, args.output_dir)

    significance = {}
    full_details = metrics["full"]["lawa_details"]
    for key in ("minus_oracle", "minus_legal14", "minus_section6", "tfidf_only", "post_truncation"):
        significance[key] = [
            paired_test(full_details, metrics[key]["lawa_details"], metric)
            for metric in ("rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "reference_fact_f1", "sentence_f1")
        ]
    save_json(args.output_dir / "significance_core.json", significance)

    # Remove in-memory per-sample rows before the compact aggregate JSON.
    compact_metrics = {
        name: {key: value for key, value in result.items() if key != "lawa_details"}
        for name, result in metrics.items()
    }
    save_json(args.output_dir / "all_metrics.json", compact_metrics)
    save_json(args.output_dir / "fit_and_inference_timings.json", {
        "fit": fit_timings, "classifier_inference": inference_timings, "decoder": decode_timings
    })
    report = build_report(tables, label_audit, significance)
    (args.output_dir / "FORMAL_ABLATION_REPORT.md").write_text(report, encoding="utf-8")
    config = {
        "experiment": EXPERIMENT,
        "status": "dev_complete_test_untouched",
        "test_argument_exists": False,
        "train_cases": len(train),
        "dev_cases": len(dev),
        "training_sentences": len(features),
        "max_chars": MAX_CHARS,
        "max_sentences": MAX_SENTENCES,
        "fact_weight": FACT_WEIGHT,
        "full_redundancy_weight": FULL_REDUNDANCY,
        "redundancy_sweep": list(REDUNDANCY_LAMBDAS),
        "budget_curve": list(BUDGETS),
        "structure_lambdas": list(STRUCTURE_LAMBDAS),
        "random_state": 42,
        "python": platform.python_version(),
        "sklearn": sklearn_version,
        **hashes,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    save_json(args.output_dir / "config.json", config)
    manifest = {
        **config,
        "outputs": {
            str(path.relative_to(args.output_dir)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(args.output_dir.rglob("*")) if path.is_file()
        },
    }
    save_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({
        "status": "complete", "output_dir": str(args.output_dir),
        "full_lawa": compact_metrics["full"]["lawa"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
