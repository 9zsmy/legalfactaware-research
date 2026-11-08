#!/usr/bin/env python3
"""Dev-only LegalFactAware v2.5 sentence-section feature ablation.

The experiment freezes v2.2 Oracle supervision, its TF-IDF vectorizer, the
v2.1 budget-aware decoder, and LR hyperparameters.  Only six deterministic
sentence-level section one-hot features are added.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression

import run_legalfactaware_v21_budget_decoder_dev as frozen
import run_legalfactaware_v2_cail_clean as base


EXPERIMENT = "LegalFactAware-v2.5-Structure-Feature-Only"
EXPECTED_TRAIN_SHA256 = "8b19e6a1aa086e55c3a1d694e86cb4cf5abfe6929ee1258a3621b291bde564a2"
EXPECTED_DEV_SHA256 = "1a056c68dcfa5f11e9574d5f1b818fa475bb5607ea7f6565e2868072a55de2c0"
EXPECTED_ORACLE_EXTRACTS_SHA256 = "de20bf0a98cb04de5f11f2b0869701fa5c940a392f5804f0b85a09fad5f19960"
EXPECTED_V22_MODEL_SHA256 = "2cc0c9fe4b1fdaca6c1b2e37e545fcece3e5da83533296c1c12ba1d8bb8f70e9"
EXPECTED_V22_PREDICTIONS_SHA256 = "de9ddcead387b71ae862fb8effb9df0b389ee386f84d9efa8d9fc80107685925"
EXPECTED_DECODER_SHA256 = "e8084e0b0ca6e1803a9bedd1c51bb36143c09136a22d70e0da2e3c19b74c93dd"

MAX_CHARS = 450
MAX_SENTENCES = 8
FACT_WEIGHT = 0.0
REDUNDANCY_WEIGHT = 0.2
SECTIONS = ("claim", "defense", "fact", "reason", "decision", "other")
SECTION_INDEX = {name: index for index, name in enumerate(SECTIONS)}

CLAIM_MARKERS = ("原告诉称", "原告起诉称", "原告请求", "诉讼请求", "请求判令", "公诉机关指控")
DEFENSE_MARKERS = ("被告辩称", "被告答辩称", "答辩称", "辩护人提出", "被告人辩解")
FACT_MARKERS = ("经审理查明", "本院查明", "经审查查明", "查明事实", "认定事实")
REASON_MARKERS = ("本院认为", "本院经审理认为", "裁判理由", "综上所述", "综上，本院认为")
DECISION_MARKERS = ("判决如下", "裁定如下", "决定如下", "判处如下")
DECISION_ACTIONS = ("驳回", "支付", "返还", "赔偿", "判处", "维持原判", "撤销", "承担", "给付")


def require(path: Path, expected: str, label: str) -> str:
    actual = frozen.sha256(path)
    if actual != expected:
        raise RuntimeError(f"unexpected {label} SHA256: {actual}; expected {expected}")
    return actual


def contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def section_labels(sentences: list[str]) -> list[str]:
    """Assign one deterministic role per sentence without using references."""
    labels: list[str] = []
    active = "other"
    total = max(len(sentences), 1)
    for index, sentence in enumerate(sentences):
        position = index / total
        # A section-opening marker changes the state for this and following
        # sentences. Priority handles rare sentences containing two markers.
        if contains_any(sentence, DECISION_MARKERS):
            active = "decision"
        elif contains_any(sentence, REASON_MARKERS):
            active = "reason"
        elif contains_any(sentence, FACT_MARKERS):
            active = "fact"
        elif contains_any(sentence, DEFENSE_MARKERS):
            active = "defense"
        elif contains_any(sentence, CLAIM_MARKERS):
            active = "claim"
        elif position >= 0.65 and contains_any(sentence, DECISION_ACTIONS):
            # Action words alone are ambiguous in claims; only treat them as
            # a decision cue in the latter 35% of a document.
            active = "decision"
        labels.append(active)
    return labels


def section_one_hot(label: str) -> list[float]:
    values = [0.0] * len(SECTIONS)
    values[SECTION_INDEX[label]] = 1.0
    return values


def document_rows(document_sentences: list[str], doc_id: int = 0) -> list[dict]:
    labels = section_labels(document_sentences)
    return [
        {
            "doc_id": doc_id,
            "index": index,
            "total": len(document_sentences),
            "sentence": sentence,
            "section": labels[index],
        }
        for index, sentence in enumerate(document_sentences)
    ]


def build_training(rows: list[dict], oracle_rows: list[dict]) -> tuple[list[dict], list[int], Counter]:
    features: list[dict] = []
    labels: list[int] = []
    counts: Counter = Counter()
    for doc, (row, oracle) in enumerate(zip(rows, oracle_rows)):
        if frozen.row_id(row) != frozen.row_id(oracle):
            raise RuntimeError("train/oracle ID mismatch")
        selected = set(map(int, oracle["oracle_selected_indices"]))
        current = document_rows(base.sentences(row), doc)
        features.extend(current)
        labels.extend(int(item["index"] in selected) for item in current)
        counts.update(item["section"] for item in current)
        if (doc + 1) % 1000 == 0 or doc + 1 == len(rows):
            print(f"loaded section/oracle rows {doc + 1}/{len(rows)}", flush=True)
    return features, labels, counts


def fit_lr(matrix, labels: list[int]):
    classifier = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42
    )
    started = time.time()
    classifier.fit(matrix, labels)
    return classifier, time.time() - started


def predict_scores(model: dict, vectorizer, sentences: list[str]) -> list[float]:
    rows = document_rows(sentences)
    if not rows:
        return []
    text = vectorizer.transform(base.decorated_text(row) for row in rows)
    section = csr_matrix([section_one_hot(row["section"]) for row in rows])
    if model["feature_mode"] == "tfidf_section":
        matrix = hstack([text, section], format="csr")
    elif model["feature_mode"] == "tfidf_legal14_section":
        legal = csr_matrix([base.explicit_features(row) for row in rows])
        matrix = hstack([text, legal, section], format="csr")
    else:
        raise ValueError(model["feature_mode"])
    return model["classifier"].predict_proba(matrix)[:, 1].tolist()


def generate(rows: list[dict], model: dict, vectorizer, name: str) -> list[dict]:
    outputs = []
    for number, row in enumerate(rows, 1):
        sentences = base.sentences(row)
        scores = predict_scores(model, vectorizer, sentences)
        selected, skipped = frozen.select_budget_greedy(
            sentences, scores, fact_weight=FACT_WEIGHT,
            redundancy_weight=REDUNDANCY_WEIGHT, max_sentences=MAX_SENTENCES,
            max_chars=MAX_CHARS,
        )
        output = frozen.prediction_row(
            row, selected, decoder="budget_greedy", fact_weight=FACT_WEIGHT,
            redundancy_weight=REDUNDANCY_WEIGHT, max_chars=MAX_CHARS,
            skipped_due_to_budget=skipped,
        )
        output["scorer"] = name
        output["feature_mode"] = model["feature_mode"]
        output["selected_sections"] = [section_labels(sentences)[i] for i in selected]
        outputs.append(output)
        if number % 100 == 0 or number == len(rows):
            print(f"generated {number}/{len(rows)} {name} dev cases", flush=True)
    return outputs


def paired_test(left: list[dict], right: list[dict], metric: str, seed: int = 42) -> dict:
    if [row["id"] for row in left] != [row["id"] for row in right]:
        raise RuntimeError("paired detail ID mismatch")
    differences = np.asarray(
        [float(a[metric]) - float(b[metric]) for a, b in zip(left, right)], dtype=float
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(10000, dtype=float)
    for start in range(0, 10000, 500):
        stop = min(start + 500, 10000)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        bootstrap[start:stop] = differences[indices].mean(axis=1)
    if np.allclose(differences, 0.0):
        p_value = 1.0
    else:
        try:
            p_value = float(wilcoxon(differences, zero_method="pratt").pvalue)
        except ValueError:
            p_value = 1.0
    return {
        "metric": metric,
        "cases": len(differences),
        "mean_delta_candidate_minus_baseline": float(differences.mean()),
        "bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "wilcoxon_p": p_value if math.isfinite(p_value) else 1.0,
        "bootstrap_samples": 10000,
        "seed": seed,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--oracle-extracts", type=Path, required=True)
    parser.add_argument("--v22-model", type=Path, required=True)
    parser.add_argument("--v22-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).resolve()
    decoder_hash = require(Path(frozen.__file__).resolve(), EXPECTED_DECODER_SHA256, "decoder")
    train_hash = require(args.train, EXPECTED_TRAIN_SHA256, "train")
    dev_hash = require(args.dev, EXPECTED_DEV_SHA256, "dev")
    oracle_hash = require(args.oracle_extracts, EXPECTED_ORACLE_EXTRACTS_SHA256, "oracle extracts")
    model_hash = require(args.v22_model, EXPECTED_V22_MODEL_SHA256, "v2.2 model")
    predictions_hash = require(args.v22_predictions, EXPECTED_V22_PREDICTIONS_SHA256, "v2.2 predictions")
    train = base.read_jsonl(args.train)
    dev = base.read_jsonl(args.dev)
    oracle_rows = base.read_jsonl(args.oracle_extracts)
    if len(train) != 7495 or len(dev) != 942 or len(oracle_rows) != 7495:
        raise RuntimeError("unexpected split size")
    with args.v22_model.open("rb") as handle:
        v22_model = pickle.load(handle)
    vectorizer = v22_model["vectorizer"]
    training_rows, labels, train_section_counts = build_training(train, oracle_rows)
    text = vectorizer.transform(base.decorated_text(row) for row in training_rows)
    legal = csr_matrix([base.explicit_features(row) for row in training_rows])
    section = csr_matrix([section_one_hot(row["section"]) for row in training_rows])
    specifications = {
        "v25_a_tfidf_section": (hstack([text, section], format="csr"), "tfidf_section"),
        "v25_b_tfidf_legal14_section": (
            hstack([text, legal, section], format="csr"), "tfidf_legal14_section"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    timings = {}
    for name, (matrix, mode) in specifications.items():
        print(f"fitting {name} on {matrix.shape}", flush=True)
        classifier, elapsed = fit_lr(matrix, labels)
        models[name] = {"classifier": classifier, "feature_mode": mode}
        timings[name] = {"fit_seconds": elapsed, "rows": matrix.shape[0], "columns": matrix.shape[1]}
        with (args.output_dir / f"{name}_model.pkl").open("wb") as handle:
            pickle.dump(models[name], handle)
        print(f"fitted {name} in {elapsed:.1f}s", flush=True)

    baseline = base.read_jsonl(args.v22_predictions)
    methods = {"baseline_frozen_v22": baseline}
    for name, model in models.items():
        methods[name] = generate(dev, model, vectorizer, name)
        frozen.write_jsonl(args.output_dir / f"dev_{name}_predictions.jsonl", methods[name])
    expected_ids = [frozen.row_id(row) for row in dev]
    for name, predictions in methods.items():
        if [frozen.row_id(row) for row in predictions] != expected_ids:
            raise RuntimeError(f"{name} dev ID mismatch")

    metrics = {name: {} for name in methods}
    details_by_unit: dict[str, dict[str, list[dict]]] = {}
    diagnostics = {}
    for unit in ("lawa", "character"):
        details_by_unit[unit] = {}
        for name, predictions in methods.items():
            aggregate, details = frozen.evaluate_predictions(
                predictions, dev, rouge_unit=unit, max_chars=MAX_CHARS,
                max_sentences=MAX_SENTENCES,
            )
            metrics[name][unit] = aggregate
            details_by_unit[unit][name] = details
            frozen.write_jsonl(args.output_dir / f"dev_{name}_{unit}_per_sample.jsonl", details)
    for name, predictions in methods.items():
        diagnostics[name] = frozen.decoder_diagnostics(predictions, max_chars=MAX_CHARS)
        metrics[name]["decoder_diagnostics"] = diagnostics[name]

    significance = {}
    selection_comparisons = {}
    for name in models:
        significance[name] = {}
        selection_comparisons[name] = frozen.compare_selections(baseline, methods[name])[0]
        for unit in ("lawa", "character"):
            significance[name][unit] = [
                paired_test(details_by_unit[unit][name], details_by_unit[unit]["baseline_frozen_v22"], metric)
                for metric in ("rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "fact_recall", "sentence_f1")
            ]

    dev_section_counts = Counter()
    for row in dev:
        dev_section_counts.update(section_labels(base.sentences(row)))
    section_audit = {
        "classes": list(SECTIONS),
        "priority": ["decision", "reason", "fact", "defense", "claim", "other"],
        "state_persists_until_next_marker": True,
        "late_decision_action_threshold": 0.65,
        "train_counts": dict(train_section_counts),
        "dev_counts": dict(dev_section_counts),
    }
    config = {
        "experiment": EXPERIMENT,
        "status": "dev_complete_test_untouched",
        "selection_basis": "train_and_dev_only",
        "test_read_or_generated": False,
        "single_changed_variable": "six deterministic sentence-section one-hot features",
        "training_supervision": "frozen_v2.2_oracle_labels",
        "vectorizer": "frozen_v2.2_tfidf_120000_char_2_4gram",
        "classifier": "LogisticRegression balanced liblinear",
        "decoder": "frozen_v2.1_budget_greedy",
        "max_chars": MAX_CHARS,
        "max_sentences": MAX_SENTENCES,
        "fact_weight": FACT_WEIGHT,
        "redundancy_weight": REDUNDANCY_WEIGHT,
        "methods": {
            "baseline_frozen_v22": "TFIDF+Legal14",
            "v25_a_tfidf_section": "TFIDF+Section6",
            "v25_b_tfidf_legal14_section": "TFIDF+Legal14+Section6",
        },
        "hyperparameter_search": False,
        "random_state": 42,
    }
    frozen.save_json(args.output_dir / "config.json", config)
    frozen.save_json(args.output_dir / "all_method_metrics.json", metrics)
    frozen.save_json(args.output_dir / "decoder_diagnostics.json", diagnostics)
    frozen.save_json(args.output_dir / "paired_significance.json", significance)
    frozen.save_json(args.output_dir / "selection_comparisons_vs_baseline.json", selection_comparisons)
    frozen.save_json(args.output_dir / "section_detector_audit.json", section_audit)
    frozen.save_json(args.output_dir / "fit_timings.json", timings)
    manifest = {
        **config,
        "script_sha256": frozen.sha256(script),
        "decoder_sha256": decoder_hash,
        "train_sha256": train_hash,
        "dev_sha256": dev_hash,
        "oracle_extracts_sha256": oracle_hash,
        "v22_model_sha256": model_hash,
        "v22_predictions_sha256": predictions_hash,
        "train_cases": len(train),
        "dev_cases": len(dev),
        "training_sentences": len(labels),
        "oracle_positive_labels": sum(labels),
        "fit_timings": timings,
        "outputs": {
            path.name: {"sha256": frozen.sha256(path), "bytes": path.stat().st_size}
            for path in sorted(args.output_dir.iterdir()) if path.is_file()
        },
    }
    frozen.save_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
