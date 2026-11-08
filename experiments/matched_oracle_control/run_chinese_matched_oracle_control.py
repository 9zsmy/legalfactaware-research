#!/usr/bin/env python3
"""Matched Chinese summary-level Oracle control on the sealed development split.

The only experimental change is whether the 450-character target budget is
enforced while constructing training labels. Both conditions retain the same
8-sentence cap and use the same representation, LR classifier, and 450-character
complete-sentence decoder. The script has no test-set argument.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import time
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.stats import wilcoxon
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression

import run_legalfactaware_formal_ablation_dev as formal
import run_legalfactaware_v21_budget_decoder_dev as decoder
import run_legalfactaware_v22_oracle_supervision_dev as oracle_impl
import run_legalfactaware_v25_structure_feature_dev as section_impl
import run_legalfactaware_v2_cail_clean as base


EXPERIMENT = "Chinese-Matched-Oracle-Character-Budget-Control-20261020"
TRAIN_CASES = 7495
DEV_CASES = 942
MAX_CHARS = 450
MAX_SENTENCES = 8
REDUNDANCY_WEIGHT = 0.2
EXPECTED = {
    "train": "8b19e6a1aa086e55c3a1d694e86cb4cf5abfe6929ee1258a3621b291bde564a2",
    "dev": "1a056c68dcfa5f11e9574d5f1b818fa475bb5607ea7f6565e2868072a55de2c0",
    "budget_oracle": "de20bf0a98cb04de5f11f2b0869701fa5c940a392f5804f0b85a09fad5f19960",
    "v22_model": "2cc0c9fe4b1fdaca6c1b2e37e545fcece3e5da83533296c1c12ba1d8bb8f70e9",
}
METRICS = ("rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "reference_fact_f1")


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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_matched_oracle(
    sentences: list[str], reference: str, max_chars: int | None
) -> tuple[list[int], float]:
    """One Oracle program; ``max_chars=None`` disables only character fit."""
    selected: list[int] = []
    remaining = set(range(len(sentences)))
    current_score = 0.0
    while remaining and len(selected) < MAX_SENTENCES:
        best_index = None
        best_score = current_score
        for index in sorted(remaining):
            ordered = sorted(selected + [index])
            candidate = "".join(sentences[item] for item in ordered)
            if max_chars is not None and len(candidate) > max_chars:
                continue
            score = oracle_impl.oracle_score(reference, candidate)
            if score > best_score + 1e-12:
                best_index = index
                best_score = score
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
        current_score = best_score
    return sorted(selected), current_score


def build_training(train: list[dict], archived_budget_rows: list[dict]):
    features: list[dict] = []
    budget_labels: list[int] = []
    unbudgeted_labels: list[int] = []
    oracle_rows: list[dict] = []
    changed_documents = 0
    for doc_id, (row, archived) in enumerate(zip(train, archived_budget_rows)):
        if decoder.row_id(row) != decoder.row_id(archived):
            raise RuntimeError(f"train/oracle ID mismatch at document {doc_id}")
        sentences = base.sentences(row)
        reference = base.reference(row)
        budget_indices, budget_score = select_matched_oracle(sentences, reference, MAX_CHARS)
        archived_indices = list(map(int, archived["oracle_selected_indices"]))
        if budget_indices != archived_indices:
            raise RuntimeError(f"recomputed budget Oracle mismatch at document {doc_id}")
        unbudgeted_indices, unbudgeted_score = select_matched_oracle(
            sentences, reference, None
        )
        budget_set = set(budget_indices)
        unbudgeted_set = set(unbudgeted_indices)
        changed_documents += int(budget_indices != unbudgeted_indices)
        sections = section_impl.section_labels(sentences)
        for index, sentence in enumerate(sentences):
            features.append({
                "doc_id": doc_id,
                "index": index,
                "total": len(sentences),
                "sentence": sentence,
                "section": sections[index],
            })
            budget_labels.append(int(index in budget_set))
            unbudgeted_labels.append(int(index in unbudgeted_set))
        oracle_rows.append({
            "id": decoder.row_id(row),
            "budget_selected_indices": budget_indices,
            "unbudgeted_selected_indices": unbudgeted_indices,
            "budget_sentence_count": len(budget_indices),
            "unbudgeted_sentence_count": len(unbudgeted_indices),
            "budget_characters": len("".join(sentences[i] for i in budget_indices)),
            "unbudgeted_characters": len("".join(sentences[i] for i in unbudgeted_indices)),
            "budget_objective": budget_score,
            "unbudgeted_objective": unbudgeted_score,
            "selection_changed": budget_indices != unbudgeted_indices,
        })
        if (doc_id + 1) % 250 == 0 or doc_id + 1 == len(train):
            print(f"constructed matched Oracles {doc_id + 1}/{len(train)}", flush=True)
    budget = np.asarray(budget_labels, dtype=np.int8)
    unbudgeted = np.asarray(unbudgeted_labels, dtype=np.int8)
    audit = {
        "training_documents": len(train),
        "training_sentences": len(features),
        "budget_positive_labels": int(budget.sum()),
        "budget_positive_rate": float(budget.mean()),
        "unbudgeted_positive_labels": int(unbudgeted.sum()),
        "unbudgeted_positive_rate": float(unbudgeted.mean()),
        "label_agreement": oracle_impl.binary_agreement(
            budget.astype(int).tolist(), unbudgeted.astype(int).tolist()
        ),
        "document_selection_change_count": changed_documents,
        "document_selection_change_rate": changed_documents / len(train),
        "budget_sentence_count_mean": mean(r["budget_sentence_count"] for r in oracle_rows),
        "unbudgeted_sentence_count_mean": mean(r["unbudgeted_sentence_count"] for r in oracle_rows),
        "budget_characters_mean": mean(r["budget_characters"] for r in oracle_rows),
        "unbudgeted_characters_mean": mean(r["unbudgeted_characters"] for r in oracle_rows),
        "budget_objective_mean": mean(r["budget_objective"] for r in oracle_rows),
        "unbudgeted_objective_mean": mean(r["unbudgeted_objective"] for r in oracle_rows),
        "archived_budget_oracle_exact_match": True,
    }
    return features, {"budget": budget, "unbudgeted": unbudgeted}, oracle_rows, audit


def fit_model(matrix, labels: np.ndarray, label_name: str):
    print(f"fitting {label_name}: matrix={matrix.shape}, positives={int(labels.sum())}", flush=True)
    started = time.perf_counter()
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    ).fit(matrix, labels)
    elapsed = time.perf_counter() - started
    print(f"fitted {label_name} in {elapsed:.2f}s", flush=True)
    return {
        "classifier": classifier,
        "feature_mode": "full",
        "training_label": label_name,
    }, elapsed


def paired_test(budget: list[dict], unbudgeted: list[dict], metric: str) -> dict:
    if [r["id"] for r in budget] != [r["id"] for r in unbudgeted]:
        raise RuntimeError("paired detail ID mismatch")
    differences = np.asarray(
        [float(a[metric]) - float(b[metric]) for a, b in zip(budget, unbudgeted)],
        dtype=float,
    )
    rng = np.random.default_rng(42)
    bootstrap = np.empty(10_000, dtype=float)
    for start in range(0, 10_000, 500):
        stop = min(start + 500, 10_000)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        bootstrap[start:stop] = differences[indices].mean(axis=1)
    p_value = 1.0 if np.allclose(differences, 0) else float(
        wilcoxon(differences, zero_method="pratt").pvalue
    )
    return {
        "metric": metric,
        "cases": len(differences),
        "mean_delta_budget_minus_unbudgeted": float(differences.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wilcoxon_p_raw": p_value if math.isfinite(p_value) else 1.0,
        "budget_win_rate": float(np.mean(differences > 0)),
        "tie_rate": float(np.mean(np.isclose(differences, 0))),
    }


def holm_adjust(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i]["wilcoxon_p_raw"])
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * rows[index]["wilcoxon_p_raw"]))
        rows[index]["wilcoxon_p_holm"] = running


def compact_metrics(name: str, result: dict) -> dict:
    lawa = result["lawa"]
    return {
        "method": name,
        "R1": lawa["rouge_1_f1"],
        "R2": lawa["rouge_2_f1"],
        "RL": lawa["rouge_l_f1"],
        "Fact_F1_macro": lawa["reference_fact_f1"],
        "Avg_chars": lawa["characters"],
        "Avg_sentences": lawa["selected_sentences"],
        "Hard_truncations": lawa["hard_truncation_count"],
        "Budget_violations": lawa["budget_violations"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--budget-oracle-extracts", type=Path, required=True)
    parser.add_argument("--v22-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_all = time.perf_counter()
    for name in ("models", "predictions", "per_sample", "tables"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    hashes = {
        "train_sha256": require(args.train, EXPECTED["train"], "train"),
        "dev_sha256": require(args.dev, EXPECTED["dev"], "dev"),
        "budget_oracle_sha256": require(
            args.budget_oracle_extracts, EXPECTED["budget_oracle"], "budget Oracle"
        ),
        "v22_model_sha256": require(args.v22_model, EXPECTED["v22_model"], "v22 model"),
    }
    train = base.read_jsonl(args.train)
    dev = base.read_jsonl(args.dev)
    archived = base.read_jsonl(args.budget_oracle_extracts)
    if (len(train), len(dev), len(archived)) != (TRAIN_CASES, DEV_CASES, TRAIN_CASES):
        raise RuntimeError("unexpected split size")
    with args.v22_model.open("rb") as handle:
        vectorizer = pickle.load(handle)["vectorizer"]

    features, labels, oracle_rows, label_audit = build_training(train, archived)
    write_jsonl(args.output_dir / "train_matched_oracles.jsonl", oracle_rows)
    save_json(args.output_dir / "training_label_audit.json", label_audit)

    print("building frozen full representation", flush=True)
    text = vectorizer.transform(base.decorated_text(row) for row in features)
    legal = csr_matrix([base.explicit_features(row) for row in features])
    section = csr_matrix([section_impl.section_one_hot(row["section"]) for row in features])
    matrix = hstack([text, legal, section], format="csr")
    models = {}
    fit_seconds = {}
    for name in ("budget", "unbudgeted"):
        models[name], fit_seconds[name] = fit_model(matrix, labels[name], name)
        with (args.output_dir / "models" / f"{name}.pkl").open("wb") as handle:
            pickle.dump(models[name], handle)
    del matrix, text, legal, section

    scores, score_timings = formal.precompute_dev_scores(dev, models, vectorizer)
    results = {}
    details = {}
    for name, display in (
        ("budget", "Budget-constrained matched Oracle"),
        ("unbudgeted", "Budget-unconstrained matched Oracle"),
    ):
        predictions, decode_timing = formal.generate_variant(
            dev,
            scores[name],
            {"name": display, "model": name, "redundancy": REDUNDANCY_WEIGHT},
        )
        write_jsonl(args.output_dir / "predictions" / f"{name}.jsonl", predictions)
        result = formal.evaluate_variant(name, predictions, dev, MAX_CHARS, args.output_dir)
        details[name] = result.pop("lawa_details")
        result["fit_seconds"] = fit_seconds[name]
        result["score_timing"] = score_timings[name]
        result["decode_timing"] = decode_timing
        results[name] = result
        print(
            f"{name}: {result['lawa']['rouge_1_f1']:.6f}/"
            f"{result['lawa']['rouge_2_f1']:.6f}/"
            f"{result['lawa']['rouge_l_f1']:.6f}",
            flush=True,
        )

    significance = [paired_test(details["budget"], details["unbudgeted"], metric) for metric in METRICS]
    holm_adjust(significance)
    comparison = [
        compact_metrics("Budget-constrained matched Oracle", results["budget"]),
        compact_metrics("Budget-unconstrained matched Oracle", results["unbudgeted"]),
    ]
    write_csv(args.output_dir / "tables" / "matched_oracle_comparison.csv", comparison)
    save_json(args.output_dir / "metrics.json", results)
    save_json(args.output_dir / "paired_significance.json", significance)

    config = {
        "experiment": EXPERIMENT,
        "status": "dev_complete_test_untouched",
        "test_argument_exists": False,
        "single_changed_variable": "450-character constraint during Oracle label construction",
        "shared_oracle_objective": "character ROUGE-1 F1 + character ROUGE-2 F1",
        "positive_marginal_gain_required": True,
        "oracle_tie_break": "lowest source index",
        "shared_oracle_max_sentences": MAX_SENTENCES,
        "budget_oracle_max_chars": MAX_CHARS,
        "unbudgeted_oracle_max_chars": None,
        "shared_representation": "frozen TF-IDF + Legal14 + Section6",
        "shared_classifier": "LogisticRegression(liblinear, balanced, seed=42, max_iter=1000)",
        "shared_decoder": "complete-sentence budget greedy",
        "decoder_max_chars": MAX_CHARS,
        "decoder_max_sentences": MAX_SENTENCES,
        "decoder_redundancy_weight": REDUNDANCY_WEIGHT,
        "train_cases": TRAIN_CASES,
        "dev_cases": DEV_CASES,
        "python": platform.python_version(),
        "sklearn": sklearn_version,
        **hashes,
    }
    save_json(args.output_dir / "config.json", config)
    manifest = {
        **config,
        "script_sha256": sha256(Path(__file__).resolve()),
        "elapsed_seconds": time.perf_counter() - started_all,
        "training_label_audit": label_audit,
        "primary_delta_budget_minus_unbudgeted": {
            row["metric"]: row["mean_delta_budget_minus_unbudgeted"] for row in significance
        },
        "outputs": {},
    }
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"][str(path.relative_to(args.output_dir))] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    save_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
