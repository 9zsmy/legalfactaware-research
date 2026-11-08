#!/usr/bin/env python3
"""Strict IN-Abs LR comparison: ordinary vs P50 budget-aware Oracle labels.

Only official-train documents are used for fitting.  The deterministic 6,325 /
703 split is inherited from phase 1, and official test contents are never read.
Both classifiers use identical rows, representation, features, LR settings and
complete-sentence MMR decoder; only the binary training target changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_inabs_phase1 as phase1

import numpy as np
from rouge_score import rouge_scorer
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


SEED = 42
CANDIDATE_CAP = 400
TRAIN_ORACLE_BUDGET = 643
BUDGETS = (423, 643, 973)
REDUNDANCY_WEIGHT = 0.2
SECTIONS = ("fact", "argument", "statute", "precedent", "ratio", "decision")

NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*(?:%|lakh|crore|million|billion)?\b", re.I)
MONEY_RE = re.compile(r"(?:Rs\.?|INR|₹)\s*\d|\b\d+(?:\.\d+)?\s*(?:rupees?|lakhs?|crores?)\b", re.I)
DATE_RE = re.compile(r"\b(?:\d{1,2}\s+)?(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\b|\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", re.I)
LAW_RE = re.compile(r"\b(?:section|article|act|rule|order|code|constitution|statute|provision)\b", re.I)
DECISION_RE = re.compile(r"\b(?:appeal|petition)\s+(?:is\s+)?(?:allowed|dismissed|disposed)|\b(?:set aside|quashed|acquitted|convicted|ordered accordingly|no order as to costs)\b", re.I)
CLAIM_RE = re.compile(r"\b(?:appellant|petitioner|plaintiff|claimant|prosecution)\b", re.I)
DEFENCE_RE = re.compile(r"\b(?:respondent|defendant|accused|learned counsel|submitted|contended|argued)\b", re.I)
COURT_RE = re.compile(r"\b(?:court|tribunal|bench|judge|justice|magistrate|high court|supreme court)\b", re.I)
CITATION_RE = re.compile(r"\b(?:AIR\s+\d{4}\s+SC|\d+\s+SCC\s+\d+|\d+\s+SCR\s+\d+|\d{4}\s+INSC\s+\d+|(?:civil|criminal)\s+appeal\s+no\.?\s*\d+)", re.I)
FACT_RE = re.compile(r"\b(?:facts?|incident|occurred|evidence|witness|complaint|fir|arrested|charged)\b", re.I)
ARGUMENT_RE = re.compile(r"\b(?:submitted|contended|argued|counsel|pleaded|according to the appellant|according to the respondent)\b", re.I)
PRECEDENT_RE = re.compile(r"\b(?:held in|relied upon|cited|precedent|supra|ratio decidendi)\b", re.I)
RATIO_RE = re.compile(r"\b(?:we find|we hold|in our view|therefore|thus|hence|it is clear|we are of the opinion|reasoning)\b", re.I)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def candidate_indices(total: int, cap: int = CANDIDATE_CAP) -> list[int]:
    """Reference-free full-document coverage with deterministic endpoints."""
    if total <= 0:
        return []
    if total <= cap:
        return list(range(total))
    values = [round(i * (total - 1) / (cap - 1)) for i in range(cap)]
    if len(set(values)) != cap:
        raise RuntimeError("evenly spaced candidate sampling produced duplicates")
    return values


def section_label(sentence: str, index: int, total: int) -> int:
    if DECISION_RE.search(sentence):
        return 5
    if PRECEDENT_RE.search(sentence) or CITATION_RE.search(sentence):
        return 3
    if LAW_RE.search(sentence):
        return 2
    if ARGUMENT_RE.search(sentence):
        return 1
    if RATIO_RE.search(sentence):
        return 4
    if FACT_RE.search(sentence) or index < max(1, round(total * 0.22)):
        return 0
    if index > round(total * 0.88):
        return 5
    return 4


def legal14(sentence: str, index: int, total: int) -> list[float]:
    wc = max(1, len(phase1.word_tokens(sentence)))
    return [
        index / max(total - 1, 1),
        (total - 1 - index) / max(total - 1, 1),
        min(math.log1p(wc) / 6.0, 1.0),
        min(wc / 80.0, 1.0),
        float(bool(NUMBER_RE.search(sentence))),
        min(len(NUMBER_RE.findall(sentence)), 5) / 5.0,
        float(bool(MONEY_RE.search(sentence))),
        float(bool(DATE_RE.search(sentence))),
        float(bool(LAW_RE.search(sentence))),
        float(bool(DECISION_RE.search(sentence))),
        float(bool(CLAIM_RE.search(sentence))),
        float(bool(DEFENCE_RE.search(sentence))),
        float(bool(COURT_RE.search(sentence))),
        float(bool(CITATION_RE.search(sentence))),
    ]


def explicit20(sentence: str, index: int, total: int) -> list[float]:
    values = legal14(sentence, index, total)
    one_hot = [0.0] * 6
    one_hot[section_label(sentence, index, total)] = 1.0
    return values + one_hot


def decorated(sentence: str, index: int, total: int) -> str:
    position_bin = min(9, int(10 * index / max(total, 1)))
    return f"position_bin_{position_bin} {sentence}"


def stable_rng(case_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}:{case_id}:negative-sampling".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def make_oracles(docs: list[dict], cache_path: Path) -> list[dict]:
    completed: list[dict] = []
    if cache_path.exists():
        completed = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if [row["id"] for row in completed] != [doc["id"] for doc in docs[:len(completed)]]:
            raise RuntimeError("oracle cache ID prefix mismatch")
    with cache_path.open("a" if completed else "w", encoding="utf-8") as handle:
        for position in range(len(completed), len(docs)):
            doc = docs[position]
            candidates = candidate_indices(len(doc["sentences"]))
            stats = phase1.sentence_statistics(doc["sentences"], doc["reference"], candidates)
            ordinary, ordinary_score, _ = phase1.greedy_oracle_sequence(
                doc["sentences"], doc["reference"], None, stats
            )
            budget, budget_score, _ = phase1.greedy_oracle_sequence(
                doc["sentences"], doc["reference"], TRAIN_ORACLE_BUDGET, stats
            )
            row = {
                "id": doc["id"], "source_sentences": len(doc["sentences"]),
                "candidate_indices": candidates, "ordinary": ordinary, "budget": budget,
                "ordinary_score": ordinary_score, "budget_score": budget_score,
            }
            completed.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if (position + 1) % 100 == 0 or position + 1 == len(docs):
                print(json.dumps({"phase": "oracles", "docs": position + 1, "total": len(docs)}), flush=True)
    return completed


def build_shared_training(docs: list[dict], oracles: list[dict]):
    texts, positions, ordinary_labels, budget_labels = [], [], [], []
    change_docs = 0
    positive_counts = Counter()
    for doc, oracle in zip(docs, oracles):
        ordinary, budget = set(oracle["ordinary"]), set(oracle["budget"])
        change_docs += ordinary != budget
        positive_counts["ordinary"] += len(ordinary)
        positive_counts["budget"] += len(budget)
        candidates = set(oracle["candidate_indices"])
        positives = ordinary | budget
        negatives = np.asarray(sorted(candidates - positives), dtype=int)
        n_negative = min(len(negatives), max(40, 5 * max(len(ordinary), len(budget), 1)))
        sampled = stable_rng(doc["id"]).choice(negatives, size=n_negative, replace=False) if n_negative else []
        selected = sorted(positives | set(map(int, sampled)))
        total = len(doc["sentences"])
        for index in selected:
            texts.append(doc["sentences"][index])
            positions.append((index, total))
            ordinary_labels.append(int(index in ordinary))
            budget_labels.append(int(index in budget))
    audit = {
        "documents": len(docs), "shared_training_rows": len(texts),
        "ordinary_positive_labels": sum(ordinary_labels), "budget_positive_labels": sum(budget_labels),
        "documents_with_changed_positive_set": change_docs,
        "document_change_rate": change_docs / len(docs),
        "row_label_agreement": float(np.mean(np.asarray(ordinary_labels) == np.asarray(budget_labels))),
    }
    return texts, positions, np.asarray(ordinary_labels, dtype=np.int8), np.asarray(budget_labels, dtype=np.int8), audit


def fit_models(texts, positions, y_ordinary, y_budget):
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 4), min_df=2, max_features=120000,
        sublinear_tf=True, dtype=np.float32,
    )
    text_matrix = vectorizer.fit_transform(
        decorated(text, index, total) for text, (index, total) in zip(texts, positions)
    )
    extras = csr_matrix(
        [explicit20(text, index, total) for text, (index, total) in zip(texts, positions)], dtype=np.float32
    )
    matrix = hstack([text_matrix, extras], format="csr")
    models = {}
    for name, labels in (("ordinary", y_ordinary), ("budget", y_budget)):
        print(json.dumps({"phase": "fit_lr", "model": name, "rows": len(labels), "features": matrix.shape[1], "positive_rate": float(labels.mean())}), flush=True)
        models[name] = LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=1000, random_state=SEED
        ).fit(matrix, labels)
    return vectorizer, models


def feature_matrix(vectorizer, sentences: list[str], candidates: list[int]):
    total = len(sentences)
    text = vectorizer.transform(decorated(sentences[i], i, total) for i in candidates)
    extras = csr_matrix([explicit20(sentences[i], i, total) for i in candidates], dtype=np.float32)
    return hstack([text, extras], format="csr")


def normalize(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    return (values - low) / (high - low) if high > low else np.full_like(values, 0.5)


def word_bigram_set(text: str) -> set[tuple[str, str]]:
    words = phase1.word_tokens(text)
    return set(zip(words, words[1:]))


def jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def decode(sentences: list[str], candidates: list[int], probabilities: np.ndarray, budget: int) -> list[int]:
    relevance = normalize(probabilities)
    lengths = [len(phase1.word_tokens(sentences[i])) for i in candidates]
    grams = [word_bigram_set(sentences[i]) for i in candidates]
    remaining, selected, used = set(range(len(candidates))), [], 0
    max_redundancy = np.zeros(len(candidates), dtype=np.float32)
    while remaining:
        fitting = [i for i in remaining if lengths[i] and used + lengths[i] <= budget]
        if not fitting:
            break
        best = max(fitting, key=lambda i: (float(relevance[i]) - REDUNDANCY_WEIGHT * float(max_redundancy[i]), float(relevance[i]), -candidates[i]))
        selected.append(candidates[best])
        used += lengths[best]
        remaining.remove(best)
        for item in remaining:
            max_redundancy[item] = max(max_redundancy[item], jaccard(grams[item], grams[best]))
    return sorted(selected)


def bootstrap_delta(rows: list[dict], budget: int, metric: str, samples: int = 10000) -> dict:
    delta = np.asarray([row["scores"][str(budget)]["budget"][metric] - row["scores"][str(budget)]["ordinary"][metric] for row in rows])
    rng = np.random.default_rng(SEED)
    means = np.empty(samples, dtype=float)
    for start in range(0, samples, 250):
        size = min(250, samples - start)
        means[start:start+size] = delta[rng.integers(0, len(delta), size=(size, len(delta)))].mean(axis=1)
    return {"mean": float(delta.mean()), "ci95_low": float(np.percentile(means, 2.5)), "ci95_high": float(np.percentile(means, 97.5)), "p_two_sided": float(min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0))))}


def evaluate(dev_docs, vectorizer, models, output_dir: Path) -> tuple[list[dict], dict]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    rows = []
    for position, doc in enumerate(dev_docs, 1):
        candidates = candidate_indices(len(doc["sentences"]))
        matrix = feature_matrix(vectorizer, doc["sentences"], candidates)
        probabilities = {name: model.predict_proba(matrix)[:, 1] for name, model in models.items()}
        row = {"id": doc["id"], "source_sentences": len(doc["sentences"]), "candidate_count": len(candidates), "scores": {}, "selected": {}}
        for budget in BUDGETS:
            row["scores"][str(budget)], row["selected"][str(budget)] = {}, {}
            for name in ("ordinary", "budget"):
                indices = decode(doc["sentences"], candidates, probabilities[name], budget)
                prediction = phase1.summary(doc["sentences"], indices)
                scores = scorer.score(doc["reference"], prediction)
                row["scores"][str(budget)][name] = {metric: scores[metric].fmeasure for metric in ("rouge1", "rouge2", "rougeLsum")}
                row["selected"][str(budget)][name] = indices
        rows.append(row)
        if position % 25 == 0 or position == len(dev_docs):
            print(json.dumps({"phase": "dev", "docs": position, "total": len(dev_docs)}), flush=True)
    with (output_dir / "dev_per_case.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    metrics = {}
    for budget in BUDGETS:
        metrics[str(budget)] = {}
        for name in ("ordinary", "budget"):
            metrics[str(budget)][name] = {metric: 100 * float(np.mean([r["scores"][str(budget)][name][metric] for r in rows])) for metric in ("rouge1", "rouge2", "rougeLsum")}
        metrics[str(budget)]["selection_change_rate"] = float(np.mean([r["selected"][str(budget)]["ordinary"] != r["selected"][str(budget)]["budget"] for r in rows]))
        metrics[str(budget)]["paired_bootstrap_budget_minus_ordinary"] = {metric: bootstrap_delta(rows, budget, metric) for metric in ("rouge1", "rouge2", "rougeLsum")}
    return rows, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to dataset/IN-Abs")
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.archive and phase1.md5(args.archive) != phase1.EXPECTED_ARCHIVE_MD5:
        raise RuntimeError("official dataset archive MD5 mismatch")
    train_part = args.dataset_dir / "train-data"
    official_ids = phase1.paired_ids(train_part)
    if len(official_ids) != phase1.EXPECTED_TRAIN:
        raise RuntimeError(f"unexpected official train count: {len(official_ids)}")
    eligible, excluded = phase1.eligible_train_ids(train_part, official_ids)
    train_ids, dev_ids = phase1.fixed_split(eligible)
    if len(train_ids) != 6325 or len(dev_ids) != 703:
        raise RuntimeError(f"unexpected fixed split: {len(train_ids)}/{len(dev_ids)}")
    all_train_lengths = [phase1.load_case(train_part, case_id)["reference_words"] for case_id in train_ids]
    derived = tuple(phase1.nearest_rank(all_train_lengths, q) for q in phase1.QUANTILES)
    if derived != BUDGETS:
        raise RuntimeError(f"train-derived budgets changed: {derived}")
    selected_train_ids = train_ids[:args.train_limit] if args.train_limit else train_ids
    selected_dev_ids = dev_ids[:args.dev_limit] if args.dev_limit else dev_ids
    train_docs = [phase1.load_case(train_part, case_id) for case_id in selected_train_ids]
    dev_docs = [phase1.load_case(train_part, case_id) for case_id in selected_dev_ids]
    started = time.time()
    write_json(args.output_dir / "split_manifest.json", {"seed": SEED, "eligible_official_train": len(eligible), "excluded": excluded, "model_train": len(train_ids), "dev": len(dev_ids), "run_train": len(train_docs), "run_dev": len(dev_docs), "test_contents_read": False, "budgets": BUDGETS})
    oracles = make_oracles(train_docs, args.output_dir / "train_oracles.jsonl")
    texts, positions, y_ordinary, y_budget, audit = build_shared_training(train_docs, oracles)
    write_json(args.output_dir / "training_label_audit.json", audit)
    vectorizer, models = fit_models(texts, positions, y_ordinary, y_budget)
    with (args.output_dir / "model_bundle.pkl").open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "models": models}, handle)
    _, metrics = evaluate(dev_docs, vectorizer, models, args.output_dir)
    write_json(args.output_dir / "dev_metrics.json", metrics)
    config = {
        "experiment": "IN-Abs LegalFactAware LR strict Oracle-label comparison", "status": "dev_complete_test_untouched",
        "single_changed_variable": "ordinary_vs_P50_budget_aware_oracle_binary_labels", "seed": SEED,
        "candidate_rule": "reference-free full-document evenly-spaced maximum 400 sentences", "train_oracle_budget_words": TRAIN_ORACLE_BUDGET,
        "dev_budgets_words": BUDGETS, "representation": "English char 2-4 TF-IDF, max_features=120000",
        "features": "Indian-English Legal14 + Section6", "classifier": "balanced liblinear logistic regression",
        "decoder": "complete-sentence dynamic fit + word-bigram MMR", "redundancy_weight": REDUNDANCY_WEIGHT,
        "official_test_contents_read": False, "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "config.json", config)
    print(json.dumps({"phase": "complete", "output": str(args.output_dir), "elapsed_seconds": config["elapsed_seconds"], "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
