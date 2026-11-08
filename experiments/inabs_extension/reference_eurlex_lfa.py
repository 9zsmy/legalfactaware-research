#!/usr/bin/env python3
"""LegalFactAware-v2.5-B transfer to EUR-Lex-Sum's official English task.

The comparison uses the official paragraph candidates and its train-derived
paragraph compression ratio.  Consequently, LegalFactAware and the paper's
LexRank baseline emit the same number of complete paragraphs for every input.
No validation or test reference is used for budgeting or model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


SECTIONS = ("context", "legal_basis", "provision", "procedure", "decision", "other")
MAX_ORACLE_CANDIDATES = 180
REDUNDANCY_WEIGHT = 0.2
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*(?:%|m|bn|million|billion)?\b", re.I)
MONEY_RE = re.compile(
    r"(?:EUR|€|£|USD|\$)\s?\d|\b\d+(?:\.\d+)?\s?(?:euros?|million|billion)\b", re.I
)
DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{4}\b|\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
    re.I,
)
LAW_RE = re.compile(
    r"\b(?:article|treaty|regulation|directive|decision|convention|protocol|annex|paragraph)\b",
    re.I,
)
DECISION_RE = re.compile(
    r"\b(?:has adopted this|have adopted this|shall enter into force|shall be binding|"
    r"done at|this decision is addressed|hereby adopts?)\b",
    re.I,
)
CONTEXT_RE = re.compile(r"\b(?:whereas|objective|in order to|with a view to|background|purpose)\b", re.I)
LEGAL_BASIS_RE = re.compile(
    r"\b(?:having regard to|pursuant to|legal basis|treaty on|article\s+\d+|directive\s+\d+|regulation\s+\([^)]*\)\s*no)\b",
    re.I,
)
PROVISION_RE = re.compile(
    r"\b(?:shall|must|may not|member states? shall|is prohibited|required to|obligation)\b", re.I
)
PROCEDURE_RE = re.compile(
    r"\b(?:european parliament|council|commission|committee|opinion of|consultation|procedure|notified)\b",
    re.I,
)
INSTITUTION_RE = re.compile(
    r"\b(?:European Union|European Parliament|European Commission|Council|Court of Justice|Member State)\b",
    re.I,
)
CITATION_RE = re.compile(
    r"\b(?:OJ\s+[LCS]\s*\d+|CELEX|ECLI)|\b(?:19|20)\d{2}/\d+(?:/EU|/EC|/EEC)?\b", re.I
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def ngrams(words: list[str], n: int) -> Counter:
    return Counter(tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1)))


def counter_f1(candidate: Counter, reference: Counter) -> float:
    total = sum(candidate.values()) + sum(reference.values())
    if not total:
        return 0.0
    overlap = sum(min(value, reference.get(key, 0)) for key, value in candidate.items())
    return 2.0 * overlap / total


def section_label(paragraph: str, index: int, total: int) -> int:
    if DECISION_RE.search(paragraph):
        return 4
    if LEGAL_BASIS_RE.search(paragraph):
        return 1
    if PROCEDURE_RE.search(paragraph):
        return 3
    if PROVISION_RE.search(paragraph):
        return 2
    if CONTEXT_RE.search(paragraph) or index < max(1, round(total * 0.22)):
        return 0
    if index > round(total * 0.82):
        return 4
    return 5


def legal14(paragraph: str, index: int, total: int) -> list[float]:
    word_count = max(1, len(tokens(paragraph)))
    return [
        index / max(total - 1, 1),
        (total - 1 - index) / max(total - 1, 1),
        min(math.log1p(word_count) / 6.0, 1.0),
        min(word_count / 80.0, 1.0),
        float(bool(NUMBER_RE.search(paragraph))),
        min(len(NUMBER_RE.findall(paragraph)), 5) / 5.0,
        float(bool(MONEY_RE.search(paragraph))),
        float(bool(DATE_RE.search(paragraph))),
        float(bool(LAW_RE.search(paragraph))),
        float(bool(DECISION_RE.search(paragraph))),
        float(bool(CONTEXT_RE.search(paragraph))),
        float(bool(PROVISION_RE.search(paragraph))),
        float(bool(INSTITUTION_RE.search(paragraph))),
        float(bool(CITATION_RE.search(paragraph))),
    ]


def extra20(paragraph: str, index: int, total: int) -> list[float]:
    values = legal14(paragraph, index, total)
    one_hot = [0.0] * len(SECTIONS)
    one_hot[section_label(paragraph, index, total)] = 1.0
    return values + one_hot


def decorated(paragraph: str, index: int) -> str:
    return f"position{index} {paragraph}"


def expected_paragraphs(paragraphs: list[str], median_ratio: float) -> int:
    return max(1, min(len(paragraphs), round(len(paragraphs) / median_ratio)))


def oracle(paragraphs: list[str], reference: str, budget: int) -> list[int]:
    reference_words = tokens(reference)
    ref1, ref2 = ngrams(reference_words, 1), ngrams(reference_words, 2)
    paragraph_words = [tokens(paragraph) for paragraph in paragraphs]
    para1 = [ngrams(words, 1) for words in paragraph_words]
    para2 = [ngrams(words, 2) for words in paragraph_words]
    ranked = sorted(
        range(len(paragraphs)),
        key=lambda i: (counter_f1(para1[i], ref1) + counter_f1(para2[i], ref2), -i),
        reverse=True,
    )[:MAX_ORACLE_CANDIDATES]
    remaining, chosen = set(ranked), []
    current, cur1, cur2 = 0.0, Counter(), Counter()
    while remaining and len(chosen) < budget:
        best, best_value = None, current
        for index in sorted(remaining):
            value = counter_f1(cur1 + para1[index], ref1) + counter_f1(cur2 + para2[index], ref2)
            if value > best_value + 1e-12:
                best, best_value = index, value
        if best is None:
            break
        chosen.append(best)
        remaining.remove(best)
        cur1 += para1[best]
        cur2 += para2[best]
        current = best_value
    return sorted(chosen)


def char_bigrams(text: str) -> set[str]:
    compact = "".join(text.lower().split())
    return {compact[i : i + 2] for i in range(max(0, len(compact) - 1))}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def normalize(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    return (values - low) / (high - low) if high > low else np.full_like(values, 0.5)


def decode(paragraphs: list[str], probabilities: np.ndarray, budget: int) -> list[int]:
    relevance = normalize(probabilities)
    grams = [char_bigrams(paragraph) for paragraph in paragraphs]
    max_redundancy = np.zeros(len(paragraphs), dtype=np.float32)
    remaining, selected = set(range(len(paragraphs))), []
    while remaining and len(selected) < budget:
        best = max(
            remaining,
            key=lambda i: (
                float(relevance[i]) - REDUNDANCY_WEIGHT * float(max_redundancy[i]),
                float(relevance[i]),
                -i,
            ),
        )
        selected.append(best)
        remaining.remove(best)
        for index in remaining:
            max_redundancy[index] = max(max_redundancy[index], jaccard(grams[index], grams[best]))
    return sorted(selected)


def read_jsonl(path: Path, segmenter, limit: int = 0) -> list[dict]:
    docs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            paragraphs = segmenter(row["reference"])
            if not paragraphs:
                raise ValueError(f"No eligible paragraphs for {row['celex_id']}")
            docs.append(
                {"id": row["celex_id"], "paragraphs": paragraphs, "reference": row["summary"]}
            )
            if limit and len(docs) >= limit:
                break
    return docs


def make_oracles(docs: list[dict], median_ratio: float, path: Path) -> list[list[int]]:
    completed = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            completed = [json.loads(line)["indices"] for line in handle if line.strip()]
    if len(completed) > len(docs):
        raise ValueError("Oracle cache is longer than the current train split")
    with path.open("a" if completed else "w", encoding="utf-8") as handle:
        for position in range(len(completed), len(docs)):
            doc = docs[position]
            budget = expected_paragraphs(doc["paragraphs"], median_ratio)
            indices = oracle(doc["paragraphs"], doc["reference"], budget)
            completed.append(indices)
            handle.write(json.dumps({"id": doc["id"], "budget": budget, "indices": indices}) + "\n")
            handle.flush()
            if (position + 1) % 100 == 0 or position + 1 == len(docs):
                print(json.dumps({"phase": "oracle", "docs": position + 1, "total": len(docs)}), flush=True)
    return completed


def fit_model(docs: list[dict], oracles: list[list[int]], seed: int):
    rng = np.random.default_rng(seed)
    paragraphs, positions, labels = [], [], []
    section_counts = Counter()
    for position, (doc, positive_list) in enumerate(zip(docs, oracles), 1):
        positive = set(positive_list)
        negative = np.asarray(sorted(set(range(len(doc["paragraphs"]))) - positive), dtype=int)
        keep = (
            rng.choice(negative, size=min(len(negative), max(30, 5 * len(positive))), replace=False)
            if len(negative)
            else []
        )
        selected = sorted(positive | set(map(int, keep)))
        total = len(doc["paragraphs"])
        for index in selected:
            paragraph = doc["paragraphs"][index]
            paragraphs.append(paragraph)
            positions.append((index, total))
            labels.append(int(index in positive))
            section_counts[SECTIONS[section_label(paragraph, index, total)]] += 1
        if position % 250 == 0 or position == len(docs):
            print(json.dumps({"phase": "training_rows", "docs": position, "rows": len(labels)}), flush=True)

    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 4), min_df=2, max_features=120000,
        sublinear_tf=True, dtype=np.float32,
    )
    text_matrix = vectorizer.fit_transform(
        decorated(paragraph, index) for paragraph, (index, _) in zip(paragraphs, positions)
    )
    extras = csr_matrix(
        [extra20(paragraph, index, total) for paragraph, (index, total) in zip(paragraphs, positions)],
        dtype=np.float32,
    )
    matrix = hstack([text_matrix, extras], format="csr")
    y = np.asarray(labels, dtype=np.int8)
    print(json.dumps({"phase": "fit_lr", "rows": len(y), "positive_rate": float(y.mean()), "features": matrix.shape[1]}), flush=True)
    classifier = LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed
    )
    classifier.fit(matrix, y)
    stats = {
        "training_documents": len(docs), "training_rows": len(y),
        "positive_labels": int(y.sum()), "positive_rate": float(y.mean()),
        "section_counts": dict(section_counts), "features": int(matrix.shape[1]),
    }
    return vectorizer, classifier, stats


def feature_matrix(vectorizer, paragraphs: list[str]):
    total = len(paragraphs)
    text_matrix = vectorizer.transform(decorated(paragraph, i) for i, paragraph in enumerate(paragraphs))
    extras = csr_matrix(
        [extra20(paragraph, i, total) for i, paragraph in enumerate(paragraphs)], dtype=np.float32
    )
    return hstack([text_matrix, extras], format="csr")


def predict_split(docs, vectorizer, classifier, median_ratio: float, output_dir: Path, split: str):
    pred_dir = output_dir / "paragraph" / "english" / split
    gold_dir = output_dir / "gold" / "english" / split
    pred_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with (output_dir / f"{split}_records.jsonl").open("w", encoding="utf-8") as handle:
        for position, doc in enumerate(docs, 1):
            paragraphs = doc["paragraphs"]
            probabilities = classifier.predict_proba(feature_matrix(vectorizer, paragraphs))[:, 1]
            budget = expected_paragraphs(paragraphs, median_ratio)
            indices = decode(paragraphs, probabilities, budget)
            prediction = "\n".join(paragraphs[index] for index in indices)
            filename = doc["id"].replace("/", "-").replace("(", "-").replace(")", "-") + ".txt"
            (pred_dir / filename).write_text(prediction, encoding="utf-8")
            (gold_dir / filename).write_text(doc["reference"], encoding="utf-8")
            record = {
                "id": doc["id"], "selected_indices": indices, "paragraph_budget": budget,
                "prediction_words": len(tokens(prediction)), "paragraph_count": len(indices),
            }
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            if position % 25 == 0 or position == len(docs):
                print(json.dumps({"phase": split, "docs": position, "total": len(docs)}), flush=True)
    return {
        "n": len(records),
        "mean_paragraph_budget": float(np.mean([row["paragraph_budget"] for row in records])),
        "mean_prediction_words": float(np.mean([row["prediction_words"] for row in records])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--official-code", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str((args.official_code / "Baselines").resolve()))
    import lexrank_baselines as official  # type: ignore

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train = read_jsonl(args.data_dir / "train.json", official.get_segmented_text, args.train_limit)
    validation = read_jsonl(args.data_dir / "validation.json", official.get_segmented_text, args.eval_limit)
    test = read_jsonl(args.data_dir / "test.json", official.get_segmented_text, args.eval_limit)
    ratios = [
        len(doc["paragraphs"]) / len(official.get_segmented_text(doc["reference"])) for doc in train
    ]
    median_ratio = float(np.median(ratios))
    print(json.dumps({"phase": "loaded", "train": len(train), "validation": len(validation), "test": len(test), "median_ratio": median_ratio}), flush=True)

    oracles = make_oracles(train, median_ratio, args.output_dir / "train_oracles.jsonl")
    vectorizer, classifier, train_stats = fit_model(train, oracles, args.seed)
    with (args.output_dir / "model_bundle.pkl").open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "classifier": classifier, "train_stats": train_stats}, handle)
    validation_stats = predict_split(validation, vectorizer, classifier, median_ratio, args.output_dir, "validation")
    test_stats = predict_split(test, vectorizer, classifier, median_ratio, args.output_dir, "test")

    manifest = {
        "experiment": "LegalFactAware-v2.5-B paragraph transfer to EUR-Lex-Sum English",
        "method": "char TF-IDF 2-4 + EU Legal14 + EU Section6 + balanced LR + oracle supervision + budget-aware decoder",
        "official_code_revision": "9f002d0ff2cc5bfd8846d52486018ee011a5d9fd",
        "dataset_revision": "f9fda7872f7ba1c433c49f29e1f49c896de6bee9",
        "candidate_units": "official filtered paragraphs",
        "budget": "same train-median paragraph compression ratio as official LexRank",
        "median_paragraph_compression_ratio": median_ratio,
        "supervision": "train-only greedy ROUGE-1+ROUGE-2 oracle",
        "oracle_candidate_cap": MAX_ORACLE_CANDIDATES,
        "redundancy_weight": REDUNDANCY_WEIGHT,
        "fact_weight": 0.0,
        "seed": args.seed,
        "hyperparameter_search": False,
        "validation_used_for_selection": False,
        "test_used_for_selection": False,
        "necessary_adaptations": ["EU legal cue regexes", "paragraph candidates", "official paragraph-count budget"],
        "train": train_stats,
        "validation": validation_stats,
        "test": test_stats,
        "dataset_sha256": {
            split: sha256(args.data_dir / f"{split}.json") for split in ("train", "validation", "test")
        },
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
