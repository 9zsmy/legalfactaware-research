#!/usr/bin/env python3
"""Train and evaluate the real LegalFactAware architecture on LCSTS.

This is a target-domain retraining of the learned method, not a rule-based
proxy. It preserves the core architecture used by LegalFactAware-BillSum:
character 2--4 gram TF-IDF, 14 explicit fact/structure features, balanced
logistic regression, and reference-fact-aware sample weights. Only Chinese
sentence splitting, tokenization, and fact patterns are adapted to LCSTS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    from evaluate_extended_summary_metrics import evaluate_row
except ModuleNotFoundError:
    from scripts.evaluate_extended_summary_metrics import evaluate_row


SENTENCE_RE = re.compile(r"(?<=[。！？；!?])")
MONEY_RE = re.compile(r"(?:人民币|RMB|￥|¥)?\s*\d[\d,.]*(?:万|亿|千)?元")
DATE_RE = re.compile(
    r"(?:19|20)\d{2}年(?:\d{1,2}月(?:\d{1,2}日)?)?"
    r"|\d{1,2}月\d{1,2}日|(?:19|20)\d{2}[-/.]\d{1,2}(?:[-/.]\d{1,2})?"
)
LAW_TITLE_RE = re.compile(r"《[^》]{2,40}(?:法|条例|规定|办法|规则|细则|解释)》")
LAW_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千万零〇两\d]+条")
ACTION_KEYWORDS = (
    "要求", "决定", "通过", "批准", "宣布", "禁止", "支持", "反对", "起诉",
    "判决", "赔偿", "承担", "发布", "实施", "取消", "增加", "减少", "建立",
)
FEATURE_NAMES = (
    "position_norm", "position_reverse", "sentence_len_log", "has_money",
    "money_count", "has_date", "date_count", "has_law_title",
    "law_title_count", "has_law_article", "law_article_count",
    "has_action_keyword", "action_keyword_count", "fact_signal_count",
)


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if limit:
        rows = rows[:limit]
    return [
        {
            "id": str(row.get("id", index)),
            "text": str(row.get("source", row.get("text", "")) or ""),
            "summary": str(row.get("reference", row.get("summary", "")) or ""),
        }
        for index, row in enumerate(rows)
    ]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", "", text or "")
    parts = [part for part in SENTENCE_RE.split(text) if part]
    return parts or ([text] if text else [])


def tokens(text: str) -> list[str]:
    return [char.lower() for char in text or "" if not char.isspace()]


def bigrams(items: list[str]) -> Counter[tuple[str, str]]:
    return Counter(zip(items, items[1:]))


def rouge2_precision(sentence: str, reference_counts: Counter[tuple[str, str]]) -> float:
    prediction_counts = bigrams(tokens(sentence))
    total = sum(prediction_counts.values())
    return sum((prediction_counts & reference_counts).values()) / total if total else 0.0


def extracted_fact_sets(text: str) -> dict[str, set[str]]:
    lowered = (text or "").lower()
    return {
        "money": {match.group(0).replace(" ", "") for match in MONEY_RE.finditer(text or "")},
        "date": {match.group(0).replace(" ", "") for match in DATE_RE.finditer(text or "")},
        "law_title": {match.group(0) for match in LAW_TITLE_RE.finditer(text or "")},
        "law_article": {match.group(0) for match in LAW_ARTICLE_RE.finditer(text or "")},
        "action": {keyword for keyword in ACTION_KEYWORDS if keyword in lowered},
    }


def fact_overlap(sentence: str, reference_facts: dict[str, set[str]]) -> int:
    sentence_facts = extracted_fact_sets(sentence)
    return sum(len(sentence_facts[name] & reference_facts[name]) for name in reference_facts)


def explicit_features(row: dict) -> list[float]:
    facts = extracted_fact_sets(row["sentence"])
    total = max(int(row["total"]), 1)
    action_count = len(facts["action"])
    signal_count = sum(bool(values) for values in facts.values())
    return [
        row["index"] / total,
        (total - row["index"]) / total,
        math.log1p(len(row["sentence"])) / 8.0,
        int(bool(facts["money"])),
        min(len(facts["money"]), 5) / 5.0,
        int(bool(facts["date"])),
        min(len(facts["date"]), 5) / 5.0,
        int(bool(facts["law_title"])),
        min(len(facts["law_title"]), 5) / 5.0,
        int(bool(facts["law_article"])),
        min(len(facts["law_article"]), 5) / 5.0,
        int(bool(action_count)),
        min(action_count, 5) / 5.0,
        signal_count / 5.0,
    ]


def decorated_text(row: dict) -> str:
    return f"position{row['index']} {row['sentence']}"


def build_training_rows(items, threshold, positive_bonus, fact_bonus, max_weight):
    rows: list[dict] = []
    labels: list[int] = []
    weights: list[float] = []
    for doc_id, item in enumerate(items):
        sentences = split_sentences(item["text"])
        reference_counts = bigrams(tokens(item["summary"]))
        reference_facts = extracted_fact_sets(item["summary"])
        for index, sentence in enumerate(sentences):
            label = int(rouge2_precision(sentence, reference_counts) > threshold)
            overlap = fact_overlap(sentence, reference_facts)
            weight = 1.0 + (positive_bonus if label else 0.0) + fact_bonus * min(overlap, 3)
            rows.append({"doc_id": doc_id, "index": index, "total": len(sentences), "sentence": sentence})
            labels.append(label)
            weights.append(min(weight, max_weight))
        if (doc_id + 1) % 1000 == 0 or doc_id + 1 == len(items):
            print(f"labeled {doc_id + 1}/{len(items)} LCSTS training rows", flush=True)
    return rows, labels, weights


def fit_models(rows, labels, weights, max_features):
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 4), min_df=2,
        max_features=max_features, sublinear_tf=True,
    )
    text_x = vectorizer.fit_transform(decorated_text(row) for row in rows)
    feature_x = csr_matrix([explicit_features(row) for row in rows])
    x = hstack([text_x, feature_x], format="csr")
    kwargs = dict(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)
    baseline = LogisticRegression(**kwargs).fit(x, labels)
    fact_aware = LogisticRegression(**kwargs).fit(x, labels, sample_weight=weights)
    return vectorizer, baseline, fact_aware


def inference_rows(sentences: list[str]) -> list[dict]:
    return [
        {"doc_id": 0, "index": index, "total": len(sentences), "sentence": sentence}
        for index, sentence in enumerate(sentences)
    ]


def predict_scores(vectorizer, classifier, rows):
    text_x = vectorizer.transform(decorated_text(row) for row in rows)
    feature_x = csr_matrix([explicit_features(row) for row in rows])
    return classifier.predict_proba(hstack([text_x, feature_x], format="csr"))[:, 1].tolist()


def select_summary(sentences: list[str], scores: list[float], k: int, max_chars: int) -> str:
    selected = sorted(sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:k])
    output: list[str] = []
    used = 0
    for index in selected:
        remaining = max_chars - used
        if remaining <= 0:
            break
        sentence = sentences[index]
        output.append(sentence[:remaining])
        used += min(len(sentence), remaining)
    return "".join(output)


def bootstrap_difference(left, right, iterations, seed):
    differences = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    estimates = [mean(rng.choice(differences) for _ in differences) for _ in range(iterations)]
    estimates.sort()
    return {
        "mean_difference": mean(differences),
        "ci95_low": estimates[int(iterations * 0.025)],
        "ci95_high": estimates[min(int(iterations * 0.975), iterations - 1)],
        "bootstrap_p": min(1.0, 2 * min(
            sum(value <= 0 for value in estimates) / iterations,
            sum(value >= 0 for value in estimates) / iterations,
        )),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/lcsts/train.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/lcsts/test.jsonl"))
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=32)
    parser.add_argument("--max-features", type=int, default=120000)
    parser.add_argument("--label-threshold", type=float, default=0.1)
    parser.add_argument("--positive-bonus", type=float, default=2.0)
    parser.add_argument("--fact-bonus", type=float, default=0.5)
    parser.add_argument("--max-weight", type=float, default=5.0)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/thesis_paper_20260708/results/legalfactaware_lcsts_725"),
    )
    args = parser.parse_args()

    train_items = read_jsonl(args.train, args.train_limit)
    test_items = read_jsonl(args.test, args.test_limit)
    rows, labels, weights = build_training_rows(
        train_items, args.label_threshold, args.positive_bonus, args.fact_bonus, args.max_weight
    )
    if len(set(labels)) < 2:
        raise ValueError("training labels contain fewer than two classes")
    vectorizer, baseline, fact_aware = fit_models(rows, labels, weights, args.max_features)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "legalfactaware_lcsts_model.pkl").open("wb") as handle:
        pickle.dump({
            "vectorizer": vectorizer,
            "baseline_classifier": baseline,
            "legalfactaware_classifier": fact_aware,
            "feature_names": FEATURE_NAMES,
        }, handle)

    details: list[dict] = []
    for index, item in enumerate(test_items):
        sentences = split_sentences(item["text"])
        inference = inference_rows(sentences)
        lead_scores = [float(len(sentences) - i) for i in range(len(sentences))]
        predictions = {"Lead-32": select_summary(sentences, lead_scores, args.k, args.max_chars)}
        predictions["LearnedRanker-Chinese"] = select_summary(
            sentences, predict_scores(vectorizer, baseline, inference), args.k, args.max_chars
        ) if inference else ""
        predictions["LegalFactAware-LCSTS"] = select_summary(
            sentences, predict_scores(vectorizer, fact_aware, inference), args.k, args.max_chars
        ) if inference else ""
        for method, prediction in predictions.items():
            details.append({
                "case_id": f"lcsts_{index}", "method": method,
                **{key: round(value, 8) for key, value in evaluate_row(
                    item["text"], item["summary"], prediction, "char"
                ).items()},
                "reference": item["summary"], "prediction": prediction,
            })
        if (index + 1) % 100 == 0 or index + 1 == len(test_items):
            print(f"evaluated {index + 1}/{len(test_items)} LCSTS cases", flush=True)

    metric_names = [
        key for key in details[0]
        if key not in {"case_id", "method", "reference", "prediction"}
    ]
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in details:
        grouped[row["method"]].append(row)
    summary = [{
        "method": method, "cases": len(group_rows),
        **{metric: round(mean(float(row[metric]) for row in group_rows), 6) for metric in metric_names},
    } for method, group_rows in sorted(grouped.items())]

    ours = {row["case_id"]: row for row in details if row["method"] == "LegalFactAware-LCSTS"}
    significance: list[dict] = []
    for other_name in ("LearnedRanker-Chinese", "Lead-32"):
        other = {row["case_id"]: row for row in details if row["method"] == other_name}
        common = sorted(set(ours) & set(other))
        for metric in ("rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "chrf_plus_plus"):
            result = bootstrap_difference(
                [float(ours[case_id][metric]) for case_id in common],
                [float(other[case_id][metric]) for case_id in common],
                args.bootstrap, args.seed,
            )
            significance.append({
                "method_a": "LegalFactAware-LCSTS", "method_b": other_name,
                "metric": metric, "common_cases": len(common),
                **{key: round(value, 6) for key, value in result.items()},
            })

    write_csv(args.output_dir / "lcsts_details.csv", details)
    write_csv(args.output_dir / "lcsts_summary.csv", summary)
    write_csv(args.output_dir / "lcsts_significance.csv", significance)
    (args.output_dir / "run_manifest.json").write_text(json.dumps({
        "method_identity": "Target-domain retraining of the learned LegalFactAware architecture",
        "not_proxy": True,
        "dataset": "LCSTS local 10,000-train / 725-test subset",
        "task_role": "Chinese general-domain stress test; not a legal benchmark",
        "train_cases": len(train_items),
        "training_sentences": len(rows),
        "positive_sentence_labels": sum(labels),
        "positive_label_rate": sum(labels) / len(labels),
        "test_cases": len(test_items),
        "architecture": "char 2-4 gram TF-IDF + 14 explicit features + balanced LogisticRegression",
        "sample_weights": {
            "positive_bonus": args.positive_bonus,
            "fact_bonus": args.fact_bonus,
            "max_weight": args.max_weight,
        },
        "selection": {"top_k": args.k, "max_chars": args.max_chars},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_dir / "lcsts_summary.csv")


if __name__ == "__main__":
    main()
