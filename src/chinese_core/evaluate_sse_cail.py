#!/usr/bin/env python3
"""Locked character-level CAIL evaluator for SSE predictions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Callable


FACT_PATTERNS = {
    "amount": re.compile(r"(?:人民币)?\d+(?:\.\d+)?(?:万)?元"),
    "date": re.compile(r"\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?"),
    "law": re.compile(r"《[^》]{2,50}》|第[一二三四五六七八九十百千万零〇\d]+条"),
    "decision": re.compile(r"驳回|支持|不予支持|支付|赔偿|返还|解除|确认|承担|连带责任|优先受偿"),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def character_tokens(text: str) -> list[str]:
    return [character for character in text or "" if not character.isspace()]


def get_tokenizer(mode: str) -> Callable[[str], list[str]]:
    if mode == "character":
        return character_tokens
    try:
        import lawa
    except ImportError as error:
        raise RuntimeError("--rouge-unit lawa requires lawa==1.2.6") from error
    return lambda text: [str(token) for token in lawa.cut(text or "") if str(token).strip()]


def rouge_n(
    reference: str,
    prediction: str,
    order: int,
    tokenizer: Callable[[str], list[str]] = character_tokens,
) -> float:
    left, right = tokenizer(reference), tokenizer(prediction)
    if len(left) < order or len(right) < order:
        return 0.0
    left_counts = Counter(tuple(left[index : index + order]) for index in range(len(left) - order + 1))
    right_counts = Counter(tuple(right[index : index + order]) for index in range(len(right) - order + 1))
    overlap = sum((left_counts & right_counts).values())
    precision = overlap / sum(right_counts.values())
    recall = overlap / sum(left_counts.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def rouge_l(
    reference: str,
    prediction: str,
    tokenizer: Callable[[str], list[str]] = character_tokens,
) -> float:
    left, right = tokenizer(reference), tokenizer(prediction)
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, 1):
            current.append(
                previous[index - 1] + 1
                if left_token == right_token
                else max(previous[index], current[-1])
            )
        previous = current
    overlap = previous[-1]
    precision, recall = overlap / len(right), overlap / len(left)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def fact_metrics(reference: str, prediction: str, source: str) -> tuple[float, float]:
    recalls, supports = [], []
    for pattern in FACT_PATTERNS.values():
        reference_values = set(pattern.findall(reference))
        prediction_values = set(pattern.findall(prediction))
        source_values = set(pattern.findall(source))
        if reference_values:
            recalls.append(len(reference_values & prediction_values) / len(reference_values))
        if prediction_values:
            supports.append(len(prediction_values & source_values) / len(prediction_values))
    return (mean(recalls) if recalls else 1.0, mean(supports) if supports else 1.0)


def sentence_metrics(selected: set[int], labels: list[int]) -> tuple[float, float, float]:
    gold = {index for index, label in enumerate(labels) if label}
    overlap = len(selected & gold)
    precision = overlap / len(selected) if selected else 1.0
    recall = overlap / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def selected_indices(row: dict) -> set[int]:
    """Normalize sentence positions emitted by SSE, MemSum, or local baselines."""
    if "selected_positions" in row:
        return {int(index) for index in row["selected_positions"]}
    if "selected_indices" in row:
        return {int(index) for index in row["selected_indices"]}
    return {
        int(index)
        for part in row.get("part_indices", [])
        for index in part
    }


def normalize_gold(row: dict) -> dict:
    if "raw_text" in row:
        return row
    source_items = row.get("text") or []
    return {
        "id": str(row.get("id") or row.get("case_id") or ""),
        "raw_text": [
            str(item.get("sentence") or item.get("text") or "")
            if isinstance(item, dict)
            else str(item)
            for item in source_items
        ],
        "raw_summary": str(row.get("summary") or row.get("reference_summary") or ""),
        "global_labels": [
            int(item.get("label", 0)) if isinstance(item, dict) else 0 for item in source_items
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True, help="Prepared SSE split JSONL.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=450)
    parser.add_argument("--max-sentences", type=int, default=8)
    parser.add_argument("--rouge-unit", choices=("character", "lawa"), default="character")
    args = parser.parse_args()
    tokenizer = get_tokenizer(args.rouge_unit)

    predictions = {str(row["id"]): row for row in read_jsonl(args.predictions)}
    normalized_gold = [normalize_gold(row) for row in read_jsonl(args.gold)]
    gold = {str(row["id"]): row for row in normalized_gold}
    if args.expected_cases is not None and len(predictions) != args.expected_cases:
        raise RuntimeError(f"expected {args.expected_cases} predictions, found {len(predictions)}")
    if set(predictions) != set(gold):
        raise RuntimeError(
            f"prediction/gold ID mismatch: missing={len(set(gold) - set(predictions))}, "
            f"extra={len(set(predictions) - set(gold))}"
        )

    details = []
    for case_id in sorted(gold):
        target, output = gold[case_id], predictions[case_id]
        prediction = str(output.get("generated_summary") or output.get("prediction") or "")
        selected = selected_indices(output)
        fact_recall, source_support = fact_metrics(
            target["raw_summary"], prediction, "".join(target["raw_text"])
        )
        precision, recall, f1 = sentence_metrics(selected, target.get("global_labels", []))
        details.append(
            {
                "id": case_id,
                "rouge_1_f1": rouge_n(target["raw_summary"], prediction, 1, tokenizer),
                "rouge_2_f1": rouge_n(target["raw_summary"], prediction, 2, tokenizer),
                "rouge_l_f1": rouge_l(target["raw_summary"], prediction, tokenizer),
                "fact_recall": fact_recall,
                "source_fact_support": source_support,
                "sentence_precision": precision,
                "sentence_recall": recall,
                "sentence_f1": f1,
                "characters": len(prediction),
                "selected_sentences": len(selected),
            }
        )
    report = {
        "cases": len(details),
        "rouge_1_f1": mean(row["rouge_1_f1"] for row in details),
        "rouge_2_f1": mean(row["rouge_2_f1"] for row in details),
        "rouge_l_f1": mean(row["rouge_l_f1"] for row in details),
        "fact_recall": mean(row["fact_recall"] for row in details),
        "source_fact_support": mean(row["source_fact_support"] for row in details),
        "sentence_precision": mean(row["sentence_precision"] for row in details),
        "sentence_recall": mean(row["sentence_recall"] for row in details),
        "sentence_f1": mean(row["sentence_f1"] for row in details),
        "average_characters": mean(row["characters"] for row in details),
        "max_characters": max(row["characters"] for row in details),
        "max_selected_sentences": max(row["selected_sentences"] for row in details),
        "budget_violations": sum(
            row["characters"] > args.max_chars or row["selected_sentences"] > args.max_sentences
            for row in details
        ),
        "rouge_unit": args.rouge_unit,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
