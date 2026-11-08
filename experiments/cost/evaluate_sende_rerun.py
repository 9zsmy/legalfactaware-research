#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean


FACT_PATTERNS = {
    "amount": re.compile(r"(?:人民币)?\d+(?:\.\d+)?(?:万)?元"),
    "date": re.compile(r"\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?"),
    "law": re.compile(r"《[^》]{2,50}》|第[一二三四五六七八九十百千万零〇\d]+条"),
    "decision": re.compile(r"驳回|支持|不予支持|支付|赔偿|返还|解除|确认|承担|连带责任|优先受偿"),
}


def source_text(row: dict) -> str:
    items = row.get("text") or row.get("raw_text") or row.get("sentences") or []
    values = []
    for item in items:
        if isinstance(item, dict):
            values.append(str(item.get("sentence") or item.get("text") or ""))
        else:
            values.append(str(item))
    return "".join(values)


def fact_scores(reference: str, prediction: str, source: str) -> tuple[float, float, float]:
    recalls, supports, f1s = [], [], []
    for pattern in FACT_PATTERNS.values():
        ref = set(pattern.findall(reference))
        pred = set(pattern.findall(prediction))
        src = set(pattern.findall(source))
        if ref:
            recalls.append(len(ref & pred) / len(ref))
        if pred:
            supports.append(len(pred & src) / len(pred))
        if ref or pred:
            overlap = len(ref & pred)
            precision = overlap / len(pred) if pred else 0.0
            recall = overlap / len(ref) if ref else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return (
        mean(recalls) if recalls else 1.0,
        mean(supports) if supports else 1.0,
        mean(f1s) if f1s else 1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--sende-root", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.sende_root))
    from evaluate_cail import rouge_l, rouge_n

    records = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    references = {}
    sources = {}
    for row in records:
        reference = next(
            (row[key] for key in ("reference_summary", "summary", "reference", "target") if key in row),
            None,
        )
        if reference is None:
            raise KeyError(f"No reference field in data row: {sorted(row)}")
        references[str(row["id"])] = reference
        sources[str(row["id"])] = source_text(row)

    reports = []
    for path in args.prediction:
        predictions = [json.loads(line) for line in path.open(encoding="utf-8")]
        if len(predictions) != len(records):
            raise ValueError(f"{path}: {len(predictions)} predictions, expected {len(records)}")
        rows = []
        for item in predictions:
            doc_id = str(item["id"])
            prediction = item.get("generated_summary", item.get("prediction", ""))
            reference = references[doc_id]
            fact_recall, fact_support, fact_f1 = fact_scores(
                reference, prediction, sources[doc_id]
            )
            rows.append(
                {
                    "r1": rouge_n(reference, prediction, 1),
                    "r2": rouge_n(reference, prediction, 2),
                    "rl": rouge_l(reference, prediction),
                    "characters": len(prediction),
                    "sentences": len(item.get("selected_indices", [])),
                    "empty": not prediction,
                    "fact_recall": fact_recall,
                    "fact_support": fact_support,
                    "fact_f1": fact_f1,
                }
            )
        reports.append(
            {
                "prediction": str(path),
                "documents": len(rows),
                "rouge_1": 100 * mean(row["r1"] for row in rows),
                "rouge_2": 100 * mean(row["r2"] for row in rows),
                "rouge_l": 100 * mean(row["rl"] for row in rows),
                "fact_recall": 100 * mean(row["fact_recall"] for row in rows),
                "source_fact_support": 100 * mean(row["fact_support"] for row in rows),
                "fact_f1": 100 * mean(row["fact_f1"] for row in rows),
                "mean_characters": mean(row["characters"] for row in rows),
                "mean_sentences": mean(row["sentences"] for row in rows),
                "empty_summaries": sum(row["empty"] for row in rows),
                "budget_violations": sum(
                    row["characters"] > 450 or row["sentences"] > 8 for row in rows
                ),
            }
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
