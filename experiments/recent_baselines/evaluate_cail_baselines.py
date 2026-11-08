#!/usr/bin/env python3
"""Unified CAIL evaluation for extractive and generative baseline outputs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median

from protocol import FACT_PATTERNS, expression_sets, read_jsonl, rouge_l, rouge_n, write_jsonl


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_delta(left: list[float], right: list[float], seed: int, iterations: int) -> dict:
    differences = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    draws = [mean(rng.choice(differences) for _ in differences) for _ in range(iterations)]
    draws.sort()
    return {
        "mean_delta": mean(differences),
        "ci95_low": draws[int(0.025 * iterations)],
        "ci95_high": draws[min(int(0.975 * iterations), iterations - 1)],
    }


def evaluate_method(dataset: list[dict], predictions: list[dict], method: str, budget: int) -> tuple[dict, list[dict]]:
    gold = {str(row["id"]): row for row in dataset}
    pred = {str(row["id"]): row for row in predictions}
    if set(gold) != set(pred):
        raise RuntimeError(
            f"{method} ID mismatch: missing={len(set(gold)-set(pred))}, extra={len(set(pred)-set(gold))}"
        )
    details = []
    type_totals = {
        name: Counter(reference_instances=0, predicted_instances=0, true_positive_instances=0,
                      reference_nonempty_documents=0)
        for name in FACT_PATTERNS
    }
    type_doc_scores = {name: {"precision": [], "recall": [], "f1": []} for name in FACT_PATTERNS}
    doc_expression_scores = []
    for case_id, row in gold.items():
        output = pred[case_id]
        prediction = str(output.get("prediction") or output.get("generated_summary") or "")
        reference, source = str(row["reference"]), str(row["source"])
        ref_sets, pred_sets, source_sets = expression_sets(reference), expression_sets(prediction), expression_sets(source)
        per_doc_type_f1 = []
        source_pred_total = source_supported = 0
        for type_name in FACT_PATTERNS:
            ref_values, pred_values = ref_sets[type_name], pred_sets[type_name]
            overlap = ref_values & pred_values
            totals = type_totals[type_name]
            totals["reference_instances"] += len(ref_values)
            totals["predicted_instances"] += len(pred_values)
            totals["true_positive_instances"] += len(overlap)
            source_pred_total += len(pred_values)
            source_supported += len(pred_values & source_sets[type_name])
            if not ref_values and not pred_values:
                continue
            precision = len(overlap) / len(pred_values) if pred_values else 0.0
            recall = len(overlap) / len(ref_values) if ref_values else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_doc_type_f1.append(f1)
            if ref_values:
                totals["reference_nonempty_documents"] += 1
                type_doc_scores[type_name]["precision"].append(precision)
                type_doc_scores[type_name]["recall"].append(recall)
                type_doc_scores[type_name]["f1"].append(f1)
        expression_f1 = mean(per_doc_type_f1) if per_doc_type_f1 else 1.0
        doc_expression_scores.append(expression_f1)
        characters = len(prediction)
        details.append({
            "id": case_id,
            "method": method,
            "rouge_1_f1": rouge_n(reference, prediction, 1),
            "rouge_2_f1": rouge_n(reference, prediction, 2),
            "rouge_l_f1": rouge_l(reference, prediction),
            "expression_f1_document_macro": expression_f1,
            "source_expression_support": source_supported / source_pred_total if source_pred_total else 1.0,
            "characters": characters,
            "length_error_to_450": abs(characters - budget),
            "budget_violation": int(characters > budget),
        })

    lengths = [row["characters"] for row in details]
    summary = {
        "method": method,
        "cases": len(details),
        "rouge_1_f1": mean(row["rouge_1_f1"] for row in details),
        "rouge_2_f1": mean(row["rouge_2_f1"] for row in details),
        "rouge_l_f1": mean(row["rouge_l_f1"] for row in details),
        "expression_f1_document_macro": mean(doc_expression_scores),
        "source_expression_support": mean(row["source_expression_support"] for row in details),
        "mean_characters": mean(lengths),
        "median_characters": median(lengths),
        "p95_characters": percentile(lengths, 0.95),
        "max_characters": max(lengths),
        "mean_absolute_error_to_450": mean(abs(value - budget) for value in lengths),
        "within_10_characters": mean(abs(value - budget) < 10 for value in lengths),
        "within_25_characters": mean(abs(value - budget) < 25 for value in lengths),
        "within_50_characters": mean(abs(value - budget) < 50 for value in lengths),
        "budget_violation_count": sum(value > budget for value in lengths),
        "budget_violation_rate": mean(value > budget for value in lengths),
        "expression_by_type": {},
    }
    for type_name in FACT_PATTERNS:
        totals = type_totals[type_name]
        precision = totals["true_positive_instances"] / totals["predicted_instances"] if totals["predicted_instances"] else 0.0
        recall = totals["true_positive_instances"] / totals["reference_instances"] if totals["reference_instances"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary["expression_by_type"][type_name] = {
            **totals,
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
            "conditional_document_macro_precision": (
                mean(type_doc_scores[type_name]["precision"])
                if type_doc_scores[type_name]["precision"] else None
            ),
            "conditional_document_macro_recall": (
                mean(type_doc_scores[type_name]["recall"])
                if type_doc_scores[type_name]["recall"] else None
            ),
            "conditional_document_macro_f1": (
                mean(type_doc_scores[type_name]["f1"])
                if type_doc_scores[type_name]["f1"] else None
            ),
        }
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, help="METHOD=path.jsonl; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=450)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="evaluate only the first N dataset rows for smoke testing; 0 keeps strict full-set evaluation",
    )
    args = parser.parse_args()

    dataset = read_jsonl(args.dataset)
    if args.limit > 0:
        dataset = dataset[:args.limit]
    evaluation_ids = {str(row["id"]) for row in dataset}
    summaries, details_by_method = {}, {}
    for spec in args.prediction:
        if "=" not in spec:
            raise ValueError(f"prediction must be METHOD=PATH: {spec}")
        method, raw_path = spec.split("=", 1)
        prediction_rows = read_jsonl(Path(raw_path))
        if args.limit > 0:
            prediction_rows = [row for row in prediction_rows if str(row["id"]) in evaluation_ids]
        summary, details = evaluate_method(dataset, prediction_rows, method, args.budget)
        summaries[method], details_by_method[method] = summary, details

    significance = []
    methods = list(summaries)
    if len(methods) > 1:
        anchor = methods[0]
        anchor_rows = details_by_method[anchor]
        for method in methods[1:]:
            rows = details_by_method[method]
            for metric in ("rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "expression_f1_document_macro"):
                significance.append({
                    "method_a": anchor,
                    "method_b": method,
                    "metric": metric,
                    **bootstrap_delta(
                        [row[metric] for row in anchor_rows],
                        [row[metric] for row in rows],
                        args.seed,
                        args.bootstrap,
                    ),
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"budget": args.budget, "methods": summaries, "paired_bootstrap": significance},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    all_details = [row for method in methods for row in details_by_method[method]]
    write_jsonl(args.output_dir / "per_document.jsonl", all_details)
    table_fields = [
        "method", "cases", "rouge_1_f1", "rouge_2_f1", "rouge_l_f1",
        "expression_f1_document_macro", "source_expression_support", "mean_characters",
        "p95_characters", "mean_absolute_error_to_450", "budget_violation_count", "budget_violation_rate",
    ]
    with (args.output_dir / "main_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_fields)
        writer.writeheader()
        for method in methods:
            writer.writerow({name: summaries[method][name] for name in table_fields})
    print(json.dumps({method: {key: value for key, value in summary.items() if key != "expression_by_type"}
                      for method, summary in summaries.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

