#!/usr/bin/env python3
"""Dev-only LegalFactAware-v2.1 budget-aware decoder experiment.

The frozen classifier, vectorizer, sentence scores, fact logic, redundancy
penalty, and output limits are reused without training.  The only experimental
change is whether the 450-character constraint is applied after selecting up
to eight sentences (legacy) or as a complete-sentence feasibility constraint
during greedy selection (budget_greedy).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_sse_cail import fact_metrics, get_tokenizer, rouge_l, rouge_n, sentence_metrics
from run_legalfactaware_v2_cail_clean import (
    char_bigrams,
    jaccard,
    minmax,
    predict_probabilities,
    read_jsonl,
    reference,
    select_indices,
    sentences,
    source_fact_items,
)


EXPERIMENT = "LegalFactAware-v2.1-budget-aware-decoder"
EXPECTED_MODEL_SHA256 = "22689296b98f398bba1f4b591db3e02e74703284da1ffdbeed1ea94cba334fac"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def row_id(row: dict) -> str:
    return str(row.get("id") or row.get("case_id") or "")


def source_labels(row: dict) -> list[int]:
    return [
        int(item.get("label", 0)) if isinstance(item, dict) else 0
        for item in (row.get("text") or [])
    ]


def candidate_score(
    index: int,
    selected: list[int],
    normalized: list[float],
    bigram_sets: list[set[tuple[str, str]]],
    fact_sets: list[set[tuple[str, str]]],
    covered: set[tuple[str, str]],
    *,
    fact_weight: float,
    redundancy_weight: float,
) -> tuple[float, float, float]:
    redundancy = max(
        (jaccard(bigram_sets[index], bigram_sets[other]) for other in selected),
        default=0.0,
    )
    marginal_facts = len(fact_sets[index] - covered)
    fact_gain = marginal_facts / max(len(fact_sets[index]), 1)
    score = normalized[index] + fact_weight * fact_gain - redundancy_weight * redundancy
    return score, redundancy, fact_gain


def select_budget_greedy(
    document_sentences: list[str],
    probabilities: list[float],
    *,
    fact_weight: float,
    redundancy_weight: float,
    max_sentences: int,
    max_chars: int,
) -> tuple[list[int], list[dict]]:
    """Run the frozen greedy objective with one added budget-feasibility test."""
    normalized = minmax(probabilities)
    bigram_sets = [char_bigrams(sentence) for sentence in document_sentences]
    fact_sets = [source_fact_items(sentence) for sentence in document_sentences]
    selected: list[int] = []
    covered: set[tuple[str, str]] = set()
    remaining = set(range(len(document_sentences)))
    skipped_events: list[dict] = []

    while remaining and len(selected) < max_sentences:
        best_index, best_score = None, -float("inf")
        for index in sorted(remaining):
            ordered = sorted(selected + [index])
            candidate_text = "".join(document_sentences[item] for item in ordered)
            score, redundancy, fact_gain = candidate_score(
                index,
                selected,
                normalized,
                bigram_sets,
                fact_sets,
                covered,
                fact_weight=fact_weight,
                redundancy_weight=redundancy_weight,
            )
            if len(candidate_text) > max_chars:
                current_text = "".join(
                    document_sentences[item] for item in sorted(selected)
                )
                skipped_events.append(
                    {
                        "selection_step": len(selected) + 1,
                        "index": index,
                        "length": len(document_sentences[index]),
                        "score": score,
                        "normalized_probability": normalized[index],
                        "redundancy": redundancy,
                        "fact_gain": fact_gain,
                        "remaining_chars": max_chars - len(current_text),
                    }
                )
                continue
            # Strict greater-than preserves the frozen sorted-index tie rule.
            if score > best_score:
                best_index, best_score = index, score

        if best_index is None:
            break
        selected.append(best_index)
        covered.update(fact_sets[best_index])
        remaining.remove(best_index)

    selected = sorted(selected)
    summary = "".join(document_sentences[index] for index in selected)
    assert len(selected) <= max_sentences
    assert len(summary) <= max_chars
    return selected, skipped_events


def prediction_row(
    source_row: dict,
    selected: list[int],
    *,
    decoder: str,
    fact_weight: float,
    redundancy_weight: float,
    max_chars: int,
    skipped_due_to_budget: list[dict] | None = None,
) -> dict:
    document_sentences = sentences(source_row)
    full_text = "".join(document_sentences[index] for index in selected)
    if decoder == "legacy":
        generated = full_text[:max_chars]
        hard_truncated = len(full_text) > max_chars
    elif decoder == "budget_greedy":
        generated = full_text
        hard_truncated = False
        assert len(generated) <= max_chars
    else:
        raise ValueError(decoder)
    return {
        "id": row_id(source_row),
        "generated_summary": generated,
        "part_indices": [selected],
        "selected_indices": selected,
        "reference_summary": reference(source_row),
        "document": "".join(document_sentences),
        "decoder": decoder,
        "fact_weight": fact_weight,
        "redundancy_weight": redundancy_weight,
        "selected_sentence_count": len(selected),
        "generated_characters": len(generated),
        "untruncated_selected_characters": len(full_text),
        "hard_truncated": hard_truncated,
        "skipped_due_to_budget": skipped_due_to_budget or [],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def distribution(values: list[float]) -> dict:
    return {
        "mean": mean(values),
        "median": median(values),
        "p25": percentile(values, 25),
        "p75": percentile(values, 75),
        "min": min(values),
        "max": max(values),
    }


def evaluate_predictions(
    predictions: list[dict],
    gold_rows: list[dict],
    *,
    rouge_unit: str,
    max_chars: int,
    max_sentences: int,
) -> tuple[dict, list[dict]]:
    tokenizer = get_tokenizer(rouge_unit)
    details: list[dict] = []
    for prediction, gold in zip(predictions, gold_rows):
        if row_id(prediction) != row_id(gold):
            raise RuntimeError(f"prediction/gold order mismatch: {row_id(prediction)} != {row_id(gold)}")
        text = prediction["generated_summary"]
        selected = {int(index) for index in prediction["selected_indices"]}
        labels = source_labels(gold)
        fact_recall, source_support = fact_metrics(
            reference(gold), text, "".join(sentences(gold))
        )
        sentence_precision, sentence_recall, sentence_f1 = sentence_metrics(selected, labels)
        details.append(
            {
                "id": row_id(gold),
                "rouge_1_f1": rouge_n(reference(gold), text, 1, tokenizer),
                "rouge_2_f1": rouge_n(reference(gold), text, 2, tokenizer),
                "rouge_l_f1": rouge_l(reference(gold), text, tokenizer),
                "fact_recall": fact_recall,
                "source_fact_support": source_support,
                "sentence_precision": sentence_precision,
                "sentence_recall": sentence_recall,
                "sentence_f1": sentence_f1,
                "characters": len(text),
                "selected_sentences": len(selected),
                "hard_truncated": int(bool(prediction["hard_truncated"])),
                "budget_violation": int(len(text) > max_chars or len(selected) > max_sentences),
            }
        )
    metrics = {
        "cases": len(details),
        "rouge_unit": rouge_unit,
        **{
            metric: mean(row[metric] for row in details)
            for metric in (
                "rouge_1_f1",
                "rouge_2_f1",
                "rouge_l_f1",
                "fact_recall",
                "source_fact_support",
                "sentence_precision",
                "sentence_recall",
                "sentence_f1",
            )
        },
        "budget_violations": sum(row["budget_violation"] for row in details),
    }
    return metrics, details


def decoder_diagnostics(predictions: list[dict], *, max_chars: int) -> dict:
    lengths = [len(row["generated_summary"]) for row in predictions]
    sentence_counts = [len(row["selected_indices"]) for row in predictions]
    utilizations = [length / max_chars for length in lengths]
    skipped = [event for row in predictions for event in row.get("skipped_due_to_budget", [])]
    count_by_sentences = Counter(sentence_counts)
    return {
        "output_characters": distribution(lengths),
        "selected_sentences": distribution(sentence_counts),
        "sentence_count_histogram": {
            str(count): count_by_sentences.get(count, 0)
            for count in range(0, 9)
        },
        "hard_truncation_count": sum(bool(row["hard_truncated"]) for row in predictions),
        "budget_utilization": distribution(utilizations),
        "skipped_due_to_budget_event_count": len(skipped),
        "cases_with_budget_skips": sum(bool(row.get("skipped_due_to_budget")) for row in predictions),
        "skipped_score": distribution([event["score"] for event in skipped]) if skipped else None,
        "skipped_length": distribution([event["length"] for event in skipped]) if skipped else None,
    }


def compare_selections(legacy: list[dict], budget: list[dict]) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    for left, right in zip(legacy, budget):
        if left["id"] != right["id"]:
            raise RuntimeError("legacy/budget prediction order mismatch")
        left_set = set(left["selected_indices"])
        right_set = set(right["selected_indices"])
        intersection = sorted(left_set & right_set)
        union = sorted(left_set | right_set)
        rows.append(
            {
                "id": left["id"],
                "legacy_selected_indices": left["selected_indices"],
                "budget_selected_indices": right["selected_indices"],
                "intersection": intersection,
                "union": union,
                "jaccard": len(intersection) / len(union) if union else 1.0,
                "added_indices": sorted(right_set - left_set),
                "removed_indices": sorted(left_set - right_set),
                "legacy_chars": len(left["generated_summary"]),
                "budget_chars": len(right["generated_summary"]),
                "legacy_hard_truncated": left["hard_truncated"],
            }
        )
    report = {
        "cases": len(rows),
        "exact_same_selection_count": sum(
            row["legacy_selected_indices"] == row["budget_selected_indices"] for row in rows
        ),
        "mean_selection_jaccard": mean(row["jaccard"] for row in rows),
        "mean_added_sentences": mean(len(row["added_indices"]) for row in rows),
        "mean_removed_sentences": mean(len(row["removed_indices"]) for row in rows),
    }
    return report, rows


def write_comparison_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def verify_frozen_legacy(rebuilt: list[dict], path: Path) -> dict:
    frozen = read_jsonl(path)
    if len(frozen) != len(rebuilt):
        raise RuntimeError(f"frozen legacy case mismatch: {len(frozen)} != {len(rebuilt)}")
    summary_mismatches = 0
    selection_mismatches = 0
    id_mismatches = 0
    for expected, actual in zip(frozen, rebuilt):
        id_mismatches += row_id(expected) != row_id(actual)
        summary_mismatches += str(expected.get("generated_summary") or "") != actual["generated_summary"]
        expected_selected = [int(index) for index in expected.get("selected_indices") or []]
        selection_mismatches += expected_selected != actual["selected_indices"]
    result = {
        "path": str(path),
        "sha256": sha256(path),
        "cases": len(frozen),
        "id_mismatches": id_mismatches,
        "summary_mismatches": summary_mismatches,
        "selection_mismatches": selection_mismatches,
    }
    if id_mismatches or summary_mismatches or selection_mismatches:
        raise RuntimeError(f"rebuilt legacy does not match frozen predictions: {result}")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frozen-dev-legacy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=942)
    parser.add_argument("--fact-weight", type=float, default=0.0)
    parser.add_argument("--redundancy-weight", type=float, default=0.2)
    parser.add_argument("--max-sentences", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=450)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.fact_weight, args.redundancy_weight, args.max_sentences, args.max_chars) != (0.0, 0.2, 8, 450):
        raise RuntimeError("This preregistered dev experiment permits only the frozen quality settings")
    model_hash = sha256(args.model)
    if model_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"unexpected frozen model SHA256: {model_hash}")

    dev_rows = read_jsonl(args.dev)
    if len(dev_rows) != args.expected_cases:
        raise RuntimeError(f"expected {args.expected_cases} dev rows, found {len(dev_rows)}")
    with args.model.open("rb") as handle:
        model = pickle.load(handle)

    legacy_predictions: list[dict] = []
    budget_predictions: list[dict] = []
    for document_number, source_row in enumerate(dev_rows, 1):
        document_sentences = sentences(source_row)
        probabilities = predict_probabilities(model, document_sentences)
        legacy_selected = select_indices(
            document_sentences,
            probabilities,
            fact_weight=args.fact_weight,
            redundancy_weight=args.redundancy_weight,
            max_sentences=args.max_sentences,
        )
        budget_selected, skipped = select_budget_greedy(
            document_sentences,
            probabilities,
            fact_weight=args.fact_weight,
            redundancy_weight=args.redundancy_weight,
            max_sentences=args.max_sentences,
            max_chars=args.max_chars,
        )
        legacy_predictions.append(
            prediction_row(
                source_row,
                legacy_selected,
                decoder="legacy",
                fact_weight=args.fact_weight,
                redundancy_weight=args.redundancy_weight,
                max_chars=args.max_chars,
            )
        )
        budget_predictions.append(
            prediction_row(
                source_row,
                budget_selected,
                decoder="budget_greedy",
                fact_weight=args.fact_weight,
                redundancy_weight=args.redundancy_weight,
                max_chars=args.max_chars,
                skipped_due_to_budget=skipped,
            )
        )
        if document_number % 100 == 0 or document_number == len(dev_rows):
            print(f"generated {document_number}/{len(dev_rows)} dev cases", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = args.output_dir / "dev_legacy_predictions.jsonl"
    budget_path = args.output_dir / "dev_budget_predictions.jsonl"
    write_jsonl(legacy_path, legacy_predictions)
    write_jsonl(budget_path, budget_predictions)

    frozen_verification = verify_frozen_legacy(legacy_predictions, args.frozen_dev_legacy)
    selection_summary, comparison_rows = compare_selections(legacy_predictions, budget_predictions)
    write_comparison_csv(args.output_dir / "dev_case_comparison.csv", comparison_rows)

    metrics: dict[str, dict] = {"legacy": {}, "budget_greedy": {}}
    per_sample: dict[str, dict] = {}
    for unit in ("lawa", "character"):
        legacy_metrics, legacy_details = evaluate_predictions(
            legacy_predictions,
            dev_rows,
            rouge_unit=unit,
            max_chars=args.max_chars,
            max_sentences=args.max_sentences,
        )
        budget_metrics, budget_details = evaluate_predictions(
            budget_predictions,
            dev_rows,
            rouge_unit=unit,
            max_chars=args.max_chars,
            max_sentences=args.max_sentences,
        )
        metrics["legacy"][unit] = legacy_metrics
        metrics["budget_greedy"][unit] = budget_metrics
        per_sample[f"legacy_{unit}"] = legacy_details
        per_sample[f"budget_greedy_{unit}"] = budget_details
        write_jsonl(args.output_dir / f"dev_legacy_{unit}_per_sample.jsonl", legacy_details)
        write_jsonl(args.output_dir / f"dev_budget_{unit}_per_sample.jsonl", budget_details)

    diagnostics = {
        "legacy": decoder_diagnostics(legacy_predictions, max_chars=args.max_chars),
        "budget_greedy": decoder_diagnostics(budget_predictions, max_chars=args.max_chars),
        "selection_comparison": selection_summary,
        "frozen_legacy_verification": frozen_verification,
    }
    metrics["legacy"]["decoder_diagnostics"] = diagnostics["legacy"]
    metrics["budget_greedy"]["decoder_diagnostics"] = diagnostics["budget_greedy"]
    save_json(args.output_dir / "dev_legacy_metrics.json", metrics["legacy"])
    save_json(args.output_dir / "dev_budget_metrics.json", metrics["budget_greedy"])
    save_json(args.output_dir / "decoder_diagnostics.json", diagnostics)

    config = {
        "experiment": EXPERIMENT,
        "decoder_a": "legacy",
        "decoder_b": "budget_greedy",
        "model_path": str(args.model),
        "model_sha256": model_hash,
        "dev_path": str(args.dev),
        "dev_sha256": sha256(args.dev),
        "training_changed": False,
        "features_changed": False,
        "classifier_changed": False,
        "fact_schema_changed": False,
        "redundancy_changed": False,
        "decoder_changed": True,
        "test_read_or_generated": False,
        "max_chars": args.max_chars,
        "max_sentences": args.max_sentences,
        "fact_weight": args.fact_weight,
        "redundancy_weight": args.redundancy_weight,
    }
    save_json(args.output_dir / "config.json", config)
    manifest = {
        **config,
        "status": "complete",
        "dev_cases": len(dev_rows),
        "outputs": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(args.output_dir.iterdir())
            if path.is_file()
        },
        "primary_delta_budget_minus_legacy": {
            metric: metrics["budget_greedy"]["lawa"][metric] - metrics["legacy"]["lawa"][metric]
            for metric in ("rouge_1_f1", "rouge_2_f1", "rouge_l_f1")
        },
    }
    save_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
