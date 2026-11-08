#!/usr/bin/env python3
"""Evaluate summaries with overlap, extractiveness, and source-support metrics.

The evaluator is deliberately dependency-light so that it can run on both the
local workstation and the experiment server.  It accepts CSV or JSONL records
and supports English word tokenization and Chinese character tokenization.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*|\d+(?:[.,:/-]\d+)*")
CJK_RE = re.compile(r"[\u3400-\u9fff]")
SENTENCE_RE = re.compile(r"(?<=[。！？!?])|(?<=[.!?])\s+|\n+")
FACT_RE = re.compile(
    r"(?:US\$|HK\$|RMB|USD|EUR|GBP|[¥￥$€£])\s?\d[\d,.]*"
    r"|\b\d{1,4}(?:[-/.]\d{1,2}){1,2}\b"
    r"|\b\d+(?:[.,]\d+)?(?:\s?(?:%|percent|years?|months?|days?|dollars?|yuan))?\b"
    r"|第[一二三四五六七八九十百千万零〇两\d]+条",
    re.I,
)
CITATION_RE = re.compile(
    r"\b\d+\s+U\.?\s*S\.?\s+\d+\b"
    r"|\b\d+\s+F\.?\s*(?:2d|3d|4th|Supp\.?\s*\d*)\s+\d+\b"
    r"|\b(?:Article|Section|Rule|Amendment|§)\s*\d+(?:\([^)]+\))*"
    r"|《[^》]{2,40}》(?:第[一二三四五六七八九十百千万零〇两\d]+条)?",
    re.I,
)


def read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() in {".jsonl", ".jsonlines"}:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    csv.field_size_limit(2**31 - 1)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def tokenize(text: str, mode: str = "auto") -> list[str]:
    text = text or ""
    if mode == "auto":
        visible = [char for char in text if not char.isspace()]
        cjk_ratio = sum(bool(CJK_RE.match(char)) for char in visible) / max(len(visible), 1)
        mode = "char" if cjk_ratio >= 0.20 else "word"
    if mode == "char":
        return [char.lower() for char in text if CJK_RE.match(char) or char.isalnum()]
    return [token.lower() for token in WORD_RE.findall(text)]


def ngrams(items: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(items) < n:
        return Counter()
    return Counter(tuple(items[index:index + n]) for index in range(len(items) - n + 1))


def overlap_prf(reference: list[str], prediction: list[str], n: int) -> tuple[float, float, float]:
    ref_counts, pred_counts = ngrams(reference, n), ngrams(prediction, n)
    overlap = sum((ref_counts & pred_counts).values())
    precision = overlap / sum(pred_counts.values()) if pred_counts else 0.0
    recall = overlap / sum(ref_counts.values()) if ref_counts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def lcs_length(left: list[str], right: list[str]) -> int:
    """Exact LCS length using Hunt-Szymanski sparse match positions."""
    if not left or not right:
        return 0
    positions: defaultdict[str, list[int]] = defaultdict(list)
    for index, token in enumerate(right):
        positions[token].append(index)
    thresholds: list[int] = []
    for token in left:
        # Reversed positions ensure one token cannot extend the same sequence
        # more than once during this iteration.
        for position in reversed(positions.get(token, [])):
            insertion = bisect.bisect_left(thresholds, position)
            if insertion == len(thresholds):
                thresholds.append(position)
            else:
                thresholds[insertion] = position
    return len(thresholds)


def rouge_l(reference: list[str], prediction: list[str]) -> tuple[float, float, float]:
    if not reference or not prediction:
        return 0.0, 0.0, 0.0
    length = lcs_length(reference, prediction)
    precision = length / len(prediction)
    recall = length / len(reference)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def sentence_rouge_l(reference: str, prediction: str, mode: str) -> float:
    """Best-aligned sentence ROUGE-L, weighted by reference sentence length."""
    ref_sentences = [part.strip() for part in SENTENCE_RE.split(reference or "") if part.strip()]
    pred_sentences = [part.strip() for part in SENTENCE_RE.split(prediction or "") if part.strip()]
    if not ref_sentences or not pred_sentences:
        return rouge_l(tokenize(reference, mode), tokenize(prediction, mode))[2]
    numerator = 0.0
    denominator = 0
    pred_tokens = [tokenize(sentence, mode) for sentence in pred_sentences]
    for sentence in ref_sentences:
        ref_tokens = tokenize(sentence, mode)
        if not ref_tokens:
            continue
        best = max((rouge_l(ref_tokens, candidate)[2] for candidate in pred_tokens), default=0.0)
        numerator += best * len(ref_tokens)
        denominator += len(ref_tokens)
    return numerator / denominator if denominator else 0.0


def sentence_bleu4(reference: list[str], prediction: list[str]) -> float:
    """Sentence BLEU-4 with add-one smoothing for stable paired tests."""
    if not prediction:
        return 0.0
    log_precision = 0.0
    effective_order = min(4, len(prediction))
    for order in range(1, effective_order + 1):
        ref_counts, pred_counts = ngrams(reference, order), ngrams(prediction, order)
        matches = sum((ref_counts & pred_counts).values())
        total = sum(pred_counts.values())
        log_precision += math.log((matches + 1.0) / (total + 1.0))
    log_precision /= max(effective_order, 1)
    brevity_penalty = 1.0
    if len(prediction) < len(reference):
        brevity_penalty = math.exp(1.0 - len(reference) / max(len(prediction), 1))
    return brevity_penalty * math.exp(log_precision)


def f_score(precision: float, recall: float, beta: float = 2.0) -> float:
    beta2 = beta * beta
    denominator = beta2 * precision + recall
    return (1 + beta2) * precision * recall / denominator if denominator else 0.0


def chrf_plus_plus(reference: str, prediction: str, mode: str) -> float:
    """Dependency-free chrF++ approximation (char 1-6 grams + token bigrams)."""
    ref_chars = [char.lower() for char in reference if not char.isspace()]
    pred_chars = [char.lower() for char in prediction if not char.isspace()]
    scores: list[float] = []
    for order in range(1, 7):
        precision, recall, _ = overlap_prf(ref_chars, pred_chars, order)
        if ngrams(ref_chars, order) and ngrams(pred_chars, order):
            scores.append(f_score(precision, recall, beta=2.0))
    ref_tokens, pred_tokens = tokenize(reference, mode), tokenize(prediction, mode)
    precision, recall, _ = overlap_prf(ref_tokens, pred_tokens, 2)
    if ngrams(ref_tokens, 2) and ngrams(pred_tokens, 2):
        scores.append(f_score(precision, recall, beta=2.0))
    return mean(scores) if scores else 0.0


def greedy_extract_fragments(source: list[str], summary: list[str]) -> list[int]:
    if not source or not summary:
        return []
    positions: defaultdict[str, list[int]] = defaultdict(list)
    for index, token in enumerate(source):
        positions[token].append(index)
    fragments: list[int] = []
    summary_index = 0
    while summary_index < len(summary):
        best = 0
        for source_index in positions.get(summary[summary_index], []):
            length = 0
            while (
                summary_index + length < len(summary)
                and source_index + length < len(source)
                and summary[summary_index + length] == source[source_index + length]
            ):
                length += 1
            best = max(best, length)
        if best:
            fragments.append(best)
            summary_index += best
        else:
            summary_index += 1
    return fragments


def normalize_fact(value: str) -> str:
    return re.sub(r"[\s,.]", "", value.lower())


def extracted_values(text: str, pattern: re.Pattern[str]) -> set[str]:
    return {normalize_fact(match.group(0)) for match in pattern.finditer(text or "")}


def set_support(values: set[str], source_values: set[str]) -> float:
    if not values:
        return 1.0
    return len(values & source_values) / len(values)


def set_prf(reference_values: set[str], prediction_values: set[str]) -> tuple[float, float, float]:
    if not reference_values and not prediction_values:
        return 1.0, 1.0, 1.0
    overlap = len(reference_values & prediction_values)
    precision = overlap / len(prediction_values) if prediction_values else 0.0
    recall = overlap / len(reference_values) if reference_values else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, score


def repetition_rate(tokens: list[str], order: int = 4) -> float:
    counts = ngrams(tokens, order)
    total = sum(counts.values())
    repeated = sum(max(value - 1, 0) for value in counts.values())
    return repeated / total if total else 0.0


def evaluate_row(source: str, reference: str, prediction: str, mode: str) -> dict[str, float]:
    ref_tokens = tokenize(reference, mode)
    pred_tokens = tokenize(prediction, mode)
    source_tokens = tokenize(source, mode) if source else []
    _, rouge1_recall, rouge1_f1 = overlap_prf(ref_tokens, pred_tokens, 1)
    _, rouge2_recall, rouge2_f1 = overlap_prf(ref_tokens, pred_tokens, 2)
    _, rouge_l_recall, rouge_l_f1 = rouge_l(ref_tokens, pred_tokens)

    fragments = greedy_extract_fragments(source_tokens, pred_tokens) if source_tokens else []
    fragment_coverage = sum(fragments) / len(pred_tokens) if pred_tokens else 0.0
    fragment_density = sum(length * length for length in fragments) / len(pred_tokens) if pred_tokens else 0.0
    novel_unigrams = 0.0
    novel_bigrams = 0.0
    if source_tokens and pred_tokens:
        source_unigrams, pred_unigrams = set(source_tokens), set(pred_tokens)
        novel_unigrams = len(pred_unigrams - source_unigrams) / len(pred_unigrams) if pred_unigrams else 0.0
        source_bigrams, pred_bigrams = set(ngrams(source_tokens, 2)), set(ngrams(pred_tokens, 2))
        novel_bigrams = len(pred_bigrams - source_bigrams) / len(pred_bigrams) if pred_bigrams else 0.0

    ref_facts = extracted_values(reference, FACT_RE)
    pred_facts = extracted_values(prediction, FACT_RE)
    source_facts = extracted_values(source, FACT_RE)
    fact_precision, fact_recall, fact_f1 = set_prf(ref_facts, pred_facts)
    pred_citations = extracted_values(prediction, CITATION_RE)
    source_citations = extracted_values(source, CITATION_RE)

    return {
        "rouge_1_f1": rouge1_f1,
        "rouge_1_recall": rouge1_recall,
        "rouge_2_f1": rouge2_f1,
        "rouge_2_recall": rouge2_recall,
        "rouge_l_f1": rouge_l_f1,
        "rouge_l_recall": rouge_l_recall,
        "sentence_rouge_l": sentence_rouge_l(reference, prediction, mode),
        "bleu_4_smoothed": sentence_bleu4(ref_tokens, pred_tokens),
        "chrf_plus_plus": chrf_plus_plus(reference, prediction, mode),
        "reference_fact_precision": fact_precision,
        "reference_fact_recall": fact_recall,
        "reference_fact_f1": fact_f1,
        "source_fact_support": set_support(pred_facts, source_facts) if source else float("nan"),
        "source_citation_support": set_support(pred_citations, source_citations) if source else float("nan"),
        "extractive_coverage": fragment_coverage if source else float("nan"),
        "extractive_density": fragment_density if source else float("nan"),
        "novel_unigram_rate": novel_unigrams if source else float("nan"),
        "novel_bigram_rate": novel_bigrams if source else float("nan"),
        "compression_ratio": len(source_tokens) / max(len(pred_tokens), 1) if source else float("nan"),
        "repetition_4gram_rate": repetition_rate(pred_tokens, 4),
        "prediction_tokens": len(pred_tokens),
        "reference_tokens": len(ref_tokens),
        "source_tokens": len(source_tokens),
    }


def bootstrap_difference(left: list[float], right: list[float], iterations: int, seed: int) -> dict[str, float]:
    differences = [a - b for a, b in zip(left, right) if not (math.isnan(a) or math.isnan(b))]
    if not differences:
        return {"mean_difference": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "bootstrap_p": float("nan")}
    rng = random.Random(seed)
    estimates = [mean(rng.choice(differences) for _ in differences) for _ in range(iterations)]
    estimates.sort()
    low = estimates[int(0.025 * iterations)]
    high = estimates[min(int(0.975 * iterations), iterations - 1)]
    p_value = 2 * min(
        sum(value <= 0 for value in estimates) / iterations,
        sum(value >= 0 for value in estimates) / iterations,
    )
    return {
        "mean_difference": mean(differences),
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_p": min(p_value, 1.0),
    }


def finite_mean(values: Iterable[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return mean(finite) if finite else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-column", default="case_id")
    parser.add_argument("--method-column", default="method")
    parser.add_argument("--source-column", default="source")
    parser.add_argument(
        "--source-lookup",
        type=Path,
        help="Optional CSV/JSONL containing one source record per case index.",
    )
    parser.add_argument("--source-lookup-column", default="source")
    parser.add_argument("--reference-column", default="reference")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--tokenization", choices=("auto", "word", "char"), default="auto")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--target", default="")
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    input_rows = read_rows(args.input)
    source_lookup_rows = read_rows(args.source_lookup) if args.source_lookup else []
    details: list[dict] = []
    for index, row in enumerate(input_rows):
        reference = str(row.get(args.reference_column, "") or "")
        prediction = str(row.get(args.prediction_column, "") or "")
        source = str(row.get(args.source_column, "") or "")
        if not source and source_lookup_rows:
            raw_id = str(row.get(args.id_column, index))
            match = re.search(r"(\d+)$", raw_id)
            lookup_index = int(match.group(1)) if match else index
            if 0 <= lookup_index < len(source_lookup_rows):
                source = str(
                    source_lookup_rows[lookup_index].get(args.source_lookup_column, "") or ""
                )
        if not reference or not prediction:
            continue
        detected_mode = args.tokenization
        if detected_mode == "auto":
            detected_mode = "char" if len(CJK_RE.findall(reference)) >= max(len(reference) * 0.2, 1) else "word"
        metrics = evaluate_row(source, reference, prediction, detected_mode)
        details.append(
            {
                "case_id": row.get(args.id_column, index),
                "method": row.get(args.method_column, "unknown"),
                "tokenization": detected_mode,
                **{name: round(value, 8) if not math.isnan(value) else "" for name, value in metrics.items()},
            }
        )

    metric_names = [name for name in details[0] if name not in {"case_id", "method", "tokenization"}] if details else []
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in details:
        grouped[str(row["method"])].append(row)
    summary: list[dict] = []
    for method, rows in sorted(grouped.items()):
        summary.append(
            {
                "method": method,
                "cases": len(rows),
                **{
                    metric: round(
                        finite_mean(float(row[metric]) for row in rows if row[metric] != ""),
                        6,
                    )
                    for metric in metric_names
                },
            }
        )

    significance: list[dict] = []
    if args.target and args.baseline:
        target_rows = {str(row["case_id"]): row for row in details if row["method"] == args.target}
        baseline_rows = {str(row["case_id"]): row for row in details if row["method"] == args.baseline}
        common_ids = sorted(set(target_rows) & set(baseline_rows))
        for metric in (
            "rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "sentence_rouge_l",
            "bleu_4_smoothed", "chrf_plus_plus", "reference_fact_f1",
            "source_fact_support", "source_citation_support", "repetition_4gram_rate",
        ):
            left = [float(target_rows[case_id][metric] or "nan") for case_id in common_ids]
            right = [float(baseline_rows[case_id][metric] or "nan") for case_id in common_ids]
            result = bootstrap_difference(left, right, args.bootstrap, args.seed)
            significance.append(
                {
                    "method_a": args.target,
                    "method_b": args.baseline,
                    "metric": metric,
                    "common_cases": len(common_ids),
                    **{key: round(value, 6) if not math.isnan(value) else "" for key, value in result.items()},
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "extended_metric_details.csv", details)
    write_csv(args.output_dir / "extended_metric_summary.csv", summary)
    if significance:
        write_csv(args.output_dir / "extended_metric_significance.csv", significance)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "input_rows": len(input_rows),
                "source_lookup": str(args.source_lookup) if args.source_lookup else "",
                "evaluated_rows": len(details),
                "methods": sorted(grouped),
                "metrics": metric_names,
                "tokenization": args.tokenization,
                "notes": [
                    "BLEU-4 is sentence-level add-one-smoothed BLEU for paired analysis.",
                    "chrF++ is a dependency-free char-1..6 plus token-bigram implementation.",
                    "Source metrics are blank when the input does not contain source text.",
                    "sentence_rouge_l is a length-weighted best-aligned sentence ROUGE-L diagnostic.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"evaluated {len(details)} rows across {len(grouped)} methods")
    print(args.output_dir / "extended_metric_summary.csv")


if __name__ == "__main__":
    main()
