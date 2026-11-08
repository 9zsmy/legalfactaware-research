#!/usr/bin/env python3
"""Shared, dependency-light protocol helpers for recent CAIL baselines."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable


FACT_PATTERNS = {
    "amount": re.compile(r"(?:人民币)?\d+(?:\.\d+)?(?:万)?元"),
    "date": re.compile(r"\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?"),
    "law": re.compile(r"《[^》]{2,50}》|第[一二三四五六七八九十百千万零〇\d]+条"),
    "decision": re.compile(r"驳回|支持|不予支持|支付|赔偿|返还|解除|确认|承担|连带责任|优先受偿"),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def visible_characters(text: str) -> list[str]:
    return [char for char in (text or "") if not char.isspace()]


def character_count(text: str) -> int:
    """Match the locked CAIL convention: Python string length, including spaces."""
    return len(text or "")


def truncate_source(text: str, max_chars: int, mode: str = "head_tail") -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if mode == "head":
        return text[:max_chars]
    if mode != "head_tail":
        raise ValueError(f"unsupported source truncation mode: {mode}")
    # Chinese judgments commonly place claims near the front and the ruling at
    # the end. Preserve both without silently increasing the context budget.
    left = max_chars // 2
    right = max_chars - left
    return text[:left] + text[-right:]


def _ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[index:index + order]) for index in range(len(tokens) - order + 1))


def rouge_n(reference: str, prediction: str, order: int) -> float:
    left, right = visible_characters(reference), visible_characters(prediction)
    left_counts, right_counts = _ngrams(left, order), _ngrams(right, order)
    if not left_counts or not right_counts:
        return 0.0
    overlap = sum((left_counts & right_counts).values())
    precision = overlap / sum(right_counts.values())
    recall = overlap / sum(left_counts.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def rouge_l(reference: str, prediction: str) -> float:
    left, right = visible_characters(reference), visible_characters(prediction)
    if not left or not right:
        return 0.0
    # Hunt-Szymanski LCS avoids a large O(mn) matrix for long Chinese strings.
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(right):
        positions.setdefault(token, []).append(index)
    thresholds: list[int] = []
    import bisect
    for token in left:
        for position in reversed(positions.get(token, [])):
            insertion = bisect.bisect_left(thresholds, position)
            if insertion == len(thresholds):
                thresholds.append(position)
            else:
                thresholds[insertion] = position
    overlap = len(thresholds)
    precision, recall = overlap / len(right), overlap / len(left)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def expression_sets(text: str) -> dict[str, set[str]]:
    return {name: set(pattern.findall(text or "")) for name, pattern in FACT_PATTERNS.items()}


def finite_mean(values: Iterable[float]) -> float:
    retained = [value for value in values if math.isfinite(value)]
    return mean(retained) if retained else float("nan")

