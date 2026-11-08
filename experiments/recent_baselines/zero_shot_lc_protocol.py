#!/usr/bin/env python3
"""Dependency-light helpers for the NAACL 2025 zero-shot LCS adaptation."""

from __future__ import annotations


VALID_OBJECTIVES = ("exact", "upper_bound")


def count_characters(text: str) -> int:
    """Use the paper and frozen CAIL convention: Python ``len(text)``."""
    return len(text or "")


def selection_key(length: int, target: int, objective: str) -> tuple[int, int]:
    """Return the deterministic key used to rank sampled summaries.

    The exact objective is Equation (7-8) in Retkowski and Waibel (2025).
    The one-sided objective is our CAIL adaptation: every compliant output is
    preferred to every over-budget output, then the closest output to the
    budget is selected. Generation order resolves remaining ties.
    """
    length, target = int(length), int(target)
    if objective == "exact":
        return (0, abs(length - target))
    if objective == "upper_bound":
        return (0, target - length) if length <= target else (1, length - target)
    raise ValueError(f"unknown objective: {objective}")


def select_candidate(lengths: list[int], target: int, objective: str) -> int:
    """Select a sample without consulting a reference summary or quality metric."""
    if not lengths:
        raise ValueError("at least one candidate is required")
    return min(
        range(len(lengths)),
        key=lambda index: (*selection_key(lengths[index], target, objective), index),
    )


def is_compliant(
    length: int,
    target: int,
    objective: str,
    tolerance: float = 0.10,
) -> bool:
    """Check the paper's tolerance or the task's hard upper bound."""
    if objective == "exact":
        return abs(int(length) - int(target)) <= float(tolerance) * int(target)
    if objective == "upper_bound":
        return int(length) <= int(target)
    raise ValueError(f"unknown objective: {objective}")


def revision_feedback(length: int, target: int, objective: str) -> str:
    """Create the measurable feedback used by the automated-revision prompt."""
    length, target = int(length), int(target)
    if length == target:
        relation = "正好达到"
    elif length > target:
        relation = f"比目标多{length - target}个"
    else:
        relation = f"比目标少{target - length}个"
    if objective == "exact":
        request = f"请将摘要修改得更接近{target}个字符"
    elif objective == "upper_bound":
        request = f"请将摘要压缩到不超过{target}个字符，并尽量保留关键事实"
    else:
        raise ValueError(f"unknown objective: {objective}")
    return f"当前摘要有{length}个字符，{relation}。{request}。"

