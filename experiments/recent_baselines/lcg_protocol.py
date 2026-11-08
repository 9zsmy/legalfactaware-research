#!/usr/bin/env python3
"""Dependency-light helpers for the ACL 2025 black-box LCG adaptation."""

from __future__ import annotations

import json
import math
import re


def upper_budget_distance(length: int, budget: int) -> int:
    """Distance to the task constraint [0, budget]."""
    return max(0, int(length) - int(budget))


def exact_target_distance(length: int, target: int) -> int:
    """Absolute distance to an exact target length."""
    return abs(int(length) - int(target))


def objective_distance(length: int, target: int, objective: str) -> int:
    """Return the distance used by the selected length-control objective."""
    if objective == "upper_bound":
        return upper_budget_distance(length, target)
    if objective == "exact":
        return exact_target_distance(length, target)
    raise ValueError(f"unknown length objective: {objective}")


def objective_satisfied(length: int, target: int, objective: str) -> bool:
    """Whether a length satisfies the selected exact or interval objective."""
    return objective_distance(length, target, objective) == 0


def select_final_chain(lengths: list[int], target: int, objective: str) -> int:
    """Select the first successful chain, otherwise the closest chain.

    The paper specifies parallel initial states but does not publish code for
    resolving final-chain ties.  This deterministic rule prioritizes the
    stated length objective and uses the generation order as the tie-breaker.
    """
    if not lengths:
        raise ValueError("at least one chain is required")
    distances = [objective_distance(length, target, objective) for length in lengths]
    return min(range(len(lengths)), key=lambda index: (distances[index], index))


def upper_budget_score(length: int, budget: int) -> float:
    """Equation (8) adapted from word length to Python-character length."""
    distance = upper_budget_distance(length, budget)
    return math.inf if distance == 0 else 1.0 / distance


def exact_target_score(length: int, target: int) -> float:
    """Inverse absolute distance used by the paper-style exact-length objective."""
    distance = exact_target_distance(length, target)
    return math.inf if distance == 0 else 1.0 / distance


def acceptance_probability(
    current_length: int,
    candidate_length: int,
    quality_ratio: float,
    budget: int,
    objective: str = "upper_bound",
) -> float:
    """Equation (12) upper bound for the chosen length objective.

    A candidate satisfying the objective terminates successfully. A current state
    satisfying it should have terminated before proposing.
    """
    current_distance = objective_distance(current_length, budget, objective)
    candidate_distance = objective_distance(candidate_length, budget, objective)
    if candidate_distance == 0:
        return 1.0
    if current_distance == 0:
        return 0.0
    ratio = max(float(quality_ratio), 0.0)
    length_ratio = current_distance / candidate_distance
    return min(1.0, length_ratio * ratio)


def extract_score_pair(text: str) -> tuple[float, float]:
    """Extract Summary 1/2 totals from JSON or the paper-style score format."""
    raw = (text or "").strip()
    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S | re.I)
    if fenced:
        candidates.insert(0, fenced.group(1))
    braced = re.search(r"\{.*\}", raw, re.S)
    if braced:
        candidates.append(braced.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        normalized = {str(key).lower().replace(" ", "_"): val for key, val in value.items()}
        left = normalized.get("summary_1_total", normalized.get("summary1_total"))
        right = normalized.get("summary_2_total", normalized.get("summary2_total"))
        if left is not None and right is not None and float(right) > 0:
            return float(left), float(right)

    patterns = (
        r"Summary\s*1[^\n\r]{0,80}?(?:Overall\s*Score|总分)[^\d]*(\d+(?:\.\d+)?)",
        r"摘要\s*1[^\n\r]{0,80}?(?:Overall\s*Score|总分)[^\d]*(\d+(?:\.\d+)?)",
    )
    reverse_patterns = tuple(pattern.replace("1", "2", 1) for pattern in patterns)
    left_match = next((m for p in patterns if (m := re.search(p, raw, re.I))), None)
    right_match = next((m for p in reverse_patterns if (m := re.search(p, raw, re.I))), None)
    if left_match and right_match and float(right_match.group(1)) > 0:
        return float(left_match.group(1)), float(right_match.group(1))
    raise ValueError("could not extract pairwise summary scores")

