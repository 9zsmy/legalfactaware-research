#!/usr/bin/env python3
"""Audit Zero-Shot LCS outputs without using reference-summary quality scores."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from protocol import read_jsonl
from zero_shot_lc_protocol import is_compliant, select_candidate


ROLE_LEAK = re.compile(r"(?:<\|im_(?:start|end)\|>|^(?:user|assistant|system)\s*$)", re.M | re.I)


def keyed(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    values = {str(row["id"]): row for row in rows}
    if len(values) != len(rows):
        raise AssertionError(f"duplicate IDs in {path}")
    return values


def audit_variant(
    expected_ids: list[str],
    sample_path: Path,
    final_path: Path,
    trace_path: Path,
    objective: str,
    target: int,
    tolerance: float,
) -> dict:
    samples, finals, traces = keyed(sample_path), keyed(final_path), keyed(trace_path)
    expected = set(expected_ids)
    for name, rows in (("sample", samples), ("final", finals), ("trace", traces)):
        if set(rows) != expected:
            raise AssertionError(
                f"{objective} {name} ID mismatch: "
                f"missing={sorted(expected-set(rows))}, extra={sorted(set(rows)-expected)}"
            )

    empty, role_leaks, selection_errors, trace_errors = [], [], [], []
    compliant = budget_violations = revisions = 0
    elapsed = 0.0
    lengths = []
    for case_id in expected_ids:
        sample, final, trace = samples[case_id], finals[case_id], traces[case_id]
        candidates = [str(value) for value in sample["sample_candidates"]]
        candidate_lengths = [len(value) for value in candidates]
        expected_index = select_candidate(candidate_lengths, target, objective)
        if sample["selected_index"] != expected_index:
            selection_errors.append(case_id)
        if sample["prediction"] != candidates[expected_index]:
            selection_errors.append(case_id + ":prediction")
        prediction = str(final["prediction"])
        if not prediction:
            empty.append(case_id)
        if ROLE_LEAK.search(prediction):
            role_leaks.append(case_id)
        if len(prediction) != int(final["characters"]):
            trace_errors.append(case_id + ":length")
        if trace["final_prediction"] != prediction:
            trace_errors.append(case_id + ":final")
        if trace["sample_candidates"] != candidates:
            trace_errors.append(case_id + ":samples")
        if final.get("posthoc_truncation") is not False:
            trace_errors.append(case_id + ":truncation")
        observed_compliance = is_compliant(
            len(prediction), target, objective, tolerance
        )
        if bool(final["objective_compliant"]) != observed_compliance:
            trace_errors.append(case_id + ":compliance")
        compliant += int(observed_compliance)
        budget_violations += int(len(prediction) > target)
        revisions += int(final["revision_count"])
        elapsed += float(final["elapsed_seconds"])
        lengths.append(len(prediction))

    return {
        "objective": objective,
        "documents": len(expected_ids),
        "empty_outputs": empty,
        "role_leaks": role_leaks,
        "selection_errors": selection_errors,
        "trace_errors": trace_errors,
        "objective_compliant_count": compliant,
        "budget_violation_count": budget_violations,
        "total_revisions": revisions,
        "total_elapsed_seconds_excluding_model_load": elapsed,
        "final_lengths": lengths,
        "passed_integrity_checks": not any(
            (empty, role_leaks, selection_errors, trace_errors)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--target", type=int, default=450)
    parser.add_argument("--tolerance", type=float, default=0.10)
    args = parser.parse_args()

    expected_ids = [str(row["id"]) for row in read_jsonl(args.dataset)[: args.limit]]
    result = {
        "dataset": str(args.dataset),
        "expected_ids": expected_ids,
        "target": args.target,
        "variants": [],
    }
    for objective, suffix in (("exact", "exact"), ("upper_bound", "upper")):
        result["variants"].append(
            audit_variant(
                expected_ids,
                args.results_dir / f"zero_shot_lc_sf3_{suffix}_smoke5.jsonl",
                args.results_dir / f"zero_shot_lc_sf3_ar5_{suffix}_smoke5.jsonl",
                args.results_dir
                / f"zero_shot_lc_sf3_ar5_{suffix}_traces_smoke5.jsonl",
                objective,
                args.target,
                args.tolerance,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

