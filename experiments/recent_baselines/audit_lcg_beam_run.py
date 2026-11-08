#!/usr/bin/env python3
"""Audit beam-level LCG traces without loading the generation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lcg_protocol import acceptance_probability, objective_satisfied, select_final_chain
from protocol import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--expected-beam-size", type=int, default=16)
    args = parser.parse_args()

    results = read_jsonl(args.result)
    initials = {str(row["id"]): row for row in read_jsonl(args.initial)}
    if not results:
        raise ValueError("result file is empty")

    successes = 0
    proposals = 0
    audit_rows = []
    for row in results:
        case_id = str(row["id"])
        beam_size = int(row["beam_size"])
        if beam_size != args.expected_beam_size:
            raise AssertionError(f"{case_id}: beam_size={beam_size}")
        initial_row = initials[case_id]
        initial_outputs = list(initial_row["beam_predictions"])
        final_outputs = list(row["beam_final_outputs"])
        traces = list(row["beam_traces"])
        if not (len(initial_outputs) == len(final_outputs) == len(traces) == beam_size):
            raise AssertionError(f"{case_id}: inconsistent beam arrays")

        initial_lengths = [len(value) for value in initial_outputs]
        final_lengths = [len(value) for value in final_outputs]
        if initial_lengths != list(row["beam_initial_characters"]):
            raise AssertionError(f"{case_id}: initial lengths do not match text")
        if final_lengths != list(row["beam_final_characters"]):
            raise AssertionError(f"{case_id}: final lengths do not match text")

        objective = str(row["length_objective"])
        target = int(row["target_chars"])
        selected = select_final_chain(final_lengths, target, objective)
        if selected != int(row["selected_beam"]):
            raise AssertionError(f"{case_id}: selected beam is not reproducible")
        if str(row["prediction"]) != final_outputs[selected]:
            raise AssertionError(f"{case_id}: prediction differs from selected beam")
        if int(row["characters"]) != final_lengths[selected]:
            raise AssertionError(f"{case_id}: selected length mismatch")

        for beam_index, beam_trace in enumerate(traces):
            for step in beam_trace:
                proposals += 1
                if int(step["beam"]) != beam_index:
                    raise AssertionError(f"{case_id}: trace assigned to wrong beam")
                expected_probability = acceptance_probability(
                    int(step["current_chars"]),
                    int(step["candidate_chars"]),
                    float(step["quality_ratio"]),
                    target,
                    objective=objective,
                )
                if abs(expected_probability - float(step["acceptance_probability"])) > 1e-12:
                    raise AssertionError(f"{case_id}: acceptance probability mismatch")
                expected_acceptance = float(step["uniform_draw"]) <= expected_probability
                if expected_acceptance != bool(step["accepted"]):
                    raise AssertionError(f"{case_id}: acceptance decision mismatch")

        success = objective_satisfied(final_lengths[selected], target, objective)
        successes += int(success)
        audit_rows.append({
            "id": case_id,
            "objective": objective,
            "selected_beam": selected,
            "characters": final_lengths[selected],
            "success": success,
            "proposals": sum(len(value) for value in traces),
        })

    print(json.dumps({
        "status": "PASS",
        "documents": len(results),
        "beam_size": args.expected_beam_size,
        "successful_documents": successes,
        "total_proposals": proposals,
        "rows": audit_rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

