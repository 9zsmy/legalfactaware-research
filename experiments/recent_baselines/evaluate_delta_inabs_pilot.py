#!/usr/bin/env python3
"""Summarize runtime and length feasibility for the IN-Abs DELTA pilot."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.predictions)
    if not rows:
        raise RuntimeError("no predictions")
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    word_errors = [int(row["word_error"]) for row in rows]
    summary = {
        "cases": len(rows),
        "mean_seconds": statistics.mean(elapsed),
        "median_seconds": statistics.median(elapsed),
        "total_gpu_hours": sum(elapsed) / 3600,
        "projected_703_gpu_hours_at_observed_mean": 703 * statistics.mean(elapsed) / 3600,
        "mean_words": statistics.mean(int(row["words"]) for row in rows),
        "median_words": statistics.median(int(row["words"]) for row in rows),
        "mean_absolute_word_error": statistics.mean(word_errors),
        "median_absolute_word_error": statistics.median(word_errors),
        "word_budget_violations": sum(bool(row["word_budget_violation"]) for row in rows),
        "unfinished_outputs": sum(int(row["remaining_masks"]) > 0 for row in rows),
        "source_truncated": sum(bool(row["source_truncated"]) for row in rows),
        "mean_peak_allocated_gib": statistics.mean(
            float(row["peak_allocated_gib"]) for row in rows
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

