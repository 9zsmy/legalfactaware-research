#!/usr/bin/env python3
"""Evaluate a frozen SENDE checkpoint at a fixed threshold for sensitivity analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sende-code", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--feature-prefix", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.sende_code))
    from evaluate_cail import load_probabilities, score_rows  # noqa: PLC0415

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, probabilities = load_probabilities(
        args.records, args.feature_prefix, args.checkpoint
    )
    report = score_rows(
        records,
        probabilities,
        args.threshold,
        args.output_dir / "test_predictions.jsonl",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
