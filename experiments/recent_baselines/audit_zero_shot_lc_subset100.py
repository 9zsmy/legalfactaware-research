#!/usr/bin/env python3
"""Integrity audit for the frozen 100-document Zero-Shot LCS run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_zero_shot_lc import audit_variant
from protocol import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_ids = [str(row["id"]) for row in read_jsonl(args.dataset)]
    variant = audit_variant(
        expected_ids,
        args.results_dir / "zero_shot_lc_sf3_upper_100.jsonl",
        args.results_dir / "zero_shot_lc_sf3_ar5_upper_100.jsonl",
        args.results_dir / "zero_shot_lc_sf3_ar5_upper_traces_100.jsonl",
        "upper_bound",
        450,
        0.10,
    )
    result = {
        "dataset": str(args.dataset),
        "expected_documents": len(expected_ids),
        "target": 450,
        "variant": variant,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

