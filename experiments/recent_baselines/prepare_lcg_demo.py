#!/usr/bin/env python3
"""Freeze a non-overlapping CAIL training example for the LCG one-shot prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import read_jsonl, write_jsonl


def source_text(value: object) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--evaluation-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    evaluation_ids = {str(row["id"]) for row in read_jsonl(args.evaluation_jsonl)}
    with args.train_jsonl.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    row = rows[args.index]
    case_id = str(row["id"])
    if case_id in evaluation_ids:
        raise RuntimeError(f"demonstration id overlaps evaluation set: {case_id}")
    summary = str(row.get("summary") or row.get("reference") or "")
    source = source_text(row.get("text", row.get("source")))
    if not source or not summary:
        raise RuntimeError("demonstration requires non-empty source and summary")
    write_jsonl(args.output, [{
        "id": case_id,
        "source": source,
        "summary": summary,
        "summary_characters": len(summary),
        "selection": f"row_index_{args.index}",
    }])
    print(f"wrote demo id={case_id} source_chars={len(source)} summary_chars={len(summary)}")


if __name__ == "__main__":
    main()

