#!/usr/bin/env python3
"""Reorder released LetSum-selected tokens in source order and apply optional L cap."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--judgements", type=Path, required=True)
parser.add_argument("--stats", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()

variants = ["grouped_all", "source_all", "grouped_refcap", "source_refcap"]
for variant in variants:
    (args.output_dir / variant / "predictions").mkdir(parents=True, exist_ok=True)

budgets = {
    fields[0]: int(fields[2])
    for fields in (line.split("\t") for line in args.stats.read_text().splitlines())
}
rows = []
for prediction_path in sorted(args.predictions.glob("*.txt")):
    name = prediction_path.name
    grouped = prediction_path.read_text(errors="ignore").split()
    remaining = Counter(grouped)
    source_order = []
    for token in args.judgements.joinpath(name).read_text(errors="ignore").split():
        if remaining[token] > 0:
            source_order.append(token)
            remaining[token] -= 1
    leftovers = []
    for token in grouped:
        if remaining[token] > 0:
            leftovers.append(token)
            remaining[token] -= 1
    source_order.extend(leftovers)
    budget = budgets[name]
    outputs = {
        "grouped_all": grouped,
        "source_all": source_order,
        "grouped_refcap": grouped[:budget],
        "source_refcap": source_order[:budget],
    }
    for variant, tokens in outputs.items():
        (args.output_dir / variant / "predictions" / name).write_text(" ".join(tokens), encoding="utf-8")
    rows.append(
        {
            "file": name,
            "budget": budget,
            "selected_tokens": len(grouped),
            "source_matched_tokens": len(source_order) - len(leftovers),
            "unmatched_tokens": len(leftovers),
        }
    )

(args.output_dir / "documents.jsonl").write_text(
    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
)
(args.output_dir / "manifest.json").write_text(
    json.dumps(
        {
            "experiment": "LetSum output-order and reference-budget audit",
            "documents": len(rows),
            "variants": variants,
            "mean_unmatched_tokens": sum(row["unmatched_tokens"] for row in rows) / len(rows),
        },
        indent=2,
    )
    + "\n"
)
