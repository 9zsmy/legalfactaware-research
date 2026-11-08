#!/usr/bin/env python3
"""Freeze a deterministic, length-stratified CAIL development subset.

The selection uses source-length and reference-length quartiles.  Allocation is
proportional to each of the 16 joint strata, and documents inside a stratum are
ordered by SHA-256(seed:id).  No model output or evaluation metric is consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from protocol import expression_sets, read_jsonl, write_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_quartiles(rows: list[dict], field: str) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (len(str(row[field])), str(row["id"])))
    total = len(ordered)
    return {
        str(row["id"]): min(3, (rank * 4) // total)
        for rank, row in enumerate(ordered)
    }


def stable_key(seed: int, case_id: str) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def proportional_allocation(groups: dict[tuple[int, int], list[dict]], sample_size: int) -> dict[tuple[int, int], int]:
    total = sum(len(items) for items in groups.values())
    exact = {key: sample_size * len(items) / total for key, items in groups.items()}
    allocation = {key: math.floor(value) for key, value in exact.items()}
    remaining = sample_size - sum(allocation.values())
    priority = sorted(
        groups,
        key=lambda key: (exact[key] - allocation[key], len(groups[key]), key),
        reverse=True,
    )
    for key in priority[:remaining]:
        allocation[key] += 1
    return allocation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--lfa-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    if not 1 <= args.sample_size <= len(rows):
        raise ValueError("sample size must be between 1 and dataset size")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("dataset IDs are not unique")

    source_bins = rank_quartiles(rows, "source")
    reference_bins = rank_quartiles(rows, "reference")
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        case_id = str(row["id"])
        groups[(source_bins[case_id], reference_bins[case_id])].append(row)

    allocation = proportional_allocation(groups, args.sample_size)
    selected_ids: set[str] = set()
    for key, items in groups.items():
        ordered = sorted(items, key=lambda row: stable_key(args.seed, str(row["id"])))
        selected_ids.update(str(row["id"]) for row in ordered[: allocation[key]])
    selected = [row for row in rows if str(row["id"]) in selected_ids]
    if len(selected) != args.sample_size:
        raise RuntimeError(f"selected {len(selected)} rows, expected {args.sample_size}")

    lfa_rows = read_jsonl(args.lfa_predictions)
    lfa_by_id = {str(row["id"]): row for row in lfa_rows}
    missing = selected_ids - set(lfa_by_id)
    if missing:
        raise RuntimeError(f"LegalFactAware predictions missing {len(missing)} selected IDs")
    selected_lfa = [lfa_by_id[str(row["id"])] for row in selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output = args.output_dir / f"cail_dev_stratified_{args.sample_size}.jsonl"
    lfa_output = args.output_dir / f"legalfactaware_stratified_{args.sample_size}.jsonl"
    manifest_output = args.output_dir / f"subset_manifest_{args.sample_size}.json"
    write_jsonl(dataset_output, selected)
    write_jsonl(lfa_output, selected_lfa)

    joint_counts = Counter(
        f"S{source_bins[str(row['id'])] + 1}-R{reference_bins[str(row['id'])] + 1}"
        for row in selected
    )
    expression_counts = []
    for row in selected:
        expressions = expression_sets(str(row["reference"]))
        expression_counts.append(sum(len(values) for values in expressions.values()))
    manifest = {
        "protocol": "deterministic proportional source/reference length quartiles",
        "selection_uses_model_outputs": False,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "population_size": len(rows),
        "dataset_sha256": sha256_file(args.dataset),
        "lfa_predictions_sha256": sha256_file(args.lfa_predictions),
        "subset_dataset_sha256": sha256_file(dataset_output),
        "subset_lfa_sha256": sha256_file(lfa_output),
        "selected_ids_in_dataset_order": [str(row["id"]) for row in selected],
        "joint_stratum_counts": dict(sorted(joint_counts.items())),
        "source_characters": {
            "min": min(len(str(row["source"])) for row in selected),
            "mean": sum(len(str(row["source"])) for row in selected) / len(selected),
            "max": max(len(str(row["source"])) for row in selected),
        },
        "reference_characters": {
            "min": min(len(str(row["reference"])) for row in selected),
            "mean": sum(len(str(row["reference"])) for row in selected) / len(selected),
            "max": max(len(str(row["reference"])) for row in selected),
        },
        "reference_expression_count": {
            "min": min(expression_counts),
            "mean": sum(expression_counts) / len(expression_counts),
            "max": max(expression_counts),
        },
    }
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "dataset": str(dataset_output),
        "lfa_predictions": str(lfa_output),
        "manifest": str(manifest_output),
        "sample_size": len(selected),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
