#!/usr/bin/env python3
"""Aggregate the five random seeds reported by the TermDiffuSum paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEEDS = (100, 101, 201, 901, 1001)
METRICS = ("rouge-1", "rouge-2", "rouge-l")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in SEEDS:
        if seed == 101:
            source_path = (
                args.phase_root
                / "outputs/phase5h_termdiffusum_source_faithful_unbounded_eval/metrics.json"
            )
            budget_path = (
                args.phase_root
                / "outputs/phase5h_termdiffusum_source_faithful_budget_eval/metrics.json"
            )
        else:
            source_path = (
                args.phase_root
                / f"outputs/phase5h_termdiffusum_official_code_seed{seed}_source_eval/metrics.json"
            )
            budget_path = (
                args.phase_root
                / f"outputs/phase5h_termdiffusum_official_code_seed{seed}_budget_eval/metrics.json"
            )
        source = load_json(source_path)
        budget = load_json(budget_path)
        rows.append(
            {
                "seed": seed,
                "source_decoder": source["f1_percent"],
                "source_budget_violations": source["budget_violations"],
                "budget_adapted_decoder": budget["f1_percent"],
                "budget_adapted_violations": budget["budget_violations"],
            }
        )

    aggregate = {
        "method": "TermDiffuSum official-public-code IN-Abs adaptation",
        "paper_seeds": list(SEEDS),
        "official_test_contents_read": False,
        "exact_paper_reproduction": False,
        "limitation": (
            "The authors state that the public package is partial. The paper reports "
            "35 candidate summaries, but the released legal records contain four "
            "candidate combinations and the released training bytecode does not consume "
            "that field. Results therefore reproduce the public-code path, not the "
            "unreleased complete implementation."
        ),
        "per_seed": rows,
    }
    for track in ("source_decoder", "budget_adapted_decoder"):
        aggregate[track] = {}
        for metric in METRICS:
            values = np.asarray([row[track][metric] for row in rows], dtype=float)
            aggregate[track][metric] = {
                "mean": float(values.mean()),
                "std_population": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    output = args.output_dir / "metrics.json"
    output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
