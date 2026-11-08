#!/usr/bin/env python3
"""Evaluate KMM retry variants with the paper-specified py-rouge 1.1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rouge import Rouge


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--default-length-limit", action="store_true")
    parser.add_argument("--preserve-newlines", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluator_args = {
        "metrics": ["rouge-n", "rouge-l"],
        "max_n": 2,
    }
    if not args.default_length_limit:
        evaluator_args["limit_length"] = False
    evaluator = Rouge(**evaluator_args)
    output = {}
    variants = sorted(path.parent.name for path in args.results_dir.glob("*/predictions"))
    for variant in variants:
        started = time.time()
        prediction_dir = args.results_dir / variant / "predictions"
        names = sorted(path.name for path in prediction_dir.glob("*.txt"))
        # The paper evaluates a document-wide summary.  Collapse both sides to
        # one logical sequence; leaving only the reference split by newlines
        # changes py-rouge's union-LCS behaviour and greatly inflates ROUGE-L.
        if args.preserve_newlines:
            hypotheses = [prediction_dir.joinpath(name).read_text(errors="ignore") for name in names]
            references = [args.references.joinpath(name).read_text(errors="ignore") for name in names]
        else:
            hypotheses = [
                " ".join(prediction_dir.joinpath(name).read_text(errors="ignore").split())
                for name in names
            ]
            references = [
                " ".join(args.references.joinpath(name).read_text(errors="ignore").split())
                for name in names
            ]
        scores = evaluator.get_scores(hypotheses, references)
        output[variant] = {
            metric: {stat: 100 * value[stat] for stat in ("p", "r", "f")}
            for metric, value in scores.items()
        }
        output[variant]["elapsed_seconds"] = time.time() - started
        print(json.dumps({variant: output[variant]}, indent=2), flush=True)
        (args.results_dir / "exact_pyrouge11_metrics.partial.json").write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
    if args.default_length_limit:
        output_name = "exact_pyrouge11_default_665bytes.json"
    elif args.preserve_newlines:
        output_name = "exact_pyrouge11_preserve_newlines.json"
    else:
        output_name = "exact_pyrouge11_metrics.json"
    (args.results_dir / output_name).write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
