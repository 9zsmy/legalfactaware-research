#!/usr/bin/env python3
"""Evaluate two UK-Abs systems using the pltrdy/rouge 1.0.1 package."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarunner", type=Path, required=True)
    parser.add_argument("--gist", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flatten-newlines", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.vendor))
    from rouge import Rouge

    result = {}
    for method, pred_dir in (("summarunner", args.summarunner), ("gist", args.gist)):
        names = sorted(path.name for path in pred_dir.glob("*.txt"))
        hypotheses = [(pred_dir / name).read_text(encoding="utf-8") for name in names]
        references = [(args.references / name).read_text(encoding="utf-8") for name in names]
        if args.flatten_newlines:
            hypotheses = [" ".join(text.splitlines()) for text in hypotheses]
            references = [" ".join(text.splitlines()) for text in references]
        raw = Rouge().get_scores(hypotheses, references, avg=True)
        result[method] = {metric: 100.0 * values["f"] for metric, values in raw.items()}
        print(json.dumps({method: result[method]}), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
