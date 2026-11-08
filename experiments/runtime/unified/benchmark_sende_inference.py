#!/usr/bin/env python3
"""Time frozen SENDE extractor inference on its archived sentence embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_dir))
    import evaluate_cail as sende

    records, probabilities = sende.load_probabilities(
        args.records, args.prefix, args.checkpoint
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row, scores in zip(records, probabilities):
            selected, prediction = sende.decode(row["sentences"], scores, args.threshold)
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "selected_indices": selected,
                        "generated_summary": prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps({"method": "SENDE", "cases": len(records)}))


if __name__ == "__main__":
    main()
