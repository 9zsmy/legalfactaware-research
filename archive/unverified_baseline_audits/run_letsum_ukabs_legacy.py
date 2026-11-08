#!/usr/bin/env python3
"""Run the released LetSum pipeline on the line-segmented UK-Abs test set."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--letsum-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def initialise(letsum_dir):
    os.chdir(letsum_dir)
    sys.path.insert(0, str(letsum_dir))
    global letsum_test
    import letsum_test

    def parse_text(file):
        lines = [line for line in Path(file).read_text(errors="ignore").splitlines(True) if line.strip()]
        return lines, list(range(1, len(lines) + 1))

    letsum_test.parse_html = parse_text


def worker(task):
    name, path = task
    started = time.time()
    prediction = letsum_test.LetSum(path)
    return {
        "file": name,
        "prediction": prediction,
        "output_words": len(prediction.split()),
        "elapsed_seconds": time.time() - started,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "official_legacy" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.judgements.glob("*.txt"))
    tasks = [(path.name, str(path.resolve())) for path in paths]
    started = time.time()
    rows = []
    with mp.get_context("spawn").Pool(
        args.workers,
        initializer=initialise,
        initargs=(args.letsum_dir.resolve(),),
    ) as pool:
        for position, row in enumerate(pool.imap_unordered(worker, tasks), 1):
            prediction_dir.joinpath(row["file"]).write_text(row.pop("prediction"), encoding="utf-8")
            rows.append(row)
            print(json.dumps({"done": position, "total": len(tasks), **row}), flush=True)
    rows.sort(key=lambda row: row["file"])
    (args.output_dir / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = {
        "experiment": "released LetSum 34-percent thematic TF-IDF pipeline",
        "documents": len(rows),
        "mean_output_words": sum(row["output_words"] for row in rows) / len(rows),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
