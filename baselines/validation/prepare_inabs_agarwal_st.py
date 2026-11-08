#!/usr/bin/env python3
"""Convert the frozen MemSum/IN-Abs preparation into Agarwal-ST JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def convert(row: dict, training: bool) -> dict:
    positives = set(row["indices"][0]) if training else set()
    sentences = row["raw_text"]
    return {
        "id": row["id"],
        "sentences": sentences,
        "labels": [int(index in positives) for index in range(len(sentences))],
        "summary": row["raw_summary"],
        "source_indices": row["source_indices"],
    }


def write_jsonl(path: Path, rows) -> tuple[int, int, int]:
    documents = sentences = positives = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            documents += 1
            sentences += len(row["sentences"])
            positives += sum(row["labels"])
    return documents, sentences, positives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memsum-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_stats = write_jsonl(
        args.output_dir / "train.jsonl",
        (convert(row, True) for row in read_jsonl(args.memsum_dir / "train.jsonl")),
    )
    dev_stats = write_jsonl(
        args.output_dir / "dev.jsonl",
        (convert(row, False) for row in read_jsonl(args.memsum_dir / "dev.jsonl")),
    )
    manifest = {
        "method": "Agarwal-ST restricted independent reimplementation",
        "source": str(args.memsum_dir),
        "train": {"documents": train_stats[0], "sentences": train_stats[1], "positive_labels": train_stats[2]},
        "dev": {"documents": dev_stats[0], "sentences": dev_stats[1]},
        "candidate_cap": 400,
        "supervision": "frozen P50 (643-word) budget oracle",
        "official_test_contents_read": False,
    }
    (args.output_dir / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
