#!/usr/bin/env python3
"""Prepare official UK-Abs train/test JSONL for SummaRuNNer validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_budgets(path: Path) -> dict[str, int]:
    budgets = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            budgets[fields[0]] = int(fields[2])
    return budgets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelled-train", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = []
    for row in read_jsonl(args.labelled_train):
        sentences = row["doc"].split("\n")
        labels = row["labels"].split("\n")
        if len(sentences) != len(labels):
            raise RuntimeError(f"train sentence/label mismatch: {row['file']}")
        train_rows.append({
            "id": row["file"], "file": row["file"], "doc": row["doc"],
            "labels": row["labels"], "summaries": row.get("summaries", ""),
            "budget": len(row.get("summaries", "").split()),
            "indices": list(range(len(sentences))),
        })

    budgets = load_budgets(args.test_data / "stats-UK-test.txt")
    judgement_dir = args.test_data / "judgement"
    reference_dir = args.test_data / "summary" / "full"
    test_rows = []
    for path in sorted(judgement_dir.glob("*.txt")):
        sentences = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        reference = (reference_dir / path.name).read_text(encoding="utf-8")
        test_rows.append({
            "id": path.name, "file": path.name, "doc": "\n".join(sentences),
            "labels": "\n".join("0" for _ in sentences), "summaries": reference,
            "budget": budgets[path.name], "indices": list(range(len(sentences))),
        })

    if len(train_rows) != 693 or len(test_rows) != 100:
        raise RuntimeError(f"unexpected split sizes: {len(train_rows)}/{len(test_rows)}")
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)
    manifest = {
        "dataset": "official UK-Abs", "train_documents": len(train_rows),
        "test_documents": len(test_rows), "training_labels": "official Avr conversion",
        "test_budget": "official per-document reference word count",
        "inabs_official_test_read": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
