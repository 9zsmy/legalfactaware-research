#!/usr/bin/env python3
"""Prepare common-label SENDE train/dev records from frozen IN-Abs assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--oracles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    oracles = {str(row["id"]): row for row in read_jsonl(args.oracles)}
    reports = {}
    for split in ("train", "dev"):
        docs = positives = sentences = 0
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in read_jsonl(args.source_dir / f"{split}.jsonl"):
                case_id = str(row["id"])
                if split == "train":
                    oracle = oracles[case_id]
                    if list(map(int, row["source_indices"])) != list(map(int, oracle["candidate_indices"])):
                        raise RuntimeError(f"candidate mismatch for {case_id}")
                    positive = set(map(int, oracle["ordinary"]))
                    labels = [int(int(index) in positive) for index in row["source_indices"]]
                else:
                    labels = [0] * len(row["sentences"])
                if len(labels) != len(row["sentences"]):
                    raise RuntimeError(f"sentence/label mismatch for {case_id}")
                output = {
                    "id": case_id, "sentences": row["sentences"], "labels": labels,
                    "summary": row["summary"], "source_indices": row["source_indices"],
                }
                stream.write(json.dumps(output, ensure_ascii=False) + "\n")
                docs += 1; sentences += len(labels); positives += sum(labels)
        reports[split] = {"documents": docs, "sentences": sentences, "positive_labels": positives}
    if reports["train"]["documents"] != 6325 or reports["dev"]["documents"] != 703:
        raise RuntimeError("frozen split count mismatch")
    manifest = {
        "method": "SENDE common-label independent reconstruction transfer",
        "supervision": "frozen Phase 2 ordinary greedy Oracle",
        "source_dir": str(args.source_dir), "oracles": str(args.oracles),
        "candidate_cap": 400, "official_test_contents_read": False, **reports,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()

