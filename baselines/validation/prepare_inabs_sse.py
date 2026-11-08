#!/usr/bin/env python3
"""Build five-part SSE proxy labels from the frozen IN-Abs train/dev records."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
PART_NAMES = ["headnote_1", "headnote_2", "headnote_3", "headnote_4", "headnote_5"]
QUOTAS = [1, 2, 1, 2, 1]


def words(text: str) -> list[str]:
    return [value.lower() for value in TOKEN_RE.findall(text)]


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def f1(left: Counter, right: Counter) -> float:
    overlap = sum((left & right).values())
    total = sum(left.values()) + sum(right.values())
    return 2 * overlap / total if total else 0.0


def part_tokens(summary: str) -> list[list[str]]:
    tokens = words(summary)
    return [tokens[len(tokens) * i // 5:len(tokens) * (i + 1) // 5] for i in range(5)]


def convert(row: dict) -> dict:
    sentences = list(row["sentences"])
    chunks = part_tokens(row["summary"])
    prepared = [words(sentence) for sentence in sentences]
    sentence_grams = [(ngrams(tokens, 1), ngrams(tokens, 2)) for tokens in prepared]
    part_indices: list[list[int]] = []
    part_labels: list[list[int]] = []
    for chunk, quota in zip(chunks, QUOTAS):
        reference = (ngrams(chunk, 1), ngrams(chunk, 2))
        scores = [
            f1(one, reference[0]) + f1(two, reference[1]) if tokens else 0.0
            for tokens, (one, two) in zip(prepared, sentence_grams)
        ]
        ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
        selected = sorted(i for i in ranked[:quota] if scores[i] > 0)
        part_indices.append(selected)
        part_labels.append([int(i in selected) for i in range(len(sentences))])
    return {
        "id": str(row["id"]),
        "text": [" ".join(tokens) for tokens in prepared],
        "raw_text": sentences,
        "raw_summary": row["summary"],
        "part_names": PART_NAMES,
        "part_indices": part_indices,
        "part_labels": part_labels,
        "source_indices": row["source_indices"],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sse-scripts", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    sys.path.insert(0, str(args.sse_scripts))
    from prepare_sse_cail import build_assets

    reports = {}
    train_rows = []
    for split, expected in (("train", 6325), ("dev", 703)):
        source = args.source_dir / f"{split}.jsonl"
        output = args.output_dir / f"{split}.jsonl"
        count = positives = overlong_tokens = 0
        with source.open(encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                converted = convert(json.loads(line))
                if len(converted["text"]) > 400:
                    raise RuntimeError("candidate cap violated")
                count += 1
                positives += sum(map(sum, converted["part_labels"]))
                overlong_tokens += sum(len(sentence.split()) > 127 for sentence in converted["text"])
                dst.write(json.dumps(converted, ensure_ascii=False) + "\n")
                if split == "train":
                    train_rows.append(converted)
                if count % 500 == 0:
                    print(json.dumps({"stage": "prepare", "split": split, "done": count}), flush=True)
        if count != expected:
            raise RuntimeError(f"{split} expected {expected}, found {count}")
        reports[split] = {
            "documents": count, "part_positive_labels": positives,
            "sentence_token_truncations": overlong_tokens,
            "source_sha256": sha256(source), "prepared_sha256": sha256(output),
        }
    build_assets(train_rows, args.output_dir, 12000, 128, 20260814)
    manifest = {
        "method": "SSE structural-proxy English adaptation",
        "part_names": PART_NAMES, "part_quotas": QUOTAS,
        "label_rule": "individual word 1+2gram F1 top-k per equal-word reference quintile",
        "candidate_cap": 400, "official_test_contents_read": False,
        **reports,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"stage": "prepare_complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
