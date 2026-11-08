#!/usr/bin/env python3
"""Protocol-compliant audit of the released UK-Abs MMR implementation."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import string
import time
from pathlib import Path

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


TABLE = str.maketrans("", "", string.punctuation + string.whitespace)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--length-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lambda-value", type=float, default=0.6)
    return parser.parse_args()


def count(doc):
    return sum(len(token.text.translate(TABLE)) > 0 for token in doc)


def select(sentences, lengths, budget, lambda_value):
    try:
        matrix = TfidfVectorizer(smooth_idf=False).fit_transform(sentences)
        similarities = (matrix @ matrix.T).toarray()
    except ValueError:
        similarities = np.eye(len(sentences))
    relevance_matrix = similarities.copy()
    np.fill_diagonal(relevance_matrix, -1.0)
    relevance = relevance_matrix.max(axis=1)

    selected = []
    remaining = set(range(len(sentences)))
    used = 0
    while remaining:
        fitting = [index for index in remaining if lengths[index] and used + lengths[index] <= budget]
        if not fitting:
            break
        if selected:
            redundancy = similarities[:, selected].max(axis=1)
        else:
            redundancy = np.zeros(len(sentences))
        # Released code resolves equal scores in favour of the later index.
        best = max(
            fitting,
            key=lambda index: (
                lambda_value * relevance[index] - (1.0 - lambda_value) * redundancy[index],
                index,
            ),
        )
        selected.append(best)
        remaining.remove(best)
        used += lengths[best]
    return selected, used


def worker(task):
    name, path, budget, lambda_value = task
    sentences = [line for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(True) if line.strip()]
    nlp = spacy.blank("en")
    lengths = [count(doc) for doc in nlp.pipe(sentences, batch_size=128)]
    selected, used = select(sentences, lengths, budget, lambda_value)
    return {
        "file": name,
        "prediction": "".join(sentences[index] for index in selected),
        "selected_indices": selected,
        "selected_sentences": len(selected),
        "source_sentences": len(sentences),
        "budget": budget,
        "used_words": used,
        "budget_violation": used > budget,
    }


def main():
    args = arguments()
    predictions = args.output_dir / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    budgets = {
        fields[0]: int(fields[1])
        for fields in (line.split("\t") for line in args.length_file.read_text().splitlines())
    }
    paths = sorted(args.judgements.glob("*.txt"))
    tasks = [(path.name, str(path), budgets[path.name], args.lambda_value) for path in paths]
    started = time.time()
    records = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for position, row in enumerate(pool.imap_unordered(worker, tasks), 1):
            (predictions / row["file"]).write_text(row.pop("prediction"), encoding="utf-8")
            records.append(row)
            print(json.dumps({"phase": "select", "done": position, "total": len(tasks), "file": row["file"]}), flush=True)
    records.sort(key=lambda row: row["file"])
    (args.output_dir / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    report = {
        "experiment": "MMR UK-Abs protocol-compliant released-code audit",
        "documents": len(records),
        "lambda": args.lambda_value,
        "representation": "released TfidfVectorizer(smooth_idf=False)",
        "corrections": ["exclude already selected sentences", "complete-sentence fit within budget"],
        "selection_order_output": True,
        "budget_violations": sum(row["budget_violation"] for row in records),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
