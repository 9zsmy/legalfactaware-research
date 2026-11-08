#!/usr/bin/env python3
"""AACL-described LetSum TF-IDF sentence-ranking variant on UK-Abs."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import spacy
from nltk.corpus import stopwords


STOP = set(stopwords.words("english"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--budget-column", type=int, default=2)
    parser.add_argument("--budget-ratio", type=float)
    return parser.parse_args()


def count_words(doc):
    return sum(any(char.isalnum() for char in token.text) for token in doc)


def scores(sentences, aggregation):
    full_text = "".join(sentences)
    rows = [[word for word in sentence.split(" ") if word] for sentence in sentences]
    vocabulary = {word for row in rows for word in row}
    weights = {}
    for word in vocabulary:
        if word.lower() in STOP:
            weights[word] = 0.0
            continue
        term_frequency = full_text.lower().count(word.lower())
        sentence_frequency = sum(word.lower() in sentence.lower() for sentence in sentences) or 0.5
        weights[word] = term_frequency * math.log(len(sentences) / sentence_frequency)
    output = []
    for row in rows:
        values = [weights[word] for word in row]
        if not values:
            output.append(0.0)
        elif aggregation == "sum":
            output.append(sum(values))
        else:
            output.append(sum(values) / len(values))
    return output


def pack(sentence_scores, lengths, budget):
    ranking = sorted(range(len(sentence_scores)), key=lambda index: sentence_scores[index], reverse=True)
    selected = []
    used = 0
    for index in ranking:
        if lengths[index] and used + lengths[index] <= budget:
            selected.append(index)
            used += lengths[index]
    return selected, used


def worker(task):
    name, path, budget = task
    sentences = [line for line in Path(path).read_text(errors="ignore").splitlines(True) if line.strip()]
    nlp = spacy.blank("en")
    lengths = [count_words(doc) for doc in nlp.pipe(sentences, batch_size=128)]
    predictions = {}
    metadata = {}
    for aggregation in ("sum", "mean"):
        selected, used = pack(scores(sentences, aggregation), lengths, budget)
        for order in ("rank", "source"):
            indices = selected if order == "rank" else sorted(selected)
            variant = f"{aggregation}_{order}"
            predictions[variant] = " ".join(sentences[index].strip() for index in indices)
            metadata[variant] = {
                "selected_sentences": len(indices),
                "used_words": used,
                "budget_violation": used > budget,
            }
    return {"file": name, "predictions": predictions, "metadata": metadata}


def main():
    args = parse_args()
    variants = [f"{aggregation}_{order}" for aggregation in ("sum", "mean") for order in ("rank", "source")]
    for variant in variants:
        (args.output_dir / variant / "predictions").mkdir(parents=True, exist_ok=True)
    budgets = {}
    for fields in (line.split("\t") for line in args.stats.read_text().splitlines()):
        value = int(fields[args.budget_column])
        if args.budget_ratio is not None:
            value = int(value * args.budget_ratio)
        budgets[fields[0]] = value
    paths = sorted(args.judgements.glob("*.txt"))
    tasks = [(path.name, str(path.resolve()), budgets[path.name]) for path in paths]
    started = time.time()
    rows = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for position, row in enumerate(pool.imap_unordered(worker, tasks), 1):
            predictions = row.pop("predictions")
            for variant, prediction in predictions.items():
                (args.output_dir / variant / "predictions" / row["file"]).write_text(prediction, encoding="utf-8")
            rows.append(row)
            print(json.dumps({"done": position, "total": len(tasks), "file": row["file"]}), flush=True)
    rows.sort(key=lambda row: row["file"])
    (args.output_dir / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = {
        "experiment": "AACL-described LetSum TF-IDF sentence ranking",
        "documents": len(rows),
        "variants": variants,
        "budget_column": args.budget_column,
        "budget_ratio": args.budget_ratio,
        "budget_violations": {
            variant: sum(row["metadata"][variant]["budget_violation"] for row in rows)
            for variant in variants
        },
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
