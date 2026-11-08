#!/usr/bin/env python3
"""Source-grounded KMM formula sensitivity grid on UK-Abs."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--length-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).parent)
    return parser.parse_args()


def initialise(base_dir):
    sys.path.insert(0, str(base_dir))
    global base
    import run_kmm_ukabs_audit as base


def formula_scores(token_rows, index, n_mode, aggregation):
    n_value = 360.0 if n_mode == "n360" else float(len(token_rows))
    scores = []
    for row in token_rows:
        counts = {}
        for token in row:
            counts[token] = counts.get(token, 0) + 1
        values = []
        for token in row:
            occurrences = index.get(token, [])
            cf = sum(count for _, count in occurrences)
            df = len(occurrences)
            tf = counts[token]
            s_value = (cf - df) / df if df else 0.0
            r_value = (cf / n_value) / s_value if s_value else 0.0
            value = (r_value / (s_value + 1.0)) * (s_value / (s_value + 1.0))
            value += (s_value / (s_value + 1.0)) ** tf
            values.append(value)
        if not values:
            score = 0.0
        elif aggregation == "sum":
            score = sum(values)
        else:
            score = sum(values) / len(values)
        scores.append(score)
    return scores


def pack(scores, lengths, budget):
    ranking = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    selected = []
    used = 0
    for index in ranking:
        if lengths[index] and used + lengths[index] <= budget:
            selected.append(index)
            used += lengths[index]
    return selected, used


def worker(task):
    name, path, budget = task
    sentences = [
        line
        for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(True)
        if line.strip()
    ]
    token_rows = [base.tokens(sentence) for sentence in sentences]
    index = base.build_index(token_rows)
    nlp = base.spacy.blank("en")
    lengths = [base.count(doc) for doc in nlp.pipe(sentences, batch_size=128)]
    variants = {}
    for n_mode in ("n360", "ndoc"):
        for aggregation in ("mean", "sum"):
            variant = f"{aggregation}_{n_mode}"
            scores = formula_scores(token_rows, index, n_mode, aggregation)
            selected, used = pack(scores, lengths, budget)
            variants[variant] = {
                "prediction": " ".join(sentences[index].strip() for index in selected),
                "selected_indices": selected,
                "selected_sentences": len(selected),
                "used_words": used,
                "budget_violation": used > budget,
            }
    return {"file": name, "budget": budget, "variants": variants}


def main():
    args = parse_args()
    initialise(args.base_dir)
    variants = [f"{aggregation}_{n_mode}" for n_mode in ("n360", "ndoc") for aggregation in ("mean", "sum")]
    for variant in variants:
        (args.output_dir / variant / "predictions").mkdir(parents=True, exist_ok=True)
    budgets = {
        fields[0]: int(fields[2])
        for fields in (line.split("\t") for line in args.length_file.read_text().splitlines())
    }
    paths = sorted(args.judgements.glob("*.txt"))
    tasks = [(path.name, str(path), budgets[path.name]) for path in paths]
    started = time.time()
    records = []
    with mp.get_context("spawn").Pool(args.workers, initializer=initialise, initargs=(args.base_dir,)) as pool:
        for position, row in enumerate(pool.imap_unordered(worker, tasks), 1):
            for variant in variants:
                prediction = row["variants"][variant].pop("prediction")
                (args.output_dir / variant / "predictions" / row["file"]).write_text(prediction, encoding="utf-8")
            records.append(row)
            print(json.dumps({"done": position, "total": len(tasks)}), flush=True)
    records.sort(key=lambda row: row["file"])
    (args.output_dir / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    manifest = {
        "experiment": "KMM source-grounded formula grid",
        "documents": len(records),
        "variants": variants,
        "decoding": "stable descending rank, complete-sentence skip-fit, single-line output",
        "budget_violations": {
            variant: sum(row["variants"][variant]["budget_violation"] for row in records)
            for variant in variants
        },
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
