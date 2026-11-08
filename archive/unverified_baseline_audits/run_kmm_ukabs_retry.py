#!/usr/bin/env python3
"""Retry KMM with the released stable tie order and budget-stop variants."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--budget-column", type=int, default=2)
    parser.add_argument("--budget-ratio", type=float)
    return parser.parse_args()


def pack(scores, lengths, budget, stop_on_no_fit):
    # Python's sort is stable: this matches the released
    # sorted(kmm.items(), key=operator.itemgetter(1), reverse=True).
    ranking = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    selected = []
    used = 0
    for index in ranking:
        if lengths[index] and used + lengths[index] <= budget:
            selected.append(index)
            used += lengths[index]
        elif stop_on_no_fit:
            break
    return selected, used


def initialise(base_dir):
    sys.path.insert(0, str(base_dir))
    global base
    import run_kmm_ukabs_audit as base


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
    score_sets = {
        "literal": base.literal_scores(token_rows, index),
        "repaired": base.repaired_scores(token_rows, index),
    }
    variants = {}
    for score_name, scores in score_sets.items():
        for policy, stop_on_no_fit in (("skip", False), ("stop", True)):
            variant = f"{score_name}_stable_{policy}"
            selected, used = pack(scores, lengths, budget, stop_on_no_fit)
            variants[variant] = {
                "prediction": " ".join(sentences[index].strip() for index in selected),
                "selected_indices": selected,
                "selected_sentences": len(selected),
                "used_words": used,
                "budget_violation": used > budget,
            }
    return {
        "file": name,
        "source_sentences": len(sentences),
        "budget": budget,
        "variants": variants,
    }


def main():
    args = parse_args()
    initialise(args.base_dir)
    variants = [
        f"{score_name}_stable_{policy}"
        for score_name in ("literal", "repaired")
        for policy in ("skip", "stop")
    ]
    for variant in variants:
        (args.output_dir / variant / "predictions").mkdir(parents=True, exist_ok=True)
    budgets = {}
    for fields in (line.split("\t") for line in args.length_file.read_text().splitlines()):
        value = int(fields[args.budget_column])
        if args.budget_ratio is not None:
            value = int(value * args.budget_ratio)
        budgets[fields[0]] = value
    paths = sorted(args.judgements.glob("*.txt"))
    tasks = [(path.name, str(path), budgets[path.name]) for path in paths]
    started = time.time()
    records = []
    with mp.get_context("spawn").Pool(
        args.workers, initializer=initialise, initargs=(args.base_dir,)
    ) as pool:
        for position, row in enumerate(pool.imap_unordered(worker, tasks), 1):
            for variant in variants:
                prediction = row["variants"][variant].pop("prediction")
                (args.output_dir / variant / "predictions" / row["file"]).write_text(
                    prediction, encoding="utf-8"
                )
            records.append(row)
            print(json.dumps({"done": position, "total": len(tasks), "file": row["file"]}), flush=True)
    records.sort(key=lambda row: row["file"])
    (args.output_dir / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    manifest = {
        "experiment": "KMM UK-Abs stable-tie retry",
        "documents": len(records),
        "variants": variants,
        "output_format": "single-line ranking-order summary",
        "budget_column": args.budget_column,
        "budget_ratio": args.budget_ratio,
        "budget_violations": {
            variant: sum(row["variants"][variant]["budget_violation"] for row in records)
            for variant in variants
        },
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
