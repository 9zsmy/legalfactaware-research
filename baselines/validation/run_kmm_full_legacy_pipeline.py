#!/usr/bin/env python3
"""Audit the complete released KMM+CRF legacy pipeline on UK-Abs text files."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import nltk
import pycrfsuite


ORDER = ["F", "I", "A", "LR", "SS", "SP", "SO", "R"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def initialise(graph_dir, base_dir):
    os.chdir(graph_dir)
    sys.path.insert(0, str(graph_dir))
    sys.path.insert(0, str(base_dir))
    global base, crf_train, tagger
    import run_kmm_ukabs_audit as base
    import crf_train

    tagger = pycrfsuite.Tagger()
    tagger.open(str(graph_dir / "crf_alltrain.model"))


def labels_for(sentences):
    tokenised = []
    kept = []
    for index, sentence in enumerate(sentences):
        tokens = nltk.word_tokenize(sentence)
        if not tokens:
            continue
        tokenised.append([(token, "-") for token in tokens])
        kept.append(index)
    postagged = crf_train.add_postag(tokenised)
    total = len(postagged)
    labels = ["-"] * len(sentences)
    for position, (source_index, sentence) in enumerate(zip(kept, postagged), 1):
        features = crf_train.sentenceToFeatures(sentence, position / total)
        prediction = tagger.tag(features)
        labels[source_index] = prediction[0] if prediction else "-"
    return labels


def legacy_select(sentences, scores, labels, document_words):
    ranking = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    visited = {}
    selected = []
    used_space_words = 0
    for index in ranking:
        label = labels[index]
        count = visited.get(label, 0)
        if count >= 2:
            continue
        visited[label] = count + 1
        selected.append(index)
        used_space_words += len(sentences[index].split(" "))
        if used_space_words > 0.34 * document_words:
            break
    rank_order = " ".join(sentences[index].strip() for index in selected)
    grouped = "\n".join(
        "".join(sentences[index].strip() for index in selected if labels[index] == category)
        for category in ORDER
    )
    return selected, rank_order, grouped, used_space_words, visited


def worker(task):
    name, path, document_words = task
    sentences = [line for line in Path(path).read_text(errors="ignore").splitlines(True) if line.strip()]
    token_rows = [base.tokens(sentence) for sentence in sentences]
    index = base.build_index(token_rows)
    scores = base.literal_scores(token_rows, index)
    labels = labels_for(sentences)
    selected, rank_order, grouped, used, visited = legacy_select(
        sentences, scores, labels, document_words
    )
    return {
        "file": name,
        "predictions": {"legacy_rank_order": rank_order, "legacy_grouped": grouped},
        "selected_indices": selected,
        "selected_sentences": len(selected),
        "used_space_words": used,
        "document_words": document_words,
        "label_counts": visited,
    }


def main():
    args = parse_args()
    variants = ["legacy_rank_order", "legacy_grouped"]
    for variant in variants:
        (args.output_dir / variant / "predictions").mkdir(parents=True, exist_ok=True)
    document_words = {
        fields[0]: int(fields[1])
        for fields in (line.split("\t") for line in args.stats.read_text().splitlines())
    }
    paths = sorted(args.judgements.glob("*.txt"))
    tasks = [(path.name, str(path.resolve()), document_words[path.name]) for path in paths]
    started = time.time()
    rows = []
    with mp.get_context("spawn").Pool(
        args.workers,
        initializer=initialise,
        initargs=(args.graph_dir.resolve(), args.base_dir.resolve()),
    ) as pool:
        for position, row in enumerate(pool.imap_unordered(worker, tasks), 1):
            predictions = row.pop("predictions")
            for variant, prediction in predictions.items():
                (args.output_dir / variant / "predictions" / row["file"]).write_text(
                    prediction, encoding="utf-8"
                )
            rows.append(row)
            print(json.dumps({"done": position, "total": len(tasks), "file": row["file"]}), flush=True)
    rows.sort(key=lambda row: row["file"])
    (args.output_dir / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = {
        "experiment": "complete released KMM+CRF legacy pipeline",
        "documents": len(rows),
        "variants": variants,
        "summary_percent": 34,
        "max_sentences_per_crf_label": 2,
        "mean_selected_sentences": sum(row["selected_sentences"] for row in rows) / len(rows),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
