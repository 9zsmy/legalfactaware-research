#!/usr/bin/env python3
"""LetSum thematic quotas with TF-IDF-ranked complete sentences on UK-Abs."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import operator
import os
import re
import sys
import time
from pathlib import Path

import spacy
from nltk.corpus import stopwords


STOP = set(stopwords.words("english"))
CATEGORIES = ["Introduction", "Context", "Analysis", "Conclusion"]
QUOTAS = {"Introduction": 10, "Context": 24, "Analysis": 60, "Conclusion": 6}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--letsum-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def initialise(letsum_dir):
    os.chdir(letsum_dir)
    sys.path.insert(0, str(letsum_dir))
    global letsum_test
    import letsum_test


def count_words(doc):
    return sum(any(char.isalnum() for char in token.text) for token in doc)


def label_sentence(line):
    score = {category: 0 for category in CATEGORIES}
    words = [item.lower() for item in line.split(" ")]
    for fire_category, letsum_category in letsum_test.mapping_to_letsum.items():
        phrases = letsum_test.cue_phrases[fire_category]
        single = [phrase for phrase in phrases if len(phrase.split(" ")) == 1]
        score[letsum_category] += sum(phrase in words for phrase in single)
        multi = [phrase for phrase in phrases if phrase not in single]
        score[letsum_category] += sum(phrase in line for phrase in multi)
    if max(score.values()) == 0:
        return "Context"
    return max(score.items(), key=operator.itemgetter(1))[0]


def tfidf_scores(sentences, aggregation):
    full_text = "".join(sentences)
    rows = [[word for word in sentence.split(" ") if word] for sentence in sentences]
    vocabulary = {word for row in rows for word in row}
    weights = {}
    for word in vocabulary:
        if word.lower() in STOP:
            weights[word] = 0.0
            continue
        tf = full_text.lower().count(word.lower())
        df = sum(word.lower() in sentence.lower() for sentence in sentences) or 0.5
        weights[word] = tf * math.log(len(sentences) / df)
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


def scoped_tfidf_scores(sentences, labels, aggregation, scope):
    if scope == "global":
        return tfidf_scores(sentences, aggregation)
    output = [0.0] * len(sentences)
    for category in CATEGORIES:
        indices = [index for index, label in enumerate(labels) if label == category]
        local_sentences = [sentences[index] for index in indices]
        if not local_sentences:
            continue
        local_scores = tfidf_scores(local_sentences, aggregation)
        for index, score in zip(indices, local_scores):
            output[index] = score
    return output


def sentence_trigrams(sentence):
    tokens = re.findall(r"[a-z0-9]+", sentence.lower())
    return set(zip(tokens, tokens[1:], tokens[2:]))


def select(sentences, labels, lengths, scores, document_words, redundancy):
    selected = []
    covered_trigrams = set()
    trigrams = [sentence_trigrams(sentence) for sentence in sentences]
    used = 0
    total_budget = int(document_words * 0.34)
    for category in CATEGORIES:
        category_budget = int(total_budget * QUOTAS[category] / 100)
        category_used = 0
        ranking = sorted(
            (index for index, label in enumerate(labels) if label == category),
            key=lambda index: scores[index],
            reverse=True,
        )
        for index in ranking:
            blocked = redundancy == "trigram" and bool(trigrams[index] & covered_trigrams)
            if (
                not blocked
                and lengths[index]
                and category_used + lengths[index] <= category_budget
            ):
                selected.append(index)
                category_used += lengths[index]
                used += lengths[index]
                covered_trigrams.update(trigrams[index])
    return selected, used, total_budget


def worker(task):
    name, path, document_words = task
    sentences = [line for line in Path(path).read_text(errors="ignore").splitlines(True) if line.strip()]
    labels = [label_sentence(sentence) for sentence in sentences]
    nlp = spacy.blank("en")
    lengths = [count_words(doc) for doc in nlp.pipe(sentences, batch_size=128)]
    predictions = {}
    metadata = {}
    for scope in ("global", "category"):
        for aggregation in ("sum", "mean"):
            sentence_scores = scoped_tfidf_scores(sentences, labels, aggregation, scope)
            for redundancy in ("none", "trigram"):
                selected, used, budget = select(
                    sentences,
                    labels,
                    lengths,
                    sentence_scores,
                    document_words,
                    redundancy,
                )
                for order in ("grouped", "source"):
                    indices = selected if order == "grouped" else sorted(selected)
                    variant = f"{scope}_{aggregation}_{redundancy}_{order}"
                    predictions[variant] = " ".join(sentences[index].strip() for index in indices)
                    metadata[variant] = {
                        "selected_sentences": len(indices),
                        "used_words": used,
                        "budget": budget,
                        "budget_violation": used > budget,
                    }
    return {"file": name, "predictions": predictions, "metadata": metadata}


def main():
    args = parse_args()
    variants = [
        f"{scope}_{aggregation}_{redundancy}_{order}"
        for scope in ("global", "category")
        for aggregation in ("sum", "mean")
        for redundancy in ("none", "trigram")
        for order in ("grouped", "source")
    ]
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
        args.workers, initializer=initialise, initargs=(args.letsum_dir.resolve(),)
    ) as pool:
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
        "experiment": "LetSum thematic quota complete-sentence audit with trigram-blocking adaptation",
        "documents": len(rows),
        "summary_percent": 34,
        "quotas": QUOTAS,
        "variants": variants,
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
