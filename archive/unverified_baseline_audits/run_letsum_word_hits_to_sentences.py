#!/usr/bin/env python3
"""Restore released LetSum word selections to their source sentences.

This is a diagnostic adaptation: it preserves the released thematic labels,
word-level TF-IDF threshold, and 10/24/60/6 allocation, then maps every
selected word occurrence back to its original sentence.  It is not presented
as the unmodified released implementation.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import multiprocessing as mp
import operator
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from nltk.corpus import stopwords


STOP = set(stopwords.words("english"))
CATEGORIES = ["Introduction", "Context", "Analysis", "Conclusion"]
QUOTAS = {"Introduction": 10, "Context": 24, "Analysis": 60, "Conclusion": 6}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--letsum-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def initialise(letsum_dir):
    os.chdir(letsum_dir)
    sys.path.insert(0, str(letsum_dir))
    global letsum_test
    import letsum_test


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


def official_word_hits(sentences, labels):
    """Return released-code selected word occurrences mapped to sentence IDs."""
    document_text = "".join(sentences)
    document_words = sum(len(sentence.split(" ")) for sentence in sentences)
    hits = {category: [] for category in CATEGORIES}
    category_budgets = {
        category: int(34 * document_words * QUOTAS[category] / 10000)
        for category in CATEGORIES
    }

    for category in CATEGORIES:
        indices = [index for index, label in enumerate(labels) if label == category]
        parts = [sentences[index] for index in indices]
        specific_text = "".join(parts)
        token_matches = list(re.finditer(r"[^ ]+", specific_text))
        words = [match.group(0) for match in token_matches]
        target = category_budgets[category]
        if not words or target == 0:
            continue

        boundaries = []
        offset = 0
        for index, part in zip(indices, parts):
            offset += len(part)
            boundaries.append((offset, index))
        ends = [end for end, _ in boundaries]

        def source_index(match):
            position = bisect.bisect_right(ends, match.start())
            if position >= len(boundaries):
                position = len(boundaries) - 1
            return boundaries[position][1]

        if target >= len(words):
            selected_positions = range(len(words))
        else:
            weights = {}
            for word in words:
                if word in weights:
                    continue
                if word.lower() in STOP:
                    weights[word] = 0.0
                    continue
                tf = document_text.lower().count(word.lower())
                df = sum(word.lower() in sentence.lower() for sentence in sentences) or 0.5
                weights[word] = tf * math.log(len(sentences) / df)
            sorted_scores = sorted(weights.values())
            threshold = sorted_scores[min(target - 1, len(sorted_scores) - 1)]
            selected_positions = []
            for position, word in enumerate(words):
                if weights[word] > threshold:
                    selected_positions.append(position)
                if len(selected_positions) >= target:
                    break

        for position in selected_positions:
            hits[category].append(source_index(token_matches[position]))
    return hits, category_budgets, document_words


def fit_hit_sentences(sentences, hits, category_budgets, policy):
    selected = []
    for category in CATEGORIES:
        if policy == "first_hit":
            ranking = list(dict.fromkeys(hits[category]))
        elif policy == "hit_count":
            counts = Counter(hits[category])
            first = {index: position for position, index in enumerate(dict.fromkeys(hits[category]))}
            ranking = sorted(counts, key=lambda index: (-counts[index], first[index]))
        else:
            raise ValueError(policy)
        used = 0
        for index in ranking:
            length = len(sentences[index].split(" "))
            if length and used + length <= category_budgets[category]:
                selected.append(index)
                used += length
    return sorted(set(selected))


def worker(task):
    name, path = task
    sentences = [
        line for line in Path(path).read_text(errors="ignore").splitlines(True) if line.strip()
    ]
    labels = [label_sentence(sentence) for sentence in sentences]
    hits, category_budgets, document_words = official_word_hits(sentences, labels)
    all_hits = sorted({index for category in CATEGORIES for index in hits[category]})
    variants = {
        "all_hits_source": all_hits,
        "budget_first_hit_source": fit_hit_sentences(
            sentences, hits, category_budgets, "first_hit"
        ),
        "budget_hit_count_source": fit_hit_sentences(
            sentences, hits, category_budgets, "hit_count"
        ),
    }
    predictions = {
        variant: " ".join(sentences[index].strip() for index in indices)
        for variant, indices in variants.items()
    }
    metadata = {}
    total_budget = sum(category_budgets.values())
    for variant, indices in variants.items():
        used = sum(len(sentences[index].split(" ")) for index in indices)
        metadata[variant] = {
            "selected_sentences": len(indices),
            "used_words": used,
            "budget": total_budget,
            "budget_violation": used > total_budget,
        }
    return {"file": name, "predictions": predictions, "metadata": metadata}


def main():
    args = parse_args()
    variants = ["all_hits_source", "budget_first_hit_source", "budget_hit_count_source"]
    for variant in variants:
        (args.output_dir / variant / "predictions").mkdir(parents=True, exist_ok=True)
    paths = sorted(args.judgements.glob("*.txt"))
    tasks = [(path.name, str(path.resolve())) for path in paths]
    rows = []
    started = time.time()
    with mp.get_context("spawn").Pool(
        args.workers, initializer=initialise, initargs=(args.letsum_dir.resolve(),)
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
        "experiment": "LetSum released word hits restored to complete source sentences",
        "identity": "diagnostic adaptation, not unmodified official reproduction",
        "documents": len(rows),
        "summary_percent": 34,
        "quotas": QUOTAS,
        "variants": variants,
        "budget_violations": {
            variant: sum(row["metadata"][variant]["budget_violation"] for row in rows)
            for variant in variants
        },
        "mean_used_words": {
            variant: sum(row["metadata"][variant]["used_words"] for row in rows) / len(rows)
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
