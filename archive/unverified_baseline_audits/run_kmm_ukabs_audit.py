#!/usr/bin/env python3
"""Audit the released K-mixture sentence ranking module on UK-Abs."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import string
import time
from collections import defaultdict
from pathlib import Path

import nltk
import spacy
from nltk.corpus import stopwords


TABLE = str.maketrans("", "", string.punctuation + string.whitespace)
STOP = set(stopwords.words("english")) | set("!#$%&()*+,-./:;<=>?@[\\]^_`{|}~")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--length-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def tokens(sentence):
    return [token for token in nltk.word_tokenize(sentence.lower()) if token not in STOP]


def build_index(token_rows):
    index = defaultdict(list)
    for doc_id, row in enumerate(token_rows, 1):
        counts = {}
        for token in row:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            index[token].append((doc_id, count))
    return index


def literal_scores(token_rows, index):
    # Preserve the released module exactly: N=360, tf/df persist between
    # tokens unless the token occurs in sentence 1, and pk is overwritten.
    n_constant = 360.0
    tf = 0
    df = 0
    scores = []
    for row in token_rows:
        pk = 0.0
        for token in row:
            occurrences = index.get(token, [])
            cf = 0
            for doc_id, count in occurrences:
                if doc_id == 1:
                    tf = count
                    df = len(occurrences)
                cf += count
            _idf = math.log(n_constant, 2) / df if df else 0.0
            s_value = (cf - df) / df if df else 0.0
            r_value = (cf / n_constant) / s_value if s_value else 0.0
            pk = (r_value / (s_value + 1.0)) * (s_value / (s_value + 1.0))
            pk += (s_value / (s_value + 1.0)) ** tf
        scores.append(pk)
    return scores


def repaired_scores(token_rows, index):
    # Minimal formula repair: current-sentence tf/current token df and mean
    # token probability, while preserving the published code's N=360.
    n_constant = 360.0
    scores = []
    for sentence_id, row in enumerate(token_rows, 1):
        token_values = []
        local_counts = {}
        for token in row:
            local_counts[token] = local_counts.get(token, 0) + 1
        for token in row:
            occurrences = index.get(token, [])
            cf = sum(count for _, count in occurrences)
            df = len(occurrences)
            tf = local_counts[token]
            s_value = (cf - df) / df if df else 0.0
            r_value = (cf / n_constant) / s_value if s_value else 0.0
            value = (r_value / (s_value + 1.0)) * (s_value / (s_value + 1.0))
            value += (s_value / (s_value + 1.0)) ** tf
            token_values.append(value)
        scores.append(sum(token_values) / len(token_values) if token_values else 0.0)
    return scores


def pack(scores, lengths, budget):
    ranking = sorted(range(len(scores)), key=lambda index: (scores[index], index), reverse=True)
    selected = []
    used = 0
    for index in ranking:
        if lengths[index] and used + lengths[index] <= budget:
            selected.append(index)
            used += lengths[index]
    return selected, used


def count(doc):
    return sum(len(token.text.translate(TABLE)) > 0 for token in doc)


def worker(task):
    name, path, budget = task
    sentences = [line for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(True) if line.strip()]
    token_rows = [tokens(sentence) for sentence in sentences]
    index = build_index(token_rows)
    nlp = spacy.blank("en")
    lengths = [count(doc) for doc in nlp.pipe(sentences, batch_size=128)]
    result = {"file": name, "source_sentences": len(sentences), "budget": budget, "variants": {}}
    for variant, scores in (
        ("literal", literal_scores(token_rows, index)),
        ("repaired", repaired_scores(token_rows, index)),
    ):
        selected, used = pack(scores, lengths, budget)
        result["variants"][variant] = {
            "prediction": "".join(sentences[index] for index in selected),
            "selected_indices": selected,
            "selected_sentences": len(selected),
            "used_words": used,
            "budget_violation": used > budget,
        }
    return result


def main():
    args = args_parser()
    for variant in ("literal", "repaired"):
        (args.output_dir / variant / "predictions").mkdir(parents=True, exist_ok=True)
    budgets = {
        fields[0]: int(fields[1])
        for fields in (line.split("\t") for line in args.length_file.read_text().splitlines())
    }
    paths = sorted(args.judgements.glob("*.txt"))
    tasks = [(path.name, str(path), budgets[path.name]) for path in paths]
    started = time.time()
    records = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for position, row in enumerate(pool.imap_unordered(worker, tasks), 1):
            for variant in ("literal", "repaired"):
                prediction = row["variants"][variant].pop("prediction")
                (args.output_dir / variant / "predictions" / row["file"]).write_text(prediction, encoding="utf-8")
            records.append(row)
            print(json.dumps({"phase": "rank", "done": position, "total": len(tasks), "file": row["file"]}), flush=True)
    records.sort(key=lambda row: row["file"])
    (args.output_dir / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    manifest = {
        "experiment": "KMM UK-Abs sentence-ranking audit",
        "documents": len(records),
        "source": "official k_mix_model_test.py ranking module",
        "rhetorical_labels_used": False,
        "variants": {
            "literal": "released variable persistence and last-token overwrite behavior",
            "repaired": "current-sentence tf/current-token df and mean token score",
        },
        "decoding": "descending KMM score; complete-sentence fit; ranking-order output",
        "budget_violations": {
            variant: sum(row["variants"][variant]["budget_violation"] for row in records)
            for variant in ("literal", "repaired")
        },
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
