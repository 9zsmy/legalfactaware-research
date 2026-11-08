#!/usr/bin/env python3
"""Run the released CaseSummarizer algorithm on UK-Abs with audit variants.

The upstream program is Python 2/Windows-only.  This runner preserves its
feature formula, corpus-level TF-IDF, sentence ranking, and hard word-budget
truncation while making path/string handling executable on Python 3.

Two score variants are emitted:
  literal  - preserves the released dictionary-newline and stale-digit bugs.
  intended - strips dictionary entries and computes the digit flag per sentence.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import re
import string
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree


G = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--length-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def normalized_sentences(text: str):
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    sentences = tokenizer.tokenize(text)
    stop = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    lower = []
    cased = []
    for sentence in sentences:
        low_tokens = [
            lemmatizer.lemmatize(word)
            for word in word_tokenize(sentence.lower())
            if word not in stop
        ]
        case_tokens = [
            lemmatizer.lemmatize(word)
            for word in word_tokenize(sentence)
            if word not in stop
        ]
        lower.append(low_tokens)
        cased.append(case_tokens)
    return sentences, lower, cased


def build_df(paths):
    df = Counter()
    cached = {}
    for position, path in enumerate(paths, 1):
        sentences, lower, cased = normalized_sentences(
            path.read_text(encoding="utf-8", errors="ignore")
        )
        cached[path.name] = (sentences, lower, cased)
        unique = {word for tokens in lower for word in tokens if word != "."}
        df.update(unique)
        if position % 10 == 0:
            print(json.dumps({"phase": "df", "done": position, "total": len(paths)}), flush=True)
    return dict(df), cached


def make_prefix_index(entries):
    index = {}
    for entry in entries:
        clean = entry.strip("\r\n")
        for size in range(1, len(clean) + 1):
            index.setdefault(clean[:size], []).append(entry)
    return index


def init_worker(df, total_docs, literal_entries, intended_entries, cached):
    G["df"] = df
    G["total_docs"] = total_docs
    G["literal_entries"] = literal_entries
    G["intended_entries"] = intended_entries
    G["literal_prefix"] = make_prefix_index(literal_entries)
    G["intended_prefix"] = make_prefix_index(intended_entries)
    G["cached"] = cached


def entities(tokens):
    chunked = ne_chunk(pos_tag(tokens))
    continuous = []
    current = []
    for item in chunked:
        if isinstance(item, Tree):
            current.append(" ".join(token for token, _ in item.leaves()))
        elif current:
            entity = " ".join(current)
            if entity not in continuous:
                continuous.append(entity)
                current = []
    # The released function does not flush a final entity; preserve that.
    return continuous


def legal_density(tokens, normalized_text, variant):
    bag = []
    prefix = G[f"{variant}_prefix"]
    for word in tokens:
        word = word.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        word = word.replace("{", "").replace("}", "")
        for item in prefix.get(word, []):
            if item in normalized_text:
                bag.extend(item.split(" "))
    return len(set(bag)) / len(tokens) if tokens else 0.0


def score_one(name):
    sentences, lower, cased = G["cached"][name]
    tf = Counter(word for tokens in lower for word in tokens if word != ".")
    total = sum(tf.values())
    tfidf = {
        word: (count / total) * math.log10(G["total_docs"] / G["df"][word])
        for word, count in tf.items()
    } if total else {}

    base = []
    for tokens in lower:
        base.append(sum(tfidf.get(word, 0.0) for word in tokens) / len(tokens) if tokens else 0.0)
    sd = float(np.std(base))
    stale_digit = any(char.isdigit() for char in (" ".join(lower[-1]) if lower else ""))

    literal = []
    intended = []
    for sentence, low_tokens, case_tokens, base_score in zip(sentences, lower, cased, base):
        entity_density = len(entities(case_tokens)) / len(low_tokens) if low_tokens else 0.0
        normalized_text = " ".join(low_tokens) + (" " if low_tokens else "")
        current_digit = any(char.isdigit() for char in normalized_text)
        literal_legal = legal_density(low_tokens, normalized_text, "literal")
        intended_legal = legal_density(low_tokens, normalized_text, "intended")
        literal_score = base_score + sd * (
            0.2 * int(stale_digit) + 0.3 * entity_density + 1.5 * literal_legal
        )
        intended_score = base_score + sd * (
            0.2 * int(current_digit) + 0.3 * entity_density + 1.5 * intended_legal
        )
        literal.append((sentence, literal_score))
        intended.append((sentence, intended_score))

    # The upstream dictionary assignment overwrites duplicate sentence strings.
    def upstream_sort(pairs):
        scores = {}
        for sentence, score in pairs:
            scores[sentence] = score
        return [sentence for sentence, _ in sorted(scores.items(), key=lambda row: row[1], reverse=True)]

    return name, upstream_sort(literal), upstream_sort(intended), len(sentences)


def hard_budget(ranked, budget):
    text = " ".join(ranked)
    no_punctuation = text.translate(str.maketrans("", "", string.punctuation))
    # Upstream uses split(" "), including possible empty fields.
    tokens = no_punctuation.split(" ")
    return " ".join(tokens[:budget])


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    literal_dir = args.output_dir / "literal" / "predictions"
    intended_dir = args.output_dir / "intended" / "predictions"
    literal_dir.mkdir(parents=True, exist_ok=True)
    intended_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.judgements.glob("*.txt"))
    budgets = {}
    for line in args.length_file.read_text(encoding="utf-8").splitlines():
        name, value = line.split("\t")[:2]
        budgets[name] = int(value)
    literal_entries = args.dictionary.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    intended_entries = [line.strip() for line in literal_entries if line.strip()]

    df, cached = build_df(paths)
    context = mp.get_context("fork")
    initializer = (df, len(paths), literal_entries, intended_entries, cached)
    records = []
    with context.Pool(args.workers, initializer=init_worker, initargs=initializer) as pool:
        for position, (name, literal, intended, sentence_count) in enumerate(
            pool.imap_unordered(score_one, [path.name for path in paths]), 1
        ):
            budget = budgets[name]
            literal_text = hard_budget(literal, budget)
            intended_text = hard_budget(intended, budget)
            (literal_dir / name).write_text(literal_text, encoding="utf-8")
            (intended_dir / name).write_text(intended_text, encoding="utf-8")
            records.append({
                "file": name,
                "budget": budget,
                "source_sentences": sentence_count,
                "literal_output_fields": len(literal_text.split(" ")),
                "intended_output_fields": len(intended_text.split(" ")),
            })
            print(json.dumps({"phase": "score", "done": position, "total": len(paths), "file": name}), flush=True)

    records.sort(key=lambda row: row["file"])
    (args.output_dir / "documents.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    manifest = {
        "experiment": "CaseSummarizer UK-Abs official-code compatibility audit",
        "documents": len(records),
        "workers": args.workers,
        "variants": {
            "literal": "preserve dictionary newline and stale digit-variable behavior",
            "intended": "strip dictionary lines and calculate digit feature per sentence",
        },
        "common": [
            "released feature formula and weights",
            "test-corpus TF-IDF as in released program",
            "NLTK sentence/token/POS/NER pipeline",
            "descending score order",
            "reference word budget with upstream hard truncation",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
