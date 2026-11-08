#!/usr/bin/env python3
"""Published-algorithm audit of linear DSDR on UK-Abs.

Primary configuration follows He et al. (AAAI 2012), Algorithm 1, using the
public reference implementation's lambda=0.1.  UK-Abs decoding uses complete
sentences and each document's reference-summary word budget.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import string
import time
from pathlib import Path

import numpy as np
import spacy
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


PUNCT_WS = str.maketrans("", "", string.punctuation + string.whitespace)
STEMMER = PorterStemmer()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--length-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lambda-value", type=float, default=0.1)
    parser.add_argument(
        "--representation",
        choices=("tfidf_l2", "tfidf_none", "tf_l2", "tf_none"),
        default="tfidf_l2",
    )
    return parser.parse_args()


def stem_tokenizer(text):
    return [STEMMER.stem(token) for token in re.findall(r"[A-Za-z]+", text.lower())]


def official_count_doc(doc):
    return sum(len(token.text.translate(PUNCT_WS)) > 0 for token in doc)


def select_dsdr(sentences, lengths, budget, lambda_value, representation):
    if not sentences:
        return []
    try:
        use_idf = representation.startswith("tfidf")
        norm = "l2" if representation.endswith("l2") else None
        vectors = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            tokenizer=stem_tokenizer,
            token_pattern=None,
            ngram_range=(1, 1),
            norm=norm,
            use_idf=use_idf,
        ).fit_transform(sentences)
        gram = (vectors @ vectors.T).toarray().astype(np.float32, copy=False)
    except ValueError:
        gram = np.eye(len(sentences), dtype=np.float32)

    matrix = gram / np.float32(lambda_value)
    available = np.asarray([length > 0 and length <= budget for length in lengths], dtype=bool)
    selected = []
    used = 0
    while np.any(available):
        fitting = available & (np.asarray(lengths) + used <= budget)
        if not np.any(fitting):
            break
        scores = np.einsum("ij,ij->j", matrix, matrix) / (1.0 + np.diag(matrix))
        scores[~fitting] = -np.inf
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            break
        selected.append(best)
        used += lengths[best]
        available[best] = False
        column = matrix[:, best].copy()
        denominator = np.float32(1.0 + matrix[best, best])
        matrix -= np.outer(column, column) / denominator
    # Algorithm 1 and the public reference implementation return the greedy
    # selection list L directly; do not restore source order in this audit.
    return selected


def worker(task):
    name, judgement_path, budget, lambda_value, representation = task
    text = Path(judgement_path).read_text(encoding="utf-8", errors="ignore")
    sentences = [line.strip() for line in text.splitlines() if line.strip()]
    nlp = spacy.blank("en")
    lengths = [official_count_doc(doc) for doc in nlp.pipe(sentences, batch_size=128)]
    selected = select_dsdr(sentences, lengths, budget, lambda_value, representation)
    prediction = " ".join(sentences[index] for index in selected)
    return {
        "file": name,
        "source_sentences": len(sentences),
        "selected_indices": selected,
        "selected_sentences": len(selected),
        "budget": budget,
        "used_words": sum(lengths[index] for index in selected),
        "prediction": prediction,
    }


def main():
    args = parse_args()
    predictions = args.output_dir / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    budgets = {}
    for line in args.length_file.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        budgets[fields[0]] = int(fields[1])

    paths = sorted(args.judgements.glob("*.txt"))
    tasks = [
        (path.name, str(path), budgets[path.name], args.lambda_value, args.representation)
        for path in paths
    ]
    started = time.time()
    records = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for position, record in enumerate(pool.imap_unordered(worker, tasks), 1):
            (predictions / record["file"]).write_text(record.pop("prediction"), encoding="utf-8")
            records.append(record)
            print(json.dumps({
                "phase": "select",
                "done": position,
                "total": len(tasks),
                "file": record["file"],
                "source_sentences": record["source_sentences"],
                "selected_sentences": record["selected_sentences"],
            }), flush=True)

    records.sort(key=lambda row: row["file"])
    (args.output_dir / "documents.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    manifest = {
        "experiment": "DSDR linear reconstruction UK-Abs published-algorithm audit",
        "documents": len(records),
        "algorithm": "He et al. 2012 Algorithm 1, linear reconstruction",
        "representation": args.representation,
        "lambda": args.lambda_value,
        "decoding": "complete sentences fitting per-document reference word budget; Algorithm-1 greedy selection order",
        "budget_violations": sum(row["used_words"] > row["budget"] for row in records),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
