#!/usr/bin/env python3
"""Prepare a 200-candidate IN-Abs sensitivity track for TermDiffuSum.

This adapter uses the already frozen 6,325/703 split and ordinary-oracle
labels.  The ``test`` file is deliberately a copy of a small dev subset for
engineering smoke tests; the official 100-document IN-Abs test is never read.
"""

from __future__ import annotations

import argparse
import itertools
import json
import string
from pathlib import Path

import spacy


def cycle_to_length(values, length=40):
    return list(itertools.islice(itertools.cycle(values), length))


def candidate_combinations(positive, sentence_count):
    main = positive[: min(40, len(positive))]
    negative = [index for index in range(sentence_count) if index not in set(positive)]
    mixed = (main[:20] + negative[:20])[:40] or main
    return [main, list(reversed(main)), main[::2] + main[1::2], mixed]


def spacy_word_count(doc):
    table = str.maketrans("", "", string.punctuation + string.whitespace)
    return sum(bool(token.text.translate(table)) for token in doc)


def convert(
    record,
    max_sentences,
    nlp,
    allow_decode_placeholder=False,
    placeholder_position="first",
):
    sentences = [str(value).strip() for value in record["sentences"][:max_sentences]]
    labels = [int(value) for value in record["labels"][: len(sentences)]]
    kept = [(sentence, label) for sentence, label in zip(sentences, labels) if sentence]
    sentences = [sentence for sentence, _ in kept]
    labels = [label for _, label in kept]
    positive = [index for index, label in enumerate(labels) if label == 1]
    if not sentences:
        return None
    placeholder = False
    if not positive:
        if not allow_decode_placeholder:
            return None
        placeholder_index = 0 if placeholder_position == "first" else len(sentences) - 1
        positive = [placeholder_index]
        labels[placeholder_index] = 1
        placeholder = True
    return {
        "id": str(record["id"]),
        "label": labels,
        "text": sentences,
        "summary": [sentences[index] for index in positive[:40]],
        "ext_idx": cycle_to_length(positive),
        "indices": candidate_combinations(positive, len(sentences)),
        "reference": record.get("summary", ""),
        "reference_words": spacy_word_count(nlp(record.get("summary", ""))),
        "sentence_word_counts": [
            spacy_word_count(doc) for doc in nlp.pipe(sentences, batch_size=128)
        ],
        "adapter": "inabs_ordinary_oracle_200_candidate_sensitivity",
        "decode_placeholder_index": placeholder,
    }


def load(
    path,
    max_sentences,
    nlp,
    max_docs=0,
    allow_decode_placeholder=False,
    placeholder_position="first",
):
    output = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            converted = convert(
                json.loads(line),
                max_sentences,
                nlp,
                allow_decode_placeholder,
                placeholder_position,
            )
            if converted is not None:
                output.append(converted)
            if max_docs and len(output) >= max_docs:
                break
    return output


def write(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sentences", type=int, default=200)
    parser.add_argument("--max-train-docs", type=int, default=0)
    parser.add_argument("--max-dev-docs", type=int, default=0)
    parser.add_argument("--smoke-test-docs", type=int, default=8)
    parser.add_argument(
        "--placeholder-position", choices=("first", "last"), default="first"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nlp = spacy.blank("en")
    train = load(args.train, args.max_sentences, nlp, args.max_train_docs)
    dev = load(
        args.dev,
        args.max_sentences,
        nlp,
        args.max_dev_docs,
        allow_decode_placeholder=True,
        placeholder_position=args.placeholder_position,
    )
    write(args.output_dir / "train.jsonl", train)
    write(args.output_dir / "val.jsonl", dev)
    write(args.output_dir / "test.jsonl", dev[: args.smoke_test_docs])
    manifest = {
        "experiment": "TermDiffuSum IN-Abs 200-candidate sensitivity adapter",
        "official_test_contents_read": False,
        "main_400_candidate_protocol": False,
        "max_sentences": args.max_sentences,
        "train_documents": len(train),
        "dev_documents": len(dev),
        "smoke_test_documents_from_dev": min(args.smoke_test_docs, len(dev)),
        "dev_placeholder_indices": len(dev),
        "placeholder_position": args.placeholder_position,
    }
    (args.output_dir / "adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
