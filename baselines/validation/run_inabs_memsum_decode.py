#!/usr/bin/env python3
"""Decode IN-Abs dev with a trained official-architecture MemSum checkpoint."""
from __future__ import annotations

import argparse
import json
import string
import sys
import time
from pathlib import Path

import spacy


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def count_words(doc) -> int:
    table = str.maketrans("", "", string.punctuation + string.whitespace)
    return sum(bool(token.text.translate(table)) for token in doc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--p-stop", type=float, default=0.6)
    parser.add_argument("--max-extracted", type=int, default=30)
    args = parser.parse_args()

    memsum_root = args.project / "third_party/MemSum_acl2022/memsum_demo"
    sys.path.insert(0, str(memsum_root))
    from my_summarizers import ExtractiveSummarizer_MemSum_Final

    summarizer = ExtractiveSummarizer_MemSum_Final(
        str(args.model), str(args.vocabulary), gpu=args.gpu,
        embed_dim=128, hidden_dim=512, N_enc_l=2, N_enc_g=2, N_dec=2,
        max_doc_len=400, max_seq_len=128,
    )
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    predictions = args.output_dir / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "decode_records.jsonl"
    started = time.perf_counter()
    rows = list(read_jsonl(args.corpus))
    with records_path.open("w", encoding="utf-8") as records:
        for number, row in enumerate(rows, 1):
            result = summarizer.extract(
                [row["text"]], p_stop_thres=args.p_stop,
                ngram_blocking=True, ngram=4, return_sentence_position=True,
                max_extracted_sentences_per_document=args.max_extracted,
            )
            selection_order = [int(value) for value in result[1][0]]
            texts = [row["raw_summary"]] + [row["raw_text"][index] for index in selection_order]
            tokenized = list(nlp.pipe(texts))
            budget = count_words(tokenized[0])
            lengths = [count_words(doc) for doc in tokenized[1:]]
            chosen, used = [], 0
            for local_index, length in zip(selection_order, lengths):
                if length and used + length <= budget:
                    chosen.append(local_index)
                    used += length
            chosen.sort()
            prediction = "\n".join(row["raw_text"][index] for index in chosen)
            actual = count_words(nlp(prediction))
            if actual > budget:
                raise RuntimeError(f"budget violation: {row['id']} {actual}>{budget}")
            (predictions / f"{row['id']}.txt").write_text(prediction, encoding="utf-8")
            record = {
                "id": row["id"],
                "candidate_count": len(row["raw_text"]),
                "selection_order_local": selection_order,
                "selected_local": chosen,
                "selected_source_indices": [row["source_indices"][index] for index in chosen],
                "reference_spacy_words": budget,
                "prediction_spacy_words": actual,
                "budget_violation": False,
                "p_stop": args.p_stop,
                "max_extracted": args.max_extracted,
            }
            records.write(json.dumps(record) + "\n")
            records.flush()
            if number % 10 == 0 or number == len(rows):
                print(json.dumps({
                    "stage": "decode", "done": number, "total": len(rows),
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                }), flush=True)
    manifest = {
        "method": "MemSum official architecture, IN-Abs P50-budget adaptation",
        "checkpoint": str(args.model),
        "documents": len(rows),
        "p_stop": args.p_stop,
        "max_extracted_sentences": args.max_extracted,
        "complete_sentence_budget": True,
        "budget_violations": 0,
        "official_test_contents_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "decode_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
