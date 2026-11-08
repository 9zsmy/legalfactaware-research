#!/usr/bin/env python3
"""Prepare split-safe Gist and SummaRuNNer inputs without reading official test."""
from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_inabs_phase1 as phase1
import run_inabs_phase2_lr as phase2

import spacy


def official_count(doc) -> int:
    table = str.maketrans("", "", string.punctuation + string.whitespace)
    return sum(len(token.text.translate(table)) > 0 for token in doc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--train-oracles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if phase1.md5(args.archive) != phase1.EXPECTED_ARCHIVE_MD5:
        raise RuntimeError("official archive MD5 mismatch")

    part = args.dataset_dir / "train-data"
    eligible, excluded = phase1.eligible_train_ids(part, phase1.paired_ids(part))
    train_ids, dev_ids = phase1.fixed_split(eligible)
    if len(train_ids) != 6325 or len(dev_ids) != 703:
        raise RuntimeError("frozen split mismatch")
    oracle_rows = [json.loads(line) for line in args.train_oracles.read_text().splitlines() if line.strip()]
    oracle_by_id = {row["id"]: row for row in oracle_rows}
    if set(oracle_by_id) != set(train_ids):
        raise RuntimeError("training oracle IDs do not match frozen train IDs")
    if args.train_limit:
        train_ids = train_ids[:args.train_limit]
    if args.dev_limit:
        dev_ids = dev_ids[:args.dev_limit]

    gist_train = args.output_dir / "gist_labelled_train"
    gist_dev = args.output_dir / "dev_candidates"
    gist_train.mkdir()
    gist_dev.mkdir()
    train_json = (args.output_dir / "summarunner_train.jsonl").open("w", encoding="utf-8")
    for position, case_id in enumerate(train_ids, 1):
        case = phase1.load_case(part, case_id)
        oracle = oracle_by_id[case_id]
        candidates = oracle["candidate_indices"]
        if candidates != phase2.candidate_indices(len(case["sentences"])):
            raise RuntimeError(f"candidate mismatch for {case_id}")
        positives = set(oracle["ordinary"])
        sentences = [case["sentences"][index] for index in candidates]
        labels = [int(index in positives) for index in candidates]
        (gist_train / f"{case_id}.txt").write_text(
            "\n".join(f"{sentence}$$${label}" for sentence, label in zip(sentences, labels)),
            encoding="utf-8",
        )
        train_json.write(json.dumps({
            "id": case_id, "doc": "\n".join(sentences),
            "labels": "\n".join(map(str, labels)), "summaries": "",
            "indices": candidates,
        }) + "\n")
        if position % 250 == 0 or position == len(train_ids):
            print(json.dumps({"stage": "prepare_train", "done": position, "total": len(train_ids)}), flush=True)
    train_json.close()

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    dev_json = (args.output_dir / "summarunner_dev.jsonl").open("w", encoding="utf-8")
    length_stream = (args.output_dir / "dev_lengths.tsv").open("w", encoding="utf-8")
    for position, case_id in enumerate(dev_ids, 1):
        case = phase1.load_case(part, case_id)
        candidates = phase2.candidate_indices(len(case["sentences"]))
        sentences = [case["sentences"][index] for index in candidates]
        budget = official_count(nlp(case["reference"]))
        (gist_dev / f"{case_id}.txt").write_text("\n".join(sentences), encoding="utf-8")
        length_stream.write(f"{case_id}.txt\t{budget}\n")
        dev_json.write(json.dumps({
            "id": case_id, "file": f"{case_id}.txt", "doc": "\n".join(sentences),
            "labels": "\n".join("0" for _ in sentences), "summaries": "",
            "indices": candidates, "budget": budget,
        }) + "\n")
        if position % 50 == 0 or position == len(dev_ids):
            print(json.dumps({"stage": "prepare_dev", "done": position, "total": len(dev_ids)}), flush=True)
    dev_json.close()
    length_stream.close()
    manifest = {
        "train_documents": len(train_ids), "dev_documents": len(dev_ids),
        "official_test_contents_read": False,
        "candidate_rule": "reference-free evenly-spaced maximum 400 sentences",
        "training_labels": "frozen Phase 2 ordinary summary-level oracle",
        "excluded": excluded,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"stage": "complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
