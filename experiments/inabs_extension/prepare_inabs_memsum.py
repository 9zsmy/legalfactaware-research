#!/usr/bin/env python3
"""Prepare the frozen 6325/703 IN-Abs split for the official MemSum architecture.

The official 100-document test split is inventoried by the shared phase-1
loader but its contents are never loaded. Candidate selection and P50 budget
oracle trajectories are reused verbatim from the frozen phase-2 run.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:[.,:/-]\d+)*|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> str:
    return " ".join("<num>" if token[0].isdigit() else token.lower() for token in TOKEN_RE.findall(text))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert(case: dict, oracle: dict, require_positive: bool = True) -> dict:
    candidates = oracle["candidate_indices"] if oracle else None
    if candidates is None:
        raise RuntimeError("every train/dev case must have frozen candidate indices")
    original_to_local = {original: local for local, original in enumerate(candidates)}
    raw_sentences = [case["sentences"][index] for index in candidates]
    positives = [original_to_local[index] for index in oracle["budget"]]
    if require_positive and not positives:
        raise RuntimeError(f"empty P50 budget oracle: {case['id']}")
    reference_lines = [line.strip() for line in case["reference"].splitlines() if line.strip()]
    return {
        "id": case["id"],
        "text": [tokenize(sentence) for sentence in raw_sentences],
        "raw_text": raw_sentences,
        "source_indices": candidates,
        "summary": [tokenize(line) for line in reference_lines],
        "raw_summary": case["reference"],
        "indices": [positives],
        "score": [float(oracle["budget_score"])],
    }


def build_vocab(rows: list[dict], output: Path, size: int, dimension: int, seed: int) -> tuple[Path, Path]:
    counts: Counter[str] = Counter()
    for row in rows:
        for sentence in row["text"] + row["summary"]:
            counts.update(sentence.split())
    reserved = ["<pad>", "<eos>", "<unk>"]
    words = reserved + [token for token, _ in counts.most_common(size - len(reserved)) if token not in reserved]
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(0.0, 0.05, (len(words), dimension)).astype(np.float32)
    embeddings[0] = 0.0
    vocab_path = output / "vocabulary_english.pkl"
    embedding_path = output / f"random_embeddings_{dimension}d.pkl"
    with vocab_path.open("wb") as stream:
        pickle.dump(words, stream)
    with embedding_path.open("wb") as stream:
        pickle.dump(embeddings, stream)
    return vocab_path, embedding_path


def write_config(args: argparse.Namespace, train: Path, dev: Path, vocab: Path, embeddings: Path) -> None:
    config = {
        "training_corpus_file_name": str(train.resolve()),
        "validation_corpus_file_name": str(dev.resolve()),
        "model_folder": str((args.output / "model").resolve()),
        "log_folder": str((args.output / "log").resolve()),
        "vocabulary_file_name": str(vocab.resolve()),
        "pretrained_unigram_embeddings_file_name": str(embeddings.resolve()),
        "n_device": 1,
        "batch_size_per_device": args.batch_size,
        "max_extracted_sentences_per_document": 30,
        "p_stop_thres": 0.6,
        "moving_average_decay": 0.999,
        "max_seq_len": args.max_seq_len,
        "max_doc_len": 400,
        "num_of_epochs": args.epochs,
        "print_every": 100,
        "save_every": (6325 + args.batch_size - 1) // args.batch_size,
        "validate_every": (6325 + args.batch_size - 1) // args.batch_size,
        "restore_old_checkpoint": False,
        "learning_rate": 0.0001,
        "warmup_step": 1000,
        "weight_decay": 0.000001,
        "dropout_rate": 0.1,
        "num_heads": 8,
        "hidden_dim": args.hidden_dim,
        "N_enc_l": 2,
        "N_enc_g": 2,
        "N_dec": 2,
        "apply_length_normalization": 1,
    }
    (args.output / "training.config").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    smoke = dict(config)
    smoke.update({
        "training_corpus_file_name": str((args.output / "train_smoke.jsonl").resolve()),
        "validation_corpus_file_name": str((args.output / "dev_smoke.jsonl").resolve()),
        "model_folder": str((args.output / "smoke_model").resolve()),
        "log_folder": str((args.output / "smoke_log").resolve()),
        "batch_size_per_device": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_of_epochs": 1,
        "print_every": 1,
        "save_every": 1,
        "validate_every": 1,
    })
    (args.output / "smoke_training.config").write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-vocab", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.phase_root.resolve()))
    import run_inabs_phase1 as phase1

    part = args.phase_root / "dataset/IN-Abs/train-data"
    eligible, excluded = phase1.eligible_train_ids(part, phase1.paired_ids(part))
    train_ids, dev_ids = phase1.fixed_split(eligible)
    oracle_rows = read_jsonl(args.phase_root / "outputs/phase2_lr_full/train_oracles.jsonl")
    oracle_by_id = {row["id"]: row for row in oracle_rows}
    if [row["id"] for row in oracle_rows] != train_ids:
        raise RuntimeError("phase-2 oracle rows do not exactly match frozen train split")

    train = [convert(phase1.load_case(part, case_id), oracle_by_id[case_id]) for case_id in train_ids]
    # Dev needs the same deterministic candidate map, constructed without using references.
    dev = []
    for case_id in dev_ids:
        case = phase1.load_case(part, case_id)
        candidates = list(range(len(case["sentences"]))) if len(case["sentences"]) <= 400 else [
            round(i * (len(case["sentences"]) - 1) / 399) for i in range(400)
        ]
        dev.append(convert(case, {"candidate_indices": candidates, "budget": [], "budget_score": 0.0}, False))

    train_path, dev_path = args.output / "train.jsonl", args.output / "dev.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    # Stress the GPU with the longest oracle trajectories, since decoder memory
    # grows with the number of selected sentences and random shuffling can hide
    # the worst case in a prefix-only smoke test.
    stress_rows = sorted(train, key=lambda row: len(row["indices"][0]), reverse=True)
    write_jsonl(args.output / "train_smoke.jsonl", stress_rows[:max(4, args.batch_size)])
    write_jsonl(args.output / "dev_smoke.jsonl", dev[:2])
    vocab, embeddings = build_vocab(train, args.output, args.max_vocab, args.embed_dim, args.seed)
    write_config(args, train_path, dev_path, vocab, embeddings)
    manifest = {
        "method": "MemSum official architecture, IN-Abs retraining",
        "train_documents": len(train),
        "dev_documents": len(dev),
        "excluded_official_train": excluded,
        "official_test_contents_read": False,
        "candidate_cap": 400,
        "candidate_policy": "same reference-free evenly-spaced mapping as frozen phase 2",
        "supervision": "frozen P50 (643-word) budget oracle from phase 2",
        "train_positive_sentences": sum(len(row["indices"][0]) for row in train),
        "vocabulary_size": len(pickle.load(vocab.open("rb"))),
        "max_doc_len": 400,
        "max_seq_len": args.max_seq_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    (args.output / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
