#!/usr/bin/env python3
"""Prepare the corrected CAIL split for a same-protocol MemSum retrain."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SUMMARY_ENDINGS = set("。！？；")
SEEDS = (20260814, 20260815, 20260816)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_summary(summary: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for character in summary:
        if character.isspace():
            continue
        current.append(character)
        if character in SUMMARY_ENDINGS:
            parts.append(" ".join(current))
            current = []
    if current:
        parts.append(" ".join(current))
    return parts or ([" ".join(summary)] if summary else [])


def rouge1_f1(reference_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not reference_tokens or not candidate_tokens:
        return 0.0
    overlap = sum((Counter(reference_tokens) & Counter(candidate_tokens)).values())
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def convert(row: dict) -> dict:
    positives = [index for index, label in enumerate(row["global_labels"]) if label]
    if not positives:
        positives = sorted({index for indices in row.get("part_indices", []) for index in indices})
    selected_tokens = [
        token
        for index in positives
        if index < len(row["text"])
        for token in row["text"][index].split()
    ]
    summary_tokens = [character for character in row["raw_summary"] if not character.isspace()]
    return {
        "id": row["id"],
        "text": row["text"],
        "raw_text": row["raw_text"],
        "summary": split_summary(row["raw_summary"]),
        "raw_summary": row["raw_summary"],
        "indices": [positives],
        "score": [rouge1_f1(summary_tokens, selected_tokens)],
        "label_source": "CAIL supplied global sentence labels; part-oracle fallback only when empty",
    }


def ids(rows: list[dict]) -> set[str]:
    return {str(row["id"]) for row in rows}


def training_config(
    *,
    train: Path,
    dev: Path,
    vocabulary: Path,
    embeddings: Path,
    output: Path,
    seed: int,
    train_cases: int,
) -> dict:
    batches_per_epoch = train_cases // 4
    return {
        "training_corpus_file_name": str(train),
        "validation_corpus_file_name": str(dev),
        "model_folder": str(output / "model"),
        "log_folder": str(output / "log"),
        "vocabulary_file_name": str(vocabulary),
        "pretrained_unigram_embeddings_file_name": str(embeddings),
        "n_device": 1,
        "batch_size_per_device": 4,
        "max_extracted_sentences_per_document": 7,
        "p_stop_thres": 0.6,
        "moving_average_decay": 0.999,
        "max_seq_len": 128,
        "max_doc_len": 512,
        "num_of_epochs": 10,
        "print_every": 100,
        "save_every": batches_per_epoch,
        "validate_every": batches_per_epoch,
        "restore_old_checkpoint": False,
        "learning_rate": 0.0001,
        "warmup_step": 1000,
        "weight_decay": 0.000001,
        "dropout_rate": 0.1,
        "num_heads": 8,
        "hidden_dim": 512,
        "N_enc_l": 2,
        "N_enc_g": 2,
        "N_dec": 2,
        "apply_length_normalization": 1,
        "pretokenized_rouge": True,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dev-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        "train": args.train_dev_dir / "train.jsonl",
        "dev": args.train_dev_dir / "dev.jsonl",
        "test": args.test_dir / "test.jsonl",
    }
    converted: dict[str, list[dict]] = {}
    outputs: dict[str, Path] = {}
    for split, source in sources.items():
        converted[split] = [convert(row) for row in read_jsonl(source)]
        outputs[split] = args.output_dir / f"{split}.jsonl"
        write_jsonl(outputs[split], converted[split])

    expected = {"train": 7495, "dev": 942, "test": 948}
    actual = {split: len(rows) for split, rows in converted.items()}
    if actual != expected:
        raise RuntimeError(f"unexpected split sizes: {actual}, expected {expected}")
    if ids(converted["train"]) & ids(converted["dev"]):
        raise RuntimeError("train/dev ID overlap")
    if ids(converted["train"]) & ids(converted["test"]):
        raise RuntimeError("train/test ID overlap")
    if ids(converted["dev"]) & ids(converted["test"]):
        raise RuntimeError("dev/test ID overlap")

    vocabulary = args.train_dev_dir / "vocabulary.pkl"
    embeddings = args.train_dev_dir / "random_embeddings_128d.pkl"
    configs = {}
    for seed in SEEDS:
        seed_dir = args.artifact_root / f"memsum_cail_strict_v2_seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        config_path = seed_dir / "training.config"
        config_path.write_text(
            json.dumps(
                training_config(
                    train=outputs["train"],
                    dev=outputs["dev"],
                    vocabulary=vocabulary,
                    embeddings=embeddings,
                    output=seed_dir,
                    seed=seed,
                    train_cases=actual["train"],
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        configs[str(seed)] = str(config_path)

    smoke_dir = args.artifact_root / "memsum_cail_strict_v2_smoke"
    smoke_train = args.output_dir / "train_smoke.jsonl"
    smoke_dev = args.output_dir / "dev_smoke.jsonl"
    write_jsonl(smoke_train, converted["train"][:8])
    write_jsonl(smoke_dev, converted["dev"][:4])
    smoke_config = training_config(
        train=smoke_train,
        dev=smoke_dev,
        vocabulary=vocabulary,
        embeddings=embeddings,
        output=smoke_dir,
        seed=SEEDS[0],
        train_cases=8,
    )
    smoke_config.update(
        {
            "max_doc_len": 64,
            "max_seq_len": 64,
            "hidden_dim": 64,
            "num_of_epochs": 1,
            "print_every": 1,
            "save_every": 2,
            "validate_every": 2,
        }
    )
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = smoke_dir / "training.config"
    smoke_path.write_text(json.dumps(smoke_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "identity": "MemSum ACL 2022 architecture, corrected CAIL strict-split Chinese adaptation",
        "split_sizes": actual,
        "sources": {
            split: {"path": str(path), "sha256": sha256(path)} for split, path in sources.items()
        },
        "prepared": {
            split: {"path": str(path), "sha256": sha256(path)} for split, path in outputs.items()
        },
        "vocabulary": {"path": str(vocabulary), "sha256": sha256(vocabulary)},
        "embeddings": {"path": str(embeddings), "sha256": sha256(embeddings)},
        "seeds": list(SEEDS),
        "configs": configs,
        "smoke_config": str(smoke_path),
        "limitations": [
            "CAIL supplied global sentence labels are used as one training episode per case.",
            "This is a Chinese adaptation of official MemSum, not an author-released CAIL checkpoint.",
        ],
    }
    manifest_path = args.output_dir / "preparation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
