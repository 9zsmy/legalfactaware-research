#!/usr/bin/env python3
"""Prepare a frozen, leakage-controlled IN-Abs DELTA pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path


PILOT_TAG = "delta-inabs-pilot-v1"
TARGET_WORDS = 643
EXPECTED_TRAIN = 6325
EXPECTED_DEV = 703


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    args = parser.parse_args()

    sys.path.insert(0, str(args.phase_root))
    import run_inabs_phase1 as phase1

    part = args.phase_root / "dataset/IN-Abs/train-data"
    eligible, excluded = phase1.eligible_train_ids(part, phase1.paired_ids(part))
    train_ids, dev_ids = phase1.fixed_split(eligible)
    if len(train_ids) != EXPECTED_TRAIN or len(dev_ids) != EXPECTED_DEV:
        raise RuntimeError(
            f"unexpected frozen split: train={len(train_ids)} dev={len(dev_ids)}"
        )

    ratios = []
    for case_id in train_ids:
        reference = (part / "summary" / f"{case_id}.txt").read_text(
            encoding="utf-8", errors="ignore"
        ).strip()
        words = len(phase1.word_tokens(reference))
        if words:
            ratios.append(len(reference) / words)
    chars_per_word = float(statistics.median(ratios))
    target_chars = int(round(TARGET_WORDS * chars_per_word))

    ranked = sorted(
        dev_ids,
        key=lambda case_id: (
            hashlib.sha256(f"{PILOT_TAG}:{case_id}".encode()).hexdigest(),
            case_id,
        ),
    )
    pilot_ids = ranked[: args.num_samples]
    inputs, references = [], []
    for rank, case_id in enumerate(pilot_ids, 1):
        case = phase1.load_case(part, case_id)
        source = "\n".join(case["sentences"])
        inputs.append(
            {
                "id": case_id,
                "pilot_rank": rank,
                "source": source,
                "source_words": case["source_words"],
                "target_words": TARGET_WORDS,
                "target_chars": target_chars,
            }
        )
        references.append(
            {
                "id": case_id,
                "reference": case["reference"],
                "reference_words": case["reference_words"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "pilot50_input.jsonl"
    reference_path = args.output_dir / "pilot50_references.jsonl"
    write_jsonl(input_path, inputs)
    write_jsonl(reference_path, references)
    manifest = {
        "pilot_tag": PILOT_TAG,
        "pilot_size": len(pilot_ids),
        "pilot_ids_in_run_order": pilot_ids,
        "official_test_contents_read": False,
        "frozen_split": {"train": len(train_ids), "dev": len(dev_ids)},
        "excluded": excluded,
        "target_words": TARGET_WORDS,
        "train_reference_chars_per_word_median": chars_per_word,
        "target_chars": target_chars,
        "input_sha256": sha256_bytes(input_path.read_bytes()),
        "references_sha256": sha256_bytes(reference_path.read_bytes()),
    }
    (args.output_dir / "pilot50_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

