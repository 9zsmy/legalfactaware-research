#!/usr/bin/env python3
"""Build the frozen 942-document dev input from the verification archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from protocol import read_jsonl, write_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=942)
    parser.add_argument("--target-chars", type=int, default=450)
    args = parser.parse_args()

    source_rows = read_jsonl(args.verification_jsonl)
    if len(source_rows) != args.expected_cases:
        raise RuntimeError(f"expected {args.expected_cases} rows, found {len(source_rows)}")

    dataset, predictions = [], []
    seen: set[str] = set()
    for index, row in enumerate(source_rows):
        case_id = str(row.get("id") or row.get("case_id") or index)
        if case_id in seen:
            raise RuntimeError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        source = str(row.get("document") or row.get("source") or "")
        reference = str(row.get("reference_summary") or row.get("reference") or "")
        prediction = str(row.get("generated_summary") or row.get("prediction") or "")
        if not source or not reference or not prediction:
            raise RuntimeError(f"missing source/reference/prediction at row {index}")
        dataset.append({
            "id": case_id,
            "split": "dev",
            "source": source,
            "reference": reference,
            "target_chars": args.target_chars,
        })
        predictions.append({
            "id": case_id,
            "method": "LegalFactAware-v2.5-B",
            "prediction": prediction,
            "characters": len(prediction),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "cail_dev_942.jsonl"
    prediction_path = args.output_dir / "legalfactaware_predictions.jsonl"
    write_jsonl(dataset_path, dataset)
    write_jsonl(prediction_path, predictions)
    manifest = {
        "status": "dev_only_test_untouched",
        "cases": len(dataset),
        "target_chars": args.target_chars,
        "source_file": str(args.verification_jsonl),
        "source_sha256": sha256(args.verification_jsonl),
        "dataset_sha256": sha256(dataset_path),
        "legalfactaware_predictions_sha256": sha256(prediction_path),
        "fields": ["id", "split", "source", "reference", "target_chars"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

