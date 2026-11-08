#!/usr/bin/env python3
"""Recompute manuscript-facing paired statistics from the frozen dev archive.

The script deliberately reads only the manifest-listed per-document Lawa
evaluation files.  It verifies file hashes, case counts, unique IDs, and row
alignment before recomputing paired bootstrap intervals and Wilcoxon tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


METRICS = ("rouge_1_f1", "rouge_2_f1", "rouge_l_f1")
VARIANTS = (
    "minus_oracle",
    "minus_legal14",
    "minus_section6",
    "tfidf_only",
    "post_truncation",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_test(full: list[dict], ablated: list[dict], metric: str) -> dict:
    differences = np.asarray(
        [float(left[metric]) - float(right[metric]) for left, right in zip(full, ablated)],
        dtype=float,
    )
    rng = np.random.default_rng(42)
    bootstrap = np.empty(10_000, dtype=float)
    for start in range(0, 10_000, 500):
        stop = min(start + 500, 10_000)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        bootstrap[start:stop] = differences[indices].mean(axis=1)
    if np.allclose(differences, 0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(wilcoxon(differences, zero_method="pratt").pvalue)
    return {
        "metric": metric,
        "cases": len(differences),
        "mean_delta_full_minus_variant": float(differences.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wilcoxon_p_raw": wilcoxon_p,
        "full_win_rate": float(np.mean(differences > 0)),
        "tie_rate": float(np.mean(np.isclose(differences, 0))),
        "bootstrap_samples": 10_000,
        "seed": 42,
        "wilcoxon_zero_method": "pratt",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_cases = int(manifest["dev_cases"])
    files = {"full": args.archive / "per_sample" / "full_lawa.jsonl"}
    files.update({name: args.archive / "per_sample" / f"{name}_lawa.jsonl" for name in VARIANTS})

    rows: dict[str, list[dict]] = {}
    file_audit: dict[str, dict] = {}
    for name, path in files.items():
        relative = str(path.relative_to(args.archive))
        expected = manifest["outputs"][relative]["sha256"]
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch for {relative}: {actual} != {expected}")
        current = read_jsonl(path)
        ids = [str(row["id"]) for row in current]
        if len(current) != expected_cases or len(set(ids)) != expected_cases:
            raise RuntimeError(f"case/ID mismatch for {relative}")
        rows[name] = current
        file_audit[name] = {
            "path": str(path),
            "sha256": actual,
            "cases": len(current),
            "unique_ids": len(set(ids)),
        }

    full_ids = [str(row["id"]) for row in rows["full"]]
    for name in VARIANTS:
        if [str(row["id"]) for row in rows[name]] != full_ids:
            raise RuntimeError(f"row-order mismatch: full vs {name}")

    results: list[dict] = []
    for variant in VARIANTS:
        for metric in METRICS:
            result = paired_test(rows["full"], rows[variant], metric)
            result["variant"] = variant
            results.append(result)

    adjusted = holm_adjust([row["wilcoxon_p_raw"] for row in results])
    for row, value in zip(results, adjusted):
        row["wilcoxon_p_holm_15_rouge_tests"] = value
        low, high = row["bootstrap_95_ci"]
        row["supported_at_0_05"] = bool(low > 0 and high > 0 and value < 0.05)

    payload = {
        "audit_date": "2026-10-10",
        "source_experiment": manifest["experiment"],
        "source_manifest_sha256": sha256(manifest_path),
        "train_cases": manifest["train_cases"],
        "dev_cases": expected_cases,
        "train_sha256": manifest["train_sha256"],
        "dev_sha256": manifest["dev_sha256"],
        "script_sha256": manifest["script_sha256"],
        "file_audit": file_audit,
        "multiple_testing_family": "5 variants x 3 primary ROUGE metrics",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
