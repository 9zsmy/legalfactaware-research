#!/usr/bin/env python3
"""Recompute paired CAIL test comparisons for the reconstructed v2.5-B output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


METRICS = ("rouge_1_f1", "rouge_2_f1", "rouge_l_f1")
METHODS = (
    "LegalFactAware-v2-quality-deterministic",
    "MemSum-official-architecture-CN-3seed-mean",
    "SSE-v2-faithful-reimplementation-3seed-mean",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired(left: dict[str, dict], right: dict[str, dict], metric: str, seed: int = 42) -> dict:
    ids = list(left)
    if set(ids) != set(right) or len(ids) != 948:
        raise RuntimeError("paired identifiers do not match the 948-case test split")
    delta = np.asarray([float(left[i][metric]) - float(right[i][metric]) for i in ids])
    rng = np.random.default_rng(seed)
    boots = np.empty(10000, dtype=float)
    for start in range(0, 10000, 500):
        stop = min(start + 500, 10000)
        sample = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        boots[start:stop] = delta[sample].mean(axis=1)
    p = 1.0 if np.allclose(delta, 0.0) else float(wilcoxon(delta, zero_method="pratt").pvalue)
    if not math.isfinite(p):
        p = 1.0
    return {
        "metric": metric,
        "cases": len(ids),
        "mean_reconstructed": float(np.mean([float(left[i][metric]) for i in ids])),
        "mean_baseline": float(np.mean([float(right[i][metric]) for i in ids])),
        "mean_delta": float(delta.mean()),
        "bootstrap_95_ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "wilcoxon_p": p,
        "win_rate": float(np.mean(delta > 0)),
        "tie_rate": float(np.mean(delta == 0)),
        "bootstrap_samples": 10000,
        "seed": seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reconstructed", type=Path, required=True)
    ap.add_argument("--baselines", type=Path, required=True)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    reconstructed = {}
    with args.reconstructed.open(encoding="utf-8-sig") as f:
        for line in f:
            row = json.loads(line)
            reconstructed[str(row["id"])] = row
    if len(reconstructed) != 948:
        raise RuntimeError(f"expected 948 reconstructed cases, found {len(reconstructed)}")

    baselines = {name: {} for name in METHODS}
    with args.baselines.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row["method"]
            if name in baselines:
                baselines[name][str(row["id"])] = row
    for name, rows in baselines.items():
        if len(rows) != 948:
            raise RuntimeError(f"{name}: expected 948 cases, found {len(rows)}")

    comparisons = []
    for name in METHODS:
        for metric in METRICS:
            row = paired(reconstructed, baselines[name], metric)
            row["baseline"] = name
            comparisons.append(row)
    adjusted = holm([row["wilcoxon_p"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["wilcoxon_p_holm_9"] = value
        row["supported"] = (
            row["bootstrap_95_ci"][0] > 0 and value < 0.05
        )

    archive = json.loads(args.archive.read_text(encoding="utf-8"))
    archived = {
        (row["method_b"], row["metric"]): row
        for row in archive["comparisons"]
        if row["metric"] in METRICS
    }
    archived_names = {
        "LegalFactAware-v2-quality-deterministic": "LegalFactAware-v2-original",
        "MemSum-official-architecture-CN-3seed-mean": "MemSum-CN-3seed-mean",
        "SSE-v2-faithful-reimplementation-3seed-mean": "SSE-v2-3seed-mean",
    }
    maximum = 0.0
    for row in comparisons:
        prior = archived[(archived_names[row["baseline"]], row["metric"])]
        row["archived_mean_delta"] = float(prior["mean_difference"])
        row["reconstructed_minus_archived_delta"] = row["mean_delta"] - float(prior["mean_difference"])
        maximum = max(maximum, abs(row["reconstructed_minus_archived_delta"]))

    result = {
        "status": "paired_test_reconstruction_complete",
        "scope": (
            "Reconstructed LegalFactAware-v2.5-B versus three locally preserved, ID-aligned "
            "test baselines. SENDE is excluded because its aligned predictions remain unavailable."
        ),
        "input_hashes": {
            "reconstructed_per_sample_sha256": sha256(args.reconstructed),
            "baseline_per_sample_sha256": sha256(args.baselines),
            "archived_comparison_sha256": sha256(args.archive),
        },
        "multiple_testing_family": "3 baselines x 3 ROUGE metrics = 9 tests",
        "maximum_absolute_change_from_archived_mean_delta": maximum,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
