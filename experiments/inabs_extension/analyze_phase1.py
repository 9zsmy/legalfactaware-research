#!/usr/bin/env python3
"""Paired diagnostics for the completed IN-Abs phase-1 dev run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np


METRICS = ("rouge1", "rouge2", "rougeLsum")
COMPARISONS = (
    ("budget_oracle", "posthoc_oracle"),
    ("budget_oracle", "lexrank"),
    ("lexrank", "lead"),
)


def bootstrap(deltas: np.ndarray, samples: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(deltas), size=(count, len(deltas)))
        means.append(deltas[indices].mean(axis=1))
    values = np.concatenate(means)
    p = min(1.0, 2.0 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))))
    return {
        "samples": samples,
        "seed": seed,
        "mean_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "two_sided_bootstrap_p": p,
        "wins": int(np.sum(deltas > 1e-12)),
        "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
        "losses": int(np.sum(deltas < -1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_key = {(row["id"], row["budget_name"], row["system"]): row for row in rows}
    budgets = sorted({row["budget_name"] for row in rows}, key=lambda x: int(x[1:]))
    ids = sorted({row["id"] for row in rows}, key=lambda x: (int(x) if x.isdigit() else 10**18, x))
    report = {"cases": len(ids), "comparisons": {}}
    for budget in budgets:
        report["comparisons"][budget] = {}
        for left_name, right_name in COMPARISONS:
            left = [by_key[(case_id, budget, left_name)] for case_id in ids]
            right = [by_key[(case_id, budget, right_name)] for case_id in ids]
            key = f"{left_name}_minus_{right_name}"
            item = {
                "selection_changed_cases": int(sum(a["selected_indices"] != b["selected_indices"] for a, b in zip(left, right))),
                "selection_changed_fraction": float(np.mean([a["selected_indices"] != b["selected_indices"] for a, b in zip(left, right)])),
                "prediction_word_delta": float(np.mean([a["prediction_words"] - b["prediction_words"] for a, b in zip(left, right)])),
                "metrics": {},
            }
            for metric in METRICS:
                deltas = np.asarray([a[metric] - b[metric] for a, b in zip(left, right)], dtype=float)
                item["metrics"][metric] = bootstrap(deltas)
            report["comparisons"][budget][key] = item
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
