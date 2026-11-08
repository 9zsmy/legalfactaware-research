#!/usr/bin/env python3
"""Dev-only strict comparison of two feasible complete-sentence selectors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn import __version__ as sklearn_version

import run_legalfactaware_formal_ablation_dev as formal
import run_legalfactaware_v21_budget_decoder_dev as decoder
import run_legalfactaware_v2_cail_clean as base


EXPERIMENT = "Chinese-Feasible-Complete-Sentence-Selector-Control-20261020"
DEV_CASES = 942
MAX_CHARS = 450
MAX_SENTENCES = 8
EXPECTED = {
    "dev": "1a056c68dcfa5f11e9574d5f1b818fa475bb5607ea7f6565e2868072a55de2c0",
    "v22_model": "2cc0c9fe4b1fdaca6c1b2e37e545fcece3e5da83533296c1c12ba1d8bb8f70e9",
    "scorer": "38d6b031a1a6faa11a8b6f952f952ad06edfdd492f0ab7671b36055d875c548f",
    "archived_greedy": "1754389518ec912635d585d65b3e50c74202b29d82f80598dd9f6973b7e01f14",
}
METRICS = ("rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "reference_fact_f1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: {actual} != {expected}")
    return actual


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_knapsack(sentences: list[str], probabilities: list[float]):
    """Maximize summed normalized scores with <=8 sentences and <=450 chars."""
    values = np.asarray(base.minmax(probabilities), dtype=np.float64)
    lengths = np.asarray([len(sentence) for sentence in sentences], dtype=np.int32)
    count = len(sentences)
    negative = -np.inf
    dp = np.full((MAX_SENTENCES + 1, MAX_CHARS + 1), negative, dtype=np.float64)
    dp[0, 0] = 0.0
    take = np.zeros(
        (count, MAX_SENTENCES + 1, MAX_CHARS + 1), dtype=np.bool_
    )
    for index, (length, value) in enumerate(zip(lengths, values)):
        if length > MAX_CHARS:
            continue
        previous = dp.copy()
        for chosen in range(1, MAX_SENTENCES + 1):
            candidates = previous[chosen - 1, : MAX_CHARS + 1 - length] + value
            current = dp[chosen, length:]
            better = candidates > current + 1e-12
            if np.any(better):
                current[better] = candidates[better]
                take[index, chosen, length:][better] = True

    best_value = float(np.max(dp))
    states = np.argwhere(np.isclose(dp, best_value, atol=1e-12, rtol=0.0))
    # Deterministic secondary rules: more sentences, then less budget use.
    best_count = int(np.max(states[:, 0]))
    best_caps = states[states[:, 0] == best_count, 1]
    capacity = int(np.min(best_caps))
    chosen = best_count
    selected: list[int] = []
    for index in range(count - 1, -1, -1):
        if chosen <= 0:
            break
        length = int(lengths[index])
        if capacity >= length and take[index, chosen, capacity]:
            selected.append(index)
            capacity -= length
            chosen -= 1
    selected.sort()
    objective = float(sum(values[index] for index in selected))
    if len(selected) > MAX_SENTENCES:
        raise AssertionError("sentence cap violated")
    if sum(len(sentences[index]) for index in selected) > MAX_CHARS:
        raise AssertionError("character budget violated")
    if not math.isclose(objective, best_value, abs_tol=1e-9):
        raise AssertionError(f"knapsack reconstruction mismatch: {objective} != {best_value}")
    return selected, objective


def objective(indices: list[int], probabilities: list[float]) -> float:
    values = base.minmax(probabilities)
    return float(sum(values[index] for index in indices))


def generate(dev: list[dict], scores: list[list[float]], archived: list[dict]):
    greedy_predictions: list[dict] = []
    knapsack_predictions: list[dict] = []
    audit_rows: list[dict] = []
    for number, (row, probabilities, archived_row) in enumerate(zip(dev, scores, archived), 1):
        sentences = base.sentences(row)
        greedy, skipped = decoder.select_budget_greedy(
            sentences,
            probabilities,
            fact_weight=0.0,
            redundancy_weight=0.0,
            max_sentences=MAX_SENTENCES,
            max_chars=MAX_CHARS,
        )
        if decoder.row_id(row) != decoder.row_id(archived_row):
            raise RuntimeError(f"archived greedy ID mismatch at {number - 1}")
        if greedy != list(map(int, archived_row["selected_indices"])):
            raise RuntimeError(f"archived greedy selection mismatch at {number - 1}")
        knapsack, knapsack_value = exact_knapsack(sentences, probabilities)
        greedy_value = objective(greedy, probabilities)
        if knapsack_value + 1e-9 < greedy_value:
            raise AssertionError(f"knapsack objective below greedy at {number - 1}")
        greedy_row = decoder.prediction_row(
            row,
            greedy,
            decoder="budget_greedy",
            fact_weight=0.0,
            redundancy_weight=0.0,
            max_chars=MAX_CHARS,
            skipped_due_to_budget=skipped,
        )
        knapsack_row = decoder.prediction_row(
            row,
            knapsack,
            decoder="budget_greedy",
            fact_weight=0.0,
            redundancy_weight=0.0,
            max_chars=MAX_CHARS,
        )
        greedy_row["method"] = "Greedy fit/skip"
        knapsack_row["method"] = "Exact 0/1 knapsack"
        greedy_predictions.append(greedy_row)
        knapsack_predictions.append(knapsack_row)
        left, right = set(greedy), set(knapsack)
        audit_rows.append({
            "id": decoder.row_id(row),
            "greedy_indices": greedy,
            "knapsack_indices": knapsack,
            "selection_changed": greedy != knapsack,
            "selection_jaccard": len(left & right) / len(left | right) if left | right else 1.0,
            "greedy_objective": greedy_value,
            "knapsack_objective": knapsack_value,
            "objective_gain": knapsack_value - greedy_value,
            "greedy_characters": len(greedy_row["generated_summary"]),
            "knapsack_characters": len(knapsack_row["generated_summary"]),
        })
        if number % 100 == 0 or number == len(dev):
            print(f"selected feasible summaries {number}/{len(dev)}", flush=True)
    return greedy_predictions, knapsack_predictions, audit_rows


def paired_test(greedy: list[dict], knapsack: list[dict], metric: str) -> dict:
    if [r["id"] for r in greedy] != [r["id"] for r in knapsack]:
        raise RuntimeError("paired ID mismatch")
    differences = np.asarray(
        [float(b[metric]) - float(a[metric]) for a, b in zip(greedy, knapsack)],
        dtype=float,
    )
    rng = np.random.default_rng(42)
    bootstrap = np.empty(10_000, dtype=float)
    for start in range(0, 10_000, 500):
        stop = min(start + 500, 10_000)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        bootstrap[start:stop] = differences[indices].mean(axis=1)
    raw = 1.0 if np.allclose(differences, 0) else float(
        wilcoxon(differences, zero_method="pratt").pvalue
    )
    return {
        "metric": metric,
        "cases": len(differences),
        "mean_delta_knapsack_minus_greedy": float(differences.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wilcoxon_p_raw": raw if math.isfinite(raw) else 1.0,
        "knapsack_win_rate": float(np.mean(differences > 0)),
        "tie_rate": float(np.mean(np.isclose(differences, 0))),
    }


def holm_adjust(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["wilcoxon_p_raw"])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["wilcoxon_p_raw"]))
        rows[index]["wilcoxon_p_holm"] = running


def compact(name: str, result: dict) -> dict:
    row = result["lawa"]
    return {
        "selector": name,
        "R1": row["rouge_1_f1"],
        "R2": row["rouge_2_f1"],
        "RL": row["rouge_l_f1"],
        "Fact_F1_macro": row["reference_fact_f1"],
        "Avg_chars": row["characters"],
        "Avg_sentences": row["selected_sentences"],
        "Hard_truncations": row["hard_truncation_count"],
        "Budget_violations": row["budget_violations"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--v22-model", type=Path, required=True)
    parser.add_argument("--scorer-model", type=Path, required=True)
    parser.add_argument("--archived-greedy-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    for name in ("predictions", "per_sample", "tables"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    hashes = {
        "dev_sha256": require(args.dev, EXPECTED["dev"], "dev"),
        "v22_model_sha256": require(args.v22_model, EXPECTED["v22_model"], "v22 model"),
        "scorer_model_sha256": require(args.scorer_model, EXPECTED["scorer"], "scorer"),
        "archived_greedy_sha256": require(
            args.archived_greedy_predictions, EXPECTED["archived_greedy"], "archived greedy"
        ),
    }
    dev = base.read_jsonl(args.dev)
    archived = base.read_jsonl(args.archived_greedy_predictions)
    if len(dev) != DEV_CASES or len(archived) != DEV_CASES:
        raise RuntimeError("unexpected dev size")
    with args.v22_model.open("rb") as handle:
        vectorizer = pickle.load(handle)["vectorizer"]
    with args.scorer_model.open("rb") as handle:
        scorer = pickle.load(handle)
    score_rows, score_timing = formal.precompute_dev_scores(
        dev, {"frozen_full": scorer}, vectorizer
    )
    greedy_predictions, knapsack_predictions, audit_rows = generate(
        dev, score_rows["frozen_full"], archived
    )
    write_jsonl(args.output_dir / "predictions" / "greedy_fit_skip.jsonl", greedy_predictions)
    write_jsonl(args.output_dir / "predictions" / "exact_knapsack.jsonl", knapsack_predictions)
    write_jsonl(args.output_dir / "selection_audit.jsonl", audit_rows)

    results = {}
    details = {}
    for name, predictions in (
        ("greedy_fit_skip", greedy_predictions),
        ("exact_knapsack", knapsack_predictions),
    ):
        result = formal.evaluate_variant(name, predictions, dev, MAX_CHARS, args.output_dir)
        details[name] = result.pop("lawa_details")
        results[name] = result
        print(
            f"{name}: {result['lawa']['rouge_1_f1']:.6f}/"
            f"{result['lawa']['rouge_2_f1']:.6f}/"
            f"{result['lawa']['rouge_l_f1']:.6f}",
            flush=True,
        )

    significance = [
        paired_test(details["greedy_fit_skip"], details["exact_knapsack"], metric)
        for metric in METRICS
    ]
    holm_adjust(significance)
    changed = [row for row in audit_rows if row["selection_changed"]]
    audit = {
        "documents": len(audit_rows),
        "archived_greedy_exact_match": True,
        "selection_change_count": len(changed),
        "selection_change_rate": len(changed) / len(audit_rows),
        "mean_selection_jaccard": float(np.mean([r["selection_jaccard"] for r in audit_rows])),
        "mean_objective_gain": float(np.mean([r["objective_gain"] for r in audit_rows])),
        "minimum_objective_gain": float(np.min([r["objective_gain"] for r in audit_rows])),
        "strict_objective_improvement_rate": float(np.mean([r["objective_gain"] > 1e-12 for r in audit_rows])),
        "mean_greedy_characters": float(np.mean([r["greedy_characters"] for r in audit_rows])),
        "mean_knapsack_characters": float(np.mean([r["knapsack_characters"] for r in audit_rows])),
    }
    table = [
        compact("Greedy fit/skip", results["greedy_fit_skip"]),
        compact("Exact 0/1 knapsack", results["exact_knapsack"]),
    ]
    write_csv(args.output_dir / "tables" / "feasible_selector_comparison.csv", table)
    save_json(args.output_dir / "metrics.json", results)
    save_json(args.output_dir / "paired_significance.json", significance)
    save_json(args.output_dir / "selection_audit.json", audit)
    config = {
        "experiment": EXPERIMENT,
        "status": "dev_complete_test_untouched",
        "test_argument_exists": False,
        "single_changed_variable": "complete-sentence feasible selector",
        "selectors": ["greedy fit/skip", "exact 0/1 knapsack"],
        "shared_sentence_scores": "frozen TF-IDF + Legal14 + Section6 LR probabilities",
        "shared_score_normalization": "document-level min-max",
        "dynamic_redundancy_enabled": False,
        "fact_bonus_enabled": False,
        "max_chars": MAX_CHARS,
        "max_sentences": MAX_SENTENCES,
        "source_order_output": True,
        "dev_cases": DEV_CASES,
        "python": platform.python_version(),
        "sklearn": sklearn_version,
        "score_timing": score_timing,
        **hashes,
    }
    save_json(args.output_dir / "config.json", config)
    manifest = {
        **config,
        "script_sha256": sha256(Path(__file__).resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "selection_audit": audit,
        "primary_delta_knapsack_minus_greedy": {
            row["metric"]: row["mean_delta_knapsack_minus_greedy"] for row in significance
        },
        "outputs": {},
    }
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"][str(path.relative_to(args.output_dir))] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    save_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
