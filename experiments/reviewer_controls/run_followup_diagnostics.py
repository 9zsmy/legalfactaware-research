#!/usr/bin/env python3
"""Post-run diagnostics for the LegalFactAware reviewer controls (dev only).

This script does not retrain a model.  It explains residual ambiguities in the
completed run: label-set length/role preferences, realized-output-length-close
subsets, and expression performance by type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import wilcoxon


SEED = 42
TRAIN_SHA256 = "8b19e6a1aa086e55c3a1d694e86cb4cf5abfe6929ee1258a3621b291bde564a2"
DEV_SHA256 = "1a056c68dcfa5f11e9574d5f1b818fa475bb5607ea7f6565e2868072a55de2c0"
ORACLE_SHA256 = "de20bf0a98cb04de5f11f2b0869701fa5c940a392f5804f0b85a09fad5f19960"
PAIR_METRICS = ("rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "reference_fact_f1")


# ---------------------------------------------------------------------------
# Reproducibility and small statistical helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripts-dir", type=Path, required=True)
    parser.add_argument("--formal-script", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--oracle-extracts", type=Path, required=True)
    parser.add_argument("--full-predictions", type=Path, required=True)
    parser.add_argument("--matched-predictions", type=Path, required=True)
    parser.add_argument("--drop-predictions", type=Path, required=True)
    parser.add_argument("--legacy-predictions", type=Path, required=True)
    parser.add_argument("--full-details", type=Path, required=True)
    parser.add_argument("--matched-details", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: {actual} != {expected}")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
    }


def paired_summary(differences: list[float]) -> dict:
    array = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(10_000, dtype=float)
    for start in range(0, 10_000, 500):
        stop = min(start + 500, 10_000)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        bootstrap[start:stop] = array[indices].mean(axis=1)
    raw = 1.0 if np.allclose(array, 0) else float(wilcoxon(array, zero_method="pratt").pvalue)
    return {
        "n": int(array.size),
        "mean_delta_left_minus_right": float(array.mean()),
        "bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "wilcoxon_p_raw": raw if math.isfinite(raw) else 1.0,
    }


def label_diagnostics(base, section_impl, decoder, train: list[dict], oracle_rows: list[dict]) -> dict:
    """Compare length, normalized position and section mix of positive labels."""
    values = {
        "oracle": {"length": [], "normalized_position": [], "section": Counter()},
        "matched": {"length": [], "normalized_position": [], "section": Counter()},
    }
    per_doc_length_delta, per_doc_position_delta = [], []
    for row, oracle_row in zip(train, oracle_rows):
        if decoder.row_id(row) != decoder.row_id(oracle_row):
            raise RuntimeError("train/oracle ID mismatch")
        sentences = base.sentences(row)
        sections = section_impl.section_labels(sentences)
        oracle_indices = sorted(map(int, oracle_row["oracle_selected_indices"]))
        k = len(oracle_indices)
        reference_counts = base.bigrams(base.tokens(base.reference(row)))
        standalone = [base.rouge2_precision(sentence, reference_counts) for sentence in sentences]
        matched_indices = sorted(sorted(range(len(sentences)), key=lambda i: (-standalone[i], i))[:k])
        doc_means = {}
        for name, indices in (("oracle", oracle_indices), ("matched", matched_indices)):
            lengths = [len(sentences[index]) for index in indices]
            denominator = max(1, len(sentences) - 1)
            positions = [index / denominator for index in indices]
            values[name]["length"].extend(lengths)
            values[name]["normalized_position"].extend(positions)
            values[name]["section"].update(sections[index] for index in indices)
            doc_means[name] = {
                "length": mean(lengths) if lengths else 0.0,
                "position": mean(positions) if positions else 0.0,
            }
        per_doc_length_delta.append(doc_means["oracle"]["length"] - doc_means["matched"]["length"])
        per_doc_position_delta.append(doc_means["oracle"]["position"] - doc_means["matched"]["position"])
    output = {}
    for name in ("oracle", "matched"):
        total = sum(values[name]["section"].values())
        output[name] = {
            "positive_candidate_length": distribution(values[name]["length"]),
            "positive_normalized_position": distribution(values[name]["normalized_position"]),
            "section_counts": dict(sorted(values[name]["section"].items())),
            "section_rates": {key: count / total for key, count in sorted(values[name]["section"].items())},
        }
    output["paired_per_document_oracle_minus_matched"] = {
        "mean_positive_length": paired_summary(per_doc_length_delta),
        "mean_positive_normalized_position": paired_summary(per_doc_position_delta),
    }
    return output


def length_close(full: list[dict], matched: list[dict]) -> dict:
    """Describe paired subsets whose realized output lengths are close.

    This is explicitly a post-output robustness diagnostic, not an unbiased
    causal estimate, because output length determines subset membership.
    """
    if [row["id"] for row in full] != [row["id"] for row in matched]:
        raise RuntimeError("paired detail IDs differ")
    output = {}
    for threshold in (10, 25, 50):
        pairs = [(left, right) for left, right in zip(full, matched) if abs(float(left["characters"]) - float(right["characters"])) <= threshold]
        row = {
            "threshold_absolute_character_difference": threshold,
            "documents": len(pairs),
            "share": len(pairs) / len(full),
            "full_mean_characters": mean(float(left["characters"]) for left, _ in pairs),
            "matched_mean_characters": mean(float(right["characters"]) for _, right in pairs),
            "metrics": {},
        }
        for metric in PAIR_METRICS:
            row["metrics"][metric] = paired_summary([float(left[metric]) - float(right[metric]) for left, right in pairs])
        output[f"within_{threshold}_characters"] = row
    return output


def expression_by_type(base, patterns, dev: list[dict], variants: dict[str, list[dict]]) -> dict:
    """Compute per-type exact-match micro and conditional document-macro scores."""
    ids = [str(row.get("id", row.get("case_id", index))) for index, row in enumerate(dev)]
    output = {}
    for variant, predictions in variants.items():
        if len(predictions) != len(dev):
            raise RuntimeError(f"unexpected prediction count for {variant}")
        output[variant] = {}
        for type_name, pattern in patterns.items():
            totals = Counter(reference_instances=0, predicted_instances=0, true_positive_instances=0, reference_nonempty_documents=0)
            doc_precision, doc_recall, doc_f1 = [], [], []
            for expected_id, gold, prediction in zip(ids, dev, predictions):
                prediction_id = str(prediction.get("id", prediction.get("case_id", expected_id)))
                if prediction_id != expected_id:
                    raise RuntimeError(f"prediction ID mismatch for {variant}")
                reference = set(pattern.findall(base.reference(gold)))
                predicted = set(pattern.findall(prediction["generated_summary"]))
                true_positive = reference & predicted
                totals["reference_instances"] += len(reference)
                totals["predicted_instances"] += len(predicted)
                totals["true_positive_instances"] += len(true_positive)
                if not reference:
                    continue
                totals["reference_nonempty_documents"] += 1
                precision = len(true_positive) / len(predicted) if predicted else 0.0
                recall = len(true_positive) / len(reference)
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                doc_precision.append(precision)
                doc_recall.append(recall)
                doc_f1.append(f1)
            micro_precision = totals["true_positive_instances"] / totals["predicted_instances"] if totals["predicted_instances"] else 0.0
            micro_recall = totals["true_positive_instances"] / totals["reference_instances"] if totals["reference_instances"] else 0.0
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
            output[variant][type_name] = {
                **totals,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
                "conditional_document_macro_precision": mean(doc_precision),
                "conditional_document_macro_recall": mean(doc_recall),
                "conditional_document_macro_f1": mean(doc_f1),
            }
    return output


def main() -> None:
    """Run the three frozen follow-up diagnostics and write hash manifests."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require(args.train, TRAIN_SHA256, "train")
    require(args.dev, DEV_SHA256, "dev")
    require(args.oracle_extracts, ORACLE_SHA256, "Oracle")
    sys.path.insert(0, str(args.scripts_dir))
    sys.path.insert(0, str(args.formal_script.parent))
    import run_legalfactaware_v21_budget_decoder_dev as decoder
    import run_legalfactaware_v25_structure_feature_dev as section_impl
    import run_legalfactaware_v2_cail_clean as base
    from evaluate_sse_cail import FACT_PATTERNS

    train = read_jsonl(args.train)
    dev = read_jsonl(args.dev)
    oracle = read_jsonl(args.oracle_extracts)
    full_details = read_jsonl(args.full_details)
    matched_details = read_jsonl(args.matched_details)
    variants = {
        "full_reference": read_jsonl(args.full_predictions),
        "matched_count_independent": read_jsonl(args.matched_predictions),
        "drop_only": read_jsonl(args.drop_predictions),
        "legacy_truncation": read_jsonl(args.legacy_predictions),
    }
    if (len(train), len(dev), len(oracle), len(full_details), len(matched_details)) != (7495, 942, 7495, 942, 942):
        raise RuntimeError("unexpected split or detail size")

    save_json(args.output_dir / "training_positive_diagnostics.json", label_diagnostics(base, section_impl, decoder, train, oracle))
    save_json(args.output_dir / "length_close_comparison.json", length_close(full_details, matched_details))
    save_json(args.output_dir / "expression_by_type.json", expression_by_type(base, FACT_PATTERNS, dev, variants))
    save_json(args.output_dir / "followup_manifest.json", {
        "status": "dev_followup_complete_test_untouched",
        "test_argument_exists": False,
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs": {
            "train": sha256(args.train),
            "dev": sha256(args.dev),
            "oracle": sha256(args.oracle_extracts),
            "full_predictions": sha256(args.full_predictions),
            "matched_predictions": sha256(args.matched_predictions),
            "drop_predictions": sha256(args.drop_predictions),
            "legacy_predictions": sha256(args.legacy_predictions),
            "full_details": sha256(args.full_details),
            "matched_details": sha256(args.matched_details),
        },
        "outputs": {
            name: {"sha256": sha256(args.output_dir / name), "bytes": (args.output_dir / name).stat().st_size}
            for name in ("training_positive_diagnostics.json", "length_close_comparison.json", "expression_by_type.json")
        },
    })


if __name__ == "__main__":
    main()
