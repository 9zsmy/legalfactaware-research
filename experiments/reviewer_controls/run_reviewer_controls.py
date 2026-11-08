#!/usr/bin/env python3
"""Reviewer-requested LegalFactAware controls on the frozen dev split only.

Pipeline overview
-----------------
1. Verify every frozen input by SHA-256.
2. Build Oracle and per-document positive-count-matched training labels.
3. Train the supervision and feature-decomposition control models.
4. Decode the frozen development set under alternative budget mechanisms.
5. Evaluate, run paired tests, and write auditable aggregate artifacts.

There is deliberately no test-set command-line argument.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.stats import wilcoxon
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression


EXPECTED = {
    "train": "8b19e6a1aa086e55c3a1d694e86cb4cf5abfe6929ee1258a3621b291bde564a2",
    "dev": "1a056c68dcfa5f11e9574d5f1b818fa475bb5607ea7f6565e2868072a55de2c0",
    "oracle": "de20bf0a98cb04de5f11f2b0869701fa5c940a392f5804f0b85a09fad5f19960",
    "vectorizer": "2cc0c9fe4b1fdaca6c1b2e37e545fcece3e5da83533296c1c12ba1d8bb8f70e9",
    "full_classifier": "38d6b031a1a6faa11a8b6f952f952ad06edfdd492f0ab7671b36055d875c548f",
}
TRAIN_CASES = 7495
DEV_CASES = 942
MAX_CHARS = 450
MAX_CANDIDATES = 8
REDUNDANCY = 0.2
FACT_WEIGHT = 0.0
SEED = 42
METRICS = ("rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "reference_fact_f1")


# ---------------------------------------------------------------------------
# Reproducibility and artifact I/O
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: {actual} != {expected}")
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
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripts-dir", type=Path, required=True)
    parser.add_argument("--formal-script", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--oracle-extracts", type=Path, required=True)
    parser.add_argument("--vectorizer-model", type=Path, required=True)
    parser.add_argument("--full-classifier-model", type=Path, required=True)
    parser.add_argument("--archived-full", type=Path, required=True)
    parser.add_argument("--archived-legacy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def fit_lr(matrix, labels: np.ndarray, name: str):
    """Fit one frozen-specification logistic-regression control model."""
    started = time.perf_counter()
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=SEED,
    ).fit(matrix, labels)
    elapsed = time.perf_counter() - started
    print(f"fitted {name}: shape={matrix.shape}, positives={int(labels.sum())}, seconds={elapsed:.2f}", flush=True)
    return model, elapsed


def build_training(base, section_impl, decoder, train: list[dict], oracle_rows: list[dict]):
    """Construct shared candidate features and two exactly count-matched labels.

    For every document, ``matched_indices`` contains exactly as many positives as
    the Oracle set, but ranks candidates independently by clipped ROUGE-2
    precision against the reference.  This is Control A in the paper.
    """
    features: list[dict] = []
    oracle_labels: list[int] = []
    matched_labels: list[int] = []
    per_document: list[dict] = []
    for doc_id, (row, oracle_row) in enumerate(zip(train, oracle_rows)):
        if decoder.row_id(row) != decoder.row_id(oracle_row):
            raise RuntimeError(f"train/oracle ID mismatch at {doc_id}")
        sentences = base.sentences(row)
        sections = section_impl.section_labels(sentences)
        oracle_indices = sorted(map(int, oracle_row["oracle_selected_indices"]))
        k = len(oracle_indices)
        reference_counts = base.bigrams(base.tokens(base.reference(row)))
        standalone = [base.rouge2_precision(sentence, reference_counts) for sentence in sentences]
        matched_indices = sorted(sorted(range(len(sentences)), key=lambda i: (-standalone[i], i))[:k])
        oracle_set, matched_set = set(oracle_indices), set(matched_indices)
        for index, sentence in enumerate(sentences):
            features.append({
                "doc_id": doc_id,
                "index": index,
                "total": len(sentences),
                "sentence": sentence,
                "section": sections[index],
            })
            oracle_labels.append(int(index in oracle_set))
            matched_labels.append(int(index in matched_set))
        union = oracle_set | matched_set
        per_document.append({
            "id": decoder.row_id(row),
            "positive_count": k,
            "oracle_indices": oracle_indices,
            "matched_indices": matched_indices,
            "positive_jaccard": len(oracle_set & matched_set) / len(union) if union else 1.0,
            "selection_equal": oracle_indices == matched_indices,
        })
        if (doc_id + 1) % 500 == 0 or doc_id + 1 == len(train):
            print(f"built matched-count labels {doc_id + 1}/{len(train)}", flush=True)
    oracle = np.asarray(oracle_labels, dtype=np.int8)
    matched = np.asarray(matched_labels, dtype=np.int8)
    return features, oracle, matched, per_document


def make_prediction(decoder, row: dict, selected: list[int], name: str, budget: int, skipped=None):
    """Convert selected candidate indices into the standard prediction record."""
    output = decoder.prediction_row(
        row,
        selected,
        decoder="budget_greedy",
        fact_weight=FACT_WEIGHT,
        redundancy_weight=REDUNDANCY,
        max_chars=budget,
        skipped_due_to_budget=skipped or [],
    )
    output["method"] = name
    output["evaluation_budget"] = budget
    return output


def generate_budget(decoder, base, dev: list[dict], score_rows: list[list[float]], name: str, budget: int, cap: int | None):
    """Run fit/skip decoding and record why selection terminated.

    ``budget_skip_observed`` and ``terminal_stop`` are intentionally separate:
    encountering an over-budget candidate does not imply that the document
    ultimately terminates because of the character budget.
    """
    outputs, stops = [], []
    for row, scores in zip(dev, score_rows):
        sentences = base.sentences(row)
        effective_cap = len(sentences) if cap is None else cap
        selected, skipped = decoder.select_budget_greedy(
            sentences,
            scores,
            fact_weight=FACT_WEIGHT,
            redundancy_weight=REDUNDANCY,
            max_sentences=effective_cap,
            max_chars=budget,
        )
        output = make_prediction(decoder, row, selected, name, budget, skipped)
        output["candidate_cap"] = cap
        outputs.append(output)
        remaining = set(range(len(sentences))) - set(selected)
        remaining_chars = budget - len(output["generated_summary"])
        any_fit = any(len(sentences[i]) <= remaining_chars for i in remaining)
        cap_hit = cap is not None and len(selected) >= cap
        if cap_hit:
            terminal = "candidate_cap"
        elif remaining and not any_fit:
            terminal = "character_budget"
        else:
            terminal = "candidate_pool"
        stops.append({
            "id": decoder.row_id(row),
            "cap_hit": cap_hit,
            "budget_skip_observed": bool(skipped),
            "terminal_stop": terminal,
            "remaining_characters": remaining_chars,
        })
    return outputs, stops


def generate_lead_tail(decoder, base, dev: list[dict], reverse: bool):
    """Generate parameter-free Lead or Tail fit/skip position baselines."""
    outputs = []
    name = "Tail fit/skip" if reverse else "Lead fit/skip"
    for row in dev:
        sentences = base.sentences(row)
        order = range(len(sentences) - 1, -1, -1) if reverse else range(len(sentences))
        selected, used = [], 0
        for index in order:
            if len(selected) >= MAX_CANDIDATES:
                break
            length = len(sentences[index])
            if used + length <= MAX_CHARS:
                selected.append(index)
                used += length
        outputs.append(make_prediction(decoder, row, sorted(selected), name, MAX_CHARS))
    return outputs


def generate_drop_only(decoder, base, dev: list[dict], score_rows: list[list[float]]):
    """Build Control D: remove complete tail candidates without replacements."""
    outputs, legacy_outputs = [], []
    for row, scores in zip(dev, score_rows):
        sentences = base.sentences(row)
        legacy = base.select_indices(
            sentences,
            scores,
            fact_weight=FACT_WEIGHT,
            redundancy_weight=REDUNDANCY,
            max_sentences=MAX_CANDIDATES,
        )
        legacy_output = decoder.prediction_row(
            row,
            legacy,
            decoder="legacy",
            fact_weight=FACT_WEIGHT,
            redundancy_weight=REDUNDANCY,
            max_chars=MAX_CHARS,
        )
        legacy_output["method"] = "Post-hoc within-candidate truncation"
        legacy_outputs.append(legacy_output)
        kept = list(legacy)
        while kept and sum(len(sentences[i]) for i in kept) > MAX_CHARS:
            kept.pop()
        output = make_prediction(decoder, row, kept, "Drop-tail complete candidates", MAX_CHARS)
        output["legacy_selected_indices"] = legacy
        output["dropped_indices"] = legacy[len(kept):]
        outputs.append(output)
    return outputs, legacy_outputs


def evaluate_variant(formal, decoder, base, section_impl, predictions: list[dict], dev: list[dict], budget: int, cap: int | None, name: str, output_dir: Path):
    """Evaluate one variant and persist paired per-document numeric details."""
    eval_cap = max(len(base.sentences(row)) for row in dev) if cap is None else cap
    aggregate, details = decoder.evaluate_predictions(
        predictions,
        dev,
        rouge_unit="lawa",
        max_chars=budget,
        max_sentences=eval_cap,
    )
    for detail, prediction, gold in zip(details, predictions, dev):
        precision, recall, f1 = formal.fact_reference_metrics(base.reference(gold), prediction["generated_summary"])
        detail["reference_fact_precision"] = precision
        detail["reference_fact_recall_macro"] = recall
        detail["reference_fact_f1"] = f1
        detail["unused_budget"] = budget - len(prediction["generated_summary"])
    for key in (
        "reference_fact_precision",
        "reference_fact_recall_macro",
        "reference_fact_f1",
        "unused_budget",
        "characters",
        "selected_sentences",
    ):
        aggregate[key] = mean(float(row[key]) for row in details)
    aggregate["hard_truncation_count"] = sum(int(row["hard_truncated"]) for row in details)
    write_jsonl(output_dir / "per_sample" / f"{name}_lawa.jsonl", details)
    return {"lawa": aggregate, "lawa_details": details, "decoder_diagnostics": decoder.decoder_diagnostics(predictions, max_chars=budget)}


def compact(result: dict) -> dict:
    """Return table-ready metrics while keeping recall conventions explicit."""
    m = result["lawa"]
    return {
        "R1": m["rouge_1_f1"],
        "R2": m["rouge_2_f1"],
        "RL": m["rouge_l_f1"],
        # Keep the two recall conventions explicit.  The paper's legacy
        # conditional recall excludes expression types absent from a reference,
        # whereas reference_fact_recall_macro averages over the fixed union of
        # expression types.  They must not share the same table label.
        "Expression_Recall_conditional_reference_types": m["fact_recall"],
        "Expression_Recall_macro_union_types": m["reference_fact_recall_macro"],
        "Expression_F1_macro_union_types": m["reference_fact_f1"],
        "Avg_chars": m["characters"],
        "Avg_candidates": m["selected_sentences"],
        "Hard_truncations": m["hard_truncation_count"],
        "Budget_violations": m["budget_violations"],
    }


def paired_test(left: list[dict], right: list[dict], metric: str, label: str) -> dict:
    """Compute paired mean delta, document bootstrap CI and Wilcoxon p-value."""
    if [r["id"] for r in left] != [r["id"] for r in right]:
        raise RuntimeError("paired IDs differ")
    differences = np.asarray([float(a[metric]) - float(b[metric]) for a, b in zip(left, right)], dtype=float)
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(10_000, dtype=float)
    for start in range(0, 10_000, 500):
        stop = min(start + 500, 10_000)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        bootstrap[start:stop] = differences[indices].mean(axis=1)
    raw = 1.0 if np.allclose(differences, 0) else float(wilcoxon(differences, zero_method="pratt").pvalue)
    return {
        "comparison": label,
        "metric": metric,
        "mean_delta_left_minus_right": float(differences.mean()),
        "bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "wilcoxon_p_raw": raw if math.isfinite(raw) else 1.0,
        "left_win_rate": float(np.mean(differences > 0)),
        "tie_rate": float(np.mean(np.isclose(differences, 0))),
    }


def holm(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i]["wilcoxon_p_raw"])
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * rows[index]["wilcoxon_p_raw"]))
        rows[index]["wilcoxon_p_holm"] = running


def expression_audit(base, patterns, rows: list[dict]) -> dict:
    """Count reference documents containing each rule-matched expression type."""
    counts = {name: 0 for name in patterns}
    all_empty = 0
    for row in rows:
        hits = {name: set(pattern.findall(base.reference(row))) for name, pattern in patterns.items()}
        all_empty += int(not any(hits.values()))
        for name, values in hits.items():
            counts[name] += int(bool(values))
    return {
        "documents": len(rows),
        "all_expression_types_empty": all_empty,
        "all_empty_rate": all_empty / len(rows),
        "reference_nonempty_documents_by_type": counts,
        "note": "When all_expression_types_empty is zero, the all-empty score convention cannot affect the document-macro mean.",
    }


def main() -> None:
    """Run Controls A--D, the B x K grid, evaluation and artifact checks."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("models", "predictions", "per_sample", "tables"):
        (args.output_dir / name).mkdir(exist_ok=True)
    sys.path.insert(0, str(args.scripts_dir))
    sys.path.insert(0, str(args.formal_script.parent))
    import run_legalfactaware_formal_ablation_dev as formal
    import run_legalfactaware_v21_budget_decoder_dev as decoder
    import run_legalfactaware_v25_structure_feature_dev as section_impl
    import run_legalfactaware_v2_cail_clean as base
    from evaluate_sse_cail import FACT_PATTERNS

    started = time.perf_counter()
    hashes = {
        "train": require(args.train, EXPECTED["train"], "train"),
        "dev": require(args.dev, EXPECTED["dev"], "dev"),
        "oracle": require(args.oracle_extracts, EXPECTED["oracle"], "Oracle"),
        "vectorizer": require(args.vectorizer_model, EXPECTED["vectorizer"], "vectorizer model"),
        "full_classifier": require(args.full_classifier_model, EXPECTED["full_classifier"], "full classifier"),
        "archived_full": sha256(args.archived_full),
        "archived_legacy": sha256(args.archived_legacy),
    }
    train = base.read_jsonl(args.train)
    dev = base.read_jsonl(args.dev)
    oracle_rows = base.read_jsonl(args.oracle_extracts)
    archived_full = base.read_jsonl(args.archived_full)
    archived_legacy = base.read_jsonl(args.archived_legacy)
    if (len(train), len(dev), len(oracle_rows), len(archived_full), len(archived_legacy)) != (TRAIN_CASES, DEV_CASES, TRAIN_CASES, DEV_CASES, DEV_CASES):
        raise RuntimeError("unexpected split or archive size")
    with args.vectorizer_model.open("rb") as handle:
        vectorizer = pickle.load(handle)["vectorizer"]
    with args.full_classifier_model.open("rb") as handle:
        full_bundle = pickle.load(handle)
    full_classifier = full_bundle["classifier"]

    features, oracle_labels, matched_labels, matched_docs = build_training(base, section_impl, decoder, train, oracle_rows)
    write_jsonl(args.output_dir / "matched_count_training_audit.jsonl", matched_docs)
    intersection = int(np.sum((oracle_labels == 1) & (matched_labels == 1)))
    union = int(np.sum((oracle_labels == 1) | (matched_labels == 1)))
    label_audit = {
        "training_candidates": len(features),
        "oracle_positives": int(oracle_labels.sum()),
        "matched_positives": int(matched_labels.sum()),
        "positive_counts_exactly_match": bool(np.array_equal(
            np.asarray([r["positive_count"] for r in matched_docs]),
            np.asarray([len(r["matched_indices"]) for r in matched_docs]),
        )),
        "candidate_label_agreement": float(np.mean(oracle_labels == matched_labels)),
        "positive_jaccard_global": intersection / union if union else 1.0,
        "document_selection_equal_count": sum(int(r["selection_equal"]) for r in matched_docs),
        "mean_document_positive_jaccard": mean(r["positive_jaccard"] for r in matched_docs),
    }
    save_json(args.output_dir / "matched_count_label_audit.json", label_audit)

    print("building training matrices", flush=True)
    decorated = vectorizer.transform(base.decorated_text(row) for row in features)
    lexical = vectorizer.transform(row["sentence"] for row in features)
    position = vectorizer.transform(f"position{row['index']} " for row in features)
    legal = csr_matrix([base.explicit_features(row) for row in features])
    section = csr_matrix([section_impl.section_one_hot(row["section"]) for row in features])
    full_matrix = hstack([decorated, legal, section], format="csr")
    lexical_full_matrix = hstack([lexical, legal, section], format="csr")

    models = {"full_reference": full_classifier}
    timings = {}
    models["matched_count_independent"], timings["matched_count_independent"] = fit_lr(full_matrix, matched_labels, "matched_count_independent")
    models["lexical_full_aux"], timings["lexical_full_aux"] = fit_lr(lexical_full_matrix, oracle_labels, "lexical_full_aux")
    models["decorated_text_only"], timings["decorated_text_only"] = fit_lr(decorated, oracle_labels, "decorated_text_only")
    models["lexical_text_only"], timings["lexical_text_only"] = fit_lr(lexical, oracle_labels, "lexical_text_only")
    models["position_marker_only"], timings["position_marker_only"] = fit_lr(position, oracle_labels, "position_marker_only")
    for name, model in models.items():
        if name != "full_reference":
            with (args.output_dir / "models" / f"{name}.pkl").open("wb") as handle:
                pickle.dump({"classifier": model, "variant": name}, handle)
    del decorated, lexical, position, legal, section, full_matrix, lexical_full_matrix

    score_rows = {name: [] for name in models}
    for number, row in enumerate(dev, 1):
        sentences = base.sentences(row)
        rows = section_impl.document_rows(sentences)
        decorated_dev = vectorizer.transform(base.decorated_text(item) for item in rows)
        lexical_dev = vectorizer.transform(item["sentence"] for item in rows)
        position_dev = vectorizer.transform(f"position{item['index']} " for item in rows)
        legal_dev = csr_matrix([base.explicit_features(item) for item in rows])
        section_dev = csr_matrix([section_impl.section_one_hot(item["section"]) for item in rows])
        full_dev = hstack([decorated_dev, legal_dev, section_dev], format="csr")
        lexical_full_dev = hstack([lexical_dev, legal_dev, section_dev], format="csr")
        matrices = {
            "full_reference": full_dev,
            "matched_count_independent": full_dev,
            "lexical_full_aux": lexical_full_dev,
            "decorated_text_only": decorated_dev,
            "lexical_text_only": lexical_dev,
            "position_marker_only": position_dev,
        }
        for name, model in models.items():
            score_rows[name].append(model.predict_proba(matrices[name])[:, 1].tolist())
        if number % 100 == 0 or number == len(dev):
            print(f"scored dev {number}/{len(dev)}", flush=True)

    results, predictions = {}, {}
    for name in models:
        rows, _ = generate_budget(decoder, base, dev, score_rows[name], name, MAX_CHARS, MAX_CANDIDATES)
        predictions[name] = rows
        results[name] = evaluate_variant(formal, decoder, base, section_impl, rows, dev, MAX_CHARS, MAX_CANDIDATES, name, args.output_dir)
        write_jsonl(args.output_dir / "predictions" / f"{name}.jsonl", rows)
        print(name, compact(results[name]), flush=True)

    if [r["selected_indices"] for r in predictions["full_reference"]] != [r["selected_indices"] for r in archived_full]:
        raise RuntimeError("regenerated full selections do not match archived full predictions")

    lead = generate_lead_tail(decoder, base, dev, reverse=False)
    tail = generate_lead_tail(decoder, base, dev, reverse=True)
    for name, rows in (("lead_fit_skip", lead), ("tail_fit_skip", tail)):
        predictions[name] = rows
        results[name] = evaluate_variant(formal, decoder, base, section_impl, rows, dev, MAX_CHARS, MAX_CANDIDATES, name, args.output_dir)
        write_jsonl(args.output_dir / "predictions" / f"{name}.jsonl", rows)

    drop_only, legacy = generate_drop_only(decoder, base, dev, score_rows["full_reference"])
    if [r["selected_indices"] for r in legacy] != [r["selected_indices"] for r in archived_legacy]:
        raise RuntimeError("regenerated legacy selections do not match archived legacy predictions")
    if [r["generated_summary"] for r in legacy] != [r["generated_summary"] for r in archived_legacy]:
        raise RuntimeError("regenerated legacy outputs do not match archived legacy outputs")
    predictions["drop_only"] = drop_only
    predictions["legacy_truncation"] = legacy
    for name, rows in (("drop_only", drop_only), ("legacy_truncation", legacy)):
        results[name] = evaluate_variant(formal, decoder, base, section_impl, rows, dev, MAX_CHARS, MAX_CANDIDATES, name, args.output_dir)
        write_jsonl(args.output_dir / "predictions" / f"{name}.jsonl", rows)

    grid_rows = []
    for budget in (300, 400, 450):
        for cap in (4, 6, 8, 10, 12, None):
            label = f"grid_b{budget}_k{'all' if cap is None else cap}"
            rows, stops = generate_budget(decoder, base, dev, score_rows["full_reference"], label, budget, cap)
            result = evaluate_variant(formal, decoder, base, section_impl, rows, dev, budget, cap, label, args.output_dir)
            compact_row = {"Budget": budget, "Candidate_cap": "all" if cap is None else cap, **compact(result)}
            compact_row.update({
                "Cap_hit_count": sum(int(r["cap_hit"]) for r in stops),
                "Cap_hit_rate": mean(int(r["cap_hit"]) for r in stops),
                "Budget_skip_count": sum(int(r["budget_skip_observed"]) for r in stops),
                "Budget_skip_rate": mean(int(r["budget_skip_observed"]) for r in stops),
                "Terminal_candidate_cap": sum(r["terminal_stop"] == "candidate_cap" for r in stops),
                "Terminal_character_budget": sum(r["terminal_stop"] == "character_budget" for r in stops),
                "Terminal_candidate_pool": sum(r["terminal_stop"] == "candidate_pool" for r in stops),
            })
            grid_rows.append(compact_row)
            if budget == 450:
                write_jsonl(args.output_dir / "predictions" / f"{label}.jsonl", rows)
            print(label, compact_row, flush=True)
    write_csv(args.output_dir / "tables" / "budget_by_candidate_cap.csv", grid_rows)

    metric_table = [{"Variant": name, **compact(result)} for name, result in results.items()]
    write_csv(args.output_dir / "tables" / "reviewer_control_metrics.csv", metric_table)
    save_json(args.output_dir / "all_metrics.json", {name: compact(result) for name, result in results.items()})

    significance = {}
    for label, left_name, right_name in (
        ("full_minus_matched_count_independent", "full_reference", "matched_count_independent"),
        ("full_minus_drop_only", "full_reference", "drop_only"),
        ("drop_only_minus_legacy_truncation", "drop_only", "legacy_truncation"),
    ):
        family = [paired_test(results[left_name]["lawa_details"], results[right_name]["lawa_details"], metric, label) for metric in METRICS]
        holm(family)
        significance[label] = family
    save_json(args.output_dir / "paired_significance.json", significance)

    audit = expression_audit(base, FACT_PATTERNS, dev)
    save_json(args.output_dir / "expression_set_audit.json", audit)
    config = {
        "experiment": "Chinese-reviewer-controls-20261028",
        "status": "dev_complete_test_untouched",
        "test_argument_exists": False,
        "train_cases": len(train),
        "dev_cases": len(dev),
        "random_seed": SEED,
        "python": platform.python_version(),
        "sklearn": sklearn_version,
        "budgets": [300, 400, 450],
        "candidate_caps": [4, 6, 8, 10, 12, "all"],
        "hashes": hashes,
        "script_sha256": sha256(Path(__file__).resolve()),
        "fit_seconds": timings,
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.output_dir / "config.json", config)
    manifest = {
        **config,
        "outputs": {
            str(path.relative_to(args.output_dir)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(args.output_dir.rglob("*")) if path.is_file()
        },
    }
    save_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir), "elapsed_seconds": config["elapsed_seconds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

