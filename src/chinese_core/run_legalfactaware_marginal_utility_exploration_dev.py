#!/usr/bin/env python3
"""Independent dev-only marginal-utility exploration for LegalFactAware-v2.5-B.

This is deliberately isolated from every frozen/formal artifact.  It has no
test argument, refuses to reuse a non-empty output directory, validates the
frozen train/dev/oracle/model hashes, and writes only beneath --output-dir.

Stage 1 contains exactly the preregistered ten configurations:
  B0, four FactNovelty decoder weights, two section-conditioned TF-IDF
  variants, and three marginal-gain-weighted Oracle variants.
No combinations are trained in this script.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import platform
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.stats import wilcoxon
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression

import run_legalfactaware_formal_ablation_dev as formal
import run_legalfactaware_v21_budget_decoder_dev as frozen
import run_legalfactaware_v22_oracle_supervision_dev as oracle_impl
import run_legalfactaware_v25_structure_feature_dev as section_impl
import run_legalfactaware_v2_cail_clean as base


EXPERIMENT = "LegalFactAware-marginal-utility-stage1-dev-20260821"
MAX_CHARS = 450
MAX_SENTENCES = 8
REDUNDANCY_WEIGHT = 0.2
FACT_NOVELTY_LAMBDAS = (0.02, 0.05, 0.10, 0.20)
WEIGHTED_ORACLE_ALPHAS = (0.25, 0.50, 1.00)
HIGH_VALUE_SECTIONS = ("fact", "reason", "decision")
ALL_SECTIONS = section_impl.SECTIONS
BOOTSTRAP_SAMPLES = 10000
SEED = 42

# Frozen before any result is observed.  Values are metric fractions.
SUCCESS_THRESHOLDS = {
    "rouge_l_gain": 0.0025,
    "rouge_2_gain": 0.0030,
    "fact_f1_gain": 0.0050,
    "max_allowed_rouge_drop": 0.0015,
}


def ensure_new_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for child in ("models", "predictions", "per_sample", "tables"):
        (path / child).mkdir(exist_ok=True)


def fit_lr(name: str, matrix, labels: np.ndarray, sample_weight=None):
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=SEED,
    )
    started = time.perf_counter()
    classifier.fit(matrix, labels, sample_weight=sample_weight)
    elapsed = time.perf_counter() - started
    print(f"fitted {name} shape={matrix.shape} in {elapsed:.2f}s", flush=True)
    return classifier, {
        "fit_seconds": elapsed,
        "matrix_rows": int(matrix.shape[0]),
        "matrix_columns": int(matrix.shape[1]),
        "positive_labels": int(labels.sum()),
    }


def replay_oracle_with_marginal_gains(sentences: list[str], reference: str):
    """Replay the frozen greedy Oracle and retain gain at the actual selection step."""
    selected: list[int] = []
    remaining = set(range(len(sentences)))
    current_score = 0.0
    gains: dict[int, float] = {}
    sequence: list[int] = []
    while remaining and len(selected) < MAX_SENTENCES:
        best_index = None
        best_score = current_score
        for index in sorted(remaining):
            ordered = sorted(selected + [index])
            candidate = "".join(sentences[item] for item in ordered)
            if len(candidate) > MAX_CHARS:
                continue
            score = oracle_impl.oracle_score(reference, candidate)
            if score > best_score + 1e-12:
                best_index = index
                best_score = score
        if best_index is None:
            break
        gains[best_index] = best_score - current_score
        sequence.append(best_index)
        selected.append(best_index)
        remaining.remove(best_index)
        current_score = best_score
    return sorted(selected), sequence, gains, current_score


def build_marginal_gain_vector(train: list[dict], oracle_rows: list[dict], slices):
    values = np.zeros(slices[-1][1], dtype=np.float64)
    records: list[dict] = []
    mismatches = 0
    for doc, (row, oracle_row, (start, stop)) in enumerate(zip(train, oracle_rows, slices)):
        sentences = base.sentences(row)
        selected, sequence, gains, objective = replay_oracle_with_marginal_gains(
            sentences, base.reference(row)
        )
        expected = list(map(int, oracle_row["oracle_selected_indices"]))
        if selected != expected or abs(objective - float(oracle_row["oracle_objective"])) > 1e-9:
            mismatches += 1
            raise RuntimeError(
                f"Oracle replay mismatch at document {doc}: selected={selected}, expected={expected}, "
                f"objective={objective}, expected_objective={oracle_row['oracle_objective']}"
            )
        for index, gain in gains.items():
            values[start + index] = gain
        records.append({
            "id": frozen.row_id(row),
            "selection_sequence": sequence,
            "selected_indices_sorted": selected,
            "marginal_gains_by_sequence": [gains[index] for index in sequence],
            "objective": objective,
        })
        if (doc + 1) % 500 == 0 or doc + 1 == len(train):
            print(f"replayed Oracle marginal gains {doc + 1}/{len(train)}", flush=True)
    positives = values[values > 0]
    if not len(positives):
        raise RuntimeError("Oracle replay produced no positive marginal gains")
    audit = {
        "documents": len(train),
        "replay_mismatches": mismatches,
        "positive_gains": int(len(positives)),
        "mean_positive_gain": float(positives.mean()),
        "median_positive_gain": float(np.median(positives)),
        "min_positive_gain": float(positives.min()),
        "max_positive_gain": float(positives.max()),
    }
    return values, records, audit


def weighted_oracle_sample_weights(labels: np.ndarray, gains: np.ndarray, alpha: float):
    positive = labels == 1
    if np.any((gains > 0) != positive):
        raise RuntimeError("marginal-gain/Oracle-positive alignment failure")
    mean_gain = float(gains[positive].mean())
    weights = np.ones(len(labels), dtype=np.float64)
    weights[positive] = np.minimum(3.0, 1.0 + alpha * gains[positive] / mean_gain)
    return weights, {
        "alpha": alpha,
        "formula": "min(3, 1 + alpha * positive_gain / mean_positive_gain); negatives=1; LR class_weight=balanced",
        "mean_positive_gain": mean_gain,
        "positive_weight_mean": float(weights[positive].mean()),
        "positive_weight_median": float(np.median(weights[positive])),
        "positive_weight_max": float(weights[positive].max()),
        "clipped_positive_count": int(np.sum(weights[positive] >= 3.0)),
        "negative_weight": 1.0,
    }


def section_interaction(text, section_names: list[str], active_sections: tuple[str, ...]):
    blocks = [text]
    names = np.asarray(section_names, dtype=object)
    for section in active_sections:
        mask = (names == section).astype(np.float64).reshape(-1, 1)
        blocks.append(text.multiply(mask).tocsr())
    return hstack(blocks, format="csr")


def assemble_matrix(text, legal, section, section_names, mode: str):
    if mode == "full":
        return hstack([text, legal, section], format="csr")
    if mode == "section_conditioned_3":
        interacted = section_interaction(text, section_names, HIGH_VALUE_SECTIONS)
        return hstack([interacted, legal, section], format="csr")
    if mode == "section_conditioned_6":
        interacted = section_interaction(text, section_names, ALL_SECTIONS)
        return hstack([interacted, legal, section], format="csr")
    raise ValueError(mode)


def dev_matrix(vectorizer, sentences: list[str], mode: str):
    rows = section_impl.document_rows(sentences)
    text = vectorizer.transform(base.decorated_text(row) for row in rows)
    legal = csr_matrix([base.explicit_features(row) for row in rows])
    section = csr_matrix([section_impl.section_one_hot(row["section"]) for row in rows])
    return assemble_matrix(text, legal, section, [row["section"] for row in rows], mode)


def score_dev(dev: list[dict], models: dict, vectorizer):
    scores = {key: [] for key in models}
    timings = {key: 0.0 for key in models}
    for number, row in enumerate(dev, 1):
        sentences = base.sentences(row)
        matrices = {}
        for key, model in models.items():
            mode = model["feature_mode"]
            if mode not in matrices:
                matrices[mode] = dev_matrix(vectorizer, sentences, mode)
            started = time.perf_counter()
            values = model["classifier"].predict_proba(matrices[mode])[:, 1]
            timings[key] += time.perf_counter() - started
            scores[key].append(values.astype(float).tolist())
        if number % 100 == 0 or number == len(dev):
            print(f"scored dev {number}/{len(dev)}", flush=True)
    return scores, timings


def generate(dev: list[dict], score_rows: list[list[float]], name: str, fact_weight: float):
    predictions = []
    started = time.perf_counter()
    for number, (row, scores) in enumerate(zip(dev, score_rows), 1):
        sentences = base.sentences(row)
        selected, skipped = frozen.select_budget_greedy(
            sentences,
            scores,
            fact_weight=fact_weight,
            redundancy_weight=REDUNDANCY_WEIGHT,
            max_sentences=MAX_SENTENCES,
            max_chars=MAX_CHARS,
        )
        output = frozen.prediction_row(
            row,
            selected,
            decoder="budget_greedy",
            fact_weight=fact_weight,
            redundancy_weight=REDUNDANCY_WEIGHT,
            max_chars=MAX_CHARS,
            skipped_due_to_budget=skipped,
        )
        output["method"] = name
        predictions.append(output)
        if number % 200 == 0 or number == len(dev):
            print(f"generated {number}/{len(dev)} {name}", flush=True)
    return predictions, time.perf_counter() - started


def add_fact_density(metrics: dict, predictions: list[dict], output_dir: Path, key: str):
    densities, counts = [], []
    for detail, prediction in zip(metrics["lawa_details"], predictions):
        summary = prediction["generated_summary"]
        count = len(base.source_fact_items(summary))
        density = 100.0 * count / max(len(summary), 1)
        detail["distinct_legal_fact_items"] = count
        detail["fact_density_per_100_chars"] = density
        counts.append(count)
        densities.append(density)
    metrics["lawa"]["distinct_legal_fact_items"] = float(np.mean(counts))
    metrics["lawa"]["fact_density_per_100_chars"] = float(np.mean(densities))
    formal.write_jsonl(output_dir / "per_sample" / f"{key}_lawa.jsonl", metrics["lawa_details"])


def paired_candidate_vs_baseline(candidate: list[dict], baseline: list[dict], metric: str):
    if [row["id"] for row in candidate] != [row["id"] for row in baseline]:
        raise RuntimeError("paired detail ID mismatch")
    differences = np.asarray(
        [float(left[metric]) - float(right[metric]) for left, right in zip(candidate, baseline)],
        dtype=float,
    )
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for start in range(0, BOOTSTRAP_SAMPLES, 500):
        stop = min(start + 500, BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        bootstrap[start:stop] = differences[indices].mean(axis=1)
    try:
        p_value = float(wilcoxon(differences, zero_method="pratt").pvalue) if not np.allclose(differences, 0) else 1.0
    except ValueError:
        p_value = 1.0
    return {
        "metric": metric,
        "mean_delta_candidate_minus_B0": float(differences.mean()),
        "bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "wilcoxon_p": p_value if math.isfinite(p_value) else 1.0,
        "candidate_win_rate": float(np.mean(differences > 0)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
    }


def result_row(key: str, label: str, metrics: dict, baseline: dict):
    current = metrics["lawa"]
    base_row = baseline["lawa"]
    deltas = {
        "delta_R1": current["rouge_1_f1"] - base_row["rouge_1_f1"],
        "delta_R2": current["rouge_2_f1"] - base_row["rouge_2_f1"],
        "delta_RL": current["rouge_l_f1"] - base_row["rouge_l_f1"],
        "delta_Fact_F1": current["reference_fact_f1"] - base_row["reference_fact_f1"],
    }
    primary = (
        deltas["delta_RL"] >= SUCCESS_THRESHOLDS["rouge_l_gain"]
        or deltas["delta_R2"] >= SUCCESS_THRESHOLDS["rouge_2_gain"]
        or deltas["delta_Fact_F1"] >= SUCCESS_THRESHOLDS["fact_f1_gain"]
    )
    no_damage = min(deltas["delta_R1"], deltas["delta_R2"], deltas["delta_RL"]) >= -SUCCESS_THRESHOLDS["max_allowed_rouge_drop"]
    return {
        "id": key,
        "method": label,
        "R1": current["rouge_1_f1"],
        "R2": current["rouge_2_f1"],
        "RL": current["rouge_l_f1"],
        "Fact_P": current["reference_fact_precision"],
        "Fact_R": current["fact_recall"],
        "Fact_F1": current["reference_fact_f1"],
        "Fact_density_per_100_chars": current["fact_density_per_100_chars"],
        "Redundancy": current["pairwise_redundancy"],
        "Avg_chars": current["characters"],
        **deltas,
        "meets_primary_gain": int(primary),
        "no_rouge_drop_over_0p15": int(no_damage),
        "preregistered_success": int(primary and no_damage),
    }


def build_report(rows: list[dict], significance: dict, output_dir: Path) -> None:
    successful = [row for row in rows if row["id"] != "b0" and row["preregistered_success"]]
    lines = [
        "# LegalFactAware 边际效用探索：Stage 1（dev-only）",
        "",
        "## 协议边界",
        "",
        "- 仅使用冻结 train=7,495 和 dev=942；脚本不存在 test 参数。",
        "- 原正式实验、冻结模型和既有输出均未修改；本目录为独立新输出。",
        "- 只做十个预注册单因素配置，不做组合，不根据结果追加超参数。",
        "- FactNovelty 使用 Legal14 同源的五类精确事实项：money/date/law_title/law_article/action。",
        "- 成功门槛：RL +0.25、R2 +0.30 或 Fact-F1 +0.50 个百分点之一，且 R1/R2/RL 任一不得下降超过 0.15 点。",
        "",
        "## 结果",
        "",
        "| ID | R1 | R2 | RL | Fact-F1 | Fact密度/100字 | ΔR2 | ΔRL | ΔFact-F1 | 成功 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {100*row['R1']:.2f} | {100*row['R2']:.2f} | {100*row['RL']:.2f} | "
            f"{100*row['Fact_F1']:.2f} | {row['Fact_density_per_100_chars']:.3f} | "
            f"{100*row['delta_R2']:+.2f} | {100*row['delta_RL']:+.2f} | "
            f"{100*row['delta_Fact_F1']:+.2f} | {'是' if row['preregistered_success'] else '否'} |"
        )
    lines += [
        "",
        "## 决策",
        "",
        f"满足预注册成功标准的单项配置：{', '.join(row['id'] for row in successful) if successful else '无'}。",
        "本阶段不自动训练任何组合；只有成功单项才有资格进入独立的第二阶段。",
        "所有候选相对 B0 的 10,000 次 paired bootstrap 与 Wilcoxon 检验见 `significance_vs_b0.json`。",
        "",
    ]
    (output_dir / "STAGE1_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--oracle-extracts", type=Path, required=True)
    parser.add_argument("--v22-model", type=Path, required=True)
    parser.add_argument("--frozen-v25-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_new_output_dir(args.output_dir)
    hashes = {
        "train_sha256": formal.require(args.train, formal.EXPECTED_TRAIN_SHA256, "train"),
        "dev_sha256": formal.require(args.dev, formal.EXPECTED_DEV_SHA256, "dev"),
        "oracle_extracts_sha256": formal.require(args.oracle_extracts, formal.EXPECTED_ORACLE_SHA256, "oracle extracts"),
        "v22_model_sha256": formal.require(args.v22_model, formal.EXPECTED_V22_MODEL_SHA256, "v2.2 model"),
        "frozen_v25_model_sha256": formal.require(args.frozen_v25_model, formal.EXPECTED_V25_MODEL_SHA256, "frozen v2.5-B model"),
    }
    preregistration = {
        "experiment": EXPERIMENT,
        "stage": 1,
        "test_argument_exists": False,
        "combinations_allowed": False,
        "configurations": ["B0", "E1-1", "E1-2", "E1-3", "E1-4", "E2-A", "E2-B", "E3-1", "E3-2", "E3-3"],
        "fact_novelty_lambdas": list(FACT_NOVELTY_LAMBDAS),
        "weighted_oracle_alphas": list(WEIGHTED_ORACLE_ALPHAS),
        "success_thresholds": SUCCESS_THRESHOLDS,
        "max_chars": MAX_CHARS,
        "max_sentences": MAX_SENTENCES,
        "redundancy_weight": REDUNDANCY_WEIGHT,
        **hashes,
        "script_sha256": formal.sha256(Path(__file__).resolve()),
    }
    formal.save_json(args.output_dir / "PREREGISTRATION.json", preregistration)

    train = base.read_jsonl(args.train)
    dev = base.read_jsonl(args.dev)
    oracle_rows = base.read_jsonl(args.oracle_extracts)
    if (len(train), len(dev), len(oracle_rows)) != (formal.TRAIN_CASES, formal.DEV_CASES, formal.TRAIN_CASES):
        raise RuntimeError("unexpected split sizes")
    with args.v22_model.open("rb") as handle:
        vectorizer = pickle.load(handle)["vectorizer"]

    features, labels, slices = formal.build_training(train, oracle_rows)
    print("building frozen train matrices", flush=True)
    text = vectorizer.transform(base.decorated_text(row) for row in features)
    legal = csr_matrix([base.explicit_features(row) for row in features])
    section = csr_matrix([section_impl.section_one_hot(row["section"]) for row in features])
    section_names = [row["section"] for row in features]
    full_matrix = hstack([text, legal, section], format="csr")

    models = {}
    fit_timings = {}
    classifier, timing = fit_lr("B0", full_matrix, labels["oracle"])
    models["b0"] = {"classifier": classifier, "feature_mode": "full"}
    fit_timings["b0"] = timing

    print("building E2-A high-value section interactions", flush=True)
    matrix_e2a = assemble_matrix(text, legal, section, section_names, "section_conditioned_3")
    classifier, timing = fit_lr("E2-A", matrix_e2a, labels["oracle"])
    models["e2a"] = {"classifier": classifier, "feature_mode": "section_conditioned_3"}
    fit_timings["e2a"] = timing
    del matrix_e2a
    gc.collect()

    print("building E2-B all-section interactions", flush=True)
    matrix_e2b = assemble_matrix(text, legal, section, section_names, "section_conditioned_6")
    classifier, timing = fit_lr("E2-B", matrix_e2b, labels["oracle"])
    models["e2b"] = {"classifier": classifier, "feature_mode": "section_conditioned_6"}
    fit_timings["e2b"] = timing
    del matrix_e2b
    gc.collect()

    gains, gain_records, gain_audit = build_marginal_gain_vector(train, oracle_rows, slices)
    formal.write_jsonl(args.output_dir / "oracle_marginal_gain_replay.jsonl", gain_records)
    formal.save_json(args.output_dir / "oracle_marginal_gain_audit.json", gain_audit)
    weight_audits = {}
    for alpha in WEIGHTED_ORACLE_ALPHAS:
        key = f"e3_{str(alpha).replace('.', 'p')}"
        weights, audit = weighted_oracle_sample_weights(labels["oracle"], gains, alpha)
        classifier, timing = fit_lr(f"E3 alpha={alpha}", full_matrix, labels["oracle"], weights)
        models[key] = {"classifier": classifier, "feature_mode": "full"}
        fit_timings[key] = timing
        weight_audits[key] = audit
    formal.save_json(args.output_dir / "weighted_oracle_audit.json", weight_audits)

    for key, model in models.items():
        with (args.output_dir / "models" / f"{key}.pkl").open("wb") as handle:
            pickle.dump(model, handle)
    del text, legal, section, full_matrix, gains
    gc.collect()

    scores, score_timings = score_dev(dev, models, vectorizer)
    specs = [
        ("b0", "B0 Full", "b0", 0.0),
        ("e1_0p02", "E1 FactNovelty lambda=0.02", "b0", 0.02),
        ("e1_0p05", "E1 FactNovelty lambda=0.05", "b0", 0.05),
        ("e1_0p10", "E1 FactNovelty lambda=0.10", "b0", 0.10),
        ("e1_0p20", "E1 FactNovelty lambda=0.20", "b0", 0.20),
        ("e2a", "E2-A conditioned TF-IDF: Fact/Reason/Decision", "e2a", 0.0),
        ("e2b", "E2-B conditioned TF-IDF: all Section6", "e2b", 0.0),
        ("e3_0p25", "E3 weighted Oracle alpha=0.25", "e3_0p25", 0.0),
        ("e3_0p5", "E3 weighted Oracle alpha=0.50", "e3_0p5", 0.0),
        ("e3_1p0", "E3 weighted Oracle alpha=1.00", "e3_1p0", 0.0),
    ]
    predictions, metrics, decode_timings = {}, {}, {}
    for key, label, model_key, fact_weight in specs:
        predictions[key], decode_timings[key] = generate(dev, scores[model_key], label, fact_weight)
        formal.write_jsonl(args.output_dir / "predictions" / f"{key}.jsonl", predictions[key])
        metrics[key] = formal.evaluate_variant(key, predictions[key], dev, MAX_CHARS, args.output_dir)
        add_fact_density(metrics[key], predictions[key], args.output_dir, key)
        lawa = metrics[key]["lawa"]
        print(f"evaluated {key}: {lawa['rouge_1_f1']:.6f}/{lawa['rouge_2_f1']:.6f}/{lawa['rouge_l_f1']:.6f} FactF1={lawa['reference_fact_f1']:.6f}", flush=True)

    rows = [result_row(key, label, metrics[key], metrics["b0"]) for key, label, _, _ in specs]
    formal.write_csv(args.output_dir / "tables" / "Stage1_all_10_configurations.csv", rows)
    significance = {}
    for key, _, _, _ in specs[1:]:
        significance[key] = [
            paired_candidate_vs_baseline(metrics[key]["lawa_details"], metrics["b0"]["lawa_details"], metric)
            for metric in ("rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "reference_fact_f1")
        ]
    formal.save_json(args.output_dir / "significance_vs_b0.json", significance)
    build_report(rows, significance, args.output_dir)

    compact_metrics = {
        key: {name: value for name, value in result.items() if name != "lawa_details"}
        for key, result in metrics.items()
    }
    formal.save_json(args.output_dir / "all_metrics.json", compact_metrics)
    formal.save_json(args.output_dir / "timings.json", {
        "fit": fit_timings,
        "classifier_score_seconds": score_timings,
        "decode_seconds": decode_timings,
    })
    config = {
        **preregistration,
        "status": "dev_complete_test_untouched",
        "train_cases": len(train),
        "dev_cases": len(dev),
        "training_sentences": len(features),
        "python": platform.python_version(),
        "sklearn": sklearn_version,
    }
    formal.save_json(args.output_dir / "config.json", config)
    manifest = {
        **config,
        "outputs": {
            str(path.relative_to(args.output_dir)): {"sha256": formal.sha256(path), "bytes": path.stat().st_size}
            for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.json"
        },
    }
    formal.save_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({
        "status": "complete",
        "output_dir": str(args.output_dir),
        "successful": [row["id"] for row in rows if row["id"] != "b0" and row["preregistered_success"]],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
