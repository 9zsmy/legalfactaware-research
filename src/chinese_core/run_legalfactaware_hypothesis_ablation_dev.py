#!/usr/bin/env python3
"""Hypothesis-driven, dev-only ablation packaging for LegalFactAware.

The script reuses already completed same-protocol formal results, adds the
one missing true hard-structure-coverage decoder, and writes paper-oriented
tables without modifying any source artifact.  It deliberately has no test
argument and refuses to overwrite a non-empty output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np

import run_legalfactaware_formal_ablation_dev as formal
import run_legalfactaware_v21_budget_decoder_dev as frozen
import run_legalfactaware_v25_structure_feature_dev as section_impl
import run_legalfactaware_v2_cail_clean as base


EXPERIMENT = "LegalFactAware-hypothesis-driven-ablation-dev-20260821"
MAX_CHARS = 450
MAX_SENTENCES = 8
REDUNDANCY_WEIGHT = 0.2
MEANINGFUL = set(formal.MEANINGFUL_SECTIONS)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_new_output(path: Path):
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for child in ("tables", "predictions", "per_sample"):
        (path / child).mkdir(exist_ok=True)


def verify_source(directory: Path, label: str):
    config = read_json(directory / "config.json")
    manifest = read_json(directory / "manifest.json")
    status = str(config.get("status", ""))
    if "test_untouched" not in status:
        raise RuntimeError(f"{label} is not a dev-only completed source: status={status}")
    if config.get("test_argument_exists") is not False:
        raise RuntimeError(f"{label} unexpectedly exposes a test argument")
    return {
        "label": label,
        "directory": str(directory),
        "config_sha256": formal.sha256(directory / "config.json"),
        "manifest_sha256": formal.sha256(directory / "manifest.json"),
        "status": status,
        "experiment": config.get("experiment"),
        "manifest_output_count": len(manifest.get("outputs", {})),
    }


def metric_row(method: str, result: dict, extra: dict | None = None):
    lawa = result["lawa"]
    row = {
        "Method": method,
        "R1": lawa["rouge_1_f1"],
        "R2": lawa["rouge_2_f1"],
        "RL": lawa["rouge_l_f1"],
        # These three are one internally consistent reference-fact family.
        "Fact_P_macro": lawa["reference_fact_precision"],
        "Fact_R_macro": lawa["reference_fact_recall_macro"],
        "Fact_F1_macro": lawa["reference_fact_f1"],
        "Legacy_conditional_fact_recall": lawa["fact_recall"],
        "Avg_chars": lawa["characters"],
        "Redundancy": lawa["pairwise_redundancy"],
        "Hard_truncations": lawa["hard_truncation_count"],
        "Budget_violations": lawa["budget_violations"],
    }
    if extra:
        row.update(extra)
    return row


def hard_coverage_select(sentences: list[str], scores: list[float]):
    """Force one feasible sentence from every present meaningful section, then fill normally."""
    normalized = base.minmax(scores)
    sections = section_impl.section_labels(sentences)
    bigrams = [base.char_bigrams(sentence) for sentence in sentences]
    present = {section for section in sections if section in MEANINGFUL}
    selected: list[int] = []
    covered: set[str] = set()
    remaining = set(range(len(sentences)))
    skipped: list[dict] = []

    def candidate_score(index: int):
        redundancy = max(
            (base.jaccard(bigrams[index], bigrams[other]) for other in selected),
            default=0.0,
        )
        return normalized[index] - REDUNDANCY_WEIGHT * redundancy, redundancy

    # Hard phase: candidates must come from an uncovered section whenever feasible.
    while present - covered and len(selected) < MAX_SENTENCES:
        uncovered = present - covered
        best_index, best_score = None, -float("inf")
        for index in sorted(remaining):
            if sections[index] not in uncovered:
                continue
            ordered = sorted(selected + [index])
            candidate = "".join(sentences[item] for item in ordered)
            score, redundancy = candidate_score(index)
            if len(candidate) > MAX_CHARS:
                skipped.append({
                    "phase": "hard_coverage", "index": index,
                    "section": sections[index], "score": score,
                    "redundancy": redundancy, "length": len(sentences[index]),
                })
                continue
            if score > best_score:
                best_index, best_score = index, score
        if best_index is None:
            break
        selected.append(best_index)
        covered.add(sections[best_index])
        remaining.remove(best_index)

    # Fill phase: use the frozen complete-sentence + redundancy objective.
    while remaining and len(selected) < MAX_SENTENCES:
        best_index, best_score = None, -float("inf")
        for index in sorted(remaining):
            ordered = sorted(selected + [index])
            candidate = "".join(sentences[item] for item in ordered)
            score, redundancy = candidate_score(index)
            if len(candidate) > MAX_CHARS:
                skipped.append({
                    "phase": "fill", "index": index,
                    "section": sections[index], "score": score,
                    "redundancy": redundancy, "length": len(sentences[index]),
                })
                continue
            if score > best_score:
                best_index, best_score = index, score
        if best_index is None:
            break
        selected.append(best_index)
        if sections[best_index] in MEANINGFUL:
            covered.add(sections[best_index])
        remaining.remove(best_index)
    return sorted(selected), skipped, sections, present, covered


def generate_hard_coverage(dev: list[dict], score_rows: list[list[float]]):
    outputs = []
    started = time.perf_counter()
    for number, (row, scores) in enumerate(zip(dev, score_rows), 1):
        sentences = base.sentences(row)
        selected, skipped, sections, present, covered = hard_coverage_select(sentences, scores)
        output = frozen.prediction_row(
            row, selected, decoder="budget_greedy",
            fact_weight=0.0, redundancy_weight=REDUNDANCY_WEIGHT,
            max_chars=MAX_CHARS, skipped_due_to_budget=skipped,
        )
        output["experimental_decoder"] = "hard_structure_coverage"
        output["selected_sections"] = [sections[index] for index in selected]
        output["available_meaningful_sections"] = sorted(present)
        output["covered_meaningful_sections"] = sorted(covered)
        output["section_coverage_rate"] = len(covered) / len(present) if present else 1.0
        outputs.append(output)
        if number % 200 == 0 or number == len(dev):
            print(f"generated hard coverage {number}/{len(dev)}", flush=True)
    return outputs, time.perf_counter() - started


def structure_diagnostics(predictions: list[dict], dev: list[dict]):
    covered_counts, available_counts, rates, complete = [], [], [], []
    for prediction, row in zip(predictions, dev):
        sections = section_impl.section_labels(base.sentences(row))
        available = {section for section in sections if section in MEANINGFUL}
        selected = set(map(int, prediction["selected_indices"]))
        covered = {sections[index] for index in selected if sections[index] in MEANINGFUL}
        rate = len(covered) / len(available) if available else 1.0
        available_counts.append(len(available))
        covered_counts.append(len(covered))
        rates.append(rate)
        complete.append(int(covered == available))
    return {
        "Avg_available_sections": float(np.mean(available_counts)),
        "Avg_covered_sections": float(np.mean(covered_counts)),
        "Macro_section_coverage_rate": float(np.mean(rates)),
        "Complete_coverage_doc_rate": float(np.mean(complete)),
    }


def main_table_rows(metrics: dict):
    return [
        metric_row("Full", metrics["full"], {"Variant": "Full"}),
        metric_row("Sentence supervision", metrics["minus_oracle"], {"Variant": "Supervision"}),
        metric_row("Official supervision", metrics["official_full"], {"Variant": "Supervision"}),
        metric_row("No legal representation", metrics["tfidf_only"], {"Variant": "Representation"}),
        metric_row("No budget decoder", metrics["post_truncation"], {"Variant": "Decoder"}),
    ]


def efficiency_rows(metrics: dict, timings: dict, dev_cases: int):
    mapping = [
        ("Logistic Regression", "oracle__full", "full"),
        ("Linear SVM", "oracle__svm_full", "svm_full"),
        ("LightGBM", "oracle__lightgbm_full", "lightgbm_full"),
    ]
    rows = []
    for label, model_key, variant_key in mapping:
        fit = float(timings["fit"][model_key]["fit_seconds"])
        score = float(timings["classifier_inference"][model_key]["classifier_score_seconds"])
        decode = float(timings["decoder"][variant_key]["decode_seconds"])
        end_to_end = score + decode
        row = metric_row(label, metrics[variant_key])
        row.update({
            "Fit_seconds": fit,
            "Classifier_score_seconds_942_docs": score,
            "Decode_seconds_942_docs": decode,
            "End_to_end_seconds_942_docs": end_to_end,
            "End_to_end_docs_per_second": dev_cases / end_to_end,
            "Relative_inference_time_vs_LR": None,
        })
        rows.append(row)
    baseline = rows[0]["End_to_end_seconds_942_docs"]
    for row in rows:
        row["Relative_inference_time_vs_LR"] = row["End_to_end_seconds_942_docs"] / baseline
    return rows


def alternative_summary(stage_rows: list[dict]):
    groups = {
        "Fact novelty reward": [row for row in stage_rows if row["id"].startswith("e1_")],
        "Section-conditioned TF-IDF": [row for row in stage_rows if row["id"].startswith("e2")],
        "Marginal-weighted Oracle": [row for row in stage_rows if row["id"].startswith("e3_")],
    }
    motivations = {
        "Fact novelty reward": "Reward newly introduced legal fact items during decoding",
        "Section-conditioned TF-IDF": "Learn section-specific lexical weights",
        "Marginal-weighted Oracle": "Preserve within-Oracle marginal-gain strength",
    }
    decisions = {
        "Fact novelty reward": "Not adopted: recall/density rise but precision, Fact-F1 and ROUGE fall",
        "Section-conditioned TF-IDF": "Not adopted: small ROUGE gain below threshold with Fact-F1 loss",
        "Marginal-weighted Oracle": "Not adopted: all weights underperform stable binary Oracle",
    }
    output = []
    for name, candidates in groups.items():
        best = max(candidates, key=lambda row: float(row["RL"]))
        output.append({
            "Alternative": name,
            "Motivation": motivations[name],
            "Best_configuration": best["method"],
            "Delta_R1": float(best["delta_R1"]),
            "Delta_R2": float(best["delta_R2"]),
            "Delta_RL": float(best["delta_RL"]),
            "Delta_Fact_F1": float(best["delta_Fact_F1"]),
            "Preregistered_success": int(best["preregistered_success"]),
            "Decision": decisions[name],
        })
    return output


def build_report(tables: dict, output_dir: Path):
    supervision = tables["supervision"]
    budget = tables["budget"]
    representation = tables["representation"]
    structure = tables["structure"]
    full, weak, official = supervision[2], supervision[0], supervision[1]
    post, length, complete = budget
    tfidf, sec6, legal = representation
    no_struct, soft, hard = structure
    lines = [
        "# Hypothesis-driven Ablation Results (dev-only)", "",
        "## Core claim", "",
        "The strongest evidence supports task alignment: summary-level budget-aware supervision and budget-consistent decoding matter more than increasing domain-feature complexity.", "",
        "## 1. Supervision target", "",
        f"Budget Oracle exceeds sentence supervision by {100*(full['RL']-weak['RL']):.2f} RL and {100*(full['Fact_F1_macro']-weak['Fact_F1_macro']):.2f} Fact-F1 points; it exceeds official supervision by {100*(full['RL']-official['RL']):.2f} RL and {100*(full['Fact_F1_macro']-official['Fact_F1_macro']):.2f} Fact-F1 points.", "",
        "## 2. Budget consistency", "",
        f"Length-aware complete-sentence selection improves over post-truncation by {100*(length['RL']-post['RL']):.2f} RL and removes {post['Hard_truncations']} hard truncations. Redundancy-aware selection adds a further {100*(complete['RL']-length['RL']):.2f} RL.", "",
        "## 3. Legal-aware representation", "",
        f"Section6 improves TF-IDF by {100*(sec6['RL']-tfidf['RL']):.2f} RL; the complete Legal14+Section6 representation improves TF-IDF by {100*(legal['RL']-tfidf['RL']):.2f} RL. Legal14 is therefore described as complementary rather than independently dominant.", "",
        "## 4. Structure utilization", "",
        f"Soft structure improves no-structure RL by {100*(soft['RL']-no_struct['RL']):.2f}. True hard coverage changes macro section coverage from {soft['Macro_section_coverage_rate']:.3f} to {hard['Macro_section_coverage_rate']:.3f}, with an RL change of {100*(hard['RL']-soft['RL']):+.2f}; this tests whether maximizing coverage is equivalent to improving summary quality.", "",
        "## Fact metric audit", "",
        "Paper tables must use the internally consistent family `reference_fact_precision`, `reference_fact_recall_macro`, and macro `reference_fact_f1`. The legacy conditional `fact_recall` averages only reference-bearing fact types and must not be paired with the other two columns. Macro Fact-F1 is the mean of per-type F1 values, not the harmonic mean of displayed macro P and macro R.", "",
        "## Paper wording", "",
        "Recommended: Legal-aware representations provide complementary gains on top of strong budget-aligned supervision. Budget Oracle is the primary mechanism; legal features guide selection; the decoder enforces feasible complete-sentence realization.", "",
        "Alternative design analysis belongs in the appendix and documents why more aggressive novelty, interaction, and marginal weighting were not adopted.", "",
    ]
    (output_dir / "HYPOTHESIS_ABLATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--v22-model", type=Path, required=True)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--alternative-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_new_output(args.output_dir)
    dev_hash = formal.require(args.dev, formal.EXPECTED_DEV_SHA256, "dev")
    v22_hash = formal.require(args.v22_model, formal.EXPECTED_V22_MODEL_SHA256, "v2.2 model")
    sources = [
        verify_source(args.formal_dir, "formal_ablation"),
        verify_source(args.alternative_dir, "alternative_design"),
    ]
    formal_metrics = read_json(args.formal_dir / "all_metrics.json")
    label_diagnostics = read_json(args.formal_dir / "label_diagnostics.json")
    timings = read_json(args.formal_dir / "fit_and_inference_timings.json")
    alternative_rows = read_csv(args.alternative_dir / "tables" / "Stage1_all_10_configurations.csv")
    dev = base.read_jsonl(args.dev)
    if len(dev) != formal.DEV_CASES:
        raise RuntimeError(f"unexpected dev size: {len(dev)}")

    # Score the frozen Full model once; no retraining is needed.
    with args.v22_model.open("rb") as handle:
        vectorizer = pickle.load(handle)["vectorizer"]
    with (args.formal_dir / "models" / "oracle__full.pkl").open("rb") as handle:
        full_model = pickle.load(handle)
    scores, score_timing = formal.precompute_dev_scores(dev, {"full": full_model}, vectorizer)
    hard_predictions, hard_decode_seconds = generate_hard_coverage(dev, scores["full"])
    formal.write_jsonl(args.output_dir / "predictions" / "hard_structure_coverage.jsonl", hard_predictions)
    hard_metrics = formal.evaluate_variant(
        "hard_structure_coverage", hard_predictions, dev, MAX_CHARS, args.output_dir
    )

    full_predictions = base.read_jsonl(args.formal_dir / "predictions" / "full.jsonl")
    no_structure_predictions = base.read_jsonl(args.formal_dir / "predictions" / "minus_section6.jsonl")
    structure_no = structure_diagnostics(no_structure_predictions, dev)
    structure_soft = structure_diagnostics(full_predictions, dev)
    structure_hard = structure_diagnostics(hard_predictions, dev)

    supervision = [
        metric_row("Sentence-level ROUGE", formal_metrics["minus_oracle"], {
            "Label_source": "Independent sentence match", "Budget_aware_label": 0,
        }),
        metric_row("Official CAIL label", formal_metrics["official_full"], {
            "Label_source": "Official sentence label", "Budget_aware_label": 0,
        }),
        metric_row("Budget Oracle", formal_metrics["full"], {
            "Label_source": "Summary-level combination optimization", "Budget_aware_label": 1,
        }),
    ]
    supervision_diag = []
    for key, label in (("weak", "Sentence-level ROUGE"), ("official", "Official CAIL label"), ("oracle", "Budget Oracle")):
        row = label_diagnostics[key]
        supervision_diag.append({
            "Method": label,
            "Positive_ratio": row["positive_rate"],
            "Oracle_precision": row["precision_vs_oracle"],
            "Oracle_recall": row["recall_vs_oracle"],
            "Oracle_F1": row["f1_vs_oracle"],
            "Avg_positive_sentences": row["average_positive_sentences_per_document"],
            "Budgeted_marginal_gain": row["budgeted_source_order_marginal_gain"]["mean"],
        })
    budget = [
        metric_row("Post-truncation", formal_metrics["post_truncation"], {"Budget_check_during_selection": 0, "Redundancy_aware": 0}),
        metric_row("Length-aware sentence selection", formal_metrics["complete_no_redundancy"], {"Budget_check_during_selection": 1, "Redundancy_aware": 0}),
        metric_row("Length + redundancy-aware", formal_metrics["full"], {"Budget_check_during_selection": 1, "Redundancy_aware": 1}),
    ]
    representation = [
        metric_row("TF-IDF", formal_metrics["tfidf_only"]),
        metric_row("TF-IDF + Section6", formal_metrics["minus_legal14"]),
        metric_row("TF-IDF + Legal14 + Section6", formal_metrics["full"]),
    ]
    representation_appendix = [
        metric_row("TF-IDF", formal_metrics["tfidf_only"], {"Legal14": 0, "Section6": 0}),
        metric_row("TF-IDF + Legal14", formal_metrics["minus_section6"], {"Legal14": 1, "Section6": 0}),
        metric_row("TF-IDF + Section6", formal_metrics["minus_legal14"], {"Legal14": 0, "Section6": 1}),
        metric_row("TF-IDF + Legal14 + Section6", formal_metrics["full"], {"Legal14": 1, "Section6": 1}),
    ]
    structure = [
        metric_row("No structure", formal_metrics["minus_section6"], structure_no),
        metric_row("Soft structure feature", formal_metrics["full"], structure_soft),
        metric_row("Hard structure coverage", hard_metrics, structure_hard),
    ]
    main_components = main_table_rows(formal_metrics)
    alt_summary = alternative_summary(alternative_rows)
    budget_curve = []
    for value in (200, 250, 300, 350, 400, 450):
        key = "full" if value == 450 else f"budget_{value}"
        budget_curve.append(metric_row(f"B={value}", formal_metrics[key], {"Budget_chars": value}))
    efficiency = efficiency_rows(formal_metrics, timings, len(dev))

    tables = {
        "supervision": supervision,
        "supervision_diagnostics": supervision_diag,
        "budget": budget,
        "representation": representation,
        "representation_appendix": representation_appendix,
        "structure": structure,
        "main_components": main_components,
        "alternative_summary": alt_summary,
        "alternative_all": alternative_rows,
        "budget_curve": budget_curve,
        "efficiency": efficiency,
    }
    filenames = {
        "supervision": "Table_1_supervision_target.csv",
        "supervision_diagnostics": "Table_1b_supervision_diagnostics.csv",
        "budget": "Table_2_budget_consistency.csv",
        "representation": "Table_3_legal_representation.csv",
        "representation_appendix": "Appendix_legal_representation_2x2.csv",
        "structure": "Table_4_structure_utilization.csv",
        "main_components": "Main_Table_component_analysis.csv",
        "alternative_summary": "Appendix_alternative_design_summary.csv",
        "alternative_all": "Appendix_alternative_design_all.csv",
        "budget_curve": "Figure_budget_curve_consistent_fact_metrics.csv",
        "efficiency": "Appendix_scorer_efficiency_corrected.csv",
    }
    for key, rows in tables.items():
        formal.write_csv(args.output_dir / "tables" / filenames[key], rows)
    build_report(tables, args.output_dir)

    compact_hard = {key: value for key, value in hard_metrics.items() if key != "lawa_details"}
    config = {
        "experiment": EXPERIMENT,
        "status": "dev_complete_test_untouched",
        "test_argument_exists": False,
        "retraining_performed": False,
        "new_experiment": "true_hard_structure_coverage_only",
        "dev_cases": len(dev),
        "dev_sha256": dev_hash,
        "v22_model_sha256": v22_hash,
        "formal_sources": sources,
        "hard_coverage_definition": "Force one feasible sentence from each meaningful section present in the document, then fill with the frozen probability-minus-redundancy objective.",
        "hard_coverage_score_seconds": score_timing["full"]["classifier_score_seconds"],
        "hard_coverage_decode_seconds": hard_decode_seconds,
        "script_sha256": formal.sha256(Path(__file__).resolve()),
    }
    formal.save_json(args.output_dir / "hard_structure_metrics.json", compact_hard)
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
        "hard_structure_lawa": compact_hard["lawa"],
        "hard_structure_diagnostics": structure_hard,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
