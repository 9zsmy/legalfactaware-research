#!/usr/bin/env python3
"""Reference-aware state diagnostics on frozen IN-Abs dev; no training/test."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import pickle
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_inabs_phase1 as phase1
import run_inabs_phase2_lr as phase2
import run_inabs_phase3c_spacy_decode as phase3c

import numpy as np
import spacy
from rouge import Rouge


SEED = 42
BUDGET = 643
EPSILON = 1e-12


def marginal(current: Counter, addition: Counter, reference: Counter) -> int:
    return sum(
        min(current.get(key, 0) + value, reference.get(key, 0))
        - min(current.get(key, 0), reference.get(key, 0))
        for key, value in addition.items()
    )


def sample_bootstrap(delta: np.ndarray, samples: int = 10000) -> dict:
    rng = np.random.default_rng(SEED)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 250):
        size = min(250, samples - start)
        draw = rng.integers(0, len(delta), size=(size, len(delta)))
        means[start:start + size] = delta[draw].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def diagnose(case: dict, candidates: list[int], lengths: list[int],
             probabilities: np.ndarray) -> tuple[list[int], dict]:
    sentences = case["sentences"]
    stats = phase1.sentence_statistics(sentences, case["reference"], candidates)
    reference1, reference2 = stats["ref1"], stats["ref2"]
    reference_total1, reference_total2 = sum(reference1.values()), sum(reference2.values())
    sentence1 = [stats["sent1"][index] for index in candidates]
    sentence2 = [stats["sent2"][index] for index in candidates]
    totals1 = [sum(value.values()) for value in sentence1]
    totals2 = [sum(value.values()) for value in sentence2]
    grams = [phase2.word_bigram_set(sentences[index]) for index in candidates]
    relevance = phase3c.normalize(probabilities)

    current1, current2 = Counter(), Counter()
    overlap1 = overlap2 = total1 = total2 = used = 0
    selected_positions: list[int] = []
    remaining = set(range(len(candidates)))
    maximum_redundancy = np.zeros(len(candidates), dtype=np.float32)
    initial_gains: dict[int, float] = {}
    diagnostics = {
        "states_after_first": 0,
        "top_choice_flips": 0,
        "pairwise_comparisons": 0,
        "pairwise_reversals": 0,
        "initial_positive_available": 0,
        "became_nonpositive": 0,
        "oracle_steps": 0,
        "mmr_agreements": 0,
    }

    while remaining:
        fitting = [position for position in sorted(remaining)
                   if lengths[position] > 0 and used + lengths[position] <= BUDGET]
        if not fitting:
            break
        current_score = phase1.oracle_objective(
            overlap1, total1, reference_total1, overlap2, total2, reference_total2
        )
        gains = {}
        increments = {}
        for position in fitting:
            inc1 = marginal(current1, sentence1[position], reference1)
            inc2 = marginal(current2, sentence2[position], reference2)
            score = phase1.oracle_objective(
                overlap1 + inc1, total1 + totals1[position], reference_total1,
                overlap2 + inc2, total2 + totals2[position], reference_total2,
            )
            gains[position] = score - current_score
            increments[position] = (inc1, inc2)
        if not initial_gains:
            initial_gains = gains.copy()
        else:
            diagnostics["states_after_first"] += 1
            initial_top = max(fitting, key=lambda p: (initial_gains[p], -candidates[p]))
            current_top = max(fitting, key=lambda p: (gains[p], -candidates[p]))
            diagnostics["top_choice_flips"] += initial_top != current_top
            positive = [p for p in fitting if initial_gains[p] > EPSILON]
            diagnostics["initial_positive_available"] += len(positive)
            diagnostics["became_nonpositive"] += sum(gains[p] <= EPSILON for p in positive)
            top = sorted(fitting, key=lambda p: (-initial_gains[p], candidates[p]))[:50]
            for left_idx, left in enumerate(top):
                for right in top[left_idx + 1:]:
                    initial_difference = initial_gains[left] - initial_gains[right]
                    current_difference = gains[left] - gains[right]
                    if abs(initial_difference) <= EPSILON or abs(current_difference) <= EPSILON:
                        continue
                    diagnostics["pairwise_comparisons"] += 1
                    diagnostics["pairwise_reversals"] += initial_difference * current_difference < 0

        oracle_choice = max(fitting, key=lambda p: (gains[p], -candidates[p]))
        if gains[oracle_choice] <= EPSILON:
            break
        mmr_choice = max(
            fitting,
            key=lambda p: (
                float(relevance[p]) - phase2.REDUNDANCY_WEIGHT * float(maximum_redundancy[p]),
                float(relevance[p]), -candidates[p],
            ),
        )
        diagnostics["oracle_steps"] += 1
        diagnostics["mmr_agreements"] += oracle_choice == mmr_choice
        selected_positions.append(oracle_choice)
        remaining.remove(oracle_choice)
        inc1, inc2 = increments[oracle_choice]
        current1.update(sentence1[oracle_choice])
        current2.update(sentence2[oracle_choice])
        overlap1 += inc1
        overlap2 += inc2
        total1 += totals1[oracle_choice]
        total2 += totals2[oracle_choice]
        used += lengths[oracle_choice]
        for position in remaining:
            maximum_redundancy[position] = max(
                maximum_redundancy[position],
                phase2.jaccard(grams[position], grams[oracle_choice]),
            )
    return sorted(candidates[p] for p in selected_positions), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--phase3-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-limit", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_part = args.dataset_dir / "train-data"
    eligible, excluded = phase1.eligible_train_ids(train_part, phase1.paired_ids(train_part))
    train_ids, dev_ids = phase1.fixed_split(eligible)
    if len(train_ids) != 6325 or len(dev_ids) != 703:
        raise RuntimeError("frozen split mismatch")
    if args.dev_limit:
        dev_ids = dev_ids[:args.dev_limit]
    with args.phase3_model.open("rb") as stream:
        bundle = pickle.load(stream)
    vectorizer = bundle["vectorizers"]["word"]
    classifier = bundle["models"]["e1_word1_2"]
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    evaluator = Rouge(metrics=["rouge-n", "rouge-l"], max_n=2,
                      limit_length=False, apply_avg=False, apply_best=False)
    records, oracle_texts, e1_texts, references = [], [], [], []
    started = time.time()
    with (args.output_dir / "dev_records.jsonl").open("w", encoding="utf-8") as output:
        for number, case_id in enumerate(dev_ids, 1):
            case = phase1.load_case(train_part, case_id)
            candidates = phase2.candidate_indices(len(case["sentences"]))
            matrix = phase3c.feature_matrix(vectorizer, case["sentences"], candidates)
            probabilities = classifier.predict_proba(matrix)[:, 1]
            lengths = phase3c.official_counts(nlp, [case["sentences"][p] for p in candidates])
            oracle_indices, diagnostics = diagnose(case, candidates, lengths, probabilities)
            e1_indices, e1_used = phase3c.decode_with_lengths(
                case["sentences"], candidates, probabilities, lengths, BUDGET
            )
            oracle_text = phase1.summary(case["sentences"], oracle_indices)
            e1_text = phase1.summary(case["sentences"], e1_indices)
            oracle_used, e1_verified = phase3c.official_counts(nlp, [oracle_text, e1_text])
            if oracle_used > BUDGET or e1_verified != e1_used or e1_used > BUDGET:
                raise RuntimeError(f"budget violation: {case_id}")
            row = {
                "id": case_id, "candidate_count": len(candidates),
                "budget_words": BUDGET, "oracle_words": oracle_used,
                "e1_words": e1_used, "oracle_selected": oracle_indices,
                "e1_selected": e1_indices, **diagnostics,
            }
            output.write(json.dumps(row) + "\n")
            output.flush()
            records.append(row)
            oracle_texts.append(oracle_text)
            e1_texts.append(e1_text)
            references.append(case["reference"])
            if number % 25 == 0 or number == len(dev_ids):
                print(json.dumps({"stage": "diagnose", "done": number,
                                  "total": len(dev_ids),
                                  "elapsed_seconds": round(time.time() - started, 1)}), flush=True)

    oracle_raw = evaluator.get_scores(oracle_texts, [[text] for text in references])
    e1_raw = evaluator.get_scores(e1_texts, [[text] for text in references])
    metrics = ("rouge-1", "rouge-2", "rouge-l")
    oracle_scores = {m: np.asarray([float(row["f"][0]) for row in oracle_raw[m]]) for m in metrics}
    e1_scores = {m: np.asarray([float(row["f"][0]) for row in e1_raw[m]]) for m in metrics}
    sum_field = lambda field: sum(row[field] for row in records)
    rate = lambda num, den: float(num / den) if den else None
    report = {
        "experiment": "IN-Abs Phase 6A summary-state diagnostics",
        "dev_documents": len(dev_ids), "train_documents": len(train_ids),
        "official_test_contents_read": False,
        "trained_model": False, "e1_model_changed": False,
        "budget_words": BUDGET, "budget_violations": 0,
        "oracle_kind": "reference-aware greedy ROUGE-1+2 multiset F1 proxy; not py-rouge upper bound",
        "top_choice_flip_rate": rate(sum_field("top_choice_flips"), sum_field("states_after_first")),
        "pairwise_reversal_rate_top50": rate(sum_field("pairwise_reversals"), sum_field("pairwise_comparisons")),
        "positive_gain_to_nonpositive_rate": rate(sum_field("became_nonpositive"), sum_field("initial_positive_available")),
        "e1_mmr_oracle_state_agreement_rate": rate(sum_field("mmr_agreements"), sum_field("oracle_steps")),
        "counts": {field: sum_field(field) for field in (
            "states_after_first", "top_choice_flips", "pairwise_comparisons",
            "pairwise_reversals", "initial_positive_available", "became_nonpositive",
            "oracle_steps", "mmr_agreements")},
        "pyrouge_f1_percent": {
            "e1": {m: 100 * float(e1_scores[m].mean()) for m in metrics},
            "greedy_reference_aware": {m: 100 * float(oracle_scores[m].mean()) for m in metrics},
        },
        "paired_greedy_minus_e1": {
            m: sample_bootstrap(oracle_scores[m] - e1_scores[m]) for m in metrics
        },
        "excluded_empty_source_pairs": excluded,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "complete", **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
