#!/usr/bin/env python3
"""Research-1 closure: E1 supervision x decoder 2x2 on frozen IN-Abs dev.

The only training-target change is ordinary versus P50 budget-aware Oracle
membership. Both models share the already frozen E1 word 1-2 gram TF-IDF
vectorizer, Legal14+Section6 features, balanced liblinear LR, candidate set,
MMR weight, and data rows. Decoder variants are complete-sentence fit/skip and
score-order selection with a final sentence fragment truncated to the exact
spaCy-counted budget. Official test contents are never read.
"""
from __future__ import annotations

import argparse
import json
import pickle
import string
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_inabs_phase1 as phase1
import run_inabs_phase2_lr as phase2
import run_inabs_phase3c_spacy_decode as phase3c

import numpy as np
import spacy
from rouge import Rouge
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression


SEED = 42
FIXED_BUDGETS = (423, 643, 973)
SUPERVISIONS = ("ordinary", "budget")
DECODERS = ("complete", "truncate")
CONFIGS = tuple(f"{supervision}_{decoder}" for supervision in SUPERVISIONS for decoder in DECODERS)
METRICS = ("rouge-1", "rouge-2", "rouge-l")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_protocol(dataset_dir: Path, train_limit: int, dev_limit: int):
    train_part = dataset_dir / "train-data"
    official_ids = phase1.paired_ids(train_part)
    if len(official_ids) != phase1.EXPECTED_TRAIN:
        raise RuntimeError(f"unexpected official train count: {len(official_ids)}")
    eligible, excluded = phase1.eligible_train_ids(train_part, official_ids)
    train_ids, dev_ids = phase1.fixed_split(eligible)
    if len(train_ids) != 6325 or len(dev_ids) != 703:
        raise RuntimeError(f"unexpected frozen split: {len(train_ids)}/{len(dev_ids)}")
    selected_train = train_ids[:train_limit] if train_limit else train_ids
    selected_dev = dev_ids[:dev_limit] if dev_limit else dev_ids
    train_docs = [phase1.load_case(train_part, case_id) for case_id in selected_train]
    return train_part, train_ids, dev_ids, train_docs, selected_dev, excluded


def load_oracles(path: Path, train_docs: list[dict]) -> list[dict]:
    rows = read_jsonl(path)[: len(train_docs)]
    if [row["id"] for row in rows] != [doc["id"] for doc in train_docs]:
        raise RuntimeError("phase-2 Oracle cache does not match frozen train prefix")
    return rows


def build_matrix(vectorizer, texts, positions):
    decorated = [phase2.decorated(text, index, total) for text, (index, total) in zip(texts, positions)]
    text_matrix = vectorizer.transform(decorated)
    extras = csr_matrix(
        [phase2.explicit20(text, index, total) for text, (index, total) in zip(texts, positions)],
        dtype=np.float32,
    )
    return hstack([text_matrix, extras], format="csr")


def fit_budget_model(matrix, labels):
    print(json.dumps({
        "phase": "fit_budget_lr",
        "rows": len(labels),
        "features": matrix.shape[1],
        "positive_rate": float(labels.mean()),
    }), flush=True)
    return LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=1000, random_state=SEED
    ).fit(matrix, labels)


def countable(token) -> bool:
    table = str.maketrans("", "", string.punctuation + string.whitespace)
    return len(token.text.translate(table)) > 0


def truncate_sentence(nlp, sentence: str, allowed_words: int) -> str:
    if allowed_words <= 0:
        return ""
    doc = nlp(sentence)
    used = 0
    end = 0
    for token in doc:
        if countable(token):
            if used >= allowed_words:
                break
            used += 1
        end = token.idx + len(token.text)
    fragment = sentence[:end].strip()
    verified = phase3c.official_counts(nlp, [fragment])[0] if fragment else 0
    if verified > allowed_words:
        raise RuntimeError("sentence truncation exceeded the official word allowance")
    return fragment


def decode_truncate(nlp, sentences, candidates, probabilities, lengths, budget):
    relevance = phase3c.normalize(probabilities)
    grams = [phase2.word_bigram_set(sentences[index]) for index in candidates]
    remaining = set(range(len(candidates)))
    selected_text: dict[int, str] = {}
    selected_indices: list[int] = []
    used = 0
    truncated_index = None
    max_redundancy = np.zeros(len(candidates), dtype=np.float32)
    while remaining and used < budget:
        valid = [item for item in remaining if lengths[item] > 0]
        if not valid:
            break
        best = max(
            valid,
            key=lambda item: (
                float(relevance[item]) - phase2.REDUNDANCY_WEIGHT * float(max_redundancy[item]),
                float(relevance[item]),
                -candidates[item],
            ),
        )
        source_index = candidates[best]
        available = budget - used
        if lengths[best] <= available:
            text = sentences[source_index]
            words = lengths[best]
        else:
            text = truncate_sentence(nlp, sentences[source_index], available)
            words = phase3c.official_counts(nlp, [text])[0] if text else 0
            truncated_index = source_index
        if text and words:
            selected_text[source_index] = text
            selected_indices.append(source_index)
            used += words
        remaining.remove(best)
        if truncated_index is not None:
            break
        for item in remaining:
            max_redundancy[item] = max(
                max_redundancy[item], phase2.jaccard(grams[item], grams[best])
            )
    prediction = " ".join(selected_text[index] for index in sorted(selected_text))
    verified = phase3c.official_counts(nlp, [prediction])[0] if prediction else 0
    if verified != used or verified > budget:
        raise RuntimeError("truncate decoder word accounting mismatch")
    return sorted(selected_indices), prediction, used, truncated_index


def bootstrap(values: np.ndarray, samples: int = 10000) -> dict:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    means = np.empty(samples, dtype=float)
    for start in range(0, samples, 250):
        size = min(250, samples - start)
        means[start : start + size] = values[
            rng.integers(0, len(values), size=(size, len(values)))
        ].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "p_two_sided": float(min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0)))),
    }


def score_arrays(hypotheses: list[str], references: list[str]) -> dict[str, np.ndarray]:
    evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False,
        apply_avg=False, apply_best=False,
    )
    raw = evaluator.get_scores(hypotheses, [[reference] for reference in references])
    return {
        metric: np.asarray([float(item["f"][0]) for item in raw[metric]], dtype=float)
        for metric in METRICS
    }


def evaluate(train_part, dev_ids, vectorizer, models, nlp, output_dir: Path):
    tracks = ("423", "643", "973", "official_reference_length")
    hypotheses = {track: {config: [] for config in CONFIGS} for track in tracks}
    references: list[str] = []
    rows: list[dict] = []
    for position, case_id in enumerate(dev_ids, 1):
        case = phase1.load_case(train_part, case_id)
        candidates = phase2.candidate_indices(len(case["sentences"]))
        matrix = phase3c.feature_matrix(vectorizer, case["sentences"], candidates)
        probabilities = {
            supervision: models[supervision].predict_proba(matrix)[:, 1]
            for supervision in SUPERVISIONS
        }
        reference_words = phase3c.official_counts(nlp, [case["reference"]])[0]
        candidate_lengths = phase3c.official_counts(
            nlp, [case["sentences"][index] for index in candidates]
        )
        budgets = {"423": 423, "643": 643, "973": 973, "official_reference_length": reference_words}
        detail = {
            "id": case_id,
            "reference_spacy_words": reference_words,
            "candidate_count": len(candidates),
            "tracks": {},
        }
        references.append(case["reference"])
        for track, budget in budgets.items():
            detail["tracks"][track] = {"budget": budget, "configs": {}}
            for supervision in SUPERVISIONS:
                selected, used = phase3c.decode_with_lengths(
                    case["sentences"], candidates, probabilities[supervision], candidate_lengths, budget
                )
                prediction = phase1.summary(case["sentences"], selected)
                verified = phase3c.official_counts(nlp, [prediction])[0] if prediction else 0
                if verified != used or verified > budget:
                    raise RuntimeError(f"complete decoder budget error for {case_id}")
                config = f"{supervision}_complete"
                hypotheses[track][config].append(prediction)
                detail["tracks"][track]["configs"][config] = {
                    "selected_indices": selected,
                    "prediction_words": verified,
                    "budget_utilization": verified / budget if budget else 0.0,
                    "budget_violation": verified > budget,
                    "truncated_index": None,
                }

                trunc_selected, trunc_prediction, trunc_used, trunc_index = decode_truncate(
                    nlp, case["sentences"], candidates, probabilities[supervision], candidate_lengths, budget
                )
                config = f"{supervision}_truncate"
                hypotheses[track][config].append(trunc_prediction)
                detail["tracks"][track]["configs"][config] = {
                    "selected_indices": trunc_selected,
                    "prediction_words": trunc_used,
                    "budget_utilization": trunc_used / budget if budget else 0.0,
                    "budget_violation": trunc_used > budget,
                    "truncated_index": trunc_index,
                }
        rows.append(detail)
        if position % 25 == 0 or position == len(dev_ids):
            print(json.dumps({"phase": "decode", "docs": position, "total": len(dev_ids)}), flush=True)

    arrays = {}
    for track in tracks:
        arrays[track] = {}
        for config in CONFIGS:
            print(json.dumps({"phase": "pyrouge", "track": track, "config": config}), flush=True)
            arrays[track][config] = score_arrays(hypotheses[track][config], references)

    for index, row in enumerate(rows):
        for track in tracks:
            for config in CONFIGS:
                row["tracks"][track]["configs"][config]["pyrouge"] = {
                    metric: float(arrays[track][config][metric][index]) for metric in METRICS
                }

    report = {}
    for track in tracks:
        report[track] = {"configs": {}, "paired_bootstrap": {}, "selection_change_rate": {}}
        for config in CONFIGS:
            word_values = np.asarray([
                row["tracks"][track]["configs"][config]["prediction_words"] for row in rows
            ], dtype=float)
            utilization = np.asarray([
                row["tracks"][track]["configs"][config]["budget_utilization"] for row in rows
            ], dtype=float)
            report[track]["configs"][config] = {
                "f1_percent": {
                    metric: 100 * float(arrays[track][config][metric].mean()) for metric in METRICS
                },
                "mean_prediction_words": float(word_values.mean()),
                "mean_budget_utilization": float(utilization.mean()),
                "budget_violations": int(sum(
                    row["tracks"][track]["configs"][config]["budget_violation"] for row in rows
                )),
                "truncated_cases": int(sum(
                    row["tracks"][track]["configs"][config]["truncated_index"] is not None for row in rows
                )),
            }
        for decoder in DECODERS:
            left, right = f"ordinary_{decoder}", f"budget_{decoder}"
            report[track]["paired_bootstrap"][f"{right}_minus_{left}"] = {
                metric: bootstrap(arrays[track][right][metric] - arrays[track][left][metric])
                for metric in METRICS
            }
            report[track]["selection_change_rate"][f"budget_minus_ordinary_{decoder}"] = float(np.mean([
                row["tracks"][track]["configs"][left]["selected_indices"]
                != row["tracks"][track]["configs"][right]["selected_indices"]
                for row in rows
            ]))
        for supervision in SUPERVISIONS:
            left, right = f"{supervision}_truncate", f"{supervision}_complete"
            report[track]["paired_bootstrap"][f"{right}_minus_{left}"] = {
                metric: bootstrap(arrays[track][right][metric] - arrays[track][left][metric])
                for metric in METRICS
            }
        report[track]["paired_bootstrap"]["supervision_x_decoder_interaction"] = {
            metric: bootstrap(
                (arrays[track]["budget_complete"][metric] - arrays[track]["ordinary_complete"][metric])
                - (arrays[track]["budget_truncate"][metric] - arrays[track]["ordinary_truncate"][metric])
            )
            for metric in METRICS
        }

    with (output_dir / "dev_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--phase2-oracles", type=Path, required=True)
    parser.add_argument("--phase3-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if phase1.md5(args.archive) != phase1.EXPECTED_ARCHIVE_MD5:
        raise RuntimeError("official archive MD5 mismatch")
    train_part, all_train_ids, all_dev_ids, train_docs, selected_dev_ids, excluded = load_protocol(
        args.dataset_dir, args.train_limit, args.dev_limit
    )
    oracles = load_oracles(args.phase2_oracles, train_docs)
    texts, positions, y_ordinary, y_budget, audit = phase2.build_shared_training(train_docs, oracles)
    with args.phase3_model.open("rb") as handle:
        phase3_bundle = pickle.load(handle)
    vectorizer = phase3_bundle["vectorizers"]["word"]
    ordinary_model = phase3_bundle["models"]["e1_word1_2"]
    started = time.time()
    matrix = build_matrix(vectorizer, texts, positions)
    if ordinary_model.n_features_in_ != matrix.shape[1]:
        raise RuntimeError("frozen E1 ordinary model feature width mismatch")
    budget_model = fit_budget_model(matrix, y_budget)
    models = {"ordinary": ordinary_model, "budget": budget_model}
    with (args.output_dir / "model_bundle.pkl").open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "models": models}, handle)
    try:
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    except Exception as error:
        raise RuntimeError("official en_core_web_sm tokenizer is required") from error
    report = evaluate(train_part, selected_dev_ids, vectorizer, models, nlp, args.output_dir)
    write_json(args.output_dir / "metrics.json", report)
    write_json(args.output_dir / "training_label_audit.json", audit)
    config = {
        "experiment": "Research-1 E1 Oracle-supervision x decoder 2x2 closure",
        "seed": SEED,
        "frozen_split": {"train": len(all_train_ids), "dev": len(all_dev_ids)},
        "run_documents": {"train": len(train_docs), "dev": len(selected_dev_ids)},
        "representation": "frozen E1 word 1-2 gram TF-IDF + Legal14 + Section6",
        "training_rows_shared": audit["shared_training_rows"],
        "supervision": ["ordinary greedy Oracle", "P50=643-word budget-aware Oracle"],
        "decoders": ["complete-sentence fit/skip", "score-order with final sentence truncation"],
        "tracks": [423, 643, 973, "per-document reference spaCy words"],
        "candidate_rule": "reference-free evenly spaced maximum 400 sentences",
        "word_counter": "official Gist spaCy en_core_web_sm punctuation-filtered tokens",
        "rouge": "py-rouge==1.1, max_n=2, limit_length=False",
        "official_test_contents_read": False,
        "excluded_empty_train_pairs": excluded,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "config.json", config)
    print(json.dumps({"phase": "complete", "output": str(args.output_dir), **config}, indent=2), flush=True)


if __name__ == "__main__":
    main()
