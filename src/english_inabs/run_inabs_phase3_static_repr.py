#!/usr/bin/env python3
"""IN-Abs phase 3: frozen ordinary-Oracle LR English representation study.

E0 is the phase-2 char 2-4 model. E1 uses word 1-2 grams. E2 combines
word 1-2 and char 3-5 grams. All other variables and all training rows are
identical. Official test contents are not read.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_inabs_phase1 as phase1
import run_inabs_phase2_lr as phase2

import numpy as np
from rouge_score import rouge_scorer
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


FIXED_BUDGETS = (423, 643, 973)
MODELS = ("e0_char2_4", "e1_word1_2", "e2_word1_2_char3_5")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_protocol(dataset_dir: Path, train_limit: int, dev_limit: int):
    train_part = dataset_dir / "train-data"
    official_ids = phase1.paired_ids(train_part)
    if len(official_ids) != phase1.EXPECTED_TRAIN:
        raise RuntimeError(f"unexpected official train count: {len(official_ids)}")
    eligible, excluded = phase1.eligible_train_ids(train_part, official_ids)
    train_ids, dev_ids = phase1.fixed_split(eligible)
    if len(train_ids) != 6325 or len(dev_ids) != 703:
        raise RuntimeError(f"unexpected fixed split: {len(train_ids)}/{len(dev_ids)}")
    selected_train = train_ids[:train_limit] if train_limit else train_ids
    selected_dev = dev_ids[:dev_limit] if dev_limit else dev_ids
    train_docs = [phase1.load_case(train_part, case_id) for case_id in selected_train]
    dev_docs = [phase1.load_case(train_part, case_id) for case_id in selected_dev]
    return train_docs, dev_docs, excluded


def load_oracles(path: Path, train_docs: list[dict]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[:len(train_docs)]
    if [row["id"] for row in rows] != [doc["id"] for doc in train_docs]:
        raise RuntimeError("phase-2 Oracle cache does not match phase-3 train IDs")
    return rows


def explicit_matrix(texts, positions):
    return csr_matrix(
        [phase2.explicit20(text, index, total) for text, (index, total) in zip(texts, positions)],
        dtype=np.float32,
    )


def fit_lr(matrix, labels, name: str):
    print(json.dumps({"phase": "fit_lr", "model": name, "rows": len(labels), "features": matrix.shape[1], "positive_rate": float(labels.mean())}), flush=True)
    return LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=1000, random_state=phase2.SEED
    ).fit(matrix, labels)


def fit_e1_e2(texts, positions, labels):
    decorated = [phase2.decorated(text, index, total) for text, (index, total) in zip(texts, positions)]
    extras = explicit_matrix(texts, positions)
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=120000,
        sublinear_tf=True, dtype=np.float32,
    )
    word_matrix = word.fit_transform(decorated)
    e1_matrix = hstack([word_matrix, extras], format="csr")
    e1 = fit_lr(e1_matrix, labels, "e1_word1_2")
    char = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=2, max_features=120000,
        sublinear_tf=True, dtype=np.float32,
    )
    char_matrix = char.fit_transform(decorated)
    e2_matrix = hstack([word_matrix, char_matrix, extras], format="csr")
    e2 = fit_lr(e2_matrix, labels, "e2_word1_2_char3_5")
    return {"word": word, "char": char}, {"e1_word1_2": e1, "e2_word1_2_char3_5": e2}


def matrices(vectorizers, e0_vectorizer, sentences: list[str], candidates: list[int]):
    total = len(sentences)
    decorated = [phase2.decorated(sentences[i], i, total) for i in candidates]
    extras = csr_matrix([phase2.explicit20(sentences[i], i, total) for i in candidates], dtype=np.float32)
    word = vectorizers["word"].transform(decorated)
    char = vectorizers["char"].transform(decorated)
    e0_text = e0_vectorizer.transform(decorated)
    return {
        "e0_char2_4": hstack([e0_text, extras], format="csr"),
        "e1_word1_2": hstack([word, extras], format="csr"),
        "e2_word1_2_char3_5": hstack([word, char, extras], format="csr"),
    }


def bootstrap(rows, track: str, left: str, right: str, metric: str, samples: int = 10000):
    delta = np.asarray([row["scores"][track][right][metric] - row["scores"][track][left][metric] for row in rows])
    rng = np.random.default_rng(phase2.SEED)
    means = np.empty(samples, dtype=float)
    for start in range(0, samples, 250):
        size = min(250, samples - start)
        means[start:start+size] = delta[rng.integers(0, len(delta), size=(size, len(delta)))].mean(axis=1)
    return {"mean": float(delta.mean()), "ci95_low": float(np.percentile(means, 2.5)), "ci95_high": float(np.percentile(means, 97.5)), "p_two_sided": float(min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0))))}


def evaluate(dev_docs, vectorizers, new_models, e0_bundle, output_dir: Path):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    models = {"e0_char2_4": e0_bundle["models"]["ordinary"], **new_models}
    rows = []
    for position, doc in enumerate(dev_docs, 1):
        candidates = phase2.candidate_indices(len(doc["sentences"]))
        feature_sets = matrices(vectorizers, e0_bundle["vectorizer"], doc["sentences"], candidates)
        probabilities = {name: models[name].predict_proba(feature_sets[name])[:, 1] for name in MODELS}
        budgets = {str(value): value for value in FIXED_BUDGETS}
        budgets["official_reference_length"] = doc["reference_words"]
        row = {"id": doc["id"], "reference_words": doc["reference_words"], "scores": {}, "selected": {}}
        for track, budget in budgets.items():
            row["scores"][track], row["selected"][track] = {}, {}
            for name in MODELS:
                indices = phase2.decode(doc["sentences"], candidates, probabilities[name], budget)
                prediction = phase1.summary(doc["sentences"], indices)
                if len(phase1.word_tokens(prediction)) > budget:
                    raise RuntimeError("complete-sentence decoder violated word budget")
                scores = scorer.score(doc["reference"], prediction)
                row["scores"][track][name] = {metric: scores[metric].fmeasure for metric in ("rouge1", "rouge2", "rougeLsum")}
                row["selected"][track][name] = indices
        rows.append(row)
        if position % 25 == 0 or position == len(dev_docs):
            print(json.dumps({"phase": "dev", "docs": position, "total": len(dev_docs)}), flush=True)
    with (output_dir / "dev_per_case.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    metrics = {}
    for track in ("423", "643", "973", "official_reference_length"):
        metrics[track] = {}
        for name in MODELS:
            metrics[track][name] = {metric: 100 * float(np.mean([r["scores"][track][name][metric] for r in rows])) for metric in ("rouge1", "rouge2", "rougeLsum")}
        metrics[track]["paired_bootstrap"] = {
            f"{right}_minus_{left}": {metric: bootstrap(rows, track, left, right, metric) for metric in ("rouge1", "rouge2", "rougeLsum")}
            for left, right in (("e0_char2_4", "e1_word1_2"), ("e0_char2_4", "e2_word1_2_char3_5"), ("e1_word1_2", "e2_word1_2_char3_5"))
        }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--phase2-oracles", type=Path, required=True)
    parser.add_argument("--phase2-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if phase1.md5(args.archive) != phase1.EXPECTED_ARCHIVE_MD5:
        raise RuntimeError("official archive MD5 mismatch")
    train_docs, dev_docs, excluded = load_protocol(args.dataset_dir, args.train_limit, args.dev_limit)
    oracles = load_oracles(args.phase2_oracles, train_docs)
    texts, positions, y_ordinary, _, audit = phase2.build_shared_training(train_docs, oracles)
    with args.phase2_model.open("rb") as handle:
        e0_bundle = pickle.load(handle)
    started = time.time()
    vectorizers, models = fit_e1_e2(texts, positions, y_ordinary)
    with (args.output_dir / "model_bundle.pkl").open("wb") as handle:
        pickle.dump({"vectorizers": vectorizers, "models": models}, handle)
    metrics = evaluate(dev_docs, vectorizers, models, e0_bundle, args.output_dir)
    write_json(args.output_dir / "dev_metrics.json", metrics)
    write_json(args.output_dir / "config.json", {
        "experiment": "IN-Abs E0/E1/E2 static representation comparison", "seed": phase2.SEED,
        "single_changed_variable": "English TF-IDF representation", "ordinary_oracle_supervision": True,
        "shared_training_rows": audit["shared_training_rows"], "train_documents": len(train_docs), "dev_documents": len(dev_docs),
        "tracks": {"fixed_words": FIXED_BUDGETS, "official_dev": "per-document reference-summary word count"},
        "evaluation_note": "development uses google rouge-score; official test comparison will additionally use frozen py-rouge protocol",
        "candidate_rule": "reference-free evenly-spaced maximum 400 sentences", "complete_sentence_budget": True,
        "official_test_contents_read": False, "excluded_empty_train_pairs": excluded, "elapsed_seconds": time.time() - started,
    })
    print(json.dumps({"phase": "complete", "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
