#!/usr/bin/env python3
"""Split-safe Linux-compatible Gist retraining on IN-Abs.

This preserves the released feature definitions and training/inference
behaviour, including global scaling for training, per-document scaling for
inference, and the released character-loop word-embedding feature.
"""

from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import joblib
import lightgbm as lgb
import nltk
import numpy as np
import spacy
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler


def vector_for_items(model: Word2Vec, items) -> np.ndarray:
    total = np.zeros(100, dtype=np.float64)
    count = 0
    vocab = model.wv.key_to_index
    for item in items:
        if item in vocab:
            total += model.wv[item]
            count += 1
    if count:
        total /= count
    return np.around(total, decimals=4)


def raw_features(sentences, cue_phrases, pos_tags, model) -> np.ndarray:
    rows = []
    punctuation_table = str.maketrans("", "", string.punctuation)
    total_sentences = len(sentences)
    for index, sentence in enumerate(sentences):
        stripped = sentence.translate(punctuation_table)
        words = nltk.word_tokenize(stripped)
        quantitative = np.asarray(
            [
                len(stripped),
                len(words),
                len(dict.fromkeys(words)),
                index + 1,
                float(f"{(index + 1) / total_sentences:.5f}"),
            ],
            dtype=np.float64,
        )
        lowered = sentence.lower()
        important = np.asarray(
            [1.0 if phrase.lower() in lowered else 0.0 for phrase in cue_phrases],
            dtype=np.float64,
        )
        # Preserve the released word_emb.py behaviour: it iterates over the
        # original sentence string, so the lookup items are characters.
        embedding = vector_for_items(model, sentence)
        tagged = nltk.pos_tag(nltk.word_tokenize(sentence))[:10]
        pos = np.zeros(len(pos_tags), dtype=np.float64)
        recognized = 0
        for _, tag in tagged:
            if tag in pos_tags:
                pos[pos_tags.index(tag)] += 1
                recognized += 1
        if recognized:
            pos /= recognized
        opening = vector_for_items(model, nltk.word_tokenize(sentence)[:5])
        rows.append(np.concatenate([quantitative, important, embedding, pos, opening]))
    matrix = np.asarray(rows)
    if matrix.ndim != 2 or matrix.shape[1] != 334:
        raise RuntimeError(f"expected 334 features, got {matrix.shape}")
    return matrix


def read_labelled(path: Path):
    sentences, labels = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        sentence, label = line.rsplit("$$$", 1)
        sentences.append(sentence.strip())
        labels.append(int(label.strip()))
    return sentences, labels


def read_plain(path: Path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def token_count(nlp, text: str) -> int:
    removable = string.punctuation + string.whitespace
    return sum(bool(t.text.translate(str.maketrans("", "", removable))) for t in nlp(text))


def load_lengths(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            result[fields[0]] = int(fields[2])
        elif len(fields) >= 2:
            result[fields[0]] = int(fields[1])
    return result


def select_summary(sentences, scores, required_words, nlp):
    score_by_sentence = {s: float(v) for s, v in zip(sentences, scores)}
    position_by_sentence = {s: i for i, s in enumerate(sentences)}
    ranked = sorted(score_by_sentence.items(), key=lambda x: x[1], reverse=True)
    selected, used = [], 0
    for sentence, _ in ranked:
        length = token_count(nlp, sentence)
        if used + length > required_words:
            continue
        used += length
        selected.append(sentence)
    selected_set = set(selected)
    ordered = [
        sentence
        for sentence, _ in sorted(position_by_sentence.items(), key=lambda x: x[1])
        if sentence in selected_set
    ]
    return ordered, used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelled-train", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--length-file", type=Path, required=True)
    parser.add_argument("--cue-phrases", type=Path, required=True)
    parser.add_argument("--pos-tags", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = args.output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    cue_phrases = args.cue_phrases.read_text(encoding="utf-8").splitlines()[:94]
    pos_tags = args.pos_tags.read_text(encoding="utf-8").splitlines()
    if len(cue_phrases) != 94 or len(pos_tags) != 35:
        raise RuntimeError("unexpected official feature dictionaries")

    files = sorted(args.labelled_train.glob("*.txt"))
    corpus, training_rows = [], []
    for path in files:
        sentences, labels = read_labelled(path)
        corpus.extend(nltk.word_tokenize(s.translate(str.maketrans("", "", string.punctuation))) for s in sentences)
        training_rows.append((path.name, sentences, labels))
    print(json.dumps({"stage": "word2vec", "documents": len(files), "sentences": len(corpus)}), flush=True)
    word2vec = Word2Vec(corpus, min_count=5, seed=args.seed, workers=12)
    word2vec.save(str(args.output_dir / "word2vec_model.bin"))

    matrices, labels = [], []
    for index, (name, sentences, doc_labels) in enumerate(training_rows, 1):
        matrices.append(raw_features(sentences, cue_phrases, pos_tags, word2vec))
        labels.extend(doc_labels)
        if index % 25 == 0 or index == len(training_rows):
            print(json.dumps({"stage": "train_features", "done": index, "total": len(training_rows)}), flush=True)
    x = np.vstack(matrices)
    y = np.asarray(labels, dtype=np.int64)
    x = StandardScaler().fit_transform(x)
    print(json.dumps({"stage": "lightgbm", "rows": len(y), "positive_rate": float(y.mean())}), flush=True)
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.001,
        num_leaves=1043,
        n_estimators=1043,
        bagging_fraction=0.8,
        feature_fraction=0.9,
        random_state=args.seed,
        n_jobs=12,
        verbosity=-1,
    )
    model.fit(x, y)
    joblib.dump(model, args.output_dir / "summary_model.pkl")

    lengths = load_lengths(args.length_file)
    nlp = spacy.blank("en")
    records = []
    test_files = sorted(args.test_dir.glob("*.txt"))
    for index, path in enumerate(test_files, 1):
        sentences = read_plain(path)
        features = StandardScaler().fit_transform(raw_features(sentences, cue_phrases, pos_tags, word2vec))
        scores = model.predict_proba(features)[:, 1]
        selected, used = select_summary(sentences, scores, lengths[path.name], nlp)
        (predictions_dir / path.name).write_text("\n".join(selected), encoding="utf-8")
        records.append({"file": path.name, "required_words": lengths[path.name], "used_words": used, "selected_sentences": len(selected)})
        print(json.dumps({"stage": "test_inference", "done": index, "total": len(test_files), "file": path.name}), flush=True)
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    manifest = {
        "experiment": "Gist-INAbs-from-frozen-ordinary-labels",
        "train_documents": len(files),
        "test_documents": len(test_files),
        "features": 334,
        "official_hyperparameters": {"learning_rate": 0.001, "num_leaves": 1043, "n_estimators": 1043},
        "compatibility_changes": [
            "POSIX paths", "current gensim/lightgbm APIs",
            "complete-sentence skip-if-not-fitting budget decoder",
        ],
        "released_character_loop_preserved": True,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
