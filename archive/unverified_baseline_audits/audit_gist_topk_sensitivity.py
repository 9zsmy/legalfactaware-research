#!/usr/bin/env python3
"""Diagnostic-only Gist Top-K sensitivity audit on saved official scores."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import string
from pathlib import Path

import numpy as np
import spacy
from rouge_score import rouge_scorer


def evaluate(item):
    k, predictions, references = item
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    values = []
    for prediction, reference in zip(predictions, references):
        score = scorer.score(reference, prediction)
        values.append([score[name].fmeasure for name in
                       ("rouge1", "rouge2", "rougeL")])
    means = 100.0 * np.asarray(values).mean(axis=0)
    return k, dict(zip(("rouge1", "rouge2", "rougeL"), means.tolist()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 15, 20, 25, 30])
    args = parser.parse_args()

    budgets = {}
    for line in args.stats_file.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            budgets[fields[0]] = int(fields[2])
    names = sorted(path.name for path in args.scores_dir.glob("*.txt"))
    references = [(args.references / name).read_text(encoding="utf-8") for name in names]
    nlp = spacy.blank("en")
    table = str.maketrans("", "", string.punctuation + string.whitespace)

    all_predictions = {k: [] for k in args.top_k}
    diagnostics = {k: {"selected_sentences": [], "used_words": []} for k in args.top_k}
    for name in names:
        score_by_sentence = {}
        position_by_sentence = {}
        for position, line in enumerate((args.scores_dir / name).read_text(
                encoding="utf-8").splitlines()):
            sentence, score = line.rsplit("$$$", 1)
            score_by_sentence[sentence] = float(score)
            position_by_sentence[sentence] = position
        ranked = sorted(score_by_sentence, key=score_by_sentence.get, reverse=True)
        lengths = {
            sentence: sum(bool(token.text.translate(table)) for token in nlp(sentence))
            for sentence in ranked
        }
        for k in args.top_k:
            selected, used = [], 0
            for sentence in ranked[:k]:
                if used + lengths[sentence] > budgets[name]:
                    break
                selected.append(sentence)
                used += lengths[sentence]
            selected_set = set(selected)
            source_order = [sentence for sentence, _ in sorted(
                position_by_sentence.items(), key=lambda item: item[1]
            ) if sentence in selected_set]
            all_predictions[k].append("\n".join(source_order))
            diagnostics[k]["selected_sentences"].append(len(source_order))
            diagnostics[k]["used_words"].append(used)

    tasks = [(k, all_predictions[k], references) for k in args.top_k]
    with mp.get_context("spawn").Pool(min(len(tasks), 5)) as pool:
        metrics = dict(pool.map(evaluate, tasks))
    report = {
        "status": "diagnostic only; Top-K is not stated for Gist in the AACL paper",
        "paper": {"rouge1": 42.7, "rouge2": 13.2, "rougeL": 21.5},
        "google_rouge_score_full_sequence": {
            str(k): {
                **metrics[k],
                "mean_selected_sentences": float(np.mean(diagnostics[k]["selected_sentences"])),
                "mean_used_words": float(np.mean(diagnostics[k]["used_words"])),
            }
            for k in args.top_k
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
