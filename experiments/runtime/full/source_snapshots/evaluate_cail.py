#!/usr/bin/env python3
"""Dev-select and test-evaluate SENDE under the locked CAIL budget protocol."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import lawa
import numpy as np
import torch

from sende_model import SENDEExtractor


def tokenized(text: str) -> list[str]:
    return [str(x) for x in lawa.cut(text) if str(x).strip()]


def rouge_n(reference: str, prediction: str, order: int) -> float:
    left, right = tokenized(reference), tokenized(prediction)
    if len(left) < order or len(right) < order:
        return 0.0
    a = Counter(tuple(left[i:i + order]) for i in range(len(left) - order + 1))
    b = Counter(tuple(right[i:i + order]) for i in range(len(right) - order + 1))
    overlap = sum((a & b).values())
    precision, recall = overlap / sum(b.values()), overlap / sum(a.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def rouge_l(reference: str, prediction: str) -> float:
    left, right = tokenized(reference), tokenized(prediction)
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for j, other in enumerate(right, 1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    precision, recall = previous[-1] / len(right), previous[-1] / len(left)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def decode(sentences, probabilities, threshold, max_chars=450, max_sentences=8):
    ranked = [int(i) for i in np.argsort(-probabilities) if probabilities[i] >= threshold]
    if not ranked and len(probabilities):
        ranked = [int(np.argmax(probabilities))]
    selected, used = [], 0
    for index in ranked:
        length = len(sentences[index])
        if len(selected) < max_sentences and used + length <= max_chars:
            selected.append(index)
            used += length
    selected.sort()
    return selected, "".join(sentences[index] for index in selected)


def score_rows(records, probabilities, threshold, output_path=None):
    details = []
    for row, scores in zip(records, probabilities):
        selected, prediction = decode(row["sentences"], scores, threshold)
        reference = row["reference_summary"]
        official = {i for i, label in enumerate(row["official_labels"]) if label}
        chosen = set(selected)
        sp = len(chosen & official) / len(chosen) if chosen else 0.0
        sr = len(chosen & official) / len(official) if official else 0.0
        details.append({
            "id": row["id"], "selected_indices": selected,
            "prediction": prediction, "reference": reference,
            "rouge_1": rouge_n(reference, prediction, 1),
            "rouge_2": rouge_n(reference, prediction, 2),
            "rouge_l": rouge_l(reference, prediction),
            "sentence_f1": 2 * sp * sr / (sp + sr) if sp + sr else 0.0,
            "characters": len(prediction), "sentences": len(selected),
        })
    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as handle:
            for row in details:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "cases": len(details), "threshold": threshold,
        "rouge_1": mean(x["rouge_1"] for x in details),
        "rouge_2": mean(x["rouge_2"] for x in details),
        "rouge_l": mean(x["rouge_l"] for x in details),
        "sentence_f1": mean(x["sentence_f1"] for x in details),
        "mean_characters": mean(x["characters"] for x in details),
        "mean_sentences": mean(x["sentences"] for x in details),
        "empty_summaries": sum(not x["prediction"] for x in details),
        "budget_violations": sum(x["characters"] > 450 or x["sentences"] > 8 for x in details),
        "hard_truncations": 0, "source_order": True, "complete_sentences": True,
    }
    report["rouge_mean"] = mean([report["rouge_1"], report["rouge_2"], report["rouge_l"]])
    return report


def load_probabilities(records_path, prefix, checkpoint_path):
    records = [json.loads(line) for line in records_path.open(encoding="utf-8")]
    embeddings = np.load(str(prefix) + ".embeddings.npy", mmap_mode="r")
    offsets = np.load(str(prefix) + ".offsets.npy")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SENDEExtractor(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    probabilities = []
    with torch.no_grad():
        for doc in range(len(records)):
            begin, end = int(offsets[doc]), int(offsets[doc + 1])
            x = torch.from_numpy(np.asarray(embeddings[begin:end], dtype=np.float32)).unsqueeze(0).to(device)
            probabilities.append(torch.sigmoid(model(x))[0].cpu().numpy())
    return records, probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-records", type=Path, required=True)
    parser.add_argument("--dev-prefix", type=Path, required=True)
    parser.add_argument("--test-records", type=Path, required=True)
    parser.add_argument("--test-prefix", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev_records, dev_probabilities = load_probabilities(args.dev_records, args.dev_prefix, args.checkpoint)
    dev_reports = []
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        report = score_rows(dev_records, dev_probabilities, threshold)
        dev_reports.append(report)
        print("dev", json.dumps(report, ensure_ascii=False), flush=True)
    best = max(dev_reports, key=lambda row: (row["rouge_mean"], -row["threshold"]))
    # Test probabilities are loaded only after the dev decision is frozen.
    test_records, test_probabilities = load_probabilities(args.test_records, args.test_prefix, args.checkpoint)
    test_report = score_rows(
        test_records, test_probabilities, best["threshold"], args.output_dir / "test_predictions.jsonl"
    )
    result = {
        "selection": "highest mean dev LAWA ROUGE-1/2/L; lower threshold breaks ties",
        "dev_reports": dev_reports,
        "selected_threshold": best["threshold"],
        "test_report": test_report,
        "test_read_after_dev_selection": True,
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("test", json.dumps(test_report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
