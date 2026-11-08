#!/usr/bin/env python3
"""Decode SENDE's frozen scores under the common IN-Abs complete-sentence budget."""

from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

import numpy as np
import spacy
import torch


def count(nlp, text: str) -> int:
    table = str.maketrans("", "", string.punctuation + string.whitespace)
    return sum(len(token.text.translate(table)) > 0 for token in nlp(text))


def decode(scores, lengths, budget, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    ranked = [int(i) for i in np.argsort(-scores) if scores[i] >= threshold]
    if not ranked and len(scores):
        ranked = [int(np.argmax(scores))]
    chosen, used = [], 0
    for index in ranked:
        length = lengths[index]
        if length and used + length <= budget:
            chosen.append(index)
            used += length
    if not chosen:
        fitting = [index for index, length in enumerate(lengths) if length and length <= budget]
        if fitting:
            best = max(fitting, key=lambda index: (float(scores[index]), -index))
            chosen = [best]
            used = lengths[best]
    return sorted(chosen), used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--features-prefix", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sende-src", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    sys.path.insert(0, str(args.sende_src))
    from sende_model import SENDEExtractor

    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    embeddings = np.load(str(args.features_prefix) + ".embeddings.npy", mmap_mode="r")
    offsets = np.load(str(args.features_prefix) + ".offsets.npy")
    if len(offsets) != len(records) + 1:
        raise RuntimeError("feature/record count mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SENDEExtractor(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty output: {args.output_dir}")
    predictions = args.output_dir / "predictions"
    predictions.mkdir(parents=True)
    details = []
    for doc, row in enumerate(records):
        start, end = int(offsets[doc]), int(offsets[doc + 1])
        if end - start != len(row["sentences"]):
            raise RuntimeError(f"sentence count mismatch for {row['id']}")
        x = torch.from_numpy(np.asarray(embeddings[start:end], dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = torch.sigmoid(model(x))[0].cpu().numpy()
        budget = count(nlp, row["summary"])
        lengths = [count(nlp, sentence) for sentence in row["sentences"]]
        selected, used = decode(scores, lengths, budget, args.threshold)
        prediction = "\n".join(row["sentences"][index] for index in selected)
        verified = count(nlp, prediction)
        if verified != used or verified > budget:
            raise RuntimeError(f"budget violation for {row['id']}: {verified}>{budget}")
        (predictions / f"{row['id']}.txt").write_text(prediction, encoding="utf-8")
        details.append({"id": str(row["id"]), "selected_indices": selected, "words": verified, "budget": budget})
        if (doc + 1) % 25 == 0 or doc + 1 == len(records):
            print(json.dumps({"stage": "decode", "done": doc + 1, "total": len(records)}), flush=True)
    with (args.output_dir / "decode_records.jsonl").open("w", encoding="utf-8") as stream:
        for row in details:
            stream.write(json.dumps(row) + "\n")
    manifest = {
        "method": "SENDE independent reconstruction common-label transfer",
        "dev_documents": len(records), "threshold": args.threshold,
        "budget_rule": "per-reference spaCy word count", "budget_violations": 0,
        "official_test_contents_read": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
