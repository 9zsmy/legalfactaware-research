#!/usr/bin/env python3
"""Decode SSE five-part scores under the frozen IN-Abs whole-sentence budget."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import string
import sys

import spacy
import torch
from torch.utils.data import DataLoader


QUOTAS = (1, 2, 1, 2, 1)
TRANSLATOR = str.maketrans("", "", string.punctuation + string.whitespace)


def count_words(nlp, text: str) -> int:
    return sum(bool(token.text.translate(TRANSLATOR)) for token in nlp.make_doc(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--sse-scripts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.sse_scripts))
    from train_sse_memsum import SSEDataset, build_model, read_jsonl

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    prepared_dir = Path(config["prepared_dir"])
    with (prepared_dir / "vocabulary.pkl").open("rb") as stream:
        vocabulary = pickle.load(stream)
    with (prepared_dir / f"random_embeddings_{config['embed_dim']}d.pkl").open("rb") as stream:
        embeddings = pickle.load(stream)
    model = build_model(
        embeddings, hidden_size=config["hidden_size"], num_heads=config["num_heads"],
        dropout=config["dropout"], token_softmax_dim=config["token_softmax_dim"],
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    rows = read_jsonl(args.corpus)
    if len(rows) != 703:
        raise RuntimeError(f"expected 703 dev rows, found {len(rows)}")
    data = SSEDataset(rows, {word: i for i, word in enumerate(vocabulary)},
                      config["max_doc_len"], config["max_seq_len"])
    loader = DataLoader(data, batch_size=4, shuffle=False, collate_fn=data.collate)
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    predictions = args.output_dir / "predictions"
    predictions.mkdir(parents=True, exist_ok=False)
    records = []
    offset = 0
    with torch.no_grad():
        for token_ids, _ in loader:
            result = model(token_ids.to(device))
            scores = result["logits"].sigmoid().cpu()
            lengths = result["valid_mask"].sum(dim=1).cpu().tolist()
            for batch_index, valid_length in enumerate(lengths):
                row = rows[offset + batch_index]
                sentences = row["raw_text"][:valid_length]
                budget = count_words(nlp, row["raw_summary"])
                selected = set()
                by_part = []
                for part_index, quota in enumerate(QUOTAS):
                    ranked = sorted(range(valid_length),
                                    key=lambda i: (-float(scores[batch_index, part_index, i]), i))
                    chosen = []
                    for index in ranked:
                        if len(chosen) >= quota:
                            break
                        if index in selected or not sentences[index].strip():
                            continue
                        trial = sorted(selected | {index})
                        trial_summary = " ".join(sentences[i].strip() for i in trial)
                        if count_words(nlp, trial_summary) <= budget:
                            selected.add(index)
                            chosen.append(index)
                    by_part.append(chosen)
                final_indices = sorted(selected)
                prediction = " ".join(sentences[i].strip() for i in final_indices)
                used = count_words(nlp, prediction)
                if used > budget:
                    raise RuntimeError(f"budget violation for {row['id']}")
                (predictions / f"{row['id']}.txt").write_text(prediction, encoding="utf-8")
                records.append({
                    "id": row["id"], "selected_indices": final_indices,
                    "selected_source_indices": [row["source_indices"][i] for i in final_indices],
                    "selected_by_part": by_part, "prediction_words": used,
                    "reference_words": budget, "unused_words": budget - used,
                })
            offset += len(lengths)
            if offset % 50 < 4 or offset == len(rows):
                print(json.dumps({"stage": "decode", "done": offset, "total": len(rows)}), flush=True)
    with (args.output_dir / "decode_records.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    manifest = {
        "method": "SSE structural-proxy English adaptation", "dev_documents": len(records),
        "budget_violations": 0, "mean_selected_sentences": sum(len(r["selected_indices"]) for r in records) / len(records),
        "mean_unused_budget_words": sum(r["unused_words"] for r in records) / len(records),
        "official_test_contents_read": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"stage": "decode_complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
