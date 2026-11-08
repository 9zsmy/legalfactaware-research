#!/usr/bin/env python3
"""Restricted Agarwal-ST transfer to the frozen IN-Abs development protocol."""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path
from statistics import mean

import numpy as np
import spacy
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class CachedDataset(Dataset):
    def __init__(self, corpus: Path, cache: Path) -> None:
        self.rows = read_rows(corpus)
        self.manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        if [str(row["id"]) for row in self.rows] != self.manifest["ids"]:
            raise RuntimeError("cache/corpus ID order mismatch")
        self.offsets = self.manifest["offsets"]
        self.embeddings = np.load(cache / "embeddings.npy", mmap_mode="r")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        start, end = self.offsets[index : index + 2]
        return np.asarray(self.embeddings[start:end], dtype=np.float32), self.rows[index]["labels"], index


def pad_documents(embeddings, labels):
    width = embeddings[0].shape[1]
    length = max(item.shape[0] for item in embeddings)
    x = np.zeros((len(embeddings), length, width), np.float32)
    y = np.zeros((len(embeddings), length), np.int64)
    mask = np.zeros((len(embeddings), length), bool)
    for row, (document, targets) in enumerate(zip(embeddings, labels)):
        if len(document) != len(targets):
            raise RuntimeError("embedding/label length mismatch")
        x[row, :len(document)] = document
        y[row, :len(document)] = targets
        mask[row, :len(document)] = True
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def collate(batch):
    x, y, indices = zip(*batch)
    px, py, mask = pad_documents(x, y)
    return px, py, mask, torch.tensor(indices)


class AgarwalSTScorer(nn.Module):
    def __init__(self, embedding_dim=768, hidden_size=128, dropout=0.5):
        super().__init__()
        self.bigru = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, 2)

    def forward(self, embeddings, mask):
        lengths = mask.sum(1).long().cpu()
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.bigru(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=embeddings.shape[1])
        return self.classifier(self.dropout(output))


def weighted_loss(logits, labels, mask, weights):
    valid = mask.bool()
    return nn.functional.cross_entropy(logits[valid], labels[valid].long(), weight=weights)


def official_count(nlp, text: str) -> int:
    table = str.maketrans("", "", string.punctuation + string.whitespace)
    return sum(len(token.text.translate(table)) > 0 for token in nlp(text))


def budget_mmr(scores, embeddings, sentences, lengths, budget, value=0.8):
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    selected, remaining, used = [], set(range(len(sentences))), 0
    while remaining:
        fitting = [index for index in remaining if lengths[index] and used + lengths[index] <= budget]
        if not fitting:
            break
        best = max(
            fitting,
            key=lambda index: (
                value * float(scores[index]) - (1.0 - value) * (
                    max(float(normalized[index] @ normalized[other]) for other in selected) if selected else 0.0
                ),
                float(scores[index]),
                -index,
            ),
        )
        selected.append(best)
        remaining.remove(best)
        used += lengths[best]
    return sorted(selected), used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.00261)
    parser.add_argument("--lambda-mmr", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    train = CachedDataset(args.train, args.train_cache)
    dev = CachedDataset(args.dev, args.dev_cache)
    loader = DataLoader(train, args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    model = AgarwalSTScorer(768, args.hidden_size, args.dropout).to(device)
    positives = sum(sum(row["labels"]) for row in train.rows)
    total = sum(len(row["labels"]) for row in train.rows)
    weights = torch.tensor([total / (2 * (total - positives)), total / (2 * positives)], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y, mask, _ in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = weighted_loss(model(x, mask), y, mask, weights)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        record = {"epoch": epoch, "train_loss": mean(losses)}
        history.append(record); print(json.dumps(record), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": vars(args), "history": history}, args.output_dir / "model_epoch5.pt")
    model.eval(); probabilities = [None] * len(dev)
    dev_loader = DataLoader(dev, args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    with torch.no_grad():
        for x, _, mask, indices in dev_loader:
            values = model(x.to(device), mask.to(device)).softmax(-1)[:, :, 1].cpu().numpy()
            lengths = mask.sum(1).tolist()
            for row, index in enumerate(indices.tolist()):
                probabilities[index] = values[row, :lengths[row]]

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    predictions_dir = args.output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    records = []
    for index, row in enumerate(dev.rows):
        start, end = dev.offsets[index:index + 2]
        embeddings = np.asarray(dev.embeddings[start:end], dtype=np.float32)
        budget = official_count(nlp, row["summary"])
        sentence_lengths = [official_count(nlp, sentence) for sentence in row["sentences"]]
        selected, used = budget_mmr(
            probabilities[index], embeddings, row["sentences"], sentence_lengths, budget, args.lambda_mmr
        )
        summary = "\n".join(row["sentences"][item] for item in selected)
        verified = official_count(nlp, summary)
        if verified != used or verified > budget:
            raise RuntimeError(f"budget violation for {row['id']}: {verified}>{budget}")
        (predictions_dir / f"{row['id']}.txt").write_text(summary, encoding="utf-8")
        records.append({"id": str(row["id"]), "selected_indices": selected, "budget": budget, "words": verified})
        if (index + 1) % 25 == 0 or index + 1 == len(dev):
            print(json.dumps({"stage": "decode", "done": index + 1, "total": len(dev)}), flush=True)

    with (args.output_dir / "decode_records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    manifest = {
        "method": "Agarwal-ST restricted IN-Abs transfer", "train_documents": len(train),
        "dev_documents": len(dev), "supervision": "P50 budget-oracle labels",
        "encoder": "nlpaueb/legal-bert-base-uncased frozen masked-mean",
        "epochs": args.epochs, "batch_size": args.batch_size, "hidden_size": args.hidden_size,
        "dropout": args.dropout, "learning_rate": args.learning_rate, "lambda_mmr": args.lambda_mmr,
        "history": history, "positive_rate": positives / total, "class_weights": weights.cpu().tolist(),
        "budget_violations": 0, "official_test_contents_read": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "budget_violations": 0}), flush=True)


if __name__ == "__main__":
    main()

