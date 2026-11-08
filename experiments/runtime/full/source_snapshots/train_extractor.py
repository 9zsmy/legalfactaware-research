#!/usr/bin/env python3
"""Train the independent SENDE sentence extractor on cached sentence embeddings."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sende_model import SENDEExtractor


class DocumentDataset(Dataset):
    def __init__(self, prefix: Path, indices: np.ndarray, max_sentences: int):
        self.embeddings = np.load(str(prefix) + ".embeddings.npy", mmap_mode="r")
        self.offsets = np.load(str(prefix) + ".offsets.npy")
        self.labels = np.load(str(prefix) + ".labels.npy", mmap_mode="r")
        self.indices = indices
        self.max_sentences = max_sentences

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        doc = int(self.indices[item])
        begin, end = int(self.offsets[doc]), int(self.offsets[doc + 1])
        end = min(end, begin + self.max_sentences)
        return (
            torch.from_numpy(np.asarray(self.embeddings[begin:end], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.labels[begin:end], dtype=np.float32)),
        )


def collate(batch):
    max_len = max(x.shape[0] for x, _ in batch)
    dim = batch[0][0].shape[1]
    embeddings = torch.zeros(len(batch), max_len, dim)
    labels = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (x, y) in enumerate(batch):
        embeddings[i, : len(x)] = x
        labels[i, : len(y)] = y
        mask[i, : len(x)] = True
    return embeddings, labels, mask


def masked_loss(logits, labels, mask, pos_weight):
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)
    return losses[mask].mean()


@torch.no_grad()
def validate(model, loader, device, pos_weight):
    model.eval()
    total, weight = 0.0, 0
    for embeddings, labels, mask in loader:
        embeddings, labels, mask = embeddings.to(device), labels.to(device), mask.to(device)
        loss = masked_loss(model(embeddings), labels, mask, pos_weight)
        n = int(mask.sum())
        total += float(loss) * n
        weight += n
    return total / max(weight, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-sentences", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    offsets = np.load(str(args.train_prefix) + ".offsets.npy")
    labels = np.load(str(args.train_prefix) + ".labels.npy", mmap_mode="r")
    doc_count = len(offsets) - 1
    order = np.random.default_rng(args.seed).permutation(doc_count)
    dev_count = max(1, round(doc_count * 0.05))
    dev_indices, train_indices = order[:dev_count], order[dev_count:]
    train_dataset = DocumentDataset(args.train_prefix, train_indices, args.max_sentences)
    dev_dataset = DocumentDataset(args.train_prefix, dev_indices, args.max_sentences)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=4, generator=generator, pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=4, pin_memory=True,
    )
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    pos_weight_value = negatives / max(positives, 1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SENDEExtractor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    pos_weight = torch.tensor(pos_weight_value, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, best, stale = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for embeddings, targets, mask in train_loader:
            embeddings, targets, mask = embeddings.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = masked_loss(model(embeddings), targets, mask, pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            n = int(mask.sum())
            running += float(loss) * n
            seen += n
        dev_loss = validate(model, dev_loader, device, pos_weight)
        entry = {"epoch": epoch, "train_loss": running / max(seen, 1), "dev_loss": dev_loss}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if dev_loss < best - 1e-5:
            best, stale = dev_loss, 0
            torch.save({"model": model.state_dict(), "config": model.config}, args.output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break

    manifest = {
        "train_prefix": str(args.train_prefix), "seed": args.seed,
        "train_documents": len(train_indices), "dev_documents": len(dev_indices),
        "positive_weight": pos_weight_value, "batch_size": args.batch_size,
        "learning_rate": args.lr, "max_sentences": args.max_sentences,
        "model": model.config, "best_dev_loss": best, "history": history,
        "paper_specified": ["hidden_size", "dilations", "batch_size", "dropout", "six_layers"],
        "independent_assumptions": ["kernel_size", "loss", "class_weight", "optimizer", "lr", "epochs", "early_stopping", "max_sentences", "seed"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

