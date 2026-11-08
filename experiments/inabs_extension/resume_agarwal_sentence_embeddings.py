#!/usr/bin/env python3
"""Resume a partially written frozen sentence-embedding memmap safely."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def read_documents(path: Path) -> list[dict]:
    documents = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            sentences = [str(item).strip() for item in row["sentences"] if str(item).strip()]
            labels = [int(value) for value in row.get("labels", [0] * len(sentences))]
            if len(sentences) != len(labels):
                raise RuntimeError(f"sentence/label mismatch for {row['id']}")
            documents.append({"id": str(row["id"]), "sentences": sentences, "labels": labels})
    return documents


def first_unwritten(embeddings: np.ndarray, chunk_size: int = 8192) -> int:
    total = len(embeddings)
    found = total
    for start in range(0, total, chunk_size):
        block = np.asarray(embeddings[start : start + chunk_size])
        zero = np.all(block == 0, axis=1)
        if zero.any():
            found = start + int(np.flatnonzero(zero)[0])
            break
    if found < total:
        for start in range(found, total, chunk_size):
            block = np.asarray(embeddings[start : start + chunk_size])
            if np.any(block != 0):
                raise RuntimeError("nonzero row found after first unwritten row; cache has a gap")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    documents = read_documents(args.input)
    offsets = [0]
    for document in documents:
        offsets.append(offsets[-1] + len(document["sentences"]))
    total = offsets[-1]
    flat = [sentence for document in documents for sentence in document["sentences"]]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    hidden = int(model.config.hidden_size)
    path = args.output_dir / "embeddings.npy"
    if path.exists():
        embeddings = np.load(path, mmap_mode="r+")
        if embeddings.shape != (total, hidden) or embeddings.dtype != np.float16:
            raise RuntimeError(f"unexpected existing cache: {embeddings.shape} {embeddings.dtype}")
        start_at = first_unwritten(embeddings)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        embeddings = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(total, hidden))
        start_at = 0
    print(json.dumps({"stage": "resume", "start": start_at, "total": total}), flush=True)

    use_amp = args.device.startswith("cuda")
    for start in range(start_at, total, args.batch_size):
        batch = flat[start : start + args.batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if use_amp else "cpu", dtype=torch.float16, enabled=use_amp
        ):
            states = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(states.dtype)
            pooled = (states * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        embeddings[start : start + len(batch)] = pooled.float().cpu().numpy().astype(np.float16)
        if start == start_at or (start // args.batch_size) % 100 == 0:
            embeddings.flush()
            print(json.dumps({"encoded": start + len(batch), "total": total}), flush=True)
    embeddings.flush()

    manifest = {
        "input": str(args.input), "model": str(args.model), "tokenizer": str(args.tokenizer),
        "pooling": "masked_mean_last_hidden_state", "encoder_frozen": True,
        "max_length": args.max_length, "dtype": "float16", "hidden_size": hidden,
        "documents": len(documents), "sentences": total,
        "ids": [document["id"] for document in documents], "offsets": offsets,
        "positive_labels": int(sum(sum(document["labels"]) for document in documents)),
        "resumed_from_sentence": start_at,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "documents": len(documents), "sentences": total}), flush=True)


if __name__ == "__main__":
    main()

