#!/usr/bin/env python3
"""Encode the segmented sentences and save contiguous, memory-mappable arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-docs", type=int)
    args = parser.parse_args()

    records = []
    with args.records.open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
            if args.max_docs is not None and len(records) >= args.max_docs:
                break
    sentences = [sentence for record in records for sentence in record["sentences"]]
    offsets = [0]
    labels = []
    for record in records:
        offsets.append(offsets[-1] + len(record["sentences"]))
        labels.extend(record["labels"])

    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        sentences,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float16)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.output_prefix) + ".embeddings.npy", embeddings)
    np.save(str(args.output_prefix) + ".offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(str(args.output_prefix) + ".labels.npy", np.asarray(labels, dtype=np.uint8))
    Path(str(args.output_prefix) + ".manifest.json").write_text(
        json.dumps(
            {
                "records": str(args.records),
                "model": args.model,
                "documents": len(records),
                "sentences": len(sentences),
                "embedding_dimension": int(embeddings.shape[1]),
                "storage_dtype": "float16",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

