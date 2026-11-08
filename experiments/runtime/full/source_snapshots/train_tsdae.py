#!/usr/bin/env python3
"""Fine-tune a legal sentence encoder with TSDAE on BillSum train sentences."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, datasets, losses


def read_sentences(path: Path, max_sentences: int | None) -> list[str]:
    sentences: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            sentences.extend(json.loads(line)["sentences"])
            if max_sentences is not None and len(sentences) >= max_sentences:
                return sentences[:max_sentences]
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sentences = read_sentences(args.train_jsonl, args.max_sentences)
    if not sentences:
        raise ValueError("No training sentences found")

    model = SentenceTransformer(args.base_model)
    # The paper reports batch size 64 on a 24 GB RTX 3090 but omits its token
    # limit.  The raw encoder default (512) does not fit; 128 is therefore an
    # explicit reconstruction assumption rather than an attributed paper value.
    model.max_seq_length = args.max_seq_length
    # SentenceTransformers otherwise performs a Hugging Face API lookup while
    # generating a model card at save time.  The experimental server cannot
    # reliably reach that API; model-card networking is unrelated to training.
    model._create_model_card = lambda *unused_args, **unused_kwargs: None
    dataset = datasets.DenoisingAutoEncoderDataset(sentences)
    generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        generator=generator,
    )
    loss = losses.DenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=args.base_model,
        tie_encoder_decoder=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=args.epochs,
        warmup_steps=min(args.warmup_steps, max(1, len(dataloader) // 10)),
        optimizer_params={"lr": args.lr},
        weight_decay=0.01,
        use_amp=torch.cuda.is_available(),
        output_path=str(args.output_dir),
        # The runtime benchmark needs the final encoder, not recovery
        # checkpoints. Disabling periodic copies avoids duplicating a 409 MB
        # model on the capacity-constrained experimental volume.
        checkpoint_path=None,
        show_progress_bar=True,
    )
    manifest = {
        "base_model": args.base_model,
        "sentences": len(sentences),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "requested_warmup_steps": args.warmup_steps,
        "effective_warmup_steps": min(args.warmup_steps, max(1, len(dataloader) // 10)),
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "paper_specified": ["TSDAE", "batch_size"],
        "independent_assumptions": ["epochs", "learning_rate", "warmup", "optimizer", "max_seq_length", "seed"],
    }
    (args.output_dir / "reconstruction_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
