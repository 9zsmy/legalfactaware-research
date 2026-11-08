#!/usr/bin/env python3
"""Train/decode a restricted TermDiffuSum IN-Abs 200-candidate track.

The recovered model is structurally fixed to 200 article and 40 summary
positions.  This script therefore evaluates a separate sensitivity track and
never claims comparability with the frozen 400-candidate main table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from imp_diff.improved_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from imp_diff.improved_diffusion.summarization_datasets_exp import load_data_summarization


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move_tensors(condition, device):
    for key, value in list(condition.items()):
        if torch.is_tensor(value):
            condition[key] = value.to(device)
    return condition


def read_adapter(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def article_signature(sentences):
    """Identify a document after the upstream loader reorders/truncates text.

    TermDiffuSum's validation loader does not preserve JSONL row order and may
    truncate very long sentence strings.  The candidate count plus prefixes of
    the first two sentences is stable under that processing.  Queues handle the
    one duplicated IN-Abs document without dropping either case.
    """
    first = sentences[0][:128] if sentences else ""
    second = sentences[1][:128] if len(sentences) > 1 else ""
    return len(sentences), first, second


def decode_budget_aware(generated, articles, adapter_rows):
    article_latents = generated[:, :200, :]
    summary_latents = generated[:, 200:240, :]
    logits = torch.softmax(
        torch.bmm(summary_latents, article_latents.transpose(1, 2)), dim=-1
    )
    results = []
    for batch_index, sentences in enumerate(articles):
        row = adapter_rows[batch_index]
        counts = row["sentence_word_counts"]
        budget = row["reference_words"]
        selected = []
        used = 0
        skipped = 0
        for scores in logits[batch_index]:
            for index in torch.argsort(scores, descending=True).tolist():
                if index >= len(sentences) or index >= len(counts) or index in selected:
                    continue
                length = counts[index]
                if not length or used + length > budget:
                    skipped += 1
                    continue
                selected.append(index)
                used += length
                break
        selected.sort()
        prediction = " ".join(sentences[index].strip() for index in selected)
        results.append(
            {
                "id": row["id"],
                "selected_indices": selected,
                "prediction": prediction,
                "prediction_spacy_words": used,
                "reference_spacy_words": budget,
                "budget_violation": used > budget,
                "skipped_due_to_budget": skipped,
            }
        )
    return results


def build_model():
    config = model_and_diffusion_defaults()
    config.update(
        {
            "model_arch": "transformer_sum",
            "diffusion_steps": 500,
            "timestep_respacing": "",
            "noise_schedule": "legal_noise",
            "in_channel": 128,
            "out_channel": 128,
            "num_channels": 128,
            "predict_xstart": True,
            "training_mode": "e2e",
            "roc_train": "legal",
        }
    )
    model, diffusion = create_model_and_diffusion(**config)
    return model, diffusion, config


def save_checkpoint(path, model, optimizer, epoch, history):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
        },
        temporary,
    )
    temporary.replace(path)


def decode_dev(model, diffusion, device, adapted_data_dir, output_dir, batch_size):
    adapter_rows = read_adapter(adapted_data_dir / "val.jsonl")
    rows_by_signature = defaultdict(deque)
    for row in adapter_rows:
        rows_by_signature[article_signature(row["text"])].append(row)
    loader = load_data_summarization(
        batch_size,
        8,
        "legal",
        split=["validation"],
        shuffle=False,
        sent_encoder_type="sbert",
        summary_type="oracle",
        summary_level="sen",
    )
    records = []
    started = time.time()
    model.eval()
    with torch.no_grad():
        for _, condition in loader:
            articles = condition["article_sens"]
            batch_rows = []
            for sentences in articles:
                signature = article_signature(sentences)
                if not rows_by_signature[signature]:
                    raise RuntimeError(
                        "Could not map validation document to its adapter row: "
                        f"signature={signature!r}"
                    )
                batch_rows.append(rows_by_signature[signature].popleft())
            condition = move_tensors(condition, device)
            generated = diffusion.p_sample_loop(
                model,
                (len(articles), 64, 128),
                clip_denoised=False,
                denoised_fn=None,
                model_kwargs=condition,
                top_p=-1,
            )
            records.extend(decode_budget_aware(generated, articles, batch_rows))
    if len(records) != len(adapter_rows):
        raise RuntimeError(
            f"Validation loader returned {len(records)} documents; "
            f"expected {len(adapter_rows)}. Choose a decode batch size that "
            "does not drop the final partial batch."
        )
    predictions_dir = output_dir / "dev_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "dev_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            (predictions_dir / f"{record['id']}.txt").write_text(
                record["prediction"], encoding="utf-8"
            )
    return {
        "predictions": len(records),
        "expected": len(adapter_rows),
        "seconds": round(time.time() - started, 3),
        "budget_violations": sum(record["budget_violation"] for record in records),
        "records_sha256": sha256(output_dir / "dev_records.jsonl"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapted-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--decode-batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    model, diffusion, model_config = build_model()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    checkpoint_path = args.output_dir / "checkpoint_latest.pt"
    history = []
    start_epoch = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"])

    config_record = {
        "experiment": "TermDiffuSum-IN-Abs-200-candidate-sensitivity",
        "official_paper_reproduction": False,
        "comparable_to_400_candidate_main_table": False,
        "official_test_contents_read": False,
        "encoder": "microsoft/mpnet-base sentence-transformer, frozen",
        "decoder_backbone": "bert-base-uncased, local offline weights",
        "supervision": "ordinary-oracle train labels",
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "decode_batch_size": args.decode_batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "model_config": model_config,
    }
    atomic_json(args.output_dir / "config.json", config_record)
    run_started = time.time()
    for epoch in range(start_epoch, args.epochs):
        loader = load_data_summarization(
            args.train_batch_size,
            8,
            "legal",
            split=["train"],
            shuffle=True,
            sent_encoder_type="sbert",
            summary_type="oracle",
            summary_level="sen",
        )
        model.train()
        totals = {}
        steps = 0
        epoch_started = time.time()
        for batch, condition in loader:
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(device)
            condition = move_tensors(condition, device)
            condition["tokenizer"] = None
            condition["output_sen"] = False
            timesteps = torch.randint(
                0, diffusion.num_timesteps, (batch.shape[0],), device=device
            )
            losses = diffusion.training_losses(
                model, batch, timesteps, model_kwargs=condition
            )
            loss = losses["loss"].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().mean())
            steps += 1
        record = {
            "epoch": epoch + 1,
            "steps": steps,
            "seconds": round(time.time() - epoch_started, 3),
            "mean_losses": {key: value / steps for key, value in totals.items()},
            "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        }
        history.append(record)
        atomic_json(args.output_dir / "train_history.json", history)
        save_checkpoint(checkpoint_path, model, optimizer, epoch + 1, history)
        print("EPOCH_COMPLETE", json.dumps(record), flush=True)

    decode_report = decode_dev(
        model,
        diffusion,
        device,
        args.adapted_data_dir,
        args.output_dir,
        args.decode_batch_size,
    )
    manifest = {
        **config_record,
        "status": "complete",
        "elapsed_seconds": round(time.time() - run_started, 3),
        "history": history,
        "decode_report": decode_report,
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print("TERMDIFFUSUM_INABS_200_COMPLETE", json.dumps(decode_report), flush=True)


if __name__ == "__main__":
    main()
