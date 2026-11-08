#!/usr/bin/env python3
"""Full independent TermDiffuSum adaptation on frozen CAIL2020-clean.

Requires the recovered Python 3.7 TermDiffuSum bytecode and the hard-coded
dataset/encoder paths prepared by the companion adapter. This is not an
official LegalAFSum reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from imp_diff.improved_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from imp_diff.improved_diffusion.summarization_datasets_exp import (
    load_data_summarization,
)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move_tensors(condition: dict, device: torch.device) -> dict:
    for key, value in list(condition.items()):
        if torch.is_tensor(value):
            condition[key] = value.to(device)
    return condition


def decode_budget_aware(
    generated: torch.Tensor,
    articles: list[list[str]],
    max_sentences: int,
    max_chars: int,
) -> list[dict]:
    article_latents = generated[:, :200, :]
    summary_latents = generated[:, 200:240, :]
    logits = torch.softmax(
        torch.bmm(summary_latents, article_latents.transpose(1, 2)), dim=-1
    )
    results = []
    for batch_index, sentences in enumerate(articles):
        selected: list[int] = []
        skipped = 0
        for row in logits[batch_index]:
            for index in torch.argsort(row, descending=True).tolist():
                if index >= len(sentences) or index in selected:
                    continue
                ordered = sorted(selected + [index])
                candidate = "".join(sentences[i] for i in ordered)
                if len(candidate) > max_chars:
                    skipped += 1
                    continue
                selected.append(index)
                break
            if len(selected) >= max_sentences:
                break
        selected.sort()
        summary = "".join(sentences[index] for index in selected)
        assert len(summary) <= max_chars
        results.append(
            {
                "selected_indices": selected,
                "prediction": summary,
                "chars": len(summary),
                "sentence_count": len(selected),
                "skipped_due_to_budget": skipped,
            }
        )
    return results


def read_ids(path: Path) -> list[str | None]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.append(json.loads(line).get("id"))
    return values


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


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict],
) -> None:
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


def decode_split(
    split: str,
    model: torch.nn.Module,
    diffusion,
    device: torch.device,
    adapted_data_dir: Path,
    output_dir: Path,
    batch_size: int,
    max_sentences: int,
    max_chars: int,
) -> dict:
    loader_split = "validation" if split == "dev" else "test"
    adapted_name = "val.jsonl" if split == "dev" else "test.jsonl"
    identifiers = read_ids(adapted_data_dir / adapted_name)
    loader = load_data_summarization(
        batch_size,
        8,
        "legal",
        split=[loader_split],
        shuffle=False,
        sent_encoder_type="sbert",
        summary_type="oracle",
        summary_level="sen",
    )
    destination = output_dir / f"{split}_predictions.jsonl"
    started = time.time()
    written = 0
    max_seen_chars = 0
    max_seen_sentences = 0
    with destination.open("w", encoding="utf-8") as handle, torch.no_grad():
        for _, condition in loader:
            articles = condition["article_sens"]
            references = condition["summary_sens"]
            condition = move_tensors(condition, device)
            generated = diffusion.p_sample_loop(
                model,
                (len(articles), 64, 128),
                clip_denoised=False,
                denoised_fn=None,
                model_kwargs=condition,
                top_p=-1,
            )
            predictions = decode_budget_aware(
                generated, articles, max_sentences=max_sentences, max_chars=max_chars
            )
            for index, prediction in enumerate(predictions):
                record_index = written + index
                record = {
                    "id": identifiers[record_index],
                    **prediction,
                    "official_extract_reference": references[index],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                max_seen_chars = max(max_seen_chars, prediction["chars"])
                max_seen_sentences = max(
                    max_seen_sentences, prediction["sentence_count"]
                )
            written += len(predictions)
    return {
        "split": split,
        "predictions": written,
        "expected": len(identifiers),
        "seconds": round(time.time() - started, 3),
        "max_chars": max_seen_chars,
        "max_sentences": max_seen_sentences,
        "hard_truncation_count": 0,
        "sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapted-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--decode-batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-sentences", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=450)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--bert-config-dir", type=Path, required=True)
    parser.add_argument(
        "--decode-splits",
        nargs="+",
        choices=("dev", "test"),
        default=("dev",),
        help="Splits decoded after training. The runtime benchmark uses dev only.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_from_pretrained = AutoConfig.from_pretrained

    def local_from_pretrained(name_or_path, *values, **options):
        if str(name_or_path) == "bert-base-uncased":
            name_or_path = str(args.bert_config_dir)
        return original_from_pretrained(name_or_path, *values, **options)

    AutoConfig.from_pretrained = staticmethod(local_from_pretrained)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    model, diffusion, model_config = build_model()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    checkpoint_path = args.output_dir / "checkpoint_latest.pt"
    history: list[dict] = []
    start_epoch = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"])

    config_record = {
        "experiment": "TermDiffuSum-CAIL-independent-full",
        "official_paper_reproduction": False,
        "data": "CAIL2020-clean frozen split",
        "supervision": "CAIL official sentence labels",
        "encoder": "hfl/chinese-roberta-wwm-ext + mean pooling, frozen",
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "decode_batch_size": args.decode_batch_size,
        "learning_rate": args.learning_rate,
        "max_sentences": args.max_sentences,
        "max_chars": args.max_chars,
        "seed": args.seed,
        "model_config": model_config,
    }
    atomic_json(args.output_dir / "config.json", config_record)

    training_started = time.time()
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
        totals: dict[str, float] = {}
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
        epoch_record = {
            "epoch": epoch + 1,
            "steps": steps,
            "seconds": round(time.time() - epoch_started, 3),
            "mean_losses": {key: value / steps for key, value in totals.items()},
            "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        }
        history.append(epoch_record)
        atomic_json(args.output_dir / "train_history.json", history)
        save_checkpoint(checkpoint_path, model, optimizer, epoch + 1, history)
        print("EPOCH_COMPLETE", json.dumps(epoch_record, ensure_ascii=False), flush=True)

    training_elapsed = time.time() - training_started
    model.eval()
    decode_reports = []
    for split in args.decode_splits:
        report = decode_split(
            split,
            model,
            diffusion,
            device,
            args.adapted_data_dir,
            args.output_dir,
            args.decode_batch_size,
            args.max_sentences,
            args.max_chars,
        )
        decode_reports.append(report)
        print("DECODE_COMPLETE", json.dumps(report, ensure_ascii=False), flush=True)

    manifest = {
        **config_record,
        "status": "complete",
        "train_seconds_this_run": round(training_elapsed, 3),
        "total_seconds_this_run": round(time.time() - training_started, 3),
        "history": history,
        "decode_reports": decode_reports,
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print("TERMDIFFUSUM_CAIL_FULL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
