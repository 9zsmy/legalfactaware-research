#!/usr/bin/env python3
"""Source-faithful TermDiffuSum adaptation for the IN-Abs dev split.

The authors did not release LegalAFSum, its term annotations, or a complete
reproduction package.  This run therefore preserves the released training and
decoding mechanics while making the minimum documented language/data changes:
an English legal-term lexicon and the existing 200-sentence IN-Abs adapter.
It emits both the authors' decoder (no word cap) and a separate word-budget
adaptation required by the unified IN-Abs comparison.
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

from imp_diff.improved_diffusion import legal_noise
from imp_diff.improved_diffusion.summarization_datasets_exp import (
    load_data_summarization,
)
from run_termdiffusum_inabs_200 import (
    article_signature,
    build_model,
    move_tensors,
    read_adapter,
)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_select(scores: torch.Tensor, sentence_count: int) -> list[int]:
    """Released legal decoder: top-5 per slot, unique choices, at most 20."""
    top = torch.topk(scores, k=5, dim=-1).indices.tolist()
    selected = []
    for candidates in top:
        for index in candidates:
            if index not in selected:
                selected.append(int(index))
                break
        if len(selected) == 20:
            break
    return sorted(index for index in selected if index < sentence_count)


def budget_select(
    scores: torch.Tensor,
    sentence_count: int,
    counts: list[int],
    budget: int,
) -> tuple[list[int], int]:
    """Minimum complete-sentence budget adaptation of the released decoder."""
    top = torch.topk(scores, k=5, dim=-1).indices.tolist()
    selected = []
    used = 0
    for candidates in top:
        for index in candidates:
            index = int(index)
            if index >= sentence_count or index >= len(counts) or index in selected:
                continue
            length = int(counts[index])
            if length <= 0 or used + length > budget:
                continue
            selected.append(index)
            used += length
            break
        if len(selected) == 20:
            break
    return sorted(selected), used


def save_recovery(path: Path, model, epoch: int, global_step: int) -> None:
    state = {
        key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"epoch": epoch, "global_step": global_step, "model": state}, temporary)
    temporary.replace(path)


def train(args, model, diffusion, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    global_step = 0
    recovery = args.output_dir / "checkpoint_recovery_fp16.pt"
    started_epoch = 0
    if recovery.exists() and args.resume_recovery:
        checkpoint = torch.load(recovery, map_location=device)
        model.load_state_dict(checkpoint["model"])
        started_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])

    for epoch in range(started_epoch, args.epochs):
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
            condition.pop("cand_combs", None)
            condition["cand_sens"] = torch.tensor(condition["cand_sens"])
            condition = move_tensors(condition, device)
            condition["tokenizer"] = None
            condition["output_sen"] = global_step % 200 == 0
            timesteps = torch.randint(
                0, diffusion.num_timesteps, (batch.shape[0],), device=device
            )
            losses = diffusion.training_losses(
                model, batch, timesteps, model_kwargs=condition
            )
            loss = losses["loss"].mean()
            loss.backward()
            # The released TrainLoop clips every step to 3, independently of
            # its nominal gradient_clipping argument.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            fraction = global_step / args.lr_anneal_steps
            current_lr = args.learning_rate * (1.0 - fraction)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().mean())
            global_step += 1
            steps += 1
        record = {
            "epoch": epoch + 1,
            "steps": steps,
            "global_step": global_step,
            "seconds": round(time.time() - epoch_started, 3),
            "learning_rate_end": optimizer.param_groups[0]["lr"],
            "mean_losses": {key: value / steps for key, value in totals.items()},
        }
        history.append(record)
        atomic_json(args.output_dir / "train_history.json", history)
        save_recovery(recovery, model, epoch + 1, global_step)
        print("EPOCH_COMPLETE", json.dumps(record), flush=True)
    return history


def write_records(root: Path, records: list[dict]) -> None:
    prediction_dir = root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    with (root / "dev_records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            (prediction_dir / f"{record['id']}.txt").write_text(
                record["prediction"], encoding="utf-8"
            )


def decode(args, model, diffusion, device):
    adapter_rows = read_adapter(args.adapted_data_dir / "val.jsonl")
    rows_by_signature = defaultdict(deque)
    for row in adapter_rows:
        rows_by_signature[article_signature(row["text"])].append(row)
    loader = load_data_summarization(
        args.decode_batch_size,
        8,
        "legal",
        split=["validation"],
        shuffle=False,
        sent_encoder_type="sbert",
        summary_type="oracle",
        summary_level="sen",
    )
    outputs = {"source_decoder": [], "budget_adapted_decoder": []}
    model.eval()
    with torch.no_grad():
        for _, condition in loader:
            articles = condition["article_sens"]
            batch_rows = []
            for sentences in articles:
                signature = article_signature(sentences)
                if not rows_by_signature[signature]:
                    raise RuntimeError(f"unmapped validation document: {signature!r}")
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
            scores = torch.softmax(
                torch.bmm(
                    generated[:, 200:240, :],
                    generated[:, :200, :].transpose(1, 2),
                ),
                dim=-1,
            )
            for index, sentences in enumerate(articles):
                row = batch_rows[index]
                source_indices = source_select(scores[index], len(sentences))
                budget_indices, used = budget_select(
                    scores[index],
                    len(sentences),
                    row["sentence_word_counts"],
                    int(row["reference_words"]),
                )
                source_used = sum(row["sentence_word_counts"][i] for i in source_indices)
                for name, selected, word_count in (
                    ("source_decoder", source_indices, source_used),
                    ("budget_adapted_decoder", budget_indices, used),
                ):
                    outputs[name].append(
                        {
                            "id": row["id"],
                            "selected_indices": selected,
                            "prediction": " ".join(sentences[i].strip() for i in selected),
                            "prediction_spacy_words": word_count,
                            "reference_spacy_words": int(row["reference_words"]),
                            "budget_violation": word_count > int(row["reference_words"]),
                        }
                    )
    for name, records in outputs.items():
        if len(records) != len(adapter_rows):
            raise RuntimeError(f"{name}: decoded {len(records)}, expected {len(adapter_rows)}")
        write_records(args.output_dir / name, records)
    return {
        name: {
            "documents": len(records),
            "budget_violations": sum(row["budget_violation"] for row in records),
            "mean_selected_sentences": float(
                np.mean([len(row["selected_indices"]) for row in records])
            ),
        }
        for name, records in outputs.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapted-data-dir", type=Path, required=True)
    parser.add_argument("--keywords-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--decode-batch-size", type=int, default=19)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-anneal-steps", type=int, default=400000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--resume-recovery", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    terms = json.loads(args.keywords_file.read_text(encoding="utf-8"))
    if not terms or not all(isinstance(term, str) and term.strip() for term in terms):
        raise RuntimeError("invalid English legal term lexicon")
    legal_noise.keyword = [term.strip().lower() for term in terms]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    model, diffusion, model_config = build_model()
    model.to(device)

    config = {
        "experiment": "TermDiffuSum-IN-Abs-source-faithful-200-candidate-adaptation",
        "exact_original_task_reproduction": False,
        "reason_not_exact": "LegalAFSum, its term annotations, checkpoints, and complete preprocessing are not public",
        "official_test_contents_read": False,
        "released_source_mechanics_preserved": True,
        "candidate_sentence_cap": 200,
        "summary_slots": 40,
        "source_decoder": "top-5 per slot; unique sentence; maximum 20 sentences; source order",
        "budget_decoder": "source decoder plus complete-sentence per-reference word cap",
        "english_legal_terms": terms,
        "english_legal_terms_count": len(terms),
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "decode_batch_size": args.decode_batch_size,
        "learning_rate": args.learning_rate,
        "lr_anneal_steps": args.lr_anneal_steps,
        "diffusion_steps": 500,
        "gradient_clip_norm": 3.0,
        "seed": args.seed,
        "model_config": model_config,
    }
    atomic_json(args.output_dir / "config.json", config)
    history = train(args, model, diffusion, device)

    checkpoint_path = args.output_dir / "model_final.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(checkpoint_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    decode_report = decode(args, model, diffusion, device)
    manifest = {
        **config,
        "status": "complete",
        "history": history,
        "decode_report": decode_report,
        "model_sha256": sha256(checkpoint_path),
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print("SOURCE_FAITHFUL_RUN_COMPLETE", json.dumps(decode_report), flush=True)


if __name__ == "__main__":
    main()
