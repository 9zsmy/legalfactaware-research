#!/usr/bin/env python3
"""Train and dev-evaluate the English CivilSum-style LED adaptation safely."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import string
import time

from datasets import Dataset
import numpy as np
from rouge_score import rouge_scorer
import spacy
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


TRANSLATOR = str.maketrans("", "", string.punctuation + string.whitespace)


def read_rows(path: Path, expected: int) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != expected:
        raise RuntimeError(f"{path}: expected {expected} rows, got {len(rows)}")
    if len({str(row["id"]) for row in rows}) != expected:
        raise RuntimeError(f"duplicate IDs in {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def word_count(nlp, text: str) -> int:
    return sum(bool(token.text.translate(TRANSLATOR)) for token in nlp.make_doc(text))


def pack_generated(nlp, generated: str, budget: int) -> tuple[str, int, int]:
    sentences = [span.text.strip() for span in nlp(generated).sents if span.text.strip()]
    chosen: list[str] = []
    skipped = 0
    for sentence in sentences:
        trial = " ".join([*chosen, sentence])
        if word_count(nlp, trial) <= budget:
            chosen.append(sentence)
        else:
            skipped += 1
    summary = " ".join(chosen)
    if word_count(nlp, summary) > budget:
        raise RuntimeError("budget packing failed")
    return summary, len(chosen), skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty output: {args.output}")
    args.output.mkdir(parents=True)
    started = time.time()
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    nlp.add_pipe("sentencizer")
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    train_rows = read_rows(args.train, 6325)
    dev_rows = read_rows(args.dev, 703)
    if args.smoke:
        train_rows, dev_rows = train_rows[:16], dev_rows[:8]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.truncation_side = "left"
    documents = lambda rows: [" ".join(row["sentences"]) for row in rows]
    train_docs, dev_docs = documents(train_rows), documents(dev_rows)
    input_truncated = {
        "train": sum(len(tokenizer(doc, add_special_tokens=True)["input_ids"]) > 1024 for doc in train_docs),
        "dev": sum(len(tokenizer(doc, add_special_tokens=True)["input_ids"]) > 1024 for doc in dev_docs),
    }

    def make_dataset(rows: list[dict], docs: list[str]) -> Dataset:
        return Dataset.from_list([
            {"document": doc, "summary": row["summary"]}
            for row, doc in zip(rows, docs)
        ])

    def preprocess(batch: dict) -> dict:
        result = tokenizer(batch["document"], max_length=1024, truncation=True)
        labels = tokenizer(text_target=batch["summary"], max_length=512, truncation=True)
        result["labels"] = labels["input_ids"]
        return result

    train_ds = make_dataset(train_rows, train_docs).map(preprocess, batched=True,
        remove_columns=["document", "summary"])
    dev_ds = make_dataset(dev_rows, dev_docs).map(preprocess, batched=True,
        remove_columns=["document", "summary"])
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, local_files_only=True)
    model.config.num_beams = 3
    model.config.max_length = 512
    model.config.min_length = 100
    model.config.length_penalty = 2.0
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True

    reference_budgets = [word_count(nlp, row["summary"]) for row in dev_rows]
    references = [row["summary"] for row in dev_rows]

    def compute_metrics(prediction) -> dict:
        generated = tokenizer.batch_decode(prediction.predictions, skip_special_tokens=True)
        if len(generated) != len(references):
            raise RuntimeError("dev prediction order/count changed")
        budgeted = [pack_generated(nlp, text, budget)[0]
                    for text, budget in zip(generated, reference_budgets)]
        return {
            "budget_rouge1": float(np.mean([
                scorer.score(ref, pred)["rouge1"].fmeasure
                for ref, pred in zip(references, budgeted)
            ])),
            "mean_words": float(np.mean([word_count(nlp, text) for text in budgeted])),
        }

    max_steps = 2 if args.smoke else 1500
    eval_steps = 2 if args.smoke else 500
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        evaluation_strategy="steps", save_strategy="steps",
        learning_rate=1e-4, per_device_train_batch_size=4,
        per_device_eval_batch_size=2, gradient_accumulation_steps=32,
        weight_decay=0.01, max_steps=max_steps,
        warmup_steps=1 if args.smoke else 150,
        logging_steps=1 if args.smoke else 20,
        eval_steps=eval_steps, save_steps=eval_steps,
        predict_with_generate=True, generation_max_length=512,
        generation_num_beams=3, load_best_model_at_end=True,
        metric_for_best_model="budget_rouge1", greater_is_better=True,
        fp16=True, save_total_limit=1, save_only_model=True,
        report_to=[], seed=2024, data_seed=2024,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=train_ds,
        eval_dataset=dev_ds, tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
    )
    manifest = {
        "method": "CivilSum LED-tail English IN-Abs generative adaptation",
        "train_documents": len(train_rows), "dev_documents": len(dev_rows),
        "train_sha256": sha256(args.train), "dev_sha256": sha256(args.dev),
        "model": str(args.model), "source_candidate_cap": 400,
        "input_tail_tokens": 1024, "target_tokens": 512,
        "input_truncated": input_truncated,
        "max_steps": max_steps, "eval_steps": eval_steps,
        "official_test_contents_read": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"stage": "training_start", **manifest}), flush=True)
    trainer.train()
    if args.smoke:
        print(json.dumps({"stage": "smoke_complete", "best_step": trainer.state.best_model_checkpoint}), flush=True)
        return

    result = trainer.predict(dev_ds, max_length=512, num_beams=3)
    generated = tokenizer.batch_decode(result.predictions, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
    if len(generated) != 703:
        raise RuntimeError("missing dev generations")
    predictions_dir = args.output / "predictions"
    predictions_dir.mkdir()
    records = []
    for row, text, budget in zip(dev_rows, generated, reference_budgets):
        summary, sentence_count, skipped = pack_generated(nlp, text, budget)
        used = word_count(nlp, summary)
        (predictions_dir / f"{row['id']}.txt").write_text(summary, encoding="utf-8")
        records.append({
            "id": str(row["id"]), "prediction_words": used,
            "reference_words": budget, "sentences": sentence_count,
            "skipped_segments": skipped, "empty": not bool(summary),
        })
    with (args.output / "dev_records.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    summary_report = {
        "stage": "decode_complete", "dev_documents": len(records),
        "budget_violations": sum(r["prediction_words"] > r["reference_words"] for r in records),
        "empty_outputs": sum(r["empty"] for r in records),
        "mean_unused_words": float(np.mean([r["reference_words"] - r["prediction_words"] for r in records])),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "elapsed_seconds": round(time.time() - started, 2),
        "official_test_contents_read": False,
    }
    (args.output / "decode_manifest.json").write_text(json.dumps(summary_report, indent=2) + "\n")
    print(json.dumps(summary_report), flush=True)


if __name__ == "__main__":
    main()
