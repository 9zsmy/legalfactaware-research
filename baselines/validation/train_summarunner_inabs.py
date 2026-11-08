#!/usr/bin/env python3
"""Train/evaluate a split-safe IN-Abs SummaRuNNer RNN_RNN adaptation."""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import spacy
import torch
from gensim.models import Word2Vec


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize(text: str):
    return text.translate(str.maketrans("", "", string.punctuation)).split()


def encode_rows(rows, word2id):
    encoded = []
    for position, row in enumerate(rows, 1):
        sentences = row["doc"].split("\n")
        labels = [int(value) for value in row["labels"].split("\n")]
        if len(sentences) != len(labels) or len(sentences) > 400:
            raise RuntimeError(f"invalid example {row.get('id')}")
        features = []
        for sentence in sentences:
            words = tokenize(sentence)[:100]
            ids = np.asarray([word2id.get(word, 1) for word in words] or [1], dtype=np.int64)
            features.append(ids)
        encoded.append({"id": row["id"], "features": features,
                        "labels": np.asarray(labels, dtype=np.float32), "row": row})
        if position % 250 == 0 or position == len(rows):
            print(json.dumps({"stage": "summarunner_encode", "done": position, "total": len(rows)}), flush=True)
    return encoded


def make_batch(examples, device):
    sentences = [sentence for example in examples for sentence in example["features"]]
    doc_lens = [len(example["features"]) for example in examples]
    max_words = max(len(sentence) for sentence in sentences)
    matrix = np.zeros((len(sentences), max_words), dtype=np.int64)
    for index, sentence in enumerate(sentences):
        matrix[index, :len(sentence)] = sentence
    labels = np.concatenate([example["labels"] for example in examples])
    return (torch.from_numpy(matrix).to(device), torch.from_numpy(labels).to(device), doc_lens)


def count_doc(doc):
    table = str.maketrans("", "", string.punctuation + string.whitespace)
    return sum(len(token.text.translate(table)) > 0 for token in doc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--dev-jsonl", type=Path, required=True)
    p.add_argument("--word2vec", type=Path, required=True)
    p.add_argument("--upstream-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--dev-limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = args.output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    sys.path.insert(0, str(args.upstream_dir))
    import models

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    word2vec = Word2Vec.load(str(args.word2vec))
    vocabulary = list(word2vec.wv.key_to_index)
    word2id = {word: index + 2 for index, word in enumerate(vocabulary)}
    embeddings = np.zeros((len(vocabulary) + 2, 100), dtype=np.float32)
    for word, index in word2id.items():
        embeddings[index] = word2vec.wv[word]
    del word2vec

    train_rows = read_jsonl(args.train_jsonl)
    dev_rows = read_jsonl(args.dev_jsonl)
    if args.train_limit:
        train_rows = train_rows[:args.train_limit]
    if args.dev_limit:
        dev_rows = dev_rows[:args.dev_limit]
    train_data = encode_rows(train_rows, word2id)
    train_document_count = len(train_data)
    del train_rows, word2id

    model_args = Namespace(
        save_dir=str(args.output_dir / "summarunner.pt"), embed_dim=100,
        embed_num=len(embeddings), pos_dim=50, pos_num=1000, seg_num=10,
        kernel_num=100, kernel_sizes=[3, 4, 5], model="RNN_RNN",
        hidden_size=200, lr=1e-5, batch_size=4, epochs=args.epochs,
        seed=seed, seq_trunc=100, max_norm=1.0,
        device=0 if device.type == "cuda" else None,
    )
    model = models.RNN_RNN(model_args, torch.from_numpy(embeddings)).to(device)
    del embeddings
    optimizer = torch.optim.Adam(model.parameters(), lr=model_args.lr)
    criterion = torch.nn.BCELoss()
    generator = np.random.default_rng(seed)
    start_epoch = 1
    checkpoint_path = args.output_dir / "summarunner.pt"
    if args.resume:
        if not checkpoint_path.exists():
            raise RuntimeError(f"resume checkpoint missing: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        required = {"model", "optimizer", "epoch", "numpy_generator_state",
                    "python_random_state", "torch_rng_state"}
        missing = required - set(checkpoint)
        if missing:
            raise RuntimeError(f"checkpoint is not exactly resumable; missing {sorted(missing)}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.bit_generator.state = checkpoint["numpy_generator_state"]
        random.setstate(checkpoint["python_random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(json.dumps({"stage": "summarunner_resume", "completed_epoch": checkpoint["epoch"],
                          "next_epoch": start_epoch}), flush=True)
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        order = generator.permutation(len(train_data))
        total_loss, batches = 0.0, 0
        for start in range(0, len(order), model_args.batch_size):
            batch = [train_data[index] for index in order[start:start + model_args.batch_size]]
            features, targets, doc_lens = make_batch(batch, device)
            probabilities = model(features, doc_lens).reshape(-1)
            loss = criterion(probabilities, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_args.max_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
            if batches % 100 == 0:
                print(json.dumps({"stage": "summarunner_train", "epoch": epoch,
                                  "batch": batches, "batches": (len(order)+3)//4,
                                  "mean_loss": total_loss / batches}), flush=True)
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "args": model_args, "epoch": epoch, "mean_loss": total_loss / batches,
            "numpy_generator_state": generator.bit_generator.state,
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }, checkpoint_path)
        print(json.dumps({"stage": "summarunner_epoch", "epoch": epoch,
                          "mean_loss": total_loss / batches,
                          "elapsed_seconds": time.time() - started}), flush=True)

    # Re-encode dev after training so train encodings can be released first.
    del train_data
    word2vec = Word2Vec.load(str(args.word2vec))
    dev_word2id = {word: index + 2 for index, word in enumerate(word2vec.wv.key_to_index)}
    del word2vec
    dev_data = encode_rows(dev_rows, dev_word2id)
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    model.eval()
    records = []
    with torch.no_grad():
        for position, example in enumerate(dev_data, 1):
            features, _, doc_lens = make_batch([example], device)
            scores = model(features, doc_lens).reshape(-1).detach().cpu().numpy()
            row = example["row"]
            sentences = row["doc"].split("\n")
            lengths = [count_doc(doc) for doc in nlp.pipe(sentences, batch_size=64)]
            ranking = sorted(range(len(sentences)), key=lambda i: (-float(scores[i]), i))
            selected, used = [], 0
            for index in ranking:
                if lengths[index] and used + lengths[index] <= row["budget"]:
                    selected.append(index)
                    used += lengths[index]
            selected.sort()
            prediction = "\n".join(sentences[index] for index in selected)
            verified = count_doc(nlp(prediction))
            if verified != used or verified > row["budget"]:
                raise RuntimeError(f"budget violation {row['id']}")
            (predictions_dir / row["file"]).write_text(prediction, encoding="utf-8")
            records.append({"id": row["id"], "selected_local_indices": selected,
                            "selected_source_indices": [row["indices"][i] for i in selected],
                            "used_words": used, "budget": row["budget"]})
            if position % 25 == 0 or position == len(dev_data):
                print(json.dumps({"stage": "summarunner_infer", "done": position,
                                  "total": len(dev_data)}), flush=True)
    with (args.output_dir / "records.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    manifest = {
        "experiment": "SummaRuNNer-INAbs-split-safe-budget-adaptation",
        "train_documents": train_document_count,
        "dev_documents": len(dev_data), "epochs": args.epochs,
        "batch_size": 4, "learning_rate": 1e-5, "hidden_size": 200,
        "candidate_cap": 400, "sentence_token_cap": 100,
        "official_test_contents_read": False, "budget_violations": 0,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"stage": "complete", **manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
