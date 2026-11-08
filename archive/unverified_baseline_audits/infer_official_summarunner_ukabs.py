#!/usr/bin/env python3
"""Run the published UK-Abs SummaRuNNer checkpoint with the original vocabulary.

The source-backed primary configuration is ``raw`` tokenization: this is exactly
the whitespace tokenization performed by hpzhao/SummaRuNNer's
``Vocab.make_features``.  Lower-casing and punctuation stripping are emitted as
diagnostics only, not as alternative primary results.
"""
from __future__ import annotations

import argparse
import json
import string
import sys
import time
from pathlib import Path

import numpy as np
import torch


def tokenize(text: str, mode: str) -> list[str]:
    if mode == "raw":
        return text.split()
    if mode == "lower":
        return text.lower().split()
    if mode == "strip-punctuation":
        return text.translate(str.maketrans("", "", string.punctuation)).split()
    raise ValueError(mode)


def encode(sentences: list[str], word2id: dict[str, int], token_cap: int,
           mode: str, device: torch.device) -> torch.Tensor:
    rows = []
    for sentence in sentences:
        words = tokenize(sentence, mode)[:token_cap]
        rows.append([word2id.get(word, 1) for word in words] or [1])
    width = max(map(len, rows))
    matrix = np.zeros((len(rows), width), dtype=np.int64)
    for index, row in enumerate(rows):
        matrix[index, :len(row)] = row
    return torch.from_numpy(matrix).to(device)


def decode(scores: np.ndarray, sentences: list[str], budget: int,
           policy: str) -> tuple[list[int], int]:
    lengths = [len(sentence.split()) for sentence in sentences]
    ranking = sorted(range(len(sentences)), key=lambda i: (-float(scores[i]), i))
    if policy.startswith("top15"):
        ranking = ranking[:15]
    if policy == "top15":
        selected = sorted(ranking)
        return selected, sum(lengths[index] for index in selected)
    selected, used = [], 0
    for index in ranking:
        length = lengths[index]
        if length <= 0:
            continue
        if used + length <= budget:
            selected.append(index)
            used += length
        elif policy in {"break", "top15-break"}:
            break
    selected.sort()
    return selected, used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--word2id", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-cap", type=int, default=100)
    parser.add_argument("--sentence-token-cap", type=int, default=50)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.upstream_dir))
    import models

    word2id = json.loads(args.word2id.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.test_jsonl.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    model_args = checkpoint["args"]
    model_args.device = 0 if device.type == "cuda" else None
    if len(word2id) != model_args.embed_num:
        raise RuntimeError(
            f"vocabulary/checkpoint mismatch: {len(word2id)} != "
            f"{model_args.embed_num}"
        )
    model = getattr(models, model_args.model)(model_args).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    configurations = [
        (tokenization, policy)
        for tokenization in ("raw", "lower", "strip-punctuation")
        for policy in ("break", "fit")
    ] + [("raw", policy) for policy in ("top15", "top15-break", "top15-fit")]
    for tokenization, policy in configurations:
        (args.output_dir / f"{tokenization}_{policy}" / "predictions").mkdir(
            parents=True, exist_ok=True
        )

    started = time.time()
    records = {f"{t}_{p}": [] for t, p in configurations}
    with torch.no_grad():
        for position, row in enumerate(rows, 1):
            sentences = row["doc"].split("\n")[:args.candidate_cap]
            scores_by_tokenization = {}
            for tokenization in ("raw", "lower", "strip-punctuation"):
                features = encode(
                    sentences, word2id, args.sentence_token_cap,
                    tokenization, device,
                )
                scores_by_tokenization[tokenization] = model(
                    features, [len(sentences)]
                ).reshape(-1).detach().cpu().numpy()
            for tokenization, policy in configurations:
                selected, used = decode(
                    scores_by_tokenization[tokenization], sentences,
                    int(row["budget"]), policy,
                )
                prediction = "\n".join(sentences[index] for index in selected)
                output = args.output_dir / f"{tokenization}_{policy}"
                (output / "predictions" / row["file"]).write_text(
                    prediction, encoding="utf-8"
                )
                records[f"{tokenization}_{policy}"].append({
                    "id": row["id"], "selected_local_indices": selected,
                    "selected_source_indices": [row["indices"][i] for i in selected],
                    "used_whitespace_words": used, "budget": int(row["budget"]),
                })
            if position % 10 == 0 or position == len(rows):
                print(json.dumps({"stage": "official_checkpoint_infer",
                                  "done": position, "total": len(rows),
                                  "elapsed_seconds": time.time() - started}),
                      flush=True)

    for tokenization, policy in configurations:
        name = f"{tokenization}_{policy}"
        output = args.output_dir / name
        with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
            for record in records[name]:
                stream.write(json.dumps(record) + "\n")
        manifest = {
            "experiment": "published UK-Abs SummaRuNNer checkpoint audit",
            "checkpoint": str(args.checkpoint),
            "word2id": str(args.word2id),
            "word2id_entries": len(word2id),
            "checkpoint_embedding_rows": model_args.embed_num,
            "tokenization": tokenization,
            "decode_policy": policy,
            "candidate_cap": args.candidate_cap,
            "sentence_token_cap": args.sentence_token_cap,
            "documents": len(rows),
            "budget_violations_whitespace": sum(
                row["used_whitespace_words"] > row["budget"]
                for row in records[name]
            ),
            "source_backed_primary": tokenization == "raw",
            "elapsed_seconds_all_configurations": time.time() - started,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"stage": "complete", "documents": len(rows),
                      "elapsed_seconds": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()
