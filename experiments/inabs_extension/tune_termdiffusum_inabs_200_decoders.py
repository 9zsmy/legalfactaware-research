#!/usr/bin/env python3
"""Decode one frozen TermDiffuSum checkpoint with controlled decoder variants.

This is a development-set decoder audit for the restricted 200-candidate
IN-Abs sensitivity track.  It never reads the official test split and never
changes or retrains the frozen checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from imp_diff.improved_diffusion.summarization_datasets_exp import (
    load_data_summarization,
)
from run_termdiffusum_inabs_200 import (
    article_signature,
    build_model,
    move_tensors,
    read_adapter,
)


VARIANTS = (
    "slot_all_max40",
    "slot_top5_max20",
    "slot_top10_max20",
    "slot_top5_max40",
    "slot_top10_max40",
    "global_max_max20",
    "global_max_max40",
    "global_mean_max20",
    "support5_max20",
    "support5_max40",
    "support10_max20",
    "support10_max40",
)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def choose_ranked(ranked, counts, budget, candidate_count, max_sentences):
    selected = []
    used = 0
    for index in ranked:
        index = int(index)
        if index >= candidate_count or index >= len(counts) or index in selected:
            continue
        length = int(counts[index])
        if length <= 0 or used + length > budget:
            continue
        selected.append(index)
        used += length
        if len(selected) >= max_sentences:
            break
    return selected, used


def choose_slotwise(scores, counts, budget, candidate_count, top_k, max_sentences):
    selected = []
    used = 0
    for slot in scores:
        ranked = np.argsort(-slot)
        if top_k is not None:
            ranked = ranked[:top_k]
        for index in ranked:
            index = int(index)
            if index >= candidate_count or index >= len(counts) or index in selected:
                continue
            length = int(counts[index])
            if length <= 0 or used + length > budget:
                continue
            selected.append(index)
            used += length
            break
        if len(selected) >= max_sentences:
            break
    return selected, used


def support_rank(scores, top_k):
    support = np.zeros(scores.shape[1], dtype=np.int32)
    rank_bonus = np.zeros(scores.shape[1], dtype=np.float64)
    for slot in scores:
        ranked = np.argsort(-slot)[:top_k]
        support[ranked] += 1
        rank_bonus[ranked] += np.arange(top_k, 0, -1)
    maximum = scores.max(axis=0)
    # np.lexsort uses the last key as the primary key.
    return np.lexsort((-maximum, -rank_bonus, -support))


def select_variant(name, scores, counts, budget, candidate_count):
    if name == "slot_all_max40":
        return choose_slotwise(scores, counts, budget, candidate_count, None, 40)
    if name.startswith("slot_top"):
        bits = name.split("_")
        top_k = int(bits[1].replace("top", ""))
        maximum = int(bits[2].replace("max", ""))
        return choose_slotwise(
            scores, counts, budget, candidate_count, top_k, maximum
        )
    maximum = int(name.rsplit("max", 1)[1])
    if name.startswith("global_max"):
        ranked = np.argsort(-scores.max(axis=0))
    elif name.startswith("global_mean"):
        ranked = np.argsort(-scores.mean(axis=0))
    elif name.startswith("support5"):
        ranked = support_rank(scores, 5)
    elif name.startswith("support10"):
        ranked = support_rank(scores, 10)
    else:
        raise ValueError(name)
    return choose_ranked(ranked, counts, budget, candidate_count, maximum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapted-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decode-batch-size", type=int, default=19)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    model, diffusion, _ = build_model()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

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
    records = {name: [] for name in VARIANTS}
    started = time.time()
    seen = 0
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
            article_latents = generated[:, :200, :]
            summary_latents = generated[:, 200:240, :]
            scores_batch = torch.softmax(
                torch.bmm(summary_latents, article_latents.transpose(1, 2)), dim=-1
            ).cpu().numpy()

            for batch_index, sentences in enumerate(articles):
                row = batch_rows[batch_index]
                counts = row["sentence_word_counts"]
                budget = int(row["reference_words"])
                scores = scores_batch[batch_index]
                for name in VARIANTS:
                    selected, used = select_variant(
                        name, scores, counts, budget, len(sentences)
                    )
                    ordered = sorted(selected)
                    prediction = " ".join(
                        sentences[index].strip() for index in ordered
                    )
                    records[name].append(
                        {
                            "id": row["id"],
                            "selected_indices": ordered,
                            "selection_order": selected,
                            "prediction": prediction,
                            "prediction_spacy_words": used,
                            "reference_spacy_words": budget,
                            "budget_violation": used > budget,
                        }
                    )
                seen += 1
            print(
                json.dumps(
                    {
                        "decoded": seen,
                        "expected": len(adapter_rows),
                        "seconds": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )

    if seen != len(adapter_rows):
        raise RuntimeError(f"decoded {seen}, expected {len(adapter_rows)}")

    summary = {
        "status": "complete",
        "official_test_contents_read": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "seed": args.seed,
        "documents": seen,
        "seconds": round(time.time() - started, 3),
        "variants": {},
    }
    for name, rows in records.items():
        variant_dir = args.output_dir / name
        prediction_dir = variant_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        with (variant_dir / "dev_records.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                (prediction_dir / f"{row['id']}.txt").write_text(
                    row["prediction"], encoding="utf-8"
                )
        counts = [len(row["selected_indices"]) for row in rows]
        utilizations = [
            row["prediction_spacy_words"] / row["reference_spacy_words"]
            if row["reference_spacy_words"]
            else 0.0
            for row in rows
        ]
        summary["variants"][name] = {
            "mean_selected_sentences": float(np.mean(counts)),
            "median_selected_sentences": float(np.median(counts)),
            "mean_budget_utilization": float(np.mean(utilizations)),
            "budget_violations": sum(row["budget_violation"] for row in rows),
        }
    atomic_json(args.output_dir / "decode_summary.json", summary)
    print("DECODER_SWEEP_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
