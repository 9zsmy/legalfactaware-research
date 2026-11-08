#!/usr/bin/env python3
"""Small original-task regression check for the official DELTA core.

This script deliberately keeps the DELTA decoder in the vendored official
repository.  It only replaces Hugging Face's online dataset loader with a
local-parquet loader so that the check can run reliably on the GPU server.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset


PROMPT = "Write a one-sentence summary of the following article in approximately {target_len} characters:\n\n{source}"


def parquet_files(dataset_dir: Path, split: str) -> list[str]:
    files = sorted(dataset_dir.glob(f"**/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet files under {dataset_dir}")
    return [str(path) for path in files]


def load_split(dataset_dir: Path, split: str):
    return load_dataset(
        "parquet",
        data_files={split: parquet_files(dataset_dir, split)},
        split=split,
    )


def estimate_chars_per_token(tokenizer, references: list[str]) -> float:
    ratios = []
    for text in references:
        token_ids = tokenizer(
            text, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        if token_ids:
            ratios.append(len(text) / len(token_ids))
    if not ratios:
        raise RuntimeError("no usable validation references")
    return float(np.mean(ratios))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--official-code-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument("--max-source-chars", type=int, default=8192)
    parser.add_argument("--quantization", choices=("4bit", "8bit"), default="4bit")
    args = parser.parse_args()

    official_delta = args.official_code_dir / "delta"
    sys.path.insert(0, str(official_delta))
    from diffusion_lm import generate_length_controlled, load_model

    test = load_split(args.dataset_dir, "test")
    validation = load_split(args.dataset_dir, "validation")
    test = test.select(range(min(args.num_samples, len(test))))
    validation = validation.select(
        range(min(args.validation_samples, len(validation)))
    )

    dlm = load_model(args.model_id, quantization=args.quantization)
    _ = dlm.token_lengths
    avg_chars_per_token = estimate_chars_per_token(
        dlm.tokenizer, [str(row["summary"]) for row in validation]
    )

    results = []
    for index, row in enumerate(test):
        source = str(row["document"])[: args.max_source_chars]
        target = str(row["summary"])
        target_len = len(target)
        prompt_ids = dlm.encode_prompt(
            PROMPT.format(target_len=target_len, source=source)
        )
        generation_start = prompt_ids.shape[1]

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        output, steps, remasks = generate_length_controlled(
            dlm=dlm,
            prompt=prompt_ids,
            target_char_length=target_len,
            max_iterations=100,
            confidence_threshold=0.65,
            remask_threshold=0.0,
            low_confidence_threshold=0.95,
            confidence_metric="probability",
            initial_gen_length=100,
            enable_length_adjustment=True,
            avg_chars_per_token=avg_chars_per_token,
            buffer_size_mode="avg_chars_per_token",
            debug=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = dlm.decode(output[0, generation_start:]).strip()
        results.append(
            {
                "sample_id": index,
                "target_len": target_len,
                "generated_len": len(generated),
                "len_error": abs(len(generated) - target_len),
                "steps": int(steps),
                "remasks": int(remasks),
                "time_sec": elapsed,
                "target_text": target,
                "generated_text": generated,
            }
        )
        print(
            f"{index + 1}/{len(test)} target={target_len} generated={len(generated)} "
            f"error={results[-1]['len_error']} sec={elapsed:.2f}",
            flush=True,
        )

    errors = [row["len_error"] for row in results]
    payload = {
        "check": "DELTA-Dream official-core XSum regression",
        "model_id": args.model_id,
        "quantization": args.quantization,
        "num_samples": len(results),
        "avg_chars_per_token": avg_chars_per_token,
        "statistics": {
            "mean_absolute_error": float(np.mean(errors)),
            "median_absolute_error": float(np.median(errors)),
            "within_10_count": sum(error < 10 for error in errors),
            "within_10_rate": float(np.mean([error < 10 for error in errors])),
            "mean_time_sec": float(np.mean([row["time_sec"] for row in results])),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload["statistics"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
