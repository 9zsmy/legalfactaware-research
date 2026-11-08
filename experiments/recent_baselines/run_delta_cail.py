#!/usr/bin/env python3
"""CAIL adaptation of the official EMNLP 2026 DELTA inference code."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from protocol import read_jsonl, truncate_source, write_jsonl


PROMPT = (
    "请将下面的中国裁判文书概括为约{target_chars}个中文字符的摘要。\n"
    "优先保留诉讼请求、关键金额和日期、法院认定、法律依据以及最终裁判结果。\n\n"
    "{source}"
)


def estimate_chars_per_token(tokenizer, references: list[str]) -> float:
    ratios = []
    for text in references:
        ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        if ids:
            ratios.append(len(text) / len(ids))
    if not ratios:
        raise RuntimeError("could not estimate characters per token")
    return float(np.mean(ratios))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-code-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--model-name", choices=("dream", "llada"), default="dream")
    parser.add_argument("--target-chars", type=int, default=450)
    parser.add_argument("--max-source-chars", type=int, default=8192)
    parser.add_argument("--source-truncation", choices=("head", "head_tail"), default="head_tail")
    parser.add_argument("--num-samples", type=int, default=0, help="0 means all rows")
    parser.add_argument("--avg-chars-per-token", type=float, default=None)
    parser.add_argument("--estimate-from-references", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.95)
    parser.add_argument("--confidence-metric", default="probability")
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--initial-gen-length", type=int, default=500)
    parser.add_argument("--quantization", choices=("4bit", "8bit"), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    official_delta = args.official_code_dir / "delta"
    if not (official_delta / "diffusion_lm.py").exists():
        raise FileNotFoundError(f"official DELTA source missing under {official_delta}")
    sys.path.insert(0, str(official_delta))
    from diffusion_lm import generate_length_controlled, load_model

    rows = read_jsonl(args.input)
    if args.num_samples > 0:
        rows = rows[:args.num_samples]
    if not rows:
        raise RuntimeError("input dataset is empty")

    existing: dict[str, dict] = {}
    if args.output.exists() and not args.no_resume:
        existing = {str(row["id"]): row for row in read_jsonl(args.output)}
    elif args.output.exists():
        args.output.unlink()

    print(f"loading {args.model_id}", flush=True)
    dlm = load_model(args.model_id, quantization=args.quantization)
    _ = dlm.token_lengths

    chars_per_token = args.avg_chars_per_token
    if args.estimate_from_references:
        chars_per_token = estimate_chars_per_token(
            dlm.tokenizer, [str(row["reference"]) for row in rows]
        )
    if chars_per_token is None:
        raise ValueError("provide --avg-chars-per-token or --estimate-from-references")
    print(f"avg_chars_per_token={chars_per_token:.6f}", flush=True)

    for index, row in enumerate(rows, 1):
        case_id = str(row["id"])
        if case_id in existing:
            continue
        source = truncate_source(str(row["source"]), args.max_source_chars, args.source_truncation)
        prompt = PROMPT.format(target_chars=args.target_chars, source=source)
        prompt_ids = dlm.encode_prompt(prompt)
        generation_start = prompt_ids.shape[1]
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        output, steps, remasks = generate_length_controlled(
            dlm=dlm,
            prompt=prompt_ids,
            target_char_length=args.target_chars,
            max_iterations=args.max_iterations,
            confidence_threshold=args.confidence_threshold,
            remask_threshold=0.0,
            low_confidence_threshold=args.low_confidence_threshold,
            confidence_metric=args.confidence_metric,
            initial_gen_length=args.initial_gen_length,
            enable_length_adjustment=True,
            avg_chars_per_token=chars_per_token,
            buffer_size_mode="avg_chars_per_token",
            debug=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        prediction = dlm.decode(output[0, generation_start:]).strip()
        result = {
            "id": case_id,
            "method": f"DELTA-{args.model_name}-CAIL-adapted",
            "prediction": prediction,
            "characters": len(prediction),
            "target_chars": args.target_chars,
            "length_error": abs(len(prediction) - args.target_chars),
            "budget_violation": len(prediction) > args.target_chars,
            "steps": int(steps),
            "remasks": int(remasks),
            "elapsed_seconds": elapsed,
            "source_characters_used": len(source),
            "source_truncation": args.source_truncation,
            "model_id": args.model_id,
            "seed": args.seed,
        }
        write_jsonl(args.output, [result], mode="a")
        print(f"{index}/{len(rows)} id={case_id} chars={len(prediction)} sec={elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()

