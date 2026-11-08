#!/usr/bin/env python3
"""IN-Abs adaptation of the official DELTA diffusion decoding core."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
PROMPT = (
    "Summarize the following Indian legal judgment in approximately "
    "{target_chars} characters (about {target_words} English words). "
    "Preserve the parties' claims, material facts, legal reasoning, cited "
    "provisions, and final decision. Do not invent facts.\n\n{source}"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate_head_tail(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    marker = "\n[... middle of judgment omitted for context limit ...]\n"
    available = max(2, limit - len(marker))
    head = available // 2
    tail = available - head
    return text[:head] + marker + text[-tail:], True


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-code-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--avg-chars-per-token", type=float, required=True)
    parser.add_argument("--max-source-chars", type=int, default=8192)
    parser.add_argument("--max-iterations", type=int, default=1200)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.95)
    parser.add_argument("--quantization", choices=("4bit", "8bit"), default="4bit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    official_delta = args.official_code_dir / "delta"
    if not (official_delta / "diffusion_lm.py").exists():
        raise FileNotFoundError(f"official DELTA source missing under {official_delta}")
    sys.path.insert(0, str(official_delta))
    from diffusion_lm import generate_length_controlled, load_model

    rows = read_jsonl(args.input)
    if args.num_samples > 0:
        rows = rows[: args.num_samples]
    if not rows:
        raise RuntimeError("empty input")
    if args.no_resume and args.output.exists():
        args.output.unlink()
    existing = (
        {str(row["id"]): row for row in read_jsonl(args.output)}
        if args.output.exists()
        else {}
    )

    print(f"loading {args.model_id}", flush=True)
    dlm = load_model(args.model_id, quantization=args.quantization)
    _ = dlm.token_lengths

    for index, row in enumerate(rows, 1):
        case_id = str(row["id"])
        if case_id in existing:
            continue
        full_source = str(row["source"])
        source, source_truncated = truncate_head_tail(full_source, args.max_source_chars)
        target_words = int(row["target_words"])
        target_chars = int(row["target_chars"])
        prompt_text = PROMPT.format(
            target_chars=target_chars, target_words=target_words, source=source
        )
        prompt_ids = dlm.encode_prompt(prompt_text)
        generation_start = prompt_ids.shape[1]
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        output, steps, remasks = generate_length_controlled(
            dlm=dlm,
            prompt=prompt_ids,
            target_char_length=target_chars,
            max_iterations=args.max_iterations,
            confidence_threshold=args.confidence_threshold,
            remask_threshold=0.0,
            low_confidence_threshold=args.low_confidence_threshold,
            confidence_metric="probability",
            initial_gen_length=1000,
            enable_length_adjustment=True,
            avg_chars_per_token=args.avg_chars_per_token,
            buffer_size_mode="avg_chars_per_token",
            debug=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated_ids = output[0, generation_start:]
        remaining_masks = int((generated_ids == dlm.mask_id).sum().item())
        prediction = dlm.decode(generated_ids).strip()
        words = word_count(prediction)
        peak_gib = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else None
        )
        result = {
            "id": case_id,
            "method": "DELTA-Dream-INABS-643w-adapted",
            "prediction": prediction,
            "characters": len(prediction),
            "words": words,
            "target_chars": target_chars,
            "target_words": target_words,
            "character_error": abs(len(prediction) - target_chars),
            "word_error": abs(words - target_words),
            "word_budget_violation": words > target_words,
            "steps": int(steps),
            "max_iterations": args.max_iterations,
            "remaining_masks": remaining_masks,
            "hit_max_iterations": int(steps) >= args.max_iterations and remaining_masks > 0,
            "remasks": int(remasks),
            "elapsed_seconds": elapsed,
            "peak_allocated_gib": peak_gib,
            "source_characters_total": len(full_source),
            "source_characters_used": len(source),
            "source_truncated": source_truncated,
            "prompt_tokens": int(prompt_ids.shape[1]),
            "avg_chars_per_token": args.avg_chars_per_token,
            "model_id": args.model_id,
            "quantization": args.quantization,
            "seed": args.seed,
        }
        append_jsonl(args.output, result)
        print(
            f"{index}/{len(rows)} id={case_id} chars={len(prediction)} "
            f"words={words} steps={steps} masks={remaining_masks} "
            f"sec={elapsed:.2f} peakGiB={peak_gib:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

