#!/usr/bin/env python3
"""Deterministic direct-prompt baseline for the same 450-character task."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from protocol import read_jsonl, truncate_source, write_jsonl


PROMPT = (
    "请对下面的中国裁判文书生成摘要。硬性要求：摘要不得超过{target_chars}个中文字符。\n"
    "摘要应优先保留诉讼请求、关键金额和日期、法院认定、法律依据以及最终裁判结果。\n"
    "只输出摘要正文，不要解释，不要列出字符数。\n\n"
    "{source}"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--target-chars", type=int, default=450)
    parser.add_argument("--max-source-chars", type=int, default=8192)
    parser.add_argument("--source-truncation", choices=("head", "head_tail"), default="head_tail")
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--quantization", choices=("4bit", "8bit"), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.num_samples > 0:
        rows = rows[:args.num_samples]
    existing: dict[str, dict] = {}
    if args.output.exists() and not args.no_resume:
        existing = {str(row["id"]): row for row in read_jsonl(args.output)}
    elif args.output.exists():
        args.output.unlink()

    load_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if args.quantization:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.quantization == "4bit",
            load_in_8bit=args.quantization == "8bit",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = "auto"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs).eval()

    for index, row in enumerate(rows, 1):
        case_id = str(row["id"])
        if case_id in existing:
            continue
        source = truncate_source(str(row["source"]), args.max_source_chars, args.source_truncation)
        prompt = PROMPT.format(target_chars=args.target_chars, source=source)
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        device = next(model.parameters()).device
        inputs = {name: value.to(device) for name, value in inputs.items()}
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated_ids = output[0, inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        result = {
            "id": case_id,
            "method": f"DirectPrompt-{args.model_id}",
            "prediction": prediction,
            "characters": len(prediction),
            "target_chars": args.target_chars,
            "length_error": abs(len(prediction) - args.target_chars),
            "budget_violation": len(prediction) > args.target_chars,
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

