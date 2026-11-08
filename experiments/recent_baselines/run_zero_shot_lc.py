#!/usr/bin/env python3
"""Run the NAACL 2025 zero-shot length-control strategies on frozen CAIL data.

This is an independent Chinese legal-domain adaptation of sample filtering
plus automated revisions. It is not an official author implementation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from protocol import read_jsonl, truncate_source, write_jsonl
from zero_shot_lc_protocol import (
    count_characters,
    is_compliant,
    revision_feedback,
    select_candidate,
)


SYSTEM_PROMPT = "你是一名助手，每次回复都只给出所请求的裁判文书摘要。"
PREFILL_EXACT = "好的！以下是恰好达到指定字符数的摘要：\n\n"
PREFILL_UPPER = "好的！以下是不超过指定字符数的摘要：\n\n"


def objective_text(target: int, objective: str) -> str:
    if objective == "exact":
        return f"恰好{target}个字符"
    if objective == "upper_bound":
        return f"不超过{target}个字符，并尽量充分利用字符预算"
    raise ValueError(f"unknown objective: {objective}")


def initial_messages(source: str, target: int, objective: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"请将以下裁判文书概括为{objective_text(target, objective)}的中文摘要。"
                "优先保留诉讼请求、关键金额和日期、法院认定、法律依据及最终裁判结果；"
                "不得编造原文没有的事实。只输出摘要正文。\n\n"
                f"{source}"
            ),
        },
    ]


def revision_messages(
    source: str,
    summary: str,
    target: int,
    objective: str,
) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"请将以下裁判文书概括为{objective_text(target, objective)}的中文摘要。"
                "优先保留诉讼请求、关键金额和日期、法院认定、法律依据及最终裁判结果；"
                "不得编造原文没有的事实。只输出摘要正文。\n\n"
                f"{source}"
            ),
        },
        {"role": "assistant", "content": summary},
        {
            "role": "user",
            "content": revision_feedback(count_characters(summary), target, objective)
            + "只输出修改后的摘要正文，不要解释，也不要报告字符数。",
        },
    ]


def prefill_for(objective: str) -> str:
    return PREFILL_EXACT if objective == "exact" else PREFILL_UPPER


def render_inputs(tokenizer, messages: list[dict], prefill: str):
    """Render an unclosed assistant prefix across Transformers versions.

    Transformers 4.41 does not support ``continue_final_message``. Rendering
    the prefill as a completed assistant turn would append Qwen's ``im_end``
    marker and make the model start a new role instead of continuing the
    summary. We therefore render the ordinary generation marker first, then
    append the paper-style prefill as raw assistant content.
    """
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer(rendered + prefill, return_tensors="pt")


def generate_summary(model, tokenizer, messages: list[dict], args, seed: int) -> str:
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inputs = render_inputs(tokenizer, messages, prefill_for(args.length_objective))
    device = next(model.parameters()).device
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    continuation = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()


def output_method(objective: str, final: bool) -> str:
    stage = "SF3-AR5" if final else "SF3"
    suffix = "exact-tol10" if objective == "exact" else "upper450-adapted"
    return f"ZeroShotLC-{stage}-Qwen2.5-7B-{suffix}"


def index_rows(path: Path) -> dict[str, dict]:
    return {str(row["id"]): row for row in read_jsonl(path)} if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--target-chars", type=int, default=450)
    parser.add_argument("--length-objective", choices=("exact", "upper_bound"), required=True)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--max-revisions", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--max-source-chars", type=int, default=8192)
    parser.add_argument("--source-truncation", choices=("head", "head_tail"), default="head_tail")
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--quantization", choices=("4bit", "8bit"), default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.num_candidates < 1:
        parser.error("--num-candidates must be at least 1")
    if args.max_revisions < 0:
        parser.error("--max-revisions cannot be negative")
    if not 0 <= args.tolerance < 1:
        parser.error("--tolerance must be in [0, 1)")

    rows = read_jsonl(args.input)
    if args.num_samples > 0:
        rows = rows[: args.num_samples]
    for path in (args.output, args.sample_output, args.trace_output):
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.no_resume and path.exists():
            path.unlink()

    completed = index_rows(args.output)
    sample_cache = index_rows(args.sample_output)

    load_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if args.quantization:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.quantization == "4bit",
            load_in_8bit=args.quantization == "8bit",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = "auto"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs).eval()

    for document_index, row in enumerate(rows):
        case_id = str(row["id"])
        if case_id in completed:
            print(f"[{document_index + 1}/{len(rows)}] {case_id}: already complete", flush=True)
            continue
        source = truncate_source(
            str(row["source"]), args.max_source_chars, args.source_truncation
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        call_index = 0

        cached = sample_cache.get(case_id)
        if cached is not None:
            candidates = [str(value) for value in cached["sample_candidates"]]
            if len(candidates) != args.num_candidates:
                raise ValueError(
                    f"cached {case_id} has {len(candidates)} candidates; "
                    f"expected {args.num_candidates}; use new output paths"
                )
            # Preserve the same revision seeds as an uninterrupted run.
            call_index = args.num_candidates
        else:
            candidates = []
            for _ in range(args.num_candidates):
                seed = args.seed + document_index * 1000 + call_index
                candidates.append(
                    generate_summary(
                        model,
                        tokenizer,
                        initial_messages(source, args.target_chars, args.length_objective),
                        args,
                        seed,
                    )
                )
                call_index += 1
            lengths = [count_characters(value) for value in candidates]
            selected_index = select_candidate(
                lengths, args.target_chars, args.length_objective
            )
            write_jsonl(
                args.sample_output,
                [
                    {
                        "id": case_id,
                        "method": output_method(args.length_objective, final=False),
                        "prediction": candidates[selected_index],
                        "characters": lengths[selected_index],
                        "budget_violation": lengths[selected_index] > args.target_chars,
                        "selected_index": selected_index,
                        "sample_candidates": candidates,
                        "sample_characters": lengths,
                        "model_id": args.model_id,
                        "objective": args.length_objective,
                        "target_chars": args.target_chars,
                        "num_candidates": args.num_candidates,
                        "seed": args.seed,
                    }
                ],
                mode="a",
            )

        lengths = [count_characters(value) for value in candidates]
        selected_index = select_candidate(lengths, args.target_chars, args.length_objective)
        current = candidates[selected_index]
        revisions = []
        for revision in range(1, args.max_revisions + 1):
            if is_compliant(
                count_characters(current),
                args.target_chars,
                args.length_objective,
                args.tolerance,
            ):
                break
            seed = args.seed + document_index * 1000 + call_index
            revised = generate_summary(
                model,
                tokenizer,
                revision_messages(
                    source, current, args.target_chars, args.length_objective
                ),
                args,
                seed,
            )
            call_index += 1
            revisions.append(
                {
                    "revision": revision,
                    "seed": seed,
                    "input_characters": count_characters(current),
                    "prediction": revised,
                    "characters": count_characters(revised),
                }
            )
            current = revised

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        final_length = count_characters(current)
        final_row = {
            "id": case_id,
            "method": output_method(args.length_objective, final=True),
            "prediction": current,
            "characters": final_length,
            "budget_violation": final_length > args.target_chars,
            "paper_tolerance_compliant": abs(final_length - args.target_chars)
            <= args.tolerance * args.target_chars,
            "objective_compliant": is_compliant(
                final_length,
                args.target_chars,
                args.length_objective,
                args.tolerance,
            ),
            "revision_count": len(revisions),
            "elapsed_seconds": elapsed,
            "model_id": args.model_id,
            "objective": args.length_objective,
            "target_chars": args.target_chars,
            "tolerance": args.tolerance,
            "num_candidates": args.num_candidates,
            "max_revisions": args.max_revisions,
            "source_truncation": args.source_truncation,
            "max_source_chars": args.max_source_chars,
            "seed": args.seed,
            "posthoc_truncation": False,
        }
        trace_row = {
            **{key: value for key, value in final_row.items() if key != "prediction"},
            "selected_sample_index": selected_index,
            "sample_candidates": candidates,
            "sample_characters": lengths,
            "revisions": revisions,
            "final_prediction": current,
        }
        write_jsonl(args.output, [final_row], mode="a")
        write_jsonl(args.trace_output, [trace_row], mode="a")
        print(
            json.dumps(
                {
                    "progress": f"{document_index + 1}/{len(rows)}",
                    "id": case_id,
                    "sample_lengths": lengths,
                    "revisions": len(revisions),
                    "final_length": final_length,
                    "compliant": final_row["objective_compliant"],
                    "seconds": round(elapsed, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

