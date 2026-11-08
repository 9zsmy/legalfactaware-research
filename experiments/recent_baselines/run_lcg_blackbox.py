#!/usr/bin/env python3
"""CAIL adaptation of ACL 2025 Length Controlled Generation for black-box LLMs."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lcg_protocol import (
    acceptance_probability,
    exact_target_distance,
    extract_score_pair,
    objective_satisfied,
    select_final_chain,
    upper_budget_distance,
)
from protocol import read_jsonl, truncate_source, write_jsonl


SYSTEM_SUMMARIZER = "你是一名能力很强的中国裁判文书摘要助手。"


def generate_chat(model, tokenizer, messages: list[dict], args, max_new_tokens: int) -> str:
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    device = next(model.parameters()).device
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def objective_instruction(length: int, objective: str) -> str:
    if objective == "exact":
        return f"生成尽可能精确达到{length}个 Python 字符"
    return f"生成不超过{length}个 Python 字符"


def initial_messages(source: str, demo: dict, budget: int, objective: str) -> list[dict]:
    demo_chars = len(str(demo["summary"]))
    return [
        {"role": "system", "content": SYSTEM_SUMMARIZER},
        {
            "role": "user",
            "content": (
                f"裁判文书：\n{demo['source']}\n\n"
                f"请根据文书{objective_instruction(demo_chars, objective)}的高质量摘要。"
                "优先保留诉讼请求、关键金额和日期、法院认定、法律依据以及最终裁判结果。"
                "只输出摘要正文。"
            ),
        },
        {"role": "assistant", "content": str(demo["summary"])},
        {
            "role": "user",
            "content": (
                f"裁判文书：\n{source}\n\n"
                f"请根据文书{objective_instruction(budget, objective)}的高质量摘要。"
                "优先保留诉讼请求、关键金额和日期、法院认定、法律依据以及最终裁判结果。"
                "不得编造文书中没有的主体、金额、日期、法律依据或裁判结论。"
                "只输出摘要正文，不要解释，不要报告字符数。"
            ),
        },
    ]


def proposal_messages(source: str, current: str, budget: int, objective: str) -> list[dict]:
    distance = (
        exact_target_distance(len(current), budget)
        if objective == "exact"
        else upper_budget_distance(len(current), budget)
    )
    if distance <= 3:
        action = "增加" if len(current) < budget else "删除"
        instruction = (
            f"当前摘要为{len(current)}个 Python 字符。请恰当地{action}{distance}个字符，"
            f"使其达到{budget}个字符，同时保持事实、裁判结论和语言完整。只输出修改后的摘要。"
        )
    elif objective == "exact" and len(current) < budget:
        instruction = (
            f"当前摘要偏短，共{len(current)}个 Python 字符。请依据原文补充遗漏的关键事实、"
            f"法院认定、法律依据和裁判结果，改写为尽可能精确达到{budget}个字符。不得编造事实。"
            "只输出修改后的摘要。"
        )
    else:
        instruction = (
            f"当前摘要过长，共{len(current)}个 Python 字符。请改写为不超过{budget}个字符，"
            "聚焦核心事实、法院认定和裁判结果，删除冗余内容，不得增加原文没有的事实。"
            "只输出修改后的摘要。"
        )
    return [
        {"role": "system", "content": SYSTEM_SUMMARIZER},
        {
            "role": "user",
            "content": (
                f"裁判文书：\n{source}\n\n"
                f"请根据文书{objective_instruction(budget, objective)}的高质量摘要。"
            ),
        },
        {"role": "assistant", "content": current},
        {"role": "user", "content": instruction},
    ]


def judge_messages(source: str, candidate: str, current: str) -> list[dict]:
    return [
        {"role": "system", "content": "你是一名严格、公正的裁判文书摘要质量评估员。"},
        {
            "role": "user",
            "content": (
                "请比较同一裁判文书的两个摘要。分别按1至10分评价：信息覆盖、语言流畅、"
                "简洁性、逻辑连贯、事实忠实。每个摘要总分最高50分。不要因为摘要位置或长度"
                "本身产生偏好。\n\n"
                f"裁判文书：\n{source}\n\n摘要1：\n{candidate}\n\n摘要2：\n{current}\n\n"
                "只输出一个 JSON 对象，不要输出其他内容："
                '{"summary_1_total": 数值, "summary_2_total": 数值}'
            ),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--target-chars", type=int, default=450)
    parser.add_argument(
        "--length-objective", choices=("upper_bound", "exact"), default="upper_bound"
    )
    parser.add_argument("--max-source-chars", type=int, default=8192)
    parser.add_argument("--max-demo-chars", type=int, default=4096)
    parser.add_argument("--source-truncation", choices=("head", "head_tail"), default="head_tail")
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--judge-max-new-tokens", type=int, default=128)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help=(
            "Number of independent initial states / MH+IS chains. The paper's "
            "white-box-model main configuration uses 16. Chains are executed "
            "sequentially here to fit a 24GB GPU."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--quantization", choices=("4bit", "8bit"), default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.beam_size < 1:
        parser.error("--beam-size must be at least 1")

    rows = read_jsonl(args.input)
    if args.num_samples > 0:
        rows = rows[:args.num_samples]
    demo = read_jsonl(args.demo)[0]
    demo["source"] = truncate_source(
        str(demo["source"]), args.max_demo_chars, args.source_truncation
    )
    existing = {} if args.no_resume or not args.output.exists() else {
        str(row["id"]): row for row in read_jsonl(args.output)
    }
    initial_existing = {} if args.no_resume or not args.initial_output.exists() else {
        str(row["id"]): row for row in read_jsonl(args.initial_output)
    }
    if args.no_resume:
        for path in (args.output, args.initial_output):
            if path.exists():
                path.unlink()

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

    for index, row in enumerate(rows, 1):
        case_id = str(row["id"])
        if case_id in existing:
            continue
        source = truncate_source(str(row["source"]), args.max_source_chars, args.source_truncation)
        torch.manual_seed(args.seed)
        randomizer = random.Random(args.seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()

        if case_id in initial_existing:
            initial_row = initial_existing[case_id]
            if "beam_predictions" in initial_row:
                initial_beams = [str(value) for value in initial_row["beam_predictions"]]
            else:
                initial_beams = [str(initial_row["prediction"])]
            if len(initial_beams) != args.beam_size:
                raise ValueError(
                    f"initial output for {case_id} contains {len(initial_beams)} beam(s), "
                    f"but --beam-size={args.beam_size}; use a new initial-output path"
                )
        else:
            initial_beams = [
                generate_chat(
                    model,
                    tokenizer,
                    initial_messages(source, demo, args.target_chars, args.length_objective),
                    args,
                    args.max_new_tokens,
                )
                for _ in range(args.beam_size)
            ]
            write_jsonl(args.initial_output, [{
                "id": case_id,
                "method": "Qwen2.5-7B-DirectPrompt-one-shot",
                "prediction": initial_beams[0],
                "characters": len(initial_beams[0]),
                "budget_violation": len(initial_beams[0]) > args.target_chars,
                "beam_size": args.beam_size,
                "beam_predictions": initial_beams,
                "beam_characters": [len(value) for value in initial_beams],
                "model_id": args.model_id,
                "seed": args.seed,
            }], mode="a")

        currents = list(initial_beams)
        beam_traces: list[list[dict]] = [[] for _ in range(args.beam_size)]
        for iteration in range(1, args.max_iterations + 1):
            if any(
                objective_satisfied(len(current), args.target_chars, args.length_objective)
                for current in currents
            ):
                break
            found_success = False
            for beam_index, current in enumerate(currents):
                if objective_satisfied(
                    len(current), args.target_chars, args.length_objective
                ):
                    found_success = True
                    break
                candidate = generate_chat(
                    model,
                    tokenizer,
                    proposal_messages(source, current, args.target_chars, args.length_objective),
                    args,
                    args.max_new_tokens,
                )
                score_1 = score_2 = None
                judge_raw = ""
                quality_ratio = 1.0
                if not objective_satisfied(
                    len(candidate), args.target_chars, args.length_objective
                ):
                    judge_raw = generate_chat(
                        model,
                        tokenizer,
                        judge_messages(source, candidate, current),
                        args,
                        args.judge_max_new_tokens,
                    )
                    try:
                        score_1, score_2 = extract_score_pair(judge_raw)
                        quality_ratio = score_1 / score_2
                    except ValueError:
                        quality_ratio = 1.0
                probability = acceptance_probability(
                    len(current),
                    len(candidate),
                    quality_ratio,
                    args.target_chars,
                    objective=args.length_objective,
                )
                draw = randomizer.random()
                accepted = draw <= probability
                beam_traces[beam_index].append({
                    "beam": beam_index,
                    "iteration": iteration,
                    "current_chars": len(current),
                    "candidate_chars": len(candidate),
                    "judge_score_1": score_1,
                    "judge_score_2": score_2,
                    "quality_ratio": quality_ratio,
                    "acceptance_probability": probability,
                    "uniform_draw": draw,
                    "accepted": accepted,
                    "judge_raw": judge_raw,
                })
                if accepted:
                    currents[beam_index] = candidate
                if objective_satisfied(
                    len(currents[beam_index]), args.target_chars, args.length_objective
                ):
                    found_success = True
                    break
            if found_success:
                break

        selected_beam = select_final_chain(
            [len(current) for current in currents],
            args.target_chars,
            args.length_objective,
        )
        initial = initial_beams[selected_beam]
        current = currents[selected_beam]
        trace = beam_traces[selected_beam]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        result = {
            "id": case_id,
            "method": (
                f"LCG-MHIS-beam{args.beam_size}-Qwen2.5-7B-CAIL-exact-adapted"
                if args.length_objective == "exact"
                else f"LCG-MHIS-beam{args.beam_size}-Qwen2.5-7B-CAIL-upper-adapted"
            ),
            "prediction": current,
            "characters": len(current),
            "target_chars": args.target_chars,
            "length_objective": args.length_objective,
            "length_error": abs(len(current) - args.target_chars),
            "budget_violation": len(current) > args.target_chars,
            "initial_characters": len(initial),
            "iterations": len(trace),
            "trace": trace,
            "beam_size": args.beam_size,
            "selected_beam": selected_beam,
            "beam_initial_characters": [len(value) for value in initial_beams],
            "beam_final_characters": [len(value) for value in currents],
            "beam_iterations": [len(value) for value in beam_traces],
            "beam_success": [
                objective_satisfied(len(value), args.target_chars, args.length_objective)
                for value in currents
            ],
            "beam_initial_outputs": initial_beams,
            "beam_final_outputs": currents,
            "beam_traces": beam_traces,
            "final_selection_rule": "first successful chain; otherwise minimum length distance",
            "beam_execution": "sequential independent chains",
            "elapsed_seconds": elapsed,
            "source_characters_used": len(source),
            "source_truncation": args.source_truncation,
            "model_id": args.model_id,
            "seed": args.seed,
        }
        write_jsonl(args.output, [result], mode="a")
        print(
            f"{index}/{len(rows)} id={case_id} beam={args.beam_size} "
            f"selected={selected_beam} initial={len(initial)} final={len(current)} "
            f"proposals={sum(len(value) for value in beam_traces)} sec={elapsed:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

