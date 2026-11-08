#!/usr/bin/env python3
"""Released Budget Recipe checkpoint on frozen IN-Abs dev, without test access."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
import string
import sys
import time

import spacy
import torch

if not hasattr(torch.utils._pytree, "register_pytree_node"):
    def _register_pytree_node_compat(cls, flatten_fn, unflatten_fn, **_kwargs):
        return torch.utils._pytree._register_pytree_node(cls, flatten_fn, unflatten_fn)
    torch.utils._pytree.register_pytree_node = _register_pytree_node_compat

from transformers import AutoModelForCausalLM, AutoTokenizer


TRANSLATOR = str.maketrans("", "", string.punctuation + string.whitespace)
TERMINAL_RE = re.compile(r"[.!?][\"'’”)]*$")


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 703 or len({str(row["id"]) for row in rows}) != 703:
        raise RuntimeError("frozen 703-case dev split not found")
    if any(len(row["sentences"]) > 400 for row in rows):
        raise RuntimeError("candidate sentence cap violated")
    return rows


def word_count(nlp, text: str) -> int:
    return sum(bool(token.text.translate(TRANSLATOR)) for token in nlp.make_doc(text))


def pack_complete(nlp, generated: str, budget: int) -> tuple[str, int, int]:
    selected = []
    incomplete = 0
    skipped_for_budget = 0
    for span in nlp(generated).sents:
        sentence = span.text.strip()
        if not sentence:
            continue
        if not TERMINAL_RE.search(sentence):
            incomplete += 1
            continue
        candidate = " ".join([*selected, sentence])
        if word_count(nlp, candidate) > budget:
            skipped_for_budget += 1
            break
        selected.append(sentence)
    summary = " ".join(selected)
    if word_count(nlp, summary) > budget:
        raise RuntimeError("budget fit failed")
    return summary, incomplete, skipped_for_budget


def completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    return {str(row["id"]) for row in records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; never use for final evaluation")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "src"))
    sys.path.insert(0, str(args.repo / "reproduction"))
    from justnlp_summ.prompting.prompt_manager import JustNLPSummPromptManager
    from run_official_reproduction import greedy_generate_chunked_prefill

    rows = read_rows(args.dev)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions"
    predictions.mkdir(exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    done = completed(records_path)
    pending = [row for row in rows if str(row["id"]) not in done]
    if args.limit:
        pending = pending[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    nlp.add_pipe("sentencizer")
    prompt_manager = JustNLPSummPromptManager()
    started = time.time()
    with records_path.open("a", encoding="utf-8") as stream:
        for position, row in enumerate(pending, 1):
            judgment = "\n".join(row["sentences"])
            messages = [
                {"role": "system", "content": prompt_manager.render_system()},
                {"role": "user", "content": prompt_manager.render_summarize(judgment, think=False)},
            ]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                   tokenize=False)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            input_tokens = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                if input_tokens >= 12000:
                    new_tokens = greedy_generate_chunked_prefill(
                        model, encoded, max_new_tokens=args.max_new_tokens,
                        eos_token_id=tokenizer.eos_token_id, chunk_size=128,
                    )[0]
                else:
                    generated_ids = model.generate(
                        **encoded, max_new_tokens=args.max_new_tokens,
                        do_sample=False, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    new_tokens = generated_ids[0, input_tokens:]
                    del generated_ids
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            budget = word_count(nlp, row["summary"])
            summary, incomplete, skipped = pack_complete(nlp, raw, budget)
            used = word_count(nlp, summary)
            (predictions / f"{row['id']}.txt").write_text(summary, encoding="utf-8")
            record = {
                "id": str(row["id"]), "input_tokens": input_tokens,
                "generated_tokens": int(new_tokens.shape[0]),
                "generation_limit_hit": int(new_tokens.shape[0]) >= args.max_new_tokens,
                "prediction_words": used, "reference_words": budget,
                "incomplete_segments": incomplete, "budget_stop": skipped,
                "empty": not bool(summary),
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if position % 10 == 0 or position == len(pending):
                print(json.dumps({"stage": "generate", "done_this_run": position,
                                  "remaining": len(pending) - position,
                                  "elapsed_seconds": round(time.time() - started, 1)}), flush=True)
            del encoded, new_tokens
            torch.cuda.empty_cache()
            gc.collect()

    if args.limit:
        print(json.dumps({"stage": "smoke_complete", "cases": args.limit}), flush=True)
        return
    with records_path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if len(records) != 703 or {r["id"] for r in records} != {str(r["id"]) for r in rows}:
        raise RuntimeError("incomplete or duplicate dev predictions")
    manifest = {
        "method": "Budget Recipe released checkpoint IN-Abs zero-shot transfer",
        "dev_documents": 703, "max_new_tokens": args.max_new_tokens,
        "budget_violations": sum(r["prediction_words"] > r["reference_words"] for r in records),
        "empty_outputs": sum(r["empty"] for r in records),
        "generation_limit_hits": sum(r["generation_limit_hit"] for r in records),
        "mean_unused_budget_words": sum(r["reference_words"] - r["prediction_words"] for r in records) / 703,
        "official_test_contents_read": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"stage": "generation_complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
