"""
Compute evaluation metrics on generated summaries.
Metrics: ROUGE-1/2/L, BERTScore, length error, LLM as Judge (scoring Relevance, Faithfulness, Coherence, Fluency)
Works with CNN/DM, arXiv, and XSum datasets.
"""
import os
import argparse
import asyncio
import inspect
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Lazy import — only needed for llm_as_judge metric
QuantizationOptions = None


def load_results(path: str) -> tuple:
    with open(path) as f:
        data = json.load(f)
    dataset = data.get("dataset", "cnndm")
    return data["results"], dataset


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if v.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def compute_rouge(*, results: list, silent=False, **kwargs) -> dict:
    print("\n=== ROUGE Scores ===")
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for r in tqdm(results, desc="Computing ROUGE"):
        s = scorer.score(r['target_text'], r['generated_text'])
        scores['rouge1'].append(s['rouge1'].fmeasure)
        scores['rouge2'].append(s['rouge2'].fmeasure)
        scores['rougeL'].append(s['rougeL'].fmeasure)

    res = {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in scores.items()}

    if not silent:
        print(f"ROUGE-1: {res['rouge1']['mean']:.4f} (±{res['rouge1']['std']:.4f})")
        print(f"ROUGE-2: {res['rouge2']['mean']:.4f} (±{res['rouge2']['std']:.4f})")
        print(f"ROUGE-L: {res['rougeL']['mean']:.4f} (±{res['rougeL']['std']:.4f})")

    return res


def compute_bertscore(*, results: list, lang: str = 'en', batch_size: int = 64, silent=False, **kwargs) -> dict:
    print("\n=== BERTScore ===")
    from bert_score import score

    refs = [r['target_text'] for r in results]
    hyps = [r['generated_text'] for r in results]

    unscaled_P, unscaled_R, unscaled_F1 = score(hyps, refs, lang=lang, verbose=True, batch_size=batch_size)
    scaled_P, scaled_R, scaled_F1 = score(hyps, refs, lang=lang, verbose=True, batch_size=batch_size,
                                          rescale_with_baseline=True)

    unscaled_result = {
        'precision': {'mean': unscaled_P.mean().item(), 'std': unscaled_P.std().item()},
        'recall': {'mean': unscaled_R.mean().item(), 'std': unscaled_R.std().item()},
        'f1': {'mean': unscaled_F1.mean().item(), 'std': unscaled_F1.std().item()},
    }

    scaled_result = {
        'precision': {'mean': scaled_P.mean().item(), 'std': scaled_P.std().item()},
        'recall': {'mean': scaled_R.mean().item(), 'std': scaled_R.std().item()},
        'f1': {'mean': scaled_F1.mean().item(), 'std': scaled_F1.std().item()},
    }

    if not silent:
        print("BERTScore unscaled:")
        print(
            f"Precision: {unscaled_result['precision']['mean']:.4f} (±{unscaled_result['precision']['std']:.4f})")
        print(f"Recall: {unscaled_result['recall']['mean']:.4f} (±{unscaled_result['recall']['std']:.4f})")
        print(f"F1: {unscaled_result['f1']['mean']:.4f} (±{unscaled_result['f1']['std']:.4f})")

        print("BERTScore scaled:")
        print(
            f"Precision: {scaled_result['precision']['mean']:.4f} (±{scaled_result['precision']['std']:.4f})")
        print(f"Recall: {scaled_result['recall']['mean']:.4f} (±{scaled_result['recall']['std']:.4f})")
        print(f"F1: {scaled_result['f1']['mean']:.4f} (±{scaled_result['f1']['std']:.4f})")

    return {"unscaled": unscaled_result, "scaled": scaled_result}


def compute_length_metrics(*, results: list, silent=False, **kwargs) -> dict:
    print("\n=== Length Metrics ===")

    target_lens = [r['target_len'] for r in results]
    gen_lens = [r['generated_len'] for r in results]

    # Always recompute from raw values to avoid format inconsistencies
    len_errors = [abs(g - t) for g, t in zip(gen_lens, target_lens)]
    signed_errors = [g - t for g, t in zip(gen_lens, target_lens)]

    res = {
        'abs_error': {'mean': np.mean(len_errors), 'std': np.std(len_errors), 'median': np.median(len_errors)},
        'signed_error': {'mean': np.mean(signed_errors), 'std': np.std(signed_errors)},
        'under_10chars': sum(1 for e in len_errors if e < 10) / len(len_errors),
        'under_25chars': sum(1 for e in len_errors if e < 25) / len(len_errors),
        'under_50chars': sum(1 for e in len_errors if e < 50) / len(len_errors),
    }
    if not silent:
        print(f"Mean absolute error: {res['abs_error']['mean']:.1f} chars")
        print(f"Median absolute error: {res['abs_error']['median']:.1f} chars")
        print(f"Mean signed error: {res['signed_error']['mean']:+.1f} chars")
        print(f"Samples <10 chars error: {res['under_10chars']:.1%}")
        print(f"Samples <25 chars error: {res['under_25chars']:.1%}")
        print(f"Samples <50 chars error: {res['under_50chars']:.1%}")

    return res

async def compute_llm_as_judge(*, results: list, model_name: str, token: str, quantization: str = "noquant",
                               silent=False, concurrency: int = 500, **kwargs) -> dict:

    from LlmAsJudge.model import get_agent, Agent, Scores, JsonEncoder, OpenAIAgent
    text_types = ("source_text", "generated_text")
    agent: Agent = get_agent(model_name, token, quantization)

    if isinstance(agent, OpenAIAgent):
        semaphore = asyncio.Semaphore(concurrency)
        async def infer_with_semaphore(r):
            async with semaphore:
                return await agent.infer((r[text_types[0]], r[text_types[1]]))
        scores = await tqdm_asyncio.gather(
            *[infer_with_semaphore(r) for r in results],
            desc=f"Comparing using {agent.__class__.__name__}({model_name})")
    else:
        scores = []
        for r in tqdm(results, desc=f"Comparing using {agent.__class__.__name__}({model_name})"):
            scores.append(agent.infer((r[text_types[0]], r[text_types[1]])))

    ### mean and std
    dc = {}
    for metric in Scores.model_fields.keys():
        values = [getattr(s, metric, 0) for s in scores]
        mean, std = np.mean(values), np.std(values)
        dc[metric] = {"mean": float(mean), "std": float(std)}

    dc["scores"] = [
        {"sample_id": r["sample_id"],
         "score": json.dumps(s, cls=JsonEncoder)} for r, s in zip(results, scores)
    ]
    return dc


def compute_all(*, metrics, results, model_name, token,
                quantization, lang, batch_size, silent=False):
    metrics['length'] = compute_length_metrics(results=results, silent=silent)
    metrics['rouge'] = compute_rouge(results=results, silent=silent)
    metrics['bertscore'] = compute_bertscore(results=results, lang=lang, batch_size=batch_size, silent=silent)
    metrics['llm_as_judge'] = asyncio.run(compute_llm_as_judge(results=results, model_name=model_name, token=token,
                                                               quantization=quantization, silent=silent))


EVAL_FUNCS_MAPPING = {
    "llm_as_judge": compute_llm_as_judge,
    "bertscore": compute_bertscore,
    "length_metrics": compute_length_metrics,
    "rouge": compute_rouge,
    "all": compute_all,
}


def run_evaluation(*methods: str, results, model_name, token,
                   quantization, lang, batch_size, silent=False):
    metrics = {}
    if "all" in methods:
        EVAL_FUNCS_MAPPING["all"](metrics=metrics, results=results, model_name=model_name, token=token,
                                  quantization=quantization, lang=lang, batch_size=batch_size, silent=silent)
        return metrics

    for method in methods:
        func = EVAL_FUNCS_MAPPING[method]
        if inspect.iscoroutinefunction(func):
            metrics[method] = asyncio.run(func(results=results, model_name=model_name, token=token,
                                               quantization=quantization, lang=lang, batch_size=batch_size,
                                               silent=silent))
        else:
            metrics[method] = func(results=results, model_name=model_name, token=token,
                                   quantization=quantization, lang=lang, batch_size=batch_size,
                                   silent=silent)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated summaries")
    parser.add_argument("--input", required=True, help="Path to results JSON")
    parser.add_argument("--output", default=None, help="Path to save metrics JSON")
    parser.add_argument("--method", nargs="+", default=["all"], choices=list(EVAL_FUNCS_MAPPING.keys()),
                        help=f"evaluation methods. Default is all. Available: {list(EVAL_FUNCS_MAPPING.keys())}")
    parser.add_argument("--model", required=False, default='gpt-5-mini', help="Model to use")
    parser.add_argument("--quant", required=False, default='noquant',
                        help="model quantization level (e.g. noquant, 4bit, 8bit)")
    parser.add_argument("--token", required=False, default=os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN"),
                    help="api-key/token for model access. Defaults to OPENAI_API_KEY env var.")
    parser.add_argument("--batch_size", required=False, type=int, default=64, help="BERTScore batch size")
    parser.add_argument("--silent", required=False, type=str2bool, default=False, help="silent mode")
    args = parser.parse_args()

    print(f"Loading results from {args.input}...")
    results, dataset = load_results(args.input)
    print(f"Loaded {len(results)} samples from {dataset}")

    lang = 'en'
    print(f"Using language: {lang}")

    metrics = run_evaluation(*set(args.method), results=results, model_name=args.model, token=args.token,
                             quantization=args.quant, lang=lang, batch_size=args.batch_size,
                             silent=args.silent)

    output_path = args.output or args.input.replace('.json', '_metrics.json')
    with Path(output_path).open('w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()

