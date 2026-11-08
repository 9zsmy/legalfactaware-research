"""
Evaluate length-controlled summarization on CNN/DailyMail, arXiv, and XSum.
Supports both LLaDA-8B and Dream-7B models.
"""

import torch
import argparse
import json
import time
import numpy as np
from datasets import load_dataset
from dataclasses import dataclass, asdict
from tqdm import tqdm

from diffusion_lm import load_model, generate_length_controlled


MODEL_CONFIGS = {
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
    "dream": "Dream-org/Dream-v0-Instruct-7B",
}

DATASET_CONFIGS = {
    "cnndm": {
        "hf_name": "cnn_dailymail",
        "hf_config": "3.0.0",
        "source_field": "article",
        "target_field": "highlights",
        "prompt_template": "Summarize the following article in approximately {target_len} characters:\n\n{source}",
        "lang": "en",
        "max_steps": 150,
        "init_tokens": 150,
    },
    "arxiv": {
        "hf_name": "ccdv/arxiv-summarization",
        "hf_config": None,
        "source_field": "article",
        "target_field": "abstract",
        "prompt_template": "Write an abstract for the following scientific paper in approximately {target_len} characters:\n\n{source}",
        "lang": "en",
        "max_steps": 500,
        "init_tokens": 500,
    },
    "xsum": {
        "hf_name": "EdinburghNLP/xsum",
        "hf_config": None,
        "source_field": "document",
        "target_field": "summary",
        "prompt_template": "Write a one-sentence summary of the following article in approximately {target_len} characters:\n\n{source}",
        "lang": "en",
        "max_steps": 100,
        "init_tokens": 100,
    },
}

CONFIDENCE_METRICS = ["margin", "entropy", "probability", "semantic", "semantic_variance"]
DECODER_MODES = ["delta", "fixed_buffer"]
BUFFER_SIZE_MODES = ["avg_chars_per_token", "manual", "target_over_2"]


@dataclass
class EvalResult:
    sample_id: int
    source_len: int
    target_len: int
    generated_len: int
    len_error: int
    steps: int
    remasks: int
    time_sec: float
    source_text: str
    target_text: str
    generated_text: str


def load_existing_results(output_path: str) -> list:
    """Load existing results for resume support."""
    try:
        with open(output_path) as f:
            data = json.load(f)
            return data.get("results", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def load_dataset_samples(
    dataset_name: str,
    split: str,
    num_samples: int,
    max_source_len: int = 8192,
):
    """Load samples from any supported dataset."""
    config = DATASET_CONFIGS[dataset_name]

    if config["hf_config"]:
        dataset = load_dataset(config["hf_name"], config["hf_config"], split=split)
    else:
        dataset = load_dataset(config["hf_name"], split=split)

    dataset = dataset.select(range(min(num_samples, len(dataset))))

    samples = []
    for idx, item in enumerate(dataset):
        source = item[config["source_field"]]
        target = item[config["target_field"]]
        if max_source_len and len(source) > max_source_len:
            source = source[:max_source_len]
        if dataset_name == "arxiv":
            if len(source) <= 100 or len(target) <= 50:
                continue
        samples.append({"id": idx, "source": source, "target": target})
    return samples


def estimate_avg_chars_per_token_from_references(
    dlm,
    dataset_name: str,
    split: str = "validation",
    num_samples: int = 1000,
    max_source_len: int = 8192,
) -> float:
    """
    Estimate chars/token from reference summaries on a non-test split.

    This is used to initialize a fixed token buffer from a character target:
        buffer_size approximately target_char_length / avg_chars_per_token

    It uses validation reference summaries, not the source document and not
    the per-sample test reference token count.
    """
    print(
        f"\nEstimating avg chars/token from {dataset_name} "
        f"{split} references, n={num_samples}..."
    )

    val_samples = load_dataset_samples(
        dataset_name=dataset_name,
        split=split,
        num_samples=num_samples,
        max_source_len=max_source_len,
    )

    ratios = []
    for sample in val_samples:
        text = sample["target"]
        if not text:
            continue

        token_ids = dlm.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        n_tokens = len(token_ids)
        if n_tokens > 0:
            ratios.append(len(text) / n_tokens)

    if not ratios:
        raise ValueError(
            f"Could not estimate avg chars/token for {dataset_name} from split={split}"
        )

    avg = float(np.mean(ratios))
    med = float(np.median(ratios))
    std = float(np.std(ratios))

    print(
        f"Estimated chars/token from validation references: "
        f"mean={avg:.4f}, median={med:.4f}, std={std:.4f}, n={len(ratios)}"
    )

    return avg


def evaluate(
    dlm,
    dataset_name: str,
    samples: list,
    output_path: str,
    confidence_threshold: float = 1.0,
    remask_threshold: float = 0.0,
    low_confidence_threshold: float = 1.0,
    confidence_metric: str = "margin",
    decoder_mode: str = "delta",
    buffer_size_mode: str = "avg_chars_per_token",
    avg_chars_per_token: float = None,
    debug: bool = False,
    resume: bool = True,
):
    config = DATASET_CONFIGS[dataset_name]
    prompt_template = config["prompt_template"]

    results = []
    start_idx = 0
    if resume:
        existing = load_existing_results(output_path)
        if existing:
            results = [EvalResult(**r) for r in existing]
            start_idx = len(results)
            print(f"Resuming from sample {start_idx} (loaded {len(results)} existing results)")

    if start_idx >= len(samples):
        print("All samples already processed!")
        _print_stats(results, dataset_name)
        return results

    print(f"\nEvaluating samples {start_idx} to {len(samples) - 1}...")
    print(
        f"Hyperparameters: confidence_threshold={confidence_threshold}, "
        f"remask_threshold={remask_threshold}, "
        f"low_confidence_threshold={low_confidence_threshold}, "
        f"confidence_metric={confidence_metric}, decoder_mode={decoder_mode}, "
        f"buffer_size_mode={buffer_size_mode}, avg_chars_per_token={avg_chars_per_token}"
    )

    for idx in tqdm(
        range(start_idx, len(samples)),
        desc="Processing samples",
        initial=start_idx,
        total=len(samples),
    ):
        sample = samples[idx]
        source = sample["source"]
        target = sample["target"]

        source_len = len(source)
        target_len = len(target)

        if debug:
            print(f"\n{'=' * 80}")
            print(f"Sample {idx + 1}/{len(samples)}")
            print(f"Source length: {source_len} chars")
            print(f"Target length: {target_len} chars")
            print(f"Source preview: {source[:100]}...")
            print(f"Target preview: {target[:200]}...")
            print(f"{'=' * 80}")

        prompt_text = prompt_template.format(source=source, target_len=target_len)
        prompt_ids = dlm.encode_prompt(prompt_text)
        gen_start = prompt_ids.shape[1]

        try:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()

            output, steps, remasks = generate_length_controlled(
                dlm=dlm,
                prompt=prompt_ids,
                target_char_length=target_len,
                max_iterations=config["max_steps"],
                confidence_threshold=confidence_threshold,
                remask_threshold=remask_threshold,
                low_confidence_threshold=low_confidence_threshold,
                confidence_metric=confidence_metric,
                initial_gen_length=config["init_tokens"],
                enable_length_adjustment=(decoder_mode == "delta"),
                avg_chars_per_token=avg_chars_per_token,
                buffer_size_mode=buffer_size_mode,
                debug=debug,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            generated_text = dlm.decode(output[0, gen_start:]).strip()
            generated_len = len(generated_text)
            len_error = abs(generated_len - target_len)

            result = EvalResult(
                sample_id=idx,
                source_len=source_len,
                target_len=target_len,
                generated_len=generated_len,
                len_error=len_error,
                steps=steps,
                remasks=remasks,
                time_sec=elapsed,
                source_text=source,
                target_text=target,
                generated_text=generated_text,
            )

            results.append(result)

            if debug:
                print(f"Generated ({generated_len} chars): {generated_text[:200]}...")
                print(
                    f"Length error: {len_error} chars, Steps: {steps}, "
                    f"Remasks: {remasks}, Time: {elapsed:.2f}s"
                )

            if (idx + 1) % 50 == 0:
                _save_results(
                    results,
                    output_path,
                    dataset_name,
                    confidence_threshold,
                    remask_threshold,
                    low_confidence_threshold,
                    confidence_metric,
                    decoder_mode,
                    buffer_size_mode,
                    avg_chars_per_token,
                )

        except Exception as e:
            print(f"\nERROR on sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    _save_results(
        results,
        output_path,
        dataset_name,
        confidence_threshold,
        remask_threshold,
        low_confidence_threshold,
        confidence_metric,
        decoder_mode,
        buffer_size_mode,
        avg_chars_per_token,
    )
    _print_stats(results, dataset_name)
    return results


def _save_results(
    results,
    output_path,
    dataset_name,
    confidence_threshold,
    remask_threshold,
    low_confidence_threshold,
    confidence_metric,
    decoder_mode,
    buffer_size_mode,
    avg_chars_per_token,
):
    if not results:
        return

    len_errors = [r.len_error for r in results]
    steps_list = [r.steps for r in results]
    remasks_list = [r.remasks for r in results]
    times = [r.time_sec for r in results]

    stats = {
        "num_samples": len(results),
        "avg_len_error": float(np.mean(len_errors)),
        "median_len_error": float(np.median(len_errors)),
        "std_len_error": float(np.std(len_errors)),
        "samples_under_10chars": int(sum(1 for e in len_errors if e < 10)),
        "samples_under_25chars": int(sum(1 for e in len_errors if e < 25)),
        "samples_under_50chars": int(sum(1 for e in len_errors if e < 50)),
        "avg_steps": float(np.mean(steps_list)),
        "avg_remasks": float(np.mean(remasks_list)),
        "avg_time_sec": float(np.mean(times)),
        "std_time_sec": float(np.std(times)),
    }

    output = {
        "dataset": dataset_name,
        "decoder_mode": decoder_mode,
        "buffer_size_mode": buffer_size_mode,
        "avg_chars_per_token": avg_chars_per_token,
        "hyperparameters": {
            "confidence_threshold": confidence_threshold,
            "remask_threshold": remask_threshold,
            "low_confidence_threshold": low_confidence_threshold,
            "confidence_metric": confidence_metric,
            "decoder_mode": decoder_mode,
            "buffer_size_mode": buffer_size_mode,
            "avg_chars_per_token": avg_chars_per_token,
        },
        "statistics": stats,
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def _print_stats(results, dataset_name):
    if not results:
        print("No results to report")
        return

    len_errors = [r.len_error for r in results]
    steps_list = [r.steps for r in results]
    remasks_list = [r.remasks for r in results]
    times = [r.time_sec for r in results]

    print(f"\n{'=' * 80}")
    print(f"{dataset_name.upper()} EVALUATION RESULTS")
    print(f"{'=' * 80}")
    print(f"Samples evaluated: {len(results)}")
    print(f"Mean absolute error: {np.mean(len_errors):.1f} chars")
    print(f"Median absolute error: {np.median(len_errors):.1f} chars")
    print(f"Std dev length error: {np.std(len_errors):.1f} chars")
    print(f"Samples <10 chars error: {sum(1 for e in len_errors if e < 10)}/{len(results)}")
    print(f"Samples <25 chars error: {sum(1 for e in len_errors if e < 25)}/{len(results)}")
    print(f"Samples <50 chars error: {sum(1 for e in len_errors if e < 50)}/{len(results)}")
    print(f"Average steps: {np.mean(steps_list):.1f}")
    print(f"Average remasks: {np.mean(remasks_list):.1f}")
    print(f"Average time: {np.mean(times):.2f}s per sample")
    print(f"Std time: {np.std(times):.2f}s")


def build_output_path(args, avg_chars_per_token):
    if args.output:
        return args.output

    conf_str = f"conf{args.confidence_threshold:.2f}".replace(".", "")
    remask_str = f"remask{args.remask_threshold:.3f}".replace(".", "")
    lowconf_str = f"lowconf{args.low_confidence_threshold:.2f}".replace(".", "")
    metric_str = args.confidence_metric
    mode_str = args.decoder_mode
    buffer_str = f"buf{args.buffer_size_mode}"

    if avg_chars_per_token is not None:
        avg_tok_str = f"avg{avg_chars_per_token:.3f}".replace(".", "")
    else:
        avg_tok_str = "avgnone"

    return (
        f"{args.dataset}_{args.model}_{mode_str}_{metric_str}_"
        f"{conf_str}_{remask_str}_{lowconf_str}_{buffer_str}_{avg_tok_str}.json"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate length-controlled summarization")
    parser.add_argument("--dataset", required=True, choices=["cnndm", "arxiv", "xsum"], help="Dataset to evaluate on")
    parser.add_argument("--model", default="llada", choices=["llada", "dream"], help="Model to use")
    parser.add_argument("--model-id", default=None, help="Override model ID")
    parser.add_argument("--split", default="test", help="Dataset split for HF datasets")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument("--max-source-len", type=int, default=8192, help="Max source length in chars")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore existing results")
    parser.add_argument("--confidence-threshold", type=float, default=1.0, help="Threshold for unmasking tokens")
    parser.add_argument("--remask-threshold", type=float, default=0.0, help="Threshold below which fixed tokens get remasked, 0 to disable")
    parser.add_argument("--low-confidence-threshold", type=float, default=1.0, help="Threshold below which mask tokens can be removed for length control")
    parser.add_argument("--confidence-metric", type=str, default="probability", choices=CONFIDENCE_METRICS, help="Confidence metric to use")
    parser.add_argument("--decoder-mode", type=str, default="delta", choices=DECODER_MODES, help="delta uses dynamic buffer adjustment; fixed_buffer disables insert/remove")
    parser.add_argument("--buffer-size-mode", type=str, default="avg_chars_per_token", choices=BUFFER_SIZE_MODES, help="How to choose the initial masked output buffer size")
    parser.add_argument("--avg-chars-per-token", type=float, default=None, help="Manually provided chars/token estimate for buffer sizing")
    parser.add_argument("--estimate-avg-chars-per-token", action="store_true", help="Estimate chars/token from validation reference summaries")
    parser.add_argument("--avg-token-estimate-split", type=str, default="validation", help="Split used to estimate chars/token")
    parser.add_argument("--avg-token-estimate-samples", type=int, default=1000, help="Number of validation references used to estimate chars/token")
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit"], default=None, help="Quantization mode")

    args = parser.parse_args()
    model_id = args.model_id or MODEL_CONFIGS[args.model]

    print(f"Dataset: {args.dataset}")
    print(f"Model: {model_id}")
    print(f"Split: {args.split}")
    print(f"Samples: {args.num_samples}")
    print(f"Confidence metric: {args.confidence_metric}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"Remask threshold: {args.remask_threshold}")
    print(f"Low confidence threshold: {args.low_confidence_threshold}")
    print(f"Decoder mode: {args.decoder_mode}")
    print(f"Buffer size mode: {args.buffer_size_mode}")
    print(f"Quantization: {args.quantization or 'None'}")

    print(f"\nLoading {args.dataset} dataset...")
    samples = load_dataset_samples(
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        max_source_len=args.max_source_len,
    )
    print(f"Loaded {len(samples)} samples")

    print(f"\nLoading model: {model_id}")
    dlm = load_model(model_id, quantization=args.quantization)

    print("Building token length vector...")
    _ = dlm.token_lengths

    avg_chars_per_token = args.avg_chars_per_token
    if args.estimate_avg_chars_per_token:
        avg_chars_per_token = estimate_avg_chars_per_token_from_references(
            dlm=dlm,
            dataset_name=args.dataset,
            split=args.avg_token_estimate_split,
            num_samples=args.avg_token_estimate_samples,
            max_source_len=args.max_source_len,
        )

    if args.buffer_size_mode == "avg_chars_per_token" and avg_chars_per_token is None:
        raise ValueError(
            "--buffer-size-mode avg_chars_per_token requires either "
            "--avg-chars-per-token VALUE or --estimate-avg-chars-per-token"
        )

    output_path = build_output_path(args, avg_chars_per_token)

    print(f"Avg chars/token: {avg_chars_per_token}")
    print(f"Output: {output_path}")

    evaluate(
        dlm=dlm,
        dataset_name=args.dataset,
        samples=samples,
        output_path=output_path,
        confidence_threshold=args.confidence_threshold,
        remask_threshold=args.remask_threshold,
        low_confidence_threshold=args.low_confidence_threshold,
        confidence_metric=args.confidence_metric,
        decoder_mode=args.decoder_mode,
        buffer_size_mode=args.buffer_size_mode,
        avg_chars_per_token=avg_chars_per_token,
        debug=args.debug,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

