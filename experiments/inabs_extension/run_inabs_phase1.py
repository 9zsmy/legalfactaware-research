#!/usr/bin/env python3
"""IN-Abs train/dev audit and extractive ceiling evaluation.

The official test filenames are inventoried for split integrity, but phase 1
never reads test document or summary contents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

VENDOR = Path(__file__).resolve().parent / "vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SEED = 42
DEV_CASES = 703
EXPECTED_TRAIN = 7030
EXPECTED_VALID_TRAIN = 7028
EXPECTED_TEST = 100
EXPECTED_ARCHIVE_MD5 = "c77948fe580d26200f5c9fe53b81485d"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
ORACLE_CANDIDATE_CAP = 400
LEXRANK_THRESHOLD = 0.10
LEXRANK_DAMPING = 0.85
QUANTILES = (0.25, 0.50, 0.75)


def word_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def paired_ids(part: Path) -> list[str]:
    docs = {path.stem for path in (part / "judgement").glob("*.txt")}
    refs = {path.stem for path in (part / "summary").glob("*.txt")}
    if docs != refs:
        raise RuntimeError(
            f"document/reference mismatch in {part}: documents_only={sorted(docs-refs)[:10]}, "
            f"references_only={sorted(refs-docs)[:10]}"
        )
    return sorted(docs, key=lambda value: (int(value) if value.isdigit() else math.inf, value))


def eligible_train_ids(part: Path, case_ids: list[str]) -> tuple[list[str], list[dict]]:
    eligible, excluded = [], []
    for case_id in case_ids:
        document_path = part / "judgement" / f"{case_id}.txt"
        summary_path = part / "summary" / f"{case_id}.txt"
        reasons = []
        if not read_nonempty_lines(document_path):
            reasons.append("empty_document")
        if not read_nonempty_lines(summary_path):
            reasons.append("empty_summary")
        if reasons:
            excluded.append({"id": case_id, "reasons": reasons})
        else:
            eligible.append(case_id)
    return eligible, excluded


def fixed_split(train_ids: list[str], seed: int = SEED, dev_cases: int = DEV_CASES) -> tuple[list[str], list[str]]:
    ranked = sorted(train_ids, key=lambda case_id: (sha256_text(f"{seed}:{case_id}"), case_id))
    dev = sorted(ranked[:dev_cases], key=lambda x: (int(x) if x.isdigit() else math.inf, x))
    dev_set = set(dev)
    train = sorted(
        (case_id for case_id in train_ids if case_id not in dev_set),
        key=lambda x: (int(x) if x.isdigit() else math.inf, x),
    )
    return train, dev


def load_case(root: Path, case_id: str) -> dict:
    sentences = read_nonempty_lines(root / "judgement" / f"{case_id}.txt")
    reference_lines = read_nonempty_lines(root / "summary" / f"{case_id}.txt")
    if not sentences or not reference_lines:
        raise RuntimeError(f"empty document or summary: {case_id}")
    return {
        "id": case_id,
        "sentences": sentences,
        "reference": "\n".join(reference_lines),
        "reference_sentences": len(reference_lines),
        "reference_words": len(word_tokens(" ".join(reference_lines))),
        "source_words": sum(len(word_tokens(sentence)) for sentence in sentences),
    }


def nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        raise ValueError("empty values")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return int(ordered[rank - 1])


def distribution(values: list[int | float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(len(array)),
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[index:index+n]) for index in range(max(0, len(tokens) - n + 1)))


def overlap(counter: Counter, reference: Counter) -> int:
    return sum(min(value, reference.get(key, 0)) for key, value in counter.items())


def f1(overlap_count: int, candidate_total: int, reference_total: int) -> float:
    denominator = candidate_total + reference_total
    return 2.0 * overlap_count / denominator if denominator else 0.0


def add_state(current: Counter, current_overlap: int, addition: Counter, reference: Counter) -> tuple[Counter, int]:
    updated = current.copy()
    delta = 0
    for key, value in addition.items():
        old = updated.get(key, 0)
        new = old + value
        ref = reference.get(key, 0)
        delta += min(new, ref) - min(old, ref)
        updated[key] = new
    return updated, current_overlap + delta


def oracle_objective(overlap1: int, total1: int, ref_total1: int,
                     overlap2: int, total2: int, ref_total2: int) -> float:
    return f1(overlap1, total1, ref_total1) + f1(overlap2, total2, ref_total2)


def sentence_statistics(sentences: list[str], reference: str,
                        candidate_indices: list[int] | None = None) -> dict:
    reference_tokens = word_tokens(reference)
    ref1, ref2 = ngrams(reference_tokens, 1), ngrams(reference_tokens, 2)
    words = [word_tokens(sentence) for sentence in sentences]
    sent1 = [ngrams(tokens, 1) for tokens in words]
    sent2 = [ngrams(tokens, 2) for tokens in words]
    if candidate_indices is None:
        individual = [
            oracle_objective(overlap(one, ref1), sum(one.values()), sum(ref1.values()),
                             overlap(two, ref2), sum(two.values()), sum(ref2.values()))
            for one, two in zip(sent1, sent2)
        ]
        ranked = sorted(range(len(sentences)), key=lambda i: (individual[i], -i), reverse=True)
        candidates = ranked[:ORACLE_CANDIDATE_CAP]
        prefiltered = len(sentences) > ORACLE_CANDIDATE_CAP
    else:
        candidates = list(dict.fromkeys(candidate_indices))
        if any(index < 0 or index >= len(sentences) for index in candidates):
            raise ValueError("candidate index outside source document")
        prefiltered = len(candidates) < len(sentences)
    return {
        "words": words,
        "sent1": sent1,
        "sent2": sent2,
        "ref1": ref1,
        "ref2": ref2,
        "candidates": candidates,
        "prefiltered": prefiltered,
    }


def greedy_oracle_sequence(sentences: list[str], reference: str, budget: int | None,
                           stats: dict | None = None) -> tuple[list[int], float, bool]:
    stats = stats if stats is not None else sentence_statistics(sentences, reference)
    ref1, ref2 = stats["ref1"], stats["ref2"]
    current1, current2 = Counter(), Counter()
    overlap1 = overlap2 = total1 = total2 = used_words = 0
    current_score = 0.0
    remaining = set(stats["candidates"])
    selected: list[int] = []
    while remaining:
        best_index = None
        best_score = current_score
        best_payload = None
        for index in sorted(remaining):
            length = len(stats["words"][index])
            if not length or (budget is not None and used_words + length > budget):
                continue
            candidate1, candidate_overlap1 = add_state(current1, overlap1, stats["sent1"][index], ref1)
            candidate2, candidate_overlap2 = add_state(current2, overlap2, stats["sent2"][index], ref2)
            candidate_total1 = total1 + sum(stats["sent1"][index].values())
            candidate_total2 = total2 + sum(stats["sent2"][index].values())
            score = oracle_objective(
                candidate_overlap1, candidate_total1, sum(ref1.values()),
                candidate_overlap2, candidate_total2, sum(ref2.values()),
            )
            if score > best_score + 1e-12:
                best_index = index
                best_score = score
                best_payload = (candidate1, candidate2, candidate_overlap1,
                                candidate_overlap2, candidate_total1, candidate_total2, length)
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
        current1, current2, overlap1, overlap2, total1, total2, length = best_payload
        used_words += length
        current_score = best_score
    return selected, current_score, bool(stats["prefiltered"])


def pack_ranked(sentences: list[str], ranking: list[int], budget: int) -> list[int]:
    selected = []
    used = 0
    lengths = [len(word_tokens(sentence)) for sentence in sentences]
    for index in ranking:
        if lengths[index] and used + lengths[index] <= budget:
            selected.append(index)
            used += lengths[index]
    return sorted(selected)


def lead(sentences: list[str], budget: int) -> list[int]:
    return pack_ranked(sentences, list(range(len(sentences))), budget)


def lexrank_scores(sentences: list[str]) -> np.ndarray:
    if len(sentences) == 1:
        return np.ones(1, dtype=float)
    try:
        matrix = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), stop_words="english",
            min_df=1, sublinear_tf=True,
        ).fit_transform(sentences)
    except ValueError:
        return np.ones(len(sentences), dtype=float)
    similarity = cosine_similarity(matrix, dense_output=True)
    np.fill_diagonal(similarity, 0.0)
    adjacency = (similarity >= LEXRANK_THRESHOLD).astype(float) * similarity
    row_sums = adjacency.sum(axis=1)
    transition = np.divide(adjacency, row_sums[:, None], out=np.zeros_like(adjacency), where=row_sums[:, None] > 0)
    dangling = row_sums == 0
    if dangling.any():
        transition[dangling, :] = 1.0 / len(sentences)
    scores = np.full(len(sentences), 1.0 / len(sentences))
    teleport = (1.0 - LEXRANK_DAMPING) / len(sentences)
    for _ in range(100):
        updated = teleport + LEXRANK_DAMPING * transition.T.dot(scores)
        if np.abs(updated - scores).sum() < 1e-8:
            scores = updated
            break
        scores = updated
    return scores


def lexrank(sentences: list[str], budget: int) -> list[int]:
    scores = lexrank_scores(sentences)
    ranking = sorted(range(len(sentences)), key=lambda index: (float(scores[index]), -index), reverse=True)
    return pack_ranked(sentences, ranking, budget)


def summary(sentences: list[str], indices: list[int]) -> str:
    return "\n".join(sentences[index] for index in sorted(indices))


def evaluate_one(scorer, reference: str, prediction: str) -> tuple[float, float, float]:
    scores = scorer.score(reference, prediction)
    return tuple(float(scores[key].fmeasure) for key in ("rouge1", "rouge2", "rougeLsum"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="dev smoke-test limit; omitted means all 703")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    archive_md5 = md5(args.archive)
    if archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise RuntimeError(f"official archive MD5 mismatch: {archive_md5}")

    official_train = args.data / "train-data"
    official_test = args.data / "test-data"
    official_train_ids = paired_ids(official_train)
    test_ids = paired_ids(official_test)
    if len(official_train_ids) != EXPECTED_TRAIN or len(test_ids) != EXPECTED_TEST:
        raise RuntimeError(f"unexpected official split sizes: train={len(official_train_ids)} test={len(test_ids)}")
    train_ids, excluded_train = eligible_train_ids(official_train, official_train_ids)
    if len(train_ids) != EXPECTED_VALID_TRAIN:
        raise RuntimeError(
            f"unexpected eligible training cases: {len(train_ids)}; exclusions={excluded_train}"
        )
    model_train_ids, dev_ids = fixed_split(train_ids)
    split_manifest = {
        "seed": SEED,
        "split_rule": "703 smallest lexicographic SHA256('42:<case_id>') values from official train",
        "official_train_file_pair_count": len(official_train_ids),
        "eligible_official_train_count": len(train_ids),
        "excluded_official_train": excluded_train,
        "model_train_count": len(model_train_ids),
        "dev_count": len(dev_ids),
        "official_test_count": len(test_ids),
        "model_train_ids": model_train_ids,
        "dev_ids": dev_ids,
        "official_test_ids_inventory_only": test_ids,
        "test_contents_loaded": False,
    }
    write_json(args.output / "split_manifest.json", split_manifest)

    started = time.perf_counter()
    model_train = [load_case(official_train, case_id) for case_id in model_train_ids]
    budgets = {
        f"p{int(q*100)}": nearest_rank([row["reference_words"] for row in model_train], q)
        for q in QUANTILES
    }
    train_stats = {
        "source_words": distribution([row["source_words"] for row in model_train]),
        "source_sentences": distribution([len(row["sentences"]) for row in model_train]),
        "reference_words": distribution([row["reference_words"] for row in model_train]),
        "reference_sentences": distribution([row["reference_sentences"] for row in model_train]),
        "frozen_word_budgets_nearest_rank": budgets,
    }
    write_json(args.output / "train_only_length_stats.json", train_stats)
    del model_train

    selected_dev_ids = dev_ids[:args.limit] if args.limit else dev_ids
    dev = [load_case(official_train, case_id) for case_id in selected_dev_ids]
    dev_stats = {
        "smoke_limit": args.limit,
        "source_words": distribution([row["source_words"] for row in dev]),
        "source_sentences": distribution([len(row["sentences"]) for row in dev]),
        "reference_words": distribution([row["reference_words"] for row in dev]),
        "reference_sentences": distribution([row["reference_sentences"] for row in dev]),
    }
    write_json(args.output / "dev_length_stats.json", dev_stats)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    systems = ["lead", "lexrank", "posthoc_oracle", "budget_oracle"]
    metric_rows = {(budget_name, system): [] for budget_name in budgets for system in systems}
    records: list[dict] = []
    prefiltered_cases = 0
    for number, row in enumerate(dev, 1):
        sentences, reference = row["sentences"], row["reference"]
        oracle_stats = sentence_statistics(sentences, reference)
        unconstrained_sequence, _, prefiltered = greedy_oracle_sequence(
            sentences, reference, None, stats=oracle_stats
        )
        lexrank_values = lexrank_scores(sentences)
        lexrank_ranking = sorted(
            range(len(sentences)),
            key=lambda index: (float(lexrank_values[index]), -index),
            reverse=True,
        )
        prefiltered_cases += int(prefiltered)
        for budget_name, budget in budgets.items():
            choices = {
                "lead": lead(sentences, budget),
                "lexrank": pack_ranked(sentences, lexrank_ranking, budget),
                "posthoc_oracle": pack_ranked(sentences, unconstrained_sequence, budget),
            }
            budget_sequence, _, budget_prefiltered = greedy_oracle_sequence(
                sentences, reference, budget, stats=oracle_stats
            )
            if budget_prefiltered != prefiltered:
                raise RuntimeError("Oracle prefilter inconsistency")
            choices["budget_oracle"] = sorted(budget_sequence)
            for system, indices in choices.items():
                prediction = summary(sentences, indices)
                words = len(word_tokens(prediction))
                if words > budget:
                    raise RuntimeError(f"budget violation: {row['id']} {system} {words}>{budget}")
                r1, r2, rl = evaluate_one(scorer, reference, prediction)
                result = {
                    "id": row["id"], "budget_name": budget_name, "budget_words": budget,
                    "system": system, "selected_indices": indices,
                    "prediction_words": words, "prediction_sentences": len(indices),
                    "rouge1": r1, "rouge2": r2, "rougeLsum": rl,
                    "oracle_prefiltered": prefiltered if "oracle" in system else None,
                }
                records.append(result)
                metric_rows[(budget_name, system)].append(result)
        if number % 10 == 0 or number == len(dev):
            print(json.dumps({"phase": "dev", "completed": number, "total": len(dev),
                              "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)

    write_jsonl(args.output / "dev_per_case.jsonl", records)
    metrics = {
        "experiment": "LegalFactAware IN-Abs phase-1 dev extractive ceiling",
        "archive_md5": archive_md5,
        "test_contents_loaded": False,
        "dev_cases": len(dev),
        "smoke_limit": args.limit,
        "budgets": budgets,
        "oracle_candidate_cap": ORACLE_CANDIDATE_CAP,
        "oracle_prefiltered_cases": prefiltered_cases,
        "oracle_prefiltered_fraction": prefiltered_cases / len(dev),
        "systems": {},
    }
    for budget_name in budgets:
        metrics["systems"][budget_name] = {}
        for system in systems:
            rows = metric_rows[(budget_name, system)]
            metrics["systems"][budget_name][system] = {
                "n": len(rows),
                "rouge1": mean(row["rouge1"] for row in rows),
                "rouge2": mean(row["rouge2"] for row in rows),
                "rougeLsum": mean(row["rougeLsum"] for row in rows),
                "mean_prediction_words": mean(row["prediction_words"] for row in rows),
                "mean_prediction_sentences": mean(row["prediction_sentences"] for row in rows),
                "budget_violations": sum(row["prediction_words"] > row["budget_words"] for row in rows),
            }
        posthoc = metric_rows[(budget_name, "posthoc_oracle")]
        aware = metric_rows[(budget_name, "budget_oracle")]
        metrics["systems"][budget_name]["budget_oracle_minus_posthoc"] = {
            key: mean(right[key] - left[key] for left, right in zip(posthoc, aware))
            for key in ("rouge1", "rouge2", "rougeLsum")
        }
    median_oracle = metrics["systems"]["p50"]["budget_oracle"]
    metrics["feasibility_gate"] = {
        "threshold_rouge1": 0.40,
        "threshold_rouge2": 0.20,
        "passed": median_oracle["rouge1"] >= 0.40 and median_oracle["rouge2"] >= 0.20,
    }
    metrics["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.output / "phase1_dev_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
