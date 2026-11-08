#!/usr/bin/env python3
"""Five-repeat, warm-start inference benchmark for the frozen CAIL models.

The model and training-only resources are loaded once.  Each measured repeat
starts by re-reading the untouched segmented development JSONL, rebuilds every
development-dependent representation, decodes summaries, and writes 942 JSONL
records.  Loading and warm-up are recorded separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def raw_sentences(row: dict) -> list[str]:
    text = row.get("text") or row.get("document") or []
    if isinstance(text, str):
        return [text]
    return [
        str(item.get("sentence") or item.get("text") or "")
        if isinstance(item, dict)
        else str(item)
        for item in text
    ]


def minimal_sse_row(row: dict, prep) -> dict:
    sentences = raw_sentences(row)
    return {
        "id": str(row.get("id") or row.get("case_id") or ""),
        "text": [prep.tokenize(sentence) for sentence in sentences],
        "raw_text": sentences,
        "global_labels": [0] * len(sentences),
        "raw_summary": str(row.get("summary") or row.get("reference_summary") or ""),
        "summary_parts": {},
        "part_names": list(prep.CAIL_PART_NAMES),
        "part_indices": [[] for _ in prep.CAIL_PART_NAMES],
        "part_labels": [[0] * len(sentences) for _ in prep.CAIL_PART_NAMES],
    }


def timed_repeat(method: str, repeat: int, out: Path, fn) -> dict:
    cuda_sync()
    started_wall = time.time()
    started = time.perf_counter()
    rows, stages = fn()
    cuda_sync()
    elapsed = time.perf_counter() - started
    destination = out / method / f"run_{repeat}" / "dev_predictions.jsonl"
    serialization_started = time.perf_counter()
    write_jsonl(destination, rows)
    serialization = time.perf_counter() - serialization_started
    elapsed += serialization
    if len(rows) != 948:
        raise RuntimeError(f"{method} repeat {repeat}: expected 948 predictions, got {len(rows)}")
    record = {
        "method": method,
        "repeat": repeat,
        "started_unix": started_wall,
        "elapsed_seconds": elapsed,
        "stages_seconds": {**stages, "serialization": serialization},
        "predictions": len(rows),
        "prediction_sha256": sha256(destination),
        "output": str(destination),
    }
    # Persist each completed repeat immediately.  A later shutdown must not
    # discard already measured wall-clock and stage timings held in memory.
    record_path = destination.parent / "run_record.json"
    temporary = record_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(record_path)
    return record


def run_lfa(args) -> tuple[float, float, list[dict]]:
    started = time.perf_counter()
    sys.path.insert(0, str(args.sse_scripts))
    sys.path.insert(0, str(args.code_dir))
    import run_legalfactaware_v2_cail_clean as base
    import run_legalfactaware_formal_ablation_dev as formal

    with (args.frozen / "legalfactaware/model.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    classifier, vectorizer = saved["classifier"], saved["vectorizer"]
    load_seconds = time.perf_counter() - started

    def infer_rows(source: Path, limit: int | None = None):
        t0 = time.perf_counter()
        rows = read_jsonl(source)
        if limit:
            rows = rows[:limit]
        read_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        matrices = [formal.matrix_for_sentences("full", vectorizer, base.sentences(row)) for row in rows]
        representation = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = [classifier.predict_proba(matrix)[:, 1].tolist() for matrix in matrices]
        scoring = time.perf_counter() - t0
        t0 = time.perf_counter()
        predictions, _ = formal.generate_variant(
            rows, scores, {"name": "LegalFactAware", "model": "oracle__full"}
        )
        decoding = time.perf_counter() - t0
        return predictions, {
            "read_preprocessing": read_seconds,
            "representation_encoding": representation,
            "scoring_forward": scoring,
            "constrained_decoding": decoding,
        }

    warm = time.perf_counter()
    infer_rows(args.train, 1)
    warmup_seconds = time.perf_counter() - warm
    reports = [timed_repeat("legalfactaware", i, args.output, lambda: infer_rows(args.dev)) for i in range(1, args.repeats + 1)]
    return load_seconds, warmup_seconds, reports


def load_sse_components(args):
    import numpy as np
    import torch

    sys.path.insert(0, str(args.sse_root / "src"))
    sys.path.insert(0, str(args.sse_scripts))
    import prepare_sse_cail as prep
    from legal_summarization.sse_extractive import decode_structured_sentences
    from train_sse_memsum import SSEDataset, build_model, resolve_device

    frozen = args.frozen / "sse"
    checkpoint = torch.load(frozen / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    with (frozen / "vocabulary.pkl").open("rb") as handle:
        vocabulary = pickle.load(handle)
    with (frozen / "random_embeddings_128d.pkl").open("rb") as handle:
        embeddings: np.ndarray = pickle.load(handle)
    device = resolve_device("cuda")
    model = build_model(
        embeddings,
        hidden_size=config["hidden_size"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        token_softmax_dim=config["token_softmax_dim"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return torch, prep, decode_structured_sentences, SSEDataset, vocabulary, config, model, device


def run_sse(args) -> tuple[float, float, list[dict]]:
    from torch.utils.data import DataLoader

    started = time.perf_counter()
    torch, prep, decode, Dataset, vocabulary, config, model, device = load_sse_components(args)
    load_seconds = time.perf_counter() - started

    def infer_rows(source: Path, limit: int | None = None):
        t0 = time.perf_counter()
        raw = read_jsonl(source)
        if limit:
            raw = raw[:limit]
        rows = [minimal_sse_row(row, prep) for row in raw]
        dataset = Dataset(
            rows,
            {word: index for index, word in enumerate(vocabulary)},
            config["max_doc_len"],
            config["max_seq_len"],
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate)
        preprocessing = time.perf_counter() - t0
        outputs, offset = [], 0
        scoring = decoding = 0.0
        with torch.no_grad():
            for token_ids, _ in loader:
                cuda_sync(); t0 = time.perf_counter()
                result = model(token_ids.to(device))
                probabilities = result["logits"].sigmoid().cpu()
                valid_lengths = result["valid_mask"].sum(dim=1).cpu().tolist()
                cuda_sync(); scoring += time.perf_counter() - t0
                t0 = time.perf_counter()
                for item_index, valid_length in enumerate(valid_lengths):
                    row = rows[offset + item_index]
                    summary, part_indices = decode(
                        probabilities[item_index, :, :valid_length],
                        row["raw_text"][:valid_length],
                        max_sentences=8,
                        max_chars=450,
                        output_order="source",
                    )
                    outputs.append({
                        "id": row["id"], "generated_summary": summary,
                        "part_indices": part_indices,
                        "reference_summary": row["raw_summary"],
                        "document": "".join(row["raw_text"]),
                    })
                decoding += time.perf_counter() - t0
                offset += len(valid_lengths)
        return outputs, {
            "read_preprocessing": preprocessing,
            "representation_encoding": 0.0,
            "scoring_forward": scoring,
            "constrained_decoding": decoding,
        }

    warm = time.perf_counter(); infer_rows(args.train, 1); warmup_seconds = time.perf_counter() - warm
    reports = [timed_repeat("sse", i, args.output, lambda: infer_rows(args.dev)) for i in range(1, args.repeats + 1)]
    return load_seconds, warmup_seconds, reports


def run_memsum(args) -> tuple[float, float, list[dict]]:
    started = time.perf_counter()
    sys.path.insert(0, str(args.sse_scripts))
    sys.path.insert(0, str(args.sse_root / "third_party/MemSum_acl2022/memsum_demo"))
    import prepare_sse_cail as prep
    from my_summarizers import ExtractiveSummarizer_MemSum_Final

    frozen = args.frozen / "memsum"
    summarizer = ExtractiveSummarizer_MemSum_Final(
        str(frozen / "model_best.pt"), str(frozen / "vocabulary.pkl"),
        gpu=0, embed_dim=128, hidden_dim=512, N_enc_l=2, N_enc_g=2, N_dec=2,
        max_doc_len=512, max_seq_len=128,
    )
    load_seconds = time.perf_counter() - started

    def infer_rows(source: Path, limit: int | None = None):
        t0 = time.perf_counter(); raw = read_jsonl(source)
        if limit: raw = raw[:limit]
        rows = [minimal_sse_row(row, prep) for row in raw]
        preprocessing = time.perf_counter() - t0
        outputs, combined = [], 0.0
        for row in rows:
            cuda_sync(); t0 = time.perf_counter()
            result = summarizer.extract(
                [row["text"]], p_stop_thres=0.6, ngram_blocking=True, ngram=4,
                return_sentence_position=True, max_extracted_sentences_per_document=7,
            )
            cuda_sync(); combined += time.perf_counter() - t0
            positions = result[1][0]
            chosen, total = [], 0
            for position in positions:
                sentence = row["raw_text"][position]
                if chosen and total + len(sentence) > 450: continue
                chosen.append(position); total += len(sentence)
            chosen.sort()
            summary = "".join(row["raw_text"][position] for position in chosen)[:450]
            outputs.append({
                "id": row["id"], "method": "MemSum-ACL2022-Chinese",
                "document": "\n".join(row["raw_text"]),
                "generated_summary": summary,
                "reference_summary": row["raw_summary"],
                "selected_positions": chosen, "summary_chars": len(summary),
            })
        return outputs, {
            "read_preprocessing": preprocessing,
            "representation_encoding": 0.0,
            "scoring_forward": combined,
            "constrained_decoding": 0.0,
        }

    warm = time.perf_counter(); infer_rows(args.train, 1); warmup_seconds = time.perf_counter() - warm
    reports = [timed_repeat("memsum", i, args.output, lambda: infer_rows(args.dev)) for i in range(1, args.repeats + 1)]
    return load_seconds, warmup_seconds, reports


def run_sende(args) -> tuple[float, float, list[dict]]:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    sys.path.insert(0, str(args.sende_root))
    from sende_model import SENDEExtractor
    import evaluate_cail as evaluate
    encoder = SentenceTransformer(str(args.frozen / "sende/tsdae"))
    checkpoint = torch.load(args.frozen / "sende/extractor_best.pt", map_location="cpu")
    extractor = SENDEExtractor(**checkpoint["config"])
    extractor.load_state_dict(checkpoint["model"])
    device = torch.device("cuda:0")
    extractor.to(device).eval()
    load_seconds = time.perf_counter() - started

    def build_records(source: Path, limit: int | None = None):
        raw = read_jsonl(source)
        if limit: raw = raw[:limit]
        records = []
        for index, row in enumerate(raw):
            sentences = ["".join(str(s).split()) for s in raw_sentences(row)]
            sentences = [s for s in sentences if s]
            records.append({
                "id": str(row.get("id", index)), "sentences": sentences,
                "reference_summary": "".join(str(row.get("summary", "")).split()),
            })
        return records

    def infer_rows(source: Path, limit: int | None = None):
        t0 = time.perf_counter(); records = build_records(source, limit); preprocessing = time.perf_counter() - t0
        all_sentences = [sentence for record in records for sentence in record["sentences"]]
        offsets = [0]
        for record in records: offsets.append(offsets[-1] + len(record["sentences"]))
        cuda_sync(); t0 = time.perf_counter()
        embeddings = encoder.encode(
            all_sentences, batch_size=256, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=False,
        ).astype(np.float32)
        cuda_sync(); encoding = time.perf_counter() - t0
        probabilities, scoring = [], 0.0
        with torch.no_grad():
            for doc in range(len(records)):
                begin, end = offsets[doc], offsets[doc + 1]
                x = torch.from_numpy(embeddings[begin:end]).unsqueeze(0).to(device)
                cuda_sync(); t0 = time.perf_counter()
                probabilities.append(torch.sigmoid(extractor(x))[0].cpu().numpy())
                cuda_sync(); scoring += time.perf_counter() - t0
        t0 = time.perf_counter(); outputs = []
        for row, scores in zip(records, probabilities):
            selected, prediction = evaluate.decode(row["sentences"], scores, 0.8)
            outputs.append({"id": row["id"], "selected_indices": selected, "generated_summary": prediction})
        decoding = time.perf_counter() - t0
        return outputs, {
            "read_preprocessing": preprocessing,
            "representation_encoding": encoding,
            "scoring_forward": scoring,
            "constrained_decoding": decoding,
        }

    warm = time.perf_counter(); infer_rows(args.train, 1); warmup_seconds = time.perf_counter() - warm
    reports = [timed_repeat("sende", i, args.output, lambda: infer_rows(args.dev)) for i in range(1, args.repeats + 1)]
    return load_seconds, warmup_seconds, reports


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def unlink_if_present(path: Path) -> None:
    """Path.unlink(missing_ok=...) is unavailable in TermDiffuSum's Python 3.7."""
    if path.is_symlink() or path.exists():
        path.unlink()


def run_termdiffusum(args) -> tuple[float, float, list[dict]]:
    import numpy as np
    import torch
    from transformers import AutoConfig

    started = time.perf_counter()
    sys.path.insert(0, str(args.term_root / "runtime"))
    term = import_file("term_runner", args.code_dir / "run_termdiffusum_cail_full.py")
    adapter = import_file("term_adapter", args.term_root / "prepare_termdiffusum_cail.py")
    original = AutoConfig.from_pretrained
    def local(name_or_path, *values, **options):
        if str(name_or_path) == "bert-base-uncased": name_or_path = str(args.bert_config)
        return original(name_or_path, *values, **options)
    AutoConfig.from_pretrained = staticmethod(local)
    random.seed(101); np.random.seed(101); torch.manual_seed(101); torch.cuda.manual_seed_all(101)
    device = torch.device("cuda:0")
    model, diffusion, _ = term.build_model(); model.to(device)
    checkpoint = torch.load(args.frozen / "termdiffusum/checkpoint_latest.pt", map_location=device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    load_seconds = time.perf_counter() - started
    link_dir = Path("/media/dxy/DiffuSum-main/datasets/legal_last40")
    originals = {name: os.readlink(link_dir / name) for name in ("train.jsonl", "val.jsonl", "test.jsonl")}

    def infer_rows(source: Path, limit: int | None = None):
        run_index = int(time.time_ns())
        adapted = args.output / "termdiffusum" / f"working_{run_index}"
        adapted.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        adapter.convert_file(source, adapted / "val.jsonl", limit or 0, 200)
        preprocessing = time.perf_counter() - t0
        for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            target = adapted / "val.jsonl"
            unlink_if_present(link_dir / name); (link_dir / name).symlink_to(target)
        cuda_sync(); t0 = time.perf_counter()
        report = term.decode_split(
            "dev", model, diffusion, device, adapted, adapted,
            6, 8, 450,
        )
        cuda_sync(); combined = time.perf_counter() - t0
        rows = read_jsonl(adapted / "dev_predictions.jsonl")
        return rows, {
            "read_preprocessing": preprocessing,
            "representation_encoding": combined,
            "scoring_forward": 0.0,
            "constrained_decoding": 0.0,
            "internal_decode_seconds": report["seconds"],
        }

    try:
        warm = time.perf_counter(); infer_rows(args.train, 1); warmup_seconds = time.perf_counter() - warm
        reports = [timed_repeat("termdiffusum", i, args.output, lambda: infer_rows(args.dev)) for i in range(1, args.repeats + 1)]
    finally:
        for name, target in originals.items():
            unlink_if_present(link_dir / name); (link_dir / name).symlink_to(target)
    return load_seconds, warmup_seconds, reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("legalfactaware", "sse", "memsum", "sende", "termdiffusum"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--sse-root", type=Path, default=Path("/root/legalfactaware_runs_20260813/sse_reimplementation"))
    parser.add_argument("--sende-root", type=Path, default=Path("/root/autodl-tmp/sende_independent_20260826"))
    parser.add_argument("--term-root", type=Path, default=Path("/root/autodl-tmp/termdiffusum_reproduction_20260817"))
    parser.add_argument("--bert-config", type=Path, default=Path("/root/autodl-tmp/fedsum_reproduction_20260826/models/bert-base-uncased"))
    args = parser.parse_args()
    args.sse_scripts = args.sse_root / "scripts"
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runner = {
        "legalfactaware": run_lfa, "sse": run_sse, "memsum": run_memsum,
        "sende": run_sende, "termdiffusum": run_termdiffusum,
    }[args.method]
    load_seconds, warmup_seconds, reports = runner(args)
    manifest = {
        "method": args.method, "status": "complete", "repeats": args.repeats,
        "model_load_seconds": load_seconds, "warmup_seconds": warmup_seconds,
        "runs": reports, "started_unix": started, "finished_unix": time.time(),
        "input_sha256": sha256(args.dev),
    }
    destination = args.output / args.method / "manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
