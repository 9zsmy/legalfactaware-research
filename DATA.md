# Data and artifact policy

This repository stores code, experiment protocols, aggregate metrics, and small audit manifests. It does **not** store restricted or reconstructable research inputs.

## Excluded from Git

- CAIL, IN-Abs, UK-Abs, XSum, LCSTS, and other datasets;
- raw judgments, reference summaries, one-shot examples derived from private corpora;
- per-document predictions and traces containing document text;
- model weights, checkpoints, embedding arrays, vectorizers, and pickle/joblib bundles;
- Hugging Face, PyTorch, spaCy, NLTK, and package caches;
- virtual environments, logs, temporary files, and remote-server credentials.

## Expected private layout

The code does not require a single hard-coded directory. A local operator may use a layout such as:

```text
/private/research-data/
├── cail/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── inabs/
│   └── dataset/IN-Abs/
├── models/
└── outputs/
```

Pass those paths explicitly to scripts. Never replace integrity checks with silent fallback data discovery.

## Aggregate results retained here

Small JSON/CSV files that contain only aggregate metrics, hashes, counts, or run configuration may be committed. Files containing source text, reference text, generated summaries, candidate sentences, or document-level traces must remain private even when small.

## Recreating a frozen run

1. Obtain the dataset and model assets through their original authorized channels.
2. Verify split identity and the hashes recorded in the relevant preregistration or manifest.
3. Install the matching requirement set.
4. run the entry point with explicit paths and a new output directory.
5. Compare aggregate metrics and integrity counts before interpreting model-quality differences.

