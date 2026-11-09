# LegalFactAware Research Repository

Public research-artifact repository for **LegalFactAware**, a fixed-budget extractive legal summarization system using summary-level membership supervision, a static sentence scorer, and redundancy-aware complete-candidate decoding.

> Repository status: public research artifact. The code, experiment identities, and aggregate results are preserved; datasets, model weights, credentials, and raw legal-document predictions are intentionally excluded.

## Repository map

```text
legalfactaware-research/
├── src/
│   ├── chinese_core/             # CAIL LegalFactAware evolution and final ablations
│   └── english_inabs/            # Frozen IN-Abs E1/Oracle/decode experiments
├── evaluation/                   # ROUGE, expression metrics, protocol audits
├── baselines/validation/         # Baseline preparation, training, decoding, audits
├── experiments/
│   ├── reviewer_controls/        # Matched-count, feature split, K sensitivity, drop-only
│   ├── matched_oracle_control/   # Oracle-budget matched control
│   ├── feasible_selector_control/# Greedy fit/skip vs exact knapsack
│   ├── cost/                     # Complete-pipeline and unified-inference timing
│   ├── runtime/                  # Runtime benchmark entry points
│   ├── recent_baselines/         # DELTA, LCG, Zero-Shot LCS adaptations
│   └── inabs_extension/          # Expanded English development experiments
├── archive/unverified_baseline_audits/
│                                   # Clearly labeled non-primary audit work
├── docs/
│   ├── provenance/               # Preregistrations, frozen manifests, handoffs
│   └── results/                  # Paper-facing evidence summaries
├── paper/                        # Current LaTeX source and editable figures
├── requirements/                 # Environment-specific dependency lists
└── tools/                        # Inventory, reconstruction, and repository audits
```

## What is authoritative

The Chinese primary method is the **combined v2.5-B pipeline**, not any single historical script:

```text
budget-consistent Oracle membership labels
    -> character 2-4 gram TF-IDF + Legal14 + Section6
    -> class-balanced Logistic Regression
    -> character-bigram redundancy control
    -> complete-candidate fit/skip decoding
    -> source-order restoration
```

Use these files to understand the final composition:

- `src/chinese_core/run_legalfactaware_v22_oracle_supervision_dev.py`
- `src/chinese_core/run_legalfactaware_v25_structure_feature_dev.py`
- `src/chinese_core/run_legalfactaware_v21_budget_decoder_dev.py`
- `src/chinese_core/run_legalfactaware_formal_ablation_dev.py`

`run_legalfactaware_v2_cail_clean.py` is a historical base version and still contains tail truncation; it must not be described as the final decoder by itself.

## Quick checks

Python 3.10 is recommended.

```bash
make syntax
make test-recent
make audit
```

`make test-recent` runs the dependency-light tests for the recent length-control adaptations. The frozen 942-document test is skipped when the private data bundle is absent.

For core CPU experiments:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/core.txt
```

For recent GPU baselines:

```bash
python -m pip install -r requirements/recent-gpu.txt
```

## Data and model setup

See [DATA.md](DATA.md). All commands require explicit local paths. Do not commit CAIL, IN-Abs, UK-Abs, XSum, checkpoints, embeddings, Hugging Face caches, or reconstructed raw predictions.

Example reviewer-control run:

```bash
python experiments/reviewer_controls/run_reviewer_controls.py \
  --train /private/path/to/cail/train.jsonl \
  --dev /private/path/to/cail/dev.jsonl \
  --output-dir /private/path/to/output
```

The formal scripts contain frozen-input integrity checks. Do not remove those checks to make mismatched data run.

## Result identities

- CAIL test (948 documents): LegalFactAware v2.5-B = 60.94/42.10/50.73 ROUGE and 70.99 legacy Fact Recall.
- Reviewer controls: CAIL development set (942 documents).
- DELTA full adaptation: CAIL development set (942 documents), official decoder core with Dream 7B 4-bit.
- LCG and Zero-Shot LCS: frozen 100-document resource-limited subset, independent CAIL adaptations.
- IN-Abs results are development-set evidence; the official 100-document test remains outside the current effectiveness claims.

Detailed claim boundaries are in `docs/results/PAPER_AND_EXPERIMENT_MASTER_CN.md` and `experiments/recent_baselines/REPRODUCTION_AUDIT.md`.

## Reproducibility and privacy

- Preregistrations and aggregate metrics are versioned.
- Raw judgments, references, predictions, model binaries, caches, and credentials are excluded.
- Vendored DELTA source retains its upstream license and source manifest.
- The repository has no open-source grant. See [LICENSE.md](LICENSE.md).
- Before every push, run `make audit`.

## Paper source

`paper/main.tex` is the current pre-revision manuscript snapshot. It intentionally preserves the state that still contains the earlier SENDE 0.8 quality numbers. Use `docs/results/PAPER_AND_EXPERIMENT_MASTER_CN.md` when revising it; do not treat every number in the snapshot as the latest result.

