# LegalFactAware IN-Abs extension: phase-1 preregistration

Frozen on 2026-08-31 before any IN-Abs development or test result was used.

## Scope

This phase is a development-set feasibility gate. It measures English length
distributions and the extractive ceiling before any full LegalFactAware model
is trained. The official 100-case IN-Abs test set remains sealed.

## Data and split

- Source: Shukla et al. (2022) official Zenodo record 7152317, `dataset.zip`.
- Required archive MD5: `c77948fe580d26200f5c9fe53b81485d`.
- Official train: 7,030 filename pairs. Two official summaries (`299.txt` and
  `4799.txt`) are zero-byte files and are excluded before splitting, leaving
  7,028 eligible pairs. This exclusion was discovered by the first end-to-end
  smoke run and recorded before any metric was produced.
- Official test: 100 pairs; filenames may be inventoried, but contents are not
  loaded by the phase-1 program.
- Development set: 703 eligible official-train cases with the smallest lexicographic
  SHA-256 value of `42:<case_id>`.
- Model-training pool: the remaining 6,325 eligible official-train cases.

## Budget protocol

Whitespace-independent English word tokens are matched with
`[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*`.

Three word budgets are the nearest-rank P25, P50 and P75 of reference-summary
word counts in the 6,327-case model-training pool. No test statistic is used.
All outputs contain complete official source lines and are restored to source
order. There is no arbitrary eight-sentence cap in the English experiment.

## Phase-1 systems

1. `lead`: take source sentences in order while they fit the word budget.
2. `lexrank`: word 1-2 gram TF-IDF, English stop words, cosine threshold 0.10,
   PageRank damping 0.85, then complete-sentence budget packing.
3. `posthoc_oracle`: greedy ROUGE-1 F1 + ROUGE-2 F1 sequence without a word
   budget, followed by complete-sentence packing at each frozen budget.
4. `budget_oracle`: greedy marginal ROUGE-1 F1 + ROUGE-2 F1 in which every
   candidate must fit the remaining word budget.

To bound runtime on unusually long judgments, Oracle search keeps all source
sentences when there are at most 400 and otherwise keeps the 400 sentences
with highest individual ROUGE-1+2 objective. This approximation is applied to
both Oracle variants, is reported per case, and must be disclosed as an
approximate greedy extractive ceiling.

## Metrics and gate

- ROUGE-1/2/Lsum F1 with Porter stemming (`rouge-score`).
- Mean output words and sentences; budget violation count.
- Fraction of cases affected by the 400-candidate Oracle prefilter.
- `budget_oracle - posthoc_oracle` paired differences.

Proceed to model training only if the median-budget `budget_oracle` reaches at
least ROUGE-1 F1 0.40 and ROUGE-2 F1 0.20 on dev. These are feasibility, not
superiority, thresholds.
