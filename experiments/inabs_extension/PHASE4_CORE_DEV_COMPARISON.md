# IN-Abs unified core development comparison

Protocol: frozen 6,325/703 split, reference-specific spaCy word budgets,
complete-sentence decoding, zero budget violations, and `py-rouge==1.1`.
The official 100-document test set was not read.

| Method | R1 | R2 | RL | Budget violations |
|---|---:|---:|---:|---:|
| SummaRuNNer split-safe adaptation | **67.73** | **40.73** | **64.46** | 0 |
| E1 | 66.42 | 38.15 | 62.95 | 0 |
| DSDR budget adaptation | 64.75 | 34.17 | 61.05 | 0 |
| EUR-Lex-Sum LexRank IN-Abs adaptation | 63.50 | 34.48 | 59.93 | 0 |
| Gist split-safe adaptation | 63.38 | 33.74 | 59.36 | 0 |
| CaseSummarizer restricted adaptation | 61.79 | 33.33 | 58.73 | 0 |
| MemSum P50-budget adaptation | 59.18 | 30.23 | 55.09 | 0 |

## Paired result: SummaRuNNer minus E1

| Metric | Mean difference (points) | Paired 95% bootstrap CI |
|---|---:|---:|
| R1 | +1.30 | [+0.97, +1.65] |
| R2 | +2.58 | [+2.07, +3.08] |
| RL | +1.51 | [+1.16, +1.86] |

The confidence intervals exclude zero on all three metrics. Therefore E1
significantly beats DSDR, CaseSummarizer, Gist, the transferred LexRank, and
the MemSum P50-budget adaptation,
but significantly loses to SummaRuNNer on the frozen development set. The preregistered A-level condition
is not met, so the official test remains sealed.

These are same-development-set implementation-adaptation results. They should
not be presented as bit-identical reproductions of the published checkpoints.
