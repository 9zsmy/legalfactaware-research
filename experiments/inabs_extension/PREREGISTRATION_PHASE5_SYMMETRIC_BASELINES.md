# IN-Abs Phase 5: symmetric CAIL-baseline transfer

Purpose: run the same paper-derived methods used in the frozen CAIL comparison
on IN-Abs, so the thesis contains a symmetric method-by-dataset matrix.

Common protocol for extractive methods:

- Official IN-Abs training partition only, frozen seed-42 split of 6,325 train
  and 703 development documents; empty pairs 299 and 4799 excluded.
- Official 100-document test contents remain unread.
- Reference-free evenly spaced maximum of 400 source sentences.
- Per-document reference-summary word budget counted with official Gist spaCy
  `en_core_web_sm` punctuation-filtered tokens.
- Complete sentences only; skip candidates that do not fit; no hard truncation.
- `py-rouge==1.1`, document macro ROUGE-1/2/L F1, zero budget violations,
  10,000 paired bootstrap samples against E1.
- Adaptation identity must be reported for every method; no adapted result is
  described as an official author result.

Symmetric methods: SENDE, SSE, Agarwal-ST, Gist, TermDiffuSum, CivilSum
LED-tail, MemSum, EUR-Lex-Sum LexRank, and Budget Recipe. E1 is the English
LegalFactAware method. Gist is already complete under this protocol.

CivilSum and Budget Recipe are reported in a separate generative/zero-shot
block because complete-sentence extractive decoding is not their native task.

## LexRank configuration

- Word unigram TF-IDF sentence vectors.
- Cosine graph threshold 0.10, PageRank damping 0.85.
- Scores are ranked descending with source-position tie breaking and packed
  under the common complete-sentence budget.
