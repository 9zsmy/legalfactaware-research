# IN-Abs Phase 5E: SSE English structural-proxy adaptation

This is an independent, constrained transfer of the existing CAIL MemSum+SSE
reimplementation, not an official SSE reproduction. The official IN-Abs test
partition remains sealed.

## Frozen protocol

- Same 6,325/703 train/dev IDs and 400-sentence candidate cap as Phase 5.
- Five *contiguous equal-word* sections of each reference headnote are used as
  deterministic English structural proxies. These are not rhetorical-role
  annotations. No learned segmentation or test reference is used.
- For each section, assign the best matching source sentences by individual
  word-unigram-plus-bigram F1, with stable source-order ties and the existing
  SSE quotas `[1, 2, 1, 2, 1]`. This is a proxy for the unavailable author
  part-oracle, not the Phase 2 ordinary Oracle.
- Preserve existing MemSum local/global encoders, five-part SSE, shared decoder,
  paper-ratio loss, sequence-axis attention, and fixed CAIL-derived training
  settings (128-dimensional random train-only embeddings, 128 tokens/sentence,
  batch 4, 10 epochs, seed 20260814). Select checkpoint on proxy-label dev F1.
- Decode part by part with the same quotas, deduplicate, enforce complete source
  sentences under the per-reference spaCy word-count ceiling, and restore source
  order. Report unused budget and sentence-token truncation.
- Evaluate all 703 dev cases using the same `py-rouge==1.1` evaluator and 10,000
  paired bootstrap samples against E1; zero budget violations required.
- Report as `SSE-structural-proxy-IN-Abs`, never as an author checkpoint result.

## Limitations declared in advance

The English headnotes have no gold five-part structure, so ordered quintiles
replace the Chinese role rules. Fixed seven-sentence quotas may underuse the
larger English reference budget. Both deviations can depress performance and
preclude attributing differences to the SSE encoder alone.
