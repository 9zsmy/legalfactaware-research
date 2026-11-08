# IN-Abs Phase 4B preregistration: supervised published baselines

Frozen before the full run.

- Official IN-Abs training partition only; deterministic seed-42 split of
  6,325 train / 703 dev, excluding empty pairs 299 and 4799.
- Official 100-document test contents are not read.
- Released checkpoints trained on all 7,030 documents are not used because
  they include the frozen development documents.
- Candidate documents use the same reference-free, evenly spaced maximum of
  400 source sentences as E1.
- Training supervision uses the already-frozen ordinary summary-level greedy
  oracle labels generated in Phase 2. No new target selection is performed.
- Each development reference supplies only its target word budget at decode
  time. The count is the official Gist spaCy `en_core_web_sm`
  punctuation-filtered token count.
- Complete sentences only, skip non-fitting candidates, zero violations.
- `py-rouge==1.1`, ROUGE-1/2/L document-level F1 macro averages, plus 10,000
  paired bootstrap samples against frozen Phase 3C E1.

## Gist

- Released 334-feature definitions and LightGBM hyperparameters:
  learning rate 0.001, 1,043 leaves, 1,043 estimators, bagging fraction 0.8,
  feature fraction 0.9.
- 100-dimensional Word2Vec is trained only from the 6,325 training documents,
  minimum count 5, seed 42.
- Released cue phrases, POS tag list, character-loop embedding behavior,
  global train scaling and per-document inference scaling are preserved.
- Compatibility changes are limited to POSIX paths and current
  gensim/lightgbm APIs. This is a restricted faithful retraining.

## SummaRuNNer

- RNN_RNN architecture from the implementation used by the released assets:
  100-dimensional word embeddings, 50-dimensional position embeddings,
  200 hidden units, 10 relative-position bins.
- Vocabulary/initial embeddings reuse the training-only Word2Vec produced by
  the immediately preceding Gist run; embeddings remain trainable.
- Absolute position capacity 1,000; maximum 400 candidate sentences; maximum
  100 tokens per sentence.
- Released IN-Abs checkpoint training settings: seed 1, batch size 4,
  learning rate 1e-5, 30 epochs, gradient norm 1.0.
- The final epoch is evaluated; epoch count and all settings are fixed without
  inspecting development ROUGE.

Both methods must be described as split-safe budget adaptations, not as
bit-identical official reproductions.
