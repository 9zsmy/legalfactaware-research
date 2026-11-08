# IN-Abs Phase 3: English static representation comparison

Phase 3 freezes the deterministic split, phase-2 shared training rows, ordinary
Oracle labels, Indian-English Legal14 and Section6 features, balanced liblinear
logistic regression, reference-free maximum-400 candidate protocol, word-bigram
MMR weight 0.2, and complete-sentence decoder. The sole changed variable is the
English TF-IDF representation:

- E0: character 2–4 grams (completed phase-2 ordinary model);
- E1: word 1–2 grams;
- E2: word 1–2 grams plus character 3–5 grams.

All three are evaluated on the same 703-document development split under four
tracks: fixed 423/643/973-word budgets and a per-document reference-summary
word budget matching the IN-Abs paper's inference-length rule. Development uses
the already frozen Google `rouge-score` diagnostics for paired comparisons.
Before the sealed official test is run, the official `py-rouge` configuration
will be reproduced and validated separately. No official-test content is read.
