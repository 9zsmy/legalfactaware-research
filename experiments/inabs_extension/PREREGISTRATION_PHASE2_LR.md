# IN-Abs Phase 2: LegalFactAware LR strict Oracle-label comparison

The official test set remains sealed. Model development and evaluation use the
fixed 6,325/703 split of the 7,028 usable official-train pairs established in
phase 1 (seed 42). Reference-summary nearest-rank train budgets are 423, 643,
and 973 words.

Two balanced logistic-regression models are trained on exactly the same sampled
sentence rows. They share English character 2–4 gram TF-IDF, Indian-English
Legal14, six heuristic judgment-section indicators, optimization settings,
redundancy control, and complete-sentence decoder. Their sole difference is the
binary label: membership in an unconstrained greedy ROUGE-1+ROUGE-2 Oracle or
membership in the corresponding 643-word budget-aware Oracle.

Every actual model uses the same reference-free candidate protocol: retain all
sentences up to 400; otherwise retain 400 deterministic, evenly spaced source
sentences including both endpoints. This avoids the reference-overlap top-400
screen used only for phase-1 Oracle-ceiling diagnostics.

Both trained models are evaluated at all three frozen budgets. We report mean
ROUGE-1/2/Lsum F1, selection-change rates, and seed-42 paired bootstrap deltas
(10,000 resamples). No dev result may change the split, budgets, features,
classifier, or decoder for this comparison.
