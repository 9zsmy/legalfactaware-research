# IN-Abs Phase 6A: summary-state diagnostics (2026-09-18)

Purpose: determine whether candidate value changes after summary sentences have
been selected, before training any state-aware model.

- Frozen split: official IN-Abs train only, 6,325 model-train / 703 dev, seed 42.
  The official 100-document test remains unread.
- E1 word 1–2 gram LR bundle is frozen; no fitting or hyperparameter selection.
- Candidates: reference-free evenly spaced maximum of 400 source sentences.
- Budget: fixed 643 spaCy-counted words (train-derived median budget).
- Oracle objective: greedy incremental ROUGE-1 F1 plus ROUGE-2 F1 proxy on word
  n-gram multisets. Reference text is used only for this **diagnostic teacher**.
  It is not a deployable model or a formal upper bound for `py-rouge`.
- At each greedy Oracle state, retain only unselected complete sentences fitting
  the remaining budget. Ties use source order. Stop when no positive gain fits.
- Diagnostics: initial-vs-current top choice flip rate, pairwise order reversals
  among at most 50 initially highest-gain still-fitting candidates, positive
  initial gains that become nonpositive, and E1 MMR's agreement with the greedy
  Oracle on the **same Oracle state**.
- Evaluate E1 and greedy Oracle outputs with the same `py-rouge==1.1` code; this
  gap is descriptive only because Oracle sees the reference. Report zero budget
  violations and 10,000 paired bootstrap samples.

Do not interpret dev diagnostics as test performance or use them to choose
official-test configurations. The next training experiment, if any, needs its
own preregistration and train-only trajectory generation.
