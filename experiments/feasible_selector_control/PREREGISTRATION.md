# Chinese feasible complete-sentence selector control

Date frozen: 2026-10-20

## Question

Given exactly the same frozen sentence scores, does the complete-sentence selection algorithm materially change development-set quality?

## Strict comparison

- **Greedy fit/skip:** repeatedly takes the highest-scoring sentence that still fits.
- **Exact 0/1 knapsack:** maximizes the sum of the same frozen normalized sentence scores under the same constraints.

Both selectors use the same LegalFactAware model scores, the same 450-character budget, at most 8 sentences, complete sentences only, and source-order output. Dynamic redundancy and fact bonuses are disabled in both arms so that the selector is the only changed variable.

The archived full system with redundancy weight 0.2 is not part of the causal comparison and is not used for model selection.

## Frozen evidence

- Development split: 942 CAIL clean documents.
- Frozen TF-IDF + Legal14 + Section6 LR scorer.
- Frozen vectorizer and evaluation implementations.
- The regenerated greedy predictions must exactly match the archived no-redundancy predictions before the comparison is accepted.
- No test-set argument exists.

## Outcomes

- LAWA ROUGE-1/2/L F1 and macro Fact F1.
- Paired bootstrap 95% confidence intervals, paired Wilcoxon tests, and Holm correction.
- Selection-change rate and exact objective audit.

No tuning will be performed after observing results.
