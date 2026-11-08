# Research 1 closure: frozen E1 Oracle supervision × decoder 2×2

Frozen before the new full run. The official 100-case IN-Abs test remains unread.

## Question

Under the final English E1 representation, does P50 budget-aware Oracle
supervision improve over ordinary greedy-Oracle supervision, and what is the
independent effect of complete-sentence budget decoding versus a final hard
sentence fragment?

## Frozen data and model components

- Official IN-Abs train only: deterministic seed-42 split, 6,325 train / 703 dev.
- Two zero-byte official-train summaries remain excluded as previously recorded.
- Reference-free, evenly spaced maximum-400 candidate protocol.
- Final E1 word 1–2 gram TF-IDF vectorizer, Indian-English Legal14, Section6,
  balanced liblinear logistic regression, word-bigram MMR weight 0.2.
- Both supervision variants use the same 663,533 sampled training rows. The
  only fitted-target change is ordinary-Oracle versus 643-word budget-Oracle
  membership.

## Frozen 2×2

1. Ordinary Oracle + complete-sentence fit/skip decoder.
2. Budget Oracle + complete-sentence fit/skip decoder.
3. Ordinary Oracle + score-order decoder with only the final sentence fragment
   truncated to the exact remaining spaCy-counted word budget.
4. Budget Oracle + the same truncation decoder.

No tuning is allowed after seeing results.

## Evaluation

- Fixed 423, 643, and 973-word tracks plus the per-document reference-summary
  spaCy word ceiling.
- Official Gist `en_core_web_sm` punctuation-filtered word counting.
- `py-rouge==1.1`, ROUGE-1/2/L F1, `max_n=2`, `limit_length=False`.
- Report mean output words, mean budget utilization, truncated-case count,
  budget violations, selection-change rate, and 10,000-sample paired bootstrap
  confidence intervals.
- Primary contrasts: Budget−Ordinary within each decoder; Complete−Truncate
  within each supervision; supervision×decoder interaction.

## Interpretation rule

Budget-aware supervision is supported only if its paired confidence interval is
above zero under the final E1 complete-sentence configuration. If it is below
zero or crosses zero, the English result is reported as a learning boundary,
not tuned away. Complete-sentence decoding is evaluated independently and must
not be credited for a supervision effect.
