# Chinese reviewer-requested controls (development set only)

Date frozen: 2026-10-28

## Purpose

This experiment addresses four causal ambiguities identified during paper review without reading or tuning on the CAIL test split.

1. Does summary-level Oracle supervision still outperform an independent label rule after matching the number of positive labels in every training document?
2. How much of the position-augmented TF--IDF block comes from lexical content and how much comes from the prepended position marker?
3. Is the eight-candidate cap, rather than the 450-character budget, frequently binding?
4. How much of the fit/skip improvement comes from avoiding within-candidate truncation, as opposed to admitting replacement candidates?

## Frozen inputs and shared components

- CAIL2020-SFZY-clean train/dev: 7,495/942 documents.
- Archived v2.2 TF--IDF vocabulary and IDF values.
- Archived v2.5-B full scorer for the reference condition.
- Logistic regression for newly trained controls: `liblinear`, `class_weight=balanced`, `max_iter=1000`, seed 42.
- Full-system decoder: redundancy weight 0.2, fact weight 0.0, source-order output.
- LAWA document-macro ROUGE and the existing rule-based expression matcher.
- No test-set argument exists in the experiment script.

## Control A: per-document positive-count-matched independent labels

For each training document, let `k` equal the number of positive Oracle membership labels. Rank every candidate independently by the existing clipped character-bigram precision against the reference and label the top `k` candidates positive. Exact ties are broken by earlier source position. This preserves the independent scoring rule while matching Oracle positive-label count separately in every document.

The matched-label scorer uses the same position-augmented TF--IDF, Legal14, Section6, LR trainer and 450-character/eight-candidate fit/skip decoder as the full system.

Primary comparison: full Oracle supervision minus matched-count independent supervision on development ROUGE-1/2/L and rule-based expression F1. A four-metric Holm family is reported.

## Control B: lexical and position separation

All variants use the same Oracle labels and frozen vectorizer.

- `decorated_text_only`: candidate text with the prepended position marker.
- `lexical_text_only`: raw candidate text without the marker.
- `position_marker_only`: the marker without candidate text.
- `lexical_full_aux`: raw candidate text plus Legal14 and Section6.
- `full_reference`: decorated text plus Legal14 and Section6.

Lead and Tail fit/skip baselines are also evaluated under 450 characters and at most eight candidates.

## Control C: character budget by candidate cap

The archived full scorer is decoded at budgets 300, 400 and 450 with candidate caps 4, 6, 8, 10, 12 and no additional cap. No model is retrained. Report ROUGE, expression recall/F1, mean characters, mean selected candidates, cap-hit rate, budget-skip rate and terminal stop reason.

## Control D: drop-only decoder

Start from the same eight candidates selected by the legacy pre-budget selector. Preserve source order and remove complete candidates from the tail until the output fits 450 characters. Do not admit replacements. Compare this condition with both post-hoc within-candidate truncation and full fit/skip.

## Expression-set audit

Report the number of development references containing each detected expression type and the number containing no detected expression of any type. If the all-empty count is zero, the all-empty score convention cannot change the reported development mean.

## Interpretation constraints

- Development results are mechanism controls, not new frozen-test rankings.
- The matched-count control isolates positive-label count but does not make the two label-generating objectives identical.
- The position-marker variants do not establish causal generalization beyond this corpus.
- The `B x K` grid is a decoder sensitivity analysis, not a basis for retroactively changing the frozen test protocol.
- No SENDE threshold or test prediction is changed by this experiment.

