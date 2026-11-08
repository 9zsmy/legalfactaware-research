# Follow-up diagnostics requested after reviewer-control results

Frozen: 2026-10-28, before inspecting the follow-up outputs.

This follow-up does not train a new model and does not accept a test-set argument. It uses the already frozen development predictions from the completed reviewer-control run.

## A. Training-positive diagnostics

Reconstruct the per-document count-matched independent labels exactly as in the completed run. For Oracle-positive and matched-independent-positive candidates, report:

- character-length distribution: mean, standard deviation, quartiles and median;
- normalized source position `index / (candidate_count - 1)`;
- Section6/role distribution;
- paired per-document differences in mean positive-candidate length and normalized position, with a 10,000-resample document bootstrap interval and Pratt-Wilcoxon p-value.

This is diagnostic evidence about label-set preferences, not a causal decomposition of the ROUGE gap.

## B. Similar-output-length comparison

Using the already generated full-Oracle and matched-independent development outputs, form paired subsets with absolute output-character difference at most 10, 25 and 50 characters. The threshold of 25 characters is the primary descriptive comparison. Report subset size, mean output lengths, and paired mean differences for ROUGE-1/2/L and fixed-union Expression F1, with 10,000-resample bootstrap intervals.

Because subset membership is determined post hoc by realized output length, this is a robustness diagnostic rather than an unbiased treatment-effect estimate.

## C. Expression results by type

For Full, matched-count independent, drop-only and legacy truncation outputs, separately report amount, date, law and decision expression results. For each type report reference support, total reference/predicted/exact-match instances, micro precision/recall/F1, and document-macro precision/recall/F1 conditional on a nonempty reference set for that type.

No empty-empty score of one is used in the conditional per-type results.

## D. Stopping reasons

Reuse the completed grid diagnostics and explicitly distinguish termination at the candidate cap from termination because no remaining candidate fits the character budget.

## Interpretation constraints

- Do not subtract old and new ablation gaps as a causal decomposition.
- Do not describe a length-close subset as eliminating all length or selection confounding.
- Do not describe exact-match expressions as legal-fact correctness.
- Do not report a paired SENDE significance test; aligned per-document predictions for a frozen, version-matched comparison are not available in this run.

