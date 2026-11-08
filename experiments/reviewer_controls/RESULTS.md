# Chinese reviewer controls: verified development results

Run date: 2026-10-28  
Status: complete on the 942-document development set; the test split was not read or supplied to the script.  
Elapsed time: 1,029.4 seconds.  
Executed-script SHA-256: `fe5f5498b1c53b5ff71b9457cf30b19e6ae55149ac98357aa18d19fa3906d4af`.

The archived aggregate files are preserved verbatim in `results/`. In those raw files, `Expression_Recall` means the fixed-union document-macro recall returned by `reference_fact_metrics`; it is **not** the legacy conditional Fact Recall used by the existing main paper table. The source has since been relabeled to expose both conventions explicitly. No model result changed.

## 1. Matched-count independent supervision

| Training labels | R-1 | R-2 | R-L | Expression F1 | Avg. chars | Avg. candidates |
|---|---:|---:|---:|---:|---:|---:|
| Oracle membership (full) | 60.27 | 41.17 | 49.84 | 47.85 | 396.49 | 7.78 |
| Independent similarity, per-document positive count matched | 58.29 | 38.92 | 47.47 | 47.79 | 338.58 | 7.97 |
| Full minus matched | **+1.99** | **+2.25** | **+2.37** | +0.06 | +57.90 | -0.19 |

The positive counts match exactly in every training document: 47,150 positives under each rule. Candidate-level agreement is 93.62%, global positive-set Jaccard is 55.04%, and the mean document-level positive-set Jaccard is 57.38%.

Paired development tests for full minus matched-count independent:

| Metric | Mean delta | Bootstrap 95% CI | Holm-adjusted Wilcoxon p |
|---|---:|---:|---:|
| R-1 | +1.99 | [1.63, 2.35] | 6.23e-23 |
| R-2 | +2.25 | [1.82, 2.69] | 1.39e-18 |
| R-L | +2.37 | [1.93, 2.81] | 4.11e-22 |
| Expression F1 | +0.06 | [-0.75, 0.89] | 0.157 |

**Supported conclusion.** Matching the number of positive labels does not remove the ROUGE advantage of summary-level Oracle supervision. The effect is statistically reliable for ROUGE, but not for expression F1.

**Required qualification.** The independent-label output is about 58 characters shorter on average. This control removes the positive-count confound but not the output-length or score-calibration confound, so it does not prove that competition alone causes the entire gain.

## 2. Lexical content versus position information

| Variant | R-1 | R-2 | R-L | Expression F1 | Avg. chars |
|---|---:|---:|---:|---:|---:|
| Full: decorated text + Legal14 + Section6 | 60.27 | 41.17 | 49.84 | 47.85 | 396.49 |
| Raw lexical text + Legal14 + Section6 | 60.00 | 40.75 | 49.38 | 47.16 | 396.01 |
| Decorated text only | 59.91 | 40.66 | 49.27 | 47.34 | 396.66 |
| Raw lexical text only | 59.86 | 40.55 | 49.20 | 47.05 | 396.68 |
| Position marker only | 38.92 | 15.85 | 24.64 | 22.31 | 341.26 |
| Lead fit/skip | 17.86 | 3.30 | 13.36 | 0.16 | 191.63 |
| Tail fit/skip | 28.26 | 10.24 | 17.29 | 25.54 | 252.21 |

**Supported conclusion.** Lexical candidate content is the dominant signal. Removing only the position marker while keeping lexical text and auxiliary features costs 0.28/0.42/0.46 ROUGE points, whereas a marker-only model collapses. Therefore, the large earlier ablation from removing the entire position-augmented TF--IDF block must not be described as a pure position effect. Legal14 and Section6 provide only a small incremental improvement over decorated text alone.

## 3. Candidate cap and character-budget sensitivity

At the frozen development setting `(B=450, K=8)`, 803/942 documents (85.24%) terminate at the candidate cap and 724/942 (76.86%) encounter at least one candidate that does not fit. Thus both constraints influence selection, but the latter count is not a terminal-stop count.

| Budget | K | R-1 | R-2 | R-L | Expression F1 | Avg. chars | Cap-hit rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 6 | 58.26 | 38.73 | 47.72 | 47.41 | 274.62 | 81.21% |
| 300 | 8 | 58.60 | 38.69 | 47.72 | 47.40 | 293.56 | 34.39% |
| 300 | all | 58.45 | 38.50 | 47.54 | 47.36 | 297.38 | 0% |
| 400 | 6 | 59.60 | 40.21 | 49.09 | 47.77 | 315.59 | 95.12% |
| 400 | 8 | **60.39** | **40.89** | **49.71** | **47.98** | 370.49 | 76.75% |
| 400 | all | 59.53 | 40.19 | 48.94 | 47.35 | 397.46 | 0% |
| 450 | 6 | 59.66 | 40.41 | 49.19 | 47.72 | 326.96 | 98.09% |
| 450 | 8 | **60.27** | **41.17** | **49.84** | **47.85** | 396.49 | 85.24% |
| 450 | 10 | 59.30 | 40.51 | 49.04 | 46.89 | 431.35 | 57.43% |
| 450 | 12 | 58.73 | 40.12 | 48.60 | 46.49 | 443.49 | 22.40% |
| 450 | all | 58.46 | 39.91 | 48.39 | 46.36 | 447.36 | 0% |

The complete 18-cell grid is in `results/budget_by_candidate_cap.csv`.

**Supported conclusion.** `K=8` is frequently binding, so it must be disclosed as a substantive decoder constraint. On development data, however, relaxing it to 10, 12, or unlimited candidates makes summaries longer and lowers ROUGE and expression F1; thus `K=8` behaves as a useful length/content regularizer rather than an obvious implementation defect. This development analysis does not authorize changing the frozen test protocol.

## 4. Decoder mechanism control

| Decoder | R-1 | R-2 | R-L | Expression F1 | Avg. chars | Hard truncations |
|---|---:|---:|---:|---:|---:|---:|
| Full fit/skip with replacements | 60.27 | 41.17 | 49.84 | 47.85 | 396.49 | 0 |
| Drop-only complete candidates, no replacements | 59.22 | 39.97 | 48.33 | 44.53 | 376.81 | 0 |
| Legacy post-hoc within-candidate truncation | 59.39 | 40.44 | 48.66 | 45.50 | 401.85 | 379 |

Full minus drop-only is +1.05/+1.20/+1.51 ROUGE and +3.32 expression-F1 points. All four bootstrap intervals exclude zero; Holm-adjusted Wilcoxon p-values are 1.26e-7, 1.74e-6, 4.20e-11, and 2.73e-21, respectively.

Drop-only versus legacy truncation is -0.17/-0.46/-0.33 ROUGE and -0.97 expression-F1 points. Bootstrap intervals are negative, but the Holm-adjusted Wilcoxon tests are not significant for ROUGE because 60.08% of document-level ROUGE differences are ties and the nonzero differences are skewed; expression F1 favors legacy truncation with adjusted p=0.00167.

**Supported conclusion.** Avoiding within-candidate truncation alone is insufficient. The result supports the importance of continuing after an over-budget candidate and admitting later complete candidates, while the full decoder also eliminates 379 hard truncations. Because these operations can interact, the experiment is not an additive contribution decomposition. The paper should describe this as a quality/candidate-boundary-integrity trade-off, not claim that complete-candidate dropping alone improves ROUGE over truncation.

## 5. Empty-expression audit and validity checks

- References with detected amount/date/law/decision expressions: 303/93/942/908 documents.
- References with no detected expression of any type: 0/942.
- Therefore, the all-empty scoring convention cannot affect the development document-macro mean.
- Every evaluated variant and every budget/cap cell has zero budget violations.
- All non-legacy variants have zero within-candidate hard truncations.

At `(B=450, K=8)`, the terminal stop reasons are distinct: 803/942 documents stop at the candidate cap and 139/942 stop because no remaining candidate fits the character budget. Separately, 724/942 documents encounter at least one over-budget candidate during selection. The latter is not itself a terminal-stop count.

## 6. Follow-up diagnostics after the reviewer-control analysis

These analyses were frozen in `FOLLOWUP_PREREGISTRATION.md` before their outputs were inspected. They reuse the completed run and do not train a new model.

### 6.1 Training-positive length, position and role preferences

| Positive-label set | Mean chars | Median chars | Mean normalized position |
|---|---:|---:|---:|
| Oracle membership | 54.18 | 44 | 0.5781 |
| Count-matched independent | 44.27 | 37 | 0.5687 |

On a paired per-document basis, Oracle positives are 11.10 characters longer on average (bootstrap 95% CI [10.70, 11.50]). The normalized-position difference is only +0.0091, although its large-sample interval excludes zero ([0.0069, 0.0114]). Thus candidate-length preference is a much more substantial difference than source position.

The largest Section6 distribution shifts are that Oracle selects more `claim` candidates (38.02% versus 35.85%) and `reason` candidates (17.66% versus 15.21%), while the independent rule selects more `other` candidates (10.64% versus 6.84%). These are descriptive label-set differences, not an additive explanation of the ROUGE gap.

### 6.2 Realized-output-length-close subsets

| Maximum absolute length difference | Documents | Full chars | Matched chars | ΔR-1 | ΔR-2 | ΔR-L | ΔExpression F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 chars | 177 | 392.10 | 390.76 | +0.37 | +0.11 | +0.18 | -0.44 |
| 25 chars (primary) | 308 | 387.93 | 382.68 | +0.76 | +0.54 | +0.78 | -0.46 |
| 50 chars | 485 | 388.42 | 373.03 | +0.91 | +0.91 | +1.07 | -0.37 |

For the primary 25-character subset, the bootstrap 95% intervals exclude zero for R-1 `[0.37, 1.15]`, R-2 `[0.06, 1.02]`, and R-L `[0.26, 1.29]`, but not for Expression F1 `[-1.54, 0.66]`. For the stricter 10-character subset, every bootstrap interval includes zero.

**Supported conclusion.** Oracle labels preferentially select longer training candidates, and realized output length explains part of the full-sample ROUGE gap. A smaller ROUGE advantage remains in the predeclared 25-character-close subset, but not reliably in the smaller 10-character-close subset. Because subset membership is determined by realized outputs, this is descriptive robustness evidence and does not eliminate selection or length confounding.

### 6.3 Expression results by type

The table below reports exact-match micro F1; support is the number of references with at least one expression of that type.

| Type | Support | Full | Matched-count independent | Drop-only | Legacy truncation |
|---|---:|---:|---:|---:|---:|
| Amount | 303 | 39.10 | 38.37 | 38.00 | 37.64 |
| Date | 93 | 23.10 | 23.95 | 21.60 | 21.41 |
| Law | 942 | 50.58 | 49.13 | 49.77 | 50.25 |
| Decision | 908 | 79.34 | 79.23 | 76.17 | 76.75 |

Conditional document-macro F1, computed only on references nonempty for the given type, is:

| Type | Full | Matched-count independent | Drop-only | Legacy truncation |
|---|---:|---:|---:|---:|
| Amount | 61.85 | 54.80 | 60.76 | 61.17 |
| Date | 51.22 | 36.48 | 51.81 | 52.80 |
| Law | 47.47 | 44.78 | 41.46 | 44.35 |
| Decision | 78.24 | 78.08 | 74.57 | 75.23 |

These results expose the effect of type frequency and averaging convention. The fixed-union aggregate Expression F1 should not be used alone to claim uniform gains across expression types. In particular, the matched-count condition has slightly higher date micro F1 despite much lower conditional document-macro date F1. These rule-based exact matches remain lexical expression coverage measures, not semantic legal-fact correctness.

## Paper-level revisions justified by these experiments

1. Add the matched-count supervision control and soften the causal wording from “competition causes the gain” to “the Oracle objective retains a ROUGE advantage after exact per-document positive-count matching.”
2. Replace the old position-feature interpretation with the lexical/marker decomposition; present the TF--IDF lexical block as the main predictor and Legal14/Section6 as small auxiliary increments.
3. Disclose the eight-candidate cap in the main method and report that it binds on 85.24% of development documents. Include the budget-by-cap sensitivity table in the appendix.
4. Add the drop-only control and attribute the full decoder gain to replacement admission plus truncation avoidance, not truncation avoidance alone.
5. State that the empty-expression convention is inactive on development because every reference contains at least one detected expression type.
6. Keep these results clearly labeled as development-set mechanism checks. Do not replace the frozen test ranking or retune SENDE from them.
7. Report a single, explicitly defined primary Expression aggregation in the main text and put the alternative aggregation plus per-type results in the appendix. Do not merge the legacy conditional recall and fixed-union macro recall by renaming them.

No paired significance comparison against SENDE is reported.

## Remaining issues not resolved by these development controls

- The contribution boundary relative to prior Oracle-supervision work still requires a direct related-work and controlled-method comparison.
- Main-table baseline versions, especially any SSE proxy, must be made identical between the score table and significance analysis.
- A paired SENDE test is not added here because this run does not contain aligned, version-matched per-document SENDE outputs; the paper must state that it is not reported.
- Rule-based Expression matching cannot establish legal-fact correctness. A blinded human evaluation of factual/case-outcome consistency remains necessary for such a claim.
- Cost, model configuration and quality comparisons still need one frozen, version-consistent accounting table.
