# Reproduction and adaptation audit

Updated: 2026-11-08

## DELTA

- Publication: EMNLP 2026 main conference.
- Source identity: the decoder core is vendored from
  `DFKI-NLP/delta-diffusion`; Git blob SHAs are recorded in
  `vendor/delta-diffusion/SOURCE_MANIFEST.json`.
- CAIL implementation: `run_delta_cail.py` imports the official
  `load_model` and `generate_length_controlled` functions directly.
- Adaptations: Chinese CAIL input, Chinese legal prompt, 450-character target,
  head-tail source truncation, Dream 7B 4-bit inference, and a frozen Chinese
  characters-per-token estimate.
- Correct label: `DELTA-Dream-CAIL-adapted (official decoding core)`.
- It is not a complete reproduction of every experiment in the paper.

Before the 942-document CAIL run, `validate_delta_official_xsum.py` performs a
50-document XSum regression using the official task prompt and paper
hyperparameters.  `run_full_delta_after_validation.sh` starts the full CAIL run
only if XSum character-length MAE is at most 5 and at least 90% of outputs are
within 10 characters of the reference target length.

The regression completed on 2026-11-03 with MAE 2.0 characters, median error
2.0, and 50/50 outputs within 10 characters.  This passes the gate and is close
to the repository's full-XSum DELTA+Dream MAE of 2.3, while remaining only a
50-document 4-bit regression rather than a reproduction of the full table.

After the gate passed, the adapted system completed all 942 frozen CAIL
development documents on 2026-11-05.  Against LegalFactAware under the same
evaluation code, DELTA obtained R-1/R-2/R-L of 34.21/16.76/19.95 and document-
macro Expression F1 of 19.57; LegalFactAware obtained 65.53/51.41/52.55 and
47.85.  Paired-bootstrap 95% intervals for all four LegalFactAware-minus-DELTA
differences exclude zero.  DELTA's mean output length was 446.49 characters,
but 55/942 outputs exceeded the 450-character hard upper bound.

## Black-box LCG

- Publication: ACL 2025 main conference, Volume 1 long paper.
- No author implementation was found in the paper's official materials during
  this audit.  The local implementation is independently derived from the
  equations and prompt templates in the paper.
- The implementation follows the MH+IS beam-1 path: iterative importance
  proposals, pairwise LLM-as-a-judge scoring, length/quality acceptance ratio,
  random acceptance, and at most five iterations.
- The paper's main open-model table uses beam size 16.  The paper separately
  reports beam-1 MH+IS ablations, including Qwen2.5.  Therefore the current
  result is a valid beam-1 adapted variant, not the paper's main-table setup.
- Adaptations: Qwen2.5-7B-Instruct 4-bit, Chinese legal prompt, Python-character
  counting instead of English NLTK word counting, CAIL input, one-shot legal
  example, and head-tail source truncation.
- Correct label: `LCG-MHIS-beam1-Qwen2.5-7B-CAIL-adapted (independent)`.

For the 450-character upper-bound run, only 2 of 100 subset documents required
an iteration.  This result mainly measures direct Qwen generation; it does not
by itself establish the quality of the MH+IS search mechanism.  The exact-450
run activates search but is an exploratory diagnostic because exact length is
not the paper task's hard upper-bound objective.

The paper's main beam-16 search setting was subsequently implemented as 16
independent initial states and MH+IS chains, with all logical chains executed
sequentially on the 24 GB GPU.  A preregistered one-document engineering audit
passed all structural checks: 16 initial states were present, acceptance
probabilities and final selection were independently recomputed, and 16 local
tests passed.  Under the upper-bound objective all initial states were already
valid, so no search proposal was made.  Under the exact-450 diagnostic, all 16
chains ran five iterations (80 proposals total), none reached 450 characters,
and the minimum-distance fallback selected a 454-character output after 856.39
seconds.  This failed the preregistered expansion gate and was not extended to
100 documents.  Its correct label is
`LCG-MHIS-beam16-Qwen2.5-7B-CAIL-adapted (independent)`.

## Zero-Shot Strategies for Length-Controllable Summarization

- Publication: Findings of NAACL 2025.
- The local implementation independently follows the published sample-
  filtering equations, three-sample main setting, character counter, automated
  revision feedback, 10% exact-target tolerance, and up-to-five-revision range.
- The paper's assistant-prefill behavior is implemented with an unclosed Qwen
  assistant prefix. A Transformers 4.41 compatibility failure in an initial
  invalid run was isolated, archived, fixed, and excluded before formal output.
- Adaptations: Qwen2.5-7B-Instruct 4-bit, Chinese legal prompt, CAIL documents,
  450 Python-character target, head-tail input truncation, and a one-sided hard-
  upper-bound selection rule.
- English length-approximation and target-adjustment coefficients were not
  transferred to Chinese and were not fitted on evaluation documents.
- Correct label: `ZeroShotLC-SF3-AR5-Qwen2.5-7B-CAIL-adapted (independent)`.

The preregistered five-document audit passed all integrity checks. The exact-
target diagnostic reached the paper's ±10% band for 2/5 documents and violated
the 450-character task budget for 1/5. The hard-upper-bound variant had no
violations but averaged only 179.8 characters.

The hard-upper-bound variant was then run without tuning on the frozen 100-
document subset. All IDs, 300 sampled candidates, selection decisions, traces,
and final outputs passed the automated audit. No revisions were triggered;
one sampled candidate exceeded 450, but a compliant candidate from the same
document was selected. Final ROUGE-1/2/L was 43.72/26.05/28.99 and document-
macro Expression F1 was 25.53, versus 65.94/52.19/53.38 and 47.30 for
LegalFactAware. All four paired-bootstrap LegalFactAware-minus-ZeroShotLC 95%
intervals exclude zero.

## Claims permitted by the current evidence

Permitted:

- official-core DELTA was evaluated on all 942 frozen CAIL development
  documents, while independent beam-1 LCG was evaluated on the frozen 100-
  document subset, using the same character-counting protocol;
- DELTA provides accurate target-length approximation on CAIL but, with Dream
  7B in this adaptation, substantially lower Chinese legal summary quality than
  LegalFactAware and a nonzero hard-budget violation rate;
- LCG removes the few upper-budget violations, while its search is rarely
  activated under this particular 450-character prompt;
- the independent ZeroShotLC hard-upper-bound adaptation satisfies the budget
  on all 100 subset documents but substantially underuses it and scores below
  LegalFactAware under the common evaluator.

Not yet permitted:

- all three papers were fully reproduced;
- the full Zero-Shot LCS paper was reproduced, including its original English
  model, datasets, length-approximation calibration, and target adjustment;
- the LCG main-table beam-16 result was reproduced on the paper's original
  English task (only the search configuration was independently implemented
  and audited on one CAIL document);
- the original methods are inferior in their original English settings;
- DELTA is inferior in its original English settings, with other diffusion
  backbones, or as a general method rather than in this frozen CAIL adaptation;
- the LCG 100-document subset result is interchangeable with a full 942-
  document development-set result.
