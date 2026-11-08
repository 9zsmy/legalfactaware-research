# DELTA–Dream IN-Abs 50-document pilot preregistration

Frozen: 2026-11-05, before any DELTA prediction on the selected IN-Abs cases.

## Purpose

This is a feasibility and runtime pilot for adapting the official DELTA
decoding core to long English legal summaries. It is not a reproduction of the
DELTA paper's full English table and is not yet a main-table comparison.

## Data discipline

- Dataset: official IN-Abs training partition only.
- Existing frozen split: 6,325 model-training cases and 703 development cases,
  seed 42, after excluding the two empty-summary cases 299 and 4799.
- Official 100-case test contents remain unread.
- Pilot: the 50 development IDs with the smallest SHA-256 values of
  `delta-inabs-pilot-v1:<case_id>`.
- References are written to a separate evaluation file and are never included
  in the generation input or prompt.

## Length protocol

The existing IN-Abs fixed-budget track uses 643 English words. DELTA controls
characters rather than words, so this pilot uses a frozen training-only
calibration:

- median characters per word over the 6,325 training references: 5.7286585;
- DELTA target: `round(643 * 5.7286585) = 3,684` characters;
- mean Dream-token characters over a deterministic 256-reference training
  calibration subset: 4.4699386;
- initial masked generation buffer: approximately 824 tokens.

The primary length diagnostics are actual English word count, absolute error
to 643 words, and the number of outputs above 643 words. Character error is a
secondary diagnostic. This conversion is an adaptation, not native word-level
DELTA control.

## Model and decoding

- Official vendored DELTA decoding core.
- Dream-v0-Instruct-7B, 4-bit inference, seed 42.
- Confidence threshold 0.65; low-confidence threshold 0.95; probability
  confidence; no remasking; dynamic mask insertion/removal enabled.
- Maximum 1,200 iterations.
- English legal summarization prompt.
- Source truncation: head and tail within 8,192 stored characters, matching the
  completed CAIL adaptation's engineering bound. The truncation rate and used
  source length are reported.
- Each prediction is appended immediately and the runner supports resume.

## Staged execution and gates

1. Prepare and hash the frozen pilot input and separate reference file.
2. Run one case and inspect runtime, GPU memory, remaining masks, word length,
   and obvious degeneration.
3. If the one-case run completes without OOM and without remaining masks,
   launch all 50 cases in `screen`.
4. Report runtime and length diagnostics before treating quality scores as
   interpretable.

Proceed to a 703-document run only if the pilot has no systematic OOM, at
least 90% of cases finish without remaining masks, and the estimated full-run
cost is acceptable. Any later quality comparison must use a clearly disclosed
adapted label and a common evaluation implementation.

## Permitted label

`DELTA-Dream-INABS-643w-adapted (official decoding core; train-calibrated character target)`

