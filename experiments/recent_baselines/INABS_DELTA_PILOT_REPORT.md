# DELTA–Dream IN-Abs 643-word pilot report

Date: 2026-11-05

## Outcome

The official DELTA decoding core can run stably on the selected 50 IN-Abs
development documents with Dream 7B 4-bit, but the current character-calibrated
adaptation is not suitable for an unqualified 643-word main-table comparison.
It has a 34% word-budget violation rate and truncates 82% of source judgments.
On the same 50 documents, the frozen E1 system is higher on all three ROUGE
metrics with paired bootstrap intervals excluding zero.

## Protocol

- Frozen IN-Abs split: 6,325 train / 703 dev; official test unread.
- Pilot IDs: deterministic SHA-256-selected 50-document dev subset.
- Existing English target: 643 words.
- DELTA target: 3,684 characters, derived from the training-reference median
  5.7286585 characters/word.
- Dream calibration: 4.4699386 characters/token from 256 deterministic
  training references.
- Official DELTA core; Dream-v0-Instruct-7B 4-bit; maximum 1,200 iterations;
  8,192-character head-tail source bound.

## Runtime and length

| Measure | Result |
|---|---:|
| Completed | 50/50 |
| Generation/evaluation exit codes | 0 / 0 |
| Total GPU time | 2.014 h |
| Mean / median time per document | 145.03 / 153.10 s |
| Projected 703-document GPU time | 28.32 h |
| Mean / median output words | 634.12 / 633.5 |
| Mean / median absolute word error | 19.56 / 16.0 |
| Outputs above 643 words | 17/50 (34%) |
| Unfinished outputs | 0/50 |
| Truncated source judgments | 41/50 (82%) |
| Mean peak allocated GPU memory | 7.42 GiB |

## Same-subset quality comparison

The comparison uses the same 50 references and the frozen
py-rouge==1.1-compatible configuration. E1 is the existing complete-sentence
fixed-643-word `ordinary_complete` configuration.

| Method | R-1 | R-2 | R-L | Word-budget violations |
|---|---:|---:|---:|---:|
| E1, fixed 643 words | **56.73** | **33.24** | **53.51** | 0/50 |
| DELTA–Dream, character-calibrated | 52.57 | 28.63 | 48.59 | 17/50 |

Paired E1-minus-DELTA differences:

| Metric | Mean difference | 95% bootstrap interval |
|---|---:|---:|
| R-1 | +4.16 | [+2.71, +5.67] |
| R-2 | +4.61 | [+2.44, +6.70] |
| R-L | +4.91 | [+3.33, +6.51] |

## Interpretation and decision

The pilot establishes engineering feasibility and a realistic full-run cost,
but not a fair native word-control comparison. DELTA controls characters, and
the train-derived conversion does not guarantee the 643-word upper bound.
Moreover, most long judgments are only partially visible to the generator.

The recommended decision is not to spend another estimated 28.32 GPU hours on
the full 703-document run with the current settings. Either retain the pilot as
a resource-limited diagnostic with all adaptations disclosed, or preregister a
new protocol that addresses both word-budget compliance and source coverage
before a full run.

