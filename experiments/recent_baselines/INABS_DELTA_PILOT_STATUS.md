# DELTA–Dream IN-Abs pilot status

Updated: 2026-11-05 13:20 CST

## Frozen protocol

- Dataset: frozen 703-document IN-Abs development split; official test unread.
- Pilot: deterministic 50-document subset selected by
  `SHA256(delta-inabs-pilot-v1:<case_id>)`.
- Length setting: 643 words, converted with the training-only median ratio to
  a 3,684-character DELTA target.
- Dream token calibration: 4.4699386 characters/token from a deterministic
  256-reference training subset.
- Model: Dream-v0-Instruct-7B, 4-bit; official DELTA decoding core.
- Maximum iterations: 1,200; source bound: 8,192 characters, head and tail.

## Launch state

- One-case smoke test passed: 57.33 seconds, 632 words, 127 iterations, zero
  remaining masks, 7.58 GiB peak allocated GPU memory, exit code 0.
- The 50-document run was launched in detached screen session
  `delta_inabs_pilot50` on 2026-11-05.
- At the first checkpoint, 4/50 documents were complete. Mean runtime was
  61.10 seconds/document, median 53.63, and all four finished with zero masks.
- Preliminary projected generation time for all 703 documents at that mean is
  11.93 GPU hours. This is provisional until the 50-document run completes.
- Preliminary length: mean 630.25 words, mean absolute error 20.25 words, and
  1/4 outputs above 643 words.
- Source truncation occurred in 3/4 initial cases and must be disclosed in any
  quality interpretation.

## Later checkpoint

- 9/50 documents complete; GPU remained fully utilized with no OOM.
- Mean/median runtime increased to 124.71/94.71 seconds because later cases
  required up to 608 iterations.
- Updated estimated completion: approximately 2026-11-05 12:46 CST; projected
  703-document generation cost at the observed mean: 24.35 GPU hours.
- All nine outputs finished with zero remaining masks. Mean output length was
  644.44 words; 5/9 exceeded 643 words. Eight of nine sources were truncated.

## Mid-run checkpoint

- 28/50 documents complete (56%); GPU utilization remained 100% with no OOM.
- Mean/median runtime: 151.30/149.04 seconds; updated completion estimate:
  approximately 2026-11-05 13:06 CST.
- Projected 703-document generation cost: 29.55 GPU hours at the observed mean.
- All 28 outputs finished with zero remaining masks; maximum observed
  iterations: 657.
- Mean output length: 635.82 words; mean absolute error: 21.04 words;
  12/28 (42.9%) exceeded the 643-word upper bound.
- 24/28 sources were truncated to the 8,192-character head-tail input.

## Final result

- Completed 50/50 documents. Generation and feasibility evaluation exit codes
  were both 0; no OOM and no unfinished output.
- Total generation time: 2.014 GPU hours. Mean/median runtime:
  145.03/153.10 seconds. Projected 703-document cost: 28.32 GPU hours.
- Mean/median output length: 634.12/633.5 words. Mean/median absolute error to
  643 words: 19.56/16.0. Word-budget violations: 17/50 (34%).
- Source truncation: 41/50 (82%). Mean peak allocated GPU memory: 7.42 GiB.
- On the same 50 documents with the same frozen py-rouge==1.1-compatible
  evaluator, DELTA obtained R-1/R-2/R-L 52.57/28.63/48.59. The existing E1
  fixed-643-word system obtained 56.73/33.24/53.51.
- Paired E1-minus-DELTA mean differences were +4.16/+4.61/+4.91 points. Their
  95% bootstrap intervals were respectively [2.71, 5.67], [2.44, 6.70], and
  [3.33, 6.51] points; all exclude zero.
- Decision: do not automatically launch the 703-document run under the current
  adaptation. First resolve strict word-budget control and the 82% source-
  truncation rate, or retain this 50-document result as a clearly labeled
  resource-limited diagnostic.

## Remote paths

- Root: `/root/autodl-tmp/legalfactaware_recent_baselines_run/experiments/recent_baselines_20261031`
- Predictions: `results/inabs_delta_pilot50/delta_dream_inabs_643w_pilot50.jsonl`
- Log: `results/inabs_delta_pilot50/pilot50.log`
- Final feasibility metrics: `results/inabs_delta_pilot50/feasibility_metrics.json`
- Exit codes: `results/inabs_delta_pilot50/pilot50.exitcodes`

