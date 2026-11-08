# IN-Abs Phase 5F: CivilSum LED-tail generative adaptation

This is a separate generative comparison, **not** a complete-sentence
extractive baseline and not an official CivilSum reproduction. Official IN-Abs
test contents must remain unread.

- Frozen 6,325/703 IN-Abs train/dev IDs, using the same 400-candidate source
  sentences already prepared for the extractive track. No test records are
  loaded or named in the runner.
- Initialize from the cached English `allenai/led-base-16384` checkpoint. Keep
  the CivilSum-style left/tail 1,024-token input and beam-3 LED generation;
  target/generation cap 512 tokens.
- Single RTX 4090 D: per-device batch 4, accumulation 32, effective batch 128,
  learning rate 1e-4, seed 2024, at most 1,500 updates. Evaluate/save at steps
  500, 1,000 and 1,500; select by dev mean ROUGE-1 after budget packing.
- For each dev document, use its reference's spaCy punctuation-filtered word
  count as an output ceiling. Segment the generated text with spaCy and pack
  whole segments in order, skipping any that do not fit. Never cut a segment
  mid-word or mid-sentence. Report unused budget and empty outputs.
- Final 703-dev result uses `py-rouge==1.1`, R1/R2/RL macro F1 and paired
  bootstrap against E1, with zero budget violations.
- Because the input is token-tail truncated and generation is not extractive,
  publish this row only in a labeled generative block, not in the main
  extractive ranking. Report input truncation rate and generated length.
