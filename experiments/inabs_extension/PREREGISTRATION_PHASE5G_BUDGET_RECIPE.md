# IN-Abs Phase 5G: released Budget Recipe checkpoint, zero-shot transfer

This is a **zero-shot released-checkpoint transfer**, not the paper's full
SFT+RLVR training recipe and not an extractive baseline. The official IN-Abs
100-case test partition stays sealed.

- Model: `VISAI-AI/Qwen3-4B-Instruct-2507-L-SUMM-fourcorners-rl-r2-ckpt500`,
  revision `2ccba598a3ca7e0491067f9950364e719f145fa1`. No IN-Abs train
  examples are used for tuning.
- Inputs: frozen 703 dev IDs, using exactly the 400 reference-free candidate
  sentences per case from the shared Phase 5 data. Prompts use the repository's
  unmodified English `system.md` and `summarize_nothink.md` templates. The
  reference text is never included in the prompt; its spaCy word count is used
  only as the output ceiling, as in the common official-protocol track.
- Decoding: deterministic greedy generation, at most 1,024 new tokens. Use
  exact chunked prefill from the existing reproduction when the prompt exceeds
  12,000 tokens, without dropping input content.
- Postprocess: sentence-segment generated English text, keep only segments
  ending in sentence-final punctuation, and take the longest generation-order
  prefix that fits the per-reference spaCy punctuation-filtered word budget.
  Never hard-truncate a segment. Report empty outputs, limit hits and unused
  budget.
- Evaluate `py-rouge==1.1` macro R1/R2/RL on all 703 dev cases, including
  zero outputs if any, plus 10,000 paired bootstrap replicates against E1.
  Require zero budget violations. This is a separate generative/zero-shot row.
- Model files may be staged in `/dev/shm` because the cloned server's persistent
  disks have insufficient free capacity; pinned revision and file integrity
  checks remain mandatory.
