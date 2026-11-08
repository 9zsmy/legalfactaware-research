# IN-Abs Phase 5C: Agarwal-ST restricted transfer

Frozen before resuming the interrupted embedding cache and starting training.

- Data: the existing seed-42 split of 6,325 train and 703 development cases;
  official test contents remain unread.
- Candidates: the reference-free, evenly spaced maximum-400 sentences already
  prepared for the MemSum/Agarwal transfer.
- Supervision: the already frozen P50 (643-word) budget-oracle sentence labels
  in `phase5c_agarwal_st_data/train.jsonl`. This differs from E1's ordinary
  oracle supervision and must be disclosed in comparisons.
- Sentence representation: frozen `nlpaueb/legal-bert-base-uncased`, masked
  mean pooling of the last hidden layer, maximum 512 wordpieces, float16 cache.
- ST scorer: one bidirectional GRU, hidden size 128 per direction, dropout 0.5,
  two-class output, class-weighted cross entropy.
- Training: seed 42, Adam learning rate 0.00261, batch size 8, five epochs. The
  final epoch is used; development ROUGE is not used for checkpoint selection.
- Decoder: Agarwal MMR-Select with lambda 0.8. At every step only complete
  sentences fitting the remaining budget are eligible. Selected sentences are
  restored to source order.
- Budget: each development case uses its own reference-summary word count under
  the official Gist spaCy `en_core_web_sm` punctuation-filtered counter.
- Evaluation: `py-rouge==1.1`, macro ROUGE-1/2/L F1, zero budget violations,
  and 10,000 paired bootstrap samples against frozen Phase 3C E1.
- Identity: restricted independent Agarwal-ST transfer, not an official author
  checkpoint reproduction.

