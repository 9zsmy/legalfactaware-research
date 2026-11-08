# IN-Abs Phase 5D: SENDE independent reconstruction transfer

Frozen before running IN-Abs TSDAE training. This is a paper-guided independent
reconstruction, not an official SENDE implementation or checkpoint.

- Data: seed-42 6,325/703 IN-Abs train/dev split; official 100-document test
  content remains unread. Exclude the two previously identified empty pairs.
- Candidate protocol: the same reference-free, evenly spaced maximum of 400
  sentences used by E1 and the completed IN-Abs baselines.
- Sentence supervision: frozen ordinary greedy Oracle labels from Phase 2.
  Although SENDE's paper describes a different recursive weighted-ROUGE Oracle,
  this common-label adaptation does not claim a reproduction of that labeler.
- Encoder: `nlpaueb/legal-bert-base-uncased`, TSDAE trained only on the 6,325
  training documents' candidate sentences. One epoch, batch size 64, max 128
  tokens, learning rate 2e-5, seed 20260826; the same reconstruction choices
  previously used in the CAIL/BillSum SENDE work.
- Extractor: six-layer dilated gated convolution network from the existing
  independent SENDE reconstruction, hidden size 384, dilations
  `[1, 2, 4, 8, 1, 1]`, batch size 64, dropout 0.1. Train only on the 6,325
  training documents, with a fixed internal 5% training-only validation split
  for early stopping. Do not use the 703 comparison documents for model choice.
- Decoding: retain the paper's probability threshold 0.2. Rank eligible
  complete sentences by score and add only those fitting the remaining
  per-document reference-summary spaCy word budget. Restore source order.
  If thresholding yields no selected sentence, use the highest-scoring
  complete sentence that fits. No hard truncation or 8-sentence limit.
- Evaluation: `py-rouge==1.1` ROUGE-1/2/L F1 macro-average on the frozen 703
  documents, zero budget violations, and 10,000 paired bootstrap samples
  against frozen Phase 3C E1.
- Storage: use the system disk for new checkpoints and sentence arrays because
  the data disk has less than 3 GB free; do not delete existing assets.
