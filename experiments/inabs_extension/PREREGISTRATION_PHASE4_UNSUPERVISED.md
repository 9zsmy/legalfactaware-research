# IN-Abs Phase 4A preregistration: unsupervised published baselines

Frozen before the full 703-document development run.

- Data: official IN-Abs training partition only.
- Split: the existing deterministic seed-42 split, 6,325 train / 703 dev;
  empty pairs 299 and 4799 remain excluded.
- Official 100-document test contents must not be read.
- Candidate protocol: the same reference-free, evenly spaced maximum of 400
  source sentences used by the frozen E1 system.
- Length protocol: each development output is limited by its own reference
  summary length, counted with the official Gist spaCy
  `en_core_web_sm` punctuation-filtered token rule.
- Decoding: ranked complete sentences are greedily packed; a sentence that
  does not fit is skipped. Hard truncation is forbidden. Required violations:
  zero.
- Evaluation: `py-rouge==1.1`, ROUGE-1/2/L F1, document macro average.

## DSDR budget adaptation

- Linear data-reconstruction DSDR (He et al., 2012).
- Per-document sentence vectors: word unigram TF-IDF, English stop-word
  removal, L2 normalization.
- Reconstruction trade-off: lambda = 0.1, the public reference
  implementation's example/default setting.
- Selection follows the published rank-one reconstruction update. At each
  step, the best currently fitting sentence is selected; selection stops when
  no complete candidate sentence fits.

## CaseSummarizer budget adaptation

- Restricted faithful Python 3 reimplementation of the public
  CaseSummarizer code because the released Python 2 program is not directly
  executable in the current environment.
- Preserved components: collection document frequency, document-normalized
  TF-IDF sentence importance, digit indicator, named-entity density, and legal
  dictionary density with released `dictionary.txt`.
- spaCy supplies English lemmatization and named entities in place of the
  unavailable legacy NLTK models. This is an implementation adaptation and
  will not be described as an official reproduction.
- Feature weights are frozen to the released code: 0.2 digit, 0.3 entity,
  1.5 legal-dictionary density, multiplied by the within-document standard
  deviation of base sentence scores.

## Reporting

- Save per-document selected source indices, word counts, and ROUGE scores.
- Report macro means, 10,000-sample bootstrap confidence intervals, and paired
  differences against the frozen Phase 3C E1 outputs.
- No parameter selection is permitted from this 703-document run.
