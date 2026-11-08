# Phase 3B: IN-Abs official-metric alignment

This phase does not load or train a model and does not inspect official-test
contents. It reconstructs the already generated E0/E1/E2 extracts for the
frozen 703-document development set's per-reference-length track.

Evaluation uses the `py-rouge==1.1` distribution linked by the AACL 2022 paper,
not the distinct `rouge==1.0.1` package that exposes the same Python module
name. Following the paper, ROUGE is initialized with `rouge-n` and `rouge-l`,
`max_n=2`, and internal length limiting disabled; all other parameters retain
the package defaults (stemming on, alpha 0.5, weight factor 1.0, compatibility
mode on). We report ROUGE-1/2/L F1, paired seed-42 bootstrap intervals, and
both regex-token and whitespace-token budget compliance.
