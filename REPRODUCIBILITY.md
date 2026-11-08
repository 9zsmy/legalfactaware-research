# Reproducibility guide

## Evidence levels

Every method should be described using one of these identities:

1. **Official artifact/checkpoint evaluation** — upstream code or checkpoint used without method reimplementation.
2. **Official-core adaptation** — upstream core algorithm used, with task/model/protocol adaptations disclosed.
3. **Independent reimplementation** — reconstructed from a paper or algorithm description.
4. **Proxy adaptation** — inspired by a paper but not a full implementation.
5. **Engineering/smoke result** — verifies that a pipeline runs; not a paper-level quality conclusion.

Never collapse these categories in a result table.

## Dataset and evaluator separation

- CAIL test: 948 documents, paper LAWA ROUGE protocol.
- CAIL development controls: 942 documents, mechanism analysis.
- Recent-baseline unified evaluator: non-whitespace character ROUGE; it must not replace LAWA values.
- Frozen recent subset: 100 documents; it must not be called a full-development result.
- IN-Abs: development evidence only unless a later frozen official-test package is explicitly added.

## Recommended reproduction order

1. Run dependency-light unit tests.
2. Rebuild/freeze the development inputs from locally authorized data.
3. Verify the LegalFactAware reference output under the selected evaluator.
4. Run a small engineering smoke test for expensive baselines.
5. Expand only when the preregistered gate passes.
6. Save aggregate metrics, exact configuration, input hashes, and integrity counts.

## Main evidence documents

- `experiments/reviewer_controls/RESULTS.md`
- `experiments/cost/RESULTS_FINAL.md`
- `experiments/recent_baselines/RUN_STATUS.md`
- `experiments/recent_baselines/REPRODUCTION_AUDIT.md`
- `docs/results/PAPER_AND_EXPERIMENT_MASTER_CN.md`

