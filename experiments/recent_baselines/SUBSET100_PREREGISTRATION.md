# Recent-method subset comparison preregistration

Frozen: 2026-11-01, before any model was run on the selected subset.

## Purpose

Run a resource-limited comparison of LegalFactAware, DELTA, and Black-box LCG
on a fixed subset.  This is a **subset comparison**, not a full 942-document
reproduction and not an official reproduction on the papers' original tasks.

## Data

- Population: frozen CAIL clean development set, 942 documents.
- Subset: 100 documents.
- Seed: `20261101`.
- Selection: proportional allocation across the 16 joint strata formed by
  source-length quartiles and reference-length quartiles.  Within each stratum,
  rows are ordered by `SHA256(seed:id)`.  The selected rows remain in original
  dataset order.
- Model outputs and quality metrics are not used in selection.
- Frozen subset SHA-256:
  `d918bc5a40e8e109f6ac1420a7e0e94d2f1d24ad233802074c3e84e9b324660d`.
- Manifest: `data/subset100/subset_manifest_100.json`.

## Methods and unchanged configurations

1. `LegalFactAware-v2.5-B`: frozen predictions restricted to the selected IDs.
2. `DELTA-Dream-CAIL-adapted`: the existing official-core adaptation, Dream
   7B, 4-bit, target 450 characters, frozen character/token ratio and thresholds.
3. `Qwen2.5-7B-DirectPrompt-one-shot`: the existing initial generator.
4. `LCG-MHIS-Qwen2.5-7B-CAIL-adapted`: upper-bound objective, at most five
   iterations, unchanged sampling parameters.
5. `LCG-MHIS-Qwen2.5-7B-CAIL-exact-exploratory`: exact-length diagnostic,
   unchanged sampling parameters.  This remains post-hoc exploratory because
   it does not match the task's hard upper-bound objective.

No prompts, thresholds, post-processing, generation settings, or selected IDs
will be changed after viewing subset outputs.  Predictions over 450 characters
remain untruncated and count as budget violations.

## Evaluation and reporting

- All methods are evaluated on the identical 100 document IDs.
- Report character ROUGE-1/2/L F1, Expression F1, source-expression support,
  output length, budget violation, and wall-clock time.
- Use LegalFactAware as the paired-bootstrap anchor.
- These results may appear in a separately labeled subset table.  They must not
  be mixed with 942-document results as though the sample sizes were identical.
- Five-document smoke results remain engineering diagnostics and are not pooled
  with the frozen 100-document comparison.

## Stop rules

- Stop a method for systematic empty/corrupt output, unrecoverable runtime
  failure, or insufficient disk/memory.  Preserve completed rows and logs.
- Engineering fixes may restore the frozen implementation but may not alter the
  method configuration.  Any configuration change requires a new labeled run.

