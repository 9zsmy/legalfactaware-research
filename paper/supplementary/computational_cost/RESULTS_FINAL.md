# Computational Cost Analysis — Final Results

完成日期：2026-10-26（Asia/Shanghai）

## A. Complete pipeline cost

The complete pipeline starts from the shared segmented CAIL2020-SFZY-clean
JSONL files and includes method-specific preparation, training, and prediction
for all 942 development documents. Values are mean ± sample standard deviation
over three successful complete runs.

| Method | Run 1 (min) | Run 2 (min) | Run 3 (min) | Mean ± SD (min) |
|---|---:|---:|---:|---:|
| SSE | 14.48 | 14.42 | 14.37 | **14.42 ± 0.06** |
| LegalFactAware | 31.28 | 31.17 | 31.10 | **31.18 ± 0.09** |
| SENDE | 34.18 | 34.02 | 34.38 | **34.19 ± 0.18** |
| MemSum | 96.10 | 95.33 | 96.15 | **95.86 ± 0.46** |
| TermDiffuSum | 155.57 | 155.52 | 155.47 | **155.52 ± 0.05** |

LegalFactAware was 8.81% faster than SENDE, 67.47% faster than MemSum, and
79.95% faster than TermDiffuSum, but required 2.16 times the pipeline time of
SSE. It therefore ranked second among the five audited implementations.

## B. Unified-start inference cost

Models and training-only fixed resources were loaded once and warmed up using
an independent training document. Each of five measured repeats re-read the
untouched segmented development JSONL, rebuilt all development-dependent
representations, scored and decoded the 942 documents, and wrote a separate
prediction file. Model loading is excluded from the main inference time and
reported separately.

| Method | Inference on 942 documents (s) | Model load (s) | Mean per document (ms) |
|---|---:|---:|---:|
| SSE | **4.00 ± 0.04** | 0.28 | 4.25 |
| LegalFactAware | **16.03 ± 0.06** | 1.00 | 17.02 |
| SENDE | **19.52 ± 0.38** | 0.68 | 20.72 |
| MemSum | **35.78 ± 0.19** | 1.60 | 37.98 |
| TermDiffuSum | **1,341.11 ± 4.11** | 4.16 | 1,423.69 |

LegalFactAware again ranked second. Its unified-start batch inference was about
17.9% faster than SENDE and 55.2% faster than MemSum, but about 4.0 times slower
than SSE. TermDiffuSum required about 22.35 minutes per 942-document repeat.

## Integrity audit

- Complete pipelines: 15/15 successful audited runs; 942 predictions each.
- Unified inference: 25/25 successful audited runs; 942 predictions each.
- Empty summaries: 0.
- Outputs longer than 450 Chinese characters: 0.
- Recorded hard truncations: 0.
- The first SENDE pipeline attempt failed because a required base-model symlink
  pointed to a removed cache. The exact official base model was restored, and
  the failed attempt was excluded; three later successful runs form the table.
- An initial TermDiffuSum inference launch failed during warm-up because Python
  3.7 does not support `Path.unlink(missing_ok=True)`. It generated no measured
  repeat and was excluded. A later user-requested shutdown interrupted another
  attempt before its in-memory aggregate timings were saved. The final five
  reported repeats were rerun from scratch and persisted individually.

## Claim boundary

These measurements support a claim that LegalFactAware has the second-lowest
pipeline and batch-inference cost among the five audited implementations on the
frozen server and configuration. They do not establish globally optimal
efficiency, single-request latency, energy efficiency, or superiority to
methods omitted because no comparable full implementation was available.
