# Unified frozen-model inference benchmark

Date: 2026-10-21  
Dataset: CAIL development set, 942 documents  
Hardware: the same server with one NVIDIA RTX 4090 D  
Repeats: 3 per method  
Reported time: median wall-clock time over the three runs

| Method | Run 1 (s) | Run 2 (s) | Run 3 (s) | Median (s) | Throughput (docs/s) | Peak GPU memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| LegalFactAware | 17.081 | 17.070 | 17.467 | 17.081 | 55.149 | 0 (CPU) |
| SSE | 5.159 | 5.007 | 5.047 | 5.047 | 186.646 | 4,745 |
| MemSum | 35.449 | 35.784 | 35.891 | 35.784 | 26.325 | 2,229 |
| SENDE | 5.134 | 5.076 | 5.048 | 5.076 | 185.579 | 473 |
| TermDiffuSum | 1,314.644 | 1,325.366 | 1,326.562 | 1,325.366 | 0.711 | 5,737 |

All 15 runs exited with code 0 and produced predictions for all 942 documents.

## Interpretation

- LegalFactAware is CPU-only in this benchmark. Its median inference time is 2.10x faster than MemSum and 77.59x faster than TermDiffuSum.
- GPU-batched SSE and SENDE have lower measured wall-clock time, but they consume prepared inputs; SENDE additionally excludes sentence-encoder computation because it reads cached embeddings.
- The table therefore supports a comparison of frozen-model inference under each archived method's available input representation. It is not a strictly equal raw-text-to-summary end-to-end latency comparison.
- LegalFactAware's 1 MiB `nvidia-smi` reading was the idle device baseline and is reported as 0 MiB because the method ran on CPU.

## Reproducibility checks

- Same 942-document development set for every method.
- Three independent timed runs per method.
- Every run produced exactly 942 prediction records.
- Every run exited successfully.
- TermDiffuSum's temporary CAIL data links were restored to the original IN-Abs paths after the benchmark.
