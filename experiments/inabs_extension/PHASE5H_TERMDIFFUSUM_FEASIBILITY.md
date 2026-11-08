# IN-Abs Phase 5H: TermDiffuSum feasibility audit (2026-09-18)

Status: **not launched as a main-protocol baseline**. No IN-Abs score has been produced.

The frozen IN-Abs comparison permits up to 400 reference-free, evenly spaced candidate sentences per document. The available TermDiffuSum compatibility run uses 200 article slots plus 40 summary slots, including fixed `generated[:, :200, :]` and `generated[:, 200:240, :]` slices. Its loader/model components are recovered Python 3.7 bytecode rather than editable source; the CAIL adapter and encoder path are Chinese-oriented. The [authors' public repository](https://github.com/huaand/TermDiffuSum-) states that the released code is partial.

On the frozen IN-Abs split, 938/6,325 train documents and 101/703 dev documents contain more than 200 candidates. Silently clipping TermDiffuSum to 200 while other methods see up to 400 changes the input problem for a material subset. Changing the entire experiment to 200 would require a separately preregistered, symmetric 200-candidate rerun of comparators; an English 400-slot reconstruction would require substantive architecture and encoder work and would no longer be a faithful direct migration.

Decision: leave TermDiffuSum as **not comparable / not run** in the 400-candidate main table. Do not use the official 100-document test to resolve this engineering limitation. If a 200-candidate sensitivity track is later requested, report it separately and label it as a restricted independent reconstruction.
