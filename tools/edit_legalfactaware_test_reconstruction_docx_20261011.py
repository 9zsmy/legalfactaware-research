#!/usr/bin/env python3
"""Update the manuscript with the audited CAIL test reconstruction."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


def insert_after(paragraph: Paragraph, text: str) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    result = Paragraph(new_p, paragraph._parent)
    if paragraph._p.pPr is not None:
        result._p.insert(0, deepcopy(paragraph._p.pPr))
    result.add_run(text)
    return result


def replace_by_prefix(document: Document, prefix: str, replacement: str) -> Paragraph:
    matches = [p for p in document.paragraphs if p.text.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one paragraph starting {prefix!r}, found {len(matches)}")
    paragraph = matches[0]
    paragraph.text = replacement
    return paragraph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    document = Document(args.source)

    replace_by_prefix(
        document,
        "Statistical analysis.",
        "Statistical analysis. Development significance was recomputed from 942 aligned "
        "document-level records on 10 October 2026. We use 10,000 paired bootstrap resamples "
        "with seed 42 for 95% percentile intervals and two-sided paired Wilcoxon signed-rank "
        "tests with Pratt zero handling. Wilcoxon p-values are Holm-corrected across the 15 "
        "primary development ROUGE comparisons (five core variants × three metrics). A result "
        "is called supported only when the 95% interval excludes zero and the Holm-adjusted "
        "p-value is below 0.05. A separately labeled test reconstruction uses the same interval "
        "and test definitions and corrects across nine ROUGE comparisons (three preserved "
        "baselines × three metrics). The exact historical final-model binary remains missing, "
        "so reconstructed paired results are evidence of computational robustness rather than "
        "proof of artifact identity. Test-table methods without restorable ID-aligned predictions, "
        "including SENDE, are compared by means only. Differences can differ by 0.01 from "
        "subtraction of rounded table values.",
    )

    replace_by_prefix(
        document,
        "LegalFactAware obtains 60.94/42.10/50.73 ROUGE",
        "LegalFactAware obtains 60.94/42.10/50.73 ROUGE and 70.99 Fact Recall, with no hard "
        "truncations or budget violations. Its ROUGE means are the highest in the extractive "
        "comparison. Against SENDE, the mean differences are 1.22/0.97/0.08 points. The small "
        "ROUGE-L margin and absence of aligned SENDE predictions support a competitive-performance "
        "statement, not a claim of statistically established superiority. In the separately "
        "labeled test reconstruction, all ROUGE differences versus the preserved LegalFactAware-v2, "
        "MemSum, and SSE-v2 records have bootstrap intervals above zero after Holm correction; "
        "this does not resolve the SENDE comparison or recover the missing historical model binary. "
        "SENDE’s missing Fact Recall also prevents a complete fact-coverage ranking.",
    )

    replace_by_prefix(
        document,
        "Appendix C preserves archived test-set means",
        "Appendix C preserves archived test-set means for the earlier LegalFactAware-v2, "
        "CAIL-adapted MemSum, and an independent SSE-v2 implementation. The exact historical "
        "v2.5-B model binary and its original prediction bundle were not recovered. We therefore "
        "generated a clearly labeled computational reconstruction from the verified split, "
        "frozen vectorizer, unchanged inference code, and independently rebuilt classifier. It "
        "reproduces the primary archived test aggregates to the reported precision and permits "
        "new paired checks against those three preserved, ID-aligned baselines. The reconstruction "
        "does not prove binary identity and cannot supply a paired SENDE comparison. SSE-v2 is "
        "distinct from the SSE proxy in Table 1 and must not be substituted for it.",
    )

    replace_by_prefix(
        document,
        "Reproducibility details.",
        "Reproducibility details. Appendix F records the frozen split hashes, experiment "
        "identifier, model and per-document artifact hashes, weak-label rule, evaluator settings, "
        "and statistical protocol. The 942-document development controls were independently "
        "rebuilt and recomputed in this audit. For the 948-document test comparison, the original "
        "final-model binary remains unavailable, but a deterministic reconstruction now restores "
        "an ID-aligned prediction and metric bundle whose primary aggregates reproduce the archive "
        "within 0.001 percentage points. Paired test evidence is consequently reported as a "
        "reconstruction audit, not as recovery of the original binary artifact. External baseline "
        "adaptations additionally require their disclosed code and checkpoints.",
    )

    replace_by_prefix(
        document,
        "Table 6 preserves archived aligned means",
        "Table 6 preserves archived aligned means for LegalFactAware-v2, CAIL-adapted MemSum, "
        "and SSE-v2. The exact historical final-model binary is still missing. Using the audited "
        "reconstruction, we regenerated 948 ID-aligned LegalFactAware-v2.5-B records and recomputed "
        "the nine ROUGE comparisons against these three baselines. Every 95% bootstrap interval "
        "remains above zero after Holm correction. The reconstructed mean differences deviate "
        "from the archived differences by at most 0.00056 percentage points. These checks support "
        "the robustness of the three comparisons but do not establish identity with the missing "
        "historical prediction file, significance against SENDE, or equivalence between SSE-v2 "
        "and the SSE structural proxy in Table 1. Sentence F1 measures sentence-selection "
        "agreement rather than contextual correctness.",
    )

    replace_by_prefix(
        document,
        "Tables 4 and 5 provide corpus statistics",
        "Tables 4 and 5 provide corpus statistics and executable-system identities. Table 6 "
        "preserves the archived aligned means. The exact historical v2.5-B prediction bundle is "
        "not treated as recovered; the separately labeled computational reconstruction is used "
        "only for the paired robustness checks described below.",
    )

    limitation = replace_by_prefix(
        document,
        "Remaining test-batch limitation.",
        "Test-batch artifact status. The verified test split contains 948 unique document "
        "identifiers and has SHA-256 "
        "5e9208643bb66bfdb0ac735f542cc64214668e53209d9373c07533997bb2490d. "
        "The exact historical final model expected at SHA-256 "
        "65fc714698f22da944cab62cbc4d28022181c31fea4f9c3f783ee11f8d776539 "
        "was not found in the local project, handoff archive, or model inventory. The original "
        "prediction bundle is likewise unavailable. We therefore do not describe the historical "
        "artifact as recovered and do not reassert paired significance against methods lacking "
        "ID-aligned predictions.",
    )
    insert_after(
        limitation,
        "Test-batch computational reconstruction. On 11 October 2026, we ran the unchanged "
        "v2.5 feature module and complete-sentence decoder with the verified v2.2 vectorizer and "
        "the independently rebuilt v2.5-B classifier (SHA-256 "
        "0858b3e80f641336ead5e70c0a4343e254677c72b2bfc9427170564744648dab) "
        "under Python 3.10.19, scikit-learn 1.7.2, SciPy 1.15.3, NumPy 2.2.6, and Lawa 1.2.6. "
        "The reconstructed prediction file (SHA-256 "
        "4c147fd6d849414ad36cd85151a5865d4cf4cf70a40784c68ce108c2e9571c62) "
        "obtains 60.9391/42.0963/50.7268 ROUGE and 70.9913 Fact Recall, preserving the reported "
        "60.94/42.10/50.73 and 70.99 values. The maximum raw ROUGE deviation from the archive is "
        "0.00056 percentage points; reconstructed Sentence F1 is 59.82 rather than the archived "
        "59.83. Against the preserved LegalFactAware-v2, MemSum, and SSE-v2 per-document records, "
        "all nine reconstructed ROUGE differences have 95% bootstrap intervals above zero and "
        "Holm-adjusted p-values below 0.05. SENDE remains mean-only because its aligned predictions "
        "are absent. The reconstruction manifest has SHA-256 "
        "1de68cc06b3dcffd9e30d9cc23176680f7f7210ca41349f805e04da7fc4f8198, "
        "and the paired-comparison record has SHA-256 "
        "6eadc832c5dbfc9c1d0943c9cd0502df55d1cb78d1f6576187b65f0f2eaf6db8.",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document.save(args.output)
    print(args.output)
    print(f"paragraphs={len(document.paragraphs)} tables={len(document.tables)}")


if __name__ == "__main__":
    main()
