from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


SOURCE = Path(
    "/Users/admin/Downloads/LegalFactAware_English_Manuscript_Revised_20261010.docx"
)
OUTPUT = Path(
    "/Users/admin/Documents/ChatGPT/1/论文写作/英文论文/"
    "LegalFactAware_English_Manuscript_Reproducibility_Audited_20261010.docx"
)


def find_paragraph(doc: Document, prefix: str) -> Paragraph:
    matches = [p for p in doc.paragraphs if p.text.strip().startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one paragraph beginning {prefix!r}, found {len(matches)}")
    return matches[0]


def clear_paragraph(paragraph: Paragraph) -> None:
    for child in list(paragraph._p):
        if child.tag.endswith("}pPr"):
            continue
        paragraph._p.remove(child)


def set_paragraph(paragraph: Paragraph, text: str) -> None:
    clear_paragraph(paragraph)
    paragraph.add_run(text)


def set_labelled(paragraph: Paragraph, label: str, body: str) -> None:
    clear_paragraph(paragraph)
    label_run = paragraph.add_run(label)
    label_run.bold = True
    paragraph.add_run(" " + body)


def replace_text(paragraph: Paragraph, old: str, new: str) -> None:
    full = paragraph.text
    if old not in full:
        raise RuntimeError(f"Missing text {old!r} in paragraph {full[:120]!r}")
    set_paragraph(paragraph, full.replace(old, new))


def insert_after(paragraph: Paragraph, text: str, style: str | None = None) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    inserted = Paragraph(new_p, paragraph._parent)
    if style:
        inserted.style = style
    inserted.add_run(text)
    return inserted


def main() -> None:
    doc = Document(SOURCE)

    phrase_replacements = [
        (
            "Fixed-budget legal summarization requires selecting useful content",
            "discourse cues have a smaller stable effect",
            "discourse cues have a small, metric-dependent effect",
        ),
        (
            "Experiments on CAIL2020-SFZY-clean assess both performance",
            "Section6 has a small stable gain",
            "Section6 has a small, metric-dependent effect",
        ),
        (
            "Evidence separating mechanisms from auxiliary choices.",
            "small and statistically unsupported feature effects",
            "small and metric-dependent feature effects",
        ),
        (
            "LegalFactAware adopts domain guidance through Legal14 and Section6",
            "a small stable contribution from Section6",
            "a small, metric-dependent contribution from Section6",
        ),
        (
            "The controlled results explain the method more directly",
            "Section6 adds a smaller structural benefit",
            "Section6 adds a small, metric-dependent mean improvement, with Holm-corrected support limited to ROUGE-2 and ROUGE-L",
        ),
        (
            "The frozen CAIL2020-SFZY-clean results show competitive performance",
            "Section6’s small stable effect",
            "Section6’s small, metric-dependent effect",
        ),
    ]
    for prefix, old, new in phrase_replacements:
        replace_text(find_paragraph(doc, prefix), old, new)

    set_labelled(
        find_paragraph(doc, "Lexical content."),
        "Lexical content.",
        "Character 2–4-gram TF–IDF represents each sentence after prepending the "
        "zero-based marker “position{index}” and one space. The vectorizer uses a "
        "character analyzer, lowercase conversion, sublinear term frequency, smoothed "
        "inverse document frequency, L2 normalization, minimum document frequency 2, "
        "no maximum-document-frequency pruning beyond 1.0, and at most 120,000 "
        "float64 features. It is fitted only on training candidates and frozen for "
        "development and test. Because the position marker is encoded in the same "
        "block, removing this block removes lexical and positional evidence together. "
        "The domain features below supplement it.",
    )

    set_paragraph(
        find_paragraph(doc, "The classifier uses inverse-frequency class weights"),
        "The classifier uses inverse-frequency class weights and L2-regularized "
        "logistic regression with C = 1.0, the liblinear solver, an intercept, "
        "tolerance 10⁻⁴, at most 1,000 iterations, and random seed 42. Continuous "
        "probabilities are retained for ranking. This simple model permits component "
        "substitutions without changing the learning architecture. Class balancing is "
        "used for sparse positives; its independent contribution is evaluated rather "
        "than assumed. Crucially, the static sentence score is fixed for a given "
        "document and does not update after each extraction decision.",
    )

    set_labelled(
        find_paragraph(doc, "Data and constraints."),
        "Data and constraints.",
        "CAIL2020-SFZY-clean is our cleaned partition of the official CAIL2020 judicial "
        "summarization resources [12]. It contains 7,495 training, 942 development, "
        "and 948 test judgments. The test set contains 53,318 candidate sentences, "
        "averaging 56.24 per judgment. Our corpus audit detected no empty documents or "
        "cross-split overlaps in identifiers, exact texts, or near duplicates. Table 4 "
        "in Appendix C gives full statistics. The 450-character limit is implemented "
        "as Python string length on the concatenation of the cleaned, stored candidate-"
        "sentence strings; whitespace is not removed for this count. The protocol also "
        "permits at most eight selected sentence indices. LegalFactAware requires "
        "complete source sentences; the tail-truncation control deliberately relaxes "
        "this requirement to test its effect.",
    )

    metrics = find_paragraph(doc, "Metrics and implementation.")
    set_labelled(
        metrics,
        "Metrics and implementation.",
        "ROUGE-1/2/L F1 is computed with Lawa 1.2.6. Each text is segmented with "
        "lawa.cut and blank tokens are discarded. ROUGE-1 and ROUGE-2 use clipped "
        "multiset overlap, ROUGE-L uses full-sequence longest common subsequence, and "
        "the reported score is the macro-average of document-level F1. The evaluator "
        "applies no stemming, stop-word removal, smoothing, or sentence-level ROUGE "
        "aggregation. The Oracle’s character-level objective is separate. Rule-based "
        "Fact Recall covers four regular-expression groups—amount, date, law name or "
        "statutory article, and decision/action—and macro-averages over reference groups "
        "that are nonempty for a document; a document with no detected reference group "
        "receives 1.0. Source Fact Support uses the corresponding nonempty predicted "
        "groups and assigns 1.0 to an empty prediction. These diagnostics measure "
        "surface coverage and output validity rather than expert legal correctness.",
    )

    weak = insert_after(metrics, "", style="Body Text")
    set_labelled(
        weak,
        "Weak-supervision control.",
        "Independent weak supervision marks a training sentence positive only when "
        "its character-bigram ROUGE-2 precision against the reference is strictly "
        "greater than 0.1. The rule labels 66.54% of the 428,766 training candidates "
        "(38.06 positives per document), compared with 11.00% and 6.29 sentences per "
        "document for Oracle membership. This control therefore changes both label "
        "construction and target density; it does not isolate the budget constraint alone.",
    )

    set_labelled(
        find_paragraph(doc, "Statistical analysis."),
        "Statistical analysis.",
        "Development significance was recomputed from 942 aligned document-level "
        "records on 10 October 2026. We use 10,000 paired bootstrap resamples with "
        "seed 42 for 95% percentile intervals and two-sided paired Wilcoxon signed-rank "
        "tests with Pratt zero handling. Wilcoxon p-values are Holm-corrected across "
        "the 15 primary ROUGE comparisons (five core variants × three metrics). A "
        "result is called supported only when the 95% interval excludes zero and the "
        "Holm-adjusted p-value is below 0.05. Test-table baselines without a restorable "
        "aligned prediction bundle are compared by means only. Differences can differ "
        "by 0.01 from subtraction of rounded table values.",
    )

    set_paragraph(
        find_paragraph(doc, "Appendix C reports the archived paired comparison"),
        "Appendix C preserves archived test-set means for the earlier LegalFactAware-v2, "
        "CAIL-adapted MemSum, and an independent SSE-v2 implementation. The currently "
        "available package does not contain the exact aligned prediction bundle and "
        "frozen final model required to recompute those test-level paired statistics. "
        "Accordingly, these entries support mean comparisons only. SSE-v2 is distinct "
        "from the SSE proxy in Table 1 and must not be substituted for it.",
    )

    set_paragraph(
        find_paragraph(doc, "Supervision contributes substantially."),
        "Supervision contributes substantially. Independent sentence-level weak labels "
        "reduce ROUGE-1/2/L by 6.24/8.28/8.68 points. The recomputed paired 95% "
        "intervals are [5.75, 6.72], [7.65, 8.87], and [8.05, 9.27], and all three "
        "Wilcoxon tests remain supported after Holm correction. Fact Recall falls by "
        "10.30 points. This supports the complete budget-consistent label-construction "
        "procedure over the tested weak-label alternative. It does not isolate the "
        "contribution of the budget constraint from all other differences between "
        "summary-membership and independent weak supervision. A matched summary-level "
        "Oracle with a changed budget would be required for that narrower attribution.",
    )

    set_paragraph(
        find_paragraph(doc, "The position-augmented text block is the main representation."),
        "The position-augmented text block is the main representation. Removing the "
        "TF–IDF block reduces ROUGE by 9.21/10.56/10.73 points; the recomputed paired "
        "95% intervals are [8.68, 9.74], [9.88, 11.24], and [10.09, 11.38], with all "
        "three Wilcoxon tests supported after Holm correction. Fact Recall falls by "
        "10.43 points. Since the input includes a source-position marker, this ablation "
        "removes encoded position as well as character n-grams. Legal and positional "
        "signals alone cannot replace the lexical evidence supplied by sentence text. "
        "The finding explains why domain indicators are auxiliary in our model; it does "
        "not establish superiority over contextual representations that were not "
        "substituted within this controlled experiment.",
    )

    set_paragraph(
        find_paragraph(doc, "Domain cues have unequal effects."),
        "Domain cues have unequal effects. Removing Section6 reduces ROUGE by "
        "0.28/0.44/0.43 points, with recomputed paired 95% intervals of [0.10, 0.45], "
        "[0.20, 0.68], and [0.20, 0.67]. After Holm correction, ROUGE-2 and ROUGE-L "
        "remain supported, but ROUGE-1 does not (adjusted p = 0.142). Removing Legal14 "
        "changes the metrics by only 0.07/0.08/0.03 points; all intervals cross zero "
        "and no adjusted test supports a gain. Removing both lowers ROUGE by "
        "0.37/0.51/0.57; Table 7 in Appendix D reports the full feature combinations. "
        "Thus Section6 has a small, metric-dependent effect, whereas a standalone "
        "Legal14 gain is not independently established.",
    )

    set_paragraph(
        find_paragraph(doc, "Fit/skip decoding preserves measurable content"),
        "Fit/skip decoding preserves measurable content as well as boundaries. Replacing "
        "it with post-hoc tail truncation reduces ROUGE by 0.88/0.74/1.19 points and "
        "Fact Recall by 4.06. The recomputed paired 95% intervals are [0.65, 1.12], "
        "[0.45, 1.02], and [0.87, 1.51], and all three Wilcoxon tests remain supported "
        "after Holm correction. Truncation occurs in 379 of 942 outputs, compared with "
        "zero for the complete-sentence decoder. Thus, the tested fit/skip strategy "
        "improves automatic coverage measures while eliminating partial-sentence output. "
        "The experiment does not verify expert-assessed legal correctness.",
    )

    set_labelled(
        find_paragraph(doc, "Reproducibility details."),
        "Reproducibility details.",
        "Appendix F records the frozen split hashes, experiment identifier, model and "
        "per-document artifact hashes, weak-label rule, evaluator settings, and "
        "statistical protocol. The 942-document development controls were independently "
        "rebuilt and recomputed in this audit. The remaining gap concerns the archived "
        "948-document test comparison: aggregate files are traceable, but the exact "
        "final prediction bundle and expected frozen model file are not present in the "
        "current local package, so test-set significance is not reasserted. External "
        "baseline adaptations additionally require their disclosed code and checkpoints.",
    )

    set_paragraph(
        find_paragraph(doc, "Tables 4 and 5 provide corpus statistics"),
        "Tables 4 and 5 provide corpus statistics and executable-system identities. "
        "Table 6 preserves an archived aligned mean comparison; it is not used for a "
        "fresh inferential claim in this audited version.",
    )
    set_paragraph(
        find_paragraph(doc, "Table 6: Strict paired comparison"),
        "Table 6: Archived aligned mean comparison of core systems on the frozen test set (%).",
    )
    set_paragraph(
        find_paragraph(doc, "The archived paired analysis reports significant sign-flip comparisons"),
        "Table 6 preserves archived aligned means for LegalFactAware-v2, CAIL-adapted "
        "MemSum, and SSE-v2. The current package does not contain the complete matched "
        "prediction bundle and exact frozen final model needed to rerun the test-level "
        "sign-flip analyses. We therefore make no new significance claim from Table 6. "
        "SSE-v2 is an independent implementation and is distinct from the SSE structural "
        "proxy in the principal test table. These means do not establish significance "
        "against SENDE. Sentence F1 measures sentence-selection agreement rather than "
        "contextual correctness.",
    )

    appendix = doc.add_paragraph(style="Heading 1")
    appendix.add_run("Appendix F Reproducibility and Batch Record")

    p = doc.add_paragraph(style="Body Text")
    set_labelled(
        p,
        "Frozen development batch.",
        "The audited development ablation is experiment "
        "LegalFactAware-v2.5-B-formal-dev-ablation-20260821, with 7,495 training and "
        "942 development documents and 428,766 training candidates. SHA-256 hashes of "
        "the cleaned JSONL splits are train "
        "8b19e6a1aa086e55c3a1d694e86cb4cf5abfe6929ee1258a3621b291bde564a2, "
        "development 1a056c68dcfa5f11e9574d5f1b818fa475bb5607ea7f6565e2868072a55de2c0, "
        "and test 5e9208643bb66bfdb0ac735f542cc64214668e53209d9373c07533997bb2490d. "
        "The full-model development per-document record has SHA-256 "
        "fba8edc13449413e446fd185f34ffab4764a79a65c4d4e71a0a8c379d60abf08; "
        "the rebuilt full model has SHA-256 "
        "0858b3e80f641336ead5e70c0a4343e254677c72b2bfc9427170564744648dab. "
        "The position-augmented TF–IDF model used by the ablation has SHA-256 "
        "2cc0c9fe4b1fdaca6c1b2e37e545fcece3e5da83533296c1c12ba1d8bb8f70e9.",
    )

    p = doc.add_paragraph(style="Body Text")
    set_labelled(
        p,
        "Weak-label rule.",
        "For every training candidate, whitespace is removed before constructing "
        "character bigrams. A sentence is positive exactly when clipped character-"
        "bigram overlap divided by the number of candidate bigrams is strictly greater "
        "than 0.1. This produces 66.54% positive candidates (38.06 per document), "
        "whereas the Oracle membership labels produce 11.00% (6.29 per document).",
    )

    p = doc.add_paragraph(style="Body Text")
    set_labelled(
        p,
        "Evaluation configuration.",
        "Primary ROUGE uses Lawa 1.2.6 tokenization, clipped multiset unigram and bigram "
        "overlap, full-sequence LCS for ROUGE-L, document-level F1, and macro-averaging. "
        "No stemming, stop-word removal, smoothing, or sentence-level aggregation is "
        "applied. The selection budget uses Python string length on concatenated cleaned "
        "candidate strings, without removing whitespace. Fact Recall and Source Fact "
        "Support use the four disclosed surface-pattern groups: amount, date, law or "
        "article, and decision/action.",
    )

    p = doc.add_paragraph(style="Body Text")
    set_labelled(
        p,
        "Recomputed inference.",
        "All 942 aligned development records were reevaluated on 10 October 2026. "
        "Intervals use 10,000 paired bootstrap resamples with seed 42. Tests use a "
        "two-sided paired Wilcoxon signed-rank test with Pratt zero handling, followed by "
        "Holm correction across 15 primary ROUGE tests (five variants × three metrics). "
        "Claims require both a 95% interval excluding zero and an adjusted p-value below "
        "0.05. This rule changes the Section6 interpretation: ROUGE-2 and ROUGE-L are "
        "supported, but ROUGE-1 is not after correction.",
    )

    p = doc.add_paragraph(style="Body Text")
    set_labelled(
        p,
        "Remaining test-batch limitation.",
        "The frozen test aggregate remains traceable to 948 document identifiers, and "
        "the test split hash above is verified. However, the exact aligned prediction "
        "bundle and the expected final model artifact (recorded SHA-256 "
        "65fc714698f22da944cab62cbc4d28022181c31fea4f9c3f783ee11f8d776539) "
        "are absent from the current local package. The audited manuscript therefore "
        "retains the reported test means but does not claim newly recomputed test-level "
        "significance.",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
