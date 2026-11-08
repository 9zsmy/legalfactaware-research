from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path("/Users/admin/Documents/ChatGPT/1")
SRC = ROOT / "论文写作/英文论文/LegalFactAware_English_Manuscript_Evidence_Audited_20261011.docx"
OUT = ROOT / "论文写作/英文论文/LegalFactAware_English_Manuscript_INABS_Added_20261017.docx"


def paragraph_by_exact_text(doc, text):
    for paragraph in doc.paragraphs:
        if paragraph.text == text:
            return paragraph
    raise ValueError(f"Paragraph not found: {text}")


def replace_in_paragraph(paragraph, old, new):
    for run in paragraph.runs:
        if old in run.text:
            run.text = run.text.replace(old, new)
            return
    full = paragraph.text
    if old not in full:
        raise ValueError(f"Text not found in paragraph: {old}")
    updated = full.replace(old, new)
    if paragraph.runs:
        paragraph.runs[0].text = updated
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(updated)


def reset_with_bold_prefix(paragraph, prefix, remainder):
    for run in paragraph.runs:
        run.text = ""
    lead = paragraph.runs[0] if paragraph.runs else paragraph.add_run()
    lead.text = prefix
    lead.bold = True
    body = paragraph.add_run(remainder)
    body.bold = False


def add_before(anchor, text, style="Normal"):
    paragraph = anchor.insert_paragraph_before(text, style=style)
    paragraph.paragraph_format.keep_together = True
    return paragraph


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=90, bottom=90, end=90):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def format_result_table(table):
    table.style = "Table Grid"
    table.autofit = False
    widths = [Inches(3.05), Inches(0.85), Inches(0.85), Inches(0.85)]
    for row_i, row in enumerate(table.rows):
        for col_i, cell in enumerate(row.cells):
            cell.width = widths[col_i]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            set_cell_shading(cell, "2F5597" if row_i == 0 else ("EAF1F8" if row_i % 2 == 0 else "FFFFFF"))
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_i == 0 else WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.space_before = Pt(0)
                for run in paragraph.runs:
                    run.font.size = Pt(9.5)
                    if row_i == 0:
                        run.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)


doc = Document(SRC)

# Abstract: add the English transfer result while keeping its development-only status explicit.
abstract = doc.paragraphs[2]
old = (
    "On the frozen CAIL2020-SFZY-clean test set, the method obtains ROUGE-1/2/L of "
    "60.94/42.10/50.73 and rule-based Fact Recall of 70.99, with no truncation or budget violations."
)
new = old + (
    " On a separately disclosed 703-document IN-Abs development split, the English E1 adaptation "
    "obtains 66.42/38.15/62.95 under per-document reference-length budgets, also with no budget "
    "violations; it remains below the retrained SummaRuNNer result, and the official 100-document "
    "test set remains sealed."
)
replace_in_paragraph(abstract, old, new)

# Introduction: state the scope and interpretation of the transfer evidence.
intro_results = paragraph_by_exact_text(doc, doc.paragraphs[10].text)
intro_old = (
    "Experiments on CAIL2020-SFZY-clean assess both performance and the proposed mechanism under a frozen "
    "450-character, eight-sentence protocol."
)
intro_new = intro_old + (
    " A separate IN-Abs extension evaluates language adaptation on 6,325 training and 703 development "
    "judgments under word-count budgets; E1 obtains 66.42/38.15/62.95 under per-document reference-length "
    "budgets, while the official 100-document test set is not used."
)
replace_in_paragraph(intro_results, intro_old, intro_new)

contrib_anchor = paragraph_by_exact_text(doc, "2 Related Work")
new_contribution = add_before(
    contrib_anchor,
    "Cross-language extension and boundary evidence. We evaluate an English E1 adaptation on IN-Abs under "
    "a common complete-sentence protocol, report its relation to adapted extractive baselines, and separate "
    "development-set transfer evidence from final official-test claims.",
    style="Normal",
)
reset_with_bold_prefix(
    new_contribution,
    "Cross-language extension and boundary evidence. ",
    "We evaluate an English E1 adaptation on IN-Abs under a common complete-sentence protocol, report its "
    "relation to adapted extractive baselines, and separate development-set transfer evidence from final "
    "official-test claims.",
)
replace_in_paragraph(paragraph_by_exact_text(doc, "This paper makes three contributions:"), "three", "four")

# Add transfer as an explicit experimental question.
questions = paragraph_by_exact_text(doc, "We organize the evaluation around three questions: whether LegalFactAware is competitive under a common output protocol; which design choices account for its performance; and how content coverage changes as the available budget shrinks. Final model comparisons use the frozen test set, while component and budget analyses use the development set. The latter diagnose the method without tuning on test results.")
questions.text = (
    "We organize the evaluation around four questions: whether LegalFactAware is competitive under a common "
    "output protocol; which design choices account for its performance; how content coverage changes as the "
    "available budget shrinks; and whether a language-adapted configuration remains competitive on English "
    "legal judgments. Chinese final-model comparisons use the frozen test set, while component and budget "
    "analyses use the development set. The English extension is development-only because the official IN-Abs "
    "test set remains sealed."
)

# Insert a new main-text English extension section before Discussion.
discussion = paragraph_by_exact_text(doc, "4.5 Discussion")
discussion.text = "4.6 Discussion"

add_before(discussion, "4.5 English Extension on IN-Abs", style="Heading 2")
add_before(
    discussion,
    "IN-Abs contains Indian Supreme Court judgments paired with headnotes [10]. We keep its official "
    "100-document test set sealed and split the 7,030 released training documents into 6,325 training and "
    "703 development documents. Candidate extraction is reference-free and retains at most 400 sentences "
    "through document-wide uniform coverage. The principal English track limits each output to the spaCy word "
    "count of that document's reference summary, follows complete-sentence fit/skip decoding, and evaluates all "
    "systems with py-rouge 1.1. Reference content is not supplied to the extractor; only the evaluation budget "
    "uses the reference length, following the dataset protocol.",
)
add_before(
    discussion,
    "The E1 adaptation replaces Chinese character n-grams with English word 1–2-gram TF–IDF, adapts Legal14 "
    "and Section6 cues to English legal expressions, and uses class-balanced logistic regression, word-bigram "
    "MMR, and complete-sentence decoding. A sparse fixed-budget membership target was less effective on the "
    "development split, so E1 retains the more stable reference-derived summary-level supervision. The English "
    "experiment therefore tests transfer of the lightweight scoring and constrained-decoding framework; it is "
    "not evidence that the Chinese Budget-consistent Oracle is universally superior across languages.",
)
add_before(
    discussion,
    "Table 3: IN-Abs development results under per-document reference-length word budgets (%). All systems use "
    "the same 703 documents and complete-sentence evaluation protocol; scores describe the disclosed retrained "
    "or adapted implementations rather than author-checkpoint reproductions.",
    style="Caption",
)

rows = [
    ("Method", "R1", "R2", "RL"),
    ("SummaRuNNer", "67.73", "40.73", "64.46"),
    ("SENDE", "66.68", "39.19", "63.16"),
    ("LegalFactAware-E1", "66.42", "38.15", "62.95"),
    ("Agarwal-ST", "66.41", "38.94", "62.77"),
    ("DSDR", "64.75", "34.17", "61.05"),
    ("LexRank", "63.50", "34.48", "59.93"),
    ("Gist", "63.38", "33.74", "59.36"),
    ("CaseSummarizer", "61.79", "33.33", "58.73"),
    ("MemSum", "59.18", "30.23", "55.09"),
    ("SSE", "54.36", "26.22", "46.85"),
]
table = doc.add_table(rows=len(rows), cols=4)
for i, row in enumerate(rows):
    for j, value in enumerate(row):
        table.cell(i, j).text = value
format_result_table(table)
discussion._p.addprevious(table._tbl)

add_before(
    discussion,
    "LegalFactAware-E1 obtains 66.42/38.15/62.95 with zero budget violations. It lies in the same development "
    "performance tier as the adapted SENDE and Agarwal-ST systems, but it does not exceed retrained "
    "SummaRuNNer, which reaches 67.73/40.73/64.46. These are mean comparisons only; aligned significance tests "
    "are not asserted for this English table. The result supports competitive cross-language adaptation of the "
    "scoring-and-decoding framework, rather than superiority over all prior extractive models.",
)
add_before(
    discussion,
    "A separate fixed-budget track uses 643 words for every development document. Under that protocol, E1 "
    "obtains 57.71/34.57/54.80, again with zero budget violations. This track serves the multi-budget research "
    "question and is not directly comparable to Table 3 because the output limits differ. Chinese and English "
    "ROUGE values are likewise not compared across datasets. Official IN-Abs test evaluation will be conducted "
    "only after the English configuration and baseline set are frozen.",
)

# Update the discussion and conclusion so the added experiment is integrated rather than appended silently.
discussion_body = paragraph_by_exact_text(doc, "The result is specific to fixed-budget Chinese extraction. Static membership scores do not respond to the meaning of every selected summary, and surface redundancy does not resolve attribution or semantic repetition. Learned summary-state modeling and budget-conditioned training address capabilities beyond this implementation. They are future extensions, rather than explanations assigned to the present results.")
discussion_body.text = (
    "The strongest causal evidence remains specific to fixed-budget Chinese extraction. The IN-Abs development "
    "result shows that an English adaptation is competitive under a unified protocol, while also revealing that "
    "the sparse fixed-budget supervision used in Chinese does not transfer unchanged. Static membership scores "
    "do not respond to the meaning of every selected summary, and surface redundancy does not resolve attribution "
    "or semantic repetition. Learned summary-state modeling and budget-conditioned training address capabilities "
    "beyond this implementation."
)

conclusion_open = paragraph_by_exact_text(doc, doc.paragraphs[[p.text for p in doc.paragraphs].index("5 Conclusion") + 1].text)
conclusion_open.text = (
    "LegalFactAware addresses fixed-budget legal summarization by connecting reference-derived training membership "
    "labels with complete-sentence constrained inference. In the Chinese configuration, a greedy reference-aware "
    "Budget-consistent Oracle supplies supervision; a lightweight lexical and domain-feature classifier estimates "
    "static membership scores; and redundancy-aware fit/skip decoding enforces feasible output. The English E1 "
    "extension retains the lightweight scoring and decoding structure while adapting its representation and using "
    "the more stable English supervision target."
)

conclusion_results = paragraph_by_exact_text(doc, doc.paragraphs[[p.text for p in doc.paragraphs].index("5 Conclusion") + 2].text)
conclusion_results.text = (
    "The frozen CAIL2020-SFZY-clean results show competitive performance, with ROUGE-1/2/L of "
    "60.94/42.10/50.73, Fact Recall of 70.99, and no hard truncation or budget violation. Controlled Chinese "
    "experiments support the tested supervision procedure and complete-sentence decoding, identify "
    "position-augmented TF–IDF as the main representation, and distinguish Section6's small, metric-dependent "
    "effect from statistically unsupported Legal14 gains. On the 703-document IN-Abs development split, E1 "
    "obtains 66.42/38.15/62.95 under reference-length budgets and 57.71/34.57/54.80 under a fixed 643-word budget, "
    "with zero violations in both tracks. Because the official IN-Abs test remains sealed, these results establish "
    "development-stage transfer evidence, not a final cross-language effectiveness claim."
)

scope = paragraph_by_exact_text(doc, "Scope and deployment. Main evidence comes from one cleaned Chinese dataset and one training budget. The 200–450-character curve changes the decoder budget while retaining the same model; it does not establish optimal multi-budget supervision. IN-Abs official-protocol experiments are not frozen and are excluded from effectiveness claims. Character features and rule lexicons require reevaluation across languages and jurisdictions. The current manuscript package does not include the executable preprocessing and fact-evaluation assets. Matched runtime measurements are not reported.")
scope.text = (
    "Scope and deployment. The strongest controlled evidence comes from one cleaned Chinese dataset and one "
    "training budget. The 200–450-character curve changes the decoder budget while retaining the same model; it "
    "does not establish optimal multi-budget supervision. The IN-Abs extension is reported on a disclosed "
    "development split only, uses a language-adapted configuration, and does not authorize claims about the sealed "
    "official test set. Character and word features, rule lexicons, and supervision density require reevaluation "
    "across languages and jurisdictions. Matched runtime measurements and expert legal evaluation are not reported."
)
reset_with_bold_prefix(
    scope,
    "Scope and deployment. ",
    "The strongest controlled evidence comes from one cleaned Chinese dataset and one training budget. The "
    "200–450-character curve changes the decoder budget while retaining the same model; it does not establish "
    "optimal multi-budget supervision. The IN-Abs extension is reported on a disclosed development split only, "
    "uses a language-adapted configuration, and does not authorize claims about the sealed official test set. "
    "Character and word features, rule lexicons, and supervision density require reevaluation across languages "
    "and jurisdictions. Matched runtime measurements and expert legal evaluation are not reported.",
)

# Renumber all pre-existing appendix tables and their in-text references after inserting Table 3.
replacements = {
    "Table 3 in Appendix B": "Table 4 in Appendix B",
    "Table 3 defines": "Table 4 defines",
    "Table 3: Legal14 features.": "Table 4: Legal14 features.",
    "Table 4 in Appendix C": "Table 5 in Appendix C",
    "Table 5 in Appendix C": "Table 6 in Appendix C",
    "Tables 4 and 5 provide": "Tables 5 and 6 provide",
    "Table 6 preserves": "Table 7 preserves",
    "Table 4: Statistics": "Table 5: Statistics",
    "Table 5: Baselines": "Table 6: Baselines",
    "Table 6: Archived": "Table 7: Archived",
    "Table 7 in Appendix D": "Table 8 in Appendix D",
    "Table 7 reports": "Table 8 reports",
    "Table 7: Interaction": "Table 8: Interaction",
    "Table 8 in Appendix E": "Table 9 in Appendix E",
    "Table 8 applies": "Table 9 applies",
    "Table 8: Development-set": "Table 9: Development-set",
}
for paragraph in doc.paragraphs:
    for run in paragraph.runs:
        updated = run.text
        for old_text, new_text in replacements.items():
            updated = updated.replace(old_text, new_text)
        run.text = updated

# Avoid a cascading renumbering replacement in the CAIL dataset-statistics reference.
for paragraph in doc.paragraphs:
    if paragraph.text.startswith("Data and constraints. CAIL2020-SFZY-clean"):
        replace_in_paragraph(paragraph, "Table 6 in Appendix C gives full statistics.", "Table 5 in Appendix C gives full statistics.")
        break

# Keep title and section hierarchy together after the additions.
for paragraph in doc.paragraphs:
    if paragraph.style.name.startswith("Heading") or paragraph.style.name == "Caption":
        paragraph.paragraph_format.keep_with_next = True

doc.save(OUT)
print(OUT)
