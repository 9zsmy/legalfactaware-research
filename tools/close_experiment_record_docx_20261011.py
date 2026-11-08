from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


SOURCE = Path(
    "/Users/admin/Documents/ChatGPT/1/论文写作/英文论文/"
    "LegalFactAware_English_Manuscript_Reproducibility_Audited_20261010.docx"
)
OUTPUT = Path(
    "/Users/admin/Documents/ChatGPT/1/论文写作/英文论文/"
    "LegalFactAware_English_Manuscript_Experiment_Record_Closed_20261011.docx"
)


def find(doc: Document, prefix: str) -> Paragraph:
    matches = [p for p in doc.paragraphs if p.text.strip().startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one paragraph beginning {prefix!r}, found {len(matches)}")
    return matches[0]


def clear(paragraph: Paragraph) -> None:
    for child in list(paragraph._p):
        if child.tag.endswith("}pPr"):
            continue
        paragraph._p.remove(child)


def set_text(paragraph: Paragraph, text: str) -> None:
    clear(paragraph)
    paragraph.add_run(text)


def set_labelled(paragraph: Paragraph, label: str, body: str) -> None:
    clear(paragraph)
    run = paragraph.add_run(label)
    run.bold = True
    paragraph.add_run(" " + body)


def insert_after(paragraph: Paragraph, style: str = "Body Text") -> Paragraph:
    element = OxmlElement("w:p")
    paragraph._p.addnext(element)
    new_paragraph = Paragraph(element, paragraph._parent)
    new_paragraph.style = style
    return new_paragraph


def main() -> None:
    doc = Document(SOURCE)

    introduction = find(doc, "Experiments on CAIL2020-SFZY-clean assess both performance")
    old = "replacing fit/skip decoding with tail truncation reduces them by 0.88/0.73/1.19 points"
    new = "replacing fit/skip decoding with tail truncation reduces them by 0.88/0.74/1.19 points"
    if old not in introduction.text:
        raise RuntimeError("Tail-truncation statement not found")
    set_text(introduction, introduction.text.replace(old, new))

    frozen = find(doc, "Frozen development batch.")
    reconciliation = insert_after(frozen)
    set_labelled(
        reconciliation,
        "TF-only batch reconciliation.",
        "The historical v2.4 record reported 60.31/41.26/50.02 ROUGE for the "
        "TF-only control, whereas the formal batch reports 59.90/40.66/49.27. "
        "The surviving v2.4 source contract specifies the same train and development "
        "split hashes, Oracle-label hash, frozen v2.2 vectorizer/model hash, decoder "
        "hash, 450-character and eight-sentence limits, redundancy coefficient 0.2, "
        "balanced liblinear logistic regression, and seed 42 as the formal batch. On "
        "11 October 2026, we re-executed this TF-only path from those verified inputs "
        "under Python 3.10.19, scikit-learn 1.7.2, SciPy 1.15.3, NumPy 2.2.6, and "
        "Lawa 1.2.6. The rerun obtains 59.9055/40.6605/49.2678, differing from the "
        "formal batch’s unrounded 59.9044/40.6600/49.2672 by at most 0.0012 points. "
        "Thus the formal value is reproducible in the audited environment, while the "
        "historical value is not explained by any documented method configuration. "
        "Because the historical classifier, aligned predictions, and runtime manifest "
        "were not preserved, the residual historical gap cannot be assigned to a "
        "specific software or artifact difference. The historical numbers are therefore "
        "superseded and excluded from claims. The reconciliation record has SHA-256 "
        "04bcaf53f16c6e254df54b76463f3e982e5059d1f425ebf141016d6d0dc9231a.",
    )

    limitation = find(doc, "Remaining test-batch limitation.")
    set_labelled(
        limitation,
        "Remaining test-batch limitation.",
        "The frozen test aggregate remains traceable to 948 document identifiers, and "
        "the test split hash above is verified. However, the exact aligned prediction "
        "bundle and the expected final model artifact (recorded SHA-256 "
        "65fc714698f22da944cab62cbc4d28022181c31fea4f9c3f783ee11f8d776539) "
        "are absent from the current local package. Consequently, the reported test "
        "means are retained as archived aggregate evidence, but test-level paired "
        "significance is not independently confirmed or reasserted. Closing this item "
        "requires restoring the exact frozen model and ID-aligned per-document "
        "predictions, then rerunning the locked evaluator and paired tests without "
        "changing the reported configuration.",
    )

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
