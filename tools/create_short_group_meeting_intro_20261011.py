from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


OUT = "/Users/admin/Documents/ChatGPT/1/论文写作/英文论文/LegalFactAware_组会简短介绍稿_20261011.docx"
FONT = "Arial Unicode MS"


def set_font(run, size=11.5, bold=False, color=(0, 0, 0)):
    run.font.name = FONT
    rfonts = run._element.get_or_add_rPr().rFonts
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), FONT)
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor(*color)


doc = Document()
section = doc.sections[0]
section.top_margin = Cm(2.1)
section.bottom_margin = Cm(2.1)
section.left_margin = Cm(2.5)
section.right_margin = Cm(2.5)

for style_name in ("Normal", "Title"):
    style = doc.styles[style_name]
    style.font.name = FONT
    style.font.color.rgb = RGBColor(0, 0, 0)
    style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT)
    style._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT)
    style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)

title_ppr = doc.styles["Title"]._element.get_or_add_pPr()
border = title_ppr.find(qn("w:pBdr"))
if border is not None:
    title_ppr.remove(border)

title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(8)
run = title.add_run("LegalFactAware论文组会简短介绍稿")
set_font(run, size=19, bold=True)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle.paragraph_format.space_after = Pt(16)
run = subtitle.add_run("约 3 分钟  可直接朗读")
set_font(run, size=10, color=(89, 89, 89))

paragraphs = [
    "各位老师、同学好，我简单汇报一下我目前正在写的论文。论文题目暂定为《LegalFactAware for Fixed Budget Legal Judgment Summarization》，主要研究固定预算条件下的中文法律裁判文书抽取式摘要。",
    "这篇论文关注的问题是，很多抽取式摘要方法先分别判断每个句子是否重要，最后才处理摘要长度。但在实际任务中，摘要只能保留有限字符和有限句子，单独看起来重要的句子可能互相重复，也可能无法同时放进预算里。如果最后直接截断，还可能把裁判结果或法律限定条件切断。因此，论文希望让训练时的监督方式和最终生成摘要时的预算要求保持一致。",
    "针对这个问题，我设计了 LegalFactAware。训练阶段使用 Budget-consistent Oracle，在给定字符数、句子数和完整句约束下构造训练标签；然后使用字符级 TF-IDF、法律事实特征和篇章位置特征，通过逻辑回归给句子评分；最后在解码时加入冗余控制，只选择能够完整放入剩余预算的句子。这里的参考摘要只用于训练标签构造，测试阶段不使用参考摘要。",
    "实验使用清洗后的 CAIL2020-SFZY 数据集，统一限制为最多 450 个汉字和 8 个完整句子。当前完整模型在测试集上的 ROUGE-1、ROUGE-2 和 ROUGE-L 分别为 60.94、42.10 和 50.73，Fact Recall 为 70.99，并且没有出现句子截断和预算违规。在统一实验协议下，结果与 2026 年的 SENDE 等方法相比具有竞争力。",
    "消融实验表明，预算约束下的训练标签和完整句解码是有效的；其中 TF-IDF 是主要的表示来源，篇章结构特征有小幅帮助，部分法律特征的独立增益还不够明确。所以论文目前的核心不是强调模型有多复杂，而是强调训练监督和最终可行摘要空间之间的对齐。",
    "目前论文已经形成完整英文稿，方法、实验、消融、统计检验和复现配置都已经整理。接下来主要是继续完善论文表达，并补充更严格的控制实验，进一步单独验证预算一致性的作用。如果时间允许，还会考虑增加专家评价。以上就是目前论文的主要内容和进展。",
]

for text in paragraphs:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0.8)
    p.paragraph_format.line_spacing = 1.45
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    set_font(r)

note = doc.add_paragraph()
note.paragraph_format.space_before = Pt(8)
note.paragraph_format.left_indent = Cm(0.7)
note.paragraph_format.right_indent = Cm(0.7)
note.paragraph_format.line_spacing = 1.2
run = note.add_run("提示：如果时间更短，可以只保留第一段、第三段、第四段和最后一段。")
set_font(run, size=9.5, color=(89, 89, 89))
ppr = note._p.get_or_add_pPr()
shd = OxmlElement("w:shd")
shd.set(qn("w:fill"), "F2F5F8")
ppr.append(shd)

doc.save(OUT)
print(OUT)
