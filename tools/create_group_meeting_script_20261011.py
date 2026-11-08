from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


OUT = "/Users/admin/Documents/ChatGPT/1/论文写作/英文论文/LegalFactAware_组会汇报稿_当前进展_20261011.docx"


def set_run_font(run, east_asia="Arial Unicode MS", latin="Arial Unicode MS", size=11, bold=None):
    # LibreOffice on macOS may ignore w:eastAsia and use w:ascii for CJK text.
    # Use the same full-coverage font in all four slots for deterministic rendering.
    run.font.name = east_asia
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), east_asia)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), east_asia)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, color="D9D9D9", size="6"):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        el = tc_borders.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            tc_borders.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), size)
        el.set(qn("w:color"), color)


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, size=14 if level == 1 else 12, bold=True)
    return p


def add_body(doc, text, bold_prefix=None, italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if bold_prefix and text.startswith(bold_prefix):
        r1 = p.add_run(bold_prefix)
        set_run_font(r1, bold=True)
        r2 = p.add_run(text[len(bold_prefix):])
        set_run_font(r2)
    else:
        r = p.add_run(text)
        set_run_font(r)
        r.italic = italic
    return p


def add_cue(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.55)
    p.paragraph_format.right_indent = Cm(0.55)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.2
    r = p.add_run(text)
    set_run_font(r, size=10)
    r.italic = True
    r.font.color.rgb = RGBColor(89, 89, 89)
    p_pr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F2F5F8")
    p_pr.append(shd)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.2
    r = p.add_run(text)
    set_run_font(r, size=10.5)
    return p


doc = Document()
sec = doc.sections[0]
sec.top_margin = Cm(1.9)
sec.bottom_margin = Cm(1.8)
sec.left_margin = Cm(2.25)
sec.right_margin = Cm(2.25)

styles = doc.styles
styles["Normal"].font.name = "Arial Unicode MS"
styles["Normal"]._element.rPr.rFonts.set(qn("w:ascii"), "Arial Unicode MS")
styles["Normal"]._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial Unicode MS")
styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial Unicode MS")
styles["Normal"].font.size = Pt(11)
for name in ("Title", "Heading 1", "Heading 2"):
    style = styles[name]
    style.font.name = "Arial Unicode MS"
    style._element.rPr.rFonts.set(qn("w:ascii"), "Arial Unicode MS")
    style._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial Unicode MS")
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial Unicode MS")
    style.font.color.rgb = RGBColor(0, 0, 0)

# Remove the decorative Title-style border inherited from some Word defaults.
title_style_ppr = styles["Title"]._element.get_or_add_pPr()
title_style_border = title_style_ppr.find(qn("w:pBdr"))
if title_style_border is not None:
    title_style_ppr.remove(title_style_border)

title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(7)
r = title.add_run("LegalFactAware论文当前进展组会汇报稿")
set_run_font(r, size=20, bold=True)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.paragraph_format.space_after = Pt(14)
r = sub.add_run("建议汇报时长 10 分钟  版本日期 2026 年 10 月 11 日")
set_run_font(r, size=10)
r.font.color.rgb = RGBColor(89, 89, 89)

add_cue(doc, "使用方式：正文可以直接朗读；灰底文字是换页或强调提示，不必逐字念。最后附有常见问题的简短回答。")

add_heading(doc, "一 开场和结论先行 约 40 秒")
add_body(doc, "各位老师、同学好，我今天汇报的是我目前正在写的论文《LegalFactAware for Fixed Budget Legal Judgment Summarization》。这项工作研究的是固定预算下的中文法律裁判文书抽取式摘要。现在论文已经形成完整英文稿，方法、主要实验、消融实验、统计检验和复现记录都已经整理完成。当前最核心的结论是：在固定字符数和句子数约束下，训练标签和最终可输出摘要之间存在不一致；如果在构造监督时就考虑可行的摘要组合，并在解码时只选择能够完整放入预算的句子，可以得到更稳定、也更容易审计的摘要结果。")
add_body(doc, "目前完整方法在冻结的 CAIL2020-SFZY-clean 测试集上取得 ROUGE-1、ROUGE-2、ROUGE-L 为 60.94、42.10、50.73，规则化 Fact Recall 为 70.99，而且没有硬截断和预算违规。它在统一抽取式比较中取得最高均值，但与最强外部基线 SENDE 的 ROUGE-L 只相差 0.08，因此我在论文中把结果表述为有竞争力，而不是笼统宣称全面显著优于所有方法。")

add_heading(doc, "二 研究问题 约 1 分钟")
add_cue(doc, "如果配合幻灯片，这里放一张“独立句子标签与预算可行摘要不一致”的简单示意图。")
add_body(doc, "这项工作的出发点是，法律裁判文书很长，事实、当事人主张、法律依据、法院说理和裁判结果分散在不同位置。抽取式方法虽然可以保留原文措辞，但通常先独立判断每个句子是否重要，最后才在解码阶段处理长度限制。这样会出现两个问题。")
add_body(doc, "第一个问题是训练目标不一致。多个句子单独看都可能与参考摘要相似，但它们可能重复同一事实，或者无法在有限预算中同时出现。独立句子标签会奖励这些句子，却没有告诉模型什么样的句子组合可以构成一个受预算约束的摘要。")
add_body(doc, "第二个问题是解码不一致。如果先选句、最后再按字符数截断，摘要末尾可能被切断，法律限定条件或裁判结果也可能不完整。因此，我把论文的研究问题概括为：能否让训练监督和推理阶段共享同一个可行输出空间，也就是相同的字符预算、句子数上限和完整句约束。")

add_heading(doc, "三 方法 约 2 分钟")
add_body(doc, "LegalFactAware 包含三个环节。第一个环节是 Budget-consistent Oracle，用于构造训练标签。它从空摘要开始，每一步选择在当前已选内容基础上能够带来正向摘要级 ROUGE 增益、并且仍满足预算的完整句子。最后进入这个贪心摘要的句子被标为正例。这里需要强调，Oracle 只在训练标签构造时使用参考摘要；开发集和测试集推理不读取参考摘要。它也不是全局最优求解器，而是一个可复现的贪心标注过程。")
add_body(doc, "第二个环节是静态句子评分。模型使用字符二到四元 TF-IDF 表示，结合十四个法律和位置特征 Legal14，以及六类篇章角色提示 Section6，再用带类别平衡的逻辑回归输出句子分数。采用逻辑回归并不是为了强调模型复杂度，而是为了让监督、表示和解码三个因素可以清楚地分开检验。")
add_body(doc, "第三个环节是完整句预算解码。解码器把静态相关性分数与冗余惩罚结合；如果某个候选句不能完整放入剩余预算，就跳过它并继续考察后面的候选句，而不是截断句子。选完以后再恢复原文顺序。")
add_body(doc, "因此，这篇论文真正想强调的并不是 Oracle 这个名字，也不是一种复杂的新模型，而是 feasible-space alignment，也就是可行输出空间对齐：训练标签和推理输出都受到相同预算与完整句约束。")

add_heading(doc, "四 实验设计 约 1 分钟")
add_body(doc, "数据使用清洗后的 CAIL2020-SFZY 数据集，冻结划分为 7,495 篇训练文档、942 篇开发文档和 948 篇测试文档。所有中文实验统一限制为最多 450 个规范化汉字和 8 个完整句子，并使用锁定的 Lawa 分词和 ROUGE 实现。")
add_body(doc, "比较对象覆盖静态、结构感知、图排序、序列式和生成式方法。其中包括 2026 年发表的 SENDE、2025 年 COLING 的 TermDiffuSum、MemSum、Agarwal-ST、Gist 和 SSE 等。需要说明的是，外部方法并不都是作者代码的精确复现：有的是独立重实现，有的是语言和预算适配，有的是代理版本。因此论文比较的是统一协议下可执行版本，而不是声称复现了所有原论文数值。")

add_heading(doc, "五 主要实验结果 约 2 分钟")
table = doc.add_table(rows=1, cols=5)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.style = "Table Grid"
headers = ["方法", "R1", "R2", "RL", "说明"]
for i, h in enumerate(headers):
    cell = table.rows[0].cells[i]
    cell.text = h
    set_cell_shading(cell, "1F4E78")
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for run in cell.paragraphs[0].runs:
        set_run_font(run, size=9.5, bold=True)
        run.font.color.rgb = RGBColor(255, 255, 255)
    set_cell_border(cell)
rows = [
    ("LegalFactAware", "60.94", "42.10", "50.73", "本文方法"),
    ("SENDE 2026", "59.72", "41.13", "50.65", "最强外部均值基线"),
    ("SSE", "57.14", "37.25", "46.25", "结构代理适配"),
    ("TermDiffuSum 2025", "54.55", "34.98", "43.80", "独立重实现"),
    ("MemSum", "49.56", "28.16", "38.52", "官方架构重训"),
]
for ri, row in enumerate(rows, start=1):
    cells = table.add_row().cells
    for ci, value in enumerate(row):
        cells[ci].text = value
        cells[ci].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        if ri % 2 == 0:
            set_cell_shading(cells[ci], "EAF2F8")
        for run in cells[ci].paragraphs[0].runs:
            set_run_font(run, size=9.3, bold=(ri == 1))
        cells[ci].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if ci in (1, 2, 3) else WD_ALIGN_PARAGRAPH.LEFT
        set_cell_border(cells[ci])
for row in table.rows:
    for idx, width in enumerate([Cm(4.0), Cm(1.6), Cm(1.6), Cm(1.6), Cm(5.2)]):
        row.cells[idx].width = width

add_body(doc, "从总体结果看，LegalFactAware 得到最高的 ROUGE 均值和 70.99 的 Fact Recall，并且没有预算违规。它相对 SSE 提升 3.80、4.85、4.48 个 ROUGE 点，相对 MemSum 提升 11.38、13.94、12.21 个点。与 SENDE 相比，R1 和 R2 分别高 1.22 和 0.97，但 RL 只高 0.08，而且没有 SENDE 的逐样本预测，所以不能进行配对显著性检验。")
add_body(doc, "更能解释方法的证据来自开发集控制实验。把预算一致的摘要成员标签替换为独立句子级弱标签以后，R1、R2、RL 分别下降 6.24、8.28、8.68，Fact Recall 下降 10.30。这个结果支持当前完整标注过程，但它还不能把摘要级边际选择、标签稀疏性和预算匹配三种因素完全拆开。")
add_body(doc, "解码实验也比较明确。把完整句 fit-and-skip 解码换成末尾硬截断后，R1、R2、RL 分别下降 0.88、0.74、1.19，Fact Recall 下降 4.06，而且 942 篇开发文档中有 379 篇出现截断。这个结果说明完整句约束不仅是格式要求，也确实影响内容保留。")
add_body(doc, "表示层的结论比较克制：TF-IDF 是主要贡献来源；Section6 带来小幅且依赖指标的改善；Legal14 单独增益没有得到统计支持。因此论文没有把全部法律特征都描述成有效创新。")

add_heading(doc, "六 当前论文完成度和证据状态 约 1 分钟")
add_body(doc, "目前英文论文已经形成完整稿，包含摘要、引言、相关工作、方法、实验、结论、局限性和复现附录。开发集的配对 bootstrap、Wilcoxon 检验和 Holm 校正已经重新计算，并补齐了数据划分、弱标签规则、模型参数、评测配置和文件哈希。")
add_body(doc, "测试集方面，原始最终模型二进制和历史预测包没有找回。为此，我使用已核验的数据划分、冻结向量器、同一特征模块和同一完整句解码器进行了确定性重建。重建结果为 60.9391、42.0963、50.7268，与归档结果在论文保留的两位小数上完全一致，最大原始差异只有 0.00056 个百分点。基于重建预测，和已保存逐样本记录的 LegalFactAware-v2、MemSum、SSE-v2 进行了九项配对比较，置信区间均高于零，Holm 校正后均显著。")
add_body(doc, "但这只能说明主要结果在计算上可重建，不能证明找回了原始二进制；也不能补出 SENDE 的配对显著性。因此论文把这部分明确称为 reconstruction audit，而不是原始实验包恢复。")

add_heading(doc, "七 当前不足和下一步 约 1 分钟")
add_body(doc, "当前最重要的不足有三点。第一，现有弱监督对照同时改变了标签构造的多个方面，还不能单独证明增益全部来自预算一致性。最有价值的补充实验是设计同为摘要级监督、只改变预算匹配条件的严格对照。")
add_body(doc, "第二，目前只有中文 CAIL 主数据集。英文 IN-Abs 已经做过扩展探索，但协议和证据还没有达到主论文核心结论的完整程度，所以当前稿件不依赖英文结果。第三，Fact Recall 只是规则可识别事实的表面覆盖，抽取句子也不等于上下文解释一定正确，仍需要法律专家评价来验证实际摘要质量。")
add_body(doc, "接下来我的优先顺序是：先完成预算一致性的严格控制变量实验；然后冻结并归档最终模型、预测、逐文档指标和环境；如果时间允许，再补充小规模专家评价。英文迁移和动态摘要状态模型放在后续扩展，不影响当前主论文闭环。")

add_heading(doc, "八 结束语 约 20 秒")
add_body(doc, "以上是目前论文的主要进展。现在方法主线和主要结果已经比较清楚，论文的核心定位是固定预算抽取式法律摘要中的可行空间对齐，而不是依赖复杂模型获得提升。我希望组会上重点听取大家对三个问题的意见：现有单数据集证据是否足够；预算一致性的严格对照应该怎样设计；以及投稿前是否必须补充专家评价。谢谢大家。")

doc.add_page_break()
add_heading(doc, "组会常见问题简短回答")
qa = [
    ("为什么核心方法叫 Oracle", "Oracle 只负责在训练阶段利用参考摘要生成成员标签，测试阶段完全不使用参考摘要。论文真正的贡献名称是可行输出空间对齐，而不是 Oracle 本身。"),
    ("为什么使用逻辑回归而不是更复杂模型", "逻辑回归使监督、表示和解码可以独立检查，计算成本低、结果可审计。论文并不声称逻辑回归在所有数据集上优于深度模型。"),
    ("这是不是动态选句模型", "不是。句子基础分数是静态的；只有冗余项和剩余预算随已选摘要变化。学习摘要状态的动态模型属于后续工作。"),
    ("有没有使用普通 Oracle", "当前核心模型只使用 Budget-consistent Oracle。对照使用的是独立句子级弱监督，不把普通 Oracle 作为最终方法。"),
    ("为什么不能直接说超过 SENDE", "SENDE 是 2026 年方法，当前均值略低于本文，但 ROUGE-L 只差 0.08，而且没有同一批文档的逐样本预测，因此没有配对显著性证据。"),
    ("是否存在测试集泄漏", "参考摘要仅在训练阶段构造标签和离线评测时使用；开发集和测试集推理只使用源文档特征和训练得到的模型。"),
    ("最大创新点是什么", "在严格字符预算、句子数上限和完整句约束下，让训练成员标签与推理阶段允许的输出空间保持一致，并用控制实验区分监督、表示和解码的作用。"),
    ("当前最需要补什么", "最优先是匹配的摘要级监督对照，以单独检验预算匹配；其次是专家评价和完整实验包归档。"),
    ("英文 IN-Abs 为什么没有放进当前核心结论", "英文实验的任务协议、预算定义和基线复现条件不同，目前更适合作为后续跨语言扩展，不能替代中文主实验的证据闭环。"),
]
for q, a in qa:
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("问：" + q)
    set_run_font(r, size=10.5, bold=True)
    p2 = doc.add_paragraph()
    p2.paragraph_format.left_indent = Cm(0.74)
    p2.paragraph_format.space_after = Pt(4)
    p2.paragraph_format.line_spacing = 1.2
    r = p2.add_run("答：" + a)
    set_run_font(r, size=10.5)

add_heading(doc, "汇报前一分钟核对")
for item in [
    "把方法名称统一说成 LegalFactAware 或可行输出空间对齐，不把 Oracle 说成推理模型。",
    "总体结果使用 60.94 42.10 50.73 和 70.99；不要混用开发集数值。",
    "面对 SENDE 只说最高均值或具有竞争力，不说全面显著超越。",
    "明确 SENDE 是 2026 年论文，但当前 CAIL 结果是受约束独立重构。",
    "明确测试集结果完成了计算重建，但原始模型二进制尚未恢复。",
    "如果时间不足，优先保留研究问题、方法三环节、三组关键实验和下一步。",
]:
    add_bullet(doc, item)

# Footer with compact page field.
for section in doc.sections:
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    r = p.add_run("LegalFactAware 组会汇报稿")
    set_run_font(r, size=8)
    r.font.color.rgb = RGBColor(127, 127, 127)

doc.save(OUT)
print(OUT)
