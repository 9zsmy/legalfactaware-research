import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "/Users/admin/Documents/ChatGPT/1/outputs/legalfactaware_master_results_20260927";
const outputPath = `${outputDir}/LegalFactAware_实验总表_20260927.xlsx`;

const rows = [
  ["LegalFactAware v2.5-B / E1", "本文方法；E1为英文适配", 60.94, 42.10, 50.73, 70.99, 66.42, 38.15, 62.95, 0, "是", "不适用", "不适用", "不适用", "本文方法", "IN-Abs官方100篇test仍封闭"],
  ["SENDE", "受限独立重建", 59.72, 41.13, 50.65, null, 66.68, 39.19, 63.16, 0, "有限可比", "无完整官方复现资产", "—", "—", "独立重建", "英文使用普通Oracle标签与TSDAE句向量"],
  ["SSE", "结构代理适配", 57.14, 37.25, 46.25, 61.47, 54.36, 26.22, 46.85, 0, "有限可比", "无完整官方复现资产", "—", "—", "独立重建", "英文固定7句配额，平均剩余316词"],
  ["Agarwal-ST", "ST+MMR受限重实现", 54.98, 35.90, 45.66, 70.69, 66.41, 38.94, 62.77, 0, "有限可比", "BVA", "数值量级基本对齐", "—", "基本验证通过", "英文使用LegalBERT句向量、MMR=0.8"],
  ["Gist", "语言与预算适配", 53.62, 34.35, 44.27, 69.42, 63.38, 33.74, 59.36, 0, "有限可比", "UK-Abs 100篇test", "61.22/24.66/23.15", "42.70/13.20/21.50", "工程通过，数值失败", "官方代码与兼容实现选句Jaccard=0.9896"],
  ["TermDiffuSum（CAIL）", "独立完整重实现", 54.55, 34.98, 43.80, 60.21, null, null, null, null, "仅CAIL", "LegalAFSum", "无法运行原任务", "64.57/52.72/63.10", "无法严格复现", "原数据、划分、术语标注和完整代码未公开"],
  ["TermDiffuSum（公开代码源码解码）", "IN-Abs 200句敏感性；seed=101", null, null, null, null, 55.88, 34.38, 49.41, 597, "否", "IN-Abs dev", "55.88/34.38/49.41", "不适用", "仅检查源码行为", "原解码无IN-Abs逐篇长度约束；五种子任务运行中"],
  ["TermDiffuSum（预算适配解码）", "公开代码迁移＋完整句预算", null, null, null, null, 61.03, 32.29, 53.22, 0, "否", "IN-Abs dev", "61.03/32.29/53.22", "不适用", "自定义适配", "固定200候选句，不能进入400句主表；五种子任务运行中"],
  ["CivilSum LED-tail", "生成式独立轨道", 53.57, 28.59, 41.35, 43.60, 48.19, 26.64, 42.01, 0, "否（生成式）", "CivilSum原任务", "数值基本对齐", "—", "基本验证通过", "英文tail-1024，平均剩余499词"],
  ["MemSum", "官方架构重训", 49.56, 28.16, 38.52, 55.66, 59.18, 30.23, 55.09, 0, "有限可比", "GovReport", "官方流程数值基本复现", "—", "基本验证通过", "英文为P50预算Oracle适配"],
  ["EUR-Lex-Sum LexRank", "图排序语言与预算适配", 42.18, 21.23, 29.84, 36.91, 63.50, 34.48, 59.93, 0, "有限可比", "EUR-Lex-Sum", "官方方法与流程数值复现", "—", "验证通过", "英文为IN-Abs统一预算适配"],
  ["Budget Recipe", "Qwen3-4B checkpoint零样本迁移", 40.19, 19.05, 26.26, 33.46, 54.44, 29.23, 48.03, 0, "否（生成式）", "作者内部验证211例", "Avg/R2/RL/BLEU 31.83/35.14/33.90/26.44", "32.71/35.60/34.43/28.11", "checkpoint基本验证通过", "未复现完整SFT+RLVR；英文5篇空摘要"],
  ["SummaRuNNer", "IN-Abs split-safe重训适配", null, null, null, null, 67.73, 40.73, 64.46, 0, "是", "UK-Abs 100篇test", "46.23/22.46/22.47", "50.20/20.50/23.70", "宽松标准基本通过", "恢复官方词表、checkpoint与topk=15"],
  ["DSDR", "IN-Abs统一预算适配", null, null, null, null, 64.75, 34.17, 61.05, 0, "是", "UK-Abs 100篇test", "43.33/21.00/38.83", "48.40/17.40/22.10", "工程通过，数值失败", "不同ROUGE口径不能同时对齐三项"],
  ["CaseSummarizer", "IN-Abs受限适配", null, null, null, null, 61.79, 33.33, 58.73, 0, "是", "UK-Abs 100篇test", "42.10/20.25/33.31", "44.50/16.60/22.70", "工程通过，数值失败", "最小Python 3兼容迁移；评测实现敏感"],
  ["FedSum", "原论文验证专用", null, null, null, null, null, null, null, null, "不适用", "CNN/DM 2K；3 seeds", "FedAvg 33.82/12.13/30.65；FedSum 33.89/12.20/30.66", "优势+1.60/+1.31/+1.52", "绝对量级通过，核心趋势失败", "FedSum−FedAvg仅+0.08/+0.06/+0.01且方向不稳定"],
  ["MMR", "原论文验证专用", null, null, null, null, null, null, null, null, "不适用", "UK-Abs 100篇test", "41.43/19.53/37.37", "44.00/15.10/20.50", "工程通过，数值失败", "修复重复选择与预算后零违规"],
  ["KMM", "原论文验证专用", null, null, null, null, null, null, null, null, "不适用", "UK-Abs 100篇test", "42.99/17.05/18.34", "43.00/13.80/20.10", "宽松标准基本通过", "完整旧管线与论文附录描述存在冲突"],
  ["LetSum", "原论文验证专用", null, null, null, null, null, null, null, null, "不适用", "UK-Abs 100篇test", "42.85/22.05/19.69", "43.70/15.80/23.30", "工程通过，部分对齐", "不存在统一配置同时对齐R1/R2/RL"],
  ["IGCS", "原论文验证待补", null, null, null, null, null, null, null, null, "不适用", "六任务验证", "五个公开任务已完成", "—", "受数据许可阻塞", "OpenAsp需要DUC/NIST合法授权"],
];

const headers = [
  "方法",
  "实验身份",
  "CAIL R1",
  "CAIL R2",
  "CAIL RL",
  "CAIL Fact Recall",
  "IN-Abs R1",
  "IN-Abs R2",
  "IN-Abs RL",
  "IN-Abs预算违规",
  "进入主表",
  "原论文验证数据",
  "本次复评（按原文指标）",
  "论文报告（按原文指标）",
  "验证结论",
  "当前状态与限制",
];

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("实验总表");
sheet.showGridLines = false;

sheet.getRange("A2").values = [["LegalFactAware 中文与英文实验及原论文验证总表"]];
sheet.getRange("A2:P2").format.font = { name: "Arial", size: 16, bold: true, color: "#1F2937" };
sheet.getRange("A3:P3").format.borders = { bottom: { style: "thin", color: "#94A3B8" } };

sheet.getRange("A4").values = [["CAIL协议"]];
sheet.getRange("B4").values = [["CAIL2020-SFZY-clean test，948篇，450中文字符、最多8句。"]];
sheet.getRange("A5").values = [["IN-Abs协议"]];
sheet.getRange("B5").values = [["冻结6325/703划分；英文主表为703篇dev；逐篇参考摘要spaCy词数预算；完整句解码；官方100篇test仍封闭。"]];
sheet.getRange("A6").values = [["比较限制"]];
sheet.getRange("B6").values = [["CAIL与IN-Abs不能跨数据集比较绝对ROUGE；“原论文验证”与“IN-Abs适配”也不是同一实验轨道。状态截至2026-09-27。"]];
sheet.getRange("A4:A6").format.font = { name: "Arial", size: 10, bold: true, color: "#334155" };
sheet.getRange("B4:P6").format.font = { name: "Arial", size: 10, italic: true, color: "#475569" };
sheet.getRange("A6:P6").format.fill = "#FFF7D6";

sheet.getRange("A8:P8").values = [headers];
sheet.getRange("A9").write(rows);
const lastRow = 8 + rows.length;
const table = sheet.tables.add(`A8:P${lastRow}`, true, "MasterResultsTable");
table.style = "TableStyleMedium2";
table.showFilterButton = true;

sheet.getRange(`A8:P${lastRow}`).format.font = { name: "Arial", size: 10, color: "#243447" };
sheet.getRange("A8:P8").format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
sheet.getRange(`C9:J${lastRow}`).format.numberFormat = "0.00";
sheet.getRange(`C9:J${lastRow}`).format.horizontalAlignment = "right";
sheet.getRange(`A9:B${lastRow}`).format.verticalAlignment = "center";
sheet.getRange(`K9:P${lastRow}`).format.verticalAlignment = "center";
sheet.getRange(`B9:B${lastRow}`).format.wrapText = true;
sheet.getRange(`L9:P${lastRow}`).format.wrapText = true;

sheet.getRange(`A${lastRow + 2}`).values = [["当前后台任务"]];
sheet.getRange(`B${lastRow + 2}`).values = [["TermDiffuSum公开代码五种子迁移正在运行：seed=101已完成；seed=100当前完成3/10轮；seed=201、901、1001待运行。完成后再填五种子均值和标准差。"]];
sheet.getRange(`A${lastRow + 3}`).values = [["来源"]];
sheet.getRange(`B${lastRow + 3}`).values = [["经审计的项目实验归档、复现日志和验证报告；IN-Abs原论文：https://aclanthology.org/2022.aacl-main.77/；TermDiffuSum论文：https://aclanthology.org/2025.coling-main.216/。"]];
sheet.getRange(`A${lastRow + 2}:A${lastRow + 3}`).format.font = { name: "Arial", size: 10, bold: true, color: "#334155" };
sheet.getRange(`B${lastRow + 2}:P${lastRow + 3}`).format.font = { name: "Arial", size: 10, italic: true, color: "#475569" };
sheet.getRange(`A${lastRow + 2}:P${lastRow + 2}`).format.fill = "#E8F1F8";

const widths = [28, 25, 11, 11, 11, 15, 11, 11, 11, 15, 13, 21, 32, 30, 25, 45];
for (let col = 0; col < widths.length; col += 1) {
  sheet.getRangeByIndexes(0, col, lastRow + 4, 1).format.columnWidth = widths[col];
}
sheet.getRange("1:1").format.rowHeight = 10;
sheet.getRange("2:2").format.rowHeight = 26;
sheet.getRange("8:8").format.rowHeight = 42;
sheet.getRange(`9:${lastRow}`).format.rowHeight = 50;
sheet.freezePanes.freezeRows(8);

workbook.recalculate();
await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({ sheetName: "实验总表", range: `A1:P${lastRow + 3}`, scale: 1, format: "png" });
await fs.writeFile(`${outputDir}/preview.png`, new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
