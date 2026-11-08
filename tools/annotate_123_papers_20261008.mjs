import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const inputPath = "/Users/admin/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_8mc5ryml21kd22_adc9/msg/file/2026-09/123.xlsx";
const mediaDir = "/tmp/codex_123_media_20261008";
const outputDir = "/Users/admin/Documents/ChatGPT/1/outputs/paper_priority_annotations_20261008";
const outputPath = `${outputDir}/123_论文年份_相关性_CCF等级标注_20261008.xlsx`;

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItemAt(0);
sheet.name = "论文复现比较_已标注";
sheet.showGridLines = false;
sheet.tabColor = "#1F4E78";

// The source workbook uses P:X as one merged screenshot area per row.
// Rebuild that area as P:T annotations and U:AC row-aligned evidence images.
for (let row = 1; row <= 14; row += 1) {
  sheet.unmergeCells(`P${row}:X${row}`);
}
sheet.deleteAllDrawings();

sheet.mergeCells("P1:T1");
sheet.getRange("P1").values = [["年份、CCF等级与比较优先级（核验至2026-10-08）"]];
sheet.getRange("P2:T2").values = [["年份", "论文 / 发表来源", "CCF等级", "重点标注", "为什么这样比较"]];

sheet.mergeCells("U1:AC1");
sheet.mergeCells("U2:AC2");
sheet.getRange("U1").values = [["逐行证据"]];
sheet.getRange("U2").values = [["原论文结果表截图（与左侧方法一一对应）"]];
for (let row = 3; row <= 14; row += 1) {
  sheet.mergeCells(`U${row}:AC${row}`);
}

const annotations = [
  [
    "2022",
    "ACL 2022",
    "A",
    "★★ 次核心比较｜动态选句最相关",
    "同为长文档抽取式；显式利用已选句历史，最适合支撑“摘要状态感知”对照。数据域不是法律，因此重要性次于SENDE。",
  ],
  [
    "2022",
    "Findings of EMNLP 2022",
    "未单列（主会B）",
    "★★ 法律抽取核心基线",
    "法律判决、抽取式、MMR与冗余控制都和本文高度同构。Findings不自动继承EMNLP主会B，但论文的方法相关性很高。",
  ],
  [
    "2024",
    "SIGIR 2024",
    "A",
    "★ A类法律领域参考",
    "同为印度法院判决和法律摘要，领域高度相关；但属于生成式且数据集不同，适合作为高等级领域参考，不宜替代直接主比较。",
  ],
  [
    "2022",
    "arXiv预印本",
    "未分级",
    "法律数据集参考｜相关性中等",
    "涉及法律长文档和跨语种摘要，但对象是欧盟立法文件而非法院判决；LexRank也是通用无监督基线，不能作为最强直接对照。",
  ],
  [
    "2026",
    "The Journal of Supercomputing",
    "C",
    "★★★ 2026最新｜首要主比较",
    "法律判决抽取式、句子选择和神经评分与本文任务形态最接近，应作为英文实验第一主比较。US与CA是同一篇论文的两个数据子集。",
  ],
  [
    "2026",
    "The Journal of Supercomputing",
    "C",
    "★★★ 2026最新｜首要主比较",
    "与上一行属于同一篇SENDE论文；本行是CA数据子集。两行应成组报告，但不能计作两篇独立论文。",
  ],
  [
    "2025",
    "TACL 2025",
    "B",
    "★★ 2025新｜方法论相关",
    "统一抽取内容选择与指令化选择，和句子重要性学习相关；但并非专门的法律摘要模型。表中42.9的“六任务宏平均”口径建议复核。",
  ],
  [
    "2025",
    "JUST-NLP 2025 Workshop",
    "未单列",
    "★ 2025新｜预算与长文档参考",
    "严格计算预算和长篇印度法院判决与预算主题贴近；但它是Qwen生成式系统且为Workshop论文，不是抽取式同构主比较。",
  ],
  [
    "2025",
    "AAAI 2025（FedSum文中基线）",
    "A（结果来源）",
    "★ 2025新｜高等级基线来源",
    "FedAvg只是FedSum论文内基线，并非本表一篇新的独立论文；A级指承载该对照结果的AAAI 2025。联邦生成式任务与本文相关性有限。",
  ],
  [
    "2025",
    "AAAI 2025",
    "A",
    "★★ 2025新｜A类训练范式参考",
    "层级高且年份新，但核心贡献是数据稀缺条件下的联邦学习，不是法律抽取或预算解码。适合作为扩展比较，不宜压过SENDE/MemSum。",
  ],
  [
    "2017",
    "AAAI 2017；表中分数源AACL 2022",
    "A",
    "★★ 经典监督抽取基线",
    "句级监督抽取的经典强基线，适合证明英文方法超过传统神经抽取模型。注意本表UK-Abs原文数值来自AACL 2022复评表。",
  ],
  [
    "2006",
    "JURIX 2006；表中分数源AACL 2022",
    "未单列",
    "★ 经典法律无监督基线",
    "法律判决、修辞角色和K-mixture具有领域针对性，但年代较早且CCF未单列。用于覆盖传统法律方法，不作为最重要比较。",
  ],
];
sheet.getRange("P3").write(annotations);

// Styling for the new annotation block.
sheet.getRange("P1:T1").format = {
  fill: "#17365D",
  font: { name: "Microsoft YaHei", size: 12, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
sheet.getRange("P2:T2").format = {
  fill: "#1F4E78",
  font: { name: "Microsoft YaHei", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#B4C6E7" },
};
sheet.getRange("P3:T14").format = {
  font: { name: "Microsoft YaHei", size: 10, color: "#243447" },
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#D9E2F3" },
};
sheet.getRange("P3:S14").format.horizontalAlignment = "center";
sheet.getRange("T3:T14").format.horizontalAlignment = "left";

// Evidence-area headers and borders.
sheet.getRange("U1:AC1").format = {
  fill: "#7F6000",
  font: { name: "Microsoft YaHei", size: 12, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
sheet.getRange("U2:AC2").format = {
  fill: "#BF9000",
  font: { name: "Microsoft YaHei", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
sheet.getRange("U3:AC14").format = {
  fill: "#FFFFFF",
  font: { name: "Microsoft YaHei", size: 10, color: "#64748B", italic: true },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#D9D9D9" },
};
sheet.getRange("U7").values = [["该行暂无独立截图：SENDE-US与SENDE-CA属于同一篇2026论文"]];
sheet.getRange("U8").values = [["该行暂无独立截图：与上一行共享SENDE论文证据"]];

// 2025/2026 recency highlights.
for (const row of [7, 8, 9, 10, 11, 12]) {
  sheet.getRange(`P${row}`).format.fill = "#E4DFEC";
  sheet.getRange(`P${row}`).format.font = { name: "Microsoft YaHei", size: 10, bold: true, color: "#7030A0" };
  sheet.getRange(`S${row}`).format.fill = "#E4DFEC";
}

// CCF tiers: A=deep red, B=orange, C=blue, unranked=gray.
for (const row of [3, 5, 11, 12, 13]) {
  sheet.getRange(`R${row}`).format.fill = "#F4CCCC";
  sheet.getRange(`R${row}`).format.font = { name: "Microsoft YaHei", size: 10, bold: true, color: "#9C0006" };
}
sheet.getRange("R9").format.fill = "#FCE5CD";
sheet.getRange("R9").format.font = { name: "Microsoft YaHei", size: 10, bold: true, color: "#B45F06" };
for (const row of [7, 8]) {
  sheet.getRange(`R${row}`).format.fill = "#D9EAF7";
  sheet.getRange(`R${row}`).format.font = { name: "Microsoft YaHei", size: 10, bold: true, color: "#1F4E78" };
}
for (const row of [4, 6, 10, 14]) {
  sheet.getRange(`R${row}`).format.fill = "#E7E6E6";
  sheet.getRange(`R${row}`).format.font = { name: "Microsoft YaHei", size: 10, bold: true, color: "#666666" };
}

// Importance/relevance highlights.
for (const row of [7, 8]) {
  sheet.getRange(`A${row}`).format.fill = "#FFF2CC";
  sheet.getRange(`S${row}`).format.fill = "#FFD966";
  sheet.getRange(`S${row}`).format.font = { name: "Microsoft YaHei", size: 10, bold: true, color: "#7F6000" };
  sheet.getRange(`P${row}:T${row}`).format.borders = { preset: "all", style: "medium", color: "#BF9000" };
}
for (const row of [3, 4, 9, 13]) {
  sheet.getRange(`S${row}`).format.fill = "#E2F0D9";
  sheet.getRange(`S${row}`).format.font = { name: "Microsoft YaHei", size: 10, bold: true, color: "#375623" };
}
sheet.getRange("T9").format.fill = "#FCE4D6";
sheet.getRange("T9").format.font = { name: "Microsoft YaHei", size: 10, bold: true, color: "#9C0006" };

// Column sizing. Existing A:O data widths are intentionally preserved.
const newWidths = { P: 11, Q: 31, R: 18, S: 32, T: 62 };
for (const [col, width] of Object.entries(newWidths)) {
  sheet.getRange(`${col}1:${col}20`).format.columnWidth = width;
}
for (let col = 20; col <= 28; col += 1) {
  sheet.getRangeByIndexes(0, col, 20, 1).format.columnWidth = 12;
}
sheet.getRange("1:1").format.rowHeight = 30;
sheet.getRange("2:2").format.rowHeight = 38;

// Restore the ten original row-aligned screenshots in the new U:AC evidence area.
const evidenceImages = [
  { row: 3, file: "image1.png", width: 442, height: 250 },
  { row: 4, file: "image2.png", width: 620, height: 250 },
  { row: 5, file: "image3.png", width: 375, height: 275 },
  { row: 6, file: "image4.png", width: 710, height: 180 },
  { row: 9, file: "image5.png", width: 710, height: 124 },
  { row: 10, file: "image6.png", width: 548, height: 200 },
  { row: 11, file: "image7.png", width: 589, height: 97 },
  { row: 12, file: "image8.png", width: 589, height: 108 },
  { row: 13, file: "image9.png", width: 594, height: 80 },
  { row: 14, file: "image10.png", width: 594, height: 80 },
];
for (const image of evidenceImages) {
  const bytes = await fs.readFile(`${mediaDir}/${image.file}`);
  const dataUrl = `data:image/png;base64,${bytes.toString("base64")}`;
  sheet.images.add({
    dataUrl,
    anchor: {
      from: { row: image.row - 1, col: 20, rowOffsetPx: 6, colOffsetPx: 8 },
      extent: { widthPx: image.width, heightPx: image.height },
    },
  });
}

// Summary and interpretation legend.
sheet.getRange("A16:AC16").merge();
sheet.getRange("A16").values = [["结论与标注说明"]];
sheet.getRange("A16:AC16").format = {
  fill: "#17365D",
  font: { name: "Microsoft YaHei", size: 12, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
const summaryRows = [
  ["推荐主比较顺序", "SENDE-US/CA（2026、任务同构） > MemSum（状态/历史感知） ≈ Agarwal（法律抽取+冗余） > SummaRuNNer（经典监督抽取） > CivilSum/IGCS/FedSum（高等级但任务不完全同构）"],
  ["CCF口径", "A/B/C按CCF推荐国际学术刊物目录标注。Findings、Workshop、arXiv、AACL和JURIX不自动继承主会等级；FedAvg的A级仅指本表对照结果来自AAAI 2025的FedSum论文。"],
  ["颜色说明", "紫色=2025/2026新论文；金色=首要主比较；绿色=方法高度相关；红色=数据口径需要复核；CCF A/B/C分别以红/橙/蓝标示。"],
  ["等级与相关性", "发表等级和方法相关性是两条不同维度：SENDE虽为CCF C，但最贴近本文；CivilSum/FedSum虽为CCF A，却不应取代同任务直接对照。"],
  ["数据口径警告", "IGCS行的42.9建议回到TACL原表复核：不要在未确认前写成“六任务宏平均”。SummaRuNNer与KMM的UK-Abs原文分数来自AACL 2022比较表。"],
];
sheet.getRange("A17").write(summaryRows);
for (let row = 17; row <= 21; row += 1) {
  sheet.mergeCells(`B${row}:AC${row}`);
}
sheet.getRange("A17:A21").format = {
  fill: "#D9EAF7",
  font: { name: "Microsoft YaHei", size: 10, bold: true, color: "#1F4E78" },
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#B4C6E7" },
};
sheet.getRange("B17:AC21").format = {
  fill: "#F7FAFC",
  font: { name: "Microsoft YaHei", size: 10, color: "#334155" },
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#D9E2F3" },
};
sheet.getRange("16:16").format.rowHeight = 28;
sheet.getRange("17:21").format.rowHeight = 48;

// Source notes live beside the visible classification instead of widening the table further.
const paperSources = {
  3: "https://aclanthology.org/2022.acl-long.450/",
  4: "https://aclanthology.org/2022.findings-emnlp.134/",
  5: "https://doi.org/10.1145/3626772.3657859",
  6: "https://arxiv.org/abs/2210.13448",
  7: "https://doi.org/10.1007/s11227-026-08248-4",
  8: "https://doi.org/10.1007/s11227-026-08248-4",
  9: "https://aclanthology.org/2025.tacl-1.74/",
  10: "https://aclanthology.org/2025.justnlp-main.11/",
  11: "https://ojs.aaai.org/index.php/AAAI/article/view/34129",
  12: "https://ojs.aaai.org/index.php/AAAI/article/view/34129",
  13: "https://ojs.aaai.org/index.php/AAAI/article/view/10958",
  14: "https://cse.iitm.ac.in/~ravi/papers/Saravanan_jurix_06.pdf",
};
for (const [rowText, url] of Object.entries(paperSources)) {
  const row = Number(rowText);
  workbook.notes.add({
    id: `${sheet.name}:Q${row}`,
    target: { cell: { sheetName: sheet.name, sheetId: sheet.sheetId, address: `Q${row}` } },
    authorId: "",
    createdAt: "",
    body: { plainText: `论文/结果来源：${url}` },
  });
}
const ccfSource = "CCF推荐目录核验：AAAI、ACL、SIGIR=A；TACL、EMNLP主会=B；The Journal of Supercomputing=C。Findings/Workshop/预印本及未列入来源不自动继承等级。";
for (let row = 3; row <= 14; row += 1) {
  workbook.notes.add({
    id: `${sheet.name}:R${row}`,
    target: { cell: { sheetName: sheet.name, sheetId: sheet.sheetId, address: `R${row}` } },
    authorId: "",
    createdAt: "",
    body: { plainText: ccfSource },
  });
}

sheet.freezePanes.freezeRows(2);
sheet.freezePanes.freezeColumns(3);
workbook.recalculate();

await fs.mkdir(outputDir, { recursive: true });
const previewAnnotations = await workbook.render({
  sheetName: sheet.name,
  range: "A1:T21",
  scale: 0.8,
  format: "png",
});
await fs.writeFile(`${outputDir}/标注区预览.png`, new Uint8Array(await previewAnnotations.arrayBuffer()));
const previewEvidence = await workbook.render({
  sheetName: sheet.name,
  range: "U1:AC14",
  scale: 0.8,
  format: "png",
});
await fs.writeFile(`${outputDir}/逐行截图预览.png`, new Uint8Array(await previewEvidence.arrayBuffer()));

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);

const inspect = await workbook.inspect({
  kind: "region,drawing,formula",
  sheetId: sheet.sheetId,
  range: "A1:AC21",
  maxChars: 12000,
  options: { maxResults: 100 },
});
await fs.writeFile(`${outputPath}.inspect.ndjson`, inspect.ndjson, "utf8");
console.log(outputPath);
