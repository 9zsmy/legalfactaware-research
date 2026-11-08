# CAIL核心方法完整流程耗时对比（补充性计算成本分析）

完成时间：2026-10-22 15:56（Asia/Shanghai）

## 对齐范围

- 数据：冻结的 CAIL2020-SFZY-clean 划分，7,495 篇训练文档、942 篇开发文档。
- 起点：已经切句并保存为 JSONL 的数据；不包括原始数据下载、清洗、公共划分和最终 ROUGE 评价。
- 终点：完成各方法自身的监督/预处理、表示生成、模型训练，以及 942 篇开发集的预算约束推理。
- 每种方法使用一个已冻结配置与随机种子；耗时来自同一台服务器上的独立完整运行。
- 各方法内部阶段并不完全同构，表中的“总耗时”表示真实可复现流水线成本，不是单一算法算子的理论复杂度。
- 本表只覆盖具有可审计本地完整流水线的五种核心方法，不代表论文效果表中的全部基线。Agarwal-ST、Gist、EUR-Lex-Sum LexRank、CivilSum LED-tail 和 Budget Recipe 尚无同口径完整计时，因此不能据此声称相对全部基线整体效率领先。

## 最终结果（一次完整运行）

| 排名 | 方法 | 完整耗时 | 秒 | 开发集输出 | 状态 |
|---:|---|---:|---:|---:|---:|
| 1 | SSE | 00:14:38.29 | 878.287 | 942 | 成功 |
| 2 | LegalFactAware | 00:31:20.36 | 1,880.359 | 942 | 成功 |
| 3 | SENDE | 00:34:15.73 | 2,055.728 | 942 | 成功 |
| 4 | MemSum | 01:37:05.59 | 5,825.591 | 942 | 成功 |
| 5 | TermDiffuSum | 02:35:48.18 | 9,348.182 | 942 | 成功 |

## 阶段分解

| 方法 | 监督/预处理 | 表示/编码 | 训练 | 开发集推理 | 备注 |
|---|---:|---:|---:|---:|---|
| LegalFactAware | 1,694.237 s | 85.892 s | 77.829 s | 17.419 s | 外层实测总耗时为 1,880.359 s |
| SSE | 405.212 s | — | 467.681 s | 5.395 s | 三阶段之和为总耗时 |
| SENDE | 60.983 s | 167.606 s | 1,822.061 s | 5.077 s | 训练含 TSDAE 1,798.189 s 和抽取器 23.872 s |
| MemSum | 476.856 s | — | 5,312.064 s | 36.671 s | 预处理含 SSE 准备和 MemSum 转换 |
| TermDiffuSum | 1.959 s | — | 8,023.507 s | 1,313.546 s | 外层实测总耗时为 9,348.182 s；无硬截断 |

## 可直接用于论文的结论

LegalFactAware 完整流水线用时 31 分 20 秒。在本次单次完整运行中，它比 SENDE 少用约 2 分 55 秒，总耗时低于 MemSum 和 TermDiffuSum，但高于 SSE。由于尚未重复完整流程测量，这些差值只作描述性报告，不表述为稳定或显著的速度优势。LegalFactAware 的主要成本不是线性模型训练或推理，而是训练集 Budget Oracle 构造：1,694.237 秒占外层实测总耗时的约 90.1%。四个已分解阶段之和为 1,875.377 秒，比外层总耗时少 4.983 秒，该部分记为进程启动、文件读写及计时封装等未分配开销。因此，论文中应同时报告完整流水线耗时与冻结模型推理耗时，避免把一次性监督构造成本和部署推理成本混为一谈。

## 论文呈现建议

这组结果适合作为补充性计算成本分析，不作为核心创新，也不支持“整体效率最优”或“最佳效率—效果权衡”的主张。正文保留一段主要观察，完整阶段表放入附录。正文精简表建议如下：

| Method | Pipeline time (min) | Dev-set inference (s) |
|---|---:|---:|
| SSE | 14.64 | 5.395 |
| LegalFactAware | 31.34 | 17.419 |
| SENDE | 34.26 | 5.077 |
| MemSum | 97.09 | 36.671 |
| TermDiffuSum | 155.80 | 1,313.546 |

建议英文正文：

> On the frozen CAIL2020-SFZY-clean split, LegalFactAware required 31.34 minutes for method-specific preparation, training, and inference on 942 development documents. Its pipeline time was lower than those of SENDE, MemSum, and TermDiffuSum, but higher than that of SSE. Budget Oracle construction accounted for 90.1% of the measured total time. Development-set inference took 17.42 seconds, compared with 5.40 seconds for SSE and 5.08 seconds for SENDE. These measurements describe one complete run per method on the same server under the frozen configurations.

表注：流水线从已切句的 CAIL JSONL 开始，包含方法特有的监督/预处理、训练和 942 篇开发集推理；不包含数据下载、公共清洗、划分和最终 ROUGE 评价。每个数字来自同一服务器上冻结配置的一次完整运行。推理阶段的表示起点并不完全一致，SENDE 使用已编码表示，因此该列不能解释为严格同口径的原始文本部署延迟。

## 完整性核验

- 五种方法的运行退出码均为 0。
- 每种方法均产生 942 条开发集预测。
- TermDiffuSum 完成 10 个 epoch，训练 8,023.507 秒，解码 1,313.546 秒，硬截断数为 0。
- TermDiffuSum 临时切换的数据链接已恢复至原 IN-Abs 数据目录。
- 服务器最终保留 5.1 GB 数据盘空间；关键模型、预测、计时文件和哈希记录均保留。

