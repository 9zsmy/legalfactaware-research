# LegalFactAware 论文与实验总整理

更新日期：2026-11-08  
用途：作为网页版 GPT 修改英文论文时的唯一实验依据。除非本文件明确允许，不得把不同数据划分、评价器或复现等级的结果合并。

## 1. 当前论文与总体状态

- 当前英文稿源文件：`paper_source/main.tex`。
- 当前论文采用 CVPR 双栏格式，主体包括方法、CAIL 中文实验、消融、案例、复现说明和计算成本。
- 当前 `main.tex` 尚未写入后来完成的 SENDE 0.9、审稿人控制实验和 2025–2026 新方法对照。
- 论文当前最需要做的是统一证据和表述，而不是继续扩展方法模块。
- 当前最稳妥的论文主线是：
  1. 用参考摘要构造摘要级 Oracle 成员标签；
  2. 用静态线性评分器学习候选重要性；
  3. 用冗余感知的完整候选 fit/skip 解码，在硬字符预算内继续寻找可放入的替代候选；
  4. 轻量词法表示是主要预测信号，Legal14 和 Section6 是辅助特征；
  5. 结果支持完整监督流程与完整候选解码，但不支持把 Oracle 字符预算、辅助特征或效率单独夸大为核心创新。

## 2. 方法的准确描述

任务是在每篇中文裁判文书中选择若干完整候选，在不超过 450 个“存储字符串字符”且最多 8 个候选的条件下生成抽取式摘要。

训练阶段：

1. 对训练文档中的候选逐步计算加入当前摘要后的边际 ROUGE 增益。
2. 在预算与候选上限下贪心构造参考引导的 Oracle 候选集合。
3. Oracle 集合中的候选得到正成员标签，其余得到负标签。
4. 用字符 n-gram TF–IDF、位置标记、Legal14 和 Section6 特征训练类别平衡的 Logistic Regression。

推理阶段：

1. 冻结线性评分器并为候选打分。
2. 使用与已选内容的重复度进行冗余感知重排序。
3. 若当前候选完整放得下则选入；若放不下则跳过，并继续检查后续候选。
4. 最终按原文顺序输出完整候选，不进行候选内部截断。

术语边界：论文中的 “complete” 或 “integrity” 只能指候选边界完整，不能外推为法律事实完整、语义完整或裁判结果完整。

## 3. 数据划分与评价口径

### 3.1 中文 CAIL 主任务

- 数据：CAIL2020-SFZY-clean。
- 训练集：7,495 篇。
- 开发集：942 篇。
- 测试集：948 篇。
- 正式约束：不超过 450 个存储字符串字符，最多 8 个完整候选。
- 948 篇测试主表使用论文现有 LAWA ROUGE 口径和 legacy conditional Fact Recall。
- 942 篇审稿人机制实验使用开发集；其中 `Expression F1` 是规则表达匹配，不等同于法律事实正确性。

### 3.2 新生成式基线的统一评价

- 新生成式基线的 ROUGE 使用“去空白字符级 F1”，不是论文旧主表的 LAWA 分词口径。
- 因此，即使同为 942 篇开发集，新生成式基线的 65.53/51.41/52.55 也不能替代旧消融表中 LegalFactAware 的 60.27/41.17/49.84。
- 新生成式方法必须与同一评价器下重新计算的 LegalFactAware 比较，并放入独立表格。
- 100 篇子集结果不能与 942 篇开发集或 948 篇测试集结果混为全量结果。

### 3.3 英文 IN-Abs

- 现有英文开发协议为 6,325 train / 703 dev；官方 100 篇 test 未读取。
- 英文 E1、DELTA 50 篇 pilot 和历史基线尚未形成可直接投稿的统一正式主表。
- 当前论文不应宣称在 IN-Abs 官方测试集达到最优，也不应把英文开发集结果写成官方测试结果。

## 4. 948 篇中文测试集主表：应采用的版本

下表中的 SENDE 已按用户决定改为固定阈值 0.9。其余方法数字不变。

| 方法 | 实验身份 | R1 | R2 | RL | Fact Recall |
|---|---|---:|---:|---:|---:|
| **LegalFactAware v2.5-B** | Proposed method | **60.94** | **42.10** | **50.73** | **70.99** |
| SENDE | Independent task reconstruction; fixed threshold 0.9 | 57.59 | 39.39 | 49.00 | 62.72 |
| SSE | Structural proxy adaptation | 57.14 | 37.25 | 46.25 | 61.47 |
| Agarwal-ST | Constrained ST+MMR | 54.98 | 35.90 | 45.66 | 70.69 |
| TermDiffuSum | Independent reimplementation | 54.55 | 34.98 | 43.80 | 60.21 |
| Gist | Language/budget adaptation | 53.62 | 34.35 | 44.27 | 69.42 |
| MemSum | Official architecture retrained | 49.56 | 28.16 | 38.52 | 55.66 |
| EUR-Lex-Sum LexRank | Graph-ranking adaptation | 42.18 | 21.23 | 29.84 | 36.91 |
| CivilSum LED-tail | Generative reference track | 53.57 | 28.59 | 41.35 | 43.60 |
| Budget Recipe | Zero-shot generative transfer | 40.19 | 19.05 | 26.26 | 33.46 |

LegalFactAware 相对 SENDE 0.9 的差值是：

- R1：+3.35；
- R2：+2.71；
- RL：+1.73。

这里没有版本匹配的 SENDE 配对显著性检验，因此只能报告均值差，不能声称显著优于 SENDE。

### SENDE 0.9 的补充审计信息

- 948 篇测试集。
- Fact F1：43.91。
- Sentence F1：53.20。
- Source Fact Support：100.00。
- 平均长度：342.93 字符。
- 平均候选数：6.13。
- 空摘要：0。
- 超预算：0。
- 候选内部硬截断：0。

必须披露：0.9 是固定配置，不是开发集选择的最优阈值。开发集选择的最优阈值是 0.8，但论文按用户决定报告固定 0.9。不能写成“在开发集选择 0.9”，也不能继续保留旧结果 59.72/41.13/50.65、66.91 或旧差值 1.22/0.97/0.08。

## 5. 942 篇开发集：机制实验的完整结论

### 5.1 基础完整系统

| 方法 | R1 | R2 | RL | Expression F1 | 平均字符 | 平均候选 |
|---|---:|---:|---:|---:|---:|---:|
| Full LegalFactAware | 60.27 | 41.17 | 49.84 | 47.85 | 396.49 | 7.78 |

### 5.2 正标签数量精确匹配

| 训练标签 | R1 | R2 | RL | Expression F1 | 平均字符 | 平均候选 |
|---|---:|---:|---:|---:|---:|---:|
| Oracle membership | 60.27 | 41.17 | 49.84 | 47.85 | 396.49 | 7.78 |
| 独立相似度标签，逐文档正例数匹配 | 58.29 | 38.92 | 47.47 | 47.79 | 338.58 | 7.97 |
| Oracle − matched | **+1.99** | **+2.25** | **+2.37** | +0.06 | +57.90 | −0.19 |

配对统计：

| 指标 | 均值差 | 95% bootstrap CI | Holm-adjusted Wilcoxon p |
|---|---:|---:|---:|
| R1 | +1.99 | [1.63, 2.35] | 6.23e-23 |
| R2 | +2.25 | [1.82, 2.69] | 1.39e-18 |
| RL | +2.37 | [1.93, 2.81] | 4.11e-22 |
| Expression F1 | +0.06 | [−0.75, 0.89] | 0.157 |

可以写：逐文档精确匹配正标签数量后，Oracle 成员标签仍保持可靠的 ROUGE 优势。

不能写：该实验证明候选竞争独立造成全部提升，或者证明法律事实质量提高。两组输出平均长度仍相差约 58 字符，Expression F1 没有可靠提升。

补充诊断显示，Oracle 训练正例平均 54.18 字符，匹配独立标签正例平均 44.27 字符；Oracle 明显偏好更长候选。输出长度相差不超过 25 字符的 308 篇子集中仍有 +0.76/+0.54/+0.78 ROUGE 优势，但这是描述性子集证据，不是完全去混杂的因果结论。

### 5.3 词法、位置和辅助特征拆分

| 变体 | R1 | R2 | RL | Expression F1 | 平均字符 |
|---|---:|---:|---:|---:|---:|
| Full: decorated text + Legal14 + Section6 | 60.27 | 41.17 | 49.84 | 47.85 | 396.49 |
| Raw lexical text + Legal14 + Section6 | 60.00 | 40.75 | 49.38 | 47.16 | 396.01 |
| Decorated text only | 59.91 | 40.66 | 49.27 | 47.34 | 396.66 |
| Raw lexical text only | 59.86 | 40.55 | 49.20 | 47.05 | 396.68 |
| Position marker only | 38.92 | 15.85 | 24.64 | 22.31 | 341.26 |
| Lead fit/skip | 17.86 | 3.30 | 13.36 | 0.16 | 191.63 |
| Tail fit/skip | 28.26 | 10.24 | 17.29 | 25.54 | 252.21 |

正确结论：词法内容是主要预测信号；位置标记和 Legal14/Section6 提供较小辅助增益。旧论文把移除整个 position-augmented TF–IDF 的大幅下降解释成“位置作用”的写法必须改掉。

### 5.4 八候选上限与字符预算

在 B=450、K=8 时，803/942 篇达到候选上限，724/942 篇在选择过程中至少遇到一个放不下的候选；最终停止原因中，803 篇因达到 K=8 停止，139 篇因所有剩余候选均放不下而停止。

| K | R1 | R2 | RL | Expression F1 | 平均字符 | 达到上限率 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | **60.27** | **41.17** | **49.84** | **47.85** | 396.49 | 85.24% |
| 10 | 59.30 | 40.51 | 49.04 | 46.89 | 431.35 | 57.43% |
| 12 | 58.73 | 40.12 | 48.60 | 46.49 | 443.49 | 22.40% |
| 不限 | 58.46 | 39.91 | 48.39 | 46.36 | 447.36 | 0% |

可以写：对冻结评分器，放宽候选上限使输出更长，但没有提高开发集 ROUGE 或 Expression F1，因此 K=8 在当前协议中表现为有效的解码约束。

不能写：K=8 对所有模型都全局最优，或在重新训练条件下仍必然最优。

### 5.5 解码机制：fit/skip、drop-only 与尾部截断

| 解码器 | R1 | R2 | RL | Expression F1 | 平均字符 | 硬截断 |
|---|---:|---:|---:|---:|---:|---:|
| Full fit/skip with replacements | 60.27 | 41.17 | 49.84 | 47.85 | 396.49 | 0 |
| Drop-only，不补入替代候选 | 59.22 | 39.97 | 48.33 | 44.53 | 376.81 | 0 |
| Legacy post-hoc tail truncation | 59.39 | 40.44 | 48.66 | 45.50 | 401.85 | 379 |

Full − drop-only 为 +1.05/+1.20/+1.51 ROUGE 和 +3.32 Expression F1；四项区间均不跨 0。

正确结论：仅取消候选内部截断并不会自动提高指标；完整 fit/skip 的优势还来自跳过放不下的候选后继续允许其他完整候选竞争。

不要使用 “gains primarily by ...” 这种暗示完成了加性因果分解的表述。可写成 “the results support the importance of admitting later feasible candidates”。

### 5.6 其他仍然有效的控制

- 去掉 Oracle 字符预算的匹配控制：60.34/41.27/49.99，Expression F1 47.78；与完整系统的四项差异均不显著。
- 无冗余重排时，greedy fit/skip 为 60.12/41.06/49.75，Expression F1 47.63；精确 0/1 knapsack 虽提高静态分数和，却没有提高摘要指标，校正检验不显著。
- 因此不能把 Oracle 字符预算或精确 knapsack 作为已有证据支持的独立显著贡献。

### 5.7 Expression 指标与分类型结果

- 开发集参考中含金额/日期/法律/裁判表达的文档数分别为 303/93/942/908。
- 942 篇中不存在所有类型均为空的参考，因此“全空时赋 1”不会影响开发集均值。
- fixed-union Expression Recall 与旧论文 legacy conditional Fact Recall 是不同定义，不能通过改名合并。
- 论文正文必须只选定一个主要定义，并为所有方法统一重算；另一种定义放附录。
- Expression 的精确匹配只能衡量词面覆盖，不能证明法律事实语义正确。

## 6. 计算成本：冻结结果

数据为同一服务器、942 篇开发集。完整流程运行 3 次；统一起点推理运行 5 次。

| 方法 | 完整流程（分钟） | 942 篇推理（秒） |
|---|---:|---:|
| SSE proxy | **14.42 ± 0.06** | **4.00 ± 0.04** |
| LegalFactAware | 31.18 ± 0.09 | 16.03 ± 0.06 |
| SENDE reconstruction | 34.19 ± 0.18 | 19.52 ± 0.38 |
| MemSum adaptation | 95.86 ± 0.46 | 35.78 ± 0.19 |
| TermDiffuSum reimplementation | 155.52 ± 0.05 | 1,341.11 ± 4.11 |

可以写：LegalFactAware 在五个可审计实现中，完整流程和批量推理成本均为第二低；相对 SENDE 分别低 8.81% 和约 17.9%。

不能写：LegalFactAware 是全局最快、能耗最低或单请求延迟最优。

## 7. 2025–2026 新方法对照

### 7.1 论文身份与当前复现等级

| 方法 | 年份与级别 | 本地实现状态 | 可进入论文的身份 |
|---|---|---|---|
| Length Controlled Generation for Black-box LLMs (LCG) | ACL 2025 主会长文 | 作者未提供方法代码；依据论文公式和提示独立实现。beam-1 完成 100 篇，beam-16 仅完成预注册单篇验收 | `LCG-MHIS-beam1-Qwen2.5-7B-CAIL-adapted (independent)` |
| Zero-Shot Strategies for Length-Controllable Summarization | Findings of NAACL 2025 | 独立实现 SF3+AR5；完成 100 篇中文适配 | `ZeroShotLC-SF3-AR5-Qwen2.5-7B-CAIL-adapted (independent)` |
| DELTA: Training-Free Character-Length Control in Summarization with Diffusion Language Models | EMNLP 2026 主会 | 使用作者官方解码核心；Dream 7B 4-bit；完成 XSum 50 篇回归与 CAIL 942 篇适配 | `DELTA-Dream-CAIL-adapted (official decoding core)` |
| HiGoE、PACO、Principled Content Selection 等 | 曾作为候选阅读 | 当前目录没有可审计的正式运行结果 | 只能用于相关工作或未来实验，不能写入结果表 |

“CAIL-adapted”表示把模型、提示、计数单位和输入处理适配到中文法律任务，不是作者在 CAIL 上发布的原始结果，也不是完整复现论文的全部英文主表。

### 7.2 DELTA：942 篇完整开发集对照

这张表使用统一的去空白字符级 ROUGE，不能与 948 篇 LAWA 主表合并。

| 方法 | R1 | R2 | RL | Expression F1 | 来源表达支持 | 平均字符 | 超预算 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LegalFactAware v2.5-B | **65.53** | **51.41** | **52.55** | **47.85** | 100.00 | 396.49 | 0/942 |
| DELTA-Dream，官方核心 CAIL 适配 | 34.21 | 16.76 | 19.95 | 19.57 | 85.96 | 446.49 | 55/942 |

LegalFactAware − DELTA 的均值差为 R1 +31.32、R2 +34.65、RL +32.60、Expression F1 +28.29，四项配对 bootstrap 95% 区间均不跨 0。

DELTA 的长度控制行为经过 XSum 50 篇官方任务回归：字符长度 MAE 2.0，50/50 输出误差小于 10 字符，接近作者仓库完整 XSum 报告的 2.3。该回归只验证解码行为，不是完整主表复现。

在 CAIL 上，DELTA 平均绝对长度误差为 3.82 字符，96.07% 输出在目标长度 ±10 字符内，但 55/942 超过 450 字符，最大 458。正确解释是它擅长接近指定目标长度，而本文任务要求严格上限；两者目标不同。

### 7.3 冻结 100 篇资源受限对照

| 方法 | R1 | R2 | RL | Expression F1 | 平均字符 | 超预算 |
|---|---:|---:|---:|---:|---:|---:|
| LegalFactAware | **65.94** | **52.19** | **53.38** | **47.30** | 388.52 | 0/100 |
| LCG-MHIS beam-1 exact（仅诊断） | 54.69 | 37.68 | 38.27 | 35.66 | 430.41 | 51/100 |
| LCG-MHIS beam-1 upper | 48.23 | 31.15 | 34.77 | 28.35 | 206.95 | 0/100 |
| ZeroShotLC SF3-AR5 upper | 43.72 | 26.05 | 28.99 | 25.53 | 185.14 | 0/100 |
| DELTA-Dream | 32.79 | 15.70 | 19.17 | 19.18 | 446.84 | 3/100 |

注意：

- LCG exact 有 51/100 超预算，只能作为搜索机制诊断，不是正式硬上限方法。
- LCG upper 仅 2/100 触发迭代，主要反映 Qwen 直接生成，而不是充分激活的 MH+IS 搜索。
- LCG 论文主配置 beam=16 的单篇验收耗时 856.39 秒，正式上限口径下初始候选均已合规、搜索 0 次；精确长度诊断最终 454 字符，未通过扩展到 100 篇的预注册门槛。
- ZeroShotLC 的 100 篇均未触发自动修订，实际表现为三次零样本生成后选择最长合规候选；预算合规但明显偏短。
- LegalFactAware − ZeroShotLC 的配对差为 +22.22/+26.14/+24.40 ROUGE 和 +21.76 Expression F1，四项 95% 区间均不跨 0。
- 该表必须标为冻结 100 篇子集，不能称为 942 篇完整结果。

### 7.4 英文 IN-Abs DELTA pilot：只能作为可行性结果

50 篇冻结开发子集、643 词轨道：

| 方法 | R1 | R2 | RL | 超过 643 词 |
|---|---:|---:|---:|---:|
| E1 complete-sentence | **56.73** | **33.24** | **53.51** | 0/50 |
| DELTA-Dream character-calibrated | 52.57 | 28.63 | 48.59 | 17/50 |

E1 − DELTA 的差为 +4.16/+4.61/+4.91，区间均不跨 0。DELTA 50 篇耗时 2.014 小时，41/50 输入因上下文限制被截断。由于 DELTA 控字符而正式协议控词，且 34% 超过 643 词，该结果不能作为无条件的正式英文主表比较。当前建议不把它写入论文主体有效性结论。

## 8. 当前证据已经解决的问题

| 质疑 | 当前状态 |
|---|---|
| Oracle 优势是否只是正标签更多 | 已明显缓解：正例数量逐文档精确匹配后仍有可靠 ROUGE 优势 |
| 词法和位置是否被混在同一特征块 | 已解决：词法是主要信号，位置和辅助特征只带来小增益 |
| 八候选上限是否随意 | 已有敏感性证据：它频繁触发，但对冻结评分器放宽后指标下降 |
| fit/skip 是否只是“不截断” | 已明显解决：drop-only 对照表明继续选择替代候选很重要 |
| 空表达赋分是否抬高开发集结果 | 已排除：开发集没有全空参考文档 |
| 计算成本是否可审计 | 已完成：5 个实现、3 次完整流程、5 次推理 |
| 是否有 2025–2026 新长度控制对照 | 已有：DELTA 942 篇；LCG 与 ZeroShotLC 100 篇资源受限适配 |

## 9. 仍未解决或不能夸大的问题

1. 与既有 Oracle 监督工作的创新边界仍需在 Related Work 和贡献表述中收紧；“使用 Oracle”本身不是新颖点。
2. 主表 SSE proxy 与历史显著性分析中的 SSE-v2 不是同一实现，不能把后者显著性直接归给主表前者。
3. SENDE 0.9 没有版本匹配的逐文档配对显著性结果。
4. Expression/Fact 指标是规则词面匹配，不能替代盲法法律事实与裁判结果人工评价。
5. 尚未完成正式人工法律事实评价。
6. IN-Abs 官方测试集未读取，英文扩展尚未闭环。
7. 新生成式结果是 CAIL 适配，不代表原论文在英文任务上的总体优劣。

## 10. 推荐的论文贡献表述

建议保留三点贡献，避免把辅助特征拆成独立贡献：

1. A reference-guided summary-level membership target construction procedure for fixed-budget legal extraction, evaluated against both ordinary independent labels and per-document positive-count-matched labels.
2. A redundancy-aware feasible decoding procedure that preserves candidate boundaries and continues selection after an infeasible candidate, evaluated against both drop-only and post-hoc truncation controls.
3. A layered empirical evaluation covering frozen test comparisons, development-set mechanism controls, budget/candidate-cap sensitivity, auditable computational cost, and recent length-control adaptations.

相关措辞应使用 “under the evaluated Chinese task/protocol”、“supports” 和 “is consistent with”，避免使用 universal、proves、state of the art、factually correct 等无充分证据的表述。

## 11. 正文修改优先顺序

### 第一优先级：纠正当前稿中的旧 SENDE 数字

全文将以下内容替换：

- 59.72/41.13/50.65 → 57.59/39.39/49.00；
- Fact Recall 66.91 或 66.90699 → 62.72；
- Fact F1 45.12947 → 43.91149；
- 差值 1.22/0.97/0.08 → 3.35/2.71/1.73；
- 删除 “small ROUGE-L difference” 或相近结论；
- 明确写 `fixed probability threshold 0.9`，并说明 paired significance is not reported。

注意不要误改其他上下文中恰好出现的 41.13。

### 第二优先级：改写摘要、引言和结论

- 摘要中的 `6.24/8.28/8.68` 只能描述完整 Oracle 标签流程相对原始独立弱监督的差异，不能再作为排除标签密度混杂的证据。
- 在摘要或主要消融段加入 matched-count 结果 `1.99/2.25/2.37`，并说明 Expression F1 无可靠差异。
- 把 “position-augmented lexical features” 的大幅下降解释改成“lexical content dominates; positional and auxiliary cues add smaller gains”。
- 把 fit/skip 的机制解释改成“avoids within-candidate truncation and admits later feasible candidates”。
- 结论中明确当前法律事实证据仍是 rule-based expression matching。

### 第三优先级：更新消融与附录

- 加入 matched-count supervision 表。
- 加入 lexical/marker/auxiliary 拆分表。
- 加入 drop-only 对照。
- 加入 K=8/10/12/unlimited 敏感性。
- 明确终止原因与“遇到放不下候选”的统计不是同一个量。
- 统一 Expression 指标定义，并在附录给出分类型结果。

### 第四优先级：加入最新方法，但必须分表

- DELTA 可作为 942 篇开发集的完整生成式对照，放在独立的 recent length-control adaptation 表。
- LCG 和 ZeroShotLC 放在独立的 100 篇 resource-limited comparison 表。
- 表标题必须写清数据规模、去空白字符级 ROUGE、CAIL adaptation 和复现身份。
- 不要把 100 篇结果补进 948 篇测试主表。

### 第五优先级：最后做全文一致性审计

- 数据集名称、划分、预算单位和候选上限是否一致。
- LAWA ROUGE 与字符级 ROUGE 是否混用。
- Fact Recall、Expression Recall、Expression F1 的定义是否一致。
- 所有主张是否能定位到表格、逐文档输出或统计文件。
- 新引用是否加入 bibliography，且 venue 分类准确。
- 页数增加后，表格是否溢出、交叉引用是否正确。

## 12. 可直接用于英文论文的安全结论句

1. `After matching the number of positive labels exactly for every training document, Oracle-membership supervision retains gains of 1.99/2.25/2.37 ROUGE points on the development set; the corresponding Expression-F1 difference is not statistically supported.`
2. `Lexical candidate content is the dominant predictive signal, while position markers and the Legal14/Section6 cues provide smaller auxiliary gains.`
3. `The fit/skip decoder eliminates within-candidate truncation and, unlike a drop-only control, continues to admit later feasible candidates.`
4. `For the frozen scorer, relaxing the eight-candidate cap increases output length but does not improve development ROUGE or Expression F1.`
5. `LegalFactAware has the second-lowest complete-pipeline and batch-inference time among the five audited implementations.`
6. `The DELTA comparison evaluates an official-decoder-core CAIL adaptation with Dream 7B in 4-bit precision; it is not a reproduction of the paper's complete English benchmark table.`
7. `The resource-limited LCG and ZeroShotLC results use a frozen 100-document subset and are reported separately from the full test and development comparisons.`

## 13. 禁止写入的结论

- `The Oracle advantage is entirely caused by content competition.`
- `LegalFactAware improves legal factual correctness.`
- `The eight-sentence cap is universally optimal.`
- `Position features are the main reason for the large TF–IDF ablation gap.`
- `Avoiding truncation alone improves ROUGE.`
- `LegalFactAware is the fastest method.`
- `LegalFactAware significantly outperforms SENDE.`
- `All recent papers were fully reproduced.`
- `DELTA/LCG/ZeroShotLC is generally inferior on its original task.`
- `The 100-document recent-baseline result is a full-dataset result.`
- `The IN-Abs official test confirms cross-lingual superiority.`

## 14. 推荐表格结构

正文建议保留：

1. Table 1：948 篇中文测试集主表，使用更新后的 SENDE 0.9。
2. Table 2：核心开发集机制控制，优先放 matched-count、lexical-only、drop-only 等最能回应审稿人的行。
3. Table 3：计算成本。
4. Table 4：942 篇 DELTA recent length-control adaptation，单独说明字符级评价。
5. Table 5：100 篇 LCG/ZeroShotLC/DELTA resource-limited comparison；若页数紧张可放附录。

附录建议放：

- 完整特征拆分；
- B×K 网格；
- Expression 分类型结果；
- SENDE 0.9 复现细节；
- 新方法复现等级、模型、量化、提示和输入截断说明；
- IN-Abs pilot，仅作为 feasibility/limitation。

## 15. 证据文件索引

- `evidence/Chinese_reviewer_controls_RESULTS.md`：matched-count、词法/位置拆分、K 敏感性、drop-only、表达分类型。
- `evidence/Computational_cost_RESULTS_FINAL.md`：五个实现的成本。
- `evidence/Recent_baselines_RUN_STATUS.md`：DELTA/LCG/ZeroShotLC 总状态。
- `evidence/Recent_baselines_REPRODUCTION_AUDIT.md`：复现等级和允许/禁止主张。
- `evidence/ZeroShotLC_SUBSET100_REPORT.md`：100 篇 ZeroShotLC 结果。
- `evidence/INABS_DELTA_PILOT_REPORT.md`：英文 50 篇可行性实验。
- `evidence/SENDE_threshold09_fact_metrics.json`：SENDE 0.9 精确结果。
- `evidence/Previous_handoff_20261028.md`：旧窗口交接及 SENDE 修改清单。

