# LegalFactAware 补充实验完整代码讲解稿

建议时长：15–20 分钟。  
讲解顺序：研究问题 → 代码结构 → 主实验执行顺序 → 跟进诊断 → 结果 → 结论边界。

---

## 一、开场：这套代码解决什么问题

可以直接这样讲：

> 这套代码不是重新寻找更高的测试集分数，而是针对审稿人提出的机制混杂，在冻结开发集上补充受控实验。我们主要回答四个问题：Oracle 监督的优势是否只是正标签更多；TF–IDF 模块的收益究竟来自词法还是位置；八候选上限是否不合理；fit/skip 的提升是否只是因为不再句内截断。之后又补了三项诊断，用来检查标签长度偏好、输出长度混杂和不同表达式类型的表现。

这里首先强调三点：

1. 主实验只有训练集和开发集参数，没有测试集参数。
2. 输入文件先经过 SHA-256 校验，不能无意中换版本。
3. 新结果用于解释机制，不用于回改冻结测试协议或重新调整 SENDE。

---

## 二、先介绍代码包结构

代码分成三层。

### 第一层：核心系统依赖

位于 `LegalFactAware_key_code_20260927/01_chinese_core/`。

讲解时重点介绍五个文件即可：

| 文件 | 作用 | 为什么重要 |
|---|---|---|
| `run_legalfactaware_v2_cail_clean.py` | 读取 CAIL 数据、生成候选、构造基础特征 | 所有实验共享同一候选与底层表示 |
| `run_legalfactaware_v21_budget_decoder_dev.py` | fit/skip 预算解码、预测记录、统一评价 | 决定如何在 450 字和 K 个候选内组成摘要 |
| `run_legalfactaware_v25_structure_feature_dev.py` | Section6 角色识别和 one-hot 特征 | 用来拆分结构辅助特征的贡献 |
| `run_legalfactaware_formal_ablation_dev.py` | 正式评价和 Expression 匹配接口 | 保证新对照与原实验使用同一评价实现 |
| `evaluate_sse_cail.py` | 金额、日期、法律和裁判结果规则 | 分类型 Expression 评价的规则来源 |

其余文件是早期或其他消融实验，也一并打包用于完整归档，但本次讲解不必逐个展开。

### 第二层：本次补充实验

位于 `experiments/chinese_reviewer_controls_20261028/`。

- `run_reviewer_controls.py`：主实验。
- `run_followup_diagnostics.py`：主实验完成后的解释性诊断。
- `summarize_for_talk.py`：现场读取汇总结果并打印关键数字。

### 第三层：结果与审计文件

- `results/all_metrics.json`：各主要变体的汇总指标。
- `results/budget_by_candidate_cap.csv`：完整的 B×K 网格。
- `results/paired_significance.json`：配对显著性结果。
- `results/manifest.json`：主实验输入、输出和已执行脚本哈希。
- `results/followup_manifest.json`：跟进诊断哈希。
- `RESULTS.md`：人工核对后的解释、限制和论文修改建议。

---

## 三、主脚本从上到下的执行顺序

文件：`run_reviewer_controls.py`。

### 第 1 步：冻结实验常量

代码位置：文件顶部 `EXPECTED`、`TRAIN_CASES`、`DEV_CASES`、`MAX_CHARS`、`MAX_CANDIDATES`、`SEED`。

重要参数：

- 训练集 7,495 篇；
- 开发集 942 篇；
- 最大 450 字；
- 最多 8 个候选；
- redundancy weight 为 0.2；
- fact weight 为 0.0；
- 随机种子为 42。

这里的重点不是参数本身，而是**所有对照使用同一组固定参数**。因此对照之间的差异不会来自临时更换解码设置。

### 第 2 步：校验输入与建立输出目录

关键函数：

- `sha256()`：逐块计算文件哈希；
- `require()`：哈希不一致立即报错；
- `parse_args()`：声明所有外部输入路径。

可以指出：`parse_args()` 中没有 `--test`，这是一条很直观的“测试集未进入脚本”证据。

进入 `main()` 后，程序创建四类输出目录：

- `models/`；
- `predictions/`；
- `per_sample/`；
- `tables/`。

然后导入同一套基础候选、预算解码、结构特征和评价模块，并校验冻结输入哈希与样本数量。

### 第 3 步：构造正例数完全匹配的监督对照

关键函数：`build_training()`。

执行逻辑：

```text
对每篇训练文档：
1. 读取 Oracle 选中的候选集合；
2. 令 k 等于该文档的 Oracle 正例数；
3. 独立计算每个候选对参考摘要的 clipped ROUGE-2 precision；
4. 按分数排序，取前 k 个作为 matched positives；
5. 分数相同时按原文位置提前者优先；
6. 保存 Oracle 与 matched 的索引、Jaccard 和是否完全相同。
```

为什么重要：原来的独立标签规则可能产生更多正例，导致比较不公平。这里在**每篇文档内部**精确匹配正例数量，因此排除了标签数量差异。

需要说明的限制：它仍没有匹配候选长度、段落角色、评分分布和最终输出长度。因此这个实验能说“正例数量不是唯一原因”，不能说“候选竞争独立造成全部提升”。

### 第 4 步：构造五种特征矩阵

代码在 `main()` 的 `building training matrices` 之后。

五种表示为：

1. `full_reference`：带 position marker 的 decorated text + Legal14 + Section6；
2. `lexical_full_aux`：raw lexical text + Legal14 + Section6；
3. `decorated_text_only`：raw text + position marker；
4. `lexical_text_only`：只有 raw lexical text；
5. `position_marker_only`：只有 position marker。

关键实现：

- `vectorizer.transform(...)` 使用同一冻结 TF–IDF 词表和 IDF；
- `base.explicit_features(...)` 生成 Legal14；
- `section_one_hot(...)` 生成 Section6；
- `hstack(...)` 组合稀疏矩阵。

为什么重要：原论文将词法和位置放在同一 TF–IDF 块里，整个块的消融不能证明位置有效。现在把两者拆开，能够判断主要信号来自哪里。

### 第 5 步：训练控制模型

关键函数：`fit_lr()`。

所有新模型统一使用：

```text
LogisticRegression
solver = liblinear
class_weight = balanced
max_iter = 1000
random_state = 42
```

完整参考模型不重新训练，而是载入冻结分类器。其余模型只改变标签或输入特征，训练器保持一致。

### 第 6 步：统一计算开发集候选分数

程序逐篇处理 942 篇开发文档：

1. 生成候选句和 Section6 行记录；
2. 构造与训练阶段对应的五种特征矩阵；
3. 调用每个模型的 `predict_proba()`；
4. 将候选正类概率保存到 `score_rows`。

这里的设计重点是：**先统一得到候选分数，再让不同解码对照复用同一分数**。这样 drop-only 和 B×K 网格不会因为重新训练而改变评分器。

### 第 7 步：fit/skip 预算解码

关键函数：`generate_budget()`。

它调用核心模块中的 `select_budget_greedy()`，在候选数上限和字符预算下选择候选。每篇文档另外记录：

- `cap_hit`：是否选满 K 个候选；
- `budget_skip_observed`：过程中是否遇到放不下的候选；
- `terminal_stop`：最终为什么停止；
- `remaining_characters`：剩余预算。

最重要的代码细节：

```text
如果已达到 K → candidate_cap
否则，如果仍有候选但没有任何一个能放入剩余预算 → character_budget
否则 → candidate_pool
```

因此，“曾经跳过候选”和“最终因预算停止”不是一回事。

### 第 8 步：Lead/Tail 位置基线

关键函数：`generate_lead_tail()`。

- Lead 从文档开头按顺序取能放入预算的候选；
- Tail 从文档结尾向前取；
- 都使用相同的 450 字和最多 8 候选约束。

它们的目的不是追求强性能，而是验证“只靠位置”是否足以解释结果。

### 第 9 步：drop-only 与旧截断对照

关键函数：`generate_drop_only()`。

执行过程：

1. 用旧版选择器先选八个候选；
2. `legacy_output` 保留旧的句内截断方式；
3. drop-only 从已选列表尾部删除完整候选，直到不超过 450 字；
4. drop-only 不允许其他候选替补。

为什么重要：这一步把两个机制区分开：

- 不在候选内部截断；
- 跳过放不下的候选后，继续让其他候选参与选择。

注意：这不是完全可加和的因果分解，因为选择路径之间可能交互。

### 第 10 步：运行 B×K 敏感性网格

循环设置：

- B ∈ {300, 400, 450}；
- K ∈ {4, 6, 8, 10, 12, all}。

总共 18 个设置，全部复用冻结完整评分器，不重新训练。

每个设置报告：

- R-1/R-2/R-L；
- Expression recall/F1；
- 平均字符数；
- 平均候选数；
- 候选上限命中率；
- 预算跳过率；
- 三种最终停止原因。

这个网格回答的是“现有 K=8 是否明显不合理”，不是用开发集重新寻找并替换测试协议。

### 第 11 步：统一评价

关键函数：`evaluate_variant()` 与 `compact()`。

`evaluate_variant()` 做三件事：

1. 调用同一 `decoder.evaluate_predictions()` 计算 LAWA ROUGE；
2. 调用同一 `fact_reference_metrics()` 计算规则表达式匹配；
3. 记录字符数、候选数、未用预算和硬截断次数。

`compact()` 将指标整理成表格格式。当前代码明确区分：

- `Expression_Recall_conditional_reference_types`；
- `Expression_Recall_macro_union_types`；
- `Expression_F1_macro_union_types`。

为什么重要：旧正文的条件式 Recall 与新汇总中的固定类型并集 Recall 不是同一指标，不能只改名字后混用。

### 第 12 步：配对统计检验

关键函数：`paired_test()` 和 `holm()`。

统计流程：

1. 先验证两种方法的文档 ID 顺序完全一致；
2. 对每篇文档计算指标差；
3. 进行 10,000 次文档 bootstrap，得到平均差的 95% CI；
4. 使用 Pratt-Wilcoxon，保留大量零差文档；
5. 同一比较中的四项指标使用 Holm 校正。

三组正式配对比较：

- full − matched-count independent；
- full − drop-only；
- drop-only − legacy truncation。

没有 SENDE 配对显著性，因为本次运行没有版本一致、逐文档对齐的 SENDE 输出。

### 第 13 步：空表达审计和结果 manifest

`expression_audit()` 统计参考摘要中四类表达式的非空文档数，并检查是否存在所有类型均为空的文档。

最后保存：

- 汇总指标；
- CSV 表；
- 配对检验；
- 配置；
- 所有输出的 SHA-256 和字节数。

`manifest.json` 的作用是证明“这组数值来自哪些输入和哪些输出”，是复现审计的核心文件。

---

## 四、跟进诊断脚本的执行顺序

文件：`run_followup_diagnostics.py`。

这个脚本不重新训练模型，只读取主实验已经生成的预测和逐文档数值。

### 第 1 步：再次校验数据版本

`require()` 校验训练、开发和 Oracle 文件哈希。脚本同样没有测试集参数。

### 第 2 步：分析训练正例偏好

关键函数：`label_diagnostics()`。

它重新构造两套标签，然后统计：

- 正例候选字符长度；
- 归一化源位置；
- claim、decision、defense、fact、other、reason 六类 Section6 分布；
- 每篇文档内 Oracle 与 matched 的平均长度差和位置差；
- bootstrap 区间与 Wilcoxon p 值。

核心发现：Oracle 正例在逐文档比较中平均长约 11.10 字，说明最终输出长度差异有明确的训练标签来源。

### 第 3 步：实际输出长度接近比较

关键函数：`length_close()`。

它选择 Full 与 matched 输出字符差不超过 10、25、50 字的文档子集，然后重新计算配对差和区间。

- 25 字是预先指定的主要描述性阈值；
- 10 字更严格但样本只有 177 篇；
- 子集由实际输出决定，因此仍存在选择偏差。

正确解释：长度解释了部分 ROUGE 差距；25 字子集仍有较小差异，但 10 字子集差异不可靠。不能说长度混杂已完全排除。

### 第 4 步：分类型 Expression 评价

关键函数：`expression_by_type()`。

分别对 amount、date、law、decision 计算：

- 参考实例数；
- 预测实例数；
- 精确匹配实例数；
- micro precision/recall/F1；
- 只在参考非空文档上计算的 conditional document-macro precision/recall/F1。

这样做的原因：日期只有 93 篇参考非空文档，而法律表达有 942 篇。只报告总体分数可能掩盖低频类型。

这些仍是规则词面匹配，不能称为语义法律事实正确性。

### 第 5 步：保存跟进 manifest

脚本保存三个纯汇总 JSON，并记录所有输入预测、逐文档数值以及输出文件的哈希。原始法律文本不会进入交付包。

---

## 五、现场展示结果的顺序

先运行：

```bash
python3 experiments/chinese_reviewer_controls_20261028/summarize_for_talk.py
```

然后按输出顺序讲。

### 结果 1：正例数量匹配

- 两种标签均为 47,150 个训练正例；
- Full：60.27 / 41.17 / 49.84；
- matched：58.29 / 38.92 / 47.47；
- 差值：+1.99 / +2.25 / +2.37；
- Expression F1 差异不显著。

结论：Oracle 的 ROUGE 优势不只来自更多正标签，但标签长度和最终输出长度仍构成混杂。

### 结果 2：词法与位置

- raw lexical only：59.86 / 40.55 / 49.20；
- position marker only：38.92 / 15.85 / 24.64。

结论：主要信号来自词汇内容；位置、Legal14 和 Section6 是较小的辅助增益。

### 结果 3：K=8

在 B=450、K=8：

- 803/942 最终因候选上限停止；
- 139/942 最终因字符预算停止；
- 放宽到 10、12 或不限候选后，输出更长但指标下降。

结论：K=8 是一个实质性约束，在当前冻结评分器的开发集上表现为有效正则；不证明对所有模型普遍最优。

### 结果 4：drop-only

- Full RL 49.84，Expression F1 47.85，硬截断 0；
- drop-only RL 48.33，Expression F1 44.53，硬截断 0；
- legacy RL 48.66，Expression F1 45.50，硬截断 379。

结论：只取消句内截断不保证自动指标提升。结果支持继续选择替代候选的重要性，同时 Full 消除了候选内硬截断。

### 结果 5：分类型表达式

Full 的 micro F1：

- amount 39.10；
- date 23.10；
- law 50.58；
- decision 79.34。

结论：不同类型差异很大，总体 Expression F1 不能证明所有法律表达类型均改善。

---

## 六、哪些代码最重要

如果时间有限，只展示以下六处：

1. `EXPECTED` 与 `parse_args()`：说明输入冻结且无测试参数。
2. `build_training()`：说明逐文档匹配正例数量。
3. 五种特征矩阵：说明如何拆分词法和位置。
4. `generate_budget()`：说明 fit/skip 与停止原因。
5. `generate_drop_only()`：说明解码机制对照。
6. `paired_test()`：说明配对 bootstrap、Wilcoxon 与 Holm 校正。

跟进诊断中只需展示：

7. `label_diagnostics()`；
8. `length_close()`；
9. `expression_by_type()`。

---

## 七、哪些内容不要过度讲

不要声称：

- “实验证明竞争机制独立造成全部提升”；
- “位置特征是系统效果的主要来源”；
- “K=8 是所有设置下的最优候选数”；
- “避免截断本身提高了 ROUGE”；
- “Expression F1 等于法律事实正确率”；
- “候选边界完整等于语义或裁判结果完整”；
- “SENDE 已通过配对显著性检验”。

更稳妥的说法：

> 在逐文档匹配正例数量后，Oracle 标签仍保持显著的开发集 ROUGE 优势；词汇内容构成主要预测信号；预算感知的替代候选选择优于只删除而不替补。上述结果是机制证据，不替代新颖性比较、主表版本统一和人工法律事实评价。

---

## 八、常见追问与回答

### 1. 为什么不直接在测试集补这些实验？

这些是机制诊断和敏感性分析。为避免使用测试集调解释或调参数，全部在冻结开发集完成；测试集只保留原定的最终报告功能。

### 2. 为什么匹配正例数后输出仍然更短？

正例数量相同不代表正例内容分布相同。跟进诊断发现 matched 正例更短，段落角色分布也不同；评分器和解码器最终因此偏好不同长度的候选。

### 3. 25 字接近子集能否证明长度不是原因？

不能。它只说明在一个实际输出较接近的子集中仍有较小 ROUGE 差异。子集由输出决定，而且 10 字子集的区间跨零，所以长度仍解释了部分差异。

### 4. K=8 经常触发是不是说明预算没作用？

不是。候选上限和字符预算分别影响选择。803 篇最终由 K 停止，139 篇最终由字符预算停止；另外有 724 篇在过程中至少跳过过超预算候选。

### 5. 为什么 drop-only 比旧截断的 ROUGE 还略低？

因为取消句内截断会保留候选边界，但也可能丢失部分高重叠文本。完整 fit/skip 的优势在于还能让后续完整候选替补，因此完整性与自动指标之间存在权衡。

### 6. 为什么 Expression 总体差异小，分类型差异却很大？

四类表达式的支持度差异很大，而且固定类型并集与条件式宏平均处理空类型的方式不同。总体平均可能掩盖金额、日期等低频类型。

### 7. 这套代码是否已经解决论文所有问题？

没有。它解决的是开发集机制证据。创新定位、主表基线版本统一、成本配置统一和盲法人工法律事实评价仍需另行完成。

---

## 九、结束语

可以用下面这段收尾：

> 这次补充实验最重要的价值不是再增加一个更高分数，而是把原来混在一起的因素拆开。正例数量匹配后 Oracle 仍有 ROUGE 优势，但长度偏好解释了其中一部分；主要有效信号是词汇内容，而不是位置标记；完整 fit/skip 的作用也不仅是取消句内截断，而是允许替代候选继续参与选择。这样修改以后，论文的机制表述更窄，但证据更可信。

