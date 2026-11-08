# LegalFactAware 补充实验代码说明

这是面向审稿意见的开发集机制实验。代码只接受冻结的训练集和开发集路径，没有测试集参数。

## 从哪里开始

如果需要进行讲解，建议依次打开：

1. `FULL_CODE_WALKTHROUGH_CN.md`：15–20 分钟的完整逐步讲解稿。
2. `CODE_WALKTHROUGH_CN.md`：约 10 分钟的精简讲解顺序与口径。
3. `run_reviewer_controls.py`：主实验，包含四组对照和预算敏感性分析。
4. `run_followup_diagnostics.py`：不重新训练的后续诊断。
5. `RESULTS.md`：可写入论文的结论、限制和表格。
6. `results/manifest.json` 与 `results/followup_manifest.json`：输入、输出和代码哈希。

现场只想展示结果时，可运行：

```bash
python3 summarize_for_talk.py
```

该脚本只读取 `results/` 中的汇总 JSON/CSV，不读取法律文书、模型或逐文档预测。

## 目录结构

```text
chinese_reviewer_controls_20261028/
├── README_CN.md                    # 本文件：入口说明
├── FULL_CODE_WALKTHROUGH_CN.md     # 15–20 分钟完整讲解稿
├── CODE_WALKTHROUGH_CN.md          # 中文讲解稿
├── PREREGISTRATION.md              # 第一轮实验冻结方案
├── FOLLOWUP_PREREGISTRATION.md     # 跟进诊断冻结方案
├── run_reviewer_controls.py        # 主实验入口
├── run_followup_diagnostics.py     # 跟进诊断入口
├── summarize_for_talk.py           # 汇总结果演示脚本
├── RESULTS.md                      # 完整结果与论文修改建议
└── results/                        # 纯汇总结果和可复现清单
```

`__pycache__/` 是本地语法检查产生的缓存，不属于实验结果。

## 两个脚本分别做什么

### `run_reviewer_controls.py`

主流程按以下顺序运行：

```text
校验冻结输入
  → 构造 Oracle 与逐文档正例数匹配标签
  → 训练监督/特征拆分对照
  → 生成 Lead、Tail、drop-only 和预算网格输出
  → 统一评价
  → 配对 bootstrap + Wilcoxon + Holm 校正
  → 保存汇总、逐文档数值记录和 manifest
```

四个实验问题与代码位置：

| 问题 | 关键函数 | 说明 |
|---|---|---|
| 标签数量是否造成优势 | `build_training` | 每篇文档精确匹配正标签数量 |
| 词法还是位置起主要作用 | `main` 中的五种矩阵 | 分离 raw text、position marker、Legal14、Section6 |
| K=8 是否不合理 | `generate_budget` | 运行 3 个预算 × 6 个候选上限 |
| fit/skip 是否只是避免截断 | `generate_drop_only` | 固定旧版八候选，只删除、不替补 |

### `run_followup_diagnostics.py`

该脚本不训练模型，只复用已完成输出：

| 诊断 | 关键函数 | 回答的问题 |
|---|---|---|
| 正例分布 | `label_diagnostics` | 两种标签是否偏好不同长度、位置和段落角色 |
| 长度接近子集 | `length_close` | 输出长度接近时 ROUGE 差距是否仍存在 |
| 分类型表达式 | `expression_by_type` | 总体 Expression F1 是否掩盖低频类型 |

## 重要指标口径

- `R1/R2/RL`：开发集 LAWA ROUGE F1。
- `Expression_F1_macro_union_types`：规则精确匹配的固定类型并集宏平均 F1。
- `Expression_Recall_conditional_reference_types`：只在参考非空类型上计算的旧条件式 Recall。
- 两种 Recall 定义不能通过改名合并。
- Expression 指标表示词面表达覆盖，不代表语义事实或裁判结果正确。

原始 `all_metrics.json` 中的 `Expression_Recall` 是固定类型并集口径；这是已完成运行的原始记录，保留不改。当前脚本已使用明确名称避免后续混淆。

## 复现与版本说明

- 已执行主实验的代码哈希记录在 `results/manifest.json`。
- 后续为了讲解和指标命名清晰，当前本地脚本增加了注释和显式指标名称，因此其字节哈希与已执行版本不同；模型数值没有被修改。
- 跟进诊断的已执行脚本哈希记录在 `results/followup_manifest.json`。
- 正式引用数值时以 `results/` 和 `RESULTS.md` 为准。

## 不应在讲解中声称的内容

- 不能说候选竞争独立造成了全部 ROUGE 提升。
- 不能把整个 TF--IDF 块的消融解释成纯位置作用。
- 不能说 K=8 对所有模型或重新训练条件都最优。
- 不能说取消候选内截断本身提高了自动指标。
- 不能把候选边界完整性说成事实或语义完整性。
- 不能报告不存在的 SENDE 配对显著性检验。

