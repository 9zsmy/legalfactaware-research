# SummaRuNNer UK-Abs 原任务验证（2026-09-23）

## 最终验证身份

本次已从“受限忠实重训练”升级为**发布 checkpoint 推理验证**。

- 官方 Zenodo 模型：`UKRNN_RNN_seed_1.pt`。
- 原始 SummaRuNNer 分叉中找回 `word2id.json`，共 153,824 个词项。
- 词表大小与发布 checkpoint 的 `embed.weight` 153,824 行完全一致。
- checkpoint 内参数包含 `topk=15`、每篇最多 100 句、每句最多 50 词。
- 上游 `Vocab.make_features` 使用原始大小写的空格分词；本次据此作为主配置。
- 数据为官方 UK-Abs 100 篇测试集；IN-Abs 官方 100 篇测试集未读取。

因此，下面的主结果不再依赖自行重训练模型。此前 693 篇训练集上的 30 轮
重训练结果仍保留为工程诊断，但不再作为原论文数值验证的主证据。

## 恢复后的协议与评测诊断

1. 使用发布的 UK-Abs SummaRuNNer checkpoint。
2. 使用与 checkpoint 嵌入行数完全匹配的原始 153,824 项词表。
3. 按上游源码对句子做原始大小写空格分词。
4. 每篇最多输入 100 句，每句最多输入 50 词。
5. 按 checkpoint 的 `topk=15` 最多选择 15 句。
6. 在逐篇参考摘要词数预算内只保留完整句；放不下的句子跳过。
7. 按原文顺序输出。
8. 使用 `py-rouge==1.1`、`max_n=2`、`limit_length=False`。

前 1—8 项均有 checkpoint、上游源码、README 或论文协议依据。另做一项评测
格式诊断：将候选摘要和参考摘要的换行展平为单行。该处理不改变 R1/R2，但使
ROUGE-L 按整篇单序列计算，解释了此前异常偏高的 RL。官方未发布最终评测
脚本，因此“展平换行”是有结果证据支持的协议恢复推断，而不是已被源码直接
确认的事实。

## 主结果

| 方法与口径 | R1-F | R2-F | RL-F | 预算违规 |
|---|---:|---:|---:|---:|
| 论文 SummaRuNNer | 50.20 | 20.50 | 23.70 | 未报告 |
| 发布 checkpoint，Top-15、预算约束、单行 py-rouge | 46.23 | 22.46 | 22.47 | 0 |
| 与论文差值（本次−论文） | -3.97 | +1.96 | -1.23 | — |

按“结果大致相当即可”的宽松验证标准，三项指标已经处于同一量级，最大绝对
差值为 3.97 分，因此可判为：**在合理的单行评测格式推断下，基本数值验证
通过，但不是逐点精确复现**。

## 差异来源审计

| 诊断配置 | R1-F | R2-F | RL-F |
|---|---:|---:|---:|
| 发布 checkpoint，填满预算、保留换行 | 63.88 | 29.55 | 61.32 |
| 发布 checkpoint，Top-15、保留换行 | 46.23 | 22.46 | 44.02 |
| 发布 checkpoint，Top-15、展平换行 | 46.23 | 22.46 | 22.47 |

审计表表明两个差异来源彼此独立：

- 恢复 checkpoint 中的 `topk=15` 解释了大部分 R1/R2 差异；此前把摘要尽量
  填满参考长度，会把 R1/R2 人为推高到约 64/30。
- 展平换行不影响 R1/R2，却把 RL 从 44.02 降到 22.47，说明此前 61 或 44
  的 RL 主要来自 summary-level/union-LCS 对换行边界的处理，并非模型质量。

额外诊断中，取消预算检查的原始 Top-15 版本为 46.18/22.47/43.98，但有 1 篇
超预算；预算约束版消除了违规，分数几乎不变。大小写、去标点以及“遇到首个
放不下句子时停止或跳过”的变化均不足 0.5 分，不能解释论文差距。

## 结论与可用表述

最终状态更新为：**官方 checkpoint 工程验证通过；在恢复 Top-15，并采用能
解释论文 RL 的单行 py-rouge 格式后，宽松数值验证基本通过；单行格式尚无最终
评测脚本直接佐证，不能称为完全复现。**

论文中可写：

> 在官方 UK-Abs 100 篇测试集上，使用发布的 SummaRuNNer checkpoint、与其
> 嵌入矩阵严格匹配的原始词表、Top-15 抽取和逐篇参考长度预算后，复评结果为
> 46.23/22.46/22.47；与论文报告的 50.20/20.50/23.70 处于相同量级。剩余差异
> 可能来自未发布的文本预处理或评测细节，因此将其视为基本数值验证，而非精确
> 复现。

## 本地证据

- `summarunner_ukabs_official_checkpoint_audit/protocol_comparison.json`
- `summarunner_ukabs_official_checkpoint_audit/evaluation_pyrouge_unflattened.json`
- `summarunner_ukabs_official_checkpoint_audit/manifest.json`
- `summarunner_ukabs_official_checkpoint_audit/records.jsonl`
- `infer_official_summarunner_ukabs.py`
- `summarunner_ukabs_validation/`（此前重训练诊断）
