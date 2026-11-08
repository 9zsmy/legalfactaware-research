# Gist UK-Abs 官方代码验证（2026-09-23）

## 验证范围

- 官方 UK-Abs 100 篇测试集。
- 官方 Zenodo UK-Abs `summary_model.pkl` 与 `word2vec_model.bin`。
- 官方仓库原始特征生成与 `generate_summary.py` 解码函数。
- 仅增加运行时兼容层：gensim 3/4 词表接口、旧 sklearn pickle 路径、
  LightGBM Booster 概率接口、Windows 路径别名；未改写官方特征或选句逻辑。
- 按逐篇参考摘要词数预算，按分数降序选句，遇到第一句超预算时停止。
- `py-rouge==1.1`，`max_n=2`，`limit_length=False`；摘要展平为单行，避免
  换行触发 summary-level union-LCS。

## 结果

| 配置 | R1-F | R2-F | RL-F |
|---|---:|---:|---:|
| 论文 Gist | 42.70 | 13.20 | 21.50 |
| 官方原始代码＋UK checkpoint | 61.22 | 24.66 | 23.15 |
| 差值 | +18.52 | +11.46 | +1.65 |
| IN checkpoint/词向量交换诊断 | 60.50 | 24.90 | 23.52 |

checkpoint 交换没有使结果接近论文，因此排除发布包中 IN/UK 模型标签互换这一
解释。论文明确要求所有方法使用逐篇参考摘要长度作为上限，不能通过任意缩短
摘要预算来调低 R1/R2。

## Top-K 截断敏感性诊断

在不改变官方句子分数、参考摘要长度预算及完整句规则的前提下，额外测试只让
得分最高的前 K 句参与解码。该实验仅用于寻找差异来源，不作为正式复现配置。

| 候选上限 | R1-F | R2-F | RL-F | 平均选句数 | 平均用词数 |
|---|---:|---:|---:|---:|---:|
| 论文 Gist | 42.70 | 13.20 | 21.50 | — | — |
| Top-10 | 27.29 | 11.23 | 14.36 | 10.00 | 236.46 |
| Top-15 | 37.72 | 15.53 | 17.94 | 14.98 | 371.67 |
| Top-20 | 45.77 | 18.42 | 20.07 | 19.93 | 512.10 |
| Top-25 | 51.75 | 20.77 | 21.42 | 24.80 | 649.83 |
| Top-30 | 55.63 | 22.17 | 22.10 | 29.41 | 779.56 |

Top-20 的三指标联合距离最小，但仍为 `+3.07/+5.22/-1.43`；Top-25 虽然
RL 几乎重合，R1/R2 却分别高出 9.05/7.57。三个指标分别需要明显不同的 K
才能靠近论文，因此不存在一个自然、统一的 Top-K 能解释论文表格，而且官方
源码和论文均没有给出这种截断设置。

## 官方代码与兼容实现一致性

- 去掉末尾换行和空行后，84/100 篇摘要逐字一致。
- 100 篇平均选句集合 Jaccard：0.9896；中位数为 1.0。
- 平均选句数：官方代码 40.30，兼容实现 40.29。
- 剩余 16 篇仅有少量边界句差异；官方代码与兼容实现的整体 ROUGE 基本相同。

因此，先前高分不是兼容重实现偏离官方代码造成的。

## 结论

最终状态：**官方 checkpoint、官方特征代码与官方解码工程验证通过；论文
ROUGE-L 接近，但 R1/R2 数值复现失败。** 当前没有源码依据支持进一步修改
候选范围、预算或解码策略，不能为了接近论文而选择任意截断值。

可使用的论文表述：

> Gist 的发布模型与官方代码已在 UK-Abs 官方测试集上成功运行，但复评结果
> 61.22/24.66/23.15 与原文 42.70/13.20/21.50 在 ROUGE-1/2 上存在显著差异。
> 模型资产交换、官方代码等价性、多种 ROUGE 格式及 Top-K 敏感性审计均无法
> 解释该差异，因而将其记为工程复现成功、论文数值未复现，并在统一协议比较中报告本次
> 可审计结果。

## 本地证据

- `results/gist_ukabs_official_code_audit/manifest.json`
- `results/gist_ukabs_official_code_audit/metrics.json`
- `results/gist_ukabs_official_code_audit/comparison.jsonl`
- `results/gist_ukabs_official_code_audit/topk_sensitivity.json`
- `results/gist_ukabs_checkpoint_swap_audit/metrics.json`
- `gist_ukabs_audit_compact.tgz`
- `run_gist_official_code_audit.py`
