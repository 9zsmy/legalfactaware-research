# ACL 2025 Black-box LCG：CAIL 适配预注册

冻结日期：2026-10-31。以下配置在查看 Qwen 输出之前确定。

## 方法身份

- 论文：Gu et al., *Length Controlled Generation for Black-box LLMs*, ACL 2025。
- 模型：`Qwen/Qwen2.5-7B-Instruct`，revision
  `a09a35458c702b33eeacc393d103063234e8bc28`。
- 名称：`LCG-MHIS-Qwen2.5-7B-CAIL-adapted`。
- 这是中文法律任务适配复现，不是作者发布的 CAIL 结果；论文页面未提供官方代码附件。

## 冻结配置

- 数据：冻结开发集 942 篇；先运行前 5 篇工程冒烟。
- 约束：不超过 450 个 Python 字符。原论文的英文 word counter 改为任务正式字符口径。
- 输入：head-tail 8192 字符；所有生成基线相同。
- one-shot：CAIL 训练集 row 0，ID `b113dd93114d50e3cc9840475cc14cdb`；
  已确认不与 942 篇开发集重叠。示例来源最多保留 4096 字符。
- 迭代：MH+IS，beam size 1，最多 5 次；同时保存初始直接提示输出。
- 采样：top-k 20、top-p 0.8、temperature 0.7、repetition penalty 1.05、seed 0。
- 模型：4bit NF4，仅属于部署适配；结果中披露。
- 不对输出事后截断。

当当前摘要超过 450 字符时，重要性分布提示模型按当前字符差距压缩；候选仍超长时，
同一模型按信息覆盖、语言流畅、简洁性、逻辑连贯、事实忠实五项进行成对评分。
接受概率使用论文式上界：

`min(1, f(candidate) / f(current) * score(candidate) / score(current))`

其中满足不超过 450 字符的候选直接接受并停止。

## 决策规则

5 篇只判断可运行性、生成退化、事实支持、长度遵守和大致质量，不做显著性结论。
只有在没有系统性退化、且 ROUGE-L 与 LegalFactAware 的差距明显小于 DELTA 冒烟的
约 29.6 点时，才考虑 942 篇全量运行。

