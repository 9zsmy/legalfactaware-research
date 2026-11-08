# NAACL 2025 Zero-Shot LCS：CAIL 适配预注册

冻结日期：2026-11-08。以下设置在查看本次 Qwen 输出之前确定。

## 方法身份

- 论文：Retkowski and Waibel, *Zero-Shot Strategies for Length-Controllable
  Summarization*, Findings of NAACL 2025, pp. 551–572。
- 官方论文：https://aclanthology.org/2025.findings-naacl.34/
- 原论文模型与数据：LLaMA-3-8B-Instruct；YTSEG、CNN/DailyMail。
- 本实验模型：`Qwen/Qwen2.5-7B-Instruct`，本地固定快照；4-bit NF4。
- 结果名称必须包含 `CAIL-adapted` 或 `independent implementation`；不得称为作者官方
  CAIL 复现。

## 选择的方法

直接复现论文中可跨语言、且与字符预算任务对应的两步：

1. Sample Filtering（SF）：每篇独立生成 3 个摘要，按长度约束选择候选。论文主表的 SF
   数值对应附录中的 3-sample 设置。
2. Automated Revisions（AR）：如候选不合规，反馈实际字符数和与目标的差值，最多修订
   5 轮。每轮仅以上一轮摘要为基础，符合论文独立修订模板。

暂不使用 Length Approximation 与 Target Adjustment。论文的字符—词转换系数与三次回归
由英文输出估计；没有独立中文校准集时直接套用会混入语言不匹配，重新在测试集拟合又会
造成泄漏。

## 冻结配置

- 数据：冻结 CAIL 开发集；先运行顺序前 5 篇，只作工程与方向验收。
- 输入：head-tail 8192 个 Python 字符，与现有生成基线一致。
- 字符计数：`len(text)`，包含空白；与论文 Listing A3 和本项目正式口径一致。
- Sample Filtering：`N=3`，只看长度，不查看参考摘要、ROUGE 或 Expression 指标。
- Automated Revisions：最多 5 轮。
- 推理：top-k 20、top-p 0.8、temperature 0.7、repetition penalty 1.05、seed 0。
- 不进行任何事后截断；保存 3 个初始候选和每一轮修订全文。
- 同时运行两个预先冻结的目标：
  - `exact`：论文忠实目标，450 字符；±10% 判为合规。
  - `upper_bound`：任务适配目标，严格不超过 450 字符；有多个合规候选时选择最接近
    450 字符者。该一侧选择规则是 CAIL 适配，不是论文原始定义。

## 5 篇停止与扩展规则

5 篇仅检查：程序完整运行、输出不是空串/乱码、轨迹可审计、没有事后截断、长度约束、
源文表达支持、ROUGE/Expression 的粗略方向。不得用 5 篇显著性或优劣结论写入论文。

只有当以下条件同时满足时，才建议扩到冻结的 100 篇子集：

1. 两个目标均 5/5 产生有效输出且轨迹字段完整；
2. `upper_bound` 不出现预算违反；
3. 没有明显模板复读或系统性事实编造；
4. 相对 LegalFactAware 的 ROUGE-L 差距不呈灾难性扩大。

