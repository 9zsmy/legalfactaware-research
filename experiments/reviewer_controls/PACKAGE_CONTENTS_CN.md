# 代码包内容说明

推荐代码包：`LegalFactAware_Reviewer_Controls_Full_Code_and_Walkthrough_20261029.zip`

## 包含内容

### 本次审稿补充实验

- `run_reviewer_controls.py`：四组核心控制实验、预算网格、评价与显著性检验。
- `run_followup_diagnostics.py`：正例分布、输出长度接近和分类型表达式诊断。
- `summarize_for_talk.py`：只读取汇总结果的现场演示脚本。
- `requirements.txt`：主要第三方 Python 依赖。
- 两份预注册说明、完整结果说明、README 和两份中文讲解稿。
- `results/` 下的汇总 JSON、CSV 和哈希 manifest。

### 主实验依赖的核心源码

压缩包同时包含 `LegalFactAware_key_code_20260927/01_chinese_core/` 下的完整 Python 源码，包括：

- `run_legalfactaware_v2_cail_clean.py`：数据读取、候选表示和基础选择逻辑；
- `run_legalfactaware_v21_budget_decoder_dev.py`：预算感知 fit/skip 解码与评价；
- `run_legalfactaware_v22_oracle_supervision_dev.py`：Oracle 监督构造；
- `run_legalfactaware_v25_structure_feature_dev.py`：Section6 结构特征；
- `run_legalfactaware_formal_ablation_dev.py`：正式消融和表达式评价函数；
- `evaluate_sse_cail.py`：Expression 规则、ROUGE 和辅助评价；
- 其余已归档的表示、类别平衡、假设和边际效用实验源码。

## 不包含内容

- CAIL 或其他原始法律文书；
- 训练集、开发集和测试集 JSONL；
- 模型权重、向量器 pickle；
- 包含文书内容的逐文档预测；
- 远程机器密码、密钥或账号信息。

这些输入由命令行参数传入。主实验会用 SHA-256 验证冻结输入，避免误用不同版本。

## 推荐阅读顺序

1. `experiments/chinese_reviewer_controls_20261028/FULL_CODE_WALKTHROUGH_CN.md`
2. `experiments/chinese_reviewer_controls_20261028/README_CN.md`
3. `experiments/chinese_reviewer_controls_20261028/run_reviewer_controls.py`
4. `experiments/chinese_reviewer_controls_20261028/run_followup_diagnostics.py`
5. `experiments/chinese_reviewer_controls_20261028/RESULTS.md`

## 快速演示

在解压目录的仓库根目录运行：

```bash
python3 experiments/chinese_reviewer_controls_20261028/summarize_for_talk.py
```

该命令不需要数据集或模型，只读取已打包的汇总结果。

