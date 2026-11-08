# LegalFactAware 关键代码本地归档

归档日期：2026-09-27  
归档位置：`/Users/admin/Documents/ChatGPT/1/LegalFactAware_key_code_20260927`

## 1. 归档目的

本目录集中保存 LegalFactAware 论文当前最关键、最能解释方法与实验结论的代码。归档内容分为中文主方法、英文 IN-Abs 扩展、统一评测和论文基线验证四部分。

本目录是“源码与实验说明归档”，不是完整训练镜像。为控制体积并避免泄露，本次没有复制：

- CAIL、IN-Abs 等数据集；
- 模型权重、pickle 模型、checkpoint；
- Hugging Face 缓存、虚拟环境、日志和中间预测；
- 服务器密码或其他登录凭据。

每个文件的大小、SHA-256 和来源记录在 `MANIFEST.tsv`。

## 2. 目录结构

```text
LegalFactAware_key_code_20260927/
├── 01_chinese_core/          中文 LegalFactAware v2.5-B 与正式消融
├── 02_english_inabs/         英文 IN-Abs 的 E1、Oracle、解码和状态诊断
├── 03_evaluation/            ROUGE、协议审计和结果汇总
├── 04_baseline_validation/   英文基线适配与原论文验证脚本
├── docs/                     预注册、结果报告和方法交接
├── requirements-core.txt     核心依赖说明
└── MANIFEST.tsv              文件清单与校验值
```

## 3. 中文主方法：代码与论文模块对应关系

中文最终方法是 LegalFactAware v2.5-B，协议为 CAIL2020-clean、450 字、最多 8 句、完整句输出并恢复原文顺序。

| 论文模块 | 关键代码 | 主要入口或函数 |
|---|---|---|
| 基础表示和句子评分 | `01_chinese_core/run_legalfactaware_v2_cail_clean.py` | `fit_model`、`predict_probabilities`；字符 2–4 gram TF-IDF 与 Legal14 特征 |
| Legal14 特征实现 | `01_chinese_core/run_legal_factaware_lcsts.py` | `explicit_features`、`decorated_text`、`extracted_fact_sets` |
| 完整句预算解码 | `01_chinese_core/run_legalfactaware_v21_budget_decoder_dev.py` | `candidate_score`、`select_budget_greedy`；450 字、8 句、冗余惩罚 0.2 |
| Budget-consistent Oracle | `01_chinese_core/run_legalfactaware_v22_oracle_supervision_dev.py` | `oracle_score`、`select_budget_oracle`、`build_oracle_training` |
| Section6 篇章结构 | `01_chinese_core/run_legalfactaware_v25_structure_feature_dev.py` | `section_labels`、`section_one_hot`、`fit_lr` |
| 最终统一消融 | `01_chinese_core/run_legalfactaware_formal_ablation_dev.py` | 统一训练标签、特征、分类器、解码器、预算和显著性检验 |
| 扩展消融 | `run_legalfactaware_class_balance_ablation_dev.py`、`run_legalfactaware_minus_tfidf_ablation_dev.py`、`run_legalfactaware_tf_only_representation_dev.py` | 类别平衡、去除 TF-IDF、TF-only 对照 |
| 机制分析 | `run_legalfactaware_hypothesis_ablation_dev.py`、`run_legalfactaware_marginal_utility_exploration_dev.py` | 假设驱动消融与边际效用分析 |

### 重要边界

`run_legalfactaware_v2_cail_clean.py` 是方法演化中的基础版本，内部仍包含末尾字符截断逻辑；它不能单独代表最终 v2.5-B。最终完整句预算逻辑以 `run_legalfactaware_v21_budget_decoder_dev.py` 的 `select_budget_greedy` 为准，最终结构特征以 v2.5 脚本为准。

中文方法的正式组合关系是：

```text
Budget-consistent Oracle
        ↓
字符 2–4 gram TF-IDF + Legal14 + Section6
        ↓
class-balanced Logistic Regression
        ↓
字符二元组冗余控制 + 完整句预算解码
        ↓
恢复原文顺序
```

## 4. 英文 IN-Abs 扩展

英文最终静态配置 E1 不直接照搬中文 Budget Oracle。已有实验表明，IN-Abs 上普通摘要级监督比稀疏 Budget Oracle 更稳定。因此英文部分采用普通监督学习静态重要性，预算信息主要用于完整句解码和后续状态建模。

推荐阅读顺序：

1. `02_english_inabs/run_inabs_phase1.py`：官方数据检查、6,325/703 固定划分、423/643/973 预算和 Oracle 诊断；不读取官方 100 篇 test。
2. `02_english_inabs/run_inabs_phase2_lr.py`：普通 Oracle 与 Budget Oracle 的严格 LR 对照；包含英文化 Legal14、Section6、词二元组 MMR 和完整句解码。
3. `02_english_inabs/run_inabs_phase3_static_repr.py`：E0/E1/E2 英文表示比较，E1 为 word 1–2 gram TF-IDF。
4. `02_english_inabs/run_inabs_phase3c_spacy_decode.py`：按官方 spaCy 计词重新解码，检查零预算违规。
5. `02_english_inabs/evaluate_phase3b_official_pyrouge.py`：官方兼容 ROUGE 评测。
6. `02_english_inabs/run_inabs_phase6a_state_diagnostics.py`：摘要状态翻转率、边际增益归零率和 Oracle 一致率诊断。

当前 Phase 6A 只证明状态信息值得建模，不是可部署动态模型。Phase 6B 状态感知重排序仍未完成。

## 5. 统一评测和结果文件

`03_evaluation/` 中保存：

- 中文与英文评测脚本；
- `py-rouge` 与 UK-Abs 协议审计；
- 结果总表生成脚本；
- 当前实验总表和“原论文—复现结果—原文截图”对照表。

评测时必须保持以下边界：

- CAIL 与 IN-Abs 的 ROUGE 绝对值不能跨数据集比较高低；
- IN-Abs 的逐篇参考长度轨道与固定 423/643/973 词轨道不能混表比较；
- 官方 100 篇 test 在模型、预算和超参数冻结前不得读取；
- 工程跑通、独立重实现、官方代码重训和官方 checkpoint 推理必须分开表述。

## 6. 基线验证代码

`04_baseline_validation/` 保存 SummaRuNNer、Gist、MemSum、Agarwal-ST、SSE、SENDE、CivilSum、Budget Recipe、TermDiffuSum、DSDR、KMM、MMR、CaseSummarizer 和 FedSum 的关键适配或审计入口。

这些文件不都等同于“原论文官方复现”：

- SummaRuNNer、KMM 等包含官方资产与协议审计；
- SENDE、SSE、Agarwal-ST 等主要是独立重建或受限适配；
- Budget Recipe 只验证公开 checkpoint，没有复现完整 SFT+RLVR；
- TermDiffuSum 在 IN-Abs 上受 200 句实现上限限制，尚未进入统一 400 句主排名；
- FedSum 是基于官方源码的修复性独立复现，核心 FedSum 优于 FedAvg 的趋势没有稳定复现。

具体身份和限制以 `docs/` 中的预注册和验证报告为准。

## 7. 环境与运行原则

建议 Python 3.10。核心 CPU 方法至少需要 NumPy、SciPy、scikit-learn、LightGBM 和 rouge-score。官方兼容评测还需要 `py-rouge==1.1` 对应的 `rouge` 模块；涉及英文句子切分的脚本需要 spaCy。

先安装依赖，再将同一子目录加入 `PYTHONPATH`。示例仅展示路径结构，数据和模型路径需替换为实际位置：

```bash
python -m pip install -r requirements-core.txt

python 01_chinese_core/run_legalfactaware_formal_ablation_dev.py \
  --train /path/to/cail/train.jsonl \
  --dev /path/to/cail/dev.jsonl \
  --oracle-extracts /path/to/oracle_train.jsonl \
  --v22-model /path/to/v22_model.pkl \
  --frozen-v25-model /path/to/v25_model.pkl \
  --output-dir /path/to/output
```

正式消融脚本包含输入文件 SHA-256 校验。若数据文件不是冻结版本，脚本会主动终止。不要为了运行而删除这些校验；应先核对数据版本和划分。

英文 Phase 1 示例：

```bash
python 02_english_inabs/run_inabs_phase1.py \
  --data /path/to/dataset/IN-Abs \
  --archive /path/to/dataset.zip \
  --output /path/to/phase1_dev
```

## 8. 当前可信结果

- CAIL test：LegalFactAware v2.5-B = R1 60.94、R2 42.10、RL 50.73、Fact Recall 70.99。
- IN-Abs 703 dev，逐篇参考长度轨道：E1 = 66.42/38.15/62.95，零预算违规。
- IN-Abs 703 dev，固定 643 词轨道：E1 = 57.71/34.57/54.80。
- Phase 6A：首选句翻转率 77.81%，成对排序翻转率 34.77%，正边际增益归零率 31.98%，E1 MMR 与参考感知贪心 Oracle 一致率 5.79%。

上述英文结果均来自开发集，官方 100 篇 test 仍应保持封闭。

## 9. 建议的后续开发入口

下一步若实现 Phase 6B，应以 `02_english_inabs/run_inabs_phase6a_state_diagnostics.py` 的状态定义和 `run_inabs_phase3_static_repr.py` 的冻结 E1 分数为基础，新建独立训练脚本，不要改写已经冻结的 Phase 1–3 文件。

动态重排序输入至少应包含：E1 静态分、候选与已选摘要重复度、事实覆盖、结构覆盖、剩余预算、句长占剩余预算比例。训练轨迹只能由 6,325 篇 train 产生，703 篇 dev 只用于选择和评估。

