# LegalFactAware 核心方法与后续实验交接（2026-09-19）

## 一、研究任务

LegalFactAware 面向法律裁判文书的抽取式摘要，在严格长度预算下选择完整句，最终按原文顺序输出，禁止硬截断。

## 二、核心方法

中文最终版本为 **LegalFactAware v2.5-B**，包括四个模块：

1. **Budget-consistent Oracle**：以参考摘要为目标，在 450 中文字符、最多 8 句约束下，用摘要级边际 ROUGE 增益生成训练标签。
2. **Sentence Representation**：字符 2–4 gram TF-IDF，加 Legal14 法律事实特征与 Section6 篇章结构特征。
3. **Sentence Scoring**：class-balanced Logistic Regression，学习句子内容重要性。
4. **Budget-aware Decoder**：按相关性选择句子，使用字符二元组冗余惩罚（lambda=0.2），仅加入完整句，最后恢复原文顺序。

整体逻辑是：

> 预算一致监督 → 法律事实与篇章表示 → 静态句子评分 → 冗余控制与完整句预算解码

## 三、三个研究方向

### 研究一：固定预算任务对齐

比较普通句级/摘要级监督、普通 Oracle、Budget-consistent Oracle 和完整句预算解码。

- 中文 CAIL 结果支持预算一致监督。
- 英文 IN-Abs 中，稀疏 Budget Oracle 不如普通 Oracle，因此英文静态模型 E1 使用普通摘要级监督。
- 正确结论是：预算一致监督存在跨语言学习边界，不能声称 Budget Oracle 在所有语言中都更优。

### 研究二：摘要状态感知

研究静态评分 `Score(s_i|B)` 与状态相关评分 `Score(s_i|S_t,B)` 的差别。计划采用：

> 普通监督的静态内容评分 → 摘要状态动态重排序 → 完整句预算解码

下一步为 Phase 6B：E1 静态模型保持冻结；动态训练轨迹只能用 6,325 篇 train 的参考摘要产生；703 篇 dev 只用于选择和评估，不能用于训练。

### 研究三：多预算适配

IN-Abs 训练集导出的固定预算为 423/643/973 词。后续比较共享模型、分预算专用模型和预算条件化模型。该阶段尚未完成。

## 四、中文主实验

- 数据：CAIL2020-clean，train/dev/test = 7,495/942/948。
- 协议：450 字、最多 8 句、完整句输出。
- LegalFactAware v2.5-B test：R1 60.94，R2 42.10，RL 50.73，Fact Recall 70.99。
- 当前中文表内居首，但相对 SENDE 仅领先 1.22/0.97/0.08；RL 不能仅凭均值宣称显著。

## 五、英文扩展

- 数据：IN-Abs 官方 train 固定拆分为 6,325 train / 703 dev。
- 官方 100 篇 test 始终封闭。
- 候选句：无参考、全篇均匀取最多 400 句。
- 评测：spaCy 计词、完整句预算、`py-rouge==1.1`。
- E1：word 1–2 gram TF-IDF + 英文化 Legal14/Section6 + balanced LR + 词二元组 MMR + 完整句解码，使用普通摘要级 Oracle 监督。
- 逐篇参考摘要长度轨道：66.42/38.15/62.95，零预算违规。
- 固定 643 词轨道：57.71/34.57/54.80。两种预算轨道不能直接比较。

## 六、状态感知诊断（Phase 6A，已完成）

703 dev、固定 643 词、零预算违规，不训练模型；参考答案仅用于诊断 teacher。

- 当前状态相对初始状态的首选句翻转率：77.81%。
- 初始 top-50 候选的成对排序翻转率：34.77%。
- 初始正边际增益后来变为非正：31.98%。
- E1 MMR 与同一 Oracle 状态下贪心 Oracle 的选句一致率：5.79%。
- E1：57.71/34.57/54.80。
- 参考感知贪心 Oracle：70.03/51.23/67.44。

诊断只说明状态信息值得建模。Oracle 看到了参考摘要，不是可部署方法，也不是正式 `py-rouge` 上界。

## 七、英文基线结果

下列均为 703 篇 dev 的逐篇参考长度轨道：

| 方法 | R1 | R2 | RL |
|---|---:|---:|---:|
| SummaRuNNer | 67.73 | 40.73 | 64.46 |
| SENDE | 66.68 | 39.19 | 63.16 |
| E1 | 66.42 | 38.15 | 62.95 |
| Agarwal-ST | 66.41 | 38.94 | 62.77 |
| DSDR | 64.75 | 34.17 | 61.05 |
| LexRank | 63.50 | 34.48 | 59.93 |
| Gist | 63.38 | 33.74 | 59.36 |
| CaseSummarizer | 61.79 | 33.33 | 58.73 |
| MemSum | 59.18 | 30.23 | 55.09 |
| SSE | 54.36 | 26.22 | 46.85 |

生成式独立轨道：Budget Recipe 54.44/29.23/48.03；CivilSum LED-tail 48.19/26.64/42.01。

TermDiffuSum 英文未跑：现有实现固定 200 句，而统一协议允许 400 句；703 dev 中 101 篇超过 200 句，不能公平进入主排名。

## 八、表述纪律

1. 原论文数值验真、数据集适配、工程跑通必须区分。
2. SENDE、SSE、Agarwal-ST、Gist 等英文结果多为独立重建或受限适配，不是作者 checkpoint 原样复现。
3. Budget Recipe 是公开 Qwen3-4B checkpoint 零样本迁移，未执行完整 SFT+RLVR。
4. CivilSum 和 Budget Recipe 不进入抽取式主排名。
5. 中英文 ROUGE 绝对值不可跨数据集比较。
6. 官方 IN-Abs test 在全部配置冻结前不得读取。

## 九、关键文件

- `inabs_extension_work/CAIL_INABS_COMBINED_RESULTS_20260912.md`
- `inabs_extension_work/PREREGISTRATION_PHASE6A_STATE_DIAGNOSTICS.md`
- `inabs_extension_work/phase6a_state_diagnostics_compact_results.tgz`
- `inabs_extension_work/PHASE5H_TERMDIFFUSUM_FEASIBILITY.md`
- `inabs_extension_work/phase5g_budgetrecipe_compact_results.tgz`

## 十、下一步

优先预注册并实现 Phase 6B 状态感知动态重排序模型。输入至少包括：E1 静态分、候选与已选摘要的重复度、事实与结构覆盖、剩余预算、句长占剩余预算比例。训练目标优先使用连续边际增益或成对排序。先做 smoke，再用后台任务运行完整 train/dev。

长任务应放入 `screen` 后台，退出 SSH 后告知预计完成时间。服务器密码不得写入项目文件；需要连接时由用户在对话中提供。
